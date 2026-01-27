"""Databases for large-scale datasets."""
import bisect
import os
import pickle
import sqlite3
from contextlib import contextmanager
from typing import *

import numpy as np
import snappy

__all__ = [
    'BytesDB',
    'BytesSqliteDB',
    'BytesMultiDB',
]


class BytesDB(object):

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            self.commit()
        self.close()

    def __len__(self) -> int:
        return self.data_count()

    def __getitem__(self, item: int):
        return self.get(item)

    def __iter__(self):
        for i in range(self.data_count()):
            yield self.get(i)

    def __repr__(self):
        desc = self.describe().strip()
        if '\n' in desc:
            desc = '\n'.join(f'  {l}' for l in desc.split('\n'))
            desc = f'\n{desc}\n'
        return f'{self.__class__.__name__}({desc})'

    def describe(self) -> str:
        raise NotImplementedError()

    def sample_n(self, n: int) -> List[bytes]:
        ret = []
        indices = np.random.randint(self.data_count(), size=n)
        for i in indices:
            ret.append(self.get(i))
        return ret

    def data_count(self) -> int:
        raise NotImplementedError()

    def get(self, item: int) -> bytes:
        raise NotImplementedError()

    def add(self, val: bytes) -> int:
        raise NotImplementedError()

    @contextmanager
    def write_batch(self):
        raise NotImplementedError()

    def commit(self):
        raise NotImplementedError()

    def optimize(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


class BytesSqliteDB(BytesDB):

    class WB(object):

        def __init__(self, conn, cur, table_name, buf_size=8192):
            self.conn = conn
            self.cur = cur
            self.table_name = table_name
            self.buf = []
            self.buf_size = buf_size

        def add(self, id, value):
            self.buf.append((id, snappy.compress(value)))
            if len(self.buf) >= self.buf_size:
                self.commit()

        def commit(self):
            if self.buf:
                self.cur.executemany(
                    f'INSERT INTO "{self.table_name}"("key", "value") VALUES (?, ?)',
                    self.buf
                )
                self.conn.commit()
                self.buf.clear()

        def rollback(self):
            self.conn.rollback()
            self.buf.clear()

    conn: sqlite3.Connection
    path: str
    file_name: str
    _data_count: int

    def __init__(self, path: str, write: bool = False, table_name: str = 'data',
                 file_name: str = '_bytes.db'):
        self.path = path
        self.table_name = table_name
        self.file_name = file_name

        if write and not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        db_path = os.path.join(self.path, file_name)
        # 检查数据库文件是否存在且可读
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
        if not os.access(db_path, os.R_OK):
            raise PermissionError(f"数据库文件无读取权限: {db_path}")
        
        try:
            self.conn = sqlite3.connect(db_path)
        except sqlite3.OperationalError as e:
            raise sqlite3.OperationalError(f"无法打开数据库文件 {db_path}: {str(e)}")
        
        self.conn.text_factory = bytes
        with self._scoped_cursor() as cur:
            cur.execute(
                f'CREATE TABLE IF NOT EXISTS "{self.table_name}" ('
                '  "key" INT PRIMARY KEY,'
                '  "value" BLOB'
                ');'
            )
            self.conn.commit()
            self._data_count = cur.execute(f'SELECT COUNT(*) FROM "{self.table_name}"').fetchone()[0]
        self._wb = None
    @contextmanager
    def _scoped_cursor(self):
        cur = self.conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def describe(self) -> str:
        p = self.path
        if self.file_name != '_bytes.db':
            p = os.path.join(p, self.file_name)
        if any(c in p for c in '(),'):
            return repr(p)
        return p

    def data_count(self) -> int:
        return self._data_count

    def get(self, item: int) -> bytes:
        with self._scoped_cursor() as cur:
            cur.execute(f'SELECT "value" FROM "{self.table_name}" WHERE "key" = {item}')
            row = cur.fetchone()
            if row is not None:
                return snappy.decompress(row[0])

    def set(self, item: int, barr: bytes):
        with self._scoped_cursor() as cur:
            cur.execute(f'UPDATE "{self.table_name}" SET "value" = (?) WHERE "key" = {item}',
            (snappy.compress(barr), ))
            self.conn.commit()

    def add(self, val: bytes) -> int:
        if self._wb is None:
            with self.write_batch():
                return self.add(val)
        else:
            key = self._data_count
            self._wb.add(key, val)
            self._data_count += 1
            return key

    @contextmanager
    def write_batch(self):
        if self._wb is not None:
            raise RuntimeError(f'Another write_batch is already open!')
        try:
            self._wb = self.WB(self.conn, self.conn.cursor(), self.table_name)
            yield self
            self._wb.commit()
            self._wb = None
        except:
            self._wb.rollback()
            self._wb = None
            raise

    def commit(self):
        if self._wb is not None:
            self._wb.commit()

    def optimize(self):
        pass

    def close(self):
        self.commit()
        self._wb = None
        self.conn.close()


class BytesMultiDB(BytesDB):

    db_list: List[BytesDB]
    db_sizes: List[int]
    _db_offset: List[int]
    _data_count: int

    def __init__(self, *db_list):
        self.db_list = list(db_list)
        self.db_sizes = [db.data_count() for db in self.db_list]
        self._db_offset = []
        i = 0
        for db in self.db_list:
            self._db_offset.append(i)
            i += db.data_count()
        self._data_count = i

    def describe(self) -> str:
        return '\n'.join(f'{db.describe()},' for db in self.db_list).rstrip(',')

    def data_count(self) -> int:
        return self._data_count

    def get(self, item: int) -> bytes:
        if item < 0 or item >= self._data_count:
            raise IndexError(item)
        i = bisect.bisect_left(self._db_offset, item + 1) - 1
        return self.db_list[i].get(item - self._db_offset[i])

    def add(self, val: bytes) -> int:
        raise RuntimeError(f'BytesMultiDB is not writeable.')

    @contextmanager
    def write_batch(self):
        raise RuntimeError(f'BytesMultiDB is not writeable.')

    def commit(self):
        pass

    def optimize(self):
        raise RuntimeError(f'BytesMultiDB is not writeable.')

    def close(self):
        for db in self.db_list:
            db.close()
