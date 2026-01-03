from typing import *

import torch
from dgl.data import DGLDataset

from tracegnn.data import *
from .config import ExpConfig
from loguru import logger
from tqdm import tqdm
import numpy as np

import os
from tracegnn.utils.host_state import load_host_infra_index, host_state_vector, DEFAULT_METRICS, DISK_METRICS
from tracegnn.utils.trace_host_fusion import load_host_topology, load_host_infra_index as load_infra_for_seq


def init_config(config: ExpConfig):
    processed_dir = os.path.join(config.dataset_root_dir, config.dataset, 'processed')
    id_manager = TraceGraphIDManager(processed_dir)

    # Set DatasetParams
    config.DatasetParams.operation_cnt = id_manager.num_operations
    config.DatasetParams.service_cnt = id_manager.num_services
    config.DatasetParams.status_cnt = id_manager.num_status

    # Set runtime info
    if config.RuntimeInfo.latency_range is None:
        tmp_latency_dict: Dict[int,List[float]] = {}

        config.RuntimeInfo.latency_range = torch.zeros([config.DatasetParams.operation_cnt + 1, 2], dtype=torch.float)
        config.RuntimeInfo.latency_p98 = torch.zeros([config.DatasetParams.operation_cnt + 1], dtype=torch.float)
        
        # TODO: Set default value
        config.RuntimeInfo.latency_range[:, :] = 50.0

        with TraceGraphDB(BytesSqliteDB(os.path.join(processed_dir, 'train'))) as db:
            logger.info('Get latency range...')
            t = db if not config.enable_tqdm else tqdm(db)
            for g in t:
                for _, nd in g.iter_bfs():
                    tmp_latency_dict.setdefault(nd.operation_id, [])
                    tmp_latency_dict[nd.operation_id].append(nd.features.avg_latency)
        for op, vals in tmp_latency_dict.items():
            vals_p99 = np.percentile(vals, 99)
            vals = np.array(vals)
            if np.any(vals < vals_p99):
                vals = vals[vals < vals_p99]

            # Set a minimum value for vals to avoid nan
            
            # TODO: set this
            vals_mean, vals_std = np.mean(vals), max(np.std(vals), 10.0)
            # vals_mean, vals_std = np.mean(vals), np.std(vals)

            config.RuntimeInfo.latency_range[op] = torch.tensor([vals_mean, vals_std])
            config.RuntimeInfo.latency_p98[op] = np.percentile(vals, 98)
        
        config.RuntimeInfo.latency_range = config.RuntimeInfo.latency_range.to(config.device)
        config.RuntimeInfo.latency_p98 = config.RuntimeInfo.latency_p98.to(config.device)


class TrainDataset(DGLDataset):
    def __init__(self, config: ExpConfig, valid=False):
        self.config = config

        # Load id_manager and basic_info
        processed_dir = os.path.join(config.dataset_root_dir, config.dataset, 'processed')
        self.id_manager = TraceGraphIDManager(processed_dir)
        # load infra index once (optional)
        try:
            self.infra_index = load_host_infra_index(processed_dir)
        except Exception:
            self.infra_index = None
        # load host topology disabled on training side (no one-hop topo approx)
        self.host_adj = {}
        # load infra index for sequence (OmniAnomaly backend)
        # try:
        #     self.infra_index_seq = load_infra_for_seq(processed_dir)
        # except Exception:
        #     self.infra_index_seq = None
        self.infra_index_seq = None # 直接设为 None，强制只用预计算数据

        if not valid:
            self.train_db = TraceGraphDB(BytesSqliteDB(os.path.join(processed_dir, 'train')))
            
        else:
            self.train_db = TraceGraphDB(BytesSqliteDB(os.path.join(processed_dir, 'val')))

        # Set config
        init_config(config)

        # Show info
        logger.info(f'{len(self.train_db)} in {"train" if not valid else "val"} dataset.')

    def process(self):
        pass

    def __len__(self):
        return len(self.train_db)

    def __getitem__(self, index):
        graph: TraceGraph = self.train_db.get(index)
        dgl_graph = graph_to_dgl(graph)

        # Phase 2: attach host_state if enabled
        if getattr(self.config.HostState, 'enable', False):
            try:
                host_ids = dgl_graph.ndata['host_id'].detach().cpu().numpy()
                
                # === 1. 尝试使用预计算数据 ===
                precomputed_map = graph.data.get('precomputed_host_state', None)
                
                if precomputed_map is not None:
                    # 获取维度 (尝试从第一个向量获取，或者根据配置计算)
                    if len(precomputed_map) > 0:
                        in_dim = next(iter(precomputed_map.values())).shape[0]
                    else:
                        # 兜底维度计算
                        metrics = list(getattr(self.config.HostState, 'metrics', DEFAULT_METRICS))
                        if getattr(self.config.HostState, 'include_disk', False):
                             for m in DISK_METRICS:
                                 if m not in metrics: metrics.append(m)
                        per_dims = int(getattr(self.config.HostState, 'per_metric_dims', 3))
                        in_dim = len(metrics) * per_dims

                    hs = np.zeros((host_ids.shape[0], in_dim), dtype=np.float32)
                    for i, hid in enumerate(host_ids):
                        hid_int = int(hid)
                        if hid_int in precomputed_map:
                            hs[i, :] = precomputed_map[hid_int]
                    
                    dgl_graph.ndata['host_state'] = torch.from_numpy(hs).to(dgl_graph.ndata['latency'].device)
                
                # === 2. 如果没有预计算数据，回退到实时计算 (原始逻辑) ===
                else:
                    # 优先使用 root span 的 StartTime
                    try:
                        st = graph.root.spans[0].start_time if (graph.root and graph.root.spans) else None
                        if isinstance(st, (int, float)):
                            v = float(st)
                            t0_ms = int(v if v > 1e12 else v * 1000.0)
                        elif hasattr(st, 'timestamp'):
                            t0_ms = int(st.timestamp() * 1000.0)
                        else:
                            v = float(dgl_graph.ndata['start_time'].min().item())
                            t0_ms = int(v * 1000.0)
                    except Exception:
                        v = float(dgl_graph.ndata['start_time'].min().item())
                        t0_ms = int(v * 1000.0)
                    t0_min_ms = (t0_ms // 60000) * 60000
                    
                    metrics = list(getattr(self.config.HostState, 'metrics', DEFAULT_METRICS))
                    if getattr(self.config.HostState, 'include_disk', False):
                        for m in DISK_METRICS:
                            if m not in metrics:
                                metrics.append(m)
                    per_dims = int(getattr(self.config.HostState, 'per_metric_dims', 3))
                    in_dim = len(metrics) * per_dims
                    
                    hid_to_vec: Dict[int, np.ndarray] = {}
                    for hid in set(map(int, host_ids.tolist())):
                        if hid <= 0: continue
                        try:
                            hname = self.id_manager.host_id.rev(int(hid))
                        except Exception:
                            hname = None
                        vec = host_state_vector(hname, self.infra_index, t0_min_ms, metrics=metrics, W=int(self.config.HostState.W), per_metric_dims=per_dims) if hname else None
                        if vec is None:
                            vec = np.zeros((in_dim,), dtype=np.float32)
                        hid_to_vec[int(hid)] = vec
                        
                    hs = np.zeros((host_ids.shape[0], in_dim), dtype=np.float32)
                    for i, hid in enumerate(host_ids):
                        vec = hid_to_vec.get(int(hid), None)
                        if vec is None: continue
                        hs[i, :] = vec
                    dgl_graph.ndata['host_state'] = torch.from_numpy(hs).to(dgl_graph.ndata['latency'].device)
            except Exception:
                pass

        # Phase 3 disabled on training side: no one-hop host-topology aggregation

        # Host sequence attachment for OmniAnomaly backend (graph-level meta)
        try:
            self._attach_host_seq(dgl_graph, graph)
        except Exception:
            pass

        return dgl_graph

    # ---------------------- helpers ----------------------
    def _attach_host_seq(self, dgl_graph, graph):
        """Attach per-host sequences [W,D] for OmniAnomaly backend as g.host_seq.
        Uses config.HostChannel.seq_window and seq_metrics.
        """
        # === [新增优化] 优先检查是否有预计算数据 ===
        # 如果 graph.data 里已经有算好的 Tensor，直接拿来用
        precomputed_seq = graph.data.get('precomputed_host_seq', None)
        if precomputed_seq is not None:
            # 直接挂载到 dgl_graph 上，无需任何计算
            try:
                dgl_graph.host_seq = precomputed_seq
            except Exception:
                # 兼容旧版 DGL 写法
                if not hasattr(dgl_graph, 'graph_data'):
                    dgl_graph.graph_data = {}
                dgl_graph.graph_data['host_seq'] = precomputed_seq
            return
        
        try:
            hc_cfg = getattr(self.config, 'HostChannel', None)
            if not hc_cfg or not bool(getattr(hc_cfg, 'enable', False)):
                return
        except Exception:
            return
        if getattr(self, 'infra_index_seq', None) is None:
            return

        import torch as _torch
        import numpy as _np

        try:
            st = graph.root.spans[0].start_time if (graph.root and graph.root.spans) else None
            if isinstance(st, (int, float)):
                v = float(st)
                t0_ms = int(v if v > 1e12 else v * 1000.0)
            elif hasattr(st, 'timestamp'):
                t0_ms = int(st.timestamp() * 1000.0)
            else:
                v = float(dgl_graph.ndata['start_time'].min().item())
                t0_ms = int(v * 1000.0)
        except Exception:
            v = float(dgl_graph.ndata['start_time'].min().item())
            t0_ms = int(v * 1000.0)
        t0_min = (t0_ms // 60000) * 60000

        host_ids = dgl_graph.ndata['host_id'].detach().cpu().numpy().astype(int)
        uniq_hosts = sorted({int(h) for h in host_ids if int(h) > 0})
        if not uniq_hosts:
            return

        def _map_metric(alias: str) -> str:
            alias = str(alias).lower().strip()
            if alias in ('cpu',):
                return 'node_cpu_usage_rate'
            if alias in ('mem', 'memory'):
                return 'node_memory_usage_rate'
            if alias in ('fs', 'filesystem'):
                return 'node_filesystem_usage_rate'
            if alias in ('disk_read',):
                return 'node_disk_read_time_seconds_total'
            if alias in ('disk_write',):
                return 'node_disk_write_time_seconds_total'
            return alias

        W = int(getattr(hc_cfg, 'seq_window', 15))
        aliases = list(getattr(hc_cfg, 'seq_metrics', ['cpu', 'mem', 'fs']))
        metrics_cols = [_map_metric(a) for a in aliases]

        def _robust_norm(x: _np.ndarray) -> _np.ndarray:
            a = x.astype(_np.float64)
            med = _np.nanmedian(a)
            q1, q3 = _np.nanpercentile(a, 25), _np.nanpercentile(a, 75)
            iqr = q3 - q1
            stdv = _np.nanstd(a)
            denom = iqr if (iqr is not None and iqr > 1e-6) else (stdv if stdv > 1e-6 else 1.0)
            z = (a - med) / denom
            z = _np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            return z

        host_seq: Dict[int, _torch.Tensor] = {}
        idx_seq = getattr(self, 'infra_index_seq', None)
        if idx_seq:
            for hid in uniq_hosts:
                try:
                    hname = self.id_manager.host_id.rev(int(hid))
                except Exception:
                    hname = None
                if not hname:
                    continue
                rec = idx_seq.get(str(hname))
                if not rec:
                    continue
                t_arr = _np.asarray(rec.get('timeMs', []), dtype=_np.int64)
                if t_arr.size == 0:
                    continue
                per_metric = []
                for mcol in metrics_cols:
                    vals = _np.asarray(rec.get('metrics', {}).get(mcol, []), dtype=_np.float64)
                    if vals.size != t_arr.size:
                        per_metric.append(_np.zeros((W,), dtype=_np.float32))
                        continue
                    seq_vals = []
                    for k in range(W):
                        target = t0_min - (W - 1 - k) * 60000
                        pos = int(_np.searchsorted(t_arr, target, side='right')) - 1
                        if pos >= 0:
                            seq_vals.append(float(vals[pos]))
                        else:
                            seq_vals.append(_np.nan)
                    seq_vals = _np.array(seq_vals, dtype=_np.float64)
                    seq_vals = _robust_norm(seq_vals)
                    per_metric.append(seq_vals.astype(_np.float32))
                if per_metric:
                    try:
                        mat = _np.stack(per_metric, axis=1)  # [W, D]
                        host_seq[int(hid)] = _torch.from_numpy(mat)
                    except Exception:
                        pass

        try:
            dgl_graph.host_seq = host_seq
        except Exception:
            try:
                if not hasattr(dgl_graph, 'graph_data'):
                    dgl_graph.graph_data = {}
                dgl_graph.graph_data['host_seq'] = host_seq
            except Exception:
                pass


class TestDataset(DGLDataset):
    def __init__(self, config: ExpConfig, test_path: str=None):
        self.config = config

        # Load id_manager and basic_info
        processed_dir = os.path.join(config.dataset_root_dir, config.dataset, 'processed')
        self.id_manager = TraceGraphIDManager(processed_dir)
        # Initialize DatasetParams and RuntimeInfo for evaluation as well
        try:
            init_config(config)
        except Exception:
            # keep evaluating even if init_config partially fails
            pass
        test_path = test_path if test_path is not None else os.path.join(processed_dir, config.test_dataset)
        self.test_db = TraceGraphDB(BytesSqliteDB(test_path))
        # load infra index once (optional)
        try:
            self.infra_index = load_host_infra_index(processed_dir)
        except Exception:
            self.infra_index = None
        # load host topology once (optional)
        try:
            self.host_adj = load_host_topology(processed_dir, self.id_manager)
        except Exception:
            self.host_adj = {}
        # load infra index for sequence (OmniAnomaly backend)
        # try:
        #     self.infra_index_seq = load_infra_for_seq(processed_dir)
        # except Exception:
        #     self.infra_index_seq = None
        self.infra_index_seq = None # 直接设为 None，强制只用预计算数据

        # Show info
        logger.info(f'{len(self.test_db)} in test dataset.')

    def process(self):
        pass

    def __len__(self):
        return len(self.test_db)

    def __getitem__(self, index):
        # graph1: TraceGraph
        # graph2: TraceGraph

        # 加载成对的图
        # graph1, graph2 = self.test_db.get(index)
        # dgl_graph1, dgl_graph2 = graph_to_dgl(graph1), graph_to_dgl(graph2)

        # 加载单个图
        graph = self.test_db.get(index)
        dgl_graph = graph_to_dgl(graph)
        # Attach original trace identifier if available for downstream diagnostics
        try:
            dgl_graph.trace_id = getattr(graph, 'trace_id', None)
        except Exception:
            pass

        # Phase 2: attach host_state if enabled (mirror train dataset)
        if getattr(self.config.HostState, 'enable', False):
            try:
                host_ids = dgl_graph.ndata['host_id'].detach().cpu().numpy()
                
                # === [新增] 1. 尝试使用预计算数据 (优化点) ===
                precomputed_map = graph.data.get('precomputed_host_state', None)
                
                if precomputed_map is not None:
                    # 维度推断逻辑
                    if len(precomputed_map) > 0:
                        in_dim = next(iter(precomputed_map.values())).shape[0]
                    else:
                        metrics = list(getattr(self.config.HostState, 'metrics', DEFAULT_METRICS))
                        if getattr(self.config.HostState, 'include_disk', False):
                             for m in DISK_METRICS:
                                 if m not in metrics: metrics.append(m)
                        per_dims = int(getattr(self.config.HostState, 'per_metric_dims', 3))
                        in_dim = len(metrics) * per_dims

                    hs = np.zeros((host_ids.shape[0], in_dim), dtype=np.float32)
                    for i, hid in enumerate(host_ids):
                        hid_int = int(hid)
                        if hid_int in precomputed_map:
                            hs[i, :] = precomputed_map[hid_int]
                    
                    dgl_graph.ndata['host_state'] = torch.from_numpy(hs).to(dgl_graph.ndata['latency'].device)
                
                # === 2. 如果没有预计算数据，回退到实时计算 (原始逻辑) ===
                else:
                    # 优先 root span StartTime；兜底用节点最小 start_time
                    try:
                        st = graph.root.spans[0].start_time if (graph.root and graph.root.spans) else None
                        if isinstance(st, (int, float)):
                            v = float(st)
                            t0_ms = int(v if v > 1e12 else v * 1000.0)
                        elif hasattr(st, 'timestamp'):
                            t0_ms = int(st.timestamp() * 1000.0)
                        else:
                            v = float(dgl_graph.ndata['start_time'].min().item())
                            t0_ms = int(v * 1000.0)
                    except Exception:
                        v = float(dgl_graph.ndata['start_time'].min().item())
                        t0_ms = int(v * 1000.0)
                    t0_min_ms = (t0_ms // 60000) * 60000
                    
                    metrics = list(getattr(self.config.HostState, 'metrics', DEFAULT_METRICS))
                    if getattr(self.config.HostState, 'include_disk', False):
                        for m in DISK_METRICS:
                            if m not in metrics:
                                metrics.append(m)
                    per_dims = int(getattr(self.config.HostState, 'per_metric_dims', 3))
                    in_dim = len(metrics) * per_dims
                    
                    hid_to_vec: Dict[int, np.ndarray] = {}
                    for hid in set(map(int, host_ids.tolist())):
                        if hid <= 0: continue
                        try:
                            hname = self.id_manager.host_id.rev(int(hid))
                        except Exception:
                            hname = None
                        vec = host_state_vector(hname, self.infra_index, t0_min_ms, metrics=metrics, W=int(self.config.HostState.W), per_metric_dims=per_dims) if hname else None
                        if vec is None:
                            vec = np.zeros((in_dim,), dtype=np.float32)
                        hid_to_vec[int(hid)] = vec
                        
                    hs = np.zeros((host_ids.shape[0], in_dim), dtype=np.float32)
                    for i, hid in enumerate(host_ids):
                        vec = hid_to_vec.get(int(hid), None)
                        if vec is None: continue
                        hs[i, :] = vec
                    dgl_graph.ndata['host_state'] = torch.from_numpy(hs).to(dgl_graph.ndata['latency'].device)
            except Exception:
                pass

        # Host sequence attachment for OmniAnomaly backend (graph-level meta)
        try:
            self._attach_host_seq(dgl_graph, graph)
        except Exception:
            pass

        # Set label of graph (one-hot label for 0 - drop, 1 - latency)
        # graph_label = torch.zeros([2], dtype=torch.bool)
        # if graph1.anomaly == 1:
        #     graph_label[0] = True
        # if graph2.anomaly == 2:
        #     graph_label[1] = True

        # 返回图和相关属性用于计算准确率
        # 返回图、标签和原始 trace_id（字符串形式，便于稳定导出）
        try:
            tid_val = getattr(graph, 'trace_id', None)
            trace_ident = str(tid_val) if tid_val is not None else None
        except Exception:
            trace_ident = None
        return dgl_graph, graph.anomaly, graph.root_cause, graph.fault_category, trace_ident
        # return dgl_graph, graph.anomaly


class DetectionDataset(DGLDataset):
    def __init__(self, config: ExpConfig, test_path: str=None):
        self.config = config

        # Load id_manager and basic_info
        processed_dir = os.path.join(config.dataset_root_dir, config.dataset, 'processed')
        self.id_manager = TraceGraphIDManager(processed_dir)
        test_path = test_path if test_path is not None else os.path.join(processed_dir, config.test_dataset)
        self.test_db = TraceGraphDB(BytesSqliteDB(test_path))

        # Get valid list
        self.valid_list: List[int] = []

        g: TraceGraph
        for i, g in enumerate(tqdm(self.test_db, desc='Get Valid List')):
            for _, nd in g.iter_bfs():
                if nd.operation_id >= config.DatasetParams.operation_cnt or \
                   nd.service_id >= config.DatasetParams.service_cnt or \
                   nd.status_id >= config.DatasetParams.status_cnt:
                   break
            else:
                self.valid_list.append(i)

    def process(self):
        pass

    def __len__(self):
        return len(self.valid_list)

    def __getitem__(self, index):
        graph: TraceGraph

        index = self.valid_list[index]
        graph = self.test_db.get(index)
        dgl_graph = graph_to_dgl(graph)

        return dgl_graph, torch.tensor(graph.trace_id[0], dtype=torch.int64), torch.tensor(graph.trace_id[1], dtype=torch.int64)

    # ---------------------- helpers ----------------------
    def _attach_host_seq(self, dgl_graph, graph):
        """Attach per-host sequences [W,D] for OmniAnomaly backend as g.host_seq.
        Uses config.HostChannel.seq_window and seq_metrics.
        """
        # Early exit if HostChannel disabled or no infra index for sequences
        try:
            hc_cfg = getattr(self.config, 'HostChannel', None)
            if not hc_cfg or not bool(getattr(hc_cfg, 'enable', False)):
                return
        except Exception:
            return
        if getattr(self, 'infra_index_seq', None) is None:
            return

        import torch as _torch
        import numpy as _np

        # t0 aligned to minute (ms)
        try:
            st = graph.root.spans[0].start_time if (graph.root and graph.root.spans) else None
            if isinstance(st, (int, float)):
                v = float(st)
                t0_ms = int(v if v > 1e12 else v * 1000.0)
            elif hasattr(st, 'timestamp'):
                t0_ms = int(st.timestamp() * 1000.0)
            else:
                v = float(dgl_graph.ndata['start_time'].min().item())
                t0_ms = int(v * 1000.0)
        except Exception:
            v = float(dgl_graph.ndata['start_time'].min().item())
            t0_ms = int(v * 1000.0)
        t0_min = (t0_ms // 60000) * 60000

        # host list
        host_ids = dgl_graph.ndata['host_id'].detach().cpu().numpy().astype(int)
        uniq_hosts = sorted({int(h) for h in host_ids if int(h) > 0})
        if not uniq_hosts:
            return

        # metric alias mapping
        def _map_metric(alias: str) -> str:
            alias = str(alias).lower().strip()
            if alias in ('cpu',):
                return 'node_cpu_usage_rate'
            if alias in ('mem', 'memory'):
                return 'node_memory_usage_rate'
            if alias in ('fs', 'filesystem'):
                return 'node_filesystem_usage_rate'
            if alias in ('disk_read',):
                return 'node_disk_read_time_seconds_total'
            if alias in ('disk_write',):
                return 'node_disk_write_time_seconds_total'
            return alias

        W = int(getattr(hc_cfg, 'seq_window', 15))
        aliases = list(getattr(hc_cfg, 'seq_metrics', ['cpu', 'mem', 'fs']))
        metrics_cols = [_map_metric(a) for a in aliases]

        def _robust_norm(x: _np.ndarray) -> _np.ndarray:
            a = x.astype(_np.float64)
            med = _np.nanmedian(a)
            q1, q3 = _np.nanpercentile(a, 25), _np.nanpercentile(a, 75)
            iqr = q3 - q1
            stdv = _np.nanstd(a)
            denom = iqr if (iqr is not None and iqr > 1e-6) else (stdv if stdv > 1e-6 else 1.0)
            z = (a - med) / denom
            z = _np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            return z

        host_seq: Dict[int, _torch.Tensor] = {}
        idx_seq = getattr(self, 'infra_index_seq', None)
        if idx_seq:
            for hid in uniq_hosts:
                try:
                    hname = self.id_manager.host_id.rev(int(hid))
                except Exception:
                    hname = None
                if not hname:
                    continue
                rec = idx_seq.get(str(hname))
                if not rec:
                    continue
                t_arr = _np.asarray(rec.get('timeMs', []), dtype=_np.int64)
                if t_arr.size == 0:
                    continue
                per_metric = []
                for mcol in metrics_cols:
                    vals = _np.asarray(rec.get('metrics', {}).get(mcol, []), dtype=_np.float64)
                    if vals.size != t_arr.size:
                        per_metric.append(_np.zeros((W,), dtype=_np.float32))
                        continue
                    seq_vals = []
                    for k in range(W):
                        target = t0_min - (W - 1 - k) * 60000
                        pos = int(_np.searchsorted(t_arr, target, side='right')) - 1
                        if pos >= 0:
                            seq_vals.append(float(vals[pos]))
                        else:
                            seq_vals.append(_np.nan)
                    seq_vals = _np.array(seq_vals, dtype=_np.float64)
                    seq_vals = _robust_norm(seq_vals)
                    per_metric.append(seq_vals.astype(_np.float32))
                if per_metric:
                    try:
                        mat = _np.stack(per_metric, axis=1)  # [W, D]
                        host_seq[int(hid)] = _torch.from_numpy(mat)
                    except Exception:
                        pass

        # attach to graph
        try:
            dgl_graph.host_seq = host_seq
        except Exception:
            try:
                if not hasattr(dgl_graph, 'graph_data'):
                    dgl_graph.graph_data = {}
                dgl_graph.graph_data['host_seq'] = host_seq
            except Exception:
                pass
