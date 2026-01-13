#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import csv
import time
import argparse
import ast
from datetime import datetime
import pandas as pd  # 核心：引入 pandas 进行数据聚合
import numpy as np

import config

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ================= 🔧 1. 在这里定义你需要的“精准指标列表” =================
TARGET_METRICS = [
    # --- 网络关键指标 ---
    "aggregate_node_net_receive_packages_errors_per_minute", # 核心：网络错包
    "aggregate_node_tcp_inuse_total_num",                    # 核心：TCP连接数
    "aggregate_node_tcp_alloc_total_num",
    
    # --- 基础资源 ---
    "aggregate_node_cpu_usage",
    "aggregate_node_memory_usage",
    "aggregate_node_disk_io_usage"
]
# =======================================================================

# 鉴权配置
os.environ.setdefault("ALIBABA_CLOUD_ROLE_SESSION_NAME", "my-sls-access")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tools.paas_entity_tools import umodel_get_entities
from tools.paas_data_tools import umodel_get_golden_metrics
from tools.common import create_cms_client, execute_cms_query
from tools.constants import REGION_ID, WORKSPACE_NAME

class BatchCustomMetricFetcher:
    def __init__(self, csv_path, output_dir=None, unified_mode=False, interval=None):
        self.csv_path = csv_path
        self.unified_mode = unified_mode
        self.interval = interval # 目标时间间隔（秒）
        self.client = create_cms_client(REGION_ID)
        
        # 确定输出目录
        if output_dir:
            self.data_dir = output_dir
        else:
            self.data_dir = os.path.join(project_root, "output_datasets" if unified_mode else "data")
            
        os.makedirs(self.data_dir, exist_ok=True)

        if self.unified_mode:
            self.global_csv_path = os.path.join(self.data_dir, "all_metrics.csv")
            self.global_headers = ['problem_id', 'fault_type', 'instance_id', 'timestamp'] + sorted(TARGET_METRICS)
            
            if not os.path.exists(self.global_csv_path):
                with open(self.global_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.global_headers)
                    writer.writeheader()
            print(f"✅ [统一模式] 结果将追加至: {self.global_csv_path}")
            if self.interval:
                print(f"⏱️  [重采样] 已启用数据聚合: 每 {self.interval} 秒一条")
        else:
            print(f"✅ [分散模式] 结果将保存至: {self.data_dir}/problem_XXX/")

    def _parse_time(self, t_str):
        if not t_str: return int(time.time())
        try:
            return int(datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S').timestamp())
        except ValueError:
            try:
                return int(t_str)
            except:
                return int(time.time())

    def fetch_metrics_for_problem(self, row):
        problem_id = row['problem_id']
        fault_type = row.get('fault_type', 'unknown')
        start_ts = self._parse_time(row['start_time']) - 180 # 提前3分钟，防止数据缺失
        end_ts = self._parse_time(row['end_time'])
        
        print(f"\n🚀 [Problem {problem_id}] 处理中... ({row['start_time']} ~ {row['end_time']})")

        # 1. 查找活跃 ECS
        entity_query = {
            "domain": "acs",
            "entity_set_name": "acs.ecs.instance",
            "from_time": start_ts,
            "to_time": end_ts,
            "limit": 100 
        }
        nodes_result = umodel_get_entities.invoke(entity_query)
        
        if not nodes_result or not nodes_result.data:
            print(f"   ⚠️ 此时段未发现活跃 ECS 节点")
            return 0, 0

        nodes = nodes_result.data
        print(f"   🔍 云端查询到活跃节点: {len(nodes)} 个")

        valid_node_metrics = {}
        
        # 2. 获取指标
        for node in nodes:
            node_name = node.get('instance_id')
            entity_id = node.get('__entity_id__')
            if not entity_id: continue

            node_data = {}

            # 策略 A: Golden Metrics
            gm_res = umodel_get_golden_metrics.invoke({
                "domain": "acs",
                "entity_set_name": "acs.ecs.instance",
                "entity_ids": [entity_id],
                "from_time": start_ts,
                "to_time": end_ts
            })
            
            if gm_res and gm_res.data:
                for item in gm_res.data:
                    m_name = item.get('metric')
                    if m_name in TARGET_METRICS:
                        self._extract_value(item, m_name, node_data)

            # 策略 B: 补缺
            missing = [m for m in TARGET_METRICS if m not in node_data]
            if missing:
                for m in missing:
                    query = f".entity_set with(domain='acs', name='acs.ecs.instance', ids=['{entity_id}']) | entity-call get_metric('{m}')"
                    try:
                        res = execute_cms_query(self.client, WORKSPACE_NAME, query, start_ts, end_ts)
                        if res and res.data:
                            vals, ts = [], []
                            for r in res.data:
                                v = r.get('value') or r.get(m)
                                t = r.get('timestamp') or r.get('ts')
                                if v is not None: vals.append(v); ts.append(t)
                            if vals:
                                node_data[m] = {"values": vals, "timestamps": ts}
                    except:
                        pass

            if node_data:
                valid_node_metrics[node_name] = node_data

        if not valid_node_metrics:
            print("   ⚠️ 未获取到任何有效指标数据")
            return 0, 0

        # 🔥【关键】在这里调用重采样
        if self.interval and self.interval > 0:
            valid_node_metrics = self._resample_data(valid_node_metrics, self.interval)

        # 统计
        batch_records = 0
        print(f"   📉 节点数据详情" + (f" (已聚合至 {self.interval}s)" if self.interval else "") + ":")
        for nid, m_data in valid_node_metrics.items():
            unique_ts = set()
            for metrics in m_data.values():
                unique_ts.update(metrics.get('timestamps', []))
            cnt = len(unique_ts)
            batch_records += cnt
            print(f"      🔹 节点 {nid:<20}: 获取 {cnt:>4} 条记录")

        # 3. 保存
        if self.unified_mode:
            self._append_to_global_csv(problem_id, fault_type, valid_node_metrics)
        else:
            self._save_separate_files(problem_id, valid_node_metrics)
            
        return len(valid_node_metrics), batch_records

    def _resample_data(self, node_metrics, interval_sec):
        """核心函数：使用 Pandas 将原始数据重采样为指定间隔"""
        resampled_metrics = {}
        
        for node_id, metrics_dict in node_metrics.items():
            df_all = pd.DataFrame()
            
            for metric_name, data in metrics_dict.items():
                ts_list = data.get('timestamps', [])
                val_list = data.get('values', [])
                
                if not ts_list: continue
                
                # 创建临时 DF
                df_temp = pd.DataFrame({'ts': ts_list, metric_name: val_list})
                # 纳秒转 datetime
                df_temp['ts'] = pd.to_datetime(df_temp['ts'], unit='ns')
                df_temp.set_index('ts', inplace=True)
                
                # Outer Join 合并
                if df_all.empty:
                    df_all = df_temp
                else:
                    df_all = df_all.join(df_temp, how='outer')
            
            if df_all.empty: continue
            
            # 重采样 (取平均值)
            # 🔥 [修正点] 使用小写 's' 替代大写 'S' 以消除 FutureWarning
            df_resampled = df_all.resample(f'{interval_sec}s').mean()
            
            # 还原为字典
            node_resul = {}
            new_timestamps = df_resampled.index.astype(np.int64).tolist()
            
            for col in df_resampled.columns:
                vals = df_resampled[col].fillna(0.0).tolist()
                node_resul[col] = {
                    "values": vals,
                    "timestamps": new_timestamps
                }
            
            resampled_metrics[node_id] = node_resul
            
        return resampled_metrics

    def _save_separate_files(self, problem_id, data):
        output_dir = os.path.join(self.data_dir, f"problem_{problem_id}")
        os.makedirs(output_dir, exist_ok=True)
        # JSON 里的 int64 可能会在某些查看器报错，但Python读取没问题
        with open(os.path.join(output_dir, "custom_ecs_metrics.json"), 'w', encoding='utf-8') as f:
            # default=str 用于处理 numpy 类型
            json.dump(data, f, indent=2, ensure_ascii=False, default=str) 
        self._save_as_csv(output_dir, "custom_ecs_metrics.csv", data)
        print(f"   ✅ 已保存至 {output_dir}")

    def _append_to_global_csv(self, problem_id, fault_type, data):
        rows_to_write = []
        for instance_id, metrics in data.items():
            time_map = {}
            for metric_name, metric_data in metrics.items():
                values = metric_data.get('values', [])
                timestamps = metric_data.get('timestamps', [])
                for v, t in zip(values, timestamps):
                    if t not in time_map:
                        time_map[t] = {
                            'problem_id': problem_id,
                            'fault_type': fault_type,
                            'instance_id': instance_id,
                            'timestamp': t
                        }
                    time_map[t][metric_name] = v
            for ts in sorted(time_map.keys()):
                row = time_map[ts]
                for m in TARGET_METRICS:
                    if m not in row: row[m] = "" 
                rows_to_write.append(row)
        if rows_to_write:
            try:
                with open(self.global_csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.global_headers)
                    writer.writerows(rows_to_write)
            except Exception as e:
                print(f"   ❌ 写入失败: {e}")

    def _save_as_csv(self, output_dir, filename, data):
        if not data: return
        all_metrics = set()
        for instance_data in data.values():
            all_metrics.update(instance_data.keys())
        headers = ['instance_id', 'timestamp'] + sorted(list(all_metrics))
        rows = []
        for instance_id, metrics in data.items():
            time_map = {}
            for metric_name, metric_data in metrics.items():
                values = metric_data.get('values', [])
                timestamps = metric_data.get('timestamps', [])
                for v, t in zip(values, timestamps):
                    if t not in time_map:
                        time_map[t] = {'instance_id': instance_id, 'timestamp': t}
                    time_map[t][metric_name] = v
            for ts in sorted(time_map.keys()):
                rows.append(time_map[ts])
        try:
            with open(os.path.join(output_dir, filename), 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            print(f"   ❌ CSV 保存失败: {e}")

    def _extract_value(self, item, metric_name, target_dict):
        try:
            vals = ast.literal_eval(item.get('__value__', '[]'))
            ts = ast.literal_eval(item.get('__ts__', '[]'))
            if vals:
                target_dict[metric_name] = {"values": vals, "timestamps": ts}
        except:
            pass

    def run(self):
        print(f"📂 读取任务列表: {self.csv_path}")
        if not os.path.exists(self.csv_path):
            print("❌ CSV 文件不存在")
            return
        total_problems, total_nodes, total_records = 0, 0, 0
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f"📋 共发现 {len(rows)} 个问题待处理...")
            start_time = time.time()
            for row in rows:
                try:
                    n_nodes, n_records = self.fetch_metrics_for_problem(row)
                    if n_nodes > 0:
                        total_problems += 1; total_nodes += n_nodes; total_records += n_records
                except Exception as e:
                    print(f"❌ 处理 Problem {row.get('problem_id')} 时出错: {e}")
            end_time = time.time()
            print("\n" + "="*50 + f"\n📊 执行完成总结 report\n" + "="*50)
            print(f"⏱️  总耗时       : {end_time - start_time:.2f} 秒")
            print(f"✅ 成功处理问题 : {total_problems} 个")
            print(f"💻 涉及节点总数 : {total_nodes} 个")
            print(f"📈 获取数据记录 : {total_records} 条")
            if self.unified_mode:
                print(f"💾 结果文件     : {self.global_csv_path}")
            else:
                print(f"💾 结果目录     : {self.data_dir}")
            print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset/b_gt.csv", help="路径指向 b_gt.csv")
    parser.add_argument("--unified", action="store_true", help="启用统一模式")
    parser.add_argument("--output-dir", default="data/NodeMetric", help="自定义输出目录")
    
    # 🔥 新增参数
    parser.add_argument("--interval", type=int, default=30, help="重采样时间间隔(秒)，例如 60 表示每分钟一条")
    
    args = parser.parse_args()
    fetcher = BatchCustomMetricFetcher(args.csv, args.output_dir, args.unified, args.interval)
    fetcher.run()