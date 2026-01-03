import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import shutil
import sqlite3
import pandas as pd
import numpy as np
import torch # 需要引入 torch
from tqdm import tqdm

# 引入项目依赖
from tracegnn.data.trace_graph import df_to_trace_graphs, TraceGraphIDManager
from tracegnn.data.trace_graph_db import TraceGraphDB, BytesSqliteDB
from tracegnn.utils.host_state import host_state_vector, DEFAULT_METRICS, DISK_METRICS

# ================= 配置区域 =================
DEFAULT_DATASET_ROOT = 'dataset/dataset_topo' 
INFRA_FILENAME = 'merged_all_infra.csv'
# Host Sequence 配置 (需与 dataset.py / config.py 保持一致)
SEQ_WINDOW = 15
SEQ_METRICS = ['cpu', 'mem', 'fs'] # 默认使用的序列指标
# ===========================================

def flexible_load_trace_csv(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(input_path)
        if 'Duration' in df.columns: df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
        if 'StartTimeMs' in df.columns: df['StartTime'] = pd.to_numeric(df['StartTimeMs'], errors='coerce')
        if 'Anomaly' in df.columns: df['Anomaly'] = df['Anomaly'].astype(bool)
        return df
    except Exception as e:
        print(f"加载CSV出错 {input_path}: {e}")
        return pd.DataFrame()

def load_infra_data_from_parent(dataset_root: str):
    parent_dir = os.path.dirname(dataset_root.rstrip(os.path.sep))
    infra_path = os.path.join(parent_dir, INFRA_FILENAME)
    if not os.path.exists(infra_path):
        infra_path_alt = os.path.join(parent_dir, 'infra', INFRA_FILENAME)
        if os.path.exists(infra_path_alt): infra_path = infra_path_alt
    
    if not os.path.exists(infra_path):
        print(f"⚠️ 警告: 未找到指标数据文件。期望路径: {infra_path}")
        return None
    
    print(f"✅ 已加载指标数据: {infra_path}")
    try:
        df = pd.read_csv(infra_path)
        if 'timeMs' not in df.columns or 'kubernetes_node' not in df.columns: return None
        all_metrics = list(set(DEFAULT_METRICS + DISK_METRICS))
        for m in all_metrics:
            if m not in df.columns: df[m] = np.nan
        try:
            df['timeMs'] = df['timeMs'].astype(np.int64)
        except:
            if 'time' in df.columns: df['timeMs'] = pd.to_datetime(df['time']).astype('int64') // 10**6
        
        cols = ['timeMs', 'kubernetes_node'] + [c for c in all_metrics if c in df.columns]
        df = df[cols].dropna(subset=['timeMs', 'kubernetes_node'])
        host_idx = {}
        for host, g in df.groupby('kubernetes_node'):
            lg = g.sort_values('timeMs')
            host_idx[str(host)] = {
                'timeMs': lg['timeMs'].to_numpy(dtype=np.int64),
                'metrics': {m: lg[m].to_numpy(dtype=np.float64) for m in lg.columns if m not in ('timeMs', 'kubernetes_node')}
            }
        return host_idx
    except Exception as e:
        print(f"解析指标数据失败: {e}")
        return None

# === 1. HostState 预计算 (GNN) ===
def precompute_host_states(trace_graphs, infra_index, id_manager, W=3):
    if infra_index is None: return
    metrics = list(DEFAULT_METRICS)
    per_metric_dims = 4  # 确保这里是 4

    for graph in tqdm(trace_graphs, desc="预计算 HostState (GNN)"):
        try:
            st = graph.root.spans[0].start_time if (graph.root and graph.root.spans) else None
            if isinstance(st, (int, float)):
                v = float(st)
                t0_ms = int(v if v > 1e12 else v * 1000.0)
            else:
                t0_ms = 0
            t0_min_ms = (t0_ms // 60000) * 60000
            
            host_ids = set(node.host_id for _, node in graph.iter_bfs() if node.host_id and node.host_id > 0)
            host_state_map = {}
            for hid in host_ids:
                hname = id_manager.host_id.rev(int(hid))
                if hname:
                    vec = host_state_vector(hname, infra_index, t0_min_ms, metrics=metrics, W=W, per_metric_dims=per_metric_dims)
                    if vec is not None:
                        host_state_map[hid] = vec
            if host_state_map:
                graph.data['precomputed_host_state'] = host_state_map
        except Exception:
            continue

# === 2. [新增] HostSequence 预计算 (OmniAnomaly) ===
def precompute_host_sequences(trace_graphs, infra_index, id_manager):
    """预先计算用于 OmniAnomaly 的时间序列数据 [Window, Metrics]"""
    if infra_index is None: return

    # 映射配置里的别名到真实列名
    def _map_metric(alias: str) -> str:
        alias = str(alias).lower().strip()
        if alias in ('cpu',): return 'node_cpu_usage_rate'
        if alias in ('mem', 'memory'): return 'node_memory_usage_rate'
        if alias in ('fs', 'filesystem'): return 'node_filesystem_usage_rate'
        return alias
    
    metrics_cols = [_map_metric(a) for a in SEQ_METRICS]
    W = SEQ_WINDOW

    def _robust_norm(x):
        med = np.nanmedian(x)
        q1, q3 = np.nanpercentile(x, 25), np.nanpercentile(x, 75)
        iqr = q3 - q1
        stdv = np.nanstd(x)
        denom = iqr if (iqr is not None and iqr > 1e-6) else (stdv if stdv > 1e-6 else 1.0)
        z = (x - med) / denom
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    for graph in tqdm(trace_graphs, desc="预计算 HostSeq (OmniAnomaly)"):
        try:
            # 计算 t0 (分钟对齐)
            st = graph.root.spans[0].start_time if (graph.root and graph.root.spans) else None
            if isinstance(st, (int, float)):
                v = float(st)
                t0_ms = int(v if v > 1e12 else v * 1000.0)
            else:
                t0_ms = 0
            t0_min = (t0_ms // 60000) * 60000

            host_ids = set(node.host_id for _, node in graph.iter_bfs() if node.host_id and node.host_id > 0)
            host_seq_map = {}

            for hid in host_ids:
                hname = id_manager.host_id.rev(int(hid))
                if not hname: continue
                
                rec = infra_index.get(str(hname))
                if not rec: continue
                
                t_arr = rec.get('timeMs', [])
                if len(t_arr) == 0: continue
                
                per_metric = []
                for mcol in metrics_cols:
                    vals = rec.get('metrics', {}).get(mcol, [])
                    seq_vals = []
                    # 简单的滑动窗口提取
                    # 优化：使用 searchsorted 批量查找范围可能更快，但这里为了逻辑一致保持循环
                    # 考虑到是预处理，稍微慢点可以接受
                    for k in range(W):
                        target = t0_min - (W - 1 - k) * 60000
                        # 找到 <= target 的最后一个点
                        pos = int(np.searchsorted(t_arr, target, side='right')) - 1
                        if pos >= 0:
                            seq_vals.append(float(vals[pos]))
                        else:
                            seq_vals.append(np.nan)
                    
                    seq_vals_np = np.array(seq_vals, dtype=np.float64)
                    norm_vals = _robust_norm(seq_vals_np)
                    per_metric.append(norm_vals.astype(np.float32))
                
                if per_metric:
                    # shape: [Window, Metrics] e.g. [15, 3]
                    mat = np.stack(per_metric, axis=1)
                    # 存储为 Tensor 以便 dataset 直接使用
                    host_seq_map[int(hid)] = torch.from_numpy(mat)
            
            if host_seq_map:
                graph.data['precomputed_host_seq'] = host_seq_map

        except Exception:
            continue

def process_split(split_name, dataset_root, id_manager, infra_index, processed_df=None):
    raw_csv = os.path.join(dataset_root, 'raw', f'{split_name}.csv')
    out_dir = os.path.join(dataset_root, 'processed', split_name)
    if not os.path.exists(raw_csv) and processed_df is None: return

    print(f"\n=== 处理 {split_name} 集 ===")
    os.makedirs(out_dir, exist_ok=True)
    
    if processed_df is not None: df = processed_df
    else: df = flexible_load_trace_csv(raw_csv)

    if df.empty: return

    trace_graphs = df_to_trace_graphs(df=df, id_manager=id_manager, min_node_count=2, max_node_count=100, summary_file=None, merge_spans=False)
    if not trace_graphs: return

    # === 执行两项预计算 ===
    precompute_host_states(trace_graphs, infra_index, id_manager)    # GNN 用
    precompute_host_sequences(trace_graphs, infra_index, id_manager) # OmniAnomaly 用
    # ====================

    db_path = os.path.join(out_dir, "_bytes.db")
    if not os.path.exists(db_path): open(db_path, 'a').close()
    db = TraceGraphDB(BytesSqliteDB(out_dir, write=True))
    try:
        with db.write_batch():
            for graph in trace_graphs:
                if hasattr(graph, 'root_cause') and graph.root_cause is None: graph.root_cause = 0
                if hasattr(graph, 'fault_category') and graph.fault_category is None: graph.fault_category = 0
                db.add(graph)
        db.commit()
        print(f"  ✅ 成功写入 {len(trace_graphs)} 个图到 {split_name} 数据库")
    finally:
        db.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=DEFAULT_DATASET_ROOT)
    args = parser.parse_args()
    
    dataset_root = args.root
    print(f"🚀 开始处理数据流 (含 Seq 优化)，根目录: {dataset_root}")
    processed_root = os.path.join(dataset_root, 'processed')
    os.makedirs(processed_root, exist_ok=True)
    infra_index = load_infra_data_from_parent(dataset_root)
    
    # 建立 ID
    print("\n[步骤 1/4] 建立统一 ID 映射...")
    combined_dfs = []
    for split in ['train', 'val', 'test']:
        path = os.path.join(dataset_root, 'raw', f'{split}.csv')
        df = flexible_load_trace_csv(path)
        if not df.empty: combined_dfs.append(df)
    if not combined_dfs: return
    temp_id_dir = os.path.join(dataset_root, 'temp_ids')
    os.makedirs(temp_id_dir, exist_ok=True)
    id_manager = TraceGraphIDManager(temp_id_dir)
    with id_manager:
        full_df = pd.concat(combined_dfs, ignore_index=True)
        for row in tqdm(full_df.itertuples(), total=len(full_df), desc="生成 ID"):
            id_manager.service_id.get_or_assign(getattr(row, 'ServiceName', '') or '')
            id_manager.operation_id.get_or_assign(getattr(row, 'OperationName', '') or '')
            id_manager.status_id.get_or_assign(str(getattr(row, 'StatusCode', '')) or '')
    id_manager.dump_to(processed_root)
    id_manager = TraceGraphIDManager(processed_root)
    if os.path.exists(temp_id_dir): shutil.rmtree(temp_id_dir)

    # 处理数据
    process_split('train', dataset_root, id_manager, infra_index)
    process_split('val', dataset_root, id_manager, infra_index)

    print("\n[步骤 3/4] 处理测试集...")
    test_csv_path = os.path.join(dataset_root, 'raw', 'test.csv')
    test_df = flexible_load_trace_csv(test_csv_path)
    if not test_df.empty:
        for col in ['RootCause', 'FaultCategory']:
            if col not in test_df.columns: test_df[col] = ''
        for idx, row in test_df.iterrows():
            if row.get('Anomaly'):
                rc_text = str(row.get('RootCause', '')).strip()
                fc_text = str(row.get('FaultCategory', '')).strip()
                mapped_id = None
                if fc_text.lower().startswith('node'):
                    rc_text = rc_text.replace('_', '-')
                    mapped_id = id_manager.host_id.get(rc_text)
                else:
                    rc_svc = rc_text.split('-')[0] if '-' in rc_text else rc_text
                    mapped_id = id_manager.service_id.get(rc_svc)
                test_df.at[idx, 'RootCause'] = mapped_id if mapped_id is not None else 0
                test_df.at[idx, 'FaultCategory'] = id_manager.fault_category.get_or_assign(fc_text) if fc_text else 0
        process_split('test', dataset_root, id_manager, infra_index, processed_df=test_df)

    id_manager.dump_to(processed_root)
    print(f"\n✨ 所有处理完成！")

if __name__ == '__main__':
    main()