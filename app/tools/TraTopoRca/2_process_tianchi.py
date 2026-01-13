import os
import argparse
import shutil
import sqlite3
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# 引入项目依赖
from tracegnn.data.trace_graph import df_to_trace_graphs, TraceGraphIDManager
from tracegnn.data.trace_graph_db import TraceGraphDB, BytesSqliteDB
from tracegnn.utils.host_state import host_state_vector

# ================= 配置区域 =================
DEFAULT_DATASET_ROOT = 'dataset/tianchi/0113' 
# 指标文件名，请确保这与你 3_allfault_nodeMetric.py 生成的文件名一致
NORMAL_METRIC_FILE = 'normal_metrics_1e5_30s.csv'  # 用于 Train/Val 集
FAULT_METRIC_FILE  = 'all_metrics_30s.csv'     # 用于 Test 集

# Host Sequence 配置 (需与 dataset.py / config.py 保持一致)
SEQ_WINDOW = 15
# [修改] 扩充序列指标别名，以包含网络和磁盘
SEQ_METRICS = ['cpu', 'mem', 'disk', 'net', 'tcp'] 

# [新增] 天池数据的真实指标列名 (请确保与 CSV 表头一致)
TIANCHI_METRICS = [
    "aggregate_node_cpu_usage",
    "aggregate_node_memory_usage",
    "aggregate_node_disk_io_usage",
    "aggregate_node_net_receive_packages_errors_per_minute",
    "aggregate_node_tcp_alloc_total_num",
    "aggregate_node_tcp_inuse_total_num"
]
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

def load_infra_data_from_parent(dataset_root: str, filename: str):
    parent_dir = os.path.dirname(dataset_root.rstrip(os.path.sep))
    
    # [修改] 使用传入的 filename 构建路径
    infra_path = os.path.join(dataset_root, filename)
    
    # 定义查找路径列表 (按优先级排序)
    paths_to_try = [
        # 1. 当前 dataset_root 下
        os.path.join(dataset_root, filename),
        
        # 2. [新增] 直接在父目录下查找 (e.g. dataset/tianchi/normal_metric.csv)
        os.path.join(parent_dir, filename),
        
        # 3. 在 dataset_root/data 下 (NormalData输出目录常见位置)
        os.path.join(dataset_root, 'data', filename),
        os.path.join(dataset_root, 'NormalData', filename),
        
        # 4. 在 parent_dir/data 下
        os.path.join(parent_dir, 'data', filename),
        os.path.join(parent_dir, 'infra', filename),
    ]

    infra_path = None
    for p in paths_to_try:
        if os.path.exists(p):
            infra_path = p
            break

    if not infra_path:
        print(f"⚠️ 警告: 未找到指标数据文件: {filename}")
        # print(f"   已尝试路径: {paths_to_try}") # 调试时可开启
        return None
    
    print(f"✅ 已加载指标数据: {infra_path}")
    try:
        df = pd.read_csv(infra_path)
        
        # [关键修改] 1. 检查关键列
        if 'timestamp' not in df.columns or 'instance_id' not in df.columns:
            print("❌ 错误: CSV缺少关键列 'timestamp' 或 'instance_id'，请检查文件格式。")
            return None

        # [关键修改] 2. 时间戳转换: 纳秒 (19位) -> 毫秒 (13位)
        # 天池 timestamp e.g., 1758036033000000000
        df['timeMs'] = df['timestamp'].astype(np.int64) // 1000000
        
        # [关键修改] 3. 节点ID映射: instance_id -> kubernetes_node
        # 这一步是为了让后续 GNN 代码能通过 'kubernetes_node' 这个标准字段找到数据
        df['kubernetes_node'] = df['instance_id'].astype(str)

        # 过滤需要的指标列，并填充缺失值
        metric_cols = [c for c in TIANCHI_METRICS if c in df.columns]
        for m in metric_cols:
            df[m] = pd.to_numeric(df[m], errors='coerce').fillna(0.0)
        
        # 构建索引 (按节点分组，按时间排序)
        # 只保留需要的列
        cols = ['timeMs', 'kubernetes_node'] + metric_cols
        df = df[cols].dropna(subset=['timeMs', 'kubernetes_node'])
        
        host_idx = {}
        for host, g in tqdm(df.groupby('kubernetes_node'), desc="构建指标索引"):
            lg = g.sort_values('timeMs')
            # 去重：防止同一毫秒有多条数据 (取最后一条)
            lg = lg.drop_duplicates(subset=['timeMs'], keep='last')
            
            host_idx[str(host)] = {
                'timeMs': lg['timeMs'].to_numpy(dtype=np.int64),
                'metrics': {m: lg[m].to_numpy(dtype=np.float64) for m in metric_cols}
            }
        return host_idx
    except Exception as e:
        print(f"解析指标数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# === 1. HostState 预计算 (GNN) ===
def precompute_host_states(trace_graphs, infra_index, id_manager, W=3):
    if infra_index is None: return
    
    # [修改] 强制使用天池指标列表作为 GNN 的输入特征
    metrics = TIANCHI_METRICS
    per_metric_dims = 4  # mean, std, max, min

    stats = {
        "total_graphs": len(trace_graphs),
        "processed_graphs": 0,
        "total_spans_with_host": 0,  # 有 HostID 的 Span 总数
        "failed_metric_spans": 0,    # 映射失败的 Span 总数
        "missing_hosts": set()       # 缺失数据的 Host 集合
    }

    for graph in tqdm(trace_graphs, desc="预计算 HostState (GNN)"):
        try:
            stats["processed_graphs"] += 1
            st = graph.root.spans[0].start_time if (graph.root and graph.root.spans) else None
            if isinstance(st, (int, float)):
                v = float(st)
                t0_ms = int(v if v > 1e12 else v * 1000.0)
            else:
                t0_ms = 0
            t0_min_ms = (t0_ms // 60000) * 60000
            
            # 找出该 Trace 中所有涉及的 Host 节点
            # iter_bfs 返回 (node_id, node_obj) 或者 node_obj，取决于具体实现
            # 这里假设 graph.iter_bfs() 返回 (_, node)
            nodes_in_graph = [node for _, node in graph.iter_bfs() if node.host_id and node.host_id > 0]
            
            host_ids = set(node.host_id for node in nodes_in_graph)
            host_state_map = {}
            
            for hid in host_ids:
                hname = id_manager.host_id.rev(int(hid))
                if hname:
                    # 尝试获取指标向量
                    vec = host_state_vector(hname, infra_index, t0_min_ms, metrics=metrics, W=W, per_metric_dims=per_metric_dims)
                    if vec is not None:
                        host_state_map[hid] = vec
                    else:
                        # [统计] 记录缺失数据的 Host
                        stats["missing_hosts"].add(hname)
            
            if host_state_map:
                graph.data['precomputed_host_state'] = host_state_map
            
            # [统计] 计算 Span 级别的缺失情况
            for node in nodes_in_graph:
                stats["total_spans_with_host"] += 1
                if node.host_id not in host_state_map:
                    stats["failed_metric_spans"] += 1

        except Exception:
            continue

    # [新增] 输出统计报告
    print("\n" + "="*50)
    print("📊 [GNN 特征映射统计报告]")
    print(f"   - 总图数: {stats['total_graphs']}")
    print(f"   - 涉及物理机的 Span 总数: {stats['total_spans_with_host']}")
    print(f"   - ❌ 指标映射失败 Span 数: {stats['failed_metric_spans']}")
    
    if stats['total_spans_with_host'] > 0:
        fail_rate = stats['failed_metric_spans'] / stats['total_spans_with_host'] * 100
        print(f"   - 📉 失败率: {fail_rate:.2f}%")
        if fail_rate > 0:
            print(f"   - ⚠️ 警告: 有 {fail_rate:.2f}% 的 Span 无法关联到性能指标！")
    
    if stats["missing_hosts"]:
        print(f"   - 缺失数据的 Host 数量: {len(stats['missing_hosts'])}")
        preview = list(stats['missing_hosts'])[:3]
        print(f"   - 缺失 Host 示例: {preview} ...")
        print("   -> 请检查 3_allfault_nodeMetric.py 的获取窗口是否足够覆盖这些 Trace 的时间点")
    else:
        print("   - ✅ 所有 Host 均成功匹配到指标数据")
    print("="*50 + "\n")

# === 2. HostSequence 预计算 (OmniAnomaly) ===
def precompute_host_sequences(trace_graphs, infra_index, id_manager):
    """预先计算用于 OmniAnomaly 的时间序列数据 [Window, Metrics]"""
    if infra_index is None: return

    # [修改] 映射配置里的别名 -> 天池真实列名
    def _map_metric(alias: str) -> str:
        alias = str(alias).lower().strip()
        
        # CPU
        if alias in ('cpu',): 
            return 'aggregate_node_cpu_usage'
            
        # Memory
        if alias in ('mem', 'memory'): 
            return 'aggregate_node_memory_usage'
            
        # Disk (对应 node diskchaos)
        if alias in ('fs', 'filesystem', 'disk', 'io'): 
            return 'aggregate_node_disk_io_usage'
            
        # Network (对应 node networkchaos, 主要是丢包/错包)
        if alias in ('net', 'network'): 
            return 'aggregate_node_net_receive_packages_errors_per_minute'
            
        # TCP (辅助网络特征)
        if alias in ('tcp',):
            return 'aggregate_node_tcp_inuse_total_num'
            
        # 如果没有匹配到，默认返回原名（以防直接使用了真实列名）
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
                    # 尝试获取指标数据，如果列名不对则返回空列表
                    vals = rec.get('metrics', {}).get(mcol, [])
                    if len(vals) == 0:
                        # 容错：如果指标不存在，填充全0序列
                        seq_vals_np = np.zeros(W, dtype=np.float64)
                    else:
                        seq_vals = []
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
                    # shape: [Window, Metrics] e.g. [15, 5]
                    mat = np.stack(per_metric, axis=1)
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
    print(f"🚀 开始处理数据流 (适配天池指标)，根目录: {dataset_root}")
    processed_root = os.path.join(dataset_root, 'processed')
    os.makedirs(processed_root, exist_ok=True)
    
    # 1. 加载指标数据
    print(f"\n[步骤 0/4] 加载指标文件...")
    normal_infra_index = load_infra_data_from_parent(dataset_root, NORMAL_METRIC_FILE)
    fault_infra_index = load_infra_data_from_parent(dataset_root, FAULT_METRIC_FILE)
    
    # 2. 建立 ID 映射
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

    # 3. 处理数据
    process_split('train', dataset_root, id_manager, normal_infra_index)
    process_split('val', dataset_root, id_manager, normal_infra_index)

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
        process_split('test', dataset_root, id_manager, fault_infra_index, processed_df=test_df)

    id_manager.dump_to(processed_root)
    print(f"\n✨ 所有处理完成！")

if __name__ == '__main__':
    main()