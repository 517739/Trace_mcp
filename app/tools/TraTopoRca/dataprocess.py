import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import random
from tracegnn.data.trace_graph import load_trace_csv, df_to_trace_graphs, TraceGraphIDManager
from tracegnn.data.trace_graph_db import TraceGraphDB, BytesSqliteDB
import pandas as pd

def precompute_host_states(trace_graphs, output_dir, id_manager, W=3):
    """在数据处理阶段预先计算 Host State 向量并存储在 graph.data 中"""
    print(f"正在尝试加载基础设施指标数据 (Output Dir: {output_dir})...")
    infra_index = load_host_infra_index(output_dir)
    
    if infra_index is None:
        print(f"警告: 未找到基础设施指标数据，跳过 SInfra 预计算。")
        return

    metrics = list(DEFAULT_METRICS)
    # for m in DISK_METRICS: # 如需开启磁盘指标请取消注释
    #     if m not in metrics: metrics.append(m)
    per_metric_dims = 4

    success_count = 0
    for graph in tqdm(trace_graphs, desc="Pre-computing HostStates"):
        try:
            st = graph.root.spans[0].start_time if (graph.root and graph.root.spans) else None
            if isinstance(st, (int, float)):
                v = float(st)
                t0_ms = int(v if v > 1e12 else v * 1000.0)
            elif hasattr(st, 'timestamp'):
                t0_ms = int(st.timestamp() * 1000.0)
            else:
                t0_ms = 0
            t0_min_ms = (t0_ms // 60000) * 60000
        except:
            continue

        host_ids = set(node.host_id for _, node in graph.iter_bfs() if node.host_id and node.host_id > 0)
        host_state_map = {}
        for hid in host_ids:
            try:
                hname = id_manager.host_id.rev(int(hid))
                if hname:
                    vec = host_state_vector(hname, infra_index, t0_min_ms, metrics=metrics, W=W, per_metric_dims=per_metric_dims)
                    if vec is not None:
                        host_state_map[hid] = vec
            except:
                pass
        
        if host_state_map:
            graph.data['precomputed_host_state'] = host_state_map
            success_count += 1
    print(f"完成 SInfra 预计算: {success_count}/{len(trace_graphs)} 个 Trace 成功关联指标数据")

def convert_csv_to_db(csv_file: str, output_dir: str, min_node_count: int = 2, max_node_count: int = 100):
    """
    将CSV文件转换为DB格式并存储
    
    Args:
        csv_file: 输入的CSV文件路径
        output_dir: 输出DB文件的目录
        min_node_count: 最小节点数阈值
        max_node_count: 最大节点数阈值
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载CSV文件
    print(f"正在加载CSV文件: {csv_file}")
    df = load_trace_csv(csv_file)
    print(f"成功加载 {len(df)} 行数据")
    
    # 创建ID管理器
    id_manager = TraceGraphIDManager(output_dir)
    
    # 将DataFrame转换为TraceGraph列表
    print("正在将数据转换为图结构...")
    trace_graphs = df_to_trace_graphs(
        df=df,
        id_manager=id_manager,
        min_node_count=min_node_count,
        max_node_count=max_node_count,
        summary_file=os.path.join(output_dir, 'conversion_summary.txt'),
        merge_spans=False
    )

    if trace_graphs is None:
        trace_graphs = []
    print(f"成功转换 {len(trace_graphs)} 个图")
    precompute_host_states(trace_graphs, output_dir, id_manager)

    # 创建数据库（直接传完整路径）
    db_path = os.path.join(output_dir, "_bytes.db")
    db = TraceGraphDB(BytesSqliteDB(db_path, write=True))
    

    # 将图数据写入数据库
    print(f"正在写入数据库到: {db_path}")
    try:
        with db.write_batch():
            for graph in trace_graphs:
                db.add(graph)
        db.commit()
        print(f"成功写入 {len(trace_graphs)} 条记录到数据库")
    finally:
        db.close()
    
    # 保存ID映射文件
    id_manager.dump_to(output_dir)
    print("已完成ID映射文件的保存")


def split_and_convert_csv(csv_file: str, output_root: str, 
                         train_ratio: float = 0.7, 
                         val_ratio: float = 0.1, 
                         test_ratio: float = 0.2,
                         min_node_count: int = 2, 
                         max_node_count: int = 100,
                         random_seed: int = 42):
    """
    将CSV文件按比例划分为训练集、验证集和测试集，并分别转换为DB格式保存
    """
    # 检查比例是否正确
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("训练集、验证集和测试集比例之和必须等于1")
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 创建输出根目录
    os.makedirs(output_root, exist_ok=True)
    
    # 加载CSV文件
    print(f"正在加载CSV文件: {csv_file}")
    df = load_trace_csv(csv_file)
    print(f"成功加载 {len(df)} 行数据")
    
    # 按TraceID进行分组
    trace_groups = df.groupby('TraceID')
    trace_ids = list(trace_groups.groups.keys())
    print(f"总共包含 {len(trace_ids)} 个不同的Trace")
    
    # 打乱TraceID顺序
    random.shuffle(trace_ids)
    
    # 计算划分索引
    total_count = len(trace_ids)
    train_end = int(total_count * train_ratio)
    val_end = int(total_count * (train_ratio + val_ratio))
    
    # 按TraceID划分数据集
    train_trace_ids = trace_ids[:train_end]
    val_trace_ids = trace_ids[train_end:val_end]
    test_trace_ids = trace_ids[val_end:]
    
    # 根据TraceID获取对应的行数据
    train_df = df[df['TraceID'].isin(train_trace_ids)]
    val_df = df[df['TraceID'].isin(val_trace_ids)]
    test_df = df[df['TraceID'].isin(test_trace_ids)]
    
    print(f"数据集划分完成:")
    print(f"  训练集: {len(train_df)} 条记录")
    print(f"  验证集: {len(val_df)} 条记录")
    print(f"  测试集: {len(test_df)} 条记录")
    
    # 为每个数据集创建目录并转换
    datasets = [
        ("train", train_df, os.path.join(output_root, "train")),
        ("val", val_df, os.path.join(output_root, "val")),
        ("test", test_df, os.path.join(output_root, "test"))
    ]
    
    # 使用第一个数据集（训练集）创建ID管理器
    train_dir = os.path.join(output_root, "train")
    id_manager = TraceGraphIDManager(train_dir)
    
    for name, df_subset, output_dir in datasets:
        print(f"\n正在处理{name}数据集...")
        os.makedirs(output_dir, exist_ok=True)
        
        # 将DataFrame转换为TraceGraph列表
        trace_graphs = df_to_trace_graphs(
            df=df_subset,
            id_manager=id_manager,  # 使用共享的ID管理器
            min_node_count=min_node_count,
            max_node_count=max_node_count,
            summary_file=os.path.join(output_dir, f'{name}_conversion_summary.txt'),
            merge_spans=False
        )
        
        if trace_graphs is None:
            trace_graphs = []
        print(f"{name}数据集成功转换 {len(trace_graphs)} 个图")

        precompute_host_states(trace_graphs, output_dir, id_manager)
        
        if len(trace_graphs) > 0:
            # 创建数据库（直接传完整路径）
            db_file_name = "_bytes.db"
            db = TraceGraphDB(BytesSqliteDB(output_dir, write=True))

            
            # 写入数据
            print(f"正在将{name}数据集写入数据库...")
            try:
                with db.write_batch():
                    for graph in trace_graphs:
                        db.add(graph)
                db.commit()
                print(f"成功写入 {len(trace_graphs)} 条记录到{name}数据库")
            finally:
                db.close()
    
    id_manager.dump_to(output_root)
    print(f"已完成ID映射文件的保存: {output_root}")


def process_aiops_dataset(raw_csv: str = "dataset/aiops/raw/data.csv", 
                         processed_dir: str = "dataset/aiops/processed",
                         train_ratio: float = 0.7, 
                         val_ratio: float = 0.1, 
                         test_ratio: float = 0.2,
                         min_node_count: int = 2, 
                         max_node_count: int = 100,
                         random_seed: int = 42):
    """
    处理aiops数据集，将原始CSV数据按比例划分并保存到processed目录
    """
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"原始CSV文件不存在: {raw_csv}")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"处理aiops数据集:")
    print(f"  原始文件: {raw_csv}")
    print(f"  输出目录: {processed_dir}")
    
    split_and_convert_csv(
        csv_file=raw_csv,
        output_root=processed_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_node_count=min_node_count,
        max_node_count=max_node_count,
        random_seed=random_seed
    )
    
    print(f"\n数据处理完成，文件已保存到: {processed_dir}")


def main():
    parser = argparse.ArgumentParser(description='将CSV文件转换为DB格式')
    parser.add_argument('csv_file', nargs='?', default='dataset/dataset_a/raw/2025-06-06_labeled_filtered.csv', help='输入的CSV文件路径')
    parser.add_argument('output_dir', nargs='?', default='dataset/dataset_a/processed', help='输出DB文件的目录')
    parser.add_argument('--min-node-count', type=int, default=2, help='最小节点数阈值 (默认: 2)')
    parser.add_argument('--max-node-count', type=int, default=100, help='最大节点数阈值 (默认: 100)')
    parser.add_argument('--split', action='store_true', help='是否按比例划分数据集')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='训练集比例 (默认: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例 (默认: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='测试集比例 (默认: 0.2)')
    parser.add_argument('--random-seed', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--aiops', action='store_true', help='处理aiops数据集 (默认路径)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"错误: 输入文件 {args.csv_file} 不存在")
        return
    
    if args.aiops:
        process_aiops_dataset(
            raw_csv=args.csv_file,
            processed_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            min_node_count=args.min_node_count,
            max_node_count=args.max_node_count,
            random_seed=args.random_seed
        )
    elif args.split:
        split_and_convert_csv(
            csv_file=args.csv_file,
            output_root=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            min_node_count=args.min_node_count,
            max_node_count=args.max_node_count,
            random_seed=args.random_seed
        )
    else:
        convert_csv_to_db(
            args.csv_file,
            args.output_dir,
            args.min_node_count,
            args.max_node_count
        )


if __name__ == '__main__':
    main()
