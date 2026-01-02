import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

def parse_csv(file_path: str) -> Dict[str, int]:
    """
    解析CSV文件，统计每个TraceID对应的记录数
    
    Args:
        file_path: CSV文件路径
        
    Returns:
        字典，key为TraceID，value为该Trace的记录数
    """
    trace_count = defaultdict(int)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            # 创建CSV阅读器，自动处理表头
            reader = csv.DictReader(csvfile)
            
            # 验证必要的列是否存在
            required_columns = {'TraceID', 'SpanId'}
            if not required_columns.issubset(reader.fieldnames):
                missing = required_columns - set(reader.fieldnames)
                raise ValueError(f"CSV文件缺少必要列：{missing}")
            
            # 统计每个TraceID的记录数
            for row in reader:
                trace_id = row['TraceID'].strip()
                if trace_id:  # 跳过空的TraceID
                    trace_count[trace_id] += 1
                    
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {file_path} 不存在")
    except Exception as e:
        raise Exception(f"解析CSV文件时出错：{str(e)}")
    
    return dict(trace_count)

def calculate_interval_distribution(
    trace_counts: Dict[str, int],
    intervals: List[Tuple[int, int]] = None
) -> Dict[str, int]:
    """
    按节点数区间统计Trace数量
    
    Args:
        trace_counts: 每个TraceID的记录数字典
        intervals: 自定义区间列表，格式为[(最小值, 最大值), ...]
        
    Returns:
        字典，key为区间描述，value为该区间内的Trace数量
    """
    # 默认区间配置
    if intervals is None:
        intervals = [
            (1, 1),          # 1个节点
            (2, 5),          # 2-5个节点
            (6, 10),         # 6-10个节点
            (11, 20),        # 11-20个节点
            (21, 50),        # 21-50个节点
            (51, 100),       # 51-100个节点
            (101, float('inf'))  # 100个以上节点
        ]
    
    # 初始化区间统计结果
    distribution = {}
    for min_val, max_val in intervals:
        if max_val == float('inf'):
            key = f">{min_val - 1}"
        elif min_val == max_val:
            key = f"{min_val}"
        else:
            key = f"{min_val}-{max_val}"
        distribution[key] = 0
    
    # 统计每个区间的Trace数量
    for count in trace_counts.values():
        for min_val, max_val in intervals:
            if min_val <= count <= max_val:
                if max_val == float('inf'):
                    key = f">{min_val - 1}"
                elif min_val == max_val:
                    key = f"{min_val}"
                else:
                    key = f"{min_val}-{max_val}"
                distribution[key] += 1
                break
    
    return distribution

def print_report(
    trace_counts: Dict[str, int],
    distribution: Dict[str, int]
) -> None:
    """
    打印统计报告
    
    Args:
        trace_counts: 每个TraceID的记录数字典
        distribution: 区间分布统计结果
    """
    print("=" * 60)
    print("TraceID 节点数统计报告")
    print("=" * 60)
    
    # 基础统计信息
    total_traces = len(trace_counts)
    total_records = sum(trace_counts.values())
    print(f"总Trace数: {total_traces}")
    print(f"总记录数: {total_records}")
    print(f"平均每个Trace的节点数: {total_records/total_traces:.2f}")
    print()
    
    # 区间分布
    print("节点数区间分布:")
    print("-" * 40)
    for interval, count in sorted(distribution.items(), key=lambda x: 
        (int(x[0].split('-')[0]) if '-' in x[0] else int(x[0].replace('>', '')))
    ):
        percentage = (count / total_traces) * 100 if total_traces > 0 else 0
        print(f"{interval:10s} | {count:6d} 个Trace | {percentage:6.2f}%")
    
    # Top 10 Trace（节点数最多）
    print()
    print("节点数最多的前10个TraceID:")
    print("-" * 40)
    top_traces = sorted(trace_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (trace_id, count) in enumerate(top_traces, 1):
        print(f"{i:2d}. {trace_id:20s} | {count:6d} 个节点")

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='统计CSV文件中TraceID的节点数分布')
    parser.add_argument('--csv_file', default="E:\ZJU\AIOps\Projects\TraDNN\dataset/aiops25\processed/2025-06-06.csv", help='输入的CSV文件路径')
    parser.add_argument(
        '--intervals', 
        help='自定义节点数区间（示例：1-1,2-5,6-10,101-∞）',
        default=None
    )
    args = parser.parse_args()
    
    # 处理自定义区间
    custom_intervals = None
    if args.intervals:
        custom_intervals = []
        for interval in args.intervals.split(','):
            interval = interval.strip()
            if '-' in interval:
                min_val, max_val = interval.split('-')
                min_val = int(min_val)
                if max_val == '∞' or max_val == 'inf':
                    max_val = float('inf')
                else:
                    max_val = int(max_val)
                custom_intervals.append((min_val, max_val))
            else:
                # 单个数值的区间
                val = int(interval)
                custom_intervals.append((val, val))
    
    try:
        # 解析CSV文件
        trace_counts = parse_csv(args.csv_file)
        
        if not trace_counts:
            print("未找到有效的TraceID记录")
            return
        
        # 计算区间分布
        distribution = calculate_interval_distribution(trace_counts, custom_intervals)
        
        # 打印报告
        print_report(trace_counts, distribution)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()