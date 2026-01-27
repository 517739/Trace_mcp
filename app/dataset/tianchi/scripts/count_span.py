import csv
import argparse
from collections import defaultdict

def count_trace_nodes(csv_file_path, trace_id_column="TraceID"):
    """
    统计CSV中每个TraceID的记录数，并按区间汇总
    
    Args:
        csv_file_path: CSV文件路径
        trace_id_column: TraceID所在的列名（默认是"TraceID"）
    """
    # 1. 初始化统计字典：key=TraceID，value=该TraceID的记录数
    trace_count = defaultdict(int)
    
    # 2. 读取CSV文件并统计每个TraceID的记录数
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            # 读取CSV表头，确认TraceID列存在
            csv_reader = csv.DictReader(f)
            headers = csv_reader.fieldnames
            
            if trace_id_column not in headers:
                print(f"❌ 错误：CSV文件中未找到列名 '{trace_id_column}'，请检查列名是否正确！")
                print(f"   当前CSV包含的列：{headers}")
                return
            
            # 遍历每一行，统计TraceID出现次数
            for row in csv_reader:
                trace_id = row[trace_id_column].strip()  # 去除首尾空格，避免空值/空格干扰
                if trace_id:  # 跳过空的TraceID
                    trace_count[trace_id] += 1
    
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 '{csv_file_path}'，请检查文件路径是否正确！")
        return
    except Exception as e:
        print(f"❌ 读取文件时发生错误：{str(e)}")
        return
    
    # 3. 定义需要统计的区间（可根据需求修改）
    # 格式：区间描述 -> 判定条件函数
    intervals = {
        "2个节点": lambda x: x == 2,
        "3个节点": lambda x: x == 3,
        "4个节点": lambda x: x == 4,
        "5个节点": lambda x: x == 5,
        "6~10个节点": lambda x: 6 <= x <= 10,
        "11~15个节点": lambda x: 11 <= x <= 15,  # 注意：原需求"10~15"易重复，调整为11~15避免重叠
        "15个节点以上": lambda x: x > 15
    }
    
    # 4. 按区间统计TraceID数量
    interval_result = defaultdict(int)
    # 先获取所有TraceID的节点数列表
    trace_node_nums = list(trace_count.values())
    
    for num in trace_node_nums:
        for interval_name, condition in intervals.items():
            if condition(num):
                interval_result[interval_name] += 1
                break  # 匹配到一个区间后跳出，避免重复统计
    
    # 5. 输出统计结果
    print("=" * 60)
    print(f"📊 统计结果（总计 {len(trace_count)} 个不同的TraceID）")
    print("=" * 60)
    
    # 按预设顺序输出区间统计（保证顺序和定义的一致）
    for interval_name in intervals.keys():
        count = interval_result.get(interval_name, 0)
        print(f"✅ {interval_name}：{count} 个TraceID")
    
    # 可选：输出前10个TraceID的详细统计（便于验证）
    print("\n🔍 前10个TraceID的详细记录数（验证用）：")
    sorted_trace = sorted(trace_count.items(), key=lambda x: x[1], reverse=True)[:10]
    for trace_id, count in sorted_trace:
        print(f"   TraceID {trace_id}：{count} 条记录")

if __name__ == "__main__":
    # 命令行参数解析（支持指定CSV路径和TraceID列名）
    parser = argparse.ArgumentParser(description="统计CSV中TraceID的节点数并按区间汇总")
    # parser.add_argument("--csv_file", default="/root/wzc/Trace_mcp/app/tools/TraTopoRca/dataset/tianchi/2e5_1622/raw/train.csv", help="CSV文件的路径（如：./trace_data.csv）")
    parser.add_argument("--csv_file", default="/root/wzc/tracezly_rca/tianchi_processed_data2.csv", help="CSV文件的路径（如：./trace_data.csv）")

    parser.add_argument("--column", "-c", default="TraceID", help="TraceID所在的列名（默认：TraceID）")
    args = parser.parse_args()
    
    # 执行统计
    count_trace_nodes(args.csv_file, args.column)