import csv
import argparse
from collections import defaultdict

def count_multi_root_traces(csv_file_path, trace_col="TraceID", span_col="SpanID", parent_col="ParentID"):
    """
    统计CSV中存在多根节点的Trace数量（根节点定义：ParentID=-1 或 ParentID在当前Trace中无匹配的SpanID）
    
    Args:
        csv_file_path: CSV文件路径
        trace_col: TraceID列名（默认TraceID）
        span_col: SpanID列名（默认SpanID）
        parent_col: ParentID列名（默认ParentID）
    """
    # 步骤1：先读取所有数据，按TraceID分组存储（SpanID和ParentID）
    trace_data = defaultdict(list)  # key=TraceID, value=[(SpanID, ParentID), ...]
    valid_traces = set()

    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            headers = csv_reader.fieldnames
            
            # 检查必要列是否存在
            required_cols = [trace_col, span_col, parent_col]
            missing_cols = [col for col in required_cols if col not in headers]
            if missing_cols:
                print(f"❌ 错误：CSV文件中缺失必要列 → {missing_cols}")
                print(f"   当前CSV包含的列：{headers}")
                return
            
            # 遍历所有行，按TraceID分组
            row_num = 0
            for row in csv_reader:
                row_num += 1
                # 提取并清洗字段
                trace_id = row[trace_col].strip() if row[trace_col] is not None else ""
                span_id = row[span_col].strip() if row[span_col] is not None else ""
                parent_id = row[parent_col].strip() if row[parent_col] is not None else ""
                
                # 跳过空TraceID或空SpanID的行（无效数据）
                if not trace_id:
                    print(f"⚠️  警告：第{row_num}行TraceID为空，已跳过")
                    continue
                if not span_id:
                    print(f"⚠️  警告：第{row_num}行SpanID为空（TraceID={trace_id}），已跳过")
                    continue
                
                trace_data[trace_id].append((span_id, parent_id))
                valid_traces.add(trace_id)

    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 '{csv_file_path}'，请检查路径！")
        return
    except Exception as e:
        print(f"❌ 读取文件时发生错误：{str(e)}")
        return

    # 步骤2：遍历每个Trace，统计根节点数量
    trace_root_count = defaultdict(int)
    multi_root_traces = []

    for trace_id in valid_traces:
        spans = trace_data[trace_id]
        # 提取当前Trace下的所有SpanID（用于判断ParentID是否存在）
        span_ids_in_trace = {span_id for span_id, _ in spans}
        root_count = 0

        # 遍历当前Trace的每个Span，判断是否为根节点
        for span_id, parent_id in spans:
            # 根节点判定规则：ParentID=-1 或 ParentID不在当前Trace的SpanID列表中
            if parent_id == "-1" or parent_id not in span_ids_in_trace:
                root_count += 1
        
        trace_root_count[trace_id] = root_count
        # 根节点数≥2则判定为多根Trace
        if root_count >= 2:
            multi_root_traces.append(trace_id)

    # 步骤3：输出统计结果
    total_trace_count = len(valid_traces)
    multi_root_count = len(multi_root_traces)

    print("=" * 70)
    print(f"📊 Trace多根节点统计结果（根节点定义：ParentID=-1 或 ParentID无匹配SpanID）")
    print("=" * 70)
    print(f"📈 总计有效Trace数量：{total_trace_count}")
    print(f"🔴 存在多根节点的Trace数量：{multi_root_count}")
    print(f"📐 多根Trace占比：{multi_root_count/total_trace_count*100:.2f}%" if total_trace_count > 0 else "📐 多根Trace占比：0.00%")
    
    # 可选：输出前10个多根Trace的根节点数（便于验证）
    if multi_root_count > 0:
        print("\n🔍 前10个多根Trace的根节点数（验证用）：")
        # 按根节点数降序排序
        sorted_multi_root = sorted(multi_root_traces, key=lambda x: trace_root_count[x], reverse=True)[:10]
        for tid in sorted_multi_root:
            print(f"   TraceID {tid}：{trace_root_count[tid]} 个根节点")

    return {
        "total_traces": total_trace_count,
        "multi_root_traces": multi_root_count,
        "multi_root_trace_list": multi_root_traces
    }

if __name__ == "__main__":
    # 命令行参数解析（支持自定义列名和默认CSV路径）
    parser = argparse.ArgumentParser(description="统计CSV中存在多根节点的Trace数量")
    # 改为可选参数，设置默认CSV路径，支持直接运行
    parser.add_argument("-f", "--file", 
                        default="/root/wzc/tracezly_rca/tianchi_processed_data2.csv",
                        dest="csv_file",
                        help="CSV文件路径（默认：normal_traces_2e5_1622_mapped.csv）")
    parser.add_argument("--trace-col", 
                        default="TraceID",
                        help="TraceID列名（默认：TraceID）")
    parser.add_argument("--span-col", 
                        default="SpanID",
                        help="SpanID列名（默认：SpanID）")
    parser.add_argument("--parent-col", 
                        default="ParentID",
                        help="ParentID列名（默认：ParentID）")
    
    args = parser.parse_args()
    
    # 执行统计
    count_multi_root_traces(
        csv_file_path=args.csv_file,
        trace_col=args.trace_col,
        span_col=args.span_col,
        parent_col=args.parent_col
    )