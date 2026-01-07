# -*- coding: utf-8 -*-
"""
CSV 故障分布统计工具
功能：读取提取好的 Trace CSV，统计不同故障类型的样本数量 (按 TraceID 去重)
"""

import csv
import argparse
import os
import collections
from prettytable import PrettyTable  # 如果没有安装，脚本会自动退化为普通打印

def count_distribution(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        return

    print(f"📂 正在读取文件: {csv_path} ...")
    
    # 计数器
    # 1. fault_type -> 唯一的 TraceID 集合
    type_stats = collections.defaultdict(set)
    # 2. problem_id -> 唯一的 TraceID 集合
    pid_stats = collections.defaultdict(set)
    
    total_rows = 0
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # 检查表头
            headers = reader.fieldnames
            if 'fault_type' not in headers or 'TraceID' not in headers:
                print("❌ CSV 缺少必要字段: fault_type 或 TraceID")
                return

            for row in reader:
                total_rows += 1
                tid = row['TraceID']
                ftype = row.get('fault_type', 'unknown')
                pid = row.get('problem_id', 'unknown')
                
                # 记录 (自动去重)
                if tid:
                    type_stats[ftype].add(tid)
                    pid_stats[pid].add(tid)
                    
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return

    # === 输出报告 ===
    print(f"\n✅ 读取完成. 总行数 (Spans): {total_rows}")
    print("\n" + "="*50)
    print("📊 故障类型分布 (按 Trace 去重)")
    print("="*50)
    
    # 尝试使用 PrettyTable 美化输出
    try:
        pt = PrettyTable()
        pt.field_names = ["Fault Type", "Trace Count", "Percentage"]
        pt.align["Fault Type"] = "l"
        pt.align["Trace Count"] = "r"
        pt.align["Percentage"] = "r"
    except ImportError:
        pt = None

    # 计算总 Trace 数
    total_traces = sum(len(s) for s in type_stats.values())
    
    # 排序输出
    sorted_types = sorted(type_stats.items(), key=lambda x: len(x[1]), reverse=True)
    
    for ftype, tids in sorted_types:
        count = len(tids)
        percent = (count / total_traces * 100) if total_traces > 0 else 0
        
        if pt:
            pt.add_row([ftype, count, f"{percent:.1f}%"])
        else:
            print(f"   🔹 {ftype:<25}: {count:>5} ({percent:.1f}%)")
            
    if pt: print(pt)
    
    print("-" * 50)
    print(f"   ∑ 总计 Trace 样本数       : {total_traces}")
    print("="*50)

    # (可选) 输出按 Problem ID 的统计
    # print("\n📋 按 Problem ID 统计 (Top 10):")
    # sorted_pids = sorted(pid_stats.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    # for pid, tids in sorted_pids:
    #     print(f"   Problem {pid:<4}: {len(tids)} Traces")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dir = "NodeFault/all_fault_traces.csv"
    # 默认读取刚才脚本生成的文件名
    # parser.add_argument("--csv", default="NodeFault/trace_node_faults_verified.csv", help="要统计的 CSV 文件路径")
    parser.add_argument("--csv", default=dir, help="要统计的 CSV 文件路径")
    args = parser.parse_args()
    
    count_distribution(args.csv)