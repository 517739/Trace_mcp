# -*- coding: utf-8 -*-
"""
Trace 数据集格式转换工具
功能：将旧版/正常数据集 CSV 转换为新版标准格式
1. 列名重命名 (如 SpanID -> SpanId, StartTime -> StartTimeMs)
2. 列顺序调整 (对齐训练数据格式)
3. 自动填充空标签 (fault_type, fault_instance, problem_id)
"""

import csv
import argparse
import os
import sys

# 增加字段大小限制，防止某些超长 Trace 报错
csv.field_size_limit(2147483647)

# ================= 配置区域 =================

# 目标 CSV 表头 (严格对应训练脚本要求的格式)
TARGET_HEADERS = [
    'TraceID', 'SpanId', 'ParentID', 
    'ServiceName', 'NodeName', 'PodName', 
    'URL', 'SpanKind', 
    'StartTimeMs', 'EndTimeMs', 'DurationMs',
    'StatusCode', 'HttpStatusCode', 
    'fault_type', 'fault_instance', 'problem_id'
]

# 列名映射字典: { "源列名": "目标列名" }
# 如果源列名和目标一致，可以不写，但为了清晰建议写全
COLUMN_MAPPING = {
    'TraceID': 'TraceID',
    'SpanID': 'SpanId',        # 注意大小写变化: ID -> Id
    'ParentID': 'ParentID',
    'NodeName': 'NodeName',
    'ServiceName': 'ServiceName',
    'PodName': 'PodName',
    'URL': 'URL',
    'HttpStatusCode': 'HttpStatusCode',
    'StatusCode': 'StatusCode',
    'SpanKind': 'SpanKind',
    'StartTime': 'StartTimeMs', # 假设源数据已经是数值，只改名
    'EndTime': 'EndTimeMs',
    'Duration': 'DurationMs'
}

# ===========================================

def convert_csv(input_path, output_path):
    print(f"🚀 开始转换: {input_path} -> {output_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ 错误: 输入文件不存在: {input_path}")
        return

    success_count = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8', newline='') as f_in, \
             open(output_path, 'w', encoding='utf-8', newline='') as f_out:
            
            # 1. 读取源文件
            reader = csv.DictReader(f_in)
            
            # 检查源文件表头是否包含我们需要的所有关键字段
            # (这里只打印警告，不强制退出，防止源文件列名有细微差别)
            source_fields = reader.fieldnames
            print(f"   ℹ️ 源文件列名: {source_fields}")
            
            # 2. 初始化写入器
            writer = csv.DictWriter(f_out, fieldnames=TARGET_HEADERS)
            writer.writeheader()
            
            # 3. 逐行处理
            for row in reader:
                new_row = {}
                
                # A. 映射已有数据
                for src_col, target_col in COLUMN_MAPPING.items():
                    # get() 防止源文件缺列报错，默认空字符串
                    # strip() 去除可能存在的首尾空格
                    val = row.get(src_col, '').strip()
                    new_row[target_col] = val
                
                # B. 填充新标签 (正常集设为空或特定标识)
                # 您说后面几个都不用打标签，这里默认留空
                new_row['fault_type'] = ''      # 或者填 "normal"
                new_row['fault_instance'] = '' 
                new_row['problem_id'] = ''      # 或者填 "0"
                
                # C. 特殊处理 (可选)
                # 如果 StartTime 是纳秒(19位)，可能需要除以 1e6 转毫秒
                # 这里提供一个简单的自动转换逻辑示例，默认注释掉
                '''
                try:
                    s_ts = float(new_row['StartTimeMs'])
                    if s_ts > 1e16: # 可能是纳秒
                        new_row['StartTimeMs'] = f"{s_ts/1e6:.3f}"
                        new_row['EndTimeMs'] = f"{float(new_row['EndTimeMs'])/1e6:.3f}"
                        new_row['DurationMs'] = f"{float(new_row['DurationMs'])/1e6:.3f}"
                except:
                    pass
                '''

                # 写入
                writer.writerow(new_row)
                success_count += 1
                
                if success_count % 10000 == 0:
                    print(f"   ⏳ 已处理 {success_count} 行...", end='\r')

        print(f"\n✅ 转换完成! 共处理 {success_count} 条数据。")
        print(f"   💾 输出文件: {output_path}")

    except Exception as e:
        print(f"\n❌ 转换过程中发生错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace CSV 格式转换工具")
    # 默认文件名，您可以直接修改这里
    parser.add_argument("--input", default="E:\ZJU\AIOps\Projects\TraDNN\Trace_mcp/app/tools/trace_sv_diag\dataset/tianchi/row_old/Normal.csv", help="源 CSV 文件路径")
    parser.add_argument("--output", default="E:\ZJU\AIOps\Projects\TraDNN\Trace_mcp/app/tools/trace_sv_diag\dataset/tianchi/row/Normal.csv", help="输出 CSV 文件路径")
    
    args = parser.parse_args()
    
    convert_csv(args.input, args.output)