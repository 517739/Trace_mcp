import pandas as pd
import numpy as np
import os

# === 配置路径 ===
TRACE_FILE = 'dataset/tianchi/0112/raw/train.csv'
METRIC_FILE = 'dataset/tianchi/all_metrics_30s.csv'

def check_time_alignment():
    print(f"🔍 正在检查时间对齐情况...")
    
    # 1. 检查 Trace 时间
    if not os.path.exists(TRACE_FILE):
        print(f"❌ 找不到 Trace 文件: {TRACE_FILE}")
        return
    df_trace = pd.read_csv(TRACE_FILE, nrows=5)
    t_trace = df_trace['StartTimeMs'].iloc[0]
    print(f"\n[Trace 样本]")
    print(f"  原始 StartTimeMs: {t_trace} (类型: {type(t_trace)})")
    print(f"  转换后时间: {pd.to_datetime(t_trace, unit='ms')}")

    # 2. 检查 Metric 时间
    if not os.path.exists(METRIC_FILE):
        print(f"❌ 找不到 Metric 文件: {METRIC_FILE}")
        return
    df_metric = pd.read_csv(METRIC_FILE, nrows=5)
    t_metric = df_metric['timestamp'].iloc[0]
    print(f"\n[Metric 样本]")
    print(f"  原始 timestamp: {t_metric} (类型: {type(t_metric)})")
    
    # 模拟当前代码的转换逻辑
    t_metric_ms_converted = int(t_metric) // 1000000
    print(f"  代码转换逻辑 (ts // 1e6): {t_metric_ms_converted}")
    print(f"  转换后时间: {pd.to_datetime(t_metric_ms_converted, unit='ms')}")

    # 3. 诊断
    print(f"\n[诊断结论]")
    if abs(t_trace - t_metric_ms_converted) > 86400 * 1000 * 365: # 差一年以上
        print("❌ 时间戳单位严重不匹配！")
        if t_metric < 1e14: # 看起来像毫秒或秒
            print("  -> 指标文件里的 timestamp 似乎不是纳秒，但代码执行了 // 1000000。")
            print("  -> 请修改 2_process_tianchi.py，去掉除法，或调整除数。")
    else:
        print("✅ 时间戳单位看起来一致。")
        print("❓ 如果单位一致但匹配失败，说明 all_metrics_30s.csv 里没有包含 train.csv 所需的时间段。")
        print("   (Train集通常是正常数据，你需要确认 metric 文件里是否合并了正常时段的指标)")

if __name__ == '__main__':
    check_time_alignment()