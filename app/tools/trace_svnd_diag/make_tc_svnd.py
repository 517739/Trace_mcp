# scripts/make_tc_svnd_complete.py
# -*- coding: utf-8 -*-
"""
Tianchi 数据集构造工具 (终极修复版)
修复说明：
1. 强制在 Core Feature 阶段注入 y_c3 和 fault_type，解决 KeyError。
2. 完善了去重聚合(Reduce)逻辑，确保标签不丢失。
3. 增强了对 NaN 和空数据的鲁棒性。
4. 包含 Orphan Span (1个悬浮根节点) 的保留逻辑。
"""

import os
import json
import argparse
import random
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================= 🔧 基础工具 =================

def http_bucket(code):
    try:
        c = int(code)
    except:
        return "other"
    if 200 <= c < 300: return "2xx"
    if 300 <= c < 400: return "3xx"
    if 400 <= c < 500: return "4xx"
    if 500 <= c < 600: return "5xx"
    return "other"

def url_template(u: str) -> str:
    if not isinstance(u, str): return "NA"
    core = u.split("?")[0].split("#")[0]
    if "://" in core: core = core.split("://")[1]
    core = core.rstrip("/")
    return core or "NA"

def make_api_key(service: str, url_tmpl: str) -> str:
    s = str(service) if pd.notna(service) else "NA_SVC"
    t = str(url_tmpl) if pd.notna(url_tmpl) else "NA_URL"
    return f"{s}||{t}"

# ================= 🚀 数据清洗与加载 =================

def load_and_clean(path, tag):
    print(f"📖 [{tag}] 读取文件: {path}")
    if not os.path.exists(path):
        print(f"   ⚠️ 文件不存在: {path}")
        return pd.DataFrame()

    dtypes = {
        "TraceID": str, "SpanId": str, "ParentID": str, 
        "ServiceName": str, "NodeName": str, 
        "StatusCode": str, "HttpStatusCode": str,
        "fault_type": str
    }
    
    try:
        df = pd.read_csv(path, dtype=dtypes, low_memory=False)
    except Exception as e:
        print(f"   ❌ 读取失败: {e}")
        return pd.DataFrame()

    df['__set__'] = tag
    
    # 填充空值
    for c in ["NodeName", "ServiceName", "URL", "fault_type"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
        else:
            df[c] = "" 

    for c in ["StartTimeMs", "EndTimeMs", "DurationMs"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    
    return df

def drop_orphan_traces(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Step 1.5] 删除断链的 Trace (允许 1 个根节点悬浮)
    """
    if df.empty: return df
    
    print(f"   🧹 正在检查 Trace 完整性 (允许 1 个根节点悬浮)...")
    valid_indices = []
    explicit_root_pids = {"", "nan", "None", "null", "-1", "0"}
    
    grouped = df.groupby("TraceID", sort=False)
    total_traces = 0
    dropped_traces = 0
    
    for tid, g in grouped:
        total_traces += 1
        span_ids = set(g["SpanId"])
        dangling_count = 0
        
        for pid in g["ParentID"]:
            pid_str = str(pid).strip()
            if pid_str in span_ids: continue
            if pid_str in explicit_root_pids: continue
            dangling_count += 1
        
        if dangling_count <= 1:
            valid_indices.extend(g.index)
        else:
            dropped_traces += 1
            
    if len(valid_indices) == len(df):
        print(f"      ✨ 所有 Trace 均结构完整。")
        return df
    
    print(f"      丢弃了 {dropped_traces} 条破碎 Trace (保留率: {100 - dropped_traces/total_traces*100:.2f}%)")
    return df.loc[valid_indices].reset_index(drop=True)

# ================= 📊 核心特征计算 =================

def per_trace_core(df_t: pd.DataFrame) -> dict:
    """计算 Trace 级的统计特征"""
    n = len(df_t)
    durs = df_t["DurationMs"].values
    
    valid_nodes = df_t[df_t["NodeName"] != ""]
    if not valid_nodes.empty:
        node_dur = valid_nodes.groupby("NodeName")["DurationMs"].sum()
        dominant_node = node_dur.idxmax() if not node_dur.empty else ""
    else:
        dominant_node = ""

    http_codes = pd.to_numeric(df_t["HttpStatusCode"], errors='coerce').fillna(0).values
    span_codes = pd.to_numeric(df_t["StatusCode"], errors='coerce').fillna(0).values
    is_err = (http_codes >= 500) | (span_codes != 0)
    
    err_rate = is_err.mean() if n > 0 else 0.0
    _5xx_frac = (http_codes >= 500).mean() if n > 0 else 0.0

    t0 = df_t["StartTimeMs"].min()
    t1 = df_t["EndTimeMs"].max()
    tmid = (t0 + t1) / 2.0 if n > 0 else 0.0

    return {
        "TraceID": df_t["TraceID"].iloc[0],
        "dominant_node": dominant_node,
        "trace_t0": t0, "trace_t1": t1, "trace_tmid": tmid,
        "span_dur_p90": np.percentile(durs, 90) if n > 0 else 0.0,
        "err_rate": err_rate,
        "_5xx_frac": _5xx_frac,
        "svc_unique": df_t["ServiceName"].nunique(),
    }

def build_window_context_fast(df_core, win_minutes=3.0):
    """[Step 3] 计算节点时间窗口上下文"""
    if df_core.empty: return df_core

    print("⏳ [3/5] 计算节点上下文 (Context)...")
    
    ctx_cols = ["ctx_traces", "ctx_services_unique", "ctx_err_rate_mean", 
                "ctx_5xx_frac_mean", "ctx_concurrency_peak", 
                "ctx_abn_ratio_error", "ctx_p90_over_baseline"]
    
    for c in ctx_cols: df_core[c] = 0.0

    groups = df_core.groupby("dominant_node")
    W_ms = win_minutes * 60 * 1000.0
    
    for dom_node, group in tqdm(groups, desc="Node Context"):
        if not dom_node: continue 
        
        sub_df = group.sort_values("trace_tmid")
        times = sub_df["trace_tmid"].values
        errs = sub_df["err_rate"].values
        f5s = sub_df["_5xx_frac"].values
        svcs = sub_df["svc_unique"].values
        
        pref_err = np.concatenate([[0], np.cumsum(errs)])
        pref_f5  = np.concatenate([[0], np.cumsum(f5s)])
        pref_svc = np.concatenate([[0], np.cumsum(svcs)])
        
        left_idxs = np.searchsorted(times, times - W_ms, side='left')
        right_idxs = np.searchsorted(times, times + W_ms, side='right')
        counts = right_idxs - left_idxs
        
        valid_mask = counts > 0
        if valid_mask.any():
            R = right_idxs[valid_mask]
            L = left_idxs[valid_mask]
            cnt = counts[valid_mask]
            
            target_indices = sub_df.index[valid_mask]
            
            df_core.loc[target_indices, "ctx_traces"] = cnt
            df_core.loc[target_indices, "ctx_services_unique"] = (pref_svc[R] - pref_svc[L]) / cnt
            df_core.loc[target_indices, "ctx_err_rate_mean"] = (pref_err[R] - pref_err[L]) / cnt
            df_core.loc[target_indices, "ctx_5xx_frac_mean"] = (pref_f5[R] - pref_f5[L]) / cnt
            df_core.loc[target_indices, "ctx_concurrency_peak"] = cnt
            df_core.loc[target_indices, "ctx_abn_ratio_error"] = (pref_err[R] - pref_err[L]) / cnt

    return df_core

def reduce_df_core_duplicates(df_core: pd.DataFrame) -> pd.DataFrame:
    """
    [Step 3.5] TraceID 去重与冲突解决
    """
    if df_core.empty: return df_core

    dup_cnt = df_core.duplicated(subset=["TraceID"]).sum()
    if dup_cnt == 0:
        return df_core

    print(f"   ⚠️ 发现 {dup_cnt} 个重复 TraceID，正在执行归并策略...")
    
    # 自定义聚合函数：取第一个非空字符串
    def first_valid_str(series):
        for s in series:
            if isinstance(s, str) and s and s.lower() != "nan": return s
        return None

    # 聚合策略：确保包含 y_c3, fault_type, dominant_node
    agg_rules = {
        "y_bin": "max",     
        "y_c3": "max",      # 优先取 Service(1)/Node(2)
        "err_rate": "mean",
        "_5xx_frac": "mean",
        "ctx_traces": "mean",
        "ctx_err_rate_mean": "mean",
        "dominant_node": first_valid_str,
        "fault_type": first_valid_str
    }
    
    # 自动补全 ctx 字段
    for col in df_core.columns:
        if col.startswith("ctx_") and col not in agg_rules:
            agg_rules[col] = "mean"
            
    # 只聚合存在的列 (防止 KeyError)
    final_agg = {k: v for k, v in agg_rules.items() if k in df_core.columns}
    
    df_red = df_core.groupby("TraceID", as_index=False).agg(final_agg)
    return df_red

def log_dataset_stats(df_core):
    """打印详细统计 (Safe Mode)"""
    print("\n=== 📊 数据集统计 (Dataset Statistics) ===")
    if df_core.empty:
        print("⚠️ 数据集为空！")
        return

    print(f"Total traces               : {len(df_core)}")
    
    if 'y_bin' in df_core.columns:
        print(f"Normal traces              : {(df_core['y_bin'] == 0).sum()}")
        print(f"Anomaly traces             : {(df_core['y_bin'] == 1).sum()}")
    
    if 'y_c3' in df_core.columns:
        print(f"Service-level faults       : {(df_core['y_c3'] == 1).sum()}")
        print(f"Node-level faults          : {(df_core['y_c3'] == 2).sum()}")
    
    if "fault_type" in df_core.columns and 'y_bin' in df_core.columns:
        print("\n--- Fault Type Breakdown ---")
        fine_cnt = df_core[df_core['y_bin'] == 1]['fault_type'].value_counts()
        for ft, cnt in fine_cnt.items():
            print(f"  {ft or 'NULL':<30} : {cnt}")
    print("========================================\n")

# ================= 📦 记录构建 =================

def build_records(df: pd.DataFrame, api_vocab, status_vocab, node_vocab, 
                  fixed_c3, min_trace_size=2):
    """[Step 4] 构建最终 JSONL 记录"""
    records = []
    if df.empty: return records
    
    df["url_tmpl"] = df["URL"].apply(url_template)
    grouped = df.groupby("TraceID", sort=False)
    
    for tid, g in grouped:
        if len(g) < min_trace_size: continue
        g = g.reset_index(drop=True)
        
        sid_map = {str(sid): i for i, sid in enumerate(g["SpanId"])}
        edges = []
        for i, pid in enumerate(g["ParentID"]):
            pid_str = str(pid)
            if pid_str in sid_map:
                edges.append([sid_map[pid_str], i])
        
        nodes_data = []
        svc_vals = g["ServiceName"].values
        url_vals = g["url_tmpl"].values
        node_vals = g["NodeName"].values
        http_vals = g["HttpStatusCode"].values 
        lat_vals = g["DurationMs"].values
        start_vals = g["StartTimeMs"].values
        end_vals = g["EndTimeMs"].values
        
        for i in range(len(g)):
            api_key = make_api_key(svc_vals[i], url_vals[i])
            if api_key not in api_vocab: api_vocab[api_key] = len(api_vocab) + 1
            api_id = api_vocab[api_key]
            
            # Status ID (Robust conversion)
            try:
                val = pd.to_numeric(http_vals[i], errors='coerce')
                skey = 0 if pd.isna(val) else int(val)
            except: skey = 0
            
            skey_str = str(skey)
            if skey_str not in status_vocab: status_vocab[skey_str] = len(status_vocab) + 1
            status_id = status_vocab[skey_str]
            
            # Node ID
            nm = str(node_vals[i]).strip()
            if not nm: node_id = 1 # <unk>
            else:
                if nm not in node_vocab: node_vocab[nm] = len(node_vocab) + 1
                node_id = node_vocab[nm]
            
            nodes_data.append({
                "api_id": int(api_id),
                "node_id": int(node_id),
                "status_id": int(status_id),
                "latency_ms": float(lat_vals[i]),
                "start_ms": float(start_vals[i]),
                "end_ms": float(end_vals[i])
            })
            
        ft = str(g.loc[0, "fault_type"]).strip().lower()
        if fixed_c3 == 0: y_bin, y_c3, ft = 0, 0, None
        else:
            y_bin, y_c3 = 1, fixed_c3
            if not ft or ft == "nan": ft = "unknown"

        records.append({
            "trace_id": str(tid),
            "nodes": nodes_data,
            "edges": edges,
            "y_bin": y_bin, 
            "y_c3": y_c3, 
            "fault_type": ft
        })
        
    return records

# ================= 🚀 主程序 =================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", default="/root/wzc/Trace_mcp/app/dataset/tianchi/data/NormalData/normal_traces_mapped.csv")
    parser.add_argument("--service", default="/root/wzc/Trace_mcp/app/dataset/tianchi/data/ServiceFault/all_fault_traces_mapped.csv")
    parser.add_argument("--node", default="/root/wzc/Trace_mcp/app/dataset/tianchi/data/NodeFault/all_fault_traces_mapped.csv")
    parser.add_argument("--outdir", default="dataset/tianchi")
    parser.add_argument("--win-minutes", type=float, default=3.0)
    parser.add_argument("--min-trace-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--drop-orphans", type=int, default=1, help="1=开启断链过滤")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)
    
    # 1. 加载
    print("[1/5] 加载 过滤 Orphans...")
    df_n = load_and_clean(args.normal, 'normal')
    df_s = load_and_clean(args.service, 'service')
    df_f = load_and_clean(args.node, 'node')
    
    # 1.5. 过滤 Orphans
    if args.drop_orphans:
        
        df_n = drop_orphan_traces(df_n)
        df_s = drop_orphan_traces(df_s)
        df_f = drop_orphan_traces(df_f)

    print("[2/5] 计算 Trace 核心指标 (包含标签注入)...")
    core_list = []
    
    # === 关键修复：显式注入 y_c3 和 fault_type ===
    
    if not df_n.empty:
        for _, g in tqdm(df_n.groupby("TraceID"), desc="Normal"):
            r = per_trace_core(g)
            r["y_bin"] = 0
            r["y_c3"] = 0
            raw_ft = str(g["fault_type"].iloc[0]) if "fault_type" in g.columns else "normal"
            ft = raw_ft.lower() if raw_ft and raw_ft != "nan" else "normal"
            r["fault_type"] = ft if ft and ft != "nan" else "normal"
            core_list.append(r)
            
    if not df_s.empty:
        for _, g in tqdm(df_s.groupby("TraceID"), desc="Service"):
            r = per_trace_core(g)
            r["y_bin"] = 1
            r["y_c3"] = 1
            raw_ft = str(g["fault_type"].iloc[0]) if "fault_type" in g.columns else "unknown"
            ft = raw_ft.lower() if raw_ft and raw_ft != "nan" else "unknown"
            r["fault_type"] = ft if ft and ft != "nan" else "unknown"
            core_list.append(r)
            
    if not df_f.empty:
        for _, g in tqdm(df_f.groupby("TraceID"), desc="Node"):
            r = per_trace_core(g)
            r["y_bin"] = 1
            r["y_c3"] = 2
            raw_ft = str(g["fault_type"].iloc[0]) if "fault_type" in g.columns else "unknown"
            ft = raw_ft.lower() if raw_ft and raw_ft != "nan" else "unknown"
            r["fault_type"] = ft if ft and ft != "nan" else "unknown"
            core_list.append(r)
            
    df_core = pd.DataFrame(core_list)
    
    # 3. 计算 Context
    df_core = build_window_context_fast(df_core, win_minutes=args.win_minutes)
    
    # 3.5. 去重归并
    print("[3/5] TraceID 去重归并...")
    df_core_red = reduce_df_core_duplicates(df_core)
    
    # 打印统计 (Safe)
    log_dataset_stats(df_core_red)
    
    if df_core_red.empty:
        print("❌ 错误：处理后无有效数据，请检查输入文件路径。")
        return

    ctx_map = df_core_red.set_index("TraceID").to_dict(orient="index")
    
    # 4. 构建样本
    print("[4/5] 构建样本...")
    api_vocab = {"<pad>": 0, "<unk>": 1}
    status_vocab = {"<pad>": 0, "<unk>": 1} 
    node_vocab = {"<pad>": 0, "<unk>": 1}   
    
    recs_n = build_records(df_n, api_vocab, status_vocab, node_vocab, 0, args.min_trace_size)
    recs_s = build_records(df_s, api_vocab, status_vocab, node_vocab, 1, args.min_trace_size)
    recs_f = build_records(df_f, api_vocab, status_vocab, node_vocab, 2, args.min_trace_size)
    
    valid_ids = set(ctx_map.keys())
    all_recs = [r for r in recs_n + recs_s + recs_f if r["trace_id"] in valid_ids]
    
    print(f"📊 最终有效样本数: {len(all_recs)}")
    
    # 计算类型词表
    type_ctr = Counter([r["fault_type"] for r in all_recs if r.get("fault_type")])
    type_names = ["Normal"] + [t for t, _ in type_ctr.most_common()]
    type2id = {t: i for i, t in enumerate(type_names)}
    
    # 5. 回填 Context & Labels
    final_data = []
    for r in all_recs:
        tid = r["trace_id"]
        info = ctx_map.get(tid, {})
        
        r["ctx"] = [
            info.get("ctx_traces", 0.0), info.get("ctx_services_unique", 0.0),
            info.get("ctx_err_rate_mean", 0.0), info.get("ctx_5xx_frac_mean", 0.0),
            info.get("ctx_concurrency_peak", 0.0), info.get("ctx_abn_ratio_error", 0.0),
            info.get("ctx_p90_over_baseline", 0.0)
        ]
        
        # 使用 Merge 后的 Labels (确保去重结果生效)
        r["y_bin"] = int(info.get("y_bin", r["y_bin"]))
        r["y_c3"]  = int(info.get("y_c3", r["y_c3"]))
        
        ft = info.get("fault_type", r["fault_type"])
        if r["y_bin"] == 0: 
            r["y_type"] = 0
        else:
            r["y_type"] = type2id.get(ft, -1) # 如果类型未知，暂标-1
            
        final_data.append(r)
        
    print("[5/5] 保存数据...")
    random.shuffle(final_data)
    n = len(final_data)
    cut1 = int(n * 0.7); cut2 = int(n * 0.85)
    
    def save_jsonl(path, data):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data: f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    save_jsonl(os.path.join(args.outdir, "train.jsonl"), final_data[:cut1])
    save_jsonl(os.path.join(args.outdir, "val.jsonl"), final_data[cut1:cut2])
    save_jsonl(os.path.join(args.outdir, "test.jsonl"), final_data[cut2:])
    
    with open(os.path.join(args.outdir, "vocab.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "api_vocab_size": len(api_vocab),
            "status_vocab_size": len(status_vocab),
            "node_vocab_size": len(node_vocab),
            "type_names": type_names,
            "ctx_dim": 7
        }, f, indent=2)
    print(f"✅ 完成！数据已保存至 {args.outdir}")

if __name__ == "__main__":
    main()