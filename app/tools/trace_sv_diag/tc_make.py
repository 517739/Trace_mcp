# make_tc_sv_b.py
# -*- coding: utf-8 -*-
"""
SV (Scheme B) 数据集构造工具 - 终极对齐版
- 仅包含 Normal + Service Faults (剔除 Node Faults)
- 核心差异：Context 计算基于 Entry Service (而非 Node)
- 修复：参数完全对齐 SVND (win-minutes, seed, min-trace-size)
- 修复：增加去重(Reduce)和详细统计(Stats)模块
"""

import os, json, argparse, random
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================= 🔧 基础工具 =================
def url_template(u: str) -> str:
    if not isinstance(u, str): return "NA"
    return u.split("?")[0].split("#")[0].split("://")[-1].rstrip("/")

def make_api_key(service: str, url_tmpl: str) -> str:
    s = str(service) if pd.notna(service) else "NA_SVC"
    t = str(url_tmpl) if pd.notna(url_tmpl) else "NA_URL"
    return f"{s}||{t}"

def load_and_clean(path, tag):
    print(f"📖 [{tag}] 读取文件: {path}")
    if not os.path.exists(path): return pd.DataFrame()
    dtypes = {"TraceID": str, "SpanId": str, "ParentID": str, "ServiceName": str, "NodeName": str, "fault_type": str}
    try: df = pd.read_csv(path, dtype=dtypes, low_memory=False)
    except: return pd.DataFrame()
    df['__set__'] = tag
    for c in ["ServiceName", "NodeName", "URL", "fault_type"]:
        df[c] = df[c].fillna("").astype(str) if c in df else ""
    for c in ["StartTimeMs", "EndTimeMs", "DurationMs"]:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0) if c in df else 0.0
    return df

# === [修正] 详细日志版断链过滤 ===
def drop_orphan_traces(df):
    if df.empty: return df
    
    print(f"   🧹 正在检查 Trace 完整性 (允许 1 个根节点悬浮)...")
    valid_indices = []
    roots = {"", "nan", "None", "null", "-1", "0"}
    
    grouped = df.groupby("TraceID", sort=False)
    total_traces = 0
    dropped_traces = 0
    
    for tid, g in grouped:
        total_traces += 1
        span_ids = set(g["SpanId"])
        dangling_count = 0
        for pid in g["ParentID"]:
            pid_str = str(pid).strip()
            if pid_str not in span_ids and pid_str not in roots:
                dangling_count += 1
        
        if dangling_count <= 1:
            valid_indices.extend(g.index)
        else:
            dropped_traces += 1
            
    if dropped_traces == 0:
        print(f"      ✨ 所有 {total_traces} 条 Trace 均结构完整。")
    else:
        keep_rate = 100 * (1 - dropped_traces / total_traces)
        print(f"      [统计] 原始: {total_traces} | 丢弃: {dropped_traces} ({dropped_traces/total_traces*100:.2f}%) | 保留率: {keep_rate:.2f}%")
        
    return df.loc[valid_indices].reset_index(drop=True)

# ================= 📊 核心特征 (Service 视角) =================
def per_trace_core(df_t: pd.DataFrame) -> dict:
    n = len(df_t)
    # [Scheme B] 使用 Entry Service 作为 Context Key
    entry_service = df_t["ServiceName"].iloc[0] if n > 0 else "unk"
    
    http_codes = pd.to_numeric(df_t["HttpStatusCode"], errors='coerce').fillna(0).values
    durs = df_t["DurationMs"].values
    t0, t1 = df_t["StartTimeMs"].min(), df_t["EndTimeMs"].max()
    
    return {
        "TraceID": df_t["TraceID"].iloc[0],
        "context_key": entry_service, 
        "trace_tmid": (t0 + t1) / 2.0,
        "span_dur_p90": np.percentile(durs, 90) if n > 0 else 0.0,
        "err_rate": (http_codes >= 500).mean() if n > 0 else 0.0,
        "_5xx_frac": (http_codes >= 500).mean() if n > 0 else 0.0,
        "svc_unique": df_t["ServiceName"].nunique(),
    }

def build_window_context_fast(df_core, win_minutes=3.0):
    if df_core.empty: return df_core
    print(f"⏳ 计算 Context (窗口={win_minutes}m, 基于 Entry Service)...")
    
    ctx_cols = ["ctx_traces", "ctx_services_unique", "ctx_err_rate_mean", 
                "ctx_5xx_frac_mean", "ctx_concurrency_peak", 
                "ctx_abn_ratio_error", "ctx_p90_over_baseline"]
    for c in ctx_cols: df_core[c] = 0.0

    groups = df_core.groupby("context_key")
    W_ms = win_minutes * 60 * 1000.0
    
    for _, group in tqdm(groups, desc="Service Context"):
        sub = group.sort_values("trace_tmid")
        times, errs = sub["trace_tmid"].values, sub["err_rate"].values
        f5s, svcs   = sub["_5xx_frac"].values, sub["svc_unique"].values
        
        pref_err = np.concatenate([[0], np.cumsum(errs)])
        pref_f5  = np.concatenate([[0], np.cumsum(f5s)])
        pref_svc = np.concatenate([[0], np.cumsum(svcs)])
        
        L = np.searchsorted(times, times - W_ms, side='left')
        R = np.searchsorted(times, times + W_ms, side='right')
        cnt = R - L
        
        valid = cnt > 0
        if valid.any():
            idx = sub.index[valid]
            df_core.loc[idx, "ctx_traces"] = cnt[valid]
            df_core.loc[idx, "ctx_services_unique"] = (pref_svc[R[valid]] - pref_svc[L[valid]]) / cnt[valid]
            df_core.loc[idx, "ctx_err_rate_mean"]   = (pref_err[R[valid]] - pref_err[L[valid]]) / cnt[valid]
            df_core.loc[idx, "ctx_5xx_frac_mean"]   = (pref_f5[R[valid]] - pref_f5[L[valid]]) / cnt[valid]
            df_core.loc[idx, "ctx_concurrency_peak"] = cnt[valid]
            df_core.loc[idx, "ctx_abn_ratio_error"] = (pref_err[R[valid]] - pref_err[L[valid]]) / cnt[valid]

    return df_core

# === [新增] 对齐 SVND 的去重归并逻辑 ===
def reduce_df_core_duplicates(df_core: pd.DataFrame) -> pd.DataFrame:
    if df_core.empty: return df_core
    dup_cnt = df_core.duplicated(subset=["TraceID"]).sum()
    if dup_cnt == 0: return df_core

    print(f"   ⚠️ 发现 {dup_cnt} 个重复 TraceID，正在执行归并策略...")
    def first_valid_str(series):
        for s in series:
            if isinstance(s, str) and s and s.lower() != "nan": return s
        return None

    agg_rules = {
        "y_bin": "max", "y_c3": "max",
        "err_rate": "mean", "_5xx_frac": "mean",
        "ctx_traces": "mean", "ctx_err_rate_mean": "mean",
        "context_key": first_valid_str, "fault_type": first_valid_str
    }
    for col in df_core.columns:
        if col.startswith("ctx_") and col not in agg_rules: agg_rules[col] = "mean"
            
    final_agg = {k: v for k, v in agg_rules.items() if k in df_core.columns}
    return df_core.groupby("TraceID", as_index=False).agg(final_agg)

# === [新增] 对齐 SVND 的统计输出 ===
def log_dataset_stats(df_core):
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
    print("========================================\n")

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

def main():
    parser = argparse.ArgumentParser()
    # [修正] 补全参数，保持与 SVND 一致
    parser.add_argument("--normal", default="E:\ZJU\AIOps\Projects\TraDNN\Trace_mcp/app\dataset/tianchi\data/NormalData/normal_traces_mapped.csv")
    parser.add_argument("--service", default="E:\ZJU\AIOps\Projects\TraDNN\Trace_mcp/app\dataset/tianchi\data/ServiceFault/all_fault_traces_mapped.csv")
    parser.add_argument("--node", default="E:\ZJU\AIOps\Projects\TraDNN\Trace_mcp/app\dataset/tianchi\data/NodeFault/all_fault_traces_mapped.csv", help="不使用，但保留参数兼容性")
    parser.add_argument("--outdir", default="dataset/tianchi/processed_0111")
    parser.add_argument("--win-minutes", type=float, default=3.0, help="Context时间窗口")
    parser.add_argument("--min-trace-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--drop-orphans", type=int, default=1)
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)
    
    print("[1/5] Loading Data (Normal + Service Only)...")
    # 加载数据 (不加载 NodeFault)
    df_n = load_and_clean(args.normal, 'n')
    df_s = load_and_clean(args.service, 's')
    
    if args.drop_orphans:
        df_n = drop_orphan_traces(df_n)
        df_s = drop_orphan_traces(df_s)
    
    print("[2/5] Core & Context...")
    core_list = []
    
    # 注入标签 logic
    if not df_n.empty:
        for _, g in tqdm(df_n.groupby("TraceID")):
            r = per_trace_core(g)
            r.update({"y_bin": 0, "y_c3": 0, "fault_type": "normal"})
            core_list.append(r)
            
    if not df_s.empty:
        for _, g in tqdm(df_s.groupby("TraceID")):
            r = per_trace_core(g)
            # 获取原始 fault_type
            ft = str(g["fault_type"].iloc[0]).lower()
            if not ft or ft=="nan": ft = "unknown"
            r.update({"y_bin": 1, "y_c3": 1, "fault_type": ft})
            core_list.append(r)
            
    df_core = pd.DataFrame(core_list)
    
    # [修正] 使用参数控制窗口大小
    df_core = build_window_context_fast(df_core, win_minutes=args.win_minutes)
    
    # [修正] 加入 SVND 的去重逻辑
    print("[3/5] Reduce Duplicates...")
    df_core = reduce_df_core_duplicates(df_core)
    log_dataset_stats(df_core)
    
    ctx_map = df_core.set_index("TraceID").to_dict(orient="index")
    
    print("[4/5] Building Records...")
    av, sv, nv = {"<pad>":0, "<unk>":1}, {"<pad>":0, "<unk>":1}, {"<pad>":0, "<unk>":1}
    
    recs = []
    recs += build_records(df_n, av, sv, nv, 0, args.min_trace_size)
    recs += build_records(df_s, av, sv, nv, 1, args.min_trace_size)
    
    final = []
    types = set()
    for r in recs:
        tid = r["trace_id"]
        if tid in ctx_map:
            info = ctx_map[tid]
            # 7维 Context 对齐
            r["ctx"] = [
                info.get("ctx_traces", 0), info.get("ctx_services_unique", 0),
                info.get("ctx_err_rate_mean", 0), info.get("ctx_5xx_frac_mean", 0),
                info.get("ctx_concurrency_peak", 0), info.get("ctx_abn_ratio_error", 0),
                info.get("ctx_p90_over_baseline", 0)
            ]
            final.append(r)
            if r["y_bin"] == 1: types.add(r["fault_type"])
            
    type_names = ["normal"] + sorted(list(types))
    t2id = {t:i for i,t in enumerate(type_names)}
    for r in final: r["y_type"] = t2id.get(r["fault_type"], 0)

    print("[5/5] Saving...")
    random.shuffle(final)
    n = len(final); c1, c2 = int(n*0.7), int(n*0.85)
    
    def dump(p, d):
        with open(p, 'w') as f: 
            for x in d: f.write(json.dumps(x)+"\n")
            
    dump(os.path.join(args.outdir, "train.jsonl"), final[:c1])
    dump(os.path.join(args.outdir, "val.jsonl"), final[c1:c2])
    dump(os.path.join(args.outdir, "test.jsonl"), final[c2:])
    
    with open(os.path.join(args.outdir, "vocab.json"), "w") as f:
        json.dump({
            "api_vocab_size": len(av), "status_vocab_size": len(sv), 
            "node_vocab_size": len(nv),
            "type_names": type_names, "ctx_dim": 7
        }, f, indent=2)
        
    print(f"✅ [Scheme B] Done. Saved to {args.outdir}")

if __name__ == "__main__":
    main()