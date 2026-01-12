# tc_test.py
# -*- coding: utf-8 -*-
"""
SV (Scheme B) 独立测试脚本
- 基于 test_without_infra.py 修改
- 从 ckpt 读取 stats 与 keep_types
- 支持 limit 参数进行物理切片 (避免内存溢出)
- 输出详细的 F1 / Precision / Recall 报告
"""
print("DEBUG: Script started. Importing libraries...", flush=True)

import os, argparse, torch, tempfile
import numpy as np
from torch.utils.data import DataLoader

# 导入 SV (Scheme B) 的模块
from tc_utils import (
    set_seed, TraceDataset, collate_multi, vocab_sizes_from_meta
)
from tc_model import TraceClassifier

def create_temp_subset(src_path, limit):
    """
    物理切片：读取源文件前 limit 行写入临时文件。
    解决加载超大测试集时的内存或超时问题。
    """
    if limit is None or limit <= 0:
        return src_path
    
    # 创建临时文件
    tf = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".jsonl", encoding="utf-8")
    print(f"[Info] Slicing dataset: reading first {limit} lines from {src_path} ...")
    
    count = 0
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= limit:
                break
            tf.write(line)
            count += 1
            
    tf.close()
    print(f"[Info] Created temp dataset at {tf.name} with {count} samples.")
    return tf.name

@torch.no_grad()
def evaluate_and_print_filtered(model, loader, device, type_names, keep_types, exclude_normal=False):
    """
    自定义评测函数：只打印 Support > 0 (实际出现过) 的类别，保持输出整洁。
    """
    model.eval()
    all_logits, all_labels = [], []
    
    # 1. 推理 (Inference)
    for g, lab, _, _ in loader:
        g = g.to(device)
        # 获取 fine-grained type labels (y_type)
        yt = lab["y_type"].to(device) # shape [B]
        mt = lab["m_type"].to(device) # mask [B]
        
        out = model(g)
        logits = out["logits_type"]
        
        # 仅评测 m_type=1 (异常且有细类标签) 的样本
        valid_idx = (mt > 0)
        if valid_idx.sum() > 0:
            all_logits.append(logits[valid_idx].detach().cpu())
            all_labels.append(yt[valid_idx].detach().cpu())

    if not all_logits:
        print("[Warn] No valid fault-type samples found in this batch/limit.")
        return

    logits = torch.cat(all_logits, 0)
    labels = torch.cat(all_labels, 0) # Ground Truth
    
    # 2. 屏蔽未保留的类型 (Masking)
    # 这一步是为了确保模型不会预测出那些训练时被过滤掉的极少样本类别
    K = logits.size(-1)
    if keep_types is not None:
        mask = torch.full((K,), float("-inf"))
        for k in keep_types:
            if k < K: mask[k] = 0.0
        logits = logits + mask

    preds = logits.argmax(1)
    
    # 3. 构建混淆矩阵
    # 确定要统计的列（keep_types 或全部）
    target_indices = sorted(list(keep_types)) if keep_types else list(range(len(type_names)))
    # 过滤掉超出 type_names 范围的索引 (防守性编程)
    target_indices = [i for i in target_indices if i < len(type_names)]
    
    idx_map = {real_id: map_id for map_id, real_id in enumerate(target_indices)}
    
    cm = np.zeros((len(target_indices), len(target_indices)), dtype=int)
    labels_np = labels.numpy()
    preds_np = preds.numpy()
    
    for t, p in zip(labels_np, preds_np):
        if t in idx_map and p in idx_map:
            cm[idx_map[t], idx_map[p]] += 1
            
    # 4. 打印报告 (过滤掉 Support=0 的行)
    print("\n===== Evaluation Result (Only Active Classes) =====")
    print(" class                 TP     Support   Pred       Precision   Recall      F1")
    
    val_p, val_r, val_f, val_s = [], [], [], []
    rows_to_print = []
    
    for i, real_id in enumerate(target_indices):
        name = type_names[real_id]
        if exclude_normal and name.lower() == "normal":
            continue
        tp = int(cm[i, i])
        support = int(cm[i, :].sum())
        predcnt = int(cm[:, i].sum())
        
        p = tp / (predcnt + 1e-9)
        r = tp / (support + 1e-9)
        f = 2*p*r/(p+r+1e-9)
        
        # 核心逻辑：只有 Support > 0 才进入最终报告
        if support > 0:
            val_p.append(p); val_r.append(r); val_f.append(f); val_s.append(support)
            rows_to_print.append((name, tp, support, predcnt, p, r, f))
    
    # 按样本数降序排列，更美观
    rows_to_print.sort(key=lambda x: x[2], reverse=True)
    
    for r in rows_to_print:
        nm, tp, sup, pred, p, r, f = r
        print(f" {nm[:20]:<20}{tp:>6d}   {sup:>8d}   {pred:>6d}     {p:>9.4f}  {r:>8.4f}  {f:>8.4f}")

    # 计算加权平均
    total_s = sum(val_s)
    acc = (labels_np == preds_np).mean()
    wp = np.average(val_p, weights=val_s) if total_s else 0
    wr = np.average(val_r, weights=val_s) if total_s else 0
    wf = np.average(val_f, weights=val_s) if total_s else 0
    
    print("-" * 75)
    print(f" Overall Acc : {acc:.4f}")
    print(f" Weighted Avg: P={wp:.4f}  R={wr:.4f}  F1={wf:.4f}  (Total Support={total_s})")
    print("-" * 75)


def main():
    parser = argparse.ArgumentParser("AIOps Trace SV Scheme B Test")
    
    # === 参数 (根据你的 tc_train.py 默认值设定) ===
    parser.add_argument("--data-root", type=str, default="dataset/tianchi/processed_0111", help="数据集目录")
    # 默认模型路径指向你刚刚 F1 EarlyStop 保存的位置
    parser.add_argument("--model-path", type=str, default="dataset/tianchi/processed_0111/f1earlystop/best_model.pt", help="模型权重路径")
    
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="仅测试前N条数据 (用于快速调试)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--exclude-normal", type=bool, default=True, help="在报告和加权平均中排除 'normal' 类")

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # 1. 加载模型 Checkpoint (获取 Stats 和 Keep Types)
    print(f"[Info] Loading model from {args.model_path} ...")
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found: {args.model_path}")
        return

    ckpt = torch.load(args.model_path, map_location=device)
    stats = ckpt["stats"]
    keep_types = set(ckpt.get("keep_types", [])) if ckpt.get("keep_types", None) is not None else None
    
    # 获取词表大小
    api_sz, st_sz, node_sz, type_names, ctx_dim = vocab_sizes_from_meta(args.data_root)
    print(f"[Info] Meta loaded. Ctx Dim={ctx_dim}, Types={len(type_names)}")

    # 2. 物理切片 (Temp File)
    origin_path = os.path.join(args.data_root, f"{args.split}.jsonl")
    if not os.path.exists(origin_path):
        print(f"❌ Error: Data file not found: {origin_path}")
        return

    actual_path = create_temp_subset(origin_path, args.limit)

    # 3. 加载数据集 (fit_stats=False, 使用 ckpt 中的全局 stats)
    ds = TraceDataset(actual_path, task="multihead", fit_stats=False, stats=stats, keep_types=keep_types)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multi)

    # 清理临时文件
    if actual_path != origin_path:
        try: os.remove(actual_path)
        except: pass

    # 4. 加载模型结构与权重
    # 注意：SV 模型不需要 Host Graph 参数，所以初始化比较简单
    model = TraceClassifier(
        api_sz, st_sz, node_sz, 
        n_types=len(type_names), 
        ctx_dim=ctx_dim
    ).to(device)
    
    # 加载权重 (SV 模型不需要像 SVND 那样做 hgc -> h_host 的键名映射，直接加载即可)
    try:
        model.load_state_dict(ckpt["state_dict"])
        print("[Info] Model weights loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Direct load failed ({e}). Trying strict=False...")
        model.load_state_dict(ckpt["state_dict"], strict=False)
    
    # 5. 运行评测
    print(f"\n[Eval] Running on {args.limit if args.limit else 'ALL'} samples from '{args.split}' set ...")
    evaluate_and_print_filtered(model, loader, device, type_names, keep_types, exclude_normal=args.exclude_normal)
    print("\n[OK] Test finished.")

if __name__ == "__main__":
    main()