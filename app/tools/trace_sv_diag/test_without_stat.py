import os, json, argparse, csv
import torch
import torch.nn as nn
from collections import Counter
import numpy as np  # 确保导入 numpy

from utils_sv import set_seed, TraceDataset, collate, vocab_sizes_from_meta
from model_sv import TraceClassifier

@torch.no_grad()
def evaluate_and_save_fine(model, loader, device, class_names, keep_types, out_dir):
    ce = nn.CrossEntropyLoss()
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n = 0.0, 0

    for g, y, order, _ in loader:
        g = g.to(device); y = y.to(device)
        logits = model(g)
        loss = ce(logits, y)
        b = y.size(0)
        total_loss += loss.item() * b
        n += b
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    if n == 0:
        print("[test] empty loader."); return

    logits = torch.cat(all_logits, 0)
    labels = torch.cat(all_labels, 0)

    K = logits.size(-1)
    
    # 屏蔽未保留的类别
    if keep_types is not None:
        mask = torch.full((K,), float("-inf"))
        for k in keep_types:
            if k < K: mask[k] = 0.0
        logits = logits + mask

    preds = logits.argmax(1)

    # 确定展示的类别名称
    keep = sorted(keep_types) if keep_types is not None else list(range(K))
    keep = [k for k in keep if k < len(class_names)]
    name_keep = [class_names[i] for i in keep]

    cm = np.zeros((len(keep), len(keep)), dtype=int)
    idx_map = {gid: lid for lid, gid in enumerate(keep)}
    
    for t, p in zip(labels.numpy().tolist(), preds.numpy().tolist()):
        if t in idx_map and p in idx_map:
            cm[idx_map[t], idx_map[p]] += 1

    rows = []
    # 用于计算加权平均的列表
    val_p, val_r, val_f, val_s = [], [], [], []

    for i, k in enumerate(keep):
        tp = int(cm[i, i])
        support = int(cm[i, :].sum())      # Support = TP + FN
        predcnt = int(cm[:, i].sum())      # Pred    = TP + FP
        
        p = tp / (predcnt + 1e-9)
        r = tp / (support + 1e-9)
        f = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        
        rows.append([name_keep[i], tp, support, predcnt, p, r, f])
        
        # 收集用于计算 Macro/Weighted 的值 (仅当 support > 0 时才有意义，但为了对其通常保留)
        if support > 0:
            val_p.append(p)
            val_r.append(r)
            val_f.append(f)
            val_s.append(support)
        else:
            # support=0 的类别不参与加权计算
            pass

    # --- 指标计算 ---
    overall_acc = float((labels.numpy() == preds.numpy()).mean())
    
    # Macro Average (算数平均)
    macro_p = float(np.mean(val_p)) if val_p else 0.0
    macro_r = float(np.mean(val_r)) if val_r else 0.0
    macro_f = float(np.mean(val_f)) if val_f else 0.0

    # Weighted Average (加权平均)
    total_s = sum(val_s)
    if total_s > 0:
        weighted_p = float(np.average(val_p, weights=val_s))
        weighted_r = float(np.average(val_r, weights=val_s))
        weighted_f = float(np.average(val_f, weights=val_s))
    else:
        weighted_p = weighted_r = weighted_f = 0.0

    os.makedirs(out_dir, exist_ok=True)
    
    # 保存 CSV
    with open(os.path.join(out_dir, "test_confusion.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow([""] + name_keep)
        for i, nm in enumerate(name_keep):
            w.writerow([nm] + cm[i, :].tolist())

    with open(os.path.join(out_dir, "test_per_class.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class","TP","Support","Pred","Precision","Recall","F1"])
        w.writerows(rows)
        # 在 CSV 末尾追加 summary
        w.writerow([])
        w.writerow(["Macro Avg", "", "", "", macro_p, macro_r, macro_f])
        w.writerow(["Weighted Avg", "", total_s, "", weighted_p, weighted_r, weighted_f])
        w.writerow(["Overall Acc", "", "", "", overall_acc, "", ""])

    # 保存 JSON
    with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "overall_acc": overall_acc,
            "macro": {"p": macro_p, "r": macro_r, "f1": macro_f},
            "weighted": {"p": weighted_p, "r": weighted_r, "f1": weighted_f},
            "kept_types": keep,
            "kept_names": name_keep,
            "loss": total_loss / max(n, 1)
        }, f, ensure_ascii=False, indent=2)

    # --- 打印输出 ---
    print("\n===== Detailed Evaluation =====")
    print("Confusion Matrix (rows=true, cols=pred):")
    print("         " + "".join([f"{n[:8]:^10}" for n in name_keep]))
    for i, nm in enumerate(name_keep):
        print(" "+f"{nm[:8]:<8}" + "".join([f"{cm[i,j]:^10d}" for j in range(len(keep))]))

    print("\nPer-class metrics:")
    print(" class                 TP     Support   Pred       Precision   Recall      F1")
    for row in rows:
        nm, tp, sup, pred, p, r, f = row
        print(f" {nm[:20]:<20}{tp:>6d}   {sup:>8d}   {pred:>6d}     {p:>9.4f}  {r:>8.4f}  {f:>8.4f}")
    
    print("-" * 75)
    print(f" Overall Acc : {overall_acc:.4f}")
    print(f" Macro Avg   : P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f:.4f}")
    print(f" Weighted Avg: P={weighted_p:.4f}  R={weighted_r:.4f}  F1={weighted_f:.4f}")
    print("-" * 75)
    print(f"[Output] Reports saved to: {out_dir}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="dataset/aiops_sv", help="包含 test.jsonl 与 vocab.json 的目录")
    ap.add_argument("--model-path", default="dataset/aiops_sv/aiops_superfine_cls.pth", help="ckpt路径")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-type-support", type=int, default=150)
    ap.add_argument("--run-name", default="trace_only")
    ap.add_argument("--task", choices=["fine", "superfine"], default="superfine", help="任务类型")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of test samples to process")
    
    args = ap.parse_args()
    set_seed(args.seed)

    api_vocab, status_vocab, fine_names, superfine_names = vocab_sizes_from_meta(args.data_root)
    if args.task == "superfine":
        if superfine_names is None:
            raise RuntimeError("vocab.json 中缺少 superfine_classes")
        class_names = superfine_names
        print(f"[Info] Task=superfine, num_classes={len(class_names)}")
    else:
        if fine_names is None:
            raise RuntimeError("vocab.json 中缺少 fine_label_map")
        class_names = fine_names
        print(f"[Info] Task=fine, num_classes={len(class_names)}")

    te_path = os.path.join(args.data_root, "test.jsonl")
    ds_te = TraceDataset(te_path, task=args.task, fit_stats=True)
    
    # === Limit logic ===
    if args.limit is not None and args.limit > 0:
        if len(ds_te.items) > args.limit:
            print(f"[Info] Limiting test dataset from {len(ds_te.items)} to {args.limit} samples.")
            ds_te.items = ds_te.items[:args.limit]

    te_loader = torch.utils.data.DataLoader(ds_te, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=0)

    label_key = "superfine_label" if args.task == "superfine" else "fine_label"
    cnt = Counter(int(r[label_key]) for r in ds_te.items if r.get(label_key) is not None and int(r[label_key]) >= 0)
    keep_types = {k for k, v in cnt.items() if v >= args.min_type_support}
    if not keep_types:
        keep_types = set(range(len(class_names)))
        print("[Warn] No types met min_support, keeping ALL.")
    print(f"[{args.task}] min_support={args.min_type_support}, keep_types={sorted(keep_types)}")

    num_classes = len(class_names)
    model = TraceClassifier(api_vocab, status_vocab, num_classes).to(args.device)
    
    print(f"[Info] Loading model from {args.model_path} ...")
    state = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state, strict=True)

    out_dir = os.path.join(args.data_root, "runs", args.run_name, f"{args.task}_test")
    evaluate_and_save_fine(model, te_loader, args.device, class_names, keep_types, out_dir)

if __name__ == "__main__":
    main()