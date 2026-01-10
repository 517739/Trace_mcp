# train_tc_sv_new.py
# -*- coding: utf-8 -*-
"""
SV (Service-View) 优化版训练脚本
- 引入 Context 特征 (如果在 make 阶段已构建)
- 输出 SVND 风格的详细报表 (混淆矩阵、F1等)
- 使用多任务 Loss 稳定训练
"""

import os, json, argparse, torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

# 假设 model_sv 已按上述第二步修改
from model_sv import TraceClassifier 
# 假设 utils_sv 已包含 dataset 定义
from utils_sv import TraceDataset, collate, set_seed, vocab_sizes_from_meta, evaluate_detailed

# ================= 主训练逻辑 =================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/tianchi/processed_sv_opt") # 你的新数据目录
    parser.add_argument("--save-dir", default="logs/sv_optimized")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ctx-dim", type=int, default=3, help="Context维度，取决于make脚本")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据
    print(f"📖 Loading data from {args.data_root}...")
    # fit_stats=True 会计算延迟的均值方差用于归一化
    ds_tr = TraceDataset(os.path.join(args.data_root, "train.jsonl"), fit_stats=True)
    stats = ds_tr.stats
    ds_va = TraceDataset(os.path.join(args.data_root, "val.jsonl"), fit_stats=False, stats=stats)
    ds_te = TraceDataset(os.path.join(args.data_root, "test.jsonl"), fit_stats=False, stats=stats)

    tr_loader = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=4)
    va_loader = DataLoader(ds_va, batch_size=args.batch, shuffle=False, collate_fn=collate)
    te_loader = DataLoader(ds_te, batch_size=args.batch, shuffle=False, collate_fn=collate)

    # 2. 获取词表大小
    api_sz, status_sz, fine_names, _ = vocab_sizes_from_meta(args.data_root)
    # 如果 vocab.json 里没有 fine_names，请手动指定或从 dataset 统计
    class_names = fine_names if fine_names else [f"Type_{i}" for i in range(10)]
    num_classes = len(class_names)
    print(f"🎯 Classes ({num_classes}): {class_names}")

    # 3. 初始化模型 (带 Context)
    model = TraceClassifier(
        api_vocab=api_sz, 
        status_vocab=status_sz, 
        num_classes=num_classes,
        ctx_dim=args.ctx_dim # 关键：传入 Context 维度
    ).to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Loss 定义
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    # 4. 训练循环
    best_f1 = 0.0
    
    for ep in range(1, args.epochs+1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(tr_loader, desc=f"Epoch {ep}/{args.epochs}")
        for g, y, _, _ in pbar:
            g = g.to(device); y = y.to(device)
            
            # Forward
            out = model(g)
            
            # 计算 Loss (多任务：分类 + 二分类辅助)
            # 假设 0 号类是 Normal
            is_anomaly = (y > 0).float()
            
            if isinstance(out, dict):
                # 推荐方式：多头 Loss
                loss_type = ce_loss(out["logits_type"], y)
                loss_bin = bce_loss(out["logit_bin"], is_anomaly)
                loss = loss_type + 0.5 * loss_bin # 权重可调
            else:
                # 兼容旧代码
                loss = ce_loss(out, y)
                
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        # Validation
        print(f"\n[Eval Epoch {ep}]")
        # 直接使用 utils_sv 中的函数，它现在能处理 dict 了
        metrics = evaluate_detailed(model, va_loader, device, class_names)
        
        # metrics 是一个字典，根据 utils_sv 的返回值获取 acc/f1
        acc = metrics["acc"]
        f1 = metrics["macro_f1"]
        print(f"Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print("✨ New Best Model Saved!")

    # 5. Final Test
    print("\n🏆 Final Test Evaluation")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pth")))
    evaluate_detailed(model, te_loader, device, class_names)
    
    # 保存 stats 用于推理
    import pickle
    with open(os.path.join(args.save_dir, "stats.pkl"), "wb") as f:
        pickle.dump(stats, f)
    print("✅ Training Complete.")

if __name__ == "__main__":
    main()