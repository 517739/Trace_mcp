#!/usr/bin/env python
# test_aiops_svnd.py  (v0.4, host GCN/GAT + host_state)
# -*- coding: utf-8 -*-
"""
独立测试脚本：
- 从 ckpt 读取 stats / keep_types / 训练时的 args
- 构造数据集（无需读取训练集），评测并打印逐类报告
"""
import os
import argparse

import torch
from torch.utils.data import DataLoader

from utils_sv import (
    set_seed,
    TraceDataset,
    collate_multi,
    vocab_sizes_from_meta,
    evaluate_detailed,
    print_per_class_reports,
)
from model_sv import TraceClassifier

try:
    # 保留兼容，但本脚本内部目前未使用 tqdm
    from tqdm import tqdm  # noqa: F401
except ImportError:
    tqdm = None  # noqa: F401


def main():
    parser = argparse.ArgumentParser("AIOps Trace Multi-Head Test")
    parser.add_argument(
        "--data_root",
        type=str,
        default="dataset/aiops_svnd_1209",
        help="包含 split.jsonl 和 vocab.json 的目录，例如 dataset/aiops_svnd_1209",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="dataset/aiops_svnd_1209/host_gcn/aiops_nodectx_multihead.pt",
        help="*.pt 模型权重路径",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="使用哪个 split 进行评测",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--host_conv",
        type=str,
        choices=["gcn", "gat"],
        default=None,
        help="主机共址通道类型；默认从 ckpt args['host_conv'] 读取，若无则用 gcn",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制测试样本数量（例如 --limit 100），默认 None 运行全量",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # ====== 加载 ckpt：stats + keep_types + 原始训练 args ======
    ckpt = torch.load(args.ckpt, map_location=device)
    stats = ckpt["stats"]
    keep_types = ckpt.get("keep_types", None)
    keep_types = set(keep_types) if keep_types is not None else None
    ckpt_args = ckpt.get("args", {}) or {}
    state_dict = ckpt.get("state_dict", {})

    # 决定 host_conv：优先命令行，其次 ckpt 中记录的配置，最后默认 gcn
    host_conv = ckpt_args.get("host_conv", "gcn")
    if args.host_conv is not None:
        host_conv = args.host_conv

    # 从 ckpt 中推断 host_state_dim，保证测试模型结构与训练时一致
    host_state_dim = 0
    if "host_state_mlp.weight" in state_dict:
        w = state_dict["host_state_mlp.weight"]
        host_state_dim = int(w.shape[1])

    # ====== 词表 / 类型名等 ======
    api_sz, st_sz, node_sz, type_names, ctx_dim = vocab_sizes_from_meta(args.data_root)

    # ====== 初始化数据集（会一次性读入 jsonl） ======
    jsonl_path = os.path.join(args.data_root, f"{args.split}.jsonl")
    print(f"[Data] Loading dataset from {jsonl_path} ...")
    ds = TraceDataset(jsonl_path, task="multihead", fit_stats=False, stats=stats, keep_types=keep_types)

    if args.limit is not None and args.limit > 0:
        original_len = len(ds)
        if args.limit < original_len:
            ds.items = ds.items[: args.limit]
            print(f"[Info] Limit applied: {original_len} -> {len(ds)} samples kept.")
        else:
            print(f"[Info] Limit ({args.limit}) >= Dataset size ({original_len}), using full dataset.")
    else:
        print(f"[Info] Using full dataset: {len(ds)} samples.")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multi)

    # ====== 模型 ======
    model = TraceClassifier(
        api_sz,
        st_sz,
        node_sz,
        n_types=len(type_names),
        ctx_dim=ctx_dim,
        host_conv=host_conv,
        host_state_dim=host_state_dim,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # ====== 评测 ======
    print(f"\n[Eval] Running {args.split}.jsonl ... (host_conv={host_conv}, host_state_dim={host_state_dim})")
    metrics = evaluate_detailed(model, loader, device, type_names, keep_types=keep_types)
    print_per_class_reports(model, loader, device, type_names, keep_types=keep_types)

    print("\n[OK] 测试完成")


if __name__ == "__main__":
    main()

