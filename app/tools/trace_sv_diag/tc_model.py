# model_sv_b.py
# -*- coding: utf-8 -*-
"""
SV (Scheme B) 模型定义
基于 model_svnd.py 修改：
1. 移除了 _make_host_graph 和 Host-GCN/GAT 通道。
2. 保留了 Node Embedding (作为普通特征融合)。
3. 保留了 Context 和 TreeLSTM 分支。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv

# ========== TreeLSTM：保持完全一致 ==========
class ChildSumTreeLSTMOp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W_iouf = nn.Linear(dim, 4 * dim, bias=False)
        self.U_iou  = nn.Linear(dim, 3 * dim, bias=False)
        self.b_iou  = nn.Parameter(torch.zeros(1, 3 * dim))
        self.U_f    = nn.Linear(dim, dim)

    def message(self, edges): return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce(self, nodes):
        h_sum = torch.sum(nodes.mailbox["h"], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox["h"]))
        c = torch.sum(f * nodes.mailbox["c"], 1)
        return {"sum": self.U_iou(h_sum), "c": c}

    def apply(self, nodes):
        iou = nodes.data["iou"] + nodes.data["sum"] + self.b_iou
        i, o, u = torch.chunk(iou, 3, dim=-1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data["c"]
        h = o * torch.tanh(c)
        return {"h": h, "c": c}

class TreeLSTMReadout(nn.Module):
    def __init__(self, dim: int, out: int):
        super().__init__()
        self.cell = ChildSumTreeLSTMOp(dim)
        self.out  = nn.Linear(dim, out)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor):
        g = dgl.reverse(dgl.remove_self_loop(g))
        g.ndata["iou"], g.ndata["f"] = torch.split(
            self.cell.W_iouf(x), [3 * x.size(-1), x.size(-1)], dim=-1
        )
        g.ndata["sum"] = torch.zeros_like(g.ndata["iou"])
        g.ndata["h"]   = torch.zeros_like(x)
        g.ndata["c"]   = torch.zeros_like(x)
        dgl.prop_nodes_topo(g, self.cell.message, self.cell.reduce, apply_node_func=self.cell.apply)
        h = g.ndata["h"]
        return self.out(F.relu(h))

# ========== 主模型：Trace Only (无 Host Graph) ==========
class TraceClassifier(nn.Module):
    def __init__(
        self,
        api_vocab,
        status_vocab,
        node_vocab,
        n_types,
        emb: int = 32,
        gc_hidden: int = 64,
        tlstm_out: int = 64,
        ctx_dim: int = 7,
        cls_hidden: int = 128,
        # 保留参数接口以兼容训练脚本调用，但实际上不使用 host_conv 等参数
        host_conv: str = "gcn", 
        host_heads: int = 4,
        **kwargs
    ):
        super().__init__()
        # 1. Embeddings
        self.api_emb    = nn.Embedding(api_vocab + 1, emb)
        self.status_emb = nn.Embedding(status_vocab + 1, emb)
        self.node_emb   = nn.Embedding(node_vocab + 1, emb) # 保留 NodeID 特征，但不构图
        self.depth_emb  = nn.Embedding(64, emb)
        self.pos_emb    = nn.Embedding(512, emb)
        self.lat_mlp    = nn.Sequential(nn.Linear(1, emb), nn.ReLU(), nn.Linear(emb, emb))
        
        # Merge: api + status + node + depth + pos + lat
        in_dim = emb * 5 + emb 
        self.merge = nn.Linear(in_dim, gc_hidden)

        # 2. Call Graph GCN (仅保留调用关系图)
        self.gcn1 = GraphConv(gc_hidden, gc_hidden, allow_zero_in_degree=True)
        self.gcn2 = GraphConv(gc_hidden, gc_hidden, allow_zero_in_degree=True)

        # [差异点] 删除了 self.h_host1, self.h_host2 (Host Graph 分支)

        # 3. TreeLSTM
        self.tlstm = TreeLSTMReadout(gc_hidden, tlstm_out)

        # 4. Context
        self.ctx_mlp = (
            nn.Sequential(nn.Linear(ctx_dim, cls_hidden // 2), nn.ReLU(), nn.Dropout(0.1))
            if ctx_dim and ctx_dim > 0 else None
        )

        # 5. Fusion: Call + Tree + Ctx (移除了 Host)
        fuse_in = gc_hidden + tlstm_out 
        if self.ctx_mlp is not None:
            fuse_in += cls_hidden // 2
            
        self.fuse = nn.Sequential(nn.Linear(fuse_in, cls_hidden), nn.ReLU(), nn.Dropout(0.2))
        
        # 6. Heads (保持三头输出，尽管 c3 退化)
        self.head_bin  = nn.Linear(cls_hidden, 1)
        self.head_c3   = nn.Linear(cls_hidden, 3)
        self.head_type = nn.Linear(cls_hidden, n_types if n_types > 0 else 1)

    def forward(self, g: dgl.DGLGraph):
        # --- Feature Embedding ---
        api    = self.api_emb(g.ndata["api_id"])
        status = self.status_emb(g.ndata["status_id"])
        nodev  = self.node_emb(g.ndata["node_id"]) # NodeID 依然作为特征输入
        depth  = self.depth_emb(torch.clamp(g.ndata["depth"], 0, 63))
        pos    = self.pos_emb(torch.clamp(g.ndata["pos"],   0, 511))
        lat    = self.lat_mlp(g.ndata["lat"])
        
        x = torch.cat([api, status, nodev, depth, pos, lat], dim=-1)
        x = F.relu(self.merge(x))

        device = g.device
        x = x.to(device).contiguous()

        # --- 1. Call Graph (GCN) ---
        src, dst = g.edges()
        N = g.num_nodes()
        # 构建双向+自环图
        g_call = dgl.add_self_loop(dgl.to_bidirected(g, copy_ndata=False))
        
        h_call = F.relu(self.gcn1(g_call, x))
        h_call = self.gcn2(g_call, h_call)

        # [差异点] 删除了 Host Graph 构建与卷积过程

        # --- 2. TreeLSTM ---
        if "parent" in g.ndata:
            sub_graphs = dgl.unbatch(g)
            sizes = [sg.num_nodes() for sg in sub_graphs]
            x_splits = torch.split(x, sizes, dim=0)
            tree_list = []
            for sg, xi in zip(sub_graphs, x_splits):
                p = sg.ndata["parent"]
                m = p >= 0
                if m.any():
                    s = p[m].to(torch.long)
                    d = torch.nonzero(m, as_tuple=False).squeeze(1).to(torch.long)
                    t = dgl.graph((s, d), num_nodes=sg.num_nodes(),
                                  idtype=torch.int64, device=sg.device)
                else:
                    t = dgl.graph(([],[]), num_nodes=sg.num_nodes(), idtype=torch.int64, device=sg.device)
                tree_list.append(t)
            g_tree = dgl.batch(tree_list)
        else:
            g_tree = dgl.graph(([],[]), num_nodes=g.num_nodes(), idtype=torch.int64, device=g.device)

        h_tl = self.tlstm(g_tree, x)

        # --- Readout & Fusion ---
        g.ndata["call"] = h_call
        g.ndata["tl"]   = h_tl
        
        mean_call = dgl.mean_nodes(g, "call")
        mean_tl   = dgl.mean_nodes(g, "tl")
        
        # [差异点] parts 中没有 mean_host
        parts = [mean_call, mean_tl]
        
        if "ctx" in g.ndata and self.ctx_mlp is not None:
            ctx = dgl.mean_nodes(g, "ctx")
            parts.append(self.ctx_mlp(ctx))

        fused = self.fuse(torch.cat(parts, dim=-1))

        return {
            "logit_bin":  self.head_bin(fused).squeeze(-1),
            "logits_c3":  self.head_c3(fused),
            "logits_type": self.head_type(fused),
        }