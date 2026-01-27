import torch
import dgl

from tracegnn.constants import MAX_SPAN_COUNT

from .trace_graph import *


def graph_to_dgl(g: TraceGraph) -> dgl.DGLGraph:
    # edge index
    u = torch.empty([g.edge_count], dtype=torch.long)
    v = torch.empty([g.edge_count], dtype=torch.long)

    # operation, service
    operation_id = torch.zeros([g.node_count], dtype=torch.long)
    service_id = torch.zeros([g.node_count], dtype=torch.long)

    # node depth
    node_depth = torch.zeros([g.node_count], dtype=torch.long)

    # node idx
    node_idx = torch.zeros([g.node_count], dtype=torch.long)

    # node feature
    span_count = torch.zeros([g.node_count], dtype=torch.long)
    latency = torch.zeros([g.node_count], dtype=torch.float)

    # status
    status = torch.zeros([g.node_count], dtype=torch.long)

    # start time
    start_time = torch.zeros([g.node_count], dtype=torch.int64)

    # Iterate
    edge_idx = 0
    for depth, idx, node, parent in g.iter_bfs(with_parent=True):
        j = node.node_id
        feat = node.features

        # node type
        operation_id[j] = node.operation_id
        service_id[j] = node.service_id

        # node depth
        node_depth[j] = depth

        # node idx
        node_idx[j] = idx

        # node feature
        span_count[j] = feat.span_count
        latency[j] = feat.avg_latency

        # status
        status[j] = node.status_id if isinstance(node.status_id, int) else 0

        # edge index
        for child in node.children:
            u[edge_idx] = node.node_id
            v[edge_idx] = child.node_id
            edge_idx += 1

        # start time
        st = node.spans[0].start_time
        if isinstance(st, int):
            start_time[j] = st
        elif hasattr(st, "timestamp"):
            start_time[j] = int(st.timestamp())
        else:
            start_time[j] = 0

    if len(u) != g.edge_count:
        raise ValueError(f'`len(u)` != `g.edge_count`: {len(u)} != {g.edge_count}')

    # Return dgl graph
    dgl_graph: dgl.DGLGraph = dgl.graph((u, v), num_nodes=g.node_count)
    dgl_graph = dgl.add_self_loop(dgl_graph)

    # Add features
    dgl_graph.ndata['operation_id'] = operation_id
    dgl_graph.ndata['service_id'] = service_id
    dgl_graph.ndata['node_idx'] = node_idx
    dgl_graph.ndata['node_depth'] = node_depth
    dgl_graph.ndata['span_count'] = torch.minimum(span_count, torch.tensor(MAX_SPAN_COUNT)).long()
    dgl_graph.ndata['latency'] = latency
    dgl_graph.ndata['status_id'] = status
    dgl_graph.ndata['start_time'] = start_time

    # 图级属性
    dgl_graph.anomaly = g.anomaly
    dgl_graph.root_cause = g.root_cause
    dgl_graph.fault_category = g.fault_category

    return dgl_graph