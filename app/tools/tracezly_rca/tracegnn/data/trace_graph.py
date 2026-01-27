import os
import pickle as pkl
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import *

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils import *

__all__ = [
    'TraceGraphNodeFeatures',
    'TraceGraphNodeReconsScores',
    'TraceGraphNode',
    'TraceGraphVectors',
    'TraceGraph',
    'TraceGraphIDManager',
    'load_trace_csv',
    'df_to_trace_graphs',
]


SERVICE_ID_YAML_FILE = 'service_id.yml'
OPERATION_ID_YAML_FILE = 'operation_id.yml'
STATUS_ID_YAML_FILE = 'status_id.yml'
FAULT_CATEGORY_YAML_FILE = 'fault_category.yml'


@dataclass
class TraceGraphNodeFeatures(object):
    __slots__ = ['span_count', 'max_latency', 'min_latency', 'avg_latency']

    span_count: int  # number of duplicates in the parent
    avg_latency: float  # for span_count == 1, avg == max == min
    max_latency: float
    min_latency: float


@dataclass
class TraceGraphNodeReconsScores(object):
    # probability of the node
    edge_logit: float
    operation_logit: float

    # probability of the latency
    avg_latency_nstd: float  # (avg_latency - avg_latency_mean) / avg_latency_std


@dataclass
class TraceGraphSpan(object):
    __slots__ = [
        'span_id', 'start_time', 'latency', 'status'
    ]

    span_id: Optional[int]
    start_time: Optional[datetime]
    latency: float
    status: str


@dataclass
class TraceGraphNode(object):
    __slots__ = [
        'node_id', 'service_id', 'operation_id', 'status_id',
        'features', 'children', 'spans', 'scores'
    ]

    node_id: Optional[int]  # the node id of the graph
    service_id: Optional[int]  # the service id
    status_id: Optional[int]
    operation_id: int  # the operation id
    features: TraceGraphNodeFeatures  # the node features
    children: List['TraceGraphNode']  # children nodes
    spans: Optional[List[TraceGraphSpan]]  # detailed spans information (from the original data)
    scores: Optional[TraceGraphNodeReconsScores]

    def __eq__(self, other):
        return other is self

    def __hash__(self):
        return id(self)

    @staticmethod
    def new_sampled(node_id: int,
                    operation_id: int,
                    status_id: int,
                    features: TraceGraphNodeFeatures,
                    scores: Optional[TraceGraphNodeReconsScores] = None
                    ):
        return TraceGraphNode(
            node_id=node_id,
            service_id=None,
            operation_id=operation_id,
            status_id=status_id,
            features=features,
            children=[],
            spans=None,
            scores=scores
        )

    def iter_bfs(self,
                 depth: int = 0,
                 with_parent: bool = False
                 ) -> Generator[
                    Union[
                        Tuple[int, 'TraceGraphNode'],
                        Tuple[int, int, 'TraceGraphNode', 'TraceGraphNode']
                    ],
                    None,
                    None
                ]:
        """Iterate through the nodes in BFS order."""
        if with_parent:
            depth = depth
            level = [(self, None, 0)]

            while level:
                next_level: List[Tuple[TraceGraphNode, TraceGraphNode, int]] = []
                for nd, parent, idx in level:
                    yield depth, idx, nd, parent
                    for c_idx, child in enumerate(nd.children):
                        next_level.append((child, nd, c_idx))
                depth += 1
                level = next_level

        else:
            depth = depth
            level = [self]

            while level:
                next_level: List[TraceGraphNode] = []
                for nd in level:
                    yield depth, nd
                    next_level.extend(nd.children)
                depth += 1
                level = next_level

    def count_nodes(self) -> int:
        ret = 0
        for _ in self.iter_bfs():
            ret += 1
        return ret


@dataclass
class TraceGraphVectors(object):
    """Cached result of `TraceGraph.graph_vectors()`."""
    __slots__ = [
        'u', 'v',
        'node_type',
        'node_depth', 'node_idx',
        'span_count', 'avg_latency', 'max_latency', 'min_latency',
        'node_features', 'status'
    ]

    # note that it is guaranteed that u[i] < v[i], i.e., upper triangle matrix
    u: np.ndarray
    v: np.ndarray

    # node type
    node_type: np.ndarray

    # node depth
    node_depth: np.ndarray

    # node idx
    node_idx: np.ndarray

    # node feature
    span_count: np.ndarray
    avg_latency: np.ndarray
    max_latency: np.ndarray
    min_latency: np.ndarray

    # status
    status: List[str]


@dataclass
class TraceGraph(object):
    __slots__ = [
        'version',
        'trace_id', 'parent_id', 'root', 'node_count', 'max_depth', 'data', 'status', 'anomaly', 'root_cause', 'fault_category'
    ]

    version: int  # version control
    trace_id: Optional[Tuple[int, int]]
    parent_id: Optional[int]
    root: TraceGraphNode
    node_count: Optional[int]
    max_depth: Optional[int]
    anomaly: int  # 0 normal, 1 abnormal
    root_cause: Optional[int]  # root cause of the anomaly
    fault_category: Optional[int]  # fault category of the anomaly
    data: Dict[str, Any]  # any data about the graph
    status: Set[str]

    @staticmethod
    def default_version() -> int:
        return 0x2

    @staticmethod
    def new_sampled(root: TraceGraphNode, node_count: int, max_depth: int):
        return TraceGraph(
            version=TraceGraph.default_version(),
            trace_id=None,
            parent_id=None,
            root=root,
            node_count=node_count,
            max_depth=max_depth,
            data={},
            anomaly=0,
            root_cause=None,
            fault_category=None,
            status=set()
        )

    @property
    def edge_count(self) -> Optional[int]:
        if self.node_count is not None:
            return self.node_count - 1

    def iter_bfs(self,
                 with_parent: bool = False
                 ):
        """Iterate through the nodes in BFS order."""
        yield from self.root.iter_bfs(with_parent=with_parent)

    def merge_spans_and_assign_id(self):
        """
        Merge spans with the same (service, operation) under the same parent,
        and re-assign node IDs.
        """
        node_count = 0
        max_depth = 0

        for depth, parent in self.iter_bfs():
            max_depth = max(max_depth, depth)

            # assign ID to this node
            parent.node_id = node_count
            node_count += 1

            # merge the children of this node
            children = []
            for child in sorted(parent.children, key=lambda o: o.operation_id):
                if children and children[-1].operation_id == child.operation_id:
                    prev_child = children[-1]

                    # merge the features
                    f1, f2 = prev_child.features, child.features
                    f1.span_count += f2.span_count
                    f1.avg_latency += (f2.avg_latency - f1.avg_latency) * (f2.span_count / f1.span_count)
                    f1.max_latency = max(f1.max_latency, f2.max_latency)
                    f1.min_latency = min(f1.min_latency, f2.min_latency)

                    # merge the children
                    if child.children:
                        if prev_child.children:
                            prev_child.children.extend(child.children)
                        else:
                            prev_child.children = child.children

                    # merge the spans
                    if child.spans:
                        if prev_child.spans:
                            prev_child.spans.extend(child.spans)
                        else:
                            prev_child.spans = child.spans
                else:
                    children.append(child)

            # re-assign the merged children
            parent.children = children

        # record node count and depth
        self.node_count = node_count
        self.max_depth = max_depth

    def assign_node_id(self):
        """Assign node IDs to the graph nodes by pre-root order."""
        node_count = 0
        max_depth = 0

        for depth, node in self.iter_bfs():
            max_depth = max(max_depth, depth)

            # assign id to this node
            node.node_id = node_count
            node_count += 1

        # record node count and depth
        self.node_count = node_count
        self.max_depth = max_depth

    def graph_vectors(self):
        # edge index
        u = np.empty([self.edge_count], dtype=np.int64)
        v = np.empty([self.edge_count], dtype=np.int64)

        # node type
        node_type = np.zeros([self.node_count], dtype=np.int64)

        # node depth
        node_depth = np.zeros([self.node_count], dtype=np.int64)

        # node idx
        node_idx = np.zeros([self.node_count], dtype=np.int64)

        # node feature
        span_count = np.zeros([self.node_count], dtype=np.int64)
        avg_latency = np.zeros([self.node_count], dtype=np.float32)
        max_latency = np.zeros([self.node_count], dtype=np.float32)
        min_latency = np.zeros([self.node_count], dtype=np.float32)
        
        # status
        status = [''] * self.node_count

        # X = np.zeros([self.node_count, x_dim], dtype=np.float32)

        edge_idx = 0
        for depth, idx, node, parent in self.iter_bfs(with_parent=True):
            j = node.node_id
            feat = node.features

            # node type
            node_type[j] = node.operation_id

            # node depth
            node_depth[j] = depth

            # node idx
            node_idx[j] = idx

            # node feature
            span_count[j] = feat.span_count
            avg_latency[j] = feat.avg_latency
            max_latency[j] = feat.max_latency
            min_latency[j] = feat.min_latency
            # X[parent.node_id, parent.operation_id] = 1   # one-hot encoded node feature
            status[j] = node.spans[0].status

            # edge index
            for child in node.children:
                u[edge_idx] = node.node_id
                v[edge_idx] = child.node_id
                edge_idx += 1

        if len(u) != self.edge_count:
            raise ValueError(f'`len(u)` != `self.edge_count`: {len(u)} != {self.edge_count}')

        return TraceGraphVectors(
            # edge index
            u=u, v=v,
            # node type
            node_type=node_type,
            # node depth
            node_depth=node_depth,
            # node idx
            node_idx=node_idx,
            # node feature
            span_count=span_count,
            avg_latency=avg_latency,
            max_latency=max_latency,
            min_latency=min_latency,
            status=status
        )

    def networkx_graph(self, id_manager: 'TraceGraphIDManager') -> nx.Graph:
        gv = self.graph_vectors()
        self_nodes = {nd.node_id: nd for _, nd in self.iter_bfs()}
        g = nx.Graph()
        # graph
        for k, v in self.data.items():
            g.graph[k] = v
        # nodes
        g.add_nodes_from(range(self.node_count))
        # edges
        g.add_edges_from([(i, j) for i, j in zip(gv.u, gv.v)])
        # node features
        for i in range(len(gv.node_type)):
            nd = g.nodes[i]
            nd['anomaly'] = self_nodes[i].anomaly
            nd['status'] = gv.status[i]
            nd['node_type'] = gv.node_type[i]
            nd['service_id'] = self_nodes[i].service_id
            nd['operation'] = id_manager.operation_id.rev(gv.node_type[i])
            for attr in TraceGraphNodeFeatures.__slots__:
                nd[attr] = getattr(gv, attr)[i]
            if self_nodes[i].scores:
                nd['avg_latency_nstd'] = self_nodes[i].scores.avg_latency_nstd
        return g

    def to_bytes(self, protocol: int = pkl.DEFAULT_PROTOCOL) -> bytes:
        return pkl.dumps(self, protocol=protocol)

    @staticmethod
    def from_bytes(content: bytes) -> 'TraceGraph':
        r = pkl.loads(content)
        return r

    def deepcopy(self) -> 'TraceGraph':
        return TraceGraph.from_bytes(self.to_bytes())


@dataclass
class TempGraphNode(object):
    __slots__ = ['trace_id', 'parent_id', 'node']

    trace_id: Tuple[int, int]
    parent_id: int
    node: 'TraceGraphNode'


class TraceGraphIDManager(object):
    __slots__ = ['root_dir', 'service_id', 'operation_id', 'status_id', 'fault_category']

    root_dir: str
    service_id: IDAssign
    operation_id: IDAssign
    status_id: IDAssign
    fault_category: IDAssign

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.service_id = IDAssign(os.path.join(self.root_dir, SERVICE_ID_YAML_FILE))
        self.operation_id = IDAssign(os.path.join(self.root_dir, OPERATION_ID_YAML_FILE))
        self.status_id = IDAssign(os.path.join(self.root_dir, STATUS_ID_YAML_FILE))
        self.fault_category = IDAssign(os.path.join(self.root_dir, FAULT_CATEGORY_YAML_FILE))

    def __enter__(self):
        self.service_id.__enter__()
        self.operation_id.__enter__()
        self.status_id.__enter__()
        self.fault_category.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.service_id.__exit__(exc_type, exc_val, exc_tb)
        self.operation_id.__exit__(exc_type, exc_val, exc_tb)
        self.status_id.__exit__(exc_type, exc_val, exc_tb)
        self.fault_category.__exit__(exc_type, exc_val, exc_tb)

    @property
    def num_operations(self) -> int:
        return len(self.operation_id)

    @property
    def num_services(self) -> int:
        return len(self.service_id)

    @property
    def num_status(self) -> int:
        return len(self.status_id)

    @property
    def num_fault_categories(self) -> int:
        return len(self.fault_category)

    def dump_to(self, output_dir: str):
        self.service_id.dump_to(os.path.join(output_dir, SERVICE_ID_YAML_FILE))
        self.operation_id.dump_to(os.path.join(output_dir, OPERATION_ID_YAML_FILE))
        self.status_id.dump_to(os.path.join(output_dir, STATUS_ID_YAML_FILE))
        self.fault_category.dump_to(os.path.join(output_dir, FAULT_CATEGORY_YAML_FILE))


def load_trace_csv(input_path: str) -> pd.DataFrame:
    dtype = {
        'TraceID': str,
        'SpanID': str,
        'ParentID': str,
        'OperationName': str,
        'ServiceName': str,
        'StartTime': int,
        'Duration': float,
        'StatusCode': str,
        'Anomaly': bool,
        'RootCause': str,
        'FaultCategory': str
    }
    return pd.read_csv(
        input_path,
        engine='c',
        usecols=list(dtype),
        dtype=dtype
    )


def df_to_trace_graphs(
    df: pd.DataFrame,
    id_manager: TraceGraphIDManager,
    min_node_count: int = 2,
    max_node_count: int = 100,
    summary_file: Optional[str] = None,
    merge_spans: bool = False,
) -> List[TraceGraph]:
    summary = []
    trace_spans = {}
    trace_info = {}

    with id_manager:
        for i, row in enumerate(tqdm(df.itertuples(), desc='Build nodes', total=len(df))):
            trace_id = row.TraceID
            span_id = row.SpanID
            parent_span_id = row.ParentID
            service_name = row.ServiceName
            operation_name = row.OperationName
            status_code = row.StatusCode

            span_dict = trace_spans.get(trace_id, None)
            if span_dict is None:
                trace_spans[trace_id] = span_dict = {}
                trace_info[trace_id] = {
                    'anomaly': getattr(row, 'Anomaly', False),
                    'root_cause': getattr(row, 'RootCause', ''),  # 默认 ''
                    'fault_category': getattr(row, 'FaultCategory', ''),  # 默认 ''
                }

            span_dict[span_id] = TempGraphNode(
                trace_id=trace_id,
                parent_id=parent_span_id,
                node=TraceGraphNode(
                    node_id=None,
                    service_id=id_manager.service_id.get_or_assign(service_name),
                    operation_id=id_manager.operation_id.get_or_assign(f'{service_name}/{operation_name}'),
                    status_id=id_manager.status_id.get_or_assign(str(status_code)),
                    features=TraceGraphNodeFeatures(
                        span_count=1,
                        avg_latency=row.Duration,
                        max_latency=row.Duration,
                        min_latency=row.Duration,
                    ),
                    children=[],
                    spans=[
                        TraceGraphSpan(
                            span_id=span_id,
                            start_time=row.StartTime,
                            latency=row.Duration,
                            status=str(status_code)
                        ),
                    ],
                    scores=None
                )
            )

    trace_graphs: List[TraceGraph] = []

    for trace_id, trace in tqdm(trace_spans.items(), total=len(trace_spans), desc='Build graphs'):
        nodes: List[TempGraphNode] = sorted(
            trace.values(),
            key=(lambda nd: (nd.node.service_id, nd.node.operation_id, nd.node.spans[0].start_time))
        )

        if len(nodes) < min_node_count or len(nodes) > max_node_count:
            continue

        status = set()
        info = trace_info[trace_id]

        # 构建树结构
        root_nodes = []
        non_root_nodes = []

        for nd in nodes:
            parent_id = nd.parent_id
            # 处理根节点：parent_id为'-1'、'0'、0、-1或不在trace中
            if (parent_id == '-1') or (parent_id == '0') or (parent_id == 0) or (parent_id == -1) or (parent_id not in trace):
                root_nodes.append(nd)
            else:
                non_root_nodes.append(nd)
                if parent_id in trace:
                    trace[parent_id].node.children.append(nd.node)
                status.update(span.status for span in nd.node.spans)

        if not root_nodes:
            continue

        primary_root = min(root_nodes, key=lambda nd: nd.node.spans[0].start_time)
        for root_nd in root_nodes:
            if root_nd != primary_root:
                primary_root.node.children.append(root_nd.node)
                status.update(span.status for span in root_nd.node.spans)

        # ✅ 正确：使用 id_manager.service_id
        root_cause_raw = info['root_cause']
        if isinstance(root_cause_raw, str) and '-' in root_cause_raw:
            root_cause_key = root_cause_raw.split('-')[0]
        else:
            root_cause_key = str(root_cause_raw) if root_cause_raw is not None else ''
        root_cause_id = id_manager.service_id.get_or_assign(root_cause_key)

        # ✅ 正确：使用 id_manager.fault_category
        fault_category_raw = info['fault_category']
        fault_category_key = str(fault_category_raw) if fault_category_raw is not None else ''
        fault_category_id = id_manager.fault_category.get_or_assign(fault_category_key)

        # ✅ 创建图
        trace_graph = TraceGraph(
            version=TraceGraph.default_version(),
            trace_id=trace_id,
            parent_id=primary_root.parent_id,
            root=primary_root.node,
            node_count=None,
            max_depth=None,
            data={},
            status=status.union(span.status for span in primary_root.node.spans),
            anomaly=1 if info['anomaly'] else 0,
            root_cause=root_cause_id,
            fault_category=fault_category_id
        )

        trace_graphs.append(trace_graph)

    # assign node id
    for trace in tqdm(trace_graphs, desc='Assign node id'):
        trace.assign_node_id()

    # 🔁 DEBUG：验证所有 trace_graph 的 root_cause 和 fault_category 都是 int 且非 None
    print("\n🔍 Running debug validation on root_cause and fault_category...")
    invalid_count = 0
    for i, graph in enumerate(trace_graphs):
        # 检查 root_cause
        if graph.root_cause is None:
            print(f"❌ [TraceGraph {i}] root_cause is None (trace_id={graph.trace_id})")
            invalid_count += 1
        elif not isinstance(graph.root_cause, (int,)):
            print(f"❌ [TraceGraph {i}] root_cause is not int: {graph.root_cause} (type={type(graph.root_cause)}, trace_id={graph.trace_id})")
            invalid_count += 1

        # 检查 fault_category
        if graph.fault_category is None:
            print(f"❌ [TraceGraph {i}] fault_category is None (trace_id={graph.trace_id})")
            invalid_count += 1
        elif not isinstance(graph.fault_category, (int,)):
            print(f"❌ [TraceGraph {i}] fault_category is not int: {graph.fault_category} (type={type(graph.fault_category)}, trace_id={graph.trace_id})")
            invalid_count += 1

    if invalid_count == 0:
        print(f"✅ All {len(trace_graphs)} TraceGraphs passed validation: root_cause and fault_category are all valid integers.")
    else:
        raise ValueError(f"❌ Validation failed: {invalid_count} graphs have invalid root_cause or fault_category!")

    return trace_graphs
