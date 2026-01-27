from typing import *

from tracegnn.models.gtrace.models.mymodel import MyTraceAnomalyModel, construct_neighbor_dict
from .config import ExpConfig
from tracegnn.utils.analyze_root_cause import evaluate_with_root_cause, evaluate_mixed_root_cause_joint
from tracegnn.utils.trace_host_fusion import (
    load_host_topology,
    load_host_infra_index,
    evaluate_with_host_topology_infra,
    evaluate_mixed_root_cause_urs,
    evaluate_mixed_root_cause_rerank,
)
from tracegnn.data.trace_graph import TraceGraphIDManager

import dgl
from loguru import logger
import torch
import numpy as np
from .utils import dgl_graph_key
from tracegnn.utils.analyze_nll import analyze_anomaly_scores



@torch.no_grad()
def evaluate(config: ExpConfig, 
             dataloader: dgl.dataloading.GraphDataLoader, 
             model: MyTraceAnomalyModel, 
             epoch: Optional[int] = None,
             # ✅ 新增下面这三个参数，默认值为 None
             id_manager=None,
             host_adj=None,
             infra_index=None):
    """
    Evaluate MyTraceAnomalyModel (基于 total_loss / structure_loss / latency_loss)
    同时进行根因定位分析
    """
    device = config.device
    n_z = config.Model.n_z

    # Train model
    logger.info('Start Evaluation with nll...')
    model.eval()

    anomaly_score_list = []
    graph_label_list = []
    all_trace_info = []
    true_root_causes_dict = {}

    # Optional limit on the total number of traces to evaluate
    try:
        max_eval_traces = int(getattr(config, 'max_eval_traces', 0) or 0)
    except Exception:
        max_eval_traces = 0

    with torch.no_grad():
        # Prepare id_manager and optional host topology / fault category mapping
        import os
        from datetime import datetime
        processed_dir = os.path.join(config.dataset_root_dir, config.dataset, 'processed')
        # ✅ 修改点 1：如果外部没传 id_manager，才自己加载
        if id_manager is None:
            id_manager = TraceGraphIDManager(processed_dir)

        # fault_category id -> name
        fc_name_map: Dict[int, str] = {}
        try:
            for i in range(len(id_manager.fault_category)):
                try:
                    fc_name_map[i] = id_manager.fault_category.rev(i)
                except Exception:
                    pass
        except Exception:
            pass
        # load host topology and infra index (optional)
        # ✅ 修改点 2：如果外部没传 host_adj，才自己加载
        if host_adj is None:
            host_adj = load_host_topology(processed_dir, id_manager)
        
        # ✅ 修改点 3：如果外部没传 infra_index，才自己加载 (这就是你最耗时的那个)
        if infra_index is None:
            infra_index = load_host_infra_index(processed_dir)
        try:
            debug_flag = bool(getattr(config.RCA, 'export_debug', False))
        except Exception:
            debug_flag = False
        if debug_flag:
            if infra_index is None:
                logger.info('Infra index: not found')
            else:
                logger.info(f'Infra index: loaded for {len(infra_index)} hosts')
        t = dataloader
        if config.enable_tqdm:
            from tqdm import tqdm
            t = tqdm(dataloader)

        for batch_idx, batch in enumerate(t):
            # batch: (graphs, labels)
            if isinstance(batch, (tuple, list)):
                if len(batch) >= 5:
                    test_graphs, graph_anomaly_labels, root_causes, fault_categories, batch_trace_ids = batch
                else:
                    test_graphs, graph_anomaly_labels, root_causes, fault_categories = batch
                    batch_trace_ids = None
            else:
                test_graphs = batch
                graph_anomaly_labels = batch.ndata['label'] if 'label' in batch.ndata else None

            # Empty cache first
            if 'cuda' in config.device:
                torch.cuda.empty_cache()

            test_graphs = test_graphs.to(device)
            if graph_anomaly_labels is not None:
                graph_anomaly_labels = graph_anomaly_labels.to(device)

            test_graph_list: List[dgl.DGLGraph] = dgl.unbatch(test_graphs)

            for i, single_test_graph in enumerate(test_graph_list):
                # Respect global evaluation trace limit if configured
                if max_eval_traces > 0 and len(all_trace_info) >= max_eval_traces:
                    break
                # Prefer original trace_id passed from dataset; then graph attribute; fallback to structural hash key
                graph_key = None
                try:
                    if batch_trace_ids is not None:
                        raw_tid = batch_trace_ids[i]
                        graph_key = str(raw_tid)
                except Exception:
                    graph_key = None
                if not graph_key:
                    try:
                        _orig_tid = getattr(single_test_graph, 'trace_id', None)
                        graph_key = str(_orig_tid) if _orig_tid is not None else None
                    except Exception:
                        graph_key = None
                if not graph_key:
                    graph_key = dgl_graph_key(single_test_graph)
                single_graph_anomaly_label = (
                    graph_anomaly_labels[i].item() if graph_anomaly_labels is not None else
                    (single_test_graph.graph_label if hasattr(single_test_graph, 'graph_label') else 0)
                )

                # 获取真实根因 / 故障类别
                groundtruth_root_cause = None
                if isinstance(batch, (tuple, list)) and len(batch) > 2:
                    # 从数据加载器的批次数据中获取root_cause
                    groundtruth_root_cause = root_causes[i].item() if hasattr(root_causes[i], 'item') else root_causes[i]
                elif hasattr(single_test_graph, 'root_cause'):
                    # 从图对象属性中获取root_cause
                    groundtruth_root_cause = single_test_graph.root_cause

                fault_category_val = None
                if isinstance(batch, (tuple, list)) and len(batch) > 3:
                    fault_category_val = fault_categories[i].item() if hasattr(fault_categories[i], 'item') else fault_categories[i]
                elif hasattr(single_test_graph, 'fault_category'):
                    fault_category_val = single_test_graph.fault_category

                # 只有在是异常trace且gt有效(>0)时才纳入RCA评估
                if int(single_graph_anomaly_label) > 0:
                    try:
                        gt_int = int(groundtruth_root_cause) if groundtruth_root_cause is not None else 0
                    except Exception:
                        gt_int = 0
                    if gt_int > 0:
                        true_root_causes_dict[graph_key] = gt_int
                    else:
                        # 补充输出：trace key、gt原值、故障类别及其名称（若可解析）
                        fc_name = None
                        try:
                            if fault_category_val is not None:
                                fc_name = fc_name_map.get(int(fault_category_val), None)
                        except Exception:
                            fc_name = None
                        try:
                            trace_ident = getattr(single_test_graph, 'trace_id', None)
                        except Exception:
                            trace_ident = None
                        rca_cfg = getattr(config, 'RCA', None)
                        if not bool(getattr(rca_cfg, 'suppress_missing_gt_warning', True)):
                            logger.warning(
                                f"Abnormal trace missing/invalid GT. key={graph_key}, trace_id={trace_ident}, gt={groundtruth_root_cause}, "
                                f"fault_category={fault_category_val}{'('+str(fc_name)+')' if fc_name else ''}. Skipped in RCA.")

                # 构造邻接矩阵
                # Build edge_index to avoid torch_sparse spmm on GPU; use scatter path in PyG
                # 边提取改为在CPU端进行，避免GPU边导出触发非法访问
                graph_cpu = single_test_graph if 'cuda' not in str(device) else single_test_graph.to('cpu')
                u, v = graph_cpu.edges()
                # 构造供 GIN 使用的 edge_index（设备端）
                u_dev = u.to(device).long().contiguous()
                v_dev = v.to(device).long().contiguous()
                edge_index = torch.stack([u_dev, v_dev], dim=0)
                N = int(single_test_graph.num_nodes())
                # 在 CPU 端计算 degree，避免 GPU 稀疏内核异常
                v_cpu = v.long().contiguous().cpu()
                if v_cpu.numel() == 0:
                    degree = torch.zeros(N, dtype=torch.long, device=device)
                else:
                    vmax = int(v_cpu.max().item())
                    if vmax >= N:
                        deg_full = torch.bincount(v_cpu, minlength=vmax + 1)
                        degree = torch.zeros(N, dtype=deg_full.dtype, device=device)
                        degree[:min(N, deg_full.numel())] = deg_full[:min(N, deg_full.numel())]
                    else:
                        degree = torch.bincount(v_cpu, minlength=N).to(device)
                neighbor_dict = construct_neighbor_dict((edge_index.cpu(), N))

                # 运行模型
                pred = model(single_test_graph, edge_index, degree, neighbor_dict, n_z=n_z)

                # 计算nll
                # 结构NLL就是loss_structure
                nll_structure = pred['loss_structure']
                # 延迟NLL就是loss_latency
                nll_latency = pred['loss_latency']

                # 组合结构/延迟 NLL；评测阶段可固定 alpha/beta（若在 config.RCA 中提供 eval_alpha/eval_beta）
                rca_cfg = getattr(config, 'RCA', None)
                alpha_override = getattr(rca_cfg, 'eval_alpha', None)
                beta_override = getattr(rca_cfg, 'eval_beta', None)
                alpha_w = float(alpha_override) if alpha_override is not None else float(pred['alpha'])
                beta_w = float(beta_override) if beta_override is not None else float(pred['beta'])
                weighted_structure = float((alpha_w * nll_structure).item())
                weighted_latency = float((beta_w * nll_latency).item())

                # 节点级分数（用于 RCA 可视化，也可用于 anomaly detection 的 topK 聚合）
                if 'node_structure_scores' in pred and 'node_latency_scores' in pred:
                    # 节点级分数：先做每条 trace 内的稳健标准化，再按评测权重组合
                    def _zscore(x: torch.Tensor) -> torch.Tensor:
                        m = torch.mean(x)
                        s = torch.std(x)
                        return (x - m) / (s + 1e-6)

                    node_struct = _zscore(pred['node_structure_scores'])
                    node_lat    = _zscore(pred['node_latency_scores'])
                    alpha_override = getattr(getattr(config, 'RCA', None), 'eval_alpha', None)
                    beta_override  = getattr(getattr(config, 'RCA', None), 'eval_beta', None)
                    alpha_w = float(alpha_override) if alpha_override is not None else float(pred['alpha'])
                    beta_w  = float(beta_override)  if beta_override  is not None else float(pred['beta'])
                    combined_node_scores = alpha_w * node_struct + beta_w * node_lat

                    # 在服务级聚合阶段再做归一化，这里保留节点级组合分数
                    single_test_graph.ndata['node_anomaly_score'] = combined_node_scores
                    single_test_graph.ndata['node_structure_score'] = pred['node_structure_scores']
                    single_test_graph.ndata['node_latency_score'] = pred['node_latency_scores']
                    current_node_scores = combined_node_scores
                else:
                    logger.warning(f"Graph {graph_key}: No node scores in prediction output.")
                    current_node_scores = torch.zeros(single_test_graph.num_nodes(), device=device)

                # ---------------- Anomaly detection graph-level score (configurable) ----------------
                # NOTE: Kendall weights (alpha/beta/gamma) are learned for balancing TRAINING losses,
                # but anomaly detection benefits from a score definition that matches the dataset's signal.
                mode = str(getattr(rca_cfg, 'detector_score_mode', 'sl')).lower() if rca_cfg is not None else 'sl'
                topk = int(getattr(rca_cfg, 'detector_topk', 5)) if rca_cfg is not None else 5
                topk = max(1, topk)

                # Host channel contribution (optional)
                host_loss_val = 0.0
                try:
                    if isinstance(pred, dict) and pred.get('loss_host', None) is not None:
                        v = pred.get('loss_host')
                        host_loss_val = float(v.item() if hasattr(v, 'item') else float(v))
                except Exception:
                    host_loss_val = 0.0

                gamma_override = getattr(rca_cfg, 'eval_gamma', None) if rca_cfg is not None else None
                try:
                    gamma_w = float(gamma_override) if gamma_override is not None else float(pred.get('gamma', 0.0))
                except Exception:
                    gamma_w = 0.0

                # node topK aggregation (if available)
                node_topk_score = 0.0
                try:
                    if current_node_scores is not None and int(single_test_graph.num_nodes()) > 0:
                        k = min(int(single_test_graph.num_nodes()), topk)
                        # Use largest scores; mean over topK for stability.
                        vals, _ = torch.topk(current_node_scores.float(), k=k, largest=True)
                        node_topk_score = float(vals.mean().item())
                except Exception:
                    node_topk_score = 0.0

                if mode == 'sl':
                    anomaly_score = weighted_structure + weighted_latency
                elif mode == 'slh':
                    anomaly_score = weighted_structure + weighted_latency + gamma_w * host_loss_val
                elif mode == 'topk_sl':
                    anomaly_score = node_topk_score
                elif mode == 'topk_slh':
                    anomaly_score = node_topk_score + gamma_w * host_loss_val
                else:
                    # Backward-compatible fallback
                    anomaly_score = weighted_structure + weighted_latency

                anomaly_score_list.append(float(anomaly_score))
                graph_label_list.append(int(single_graph_anomaly_label))

                # 收集trace信息
                trace_info = {
                    'graph': single_test_graph,
                    'trace_id': graph_key,
                    'is_anomalous': int(single_graph_anomaly_label) > 0,
                    'node_scores': current_node_scores,
                    'fault_category': fault_category_val,
                }
                # 透传 HostNLL（若模型提供）
                try:
                    if isinstance(pred, dict) and ('host_nll_dict' in pred) and (pred['host_nll_dict'] is not None):
                        trace_info['host_nll'] = pred['host_nll_dict']
                except Exception:
                    pass
                all_trace_info.append(trace_info)

            # Stop consuming further batches if we've reached the configured maximum
            if max_eval_traces > 0 and len(all_trace_info) >= max_eval_traces:
                break

        # Convert to numpy arrays
        anomaly_score_array = np.array(anomaly_score_list, dtype=np.float32)
        graph_label_array = np.array(graph_label_list, dtype=np.int64)

        # Debug information
        # logger.debug(f'Combined NLL range: [{combined_nll_array.min():.2f}, {combined_nll_array.max():.2f}]')
        # logger.debug(f'Graph labels distribution: {np.bincount(graph_label_array)}')

        # Check for any abnormally large NLL values
        # normal_nll_values = anomaly_score_array[graph_label_array == 0]
        # if len(normal_nll_values) > 0:
        #     logger.debug(f'Normal NLL range: [{normal_nll_values.min():.2f}, {normal_nll_values.max():.2f}]')
        #     logger.debug(f'Normal NLL mean: {np.mean(normal_nll_values):.2f}')

        # Set evaluation output
        logger.info('-------------------Graph Level Overall-----------------------')
        # Get overall graph level result
        overall_result = analyze_anomaly_scores(
            score_list=anomaly_score_array,
            label_list=graph_label_array
        )
        logger.info(overall_result)

        # Compute confusion matrix (TP/FP/TN/FN) on test set using the chosen threshold
        # Prefer best_threshold (max F1) to improve F-score; fallback to MAD-based 'threshold'
        try:
            thr = float(overall_result.get('best_threshold', overall_result.get('threshold', 0.0)))
        except Exception:
            thr = 0.0
        y_true = (graph_label_array > 0).astype(int)
        y_pred = (anomaly_score_array > thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        # Build RCA trace list according to filter mode ('gt'/'pred'/'all')
        rca_filter_mode = str(getattr(getattr(config, 'RCA', None), 'rca_filter', 'gt'))
        pred_flags = {all_trace_info[i]['trace_id']: bool(y_pred[i]) for i in range(len(all_trace_info))}
        rca_traces: List[Dict[str, Any]] = []
        for rec in all_trace_info:
            if rca_filter_mode == 'gt':
                use = bool(rec.get('is_anomalous', False))
            elif rca_filter_mode == 'pred':
                use = bool(pred_flags.get(rec['trace_id'], False))
            else:
                use = True
            if use:
                r = dict(rec)
                r['is_anomalous'] = True
                rca_traces.append(r)

        # 根因定位评估（服务级 / 主机级 / 混合级）
        if rca_traces:
            logger.info("-------------------Root Cause Analysis-----------------------")
            # RCA params
            rca_cfg = getattr(config, 'RCA', None)
            rca_topk = int(getattr(rca_cfg, 'topk', 5))
            svc_pool_k = int(getattr(rca_cfg, 'svc_pool_k', 2))
            host_pool_k = int(getattr(rca_cfg, 'host_pool_k', 3))
            lambda_host = float(getattr(rca_cfg, 'lambda_host', 1.0))
            alpha = float(getattr(rca_cfg, 'alpha', 0.7))
            steps = int(getattr(rca_cfg, 'steps', 1))
            eval_k = min(rca_topk, 5)

            # 方便后续按 trace_id 反查原始图信息
            rca_trace_map: Dict[str, Dict[str, Any]] = {r['trace_id']: r for r in rca_traces}

            # 仅在“服务类故障”子集上评估服务级RCA，避免与主机类样本混入
            def _is_host_fault(trace_rec: Dict[str, Any]) -> bool:
                fc_id = trace_rec.get('fault_category', None)
                if fc_id is not None and fc_name_map:
                    try:
                        name = fc_name_map.get(int(fc_id), '').lower()
                        return name.startswith('node') or ('host' in name)
                    except Exception:
                        return False
                # 退化：若无法解析类别，则尝试用 gt 是否属于 host_id 集判断
                try:
                    gt_val = int(true_root_causes_dict.get(trace_rec['trace_id'], 0))
                    hids = trace_rec['graph'].ndata['host_id'].detach().cpu().numpy().tolist()
                    return gt_val in set(map(int, hids))
                except Exception:
                    return False

            # 计算 AVG5（MAP@5 风格）辅助函数
            def _compute_svc_avg5(results: List[Dict[str, Any]], topk: int = 5) -> float:
                scores: List[float] = []
                for r in results:
                    gt = r.get('groundtruth')
                    try:
                        gt_int = int(gt) if gt is not None else 0
                    except Exception:
                        gt_int = 0
                    if gt_int <= 0:
                        continue
                    cands = r.get('top_candidates') or []
                    svc_ids: List[int] = []
                    for sid, _score in cands:
                        try:
                            s_int = int(sid)
                        except Exception:
                            continue
                        if s_int not in svc_ids:
                            svc_ids.append(s_int)
                        if len(svc_ids) >= topk:
                            break
                    if not svc_ids:
                        scores.append(0.0)
                        continue
                    rank = None
                    for idx, sid in enumerate(svc_ids):
                        if sid == gt_int:
                            rank = idx + 1
                            break
                    if rank is None or rank > topk:
                        scores.append(0.0)
                    else:
                        scores.append(1.0 / float(rank))
                return float(np.mean(scores)) if scores else 0.0

            def _compute_host_avg5(results: List[Dict[str, Any]], topk: int = 5) -> float:
                scores: List[float] = []
                for r in results:
                    gt = r.get('groundtruth')
                    try:
                        gt_int = int(gt) if gt is not None else 0
                    except Exception:
                        gt_int = 0
                    if gt_int <= 0:
                        continue
                    cands = r.get('top_host') or []
                    host_ids: List[int] = []
                    for hid, _score in cands:
                        try:
                            h_int = int(hid)
                        except Exception:
                            continue
                        host_ids.append(h_int)
                        if len(host_ids) >= topk:
                            break
                    if not host_ids:
                        scores.append(0.0)
                        continue
                    rank = None
                    for idx, hid in enumerate(host_ids):
                        if hid == gt_int:
                            rank = idx + 1
                            break
                    if rank is None or rank > topk:
                        scores.append(0.0)
                    else:
                        scores.append(1.0 / float(rank))
                return float(np.mean(scores)) if scores else 0.0

            def _compute_mixed_avg5(results: List[Dict[str, Any]], topk: int = 5) -> float:
                scores: List[float] = []
                for r in results:
                    tid = r.get('trace_id')
                    gt = r.get('groundtruth')
                    try:
                        gt_int = int(gt) if gt is not None else 0
                    except Exception:
                        gt_int = 0
                    if gt_int <= 0 or tid is None:
                        continue
                    # 判定该样本是主机类故障还是服务类故障
                    is_host_fault = False
                    src = rca_trace_map.get(tid)
                    if src is not None:
                        try:
                            is_host_fault = _is_host_fault(src)
                        except Exception:
                            is_host_fault = False
                    cands = r.get('top_mixed') or []
                    ids: List[int] = []
                    for kind, ident, _score in cands:
                        if is_host_fault and kind != 'host':
                            continue
                        if (not is_host_fault) and kind != 'svc':
                            continue
                        try:
                            i_int = int(ident)
                        except Exception:
                            continue
                        ids.append(i_int)
                        if len(ids) >= topk:
                            break
                    if not ids:
                        scores.append(0.0)
                        continue
                    rank = None
                    for idx, i_val in enumerate(ids):
                        if i_val == gt_int:
                            rank = idx + 1
                            break
                    if rank is None or rank > topk:
                        scores.append(0.0)
                    else:
                        scores.append(1.0 / float(rank))
                return float(np.mean(scores)) if scores else 0.0

            # 服务级 RCA：仅在“服务类故障”子集上评估
            svc_only_traces: List[Dict[str, Any]] = []
            svc_only_gt: Dict[str, Any] = {}
            for rec in rca_traces:
                if not rec.get('is_anomalous', False):
                    continue
                tid = rec['trace_id']
                gt = true_root_causes_dict.get(tid, None)
                if gt is None:
                    continue
                if not _is_host_fault(rec):
                    svc_only_traces.append(rec)
                    svc_only_gt[tid] = gt

            root_cause_results, acc_top1, acc_top5 = evaluate_with_root_cause(
                svc_only_traces,
                true_root_causes=svc_only_gt,
                topk=rca_topk,
                svc_pool_k=int(getattr(rca_cfg, 'svc_pool_k', 3)),
                svc_len_p=float(getattr(rca_cfg, 'svc_len_p', 0.25)),
                svc_tau=float(getattr(rca_cfg, 'svc_tau', 0.8)),
            )
            svc_avg5 = _compute_svc_avg5(root_cause_results, topk=eval_k)
            logger.info(f"服务级根因 Top1: {acc_top1:.4f}，Top5: {acc_top5:.4f}，AVG5: {svc_avg5:.4f}")

            # 主机级（物理拓扑融合）：使用共址+拓扑传播进行 host 级评估
            host_kwargs = dict(
                true_root_causes=true_root_causes_dict,
                host_adj=host_adj,
                topk=rca_topk,
                pool_k=host_pool_k,
                alpha=alpha,
                steps=steps,
                fault_category_names=fc_name_map,
                id_manager=id_manager,
                infra_index=infra_index,
                W=int(getattr(rca_cfg, 'W', 3)),
                eta=float(getattr(rca_cfg, 'eta', 0.3)),
                host_nll_key='host_nll',
                lambda_host=float(getattr(rca_cfg, 'lambda_host_host', 0.45)),
                rho_mode=str(getattr(rca_cfg, 'rho_mode', 'count')),
                debug=bool(getattr(rca_cfg, 'export_debug', False)),
            )
            if bool(getattr(rca_cfg, 'enable_host_ms_infra', False)):
                host_kwargs.update(
                    ms_W_list=list(getattr(rca_cfg, 'ms_W_list', [1,5,15])),
                    lse_tau=float(getattr(rca_cfg, 'lse_tau', 2.0)),
                    sinfra_w=dict(getattr(rca_cfg, 'sinfra_w', {'z_point':0.6,'z_win':0.3,'peer':0.1})),
                    peer_mode=str(getattr(rca_cfg, 'peer_mode', 'trace')),
                )
            host_results, acc_host_top1, acc_host_top5 = evaluate_with_host_topology_infra(
                rca_traces, **host_kwargs
            )
            host_avg5 = _compute_host_avg5(host_results, topk=eval_k)
            logger.info(f"主机级根因Top1: {acc_host_top1:.4f}，Top5: {acc_host_top5:.4f}，AVG5: {host_avg5:.4f}")

            # 服务+主机混合 Top-K：按开关选择 rerank 或 base URS 路径
            if bool(getattr(rca_cfg, 'enable_mixed_rerank', False)):
                rerank_kwargs = dict(
                    true_root_causes=true_root_causes_dict,
                    id_manager=id_manager,
                    infra_index=infra_index,
                    topk=rca_topk,
                    svc_pool_k=svc_pool_k,
                    svc_len_p=float(getattr(rca_cfg, 'svc_len_p', 0.25)),
                    svc_tau=float(getattr(rca_cfg, 'svc_tau', 0.8)),
                    host_W=int(getattr(rca_cfg, 'W', 3)),
                    eta=float(getattr(rca_cfg, 'eta', 0.3)),
                    lambda_host=float(getattr(rca_cfg, 'lambda_host_mixed', 0.35)),
                    rho_mode=str(getattr(rca_cfg, 'rho_mode', 'count')),
                    urs_alpha=float(getattr(rca_cfg, 'urs_alpha', 0.6)),
                    gamma_svc=float(getattr(rca_cfg, 'gamma_svc', 0.3)),
                    gamma_host=float(getattr(rca_cfg, 'gamma_host', 0.3)),
                    host_nll_key='host_nll',
                    peer_mode=str(getattr(rca_cfg, 'peer_mode', 'trace')),
                )
                if bool(getattr(rca_cfg, 'enable_host_ms_infra', False)):
                    rerank_kwargs.update(
                        ms_W_list=list(getattr(rca_cfg, 'ms_W_list', [1,5,15])),
                        lse_tau=float(getattr(rca_cfg, 'lse_tau', 2.0)),
                        sinfra_w=dict(getattr(rca_cfg, 'sinfra_w', {'z_point':0.6,'z_win':0.3,'peer':0.1})),
                    )
                rerank_kwargs.update(
                    mixed_pool_svc_k=int(getattr(rca_cfg, 'mixed_pool_svc_k', 3)),
                    mixed_pool_host_k=int(getattr(rca_cfg, 'mixed_pool_host_k', 3)),
                    topo_boost_beta=float(getattr(rca_cfg, 'topo_boost_beta', 0.2)),
                    enable_hetero_prop=bool(getattr(rca_cfg, 'enable_hetero_prop', False)),
                    enable_causal_pruning=bool(getattr(rca_cfg, 'enable_causal_pruning', False)),
                    pruning_threshold=float(getattr(rca_cfg, 'pruning_threshold', 0.01)),
                )
                mixed_results, acc_mix_top1, acc_mix_top5, acc_mix_svc_t1, acc_mix_svc_t5, acc_mix_host_t1, acc_mix_host_t5 = evaluate_mixed_root_cause_rerank(
                    rca_traces, **rerank_kwargs
                )
            else:
                # Determine path for RCA conflicts CSV (same directory as reports)
                try:
                    cfg_dir = getattr(config, 'report_dir', 'reports')
                    reports_dir = cfg_dir if os.path.isabs(cfg_dir) else os.path.join(processed_dir, cfg_dir)
                    os.makedirs(reports_dir, exist_ok=True)
                    conflict_csv_path = os.path.join(reports_dir, 'rca_conflicts.csv')
                except Exception:
                    conflict_csv_path = None

                mixed_results, acc_mix_top1, acc_mix_top5, acc_mix_svc_t1, acc_mix_svc_t5, acc_mix_host_t1, acc_mix_host_t5 = evaluate_mixed_root_cause_urs(
                    rca_traces,
                    true_root_causes=true_root_causes_dict,
                    host_adj=host_adj,
                    topk=rca_topk,
                    svc_pool_k=svc_pool_k,
                    host_pool_k=host_pool_k,
                    alpha=alpha,
                    steps=steps,
                    fault_category_names=fc_name_map,
                    lambda_host=float(getattr(rca_cfg, 'lambda_host_mixed', 0.35)),
                    id_manager=id_manager,
                    infra_index=infra_index,
                    W=int(getattr(rca_cfg, 'W', 3)),
                    urs_alpha=float(getattr(rca_cfg, 'urs_alpha', 0.6)),
                    debug=bool(getattr(rca_cfg, 'export_debug', False)),
                    ms_W_list=(list(getattr(rca_cfg, 'ms_W_list', [1,5,15])) if bool(getattr(rca_cfg, 'enable_host_ms_infra', False)) else None),
                    lse_tau=(float(getattr(rca_cfg, 'lse_tau', 2.0)) if bool(getattr(rca_cfg, 'enable_host_ms_infra', False)) else 2.0),
                    sinfra_w=(dict(getattr(rca_cfg, 'sinfra_w', {'z_point':0.6,'z_win':0.3,'peer':0.1})) if bool(getattr(rca_cfg, 'enable_host_ms_infra', False)) else None),
                    peer_mode=str(getattr(rca_cfg, 'peer_mode', 'trace')),
                    conflict_csv_path=conflict_csv_path,
                    epoch=int(epoch) if epoch is not None else None,
                )
            mixed_avg5 = _compute_mixed_avg5(mixed_results, topk=eval_k)
            logger.info(f"混合根因 Top1: {acc_mix_top1:.4f}，Top5: {acc_mix_top5:.4f}，AVG5: {mixed_avg5:.4f} (svc_t1={acc_mix_svc_t1:.4f}, svc_t5={acc_mix_svc_t5:.4f}; host_t1={acc_mix_host_t1:.4f}, host_t5={acc_mix_host_t5:.4f})")

            # 控制台简化输出：不逐条打印单个 Trace 级HostTop1 结果

            # 展示混合Top1候选（包含类型）
            try:
                service_name = {i: id_manager.service_id.rev(i) for i in range(len(id_manager.service_id))}
            except Exception:
                service_name = {}
            # for rec in mixed_results[:10]:
            #     if rec.get('top_mixed'):
            #         k, i, score = rec['top_mixed'][0]
            #         if k == 'host':
            #             name = host_name.get(int(i), '')
            #             logger.info(f"Trace {rec['trace_id']} MixedTop1: host_id={int(i)}({name}) score={score:.4f} gt={rec['groundtruth']}")
            #         else:
            #             name = service_name.get(int(i), '')
            #             logger.info(f"Trace {rec['trace_id']} MixedTop1: service_id={int(i)}({name}) score={score:.4f} gt={rec['groundtruth']}")

            # ------------------- Persist a detailed report (save_final_only) -----------------------
            try:
                cfg_dir = getattr(config, 'report_dir', 'reports')
                reports_dir = cfg_dir if os.path.isabs(cfg_dir) else os.path.join(processed_dir, cfg_dir)
                os.makedirs(reports_dir, exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                max_epochs = int(getattr(config, 'max_epochs', -1))
                is_last_epoch = (epoch is not None and max_epochs > 0 and int(epoch) == max_epochs - 1)
                is_final_eval = (epoch is None)
                save_final_only = bool(getattr(rca_cfg, 'save_final_only', True))
                should_save = (not save_final_only) or (is_last_epoch or is_final_eval)
                if should_save:
                    if getattr(config, 'include_epoch_in_report_name', True) and epoch is not None:
                        report_name = f'eval_report_epoch{epoch}_{ts}.md'
                    else:
                        report_name = f'eval_report_{ts}.md'
                    report_path = os.path.join(reports_dir, report_name)

                # Build id -> name dicts
                try:
                    service_name = {i: id_manager.service_id.rev(i) for i in range(len(id_manager.service_id))}
                except Exception:
                    service_name = {}
                try:
                    host_name = {i: id_manager.host_id.rev(i) for i in range(len(id_manager.host_id))}
                except Exception:
                    host_name = {}

                # Index RCA results by trace_id for merging
                svc_map = {r['trace_id']: r for r in root_cause_results}
                host_map = {r['trace_id']: r for r in host_results}
                mixed_map = {r['trace_id']: r for r in mixed_results}

                # Choose up to 20 sample traces (prefer abnormal with valid GT)
                sample_count = int(getattr(rca_cfg, 'sample_count', 20))
                sample_ids: List[str] = []
                for tid in true_root_causes_dict.keys():
                    if tid in mixed_map or tid in host_map or tid in svc_map:
                        sample_ids.append(tid)
                    if len(sample_ids) >= sample_count:
                        break
                if not sample_ids:
                    # fallback: take any keys present
                    sample_ids = list(mixed_map.keys())[:sample_count] or list(host_map.keys())[:sample_count] or list(svc_map.keys())[:sample_count]

                lines: List[str] = []
                lines.append('# Evaluation Report')
                lines.append('')
                lines.append(f'- Timestamp: {ts}')
                lines.append(f'- Dataset: {config.dataset} ({config.test_dataset})')
                lines.append(f'- Threshold: {thr:.6f}')
                lines.append(f'- Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}')
                lines.append('')
                lines.append('## Graph-level Metrics')
                for k, v in overall_result.items():
                    try:
                        lines.append(f'- {k}: {v}')
                    except Exception:
                        pass
                lines.append('')
                lines.append('## RCA Metrics')
                lines.append(f'- Service-level: Top1={acc_top1:.4f}, Top5={acc_top5:.4f}, Avg5={svc_avg5:.4f}')
                lines.append(f'- Host-level:    Top1={acc_host_top1:.4f}, Top5={acc_host_top5:.4f}, Avg5={host_avg5:.4f}')
                lines.append(f'- Mixed:         Top1={acc_mix_top1:.4f}, Top5={acc_mix_top5:.4f}, Avg5={mixed_avg5:.4f} (svc_t1={acc_mix_svc_t1:.4f}, svc_t5={acc_mix_svc_t5:.4f}; host_t1={acc_mix_host_t1:.4f}, host_t5={acc_mix_host_t5:.4f})')
                lines.append('')
                lines.append('## Sample Traces (up to 20)')

                def fmt_svc_list(lst):
                    return ', '.join([f"{int(sid)}({service_name.get(int(sid),'')})@{score:.4f}" for sid, score in lst])

                def fmt_host_list(lst):
                    return ', '.join([f"{int(hid)}({host_name.get(int(hid),'')})@{score:.4f}" for hid, score in lst])

                for tid in sample_ids:
                    gt = true_root_causes_dict.get(tid, None)
                    # fault category name if available
                    fc_id = None
                    # find fc_id from any map
                    if tid in mixed_map:
                        fc_id = mixed_map[tid].get('fault_category', None)
                    elif tid in host_map:
                        fc_id = host_map[tid].get('fault_category', None)
                    elif tid in svc_map:
                        fc_id = svc_map[tid].get('fault_category', None)
                    fc_nm = None
                    try:
                        if fc_id is not None:
                            fc_nm = fc_name_map.get(int(fc_id), None)
                    except Exception:
                        fc_nm = None

                    lines.append(f'- Trace: {tid}')
                    lines.append(f'  - GT: {gt}  FaultCategory: {fc_id}({fc_nm if fc_nm else ""})')

                    # service-level topk
                    svc_line = '  - Service TopK: '
                    if tid in svc_map and svc_map[tid].get('top_candidates'):
                        svc_line += fmt_svc_list(svc_map[tid]['top_candidates'])
                        svc_line += f"  (Top1Correct={svc_map[tid].get('is_correct_top1', False)})"
                    else:
                        svc_line += '(none)'
                    lines.append(svc_line)

                    # host-level topk
                    host_line = '  - Host TopK: '
                    if tid in host_map and host_map[tid].get('top_host'):
                        host_line += fmt_host_list(host_map[tid]['top_host'])
                        host_line += f"  (Top1Correct={host_map[tid].get('is_correct_host_top1', False)})"
                    else:
                        host_line += '(none)'
                    lines.append(host_line)

                    # mixed topk
                    mixed_line = '  - Mixed TopK: '
                    if tid in mixed_map and mixed_map[tid].get('top_mixed'):
                        parts = []
                        for knd, idx, sc in mixed_map[tid]['top_mixed']:
                            if knd == 'host':
                                nm = host_name.get(int(idx), '')
                                parts.append(f"host:{int(idx)}({nm})@{sc:.4f}")
                            else:
                                nm = service_name.get(int(idx), '')
                                parts.append(f"svc:{int(idx)}({nm})@{sc:.4f}")
                        mixed_line += ', '.join(parts)
                        mixed_line += f"  (Top1Correct={mixed_map[tid].get('is_correct_top1', False)})"
                    else:
                        mixed_line += '(none)'
                    lines.append(mixed_line)

                if should_save:
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines))
                    logger.info(f"Detailed eval report saved to {report_path}")

                # Export per-trace CSV if enabled
                if should_save and bool(getattr(rca_cfg, 'export_csv', True)):
                    import csv
                    csv_name = (
                        f'rca_results_epoch{epoch}_{ts}.csv' if getattr(config, 'include_epoch_in_report_name', True) and epoch is not None
                        else f'rca_results_{ts}.csv'
                    )
                    csv_path = os.path.join(reports_dir, csv_name)

                    # Index RCA results by trace_id for merging
                    svc_map = {r['trace_id']: r for r in root_cause_results}
                    host_map = {r['trace_id']: r for r in host_results}
                    mixed_map = {r['trace_id']: r for r in mixed_results}

                    keys = set(svc_map.keys()) | set(host_map.keys()) | set(mixed_map.keys())
                    fieldnames = [
                        'trace_id','groundtruth','fault_category',
                        'svc_top1_id','svc_top1_score','svc_top1_correct','svc_topk',
                        'host_top1_id','host_top1_score','host_top1_correct','host_topk',
                        'mixed_top1_kind','mixed_top1_id','mixed_top1_score','mixed_top1_correct','mixed_topk',
                    ]
                    if bool(getattr(rca_cfg, 'export_debug', False)):
                        fieldnames += [
                            'infra_total_hosts','infra_hit_hosts','infra_hit_ratio','minute_key_ms','W','rho_mode','lambda_host','eta','alpha',
                            'top1_pre_host','top1_pre_model_n','top1_pre_infra_n','top1_pre_rho','top1_pre_fused','top1_pre_sinfra_components',
                            'urs_alpha','pair_count','pair_nonzero_j_ratio','top_pairs',
                        ]

                    def fmt_topk(lst, kind='svc'):
                        if not lst:
                            return ''
                        if kind == 'mixed':
                            return ';'.join([f"{k}:{int(i)}@{s:.4f}" for (k, i, s) in lst])
                        else:
                            return ';'.join([f"{int(i)}@{s:.4f}" for (i, s) in lst])

                    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
                        writer = csv.DictWriter(cf, fieldnames=fieldnames)
                        writer.writeheader()
                        for tid in keys:
                            row = {'trace_id': tid}
                            row['groundtruth'] = true_root_causes_dict.get(tid, None)
                            fc_id = None
                            if tid in mixed_map:
                                fc_id = mixed_map[tid].get('fault_category', None)
                            elif tid in host_map:
                                fc_id = host_map[tid].get('fault_category', None)
                            elif tid in svc_map:
                                fc_id = svc_map[tid].get('fault_category', None)
                            row['fault_category'] = fc_id

                            if tid in svc_map and svc_map[tid].get('top_candidates'):
                                sid, sscore = svc_map[tid]['top_candidates'][0]
                                row['svc_top1_id'] = int(sid)
                                row['svc_top1_score'] = float(sscore)
                                row['svc_top1_correct'] = bool(svc_map[tid].get('is_correct_top1', False))
                                row['svc_topk'] = fmt_topk(svc_map[tid]['top_candidates'], 'svc')
                            if tid in host_map and host_map[tid].get('top_host'):
                                hid, hscore = host_map[tid]['top_host'][0]
                                row['host_top1_id'] = int(hid)
                                row['host_top1_score'] = float(hscore)
                                row['host_top1_correct'] = bool(host_map[tid].get('is_correct_host_top1', False))
                                row['host_topk'] = fmt_topk(host_map[tid]['top_host'], 'host')
                            if tid in mixed_map and mixed_map[tid].get('top_mixed'):
                                knd, mid, mscore = mixed_map[tid]['top_mixed'][0]
                                row['mixed_top1_kind'] = str(knd)
                                row['mixed_top1_id'] = int(mid)
                                row['mixed_top1_score'] = float(mscore)
                                row['mixed_top1_correct'] = bool(mixed_map[tid].get('is_correct_top1', False))
                                row['mixed_topk'] = fmt_topk(mixed_map[tid]['top_mixed'], 'mixed')

                            if bool(getattr(rca_cfg, 'export_debug', False)):
                                if tid in host_map:
                                    for k in ['infra_total_hosts','infra_hit_hosts','infra_hit_ratio','minute_key_ms','W','rho_mode','lambda_host','eta','alpha','top1_pre_host','top1_pre_model_n','top1_pre_infra_n','top1_pre_rho','top1_pre_fused','top1_pre_sinfra_components']:
                                        if k in host_map[tid]:
                                            row[k] = host_map[tid][k]
                                if tid in mixed_map:
                                    for k in ['urs_alpha','pair_count','pair_nonzero_j_ratio','top_pairs']:
                                        if k in mixed_map[tid]:
                                            row[k] = mixed_map[tid][k]

                            writer.writerow(row)

                    logger.info(f"RCA CSV exported to {csv_path}")

                # ------------------- Append epoch metrics summary CSV -----------------------
                try:
                    import csv
                    summary_name = 'epoch_metrics.csv'
                    summary_path = os.path.join(reports_dir, summary_name)
                    headers = [
                        'timestamp','epoch',
                        'auc','best_threshold','threshold','best_fscore','best_pr','best_rc','fscore','precision','recall',
                        'svc_top1','svc_top5','svc_avg5',
                        'host_top1','host_top5','host_avg5',
                        'mixed_top1','mixed_top5','mixed_avg5',
                        'mixed_svc_top1','mixed_svc_top5','mixed_host_top1','mixed_host_top5'
                    ]
                    row = {
                        'timestamp': ts,
                        'epoch': int(epoch) if epoch is not None else None,
                        'auc': float(overall_result.get('auc', 0.0)),
                        'best_threshold': float(overall_result.get('best_threshold', 0.0)),
                        'threshold': float(overall_result.get('threshold', 0.0)),
                        'best_fscore': float(overall_result.get('best_fscore', 0.0)),
                        'best_pr': float(overall_result.get('best_pr', 0.0)),
                        'best_rc': float(overall_result.get('best_rc', 0.0)),
                        'fscore': float(overall_result.get('fscore', 0.0)),
                        'precision': float(overall_result.get('precision', 0.0)),
                        'recall': float(overall_result.get('recall', 0.0)),
                        'svc_top1': float(acc_top1),
                        'svc_top5': float(acc_top5),
                        'svc_avg5': float(svc_avg5),
                        'host_top1': float(acc_host_top1),
                        'host_top5': float(acc_host_top5),
                        'host_avg5': float(host_avg5),
                        'mixed_top1': float(acc_mix_top1),
                        'mixed_top5': float(acc_mix_top5),
                        'mixed_avg5': float(mixed_avg5),
                        'mixed_svc_top1': float(acc_mix_svc_t1),
                        'mixed_svc_top5': float(acc_mix_svc_t5),
                        'mixed_host_top1': float(acc_mix_host_t1),
                        'mixed_host_top5': float(acc_mix_host_t5),
                    }
                    write_header = not os.path.isfile(summary_path)
                    with open(summary_path, 'a', newline='', encoding='utf-8') as sf:
                        writer = csv.DictWriter(sf, fieldnames=headers)
                        if write_header:
                            writer.writeheader()
                        writer.writerow(row)
                except Exception as e:
                    logger.warning(f"Failed to append epoch metrics CSV: {e}")
            except Exception as e:
                logger.warning(f"Failed to write detailed eval report: {e}")

            # ----------- 最后一轮/最终评估：按需生成单页总览图（dashboard） -----------
            try:
                if should_save and bool(getattr(rca_cfg, 'enable_plots', False)):
                    try:
                        from tools.plot_reports import plot_dashboard
                        outs_dash = plot_dashboard(reports_dir)
                        if outs_dash:
                            logger.info(f"Report dashboard saved: {outs_dash[0]}")
                    except Exception as pe:
                        logger.debug(f"Skip report dashboard: {pe}")
            except Exception as e:
                logger.debug(f"Plotting guard failed: {e}")

    model.train()
