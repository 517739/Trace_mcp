import mltk
import torch
from typing import *


# 实验配置（训练/评估运行所需参数与路径）
class ExpConfig(mltk.Config):
    # 基础训练配置
    device: str = 'cpu'
    dataset: str = '2e5_1622_minspan5'
    # 提示：为了快速验证可以把 test_dataset 暂时设为 'val'，正式评估应为 'test'
    test_dataset: str = 'test'
    seed: int = 1234

    batch_size: int = 32
    test_batch_size: int = 64
    # Maximum number of traces to evaluate (None or <=0 means no limit)
    max_eval_traces: Optional[int] = None
    max_epochs: int = 2
    enable_tqdm: bool = True
    # Debug helpers: set True to print/check NaN/Inf sources and fail fast.
    debug_nan: bool = True

    # 数据集根目录（相对工程根目录或绝对路径）
    dataset_root_dir: str = 'dataset/tianchi'
    # 模型权重保存/读取路径：
    model_path: str = 'save/demo/model.pth'

    # 报告输出目录（相对 processed 目录或绝对路径）
    report_dir: str = 'reports_0126_demo'
    include_epoch_in_report_name: bool = True  # 报告文件名中是否包含 epoch 序号

    # 模型相关配置
    class Latency(mltk.Config):
        embedding_type: str = 'normal'        # 延迟特征嵌入方式
        latency_feature_length: int = 1
        latency_embedding: int = 10
        latency_max_value: float = 50.0

    class Model(mltk.Config):
        vae: bool = True
        anneal: bool = False
        kl_weight: float = 1e-2
        n_z: int = 5

        latency_model: str = 'bert'          # 延迟分支：tree / bert
        structure_model: str = 'isoc_vgae'   # 结构分支：tree / isoc_vgae

        # TraceBERT 参数
        bert_n_layers: int = 2
        bert_n_heads: int = 4
        bert_dim_feedforward: int = 256
        bert_max_len: int = 100

        num_features: int = 13               # 节点特征维度
        hidden_dim: List[int] = [512, 512]   # 隐藏层维度
        latency_input: bool = False
        embedding_size: int = 4
        graph_embedding_size: int = 4
        decoder_feature_size: int = 4

        latency_feature_size: int = 64        # tracebert设置为64，treelstm设置为4
        latency_gcn_layers: int = 5
        # 训练端：不确定性权重 log_sigma 的预热冻结轮数（0 表示不冻结）
        freeze_sigma_epochs: int = 4
        # 训练端主机拓扑特征（一般评估端使用）
        enable_hetero: bool = False          # 是否启用一跳拓扑近似
        host_topo_out_dim: int = 16          # HostTopoEncoder 输出维度
        host_topo_alpha: float = 0.5         # 主机拓扑一次聚合强度 alpha（I + alpha A）

    decoder_max_nodes: int = 100

    # 主机通道（训练端：时序/快照 VAE）
    class HostChannel(mltk.Config):
        enable: bool = True                  # 是否启用主机通道分支
        # 后端：'omni'（时序：GRU+VAE）或 'vae'（快照：基于 host_state）
        backend: str = 'omni'                # 'omni' or 'anomaly_transformer'

        # Anomaly Transformer 参数
        at_layers: int = 3
        at_heads: int = 8

        # 'omni' 模式：分钟级窗口与指标
        seq_window: int = 15                 # 回看窗口 W（分钟）
        # seq_metrics: List[str] = ['cpu', 'mem', 'fs']  # 指标别名
        seq_metrics: List[str] = ['cpu', 'mem', 'disk', 'net']
        # 模型维度（两种后端通用）
        hidden_dim: int = 64                 # 主机头隐藏维度
        latent_dim: int = 16                 # 主机头潜变量维度
        beta_kl: float = 1e-3                # 主机通道 KL 额外权重
        kendall_init_logvar: float = 0.2     # Kendall 头初始 log-sigma

    # 主机状态特征（评估/训练端）
    class HostState(mltk.Config):
        enable: bool = True                 # 是否启用 host_state 特征
        # metrics: List[str] = [
        #     'node_cpu_usage_rate',
        #     'node_memory_usage_rate',
        #     'node_filesystem_usage_rate',
        # ]

        metrics: List[str] = [
            "aggregate_node_cpu_usage",
            "aggregate_node_memory_usage",
            "aggregate_node_disk_io_usage",
            "aggregate_node_net_receive_packages_errors_per_minute",
            "aggregate_node_tcp_alloc_total_num",
            "aggregate_node_tcp_inuse_total_num"
        ]
        include_disk: bool = False          # 是否包含磁盘时间类指标
        W: int = 3                          # 回看窗口（分钟）
        # host_state 向量的每指标维度：
        # 3 -> [z0, delta, max_z]
        # 4 -> [z0, delta, max_z, mean_z]（推荐）
        per_metric_dims: int = 4
        out_dim: int = 16                   # HostStateEncoder 输出维度

    # RCA / 评估参数
    class RCA(mltk.Config):
        # 主机相关融合权重
        lambda_host_host: float = 0.45   # 主机榜：HostNLL 与 SInfra 的融合权重
        lambda_host_mixed: float = 0.35  # 混合榜：host 与 service 分支权重
        topk: int = 5                    # RCA 指标 Top-K
        svc_pool_k: int = 3              # 服务内池化 K（更尖锐）
        host_pool_k: int = 3             # 主机池化 K
        lambda_host: float = 0.7         # 简化混合权重（兼容旧逻辑）
        alpha: float = 0.7               # 主机拓扑传播强度
        steps: int = 1                   # 主机拓扑传播步数
        sample_count: int = 20           # 报告示例条数
        # RCA 样本选择：'gt'（标注异常）、'pred'（预测异常）、'all'（全部）
        rca_filter: str = 'gt'

        # 基础设施指标 + URS（统计增强）
        W: int = 3                         # 基础设施指标回看窗口（分钟）
        eta: float = 0.3                   # 主机注意力强度（ρ_h）
        urs_alpha: float = 0.5             # URS 中 Jaccard 与 SInfra 的融合权重
        rho_mode: str = 'count'            # 注意力模式：'count' 或 'duration'
        # 多尺度 SInfra + 同伴归一
        ms_W_list: List[int] = [1, 5, 15]   # 多尺度窗口（分钟）
        lse_tau: float = 1.5                # log-sum-exp 平滑温度
        sinfra_w: Dict[str, float] = {'z_point': 0.6, 'z_win': 0.3, 'peer': 0.1}
        peer_mode: str = 'global'           # 同伴集合：'trace' 或 'global'
        # 服务聚合（评估端）：两段式聚合
        svc_len_p: float = 0.25             # 节点数校正指数 p（sum_topK / n_s^p）
        svc_tau: float = 0.8                # 跨服务 softmax 温度
        # 评估端固定结构/延迟权重，避免训练期漂移
        eval_alpha: float = 0.7
        eval_beta: float = 0.3
        # 评估端 anomaly detection 的图级打分方式：
        # - 'sl'      : alpha*StructNLL + beta*LatencyNLL（旧默认）
        # - 'slh'     : alpha*StructNLL + beta*LatencyNLL + gamma*HostLoss（适合天池这类 host/infra 注入）
        # - 'topk_sl' : topK(node_score)（node_score 由节点级结构/时延分数组合得到）
        # - 'topk_slh': topK(node_score) + gamma*HostLoss（推荐优先尝试）
        detector_score_mode: str = 'topk_slh'
        detector_topk: int = 5
        # 如需固定 host 权重，可设置 eval_gamma；为 None 时使用模型输出 pred['gamma']（Kendall 对应）
        eval_gamma: Optional[float] = None

        # 导出控制
        export_csv: bool = True             # 是否导出每条 trace 的 RCA 结果 CSV
        export_debug: bool = True           # 是否导出详细调试信息

        # 混合榜增强（base + URS + rerank）
        gamma_svc: float = 0.3              # URS 对服务分支增强权重
        gamma_host: float = 0.3             # URS 对主机分支增强权重

        # 评估增强（默认保持关闭，不改变现有结果）
        enable_mixed_rerank: bool = True    # 启用 rerank（截断 + 拓扑一致性加权）
        mixed_pool_svc_k: int = 3           # 混合前服务候选截断数
        mixed_pool_host_k: int = 3          # 混合前主机候选截断数
        topo_boost_beta: float = 0.2        # 同机拓扑加权系数

        enable_host_ms_infra: bool = True  # 主机侧启用多尺度 SInfra + 同伴归一

        # 异构传播（service <- host）与主机波动性剪枝
        enable_hetero_prop: bool = True    # 是否启用 Host→Service 传播（rerank 路径）
        enable_causal_pruning: bool = True # 是否启用主机波动性剪枝
        pruning_threshold: float = 0.01     # 剪枝阈值（时序标准差阈值）

        # 评估端其它开关
        suppress_missing_gt_warning: bool = True  # 是否静默“异常但无 GT”的告警
        save_final_only: bool = True              # 仅在最后一轮/最终评估保存 MD/CSV
        enable_plots: bool = False                # 最后一轮/最终评估是否生成图表
        export_unlabeled: bool = False            # 是否导出未标注样本（不计入指标）

    # 运行期数据缓存（从数据集统计/加载）
    class RuntimeInfo:
        latency_range: torch.Tensor = None   # 操作延迟的均值/方差（或均值/尺度）
        latency_p98: torch.Tensor = None     # 操作延迟的 p98 分位

    # 数据集元信息（由 TraceGraphIDManager 提供）
    class DatasetParams:
        operation_cnt: int = None
        service_cnt: int = None
        status_cnt: int = None
