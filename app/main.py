"""
Trace_mcp/app/main.py

FastMCP 入口：把 app/tools 下的脚本/模块封装为 MCP tools。

约定：真正的“脚本选择 + 命令拼接 + 子进程执行 + 日志落盘”都在 app/handler.py，
main.py 只负责：
- 注册 MCP tools（对外可调用）
- 组合参数并调用 handler.run_tool
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.handler import list_tools, run_tool
from app.schemas import ToolId, ToolRequest, Stage
from app.config_manager import read_gtrace_config, update_gtrace_config as update_gtrace_config_file


mcp = FastMCP("trace-tools")


def _as_dict(model_obj: Any) -> Dict[str, Any]:
    # 兼容 pydantic v1/v2
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    return model_obj.dict()


def _raise_on_failure(result: Any, *, title: str) -> None:
    if getattr(result, "exit_code", -1) == 0:
        return
    # 返回日志是 tail；完整日志在 log_path
    log_path = getattr(result, "log_path", "")
    log_tail = getattr(result, "log", "")
    raise RuntimeError(f"{title} 失败 (exit_code={result.exit_code})\nlog_path={log_path}\n\n{log_tail}")


@mcp.tool()
def ping() -> str:
    """基础连通性检查（不启动子进程）。"""
    return "pong"


@mcp.tool()
def list_trace_tools() -> Dict[str, Any]:
    """列出当前 MCP Server 暴露的底层工具目录与 stage 映射。"""
    return list_tools()


@mcp.tool()
def get_gtrace_config(
    tool_id: Annotated[str, Field(description="仅支持 tratoporca / tracezly_rca")],
) -> Dict[str, Any]:
    """读取 gtrace 的 config.py 可控参数快照（白名单字段）。

    用途：让用户/LLM 知道“当前有哪些参数”“现在是什么值”，再决定怎么改。
    """
    if tool_id not in ("tratoporca", "tracezly_rca"):
        raise ValueError("tool_id 仅支持 tratoporca / tracezly_rca")
    return read_gtrace_config(tool_id)  # type: ignore[arg-type]


@mcp.tool()
def update_gtrace_config(
    tool_id: Annotated[str, Field(description="仅支持 tratoporca / tracezly_rca")],
    updates: Annotated[Dict[str, Any], Field(description="要修改的键值对；仅允许白名单字段（可先 get_gtrace_config 查看）")],
    dry_run: Annotated[bool, Field(description="True: 只返回 diff 不落盘；False: 写回 config.py 并自动备份 .bak")]=True,
) -> Dict[str, Any]:
    """更新 gtrace 的 config.py（支持 dry-run）。

    典型用法：
    - 先 `get_gtrace_config(tratoporca)` 看当前值与支持字段
    - 再 `update_gtrace_config(tratoporca, {"device":"cpu","batch_size":16}, dry_run=True)` 看 diff
    - 最后 dry_run=False 真正写入，然后再调用 train/test
    """
    if tool_id not in ("tratoporca", "tracezly_rca"):
        raise ValueError("tool_id 仅支持 tratoporca / tracezly_rca")
    return update_gtrace_config_file(tool_id, updates, dry_run=dry_run)  # type: ignore[arg-type]


@mcp.tool()
def run_trace_tool(
    tool_id: ToolId,
    stage: Stage,
    args: Annotated[Dict[str, Any], Field(description="传给脚本/模块的参数字典，会被转为命令行参数")] = Field(
        default_factory=dict
    ),
    timeout_sec: Annotated[Optional[int], Field(description="可选超时（秒），None 表示不设置超时")] = None,
    tail_lines: Annotated[int, Field(description="返回日志的行数（完整日志会落到 log_path）")] = 400,
    max_log_chars: Annotated[int, Field(description="返回日志的最大字符数（兜底限制）")] = 20_000,
) -> Dict[str, Any]:
    """通用入口：运行任意 tool_id + stage。

    对调试/扩展友好：新增工具/脚本时只需要改 handler 里的 registry。
    """
    req = ToolRequest(
        tool_id=tool_id,
        stage=stage,
        extra_args=args,
        timeout_sec=timeout_sec,
        tail_lines=tail_lines,
        max_log_chars=max_log_chars,
    )
    result = run_tool(req)
    return _as_dict(result)


@mcp.tool()
def check_env() -> Dict[str, Any]:
    """环境诊断探针：验证子进程能否正常启动与输出日志。"""
    req = ToolRequest(tool_id="aiops_svnd", stage="debug", extra_args={}, timeout_sec=60, tail_lines=200)
    result = run_tool(req)
    _raise_on_failure(result, title="check_env")
    return _as_dict(result)


# ================= Group 1: Trace + Topology Fusion (SVND) =================

@mcp.tool()
def preprocess_aiops_svnd(**kwargs: Any) -> Dict[str, Any]:
    """[SVND] 数据预处理（Trace + 拓扑/节点信息融合）。"""
    req = ToolRequest(tool_id="aiops_svnd", stage="preprocess", extra_args=kwargs)
    result = run_tool(req)
    _raise_on_failure(result, title="preprocess_aiops_svnd")
    return _as_dict(result)


@mcp.tool()
def train_aiops_svnd(**kwargs: Any) -> Dict[str, Any]:
    """[SVND] 训练（Trace + 拓扑/节点信息融合）。"""
    req = ToolRequest(tool_id="aiops_svnd", stage="train", extra_args=kwargs)
    result = run_tool(req)
    _raise_on_failure(result, title="train_aiops_svnd")
    return _as_dict(result)


@mcp.tool()
def test_aiops_svnd(
    model_path: Annotated[str, Field(description="模型权重文件路径 (.pt)")] = "dataset/aiops_svnd/1019/aiops_nodectx_multihead.pt",
    data_root: Annotated[str, Field(description="数据集目录")] = "dataset/aiops_svnd",
    batch_size: Annotated[int, Field(description="批大小 (Batch Size)")] = 128,
    seed: Annotated[int, Field(description="随机种子")] = 2025,
    device: Annotated[str, Field(description="运行设备 ('cuda' 或 'cpu')")] = "cuda",
    limit: Annotated[int, Field(description="测试样本数量限制（建议设置以避免超时，如 50）")] = 50,
    timeout_sec: Annotated[Optional[int], Field(description="可选超时（秒）")] = None,
) -> Dict[str, Any]:
    """[SVND] 测试/评估（建议使用 limit 控制规模，避免 MCP 超时）。"""
    extra_args = {
        "model_path": model_path,
        "data_root": data_root,
        "batch_size": batch_size,
        "seed": seed,
        "device": device,
        "limit": limit,
    }
    req = ToolRequest(tool_id="aiops_svnd", stage="test", extra_args=extra_args, timeout_sec=timeout_sec)
    result = run_tool(req)
    _raise_on_failure(result, title="test_aiops_svnd")
    return _as_dict(result)


# ================= Group 2: Single Trace (SV) =================

@mcp.tool()
def preprocess_aiops_sv(**kwargs: Any) -> Dict[str, Any]:
    """[SV] 数据预处理（仅 Trace 调用链结构）。"""
    req = ToolRequest(tool_id="aiops_sv", stage="preprocess", extra_args=kwargs)
    result = run_tool(req)
    _raise_on_failure(result, title="preprocess_aiops_sv")
    return _as_dict(result)


@mcp.tool()
def train_aiops_sv(**kwargs: Any) -> Dict[str, Any]:
    """[SV] 训练（仅 Trace 调用链结构）。"""
    req = ToolRequest(tool_id="aiops_sv", stage="train", extra_args=kwargs)
    result = run_tool(req)
    _raise_on_failure(result, title="train_aiops_sv")
    return _as_dict(result)


@mcp.tool()
def test_aiops_sv(
    model_path: Annotated[str, Field(description="模型权重文件路径 (.pth)")] = "dataset/aiops_sv/aiops_superfine_cls.pth",
    data_root: Annotated[str, Field(description="数据集目录（包含 test.jsonl 与 vocab.json）")] = "dataset/aiops_sv",
    task: Annotated[str, Field(description="任务类型：fine/superfine")] = "superfine",
    batch: Annotated[int, Field(description="批大小 (Batch Size)")] = 64,
    seed: Annotated[int, Field(description="随机种子")] = 2025,
    device: Annotated[str, Field(description="运行设备 ('cuda' 或 'cpu')")] = "cuda",
    min_type_support: Annotated[int, Field(description="过滤小样本类别阈值")] = 150,
    run_name: Annotated[str, Field(description="运行名称（影响输出目录名）")] = "trace_only",
    limit: Annotated[int, Field(description="测试样本数量限制（可选，用于快速验证/避免超时）")] = 100,
    timeout_sec: Annotated[Optional[int], Field(description="可选超时（秒）")] = None,
) -> Dict[str, Any]:
    """[SV] 测试/评估（建议使用 limit 控制规模，避免 MCP 超时）。"""
    extra_args = {
        "model_path": model_path,
        "data_root": data_root,
        "task": task,
        "batch": batch,
        "seed": seed,
        "device": device,
        "min_type_support": min_type_support,
        "run_name": run_name,
        "limit": limit,
    }
    req = ToolRequest(tool_id="aiops_sv", stage="test", extra_args=extra_args, timeout_sec=timeout_sec)
    result = run_tool(req)
    _raise_on_failure(result, title="test_aiops_sv")
    return _as_dict(result)


# ================= Group 3: TraTopoRca (GTrace) =================

@mcp.tool()
def train_tracerca(
    timeout_sec: Annotated[Optional[int], Field(description="可选超时（秒）")] = None,
    args: Annotated[Dict[str, Any], Field(description="可选参数：建议使用 _raw_args 传 mltk/config 覆盖参数；默认参数主要在 config.py")] = Field(
        default_factory=dict
    ),
) -> Dict[str, Any]:
    """[TraTopoRca] 训练：在工具目录下运行 `python -m tracegnn.models.gtrace.mymodel_main`。

    注意：该训练入口通常通过 `tracegnn/models/gtrace/config.py` 控制参数。
    如果你确定该训练框架支持命令行覆盖（取决于 mltk 的配置方式），可以通过：
    - args={"_raw_args": ["--some-override", "value"]}
    的形式传入原样参数。
    """
    req = ToolRequest(tool_id="tratoporca", stage="train", extra_args=args, timeout_sec=timeout_sec)
    result = run_tool(req)
    _raise_on_failure(result, title="train_tracerca")
    return _as_dict(result)


@mcp.tool()
def test_tracerca(
    model: Annotated[Optional[str], Field(description="模型路径 (.pth)，不填则用 config 默认")] = None,
    test_dataset: Annotated[Optional[str], Field(description="测试子集")] = None,
    limit: Annotated[Optional[str], Field(description="限制测试数量（例如 10）")] = None,
    report_dir: Annotated[str, Field(description="报告输出目录")] = "reports_mcp",
    batch_size: Annotated[int, Field(description="批大小（会传给脚本）")] = 32,
    export_debug: Annotated[bool, Field(description="是否导出调试信息")] = True,
    timeout_sec: Annotated[Optional[int], Field(description="可选超时（秒）")] = None,
) -> Dict[str, Any]:
    """[TraTopoRca] 测试/评估：python -m tracegnn.models.gtrace.mymodel_test。"""
    extra_args = {
        "model": model,
        "test_dataset": test_dataset,
        "limit": limit,
        "report_dir": report_dir,
        "batch_size": batch_size,
        "export_debug": export_debug,
    }
    req = ToolRequest(tool_id="tratoporca", stage="test", extra_args=extra_args, timeout_sec=timeout_sec)
    result = run_tool(req)
    _raise_on_failure(result, title="test_tracerca")
    return _as_dict(result)


# ================= Group 4: tracezly_rca（实验性） =================

@mcp.tool()
def preprocess_tracezly_rca(**kwargs: Any) -> Dict[str, Any]:
    """[tracezly_rca] 数据转换/预处理（convert_trace_dataset.py）。"""
    req = ToolRequest(tool_id="tracezly_rca", stage="preprocess", extra_args=kwargs)
    result = run_tool(req)
    _raise_on_failure(result, title="preprocess_tracezly_rca")
    return _as_dict(result)


@mcp.tool()
def train_tracezly_rca(**kwargs: Any) -> Dict[str, Any]:
    """[tracezly_rca] 训练：在工具目录下运行 `python -m tracegnn.models.gtrace.mymodel_main`。

    注意：主要参数通常由 `tracegnn/models/gtrace/config.py` 控制。
    """
    req = ToolRequest(tool_id="tracezly_rca", stage="train", extra_args=kwargs)
    result = run_tool(req)
    _raise_on_failure(result, title="train_tracezly_rca")
    return _as_dict(result)


@mcp.tool()
def test_tracezly_rca(**kwargs: Any) -> Dict[str, Any]:
    """[tracezly_rca] 测试/评估。

    说明：tracezly_rca 目录下未提供标准的 `mymodel_test.py` 模块入口，
    当前使用目录内的 `evaluate_model.py` 脚本作为评估入口。
    """
    req = ToolRequest(tool_id="tracezly_rca", stage="test", extra_args=kwargs)
    result = run_tool(req)
    _raise_on_failure(result, title="test_tracezly_rca")
    return _as_dict(result)


if __name__ == "__main__":
    # 默认 transport 由 fastmcp 决定；如需 SSE 可用：mcp.run(transport="sse")
    mcp.run()
