"""
Trace_mcp/app/schemas.py

这里定义 MCP 侧使用的请求/响应模型。

这些模型的目标是给 handler 一个稳定、统一的“调用描述”，以便：
- 选择工具（对应 app/tools 下的目录）
- 选择阶段（preprocess/train/test/debug）
- 将任意参数字典转换为命令行参数并执行
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


# 这里的 tool_id 与 app/tools 下的目录一一对应（或语义对应）
ToolId = Literal["aiops_svnd", "aiops_sv", "tratoporca", "tracezly_rca"]
Stage = Literal["preprocess", "train", "test", "debug"]


class ToolRequest(BaseModel):
    """统一的“运行工具脚本”请求。"""

    tool_id: ToolId
    stage: Stage

    # 透传给脚本/模块的参数（会转换为命令行：--k v / --flag）
    extra_args: Dict[str, Any] = Field(default_factory=dict)

    # 执行控制参数（不参与命令行拼接）
    timeout_sec: Optional[int] = None
    tail_lines: int = 400
    max_log_chars: int = 20_000


class ScriptResult(BaseModel):
    """子进程执行结果（给 MCP 返回用）。

    为了避免一次性返回超大日志：
    - log 仅返回 tail（由 ToolRequest.tail_lines / max_log_chars 控制）
    - log_path 指向完整日志文件，便于本地排查
    """

    exit_code: int
    log: str

    command: str
    cwd: str
    log_path: str
