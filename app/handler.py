"""
Trace_mcp/app/handler.py

该模块负责把 MCP 的“工具调用”转换为子进程命令，并在 app/tools 下执行。

设计目标：
- 不把每个工具脚本的细节散落在 main.py；main.py 只负责注册 MCP tool
- 工具脚本/模块的映射集中在这里（tool_id + stage -> (cwd, target, kind)）
- 统一日志、环境变量与参数转换逻辑
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

from app.schemas import ScriptResult, ToolRequest


TargetKind = Literal["script", "module"]


@dataclass(frozen=True)
class StageTarget:
    kind: TargetKind  # "script" => python xxx.py, "module" => python -m pkg.mod
    target: str       # "xxx.py" or "pkg.mod"


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    work_dir: Path
    description: str
    stages: Mapping[str, StageTarget]

    # 如果工具以 package 形式存在（例如 tracegnn），需要将该目录注入 PYTHONPATH。
    pythonpath: Optional[Path] = None


def _tools_root() -> Path:
    return (Path(__file__).parent / "tools").resolve()


def get_tool_registry() -> Dict[str, ToolSpec]:
    root = _tools_root()

    # 显式映射：避免“拼文件名”的隐式规则导致找不到脚本（例如 preprocess 实际叫 make_xxx.py）
    return {
        "aiops_svnd": ToolSpec(
            tool_id="aiops_svnd",
            work_dir=(root / "trace_svnd_diag").resolve(),
            description="Trace+Topology 融合诊断（SVND）：make/train/test/debug",
            stages={
                "preprocess": StageTarget(kind="script", target="make_aiops_svnd.py"),
                "train": StageTarget(kind="script", target="train_aiops_svnd.py"),
                "test": StageTarget(kind="script", target="test_aiops_svnd.py"),
                "debug": StageTarget(kind="script", target="debug_check_svnd.py"),
            },
        ),
        "aiops_sv": ToolSpec(
            tool_id="aiops_sv",
            work_dir=(root / "trace_sv_diag").resolve(),
            description="单 Trace 诊断（SV）：make/train/test",
            stages={
                "preprocess": StageTarget(kind="script", target="make_aiops_sv.py"),
                "train": StageTarget(kind="script", target="train_aiops_sv.py"),
                "test": StageTarget(kind="script", target="test_aiops_sv.py"),
            },
        ),
        "tratoporca": ToolSpec(
            tool_id="tratoporca",
            work_dir=(root / "TraTopoRca").resolve(),
            description="TraTopoRca (GTrace)：python -m tracegnn...",
            stages={
                "train": StageTarget(kind="module", target="tracegnn.models.gtrace.mymodel_main"),
                "test": StageTarget(kind="module", target="tracegnn.models.gtrace.mymodel_test"),
            },
            pythonpath=(root / "TraTopoRca").resolve(),
        ),
        "tracezly_rca": ToolSpec(
            tool_id="tracezly_rca",
            work_dir=(root / "tracezly_rca").resolve(),
            description="tracezly_rca（实验性）：convert + gtrace train/test",
            stages={
                "preprocess": StageTarget(kind="script", target="convert_trace_dataset.py"),
                "train": StageTarget(kind="module", target="tracegnn.models.gtrace.mymodel_main"),
                # tracezly_rca 的 tracegnn/models/gtrace 下没有 mymodel_test.py；这里使用仓库内提供的评估脚本
                "test": StageTarget(kind="script", target="evaluate_model.py"),
            },
            pythonpath=(root / "tracezly_rca").resolve(),
        ),
    }


def list_tools() -> Dict[str, Any]:
    """给 MCP/调试使用：列出当前可用工具与 stage 映射。"""
    reg = get_tool_registry()
    return {
        "tools": [
            {
                "tool_id": spec.tool_id,
                "work_dir": str(spec.work_dir),
                "description": spec.description,
                "stages": {k: {"kind": v.kind, "target": v.target} for k, v in spec.stages.items()},
            }
            for spec in reg.values()
        ]
    }


def _iter_cli_args(extra_args: Mapping[str, Any]) -> Iterable[str]:
    """将 dict 参数转为命令行参数。

    规则：
    - None: 忽略
    - bool: True => --flag；False => 忽略
    - list/tuple/set: --k v1 --k v2 ...
    - dict/其他复杂对象: JSON 序列化后作为单个值传入
    - 其余类型: str(v)

    约定：额外支持 _raw_args: ["--foo", "bar"] 这种“原样附加”的参数（不做 key->--key 转换）。
    """
    raw = extra_args.get("_raw_args")
    if raw is not None:
        if not isinstance(raw, (list, tuple)):
            raise TypeError("_raw_args 必须是 list/tuple")
        for item in raw:
            yield str(item)

    for k in sorted(extra_args.keys()):
        if k == "_raw_args":
            continue
        v = extra_args[k]
        if v is None:
            continue

        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                yield flag
            continue

        if isinstance(v, (list, tuple, set)):
            for item in v:
                yield flag
                yield str(item)
            continue

        if isinstance(v, dict):
            yield flag
            yield json.dumps(v, ensure_ascii=False)
            continue

        yield flag
        yield str(v)


def _build_env(base_env: Mapping[str, str], *, pythonpath: Optional[Path]) -> Dict[str, str]:
    env = dict(base_env)

    # 统一 UTF-8 输出，避免 Windows 控制台/子进程编码导致日志乱码
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    # 一些用户环境下 OpenMP 会报重复加载；保守处理（若用户显式设置则不覆盖）
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    if pythonpath is not None:
        old = env.get("PYTHONPATH", "")
        if old:
            env["PYTHONPATH"] = f"{pythonpath}{os.pathsep}{old}"
        else:
            env["PYTHONPATH"] = str(pythonpath)

    # 确保当前 python 环境的 Scripts/Library/bin 在 PATH 前面（Windows 下常见）
    python_root = Path(sys.executable).resolve().parent
    path_entries = [
        python_root / "Library" / "bin",
        python_root / "Scripts",
        python_root,
    ]
    prefix = os.pathsep.join(str(p) for p in path_entries if p.exists())
    if prefix:
        env["PATH"] = f"{prefix}{os.pathsep}{env.get('PATH', '')}"

    return env


def _build_command(*, stage_target: StageTarget, work_dir: Path, extra_args: Mapping[str, Any]) -> List[str]:
    cmd = [sys.executable, "-u"]
    if stage_target.kind == "module":
        cmd.extend(["-m", stage_target.target])
    else:
        cmd.append(str((work_dir / stage_target.target).resolve()))

    cmd.extend(list(_iter_cli_args(extra_args)))
    return cmd


def _run_and_tail_log(
    *,
    cmd: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    timeout_sec: Optional[int],
    tail_lines: int,
    max_log_chars: int,
) -> Tuple[int, str]:
    """运行子进程，将完整日志写入文件，并返回 tail。"""
    # 逐行写文件 + 维护一个 tail buffer，避免 stdout 巨大导致内存爆
    tail: List[str] = []

    def _append_tail(line: str) -> None:
        tail.append(line)
        if len(tail) > max(1, tail_lines):
            del tail[0 : len(tail) - tail_lines]

    with open(log_path, "w", encoding="utf-8") as f_log:
        f_log.write("=== Trace_mcp Subprocess ===\n")
        f_log.write(f"Time: {datetime.now().isoformat(timespec='seconds')}\n")
        f_log.write(f"CWD: {cwd}\n")
        f_log.write(f"CMD: {' '.join(cmd)}\n\n")
        f_log.flush()

        try:
            proc = subprocess.Popen(
                list(cmd),
                cwd=str(cwd),
                env=dict(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert proc.stdout is not None

            for line in proc.stdout:
                f_log.write(line)
                _append_tail(line)

            exit_code = proc.wait(timeout=timeout_sec)
            return exit_code, "".join(tail)[-max_log_chars:]
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            f_log.write("\n\n[MCP] TimeoutExpired\n")
            return -1, ("".join(tail) + "\n[MCP] TimeoutExpired\n")[-max_log_chars:]
        except Exception as e:
            f_log.write(f"\n\n[MCP] Handler exception: {e}\n")
            return -1, ("".join(tail) + f"\n[MCP] Handler exception: {e}\n")[-max_log_chars:]


def run_tool(req: ToolRequest) -> ScriptResult:
    reg = get_tool_registry()
    spec = reg.get(req.tool_id)
    if spec is None:
        return ScriptResult(
            exit_code=-1,
            log=f"未知 tool_id={req.tool_id}，可用值：{sorted(reg.keys())}",
            command="",
            cwd="",
            log_path="",
        )

    stage_target = spec.stages.get(req.stage)
    if stage_target is None:
        return ScriptResult(
            exit_code=-1,
            log=f"tool_id={req.tool_id} 不支持 stage={req.stage}，可用：{sorted(spec.stages.keys())}",
            command="",
            cwd=str(spec.work_dir),
            log_path="",
        )

    cmd = _build_command(stage_target=stage_target, work_dir=spec.work_dir, extra_args=req.extra_args)
    env = _build_env(os.environ, pythonpath=spec.pythonpath)

    log_name = f"mcp_{req.tool_id}_{req.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = (spec.work_dir / log_name).resolve()

    exit_code, tail = _run_and_tail_log(
        cmd=cmd,
        cwd=spec.work_dir,
        env=env,
        log_path=log_path,
        timeout_sec=req.timeout_sec,
        tail_lines=req.tail_lines,
        max_log_chars=req.max_log_chars,
    )

    return ScriptResult(
        exit_code=exit_code,
        log=tail,
        command=" ".join(cmd),
        cwd=str(spec.work_dir),
        log_path=str(log_path),
    )


# 兼容旧命名：main.py 里之前叫 run_script
def run_script(req: ToolRequest) -> ScriptResult:
    return run_tool(req)
