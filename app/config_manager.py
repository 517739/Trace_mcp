"""
Trace_mcp/app/config_manager.py

为 tracegnn 的 gtrace 工具提供“可控参数”的读取与更新能力。

目标：让 MCP Client（或 LLM）能够通过结构化参数（由自然语言解析而来）
来修改各工具目录下的 `tracegnn/models/gtrace/config.py`，然后再启动训练/测试。

设计原则：
- 只允许修改白名单字段（避免自然语言误改代码逻辑）
- 通过“行级解析 + 定位到 class 块”的方式做最小修改（不依赖 import 执行 config.py）
- 支持 dry_run 返回 diff，不落盘
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple


ToolId = Literal["tratoporca", "tracezly_rca"]


@dataclass(frozen=True)
class ConfigLocation:
    # 形如：ExpConfig.device 或 ExpConfig.RCA.export_debug
    class_path: Tuple[str, ...]  # e.g. ("ExpConfig",) or ("ExpConfig", "RCA")
    field_name: str              # e.g. "device" / "export_debug"


def _repo_root() -> Path:
    # Trace_mcp/app/config_manager.py -> Trace_mcp
    return Path(__file__).resolve().parents[1]


def get_config_py_path(tool_id: ToolId) -> Path:
    repo = _repo_root()
    if tool_id == "tratoporca":
        return (repo / "app" / "tools" / "TraTopoRca" / "tracegnn" / "models" / "gtrace" / "config.py").resolve()
    if tool_id == "tracezly_rca":
        return (repo / "app" / "tools" / "tracezly_rca" / "tracegnn" / "models" / "gtrace" / "config.py").resolve()
    raise ValueError(f"unsupported tool_id={tool_id}")


def _allowed_fields(tool_id: ToolId) -> Dict[str, ConfigLocation]:
    # 说明：这里的 key 是对外的“可控参数名”。
    # - 对 top-level 用 "device"/"batch_size" 这种简单键
    # - 对嵌套用 "RCA.export_debug" 这种点号表示法
    if tool_id == "tratoporca":
        return {
            "device": ConfigLocation(("ExpConfig",), "device"),
            "dataset": ConfigLocation(("ExpConfig",), "dataset"),
            "test_dataset": ConfigLocation(("ExpConfig",), "test_dataset"),
            "seed": ConfigLocation(("ExpConfig",), "seed"),
            "batch_size": ConfigLocation(("ExpConfig",), "batch_size"),
            "test_batch_size": ConfigLocation(("ExpConfig",), "test_batch_size"),
            "max_epochs": ConfigLocation(("ExpConfig",), "max_epochs"),
            "max_eval_traces": ConfigLocation(("ExpConfig",), "max_eval_traces"),
            "dataset_root_dir": ConfigLocation(("ExpConfig",), "dataset_root_dir"),
            "model_path": ConfigLocation(("ExpConfig",), "model_path"),
            "report_dir": ConfigLocation(("ExpConfig",), "report_dir"),
            "RCA.export_debug": ConfigLocation(("ExpConfig", "RCA"), "export_debug"),
        }

    # tracezly_rca 的 config.py 相对精简（目前文件里没有 report_dir/model_path/max_eval_traces 等字段）
    return {
        "device": ConfigLocation(("ExpConfig",), "device"),
        "dataset": ConfigLocation(("ExpConfig",), "dataset"),
        "test_dataset": ConfigLocation(("ExpConfig",), "test_dataset"),
        "seed": ConfigLocation(("ExpConfig",), "seed"),
        "batch_size": ConfigLocation(("ExpConfig",), "batch_size"),
        "test_batch_size": ConfigLocation(("ExpConfig",), "test_batch_size"),
        "max_epochs": ConfigLocation(("ExpConfig",), "max_epochs"),
        "dataset_root_dir": ConfigLocation(("ExpConfig",), "dataset_root_dir"),
    }


def _py_repr(v: Any) -> str:
    # 生成可写回 config.py 的字面量表示
    # - str 用单引号 repr，避免破坏原格式
    # - None/bool/int/float 直接 repr
    return repr(v)


def _find_class_block(lines: List[str], class_name: str, base_indent: int) -> Optional[Tuple[int, int]]:
    """找到 `class X` 这一块的 [start, end) 行号范围（按缩进判断块结束）。"""
    class_re = re.compile(rf"^\s{{{base_indent}}}class\s+{re.escape(class_name)}\b")
    start = None
    for i, ln in enumerate(lines):
        if class_re.match(ln):
            start = i
            break
    if start is None:
        return None

    # 块结束：遇到缩进 < base_indent 的非空行；或文件结束
    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if not ln.strip():
            continue
        indent = len(ln) - len(ln.lstrip(" "))
        if indent < base_indent:
            end = j
            break
    return start, end


def _update_field_in_block(
    lines: List[str],
    block_start: int,
    block_end: int,
    field_name: str,
    new_value_repr: str,
    field_indent: int,
) -> bool:
    """在指定 block 内替换 `field_name: xxx = yyy` 的 yyy。"""
    # 支持：
    #   field: type = value
    #   field=value （少见，但尽量兼容）
    # 同时保留行末注释
    pat = re.compile(
        rf"^(?P<indent>\s{{{field_indent}}})"
        rf"(?P<lhs>{re.escape(field_name)}\s*(?::\s*[^=]+)?\s*=\s*)"
        rf"(?P<rhs>.+?)"
        rf"(?P<comment>\s+#.*)?$"
    )
    for i in range(block_start, block_end):
        m = pat.match(lines[i])
        if not m:
            continue
        indent = m.group("indent")
        lhs = m.group("lhs")
        comment = m.group("comment") or ""
        lines[i] = f"{indent}{lhs}{new_value_repr}{comment}\n"
        return True
    return False


def read_gtrace_config(tool_id: ToolId) -> Dict[str, Any]:
    """读取白名单字段的当前值（以字符串形式返回，避免执行 import）。"""
    path = get_config_py_path(tool_id)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    allowed = _allowed_fields(tool_id)
    out: Dict[str, Any] = {"tool_id": tool_id, "config_path": str(path), "values": {}, "unsupported": []}

    # 仅做轻量解析：抓取对应字段所在的“行文本 rhs”
    # 目前只实现 ExpConfig top-level 字段 + TraTopoRca 的 RCA.export_debug
    exp_block = _find_class_block(lines, "ExpConfig", base_indent=0)
    if exp_block is None:
        out["unsupported"].append("class ExpConfig not found")
        return out

    exp_start, exp_end = exp_block

    for key, loc in allowed.items():
        # 定位到目标 class block
        cur_start, cur_end = exp_start, exp_end
        base_indent = 4
        if len(loc.class_path) >= 2:
            # 进入嵌套类：假设嵌套类缩进 +4
            for nested in loc.class_path[1:]:
                nested_block = _find_class_block(lines[cur_start:cur_end], nested, base_indent=base_indent)
                if nested_block is None:
                    out["values"][key] = None
                    break
                ns, ne = nested_block
                # 把相对行号转为绝对行号
                cur_start = cur_start + ns
                cur_end = cur_start + (ne - ns)
                base_indent += 4
            else:
                # 找字段行
                field_pat = re.compile(
                    rf"^\s{{{base_indent}}}{re.escape(loc.field_name)}\s*(?::\s*[^=]+)?\s*=\s*(?P<rhs>.+?)(\s+#.*)?$"
                )
                for i in range(cur_start, cur_end):
                    m = field_pat.match(lines[i])
                    if m:
                        out["values"][key] = m.group("rhs").strip()
                        break
                else:
                    out["values"][key] = None
        else:
            field_pat = re.compile(
                rf"^\s{{4}}{re.escape(loc.field_name)}\s*(?::\s*[^=]+)?\s*=\s*(?P<rhs>.+?)(\s+#.*)?$"
            )
            for i in range(exp_start, exp_end):
                m = field_pat.match(lines[i])
                if m:
                    out["values"][key] = m.group("rhs").strip()
                    break
            else:
                out["values"][key] = None

    return out


def update_gtrace_config(
    tool_id: ToolId,
    updates: Dict[str, Any],
    *,
    dry_run: bool = True,
    backup: bool = True,
) -> Dict[str, Any]:
    """更新 config.py 中白名单字段。

    返回：diff + 实际生效字段。
    """
    allowed = _allowed_fields(tool_id)
    unknown = sorted(k for k in updates.keys() if k not in allowed)
    if unknown:
        return {
            "ok": False,
            "error": f"存在不允许修改的字段：{unknown}",
            "allowed_keys": sorted(allowed.keys()),
        }

    path = get_config_py_path(tool_id)
    old_text = path.read_text(encoding="utf-8")
    lines = old_text.splitlines(keepends=True)

    exp_block = _find_class_block(lines, "ExpConfig", base_indent=0)
    if exp_block is None:
        return {"ok": False, "error": "未找到 class ExpConfig，无法更新"}
    exp_start, exp_end = exp_block

    applied: Dict[str, Any] = {}

    # 逐个字段应用；避免“同名字段在不同 block”造成误替换
    for key, new_value in updates.items():
        loc = allowed[key]
        new_repr = _py_repr(new_value)

        cur_start, cur_end = exp_start, exp_end
        class_indent = 0
        field_indent = 4

        if len(loc.class_path) >= 2:
            # 进入嵌套类
            base_indent = 4
            for nested in loc.class_path[1:]:
                nested_block = _find_class_block(lines[cur_start:cur_end], nested, base_indent=base_indent)
                if nested_block is None:
                    return {"ok": False, "error": f"未找到嵌套类 {'.'.join(loc.class_path)}，无法更新 {key}"}
                ns, ne = nested_block
                cur_start = cur_start + ns
                cur_end = cur_start + (ne - ns)
                base_indent += 4
            field_indent = base_indent

        ok = _update_field_in_block(
            lines,
            block_start=cur_start,
            block_end=cur_end,
            field_name=loc.field_name,
            new_value_repr=new_repr,
            field_indent=field_indent,
        )
        if not ok:
            return {"ok": False, "error": f"未找到字段 {key} 对应的代码行，无法更新（字段名={loc.field_name}）"}
        applied[key] = new_value

    new_text = "".join(lines)
    diff = "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )

    if not dry_run:
        if backup:
            bak = path.with_suffix(path.suffix + ".bak")
            if not bak.exists():
                bak.write_text(old_text, encoding="utf-8")
        path.write_text(new_text, encoding="utf-8")

    return {
        "ok": True,
        "tool_id": tool_id,
        "config_path": str(path),
        "dry_run": dry_run,
        "applied": applied,
        "diff": diff,
    }

