# Trace_mcp（MCP Server）

该目录提供一个基于 `mcp.server.fastmcp.FastMCP` 的 MCP Server，用于把 `Trace_mcp/app/tools` 下的故障诊断/根因分析脚本封装为可被 LLM 或其他 MCP Client 调用的工具。


## 目录结构（核心）

| 路径 | 作用 | 备注 |
|---|---|---|
| `Trace_mcp/app/main.py` | MCP Server 入口，注册对外 tools | 只做“工具注册 + 参数打包” |
| `Trace_mcp/app/handler.py` | 子进程执行器：把 tool 调用映射到具体脚本/模块并执行 | 统一日志、环境变量与参数转换 |
| `Trace_mcp/app/schemas.py` | MCP 侧请求/响应模型（Pydantic） | `ToolRequest` / `ScriptResult` |
| `Trace_mcp/app/tools/` | 四个“故障处理工具”目录 | 每个目录内部是独立的脚本/模型代码 |
| `Trace_mcp/app/data/` | 数据集处理脚本（按数据集划分） | `aiops25/`、`tianchi/` |


## 四个工具目录一览

| tool_id（handler 使用） | 目录 | 说明 | stage -> 目标 |
|---|---|---|---|
| `aiops_svnd` | `Trace_mcp/app/tools/trace_svnd_diag/` | Trace + 拓扑/节点信息融合（SVND） | `preprocess->make_aiops_svnd.py` / `train->train_aiops_svnd.py` / `test->test_aiops_svnd.py` / `debug->debug_check_svnd.py` |
| `aiops_sv` | `Trace_mcp/app/tools/trace_sv_diag/` | 单 Trace 结构诊断（SV） | `preprocess->make_aiops_sv.py` / `train->train_aiops_sv.py` / `test->test_aiops_sv.py` |
| `tratoporca` | `Trace_mcp/app/tools/TraTopoRca/` | GTrace 风格根因分析（tracegnn） | `train->python -m tracegnn.models.gtrace.mymodel_main` / `test->python -m tracegnn.models.gtrace.mymodel_test` |
| `tracezly_rca` | `Trace_mcp/app/tools/tracezly_rca/` | 实验性脚本集合（含 gtrace train 与数据转换） | `preprocess->convert_trace_dataset.py` / `train->python -m tracegnn.models.gtrace.mymodel_main` / `test->evaluate_model.py` |

说明：上表的映射在 `Trace_mcp/app/handler.py` 的 `get_tool_registry()` 中集中维护。


## MCP 暴露的工具列表（对外）

| MCP tool 名称 | 用途 | 推荐场景 |
|---|---|---|
| `ping` | 连通性检查（不启动子进程） | 验证 MCP 通信链路 |
| `list_trace_tools` | 列出 tool_id/stage 映射 | 调试/排查“脚本到底会跑哪个” |
| `get_gtrace_config` | 读取 gtrace 的 config.py（白名单字段） | 在训练/测试前查看当前可控参数与取值 |
| `update_gtrace_config` | 更新 gtrace 的 config.py（支持 dry-run + 自动备份） | 用自然语言生成结构化 updates，安全落盘后再训练/测试 |
| `run_trace_tool` | 通用运行入口：`tool_id + stage + args` | 统一入口，适合自动化/批量 |
| `check_env` | 环境诊断探针（走 `aiops_svnd:debug`） | 排查子进程启动/依赖问题 |
| `preprocess_aiops_svnd` / `train_aiops_svnd` / `test_aiops_svnd` | SVND 三件套 | Trace+拓扑融合诊断 |
| `preprocess_aiops_sv` / `train_aiops_sv` / `test_aiops_sv` | SV 三件套 | 单 Trace 结构诊断 |
| `train_tracerca` / `test_tracerca` | TraTopoRca(GTrace) 训练/评估 | 根因分析实验 |
| `preprocess_tracezly_rca` / `train_tracezly_rca` / `test_tracezly_rca` | tracezly_rca 相关入口 | 实验性脚本 |


## 运行方式

| 方式 | 命令 | 说明 |
|---|---|---|
| 直接启动 | `python -m app.main` | 在 `Trace_mcp/` 目录下执行 |
| Inspector 启动 | `npx @modelcontextprotocol/inspector python -m app.main` | 适合本地调试/查看 schema |


## TraTopoRca / tracezly_rca 的启动方式与配置说明（重要）

你提到的两点这里做显式说明：

| 工具目录 | 训练命令（在该目录下运行） | 测试命令（在该目录下运行） | 参数控制方式 |
|---|---|---|---|
| `Trace_mcp/app/tools/TraTopoRca/` | `python -m tracegnn.models.gtrace.mymodel_main` | `python -m tracegnn.models.gtrace.mymodel_test` | 主要由 `Trace_mcp/app/tools/TraTopoRca/tracegnn/models/gtrace/config.py` 控制；`mymodel_test` 额外支持命令行覆盖（如 `--model/--report-dir/--limit` 等） |
| `Trace_mcp/app/tools/tracezly_rca/` | `python -m tracegnn.models.gtrace.mymodel_main` | （无标准 `mymodel_test.py`）可用 `python evaluate_model.py` | 主要由 `Trace_mcp/app/tools/tracezly_rca/tracegnn/models/gtrace/config.py` 控制 |


## 调用示例（通用入口 run_trace_tool）

```json
{
  "tool_id": "aiops_svnd",
  "stage": "test",
  "args": {
    "model_path": "dataset/aiops_svnd/1019/aiops_nodectx_multihead.pt",
    "data_root": "dataset/aiops_svnd",
    "batch_size": 128,
    "device": "cuda",
    "limit": 50
  },
  "timeout_sec": 600,
  "tail_lines": 200
}
```


## 日志与排查

| 项 | 位置 | 说明 |
|---|---|---|
| 子进程完整日志 | 各工具目录下的 `mcp_<tool_id>_<stage>_<timestamp>.log` | `handler.py` 自动落盘 |
| MCP 返回日志 | `ScriptResult.log` | 仅返回 tail，避免响应过大 |


## 数据处理（app/data）

| 目录 | 数据集 | 说明 |
|---|---|---|
| `Trace_mcp/app/data/aiops25/` | AIOps25 | 该目录下是按比赛/数据集定制的处理脚本（如 trace 切分、groundtruth 对齐等） |
| `Trace_mcp/app/data/tianchi/` | 天池 | 包含数据/脚本/src/tools 等子目录，适合复用既有 pipeline |
