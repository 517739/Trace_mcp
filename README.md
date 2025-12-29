# Trace_mcp MCP Server

This directory provides an MCP-style HTTP server that exposes trace diagnosis **training and testing** scripts as tools for LLMs or other clients.

## Supported Tools

### Training Tools
- `aiops_3c6c`: runs `TraDiag/trace_service_diag/train_aiops_sv.py`
- `aiops_svnd`: runs `TraDiag/trace_svnd_diag/train_aiops_svnd.py`

### Testing / Evaluation Tools
- `test_aiops_3c6c`: runs `app/tools/trace_sv_diag/test_aiops_sv.py` (Evaluates trace service diagnosis)
- `test_aiops_svnd`: runs `app/tools/trace_svnd_diag/test_aiops_svnd.py` (Evaluates multi-modal host+trace diagnosis)
- `test_gtrace`: runs `app/tools/TraTopoRca/tracegnn/models/gtrace/mymodel_test.py` (Evaluates Graph Anomaly Detection model)

## Build and run with Docker

From the repository root:

```bash
npx @modelcontextprotocol/inspector python -m app.main