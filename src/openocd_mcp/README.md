# openocd_mcp

`openocd_mcp` is an independent module that provides OpenOCD MCP runtime/service capabilities.

## Layout

- `src/openocd_mcp/src/openocd_mcp/`: service/runtime implementation
- `src/openocd_mcp/tests/openocd_mcp/`: pytest test suite
- `src/openocd_mcp/docs/debug_runtime/`: runtime API/state/error docs
- `src/openocd_mcp/skills/`: MCP skill package migrated from root project

## Run

```bash
uv run python src/openocd_mcp/scripts/openocd_mcp.py
```

## Test

```bash
uv run python -m pytest -q src/openocd_mcp/tests/openocd_mcp
```

## Build EXE

```bash
pyinstaller src/openocd_mcp/OpenOCDMCP.spec
```
