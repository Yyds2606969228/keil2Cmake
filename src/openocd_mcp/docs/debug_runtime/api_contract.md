# API Contract (v0.2)

## Runtime Model

当前调试运行时默认按 **本地 `stdio` MCP 服务** 设计：

- 由 MCP client 按需拉起进程
- 通过标准输入/输出通信
- 默认不依赖固定 TCP 端口

因此，在 EXE 分发场景下推荐：

- `Keil2Cmake.exe` 作为主入口
- `openocd-mcp.exe` 作为独立 MCP 服务入口

这不会使全局编排失效；编排仍保留在 `keil2cmake.orchestrator`，MCP 只负责暴露调试与运行时能力。

## System & Connectivity

- `list_debug_probes()`
- `list_serial_ports()`
- `connect_debugger(config: dict)`
- `connect_serial(port: str, baud: int, config?: dict)`
- `reset_target(reset_type: str)` where `reset_type in halt|init|run`
- `emergency_stop()`

## Debugging (JTAG/SWD)

- `execute_raw_tcl(cmd: str)`
- `control_target(action: str)` where `action in halt|resume|step|reset|init|run`
- `read_memory(address: str, count: int, width?: int)`
- `write_memory(address: str, value: int, width?: int, dry_run?: bool, confirm?: bool)`
  - Critical peripheral-range writes (`0x40000000` - `0x5FFFFFFF`) require `confirm=true`
  - Use `dry_run=true` for preflight preview
- `read_peripheral(name: str)` where `name` format is `PERIPH->REG`
- `manage_breakpoint(point_type: str, addr: str, action: str)` where
  - `point_type in bp|wp`
  - `action in add|del`

## Serial Assistant

- `serial_write(data: str, mode?: str)` where `mode in ascii|hex`
- `serial_read_buffer(lines: int, keyword?: str)`
- `serial_set_trigger(keyword: str, action: str)`

## Scripting

- `submit_task(code: str, timeout_ms?: int)` returns task id
- `get_task_result(id: str)`
- `cancel_task(id: str, force?: bool)`

## Safety Extensions

These are implementation safety helpers and do not conflict with PRD scope:

- `set_flash_mode(enabled: bool)`
- `flash_program(file_path: str, address?: str, verify?: bool, reset?: bool, dry_run?: bool)`

## Response Envelope

All tools return:

```json
{
  "success": true,
  "status": "success",
  "data": {},
  "error_code": null,
  "message": "ok",
  "raw_output": null,
  "timestamp": "2026-02-28T00:00:00+00:00"
}
```
