"""FastMCP server wiring."""

from __future__ import annotations

from typing import Any

from .tools import OpenOCDMCPService

try:
    from fastmcp import FastMCP  # type: ignore[import-untyped]
    _FASTMCP_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - runtime dependency
    FastMCP = None
    _FASTMCP_IMPORT_ERROR = exc


def create_server(service: OpenOCDMCPService | None = None) -> Any:
    if FastMCP is None:
        raise RuntimeError(f"fastmcp import failed: {_FASTMCP_IMPORT_ERROR!r}")
    svc = service or OpenOCDMCPService()
    mcp = FastMCP("openocd-mcp")

    @mcp.tool()
    def list_debug_probes() -> dict[str, Any]:
        return svc.list_debug_probes()

    @mcp.tool()
    def list_serial_ports() -> dict[str, Any]:
        return svc.list_serial_ports()

    @mcp.tool()
    def connect_debugger(config: dict[str, Any]) -> dict[str, Any]:
        return svc.connect_debugger(config)

    @mcp.tool()
    def connect_serial(port: str, baud: int, config: dict[str, Any] | None = None) -> dict[str, Any]:
        return svc.connect_serial(port, baud, config)

    @mcp.tool()
    def execute_raw_tcl(cmd: str) -> dict[str, Any]:
        return svc.execute_raw_tcl(cmd)

    @mcp.tool()
    def control_target(action: str) -> dict[str, Any]:
        return svc.control_target(action)

    @mcp.tool()
    def reset_target(reset_type: str) -> dict[str, Any]:
        return svc.reset_target(reset_type)

    @mcp.tool()
    def read_memory(address: str, count: int, width: int = 32) -> dict[str, Any]:
        return svc.read_memory(address, count, width)

    @mcp.tool()
    def write_memory(
        address: str,
        value: int,
        width: int = 32,
        dry_run: bool = False,
        confirm: bool = False,
    ) -> dict[str, Any]:
        return svc.write_memory(address, value, width, dry_run=dry_run, confirm=confirm)

    @mcp.tool()
    def read_peripheral(name: str) -> dict[str, Any]:
        return svc.read_peripheral(name)

    @mcp.tool()
    def manage_breakpoint(point_type: str, addr: str, action: str) -> dict[str, Any]:
        return svc.manage_breakpoint(point_type=point_type, addr=addr, action=action)

    @mcp.tool()
    def serial_write(data: str, mode: str = "ascii") -> dict[str, Any]:
        return svc.serial_write(data, mode)

    @mcp.tool()
    def serial_read_buffer(lines: int, keyword: str | None = None) -> dict[str, Any]:
        return svc.serial_read_buffer(lines, keyword)

    @mcp.tool()
    def serial_set_trigger(keyword: str, action: str) -> dict[str, Any]:
        return svc.serial_set_trigger(keyword, action)

    @mcp.tool()
    def set_flash_mode(enabled: bool) -> dict[str, Any]:
        return svc.set_flash_mode(enabled)

    @mcp.tool()
    def flash_program(
        file_path: str,
        address: str | None = None,
        verify: bool = True,
        reset: bool = True,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        return svc.flash_program(file_path, address, verify=verify, reset=reset, dry_run=dry_run)

    @mcp.tool()
    def submit_task(code: str, timeout_ms: int = 30_000) -> dict[str, Any]:
        return svc.submit_task(code, timeout_ms=timeout_ms)

    @mcp.tool()
    def get_task_result(id: str) -> dict[str, Any]:
        return svc.get_task_result(task_id=id)

    @mcp.tool()
    def cancel_task(id: str, force: bool = False) -> dict[str, Any]:
        return svc.cancel_task(task_id=id, force=force)

    @mcp.tool()
    def emergency_stop() -> dict[str, Any]:
        return svc.emergency_stop()

    return mcp


def main() -> None:
    mcp = create_server()
    mcp.run()


if __name__ == "__main__":
    main()
