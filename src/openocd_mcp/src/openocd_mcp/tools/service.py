"""Service facade that implements all tool operations."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import re
from threading import RLock
from typing import Any, Callable

from ..core import MCPServiceError, fail, ok, partial
from ..core.openocd_path import resolve_openocd_binary
from ..core.models import ToolResult
from ..core.session import SessionState
from ..parsers import ELFResolver, SVDResolver
from ..runtime.context import DebugContext
from ..runtime.task_runtime import TaskRuntime
from ..transport.openocd_tcl import OpenOCDClient, OpenOCDConfig
from ..transport.probe_discovery import discover_debug_probes
from ..transport.serial import SerialConfig, SerialManager


def _parse_dump_values(raw: str) -> list[int]:
    values: list[int] = []
    for line in raw.splitlines():
        _, sep, rhs = line.partition(":")
        content = rhs if sep else line
        hex_prefixed = re.findall(r"0x[0-9a-fA-F]+", content)
        if hex_prefixed:
            for token in hex_prefixed:
                values.append(int(token, 16))
            continue
        for token in re.findall(r"\b[0-9a-fA-F]{1,16}\b", content):
            values.append(int(token, 16))
    return values


def _extract_hex_tokens(raw: str) -> list[str]:
    tokens = re.findall(r"0x[0-9a-fA-F]+", raw)
    return [t.lower() for t in tokens]


def _looks_like_openocd_error(raw: str) -> bool:
    return bool(re.search(r"(^|\s)error[:\s]", raw, flags=re.IGNORECASE))


def _looks_like_breakpoint_exhaustion(raw: str) -> bool:
    patterns = (
        r"no free breakpoint",
        r"resource.*(exhausted|unavailable)",
        r"(cannot|can't|failed to)\s+add\s+(hardware\s+)?breakpoint",
        r"watchpoint.*(full|unavailable|failed)",
    )
    lowered = raw.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def _parse_flash_mismatch(raw: str) -> dict[str, Any] | None:
    normalized = " ".join(raw.split())
    patterns = [
        re.compile(
            r"(?:verify failed|mismatch).*?(0x[0-9a-fA-F]+).*?(?:expected|want)\s*(0x[0-9a-fA-F]+)"
            r".*?(?:got|actual|read)\s*(0x[0-9a-fA-F]+)",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"at\s+(0x[0-9a-fA-F]+).*?(0x[0-9a-fA-F]+)\s*(?:!=|does not match)\s*(0x[0-9a-fA-F]+)",
            flags=re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(normalized)
        if not match:
            continue
        address, expected, actual = (token.lower() for token in match.groups())
        return {
            "detected": True,
            "message": "Flash verify mismatch detected.",
            "address": address,
            "expected": expected,
            "actual": actual,
            "raw_tokens": _extract_hex_tokens(raw),
        }
    lowered = normalized.lower()
    if "verify failed" in lowered or "mismatch" in lowered or "does not match" in lowered:
        return {
            "detected": True,
            "message": "Flash verify mismatch detected, but address/data could not be parsed.",
            "raw_tokens": _extract_hex_tokens(raw),
        }
    return None


class OpenOCDMCPService:
    def __init__(
        self,
        *,
        openocd: OpenOCDClient | None = None,
        serial_mgr: SerialManager | None = None,
        svd: SVDResolver | None = None,
        elf: ELFResolver | None = None,
        tasks: TaskRuntime | None = None,
    ) -> None:
        self.session = SessionState()
        self.openocd = openocd or OpenOCDClient()
        self.serial_mgr = serial_mgr or SerialManager()
        self.svd = svd or SVDResolver()
        self.elf = elf or ELFResolver()
        self.tasks = tasks or TaskRuntime()
        self._lock = RLock()
        self._task_seq = 0
        self._bp_seq = 0
        self._breakpoints: dict[int, dict[str, Any]] = {}
        self._trigger_history: list[dict[str, Any]] = []
        if hasattr(self.openocd, "add_log_listener"):
            self.openocd.add_log_listener(self._on_openocd_log)

    def _guard(self, fn: Callable[[], ToolResult]) -> dict[str, Any]:
        try:
            return fn().to_dict()
        except MCPServiceError as exc:
            return fail(
                error_code=exc.error_code,
                message=exc.message,
                raw_output=exc.raw_output,
            ).to_dict()
        except Exception as exc:  # noqa: BLE001
            return fail(error_code="SYS_UNEXPECTED", message=f"{exc.__class__.__name__}: {exc}").to_dict()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _next_task_id(self) -> str:
        with self._lock:
            self._task_seq += 1
            return f"task-{self._task_seq:04d}"

    def _next_breakpoint_id(self) -> int:
        with self._lock:
            self._bp_seq += 1
            return self._bp_seq

    def _on_openocd_log(self, line: str, event: dict[str, Any] | None) -> None:
        if not event:
            return
        event_name = str(event.get("event", ""))
        if event_name == "TARGET_HALTED":
            self.session.update(target_state="halted")
            self.tasks.emit("TARGET_HALTED", event)
        elif event_name == "WATCHPOINT_HIT":
            self.tasks.emit("WATCHPOINT_HIT", event)
        self.tasks.emit("OPENOCD_LOG", {"line": line, "event": event})

    def list_debug_probes(self) -> dict[str, Any]:
        def action() -> ToolResult:
            serial_ports = SerialManager.list_ports()
            probes = discover_debug_probes(serial_ports)
            data = {
                "probes": probes,
                "count": len(probes),
                "note": (
                    "Best-effort discovery. If probe is missing, use connect_debugger "
                    "to validate actual OpenOCD connectivity."
                ),
            }
            return ok(data=data)

        return self._guard(action)

    def list_serial_ports(self) -> dict[str, Any]:
        return self._guard(lambda: ok(data={"ports": SerialManager.list_ports()}))

    def connect_debugger(self, config: dict[str, Any]) -> dict[str, Any]:
        def action() -> ToolResult:
            if config.get("flash_mode") is not None:
                self.session.update(flash_mode=bool(config["flash_mode"]))
            svd_path = config.get("svd_path")
            elf_path = config.get("elf_path")
            if config.get("auto_start", True):
                openocd_bin = config.get("openocd_bin") or resolve_openocd_binary() or "openocd"
                openocd_config = OpenOCDConfig(
                    adapter=config["adapter"],
                    target=config["target"],
                    transport=config.get("transport", "swd"),
                    tcl_port=int(config.get("tcl_port", 6666)),
                    telnet_port=int(config.get("telnet_port", 4444)),
                    openocd_bin=openocd_bin,
                    extra_args=list(config.get("extra_args", [])),
                    port_scan_limit=int(config.get("port_scan_limit", 20)),
                )
                self.openocd.start(openocd_config)
            else:
                host = config.get("host", "127.0.0.1")
                tcl_port = int(config.get("tcl_port", 6666))
                telnet_port = int(config.get("telnet_port", 4444))
                self.openocd.connect(
                    host=host,
                    port=tcl_port,
                )
                if hasattr(self.openocd, "start_log_monitor"):
                    self.openocd.start_log_monitor(host=host, port=telnet_port)
            if svd_path:
                self.svd.load(svd_path)
            if elf_path:
                self.elf.load(elf_path)
            version_raw = self.openocd.execute("version")
            self.session.update(debugger_connected=True, target_state="connected")
            return ok(
                data={
                    "status": "connected",
                    "version": version_raw.strip(),
                    "tcl_port": getattr(self.openocd, "tcl_port", None),
                    "telnet_port": getattr(self.openocd, "telnet_port", None),
                    "session": self.session.snapshot(),
                },
                raw_output=version_raw,
            )

        return self._guard(action)

    def connect_serial(self, port: str, baud: int, config: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg = config or {}

        def action() -> ToolResult:
            serial_cfg = SerialConfig(
                port=port,
                baud=baud,
                bytesize=int(cfg.get("bytesize", 8)),
                parity=str(cfg.get("parity", "N")),
                stopbits=int(cfg.get("stopbits", 1)),
                timeout=float(cfg.get("timeout", 0.1)),
            )
            self.serial_mgr.connect(serial_cfg)
            self.session.update(serial_connected=True)
            return ok(data={"serial": asdict(serial_cfg), "session": self.session.snapshot()})

        return self._guard(action)

    def execute_raw_tcl(self, cmd: str) -> dict[str, Any]:
        def run() -> ToolResult:
            raw = self.openocd.execute(cmd)
            return ok(data={"raw_output": raw}, raw_output=raw)

        return self._guard(run)

    def reset_target(self, reset_type: str = "halt") -> dict[str, Any]:
        def run() -> ToolResult:
            allowed = {"halt", "init", "run"}
            if reset_type not in allowed:
                raise MCPServiceError("DBG_INVALID_RESET_TYPE", f"Unsupported reset type: {reset_type}")
            raw = self.openocd.execute(f"reset {reset_type}")
            state = "halted" if reset_type in {"halt", "init"} else "running"
            self.session.update(target_state=state)
            return ok(data={"success": True, "state": state}, raw_output=raw)

        return self._guard(run)

    def control_target(self, action: str) -> dict[str, Any]:
        def run() -> ToolResult:
            current_state = self.session.snapshot().get("target_state")
            if action == "halt" and current_state == "halted":
                return ok(data={"state": "halted", "pc": self.openocd.get_pc()}, message="Target already halted.")
            if action == "resume" and current_state == "running":
                return ok(data={"state": "running", "pc": self.openocd.get_pc()}, message="Target already running.")
            raw = self.openocd.control_target(action)
            target_state = "halted" if action in {"halt", "step"} else "running"
            pc = self.openocd.get_pc()
            self.session.update(target_state=target_state)
            if action == "halt":
                self.tasks.emit("TARGET_HALTED", {"state": target_state, "pc": pc, "time": self._utc_now()})
            return ok(data={"state": target_state, "pc": pc}, raw_output=raw)

        return self._guard(run)

    def _resolve_address(self, address: str | int) -> int:
        if isinstance(address, int):
            return address
        token = address.strip()
        if "->" in token:
            info = self.svd.resolve(token)
            return info.address
        return self.elf.resolve(token)

    def read_memory(self, address: str | int, count: int, width: int = 32) -> dict[str, Any]:
        def run() -> ToolResult:
            if width not in {8, 16, 32}:
                raise MCPServiceError("DBG_INVALID_WIDTH", "Width must be 8, 16, or 32.")
            if count <= 0:
                raise MCPServiceError("DBG_INVALID_COUNT", "Count must be > 0.")
            addr = self._resolve_address(address)
            cmd = {8: "mdb", 16: "mdh", 32: "mdw"}[width]
            raw = self.openocd.execute(f"{cmd} 0x{addr:x} {count}")
            parsed = _parse_dump_values(raw)[:count]
            if not parsed:
                return partial(
                    data={"base": f"0x{addr:x}", "width": width, "count": count},
                    message="Memory read returned no parsable values.",
                    raw_output=raw,
                )
            return ok(
                data={"base": f"0x{addr:x}", "width": width, "count": count, "data": parsed},
                raw_output=raw,
            )

        return self._guard(run)

    def write_memory(
        self,
        address: str | int,
        value: int,
        width: int = 32,
        *,
        dry_run: bool = False,
        confirm: bool = False,
    ) -> dict[str, Any]:
        def run() -> ToolResult:
            if width not in {8, 16, 32}:
                raise MCPServiceError("DBG_INVALID_WIDTH", "Width must be 8, 16, or 32.")
            addr = self._resolve_address(address)
            if 0x08000000 <= addr < 0x0A000000 and not self.session.flash_mode:
                raise MCPServiceError(
                    "SAFE_FLASH_MODE_REQUIRED",
                    "Flash write blocked. Enable flash_mode to proceed.",
                )
            is_critical = 0x40000000 <= addr < 0x60000000
            command_preview = {
                "action": "write_memory",
                "address": f"0x{addr:x}",
                "value": value,
                "width": width,
                "critical_region": is_critical,
            }
            if dry_run:
                return ok(
                    data={"dry_run": True, "command": command_preview},
                    message="Dry-run only. Re-run with dry_run=false to execute.",
                )
            if is_critical and not confirm:
                raise MCPServiceError(
                    "SAFE_CONFIRM_REQUIRED",
                    "Critical peripheral register write requires confirm=true or dry_run=true.",
                )
            cmd = {8: "mwb", 16: "mwh", 32: "mww"}[width]
            raw = self.openocd.execute(f"{cmd} 0x{addr:x} 0x{value:x}")
            return ok(
                data={
                    "dry_run": False,
                    "address": f"0x{addr:x}",
                    "value": value,
                    "width": width,
                    "critical_region": is_critical,
                },
                raw_output=raw,
            )

        return self._guard(run)

    def set_flash_mode(self, enabled: bool) -> dict[str, Any]:
        def run() -> ToolResult:
            self.session.update(flash_mode=bool(enabled))
            return ok(data={"flash_mode": self.session.snapshot()["flash_mode"]})

        return self._guard(run)

    def flash_program(
        self,
        file_path: str,
        address: str | int | None = None,
        *,
        verify: bool = True,
        reset: bool = True,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        def run() -> ToolResult:
            if not self.session.flash_mode:
                raise MCPServiceError(
                    "SAFE_FLASH_MODE_REQUIRED",
                    "Flash programming blocked. Enable flash_mode first.",
                )
            if not Path(file_path).exists():
                raise MCPServiceError("SAFE_FLASH_FILE_NOT_FOUND", f"Flash image not found: {file_path}")
            command_preview = {
                "action": "program",
                "file_path": file_path,
                "address": address,
                "verify": verify,
                "reset": reset,
            }
            if dry_run:
                return ok(
                    data={"dry_run": True, "command": command_preview},
                    message="Dry-run only. Re-run with dry_run=false to execute.",
                )
            raw = self.openocd.program(file_path, address=address, verify=verify, reset=reset, timeout_s=180.0)
            mismatch = _parse_flash_mismatch(raw) if verify else None
            if mismatch is not None:
                return fail(
                    error_code="DBG_FLASH_VERIFY_MISMATCH",
                    message="Flash verification failed.",
                    raw_output=raw,
                    data={"dry_run": False, "programmed": False, "command": command_preview, "mismatch": mismatch},
                )
            return ok(
                data={"dry_run": False, "programmed": True, "command": command_preview},
                raw_output=raw,
                message="Flash programming completed.",
            )

        return self._guard(run)

    def manage_breakpoint(
        self,
        point_type: str,
        addr: str | int,
        action: str,
        *,
        length: int = 4,
        access: str = "w",
    ) -> dict[str, Any]:
        def run() -> ToolResult:
            normalized_action = action.lower()
            address = self._resolve_address(addr)
            raw = self.openocd.manage_breakpoint(
                point_type=point_type,
                address=address,
                action=normalized_action,
                length=length,
                access=access,
            )
            if normalized_action == "add" and _looks_like_breakpoint_exhaustion(raw):
                raise MCPServiceError(
                    "DBG_BREAKPOINT_RESOURCE_EXHAUSTED",
                    "No available hardware breakpoint/watchpoint resource.",
                    raw_output=raw,
                )
            if _looks_like_openocd_error(raw):
                raise MCPServiceError(
                    "DBG_BREAKPOINT_OPERATION_FAILED",
                    "OpenOCD reported breakpoint/watchpoint operation error.",
                    raw_output=raw,
                )
            if normalized_action == "add":
                bp_id = self._next_breakpoint_id()
                self._breakpoints[bp_id] = {
                    "id": bp_id,
                    "type": point_type,
                    "address": address,
                    "length": length,
                    "access": access,
                }
                return ok(data={"id": bp_id, "type": point_type, "address": f"0x{address:x}"}, raw_output=raw)
            removed_id = None
            for key, value in list(self._breakpoints.items()):
                if value["type"] == point_type and value["address"] == address:
                    removed_id = key
                    del self._breakpoints[key]
                    break
            return ok(
                data={"id": removed_id, "type": point_type, "address": f"0x{address:x}", "deleted": True},
                raw_output=raw,
            )

        return self._guard(run)

    def read_peripheral(self, name: str) -> dict[str, Any]:
        def run() -> ToolResult:
            try:
                info = self.svd.resolve(name)
            except MCPServiceError:
                fallback = self.svd.resolve_best_effort(name)
                address = fallback.get("address")
                snippet = fallback.get("raw_xml_snippet")
                if isinstance(address, int):
                    raw = self.openocd.execute(f"mdw 0x{address:x} 1")
                    values = _parse_dump_values(raw)
                    raw_hex = f"0x{values[0]:x}" if values else None
                    return partial(
                        data={
                            "status": "partial_success",
                            "parsed": None,
                            "raw_xml_snippet": snippet,
                            "memory_dump": {f"0x{address:x}": raw_hex},
                            "note": str(fallback.get("error") or "SVD fallback mode"),
                        },
                        message="SVD semantic parsing failed. Returned XML snippet and raw memory dump.",
                        raw_output=raw,
                    )
                return partial(
                    data={
                        "status": "partial_success",
                        "parsed": None,
                        "raw_xml_snippet": snippet,
                        "memory_dump": None,
                        "note": str(fallback.get("error") or "SVD fallback mode"),
                    },
                    message="SVD semantic parsing failed and address could not be resolved.",
                )

            raw = self.openocd.execute(f"mdw 0x{info.address:x} 1")
            values = _parse_dump_values(raw)
            if not values:
                snippet = self.svd.raw_xml_snippet(name)
                return partial(
                    data={
                        "status": "partial_success",
                        "parsed": None,
                        "raw_xml_snippet": snippet,
                        "memory_dump": {f"0x{info.address:x}": None},
                        "note": "SVD resolved address, but memory value could not be parsed.",
                    },
                    message="SVD resolved address, but memory value could not be parsed.",
                    raw_output=raw,
                )
            value = values[0]
            fields = self.svd.decode_fields(info, value)
            return ok(
                data={
                    "status": "success",
                    "register": {
                        "name": info.register,
                        "peripheral": info.peripheral,
                        "address": f"0x{info.address:x}",
                        "value": value,
                        "fields": fields,
                    },
                },
                raw_output=raw,
            )

        return self._guard(run)

    def serial_write(self, data: str, mode: str = "ascii") -> dict[str, Any]:
        return self._guard(lambda: self._serial_write_impl(data, mode))

    def _serial_write_impl(self, data: str, mode: str = "ascii") -> ToolResult:
        self.serial_mgr.write(data, mode=mode)
        return ok(data={"written": True, "mode": mode})

    def serial_read_buffer(self, lines: int = 50, keyword: str | None = None) -> dict[str, Any]:
        return self._guard(lambda: ok(data={"lines": self.serial_mgr.read_buffer(lines=lines, keyword=keyword)}))

    def serial_set_trigger(self, keyword: str, action: str) -> dict[str, Any]:
        def on_trigger(trigger_action: str, line: str) -> None:
            triggered_at = self._utc_now()
            self._trigger_history.append(
                {"keyword": keyword, "action": trigger_action, "line": line, "triggered_at": triggered_at}
            )
            if trigger_action == "halt":
                self.openocd.control_target("halt")
                self.session.update(target_state="halted")
                self.tasks.emit("TARGET_HALTED", {"line": line, "time": triggered_at})
                return
            self.openocd.execute(trigger_action)

        def run() -> ToolResult:
            self.serial_mgr.set_trigger(regex=keyword, action=action, callback=on_trigger)
            return ok(data={"keyword": keyword, "action": action})

        return self._guard(run)

    def _read32_internal(self, address: int) -> int:
        raw = self.openocd.execute(f"mdw 0x{address:x} 1")
        parsed = _parse_dump_values(raw)
        if not parsed:
            raise MCPServiceError("DBG_READ_FAILED", f"Cannot parse value from: {raw}", raw_output=raw)
        return parsed[0]

    def _write32_internal(self, address: int, value: int) -> None:
        self.openocd.execute(f"mww 0x{address:x} 0x{value:x}")

    def submit_task(self, code: str, timeout_ms: int = 30_000) -> dict[str, Any]:
        def context_factory(cancel_event: Any) -> DebugContext:
            return DebugContext(
                read32=self._read32_internal,
                write32=self._write32_internal,
                halt=lambda: self.openocd.control_target("halt"),
                resume=lambda: self.openocd.control_target("resume"),
                serial_write=lambda data: self.serial_mgr.write(data, mode="ascii"),
                cancel_event=cancel_event,
            )

        def run() -> ToolResult:
            task_id = self._next_task_id()
            self.tasks.submit(name=task_id, code=code, context_factory=context_factory, timeout_ms=timeout_ms)
            return ok(data={"id": task_id, "timeout_ms": timeout_ms})

        return self._guard(run)

    def get_task_result(self, task_id: str) -> dict[str, Any]:
        return self._guard(lambda: ok(data=self.tasks.get(task_id)))

    def cancel_task(self, task_id: str, *, force: bool = False) -> dict[str, Any]:
        return self._guard(
            lambda: ok(
                data={"cancelled": self.tasks.cancel(task_id, force=force), "id": task_id, "force": force}
            )
        )

    def emergency_stop(self) -> dict[str, Any]:
        def run() -> ToolResult:
            task_report = self.tasks.cancel_all(force=True, wait_timeout_s=0.5)
            errors: list[str] = []
            if task_report["alive"]:
                errors.append(f"task_stop_failed={task_report['alive']}")
            tracked = list(self._breakpoints.values())
            bp_report: dict[str, Any] = {"verified": False, "errors": [], "remaining": {"bp": [], "wp": []}}
            try:
                bp_report = self.openocd.clear_all_breakpoints(tracked_points=tracked)
                if not bp_report.get("verified"):
                    errors.append("breakpoint_clear_not_verified")
            except MCPServiceError as exc:
                errors.append(f"breakpoint_clear_error={exc.message}")
            self._breakpoints.clear()
            try:
                self.openocd.execute("reset halt")
            except MCPServiceError as exc:
                errors.append(f"reset_halt_failed={exc.message}")
            finally:
                self.openocd.close()
                self.serial_mgr.close()
            self.session.update(
                debugger_connected=False,
                serial_connected=False,
                target_state="halted",
            )
            if errors:
                raise MCPServiceError(
                    "SYS_EMERGENCY_STOP_INCOMPLETE",
                    "Emergency stop executed with unresolved issues.",
                    raw_output=str({"errors": errors, "tasks": task_report, "breakpoints": bp_report}),
                )
            return ok(
                data={
                    "session": self.session.snapshot(),
                    "tasks": task_report,
                    "breakpoints": bp_report,
                },
                message="Emergency stop completed.",
            )

        return self._guard(run)
