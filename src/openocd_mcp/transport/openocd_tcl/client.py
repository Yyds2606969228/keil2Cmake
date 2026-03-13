"""OpenOCD TCL client with optional telnet log monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import re
import socket
import subprocess
import tempfile
from threading import Event, RLock, Thread
import time
from typing import Any, Callable

from ...core.errors import MCPServiceError

EOM = b"\x1a"
LogCallback = Callable[[str, dict[str, Any] | None], None]


@dataclass(slots=True)
class OpenOCDConfig:
    adapter: str
    target: str
    transport: str = "swd"
    tcl_port: int = 6666
    telnet_port: int = 4444
    openocd_bin: str = "openocd"
    extra_args: list[str] = field(default_factory=list)
    port_scan_limit: int = 20


class OpenOCDClient:
    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._process_log_handle: Any | None = None
        self._process_log_path: str | None = None
        self._socket: socket.socket | None = None
        self._telnet_socket: socket.socket | None = None
        self._telnet_thread: Thread | None = None
        self._stop_event = Event()
        self._lock = RLock()
        self._log_callbacks: list[LogCallback] = []
        self.tcl_port = 6666
        self.telnet_port = 4444

    @staticmethod
    def _is_port_free(port: int) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def _choose_ports(self, start_tcl: int, start_telnet: int, limit: int) -> tuple[int, int]:
        for offset in range(max(limit, 1)):
            tcl = start_tcl + offset
            telnet = start_telnet + offset
            if self._is_port_free(tcl) and self._is_port_free(telnet):
                return tcl, telnet
        raise MCPServiceError("SYS_PORT_UNAVAILABLE", "No free TCL/Telnet port pair found.")

    def add_log_listener(self, callback: LogCallback) -> None:
        self._log_callbacks.append(callback)

    def _emit_log(self, line: str) -> None:
        event = self._parse_log_event(line)
        for callback in list(self._log_callbacks):
            callback(line, event)

    @staticmethod
    def _parse_log_event(line: str) -> dict[str, Any] | None:
        lower = line.lower()
        now = datetime.now(timezone.utc).isoformat()
        if "target halted" in lower:
            pc = None
            match = re.search(r"\bpc[:=]\s*(0x[0-9a-fA-F]+)\b", line)
            if match:
                pc = match.group(1)
            return {"event": "TARGET_HALTED", "pc": pc, "raw": line, "time": now}
        if "watchpoint" in lower and ("hit" in lower or "trigger" in lower):
            addr = None
            match = re.search(r"(0x[0-9a-fA-F]+)", line)
            if match:
                addr = match.group(1)
            return {"event": "WATCHPOINT_HIT", "address": addr, "raw": line, "time": now}
        return None

    def _read_process_log_tail(self, max_chars: int = 4000) -> str | None:
        path = self._process_log_path
        if not path or not os.path.exists(path):
            return None
        if self._process_log_handle is not None:
            try:
                self._process_log_handle.flush()
            except Exception:
                pass
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            return None
        if not text:
            return None
        return text[-max_chars:]

    def _start_telnet_monitor(self, host: str, port: int) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect((host, port))
        except OSError as exc:
            raise MCPServiceError(
                "SYS_TELNET_CONNECT_FAILED",
                f"Failed to connect OpenOCD telnet/log port: {exc}",
                raw_output=self._read_process_log_tail(),
            ) from exc
        self._telnet_socket = sock
        self._stop_event.clear()
        self._telnet_thread = Thread(target=self._telnet_loop, name="openocd-telnet-log", daemon=True)
        self._telnet_thread.start()

    def start_log_monitor(self, host: str = "127.0.0.1", port: int | None = None) -> None:
        self._start_telnet_monitor(host, self.telnet_port if port is None else port)

    def _telnet_loop(self) -> None:
        sock = self._telnet_socket
        if sock is None:
            return
        buffer = b""
        while not self._stop_event.is_set():
            try:
                data = sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            buffer += data
            while b"\n" in buffer:
                raw_line, buffer = buffer.split(b"\n", 1)
                line = raw_line.decode("utf-8", errors="replace").strip("\r")
                if line.strip():
                    self._emit_log(line.strip())

    def start(self, config: OpenOCDConfig, startup_timeout_s: float = 8.0) -> None:
        with self._lock:
            if self._socket is not None:
                raise MCPServiceError("SYS_ALREADY_CONNECTED", "OpenOCD TCL is already connected.")
            chosen_tcl, chosen_telnet = self._choose_ports(
                start_tcl=config.tcl_port,
                start_telnet=config.telnet_port,
                limit=config.port_scan_limit,
            )
            self.tcl_port = chosen_tcl
            self.telnet_port = chosen_telnet
            cmd = [
                config.openocd_bin,
                "-f",
                f"interface/{config.adapter}.cfg",
                "-f",
                f"target/{config.target}.cfg",
                "-c",
                f"transport select {config.transport}",
                "-c",
                f"tcl_port {self.tcl_port}",
                "-c",
                f"telnet_port {self.telnet_port}",
            ]
            cmd.extend(config.extra_args)
            try:
                log_file = tempfile.NamedTemporaryFile(  # noqa: PTH123
                    mode="w+",
                    encoding="utf-8",
                    errors="replace",
                    delete=False,
                    prefix="openocd-mcp-",
                    suffix=".log",
                )
                self._process_log_handle = log_file
                self._process_log_path = log_file.name
                self._process = subprocess.Popen(  # noqa: S603
                    cmd,
                    stdout=log_file,
                    stderr=log_file,
                    text=True,
                )
            except OSError as exc:
                raise MCPServiceError("SYS_OPENOCD_START_FAILED", str(exc)) from exc

        deadline = time.monotonic() + startup_timeout_s
        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                output = self._read_process_log_tail()
                self.close()
                raise MCPServiceError(
                    "SYS_OPENOCD_EXITED",
                    "OpenOCD process exited before TCL socket became available.",
                    raw_output=output,
                )
            try:
                self.connect(port=self.tcl_port)
                self.start_log_monitor("127.0.0.1", self.telnet_port)
                return
            except MCPServiceError:
                time.sleep(0.1)

        output = self._read_process_log_tail()
        self.close()
        raise MCPServiceError("SYS_OPENOCD_TIMEOUT", "Timed out waiting for OpenOCD TCL socket.", raw_output=output)

    def connect(self, host: str = "127.0.0.1", port: int = 6666, timeout_s: float = 2.0) -> None:
        with self._lock:
            if self._socket is not None:
                return
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout_s)
            try:
                sock.connect((host, port))
            except OSError as exc:
                sock.close()
                raise MCPServiceError("SYS_TCL_CONNECT_FAILED", f"Failed to connect TCL socket: {exc}") from exc
            self._socket = sock
            self.tcl_port = port

    def execute(self, command: str, timeout_s: float = 2.0) -> str:
        with self._lock:
            if self._socket is None:
                raise MCPServiceError("SYS_NOT_CONNECTED", "TCL socket is not connected.")
            payload = f"{command}\x1a".encode("utf-8")
            self._socket.settimeout(timeout_s)
            try:
                self._socket.sendall(payload)
                chunks: list[bytes] = []
                while True:
                    piece = self._socket.recv(4096)
                    if not piece:
                        break
                    if EOM in piece:
                        before, _, _ = piece.partition(EOM)
                        chunks.append(before)
                        break
                    chunks.append(piece)
            except OSError as exc:
                raise MCPServiceError("DBG_TCL_IO_ERROR", f"TCL RPC failed: {exc}") from exc
        return b"".join(chunks).decode("utf-8", errors="replace").strip()

    def control_target(self, action: str) -> str:
        allowed = {"halt", "resume", "step", "reset", "init", "run"}
        if action not in allowed:
            raise MCPServiceError("DBG_INVALID_ACTION", f"Unsupported control action: {action}")
        command = "reset run" if action == "run" else action
        return self.execute(command)

    def get_pc(self) -> str | None:
        raw = self.execute("reg pc")
        match = re.search(r"(0x[0-9a-fA-F]+|[0-9a-fA-F]{8,16})", raw)
        if not match:
            return None
        token = match.group(1)
        if token.startswith("0x"):
            return token.lower()
        return f"0x{token.lower()}"

    def manage_breakpoint(
        self,
        *,
        point_type: str,
        address: int,
        action: str,
        length: int = 4,
        access: str = "w",
    ) -> str:
        ptype = point_type.lower()
        op = action.lower()
        if ptype not in {"bp", "wp"}:
            raise MCPServiceError("DBG_INVALID_BP_TYPE", "type must be 'bp' or 'wp'.")
        if op not in {"add", "del"}:
            raise MCPServiceError("DBG_INVALID_BP_ACTION", "action must be 'add' or 'del'.")
        if ptype == "bp":
            if op == "add":
                return self.execute(f"bp 0x{address:x}")
            return self.execute(f"rbp 0x{address:x}")
        if op == "add":
            return self.execute(f"wp 0x{address:x} {length} {access}")
        return self.execute(f"rwp 0x{address:x}")

    @staticmethod
    def _parse_addresses(raw: str) -> set[int]:
        return {int(token, 16) for token in re.findall(r"0x[0-9a-fA-F]+", raw)}

    def list_breakpoints(self) -> dict[str, set[int]]:
        bp_out = self.execute("bp")
        wp_out = self.execute("wp")
        return {"bp": self._parse_addresses(bp_out), "wp": self._parse_addresses(wp_out)}

    def clear_all_breakpoints(
        self,
        tracked_points: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        points = self.list_breakpoints()
        if tracked_points:
            for item in tracked_points:
                try:
                    address = int(item.get("address", 0))
                except (TypeError, ValueError):
                    continue
                ptype = str(item.get("type", "")).lower()
                if ptype in {"bp", "wp"}:
                    points[ptype].add(address)
        errors: list[str] = []
        cleared = {"bp": 0, "wp": 0}
        for address in sorted(points["bp"]):
            try:
                self.execute(f"rbp 0x{address:x}")
                cleared["bp"] += 1
            except MCPServiceError as exc:
                errors.append(f"rbp 0x{address:x}: {exc.message}")
        for address in sorted(points["wp"]):
            try:
                self.execute(f"rwp 0x{address:x}")
                cleared["wp"] += 1
            except MCPServiceError as exc:
                errors.append(f"rwp 0x{address:x}: {exc.message}")
        remaining_raw = self.list_breakpoints()
        verified = not remaining_raw["bp"] and not remaining_raw["wp"]
        remaining = {
            "bp": [f"0x{addr:x}" for addr in sorted(remaining_raw["bp"])],
            "wp": [f"0x{addr:x}" for addr in sorted(remaining_raw["wp"])],
        }
        return {"cleared": cleared, "remaining": remaining, "errors": errors, "verified": verified}

    def program(
        self,
        file_path: str,
        *,
        address: str | int | None = None,
        verify: bool = True,
        reset: bool = True,
        timeout_s: float = 60.0,
    ) -> str:
        segments = [f"\"{file_path}\""]
        if address is not None:
            if isinstance(address, int):
                segments.append(f"0x{address:x}")
            else:
                segments.append(str(address))
        if verify:
            segments.append("verify")
        if reset:
            segments.append("reset")
        command = "program " + " ".join(segments)
        return self.execute(command, timeout_s=timeout_s)

    def close(self) -> None:
        self._stop_event.set()
        if self._telnet_socket is not None:
            try:
                self._telnet_socket.close()
            finally:
                self._telnet_socket = None
        if self._telnet_thread and self._telnet_thread.is_alive():
            self._telnet_thread.join(timeout=0.5)
        with self._lock:
            if self._socket is not None:
                try:
                    self._socket.close()
                finally:
                    self._socket = None
            if self._process is not None:
                if self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                self._process = None
            if self._process_log_handle is not None:
                try:
                    self._process_log_handle.close()
                except Exception:
                    pass
                self._process_log_handle = None
            if self._process_log_path is not None:
                try:
                    os.remove(self._process_log_path)
                except OSError:
                    pass
                self._process_log_path = None

    def is_connected(self) -> bool:
        with self._lock:
            return self._socket is not None
