"""Serial manager with trigger support and line ring buffer."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, RLock, Thread
import re
import time
from typing import Callable

from ...core.errors import MCPServiceError
from ...core.ring_buffer import LineRingBuffer

try:
    import serial  # type: ignore[import-untyped]
    from serial.tools import list_ports  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - dependency is optional in tests
    serial = None
    list_ports = None


@dataclass(slots=True)
class SerialConfig:
    port: str
    baud: int = 115200
    bytesize: int = 8
    parity: str = "N"
    stopbits: int = 1
    timeout: float = 0.1


@dataclass(slots=True)
class TriggerRule:
    regex: re.Pattern[str]
    action: str
    callback: Callable[[str, str], None]
    cooldown_ms: int = 100
    last_fired_ms: int = 0

    def should_fire(self, line: str, now_ms: int) -> bool:
        if not self.regex.search(line):
            return False
        if now_ms - self.last_fired_ms < self.cooldown_ms:
            return False
        self.last_fired_ms = now_ms
        return True


class SerialManager:
    def __init__(self, buffer_capacity_bytes: int = 100 * 1024) -> None:
        self.buffer = LineRingBuffer(capacity_bytes=buffer_capacity_bytes)
        self._serial: object | None = None
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lock = RLock()
        self._triggers: list[TriggerRule] = []

    @staticmethod
    def list_ports() -> list[dict[str, str]]:
        if list_ports is None:
            return []
        output: list[dict[str, str]] = []
        for port in list_ports.comports():
            output.append({"port": port.device, "description": port.description, "hwid": port.hwid})
        return output

    def connect(self, config: SerialConfig) -> None:
        if serial is None:
            raise MCPServiceError("UART_DEPENDENCY_MISSING", "pyserial is not installed.")
        with self._lock:
            if self._serial is not None:
                raise MCPServiceError("UART_ALREADY_CONNECTED", "Serial is already connected.")
            try:
                self._serial = serial.Serial(  # type: ignore[operator]
                    port=config.port,
                    baudrate=config.baud,
                    bytesize=config.bytesize,
                    parity=config.parity,
                    stopbits=config.stopbits,
                    timeout=config.timeout,
                )
            except Exception as exc:  # pragma: no cover - hardware dependent
                raise MCPServiceError("UART_CONNECT_FAILED", f"Failed to open serial port: {exc}") from exc
            self._stop_event.clear()
            self._thread = Thread(target=self._read_loop, name="serial-reader", daemon=True)
            self._thread.start()

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            serial_obj = self._serial
            if serial_obj is None:
                return
            try:
                waiting = getattr(serial_obj, "in_waiting", 0)
                if waiting:
                    raw = serial_obj.readline()
                else:
                    time.sleep(0.01)
                    continue
            except Exception:  # pragma: no cover - hardware dependent
                time.sleep(0.05)
                continue
            if not raw:
                continue
            if isinstance(raw, str):
                line = raw.strip()
            else:
                line = self._decode_line(raw)
            if line:
                self._consume_line(line)

    @staticmethod
    def _decode_line(raw: bytes) -> str:
        try:
            return raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            try:
                return raw.decode("latin-1").strip()
            except UnicodeDecodeError:
                return repr(raw)

    def _consume_line(self, line: str) -> None:
        self.buffer.append(line)
        now_ms = int(time.monotonic() * 1000)
        with self._lock:
            triggers = list(self._triggers)
        for trigger in triggers:
            if trigger.should_fire(line, now_ms):
                Thread(
                    target=trigger.callback,
                    args=(trigger.action, line),
                    name="serial-trigger",
                    daemon=True,
                ).start()

    def feed_line_for_test(self, line: str) -> None:
        self._consume_line(line)

    def write(self, data: str, mode: str = "ascii") -> None:
        serial_obj = self._serial
        if serial_obj is None:
            raise MCPServiceError("UART_NOT_CONNECTED", "Serial port is not connected.")
        if mode not in {"ascii", "hex"}:
            raise MCPServiceError("UART_INVALID_MODE", f"Unsupported serial write mode: {mode}")
        payload = data.encode("utf-8") if mode == "ascii" else bytes.fromhex(data)
        try:
            serial_obj.write(payload)
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise MCPServiceError("UART_WRITE_FAILED", f"Failed to write serial data: {exc}") from exc

    def read_buffer(self, lines: int = 50, keyword: str | None = None) -> list[str]:
        return self.buffer.tail(lines=lines, keyword=keyword)

    def set_trigger(
        self,
        regex: str,
        action: str,
        callback: Callable[[str, str], None],
        *,
        cooldown_ms: int = 100,
    ) -> None:
        try:
            compiled = re.compile(regex)
        except re.error as exc:
            raise MCPServiceError("UART_REGEX_INVALID", f"Invalid regex: {exc}") from exc
        with self._lock:
            self._triggers.append(
                TriggerRule(regex=compiled, action=action, callback=callback, cooldown_ms=cooldown_ms)
            )

    def clear_triggers(self) -> None:
        with self._lock:
            self._triggers.clear()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        serial_obj = self._serial
        self._serial = None
        if serial_obj is not None:
            try:
                serial_obj.close()
            except Exception:  # pragma: no cover - hardware dependent
                pass

    def is_connected(self) -> bool:
        return self._serial is not None
