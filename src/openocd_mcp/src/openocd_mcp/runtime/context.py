"""Script execution context exposed to user code."""

from __future__ import annotations

from threading import Event, RLock
from typing import Any, Callable

from ..core.ring_buffer import LineRingBuffer, RecordRingBuffer


class DebuggerProxy:
    def __init__(self, ctx: "DebugContext") -> None:
        self._ctx = ctx

    def read32(self, addr: int) -> int:
        return self._ctx.read32(addr)

    def write32(self, addr: int, value: int) -> None:
        self._ctx.write32(addr, value)

    def halt(self) -> None:
        self._ctx.halt()

    def resume(self) -> None:
        self._ctx.resume()


class SerialProxy:
    def __init__(self, ctx: "DebugContext") -> None:
        self._ctx = ctx

    def write(self, data: str) -> None:
        self._ctx.serial_write(data)


class DebugContext:
    def __init__(
        self,
        *,
        read32: Callable[[int], int],
        write32: Callable[[int, int], None],
        halt: Callable[[], None],
        resume: Callable[[], None],
        serial_write: Callable[[str], None],
        cancel_event: Event,
        record_capacity_bytes: int = 256 * 1024,
        log_capacity_bytes: int = 64 * 1024,
    ) -> None:
        self._read32 = read32
        self._write32 = write32
        self._halt = halt
        self._resume = resume
        self._serial_write = serial_write
        self._cancel_event = cancel_event
        self._lock = RLock()
        self._records = RecordRingBuffer(capacity_bytes=record_capacity_bytes)
        self._logs = LineRingBuffer(capacity_bytes=log_capacity_bytes)
        self.debugger = DebuggerProxy(self)
        self.serial = SerialProxy(self)

    def read32(self, addr: int) -> int:
        self.raise_if_cancelled()
        return self._read32(addr)

    def write32(self, addr: int, value: int) -> None:
        self.raise_if_cancelled()
        self._write32(addr, value)

    def halt(self) -> None:
        self.raise_if_cancelled()
        self._halt()

    def resume(self) -> None:
        self.raise_if_cancelled()
        self._resume()

    def serial_write(self, data: str) -> None:
        self.raise_if_cancelled()
        self._serial_write(data)

    def record(self, data: dict[str, Any]) -> None:
        self.raise_if_cancelled()
        self._records.append(data)

    def log_data(self, data: dict[str, Any]) -> None:
        self.record(data)

    def log(self, msg: str) -> None:
        self.raise_if_cancelled()
        self._logs.append(msg)

    def raise_if_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise RuntimeError("task_cancelled")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"records": self._records.snapshot(), "logs": self._logs.tail(lines=len(self._logs))}
