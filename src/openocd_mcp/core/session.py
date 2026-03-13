"""Session state container."""

from __future__ import annotations

from threading import RLock
from typing import Any


class SessionState:
    def __init__(self) -> None:
        self._lock = RLock()
        self.debugger_connected = False
        self.serial_connected = False
        self.target_state = "unknown"
        self.flash_mode = False

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "debugger_connected": self.debugger_connected,
                "serial_connected": self.serial_connected,
                "target_state": self.target_state,
                "flash_mode": self.flash_mode,
            }
