"""Simple event bus for in-process callbacks."""

from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Any, Callable

EventCallback = Callable[[dict[str, Any]], None]


class EventBus:
    def __init__(self) -> None:
        self._listeners: dict[str, list[EventCallback]] = defaultdict(list)
        self._lock = RLock()

    def on(self, event_name: str, callback: EventCallback) -> None:
        with self._lock:
            self._listeners[event_name].append(callback)

    def emit(self, event_name: str, payload: dict[str, Any] | None = None) -> None:
        data = payload or {}
        with self._lock:
            listeners = list(self._listeners[event_name])
        for callback in listeners:
            callback(data)
