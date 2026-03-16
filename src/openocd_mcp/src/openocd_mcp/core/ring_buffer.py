"""Simple thread-safe line ring buffer."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from threading import RLock
from typing import Any


class LineRingBuffer:
    def __init__(self, capacity_bytes: int = 100 * 1024) -> None:
        self.capacity_bytes = capacity_bytes
        self._lines: deque[str] = deque()
        self._size_bytes = 0
        self._lock = RLock()

    def append(self, line: str) -> None:
        encoded_size = len(line.encode("utf-8", errors="replace")) + 1
        with self._lock:
            self._lines.append(line)
            self._size_bytes += encoded_size
            while self._size_bytes > self.capacity_bytes and self._lines:
                removed = self._lines.popleft()
                self._size_bytes -= len(removed.encode("utf-8", errors="replace")) + 1

    def tail(self, lines: int = 50, keyword: str | None = None) -> list[str]:
        with self._lock:
            selected = list(self._lines)[-max(lines, 0) :]
        if keyword:
            return [line for line in selected if keyword in line]
        return selected

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
            self._size_bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._lines)


class RecordRingBuffer:
    """Thread-safe ring buffer for structured records, bounded by byte size."""

    def __init__(self, capacity_bytes: int = 256 * 1024) -> None:
        self.capacity_bytes = capacity_bytes
        self._records: deque[dict[str, Any]] = deque()
        self._sizes: deque[int] = deque()
        self._size_bytes = 0
        self._lock = RLock()

    @staticmethod
    def _estimate_size(record: dict[str, Any]) -> int:
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return len(encoded.encode("utf-8", errors="replace")) + 1

    def append(self, record: dict[str, Any]) -> None:
        stored = deepcopy(record)
        size = self._estimate_size(stored)
        with self._lock:
            self._records.append(stored)
            self._sizes.append(size)
            self._size_bytes += size
            while self._size_bytes > self.capacity_bytes and self._records:
                self._records.popleft()
                removed_size = self._sizes.popleft()
                self._size_bytes -= removed_size

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(item) for item in self._records]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._sizes.clear()
            self._size_bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
