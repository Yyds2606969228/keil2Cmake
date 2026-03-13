from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import ArtifactRecord


class ArtifactRegistry:
    def __init__(self) -> None:
        self._records: list[ArtifactRecord] = []

    def register(
        self,
        *,
        kind: str,
        path: str,
        role: str,
        producer: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        record = ArtifactRecord(
            kind=kind,
            path=path,
            role=role,
            producer=producer,
            metadata=metadata or {},
        )
        self._records.append(record)
        return record

    def latest(self, role: str) -> ArtifactRecord | None:
        for record in reversed(self._records):
            if record.role == role:
                return record
        return None

    def snapshot(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._records]

    def grouped(self) -> dict[str, list[dict[str, Any]]]:
        output: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in self._records:
            output[record.role].append(record.to_dict())
        return dict(output)
