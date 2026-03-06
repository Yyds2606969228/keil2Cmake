"""Common API response model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Status = Literal["success", "partial_success", "failed"]


@dataclass(slots=True)
class ToolResult:
    success: bool
    status: Status
    data: dict[str, Any] | None = None
    error_code: str | None = None
    message: str = "ok"
    raw_output: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "data": self.data,
            "error_code": self.error_code,
            "message": self.message,
            "raw_output": self.raw_output,
            "timestamp": self.timestamp.isoformat(),
        }


def ok(
    data: dict[str, Any] | None = None,
    *,
    message: str = "ok",
    raw_output: str | None = None,
) -> ToolResult:
    return ToolResult(
        success=True,
        status="success",
        data=data,
        message=message,
        raw_output=raw_output,
    )


def partial(
    data: dict[str, Any] | None = None,
    *,
    message: str,
    raw_output: str | None = None,
) -> ToolResult:
    return ToolResult(
        success=False,
        status="partial_success",
        data=data,
        message=message,
        raw_output=raw_output,
    )


def fail(
    *,
    error_code: str,
    message: str,
    raw_output: str | None = None,
    data: dict[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        success=False,
        status="failed",
        data=data,
        error_code=error_code,
        message=message,
        raw_output=raw_output,
    )
