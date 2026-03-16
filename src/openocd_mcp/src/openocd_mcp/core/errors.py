"""Error types used by the service layer."""

from __future__ import annotations


class MCPServiceError(Exception):
    """Typed service exception that maps directly to API errors."""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        raw_output: str | None = None,
        status: str = "failed",
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.raw_output = raw_output
        self.status = status
