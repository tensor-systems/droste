from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .redaction import redact_secrets

_BATCH_ERROR_DETAIL_STRING_LIMITS = {
    "request_id": 256,
    "batch_id": 256,
    "item_id": 256,
    "layer": 128,
    "cause": 128,
    "code": 128,
}


class SandboxError(Exception):
    """Raised when environment execution fails."""

    pass


class PolicyError(SandboxError):
    """Raised when generated code violates the RLM execution contract."""

    pass


@dataclass(frozen=True, slots=True)
class BatchItemErrorDetails:
    """Allowlisted, payload-free metadata for one failed batch item.

    Providers may expose much larger error objects. This value deliberately
    admits only bounded scalar correlation and classification fields so it can
    cross broker and runner serialization boundaries without retaining request
    or response payloads by accident.
    """

    request_id: str | None = None
    batch_id: str | None = None
    item_id: str | None = None
    layer: str | None = None
    cause: str | None = None
    status_code: int | None = None
    code: str | None = None
    retryable: bool | None = None

    def __post_init__(self) -> None:
        for name, limit in _BATCH_ERROR_DETAIL_STRING_LIMITS.items():
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, redact_secrets(value.strip())[:limit])
        if self.status_code is not None and (
            not isinstance(self.status_code, int)
            or isinstance(self.status_code, bool)
            or not 100 <= self.status_code <= 599
        ):
            raise ValueError("status_code must be an integer from 100 through 599")
        if self.retryable is not None and not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a boolean")

    def to_dict(self) -> dict[str, str | int | bool]:
        """Return the compact JSON-compatible public representation."""

        return {
            name: value
            for name in (
                "request_id",
                "batch_id",
                "item_id",
                "layer",
                "cause",
                "status_code",
                "code",
                "retryable",
            )
            if (value := getattr(self, name)) is not None
        }


class BatchItemError(RuntimeError):
    """Batch-item failure retaining typed safe details alongside its string."""

    def __init__(self, message: str, details: BatchItemErrorDetails) -> None:
        if not isinstance(details, BatchItemErrorDetails):
            raise TypeError("details must be BatchItemErrorDetails")
        self.details = details
        super().__init__(message)


@dataclass
class RLMError:
    """Structured error for RLM execution."""

    type: str
    message: str
    code: str | None = None
    details: dict[str, Any] | None = None
