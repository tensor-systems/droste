from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SandboxError(Exception):
    """Raised when environment execution fails."""
    pass


class BatchLLMError(SandboxError):
    """Raised when a batch sub-LLM request returns one or more errors."""

    def __init__(self, message: str, errors: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.errors = errors


@dataclass
class RLMError:
    """Structured error for RLM execution."""
    type: str
    message: str
    code: str | None = None
    details: dict[str, Any] | None = None
