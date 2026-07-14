from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SandboxError(Exception):
    """Raised when environment execution fails."""

    pass


class PolicyError(SandboxError):
    """Raised when generated code violates the RLM execution contract."""

    pass


class SubcallBudgetExceeded(RuntimeError):
    """A subcall dispatch was rejected before exceeding its call budget."""

    pass


@dataclass
class RLMError:
    """Structured error for RLM execution."""

    type: str
    message: str
    code: str | None = None
    details: dict[str, Any] | None = None
