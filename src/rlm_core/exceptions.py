from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RLMError:
    """Structured error for RLM execution."""
    type: str
    message: str
    code: str | None = None
