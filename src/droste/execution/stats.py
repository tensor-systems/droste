from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExecutionStats:
    """Mutable execution statistics."""

    depth: int = 0
    calls_made: int = 0
    total_tokens: int = 0
    retrieved_ids: list[str] = field(default_factory=list)
