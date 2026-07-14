from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class UsageBreakdown:
    """Trusted token and request totals for one billing scope."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    requests: int
    successes: int

    def __post_init__(self) -> None:
        values = (
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
            self.requests,
            self.successes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError("usage breakdown counts must be non-negative integers")
        if self.successes > self.requests:
            raise ValueError("usage successes cannot exceed requests")

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "requests": self.requests,
            "successes": self.successes,
        }


@dataclass(frozen=True, slots=True)
class ResolvedUsage:
    """Immutable billing projection reconciled with the legacy total."""

    root: UsageBreakdown
    subcall: UsageBreakdown
    unattributed_tokens: int
    total_tokens: int
    wall_time_ms: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.unattributed_tokens, self.total_tokens, self.wall_time_ms)
        ):
            raise ValueError("resolved usage totals must be non-negative integers")
        if (
            self.root.total_tokens + self.subcall.total_tokens + self.unattributed_tokens
            != self.total_tokens
        ):
            raise ValueError("usage token scopes must reconcile to total_tokens")

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "resolved",
            "root": self.root.as_dict(),
            "subcall": self.subcall.as_dict(),
            "unattributed": {"total_tokens": self.unattributed_tokens},
            "total_tokens": self.total_tokens,
            "wall_time_ms": self.wall_time_ms,
        }


@dataclass
class ExecutionStats:
    """Mutable execution statistics."""

    depth: int = 0
    calls_made: int = 0
    successful_calls: int = 0
    total_tokens: int = 0
    root_input_tokens: int = 0
    root_output_tokens: int = 0
    root_total_tokens: int = 0
    root_requests: int = 0
    root_successes: int = 0
    subcall_input_tokens: int = 0
    subcall_output_tokens: int = 0
    subcall_total_tokens: int = 0
    retrieved_ids: list[str] = field(default_factory=list)

    def resolved_usage(self, wall_time_ms: int) -> ResolvedUsage:
        attributed = self.root_total_tokens + self.subcall_total_tokens
        return ResolvedUsage(
            root=UsageBreakdown(
                self.root_input_tokens,
                self.root_output_tokens,
                self.root_total_tokens,
                self.root_requests,
                self.root_successes,
            ),
            subcall=UsageBreakdown(
                self.subcall_input_tokens,
                self.subcall_output_tokens,
                self.subcall_total_tokens,
                self.calls_made,
                self.successful_calls,
            ),
            unattributed_tokens=self.total_tokens - attributed,
            total_tokens=self.total_tokens,
            wall_time_ms=wall_time_ms,
        )
