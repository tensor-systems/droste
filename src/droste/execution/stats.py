from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class UsageBreakdown:
    """Trusted token and request totals for one billing scope."""

    input_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    output_tokens: int
    total_tokens: int
    requests: int
    successes: int
    complete: bool = True

    def __post_init__(self) -> None:
        values = (
            self.input_tokens,
            self.cache_read_tokens,
            self.cache_creation_tokens,
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
        if not isinstance(self.complete, bool):
            raise TypeError("usage completeness must be a bool")
        if (
            self.complete
            and self.cache_read_tokens + self.cache_creation_tokens > self.input_tokens
        ):
            raise ValueError("complete cache token breakdown cannot exceed input tokens")

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "input_tokens": self.input_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "requests": self.requests,
            "successes": self.successes,
            "complete": self.complete,
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
            "kind": "resolved" if self.root.complete and self.subcall.complete else "partial",
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
    root_cache_read_tokens: int = 0
    root_cache_creation_tokens: int = 0
    root_reasoning_tokens: int = 0
    root_output_tokens: int = 0
    root_total_tokens: int = 0
    root_requests: int = 0
    root_successes: int = 0
    root_usage_complete: bool = True
    subcall_input_tokens: int = 0
    subcall_cache_read_tokens: int = 0
    subcall_cache_creation_tokens: int = 0
    subcall_reasoning_tokens: int = 0
    subcall_output_tokens: int = 0
    subcall_total_tokens: int = 0
    subcall_usage_complete: bool = True
    retrieved_ids: list[str] = field(default_factory=list)

    def resolved_usage(self, wall_time_ms: int) -> ResolvedUsage:
        attributed = self.root_total_tokens + self.subcall_total_tokens
        return ResolvedUsage(
            root=UsageBreakdown(
                input_tokens=self.root_input_tokens,
                cache_read_tokens=self.root_cache_read_tokens,
                cache_creation_tokens=self.root_cache_creation_tokens,
                output_tokens=self.root_output_tokens,
                total_tokens=self.root_total_tokens,
                requests=self.root_requests,
                successes=self.root_successes,
                complete=self.root_usage_complete,
            ),
            subcall=UsageBreakdown(
                input_tokens=self.subcall_input_tokens,
                cache_read_tokens=self.subcall_cache_read_tokens,
                cache_creation_tokens=self.subcall_cache_creation_tokens,
                output_tokens=self.subcall_output_tokens,
                total_tokens=self.subcall_total_tokens,
                requests=self.calls_made,
                successes=self.successful_calls,
                complete=self.subcall_usage_complete,
            ),
            unattributed_tokens=self.total_tokens - attributed,
            total_tokens=self.total_tokens,
            wall_time_ms=wall_time_ms,
        )
