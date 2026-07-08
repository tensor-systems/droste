from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import (
    DEFAULT_MAX_CALLS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_OUTPUT_CHARS,
    ExecutionConfig,
)
from .progress import EventCallback, ProgressCallback, emit_event, emit_progress
from .stats import ExecutionStats


@dataclass
class ExecutionContext:
    """Context for tracking recursive LLM calls within sandbox execution."""

    config: ExecutionConfig = field(default_factory=ExecutionConfig)
    stats: ExecutionStats = field(default_factory=ExecutionStats)

    def emit_progress(self, status: str) -> None:
        """Emit progress via callback if provided, otherwise use default emitter."""
        if self.config.on_progress is not None:
            self.config.on_progress(status)
        else:
            emit_progress(status)

    def emit_event(self, event: dict[str, Any]) -> None:
        """Emit a structured loop event (#2) via callback if provided, else stderr."""
        if self.config.on_event is not None:
            self.config.on_event(event)
        else:
            emit_event(event)

    @property
    def depth(self) -> int:
        return self.stats.depth

    @depth.setter
    def depth(self, value: int) -> None:
        self.stats.depth = value

    @property
    def max_depth(self) -> int:
        return self.config.max_depth

    @property
    def max_calls(self) -> int:
        return self.config.max_calls

    @property
    def max_iterations(self) -> int:
        return self.config.max_iterations

    @property
    def max_output_chars(self) -> int:
        return self.config.max_output_chars

    @property
    def calls_made(self) -> int:
        return self.stats.calls_made

    @calls_made.setter
    def calls_made(self, value: int) -> None:
        self.stats.calls_made = value

    @property
    def total_tokens(self) -> int:
        return self.stats.total_tokens

    @total_tokens.setter
    def total_tokens(self, value: int) -> None:
        self.stats.total_tokens = value

    @property
    def verbose(self) -> bool:
        return self.config.verbose

    @property
    def on_progress(self) -> ProgressCallback | None:
        return self.config.on_progress


def create_execution_context(
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_calls: int = DEFAULT_MAX_CALLS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    verbose: bool = False,
    on_progress: ProgressCallback | None = None,
    on_event: EventCallback | None = None,
) -> ExecutionContext:
    config = ExecutionConfig(
        max_depth=max_depth,
        max_calls=max_calls,
        max_iterations=max_iterations,
        max_output_chars=max_output_chars,
        verbose=verbose,
        on_progress=on_progress,
        on_event=on_event,
    )
    return ExecutionContext(config=config, stats=ExecutionStats())
