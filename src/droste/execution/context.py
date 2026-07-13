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
from .progress import EVENT_TYPES, EventCallback, ProgressCallback
from .stats import ExecutionStats


@dataclass
class ExecutionContext:
    """Context for tracking recursive LLM calls within sandbox execution."""

    config: ExecutionConfig = field(default_factory=ExecutionConfig)
    stats: ExecutionStats = field(default_factory=ExecutionStats)

    def emit_progress(self, status: str) -> None:
        """Deliver a progress line to the attached sink; silent when none.

        No default emitter (#35): a bare engine call performs no I/O. Entry
        points attach sinks explicitly (droste_runner and the relay: the
        stderr NDJSON sinks; the CLI: its --verbose echo)."""
        if self.config.on_progress is not None:
            self.config.on_progress(status)

    def emit_event(self, event: dict[str, Any]) -> None:
        """Deliver a structured loop event (#1) to the attached sink.

        The type is validated against the shared vocabulary even when no
        sink is attached, so an off-vocabulary emitter fails the first test
        that exercises it instead of shipping an event the relay's
        forwarding filter silently drops (#35)."""
        event_type = event.get("type")
        if event_type not in EVENT_TYPES:
            raise ValueError(
                f"unknown RLM event type {event_type!r}: add it to "
                "droste.execution.progress.EVENT_TYPES and the relay's events.ts "
                "vocabulary (kept in lockstep by tests/test_event_vocabulary.py)"
            )
        if self.config.on_event is not None:
            self.config.on_event(event)

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
    def successful_calls(self) -> int:
        return self.stats.successful_calls

    @successful_calls.setter
    def successful_calls(self, value: int) -> None:
        self.stats.successful_calls = value

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
