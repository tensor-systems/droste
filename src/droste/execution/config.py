from __future__ import annotations

from dataclasses import dataclass

from .budget import Budget
from .progress import EventCallback, ProgressCallback
from .trace import RunRecordCallback

DEFAULT_OUTPUT_CHARS = 25_000
DEFAULT_SUBCALL_CONCURRENCY = 5


def validate_subcall_concurrency(value: object) -> int:
    """Return one valid, positive subcall batch-concurrency value."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("subcall concurrency must be an integer")
    if value < 1:
        raise ValueError("subcall concurrency must be positive")
    return value


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """Local REPL guardrails, deliberately separate from compute spend."""

    output_chars: int = DEFAULT_OUTPUT_CHARS
    execution_timeout_ms: int = 0
    capture_output_chars: int | None = None

    def __post_init__(self) -> None:
        values = (self.output_chars, self.execution_timeout_ms)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError("sandbox limits must be non-negative integers")
        if self.capture_output_chars is not None and (
            isinstance(self.capture_output_chars, bool)
            or not isinstance(self.capture_output_chars, int)
            or self.capture_output_chars < self.output_chars
        ):
            raise ValueError("capture_output_chars must be at least output_chars")

    @property
    def resolved_capture_output_chars(self) -> int:
        return self.output_chars if self.capture_output_chars is None else self.capture_output_chars


@dataclass(frozen=True)
class ExecutionConfig:
    """Immutable configuration for RLM execution."""

    budget: Budget = Budget()
    sandbox: SandboxLimits = SandboxLimits()
    # The core does not read this host-facing hint (#35); verbose views are rendered by shells from
    # structured events (droste.execution.progress.render_verbose).
    verbose: bool = False
    on_progress: ProgressCallback | None = None
    # Structured loop events (#1). None means NO emission (#35): entry points
    # attach a sink explicitly (droste_runner/relay: the stderr NDJSON sink
    # droste.execution.progress.emit_event). Independent of on_progress.
    on_event: EventCallback | None = None
    # Optional host I/O shell. Droste supplies the policy-resolved immutable
    # value; the callback decides whether and where to persist it.
    on_run_record: RunRecordCallback | None = None
