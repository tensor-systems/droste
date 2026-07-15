from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..capabilities import CapabilityAttemptEvent, CapabilityResult

from ..protocols.llm_client import TokenUsage
from .budget import Budget, BudgetLedger
from .config import ExecutionConfig, SandboxLimits
from .progress import EVENT_TYPES, EventCallback, ProgressCallback, progress_event
from .stats import ExecutionStats
from .trace import (
    Clock,
    DataUseAuthorization,
    MonotonicClock,
    RunRecord,
    RunRecordCallback,
    TraceRecorder,
    TraceRetentionPolicy,
)


@dataclass
class ExecutionContext:
    """Context for tracking recursive LLM calls within sandbox execution."""

    config: ExecutionConfig = field(default_factory=ExecutionConfig)
    stats: ExecutionStats = field(default_factory=ExecutionStats)
    trace: TraceRecorder = field(default_factory=TraceRecorder)
    ledger: BudgetLedger = field(default_factory=lambda: BudgetLedger(Budget()))
    _emission_lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _iteration: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.ledger.budget != self.config.budget:
            raise ValueError("ExecutionContext config and ledger budgets must match")

    def emit_progress(self, status: str) -> None:
        """Deliver a progress line to the attached sink; silent when none.

        No default emitter (#35): a bare engine call performs no I/O. Entry
        points attach sinks explicitly (droste_runner and the relay: the
        stderr NDJSON sinks; the CLI: its --verbose echo)."""
        if self.config.on_progress is not None:
            self.config.on_progress(status)
        self.emit_event(progress_event(status))

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
        with self._emission_lock:
            value = self.trace.append(event)
            if self.config.on_event is not None:
                self.config.on_event(value.as_dict())

    def finish_trace(self, terminal: dict[str, Any]) -> RunRecord:
        """Emit the terminal value and return one policy-resolved run record."""
        with self._emission_lock:
            prior_count = len(self.trace.events)
            record = self.trace.finish(terminal)
            if len(self.trace.events) > prior_count:
                if self.config.on_event is not None:
                    self.config.on_event(self.trace.events[-1].as_dict())
                if self.config.on_run_record is not None:
                    self.config.on_run_record(record)
            return record

    def record_root_attempt(self) -> None:
        self.stats.root_requests += 1

    def record_root_success(self, usage: TokenUsage) -> None:
        self._validate_usage(usage)
        if self.stats.root_successes >= self.stats.root_requests:
            raise ValueError("root successes cannot exceed requests")
        self.stats.root_successes += 1
        self.stats.root_input_tokens += usage.prompt_tokens
        self.stats.root_output_tokens += usage.completion_tokens
        self.stats.root_total_tokens += usage.total_tokens
        self.stats.total_tokens += usage.total_tokens

    def record_subcall_attempts(self, count: int = 1) -> None:
        self._validate_count(count)
        self.stats.calls_made += count

    def record_subcall_successes(self, count: int = 1) -> None:
        self._validate_count(count)
        if self.stats.successful_calls + count > self.stats.calls_made:
            raise ValueError("subcall successes cannot exceed attempts")
        self.stats.successful_calls += count

    def record_subcall_usage(self, usage: TokenUsage) -> None:
        self._validate_usage(usage)
        self.stats.subcall_input_tokens += usage.prompt_tokens
        self.stats.subcall_output_tokens += usage.completion_tokens
        self.stats.subcall_total_tokens += usage.total_tokens
        self.stats.total_tokens += usage.total_tokens

    @staticmethod
    def _validate_count(count: int) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("accounting counts must be non-negative integers")

    @staticmethod
    def _validate_usage(usage: TokenUsage) -> None:
        values = (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError("token usage must contain non-negative integers")

    def observe_capability(self, result: CapabilityResult) -> None:
        """Append the broker-owned content-free capability projection."""
        self.emit_event({"type": "capability", "outcome": result.to_trace_dict()})

    def begin_iteration(self, iteration: int) -> None:
        """Publish the current loop iteration before model-authored code may run."""

        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
            raise ValueError("execution iteration must be positive")
        self._iteration = iteration

    def observe_capability_attempt(self, event: CapabilityAttemptEvent) -> None:
        """Project one broker attempt fact into the canonical subcall event."""

        from ..capabilities import CapabilityAttemptPhase, CapabilityKind, thaw_value

        capability_id = event.call.capability_id
        if (
            capability_id.kind is not CapabilityKind.INFERENCE
            or capability_id.provider_type != "subcall"
            or self._iteration == 0
        ):
            return
        value: dict[str, Any] = {
            "type": "subcall",
            "phase": event.phase.value,
            "call_id": event.call.call_id,
            "operation": capability_id.operation,
            "iteration": self._iteration,
        }
        if event.reservation is not None:
            value["reservation"] = event.reservation.to_dict()
        if event.checkpoint is not None:
            value["checkpoint"] = event.checkpoint.to_dict()
        if event.error is not None:
            value["error"] = {"code": event.error.code, "type": event.error.type}
        if capability_id.operation in {"llm_batch", "llm_batch_with_errors"}:
            value["batch_id"] = event.call.call_id
            raw_prompts = thaw_value(
                event.call.args[0]
                if event.call.args
                else event.call.kwargs.get("prompts", ())
            )
            if isinstance(raw_prompts, (list, tuple)) and raw_prompts:
                value["batch_count"] = len(raw_prompts)
        self.emit_event(value)

    @property
    def depth(self) -> int:
        return self.stats.depth

    @depth.setter
    def depth(self, value: int) -> None:
        self.stats.depth = value

    @property
    def budget(self) -> Budget:
        return self.config.budget

    @property
    def sandbox(self) -> SandboxLimits:
        return self.config.sandbox

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
    budget: Budget | None = None,
    sandbox: SandboxLimits | None = None,
    verbose: bool = False,
    on_progress: ProgressCallback | None = None,
    on_event: EventCallback | None = None,
    on_run_record: RunRecordCallback | None = None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    trace_depth: int = 0,
    trace_retention: TraceRetentionPolicy | None = None,
    data_use: DataUseAuthorization | None = None,
    trace_clock: Clock | None = None,
    trace_monotonic_clock: MonotonicClock | None = None,
) -> ExecutionContext:
    resolved_budget = budget or Budget()
    resolved_sandbox = sandbox or SandboxLimits()
    config = ExecutionConfig(
        budget=resolved_budget,
        sandbox=resolved_sandbox,
        verbose=verbose,
        on_progress=on_progress,
        on_event=on_event,
        on_run_record=on_run_record,
    )
    trace_kwargs: dict[str, Any] = {
        "parent_run_id": parent_run_id,
        "depth": trace_depth,
        "retention": trace_retention or TraceRetentionPolicy(),
        "data_use": data_use or DataUseAuthorization(),
    }
    if run_id is not None:
        trace_kwargs["run_id"] = run_id
    if trace_clock is not None:
        trace_kwargs["clock"] = trace_clock
    if trace_monotonic_clock is not None:
        trace_kwargs["monotonic_clock"] = trace_monotonic_clock
    context = ExecutionContext(
        config=config,
        stats=ExecutionStats(),
        trace=TraceRecorder(**trace_kwargs),
        ledger=BudgetLedger(resolved_budget),
    )
    context.ledger.on_event = context.emit_event
    return context
