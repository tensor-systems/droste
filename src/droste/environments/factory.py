"""Pure substrate selection and thin host-wiring constructors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..capabilities import CapabilityAnnotator, CapabilityGuard, CapabilityObserver
from ..execution.broker_budget import BrokerBudget
from ..execution.budget import Budget
from ..execution.config import SandboxLimits
from ..execution.context import ExecutionContext, create_execution_context
from ..execution.progress import EventCallback, ProgressCallback
from ..execution.trace import (
    DataUseAuthorization,
    RunRecordCallback,
    TraceRetentionPolicy,
)
from ..protocols.environment import RLMEnvironment
from ..protocols.subcall_client import SubcallClient
from ..providers import ProviderRegistry
from .inprocess import RunnerEnvironment
from .pyodide import PyodideEnvironment

EnvironmentKind = Literal["native", "pyodide"]
EnvironmentType = type[RunnerEnvironment]


def select_environment(kind: str) -> EnvironmentType:
    """Purely select the implementation for a supported substrate name."""
    if kind == "native":
        return RunnerEnvironment
    if kind == "pyodide":
        return PyodideEnvironment
    raise ValueError(f"unsupported environment kind: {kind!r}")


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Immutable host-owned budgets and substrate safety declarations."""

    kind: EnvironmentKind
    budget: Budget = Budget()
    sandbox: SandboxLimits = SandboxLimits()
    host_managed_timeout: bool = False
    host_managed_isolation: bool = False

    def __post_init__(self) -> None:
        select_environment(self.kind)
        if not isinstance(self.budget, Budget):
            raise TypeError("EnvironmentConfig budget must be a Budget")
        if not isinstance(self.sandbox, SandboxLimits):
            raise TypeError("EnvironmentConfig sandbox must be SandboxLimits")
        if self.kind == "pyodide":
            if self.sandbox.execution_timeout_ms != 0:
                raise ValueError(
                    "pyodide cannot enforce execution_timeout_ms; set it to 0 and "
                    "enforce a wall-clock timeout in the host"
                )
            if not self.host_managed_timeout:
                raise ValueError("pyodide requires host_managed_timeout=True")
            if not self.host_managed_isolation:
                raise ValueError("pyodide requires host_managed_isolation=True")
            if self.sandbox.capture_output_chars not in (None, self.sandbox.output_chars):
                raise ValueError(
                    "pyodide has one loop-owned output limit; "
                    "capture_output_chars must match output_chars"
                )
        elif self.host_managed_timeout or self.host_managed_isolation:
            raise ValueError("host-managed safety declarations are only valid for pyodide")


def create_environment_context(
    config: EnvironmentConfig,
    *,
    verbose: bool = False,
    on_progress: ProgressCallback | None = None,
    on_event: EventCallback | None = None,
    on_run_record: RunRecordCallback | None = None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    trace_depth: int = 0,
    trace_retention: TraceRetentionPolicy | None = None,
    data_use: DataUseAuthorization | None = None,
) -> ExecutionContext:
    """Create the loop context from the same immutable budgets as the environment."""
    return create_execution_context(
        budget=config.budget,
        sandbox=config.sandbox,
        verbose=verbose,
        on_progress=on_progress,
        on_event=on_event,
        on_run_record=on_run_record,
        run_id=run_id,
        parent_run_id=parent_run_id,
        trace_depth=trace_depth,
        trace_retention=trace_retention,
        data_use=data_use,
    )


def create_environment(
    config: EnvironmentConfig,
    *,
    context: Any,
    registry: ProviderRegistry | None,
    subcalls: SubcallClient,
    execution_context: ExecutionContext,
    capability_run_id: str | None = None,
    capability_parent_run_id: str | None = None,
    capability_guard: CapabilityGuard | None = None,
    capability_annotator: CapabilityAnnotator | None = None,
    capability_observer: CapabilityObserver | None = None,
) -> RLMEnvironment:
    """Construct an environment and take ownership of ``registry`` immediately.

    A successful environment closes the registry with the run. Construction
    failures close it before propagating, so callers must not retain a second
    owner after passing it here.
    """

    try:
        environment_type = select_environment(config.kind)
        if execution_context.budget != config.budget or execution_context.sandbox != config.sandbox:
            raise ValueError("environment config must match its execution context")
        accounting = BrokerBudget(
            execution_context.ledger,
            on_inference_settlement=execution_context.record_subcall_settlement,
        )
        return environment_type(
            context=context,
            registry=registry,
            subcalls=subcalls,
            max_output_chars=config.sandbox.resolved_capture_output_chars,
            exec_timeout_ms=config.sandbox.execution_timeout_ms,
            budget_ledger=execution_context.ledger,
            capability_run_id=capability_run_id,
            capability_parent_run_id=capability_parent_run_id,
            capability_guard=capability_guard,
            capability_annotator=capability_annotator,
            capability_observer=capability_observer,
            capability_attempt_observer=execution_context.observe_capability_attempt,
            capability_attempt_authority=accounting,
            subcall_usage_callback=execution_context.record_subcall_usage,
        )
    except BaseException as exc:
        if registry is None:
            raise
        try:
            registry.close()
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "environment construction and provider cleanup failed",
                [exc, cleanup_error],
            ) from None
        raise
