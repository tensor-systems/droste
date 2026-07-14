"""Pure substrate selection and thin host-wiring constructors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..execution.config import (
    DEFAULT_MAX_CALLS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_OUTPUT_CHARS,
)
from ..execution.context import ExecutionContext, create_execution_context
from ..execution.progress import EventCallback, ProgressCallback
from ..protocols.environment import RLMEnvironment
from ..protocols.subcall_client import SubcallClient
from ..registry import DataSourceRegistry
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
    max_depth: int = DEFAULT_MAX_DEPTH
    max_calls: int = DEFAULT_MAX_CALLS
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    # Normally omitted so the executor and loop share one cap. Native hosts
    # with a deliberate two-stage policy may keep a looser capture-buffer cap
    # while the loop owns the lower, repairable output-budget error.
    executor_max_output_chars: int | None = None
    exec_timeout_ms: int = 0
    host_managed_timeout: bool = False
    host_managed_isolation: bool = False

    def __post_init__(self) -> None:
        select_environment(self.kind)
        if self.exec_timeout_ms < 0:
            raise ValueError("exec_timeout_ms must be non-negative")
        if self.executor_max_output_chars is not None and self.executor_max_output_chars < 0:
            raise ValueError("executor_max_output_chars must be non-negative")
        if self.kind == "pyodide":
            if self.exec_timeout_ms != 0:
                raise ValueError(
                    "pyodide cannot enforce exec_timeout_ms; set it to 0 and "
                    "enforce a wall-clock timeout in the host"
                )
            if not self.host_managed_timeout:
                raise ValueError("pyodide requires host_managed_timeout=True")
            if not self.host_managed_isolation:
                raise ValueError("pyodide requires host_managed_isolation=True")
            if self.executor_max_output_chars not in (None, self.max_output_chars):
                raise ValueError(
                    "pyodide has one loop-owned output limit; "
                    "executor_max_output_chars must match max_output_chars"
                )
        elif self.host_managed_timeout or self.host_managed_isolation:
            raise ValueError("host-managed safety declarations are only valid for pyodide")

    @property
    def resolved_executor_max_output_chars(self) -> int:
        """Executor cap, defaulting to the loop's output budget."""
        if self.executor_max_output_chars is None:
            return self.max_output_chars
        return self.executor_max_output_chars


def create_environment_context(
    config: EnvironmentConfig,
    *,
    verbose: bool = False,
    on_progress: ProgressCallback | None = None,
    on_event: EventCallback | None = None,
) -> ExecutionContext:
    """Create the loop context from the same immutable budgets as the environment."""
    return create_execution_context(
        max_depth=config.max_depth,
        max_calls=config.max_calls,
        max_iterations=config.max_iterations,
        max_output_chars=config.max_output_chars,
        verbose=verbose,
        on_progress=on_progress,
        on_event=on_event,
    )


def create_environment(
    config: EnvironmentConfig,
    *,
    context: Any,
    registry: DataSourceRegistry | None,
    subcalls: SubcallClient,
) -> RLMEnvironment:
    """Construct the selected environment around host-supplied live dependencies."""
    environment_type = select_environment(config.kind)
    return environment_type(
        context=context,
        registry=registry,
        subcalls=subcalls,
        max_output_chars=config.resolved_executor_max_output_chars,
        exec_timeout_ms=config.exec_timeout_ms,
    )
