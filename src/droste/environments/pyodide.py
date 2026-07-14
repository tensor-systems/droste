"""Pyodide/WASM environment backed by the host-isolated raw executor."""

from __future__ import annotations

from typing import Any

from ..capabilities import CapabilityAnnotator, CapabilityGuard, CapabilityObserver
from ..protocols.environment import ExecutionResult
from ..protocols.subcall_client import SubcallClient
from ..registry import DataSourceRegistry
from ..substrates.pyodide import RawExecutor
from .inprocess import RunnerEnvironment


class PyodideEnvironment(RunnerEnvironment):
    """RLM environment for a Pyodide interpreter isolated by its host.

    Pyodide cannot enforce native signal timers. The factory requires the host
    to acknowledge both the external isolation boundary and wall-clock timeout
    before it will construct this environment.
    """

    def __init__(
        self,
        *,
        context: Any,
        registry: DataSourceRegistry | None,
        subcalls: SubcallClient,
        max_output_chars: int,
        exec_timeout_ms: int = 0,
        capability_run_id: str | None = None,
        capability_parent_run_id: str | None = None,
        capability_guard: CapabilityGuard | None = None,
        capability_annotator: CapabilityAnnotator | None = None,
        capability_observer: CapabilityObserver | None = None,
    ) -> None:
        if exec_timeout_ms != 0:
            raise ValueError("PyodideEnvironment cannot enforce exec_timeout_ms")
        super().__init__(
            context=context,
            registry=registry,
            subcalls=subcalls,
            max_output_chars=max_output_chars,
            exec_timeout_ms=0,
            capability_run_id=capability_run_id,
            capability_parent_run_id=capability_parent_run_id,
            capability_guard=capability_guard,
            capability_annotator=capability_annotator,
            capability_observer=capability_observer,
        )
        self._executor = RawExecutor(
            db=None,
            max_output_chars=max_output_chars,
            namespace=self._globals,
        )

    def execute(self, code: str) -> ExecutionResult:
        output = self._executor.execute_with_output(code)
        return ExecutionResult(
            stdout=output.stdout,
            stderr=output.stderr,
            timed_out=False,
            exit_code=0,
            files_written=[],
        )

    def close(self) -> None:
        self._executor.close()
