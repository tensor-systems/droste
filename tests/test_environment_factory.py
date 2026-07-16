"""Substrate factory boundaries and safety declarations (#12)."""

from __future__ import annotations

import signal
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from droste import RLMConfig, run_rlm
from droste.environments import (
    EnvironmentConfig,
    PyodideEnvironment,
    RunnerEnvironment,
    create_environment,
    create_environment_context,
    select_environment,
)
from droste.execution import Budget, SandboxLimits
from droste.protocols.llm_client import TokenUsage
from droste.testing import (
    MockLLMClient,
    MockResponse,
    MockSubcallClient,
    require_ordered_terminal_events,
)


def _environment(config: EnvironmentConfig):
    return create_environment(
        config,
        context={"files": []},
        registry=None,
        subcalls=MockSubcallClient(),
        execution_context=create_environment_context(config),
    )


def test_selection_is_pure_and_rejects_unknown_kinds() -> None:
    assert select_environment("native") is RunnerEnvironment
    assert select_environment("pyodide") is PyodideEnvironment
    with pytest.raises(ValueError, match="unsupported environment kind"):
        select_environment("browser")
    with pytest.raises(ValueError, match="unsupported environment kind"):
        EnvironmentConfig(kind="browser")  # type: ignore[arg-type]


def test_environment_config_is_immutable_and_drives_execution_context() -> None:
    config = EnvironmentConfig(
        kind="native",
        budget=Budget(
            tokens=2_000,
            subcalls=4,
            depth=3,
            wall_ms=8_000,
            root_output_tokens=512,
            subcall_output_tokens=256,
        ),
        sandbox=SandboxLimits(output_chars=6, execution_timeout_ms=7),
    )
    context = create_environment_context(config)

    assert context.budget == config.budget
    assert context.sandbox == config.sandbox
    with pytest.raises(FrozenInstanceError):
        config.budget = Budget()  # type: ignore[misc]


def test_native_config_can_preserve_distinct_loop_and_executor_output_caps() -> None:
    config = EnvironmentConfig(
        kind="native",
        sandbox=SandboxLimits(output_chars=25_000, capture_output_chars=100_000),
    )

    context = create_environment_context(config)
    environment = _environment(config)

    assert context.sandbox.output_chars == 25_000
    assert environment.capabilities()["max_output_chars"] == 100_000


def test_native_config_rejects_an_executor_cap_below_the_loop_cap() -> None:
    with pytest.raises(ValueError, match="at least output_chars"):
        EnvironmentConfig(
            kind="native",
            sandbox=SandboxLimits(output_chars=25_000, capture_output_chars=10_000),
        )


def test_native_environment_owns_signal_timeout_wiring(monkeypatch) -> None:
    timer_calls: list[tuple[int, float]] = []
    signal_calls: list[tuple[int, object]] = []
    previous_handler = object()

    def fake_signal(signum: int, handler: object) -> object:
        signal_calls.append((signum, handler))
        return previous_handler

    monkeypatch.setattr("droste.environments.inprocess.signal.signal", fake_signal)
    monkeypatch.setattr(
        "droste.environments.inprocess.signal.setitimer",
        lambda which, seconds: timer_calls.append((which, seconds)),
    )
    environment = _environment(
        EnvironmentConfig(
            kind="native",
            sandbox=SandboxLimits(execution_timeout_ms=25),
        )
    )

    result = environment.execute("print('ok')")

    assert isinstance(environment, RunnerEnvironment)
    assert result.stdout == "ok\n"
    assert timer_calls == [(signal.ITIMER_REAL, 0.025), (signal.ITIMER_REAL, 0)]
    assert signal_calls[-1] == (signal.SIGALRM, previous_handler)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({}, "host_managed_timeout=True"),
        ({"host_managed_timeout": True}, "host_managed_isolation=True"),
        (
            {
                "host_managed_timeout": True,
                "host_managed_isolation": True,
                "sandbox": SandboxLimits(execution_timeout_ms=1),
            },
            "cannot enforce execution_timeout_ms",
        ),
    ],
)
def test_pyodide_requires_explicit_host_timeout_and_isolation(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        EnvironmentConfig(kind="pyodide", **overrides)  # type: ignore[arg-type]


def test_pyodide_environment_uses_shared_raw_namespace_without_signal_timers(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "droste.environments.inprocess.signal.setitimer",
        lambda *args: pytest.fail("Pyodide must not use native signal timers"),
    )
    config = EnvironmentConfig(
        kind="pyodide",
        host_managed_timeout=True,
        host_managed_isolation=True,
    )
    environment = _environment(config)

    result = environment.execute(
        "import sys\n"
        "answer = {'content': 'done', 'ready': True}\n"
        "print('unbounded here')\n"
        "print('warning', file=sys.stderr)"
    )

    assert isinstance(environment, PyodideEnvironment)
    assert result.stdout == "unbounded here\n"
    assert result.stderr == "warning\n"
    assert environment.globals()["answer"] == {"content": "done", "ready": True}
    assert result.timed_out is False


def test_native_and_pyodide_deliver_the_same_ordered_terminal_event_lifecycle() -> None:
    event_orders: list[tuple[str, ...]] = []
    for config in (
        EnvironmentConfig(kind="native"),
        EnvironmentConfig(
            kind="pyodide",
            host_managed_timeout=True,
            host_managed_isolation=True,
        ),
    ):
        events: list[dict[str, object]] = []
        run_id = f"lifecycle-{config.kind}"
        context = create_environment_context(config, on_event=events.append, run_id=run_id)
        subcalls = MockSubcallClient()
        environment = create_environment(
            config,
            context={},
            registry=None,
            subcalls=subcalls,
            execution_context=context,
        )
        result = run_rlm(
            "question",
            environment=environment,
            root_llm=MockLLMClient(
                [
                    MockResponse(
                        "```python\nanswer['content'] = 'done'\nanswer['ready'] = True\n```",
                        TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                    )
                ]
            ),
            subcalls=subcalls,
            config=RLMConfig(run_id=run_id),
            context=context,
        )
        assert result.answer == "done"
        event_orders.append(require_ordered_terminal_events(events))

    assert event_orders[0] == event_orders[1]


def test_native_rejects_pyodide_only_safety_declarations() -> None:
    with pytest.raises(ValueError, match="only valid for pyodide"):
        EnvironmentConfig(kind="native", host_managed_timeout=True)


def test_pyodide_rejects_a_distinct_executor_output_cap() -> None:
    with pytest.raises(ValueError, match="one loop-owned output limit"):
        EnvironmentConfig(
            kind="pyodide",
            sandbox=SandboxLimits(output_chars=25_000, capture_output_chars=50_000),
            host_managed_timeout=True,
            host_managed_isolation=True,
        )


def test_host_entrypoints_use_factory_instead_of_copying_environment_wiring() -> None:
    root = Path(__file__).parents[1]
    host_paths = (
        root / "src/droste_cli/main.py",
        root / "src/droste_runner/run.py",
        root / "benchmarks/live.py",
        root / "examples/pyodide-host/pyodide_host_adapter.py",
    )
    for path in host_paths:
        source = path.read_text(encoding="utf-8")
        assert "RunnerEnvironment(" not in source, path
        assert "create_execution_context(" not in source, path

    for path in host_paths:
        source = path.read_text(encoding="utf-8")
        assert "execution_context=" in source, path

    pyodide_adapter = host_paths[-1].read_text(encoding="utf-8")
    assert 'kind="pyodide"' in pyodide_adapter
    assert "host_managed_timeout=True" in pyodide_adapter
    assert "host_managed_isolation=True" in pyodide_adapter
