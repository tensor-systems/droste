"""Substrate factory boundaries and safety declarations (#12)."""

from __future__ import annotations

import signal
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from droste.environments import (
    EnvironmentConfig,
    PyodideEnvironment,
    RunnerEnvironment,
    create_environment,
    create_environment_context,
    select_environment,
)
from droste.testing import MockSubcallClient


def _environment(config: EnvironmentConfig):
    return create_environment(
        config,
        context={"files": []},
        registry=None,
        subcalls=MockSubcallClient(),
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
        max_depth=3,
        max_calls=4,
        max_iterations=5,
        max_output_chars=6,
        exec_timeout_ms=7,
    )
    context = create_environment_context(config)

    assert context.max_depth == 3
    assert context.max_calls == 4
    assert context.max_iterations == 5
    assert context.max_output_chars == 6
    with pytest.raises(FrozenInstanceError):
        config.max_calls = 99  # type: ignore[misc]


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
    environment = _environment(EnvironmentConfig(kind="native", exec_timeout_ms=25))

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
                "exec_timeout_ms": 1,
            },
            "cannot enforce exec_timeout_ms",
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
        max_output_chars=4,
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


def test_native_rejects_pyodide_only_safety_declarations() -> None:
    with pytest.raises(ValueError, match="only valid for pyodide"):
        EnvironmentConfig(kind="native", host_managed_timeout=True)


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

    for path in host_paths[:3]:
        source = path.read_text(encoding="utf-8")
        assert "max_output_chars=environment_config.max_output_chars" in source, path

    pyodide_adapter = host_paths[-1].read_text(encoding="utf-8")
    assert 'kind="pyodide"' in pyodide_adapter
    assert "host_managed_timeout=True" in pyodide_adapter
    assert "host_managed_isolation=True" in pyodide_adapter
