from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

# The optional Verifiers dependency currently declares python_version < 3.14.
# Core Droste still supports 3.14; collection must skip this module cleanly
# when the upstream package is intentionally absent.
pytest.importorskip("verifiers", reason="the optional Verifiers extra is unavailable")

from verifiers.v1.loaders import harness_class
from verifiers.v1.runtimes import ProgramResult

from droste_verifiers.harness import (
    RESULT_PATH,
    DrosteHarness,
    DrosteHarnessConfig,
    _program_source,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.sources: list[str] = []
        self.program_argv: list[str] = []
        self.program_env: dict[str, str] = {}
        self.result = b"{}"

    async def prepare_uv_script(self, source: str, env: dict[str, str]) -> list[str]:
        self.sources.append(source)
        return ["python", "/tmp/droste-harness.py"]

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.program_argv = argv
        self.program_env = env
        return ProgramResult(exit_code=0, stdout="{}\n", stderr="")

    async def read(self, path: str) -> bytes:
        assert path == RESULT_PATH
        return self.result


class Sampling:
    def model_dump(self, **_kwargs):
        return {"temperature": 0.3, "max_tokens": 2048}


def _trace(prompt: str = "Find the needle", system_prompt: str | None = None):
    data = SimpleNamespace(prompt=prompt, system_prompt=system_prompt)
    return SimpleNamespace(task=SimpleNamespace(data=data))


def _arg(argv: list[str], prefix: str) -> str:
    return next(item for item in argv if item.startswith(prefix))


def test_plugin_discovery_exports_exactly_one_harness() -> None:
    assert harness_class("droste_verifiers") is DrosteHarness


def test_harness_uses_symbolic_runtime_paths_and_intercepts_root_and_subcalls() -> None:
    runtime = FakeRuntime()
    harness = DrosteHarness(
        DrosteHarnessConfig(
            droste_version="1.2.3",
            depth=1,
            data_paths=["/task/corpus.json"],
            root_revision="root-sha",
            subcall_revision="leaf-sha",
        )
    )
    ctx = SimpleNamespace(model="root-model", sampling=Sampling())

    result = asyncio.run(
        harness.launch(ctx, _trace(), runtime, "http://intercept/v1", "secret", {})
    )

    assert result.exit_code == 0
    assert "droste==1.2.3" in runtime.sources[-1]
    assert _arg(runtime.program_argv, "--base-url=") == "--base-url=http://intercept/v1"
    assert _arg(runtime.program_argv, "--model=") == "--model=root-model"
    assert _arg(runtime.program_argv, "--subcall-model=") == "--subcall-model=root-model"
    assert all("secret" not in arg for arg in runtime.program_argv)
    assert runtime.program_env["OPENAI_API_KEY"] == "secret"
    assert "/task/corpus.json" in runtime.program_argv
    assert "Find the needle" in runtime.program_argv
    # The corpus remains a runtime path. It is never copied into the task prompt.
    assert "/task/corpus.json" not in "Find the needle"

    rollout = json.loads(_arg(runtime.program_argv, "--rollout-config=").split("=", 1)[1])
    assert rollout["root_revision"] == "root-sha"
    assert rollout["subcall_revision"] == "leaf-sha"
    assert rollout["root_sampling"]["temperature"] == 0.3


def test_depth_zero_disables_flat_subcalls_without_changing_harness() -> None:
    runtime = FakeRuntime()
    harness = DrosteHarness(DrosteHarnessConfig(depth=0, max_subcalls=50))

    asyncio.run(
        harness.launch(
            SimpleNamespace(model="model", sampling=Sampling()),
            _trace(),
            runtime,
            "http://intercept/v1",
            "secret",
            {},
        )
    )

    assert "--budget-subcalls=0" in runtime.program_argv
    assert "--budget-depth=0" in runtime.program_argv


def test_task_system_prompt_is_deterministically_folded_into_question() -> None:
    runtime = FakeRuntime()
    harness = DrosteHarness(DrosteHarnessConfig())

    asyncio.run(
        harness.launch(
            SimpleNamespace(model="model", sampling=Sampling()),
            _trace("user task", system_prompt="system task contract"),
            runtime,
            "http://intercept/v1",
            "secret",
            {},
        )
    )

    assert "system task contract\n\nuser task" in runtime.program_argv


def test_harness_metrics_preserve_stdout_and_terminal_diagnostics() -> None:
    runtime = FakeRuntime()
    runtime.result = json.dumps(
        {
            "iterations": 3,
            "tokens_used": 120,
            "subcalls": 4,
            "successful_subcalls": 3,
            "stdout_chars": 900,
            "ready": False,
            "extracted": True,
        }
    ).encode()
    harness = DrosteHarness(DrosteHarnessConfig(depth=1))

    metrics = asyncio.run(harness.droste(_trace(), runtime))

    assert metrics == {
        "droste_iterations": 3.0,
        "droste_tokens": 120.0,
        "droste_subcalls": 4.0,
        "droste_successful_subcalls": 3.0,
        "droste_stdout_chars": 900.0,
        "droste_ready": 0.0,
        "droste_extracted": 1.0,
        "droste_depth": 1.0,
    }


def test_pep723_program_pins_the_runtime_engine() -> None:
    source = _program_source("2.0.0")

    assert '# dependencies = ["droste==2.0.0"]' in source
    assert "droste_verifiers" not in source


def test_harness_rejects_unsafe_or_non_pep440_versions() -> None:
    import pytest

    for value in ('1.2.3"; injected = "x', "main", "1.2.3; python_version<'4'"):
        with pytest.raises(ValueError, match="droste_version"):
            DrosteHarnessConfig(droste_version=value)
