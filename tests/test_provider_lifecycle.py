"""Explicit ownership and deterministic provider-runtime cleanup."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import replace
from threading import Thread

import pytest

from droste import (
    Budget,
    ConfiguredSource,
    EnvironmentConfig,
    ProviderCatalog,
    ProviderRuntime,
    RLMConfig,
    create_environment,
    create_environment_context,
    run_rlm,
)
from droste.protocols.llm_client import TokenUsage
from droste.sources.bridge import ProviderService
from droste.sources.filesystem_text import filesystem_text_provider
from droste.sources.sql_local import sqlite_provider
from droste.testing import (
    MockEnvironment,
    MockLLMClient,
    MockResponse,
    MockSubcallClient,
    fake_records_provider,
)


def _registration(binder):
    return replace(fake_records_provider(), binder=binder)


def _runtime(source_id: str, closed: list[str], *, fail_close: bool = False):
    handlers = (
        fake_records_provider().binder(ConfiguredSource(source_id, "fake_records"), None).handlers
    )

    def close() -> None:
        closed.append(source_id)
        if fail_close:
            raise RuntimeError(f"close failed: {source_id}")

    return ProviderRuntime(handlers, close_callback=close)


def test_resource_free_runtime_needs_no_close_boilerplate() -> None:
    runtime = ProviderRuntime(
        fake_records_provider().binder(ConfiguredSource("records", "fake_records"), None).handlers
    )

    runtime.close()
    runtime.close()


def test_registry_closes_multiple_sources_once_in_reverse_bind_order_under_concurrency() -> None:
    closed: list[str] = []
    registration = _registration(lambda source, context=None: _runtime(source.source_id, closed))
    registry = ProviderCatalog((registration,)).bind(
        (
            ConfiguredSource("first", "fake_records"),
            ConfiguredSource("second", "fake_records"),
        )
    )

    threads = [Thread(target=registry.close) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    registry.sources[0].close()

    assert closed == ["second", "first"]


def test_registry_attempts_every_close_and_does_not_retry_failures() -> None:
    closed: list[str] = []
    registration = _registration(
        lambda source, context=None: _runtime(source.source_id, closed, fail_close=True)
    )
    registry = ProviderCatalog((registration,)).bind(
        (
            ConfiguredSource("first", "fake_records"),
            ConfiguredSource("second", "fake_records"),
        )
    )

    with pytest.raises(ExceptionGroup, match="provider runtime cleanup failed") as raised:
        registry.close()
    with pytest.raises(ExceptionGroup, match="provider runtime cleanup failed"):
        registry.close()

    assert closed == ["second", "first"]
    assert len(raised.value.exceptions) == 2


def test_partial_bind_failure_closes_every_acquired_runtime() -> None:
    closed: list[str] = []

    def bind(source, context=None):
        del context
        if source.source_id == "third":
            raise RuntimeError("third bind failed")
        return _runtime(source.source_id, closed)

    with pytest.raises(RuntimeError, match="third bind failed"):
        ProviderCatalog((_registration(bind),)).bind(
            (
                ConfiguredSource("first", "fake_records"),
                ConfiguredSource("second", "fake_records"),
                ConfiguredSource("third", "fake_records"),
            )
        )

    assert closed == ["second", "first"]


def test_invalid_runtime_is_closed_before_prior_bindings() -> None:
    closed: list[str] = []

    def bind(source, context=None):
        del context
        if source.source_id == "invalid":
            return ProviderRuntime(
                {"wrong": lambda execution: None},
                close_callback=lambda: closed.append("invalid"),
            )
        return _runtime(source.source_id, closed)

    with pytest.raises(ValueError, match="exactly match"):
        ProviderCatalog((_registration(bind),)).bind(
            (
                ConfiguredSource("valid", "fake_records"),
                ConfiguredSource("invalid", "fake_records"),
            )
        )

    assert closed == ["invalid", "valid"]


def test_runtime_identity_cannot_be_shared_between_independent_sources() -> None:
    closed: list[str] = []
    shared = _runtime("shared", closed)
    registration = _registration(lambda source, context=None: shared)

    with pytest.raises(ValueError, match="separate runtime lease"):
        ProviderCatalog((registration,)).bind(
            (
                ConfiguredSource("first", "fake_records"),
                ConfiguredSource("second", "fake_records"),
            )
        )

    assert closed == ["shared"]


@pytest.mark.parametrize(
    "config",
    (
        EnvironmentConfig(kind="native"),
        EnvironmentConfig(
            kind="pyodide",
            host_managed_timeout=True,
            host_managed_isolation=True,
        ),
    ),
)
def test_environment_owns_registry_for_native_and_pyodide(config: EnvironmentConfig) -> None:
    closed: list[str] = []
    registration = _registration(lambda source, context=None: _runtime(source.source_id, closed))
    registry = ProviderCatalog((registration,)).bind((ConfiguredSource("records", "fake_records"),))
    environment = create_environment(
        config,
        context={},
        registry=registry,
        subcalls=MockSubcallClient(),
        execution_context=create_environment_context(config),
    )

    environment.close()
    environment.close()

    assert closed == ["records"]


def test_environment_factory_closes_registry_when_construction_fails() -> None:
    closed: list[str] = []
    registration = _registration(lambda source, context=None: _runtime(source.source_id, closed))
    registry = ProviderCatalog((registration,)).bind((ConfiguredSource("records", "fake_records"),))
    config = EnvironmentConfig(kind="native", budget=Budget(tokens=10_000))

    with pytest.raises(ValueError, match="must match"):
        create_environment(
            config,
            context={},
            registry=registry,
            subcalls=MockSubcallClient(),
            execution_context=create_environment_context(EnvironmentConfig(kind="native")),
        )

    assert closed == ["records"]


def test_run_closes_environment_on_process_control_during_scaffold_resolution() -> None:
    class ProcessControlEnvironment(MockEnvironment):
        def __init__(self) -> None:
            super().__init__()
            self.close_count = 0

        def globals(self):
            raise KeyboardInterrupt

        def close(self) -> None:
            self.close_count += 1

    environment = ProcessControlEnvironment()

    with pytest.raises(KeyboardInterrupt):
        run_rlm(
            "question",
            environment=environment,
            root_llm=MockLLMClient([]),
            subcalls=MockSubcallClient(),
            config=RLMConfig(),
        )

    assert environment.close_count == 1


def test_run_closes_owned_registry_on_process_control_during_context_binding() -> None:
    closed: list[str] = []
    registration = _registration(lambda source, context=None: _runtime(source.source_id, closed))
    registry = ProviderCatalog((registration,)).bind((ConfiguredSource("records", "fake_records"),))

    class ProcessControlSubcalls(MockSubcallClient):
        def bind_context(self, context) -> None:
            del context
            raise KeyboardInterrupt

    subcalls = ProcessControlSubcalls()
    config = EnvironmentConfig(kind="native")
    environment = create_environment(
        config,
        context={},
        registry=registry,
        subcalls=subcalls,
        execution_context=create_environment_context(config),
    )

    with pytest.raises(KeyboardInterrupt):
        run_rlm(
            "question",
            environment=environment,
            root_llm=MockLLMClient([]),
            subcalls=subcalls,
            config=RLMConfig(),
        )

    assert closed == ["records"]


def test_run_preserves_successful_result_when_environment_cleanup_fails() -> None:
    class CloseFailingEnvironment(MockEnvironment):
        def close(self) -> None:
            raise RuntimeError("provider close failed")

    root = MockLLMClient(
        [
            MockResponse(
                "```python\nanswer['content'] = 'kept'\nanswer['ready'] = True\n```",
                TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, exact=True),
            )
        ]
    )

    with pytest.warns(RuntimeWarning, match="result preserved.*provider close failed"):
        result = run_rlm(
            "question",
            environment=CloseFailingEnvironment(),
            root_llm=root,
            subcalls=MockSubcallClient(),
            config=RLMConfig(),
        )

    assert result.ready is True
    assert result.answer == "kept"


def test_run_does_not_adopt_an_exception_from_its_callers_handler() -> None:
    class CloseFailingEnvironment(MockEnvironment):
        def close(self) -> None:
            raise RuntimeError("provider close failed")

    root = MockLLMClient(
        [
            MockResponse(
                "```python\nanswer['content'] = 'fallback kept'\nanswer['ready'] = True\n```",
                TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, exact=True),
            )
        ]
    )

    try:
        raise ValueError("unrelated primary attempt")
    except ValueError:
        with pytest.warns(RuntimeWarning, match="result preserved.*provider close failed"):
            result = run_rlm(
                "fallback question",
                environment=CloseFailingEnvironment(),
                root_llm=root,
                subcalls=MockSubcallClient(),
                config=RLMConfig(),
            )

    assert result.ready is True
    assert result.answer == "fallback kept"


def test_run_groups_primary_and_environment_cleanup_failures() -> None:
    class FailingEnvironment(MockEnvironment):
        def globals(self):
            raise KeyboardInterrupt

        def close(self) -> None:
            raise RuntimeError("provider close failed")

    with pytest.raises(BaseExceptionGroup, match="setup and environment cleanup") as raised:
        run_rlm(
            "question",
            environment=FailingEnvironment(),
            root_llm=MockLLMClient([]),
            subcalls=MockSubcallClient(),
            config=RLMConfig(),
        )

    assert [type(exc) for exc in raised.value.exceptions] == [KeyboardInterrupt, RuntimeError]


def test_run_groups_execution_and_environment_cleanup_failures() -> None:
    class FailingEnvironment(MockEnvironment):
        def execute(self, code):
            del code
            raise KeyboardInterrupt

        def close(self) -> None:
            raise RuntimeError("provider close failed")

    root = MockLLMClient(
        [
            MockResponse(
                "```python\nanswer['content'] = 'unused'\n```",
                TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, exact=True),
            )
        ]
    )

    with pytest.raises(BaseExceptionGroup, match="execution and environment cleanup") as raised:
        run_rlm(
            "question",
            environment=FailingEnvironment(),
            root_llm=root,
            subcalls=MockSubcallClient(),
            config=RLMConfig(),
        )

    assert [type(exc) for exc in raised.value.exceptions] == [KeyboardInterrupt, RuntimeError]


def test_provider_service_and_registry_share_one_idempotent_runtime_owner() -> None:
    closed: list[str] = []
    registration = _registration(lambda source, context=None: _runtime(source.source_id, closed))
    registry = ProviderCatalog((registration,)).bind((ConfiguredSource("records", "fake_records"),))
    service = ProviderService(registry.sources[0])

    service.close()
    registry.close()
    service.close()

    assert closed == ["records"]


def test_filesystem_runtime_closes_pinned_root_without_gc(tmp_path, monkeypatch) -> None:
    real_close = os.close
    closed: list[int] = []

    def tracked_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr("droste.sources._filesystem_text_runtime.os.close", tracked_close)
    registry = ProviderCatalog((filesystem_text_provider(),)).bind(
        (
            ConfiguredSource(
                "docs",
                "filesystem_text",
                {"root": str(tmp_path)},
            ),
        )
    )

    registry.close()
    registry.close()

    assert len(closed) == 1


def test_sqlite_runtime_does_not_close_a_host_owned_connection() -> None:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    registry = ProviderCatalog((sqlite_provider(),)).bind(
        (ConfiguredSource("db", "sqlite"),),
        context=connection,
    )

    registry.close()

    assert connection.execute("SELECT 1").fetchone() == (1,)
    connection.close()


def test_sqlite_runtime_rejects_calls_after_registry_close(tmp_path) -> None:
    path = tmp_path / "facts.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE facts(value TEXT)")
    connection.close()
    registry = ProviderCatalog((sqlite_provider(),)).bind(
        (ConfiguredSource("db", "sqlite", {"sqlite_path": str(path)}),)
    )
    runtime = registry.sources[0].runtime

    registry.close()

    class Execution:
        @staticmethod
        def check() -> None:
            return None

    with pytest.raises(RuntimeError, match="runtime is closed"):
        runtime.handlers["schema"](Execution())
