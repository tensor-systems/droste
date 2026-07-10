"""Unified data sources: registry-through-runner + WrapperV1DataSource.

Covers the source-unification plumbing — build_data_sources(), the WrapperV1DataSource
remote adapter, and that RunnerEnvironment now sources its globals/prompt from a
DataSourceRegistry instead of the old flat data_source_* special-case.
"""

from __future__ import annotations

import pytest

from droste.registry import DataSourceRegistry
from droste.testing import MockDataSource, MockSubcallClient
from droste_runner.runner import (
    SOURCE_PROTOCOL_VERSION,
    RunnerEnvironment,
    WrapperV1DataSource,
    _reset_source_types,
    build_data_sources,
    register_source_type,
)


@pytest.fixture(autouse=True)
def _clean_source_registry():
    """register_source_type is process-global; isolate it per test."""
    _reset_source_types()
    yield
    _reset_source_types()


# --- build_data_sources ----------------------------------------------------


def test_legacy_singular_data_source_is_wrapper_sugar() -> None:
    sources, default = build_data_sources({"data_source": {"base_url": "https://x", "token": "t"}})
    assert len(sources) == 1
    assert isinstance(sources[0], WrapperV1DataSource)
    assert sources[0].name() == "wrapper"
    # The single legacy source flattens to top-level globals by default.
    assert default == "wrapper"


def test_data_sources_list_builds_named_wrapper() -> None:
    sources, default = build_data_sources(
        {
            "data_sources": [
                {"type": "wrapper_v1", "name": "partner", "base_url": "https://x", "token": "t"}
            ],
            "default_source": "partner",
        }
    )
    assert [s.name() for s in sources] == ["partner"]
    assert default == "partner"


def test_no_data_source_yields_empty() -> None:
    sources, default = build_data_sources({})
    assert sources == []
    assert default is None


def test_unknown_type_raises() -> None:
    with pytest.raises(ValueError, match="unknown data source type"):
        build_data_sources({"data_sources": [{"type": "redis", "name": "r"}]})


def test_sql_and_fs_unregistered_raise_loudly() -> None:
    for stype in ("sql", "fs"):
        with pytest.raises(ValueError, match="no registered factory"):
            build_data_sources({"data_sources": [{"type": stype, "name": "db"}]})


def test_data_sources_entry_must_be_object() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        build_data_sources({"data_sources": ["not-a-dict"]})


# --- register_source_type (Option C, source unification) -----------------


def test_registered_factory_builds_source_with_config_and_ctx() -> None:
    seen: dict[str, object] = {}

    def factory(config, ctx):
        seen["config"] = config
        seen["ctx"] = ctx
        return MockDataSource(schema="tables: t")

    register_source_type("sql", factory, protocol=SOURCE_PROTOCOL_VERSION)
    ctx = object()
    sources, default = build_data_sources(
        {
            "data_sources": [{"type": "sql", "name": "db", "profile_id": "p1"}],
            "default_source": "db",
        },
        ctx,
    )
    assert len(sources) == 1 and default == "db"
    assert seen["config"] == {"type": "sql", "name": "db", "profile_id": "p1"}
    assert seen["ctx"] is ctx


def test_factory_dispatch_is_not_limited_to_sql_fs() -> None:
    register_source_type(
        "messages", lambda config, ctx: MockDataSource(schema="m"), protocol=SOURCE_PROTOCOL_VERSION
    )
    sources, _ = build_data_sources({"data_sources": [{"type": "messages", "name": "chats"}]})
    assert len(sources) == 1


def test_factory_returning_none_raises() -> None:
    register_source_type("sql", lambda config, ctx: None, protocol=SOURCE_PROTOCOL_VERSION)
    with pytest.raises(ValueError, match="returned no source"):
        build_data_sources({"data_sources": [{"type": "sql", "name": "db"}]})


def test_duplicate_registration_raises() -> None:
    register_source_type(
        "sql", lambda config, ctx: MockDataSource(schema="t"), protocol=SOURCE_PROTOCOL_VERSION
    )
    with pytest.raises(ValueError, match="already registered"):
        register_source_type(
            "sql", lambda config, ctx: MockDataSource(schema="t"), protocol=SOURCE_PROTOCOL_VERSION
        )


def test_wrapper_v1_cannot_be_reregistered() -> None:
    with pytest.raises(ValueError, match="built in"):
        register_source_type(
            "wrapper_v1", lambda config, ctx: None, protocol=SOURCE_PROTOCOL_VERSION
        )


def test_blank_type_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        register_source_type("  ", lambda config, ctx: None, protocol=SOURCE_PROTOCOL_VERSION)


def test_protocol_mismatch_fails_at_registration() -> None:
    with pytest.raises(RuntimeError, match="protocol"):
        register_source_type(
            "sql",
            lambda config, ctx: MockDataSource(schema="t"),
            protocol=SOURCE_PROTOCOL_VERSION + 1,
        )


def test_request_cannot_name_a_module() -> None:
    # The request stays declarative: an unregistered type never triggers an
    # import, it just fails. (This is the whole point of Option C.)
    with pytest.raises(ValueError, match="unknown data source type"):
        build_data_sources({"data_sources": [{"type": "some.module.path", "name": "x"}]})


def test_run_threads_source_ctx_to_factories(monkeypatch) -> None:
    # Guard the run() -> build_data_sources(request, source_ctx) passthrough:
    # a regression back to build_data_sources(request) would silently drop the
    # host-supplied edge context.
    import droste_runner.runner as runner_mod

    seen: dict[str, object] = {}

    def factory(config, ctx):
        seen["ctx"] = ctx
        return MockDataSource(schema="t")

    register_source_type("sql", factory, protocol=SOURCE_PROTOCOL_VERSION)

    from types import SimpleNamespace

    def fake_run_rlm(question, **kwargs):
        return SimpleNamespace(
            answer="ok",
            ready=True,
            iterations=1,
            tokens_used=0,
            sub_calls_made=0,
            trajectory=[],
            error=None,
        )

    monkeypatch.setattr(runner_mod, "run_rlm", fake_run_rlm)

    ctx = object()
    response = runner_mod.run(
        {
            "protocol_version": 1,
            "question": "q",
            "token": "t",
            "root_endpoint": "https://cloud/root",
            "subcall_endpoint": "https://cloud/subcall",
            "data_sources": [{"type": "sql", "name": "db"}],
        },
        source_ctx=ctx,
    )
    assert seen["ctx"] is ctx
    assert response["answer"] == "ok"


def test_main_rejects_adapter_module_from_request_file(monkeypatch, tmp_path) -> None:
    # The request file is the untrusted boundary: it must never name code to
    # import. Trusted in-process callers of run() keep the adapter seam.
    import droste_runner.runner as runner_mod

    request_path = tmp_path / "request.json"
    # protocol_version present so the request reaches the security check —
    # the version gate deliberately runs first (see test_rlm_runner_adapter's
    # test_main_version_gate_precedes_adapter_module_rejection).
    request_path.write_text('{"protocol_version": 1, "adapter_module": "evil.module"}')
    monkeypatch.setenv("RLM_RUNNER_REQUEST_PATH", str(request_path))
    with pytest.raises(RuntimeError, match="not accepted from the request file"):
        runner_mod.main()


# --- WrapperV1DataSource + registry wiring ---------------------------------


def test_wrapper_capabilities_and_schema() -> None:
    src = WrapperV1DataSource(
        {
            "base_url": "https://x",
            "token": "t",
            "allowed_hosts": ["x"],
            "limits": {"max_requests": 5},
        },
        name="partner",
    )
    caps = src.capabilities()
    assert caps["search"] and caps["get"] and caps["stats"]
    assert caps["sql"] is False
    schema = src.get_schema()
    assert "wrapper_v1" in schema
    assert "Allowed hosts: x" in schema
    assert "max_requests" in schema


def test_registry_exposes_wrapper_verbs_including_content() -> None:
    src = WrapperV1DataSource({"base_url": "https://x", "token": "t"}, name="partner")
    env = DataSourceRegistry([src], default_source_name="partner").globals()
    ns = env["partner"]
    # Namespace is attribute-accessible (db.search), not a dict (db["search"]).
    assert all(hasattr(ns, v) for v in ("search", "get", "content", "get_stats"))
    assert not hasattr(ns, "query")  # no sql capability
    # default_source flattens the verbs to top level.
    assert env["search"] is ns.search
    assert env["content"] is ns.content


def test_registry_sql_source_exposes_query_and_schema() -> None:
    env = DataSourceRegistry([MockDataSource(schema="t")], default_source_name="mock").globals()
    assert hasattr(env["mock"], "query")
    assert hasattr(env["mock"], "get_schema")  # schema capability -> callable
    assert env["query"] is env["mock"].query


def test_reserved_and_duplicate_names_rejected() -> None:
    class Named(MockDataSource):
        def __init__(self, n: str) -> None:
            super().__init__(schema="t")
            self._n = n

        def name(self) -> str:
            return self._n

    with pytest.raises(ValueError, match="reserved"):
        DataSourceRegistry([Named("context")]).globals()
    with pytest.raises(ValueError, match="duplicate"):
        DataSourceRegistry([Named("db"), Named("db")]).globals()


def test_unknown_default_source_rejected() -> None:
    with pytest.raises(ValueError, match="not a defined source"):
        DataSourceRegistry([MockDataSource(schema="t")], default_source_name="nope").globals()


def test_non_list_data_sources_rejected() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        build_data_sources({"data_sources": {"type": "wrapper_v1", "base_url": "x", "token": "t"}})


# --- RunnerEnvironment integration -----------------------------------------


def _env(registry: DataSourceRegistry | None) -> RunnerEnvironment:
    return RunnerEnvironment(
        context={"files": []},
        registry=registry,
        subcalls=MockSubcallClient(),
        max_output_chars=10000,
        exec_timeout_ms=0,
    )


def test_environment_merges_registry_globals() -> None:
    src = WrapperV1DataSource({"base_url": "https://x", "token": "t"}, name="partner")
    env = _env(DataSourceRegistry([src], default_source_name="partner"))
    g = env.globals()
    assert "partner" in g and "search" in g  # namespaced + flattened
    assert callable(g["llm_query"])  # base globals still present


def test_query_is_attribute_callable_in_sandbox() -> None:
    """The defining RLM property: `db.query(...)` runs in the sandbox and its
    result is a Python value the model computes over — not a tool call whose
    result returns to the context window. Drive it through the real executor."""
    src = MockDataSource(schema="t", query_results={"SELECT": [{"x": 1}, {"x": 2}]})
    environment = _env(DataSourceRegistry([src]))
    # Exactly what the prompt tells the model to write — attribute access, then
    # arbitrary Python over the returned rows.
    result = environment.execute("rows = mock.query('SELECT x')\ntotal = sum(r['x'] for r in rows)")
    assert result.exit_code == 0
    g = environment.globals()
    assert g["rows"] == [{"x": 1}, {"x": 2}]
    assert g["total"] == 3


def test_environment_prompt_fragment_reflects_sources() -> None:
    with_src = _env(DataSourceRegistry([MockDataSource(schema="my-schema")]))
    frag = with_src.prompt_fragment()
    assert "## Data Sources" in frag
    assert "my-schema" in frag
    # RLM framing: data is pulled into persistent Python variables, not tool-called
    # into the context window. Guard against regressing to a tool-menu prompt.
    assert "Python variables" in frag
    assert "persist across iterations" in frag

    without = _env(None)
    bare = without.prompt_fragment()
    assert "## Data Sources" not in bare
    assert "context" in bare  # the context guidance line remains
