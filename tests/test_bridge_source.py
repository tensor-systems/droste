"""Bridge-backed DataSource (droste#3, A'-2): DataSourceService <-> BridgeDataSource.

A pure in-process loopback (`BridgeDataSource(bridge_call=service.handle, ...)`)
stands in for the real cross-interpreter transport (a Deno JSON-RPC bridge
between two Pyodide contexts) — the wire contract is exactly the same either
way, so this is a faithful test of the boundary without needing Pyodide.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from droste.registry import DataSourceRegistry
from droste.sources.bridge import BridgeDataSource, DataSourceService
from droste.sources.sql_local import LocalSqlDataSource, SqlPolicyError
from droste.testing import MockDataSource

# --- loopback round-trip over a full-capability source ----------------------


def test_describe_reports_capabilities_schema_and_optional_methods() -> None:
    source = MockDataSource(schema="mock schema", stats={"rows": 3})
    service = DataSourceService(source)
    bridged = BridgeDataSource(service.handle, name="mock")

    assert bridged.name() == "mock"
    assert bridged.capabilities() == source.capabilities()
    assert bridged.get_schema() == "mock schema"
    # MockDataSource defines these unconditionally -> hasattr is true -> bound.
    for method in ("get_messages", "get_chats", "get_chat_messages", "sample"):
        assert hasattr(bridged, method)
    # MockDataSource never defines "find" or "content".
    assert not hasattr(bridged, "find")
    assert not hasattr(bridged, "content")


def test_core_verbs_round_trip_through_the_bridge() -> None:
    source = MockDataSource(
        schema="s",
        stats={"rows": 1},
        query_results={"SELECT 1": [{"a": 1}]},
    )
    service = DataSourceService(source)
    bridged = BridgeDataSource(service.handle, name="mock")

    assert bridged.query("SELECT 1") == [{"a": 1}]
    assert bridged.search("hi") == []
    assert bridged.get("id1") is None
    assert bridged.get_recent(days=1, limit=5) == []
    assert bridged.get_stats() == {"rows": 1}


def test_search_does_not_inject_defaults_for_a_narrower_signature() -> None:
    """A caller-omitted search() kwarg must not be forwarded at all — only what
    the caller actually passed. Otherwise a source whose search() doesn't
    accept the chat-archive-style extras (e.g. WrapperV1DataSource.search(query,
    filters=None, page=None), no limit/sender/chat/days/table) breaks with a
    TypeError on every bridged call, even though calling it directly works."""

    class NarrowSearchSource(MockDataSource):
        def search(self, query, filters=None, page=None):
            return [{"query": query, "filters": filters, "page": page}]

    service = DataSourceService(NarrowSearchSource())
    bridged = BridgeDataSource(service.handle, name="mock")

    assert bridged.search("hi") == [{"query": "hi", "filters": None, "page": None}]
    # An explicitly passed kwarg the narrow source *does* accept still forwards.
    assert bridged.search("hi", filters={"a": 1}) == [
        {"query": "hi", "filters": {"a": 1}, "page": None}
    ]


def test_optional_verb_forwards_through_the_bridge() -> None:
    source = MockDataSource()
    service = DataSourceService(source)
    bridged = BridgeDataSource(service.handle, name="mock")

    assert bridged.get_chats() == []
    assert bridged.get_chat_messages("chat1", limit=10) == []
    assert bridged.sample(n=5) == []


def test_registry_composes_a_bridged_source_like_any_other() -> None:
    source = MockDataSource(query_results={"SELECT": [{"row": 1}]})
    service = DataSourceService(source)
    bridged = BridgeDataSource(service.handle, name="mock")

    registry = DataSourceRegistry([bridged], default_source_name="mock")
    env = registry.globals()

    assert env["mock"].query("SELECT") == [{"row": 1}]
    # Default source is also flattened unprefixed.
    assert env["query"]("SELECT") == [{"row": 1}]
    assert "## Data Sources" in registry.prompt_fragment()


# --- optional methods are bound only when the remote side reports them -----


def test_optional_methods_not_bound_when_source_lacks_them(tmp_path) -> None:
    db_path = tmp_path / "t.sqlite"
    sqlite3.connect(str(db_path)).execute("CREATE TABLE t (a INTEGER)").connection.commit()
    source = LocalSqlDataSource({"sqlite_path": str(db_path)})
    service = DataSourceService(source)
    bridged = BridgeDataSource(service.handle, name="db")

    # LocalSqlDataSource has none of the optional verbs.
    for method in ("find", "content", "get_messages", "get_chats", "get_chat_messages", "sample"):
        assert not hasattr(bridged, method)
    # And no class-level definitions leak through either — a second bridged
    # instance over a source that DOES have them must differ per-instance.
    other = BridgeDataSource(DataSourceService(MockDataSource()).handle, name="mock")
    assert hasattr(other, "get_chats")
    assert not hasattr(bridged, "get_chats")


# --- security: the fixed method allowlist is the real boundary -------------


def test_unknown_method_name_is_rejected_not_getattred() -> None:
    service = DataSourceService(MockDataSource())
    raw = service.handle("__init__", "{}")
    envelope = json.loads(raw)
    assert envelope["ok"] is False
    assert envelope["error"]["type"] == "ValueError"
    assert "unknown bridge method" in envelope["error"]["message"]


def test_capability_gated_method_rejected_when_disabled(tmp_path) -> None:
    db_path = tmp_path / "t.sqlite"
    sqlite3.connect(str(db_path)).execute("CREATE TABLE t (a INTEGER)").connection.commit()
    source = LocalSqlDataSource({"sqlite_path": str(db_path)})
    assert source.capabilities()["search"] is False
    service = DataSourceService(source)
    bridged = BridgeDataSource(service.handle, name="db")

    # search is disabled -> BridgeDataSource never binds it, same as any other
    # DataSource that simply doesn't implement an unsupported verb.
    assert not hasattr(bridged, "search")

    # But nothing stops generated code from reaching the raw bridge_call
    # directly (e.g. via `db.query.__self__._call`), bypassing BridgeDataSource
    # entirely — so the SERVICE must independently reject it too.
    raw = service.handle("search", json.dumps({"args": ["anything"], "kwargs": {}}))
    envelope = json.loads(raw)
    assert envelope["ok"] is False
    assert envelope["error"]["type"] == "PermissionError"


def test_optional_method_rejected_when_source_lacks_it() -> None:
    service = DataSourceService(MockDataSource())
    raw = service.handle("get_messages", "{}")
    envelope = json.loads(raw)
    assert envelope["ok"] is True  # MockDataSource does implement it

    class NoOptional:
        def name(self) -> str:
            return "bare"

        def capabilities(self) -> dict[str, bool]:
            return {
                "sql": False,
                "search": False,
                "get": False,
                "recent": False,
                "schema": False,
                "stats": False,
            }

        def get_schema(self) -> str:
            return ""

        def get_stats(self) -> dict[str, Any]:
            return {}

        def search(self, *a, **k):
            return []

        def query(self, sql):
            return []

        def get(self, id):
            return None

        def get_recent(self, days=7, limit=100):
            return []

    bare_service = DataSourceService(NoOptional())
    raw = bare_service.handle("get_messages", "{}")
    envelope = json.loads(raw)
    assert envelope["ok"] is False
    assert envelope["error"]["type"] == "PermissionError"


def test_bad_params_json_rejected() -> None:
    service = DataSourceService(MockDataSource())
    raw = service.handle("query", "not json")
    envelope = json.loads(raw)
    assert envelope["ok"] is False

    raw = service.handle("query", "[1, 2]")  # valid JSON, not an object
    envelope = json.loads(raw)
    assert envelope["ok"] is False
    assert "must be a JSON object" in envelope["error"]["message"]


# --- real SQLite behind the bridge: fidelity + error propagation -----------


def test_real_sqlite_query_round_trips_through_the_bridge(tmp_path) -> None:
    db_path = tmp_path / "t.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE people (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO people VALUES (1, 'ada'), (2, 'grace')")
    conn.commit()
    conn.close()

    source = LocalSqlDataSource({"sqlite_path": str(db_path)})
    service = DataSourceService(source)
    bridged = BridgeDataSource(service.handle, name="db")

    direct = source.query("SELECT * FROM people ORDER BY id")
    via_bridge = bridged.query("SELECT * FROM people ORDER BY id")
    assert (
        direct
        == via_bridge
        == [
            {"id": 1, "name": "ada"},
            {"id": 2, "name": "grace"},
        ]
    )
    assert "people(id INTEGER, name TEXT)" in bridged.get_schema()


def test_sql_policy_violation_message_survives_the_bridge(tmp_path) -> None:
    db_path = tmp_path / "t.sqlite"
    sqlite3.connect(str(db_path)).execute("CREATE TABLE t (a INTEGER)").connection.commit()
    source = LocalSqlDataSource({"sqlite_path": str(db_path)})
    service = DataSourceService(source)
    bridged = BridgeDataSource(service.handle, name="db")

    # Sanity: the same statement is rejected identically without the bridge.
    with pytest.raises(SqlPolicyError, match="only SELECT statements"):
        source.query("DELETE FROM t")

    with pytest.raises(RuntimeError, match="only SELECT statements"):
        bridged.query("DELETE FROM t")


# --- JSON fidelity: non-JSON-native values are coerced, not fatal ----------


def test_non_json_native_value_is_coerced_via_default_str() -> None:
    class BytesSource(MockDataSource):
        def query(self, sql: str) -> list[dict[str, Any]]:
            return [{"blob": b"\x00\x01"}]

    service = DataSourceService(BytesSource())
    bridged = BridgeDataSource(service.handle, name="mock")
    row = bridged.query("SELECT 1")[0]
    assert row["blob"] == str(b"\x00\x01")


# --- malformed / non-JSON transport responses surface as RuntimeError ------


def test_non_json_bridge_response_raises_runtime_error() -> None:
    service = DataSourceService(MockDataSource())

    def broken_call(method: str, params_json: str) -> str:
        if method == "describe":
            return service.handle(method, params_json)
        return "not json at all"

    bridged = BridgeDataSource(broken_call, name="mock")
    with pytest.raises(RuntimeError, match="non-JSON response"):
        bridged.query("SELECT 1")
