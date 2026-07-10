"""Bridge-backed DataSource (A'-2 sandbox split): DataSourceService <-> BridgeDataSource.

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
    # MockDataSource defines sample unconditionally -> hasattr is true -> bound.
    assert hasattr(bridged, "sample")
    # MockDataSource never defines "find" or "content".
    assert not hasattr(bridged, "find")
    assert not hasattr(bridged, "content")
    # Domain verbs are no longer droste vocabulary (#10): nothing binds them
    # unless a source declares them via extra_methods.
    for method in ("get_messages", "get_chats", "get_chat_messages"):
        assert not hasattr(bridged, method)


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

    assert bridged.sample(n=5) == []


def test_source_declared_extra_methods_forward_and_reach_the_registry() -> None:
    """The #10 seam end to end: a source declares its domain verbs in an
    `extra_methods` attribute; DataSourceService picks them up with no
    extra_methods= param, describe() reports them, BridgeDataSource binds
    and re-exposes them, and a registry over the bridged source flattens
    them into sandbox globals like any local source's."""

    class ChatArchiveSource(MockDataSource):
        extra_methods = ("get_messages", "get_chats")

        def get_messages(self, limit=10000):
            return [{"text": "hi"}]

        def get_chats(self):
            return [{"chat": "c1"}]

    service = DataSourceService(ChatArchiveSource())
    assert service.describe()["extra_methods"] == ["get_chats", "get_messages"]

    bridged = BridgeDataSource(service.handle, name="mock")
    assert bridged.get_messages(limit=5) == [{"text": "hi"}]
    assert bridged.extra_methods == ("get_chats", "get_messages")

    env = DataSourceRegistry([bridged], default_source_name="mock").globals()
    assert env["get_chats"]() == [{"chat": "c1"}]
    assert env["mock"].get_messages() == [{"text": "hi"}]


def test_service_validates_declared_extras_at_construction() -> None:
    """Transport parity (codex review on #10): the registry raises a config
    error for a bad declared extra; the service must too — not advertise it
    through describe() and fail with a late TypeError inside a dispatch."""
    import pytest

    class DeclaredNotCallable(MockDataSource):
        extra_methods = ("stats_blob",)
        stats_blob = {"not": "callable"}

    with pytest.raises(ValueError, match="is not a callable"):
        DataSourceService(DeclaredNotCallable())

    class DeclaredMissing(MockDataSource):
        extra_methods = ("nonexistent_verb",)

    with pytest.raises(ValueError, match="is not a callable"):
        DataSourceService(DeclaredMissing())

    class ParamNotCallable(MockDataSource):
        weird = 42

    with pytest.raises(ValueError, match="is not a callable"):
        DataSourceService(ParamNotCallable(), extra_methods=("weird",))

    class PresentButNone(MockDataSource):
        maybe_verb = None

    with pytest.raises(ValueError, match="is not a callable"):
        DataSourceService(PresentButNone(), extra_methods=("maybe_verb",))

    # Same vocabulary rule as the registry (transport parity): an extra may
    # not reuse an engine verb, even a disabled one — the bridge dispatches
    # core names first, so it would be advertised yet unreachable.
    class ShadowsCoreVerb(MockDataSource):
        extra_methods = ("query",)

    with pytest.raises(ValueError, match="collides with an engine verb"):
        DataSourceService(ShadowsCoreVerb())

    # Machinery/protocol names and underscored attributes are never valid
    # extras — an extra named `_request` would let the bridged client
    # setattr over its own proxy machinery (codex review).
    class ShadowsMachinery(MockDataSource):
        extra_methods = ("_request",)

        def _request(self):  # pragma: no cover - never reached
            return None

    with pytest.raises(ValueError, match="underscore"):
        DataSourceService(ShadowsMachinery())

    class ShadowsDescribe(MockDataSource):
        extra_methods = ("describe",)

        def describe(self):  # pragma: no cover - never reached
            return {}

    with pytest.raises(ValueError, match="collides"):
        DataSourceService(ShadowsDescribe())

    # Builtins: a flattened default-source verb named `len` would hijack
    # every len() call in generated code (codex review).
    class ShadowsBuiltin(MockDataSource):
        extra_methods = ("len",)

        def len(self):  # pragma: no cover - never reached  # noqa: A003
            return 0

    with pytest.raises(ValueError, match="builtin"):
        DataSourceService(ShadowsBuiltin())

    # Non-identifiers and keywords: generated code could never call
    # `fetch-page()` or `class()` — reject at declaration (codex review).
    for bad_name in ("fetch-page", "", "class"):

        class BadName(MockDataSource):
            extra_methods = (bad_name,)

        with pytest.raises(ValueError, match="not a valid Python identifier"):
            DataSourceService(BadName())


def test_bridged_client_rejects_unsafe_advertised_names() -> None:
    """Defense in depth: even if the service-side validation is bypassed
    (spoofed/buggy remote), BridgeDataSource must refuse to setattr over its
    own machinery from describe()'s advertised names."""
    import pytest

    def spoofed_bridge_call(method: str, params_json: str) -> str:
        assert method == "describe"
        return json.dumps(
            {
                "ok": True,
                "result": {
                    "name": "evil",
                    "capabilities": {},
                    "schema": "",
                    "optional_methods": ["_request"],
                    "extra_methods": [],
                },
            }
        )

    with pytest.raises(ValueError, match="unsafe optional method name"):
        BridgeDataSource(spoofed_bridge_call, name="evil")


def test_extra_method_may_not_shadow_a_disabled_core_verb() -> None:
    """Transport parity (codex review on #10): the bridge dispatches core
    names before extras, so an extra shadowing a DISABLED core verb would
    work in-process but be rejected across the bridge. The registry must
    reject it against the full engine vocabulary, not just enabled verbs."""
    import pytest

    class ShadowingSource(MockDataSource):
        extra_methods = ("query",)

        def capabilities(self):
            caps = dict(super().capabilities())
            caps["sql"] = False  # `query` core verb disabled...
            return caps

    with pytest.raises(ValueError, match="collides with an engine verb"):
        DataSourceRegistry([ShadowingSource()]).globals()


def test_extra_methods_forward_like_any_other_optional_verb() -> None:
    """A host's own optional verb (e.g. a host app's get_retrieved_guids(), not part
    of the DataSource Protocol or droste's own _OPTIONAL_METHODS) round-trips
    through extra_methods= with zero BridgeDataSource changes — it just binds
    dynamically from whatever describe() reports, same as any other optional
    verb."""

    class HostSource(MockDataSource):
        def get_retrieved_guids(self) -> list[str]:
            return ["guid-1", "guid-2"]

    service = DataSourceService(HostSource(), extra_methods=("get_retrieved_guids",))
    bridged = BridgeDataSource(service.handle, name="mock")

    assert hasattr(bridged, "get_retrieved_guids")
    assert bridged.get_retrieved_guids() == ["guid-1", "guid-2"]

    # A source that doesn't implement the extra method doesn't get it bound,
    # same as any other optional verb.
    plain_service = DataSourceService(MockDataSource(), extra_methods=("get_retrieved_guids",))
    plain_bridged = BridgeDataSource(plain_service.handle, name="mock")
    assert not hasattr(plain_bridged, "get_retrieved_guids")

    # And it's rejected server-side if a caller tries to reach it anyway on a
    # source that doesn't implement it — same allowlist discipline as any
    # other verb, droste-defined or host-defined.
    raw = plain_service.handle("get_retrieved_guids", json.dumps({"args": [], "kwargs": {}}))
    envelope = json.loads(raw)
    assert envelope["ok"] is False
    assert envelope["error"]["type"] == "PermissionError"


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
    for method in ("find", "content", "sample"):
        assert not hasattr(bridged, method)
    # And no class-level definitions leak through either — a second bridged
    # instance over a source that DOES have them must differ per-instance.
    other = BridgeDataSource(DataSourceService(MockDataSource()).handle, name="mock")
    assert hasattr(other, "sample")
    assert not hasattr(bridged, "sample")


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
    raw = service.handle("sample", "{}")
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
    raw = bare_service.handle("sample", "{}")
    envelope = json.loads(raw)
    assert envelope["ok"] is False
    assert envelope["error"]["type"] == "PermissionError"

    # A domain verb nobody declared is not even a known method name — the
    # allowlist rejects it as unknown, not merely unimplemented (#10).
    raw = bare_service.handle("get_messages", "{}")
    envelope = json.loads(raw)
    assert envelope["ok"] is False
    assert envelope["error"]["type"] == "ValueError"
    assert "unknown bridge method" in envelope["error"]["message"]


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
