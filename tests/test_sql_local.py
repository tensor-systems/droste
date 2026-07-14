"""Local-mode SQL data source.

Covers the local policy gate (SELECT-only, single statement, aggregate and
subquery allowances, LIMIT injection, row cap, timeout) and the end-to-end
path ported from the hosted platform's end-to-end suite: register() ->
build_data_sources -> DataSourceRegistry -> query()/get_schema() over a real
SQLite file opened mode=ro.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from droste.capabilities import CapabilityBroker, CapabilityCallError
from droste.registry import DataSourceRegistry
from droste.sources.sql_local import (
    DEFAULT_LOCAL_SQL_POLICY,
    LocalSqlDataSource,
    LocalSqlPolicy,
    SqlPolicyError,
    local_sql_source_factory,
    register,
    validate_local_sql,
)
from droste_runner.runner import _reset_source_types, build_data_sources


@pytest.fixture(autouse=True)
def _clean_source_registry():
    """register_source_type is process-global; isolate it per test."""
    _reset_source_types()
    yield
    _reset_source_types()


DEFAULT_POLICY = LocalSqlPolicy.from_spec(None)


def _make_db(tmp_path, script: str | None = None) -> str:
    path = str(tmp_path / "app.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        script
        or """
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, plan TEXT);
        INSERT INTO users VALUES (1,'ada','pro'),(2,'bob','free');
        """
    )
    conn.commit()
    conn.close()
    return path


# --- policy parsing ----------------------------------------------------------


def test_no_policy_uses_permissive_defaults() -> None:
    assert DEFAULT_POLICY.aggregations_allowed
    assert DEFAULT_POLICY.subqueries_allowed
    assert DEFAULT_POLICY.default_limit == 1000
    assert DEFAULT_POLICY.max_limit == 10000
    assert DEFAULT_POLICY.timeout_ms == 5000
    assert "group_concat" in DEFAULT_POLICY.aggregation_functions


def test_supplied_policy_without_aggregations_rejects_them() -> None:
    # Mirrors the cloud validator: an empty aggregations/subqueries policy
    # REJECTS aggregates and subqueries; permissiveness is opt-in.
    policy = LocalSqlPolicy.from_spec({"dialect": "sqlite", "read_only": True})
    assert not policy.aggregations_allowed
    assert not policy.subqueries_allowed


def test_read_only_false_rejected() -> None:
    with pytest.raises(ValueError, match="read-only"):
        LocalSqlPolicy.from_spec({"read_only": False})


def test_non_sqlite_dialect_rejected() -> None:
    with pytest.raises(ValueError, match="sqlite"):
        LocalSqlPolicy.from_spec({"dialect": "postgres"})


def test_explicit_zero_timeout_ms_is_preserved_not_defaulted() -> None:
    # `0 or 5000` would silently coerce an explicit 0 back to the 5000ms
    # default (0 is falsy) — timeout_ms=0 is a real, validated value (see
    # test_bad_limits_rejected: only < 0 is rejected) meaning "no timer"
    # (query()'s own `if timeout_ms > 0` gate). Regression for #8.
    policy = LocalSqlPolicy.from_spec({"limits": {"timeout_ms": 0}})
    assert policy.timeout_ms == 0


def test_bad_limits_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        LocalSqlPolicy.from_spec({"limits": {"default_limit": -1}})
    with pytest.raises(ValueError, match="timeout_ms"):
        LocalSqlPolicy.from_spec({"limits": {"timeout_ms": -5}})


# --- statement gate ----------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM users",
        "UPDATE users SET plan = 'pro'",
        "INSERT INTO users VALUES (3,'eve','pro')",
        "DROP TABLE users",
        "PRAGMA journal_mode",
        "WITH t AS (SELECT 1) SELECT * FROM t",  # guardrail is SELECT-only; no CTEs
    ],
)
def test_non_select_rejected(sql: str) -> None:
    with pytest.raises(SqlPolicyError, match="only SELECT statements are allowed"):
        validate_local_sql(sql, DEFAULT_POLICY)


def test_multi_statement_rejected() -> None:
    with pytest.raises(SqlPolicyError, match="multiple statements"):
        validate_local_sql("SELECT 1; SELECT 2", DEFAULT_POLICY)
    with pytest.raises(SqlPolicyError, match="multiple statements"):
        validate_local_sql("SELECT 1;", DEFAULT_POLICY)  # even a trailing semicolon


def test_comments_rejected() -> None:
    with pytest.raises(SqlPolicyError, match="comments"):
        validate_local_sql("SELECT 1 -- sneaky", DEFAULT_POLICY)
    with pytest.raises(SqlPolicyError, match="comments"):
        validate_local_sql("SELECT /* hidden */ 1", DEFAULT_POLICY)


def test_gate_ignores_string_literal_contents() -> None:
    # Semicolons, comment markers, and keywords inside literals must not trip
    # the gate: the scan runs over a literal-masked statement.
    sql = "SELECT name FROM users WHERE note = 'a;b -- (select c)'"
    normalized = validate_local_sql(sql, DEFAULT_POLICY)
    assert normalized.endswith("LIMIT 1000")


def test_escaped_quotes_inside_literal() -> None:
    normalized = validate_local_sql("SELECT 'it''s; fine' AS v", DEFAULT_POLICY)
    assert normalized.startswith("SELECT 'it''s; fine'")


def test_unterminated_literal_rejected() -> None:
    with pytest.raises(SqlPolicyError, match="unterminated"):
        validate_local_sql("SELECT 'oops", DEFAULT_POLICY)


def test_empty_sql_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        validate_local_sql("   ", DEFAULT_POLICY)


def test_aggregates_allowed_by_default_policy() -> None:
    validate_local_sql("SELECT count(*), sum(id) FROM users GROUP BY plan", DEFAULT_POLICY)
    validate_local_sql("SELECT GROUP_CONCAT(name) FROM users", DEFAULT_POLICY)


def test_aggregates_rejected_when_policy_omits_them() -> None:
    strict = LocalSqlPolicy.from_spec({"read_only": True})
    with pytest.raises(SqlPolicyError, match="aggregate functions are not allowed"):
        validate_local_sql("SELECT count(*) FROM users", strict)


def test_aggregate_outside_allowlist_rejected() -> None:
    policy = LocalSqlPolicy.from_spec(
        {"read_only": True, "aggregations": {"allowed": True, "functions": ["count"]}}
    )
    validate_local_sql("SELECT COUNT(*) FROM users", policy)
    with pytest.raises(SqlPolicyError, match="'sum' is not allowed"):
        validate_local_sql("SELECT sum(id) FROM users", policy)


def test_non_aggregate_functions_pass_regardless() -> None:
    strict = LocalSqlPolicy.from_spec({"read_only": True})
    validate_local_sql("SELECT lower(name), length(plan) FROM users", strict)


def test_subqueries_allowed_by_default_rejected_when_omitted() -> None:
    sql = "SELECT * FROM users WHERE id IN (SELECT id FROM users)"
    validate_local_sql(sql, DEFAULT_POLICY)
    strict = LocalSqlPolicy.from_spec({"read_only": True})
    with pytest.raises(SqlPolicyError, match="subqueries are not allowed"):
        validate_local_sql(sql, strict)


def test_limit_injected_when_absent_and_preserved_when_present() -> None:
    assert validate_local_sql("SELECT * FROM users", DEFAULT_POLICY).endswith("LIMIT 1000")
    kept = validate_local_sql("SELECT * FROM users LIMIT 5", DEFAULT_POLICY)
    assert kept == "SELECT * FROM users LIMIT 5"


def test_limit_inside_subquery_is_not_top_level() -> None:
    sql = "SELECT * FROM (SELECT id FROM users LIMIT 5) sub"
    assert validate_local_sql(sql, DEFAULT_POLICY).endswith("LIMIT 1000")


def test_limit_inside_string_is_not_top_level() -> None:
    sql = "SELECT * FROM users WHERE note = 'limit 5'"
    assert validate_local_sql(sql, DEFAULT_POLICY).endswith("LIMIT 1000")


# --- codex adversarial-review regressions ---------------------------
# Each exact bypass where the scanner disagreed with SQLite's lexer.


def test_quoted_aggregate_function_names_hit_the_policy() -> None:
    # Bypass 1: SQLite accepts "count"(*), [count](*), and `count`(*) as calls
    # to count; identifier quoting must not hide a call from the gate.
    strict = LocalSqlPolicy.from_spec({"read_only": True})
    for sql in (
        'SELECT "count"(*) FROM users',
        "SELECT [count](*) FROM users",
        "SELECT `count`(*) FROM users",
        'SELECT "COUNT" (*) FROM users',  # case + whitespace before the paren
    ):
        with pytest.raises(SqlPolicyError, match="aggregate functions are not allowed"):
            validate_local_sql(sql, strict)


def test_quoted_aggregate_names_respect_allowlist() -> None:
    policy = LocalSqlPolicy.from_spec(
        {"read_only": True, "aggregations": {"allowed": True, "functions": ["count"]}}
    )
    validate_local_sql('SELECT "count"(*) FROM users', policy)
    with pytest.raises(SqlPolicyError, match="'sum' is not allowed"):
        validate_local_sql("SELECT [sum](id) FROM users", policy)


def test_quoted_aggregate_name_without_call_is_not_flagged() -> None:
    # A *column* named count/sum/avg is data, not an aggregate call.
    strict = LocalSqlPolicy.from_spec({"read_only": True})
    validate_local_sql("SELECT [count] FROM users", strict)
    validate_local_sql('SELECT "sum", `avg` FROM users', strict)


def test_identifier_named_limit_does_not_suppress_injection() -> None:
    # Bypass 2: an identifier named limit ([limit], `limit`, "limit") is not a
    # LIMIT clause; the statement must still get the injected cap.
    for sql in (
        "SELECT x AS [limit] FROM t",
        "SELECT x AS `limit` FROM t",
        'SELECT x AS "limit" FROM t',
    ):
        assert validate_local_sql(sql, DEFAULT_POLICY).endswith("LIMIT 1000")


def test_unterminated_bracket_identifier_rejected() -> None:
    with pytest.raises(SqlPolicyError, match="unterminated"):
        validate_local_sql("SELECT x AS [limit FROM t", DEFAULT_POLICY)


def test_parenthesized_cte_and_values_count_as_subqueries() -> None:
    # Bypass 3: SQLite executes parenthesized CTEs and VALUES rows as
    # subqueries; "(select" alone is not the whole subquery grammar.
    strict = LocalSqlPolicy.from_spec({"read_only": True})
    for sql in (
        "SELECT * FROM (WITH t AS (VALUES(1)) SELECT * FROM t)",
        "SELECT * FROM (VALUES(1),(2))",
        "SELECT * FROM (  values (1))",  # whitespace-tolerant
        "SELECT * FROM (\n\tWITH t AS (VALUES(1)) SELECT * FROM t)",
    ):
        with pytest.raises(SqlPolicyError, match="subqueries are not allowed"):
            validate_local_sql(sql, strict)
    # Still fine when the policy allows subqueries.
    validate_local_sql("SELECT * FROM (VALUES(1),(2))", DEFAULT_POLICY)


def test_writable_ctx_connection_forced_read_only(tmp_path) -> None:
    # Bypass 4: a host handing a read-write connection must not silently
    # bypass the documented read-only defense — the source enforces
    # PRAGMA query_only=ON on the supplied connection.
    path = _make_db(tmp_path)
    ctx = sqlite3.connect(path, check_same_thread=False)  # read-write handle
    src = LocalSqlDataSource({"name": "db"}, ctx)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        src._connection().execute("INSERT INTO users VALUES (9,'mallory','free')")
    assert src.query("SELECT count(*) AS n FROM users") == [{"n": 2}]


# --- execution: caps, timeout, read-only ------------------------------------


def _source(tmp_path, policy: dict | None = None, script: str | None = None) -> LocalSqlDataSource:
    path = _make_db(tmp_path, script)
    config: dict = {"type": "sql", "name": "db", "sqlite_path": path}
    if policy is not None:
        config["policy"] = policy
    return LocalSqlDataSource(config)


def test_default_limit_caps_unbounded_queries(tmp_path) -> None:
    script = "CREATE TABLE t (x INTEGER);" + "".join(
        f"INSERT INTO t VALUES ({i});" for i in range(20)
    )
    src = _source(
        tmp_path,
        policy={
            "read_only": True,
            "limits": {"default_limit": 5, "max_limit": 10},
        },
        script=script,
    )
    assert len(src.query("SELECT x FROM t")) == 5  # injected default_limit
    assert len(src.query("SELECT x FROM t LIMIT 8")) == 8  # explicit LIMIT kept


def test_explicit_limit_over_max_fails_fast(tmp_path) -> None:
    script = "CREATE TABLE t (x INTEGER);" + "".join(
        f"INSERT INTO t VALUES ({i});" for i in range(20)
    )
    src = _source(
        tmp_path,
        policy={"read_only": True, "limits": {"default_limit": 5, "max_limit": 10}},
        script=script,
    )
    with pytest.raises(SqlPolicyError, match="max_limit"):
        src.query("SELECT x FROM t LIMIT 15")


def test_timeout_interrupts_long_query(tmp_path) -> None:
    script = "CREATE TABLE t (x INTEGER);" + "".join(
        f"INSERT INTO t VALUES ({i});" for i in range(500)
    )
    src = _source(
        tmp_path,
        policy={
            "read_only": True,
            "limits": {"timeout_ms": 50},
            "aggregations": {"allowed": True},
        },
        script=script,
    )
    with pytest.raises(RuntimeError, match="time limit"):
        # ~1.25e8 nested-loop rows: comfortably outlives a 50ms budget.
        src.query("SELECT count(*) FROM t a, t b, t c WHERE a.x != b.x")


def test_zero_timeout_never_arms_a_timer(tmp_path) -> None:
    # threading.Timer.start() spawns a real OS thread — unavailable under
    # Pyodide/WASM ("RuntimeError: can't start new thread" on the very first
    # query with the DEFAULT policy, since 5000 > 0 always arms one). A host
    # embedding droste under Pyodide relies on timeout_ms=0 to opt out of the
    # timer entirely (the host's own wall-clock kill covers it instead).
    # Regression for #8.
    from unittest import mock

    src = _source(tmp_path, policy={"read_only": True, "limits": {"timeout_ms": 0}})
    with mock.patch.object(threading.Timer, "start") as mock_start:
        rows = src.query("SELECT 1 AS x")
    assert rows == [{"x": 1}]
    mock_start.assert_not_called()


def test_threadless_platform_degrades_to_no_timer(tmp_path) -> None:
    # Under Pyodide/WASM (sys.platform == "emscripten"), threading imports
    # fine but Timer.start() raises RuntimeError("can't start new thread").
    # The DEFAULT policy (5000ms) must degrade to no timer — with a
    # RuntimeWarning naming the host's wall-clock timeout as the enforcement
    # — instead of failing every query. Fix for #8.
    from unittest import mock

    import droste.sources.sql_local as sql_local_mod

    src = _source(tmp_path)  # default policy: timeout_ms=5000, arms a timer
    with mock.patch.object(sql_local_mod.sys, "platform", "emscripten"):
        with mock.patch.object(
            threading.Timer, "start", side_effect=RuntimeError("can't start new thread")
        ):
            with pytest.warns(RuntimeWarning, match="wall-clock"):
                rows = src.query("SELECT 1 AS x")
    assert rows == [{"x": 1}]
    # Back on a threaded runtime, the very next query arms a timer again —
    # the degradation is per-attempt, not a sticky global.
    with mock.patch.object(threading.Timer, "start") as mock_start:
        src.query("SELECT 1 AS x")
    mock_start.assert_called_once()


def test_thread_exhaustion_on_threaded_platform_reraises(tmp_path) -> None:
    # On a NORMAL platform the same RuntimeError means thread exhaustion —
    # the process is resource-constrained, which is exactly when silently
    # dropping the query timeout would hurt most. Re-raise, don't degrade
    # (codex review on the #8 fix).
    from unittest import mock

    src = _source(tmp_path)
    with mock.patch.object(
        threading.Timer, "start", side_effect=RuntimeError("can't start new thread")
    ):
        with pytest.raises(RuntimeError, match="can't start new thread"):
            src.query("SELECT 1 AS x")


def test_timer_arms_inside_lock(tmp_path) -> None:
    # The codex race fix: a queued query's interrupt timer must be armed only
    # after it acquires the connection lock. If it were armed before, its
    # timeout would elapse while it is still blocked and conn.interrupt()
    # would abort the *lock holder's* query running on the shared connection.
    script = "CREATE TABLE t (x INTEGER);" + "".join(
        f"INSERT INTO t VALUES ({i});" for i in range(2000)
    )
    src = _source(
        tmp_path,
        policy={
            "read_only": True,
            "limits": {"timeout_ms": 30},
            "aggregations": {"allowed": True},
        },
        script=script,
    )
    conn = src._connection()
    entered = threading.Event()
    result: dict = {}

    def queued():
        entered.set()
        result["rows"] = src.query("SELECT 1 AS one")

    with src._lock:  # simulate an in-flight query holding the lock
        t = threading.Thread(target=queued)
        t.start()
        entered.wait(timeout=2)
        # The queued query's 30ms timeout elapses while it waits for the lock.
        # Run a long statement on the shared connection across that window: a
        # pre-lock-armed timer would interrupt it (sqlite3.OperationalError).
        conn.execute("SELECT count(*) FROM t a, t b WHERE a.x != b.x").fetchall()
    t.join(timeout=5)
    assert result["rows"] == [{"one": 1}]  # queued query ran cleanly after release


def test_writes_fail_at_sqlite_layer(tmp_path) -> None:
    src = _source(tmp_path)
    conn = src._connection()
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("INSERT INTO users VALUES (9,'mallory','free')")


def test_query_counts_and_returns_dict_rows(tmp_path) -> None:
    src = _source(tmp_path)
    rows = src.query("SELECT name FROM users WHERE plan = 'pro'")
    assert rows == [{"name": "ada"}]
    assert src.queries_made == 1


def test_ctx_connection_is_used_directly(tmp_path) -> None:
    path = _make_db(tmp_path)
    ctx = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    src = LocalSqlDataSource({"name": "db"}, ctx)
    assert src._connection() is ctx
    assert src.query("SELECT count(*) AS n FROM users") == [{"n": 2}]


def test_missing_sqlite_path_and_ctx_rejected() -> None:
    with pytest.raises(ValueError, match="sqlite_path"):
        LocalSqlDataSource({"name": "db"})


# --- schema + capabilities ----------------------------------------------------


def test_get_schema_lists_tables_and_columns(tmp_path) -> None:
    src = _source(tmp_path)
    schema = src.get_schema()
    assert "users(" in schema
    assert "name TEXT" in schema
    assert "read-only" in schema


def test_capabilities_are_sql_and_schema_only(tmp_path) -> None:
    caps = _source(tmp_path).capabilities()
    assert caps["sql"] and caps["schema"]
    assert not (caps["search"] or caps["get"] or caps["recent"] or caps["stats"])


# --- register() + end-to-end through the runner registry ---------------------
# Ported from ModelRelay's TestSqlDataSource_PythonEndToEnd (local half: the
# cloud /sql/validate stub is replaced by the in-process policy gate).


def test_register_end_to_end(tmp_path) -> None:
    register()
    path = _make_db(tmp_path)
    spec = {"type": "sql", "name": "db", "sqlite_path": path}
    sources, default = build_data_sources({"data_sources": [spec], "default_source": "db"}, None)
    assert len(sources) == 1 and default == "db"
    registry = DataSourceRegistry(sources, default_source_name=default)
    broker = CapabilityBroker(registry.capability_registrations())
    env = registry.broker_globals(broker)

    # Schema introspection describes the users table.
    assert "users(" in env["db"].get_schema()

    # SELECT works; rows come back as Python values.
    assert env["db"].query("SELECT name FROM users WHERE plan = 'pro'") == [{"name": "ada"}]

    # Non-SELECT is rejected with the policy message surfaced to the model.
    with pytest.raises(CapabilityCallError) as exc_info:
        env["db"].query("DELETE FROM users")
    assert exc_info.value.error.type == "SqlPolicyError"
    assert "only SELECT" in str(exc_info.value)

    # Writes fail at the SQLite layer (mode=ro defense in depth).
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        sources[0]._connection().execute("INSERT INTO users VALUES (9,'mallory','free')")

    # default_source flattens the verbs to top level.
    assert env["query"] is env["db"].query
    assert env["get_schema"] is env["db"].get_schema


def test_register_twice_fails_loudly() -> None:
    # register_source_type is fail-fast on duplicates; register() propagates.
    register()
    with pytest.raises(ValueError, match="already registered"):
        register()


def test_factory_passes_ctx_connection(tmp_path) -> None:
    path = _make_db(tmp_path)
    register()
    ctx = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    sources, _ = build_data_sources({"data_sources": [{"type": "sql", "name": "db"}]}, ctx)
    assert sources[0].query("SELECT count(*) AS n FROM users") == [{"n": 2}]


def test_factory_direct(tmp_path) -> None:
    path = _make_db(tmp_path)
    src = local_sql_source_factory({"name": "mydb", "sqlite_path": path}, None)
    assert src.name() == "mydb"


def test_default_policy_dict_matches_reference_shape() -> None:
    # Guard the exported constant against drift from ModelRelay's
    # defaultLocalSQLPolicy (cmd/mrl/commands_rlm.go).
    assert DEFAULT_LOCAL_SQL_POLICY["read_only"] is True
    assert DEFAULT_LOCAL_SQL_POLICY["dialect"] == "sqlite"
    assert DEFAULT_LOCAL_SQL_POLICY["limits"] == {
        "default_limit": 1000,
        "max_limit": 10000,
        "timeout_ms": 5000,
    }
    assert DEFAULT_LOCAL_SQL_POLICY["aggregations"]["allowed"] is True
    assert DEFAULT_LOCAL_SQL_POLICY["subqueries"]["allowed"] is True
