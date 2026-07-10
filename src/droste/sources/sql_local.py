"""Local-mode SQL data source: SQLite, read-only, policy-gated in-process.

The one principle: SQL data behaves like a variable in a Python REPL, not a
tool you call. The surface is thin — ``query(sql)`` (arbitrary read-only
SELECT, rows come back as ``list[dict]``) plus ``get_schema()`` for
navigation. No search/get_recent/sample verbs: the model writes those itself
in Python over the returned values.

``query`` is the policy application point: every statement passes a local
read-only gate (SELECT-only, single statement, no comments, aggregate and
subquery allowances per the policy) and executes with a top-level LIMIT
injected when absent, a hard result-row cap, and a wall-clock timeout, against
a SQLite connection opened ``mode=ro`` (a live connection supplied by the host
via ``ctx`` is forced read-only with ``PRAGMA query_only=ON``).

Scope: this module is the **local half** of the SQL source only. The cloud
validate-URL path (``profile_id`` + stateless ``POST /sql/validate`` — the
hosted "validated SQL policies, audited, per-user billing" upgrade) is
explicitly out of scope here; that stays a ModelRelay integration which wraps
this module.

Threat model (important — do not overclaim):
    In the **in-process / local** host (the beachhead, e.g. ``mrl rlm --db``)
    the sandbox runs the model's *arbitrary Python* on the same machine, so
    nothing in this class is a security boundary against the model: it can
    open its own ``sqlite3`` connection to ``_sqlite_path`` (read-write,
    ignoring our ``mode=ro``), reach the raw handle via ``query().__self__``,
    or skip ``db`` entirely. ``mode=ro`` only guarantees *our* accessor never
    mutates; the SQL policy is a **guardrail** that injects LIMITs and shapes
    what the model is guided to do, not an allowlist that constrains it. The
    real controls in local mode are the same ones that bound any local script
    the user runs: **OS file permissions** and the user's choice of which
    database to point at. In a **hosted / multi-tenant** host the sandbox is a
    real isolation boundary and the edge connection handed to the child must
    itself be a scoped, short-lived, least-privilege read-only connection —
    the policy is enforced there by that constrained connection plus the
    sandbox, not by this class alone.

Threadless runtimes (Pyodide/WASM):
    The per-query ``timeout_ms`` is enforced with a ``threading.Timer``. On
    platforms that are threadless by design (``sys.platform`` in
    ``emscripten``/``wasi`` — ``threading`` imports but ``start()`` raises
    ``RuntimeError``), queries still run: the timer is skipped with a one-time
    ``RuntimeWarning`` and the substrate's own wall-clock kill (e.g. the Deno
    relay's process timeout) is the enforcement. On normal threaded platforms
    the same ``RuntimeError`` means thread exhaustion and is re-raised.
"""

from __future__ import annotations

import re
import sqlite3
import sys
import threading
import warnings
from dataclasses import dataclass
from typing import Any

from .registration import SOURCE_PROTOCOL_VERSION, register_source_type

# Read-only policy used when the spec carries no policy: SELECT-only with sane
# limits, but otherwise permissive (aggregates, joins, subqueries) — it's the
# user's own local file, and the restrictive knobs exist for locked-down
# profiles, not for this. Mirrors the hosted platform's default SQL policy.
DEFAULT_LOCAL_SQL_POLICY: dict[str, Any] = {
    "dialect": "sqlite",
    "read_only": True,
    "limits": {"default_limit": 1000, "max_limit": 10000, "timeout_ms": 5000},
    "aggregations": {
        "allowed": True,
        "functions": ["count", "sum", "avg", "min", "max", "total", "group_concat"],
    },
    "subqueries": {"allowed": True},
}

# Aggregate names the gate recognizes (SQLite built-ins). min/max also have
# scalar forms; treating every call as an aggregate is conservative and the
# default policy allows them anyway.
_AGGREGATE_NAMES = frozenset(
    {"count", "sum", "avg", "min", "max", "total", "group_concat", "string_agg"}
)

_SELECT_RE = re.compile(r"\s*select\b", re.IGNORECASE)
# Parenthesized SELECTs, but also parenthesized CTEs and VALUES rows — SQLite
# executes `SELECT * FROM (WITH t AS (VALUES(1)) SELECT ...)` and
# `SELECT * FROM (VALUES(1),(2))` as subqueries just the same.
_SUBQUERY_RE = re.compile(r"\(\s*(select|with|values)\b", re.IGNORECASE)
_FUNC_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_LIMIT_RE = re.compile(r"\blimit\b", re.IGNORECASE)


class SqlPolicyError(RuntimeError):
    """The local policy gate rejected the statement (surfaced to the model)."""


@dataclass(frozen=True)
class LocalSqlPolicy:
    """Parsed local policy. Semantics mirror the cloud validator's shape:

    a policy that *omits* ``aggregations``/``subqueries`` REJECTS aggregates
    and subqueries — permissive behavior must be opted into (as
    :data:`DEFAULT_LOCAL_SQL_POLICY` does). Only when the spec carries no
    policy at all do the permissive defaults apply.
    """

    default_limit: int = 1000
    max_limit: int = 10000
    timeout_ms: int = 5000
    aggregations_allowed: bool = False
    # Lowercased allowlist; empty means "any aggregate" when aggregations are allowed.
    aggregation_functions: frozenset[str] = frozenset()
    subqueries_allowed: bool = False

    @classmethod
    def from_spec(cls, policy: dict[str, Any] | None) -> "LocalSqlPolicy":
        if policy is None:
            policy = DEFAULT_LOCAL_SQL_POLICY
        if not isinstance(policy, dict):
            raise ValueError("sql policy must be an object")

        dialect = str(policy.get("dialect") or "sqlite").lower()
        if dialect != "sqlite":
            raise ValueError(f"local sql source only supports the sqlite dialect, got {dialect!r}")
        # The local source is read-only, full stop. A policy asking for writes
        # is a configuration error, not something to quietly ignore.
        if policy.get("read_only") is False:
            raise ValueError(
                "local sql source is read-only; policy read_only=false is not supported"
            )

        limits = policy.get("limits")
        limits = limits if isinstance(limits, dict) else {}
        default_limit = int(limits.get("default_limit") or 1000)
        max_limit = int(limits.get("max_limit") or 10000)
        # NOT `or 5000`: 0 is a valid, explicitly-supported value (query()'s
        # own `if timeout_ms > 0` gate treats it as "no timer", and the
        # validation two lines below allows it) — `or` would silently coerce
        # an explicit 0 back to 5000 since 0 is falsy in Python.
        raw_timeout_ms = limits.get("timeout_ms")
        timeout_ms = 5000 if raw_timeout_ms is None else int(raw_timeout_ms)
        if default_limit < 1 or max_limit < 1:
            raise ValueError("sql policy limits must be positive")
        if timeout_ms < 0:
            raise ValueError("sql policy timeout_ms must be >= 0")

        aggregations = policy.get("aggregations")
        aggregations = aggregations if isinstance(aggregations, dict) else {}
        subqueries = policy.get("subqueries")
        subqueries = subqueries if isinstance(subqueries, dict) else {}

        functions = aggregations.get("functions") or []
        return cls(
            default_limit=default_limit,
            max_limit=max_limit,
            timeout_ms=timeout_ms,
            aggregations_allowed=bool(aggregations.get("allowed")),
            aggregation_functions=frozenset(str(f).lower() for f in functions),
            subqueries_allowed=bool(subqueries.get("allowed")),
        )


def _mask_quoted(sql: str) -> tuple[str, list[tuple[int, str]]]:
    """Blank out the *contents* of quoted regions so keyword scans never match
    inside them. Length-preserving.

    SQLite has four quoting forms: ``'...'`` string literals, and three quoted
    *identifier* forms — ``"..."``, ``[...]``, and ``` `...` ``` (doubled-quote
    escaping applies to all but brackets). All are masked, so an identifier
    named ``limit`` or ``select`` can never satisfy a keyword scan. But a
    quoted identifier can still *name a function* — SQLite accepts
    ``"count"(*)`` / ``[count](*)`` / ``` `count`(*) ``` as calls to count —
    so the unquoted, lowercased identifier names are returned alongside the
    masked text as ``(offset just past the closing delimiter, name)`` pairs
    for the aggregate gate to inspect.
    """
    out: list[str] = []
    identifiers: list[tuple[int, str]] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in ("'", '"', "`"):
            quote = ch
            out.append(quote)
            i += 1
            name_chars: list[str] = []
            closed = False
            while i < n:
                if sql[i] == quote:
                    # Doubled quote is an escaped quote inside the region.
                    if i + 1 < n and sql[i + 1] == quote:
                        out.append("  ")
                        name_chars.append(quote)
                        i += 2
                        continue
                    out.append(quote)
                    i += 1
                    closed = True
                    break
                out.append(" ")
                name_chars.append(sql[i])
                i += 1
            if not closed:
                raise SqlPolicyError("sql rejected by policy: unterminated quoted string")
            if quote != "'":  # '...' is a value; the rest are identifiers
                identifiers.append((i, "".join(name_chars).lower()))
        elif ch == "[":
            end = sql.find("]", i + 1)
            if end == -1:
                raise SqlPolicyError("sql rejected by policy: unterminated bracketed identifier")
            out.append("[")
            out.append(" " * (end - i - 1))
            out.append("]")
            identifiers.append((end + 1, sql[i + 1 : end].lower()))
            i = end + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out), identifiers


def _called_function_names(masked: str, identifiers: list[tuple[int, str]]) -> "list[str]":
    """All function names invoked in the statement, lowercased: bare-identifier
    calls found by regex over the masked text, plus quoted-identifier calls
    (a quoted identifier whose next non-space character is an open paren)."""
    names = [m.group(1).lower() for m in _FUNC_CALL_RE.finditer(masked)]
    for end, name in identifiers:
        rest = masked[end:].lstrip()
        if rest.startswith("("):
            names.append(name)
    return names


def _has_top_level_limit(masked: str) -> bool:
    for match in _LIMIT_RE.finditer(masked):
        prefix = masked[: match.start()]
        if prefix.count("(") - prefix.count(")") == 0:
            return True
    return False


def validate_local_sql(sql: str, policy: LocalSqlPolicy) -> str:
    """Run the local read-only gate; return the normalized statement.

    Mirrors the cloud validator's local-policy semantics as a guardrail (see
    the module threat model — this is not a security boundary): SELECT-only,
    single statement (semicolons and comments rejected outright), aggregate
    and subquery allowances from the policy, and a top-level LIMIT injected
    when the statement has none.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("query requires a non-empty SQL string")

    masked, quoted_identifiers = _mask_quoted(sql)

    if "--" in masked or "/*" in masked:
        raise SqlPolicyError("sql rejected by policy: comments are not allowed")
    if ";" in masked:
        raise SqlPolicyError(
            "sql rejected by policy: multiple statements are not allowed (remove semicolons)"
        )
    if not _SELECT_RE.match(masked):
        raise SqlPolicyError("sql rejected by policy: only SELECT statements are allowed")

    if not policy.subqueries_allowed and _SUBQUERY_RE.search(masked):
        raise SqlPolicyError("sql rejected by policy: subqueries are not allowed")

    for name in _called_function_names(masked, quoted_identifiers):
        if name not in _AGGREGATE_NAMES:
            continue
        if not policy.aggregations_allowed:
            raise SqlPolicyError("sql rejected by policy: aggregate functions are not allowed")
        if policy.aggregation_functions and name not in policy.aggregation_functions:
            raise SqlPolicyError(
                f"sql rejected by policy: aggregate function {name!r} is not allowed"
            )

    normalized = sql.strip()
    if not _has_top_level_limit(masked):
        normalized = f"{normalized} LIMIT {policy.default_limit}"
    return normalized


class LocalSqlDataSource:
    """Read-only SQL over a local SQLite file (or a host-supplied connection)."""

    def __init__(self, config: dict[str, Any], ctx: Any = None) -> None:
        self._name = str(config.get("name") or "db")
        raw_policy = config.get("policy")
        if raw_policy is not None and not isinstance(raw_policy, dict):
            raise ValueError("sql policy must be an object")
        self._policy = LocalSqlPolicy.from_spec(raw_policy)

        # Edge connection: an in-process host may hand a live sqlite3
        # connection via ctx; otherwise the spec supplies a path and we open
        # it read-only ourselves (mode=ro — the gate is not the only wall).
        self._conn: sqlite3.Connection | None = None
        if ctx is not None and isinstance(ctx, sqlite3.Connection):
            # A host-supplied connection may have been opened read-write —
            # silently accepting it would bypass the documented mode=ro
            # defense. Enforce read-only at the SQLite layer for the lifetime
            # of the connection (a deliberate, persistent side effect on the
            # host's handle): writes through it now fail with SQLITE_READONLY,
            # same as our own mode=ro open. Still a guardrail, not a security
            # boundary — code holding the raw handle can flip the pragma back.
            ctx.execute("PRAGMA query_only=ON")
            self._conn = ctx
        else:
            path = str(config.get("sqlite_path") or "").strip()
            if not path:
                raise ValueError("sql data source requires sqlite_path (or a live ctx connection)")
            self._sqlite_path = path
        self._lock = threading.Lock()
        self.queries_made = 0

    # -- DataSource protocol -------------------------------------------------

    def name(self) -> str:
        return self._name

    def capabilities(self) -> dict[str, bool]:
        return {
            "sql": True,
            "schema": True,
            "search": False,
            "get": False,
            "recent": False,
            "stats": False,
        }

    def get_schema(self) -> str:
        conn = self._connection()
        lines: list[str] = ["SQLite database. Tables:"]
        with self._lock:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            for (table,) in tables:
                cols = conn.execute(f"PRAGMA table_info({self._quote_ident(table)})").fetchall()
                col_desc = ", ".join(f"{c[1]} {c[2]}".strip() for c in cols)
                lines.append(f"- {table}({col_desc})")
        lines.append(
            'Query with query("SELECT ...") — read-only, single SELECT statement; '
            "rows return as list[dict] for you to compute over in Python."
        )
        return "\n".join(lines)

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Gate the statement through the local policy, then execute read-only."""
        normalized = validate_local_sql(sql, self._policy)
        conn = self._connection()
        timeout_ms = self._policy.timeout_ms
        max_rows = self._policy.max_limit

        # Hold the connection lock across the *whole* execute+fetch, and arm the
        # interrupt timer only inside it. sqlite3 connections are not
        # thread-safe and llm_batch fans out threads: if the timer were armed
        # before acquiring the lock, a queued query's timer could interrupt a
        # different query already running on the shared connection.
        with self._lock:
            self.queries_made += 1
            timer: threading.Timer | None = None
            if timeout_ms > 0:
                timer = threading.Timer(timeout_ms / 1000.0, conn.interrupt)
                timer.daemon = True
                try:
                    timer.start()
                except RuntimeError:
                    # Threadless platform (Pyodide/WASM: threading imports but
                    # thread *creation* raises "can't start new thread").
                    # There is no thread to arm the interrupt from, so the
                    # policy's timeout_ms cannot be enforced at this layer —
                    # the substrate's own wall-clock kill (e.g. the host
                    # process timeout in the Deno+Pyodide relay) is the
                    # enforcement there. Degrade to no timer instead of
                    # failing every query under the default policy. Only on
                    # platforms that are threadless BY DESIGN: on a normal
                    # threaded runtime the same RuntimeError means thread
                    # exhaustion, and silently dropping the timeout right
                    # when the process is resource-constrained would be the
                    # opposite of what the policy asks for — re-raise there.
                    if sys.platform not in ("emscripten", "wasi"):
                        raise
                    timer = None
                    warnings.warn(
                        "sql_local: threads unavailable on this platform "
                        f"({sys.platform}); the policy timeout ({timeout_ms} "
                        "ms) is not enforced — rely on the host's wall-clock "
                        "timeout",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            try:
                cursor = conn.execute(normalized)
                columns = [d[0] for d in cursor.description or []]
                # Result-row cap: the injected LIMIT bounds statements without
                # one, but an explicit LIMIT can exceed max_limit — fail fast
                # with guidance instead of silently truncating.
                rows = cursor.fetchmany(max_rows + 1)
                if len(rows) > max_rows:
                    raise SqlPolicyError(
                        f"sql rejected by policy: query returned more than {max_rows} rows "
                        f"(max_limit); add a tighter LIMIT or aggregate in SQL"
                    )
                return [dict(zip(columns, row)) for row in rows]
            except sqlite3.OperationalError as exc:
                if "interrupted" in str(exc).lower():
                    raise RuntimeError(
                        f"query exceeded the policy time limit ({timeout_ms} ms)"
                    ) from exc
                raise
            finally:
                if timer is not None:
                    timer.cancel()

    # -- internals -------------------------------------------------------------

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            # mode=ro: the database file is opened read-only at the SQLite
            # layer — even a statement that slipped past the gate cannot
            # mutate. check_same_thread=False + self._lock: llm_batch fans out
            # threads, sqlite connections are not thread-safe by default.
            self._conn = sqlite3.connect(
                f"file:{self._sqlite_path}?mode=ro", uri=True, check_same_thread=False
            )
        return self._conn

    @staticmethod
    def _quote_ident(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'


def local_sql_source_factory(config: dict[str, Any], ctx: Any = None) -> LocalSqlDataSource:
    """Option C factory: build a LocalSqlDataSource from a declarative spec + host ctx."""
    return LocalSqlDataSource(config, ctx)


def register() -> None:
    """Register the local SQL source as type ``"sql"`` with the runner.

    Call this from the runner *entrypoint* at startup (never from a request —
    requests stay declarative). Spec shape: ``{"type": "sql", "name": ...,
    "sqlite_path": ..., "policy": {...optional overrides...}}``.
    """
    register_source_type("sql", local_sql_source_factory, protocol=SOURCE_PROTOCOL_VERSION)
