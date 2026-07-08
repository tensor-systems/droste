"""Droste's own minimal Pyodide host adapter — reference + CI proof, not a
product.

This is what `pyodide/relay.ts` (droste's droste-general Deno entrypoint)
means by "a host-supplied adapter module": a small Python module, staged next
to the `droste` package into the sources directory relay.ts mounts, exposing
exactly two functions:

    build_db_service(db_path, contacts_db_path=None) -> (service, meta)
    run_for_host_pyodide(request, host_fetch, bridge_call=None, meta=None) -> dict

`meta` is opaque to relay.ts — whatever `build_db_service` returns crosses the
process/interpreter boundary as JSON and comes back to `run_for_host_pyodide`
unexamined. This adapter's own `meta` is deliberately empty (`{}`): it has
nothing host-specific to carry (contrast a product host, which might carry a
`has_contacts`-shaped fact only computable where its data file is visible).

Everything here is droste-general: `droste.substrates.pyodide` for the
Pyodide-safe LLM client, `droste.sources.sql_local` for a read-only SQLite
data source, `droste.sources.bridge` for the untrusted/trusted interpreter
split, `droste_runner.runner.RunnerEnvironment` for the same in-process REPL
environment droste's own CLI (`droste_cli.main.run_ask`) and `droste_runner`
(ModelRelay's hosted runner) already use. No third-party or product-specific
imports. A real host (e.g. Cozy's `rcl_rlm.pyodide_adapter`) looks like this
file with its own data source and result shape substituted in.

Subcalls (`llm_query`) are stubbed with `droste.testing.MockSubcallClient` —
this example's job is to prove the relay's adapter-loading mechanism end to
end (dynamic import, the DB-service bridge, the opaque meta blob, the
host_fetch wiring), not to demonstrate recursive subcalls.
"""

from __future__ import annotations

from typing import Any

from droste import DataSourceRegistry, RLMConfig, run_rlm
from droste.execution.context import create_execution_context
from droste.sources.bridge import BridgeDataSource, DataSourceService
from droste.sources.sql_local import local_sql_source_factory
from droste.substrates.pyodide import BridgedLLMClient, HostFetch, serialize_error
from droste.testing import MockSubcallClient
from droste_runner.runner import RunnerEnvironment

# Default kept in sync with BridgedLLMClient's own default; overridable per
# request (request["base_url"]) so tests can point this adapter at a local
# mock server instead of the real ModelRelay endpoint.
_DEFAULT_BASE_URL = "https://api.modelrelay.ai/api/v1"

# LocalSqlDataSource.query()'s default policy enforces its per-query timeout
# with threading.Timer, which needs real OS threads — unavailable under
# Pyodide/WASM ("RuntimeError: can't start new thread" on the very first
# query). timeout_ms=0 skips that branch entirely; the host's own wall-clock
# kill (Deno's process timeout) is the real timeout enforcement here, exactly
# like RunnerEnvironment(exec_timeout_ms=0, ...) below for exec timeouts.
# Tracked as droste#82 — this is a real gap in sql_local.py's Pyodide
# compatibility, not a permanent design choice.
_PYODIDE_SAFE_SQL_POLICY = {
    "dialect": "sqlite",
    "read_only": True,
    "limits": {"default_limit": 1000, "max_limit": 10000, "timeout_ms": 0},
    "aggregations": {
        "allowed": True,
        "functions": ["count", "sum", "avg", "min", "max", "total", "group_concat"],
    },
    "subqueries": {"allowed": True},
}


def _sql_source(db_path: str) -> Any:
    return local_sql_source_factory({"name": "db", "sqlite_path": db_path, "policy": _PYODIDE_SAFE_SQL_POLICY})


def build_db_service(
    db_path: str, contacts_db_path: str | None = None
) -> tuple[DataSourceService, dict[str, Any]]:
    """Build the trusted-side `DataSourceService` for the A'-2 split.

    `contacts_db_path` is accepted (matching the adapter contract's shape)
    but unused — this example has exactly one data source, a read-only SQL
    table. A real host with a second file to protect would build a second
    source here and describe it in `meta`.
    """
    source = _sql_source(db_path)
    service = DataSourceService(source)
    return service, {}


def run_for_host_pyodide(
    request: dict[str, Any],
    host_fetch: HostFetch,
    bridge_call: Any = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The Pyodide equivalent of an in-process `run_rlm` call.

    `bridge_call` is the droste#3/A'-2 seam: when the host wires a second,
    trusted Pyodide interpreter running a `DataSourceService` built by
    `build_db_service`, it passes the resulting `bridge_call(method,
    params_json)` here instead of a raw `db_path`, and the DB never opens
    inside this (untrusted) interpreter. `bridge_call` is `None` when the
    host opts out (relay.ts: `RLM_DB_SERVICE=0`) — the single-interpreter,
    `db_path`-in-sandbox behavior below is unchanged in that case.
    """
    # customer_token only when auth_type says so — BridgedLLMClient._auth_headers
    # prefers a customer token (bearer) over an api_key (X-ModelRelay-Api-Key)
    # whenever one is present, so an unconditional pass-through would pick the
    # wrong credential for an api_key-authenticated request that happens to
    # also carry a stale/irrelevant customer_token field.
    auth_type = request.get("auth_type", "api_key")
    customer_token = request.get("customer_token") if auth_type == "customer_token" else None
    client = BridgedLLMClient(
        host_fetch,
        api_key=request.get("api_key"),
        customer_token=customer_token,
        base_url=request.get("base_url") or _DEFAULT_BASE_URL,
    )

    if bridge_call is not None:
        source = BridgeDataSource(bridge_call, name="db")
    else:
        source = _sql_source(request["db_path"])

    registry = DataSourceRegistry([source], default_source_name=source.name())
    subcalls = MockSubcallClient()
    exec_context = create_execution_context(
        max_depth=int(request.get("max_depth") or 1),
        max_calls=int(request.get("max_calls") or 0),
        max_iterations=int(request.get("max_iterations") or 8),
        max_output_chars=int(request.get("max_output_chars") or 8000),
    )
    environment = RunnerEnvironment(
        context=None,
        registry=registry,
        subcalls=subcalls,
        max_output_chars=int(request.get("max_output_chars") or 8000),
        exec_timeout_ms=0,  # host (Deno) enforces the wall-clock kill, not this interpreter
    )

    config = RLMConfig(
        max_iterations=int(request.get("max_iterations") or 8),
        max_depth=int(request.get("max_depth") or 1),
        max_calls=int(request.get("max_calls") or 0),
        max_output_chars=int(request.get("max_output_chars") or 8000),
        root_model=request.get("root_model"),
    )

    res = run_rlm(
        request["question"],
        environment=environment,
        root_llm=client,
        subcalls=subcalls,
        config=config,
        context=exec_context,
    )
    return {
        "answer": res.answer,
        "sub_calls_made": res.sub_calls_made,
        "total_tokens": res.tokens_used,
        "iterations": res.iterations,
        "error": serialize_error(res.error),
        "extracted": bool(res.extracted),
        "extract_error": serialize_error(res.extract_error),
    }
