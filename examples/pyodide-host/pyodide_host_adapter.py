"""Droste's own minimal Pyodide host adapter — reference + CI proof, not a
product.

This is what `pyodide/relay.ts` (droste's droste-general Deno entrypoint)
means by "a host-supplied adapter module": a small Python module, staged next
to the `droste` package into the sources directory relay.ts mounts, exposing
exactly two functions:

    build_db_service(db_path, contacts_db_path=None) -> (service, meta)
    run_for_host_pyodide(
        request, host_fetch, bridge_call, duplex_bridge_call=None, meta=None
    ) -> dict

`meta` is opaque to relay.ts — whatever `build_db_service` returns crosses the
process/interpreter boundary as JSON and comes back to `run_for_host_pyodide`
unexamined. This adapter's own `meta` is deliberately empty (`{}`): it has
nothing host-specific to carry (contrast a product host, which might carry a
`has_contacts`-shaped fact only computable where its data file is visible).

Everything here is droste-general: `droste.substrates.pyodide` for the
Pyodide-safe LLM client, `droste.sources.sql_local` for a read-only SQLite
data source, `droste.sources.bridge` for the untrusted/trusted interpreter
split, and `droste.create_environment` for substrate-specific REPL wiring.
No third-party or product-specific imports. A real host's production adapter
looks like this file with its own data source and result shape substituted in.

Subcalls (`llm_query`) are stubbed with `droste.testing.MockSubcallClient` —
this example's job is to prove the relay's adapter-loading mechanism end to
end (dynamic import, the DB-service bridge, the opaque meta blob, the
host_fetch wiring), not to demonstrate recursive subcalls.
"""

from __future__ import annotations

from typing import Any

from droste import (
    Budget,
    ConfiguredSource,
    EnvironmentConfig,
    ProviderCatalog,
    RLMConfig,
    RolloutConfiguration,
    SandboxLimits,
    SideEffect,
    create_environment,
    create_environment_context,
    run_rlm,
)
from droste.execution.progress import emit_event, emit_progress
from droste.sources.bridge import BridgeProvider, ProviderService
from droste.sources.sql_local import sqlite_provider
from droste.substrates.pyodide import BridgedLLMClient, HostFetch, serialize_error
from droste.testing import MockSubcallClient

# Default kept in sync with BridgedLLMClient's own default; overridable per
# request (request["base_url"]) so tests can point this adapter at a local
# mock server instead of the real ModelRelay endpoint.
_DEFAULT_BASE_URL = "https://api.modelrelay.ai/api/v1"


# The default SQL policy just works here: LocalSqlRuntime enforces its
# per-query timeout with threading.Timer, and where thread creation is
# unavailable (Pyodide/WASM) it degrades to no timer with a RuntimeWarning —
# the host's own wall-clock kill (Deno's process timeout) is the real
# enforcement in this substrate, exactly like the factory's Pyodide policy
# below for exec timeouts.
def _sql_source(db_path: str) -> ConfiguredSource:
    return ConfiguredSource("db", "sqlite", {"sqlite_path": db_path})


def _sql_registry(db_path: str):
    return ProviderCatalog((sqlite_provider(),)).bind(
        (_sql_source(db_path),),
        default_source_id="db",
    )


def build_db_service(
    db_path: str, contacts_db_path: str | None = None
) -> tuple[ProviderService, dict[str, Any]]:
    """Build the trusted-side `ProviderService` for the A'-2 split.

    `contacts_db_path` is accepted (matching the adapter contract's shape)
    but unused — this example has exactly one data source, a read-only SQL
    table. A real host with a second file to protect would build a second
    source here and describe it in `meta`.
    """
    source = _sql_registry(db_path).sources[0]
    service = ProviderService(source)
    return service, {}


def run_for_host_pyodide(
    request: dict[str, Any],
    host_fetch: HostFetch,
    bridge_call: Any,
    duplex_bridge_call: Any = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The Pyodide equivalent of an in-process `run_rlm` call.

    `bridge_call` is the A'-2 seam: the host wires a second, trusted Pyodide
    interpreter running a `ProviderService` built by `build_db_service` and
    passes its `bridge_call(method, params_json)` here. The database never
    opens inside this untrusted interpreter. This database-backed reference
    adapter intentionally has no direct-access fallback and is not a
    context-only adapter; providerless traffic should use a separate adapter
    module with no database bindings.
    `duplex_bridge_call` is the explicitly selected bridge-v2 message pump;
    omitting it keeps the same bridge provider on unary protocol 4.
    """
    if not callable(bridge_call):
        raise TypeError(
            "bridge_call must be callable; this database-backed adapter cannot "
            "serve providerless requests or open request paths directly"
        )

    # customer_token only when auth_type says so — BridgedLLMClient._auth_headers
    # prefers a customer token (bearer) over an api_key (X-ModelRelay-Api-Key)
    # whenever one is present, so an unconditional pass-through would pick the
    # wrong credential for an api_key-authenticated request that happens to
    # also carry a stale/irrelevant customer_token field.
    auth_type = request.get("auth_type", "api_key")
    customer_token = request.get("customer_token") if auth_type == "customer_token" else None
    root_reasoning_effort = request.get("root_reasoning_effort")
    if root_reasoning_effort is not None and (
        not isinstance(root_reasoning_effort, str) or not root_reasoning_effort
    ):
        raise ValueError("request.root_reasoning_effort must be a non-empty string or null")
    client = BridgedLLMClient(
        host_fetch,
        api_key=request.get("api_key"),
        customer_token=customer_token,
        base_url=request.get("base_url") or _DEFAULT_BASE_URL,
        reasoning_effort=root_reasoning_effort or "",
    )

    bridge = BridgeProvider(bridge_call, duplex_call=duplex_bridge_call)
    registration = bridge.registration(
        # Effects are deliberately supplied by this receiving host, not
        # trusted from transport metadata.
        effects={"query": SideEffect.READ, "schema": SideEffect.READ},
        policy_metadata={"query": {"read_only": True}},
    )
    registry = ProviderCatalog((registration,)).bind(
        (ConfiguredSource(bridge.source_id, bridge.manifest.provider_type),),
        default_source_id=bridge.source_id,
    )
    subcalls = MockSubcallClient()
    raw_budget = request.get("budget")
    budget = Budget.from_dict(raw_budget) if isinstance(raw_budget, dict) else Budget()
    sandbox = SandboxLimits(output_chars=8_000)
    environment_config = EnvironmentConfig(
        kind="pyodide",
        budget=budget,
        sandbox=sandbox,
        # The Deno/WASM process owns both boundaries; constructing a Pyodide
        # environment without these explicit declarations fails loudly.
        host_managed_timeout=True,
        host_managed_isolation=True,
    )
    exec_context = create_environment_context(
        environment_config,
        # Stderr NDJSON is how loop events reach the relay's forwarding
        # filter under Pyodide — attached explicitly (#35: no default sink).
        on_progress=emit_progress,
        on_event=emit_event,
    )
    environment = create_environment(
        environment_config,
        context=None,
        registry=registry,
        subcalls=subcalls,
        execution_context=exec_context,
        capability_run_id=exec_context.trace.run_id,
        capability_parent_run_id=exec_context.trace.parent_run_id,
        capability_observer=exec_context.observe_capability,
    )

    config = RLMConfig(
        budget=budget,
        sandbox=sandbox,
        root_model=request.get("root_model"),
        rollout=RolloutConfiguration(
            root_sampling=(
                {"reasoning_effort": root_reasoning_effort} if root_reasoning_effort else {}
            )
        ),
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
