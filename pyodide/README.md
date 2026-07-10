# Deno + Pyodide RLM substrate

The Deno + Pyodide (CPython-on-WASM) substrate runs droste's RLM engine under
one Deno binary + an offline Pyodide runtime — no `Python.framework`, no
wheelhouse, no per-`.so` code signing. It started as a feasibility spike (see
History, below) and now ships as the default substrate inside a production
macOS `.app` bundle.

This substrate has two layers, and the split matters if you're adopting droste
for your own host:

- **The substrate itself** — droste-general, reusable, ModelRelay-shaped but
  host-agnostic: `relay.ts`, `broker.ts`, `events.ts`, `stream.ts`, and the
  Python adapters in `droste.substrates.pyodide` (which ship inside `pip install
  droste`). Plus the A′-1 / A′-2 security machinery, whose droste-side half
  (`droste.sources.bridge`) is also part of the installable package. **None of
  this knows anything about any particular host's data layer** — `relay.ts`
  included. It's fully adapter-agnostic: it takes the name of a host adapter
  module as a CLI argument and drives it through a fixed two-function contract
  (below), never referencing any specific host.
- **A host adapter** — one small Python module per host that wires the
  substrate up to that host's own data source and request/response shape. This
  is *not* part of the droste package; each host writes its own. Two exist:
  - `examples/pyodide-host/pyodide_host_adapter.py` — droste's own minimal
    reference adapter, built entirely from droste-general pieces, with zero
    third-party or product dependencies. It ships in this repo as the worked
    example and the CI proof that `relay.ts` actually works end to end.
    Copy the *shape* of this file, substituting your own data source and result
    format.
  - A production host's adapter — backed by the host's own data-layer package
    instead of `droste.sources.sql_local`. It lives in the host's repo, **not
    here** — droste's repo has no host-specific code anywhere under `pyodide/`.
    Such an adapter is analogous in shape to the example adapter, just with the
    host's own request contract and data layer.

If you're reading this to embed droste elsewhere: everything in "the substrate"
is yours to reuse as-is; write one adapter module of your own and point
`relay.ts` at it.

## The substrate (droste-general)

Deno-side modules, all ModelRelay-aware but host-data-layer-agnostic:

- `relay.ts` — the Deno process a host spawns. Reads a request JSON on stdin,
  runs the RLM in Pyodide via a **host-supplied adapter module** (named on the
  command line), writes a response JSON on stdout. Deno holds the only ambient
  capabilities (narrow net to ModelRelay + read of the DB dir); the Pyodide
  sandbox has none. Knows nothing about any host's data layer — see "Writing a
  host adapter".
- `broker.ts` — the A′-1 credential broker (see Security model).
- `events.ts` — the structured NDJSON event vocabulary + stderr forwarding filter.
- `stream.ts` — reconstructs a complete `/responses` reply from a ModelRelay SSE stream.

Python-side adapters, in **`droste.substrates.pyodide`** (part of the droste
package — installed by `pip install droste`, so a host that depends on droste
gets these for free, nothing to stage separately):

- `RawExecutor` — a plain-interpreter executor that replaces droste's
  `RestrictedExecutor`. The Deno/WASM jail *is* the sandbox here, so
  RestrictedPython would be redundant; `RawExecutor` is also Pyodide-safe (no
  signals, no threads).
- `BridgedLLMClient` — an `LLMClient` (droste's protocol) that calls ModelRelay
  over a host-injected `host_fetch` callable instead of a real socket
  (Pyodide has no sockets). Implements `responses_create`,
  `get_model_context_window`, `get_model_max_output_tokens`. Under Pyodide the
  host fetch is async; `_post` blocks the synchronous RLM loop on it via
  `run_sync`.
- `HostFetch` — the callable type alias for that injected fetch,
  `(method, url, headers_json, body) -> response_text`.
- `serialize_error` — makes `run_rlm`'s error *dataclass* JSON-serializable
  for a host's stdio wire contract (otherwise `json.dumps` raises on the
  dataclass and the host gets no output at all). Public API (no leading
  underscore) because adapter modules in other packages call it across the
  module boundary; it isn't a private helper anymore.

These have zero third-party deps and zero knowledge of any host's product
wiring — which is exactly what lets them import under Pyodide and be reused.

The A′-2 DB-service split's generic half — `droste.sources.bridge`
(`DataSourceService`, `BridgeDataSource`) — is likewise droste core; see
Security model. So is `droste.sources.sql_local` (`local_sql_source_factory`,
a read-only SQLite data source), which the reference adapter uses — though note
its Pyodide timeout caveat under "Writing a host adapter" and Known gaps.

**One honest caveat on reuse:** `BridgedLLMClient._post` and `broker.ts`'s
`isModelRelayResponsesCall` scoping are ModelRelay-specific today. A host
targeting a different LLM backend would need to extend the broker (the URL/auth
scoping) and the client itself. That's real future work adjacent to droste#68
(MCP-as-transport), not solved here.

## Writing a host adapter

To embed droste under this substrate, write one Python module exposing exactly
two functions, stage it into the sources directory next to the `droste`
package, and pass its module name to `relay.ts`. That's the whole contract.
`examples/pyodide-host/pyodide_host_adapter.py` is a complete, runnable
implementation of everything below (~150 lines, thoroughly commented); read it
alongside this section.

### The two-function contract

```python
def build_db_service(db_path, contacts_db_path=None) -> tuple[DataSourceService, dict]:
    ...

def run_for_host_pyodide(request, host_fetch, bridge_call=None, meta=None) -> dict:
    ...
```

- **`build_db_service(db_path, contacts_db_path=None)`** runs in the *trusted*
  DB-service interpreter (the one that boots first and actually holds the DB
  file). It builds a `droste.sources.bridge.DataSourceService` wrapping your
  real data source, and returns `(service, meta)`. `meta` is a plain dict of
  whatever facts are only computable where the DB file is visible; it is
  **opaque to `relay.ts`** (below). The reference adapter has nothing
  host-specific to carry, so its `meta` is just `{}`; a production adapter
  might carry things like a filesystem-probe result or a default call budget
  computed from a `SELECT COUNT(*)`.
- **`run_for_host_pyodide(request, host_fetch, bridge_call=None, meta=None)`**
  runs in the *untrusted* REPL interpreter. It wires a `BridgedLLMClient` (over
  the injected `host_fetch`) and an environment into `droste.run_rlm`, and
  returns a response dict. When `bridge_call` is non-`None` (DB-service mode,
  the default), it reaches the DB only through a `BridgeDataSource(bridge_call)`
  RPC — the DB never opens in this interpreter. When it's `None`
  (`RLM_DB_SERVICE=0`), the same function opens `request["db_path"]` directly in
  a single interpreter. `meta` is the same opaque blob `build_db_service`
  returned, ferried back verbatim.

### How the adapter module is selected (and why it's safe)

`relay.ts` takes the adapter module name as its **second CLI argument**:

```
deno run --allow-net=api.modelrelay.ai --allow-read --allow-env \
    relay.ts <sources> <adapter_module>
```

The name is validated against `^[A-Za-z_][A-Za-z0-9_.]*$`, then set as a Pyodide
Python global and resolved with `importlib.import_module` — never
string-interpolated into a Python source template. Crucially, the name comes
from `Deno.args` (the same trust class as `<sources>`: set by whatever spawns
the process — the control plane), **never from the request JSON on stdin** (the
data plane). This mirrors droste's native path, where `droste_runner.run()`'s
request-file entrypoint explicitly rejects an `adapter_module` in the request
body: code-selection is a trusted-caller decision, not something an LLM-facing
request may steer.

The module just needs to be importable inside Pyodide, so `relay.ts` stages it
into the same sources directory as the `droste` package (mounted at `/app`,
which is on `sys.path`) — `importlib.import_module("your_adapter_module")` then
resolves it like any other top-level module.

### The opaque `meta` blob

`meta` is the adapter's private channel between its two halves, which run in two
*different interpreters*. `relay.ts` never looks inside it: whatever
`build_db_service` returns is `json.dumps`'d in the trusted interpreter, carried
across the boundary, `json.loads`'d, and handed back to
`run_for_host_pyodide(..., meta=...)` completely unexamined. Only your adapter
(present in both interpreters) knows or cares what's in it. Put anything
JSON-serializable there that the trusted side computes and the untrusted side
needs. (An earlier version of `relay.ts` hardcoded one host's field names into
its own Python template; making `meta` opaque is what removed the last
host-specific knowledge from the relay.)

### Gotcha 1 — JsNull normalization (already handled for you)

A JS `null` passed across the Pyodide FFI via `globals.set(name, null)` does
**not** arrive as Python's `None`. It arrives as a `JsProxy`-wrapped `JsNull`
sentinel, which passes an `is not None` check and then blows up the first time
an adapter tries to use it (e.g. `'JsNull' object is not callable`). **You do
not need to defend against this** — `relay.ts` normalizes it inside its own
embedded Python template, at both boundary crossings, before your adapter is
called:

- `bridge_call` → `_bridge_call = bridge_call if callable(bridge_call) else None`
- `contacts_db_path` → `_contacts_db_path = svc_contacts_db_path if isinstance(svc_contacts_db_path, str) else None`

So the contract your adapter sees is a clean "real Python `None`, or a real
value." This is called out here only so nobody re-discovers and re-solves it in
their own adapter — the relay owns it.

### Gotcha 2 — `sql_local` timeouts under Pyodide (your responsibility)

If your adapter uses `droste.sources.sql_local`, you **must** set the policy's
`timeout_ms` to `0`. `LocalSqlDataSource.query()` enforces its per-query timeout
with `threading.Timer`, which needs real OS threads — unavailable under
Pyodide/WASM — so any query with a nonzero `timeout_ms` raises
`RuntimeError: can't start new thread` on the very first call. `timeout_ms: 0`
skips the timer branch entirely (the code treats `0` as "no timer"); the host's
own wall-clock kill (Deno's process timeout) is the real enforcement in this
substrate, exactly as `RunnerEnvironment(exec_timeout_ms=0, ...)` already is for
exec timeouts. The reference adapter does this explicitly via its
`_PYODIDE_SAFE_SQL_POLICY` constant. This is tracked as droste#82 (still open —
until it's fixed in `sql_local.py` itself, `timeout_ms: 0` is the required
opt-out for any Pyodide host using that source). See Known gaps.

## Configuration flags

`relay.ts` reads three Deno environment variables once at process start, each
an independent kill switch / bisect — the four combinations of `RLM_BRIDGE` ×
`RLM_DB_SERVICE` are all coherent, and `RLM_STREAM` is orthogonal to both.
(These are `relay.ts`'s flags; a different host would define its own.)

| Flag | Default | Set it to... | ...to get |
|------|---------|---------------|-----------|
| `RLM_BRIDGE` | (unset) | `legacy` | Pre-A′-1 behavior: the ModelRelay credential is a visible global inside the untrusted REPL interpreter, which assembles its own auth header. Kill switch only — the split is one `host_fetch` call site catching a real design mistake, not a feature. |
| `RLM_DB_SERVICE` | on | `0` | Single-interpreter mode: the untrusted REPL interpreter mounts the DB directly (`db_path` in the sandbox request), instead of routing through the trusted DB-service interpreter over a bridge call. |
| `RLM_STREAM` | on | `0` | Legacy unary ModelRelay call (no SSE), for when NDJSON streaming from `/responses` is suspected of causing an issue. |

## Security model

Two orthogonal splits between untrusted (LLM-generated code) and trusted
(host-controlled) territory. The mechanisms live in the substrate; `relay.ts`
wires them and the host adapter fills in the host-specific pieces.

**A′-1 — credential broker (`broker.ts`).** The untrusted REPL interpreter
never holds the ModelRelay API key or customer token. `splitCredentials`
strips them from the request before it becomes a sandbox global; the broker
holds them host-side and `stripAndInjectAuth` overwrites whatever auth header
the sandbox's `host_fetch` call tried to set — scoped to the exact
`POST https://api.modelrelay.ai/api/v1/responses(/batch)` calls
(`isModelRelayResponsesCall`), never a general-purpose credential. Proven by
`broker_test.ts` (pure unit tests) and `broker_integration_test.ts` (real
Pyodide interpreter, a scripted *sync* `host_fetch`). The async branch —
`relay.ts`'s real `host_fetch` is `async`, so `BridgedLLMClient._post` must
`run_sync` the awaitable — isn't exercised by a dedicated in-repo broker test
anymore (that was `broker_batch_integration_test.ts`, which moved out to the
host repo it depended on);
it's still covered for real by `examples/pyodide-host/e2e_test.ts`, which
spawns the actual `relay.ts` subprocess with its actual async `host_fetch`.

**A′-2 — DB-service split (`droste.sources.bridge`, wired by `relay.ts` + the
host adapter's `build_db_service`).** The untrusted REPL interpreter never holds
the corpus DB either. A second, trusted Pyodide interpreter boots first, holds
a real data source behind a `DataSourceService` (a fixed method allowlist, gated
by capabilities — never a generic `getattr` on caller-controlled strings), and
the REPL interpreter reaches it only through a `BridgeDataSource` RPC call.
`DataSourceService` and `BridgeDataSource` are droste-general
(`droste.sources.bridge`); the adapter's `build_db_service` is the host-specific
step that puts a real data source behind the service.
`bridge_source_integration_test.ts` proves the generic wire contract, and
`examples/pyodide-host/e2e_test.ts` proves the full path end to end against a
real SQLite file (see Testing).

With `RLM_DB_SERVICE` at its default (on), the sandbox that runs LLM-generated
code has no filesystem access to the corpus at all, no network of its own
(Pyodide has no sockets), and no live credential — its only channels to the
outside world are the bridged data-source RPC and the brokered ModelRelay
call, both host-mediated and host-scoped.

**`RLM_DB_SERVICE=0` (legacy single-interpreter mode) does not have this
property** — see Known gaps.

## Testing

Deno suite (the substrate's own tests, all in this directory):

```bash
cd pyodide
deno test --allow-read --allow-env --allow-ffi .
```

droste's end-to-end proof of the whole relay + adapter path (in `examples/`,
needs `--allow-run` to spawn the relay subprocess, `--allow-write` for its temp
sources/DB dirs, and loopback net for its mock ModelRelay server):

```bash
deno test --allow-run --allow-read --allow-write --allow-net=127.0.0.1 --allow-env \
    examples/pyodide-host/e2e_test.ts
```

Python tests (from the repo root, auto-discovered — no special config):

```bash
uv run pytest
```

Roughly three tiers of coverage:

- **Pure / unit** (no Pyodide, no network): `broker_test.ts`, `events_test.ts`,
  `stream_test.ts` (Deno); `tests/test_pyodide_error_serialization.py` and
  `tests/test_sql_local.py` (Python) — the latter includes the droste#82
  regressions proving an explicit `timeout_ms: 0` survives (isn't defaulted back
  to 5000) and that a query then runs under it. `examples/pyodide-host/`
  `test_pyodide_host_adapter.py` is a fast *native* (no-Pyodide) sanity check of
  the reference adapter's three code paths: single-interpreter mode, DB-service
  bridge mode, and error serialization on a root LLM failure.
- **Real-Pyodide, substrate-only** (no host adapter, no host data layer):
  `broker_integration_test.ts` and `bridge_source_integration_test.ts` load a
  real Pyodide interpreter with a scripted `host_fetch` and prove the broker /
  bridge wire contracts directly. A few seconds each.
- **Real-Pyodide against the real `relay.ts`** (droste's own E2E):
  `examples/pyodide-host/e2e_test.ts` spawns the actual `relay.ts` as a Deno
  subprocess — real process boundary, real Pyodide interpreter, real dynamic
  `importlib.import_module` of `pyodide_host_adapter`, real stdin/stdout
  request/response contract — against a local mock HTTP server standing in for
  ModelRelay. It builds a real temp SQLite fixture and asserts a real `query()`
  round-trip (`SELECT COUNT(*)` → answer `"3"`), in both DB-service (default)
  and `RLM_DB_SERVICE=0` single-interpreter modes. **Zero real network and zero
  sibling checkout**, so it runs unconditionally in CI — no skip. This replaces
  two former host-coupled tests (`db_service_integration_test.ts`,
  `broker_batch_integration_test.ts`), which required a sibling host-repo
  checkout and moved to that host's repo alongside its own adapter.

## Known gaps

- **`RLM_DB_SERVICE=0` mounts the whole data directory into the untrusted
  interpreter, not just the DB (#79).** `relay.ts`'s legacy-mode branch
  (`py.mountNodeFS("/data", dbDir)`) mounts the corpus DB's *parent directory*
  wholesale — no narrowing to just the DB files. (Pyodide's
  `mountNodeFS(path, hostPath)` has no read-only option at all, so narrowing to
  individual files, or copying just those files into a scratch directory first,
  would be the only available mitigation.) If that directory also holds
  application state or config, LLM-generated code running in this mode could
  read or write those siblings directly via plain `open()` calls — an opened DB
  file's own read-only mode protects only the DB, not the sibling files sharing
  the mount. This is exactly the risk the original spike flagged as a "do before
  shipping" item; it was never actually implemented — superseded instead by
  defaulting `RLM_DB_SERVICE` to on (the DB-service split removes the DB mount
  from the untrusted interpreter entirely, a strictly stronger fix), but the
  legacy path itself is still open. Not triggered by default; only reachable via
  the `RLM_DB_SERVICE=0` kill switch. This is a `relay.ts` gap in its own
  legacy mode, not a droste-package one.
- **`sql_local.py`'s per-query timeout is incompatible with Pyodide (#82,
  open).** `LocalSqlDataSource.query()` enforces `timeout_ms` with
  `threading.Timer`, which needs real OS threads — so under Pyodide any nonzero
  `timeout_ms` (including the default policy's `5000`) raises
  `RuntimeError: can't start new thread` on the first query. A Pyodide host must
  opt out with `timeout_ms: 0` (see "Writing a host adapter", Gotcha 2). The
  branch fix that made an explicit `0` actually stick (it used to be silently
  coerced back to `5000` by an `or` idiom) has landed; #82 tracks the remaining
  work of properly hardening / documenting the threadless-runtime path in
  `sql_local.py` itself so the *default* policy doesn't break under Pyodide.
- **droste#74** — droste's own `LLMClient` protocol (`protocols/llm_client.py`)
  still declares a stale `batch_responses(requests) -> list[str]` method that
  droste's own core loop never calls (`subcalls.llm_batch` is the real path).
  The actually-used batch contract, `batch_responses_typed(...) -> BatchResponse`,
  is a ModelRelay-specific extension (only ModelRelay, among droste's clients,
  has a real server-side batch endpoint). As of the droste#80 split it no longer
  lives on the substrate's shared `BridgedLLMClient` at all — it belongs on a
  host adapter's own client subclass, in the host's repo. That resolves the
  worse half of the awkwardness: the batch method is a genuine host extension,
  not a host-specific method squatting on droste's general client. What remains
  is only that droste's *public* protocol still advertises the stale
  `batch_responses` it never uses. Tracked, not yet resolved.
- **Extract-fallback failure rate is unknown.** When `max_iterations` is
  exhausted without `answer["ready"]`, one more LLM call tries to synthesize a
  best-effort answer; a failure there now surfaces as a structured
  `extract_error` (result field + `extract_error` NDJSON event) instead of
  silently falling back to raw loop output, but there's no data yet on how
  often that call actually fails or why. No retry has been added — that's a
  decision for once real failure data exists, not before.

## History

This substrate started as a feasibility spike: proving droste imports cleanly
under Pyodide, that a real corpus returns byte-identical results across sqlite
engines (native 3.53.1 vs Pyodide's bundled 3.39.0) against a large real-world
benchmark corpus, and that packaging a Deno binary + an offline
`--cached-only` `DENO_DIR` (~14MB) beats shipping a signed `Python.framework` +
wheelhouse per architecture. That work — plus the two security hardening passes
(A′-1 and A′-2, the latter now on by default) and the droste#80 split that made
`relay.ts` fully adapter-agnostic and gave droste its own example host — is in
this repo's git/PR history, not duplicated here. A few standalone investigation
scripts from that era (`spike_topology.ts`, `probe_dual_sqlite.ts`,
`verify_16_threading.ts`) still live in this directory as reference, outside
the `deno test` suite.
