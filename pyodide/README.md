# Deno + Pyodide RLM substrate

The Deno + Pyodide (CPython-on-WASM) substrate runs droste's RLM engine under
one Deno binary + an offline Pyodide runtime — no `Python.framework`, no
wheelhouse, no per-`.so` code signing. It started as a feasibility spike (see
History, below) and now ships as the default substrate inside a production
macOS `.app` bundle.

This substrate has two layers, and the split matters if you're adopting droste
for your own host:

- **The substrate itself** — droste-general, reusable, ModelRelay-shaped but
  host-agnostic: `relay.ts`, `event_channel.ts`, `broker.ts`, `events.ts`,
  `stream.ts`, and the
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

**Getting the Deno half:** the relay ships **inside the droste wheel** (#33)
as package data under `droste/substrates/_relay/` — `relay.ts`,
`event_channel.ts`, `stream.ts`, `broker.ts`, `events.ts`, `offline-probe.ts`,
and `deps.ts` (the single Pyodide version pin). Stage it in your build from the
installed package:

    relay_dir="$(droste relay-path)"
    cp "$relay_dir"/*.ts <your-build-staging>/

(Or `python -c 'from droste.substrates import relay_dir; print(relay_dir())'`.)
The wheel is already the one pinned, hash-verified artifact in your lockfile,
so relay and engine are version-locked by construction — no separate tarball
download, no sha pin to keep in sync. On an admitted run, the relay prefixes
the first canonical event with a `startup` event
(`{engine_version, runner_protocol, provider_protocol}`) so contract adoption
is a structured signal, not a changelog audit. Preflight and pre-admission
refusal leave the event descriptor empty. (Tagged releases
still attach `droste-relay-vX.Y.Z.tar.gz` as a convenience for non-Python
consumers; it is no longer the embedder path.)

In this repo the relay sources live at `src/droste/substrates/_relay/`; this
directory (`pyodide/`) keeps the substrate's tests, spikes, and this README.

## The substrate (droste-general)

Deno-side modules, all ModelRelay-aware but host-data-layer-agnostic:

- `relay.ts` — the Deno process a host spawns. Reads a request JSON on stdin,
  runs the RLM in Pyodide via a **host-supplied adapter module** (named on the
  command line), writes a response JSON on stdout, and requires a dedicated
  host-provided descriptor for Trace ABI NDJSON. Deno holds the only ambient
  capabilities (narrow net to ModelRelay plus trusted provider files); the
  Pyodide sandbox has none. Knows nothing about any host's data layer — see
  "Writing a host adapter".
- `event_channel.ts` — the single fail-closed writer for that host event
  descriptor. It does not define or transform event bodies.
- `broker.ts` — the A′-1 credential broker (see Security model).
- `events.ts` — the structured NDJSON event vocabulary + Pyodide-stderr
  forwarding filter.
- `stream.ts` — reconstructs a complete `/responses` reply from a ModelRelay SSE stream.

Python-side adapters, in **`droste.substrates.pyodide`** (part of the droste
package — installed by `pip install droste`, so a host that depends on droste
gets these for free, nothing to stage separately):

- `RawExecutor` — a plain-interpreter executor that replaces droste's
  `RestrictedExecutor`. The Deno/WASM jail *is* the sandbox here, so
  RestrictedPython would be redundant; `RawExecutor` is also Pyodide-safe (no
  signals, no threads).
- `EnvironmentConfig(kind="pyodide", ...)` + `create_environment(...)` — the
  supported host-wiring path. It selects `RawExecutor`, shares one immutable
  budget record with the loop execution context, rejects native signal
  timeouts, and requires the adapter to declare that its host owns both WASM
  isolation and the wall-clock deadline. The declarations are loud contract
  checks, not substitutes for the actual Deno boundary.
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
(`ProviderService`, `BridgeProvider`) — is likewise droste core; see
Security model. So is `droste.sources.sql_local` (`sqlite_provider()`,
a read-only SQLite provider), which the reference adapter uses — though note
its Pyodide timeout caveat under "Writing a host adapter" and Known gaps.

**One honest caveat on reuse:** `BridgedLLMClient._post` and `broker.ts`'s
`isModelRelayResponsesCall` scoping are ModelRelay-specific today. A host
targeting a different LLM backend would need to extend the broker (the URL/auth
scoping) and the client itself. That's real future work adjacent to #5
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
def build_db_service(db_path, contacts_db_path=None) -> tuple[ProviderService, dict]:
    ...

def run_for_host_pyodide(
    request, host_fetch, bridge_call, duplex_bridge_call=None, meta=None
) -> dict:
    ...
```

Inside `run_for_host_pyodide`, build one frozen config and use it for both
pieces of loop wiring:

```python
config = EnvironmentConfig(
    kind="pyodide",
    budget=Budget(
        tokens=token_budget,
        subcalls=subcall_budget,
        depth=depth_budget,
        wall_ms=wall_time_ms,
        root_output_tokens=root_output_tokens,
        subcall_output_tokens=subcall_output_tokens,
    ),
    sandbox=SandboxLimits(output_chars=max_output_chars, execution_timeout_ms=0),
    host_managed_timeout=True,
    host_managed_isolation=True,
)
execution_context = create_environment_context(config, on_event=emit_event)
environment = create_environment(
    config,
    context=data,
    registry=registry,
    subcalls=subcalls,
    execution_context=execution_context,
)
```

Do not instantiate the native `RunnerEnvironment` in a Pyodide adapter. That
silently selects the wrong executor and leaves substrate assumptions scattered
through host code.

- **`build_db_service(db_path, contacts_db_path=None)`** runs in the *trusted*
  DB-service interpreter (the one that boots first and actually holds the DB
  file). It builds a `droste.sources.bridge.ProviderService` wrapping one
  already-bound source, and returns `(service, meta)`. `meta` is a plain dict of
  whatever facts are only computable where the DB file is visible; it is
  **opaque to `relay.ts`** (below). The reference adapter has nothing
  host-specific to carry, so its `meta` is just `{}`; a production adapter
  might carry things like a filesystem-probe result or a default call budget
  computed from a `SELECT COUNT(*)`.
- **`run_for_host_pyodide(request, host_fetch, bridge_call, duplex_bridge_call=None, meta=None)`**
  runs in the *untrusted* REPL interpreter. It wires a `BridgedLLMClient` (over
  the injected `host_fetch`) and an environment into `droste.run_rlm`, and
  returns a response dict. A database-backed adapter requires a callable
  `bridge_call` and reaches the DB only through a
  `BridgeProvider(bridge_call, duplex_call=duplex_bridge_call)` registration
  with receiving-host effect policy — the DB never opens in this interpreter.
  The relay supplies its bounded bridge-v2 pump as `duplex_bridge_call`;
  custom hosts may omit it to retain unary provider-protocol-4 invocation.
  A context-only adapter may accept `bridge_call=None`, but it must not use a
  request path to open a provider inside the untrusted interpreter. `meta` is
  the same opaque blob `build_db_service` returned, ferried back verbatim.

In two-interpreter mode, a trusted host may send `SIGUSR1` to request
cooperative cancellation of the active provider call. The relay keys that
request to the invocation's `call_id` before the first frame and the receiving broker decides
the next frame acknowledgement. `SIGTERM` and `SIGKILL` remain hard-stop paths;
a handler that never reaches `check()`/`checkpoint()` still requires the host's
ordinary hard timeout.

### How the adapter module is selected (and why it's safe)

`relay.ts` takes the adapter module name as its **second CLI argument**:

```bash
DENO_EXTRA_STDIO_FDS=3 DROSTE_RELAY_EVENT_FD=3 \
  deno run --allow-net=api.modelrelay.ai --allow-read --allow-env \
  relay.ts <sources> <adapter_module> 3>events.ndjson
```

The file redirect is only a shell demonstration. A host using pipes must drain
fd2 and the event descriptor concurrently while it waits for the unary response
and process exit.

The host must open the named descriptor before launch; fd3 is the convention.
An external launcher must also include that descriptor in Deno's
`DENO_EXTRA_STDIO_FDS` startup marker. Deno consumes the marker before user
JavaScript starts and registers the inherited number for `node:fs`; merely
passing fd3 from a shell or Go `exec.Cmd.ExtraFiles` is not sufficient. Deno's
own `node:child_process` compatibility layer sets the marker automatically,
but production launchers must set it explicitly.

The relay rejects fd0, fd1, fd2, malformed values, and descriptors it cannot
write. It never falls back to fd2. A missing marker makes the otherwise-open
descriptor unavailable to the relay and produces the same fail-closed result.
A missing or unavailable channel returns one unary response with `error.type =
"RelayEventChannelError"` when stdout remains available, writes only an
allowlisted reason code to fd2, and exits before Pyodide work begins.

That transport failure body is relay-level and intentionally does not claim a
runner protocol or operation: descriptor validation occurs before the
host-selected adapter owns its response schema. Consumers branch on the closed
`RelayEventChannelError` type/code, not on adapter or runner fields.

The process has three independent output lanes:

| Descriptor | Contract |
|------------|----------|
| fd1 | Exactly one unary response JSON line. Adapter-owned responses use its HostResponse schema; pre-adapter event-channel failures use the relay-level error above. |
| configured event descriptor (fd3 by convention) | Canonical Trace ABI v2 NDJSON only. |
| fd2 | Diagnostics only; never parse or promote these bytes as events. |

Drain fd2 and the event descriptor concurrently. A hard cancellation or
process failure may end fd3 after a valid nonterminal prefix; the transport
does not invent a `done` event. Reconcile terminal state from the unary response
when one exists, otherwise report a transport failure. A descriptor can also
fail after part of a frame was written; treat a final unterminated or invalid
line as a typed transport failure and never promote it to a Trace event. An
empty preflight/refusal stream intentionally performs no peer-liveness write;
peer or access-mode loss is detected fail-closed on the first admitted-run
frame.

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
- `duplex_bridge_call` → `_duplex_bridge_call = duplex_bridge_call if callable(duplex_bridge_call) else None`
- `contacts_db_path` → `_contacts_db_path = svc_contacts_db_path if isinstance(svc_contacts_db_path, str) else None`

So the contract your adapter sees is a clean "real Python `None`, or a real
value." This is called out here only so nobody re-discovers and re-solves it in
their own adapter — the relay owns it.

### Gotcha 2 — `sql_local` timeouts under Pyodide (handled, but know the limit)

`LocalSqlRuntime.query()` enforces its per-query timeout with
`threading.Timer`, which needs real OS threads — unavailable under
Pyodide/WASM. Since the #8 fix, the source degrades gracefully there: the
timer is skipped with a one-time `RuntimeWarning`, and queries run normally
under the **default** policy. The thing to know is what that means: the
policy's `timeout_ms` is simply **not enforced** in this substrate — the
host's own wall-clock kill (Deno's process timeout) is the real enforcement,
exactly as the Pyodide environment factory requires for exec timeouts. Make
sure your host actually has one.

## Configuration flags

`relay.ts` reads three independent Deno environment variables at process start.
(These are `relay.ts`'s flags; a different host would define its own.)

| Flag | Default | Set it to... | ...to get |
|------|---------|---------------|-----------|
| `DENO_EXTRA_STDIO_FDS` | none (required for external launchers) | A comma-separated list that includes the event descriptor (`3` conventionally). | Registers extra inherited numeric stdio with Deno before relay code starts; Deno consumes this marker. |
| `DROSTE_RELAY_EVENT_FD` | none (required) | A decimal inherited descriptor of 3 or greater; fd3 is conventional. | The sole canonical Trace ABI NDJSON output lane. Missing, malformed, or unwritable descriptors fail closed. |
| `RLM_BRIDGE` | (unset) | `legacy` | Pre-A′-1 behavior: the ModelRelay credential is a visible global inside the untrusted REPL interpreter, which assembles its own auth header. Kill switch only — the split is one `host_fetch` call site catching a real design mistake, not a feature. |
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

**A′-2 — provider-service split (`droste.sources.bridge`, wired by `relay.ts` + the
host adapter's `build_db_service`).** The untrusted REPL interpreter never holds
the corpus DB either. A second, trusted Pyodide interpreter boots first, holds
a bound provider behind a `ProviderService` (a fixed operation allowlist from
the verified manifest — never a generic `getattr` on caller-controlled strings),
and the REPL interpreter reaches it only through a `BridgeProvider` registration.
`ProviderService` and `BridgeProvider` are droste-general
(`droste.sources.bridge`); the adapter's `build_db_service` is the host-specific
step that puts a real data source behind the service.
`bridge_source_integration_test.ts` proves the generic wire contract.
`filesystem_provider_integration_test.ts` mounts real SQLite and
`filesystem_text` sources in the trusted interpreter and proves the separate
REPL interpreter can read the configured text root only through generated
broker bindings; the root path and provider runtime are non-ambient there. And
`examples/pyodide-host/e2e_test.ts` proves the full path end to end against a
real SQLite file (see Testing).

The sandbox that runs LLM-generated code has no filesystem access to the
corpus, no network of its own (Pyodide has no sockets), and no live credential.
Its only channels to the outside world are the brokered provider call and the
brokered ModelRelay call, both host-mediated and host-scoped. Database paths
are consumed by the trusted interpreter and removed before the sandbox request
is constructed.

## Testing

Deno suite (the substrate's own tests, all in this directory):

```bash
cd pyodide
deno test --allow-read --allow-env --allow-ffi .
```

First run on a fresh machine also needs `--allow-net=cdn.jsdelivr.net` plus
`--allow-write` pointed at your Deno cache dir (`deno info` prints it): the
npm pyodide package unvendors stdlib `sqlite3`, so `loadPackage("sqlite3")`
fetches the wheel from the CDN once and writes it alongside the pyodide npm
install (neither flag needed thereafter — CI warms its cache the same way).

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
  `event_channel_test.ts`, `stream_test.ts` (Deno);
  `tests/test_pyodide_error_serialization.py` and
  `tests/test_sql_local.py` (Python) — the latter includes the #8
  regressions proving an explicit `timeout_ms: 0` survives (isn't defaulted back
  to 5000) and that a query then runs under it. `examples/pyodide-host/`
  `test_pyodide_host_adapter.py` is a fast *native* (no-Pyodide) sanity check of
  the reference adapter's brokered provider path, rejection of direct database
  access, and error serialization on a root LLM failure.
- **Real-Pyodide, substrate-only** (no host adapter, no host data layer):
  `broker_integration_test.ts`, `bridge_source_integration_test.ts`, and
  `filesystem_provider_integration_test.ts` load one or two
  real Pyodide interpreters with a scripted `host_fetch` and prove the broker /
  bridge wire contracts directly. The filesystem test uses the real first-party
  provider and proves its configured root is absent from the generated-code
  interpreter. A few seconds each.
- **Real-Pyodide against the real `relay.ts`** (droste's own E2E):
  `examples/pyodide-host/e2e_test.ts` spawns the actual `relay.ts` as a Deno
  subprocess — real process boundary, real Pyodide interpreter, real dynamic
  `importlib.import_module` of `pyodide_host_adapter`, real stdin/stdout
  request/response contract — against a local mock HTTP server standing in for
  ModelRelay. It builds a real temp SQLite fixture, asserts a real brokered
  `query()` round-trip (`SELECT COUNT(*)` → answer `"3"`), and proves a sibling
  file in the database directory is not visible to generated code. **Zero real
  network and zero sibling checkout**, so it runs unconditionally in CI — no
  skip. This replaces
  two former host-coupled tests (`db_service_integration_test.ts`,
  `broker_batch_integration_test.ts`), which required a sibling host-repo
  checkout and moved to that host's repo alongside its own adapter.

## Known gaps

- **Custom hosts that select only the unary provider bridge cannot deliver a
  new soft-cancellation request during a synchronous remote call.** Provider protocol 4 carries the
  cancellation snapshot and remaining deadline at dispatch, and returns a
  validated cumulative checkpoint at completion. A remote handler can poll
  `CapabilityExecutionContext.check()` and enforce that dispatched deadline,
  but the current request/response bridge has no reverse channel for a
  cancellation requested afterward and cannot stream mid-call checkpoints.
  The bundled two-interpreter Deno relay explicitly selects bridge v2 to add
  that channel. Custom unary-only hosts still require a wall-clock timeout and
  process termination as their hard Pyodide boundary.

- **`sql_local.py`'s per-query `timeout_ms` is unenforced under Pyodide (#8,
  fixed to degrade).** The timeout is a `threading.Timer`; where thread
  creation is unavailable the source now skips the timer with a
  `RuntimeWarning` instead of crashing on the first query, so the default
  policy works. But no in-interpreter timeout exists in that substrate — the
  host's wall-clock kill is the only enforcement (see "Writing a host
  adapter", Gotcha 2).
- **Typed server-side batch is a host extension today (#21).** droste's
  `LLMClient` protocol deliberately has no batch method (#6 removed the stale,
  never-called `batch_responses`): the loop parallelizes via
  `subcalls.llm_batch`, and how that is transported is each client's own
  implementation. The typed `batch_responses_typed(...) -> BatchResponse`
  contract against a real server-side batch endpoint lives on host adapters'
  own client subclasses; #21 designs its first-class home (one wire call on
  batch-capable gateway clients — no fallbacks, no capability sniffing in the
  loop).
- **Extract-fallback failure rate is unknown.** When `max_iterations` is
  exhausted without `answer["ready"]`, one more LLM call tries to synthesize a
  best-effort answer; a failure there now surfaces as a structured
  `extract_error` (result field + the Trace ABI v2 `extract` failure event) instead of
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
(A′-1 and A′-2, with A′-2 now the sole database-backed path) and the
adapter-agnostic split that made `relay.ts` fully adapter-agnostic and gave
droste its own example host — is in
this repo's git/PR history, not duplicated here. A few standalone investigation
scripts from that era (`spike_topology.ts`, `probe_dual_sqlite.ts`,
`verify_16_threading.ts`) still live in this directory as reference, outside
the `deno test` suite.
