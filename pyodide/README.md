# Deno + Pyodide RLM substrate

The Deno + Pyodide (CPython-on-WASM) substrate is how droste's RLM engine runs
inside Cozy's shipped `.app` bundle: one Deno binary + an offline Pyodide
runtime, no `Python.framework`, no wheelhouse, no per-`.so` code signing. It
started as a feasibility spike (see History, below) and is now the substrate
Cozy runs by default in production.

## Entrypoint

`relay.ts` is the process the Swift host spawns. Contract: read a `HostRequest`
JSON object on stdin, run the RLM in Pyodide, write a `HostResponse` JSON
object on stdout — the same contract the native PyO3 helper uses. It mounts
the bundled sources directory (droste + cozy's `rcl_rlm` data layer, staged by
cozy's own build) into Pyodide's `/app`, and the corpus DB directory
(`request.db_path`'s parent) into `/data`.

Supporting modules `relay.ts` imports directly:
- `broker.ts` — the A′-1 credential broker (see Security model).
- `events.ts` — the structured NDJSON event vocabulary + stderr forwarding filter.
- `stream.ts` — reconstructs a complete `/responses` reply from a ModelRelay SSE stream.
- `pyodide_runtime.py` — the Python-side adapters this substrate needs that the
  native substrate doesn't: `BridgedLLMClient` (an `LLMClient` that calls
  ModelRelay over `host_fetch` instead of a real socket), `build_db_service`
  (the trusted DB-service interpreter's setup), `run_for_host_pyodide` (the
  substrate's version of `rcl_rlm.host.run_for_host`).

## Configuration flags

All three are Deno environment variables read once at process start, each an
independent kill switch/bisect — the four combinations of `RLM_BRIDGE` ×
`RLM_DB_SERVICE` are all coherent, and `RLM_STREAM` is orthogonal to both.

| Flag | Default | Set it to... | ...to get |
|------|---------|---------------|-----------|
| `RLM_BRIDGE` | (unset) | `legacy` | Pre-A′-1 behavior: the ModelRelay credential is a visible global inside the untrusted REPL interpreter, which assembles its own auth header. Kill switch only — the split is one host_fetch call site catching a real design mistake, not a feature. |
| `RLM_DB_SERVICE` | on | `0` | Single-interpreter mode: the untrusted REPL interpreter mounts the DB directly (`db_path` in the sandbox request), instead of routing through the trusted DB-service interpreter over a bridge call. |
| `RLM_STREAM` | on | `0` | Legacy unary ModelRelay call (no SSE), for when NDJSON streaming from `/responses` is suspected of causing an issue. |

## Security model

Two orthogonal splits between untrusted (LLM-generated code) and trusted
(host-controlled) territory:

**A′-1 — credential broker (`broker.ts`).** The untrusted REPL interpreter
never holds the ModelRelay API key or customer token. `splitCredentials`
strips them from the request before it becomes a sandbox global; the broker
holds them host-side and `stripAndInjectAuth` overwrites whatever auth header
the sandbox's `host_fetch` call tried to set — scoped to the exact
`POST https://api.modelrelay.ai/api/v1/responses(/batch)` calls
(`isModelRelayResponsesCall`), never a general-purpose credential. Proven by
`broker_test.ts` (pure unit tests) and `broker_integration_test.ts` /
`broker_batch_integration_test.ts` (real Pyodide interpreter, scripted
`host_fetch`, both the sync and async code paths).

**A′-2 — DB-service split (`pyodide_runtime.py` + `droste/sources/bridge.py`).**
The untrusted REPL interpreter never holds the corpus DB either. A second,
trusted Pyodide interpreter boots first, holds a real `MessageDatabase` behind
a `DataSourceService` (a fixed method allowlist, gated by capabilities —
never a generic `getattr` on caller-controlled strings), and the REPL
interpreter reaches it only through a `BridgeDataSource` RPC call
(`droste.sources.bridge`). `bridge_source_integration_test.ts` proves the
generic wire contract; `db_service_integration_test.ts` proves the full path
against real `rcl_rlm` (not a stub) — a real `MessageDatabase`, generated
code's `query()` calls reaching it over the bridge, `retrieved_guids`
surviving via `extra_methods`, and the REPL interpreter never seeing the DB
file at all.

With `RLM_DB_SERVICE` at its default (on), the sandbox that runs LLM-generated
code has no filesystem access to the corpus at all, no network of its own
(Pyodide has no sockets), and no live credential — its only channels to the
outside world are the bridged data-source RPC and the brokered ModelRelay
call, both host-mediated and host-scoped.

**`RLM_DB_SERVICE=0` (legacy single-interpreter mode) does not have this
property** — see Known gaps.

## Testing

```bash
cd pyodide
deno test --allow-read --allow-env --allow-ffi .
```

`broker_test.ts` and `events_test.ts` are pure/unit (no Pyodide, no network).
Everything else loads a real Pyodide interpreter and takes a few seconds each.
`db_service_integration_test.ts` and `broker_batch_integration_test.ts` mount a
sibling cozy checkout's `tools/rcl-rlm/src` (`RCL_RLM_SRC` env override,
default `../../cozy/tools/rcl-rlm/src`) and are skipped (not failed) if it's
absent.

The Deno suite is where `pyodide_runtime.py`'s real logic (`BridgedLLMClient`,
`build_db_service`, `run_for_host_pyodide`) actually gets exercised, inside a
real Pyodide interpreter. `uv run pytest` from the repo root additionally
covers one small, substrate-independent piece —
`tests/test_pyodide_error_serialization.py`'s `_serialize_error` helper — but
that's a slice, not a substitute for the Deno suite.

## Known gaps

- **`RLM_DB_SERVICE=0` mounts the whole data directory into the untrusted
  interpreter, not just the DB (#79).** `relay.ts`'s legacy-mode branch
  (`py.mountNodeFS("/data", dbDir)`) mounts the corpus DB's *parent directory*
  wholesale — no narrowing to just `shadow.db{,-wal,-shm}` +
  `contacts.db{,-wal,-shm}`. (Pyodide's `mountNodeFS(path, hostPath)` has no
  read-only option at all, so narrowing to individual files would be the only
  available mitigation, short of copying just those files into a scratch
  directory first.) That directory also holds `config.json` / `sessions.json`
  (settings, dev keys, session state), so LLM-generated code running in this
  mode could read or write them directly via plain `open()` calls — `MessageDatabase`
  itself still opens the DB read-only (`file:{path}?mode=ro`), but that only
  protects the DB file, not the sibling files sharing the mount. This is
  exactly the risk the original spike flagged as a "do before shipping" item;
  it was never actually implemented — superseded instead by defaulting
  `RLM_DB_SERVICE` to on (the DB-service split removes the DB mount from the
  untrusted interpreter entirely, a strictly stronger fix), but the legacy
  path itself is still open. Not triggered by default; only reachable via the
  `RLM_DB_SERVICE=0` kill switch.
- **droste#74** — droste's own `LLMClient` protocol (`protocols/llm_client.py`)
  still declares a stale `batch_responses(requests) -> list[str]` method that
  droste's own core loop never calls (`subcalls.llm_batch` is the real path).
  `BridgedLLMClient` implements the *actually*-used
  `batch_responses_typed(requests, options=None) -> BatchResponse` contract
  (a cozy-`rcl_rlm`-specific extension, since only ModelRelay among droste's
  clients has a real server-side batch endpoint), but droste's public protocol
  hasn't caught up. Tracked, not yet resolved.
- **Extract-fallback failure rate is unknown.** When `max_iterations` is
  exhausted without `answer["ready"]`, one more LLM call tries to synthesize a
  best-effort answer; a failure there now surfaces as a structured
  `extract_error` (result field + `extract_error` NDJSON event) instead of
  silently falling back to raw loop output, but there's no data yet on how
  often that call actually fails or why. No retry has been added — that's a
  decision for once real failure data exists, not before.

## History

This substrate started as a feasibility spike: proving droste imports cleanly
under Pyodide, that `MessageDatabase` returns byte-identical results across
sqlite engines (native 3.53.1 vs Pyodide's bundled 3.39.0) against a real
300k+-message corpus, and that packaging a Deno binary + an offline
`--cached-only` `DENO_DIR` (~14MB) beats shipping a signed `Python.framework` +
wheelhouse per architecture. That work (and the two security hardening passes,
A′-1 and A′-2, that followed) is in this repo's git/PR history, not duplicated
here. A few standalone investigation scripts from that
era (`spike_topology.ts`, `probe_dual_sqlite.ts`, `verify_16_threading.ts`,
`score_pyodide_parity.py`) still live in this directory as reference, outside
the `deno test` suite.
