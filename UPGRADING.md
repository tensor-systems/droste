# Upgrading droste

Embedder-facing notes for moving between engine versions: what changed at the
integration surface, what breaks loudly, and — more importantly — what would
degrade *silently* if you skip a step. Loud breaks announce themselves (the
runner refuses mismatched `protocol_version` requests with a structured error;
the relay's `startup` event reports the engine + protocol versions it speaks);
this file exists for everything that doesn't.

Ordered newest first. "Embedder" means anything that builds on the engine
beyond the `droste` CLI: hosts calling `run_rlm` in-process, `droste_runner`
consumers, and Pyodide-substrate integrations staging the Deno relay.

## Unreleased (post-0.10.6)

### Trajectory execution status is explicit

`IterationRecord` and each built-in runner trajectory entry now include the
additive string field `execution_status`, currently `"success"` or `"error"`.
Use it instead of interpreting the text in `execution_result`: successful
stdout may legitimately begin with `ERROR:`. The existing `execution_result`
field and runner protocol version are unchanged.

### ModelRelay root request accounting

`ModelRelayClient.root_requests_issued` exposes a thread-safe cumulative count
of HTTP root requests dispatched by that client, including repair, extraction,
streaming, and failed requests. Payload validation and request-construction
failures before dispatch do not increment the count. This additive client-level
metric does not change `RLMResult` or the runner protocol.

## 0.10.6 (from 0.10.5)

### Confirmed answers can carry structured metadata

Sandbox code may set a JSON object at `answer["metadata"]`. A confirmed
result exposes a validated, defensively copied version as
`RLMResult.answer_metadata`; the built-in runner and CLI JSON output include
the same additive `answer_metadata` field. Metadata is limited to 64 KiB.

`answer["metadata"]` is now reserved and validated whenever an answer claims
readiness, even when contract enforcement is disabled. Invalid metadata blocks
confirmation and is returned to the model for repair. Emit only plain JSON
objects, arrays, strings, booleans, null, finite numbers, and integers within
JavaScript's safe range; tuples, custom scalar types, non-string object keys,
cycles, excessive nesting, and oversized structures are rejected.

The text-only terminal extraction fallback intentionally returns empty
metadata: partial structured values are not evidence for newly synthesized
text. Embedders that construct `RLMResult` directly need no change because the
new field defaults to an empty object.

### Successful semantic evidence and structured subcalls

`ExecutionStats` now distinguishes attempted `calls_made` from
`successful_calls`. Subcall clients must reserve attempted calls before
dispatch as before, and increment successful calls only for items that return
usable text. Semantic policy hints now require successful evidence; an
all-failed batch no longer satisfies the ready gate. `RLMResult`, CLI JSON, and
runner protocol responses expose the successful count as
`sub_calls_succeeded` / `successful_subcalls` while preserving existing
attempted-call fields.

Sandboxes now expose `llm_batch_json` (also `llm_query_batched_json`) for
opt-in, locally validated structured output. It supports a documented
deterministic JSON-schema subset, caller validators, ordered per-item errors,
and malformed-only bounded repair. Provider errors are attributable and are
never converted into parse retries. ModelRelay clients continue to use one
native batch request for each initial or repair batch.

Subcall call-budget rejection now raises `SubcallBudgetExceeded`, a dedicated
`RuntimeError` subclass. Existing `except RuntimeError` handlers remain
compatible, while structured callers can distinguish budget exhaustion from
provider failures without matching error text. For third-party clients from
before this type existed, `structured_batch` recognizes only the exact legacy
`RuntimeError("max subcalls exceeded")` form. Structured batch `errors` is
authoritative for item failure: a valid JSON `null` and a failed value slot are
both represented as JSON-serializable `None` in `values`.

### Semantic policy is enforced when confirming an answer

With `PolicyHints(semantic=True)`, inspection and local aggregation blocks may
now execute before a semantic subcall. The ready-time gate still refuses to
confirm an answer until at least one `llm_query` or batched equivalent succeeds.
Hosts get the same final-answer contract without forcing harmless preparation
steps through policy-repair iterations.

## 0.10.5 (from 0.10.4)

The Pyodide credential broker now recognizes an optional exact
`data_source_endpoint` as a short-lived runner callback. The held runner token
is injected only for that exact URL, matching the root/subcall callback rules;
data-source bearer credentials remain in the trusted host.

## 0.10.4 (from 0.10.3)

### ModelRelay batches use the synchronous batch endpoint

`ModelRelaySubcallClient.llm_batch` now sends one `POST /responses/batch`
request instead of spawning per-item worker threads. Hosts that mock ModelRelay
must implement the typed batch response envelope (`results[].id/status/response/error`).
There is deliberately no runtime fallback to individual requests.

### Pyodide relay supports hosted runners without a database

The Deno relay no longer assumes every host request has `db_path`. Hosts may
stage a pure context/inference adapter with no DB service. A short-lived
`token` in the request is stripped before the sandbox starts and is injected
only for exact `root_endpoint`, `subcall_endpoint`, and
`subcall_batch_endpoint` callback URLs.

## 0.10.3 (from 0.10.2)

### Failed-only trajectories are not extraction evidence

Failed-only trajectories with no retained draft are not extraction evidence;
their terminal error remains fatal. Because no extract call is attempted,
`extract_error` remains `None` on this path.

## 0.10.2 (from 0.10.1)

### Terminal failures can recover a typed best-effort answer

When a terminal execution error, exhausted subcall budget, or unresolved
ready-policy hint leaves usable partial work, `run_rlm` now gives the existing
extract fallback one bounded chance to synthesize an answer. On success,
`RLMResult.extracted` is `True`, fatal `error` is `None`, and the superseded
step error is preserved as `recovered_error`. The runner and CLI JSON surfaces
include the same additive `recovered_error` object. Hosts should present these
answers as unconfirmed and may use `recovered_error.type` for telemetry.

Any failed execution whose repair also fails (or returns no code) is now
retained in `trajectory`, including on mid-run iterations that later recover.
Consumers must tolerate duplicate iteration numbers and `execution_result`
values beginning with `ERROR:` even when the final run is ready.

The exact extract response `unable to determine from the work so far` is not a
successful recovery: it produces `extract_error.type == "InsufficientEvidence"`
and retains the fatal terminal error.

Any failed sandbox execution now revokes `answer["ready"]`, including when the
block set readiness or rebound the answer dict before raising. Such a run may
consume another root iteration instead of returning `ready=True` alongside a
fatal execution error; retained `answer["content"]` remains available to repair
or extraction.

## 0.10.1 (from 0.10.0)

### Event emission is now opt-in — attach sinks or loop events go silent

`run_rlm` no longer prints NDJSON events to stderr by default (#35). If you
relied on the default stderr stream — for example to feed the relay's event
forwarder or a "watch it think" UI — attach the sinks explicitly:

```python
from droste.execution.progress import emit_event, emit_progress

run_rlm(..., on_progress=emit_progress, on_event=emit_event)
```

If you build your own `ExecutionContext` and pass it as `context=`, attach the
sinks THERE — `run_rlm`'s `on_progress`/`on_event` arguments apply only when
it creates the context for you:

```python
context = create_execution_context(..., on_progress=emit_progress, on_event=emit_event)
run_rlm(..., context=context)
```

`droste_runner.run()`'s built-in HTTP path attaches them itself. An
`adapter_module` does NOT inherit that: the runner delegates to the adapter
before any sink-configured context exists, and the Deno relay likewise calls
your adapter directly — an adapter that calls `run_rlm` must attach the sinks
itself (as `examples/pyodide-host/pyodide_host_adapter.py` now does). **This
is a silent degradation if missed**: the run still succeeds; the event stream
is just empty.

Related changes in the same release:

- New event types `llm_response` and `execution_error`; `output` events gained
  `calls_made` / `answer_ready` / `answer_content_chars`. Consumers switching
  on event type should ignore unknown types (additive by design).
- `RLMConfig.verbose` is no longer read by the core. Verbose views are pure
  projections of the event stream — apply
  `droste.execution.progress.render_verbose(event)` in your own sink.
- Custom event emitters are validated: an event `type` outside
  `droste.execution.progress.EVENT_TYPES` raises `ValueError`.

### Relay stream + executor fixes (behavioral, no action needed)

- A streamed LLM response that ends without the protocol's terminal event, or
  carries a mid-stream `error` record, now **fails the call** instead of
  returning accumulated partial text as a clean answer (#43).
- Hosted `droste_runner` subcalls now request and consume ModelRelay's
  `responses-stream/v2` NDJSON contract. Callback servers should honor that
  Accept header so slow generations keep bytes moving through reverse proxies;
  plain JSON remains supported for local callback handlers.
- The Pyodide `RawExecutor` no longer silently truncates oversized sandbox
  output; over-budget prints raise the same `SandboxError` on every substrate,
  giving the model its narrow-your-query feedback (#44).

## 0.10.0 (from 0.9.0)

### The Deno relay ships inside the wheel — drop the tarball + SHA pin

The relay (`relay.ts` and friends) is package data under
`droste/substrates/_relay/`. Stage it from the installed wheel:

```sh
cp "$(droste relay-path)"/*.ts <your-build-staging>/
```

(or `python -c 'from droste.substrates import relay_dir; print(relay_dir())'`).
The wheel is the one pinned, hash-verified artifact in your lockfile, so relay
and engine can no longer drift; delete any release-tarball download and
SHA-256 pin from your build. **Copy every `.ts` file** — 0.10.0 added
`deps.ts` (the single Pyodide version pin), which `relay.ts` imports at
runtime; a hardcoded file list from 0.9.0 ships a relay that cannot resolve
its own imports. The release tarball still exists as a convenience for
non-Python consumers; it is no longer the embedder path.

At startup the relay emits a `startup` event —
`{engine_version, runner_protocol, source_protocol}` — so "which contract is
this app actually running?" is a log line, not a bundle autopsy.

### Accessor discovery reads an explicit manifest — forward it or lose it

The `_droste_data_source` namespace marker is gone. The count contract's
`len()`-over-accessor check now reads accessor names through an optional
environment method (#31). If you compose data sources into a **custom**
`RLMEnvironment` around `DataSourceRegistry.globals()`, forward the registry's
manifest:

```python
def accessor_manifest(self):
    return self._registry.accessor_manifest()
```

**This is a silent degradation if missed**: enforcement falls back to the
static generic verbs and stops covering your sources' actual accessor names
(including `extra_methods`). Environments built on `droste_runner`'s
`RunnerEnvironment` need no change.

### Smaller 0.10.0 deltas

- `register_source_type` / `SOURCE_PROTOCOL_VERSION` moved down to
  `droste.sources.registration`; `droste_runner` re-exports the old names, so
  existing imports keep working. Pass the protocol version you were written
  against as a literal — echoing the engine's own constant back defeats the
  startup compatibility check.
- Trajectory `llm_input` is structured (the message list). On the runner wire
  it is serialized as a **JSON string** — same type as before, finally
  parseable (it was a Python repr).
- `llm_batch_with_errors` is now bounded (worker pool + the same 50-prompt cap
  as `llm_batch`) and reports per-item errors in index order.

## 0.9.0 (from 0.8.x)

### `protocol_version` is required on every runner request

Every `droste_runner` request must carry `"protocol_version": 1`. A missing or
mismatched version gets a structured refusal (`protocol_version_missing` /
`protocol_version_mismatch`) naming both sides — no partial work. See
[docs/architecture.md](docs/architecture.md), "The runner protocol".

### The engine is domain-blind — declare your verbs

Domain-specific verbs are no longer auto-bound by `hasattr`. A source declares
them itself:

```python
class MySource:
    extra_methods = ("get_threads", "get_labels")
```

Both transports honor the declaration (the registry binds them in-process;
the bridge advertises and dispatches them). `search()` also lost its
domain-specific kwargs — sources own their signatures. This is the
`SOURCE_PROTOCOL_VERSION` 1 → 2 bump: a source registering with `protocol=1`
fails loudly at startup rather than silently losing its accessors.
