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

## Unreleased (post-0.10.2)

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
