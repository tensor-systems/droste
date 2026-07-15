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

## Unreleased (post-0.13.0)

### Runner exceptions retain their selected operation

After runner protocol v4 accepts `run` or `preflight`, top-level worker
exceptions now carry that operation instead of `null`, so hosts can retain the
structured error without guessing which response schema applies. Missing or
mismatched protocol versions still refuse first with `operation: null`.
Preflight exceptions use the exact closed five-field preflight envelope with
`status: "error"` and `preflight: null`; they are not run envelopes relabeled
as preflight. Custom-catalog process hosts should replace outer exception
wrappers with `run_worker_request(...)`, then write `WorkerOutcome.response`
and use `WorkerOutcome.exit_code`.

Runner requests also accept one optional non-empty `root_reasoning_effort`.
The native and Pyodide root clients send the exact value on every root callback,
and scaffold evidence records the same value under `root_sampling`. Hosts should
derive this field from their immutable run specification. A conflicting
`root_sampling.reasoning_effort` is rejected before inference.
The bundled Deno/Pyodide relay applies the same operation-specific exception
projection when an adapter raises.

## 0.13.0 (from 0.12.1)

### Runner protocol v4 adds safe, content-free scaffold preflight

`RUNNER_PROTOCOL_VERSION` is now 4. Requests accept an explicit `operation`
discriminant: `run` (the default within v4) or `preflight`. Preflight resolves
the exact effective PromptPack, environment globals, capability manifest,
rollout configuration, and scaffold, then checks checkpoint requirements
without model or provider calls. It does not require endpoints or credentials.
Success returns the complete scaffold in a separately versioned, content-free
preflight value. Compatibility refusal preserves the public
`ScaffoldCompatibilityError` mismatch paths and has stable code
`scaffold_incompatible`.

Upgrade request writers and runners atomically. The version bump is necessary:
an older v3 runner may ignore an unknown `operation` field and execute a request
that a newer host intended only to inspect. A v3 runner instead refuses the v4
envelope before work. In-process hosts may call `preflight_rlm(...)`; it and
`run_rlm(...)` share one scaffold resolver.

## 0.12.1 (from 0.12.0)

### Scaffold concurrency now controls built-in subcall execution

`RolloutConfiguration.concurrency` is now the effective maximum number of
in-flight items in every built-in subcall batch, rather than provenance-only
metadata. The compatibility default is 5, matching the earlier native and
runner behavior. In-process callers that choose another value must pass that
same immutable value to the built-in client's `max_parallel` constructor
argument; `run_rlm` rejects a mismatch before the first model request. Custom
third-party clients remain compatible, but are responsible for honoring the
declared value.

Runner requests may set the positive integer `subcall_concurrency`; omitted
values resolve once to 5. Native CLI `--rollout-config` values now configure
the client as well as the scaffold manifest. The Pyodide relay preserves the
field unchanged. The effective value remains content-free provenance at
`scaffold_manifest.inference.concurrency`. The runner protocol remains v3
because the request field is optional and additive.

## 0.12.0 (from 0.11.0)

### Optional provider bridge v2 duplex sessions

`BridgeProvider` keeps its unary provider-protocol-4 behavior unless a host
explicitly passes `duplex_call`. The callable must return one per-invocation
session with `receive`, `send`, `cancellation_requested(call_id)`, and `close`
methods. Bridge v2 streams cumulative checkpoints and cooperative cancellation
through a bounded pull-based pump; do not implement it by calling back into a
suspended Pyodide interpreter.

The bundled Deno relay now selects this duplex path for its two-interpreter
provider mode. Host adapters invoked by that relay must accept the new
`duplex_bridge_call` keyword and pass it to
`BridgeProvider(bridge_call, duplex_call=duplex_bridge_call)`. Upgrade the staged
relay, adapter, and Python package atomically. Single-interpreter mode and hosts
that construct `BridgeProvider(bridge_call)` directly remain unary.

For the bundled relay, `SIGUSR1` requests cooperative cancellation of its one
active duplex provider call. Existing `SIGTERM`/`SIGKILL` process-control
semantics are unchanged and remain the fallback for non-cooperative handlers.

Remote provider loss before terminal delivery now yields the stable
`bridge.transport_lost` capability error and settles once in the receiving
broker. Acknowledged checkpoints remain committed; unacknowledged remote facts
are not trusted.

### Python 3.14 is supported by the core package

Droste continues to require Python 3.11 or newer and now tests both the 3.11
compatibility floor and latest stable Python 3.14 in CI. The optional Verifiers
v1 harness remains limited to Python 3.11–3.13 by its upstream dependency
markers; installing the extra on 3.14 leaves it absent and its dedicated tests
skip cleanly.

### Rollouts expose one content-addressed scaffold identity

`RLMResult`, CLI JSON, and runner responses add `scaffold_manifest` and
`stdout_chars`, the sum of exact returned stdout lengths on retained trajectory
entries. The existing prompt-pack
`content_sha256` participates in the manifest as its prefixed `content_hash`.
Trajectory entries add `attempt_kind` and `stdout_chars`; consumers should use
the typed attempt/status fields rather than infer them from text. Droste rejects
oversized output instead of silently truncating it, so it does not publish a
truncation flag or count. Default durable terminal records retain only the
manifest ID and schema version.

Hosts may provide resolved model/source revisions, sampling, concurrency, seed,
and runner version through `RolloutConfiguration`, plus checkpoint requirements
through `RLMConfig.checkpoint_requirements`. Incompatible requirements fail
before inference. The runner accepts the corresponding optional request fields;
the protocol remains v3 because all envelope additions are optional/additive.
The CLI adds `--prompt-profile` and `--rollout-config`.

The `verifiers` extra adds the Verifiers v1 `droste_verifiers` harness package.
RLM skill artifacts and `rlm_skills_provider` are opt-in; no skill provider or
prompt text is registered automatically. See `docs/scaffold-manifest.md`,
`docs/verifiers-harness.md`, and `docs/rlm-skills.md`.

## 0.11.0 (from 0.10.6)

### Prompt-pack provenance includes canonical content identity

Every resolved `PromptPackRecord` and its CLI, runner, report, and trace result
projection now includes the additive `content_sha256` field. It is the lowercase
SHA-256 digest of a pure canonical UTF-8 JSON serialization of the complete
validated pack value; prompt text itself is not added to result provenance.
Callers that construct `PromptPackRecord` directly remain source-compatible
because the new field is optional there, while records produced by the resolver
always populate it.

Prompt-pack TOML may declare an optional top-level `content_sha256`. Loaders now
reject malformed declarations and declarations that do not match the validated
content. Existing artifacts without a declaration continue to load unchanged.
See `docs/prompt-packs.md` for the canonicalization rules and public helper
functions. The runner protocol version is unchanged.

### Capability handlers receive one execution context (breaking)

Trusted capability and `ProviderRuntime` handlers now have the single required
signature `handler(CapabilityExecutionContext, *args, **kwargs)`. Migrate every
registration atomically; there is no signature introspection or legacy adapter.
Generated sandbox bindings are unchanged and never receive the context.

The frozen context carries call/run identity, the caller-authorized monotonic
deadline, and immutable reservation facts. Long-running trusted handlers call
`check()` to observe cancellation/deadlines and may report cumulative token and
subcall usage with `checkpoint()`. Providers do not receive the ledger or trace.
Cancellation and deadline results use stable `cancelled` and
`deadline_exceeded` codes. Admission begins the exactly-once settlement
boundary, including policy denial before handler dispatch.

The provider protocol is now 4. `ProviderService` bridge invokes require the
versioned execution facts and return a validated cumulative checkpoint. Upgrade
both bridge interpreters and the staged relay atomically; the startup event's
`provider_protocol` detects drift.

### Batch-item errors expose safe structured details

Native response-batch item failures keep the existing human-readable error
string and now add an optional `details` object to `llm_batch_with_errors` and
structured JSON batch error entries. The additive fields are `request_id`,
`batch_id`, `item_id`, `layer`, `cause`, `status_code`, `code`, and
`retryable`; the frozen `BatchItemErrorDetails` value itself redacts and bounds
every accepted string, including when hosts construct it directly. Unknown
fields and payload data are not retained. The same object survives built-in
broker and environment boundaries.

Direct `llm_batch` calls still raise a `RuntimeError`, now using the compatible
`BatchItemError` subclass when typed item details are available. Hosts can
inspect its frozen `BatchItemErrorDetails` value without parsing the unchanged
message. Custom `SubcallClient` implementations may keep returning only
`index` and `error`; no new method or required field was added. The runner
protocol version is unchanged.

### Compute limits are one strict budget object (breaking)

`max_iterations`, `max_calls`, `max_depth`, `max_subcalls`, and
`subcall_max_output_tokens` are removed from engine, client, CLI, and runner
configuration. There are intentionally no aliases or translation layer. Pass
one complete `Budget(tokens, subcalls, depth, wall_ms, root_output_tokens,
subcall_output_tokens)` value. Pass local REPL guardrails separately as
`SandboxLimits`.

All root and brokered capability work reserves its maximum authorized vector
before dispatch and reconciles actual work afterward. Rejections use typed
`BudgetExhausted` values. Batch reservations are atomic; strict child ledgers
reserve from the parent and refund unused authorization on close. Runner
requests now require `protocol_version: 3` and a complete `budget` object.

Budget mutations are durable Trace ABI v1 values from
`source="budget_ledger"`, keyed by `call_id`, with `reserve`, `commit`,
`refund`, and `exhaust` actions. See `docs/budgets.md`.

### Subcall output limits are visible to the root model

The built-in subcall clients now expose read-only `output_token_limit` metadata
when they know the effective value. The root authorized-compute prompt renders
positive limits as bounded, `None` as deliberately unbounded, and unavailable
metadata as unknown. The HTTP-backed runner reports an explicit positive
request override; when the field is omitted, its callback owns the default and
the limit stays unknown. This keeps custom `SubcallClient` implementations
compatible: the base protocol has no new required member.

Custom clients may opt in by implementing the additive
`SubcallOutputTokenLimitProvider` companion protocol. Return a positive integer
for the effective per-call ceiling or `None` only when the client deliberately
leaves output unbounded. Do not report an endpoint default unless the client can
identify its effective value. The runner envelope and protocol version are
unchanged.

The built-in generic prompt packs are revised to `1.0.2`; they distinguish
subcall input capacity from output capacity when planning structured and
map-reduce work.

### Data providers are manifest-driven (breaking)

The fixed `DataSource` protocol, capability booleans, universal verb table,
`extra_methods`, process-global source factories, and singular `data_source`
request sugar are removed. There are intentionally no compatibility aliases.
Hosts must construct an explicit `ProviderCatalog` from
`ProviderRegistration` values and bind declarative `ConfiguredSource` values
for each run. The bundled local provider is now `sqlite_provider()` with type
`sqlite`; its Python bindings remain `query()` and `get_schema()`, while the
stable raw operation IDs are `query` and `schema`.

Each `ProviderManifest` is source-agnostic, immutable, revisioned, and
SHA-256-digested. Operations declare separate raw IDs and Python binding names,
schema dialect/provenance, cursor behavior, inline/handle/untyped delivery,
budget class, and descriptions. Hosts classify every operation's effect and
own policy metadata; do not trust effect annotations received over a bridge.
`ProviderService` and `BridgeProvider` replace `DataSourceService` and
`BridgeDataSource`.

Remove imports of `DataSource`, `DataSourceCapabilities`, `SearchResult`,
`DataSourceRegistry`, `register_source_type`, `SOURCE_PROTOCOL_VERSION`, and
the old SQL factory. Use `ProviderManifest`, `ProviderOperation`,
`ProviderRegistration`, `ProviderCatalog`, `ProviderRegistry`, and
`PROVIDER_PROTOCOL_VERSION` instead. `EvidenceRef` is also replaced by
structured `EvidenceLocation`/`EvidenceRange` values.

`RUNNER_PROTOCOL_VERSION` is now 3. The manifest migration introduced provider
protocol 3; the context-first handler migration above makes the current provider
protocol 4. Requests
must use `data_sources` as a list of `{type, name, ...config}` objects. The
relay startup event now reports `provider_protocol` instead of
`source_protocol`. Upgrade the runner request, staged relay, and provider
catalog atomically; mismatches fail before work begins.

### Built-in sandbox capabilities use one brokered ABI

`RunnerEnvironment` and `PyodideEnvironment` now generate their existing
`llm_query*` and provider Python APIs from one immutable capability manifest.
Built-in environments no longer put raw `SubcallClient` or provider bound
methods in the sandbox globals mapping. The loop's structured JSON
batch replacement also uses the environment's broker-backed adapter.

Hosts that need correlation may pass additive `capability_run_id` and
`capability_parent_run_id` arguments to `create_environment`. Optional typed
`capability_guard`, `capability_annotator`, and `capability_observer` callables
are integration seams for policy/budget facts and trace projection; no default
policy, shared ledger, persistence, or transport change is implied.
Capability calls carry a frozen `CapabilityId` made of `kind`, `provider_type`,
`source_id`, and `operation`; the broker resolves the full descriptor from its
manifest so future schema, documentation, and policy metadata do not change the
wire identity. Trace integrations should persist
`CapabilityResult.to_trace_dict()`, the content-free projection. The full
`to_dict()` includes arguments, inline results, error messages, evidence
references, and result-handle locators and is suitable only for explicitly
configured replay retention.
`capability_annotator` is an exactly-once post-attempt finalizer: it runs after
each attempted handler outcome, including invalid results and propagated
cancellation, and is skipped for validation or guard exits before an attempt.
Accounting integrations can therefore reconcile a guard reservation by
`call_id` without a second broker callback.
Trusted handlers may return `CapabilityOutcome(result=..., metadata=...)` or
`CapabilityOutcome(error=CapabilityError(...), metadata=...)` to attach typed
provider failures, usage, or evidence without raising or requiring a transport
parser. Error `code` is now an extensible stable string; `CapabilityErrorCode`
remains the set of broker-defined string constants. Raw handler return values
are normalized automatically at registration. Provider sequence metadata is
preserved first and finalizer sequence metadata is appended; conflicting
singular handle/child-run facts become an annotator error rather than silently
choosing one.
The unused `BatchLLMError` compatibility type and repair branch are removed.
Plain `llm_batch` reports its brokered typed failure as `CapabilityCallError`;
callers that need ordered per-item failures use `llm_batch_json` or the trusted
`llm_batch_with_errors` adapter, both of which remain single atomic broker calls.
`DataSourceRegistry` no longer has a standalone `globals()` projection. Hosts
construct one run broker from `capability_registrations()` and pass it to
`broker_globals()`, preventing an accidental second broker without the run's
identity, guard, accounting annotator, or observer.
Custom environments must expose the resulting registry's
`accessor_manifest()` as well. There is no fixed generic-verb fallback: an
environment that omits this manifest supplies no provider accessors to the
count-policy check, so its custom bindings are not covered.

Custom `RLMEnvironment` implementations must now implement
`sandbox_subcalls(subcalls)`. Return a broker-backed `SubcallClient`; the
`droste.capabilities.broker_subcalls()` helper supplies the standard standalone
adapter. `run_rlm` replaces all canonical subcall globals from that method and
does not retain a raw-client fallback.
### Trace ABI v1 unifies live events and terminal run records

Every structured event now carries the required v1 envelope: `run_id`, positive
monotonic `seq`, UTC `timestamp`, `type`, `version`, `persistence_class`, and
required `depth` (`0` for roots), with optional `parent_run_id`. Partial pre-envelope dictionaries are not a
supported wire format. Pyodide relay telemetry is a child run correlated by
`parent_run_id`, so the relay and engine do not race to own one sequence.

`RLMResult`, runner responses, and CLI JSON expose `run_record`. Durable
terminal/usage/budget/policy/capability facts are always selected;
code/output/error/repair/result/replay content requires an explicit
`TraceRetentionPolicy`; progress/deltas remain transient. The canonical
trajectory-free `result` is nevertheless always delivered live before `done`;
full `replay` is emitted only when selected. The default record retains no
configurable content. Training authorization is independent and defaults to
denied; enabling it requires an authorization reference plus the `training`
purpose. Retention policies now carry a stable `policy_id` and may record an
absolute host-managed expiry.

Attach `on_run_record` to `RLMConfig` or `create_execution_context` to hand the
resolved immutable record to local persistence. If an existing context is
passed to `run_rlm`, it owns trace settings; explicit conflicting `RLMConfig`
settings raise `ValueError`. See [docs/trace-abi.md](docs/trace-abi.md).

The runner protocol is now version 2. Success, refusal, and exception responses
share one field set. The unary response does not include a full trajectory;
when replay is retained, `replay.result.trajectory[].llm_input` is a structured
message list instead of a JSON-encoded string. Requests must send
`"protocol_version": 2`; mismatched v1 requests fail before work begins.

### Hosts select environments through one substrate factory

New in-process hosts should build an immutable `EnvironmentConfig`, then call
`create_environment_context(config, ...)` and `create_environment(config,
...)`. The CLI, HTTP runner, benchmark harness, and reference Pyodide adapter
now use this path, so execution budgets and substrate selection have one owner.

Existing direct `RunnerEnvironment` imports remain compatible. Pyodide hosts
should migrate: `kind="pyodide"` selects the signal-free `RawExecutor` path and
requires `host_managed_timeout=True` plus `host_managed_isolation=True`. Those
flags assert that the host already supplies the external deadline and WASM
jail; they do not create either boundary. A Pyodide config with a nonzero
`exec_timeout_ms` fails loudly instead of pretending to enforce a signal timer.

### Runner implementation modules are focused

The former `droste_runner.runner` monolith is split into `run`, `protocol`,
`http_clients`, `sources`, and a small `environment` compatibility shim. The
generic native environment now lives at `droste.environments.RunnerEnvironment`;
the CLI and in-tree embedders use that canonical import.

Existing imports from `droste_runner.runner` remain valid, including `run`,
`RunnerEnvironment`, HTTP clients, source helpers, and protocol constants.
This is a structure-only change: request/response fields, protocol versions,
adapter dispatch, and process entrypoint behavior are unchanged.
The supported process entrypoint remains `python -m droste_runner` from an
installed package. Direct execution of an extracted `runner.py` file is not a
supported entrypoint; the old repository-layout `sys.path` mutation was
intentionally removed.

### Harness prompts resolve from versioned prompt packs

The built-in system, user, refinement, repair, and extraction prompts now load
from complete TOML prompt packs and resolve once per run by `(model, profile)`.
The default generic `full` pack preserves the prior harness behavior;
`minimal` and `none` preserve the existing tips profiles. Existing complete
`system_prompt` and user/refinement template overrides remain accepted.

`RLMConfig.prompt_profile` is the new profile spelling; when omitted,
`tips_profile` remains compatible. `RLMConfig.enforce_contract=None` delegates
to the resolved pack's policy default, while explicit booleans still win.
`run_rlm` accepts an immutable caller pack or consumer catalog. Invalid packs
fail before the first model call.

`RLMResult.prompt_pack`, CLI JSON, and the built-in runner response now expose
the additive resolved ID, revision, profile, resolution tier, model family, and
provenance. Runner requests may send additive `prompt_profile`; protocol version
1 is unchanged. See `docs/prompt-packs.md` for authoring and fallback rules.

### Trajectory execution status is explicit

`IterationRecord` and each built-in runner trajectory entry now include the
additive string field `execution_status`, currently `"success"` or `"error"`.
Use it instead of interpreting the text in `execution_result`: successful
stdout may legitimately begin with `ERROR:`. The existing `execution_result`
field and runner protocol version are unchanged.
Direct positional construction that omits the new field remains accepted and
defaults conservatively to `"error"`; engine-created records always set the
status from their typed step outcome.

### Semantic structured batches fail closed when incomplete

`PolicyHints(semantic=True)` now keeps any `llm_batch_json` result with
unresolved item errors from confirming `answer["ready"]`. The loop returns the
violation for repair; if the iteration budget ends, the existing bounded
extraction path either produces an explicitly unconfirmed answer with
`recovered_error.type == "PolicyError"` or leaves the policy error fatal.

A later error-free call clears an incomplete result only when it repeats the
exact prompts, contexts, schema, and validator object. A different successful
batch is not completion evidence for earlier partial work. This is a behavior
change only for callers that explicitly enable semantic policy; runs without
that hint are unchanged.

If the remaining total subcall budget is smaller than the minimum needed to
replay every unresolved recorded exact request, the loop now stops requesting
impossible repairs and enters the existing bounded extraction/failure path
immediately. In the additive diagnostic details, `required_subcalls` is the sum
of the full batch cardinality for each unresolved recorded exact request, not
the number of failed items. Validator object identity participates in exactness,
so otherwise-identical calls made with distinct validator objects are distinct
recorded requests and each contributes its full batch cardinality.

Successful extraction remains unconfirmed and preserves a `PolicyError` in
`recovered_error`; without extraction evidence, the same policy error remains
fatal. Its additive `details.reason` is
`"semantic_exact_retry_budget_exhausted"` and also includes the remaining
subcall count plus unresolved recorded-request and item counts. Provider errors
do not trigger this handoff while enough budget for the recorded exact retries
remains.

When that terminal handoff has no retained `answer["content"]`, the loop now
makes one root finalization request and executes at most one returned code block
in the existing persistent REPL. No missing-code or execution repair is made.
All model-visible subcall bindings are disabled before that code executes, so
the step cannot spend further subcall budget and can synthesize already retained
work; any resulting draft still passes through the normal unconfirmed extraction
path. Incomplete exact semantic evidence continues to revoke readiness, and a
finalization that retains no draft leaves the original typed policy failure
fatal. If the root finalization request itself fails, the event stream emits an
additive `finalization_error` event with `error_type` and `message`; the original
policy failure remains authoritative and no finalization retry is made.

To make this completeness check enforceable, `run_rlm` continues to replace any
`llm_batch_json` and `llm_query_batched_json` entries in the mapping returned by
`environment.globals()` with Droste's tracked bindings before sandbox execution.
While semantic contract enforcement is active, it now also replaces
`llm_query`, `llm_batch`, `batch_llm_query`, and `llm_query_batched` with
revocable bindings so saved aliases can be disabled during terminal
finalization. Embedders that enable semantic enforcement must treat all six
names as reserved: expose custom helpers under different names, or customize
the `SubcallClient` passed to `run_rlm` instead. Runs without semantic
enforcement retain the prior direct-binding behavior.

The three built-in prompt packs are revised from `1.0.0` to `1.0.1` to describe
the bounded terminal mode in their error-repair templates. Caller-provided packs
remain valid without changes; the engine still enforces the no-subcall boundary
even when a custom template does not describe it.

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

Budget rejection uses `BudgetExhausted`, carrying the exhausted resource,
requested amount, and remaining authorization. Structured batch errors expose
the stable `budget_exhausted` type without matching exception text.

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
