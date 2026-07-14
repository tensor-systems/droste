# Architecture

## The loop

```

The harness strategy is resolved once at run start as one immutable
[prompt pack](prompt-packs.md). Package/file I/O ends at the loader boundary;
slot validation, `(model, profile)` fallback, and rendering are pure value
transformations. The engine never merges partial packs or changes strategy
mid-run.

Optional [RLM skills](rlm-skills.md) are immutable, content-addressed strategy
data loaded through an explicitly registered read-only provider. They never
change the prompt-pack or loop contract. The resolved prompt, capability,
contract, inference, budget, and sandbox facts form one content-addressed
[scaffold manifest](scaffold-manifest.md), checked before inference when a
checkpoint declares requirements.
question ──▶ root LLM ──▶ python code ──▶ sandboxed REPL ──▶ output
                ▲                             │
                └── refinement prompt ◀───────┘
                        (loop until answer["ready"], budget-bounded)
```

The root model's *initial prompt* contains a description of your data
(type, size, a preview for string contexts; names and sizes for files) —
not the data itself, which lives as REPL variables (`context`, or named
sources like `db`). What flows back to the root each iteration is exactly
what the executed code prints: narrow deliberately, because printed output
is root-visible by design. Each iteration it writes Python;
the sandbox executes it; stdout feeds the next iteration. Three functions
bridge back to language models:

- `llm_query(prompt)` — one subcall.
- `llm_query_batched(prompts)` — concurrent subcalls (bounded workers),
  order-preserving. Aliases: `llm_batch`, `batch_llm_query`.
- `llm_batch_json(prompts, schema, ...)` — ordered local JSON validation and
  malformed-only bounded repair, with per-item errors (the error indices are
  authoritative because valid JSON `null` and failed slots are both `None`).
  Alias: `llm_query_batched_json`. With explicit semantic policy, an incomplete
  result blocks answer confirmation until the exact prompts, contexts, schema,
  and validator object later produce an error-free result.

## Capability broker

Generated code reaches data sources and sub-LLMs through one brokered path.
At environment construction, trusted source verbs and the subcall client become
an immutable `CapabilityManifest`. The native and Pyodide environments generate
the same model-facing Python functions from that manifest; those functions
submit immutable `CapabilityCall` values to an exact allowlist and unwrap the
single typed `CapabilityResult` envelope for compatibility with existing code.
Calls carry only a frozen `CapabilityId`—kind, reusable provider type,
configured source ID, and operation. The broker resolves the complete descriptor
from the manifest, keeping stable dispatch identity separate from documentation,
schema, pagination, budget, and policy metadata that descriptors may gain.

Each result envelope carries the capability kind, configured source or provider
identifier, operation, call and run IDs, typed status/error, value or bounded
handle, usage and budget deltas, evidence references, and optional parent/child
run IDs. Most fields are empty until a host supplies facts through the broker's
narrow annotator seam. A guard may deny a call before dispatch, and an observer
may project the completed immutable envelope into a trace. Those are interfaces,
not policy, ledger, or trace implementations; the corresponding subsystems own
their semantics. Durable capability events use `CapabilityResult.to_trace_dict()`:
it retains IDs, status, typed error code/type, accounting facts, evidence count,
and optional result-handle media type and size, but no parameters, inline result,
error message, evidence reference, or handle locator. Full envelopes are replay
content and require an explicit content-retention policy.
At registration, raw trusted handler values are normalized into one
`CapabilityOutcome` convention. A provider can return a result or a typed
provider failure with an extensible stable error code, together with usage and
evidence metadata; unexpected exceptions remain broker `handler_error` values.
Provider sequence facts precede the exactly-once finalizer's appended facts.
Conflicting singular handle or child-run facts fail closed, so dispatch never
needs a provider-specific parser or precedence convention.
The annotator has exactly-once post-attempt semantics: it runs after success,
handler error, invalid result, or propagated cancellation, but not when run
identity, allowlist, arguments, or the guard reject a call before its handler is
attempted. The separate attempt authority starts at admission and therefore
settles even those post-admission guard exits without changing annotator scope.

Every trusted registration has one context-first handler signature:
`handler(CapabilityExecutionContext, *args, **kwargs)`. The context is a frozen
view of call/run identity, deadline, and reservation plus two narrow mechanisms:
`check()` for cooperative cancellation/deadline observation and cumulative
`checkpoint()` for token/subcall progress. It exposes no ledger, trace, or
callback registration. A broker-owned mutable attempt controller closes the
cancellation/finalization race; admission begins the exactly-once settlement
boundary even when a later policy guard denies dispatch.

`call_id` is an in-flight identity, not a durable idempotency key. After call
validation, the broker atomically claims a caller-supplied ID before asking the
attempt authority for admission and retains that claim through result delivery.
A concurrent duplicate is rejected without admission, checkpointing, or
settlement. The claim is released when `dispatch` returns or raises, including
after admission refusal, so a later dispatch may deliberately reuse the ID.
Hosts that require replay deduplication must enforce that separate policy before
dispatch rather than turning the budget authority into an identity registry.

`llm_batch` is one broker operation and invokes the subcall client's batch method
once. It is not decomposed into nested `llm_query` calls, so ordering,
reservation, concurrency, and provider-native batch semantics stay intact.
`llm_batch_json` remains local deterministic composition over the broker-backed
batch adapter. `ProviderService` is a fixed bridge transport behind registered
handlers and is deliberately not a second capability ABI.

The loop ends when the model sets `answer["ready"] = True`, or the resolved
compute budget cannot authorize more work. On a supported terminal handoff,
a trajectory containing executed work may use a single **extract pass** to
produce a best-effort answer, flagged `extracted=True` and never
presented as confirmed. Policy violations withhold gated content from the
final answer rather than silently passing it through.

Policy is caller-supplied rather than inferred from question text. In-process
callers opt into semantic enforcement with
`RLMConfig(policy_hints=PolicyHints(semantic=True))`; without that hint,
structured batch results remain prompt-driven and the loop does not add this
completeness gate.

## Budgets and cost

Everything expensive is authorized by one immutable `Budget`: total tokens,
subcalls, child depth, wall time, and per-request root/subcall output ceilings.
One run-scoped `BudgetLedger` owns mutable accounting. Root calls and brokered
capabilities atomically reserve before dispatch, then commit actual work and
refund unused authorization. A failed vector reservation changes nothing;
concurrent work therefore cannot oversubscribe one dimension while another
passes. See [Budgets](budgets.md).

The root prompt reports the effective per-call subcall output ceiling separately
from input capacity. Built-in clients implement the optional
`SubcallOutputTokenLimitProvider` companion protocol: a positive
`output_token_limit` is bounded, `None` is deliberately unbounded, and absence
is rendered as unknown rather than guessed. HTTP-backed clients report only an
explicit positive limit because a callback-owned default is not knowable from
the runner. The base `SubcallClient` protocol is unchanged, and broker-backed
and semantic-policy wrappers forward the optional metadata.

## Providers and configured sources

Provider metadata is immutable, source-agnostic data. `ProviderManifest`
contains a revision, canonical SHA-256 digest, and ordered `ProviderOperation`
values. Each operation carries a stable raw ID, a separately validated Python
binding name, documentation, parameter/result `SchemaSpec` values with explicit
dialect and provenance, cursor semantics, inline/handle/untyped delivery, and
a budget class. Cursor operations must describe both the input cursor and
output `next_cursor`.

The host owns the imperative edge: a `ProviderRegistration` pairs the manifest
with an exact side-effect map, optional policy metadata, and a binder returning
live handlers. Effects are never accepted from a transport. An explicit
`ProviderCatalog` binds declarative `ConfiguredSource` values into one
per-run `ProviderRegistry`; there is no process-global registry and requests
cannot name code to import. The registry snapshots descriptors once for the
run, then projects broker registrations, prompt text, Python bindings, and the
policy accessor manifest from those same values.

`ProviderService`/`BridgeProvider` carry a verified manifest plus raw operation
calls across an interpreter boundary. Invoke requests also carry exact portable
execution facts; responses carry a separately validated cumulative checkpoint
which the receiving broker applies before its sole final settlement. The
receiving host supplies its own effects and policy before binding. Unknown
operations are rejected against the bound handler map, and schema/digest
mismatches fail before dispatch.

The bundled SQLite source is local-mode: SELECT-only policy gate (single
statement, masked-identifier keyword scanning, LIMIT injection, row caps,
statement timeout), database opened `mode=ro`, and `PRAGMA query_only=ON`
enforced on host-supplied connections.

## The runner protocol (embedding)

`python -m droste_runner` is a one-shot JSON worker: a host writes a request
file (question, budgets, endpoints or context, declarative data sources) and
reads a response (answer, `ready`, `extracted`, iterations, attempted and
successful subcalls,
trajectory, usage). This is how non-Python hosts embed the engine.
Each trajectory entry carries `execution_status` (`success` or `error`) beside
the unchanged `execution_result` text, an `attempt_kind` (`initial`, `repair`,
or `terminal`), and the exact character count of stdout returned by the
environment. Consumers must not infer any of these from output text. Oversized
stdout is rejected rather than silently truncated. Result-level `stdout_chars`
is the sum of these exact counts across the retained trajectory.

The Python implementation is split by ownership: `droste.environments`
provides the generic native in-process environment; `droste_runner` keeps
HTTP clients, remote source transport, envelope helpers, and orchestration in
`http_clients`, `sources`, `protocol`, and `run` modules respectively.
`droste_runner.runner` is a convenience facade, not a second implementation.
Both successful and refused envelopes are shaped by
`protocol.build_response`, so their shared fields cannot drift.

### Scaffold preflight

Hosts may set `"operation": "preflight"` to resolve and validate the exact
effective scaffold before authorizing inference. Omitting `operation` preserves
the normal `"run"` behavior. Preflight requires the same budget, model,
rollout, PromptPack selector, provider configuration, sandbox configuration,
and optional `checkpoint_scaffold_requirements` as execution, but it does not
require root/subcall endpoints or a token.

The engine and runner share one resolver for PromptPack selection, policy
defaults, environment globals, capability manifest, rollout configuration,
scaffold construction, and `ScaffoldCompatibilityError`. A successful response
is intentionally a separate, content-free envelope:

```json
{
  "protocol_version": 4,
  "operation": "preflight",
  "status": "success",
  "preflight": {
    "schema_version": 1,
    "scaffold_manifest": {"schema_version": 1, "id": "sha256:..."}
  },
  "error": null
}
```

The abbreviated manifest above is illustrative; the actual response carries
the complete `ScaffoldManifest`. It never carries the question, context,
rendered prompt prose, answer, trajectory, trace, run identity, or usage. A
checkpoint mismatch returns `status: "refusal"`, no preflight result, and a
typed `ScaffoldCompatibilityError` with code `scaffold_incompatible` and
structured mismatch paths. Hosts can therefore distinguish a policy refusal
from worker or transport failure without parsing an error message. Preflight
installs no event sink and uses a fail-if-called capability placeholder; root,
subcall, and configured-provider callbacks cannot be dispatched.

Configured providers are projected from their immutable registration manifest,
effect policy, source type, and source name. Preflight deliberately does not
invoke live provider binders: binders may open files, databases, or network
connections, while those runtime resources and task data do not participate in
the scaffold identity. Normal execution binds the same descriptors to their
live handlers for the run. Refusals or worker errors that occur before
an operation result is produced carry `operation: null`; successful runs and
typed preflight compatibility refusals carry their accepted discriminant.

Completed responses also carry the policy-resolved
[Trace ABI v1](trace-abi.md) `run_record`. Live events and terminal records use
the same strict envelope and projection. Persistence remains a host I/O
decision; the engine never opens a trace store.

Completed results also expose the full scaffold manifest and aggregate stdout
facts. Default durable retention stores only the manifest ID/version; trainer
outcomes join externally by run ID and manifest ID. The optional
[Verifiers v1 harness](verifiers-harness.md) sends root and subcall traffic
through one interception endpoint without moving runtime data into the prompt.

The optional positive `subcall_concurrency` request field resolves once before
client construction (default 5), controls every HTTP-backed subcall batch, and
is recorded at `scaffold_manifest.inference.concurrency`. Native CLI rollout
configuration follows the same path. The Pyodide relay forwards the value
unchanged; it does not choose another concurrency policy.

**Versioned boundary**: the request/response schema and provider contract
are versioned, each by a single integer:

- `RUNNER_PROTOCOL_VERSION` (currently 4) governs the request/response
  envelope. Every request **must** carry `protocol_version` — requests are
  self-describing, the same discipline as JSON-RPC's mandatory `"jsonrpc"`
  field. A missing or mismatched version is answered with a structured
  error (`protocol_version_missing` / `protocol_version_mismatch`) naming
  both sides — a host detects incompatibility explicitly instead of
  failing on a missing field. Responses carry `protocol_version`: the
  engine stamps its own everywhere except an `adapter_module` response
  that already claimed one (adapters own their response shape).
  Protocol v4 requires one exact six-field `budget` object; missing,
  unknown, or invalid fields fail before endpoint dispatch.
  Protocol v4 adds the `operation` discriminant and typed preflight response.
  The bump is required for safety: a v3 runner could ignore an unknown
  `operation` field and execute a request that a v4 host intended only to
  inspect. Old runners instead reject v4 at the version gate before endpoints,
  credentials, provider binding, or work.
- `PROVIDER_PROTOCOL_VERSION` (currently 4) governs manifest parsing,
  context-first provider binding, and bridge invocation facts. A mismatched
  manifest fails before a source is live.

Provider bridge v2 is a separate, explicitly selected transport contract; it
does not change provider manifest protocol 4. It adds one bounded per-call
message pump for ordered checkpoints, cancellation acknowledgement, and a
terminal outcome while preserving the same capability identity, manifest,
generated binding, and broker finalization path. Unary bridge invocation stays
the default when a host does not supply the duplex session factory.

The rules: **adding an optional field is not a version bump** (the 0.5.x
subcall cost-control knobs are the worked example — older engines ignore
them, newer engines honor them); renaming or removing a field, or changing
a field's semantics, bumps the integer. A control plane that serves
embedded engines must tolerate a stated window of engine versions
(embedded engines in the field can't be force-upgraded).

## Sandboxing — the honest version

The REPL is a **guardrail, not a security boundary**. It bounds output size
and execution time and keeps well-behaved models on the rails. A hostile
workload needs real isolation, which belongs to the host: OS permissions
own file access (the SQL source's read-only policy assumes this), and
process/container/WASM isolation owns escape (hosts run the engine in a
subprocess, jail, or Pyodide/WASM runtime as their threat model requires).
The engine will not pretend otherwise, and neither should you.

### Host environment factory

Hosts select the execution substrate through one immutable configuration and
derive the loop context and environment from it:

```python
from droste import (
    Budget,
    EnvironmentConfig,
    SandboxLimits,
    create_environment,
    create_environment_context,
)

config = EnvironmentConfig(
    kind="native",
    budget=Budget(subcalls=50, depth=1),
    sandbox=SandboxLimits(output_chars=25_000, execution_timeout_ms=5_000),
)
execution_context = create_environment_context(config)
environment = create_environment(
    config,
    context=data,
    registry=registry,
    subcalls=subcalls,
    execution_context=execution_context,
)
```

`native` selects the signal-timed in-process environment. `pyodide` selects
the thread- and signal-free raw executor; because WASM isolation and
wall-clock termination live outside Python, that config requires explicit
`host_managed_isolation=True` and `host_managed_timeout=True` declarations and
rejects a nonzero `exec_timeout_ms`. These booleans are assertions by the host,
not security mechanisms. The host still has to provide the Deno/WASM jail and
kill deadline. This keeps substrate selection pure while the constructor is a
thin shell around live data, registry, and subcall dependencies.
