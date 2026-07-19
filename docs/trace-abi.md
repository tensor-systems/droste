# Trace ABI v4

Droste exposes one append-only event stream and one policy-resolved terminal
`RunRecord`. The engine creates values; it does not choose a database or write
files. Hosts attach `on_event` for live delivery and optionally
`on_run_record` as their persistence I/O shell.

The integer `version` names the complete strict contract: envelope, event
bodies, persistence classification, and ordering/terminal invariants. Changing
any of those requires a new Trace ABI version and an atomic consumer migration;
even adding an optional field is breaking because current strict readers reject
unknown fields. The runner protocol changes only when its own request/response
or negotiation contract changes, including an embedded run-record version. It
does not change merely because a released fixture is added.

## Event envelope

Every event is a strict v4 value with these fields:

```json
{
  "run_id": "e0f7...",
  "seq": 1,
  "timestamp": "2026-07-14T05:00:00Z",
  "type": "progress",
  "version": 4,
  "persistence_class": "transient",
  "parent_run_id": "optional-parent",
  "depth": 0,
  "status": "Iteration 1/20: Generating code..."
}
```

`seq` is positive and monotonic within one `run_id`. `depth` is required: roots
use `0`; child runs use their deterministic parent depth plus one and carry
`parent_run_id`. `parse_event` validates a UTC timestamp and rejects partial,
misclassified, unsupported-version, missing-body-field, and unknown-body-field
values.

Relay-owned Pyodide telemetry uses a child `run_id` whose `parent_run_id` is
the engine run. The relay and engine therefore each own one sequence; their
events correlate without two writers racing to stamp the same counter.

## Retention is data selection

The persistence class is exhaustive and fixed by event type:

| Class | Event types | Rule |
| --- | --- | --- |
| `durable` | `usage`, `budget`, `policy`, `capability`, `done` | Always in the terminal record |
| `configurable` | `iteration_start`, `llm_response`, `code`, `output`, `execution_error`, `subcall`, `repair`, `extract`, `result`, `replay` | Included only when named by `TraceRetentionPolicy.retain` |
| `transient` | `startup`, `progress`, `reasoning_delta` | Live delivery only; never in the terminal record |

Retention governs the terminal record, not the live channel. `result` is
always delivered once before `done`, even when it is not retained. `replay` is
different: it is emitted only when the host explicitly selects replay
retention.

## Exhaustive v4 bodies

Every event body has a fixed top-level schema. Optional fields are marked `?`.
Objects named below are JSON objects; all other types are primitive.

| Type | Body |
| --- | --- |
| `startup` | `engine_version: string`, `runner_protocol?: integer|null`, `provider_protocol?: integer|null` |
| `progress` | `status: string` |
| `iteration_start` | `iteration: integer`, `remaining_tokens: integer` |
| `llm_response` | `iteration: integer`, `response: string` |
| `code` | `iteration: integer`, `code: string` |
| `output` | `iteration: integer`, `stdout: string`, `calls_made: integer`, `answer_ready: boolean`, `answer_content_chars: integer` |
| `execution_error` | `iteration: integer`, `error_type: string`, `message: string` |
| `reasoning_delta` | `text: string` |
| `subcall` | `phase`, `call_id`, `operation`, `iteration`, plus phase-specific reservation/checkpoint/error and optional batch metadata |
| `repair` | `phase: "start"|"completion"|"failure"`, `kind: "missing_code"|"execution_error"|"terminal"`, `iteration`, `error?` only on failure |
| `extract` | `phase: "start"|"completion"|"failure"`, `iteration`, `extract_error?` only on failure |
| `result`, `replay` | `result: object` |
| `usage` | `kind: "resolved"|"partial"`, `root: object`, `subcall: object`, `unattributed: object`, `total_tokens: integer`, `wall_time_ms: integer` |
| `budget` | `kind: "snapshot"|"mutation"`, `source: string`, plus the kind-specific fields below |
| `policy` | `contract_enforced: boolean`, `outcome: string`, `violation_type: string|null` |
| `capability` | `outcome: object` in the exact `CapabilityResult.to_trace_dict()` shape |
| `done` | `status: "success"|"error"|"cancelled"`, `ready`, `extracted`, `iterations`, and the content-free reconciliation objects/errors |

The canonical `result.result` fields are `answer`, `answer_metadata`, `ready`,
`iterations`, `tokens_used`, `subcalls`, `successful_subcalls`, `extracted`,
`error`, `extract_error`, `recovered_error`, and `prompt_pack`. It deliberately
has no trajectory or `run_record`. A retained `replay.result` adds the full
trajectory; each `llm_input` remains a structured message list.

The durable `done` value contains content-free terminal status, typed error
presence, and reconciled usage/budget/policy facts. It never copies answer,
code, output, trajectory, error messages, error details, or executed source.
The configurable `replay` value is the complete result snapshot for hosts that
explicitly retain replay content. Its `result.usage` is the same resolved or
partial projection emitted durably in `usage` and `done`, including both cache
token classes.

Billing consumes the durable `usage` value. Both `root` and `subcall` carry
`input_tokens`, `cache_read_tokens`, `cache_creation_tokens`, `output_tokens`,
`total_tokens`, `requests`, `successes`, and `complete`. Cache tokens are
disjoint breakdowns inside inclusive input totals; complete scopes therefore
require their sum not to exceed `input_tokens`. They are not added again to
`input_tokens` or `total_tokens`. The top-level kind is `resolved` exactly when
both scopes are complete; otherwise it is `partial`. Partial usage preserves
known provider counts without substituting conservative budget reservations for actual usage.
Legacy/custom-client tokens that cannot be assigned safely appear under
`unattributed.total_tokens`; the three token scopes must sum to
`total_tokens`. None of these facts is inferred by counting stream events.

The budget body remains a discriminated event in Trace ABI v4. The terminal snapshot uses
`kind="snapshot"`, `source="budget_ledger"`, and `configured`, `consumed`,
and `remaining` objects. The ledger may emit any number of
`kind="mutation"` values with `action` (`reserve`, `commit`, `refund`, or
`exhaust`), `resource`, non-negative `amount`, and optional `call_id`.
Consumers must not assume exactly one mutation.

```python
from droste import DataUseAuthorization, RLMConfig, TraceRetentionPolicy

config = RLMConfig(
    trace_retention=TraceRetentionPolicy(
        retain=frozenset({"code", "output", "repair", "replay"}),
        policy_id="local-training-corpus-v1",
        expires_at="2026-10-14T00:00:00Z",
        host_managed_expiry=True,
    ),
    data_use=DataUseAuthorization(
        training_allowed=True,
        authorization_ref="consent://training/trace-policy-1",
        purposes=frozenset({"training"}),
    ),
    on_run_record=save_record_locally,
)
```

Retention never grants training permission. `DataUseAuthorization` defaults to
denied and an empty purpose set. Training requires both an auditable
`authorization_ref` and the explicit `training` purpose. `expires_at` is an
absolute, timezone-aware timestamp and is valid only with
`host_managed_expiry=True`: Droste records the policy fact while the local host
owns expiry/deletion. A local host can serialize `record.as_dict()` to its own
protected storage; cloud transport and governance integrations remain host
concerns.

When `run_rlm` creates its context, trace settings come from `RLMConfig`. When
a caller injects an existing `ExecutionContext`, that context owns trace
identity, retention, data-use authorization, and the record sink because it
may already contain shared accounting/events. Explicit conflicting config
values fail instead of mutating a live sequence.

## Capability outcomes

The broker owns immutable capability call/result values and their content-free
accounting/evidence projection. Trace observation wraps only that broker-owned
projection as the `outcome` of a durable `capability` event; full parameters,
results, messages, and free-form evidence never enter the durable value. Trace
code does not copy the capability schema, dispatch calls, or participate in
authorization. This keeps tracing observational and gives native and Pyodide
paths the same identity and evidence facts.

## Subcall, repair, and extract lifecycles

`subcall` is the live and retainable projection of the broker-owned attempt;
it is not another dispatch or accounting protocol. `start` carries the exact
broker reservation. `progress` carries a trusted cumulative checkpoint.
`completion` and `failure` carry the last accepted checkpoint; failure adds
only the stable provider error code and type. Prompts, contexts, results, and
provider error messages never enter this event.

Every subcall body carries the loop `iteration`; run nesting remains the
envelope's `depth`/`parent_run_id`. The broker `call_id` is the attempt identity,
while envelope `seq` is the only event order. An atomic `llm_batch` or
`llm_batch_with_errors` reports `batch_count`; its `call_id` is already the
batch identity. Droste does not synthesize item completion or IDs when the
atomic transport cannot observe them.

`repair` and `extract` use closed phase discriminators instead of progress
strings. Missing-code, execution-error, and terminal repair paths each emit a
start and exactly one completion or failure once entered. Extract fallback does
the same; a failure carries the typed `extract_error`. The canonical `result`
still carries the unary-equivalent answer and `done` remains content-free.

## Terminal reconciliation

Finalization emits resolved `usage`, `budget`, and `policy`; always delivers the
canonical unary-equivalent `result`; optionally emits the complete `replay`
snapshot when replay retention is selected; then emits durable `done`.
`RunRecord` retains the selected subset without
renumbering it, so gaps truthfully show discarded transient or configurable
values. Its terminal projection must equal the body of its final `done` event,
and its usage totals reconcile with `RLMResult`.

## Released conformance corpus

Droste 0.15.1 and later ship the authoritative fixture bytes inside the wheel
and sdist. Python consumers load them through package resources:

```python
from droste.testing import (
    runner_v8_refusal_ndjson,
    trace_v4_execution_ndjson,
    trace_v4_lifecycle_ndjson,
)

execution_lines = trace_v4_execution_ndjson().splitlines()
event_lines = trace_v4_lifecycle_ndjson().splitlines()
pre_admission_refusal = runner_v8_refusal_ndjson()
```

The compact execution NDJSON contains a two-iteration root trace plus a
depth-one child trace. Its producer-stamped `llm_response`, `code`, successful
`output`, and `execution_error` events let consumers test iteration and depth
projection.
The successful root output deliberately starts with `ERROR:` so consumers must
use the event discriminant rather than interpreting stdout prose as status.

The lifecycle NDJSON contains five contiguous runs whose per-run sequences
restart at one: ordinary success, successful and failed extract fallback, loud
output-limit failure, and cancellation. Together they cover unary and
atomic-batch subcall outcomes with stable call attribution, structured
execution error, repair and extract completion/failure, and exact
`result`/`usage`/`budget`/`policy`/`done` reconciliation. An output-limit
failure has no shortened `output` event:
Droste rejects oversized stdout rather than presenting clipped content as a
successful value. `stdout_chars` preserves the rejected size in the terminal
result.

A runner protocol refusal occurs before admission, so it has no `run_id`, no
`RunRecord`, and no Trace ABI event sequence. Its fixture lives beside the
corpus so hosts test the boundary explicitly; an event relay must reject those
bytes. The GitHub release's `droste-relay-vX.Y.Z.tar.gz` exposes these same
files under `conformance/` for non-Python consumers. Consumers should pin the
release, read the fixture bytes, and use the released Python parser or Deno
filter rather than copying either schema.
