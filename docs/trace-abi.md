# Trace ABI v1

Droste exposes one append-only event stream and one policy-resolved terminal
`RunRecord`. The engine creates values; it does not choose a database or write
files. Hosts attach `on_event` for live delivery and optionally
`on_run_record` as their persistence I/O shell.

## Event envelope

Every event is a strict v1 value with these fields:

```json
{
  "run_id": "e0f7...",
  "seq": 1,
  "timestamp": "2026-07-14T05:00:00Z",
  "type": "progress",
  "version": 1,
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
| `configurable` | `iteration_start`, `llm_response`, `code`, `output`, `execution_error`, `finalization_error`, `extract_error`, `repair`, `result`, `replay` | Included only when named by `TraceRetentionPolicy.retain` |
| `transient` | `startup`, `progress`, `reasoning_delta` | Live delivery only; never in the terminal record |

Retention governs the terminal record, not the live channel. `result` is
always delivered once before `done`, even when it is not retained. `replay` is
different: it is emitted only when the host explicitly selects replay
retention.

## Exhaustive v1 bodies

Every event body has a fixed top-level schema. Optional fields are marked `?`.
Objects named below are JSON objects; all other types are primitive.

| Type | Body |
| --- | --- |
| `startup` | `engine_version: string`, `runner_protocol?: integer|null`, `source_protocol?: integer|null` |
| `progress` | `status: string` |
| `iteration_start` | `iteration: integer`, `max_iterations: integer` |
| `llm_response` | `iteration: integer`, `response: string` |
| `code` | `iteration: integer`, `code: string` |
| `output` | `iteration: integer`, `stdout: string`, `calls_made: integer`, `answer_ready: boolean`, `answer_content_chars: integer` |
| `execution_error` | `iteration: integer`, `error_type: string`, `message: string` |
| `reasoning_delta` | `text: string` |
| `finalization_error`, `extract_error` | `error_type: string`, `message: string` |
| `repair` | `iteration: integer`, `reason: string`, `error_type?: string` |
| `result`, `replay` | `result: object` |
| `usage` | `kind: "resolved"`, `root: object`, `subcall: object`, `unattributed: object`, `total_tokens: integer`, `wall_time_ms: integer` |
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
explicitly retain replay content.

Billing consumes the durable `usage` value. Both `root` and `subcall` carry
`input_tokens`, `output_tokens`, `total_tokens`, `requests`, and `successes`.
Legacy/custom-client tokens that cannot be assigned safely appear under
`unattributed.total_tokens`; the three token scopes must sum to
`total_tokens`. None of these facts is inferred by counting stream events.

Budget v1 is a discriminated event. A pre-ledger compatibility snapshot uses
`kind="snapshot"`, `source="legacy_execution_stats"`, and `configured`,
`consumed`, and `remaining` objects. A ledger may emit any number of
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

## Terminal reconciliation

Finalization emits resolved `usage`, `budget`, and `policy`; always delivers the
canonical unary-equivalent `result`; optionally emits the complete `replay`
snapshot when replay retention is selected; then emits durable `done`.
`RunRecord` retains the selected subset without
renumbering it, so gaps truthfully show discarded transient or configurable
values. Its terminal projection must equal the body of its final `done` event,
and its usage totals reconcile with `RLMResult`.
