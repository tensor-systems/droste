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
  "depth": 1,
  "status": "Iteration 1/20: Generating code..."
}
```

`seq` is positive and monotonic within one `run_id`. `parent_run_id` and
`depth` describe nested runs without making identity depend on mutable loop
state. `parse_event` is the authoritative parser and rejects partial,
misclassified, or unsupported-version envelopes.

Relay-owned Pyodide telemetry uses a child `run_id` whose `parent_run_id` is
the engine run. The relay and engine therefore each own one sequence; their
events correlate without two writers racing to stamp the same counter.

## Retention is data selection

The persistence class is exhaustive and fixed by event type:

| Class | Event types | Rule |
| --- | --- | --- |
| `durable` | `usage`, `budget`, `policy`, `capability`, `done` | Always in the terminal record |
| `configurable` | `iteration_start`, `llm_response`, `code`, `output`, `execution_error`, `subcall`, `extract_error`, `repair`, `replay` | Included only when named by `TraceRetentionPolicy.retain` |
| `transient` | `startup`, `progress`, `reasoning_delta`, `heartbeat` | Live delivery only; never in the terminal record |

The durable `done` value contains content-free terminal status, typed error
presence, and reconciled usage/budget/policy facts. It never copies answer,
code, output, trajectory, error messages, error details, or executed source.
The configurable `replay` value is the complete result snapshot for hosts that
explicitly retain replay content.

Billing consumes the durable `usage` value: `kind="resolved"`, structured
token and subcall totals, and measured wall time. The durable `budget` value
records configured, consumed, remaining, and mutation summaries. Neither is
derived by counting progress events. `done` repeats the same resolved values
and names the configurable event types retained, so reconciliation is direct.

```python
from droste import DataUseAuthorization, RLMConfig, TraceRetentionPolicy

config = RLMConfig(
    trace_retention=TraceRetentionPolicy(
        frozenset({"code", "output", "repair", "replay"})
    ),
    # Independent, host-supplied authorization. It defaults to False.
    data_use=DataUseAuthorization(training_allowed=False),
    on_run_record=save_record_locally,
)
```

Retention never grants training permission. `DataUseAuthorization` is a
separate host value and defaults to denied even when replay content is
retained. A local host can serialize `record.as_dict()` to its own protected
storage; cloud transport and governance integrations remain host concerns.

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

Finalization emits resolved `usage`, `budget`, and `policy`; emits the complete
configurable `replay` snapshot only when replay retention is selected; then
emits durable `done`. `RunRecord` retains the selected subset without
renumbering it, so gaps truthfully show discarded transient or configurable
values. Its terminal projection must equal the body of its final `done` event,
and its usage totals reconcile with `RLMResult`.
