# Trace ABI v10

Droste exposes one append-only live event stream and one policy-resolved
terminal `RunRecord`. The engine creates these values; hosts choose delivery
and persistence.

Every event has `run_id`, positive monotonic `seq`, UTC `timestamp`, `type`,
`version: 10`, `persistence_class`, `depth`, optional `parent_run_id`, and its
type-specific body. `parse_event()` rejects unsupported versions, unknown or
missing fields, invalid classifications, and invalid terminal shapes.

## Retention classes

| Class | Event types | Terminal record |
| --- | --- | --- |
| Durable | `usage`, `budget`, `policy`, `capability`, `done` | Always retained |
| Configurable | `iteration_start`, `llm_response`, `code`, `output`, `execution_error`, `subcall`, `repair`, `extract`, `result`, `replay`, `checkpoint` | Retained only when selected |
| Transient | `startup`, `progress`, `reasoning_delta`, `usage_progress`, `heartbeat` | Live only |

Retention does not control ordinary live delivery. `result` is always delivered
before `done`; `replay` is emitted only when replay retention is selected.
Training authorization is separate and denied by default.

## Current event contract

- `startup` identifies engine, runner, provider, and scaffold versions and may
  list the armed `ready_gates`.
- `heartbeat` reports content-free provider-call liveness as `elapsed_ms`.
- `checkpoint` carries the current draft and optional opaque host payload.
- `subcall`, `repair`, and `extract` use closed phase discriminators and exactly
  one terminal completion or failure after their start.
- `usage_progress` is a cumulative snapshot emitted only after numeric provider
  usage settles; it is never inferred from visible output.
- `capability` is the broker's content-free trace projection. Parameters,
  results, error messages, and evidence locators are excluded.
- `done` includes terminal status, readiness, extraction, `degradations`, usage,
  budget, policy, retention, typed error presence, scaffold identity, and total
  stdout characters. It never includes answer, code, stdout, or error details.

Root and subcall usage scopes separately report inclusive input, cache-read,
cache-creation, output, and total tokens plus request/success counts and
completeness. The top-level kind is `resolved` only when both scopes are
complete; otherwise known counters are preserved as `partial` evidence.

## Ordering and reconciliation

An execution context serializes stamping and callback delivery. Event `seq` is
the only event order; broker `call_id` identifies an attempt, not an event
sequence or durable idempotency key.

Finalization emits `usage`, `budget`, and `policy`; delivers `result`; may emit
`replay`; and ends with `done`. The terminal record's final projection must
equal that `done` body. Retention can leave sequence gaps but never renumbers
events.

## Conformance corpus

The wheel and sdist include the authoritative strict fixture bytes. Consumers
should use these instead of copying the prose schema:

```python
from droste.testing import (
    runner_v10_refusal_ndjson,
    trace_v10_execution_ndjson,
    trace_v10_lifecycle_ndjson,
)
```

The runner refusal fixture is intentionally not a trace event. The two Trace
ABI fixtures cover representative execution and lifecycle phases, partial and
resolved usage, retention, and terminal reconciliation.
