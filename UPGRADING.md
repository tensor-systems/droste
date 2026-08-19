# Upgrading Droste

This guide covers upgrades to 0.25 from the two preceding minor releases. For
older installations, upgrade one minor version at a time using the corresponding
GitHub release notes before applying this guide. Runner and strict Trace ABI
consumers must upgrade their decoders and packaged fixtures with the engine.

## 0.25.0 from 0.24.x

### Trace ABI v10 reports degraded execution

Results and terminal `done` events now include `degradations`. An empty list is
a clean run. Each item identifies the callback site, error type, bounded detail,
and consequence of work that continued without it.

Hosts should surface, log, or reject non-empty degradations according to what
was lost. The affected callbacks remain observational and do not terminate the
run.

Update strict consumers and replace `trace_v9_*` helpers and `trace-v9-*`
fixtures with their `v10` equivalents.

## 0.24.0 from 0.23.x

### Trace ABI v9 reports armed ready gates

The optional `startup.ready_gates` array names the ready-time enforcement
enabled for the run: `policy_hints`, `ready_metadata_validator`, and/or
`ready_answer_validator`. An empty array means no gate was armed.

Hosts that require enforcement should assert the expected gates rather than
assuming configuration from a successful result.

`ready_answer_validator` receives `ReadyAnswerState`, which includes metadata,
answer content, iteration, calls made, and successful calls. The older
metadata-only validator remains supported; both run when both are configured.

Update strict consumers and packaged fixtures to Trace ABI v9.

## 0.23.0 from 0.22.x

### Trace ABI v8 adds provider-call heartbeat

The transient `heartbeat` event reports `elapsed_ms` every 15 seconds while a
provider call remains outstanding. It is liveness only: it is never retained
and does not affect budget accounting.

No-progress watchdogs should count `heartbeat` and `reasoning_delta` as
activity. The relay cannot emit a heartbeat while generated WASM code itself is
blocked, so the event does not mask an interpreter wedge.

Update strict consumers and packaged fixtures to Trace ABI v8.
