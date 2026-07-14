# Budgets

Droste authorizes compute with one immutable value:

```python
Budget(
    tokens=500_000,
    subcalls=50,
    depth=1,
    wall_ms=300_000,
    root_output_tokens=4_096,
    subcall_output_tokens=2_048,
)
```

The caller resolves presets or product policy before constructing this value.
Enforcement never interprets presets and never merges host-specific counters.
`SandboxLimits` is separate: output capture and local code timeouts describe
the REPL boundary, not provider/model spend.

## One mutable authority

Each run has one `BudgetLedger`. A root request or brokered capability reserves
its maximum token/subcall/wall/depth vector atomically before dispatch. If any
dimension cannot fit, the reservation is rejected without mutation as a typed
`BudgetExhausted` fact. On completion, the same `call_id` commits conservative
actual spend and refunds the unused reservation.

Provider transports do not enforce budgets. They issue requests and report
mechanism usage. The capability broker is the admission/finalization boundary,
so handler errors, invalid results, annotator errors, and process-control exits
all settle the reservation exactly once.

Trusted handlers receive a frozen `CapabilityExecutionContext`. Its reservation
is identified only by the call's `call_id`; providers never receive the ledger.
Long-running work may report cumulative token/subcall usage with
`context.checkpoint(tokens=..., subcalls=...)`. Equal values are idempotent,
dimensions cannot move backward, and values cannot exceed the admitted vector.
Checkpoint deltas become committed ledger facts immediately; final
reconciliation cannot retract them and refunds only authorization still unused.
Wall time is always measured by the broker rather than reported by a provider.

`context.check()` observes cooperative cancellation and the caller-authorized
monotonic deadline. Cancellation before handler dispatch refunds the admission;
after dispatch it follows the same deterministic final settlement as any other
attempt. Results distinguish the stable `cancelled` and `deadline_exceeded`
error codes under one `CapabilityStatus.CANCELLED` terminal class.

Inference batches reserve every item together. Concurrent callers therefore
cannot pass separate check-then-increment races. The broker preserves one root
output allocation while admitting inference work so early subcalls cannot
consume the parent's ability to synthesize.

## Child runs

`ledger.child(call_id, child_budget)` creates a strict sub-ledger only after the
parent atomically reserves the child's token and subcall allocation plus one
depth unit. The child's own deadline must fit within the parent's shared
remaining deadline; wall time is not additive capacity. The child may spend no
more than its budget. `child.close()` reconciles its actual spend into the
parent and refunds everything unused. Closing is idempotent; closing with active
reservations fails loudly.

## Trace facts

Every mutation is a durable Trace ABI v1 `budget` event from
`source="budget_ledger"`: `reserve`, `commit`, `refund`, or `exhaust`.
Mutation events carry `resource`, non-negative `amount`, and `call_id`. The
terminal snapshot records the configured, consumed, and remaining vectors.
Event callbacks run outside the ledger's state lock; event observation cannot
become a second accounting authority.

## Runner and CLI

Runner protocol v4 requires an exact `budget` object with all six fields.
Missing and unknown fields fail before work. The CLI resolves its six budget
flags into the same value. There are no legacy `max_*` aliases or translation
rules.
