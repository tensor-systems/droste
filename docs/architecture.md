# Architecture

Droste keeps source data in an execution environment instead of copying it into
the root model prompt. The root model writes Python, the environment executes
it, and only bounded stdout returns for the next iteration.

```text
question ──▶ root model ──▶ Python ──▶ environment ──▶ stdout
               ▲                                      │
               └─────── next iteration ◀──────────────────┘
```

The loop stops when the model sets `answer["ready"] = True`, reaches its
iteration limit, or cannot reserve more budget. A terminal extraction call may
return a best-effort answer from completed work; extracted answers remain
distinct from confirmed answers.

## Components

- **Prompt pack:** one immutable strategy artifact selected before inference.
  It defines the initial, refinement, repair, and extraction templates.
- **Execution environment:** a persistent Python namespace containing the
  supplied context and generated capability bindings.
- **Capability broker:** the only route from generated bindings to trusted
  subcall or data-provider handlers. It owns allowlisting, attempt lifecycle,
  cancellation, and accounting integration.
- **Budget ledger:** the single mutable authority for tokens, subcalls, depth,
  wall time, iterations, and root/subcall output ceilings.
- **Scaffold manifest:** the content-addressed identity of the resolved prompt,
  capabilities, inference settings, budget, sandbox, and protocol versions.
- **Trace context:** stamps and serializes the versioned live event stream and
  policy-selected terminal record.

Generated code can call `llm_query`, ordered batch helpers, and any bindings
created by the provider registry. These are flat subcalls within the current
run; Droste does not expose a built-in model-facing child-RLM helper.

## Trust boundaries

Provider handlers are trusted host code. They receive a frozen
`CapabilityExecutionContext` with cooperative `check()` and cumulative
`checkpoint()` methods, not the mutable ledger or trace recorder. The broker
normalizes every attempted outcome and settles it exactly once.

The native REPL is a guardrail, not a security boundary. It limits execution
time and captured output, but arbitrary Python may retain the ambient authority
of its process. Hosts that execute untrusted workloads must add OS, container,
or WASM isolation. The Pyodide substrate keeps credentials and provider data in
trusted host/interpreter contexts and exposes only brokered calls to generated
code.

Policy is explicit. Droste does not infer semantic or count requirements from
question wording; callers arm policy hints or ready-time validators when a
product requires them.

## Versioned boundaries

Current host-facing contracts are documented separately:

- [Providers](providers.md)
- [Runner protocol](reference/runner.md)
- [Trace ABI](reference/trace.md)
- [Prompt packs and RLM skills](reference/prompt-packs.md)
- [Scaffold manifest](reference/scaffold.md)

Those references describe current behavior only. Migration history belongs in
[UPGRADING.md](../UPGRADING.md).
