# Architecture

## The loop

```

The harness strategy is resolved once at run start as one immutable
[prompt pack](prompt-packs.md). Package/file I/O ends at the loader boundary;
slot validation, `(model, profile)` fallback, and rendering are pure value
transformations. The engine never merges partial packs or changes strategy
mid-run.
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

The loop ends when the model sets `answer["ready"] = True`, or the iteration
budget runs out — in which case, when the trajectory contains executed work,
a single **extract pass** produces a best-effort answer from it, flagged `extracted=True` and never
presented as confirmed. Policy violations withhold gated content from the
final answer rather than silently passing it through.

Policy is caller-supplied rather than inferred from question text. In-process
callers opt into semantic enforcement with
`RLMConfig(policy_hints=PolicyHints(semantic=True))`; without that hint,
structured batch results remain prompt-driven and the loop does not add this
completeness gate.

## Budgets and cost

Everything expensive is bounded and explicit: `max_iterations`, `max_calls`
(subcalls; attempted calls are counted only when issued and successful calls
are tracked separately, both enforced under a lock), `max_depth`,
per-subcall `max_output_tokens` (default 2048), and `reasoning_effort`
passthrough. Subcall usage is added to `result.tokens_used`. The defaults
encode a measured lesson: unbounded subcall output (thinking tokens
especially) is where RLM runs go from cents to dollars.

## Data sources (registry)

Hosts register source *types* at startup — never from a request:

```python
from droste.sources.sql_local import register
register()  # exposes type "sql": {"sqlite_path": ..., "policy": {...}}
```

A source declares capabilities (`{sql, schema}`, `{search}`, ...) and
becomes a named variable in the REPL (`db.query("SELECT ...")`). Requests
stay declarative: they can name a registered type and its config, never code
to import. The registry rejects reserved names and protocol mismatches
(`SOURCE_PROTOCOL_VERSION`; registrants pass the version they implement).

The protocol is **domain-blind**: core verbs only, plus the generic
optionals `find`/`content`/`sample`. Domain-specific verbs are declared by
the source itself via an `extra_methods` attribute (a tuple of method
names); the registry binds exactly those callables into the sandbox —
validated against engine verbs, reserved globals, and Python builtins,
and a flattened default-source verb additionally may not collide with
another source's namespace — and the bridge's `DataSourceService` honors
the same declaration, so a source behaves identically in-process and
across the Pyodide bridge.

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
the unchanged `execution_result` text; consumers must not infer status from an
output prefix.

The Python implementation is split by ownership: `droste.environments`
provides the generic native in-process environment; `droste_runner` keeps
HTTP clients, remote source transport, envelope helpers, and orchestration in
`http_clients`, `sources`, `protocol`, and `run` modules respectively.
`droste_runner.runner` is a compatibility facade for existing embedders, not a
second implementation. Both successful and refused envelopes are shaped by
`protocol.build_response`, so their shared fields cannot drift.

**Compatibility window**: the request/response schema and the source
protocol are versioned, each by a single integer:

- `RUNNER_PROTOCOL_VERSION` (currently 1) governs the request/response
  envelope. Every request **must** carry `protocol_version` — requests are
  self-describing, the same discipline as JSON-RPC's mandatory `"jsonrpc"`
  field. A missing or mismatched version is answered with a structured
  error (`protocol_version_missing` / `protocol_version_mismatch`) naming
  both sides — a host detects incompatibility explicitly instead of
  failing on a missing field. Responses carry `protocol_version`: the
  engine stamps its own everywhere except an `adapter_module` response
  that already claimed one (adapters own their response shape).
- `SOURCE_PROTOCOL_VERSION` (currently 2; v2 made the contract
  domain-blind — domain verbs are source-declared `extra_methods`, no
  longer auto-bound) governs the data-source registration contract and
  fails at startup, not per-request.

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
