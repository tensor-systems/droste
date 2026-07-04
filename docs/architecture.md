# Architecture

## The loop

```
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
the sandbox executes it; stdout feeds the next iteration. Two functions
bridge back to language models:

- `llm_query(prompt)` — one subcall.
- `llm_query_batched(prompts)` — concurrent subcalls (bounded workers),
  order-preserving. Aliases: `llm_batch`, `batch_llm_query`.

The loop ends when the model sets `answer["ready"] = True`, or the iteration
budget runs out — in which case, when the trajectory contains executed work,
a single **extract pass** produces a best-effort answer from it, flagged `extracted=True` and never
presented as confirmed. Policy violations withhold gated content from the
final answer rather than silently passing it through.

## Budgets and cost

Everything expensive is bounded and explicit: `max_iterations`, `max_calls`
(subcalls; counted only when issued, enforced under a lock), `max_depth`,
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
(`SOURCE_PROTOCOL_VERSION`).

The bundled SQLite source is local-mode: SELECT-only policy gate (single
statement, masked-identifier keyword scanning, LIMIT injection, row caps,
statement timeout), database opened `mode=ro`, and `PRAGMA query_only=ON`
enforced on host-supplied connections.

## The runner protocol (embedding)

`python -m droste_runner` is a one-shot JSON worker: a host writes a request
file (question, budgets, endpoints or context, declarative data sources) and
reads a response (answer, `ready`, `extracted`, iterations, subcalls,
trajectory, usage). This is how non-Python hosts embed the engine.

**Compatibility window**: the request/response schema and the source
protocol are versioned. Fields are added optional-first; a control plane
that serves embedded engines must tolerate a stated window of engine
versions (embedded engines in the field can't be force-upgraded). Breaking
changes require a protocol version bump.

## Sandboxing — the honest version

The REPL is a **guardrail, not a security boundary**. It bounds output size
and execution time and keeps well-behaved models on the rails. A hostile
workload needs real isolation, which belongs to the host: OS permissions
own file access (the SQL source's read-only policy assumes this), and
process/container/WASM isolation owns escape (hosts run the engine in a
subprocess, jail, or Pyodide/WASM runtime as their threat model requires).
The engine will not pretend otherwise, and neither should you.
