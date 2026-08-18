# Architecture overview

Droste keeps the source data in an execution environment instead of placing it
in the root model prompt. On each iteration, the root model writes Python. The
environment runs that code and returns only its printed output.

```text
question -> root model -> Python -> execution environment -> printed output
               ^                                              |
               +---------------- next iteration ---------------+
```

The generated program can use ordinary Python for exact work and call a
smaller model for bounded interpretation:

- `llm_query(prompt)` makes one model call.
- `llm_query_batched(prompts)` evaluates an ordered batch concurrently.
- `llm_batch_json(prompts, schema, ...)` validates structured batch results.

The host decides which data and capabilities the program receives. Built-in
sources cover text files and SQLite. Provider manifests and the capability
broker allow hosts to expose additional sources without giving generated code
direct access to transport clients.

One run-scoped ledger authorizes tokens, subcalls, depth, wall time,
iterations, and output ceilings. A separate sandbox configuration limits local
execution and captured output. Every run can emit a versioned event stream for
live progress, auditing, and replay.

See the [technical architecture](architecture.md) for the complete broker,
runner, provider, budget, sandbox, and trace contracts.
