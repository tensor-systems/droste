# Embed Droste

The dependency-free `droste` package can run with your model clients and data.
Install it with `uv add droste`, then start from the tested
[embedding example](../examples/embedding.py).

An embedded run has four shared pieces: a `Budget`, an `ExecutionContext`, model
clients, and an execution environment. Construct the context and environment
from the same `EnvironmentConfig`; conflicting budgets fail before inference.

`OpenAICompatClient` reads `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
`ModelRelayClient` and `ModelRelaySubcallClient` read `MODELRELAY_API_KEY` and
`MODELRELAY_BASE_URL`. Anthropic clients are also included. Explicit constructor
arguments override environment values.

## Compute and local execution limits

`Budget` is the model-compute authorization:

```python
Budget(
    tokens=500_000,
    subcalls=50,
    depth=1,
    wall_ms=300_000,
    max_iterations=30,
    root_output_tokens=4_096,
    subcall_output_tokens=2_048,
)
```

One `BudgetLedger` atomically reserves and reconciles root calls and brokered
subcalls. Failed work still settles its reservation. `depth` authorizes child
ledgers for hosts that construct child runs; Droste's built-in model-facing
helpers currently make only flat `llm_query` and batch subcalls.

`SandboxLimits` is separate. It bounds local execution time and captured
output; it is not provider/model spend and the native REPL is not a security
boundary.

Built-in subcall clients default to five concurrent batch items. If you change
their `max_parallel`, record the same value in `RLMConfig.rollout`; a mismatch
fails before the first model request.

## Data, policy, and events

Pass plain data through `context`, or use a `ProviderRegistry` for named data
sources. Droste includes read-only SQLite and filesystem-text providers; see
[Providers](providers.md).

Droste does not infer enforcement rules from question text. Products that need
semantic-subcall or answer requirements must pass explicit `PolicyHints` or a
ready-time validator.

Attach `on_event` for live events and configure `TraceRetentionPolicy` for the
terminal record. See the [Trace ABI](reference/trace.md). Non-Python hosts use
the [runner protocol](reference/runner.md); the Pyodide host path is documented
beside its implementation in [pyodide/README.md](../pyodide/README.md).
