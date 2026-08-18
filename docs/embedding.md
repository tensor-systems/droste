# Embed Droste

The `droste` package has no runtime dependencies. An embedded run needs a root
model client, a subcall client, an execution environment, and one shared
execution context.

```bash
uv add droste
```

This example uses an OpenAI-compatible endpoint and a string as the dataset:

```python
from droste import (
    Budget,
    EnvironmentConfig,
    OpenAICompatClient,
    OpenAICompatSubcallClient,
    RLMConfig,
    create_environment,
    create_environment_context,
    run_rlm,
)

model = "YOUR_MODEL_ID"
environment_config = EnvironmentConfig(
    kind="native",
    budget=Budget(subcalls=50, depth=1),
)
context = create_environment_context(environment_config)

root = OpenAICompatClient(model=model)
subcalls = OpenAICompatSubcallClient(model=model, context=context)
environment = create_environment(
    environment_config,
    context="your source data",
    registry=None,
    subcalls=subcalls,
    execution_context=context,
)

result = run_rlm(
    "What happened?",
    environment=environment,
    root_llm=root,
    subcalls=subcalls,
    config=RLMConfig(root_model=model),
    context=context,
)
print(result.answer)
```

`OpenAICompatClient` reads `OPENAI_API_KEY` and `OPENAI_BASE_URL`. Explicit
constructor arguments take precedence. The package also provides
`AnthropicClient` and `AnthropicSubcallClient`.

## Data providers

Pass plain data through `context`, or bind a `ProviderRegistry` for named data
sources. Droste includes filesystem text and read-only SQLite providers. See
[provider manifests](provider-manifests.md) when exposing an application data
source or MCP server.

## Budgets and concurrency

`EnvironmentConfig` and its execution context must share one `Budget`. The
same ledger reserves and settles root calls, subcalls, child runs, and wall
time.

Subcall clients default to five concurrent batch items. If you change
`OpenAICompatSubcallClient(max_parallel=...)`, set the same value in
`RLMConfig.rollout`. A mismatch fails before inference.

See [budgets](budgets.md) for the full authorization model.

## Policy and traces

Droste does not infer enforcement rules from the wording of a question. Pass
explicit `PolicyHints` when a product requires semantic subcalls or another
answer contract.

Attach `on_event` or configure trace retention through the execution context
to consume structured events. The [Trace ABI](trace-abi.md) defines the event
and retention contracts.

Hosts that run Droste through HTTP or Pyodide should use the runner protocol in
the [technical architecture](architecture.md#the-runner-protocol-embedding).
