<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/droste-dark.svg">
  <img src="docs/assets/droste.svg" alt="Droste — nested frames converging on the answer" width="96">
</picture>

# Droste

**A recursive analysis engine for data too large for a context window.**

Droste implements the Recursive Language Model (RLM) technique. Rather than
placing an entire corpus in the root model's context, it exposes the corpus
through a sandboxed Python REPL. The model writes programs over that data and
delegates bounded semantic judgments through `llm_query` and
`llm_query_batched`.

## Not a general-purpose agent

General coding and tool agents choose actions across open-ended tasks. When
they retrieve material, the observations return through the model's
conversation context. Droste has a narrower job: run model-written programs
over a corpus, use Python or SQL for exact computation, and send only selected
semantic work to model subcalls. Its program may print excerpts into context,
but the full corpus does not have to pass through the context window.

This separation keeps the mechanism explicit: code locates and aggregates;
subcalls interpret bounded inputs; the root model assembles the answer within
configured iteration, subcall, and output limits.

```bash
uvx droste "which customer had a failed charge, and why?" server.log
uvx droste "which plan has the highest refund rate vs its MRR?" shop.db
uvx droste "how do the authentication flows differ?" ./docs
```

![droste answering a two-part question over a 444 kB server log, streaming its code as it works](docs/assets/demo.gif)

The first example runs against a 444 kB log:

```
$ droste "Which customer had a failed charge, for what amount, and why?
  How many timeout errors are there, and which upstream do they blame?" server.log

1. **Failed Charge Details**:
   - **Customer**: `cus_9982`
   - **Amount**: 1499 (USD, which is $14.99)
   - **Reason**: The card was declined due to insufficient funds
     (`reason=card_declined decline_code=insufficient_funds`).

2. **Timeout Errors**:
   - **Count**: There are exactly 66 timeout errors in the log.
   - **Upstream blamed**: They blame `payments-v2` (`upstream=payments-v2`).
```

The counts are exact because the model *counted them in Python* — it never
read 3,400 log lines through its attention. In `--db` mode the model
introspects your schema, writes read-only SQL, and computes over the rows;
in the demo above it noticed the free plan makes refund-rate-vs-MRR
undefined and answered for the paid plans instead.

## Why this structure

Mechanical work stays mechanical: regex and SQL find *where*, model subcalls
interpret *what*, and code combines the results. The root can inspect the
shape of the corpus, narrow it without model calls, and fan out only when a
step requires semantic judgment.

Execution is bounded by explicit iteration, subcall, and output limits. Root
and subcall models can be configured independently. These controls make the
work observable and limitable; they are not a promise of a particular answer
quality, latency, or price, which depend on the data, models, and endpoint.

## Reproducible evidence

The repository ships a [versioned benchmark harness](benchmarks/README.md),
immutable per-task artifact schemas, deterministic scorers, and report
generation. A zero-cost smoke run checks the artifact and reporting path
without making network calls:

```bash
output="$(mktemp -d)/droste-benchmark-smoke"
uv run python -m benchmarks smoke --output "$output"
```

The smoke run validates the machinery, not model quality. Today the
[suite manifest](benchmarks/manifests/rlm-paper-v1.json) has one ready dataset:
a pinned 50-task, 131K-token OOLONG slice that must be materialized from its
public source. Its live arms remain blocked until a public model configuration
and immutable results are published; the other named benchmark families remain
`planned`. This README therefore does not present score, cost, or latency
numbers as reproducible results. Publishing the configuration, artifacts, and
reports is tracked in [#81](https://github.com/tensor-systems/droste/issues/81).

## Use it

Ask questions over files, folders, and SQLite from the terminal. The
contract: **args that exist are data, the one that doesn't is the question,
no args means the current directory, pipes are data too — and it always
prints one line saying what it read.**

```bash
uvx droste "…" ./docs        # zero-install, npx-style
uv tool install droste       # or keep the binary around
pipx install droste          # the older equivalent
```

```bash
droste login                 # one-time setup: free credits, or your own key
droste "what changed between these?" report.txt logs.txt
droste "which customers churned last month?" app.db
droste "how does auth work here?" ./docs
cd ~/notes && droste "what did I decide about pricing?"
tail -5000 app.log | droste "why did it crash?"
```

SQLite files are recognized by their magic bytes — no flag needed (`--db`
remains as an explicit override). Directory walks skip binaries, dotfiles,
and the usual junk (`.git`, `node_modules`, …) and cap sizes
(`--max-file-bytes`, `--max-bytes`); every skip is counted in the report
line. `droste ask …` still works as an alias.

Files are materialized as the sandbox's `context` variable — the model is
told each file's name and size (not its contents) and pulls data in via
code, so multi-MB files are fine. What the model reads is whatever its code
chooses to print. `--db` uses the engine's local-mode SQL data source (read-only
policy as a guardrail, not a boundary; OS permissions are the boundary).

Engine knobs mirror `RLMConfig`: `--subcall-model`,
`--subcall-max-output-tokens` (default 2048), `--reasoning-effort`,
`--max-iterations`, `--max-subcalls`. `--json` prints a result object for
scripting; `--verbose` streams one-line progress to stderr (watch it think);
`--trace` renders the full structured event stream — generated code, execution
output with per-iteration sub-call counts and answer state, LLM responses,
execution errors. Exit code 0 means a confirmed (or extracted-with-note)
answer.

The strict [Trace ABI v1](docs/trace-abi.md) gives every event one run identity,
sequence, timestamp, and retention class, then returns a policy-resolved
terminal record for host-owned local persistence. A trajectory-free canonical
result is always delivered live before the content-free terminal event; full
trajectory replay is emitted only under explicit retention. Retaining replay
content and authorizing training use are separate, default-denied decisions;
training also requires an auditable authorization reference and purpose.

Three worked starting points live in [docs/recipes.md](docs/recipes.md)
(logs, chat archives, SQLite).

Droste is the open execution engine. Compatible hosted gateways and control
planes can add authentication, server-enforced policy and cost limits, and
audit around it; those services are integrations, not part of the engine.
Use `--base-url` to select a compatible endpoint.

## Embed it

The same wheel is the engine as a library — zero runtime dependencies,
`urllib`-only. Add it to your app and point the loop at your own data
sources:

```bash
uv add droste        # or: pip install droste
```

Using is asking over *your* data; embedding is building RLM answers into a
product for *your users*.

### BYOK: compatible endpoints

The engine includes an OpenAI-compatible client and an Anthropic Messages
client. Configure the corresponding API key and model identifier; an explicit
base URL selects a compatible endpoint. Bring your own key — no hosted account
required. The CLI detects the protocol from credential and endpoint
configuration, and an explicit `--base-url`/`OPENAI_BASE_URL` always wins.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
droste "why did it crash?" ./logs --model claude-opus-4-8
```

```python
from droste import (
    Budget,
    EnvironmentConfig,
    OpenAICompatClient,
    OpenAICompatSubcallClient,
    SandboxLimits,
    create_environment,
    create_environment_context,
    run_rlm,
)

environment_config = EnvironmentConfig(
    kind="native",
    budget=Budget(subcalls=50, depth=1),
    sandbox=SandboxLimits(output_chars=25_000),
)
context = create_environment_context(environment_config)
root = OpenAICompatClient(model="gpt-5.2-mini")  # OPENAI_API_KEY / OPENAI_BASE_URL from env
subcalls = OpenAICompatSubcallClient(
    model="gpt-5.2-mini",
    context=context,               # shared call/token accounting
    max_output_tokens=2048,        # per-subcall output bound (cost control)
)

env = create_environment(
    environment_config,
    context=data,
    registry=registry,
    subcalls=subcalls,
    execution_context=context,
)
result = run_rlm(question, environment=env, root_llm=root, subcalls=subcalls, context=context)
```

Explicit `base_url=` / `api_key=` constructor args win over the environment
variables. Subcall batches use the immutable rollout concurrency (default 5),
and every subcall's usage block is added to `result.tokens_used`. When choosing
a non-default value in-process, pass the same value as the built-in subcall
client's `max_parallel` and `RolloutConfiguration.concurrency`; a mismatch
fails before inference.

`reasoning_effort` and `extra_body` pass through to the endpoint as-is.
Disabling thinking per-subcall is a gateway capability: a compatible gateway
may enforce it server-side, while raw endpoints may ignore a client-side
disable.

### Runner architecture (droste_runner)

The `droste_runner` package is a thin orchestration layer that wires `droste` to
HTTP-backed root LLM calls and subcalls. It is shared across hosted and
in-process embedders so the loop logic stays in one place. For custom environments,
set `adapter_module` in the runner request to delegate to an adapter module's
`run(request)` function.

```mermaid
flowchart LR
    Host[Host App] --> Runner[droste_runner]
    Runner --> Core[droste run_rlm]
    Runner --> Env[RunnerEnvironment]
    Env --> Sandbox[Python REPL execute]

    Core --> RootLLM[LLMClient responses_create]
    RootLLM --> Responses[Host /responses]

    Core --> Subcalls[SubcallClient llm_query llm_batch]
    Subcalls --> SubcallAPI[Host /rlm/subcall]
```

**Runner Inputs**
- `protocol_version`: **required** on every request (currently `3`) — a
  missing or mismatched version gets a structured refusal, so hosts detect
  incompatibility instead of failing on a missing field. See
  [docs/architecture.md](docs/architecture.md) for the compatibility rules and
  [UPGRADING.md](UPGRADING.md) for per-release embedder migration notes.
- `budget`: **required** complete six-field compute authorization object.
- `subcall_concurrency`: optional positive batch limit (default: `5`), recorded
  in the returned scaffold manifest.
- `root_endpoint` + `subcall_endpoint` + `token`: required for HTTP-backed runs.
- `adapter_module`: optional Python module path to override the runner entirely.

### Core concepts

#### Protocols

Implement these to integrate with your infrastructure:

- **`RLMEnvironment`** - Sandboxed Python REPL with data access
- **`LLMClient`** - Chat completion interface for the root LLM
- **`SubcallClient`** - Provides `llm_query()` and `llm_batch()` for sub-LLM calls
- **`SubcallOutputTokenLimitProvider`** - Optional companion protocol exposing a
  read-only `output_token_limit`: a positive per-call token ceiling or `None`
  when deliberately unbounded. Clients that omit it remain compatible and are
  reported to the root model as having an unknown limit.
- **`ProviderManifest`** - Immutable data-operation metadata

#### Providers are descriptor-driven

Droste has no universal data verb table. A reusable `ProviderManifest`
declares each provider's stable raw operation ID, Python binding name,
description, parameter/result schemas with dialect and provenance, pagination,
delivery mode, and budget class. A host combines the manifest with its own
authoritative side-effect classifications and policy metadata in a
`ProviderRegistration`, then binds named `ConfiguredSource` values through an
explicit `ProviderCatalog`.

The resulting immutable per-run descriptors generate the prompt, Python
bindings, policy accessor inventory, and broker allowlist. Provider metadata
can evolve without changing `CapabilityId`; raw operation IDs remain separate
from Python names. The bridge transports manifests and operation calls but not
authoritative effects or policy, which the receiving host must supply.
See [Provider manifests](docs/provider-manifests.md) for the value model,
ownership boundaries, bridge contract, and migration example.

#### Configuration

```python
RLMConfig(
    budget=Budget(
        tokens=500_000,
        subcalls=50,
        depth=1,
        wall_ms=300_000,
        root_output_tokens=4_096,
        subcall_output_tokens=2_048,
    ),
    sandbox=SandboxLimits(output_chars=25_000),
    prompt_profile="full",  # Versioned prompt-pack profile (full/minimal/none)
    policy_hints=PolicyHints(semantic=True), # Optional explicit contract
)
```

Compute authorization is one immutable vector, reconciled by one run-scoped
ledger. See [Budgets](docs/budgets.md). Sandbox output and execution guardrails
are separate because they describe the local REPL, not model/provider spend.

Harness prompts resolve once per run from immutable, versioned data. See
[Prompt packs](docs/prompt-packs.md) for the stable five-slot contract, custom
pack loading, deterministic fallback order, and provenance records.

Droste does not infer semantic intent from the question. When a caller supplies
`PolicyHints(semantic=True)`, at least one semantic subcall must succeed and any
incomplete `llm_batch_json` result blocks confirmation. Only an error-free
repeat with the exact prompts, contexts, schema, and validator object resolves
that partial evidence. Omit the hint to retain purely prompt-driven behavior.

#### Result

```python
RLMResult(
    answer="...",           # Final answer from answer["content"]
    ready=True,             # Whether answer["ready"] was set
    iterations=3,           # Iterations used
    tokens_used=1500,       # Total tokens consumed
    sub_calls_made=12,      # Total llm_query/llm_batch calls
    trajectory=[...],       # Full execution history
    extracted=False,        # True if the answer came from the post-exhaustion
                            # extract pass (best-effort, not confirmed)
    prompt_pack=...,        # Frozen resolved pack identity + provenance
)
```

## Development

```bash
uv sync --extra verifiers  # Install the full test surface when supported
uv run pytest              # Verifiers tests skip when its extra is unavailable
uv build                   # Build wheel
```

## The name

The [Droste effect](https://en.wikipedia.org/wiki/Droste_effect) is the
picture that contains itself. M.C. Escher's *Print Gallery* pushed it to its
limit — a man in a gallery viewing a print that contains the gallery he is
standing in — and Escher left the center of the spiral famously blank,
signed but uncompleted, where the recursion outran his hand. Fifty years
later, mathematicians completed it; their project was titled *"The
Mathematics Behind the Droste Effect."*

The answer at the center of the spiral — the part the picture couldn't hold
— is what recursion computes.

## License

Apache-2.0. See [LICENSE](LICENSE). Contributions welcome —
[CONTRIBUTING.md](CONTRIBUTING.md). Versioning is semver; the runner
protocol and source-registry contract carry an explicit compatibility
window (see [docs/architecture.md](docs/architecture.md)).
