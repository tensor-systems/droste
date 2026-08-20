<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/droste-dark.svg">
  <img src="docs/assets/droste.svg" alt="Droste" width="96">
</picture>

# Droste

**Ask questions about data too large for a context window.**

Droste is an open source Recursive Language Model (RLM) engine. It gives a
model a Python REPL over your data, so the model can search, filter, count, and
delegate small pieces of interpretation without loading the full corpus into
its context.

```bash
uvx droste "which customer had a failed charge, and why?" server.log
uvx droste "which plan has the highest refund rate vs its MRR?" shop.db
uvx droste "how do the authentication flows differ?" ./docs
```

![droste answering a two-part question over a 444 kB server log, streaming its code as it works](docs/assets/demo.gif)

## Why Droste

Code handles work that should be exact, such as searching, counting, joining,
and aggregating. Model calls handle the parts that require interpretation.
Droste combines both while keeping the full dataset outside the model's
context window.

The root model sees only what its code prints and the bounded results of any
subcalls it makes. Runs have explicit limits for iterations, subcalls, time,
and model output.

## Use it

Install the command and sign in once:

```bash
uv tool install droste
droste login
```

Then ask a question about files, directories, SQLite databases, or piped text:

```bash
droste "what changed between these?" report.txt logs.txt
droste "which customers churned last month?" app.db
tail -5000 app.log | droste "why did it crash?"
```

For a one-off run, use `uvx droste` instead of installing it. The command also
supports an existing ModelRelay key or your own OpenAI-compatible or Anthropic
endpoint. See the [quickstart](docs/quickstart.md) for credentials, directories,
SQLite, and useful flags.

## Good fits

- Exact counts, aggregates, and joins over logs, exports, or databases.
- Classification or review across many records using bounded model batches.
- Questions that need both computation and close reading.

If the data already fits comfortably in a context window, or the task is
open-ended agent work rather than a question about a corpus, use a general
agent instead. The [quickstart](docs/quickstart.md) includes examples for logs,
archives, and SQLite.

## Results

Published runs show the largest gains on tasks that require aggregation across
scattered or oversized context.

| Benchmark | Direct baseline | Droste | Recorded cost |
|---|---:|---:|---:|
| OOLONG, 131K tokens | 0.6020 | **0.6432** | $10.16 vs $26.18 |
| OOLONG-Pairs, 32K tokens | 0.034 | **0.80** | $2.14 vs $2.50 |
| BrowseComp-Plus, 6.0M to 11.1M tokens | Could not run | **0.9400** | $24.54 |

These are individual published runs, not universal quality or cost guarantees.
See the [results and caveats](docs/benchmarks.md), or open the repository's
[benchmark guide](benchmarks/README.md) for artifacts and reproduction steps.

## Use it as a library

The same package is a dependency-free Python library. You can supply your own
models and data providers, set hard compute budgets, and consume structured
traces. Start with the [embedding guide](docs/embedding.md). Protocol and host
implementers can use the [technical reference](docs/README.md#technical-reference).

## Development

```bash
uv sync
uv run pytest
uv build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## License

Apache-2.0. See [LICENSE](LICENSE).
