# Quickstart

Droste answers questions about files, directories, SQLite databases, and piped
text.

## Install and sign in

```bash
uv tool install droste
droste login
```

`droste login` can use ModelRelay credits or store your own key. For scripts
and CI, set a provider key directly:

```bash
export OPENAI_API_KEY=sk-...
# Or: export ANTHROPIC_API_KEY=sk-ant-...
```

Set `OPENAI_BASE_URL` when using another OpenAI-compatible endpoint. You can
also pass `--api-key` and `--base-url` for one run.

For a one-off run without installation:

```bash
uvx droste "summarize the failures" server.log
```

## Ask a question

```bash
droste "what changed between these?" report.txt logs.txt
droste "which customers churned last month?" app.db
droste "how does authentication work?" ./src
tail -5000 app.log | droste "why did it crash?"
```

Arguments that exist on disk are input data. The quoted argument that does not
exist is the question. With no data argument, Droste reads the current
directory. SQLite files are detected automatically.

Directory reads skip dotfiles, binary files, and common generated directories
such as `.git` and `node_modules`. Droste reports what it loaded and skipped
before the run starts.

## Useful options

```bash
droste "..." ./data --verbose
droste "..." ./data --json
droste "..." ./data --model MODEL --subcall-model SMALLER_MODEL
droste "..." ./data --budget-subcalls 25 --max-iterations 15
```

`--verbose` streams progress and generated code to stderr. `--trace` includes
the complete structured execution trace. Run `droste --help` for every budget,
input, and output option.

The generated Python code can read the data you provide. Database access is
opened read-only and policy-gated, but process isolation and operating-system
permissions remain the security boundary.

## Examples

Interrogate a large log without putting the whole file in the model prompt:

```bash
droste "Which failure caused the outage, and what happened immediately before it?" server.log
```

Ask across an export or archive. Droste recursively reads supported text files
while skipping hidden and common generated directories:

```bash
droste "Which decisions were reversed, and why?" ./chat-export
```

Use SQL for exact aggregation and model calls only where interpretation is
needed:

```bash
droste "Which plan has the highest refund rate relative to MRR?" shop.db
```

For multiple inputs, pass every path after the question:

```bash
droste "What changed between these reports?" january.md february.md
```
