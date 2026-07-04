---
name: droste
description: Recursive analysis over large local datasets (files, SQLite) that are too big to read into context. Use when a question needs computing/aggregating across an entire corpus — counts, trends, cross-file reasoning, "analyze all of X" — not a quick lookup.
homepage: https://github.com/tensor-systems/droste
metadata:
  {
    "openclaw":
      {
        "emoji": "🌀",
        "os": ["darwin", "linux"],
        "requires": { "bins": ["droste"] },
        "install":
          [
            {
              "id": "uv",
              "kind": "uv",
              "package": "droste",
              "bins": ["droste"],
              "label": "Install droste (uv)",
            },
          ],
      },
  }
---

# droste

`droste` is a Recursive Language Model (RLM) engine. Instead of reading a dataset
into the context window, it hands the model the data as a variable in a sandboxed
Python REPL; the model writes code over it and fans out bounded sub-LLM calls
(map-reduce), then aggregates. This computes answers across data far larger than
any context window.

## When to use this (and when not to)

Reach for `droste` when the question needs reasoning **across a whole corpus**
that won't fit in context, or needs computation you'd otherwise fake:

- "How many refunds last quarter, and what did the churned customers have in common?" over a database
- "Summarize how the tone of this 300k-line chat export changed over the year"
- "Which of these 400 log files show the same root-cause error, and which failed first?"
- Counting, grouping, trend, or cross-file/cross-row questions where reading everything into the prompt would truncate or lie.

Do **not** use it for a quick single-file read, a one-off lookup, or a chat where
the content already fits comfortably in context — answer those directly. `droste`
costs an LLM loop; it earns its keep only when scale or computation is the point.

## How to call it

```bash
# Over files (large files OK — subject to local memory; the model pulls in what it needs via code):
droste ask <file...> "<question>" --model <model>

# Over a SQLite database (read-only, policy-gated; the model reads the schema
# and writes SELECTs):
droste ask --db <path.db> "<question>" --model <model>
```

Key flags: `--model` (or env `DROSTE_MODEL`); `--json` for a machine-readable
result to parse; `--subcall-model <cheap-model>` to run the fan-out subcalls on a
cheaper model than the root; `--max-subcalls`, `--max-iterations` to bound cost.
BYOK via `OPENAI_API_KEY`/`OPENAI_BASE_URL` (any OpenAI-compatible endpoint,
including local Ollama). Exit code is 0 for a usable answer; when the budget ran
out, the answer is still printed but a stderr note (`droste: note: max iterations
reached; answer extracted from partial work (unconfirmed)`) flags it as
unconfirmed — check for that note before trusting the result.

## Notes

- SQLite access is read-only (opened `mode=ro`, SELECT-only policy) — it will not
  modify the database. The policy is a guardrail, not a security boundary; the
  file's OS permissions are the real control.
- iMessage / chat data: point `--db` at the Messages `chat.db` (needs Full Disk
  Access for the terminal, like any Messages reader). `droste` then reasons over
  the *whole* history in SQL rather than paging a fixed number of recent messages.
- Prefer `--json` when you need to lift the answer into a reply programmatically.
