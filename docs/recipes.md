# Recipes

Three copy-paste starting points. Every recipe works with any
OpenAI-compatible endpoint — set `OPENAI_API_KEY` (and `OPENAI_BASE_URL` for
non-OpenAI providers; see the README's BYOK section).

## 1. Interrogate a huge log file

```bash
droste ask server.log \
  "Group the errors by service and root cause. Which failure started first,
   and did it cascade? Give me counts, not vibes."
```

Multi-megabyte files are fine: the model is told the file's name and size,
then pulls slices in via code. Counting, grouping, and time-ordering happen
in Python, so the numbers are exact. Add `--verbose` to watch it work.

## 2. Ask questions across a chat/export archive

Point it at an export directory's files — Slack, WhatsApp, or any
line-oriented dump:

```bash
droste ask export/channel-eng.txt export/channel-support.txt \
  "What did we promise customers about the migration, and did engineering's
   internal discussion match what support was telling people?"
```

This is the shape retrieval can't do: the answer requires reading *both*
files, aligning them in time, and judging consistency — the model narrows
mechanically, then fans out `llm_query_batched` over the sections that need
actual reading.

## 3. Analyze a SQLite database

```bash
droste ask --db app.db \
  "Which customers churned in Q2, what did they have in common, and how much
   MRR walked out the door?"
```

`--db` exposes the database as a read-only, policy-gated source (SELECT-only,
single statement, bounded rows). The model reads the schema, iterates on
queries, and computes in code. The policy is a guardrail, not a security
boundary — the file is opened read-only, and OS permissions are the real
control. Cheap subcalls under a stronger root: add
`--subcall-model gpt-5.2-mini` (or any cheaper model on your endpoint).
