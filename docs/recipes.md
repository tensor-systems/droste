# Recipes

These examples work with the account saved by `droste login` or any
OpenAI-compatible endpoint configured with `OPENAI_API_KEY` and, when needed,
`OPENAI_BASE_URL`.

## 1. Interrogate a huge log file

```bash
droste "Group errors by service and cause. Which failure started first, and
did it cascade? Include exact counts." server.log
```

The model inspects slices of the file through code. Python handles counting,
grouping, and time ordering. Add `--verbose` to watch the run.

## 2. Ask questions across a chat/export archive

Point it at Slack, WhatsApp, or another line-oriented export:

```bash
droste export/channel-eng.txt export/channel-support.txt \
  "What did we promise customers about the migration, and did engineering's
   internal discussion match what support was telling people?"
```

Droste searches both files, aligns relevant passages in time, and uses model
subcalls where judging consistency requires close reading.

## 3. Analyze a SQLite database

```bash
droste app.db \
  "Which customers churned in Q2, what did they have in common, and how much
   MRR walked out the door?"
```

Droste detects SQLite files, reads the schema, writes bounded read-only
queries, and computes over the results. Operating-system permissions remain
the security boundary. To use a cheaper model for subcalls, add
`--subcall-model MODEL`.
