# Contributing to Droste

Thanks for your interest. Ground rules, kept short:

- **Bugs and ideas**: open an issue with a reproduction (for engine behavior,
  the trajectory JSON from `--json`/`--verbose` output is the reproduction).
- **PRs**: all code changes need tests. Run `uv run pytest -q` — the suite is
  fast and must stay green. No new runtime dependencies without discussion
  (the engine is deliberately stdlib-only).
- **Security-sensitive areas**: the SQL policy gate and the sandbox are
  guardrails with documented threat models — read the docstrings in
  `droste/sources/sql_local.py` before changing them, and include
  adversarial tests (bypass attempts) with any change there.
- **Protocol changes**: the runner request/response and source-registry
  protocol are versioned compatibility surfaces (hosts embed old engines).
  Additive and optional by default; breaking changes need a protocol bump and
  a documented migration.
- **Benchmarks**: claims about accuracy, cost, or latency require checked-in
  immutable artifacts and a version-matched generated report. Unpublished runs
  must not support public comparative claims.
- **Documentation**: document shipped behavior, not roadmap work. Give each
  concept one canonical page and link to it instead of copying contract prose.
  Version history belongs in `UPGRADING.md`; implementation rationale belongs
  in code, tests, issues, or pull requests. Add every maintained page to
  `docs/README.md`. Runnable examples live under `examples/` and need tests.
- **Documentation contracts**: `tests/test_docs.py` checks local links, anchors,
  orphaned pages, and version headings against runtime constants. A protocol or
  schema bump updates its implementation, reference heading, fixtures, and
  migration note in one change.

Development:

```bash
uv sync
uv run pytest -q
uv run droste --help
```
