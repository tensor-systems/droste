# Contributing to Droste

Thanks for your interest. Ground rules, kept short:

- **Bugs and ideas**: open an issue with a reproduction (for engine behavior,
  the trajectory JSON from `--json`/`--verbose` output is the reproduction).
- **PRs**: all code changes need tests. Run `uv run pytest -q` — the suite is
  fast and must stay green. No new runtime dependencies without discussion
  (the engine is deliberately stdlib-only).
- **Security-sensitive areas**: the SQL policy gate and the sandbox are
  guardrails with documented threat models — read the docstrings in
  `rlm_core/sources/sql_local.py` before changing them, and include
  adversarial tests (bypass attempts) with any change there.
- **Protocol changes**: the runner request/response and source-registry
  protocol are versioned compatibility surfaces (hosts embed old engines).
  Additive and optional by default; breaking changes need a protocol bump and
  a documented migration.
- **Benchmarks**: claims about accuracy or cost need artifacts. The benchmark
  harness lives with the ModelRelay platform; results referenced in the
  README link to per-task JSONs.

Development:

```bash
uv sync
uv run pytest -q
uv run droste --help
```
