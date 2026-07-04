# Agent Guidelines

## Package Management

This project uses **uv** as the default package manager.

- Install dependencies: `uv sync`
- Add a dependency: `uv add <package>`
- Add a dev dependency: `uv add --group dev <package>`
- Run tests: `uv run pytest`

## Policy Hints (Contract Enforcement)

droste does **not** infer semantic or count intent from the question text.
If you want contract enforcement (e.g., require `llm_query` for semantic tasks
or require SQL aggregates + numeric-only output for counts), the caller must
pass explicit policy hints. Benchmarks should supply these hints; general use
can omit them to keep behavior purely prompt-driven.

## LLM Client Protocol

- `LLMClient` now exposes `responses_create(...)` (message-based) to avoid chat/completions terminology.
- The core loop calls `responses_create` and expects it to wrap `/responses` semantics, not OpenAI-style completions.

## droste_runner Package

- `droste_runner` is a shared HTTP-backed runner used by host apps (ModelRelay, Recall). It reads the request JSON from `RLM_RUNNER_REQUEST_PATH` (or argv) and returns a JSON response payload.
- The runner wraps `droste` and supplies an HTTP `LLMClient` + `SubcallClient` plus a sandboxed `RunnerEnvironment`.
- Timeouts in `RunnerEnvironment.execute` use `signal.setitimer` and restore the previous handler (`old_handler`) after each execution to avoid clobbering host signal handlers.
- `droste_runner` expects HTTP endpoints for root and subcall execution (`root_endpoint`/`subcall_endpoint` + `token`).
- `adapter_module` lets callers delegate the runner to a custom module with `run(request)` (used for non-HTTP environments like Recall).

## Installing from Private Index

To install `droste` from the private PyPI index:

```bash
uv pip install --index-url https://${PYPI_TOKEN}:x@rlm-pypi.hyperpredict.workers.dev/simple droste
```

Or add to your `pyproject.toml`:

```toml
[[tool.uv.index]]
url = "https://rlm-pypi.hyperpredict.workers.dev/simple"
```

Then set `UV_INDEX_PYPI_TOKEN` or use a `.netrc` file for authentication.

## Offline Wheelhouse (macOS App Builds)

For reproducible offline builds, download wheels to a local directory:

```bash
# Download wheels
uv pip download --dest wheelhouse --index-url "https://${PYPI_TOKEN}:x@rlm-pypi.hyperpredict.workers.dev/simple" droste

# Install offline (no network needed)
uv pip install --no-index --find-links wheelhouse droste
```

## Publishing

Publishing is automated via GitHub Actions on `v*` tags. To publish manually:

```bash
uv build
uv run python scripts/publish.py
```

Requires `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` environment variables.
