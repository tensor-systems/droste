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

- `droste_runner` is a shared HTTP-backed runner used by host apps (ModelRelay's hosted runner, in-process embedders like Cozy). It reads the request JSON from `RLM_RUNNER_REQUEST_PATH` (or argv) and returns a JSON response payload.
- The runner wraps `droste` and supplies an HTTP `LLMClient` + `SubcallClient` plus a sandboxed `RunnerEnvironment`.
- Timeouts in `RunnerEnvironment.execute` use `signal.setitimer` and restore the previous handler (`old_handler`) after each execution to avoid clobbering host signal handlers.
- `droste_runner` expects HTTP endpoints for root and subcall execution (`root_endpoint`/`subcall_endpoint` + `token`).
- `adapter_module` lets callers delegate the runner to a custom module with `run(request)` (used by in-process embedders).


## Login (droste#55)

- `droste login` runs loopback OAuth (RFC 8252) against ModelRelay: local
  127.0.0.1 server -> `POST /auth/oauth/start` -> browser -> the OAuth
  callback form-POSTs tokens (+ `issued_key_*` on fresh signups) to the
  loopback. Nonce-checked; 3-minute timeout. Unsolicited loopback POSTs
  (wrong path, or no `handoff_nonce` field) are rejected without aborting
  the wait; a well-formed callback with the wrong nonce is terminal (CSRF).
- Free credits require a $0 card check: the CLI opens the checkout URL and
  polls `POST /account/card-verification/confirm` (409 = still open, 422 =
  prepaid). All card UX is web-side; the CLI only reads the outcome.
  Prepaid/timeouts still complete login (honest warning, no credits) —
  never strand the user. Servers without the endpoints (404) skip quietly.
- Credentials: ONE long-lived `mr_sk_*` API key in
  `$XDG_CONFIG_HOME/droste/credentials.json` (0600, atomic mkstemp +
  fsync + rename). No OAuth tokens stored — nothing to refresh; logout is
  deleting the file.
- `droste login` on a TTY is a chooser: ModelRelay (default) or your own
  key; BOTH are stored in the credentials file (`provider: modelrelay |
  byok`). Choosing how droste runs is a deliberate setup step — never a
  side effect of exported env vars (ambient keys must not bypass the
  free-credits choice).
- Resolution order: `--api-key`/`--base-url` flags -> stored credentials
  (either provider) -> interactive setup in-line when on a TTY -> env keys
  as a non-interactive (scripts/CI) fallback -> terse error pointing at
  `droste login`.
- The ModelRelay API rejects `Authorization: Bearer mr_sk_...` — API keys
  go in `X-ModelRelay-Api-Key`; bearer is only for OAuth access tokens.
- Logged-in runs use `droste.clients.modelrelay` (native `/responses`,
  NDJSON streaming for --verbose); subcalls default
  `reasoning_effort="none"` + a 2048-token output bound (the same defaults
  the platform applies server-side).
- Tests isolate `XDG_CONFIG_HOME` + provider env via `tests/conftest.py`
  (autouse) — without it, tests read the developer's real credentials.

## Offline Wheelhouse (macOS App Builds)

For reproducible offline builds, download wheels to a local directory:

```bash
# Download droste + its deps as wheels (from public PyPI)
pip download droste --dest wheelhouse

# Install offline (no network needed)
uv pip install --no-index --find-links wheelhouse droste
```

## Pyodide Credential Broker

- `pyodide/broker.ts` must remove secret credential values (`api_key` and
  `customer_token`) before creating the sandbox request.
- Preserve a normalized `auth_type` in that request. It is nonsecret routing
  metadata required by adapters that distinguish customer-tier defaults from
  tierless API-key requests without receiving either credential.

## Publishing

The package ships on public PyPI. To build and publish a release:

```bash
uv build
uv publish        # requires a PyPI token (UV_PUBLISH_TOKEN)
```

Tag releases as `vX.Y.Z` and bump the version in `pyproject.toml`.
