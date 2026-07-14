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
With `PolicyHints(semantic=True)`, an incomplete structured batch remains
unconfirmed until the exact prompts, contexts, schema, and validator object are
rerun without errors; a different successful batch is not completion evidence.

Terminal step failures with usable partial work go through the bounded extract
fallback. A successful recovery is `extracted=True`, `error=None`, with the
superseded typed failure in `recovered_error`; hosts must treat it as
unconfirmed, while benchmarks may inspect the recovered error provenance.
Failed attempts are retained in the trajectory when their repair also fails,
including on mid-run iterations that later recover successfully.
A failed-only trajectory with no retained `answer["content"]` must stay fatal;
only a retained draft or at least one successful step is extraction evidence.
Trajectory execution state is explicit in `IterationRecord.execution_status`;
never infer success or failure from `execution_result` text because successful
stdout may begin with `ERROR:`. Build loop records from the typed `StepOutcome`
so output and status cannot drift apart, and label extract-fallback trajectory
evidence with that status rather than leaving the model to interpret prefixes.

## Prompt Packs

- Harness strategy lives in complete versioned TOML artifacts under
  `src/droste/prompts/packs/`; do not add prompt prose back to loop constants.
- The stable slots are `capabilities`, `budget`, `question`, `history`, and
  `output_contract`. Unknown/missing slots fail during pack validation.
- Keep packs, catalogs, resolved selections, and provenance frozen values.
  Parsing, validation, resolution, and rendering stay pure; filesystem and
  package-resource reads belong only in loader functions.
- Resolve exactly one pack per run. Never merge partial packs or mutate strategy
  mid-run. Broker-loaded RLM skills are separate work under #3.
- New artifacts need a wheel-resource check and must preserve generic fallback
  compatibility unless an upgrading note explicitly calls out a change.

## Reproducible Benchmarks

- Repository-local benchmark tooling lives in `benchmarks/` and is deliberately
  excluded from the published wheel. Run its zero-cost integrity check with
  `uv run python -m benchmarks smoke --output <new-directory>`.
- Per-task artifacts are immutable: the runner refuses to overwrite an existing
  artifact. Reports reject artifacts whose suite version or manifest SHA-256
  differs from the selected manifest.
- Store cost as integer micro-US dollars and root/subcall usage separately.
  Failures and timeouts remain typed artifacts; never omit them from aggregates.
- Keep unpublished live arms blocked. Enabling them requires a public model
  configuration and an end-to-end check that response usage reflects the
  requested reasoning settings.

## LLM Client Protocol

- `LLMClient` now exposes `responses_create(...)` (message-based) to avoid chat/completions terminology.
- The core loop calls `responses_create` and expects it to wrap `/responses` semantics, not OpenAI-style completions.
- `ModelRelayClient.root_requests_issued` is a thread-safe cumulative count of root HTTP requests at the dispatch boundary. It includes streaming and non-streaming requests that later fail, including repair and extraction calls, but excludes payload or request-construction failures before dispatch.
- `ModelRelaySubcallClient.llm_batch` uses one typed `/responses/batch` request and never falls back to per-item fan-out. Batch ids are parsed back into caller order, per-item errors remain attributable, and the entire call budget is reserved atomically before dispatch. BYOK clients keep bounded concurrent fan-out because their synchronous APIs have no equivalent endpoint.

## Trace ABI

- Every structured event is a strict Trace ABI v1 value. Stamp it exactly once
  through `ExecutionContext`; do not emit raw or partially enveloped event
  dictionaries at host boundaries.
- `execution.trace` owns immutable values, parsing, classification, retention
  selection, and terminal-record invariants. Keep persistence I/O in an
  injected host callback; core code must not choose files, databases, or cloud
  services.
- Durable events are `usage`, `budget`, `policy`, `capability`, and `done`.
  Never put answer/code/output/trajectory, error messages/details, or executed
  source inside them. Those belong only to configurable events.
- Deliver the configurable canonical `result` live before `done` regardless of
  retention. Emit full `replay`/trajectory only when replay retention is
  selected; retention controls storage, not ordinary live delivery.
- Retention and training authorization are independent values. Training is
  denied by default, requires an authorization reference plus the `training`
  purpose, and must never be inferred from retained content. Expiry is recorded
  by the ABI and enforced by the host persistence shell.
- An injected `ExecutionContext` owns trace identity and policy. Reject
  conflicting `RLMConfig` values rather than replacing a recorder that may
  already contain events.
- Capability tracing is observational: wrap only the broker-owned content-free
  accounting/evidence projection; do not duplicate its schema or let tracing
  participate in dispatch or authorization.

## droste_runner Package

- `droste_runner` is a shared HTTP-backed runner used by host apps (ModelRelay's hosted runner, in-process embedders). It reads the request JSON from `RLM_RUNNER_REQUEST_PATH` (or argv) and returns a JSON response payload.
- Invoke the process runner as `python -m droste_runner` from an installed
  package. Do not restore repository-layout `sys.path` mutation or rely on
  direct execution of `runner.py`.
- Module ownership is strict: `run.py` orchestrates, `protocol.py` shapes both
  refusal and success envelopes, `http_clients.py` owns network clients, and
  `sources.py` owns the remote wrapper plus declarative source construction.
  `runner.py` only re-exports compatibility names; focused modules must not
  import that facade or each other in a cycle.
- The generic native `RunnerEnvironment` lives in
  `droste.environments.inprocess`; `droste_runner.environment` and
  `droste_runner.runner` are compatibility shims. Core/CLI code must not import
  the runner package to obtain an execution environment.
- Host entrypoints must use one frozen `EnvironmentConfig` with
  `create_environment_context` and `create_environment`; do not copy budget,
  registry, timeout, and executor selection across host modules. Direct
  `RunnerEnvironment` construction remains compatibility-only.
- `kind="pyodide"` selects `PyodideEnvironment`/`RawExecutor` and requires
  explicit `host_managed_timeout` and `host_managed_isolation` declarations.
  They are assertions that the Deno/WASM host supplies those boundaries, not
  Python-side enforcement. Never weaken them or silently accept a native
  signal timeout for Pyodide.
- Every request MUST carry `"protocol_version"` (currently 2) — missing/mismatched versions get a structured refusal, never partial work. See docs/architecture.md, "The runner protocol".
- The runner wraps `droste` and supplies an HTTP `LLMClient` + `SubcallClient` plus a sandboxed `RunnerEnvironment`.
- Timeouts in `RunnerEnvironment.execute` use `signal.setitimer` and restore the previous handler (`old_handler`) after each execution to avoid clobbering host signal handlers.
- `droste_runner` expects HTTP endpoints for root and subcall execution (`root_endpoint`/`subcall_endpoint` + `token`).
- Hosted subcalls negotiate `responses-stream/v2` NDJSON, ignore keepalive and reasoning events, assemble text from `update.delta`, require a terminal `completion`, and fail on error or truncated streams. Servers may still return plain JSON for compatibility with non-streaming local callback handlers.
- `adapter_module` lets callers delegate the runner to a custom module with `run(request)` (used by in-process embedders).

## Capability Broker ABI

- Built-in environments expose generated compatibility bindings, never raw
  `SubcallClient` or data-source bound methods. Loop-installed structured batch
  helpers must use the required `environment.sandbox_subcalls(subcalls)` result
  so they do not recreate a direct egress path. Custom environments can use
  `droste.capabilities.broker_subcalls()`; there is no raw-client fallback.
- `CapabilityId`, `CapabilityManifest`, `CapabilityCall`, and
  `CapabilityResult` are immutable values. Calls carry only the stable
  `(kind, provider_type, source_id, operation)` identity; the broker resolves
  evolving descriptor documentation and policy metadata from its manifest.
  Exact identity validation is the allowlist; transports stay behind registered
  trusted handlers and must not add a parallel decoder or dispatch protocol.
- Keep `llm_batch` atomic: one broker call invokes one client batch operation.
  Do not expand it into nested `llm_query` calls.
- Registrations normalize raw handler return values once into
  `CapabilityOutcome`. Trusted providers may return that value directly to
  attach either a result or an extensible stable-code `CapabilityError` plus
  provider usage/evidence metadata. Dispatch must consume only the normalized
  outcome convention; unexpected exceptions remain `handler_error`.
- The guard, annotator, and observer are seams only. Budget ownership, policy
  semantics, trace ordering/storage/retention, provider generalization, and MCP
  transport belong to their own issues. Observers are observational and must
  never become an authority or alternate dispatch path. Durable traces consume
  `CapabilityResult.to_trace_dict()`, which excludes parameters, inline results,
  error messages, evidence references, and result-handle locators; full
  `to_dict()` envelopes are replay content and require a separately configured
  retention policy.
- The annotator is also the exactly-once post-attempt finalizer. It runs once
  after every attempted handler outcome (success, handler error, invalid result,
  or propagated cancellation) and never on run/allowlist/argument/guard exits
  where the handler was not attempted. Keep reservation/reconciliation logic
  keyed by the immutable `call_id`; do not add a parallel finalization path.
  Provider metadata is ordered before finalizer metadata; sequence facts append,
  while conflicting singular result handles or child-run IDs fail closed.


## Login

- `droste login` runs loopback OAuth (RFC 8252) against ModelRelay: local
  127.0.0.1 server -> `POST /auth/oauth/start` -> browser -> the OAuth
  callback form-POSTs tokens (+ `issued_key_*` on fresh signups) to the
  loopback. Nonce-checked; 3-minute timeout. Unsolicited loopback POSTs
  (wrong path, or no `handoff_nonce` field) are rejected without aborting
  the wait; a well-formed callback with the wrong nonce is terminal (CSRF).
- Browser launching is suppressed over SSH only when no opener is supplied.
  Explicit opener callbacks still run, and the fallback URL is always printed.
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

- The relay broker (`src/droste/substrates/_relay/broker.ts`) must remove secret credential values (`api_key` and
  `customer_token`) before creating the sandbox request.
- Preserve a normalized `auth_type` in that request. It is nonsecret routing
  metadata required by adapters that distinguish customer-tier defaults from
  tierless API-key requests without receiving either credential.
- Hosted runners may pass a short-lived `token`; the relay strips it from the
  sandbox request and injects it only for exact root/subcall callback URLs from
  the trusted envelope. Requests without `db_path` skip the DB-service setup
  entirely, allowing context-only hosted adapters.
- `data_source_endpoint` is part of that same exact-match runner callback set;
  it lets a host proxy source calls without putting the source credential in
  the sandbox request.

## Repo Hygiene (manual, before pushing docs/comments)

This repo is public-facing: no references to internal strategy, private
repos, or closed-product internals (host-app package/binary names, private
corpus/db names, sibling-checkout paths). There is deliberately no CI term
gate — a public workflow would have to enumerate the very vocabulary it
exists to keep out. Check manually with `git grep -i` over the terms you
know are internal before pushing prose or comments that discuss hosts or
embedders; CI still scans for committed key material.

Keep public positioning consistent across the README, package description, and
repository description: Droste is a recursive analysis engine for data too
large for a context window, and RLM is the technique. Explain its structure
without naming buyers, hosts, or competitors. Quantitative claims must link to
version-matched reports and immutable artifacts; otherwise state the evidence
limit instead. Factual compatibility documentation may name a protocol or API
class when readers need it to configure the engine.

## Publishing

The package ships on public PyPI, released by CI — never by hand:

0. Update `UPGRADING.md`: retitle the "Unreleased" section to the new version
   and start a fresh empty one. Every embedder-facing change (new required
   field, changed default, moved import, anything that degrades silently if a
   host skips a step) must have an entry — the loud contract signals
   (protocol refusals, the relay's startup handshake) don't cover silent
   degradations, and PR bodies don't reach public users.
1. Bump `version` in `pyproject.toml`, run `uv lock` (the lockfile pins the
   project's own version — `uv sync --locked` in the release job aborts on a
   stale one), and merge both to main.
2. Tag the merge commit `vX.Y.Z` and push the tag.
3. `.github/workflows/release.yml` runs tests, builds sdist+wheel,
   publishes to PyPI via **trusted publishing** (OIDC — no token secret),
   and creates the GitHub release with artifacts attached. The job fails
   fast if the tag and `pyproject.toml` version disagree.
4. Live smoke against production (manual — CI deliberately holds no LLM
   keys): `uvx droste@X.Y.Z "…" <some file>` with real credentials must
   return exit 0. The mocked e2e suite cannot see edge/WAF behavior — a
   Cloudflare rule once blocked every fresh install by User-Agent (#49)
   while the whole suite stayed green.

The PyPI trusted publisher (project `droste` → Publishing) must name this
repo and `release.yml`. A local `uv build && uv publish` remains possible
in an emergency but needs a `UV_PUBLISH_TOKEN`, which is deliberately not
kept as a repo secret.
