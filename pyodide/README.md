# Spike: Deno + Pyodide as the RLM execution substrate

Feasibility spike for running droste's RLM in a **Deno + Pyodide (CPython-on-WASM)**
sandbox instead of the native CPython-framework + wheelhouse it ships today — the
goal being a much simpler/safer macOS app bundle (one Deno binary + data, no native
`.so` signing, arch-independent WASM).

Run:
```bash
./run.sh spike.ts      # Phase 0: WASM viability (imports, host-tool bridge, blockers)
./run.sh phase1.ts     # Phase 1: data-layer fidelity, native vs Pyodide, on the corpus
./run.sh phase2.ts     # Phase 2: RunnerEnvironment.execute() runs under Pyodide
MODELRELAY_API_KEY=... ./run.sh phase3.ts   # Phase 2 (step 3): a live RLM query end-to-end
MODELRELAY_API_KEY=... ./run.sh phase4.ts   # Phase 4: 8-query semantic parity vs baseline
deno run --cached-only --allow-read --allow-env offline-probe.ts   # Phase 3: offline Pyodide
```

**Status (all green / committed):** Phase 0 viability ✓ · Phase 1 fidelity ✓ · executor port ✓ ·
a live Recall RLM query runs end-to-end in Deno+Pyodide ✓ · parity shows no quality-regression
signal (Pyodide ≥ native on 6/8 semantic queries; single-shot metric is noisy) ✓.

**Phase 3 — packaging (all technical risks resolved):**
- *Offline Pyodide* ✓ — the runtime (`pyodide.asm.wasm`, `python_stdlib.zip`, `sqlite3` wheel) is
  ~13MB of **data** and runs with **zero network** (`deno run --cached-only`).
- *Relay drop-in* ✓ (`relay.ts`, Phase 3a) — reads a HostRequest JSON on stdin, runs the RLM in
  Pyodide, writes a clean HostResponse JSON on stdout. Same contract as the native PyO3 helper.
- *Bundle strategy* ✓ — `deno compile` is **NOT supported** for Pyodide (runtime fs/WASM loading).
  Use an **isolated `DENO_DIR` + `--cached-only`**: a self-contained ~14MB cache runs the relay
  fully offline. Shipped bundle = one Deno binary (~40MB, only signed native artifact) + ~14MB
  `DENO_DIR` data + Python sources + `relay.ts` ≈ 54MB. No `Python.framework`, no wheelhouse, no
  per-`.so` signing — the "easier to package" payoff.

Remaining Phase 3 is mechanical productionization: assemble the bundle into the `.app`
(`package_app.sh`), point Swift `RLMHelperRunner` at the Deno relay (replace `recall-rlm-helper`,
tight read-only DB-only mount), sign Deno + add the JIT entitlement, build, smoke-test.
`relay.ts` mounts the bundled sources dir (droste + the cozy `rcl_rlm` data layer,
staged by cozy's own build) into Pyodide's `/app` and the DB directory into `/data`.
Requires Deno; the corpus DB path comes from the host request (`db_path`).

## Phase 0 — viability (PASS)
- All of droste imports cleanly under Pyodide, incl. `droste_runner.runner`.
- The RLM premise works: model-style Python in the REPL calling an injected host
  `query()` tool over the JS⇄WASM boundary, then doing arbitrary Python on the rows.
- Confirmed WASM blockers (all expected, all → move to the trusted host):
  - `signal.setitimer`/SIGALRM exec timeout → **MISSING** → host wall-clock kill.
  - `threading.Thread` / `ThreadPoolExecutor` → **"can't start new thread"** → batch
    subcall parallelism moves host-side; depth `threading.local` → plain int.
  - `urllib`/sockets → no network → brokered host-side via the `llm_query` tool.

## Phase 1 — data-layer fidelity + volume (PASS)
Against the real **300,649-message** corpus, `MessageDatabase` run **verbatim** in both
runtimes returns **byte-identical** results:
```
sqlite native/pyodide : 3.53.1 / 3.39.0   (engines differ)
messages              : 300649 / 300649
get_messages digest   : 5c974821e911 == 5c974821e911   (500 rows)
view(query) digest    : 918fb043cbeb == 918fb043cbeb   (300 rows, ATTACH+view)
=> FIDELITY PARITY
```
- **Volume:** the 156MB DB is **FS-mounted** into Pyodide via `mountNodeFS` (read in
  place, NOT copied into the WASM heap).
- **SQLite version skew** (3.53.1 vs 3.39.0) did not change results here, but it's the
  thing to watch — the full benchmark suite (Phase 4) is the real parity gate.

## Findings that shape v1
- **Bundle the `sqlite3` Pyodide package** — it's *unvendored* from the base distribution
  (`loadPackage("sqlite3")`), not present by default.
- **The data layer must import without the network stack.** ✅ DONE (cozybot `e28f7ad`,
  branch `feat/deno-pyodide-rlm`): `rcl_rlm/__init__.py` eagerly imported `.modelrelay` →
  httpx. Refactored to a lazy PEP-562 `__init__` — data layer eager, network/droste lazy.
  The spike now stages the **full** `rcl_rlm` (real `__init__`) and the data layer imports
  cleanly under Pyodide.
- Keep `MessageDatabase` **verbatim**; fidelity is preserved when the same code runs over
  the same file. Reimplementing query()/the view in TS/Swift would reintroduce drift risk.

## Security model

The RLM queries `shadow.db` — a **read-only, app-maintained derived copy** of iMessage
data (`MessageDatabase` opens `file:{path}?mode=ro`). The live `~/Library/Messages/chat.db`
is read only by the trusted Swift host (to build the shadow) and **never reaches the
sandbox**. Combined with Pyodide having **no sockets** (no network egress), this makes the
data-layer-in-sandbox posture acceptable for v1: generated code can only read a read-only
copy of the corpus it's already meant to query, and can't exfiltrate it (except via the
`llm_query` sub-call to the user's own ModelRelay).

**v1 mitigation — tight, read-only, DB-only mount (do in Phase 2):** the data dir also holds
`config.json` / `sessions.json` (settings / dev keys / session state). Do NOT mount the whole
directory (the Phase 1 spike does, for convenience). Mount only:
```
read-only:  shadow.db{,-wal,-shm}  +  contacts.db{,-wal,-shm}
exclude:    config.json, sessions.json, everything else
```
(WAL mode → include the `-wal`/`-shm` siblings, or open with `immutable=1`.) This removes the
only real leak vector while keeping `MessageDatabase` verbatim.

**Later hardening (Design B / A'-2) — tracked in tensor-systems/droste#3:** move the data
layer onto the trusted host as a separate Pyodide "DB service" context (verbatim
`MessageDatabase` + read-only mount + ENFORCED `SqlValidator`), with the untrusted REPL
context getting no FS/no net and only the bridged tools. Matches DSPy's tools-only posture.
Defense-in-depth, post-v1.

**Progress (droste#9 / droste#3 A'-2, cozy#807 — done, on by default):** the full cross-interpreter
split is wired and tested end-to-end. `droste/sources/bridge.py` (`DataSourceService` server
half + `BridgeDataSource` client half; unit tests in `tests/test_bridge_source.py`, a
real-two-Pyodide-interpreters proof in `bridge_source_integration_test.ts`); cozy's
`rcl_rlm.rlm.run_rlm` gained the `data_source=`/`has_contacts=` keywords and
`rcl_rlm.pyodide_service.build_service_source` (cozy#808); `pyodide_runtime.py` gained
`build_db_service(db_path, contacts_db_path)` — builds the real `MessageDataSource` +
`DataSourceService` for the trusted interpreter, including corpus-scaled
`default_max_calls` and a `has_contacts` probe (both computed where the DB is actually
visible, then threaded across — the REPL interpreter has no `/data` mount to probe either
from) — and `run_for_host_pyodide` accepts `bridge_call`/`has_contacts`/`default_max_calls`.
`relay.ts` boots the trusted "DB service" interpreter FIRST (a bootstrap failure then costs
one interpreter, not two), mounts `/data` there only, and forwards a `bridge_call` into the
REPL interpreter — on by default as of 2026-07-08 (`RLM_DB_SERVICE=0` reverts to the
single-interpreter behavior), orthogonal to the `RLM_BRIDGE=legacy` kill-switch from A'-1.
`db_service_integration_test.ts` proves the whole
path against real `rcl_rlm` (not a stub): a real `MessageDatabase`, generated code's
`query()` calls reaching it over the bridge, `retrieved_guids` surviving via
`extra_methods`, and the REPL interpreter never seeing the DB. The cozy-side engine-pin bump
this depends on (`rlm-core`→`droste`) landed separately (cozy#812).

**Live-corpus parity check (done):** ran 3 representative CLI queries (`rcl-rlm --deno`) —
one factual count, one discovery yes/no, one semantic query that fans out an `llm_query`
sub-call — against the real 300k+-message benchmark corpus, once with `RLM_DB_SERVICE`
unset and once with `RLM_DB_SERVICE=1`, real `MODELRELAY_API_KEY`, real ModelRelay calls.
All three: **byte-for-byte identical** answer text, `retrieved_guids` (including full GUID
list + order on the semantic query), `iterations`, `sub_calls_made`, and `total_tokens`
between the two modes. This is the strongest available evidence that the DB-service split
doesn't change behavior — same corpus, same questions, same model, only the DB's interpreter
locality differs. On the strength of this parity result, `RLM_DB_SERVICE` now defaults to on
(`RLM_DB_SERVICE=0` is the escape hatch back to the single-interpreter mode).

**Fixed (droste#16): `batch_llm_query` under the Deno+Pyodide relay was a hard crash.**
`BridgedLLMClient` (this substrate's LLM client) only had a stale, unused
`batch_responses(requests) -> list[str]` method; cozy's `rcl_rlm.llm_orchestrator`
actually calls `batch_responses_typed(requests, options=None) -> BatchResponse` (a
single POST to ModelRelay's server-side `/responses/batch` — the fan-out happens on
ModelRelay's end, there's no client-side threading to port to Pyodide, which
debunks the original fan-out-under-Pyodide framing of #16). Any `batch_llm_query`/
`llm_batch_with_errors` call made while running under `--deno` hit an
`AttributeError`. Root cause was protocol drift: droste's own `LLMClient` protocol
(`src/droste/protocols/llm_client.py`) still declares the old `batch_responses`
signature; `BridgedLLMClient` faithfully implemented that stale protocol while the
actual runtime consumer (cozy's separate, evolved `rcl_rlm.llm_client.LLMClient`)
had moved on. Fixed by adding `batch_responses_typed` (lazily importing and reusing
`rcl_rlm.modelrelay.BatchResponse.from_dict`, matching this module's existing
lazy-rcl_rlm-import pattern) and deleting the dead method. `broker.ts`'s credential
scoping was widened from a single exact path to a two-entry exact-match set
(`/api/v1/responses`, `/api/v1/responses/batch`) so the held ModelRelay credential
is actually injected on batch calls too — without this, batch calls made by
sandbox-side code (which never holds a real credential; see A′ above) would get a
blank auth header and fail with 401, regardless of the method existing.
`broker_batch_integration_test.ts` proves both fixes together against a real
Pyodide interpreter with an *async* fake `host_fetch` (the existing
`broker_integration_test.ts` case predates this fix and uses a sync-string
`host_fetch`, which never exercises `BridgedLLMClient._post`'s `run_sync(awaitable)`
branch — the actual code path the real Deno relay always takes). The stale
`LLMClient` protocol drift itself (droste#16's real root cause) is tracked
separately — see the protocol-drift issue linked from #16.

**Fixed: the post-exhaustion extract-fallback silently swallowed failures.**
Found via a real user report through Cozy: a query that ran out of
`max_iterations` without the sandbox ever setting `answer["ready"]` showed raw
Python debug `print()` output as the "answer," instead of the synthesized
best-effort answer the engine's extract-fallback pass (`_extract_final_answer`,
issue #21) exists to produce. Root cause: that function wrapped its one LLM
call in a bare `try/except Exception: return ""`, so ANY failure (provider
error, empty response, anything) was silently swallowed, and the caller only
overwrote the raw-stdout fallback (`_best_answer`) when extraction *succeeded*
— when it silently failed, raw debug output stayed as the shown answer with
zero trace anywhere of why. Fixed by having `_extract_final_answer` return
`(text, RLMError | None)` instead of swallowing the exception, threading a new
`RLMResult.extract_error` field through `droste_runner`'s response mapping and
`droste_cli`'s JSON/stderr output, and emitting a new `extract_error` structured
event (added to the Pyodide/Deno relay's event vocabulary in `events.ts`) so
hosts watching the event stream see the failure live, not just in the final
result. Deliberately does NOT retry or change what `_best_answer` returns for
the many *other*, non-max-iterations call sites that intentionally rely on
"if you just print(), the printed output becomes the final answer" (see
`prompts.py`'s documented backward-compatibility contract) — this fix is scoped
to the max-iterations-exhausted + extraction-attempted path only, where raw
debug output was never an intentional answer.

Separately, `pyodide_runtime.py`'s `run_for_host_pyodide` response never
surfaced `extracted`/`extract_error` to the host AT ALL (a second, adjacent
gap, pre-existing and independent of the above): the native (`droste_runner`)
substrate's `runner_adapter.py` already mapped `extracted` into its response,
but the Pyodide/Deno relay substrate — the one Cozy actually runs by default —
silently dropped it. Both fields are now included in the Pyodide substrate's
response, matching the native substrate.
