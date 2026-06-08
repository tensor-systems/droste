# Spike: Deno + Pyodide as the RLM execution substrate

Feasibility spike for running rlm-core's RLM in a **Deno + Pyodide (CPython-on-WASM)**
sandbox instead of the native CPython-framework + wheelhouse it ships today — the
goal being a much simpler/safer macOS app bundle (one Deno binary + data, no native
`.so` signing, arch-independent WASM).

Run:
```bash
./run.sh spike.ts     # Phase 0: WASM viability (imports, host-tool bridge, blockers)
./run.sh phase1.ts    # Phase 1: data-layer fidelity, native vs Pyodide, on the corpus
```
`run.sh` stages `rlm-core/src` + the **verbatim** rcl_rlm data layer (`message_database.py`,
`sql_validator.py`, `exceptions.py`) into a zip Pyodide loads. Requires Deno; the
corpus DB is expected at `~/Library/Application Support/RecallRLM/` (override as arg 2).

## Phase 0 — viability (PASS)
- All of rlm-core imports cleanly under Pyodide, incl. `rlm_runner.runner`.
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
  httpx. Refactored to a lazy PEP-562 `__init__` — data layer eager, network/rlm-core lazy.
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

**Later hardening (Design B) — tracked in tensor-systems/rlm-core#3:** move the data layer
onto the trusted host as a separate Pyodide "DB service" context (verbatim `MessageDatabase`
+ read-only mount + ENFORCED `SqlValidator`), with the untrusted REPL context getting no
FS/no net and only the bridged tools. Matches DSPy's tools-only posture. Defense-in-depth,
post-v1.
