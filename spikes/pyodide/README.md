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
- **The data layer must import without the network stack.** `rcl_rlm/__init__.py` eagerly
  imports `.modelrelay` → httpx, which won't load under Pyodide. The in-sandbox data layer
  needs to be importable standalone (this spike stages a minimal `rcl_rlm` with a stub
  `__init__`). Refactor target for v1.
- Keep `MessageDatabase` **verbatim**; fidelity is preserved when the same code runs over
  the same file. Reimplementing query()/the view in TS/Swift would reintroduce drift risk.

## Open design decision (Phase 2/3)
Phase 1 ran the data layer *inside* Pyodide with the DB mounted (parity with today's
isolation: the data layer shares the interpreter with generated code; network is blocked).
For stronger isolation, move the DB read out of the untrusted REPL onto the trusted host
(Swift) behind the `query()` tool — at the cost of either reimplementing MessageDatabase
host-side or running it in a separate trusted Python context. Network (`llm_query`) is
host-side either way (no sockets in WASM).
