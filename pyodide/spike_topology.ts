// Spike: measure the cost of the sandbox-split topology options.
//
//   Option A — TWO Pyodide contexts in one Deno process:
//              untrusted REPL  +  trusted "DB service".
//              query() in the REPL run_sync's a bridge call that Deno forwards
//              into the DB-service interpreter, which validates + runs the query.
//
//   Option B — ONE Pyodide context (untrusted REPL) + DB service in Deno/JS.
//              query() run_sync's a bridge call answered by a host-side JS handler.
//
// Measures: (1) can two Pyodide interpreters coexist + bridge (feasibility of A),
// (2) isolation between them (globals + FS), (3) RSS cost of the 2nd interpreter,
// (4) per-call bridge round-trip latency for A vs B (a query-heavy RLM makes many
// data.* calls per run, so per-call overhead compounds).
//
// NOTE: uses a pure-Python dict "corpus" + validator, NOT sqlite3 — the sqlite3
// Pyodide package won't load in this env (0.29.4 vs 314.0.2 cache drift; it fails
// in a SINGLE interpreter too, so it's an env issue, not a topology property).
// Phase 1 already proved sqlite3 + MessageDatabase run verbatim in one Pyodide;
// what this spike isolates is the *interpreter/bridge* cost, which is orthogonal
// to which SQL engine the trusted side uses.
//
// Run: deno run --allow-read --allow-env --allow-ffi spike_topology.ts

import { loadPyodide } from "npm:pyodide@0.29.4";

const CALLS = 300; // bridge round-trips to benchmark
const SQL = "SELECT id, body FROM messages WHERE id < 3";
const EXPECT = JSON.stringify([{ id: 0, body: "hi" }, { id: 1, body: "there" }]);

const rssMB = () => Math.round((Deno.memoryUsage().rss / 1024 / 1024) * 10) / 10;
const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const p95 = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(s.length * 0.95))]; };
const now = () => performance.now();
const quiet = { stdout: () => {}, stderr: () => {} };

// A pure-Python trusted data layer: a dict "corpus" + an ENFORCED validator.
// Stands in for MessageDatabase + SqlValidator. `db_query` is the boundary the
// untrusted REPL cannot bypass in option A (it lives in the OTHER interpreter).
const DB_SERVICE_PY = `
import json
_MESSAGES = [{"id":0,"body":"hi"},{"id":1,"body":"there"},{"id":2,"body":"friend"}]
_DANGER = ("DROP","DELETE","INSERT","UPDATE","CREATE","ALTER","ATTACH","PRAGMA")
def db_query(sql):
    up = sql.upper()
    if any(k in up for k in _DANGER):
        raise ValueError("SqlValidator: rejected non-read SQL")
    # Toy "engine": support the one benchmark query + the write-attempt path.
    if "ID < 3" in up:
        return json.dumps([m for m in _MESSAGES if m["id"] < 3][:2])
    return json.dumps(_MESSAGES)
`;
// The untrusted REPL's query() proxy — identical body in both options; only what
// _js_query resolves to differs (2nd interp vs host JS).
const REPL_PROXY_PY = `
import json
from pyodide.ffi import run_sync
def query(sql):
    r = run_sync(_js_query(sql))
    return json.loads(r if isinstance(r, str) else r)
`;

const results: Record<string, unknown> = {};
console.error("== baseline =="); const m0 = rssMB();
console.error(`RSS baseline (module loaded, no interpreter): ${m0} MB`);

// -- REPL interpreter (needed by BOTH options) --------------------------------
console.error("\n== loading REPL (untrusted) interpreter ==");
let t = now();
const repl = await loadPyodide(quiet);
const tReplLoad = Math.round(now() - t);
const mRepl = rssMB();
console.error(`REPL loaded in ${tReplLoad}ms; RSS ${mRepl} MB (+${(mRepl - m0).toFixed(1)})`);
await repl.runPythonAsync(REPL_PROXY_PY);

// -- OPTION A: second DB-service interpreter ---------------------------------
console.error("\n== OPTION A: loading 2nd (DB-service) interpreter ==");
t = now();
const dbsvc = await loadPyodide(quiet);
const tDbLoad = Math.round(now() - t);
const mBoth = rssMB();
console.error(`2nd interp loaded in ${tDbLoad}ms; RSS ${mBoth} MB (+${(mBoth - mRepl).toFixed(1)} for 2nd interp)`);
await dbsvc.runPythonAsync(DB_SERVICE_PY);
const dbQuery = dbsvc.globals.get("db_query");

// Bridge: REPL query() -> run_sync(_js_query) -> Deno forwards into dbsvc.
// This call runs on Deno's loop WHILE the REPL is suspended in run_sync — the
// crux feasibility question for A. dbQuery is a sync PyProxy from dbsvc.
// async to match production host_fetch: run_sync requires an awaitable.
const jsQueryA = async (sql: string): Promise<string> => dbQuery(sql) as string;
repl.globals.set("_js_query", jsQueryA);

let feasibleA = false, validatorA = false;
try {
  // Compare structure, not JSON formatting (Python json.dumps spaces != JS).
  feasibleA = await repl.runPythonAsync(`query(${JSON.stringify(SQL)}) == [{"id":0,"body":"hi"},{"id":1,"body":"there"}]`);
  try { await repl.runPythonAsync(`query("DELETE FROM messages")`); } catch { validatorA = true; }
} catch (e) {
  console.error("OPTION A bridge FAILED:", e instanceof Error ? e.message.split("\n")[0] : String(e));
}
console.error(`A feasible (cross-interp query correct): ${feasibleA}; validator enforced: ${validatorA}`);

// Isolation: globals + FS must not cross between the two interpreters.
await repl.runPythonAsync(`SECRET_IN_REPL = 1234`);
const leakGlobals = await dbsvc.runPythonAsync(`"SECRET_IN_REPL" in dir()`);
await repl.runPythonAsync(`open("/tmp/repl_only.txt","w").write("x")`);
const leakFS = await dbsvc.runPythonAsync(`__import__("os").path.exists("/tmp/repl_only.txt")`);
console.error(`isolation: globals leak=${leakGlobals} (want false), FS leak=${leakFS} (want false)`);

const latA: number[] = [];
for (let i = 0; i < CALLS; i++) { const s = now(); await repl.runPythonAsync(`query(${JSON.stringify(SQL)})`); latA.push(now() - s); }
console.error(`A latency/${CALLS}: median ${median(latA).toFixed(3)}ms  p95 ${p95(latA).toFixed(3)}ms`);

results.optionA = {
  feasible: feasibleA, validator_enforced: validatorA,
  globals_isolated: !leakGlobals, fs_isolated: !leakFS,
  second_interp_rss_mb: Math.round((mBoth - mRepl) * 10) / 10,
  second_interp_load_ms: tDbLoad,
  bridge_median_ms: Math.round(median(latA) * 1000) / 1000,
  bridge_p95_ms: Math.round(p95(latA) * 1000) / 1000,
};

// -- OPTION B: DB service in Deno/JS (reuse the same REPL interpreter) --------
console.error("\n== OPTION B: DB service in Deno/JS ==");
const MESSAGES = [{ id: 0, body: "hi" }, { id: 1, body: "there" }, { id: 2, body: "friend" }];
const DANGER = ["DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", "ATTACH", "PRAGMA"];
const jsQueryB = async (sql: string): Promise<string> => {
  const up = sql.toUpperCase();
  if (DANGER.some((k) => up.includes(k))) throw new Error("SqlValidator: rejected non-read SQL");
  return JSON.stringify(up.includes("ID < 3") ? MESSAGES.filter((m) => m.id < 3).slice(0, 2) : MESSAGES);
};
repl.globals.set("_js_query", jsQueryB);
let feasibleB = false, validatorB = false;
feasibleB = await repl.runPythonAsync(`query(${JSON.stringify(SQL)}) == [{"id":0,"body":"hi"},{"id":1,"body":"there"}]`);
try { await repl.runPythonAsync(`query("DELETE FROM messages")`); } catch { validatorB = true; }
const latB: number[] = [];
for (let i = 0; i < CALLS; i++) { const s = now(); await repl.runPythonAsync(`query(${JSON.stringify(SQL)})`); latB.push(now() - s); }
console.error(`B feasible: ${feasibleB}; validator: ${validatorB}`);
console.error(`B latency/${CALLS}: median ${median(latB).toFixed(3)}ms  p95 ${p95(latB).toFixed(3)}ms`);

results.optionB = {
  feasible: feasibleB, validator_enforced: validatorB, second_interp_rss_mb: 0,
  bridge_median_ms: Math.round(median(latB) * 1000) / 1000,
  bridge_p95_ms: Math.round(p95(latB) * 1000) / 1000,
};
results.memory = { baseline_mb: m0, one_interp_mb: mRepl, two_interp_mb: mBoth, repl_load_ms: tReplLoad, second_interp_load_ms: tDbLoad };
results.env = { deno: Deno.version.deno, v8: Deno.version.v8, calls: CALLS, note: "pure-Python data layer; sqlite3 pkg unloadable in this env (cache drift), orthogonal to topology cost" };

console.error("\n== RESULT (JSON) ==");
console.log(JSON.stringify(results, null, 2));
