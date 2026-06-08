// Phase 4 — parity: run the 8 semantic benchmark queries through the Pyodide
// substrate; the native semantic_extraction scoring is then applied to these
// answers (score_pyodide_parity.py) for an apples-to-apples vs the baseline.
//   MODELRELAY_API_KEY=... ./run.sh phase4.ts   (reads /tmp/sem_queries.json)
import { loadPyodide } from "npm:pyodide";

const [zipPath, dbDir] = Deno.args;
const apiKey = Deno.env.get("MODELRELAY_API_KEY");
if (!apiKey) { console.error("MODELRELAY_API_KEY not set"); Deno.exit(1); }
const queries = JSON.parse(await Deno.readTextFile("/tmp/sem_queries.json"));

console.log("[pyodide] loading runtime + sqlite3...");
const pyodide = await loadPyodide();
await pyodide.loadPackage("sqlite3");
const zip = await Deno.readFile(zipPath);
pyodide.unpackArchive(zip, "zip", { extractDir: "/app" });
await pyodide.runPythonAsync(`import sys; sys.path.insert(0, "/app")`);
pyodide.mountNodeFS("/data", dbDir);
pyodide.globals.set("host_fetch", async (m: string, u: string, h: string, b: string) => {
  const r = await fetch(u, { method: m, headers: JSON.parse(h), body: b });
  return await r.text();
});
pyodide.globals.set("api_key", apiKey);
pyodide.globals.set("queries_json", JSON.stringify(queries));

console.log(`[run] ${queries.length} semantic queries through Pyodide RLM...`);
const out = await pyodide.runPythonAsync(`
import json, time
from pyodide_runtime import RawExecutor, BridgedLLMClient
from rcl_rlm.rlm import run_rlm

client = BridgedLLMClient(host_fetch, api_key)
results = []
for q in json.loads(queries_json):
    t0 = time.time()
    try:
        # run_rlm defaults (match the native benchmark); only the two adapters differ.
        res = run_rlm(
            question=q["question"],
            db_path="/data/shadow_corpus_test.db",
            contacts_db_path="/data/contacts.db",
            llm_client=client,
            executor_factory=RawExecutor,
        )
        rec = {"id": q["id"], "answer": res.answer or "", "subcalls": res.sub_calls_made,
               "tokens": res.total_tokens, "error": res.error}
    except Exception as e:
        rec = {"id": q["id"], "answer": "", "subcalls": 0, "tokens": 0, "error": f"{type(e).__name__}: {e}"}
    rec["elapsed"] = round(time.time() - t0, 1)
    print(f"  [done] {q['id']}: subcalls={rec['subcalls']} tokens={rec['tokens']} {rec['elapsed']}s err={rec['error']}", flush=True)
    results.append(rec)
json.dumps(results)
`);

await Deno.writeTextFile("/tmp/pyodide_sem_results.json", out);
console.log("\nwrote /tmp/pyodide_sem_results.json (" + JSON.parse(out).length + " queries)");
