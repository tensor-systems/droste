// Phase 2 step 3 — the live run: a real Recall RLM query end-to-end in Pyodide,
// driving the UNCHANGED rcl_rlm.rlm.run_rlm with the two Pyodide adapters
// (RawExecutor + BridgedLLMClient) and live ModelRelay calls.
//   MODELRELAY_API_KEY=... RLM_QUESTION="..." ./run.sh phase3.ts
import { loadPyodide } from "npm:pyodide@0.29.4";

const [zipPath, dbDir] = Deno.args;
const apiKey = Deno.env.get("MODELRELAY_API_KEY");
if (!apiKey) {
  console.error("MODELRELAY_API_KEY not set");
  Deno.exit(1);
}
const question = Deno.env.get("RLM_QUESTION") ||
  "Find messages where someone expresses gratitude or appreciation, and summarize what people are grateful for.";

console.log("[pyodide] loading runtime + sqlite3...");
const pyodide = await loadPyodide();
await pyodide.loadPackage("sqlite3");

const zip = await Deno.readFile(zipPath);
pyodide.unpackArchive(zip, "zip", { extractDir: "/app" });
await pyodide.runPythonAsync(`import sys; sys.path.insert(0, "/app")`);
pyodide.mountNodeFS("/data", dbDir);

// Async host fetch -> ModelRelay. BridgedLLMClient run_sync()s this so the
// synchronous RLM loop blocks for the result.
pyodide.globals.set(
  "host_fetch",
  async (method: string, url: string, headersJson: string, body: string) => {
    const resp = await fetch(url, { method, headers: JSON.parse(headersJson), body });
    return await resp.text();
  },
);
pyodide.globals.set("api_key", apiKey);
pyodide.globals.set("question", question);

console.log(`[run] "${question}"`);
const t0 = Date.now();
const result = await pyodide.runPythonAsync(`
import json
from pyodide_runtime import RawExecutor, BridgedLLMClient
from rcl_rlm.rlm import run_rlm

client = BridgedLLMClient(host_fetch, api_key)
res = run_rlm(
    question=question,
    db_path="/data/shadow_corpus_test.db",
    contacts_db_path="/data/contacts.db",
    llm_client=client,
    executor_factory=RawExecutor,   # <- the only two swaps vs native
    max_calls=8,
    max_depth=2,
    max_iterations=3,
)
json.dumps({
    "answer": res.answer,
    "sub_calls": res.sub_calls_made,
    "iterations": res.iterations,
    "tokens": res.total_tokens,
    "retrieved": len(res.retrieved_guids or []),
    "error": res.error,
})
`);

const parsed = JSON.parse(result);
console.log("\n────────────── LIVE PYODIDE RLM RESULT ──────────────");
console.log("elapsed       :", ((Date.now() - t0) / 1000).toFixed(1) + "s");
console.log("sub_calls     :", parsed.sub_calls, " iterations:", parsed.iterations, " tokens:", parsed.tokens);
console.log("retrieved msgs:", parsed.retrieved);
console.log("error         :", parsed.error);
console.log("\nANSWER:\n" + (parsed.answer || "(empty)"));
console.log("─────────────────────────────────────────────────────");
