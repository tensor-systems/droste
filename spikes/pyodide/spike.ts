// Phase 0 spike: can Deno + Pyodide host rlm-core, round-trip a host tool, and
// what breaks under WASM?   Run:  ./run.sh spike.ts
import { loadPyodide } from "npm:pyodide";

const zipPath = Deno.args[0];

console.log("[1] loading Pyodide in Deno...");
const pyodide = await loadPyodide();
console.log("    Pyodide version:", pyodide.version);

// Host tool: stubbed query() standing in for the trusted host's sqlite layer.
pyodide.globals.set("__host_query", (_sql: string) =>
  JSON.stringify([
    { contact: "Winona", body: "see you at 7", date: "2026-06-01" },
    { contact: "Winona", body: "thanks!!", date: "2026-06-02" },
    { contact: "Winona", body: "", date: "2026-06-03" },
  ])
);

const zip = await Deno.readFile(zipPath);
pyodide.unpackArchive(zip, "zip", { extractDir: "/app" });
await pyodide.runPythonAsync(`import sys; sys.path.insert(0, "/app")`);

console.log("\n[2] IMPORTS:", await pyodide.runPythonAsync(`
import json
out = {}
for mod in ["rlm_core","rlm_core.protocols.environment","rlm_core.loop.rlm","rlm_runner","rlm_runner.runner"]:
    try: __import__(mod); out[mod] = "OK"
    except Exception as e: out[mod] = f"{type(e).__name__}: {e}"
json.dumps(out)
`));

console.log("\n[3] BRIDGE ROUND-TRIP:", await pyodide.runPythonAsync(`
import json
def query(sql): return json.loads(__host_query(sql))
rows = query("SELECT body,date FROM messages_with_contacts WHERE contact='Winona'")
json.dumps({"rows": len(rows), "non_empty": len([r for r in rows if r['body']])})
`));

console.log("\n[4] WASM PROBES:", await pyodide.runPythonAsync(`
import json
r = {}
import signal
r["signal.setitimer"] = "present" if hasattr(signal,"setitimer") else "MISSING"
import threading
try:
    t = threading.Thread(target=lambda: None); t.start(); t.join(); r["threading.Thread"] = "ran"
except Exception as e: r["threading.Thread"] = f"{type(e).__name__}: {e}"
try:
    import urllib.request; urllib.request.urlopen("https://example.com", timeout=2); r["urllib"] = "works"
except Exception as e: r["urllib"] = f"{type(e).__name__}: {e}"
json.dumps(r)
`));
console.log("\n[done]");
