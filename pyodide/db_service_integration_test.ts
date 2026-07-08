// A'-2 DB-service split, real rcl_rlm/droste E2E (droste#3, cozy#807 items 4-5):
// two REAL Pyodide interpreters wired exactly like relay.ts's RLM_DB_SERVICE=1
// path, running the ACTUAL cozy rcl_rlm.rlm.run_rlm(data_source=...) seam
// against a real MessageDatabase — not a droste-only stub (that's
// bridge_source_integration_test.ts). Proves: has_contacts/default_max_calls
// thread correctly from the service interpreter to the REPL side (the two bugs
// the advisor caught before any code was written), the REPL interpreter never
// gets the DB, generated code's query() calls really reach the service-side
// MessageDatabase, and retrieved_guids survives the bridge via extra_methods.
//
// Requires a sibling cozy checkout (RCL_RLM_SRC env override, default
// ../../cozy/tools/rcl-rlm/src) — skipped if absent, matching
// build-rlm-deno.sh's own sibling-checkout convention.
//
// Run: deno test --allow-read --allow-env --allow-ffi db_service_integration_test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { loadPyodide } from "npm:pyodide@0.29.4";

const DROSTE_SRC = new URL("../src", import.meta.url).pathname;
const RUNTIME_DIR = new URL(".", import.meta.url).pathname;
const RCL_RLM_SRC = Deno.env.get("RCL_RLM_SRC") ??
  new URL("../../cozy/tools/rcl-rlm/src", import.meta.url).pathname;
const quiet = { stdout: () => {}, stderr: () => {} };

let rclRlmAvailable = false;
try {
  rclRlmAvailable = (await Deno.stat(RCL_RLM_SRC)).isDirectory;
} catch {
  rclRlmAvailable = false;
}

async function loadWithSources(): Promise<any> {
  const interp = await loadPyodide(quiet);
  await interp.loadPackage("sqlite3", { messageCallback: () => {}, errorCallback: () => {} });
  interp.mountNodeFS("/droste_src", DROSTE_SRC);
  interp.mountNodeFS("/rcl_rlm_src", RCL_RLM_SRC);
  interp.mountNodeFS("/pyodide_runtime", RUNTIME_DIR);
  await interp.runPythonAsync(
    `import sys
for p in ("/droste_src", "/rcl_rlm_src", "/pyodide_runtime"):
    if p not in sys.path:
        sys.path.insert(0, p)`,
  );
  return interp;
}

Deno.test({
  name: "A'-2 E2E: run_for_host_pyodide(bridge_call=...) against a real MessageDatabase, real rcl_rlm",
  ignore: !rclRlmAvailable,
  fn: async () => {
    // -- trusted "DB service" interpreter: holds the real MessageDatabase. -----
    const dbsvc = await loadWithSources();
    await dbsvc.runPythonAsync(`
import sqlite3
conn = sqlite3.connect("/tmp/shadow.db")
conn.executescript(
    "CREATE TABLE messages (rowid INTEGER PRIMARY KEY, guid TEXT NOT NULL, "
    "sender TEXT NOT NULL, text TEXT NOT NULL, timestamp INTEGER NOT NULL, "
    "chat_name TEXT, chat_id TEXT, is_from_me INTEGER DEFAULT 0, is_group INTEGER DEFAULT 0);"
    "INSERT INTO messages VALUES (1, 'g1', 'me', 'hello world', 1000, 'Test Chat', 'c1', 1, 0);"
    "INSERT INTO messages VALUES (2, 'g2', 'friend', 'hi there', 2000, 'Test Chat', 'c1', 0, 0);"
)
conn.commit()
conn.close()
`);
    const metaJson = await dbsvc.runPythonAsync(`
import json
from pyodide_runtime import build_db_service
_service, _meta = build_db_service("/tmp/shadow.db", None)
json.dumps(_meta)
`);
    const meta = JSON.parse(metaJson);
    assertEquals(meta.has_contacts, false);
    assertEquals(meta.default_max_calls, 12); // MIN_SUBCALLS clamp for a 2-row corpus

    const handle = dbsvc.globals.get("_service").handle;
    const bridgeCall = async (method: string, paramsJson: string) => handle(method, paramsJson) as string;

    // -- untrusted "REPL" interpreter: never sees the DB. ----------------------
    const repl = await loadWithSources();
    repl.globals.set("bridge_call", bridgeCall);

    // Scripted fake ModelRelay: one call, returns code that queries the REAL
    // service-side MessageDatabase (both a count and a guid-bearing row, to
    // prove retrieved_guids survives the bridge) and terminates in one
    // iteration.
    const fakeHostFetch = async (_m: string, _u: string, _h: string, _b: string) => {
      const code = [
        "```python",
        'rows = query("SELECT COUNT(*) AS n FROM messages")',
        'query("SELECT guid FROM messages ORDER BY rowid LIMIT 1")',
        'answer["content"] = str(rows[0]["n"])',
        'answer["ready"] = True',
        "```",
      ].join("\n");
      return JSON.stringify({
        output: [{ type: "message", role: "assistant", content: [{ type: "text", text: code }] }],
      });
    };
    repl.globals.set("host_fetch", fakeHostFetch);
    repl.globals.set(
      "request_json",
      JSON.stringify({ question: "how many messages are there", root_model: "test-model", max_iterations: 1 }),
    );

    const out = await repl.runPythonAsync(`
import json
from pyodide_runtime import run_for_host_pyodide
resp = run_for_host_pyodide(
    json.loads(request_json), host_fetch,
    bridge_call=bridge_call, has_contacts=False, default_max_calls=12,
)
json.dumps(resp)
`);
    const resp = JSON.parse(out);
    assertEquals(resp.error, null);
    assertEquals(resp.answer, "2"); // real COUNT(*) from the service-side DB
    assertEquals(resp.retrieved_guids, ["g1"]); // extra_methods wiring proof

    // -- isolation: the REPL interpreter never had the DB mounted. ------------
    const dbVisibleInRepl = await repl.runPythonAsync(`__import__("os").path.exists("/tmp/shadow.db")`);
    assert(!dbVisibleInRepl, "the DB file must not be reachable from the REPL interpreter");

    // -- security: a forged method name is rejected, not getattr'd. -----------
    const forged = await repl.runPythonAsync(`
import json
from pyodide.ffi import run_sync
raw = run_sync(bridge_call("__init__", "{}"))
json.dumps(json.loads(raw))
`);
    const forgedEnvelope = JSON.parse(forged);
    assertEquals(forgedEnvelope.ok, false);
    assertEquals(forgedEnvelope.error.type, "ValueError");
  },
});
