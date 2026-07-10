// Bridge-backed DataSource E2E (A'-2 sandbox split): two REAL Pyodide interpreters,
// wired the way relay.ts's eventual DB-service split will wire them, running the
// actual droste.sources.bridge contract (not the toy validator spike_topology.ts
// used) — proves DataSourceService <-> BridgeDataSource works over the real
// cross-interpreter transport, not just the in-process Python loopback the unit
// tests in tests/test_bridge_source.py exercise.
//
// Deliberately droste-only: the "trusted" interpreter hosts a small stub
// DataSource shaped like a real one (asymmetric capabilities: sql+schema, no
// search/get/recent), not a host app's product-specific data source (which
// isn't available outside that host's bundle), and not droste's own
// LocalSqlDataSource, whose query() arms a threading.Timer for its statement
// timeout — threading is unavailable under Pyodide/WASM, a real but SEPARATE
// constraint tracked by #8, orthogonal to the bridge wiring under test
// here. This validates the same wire contract a host adapter drives with a
// different source on the other end.
//
// Run: deno test --allow-read --allow-env --allow-ffi bridge_source_integration_test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { loadPyodide } from "npm:pyodide@0.29.4";

const SRC_DIR = new URL("../src", import.meta.url).pathname;
const quiet = { stdout: () => {}, stderr: () => {} };

async function loadWithDroste(): Promise<any> {
  const interp = await loadPyodide(quiet);
  // droste.sources.__init__ eagerly imports sql_local.py, which imports the
  // stdlib sqlite3 module — needed by BOTH interpreters here just to import
  // droste.sources.bridge, even though neither source below touches sqlite3.
  await interp.loadPackage("sqlite3", { messageCallback: () => {}, errorCallback: () => {} });
  interp.mountNodeFS("/app", SRC_DIR);
  await interp.runPythonAsync(`import sys; sys.path.insert(0, "/app")`);
  return interp;
}

Deno.test("A'-2 wiring: BridgeDataSource in the REPL interpreter round-trips a real source in the DB-service interpreter", async () => {
  // -- trusted "DB service" interpreter: holds the real DataSource. -----------
  const dbsvc = await loadWithDroste();
  await dbsvc.runPythonAsync(`
from droste.sources.bridge import DataSourceService

# Asymmetric capabilities (sql+schema only, no search/get/recent) — shaped
# like a real source (e.g. LocalSqlDataSource), so the capability-gating test
# below exercises the same rejection path a real source would hit.
class PeopleSource:
    def name(self):
        return "people"

    def capabilities(self):
        return {"sql": True, "search": False, "get": False, "recent": False, "schema": True, "stats": False}

    def get_schema(self):
        return "people(id INTEGER, name TEXT)"

    def get_stats(self):
        return {}

    def search(self, *args, **kwargs):
        return []

    def query(self, sql):
        return [{"id": 1, "name": "ada"}, {"id": 2, "name": "grace"}]

    def get(self, id):
        return None

    def get_recent(self, days=7, limit=100):
        return []

_source = PeopleSource()
_service = DataSourceService(_source)
_DBSVC_MARKER = "only visible in the DB-service interpreter"
`);
  const handle = dbsvc.globals.get("_service").handle;

  // -- untrusted "REPL" interpreter: never sees the real source. ---------------
  const repl = await loadWithDroste();

  // The bridge: an async JS forwarder into the OTHER interpreter, set as a
  // global the REPL's Python side calls synchronously via run_sync — same
  // shape as relay.ts's host_fetch and spike_topology.ts's _js_query.
  const bridgeCall = async (method: string, paramsJson: string): Promise<string> => {
    return handle(method, paramsJson) as string;
  };
  repl.globals.set("bridge_call", bridgeCall);

  const result = await repl.runPythonAsync(`
import json
from droste.sources.bridge import BridgeDataSource

db = BridgeDataSource(bridge_call, name="db")
rows = db.query("SELECT * FROM people ORDER BY id")
schema = db.get_schema()
json.dumps({"rows": rows, "schema": schema})
`);
  const parsed = JSON.parse(result);
  assertEquals(parsed.rows, [
    { id: 1, name: "ada" },
    { id: 2, name: "grace" },
  ]);
  assertEquals(parsed.schema, "people(id INTEGER, name TEXT)");

  // -- isolation: the REPL interpreter has its own globals/FS, entirely
  // separate from the DB-service interpreter (mirrors probe_dual_sqlite.ts's
  // isolation check). The REPL sees only what came back over the bridge —
  // never the source object, the marker global, or dbsvc's memory space.
  const markerLeaked = await repl.runPythonAsync(`"_DBSVC_MARKER" in dir()`);
  assert(!markerLeaked, "DB-service interpreter globals must not leak into the REPL interpreter");
  const sourceLeaked = await repl.runPythonAsync(`"_source" in dir() or "_service" in dir()`);
  assert(!sourceLeaked, "the real DataSource/DataSourceService must not exist in the REPL interpreter");

  // -- security: a forged/unknown method name is rejected, not getattr'd. -----
  const forged = await repl.runPythonAsync(`
import json
from pyodide.ffi import run_sync
raw = run_sync(bridge_call("__init__", "{}"))
json.dumps(json.loads(raw))
`);
  const forgedEnvelope = JSON.parse(forged);
  assertEquals(forgedEnvelope.ok, false);
  assertEquals(forgedEnvelope.error.type, "ValueError");
  assert(forgedEnvelope.error.message.includes("unknown bridge method"));

  // -- security: a capability-gated method (search, disabled on this source) --
  // is rejected by the SERVICE even when called directly, not merely absent
  // from the BridgeDataSource-composed sandbox namespace.
  const gated = await repl.runPythonAsync(`
import json
from pyodide.ffi import run_sync
raw = run_sync(bridge_call("search", json.dumps({"args": ["anything"], "kwargs": {}})))
json.dumps(json.loads(raw))
`);
  const gatedEnvelope = JSON.parse(gated);
  assertEquals(gatedEnvelope.ok, false);
  assertEquals(gatedEnvelope.error.type, "PermissionError");
});
