// Bridge-backed provider E2E (A'-2 sandbox split): two REAL Pyodide interpreters,
// wired the way relay.ts's eventual DB-service split will wire them, running the
// actual droste.sources.bridge contract (not the toy validator spike_topology.ts
// used) — proves ProviderService <-> BridgeProvider works over the real
// cross-interpreter transport, not just the in-process Python loopback the unit
// tests in tests/test_bridge_source.py exercise.
//
// Deliberately droste-only: the "trusted" interpreter hosts a small stub runtime
// behind Droste's SQLite provider manifest (query+schema only), not a host app's
// product-specific provider (which isn't available outside that host's bundle),
// and not droste's own LocalSqlRuntime, whose query() arms a threading.Timer for
// its statement timeout — threading is unavailable under Pyodide/WASM, a real
// but SEPARATE constraint tracked by #8, orthogonal to the bridge wiring under
// test here. This validates the same wire contract a host adapter drives with a
// different provider runtime on the other end.
//
// Run: deno test --allow-read --allow-env --allow-ffi bridge_source_integration_test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { loadPyodide } from "../src/droste/substrates/_relay/deps.ts";
import { startProviderDuplex } from "../src/droste/substrates/_relay/provider_duplex.ts";

const SRC_DIR = new URL("../src", import.meta.url).pathname;
const quiet = { stdout: () => {}, stderr: () => {} };

async function loadWithDroste(): Promise<any> {
  const interp = await loadPyodide(quiet);
  // droste.sources.__init__ eagerly imports sql_local.py, which imports the
  // stdlib sqlite3 module — needed by BOTH interpreters here just to import
  // droste.sources.bridge, even though neither source below touches sqlite3.
  // errorCallback must surface: a swallowed load failure (e.g. no network to
  // fetch the unvendored wheel) otherwise resurfaces later as a baffling
  // ModuleNotFoundError inside the interpreter.
  await interp.loadPackage("sqlite3", {
    messageCallback: () => {},
    errorCallback: (msg: string) =>
      console.error(`loadPackage(sqlite3): ${msg}`),
  });
  interp.mountNodeFS("/app", SRC_DIR);
  await interp.runPythonAsync(`import sys; sys.path.insert(0, "/app")`);
  return interp;
}

Deno.test("A'-2 wiring: BridgeProvider in the REPL interpreter round-trips a provider runtime in the DB-service interpreter", async () => {
  // -- trusted "DB service" interpreter: holds the live provider runtime. -----
  const dbsvc = await loadWithDroste();
  await dbsvc.runPythonAsync(`
from droste import ConfiguredSource, ProviderCatalog, ProviderRegistration, ProviderRuntime, SideEffect
from droste.sources.bridge import ProviderService
from droste.sources.sql_local import SQLITE_PROVIDER_MANIFEST

def _query(execution, sql):
    execution.check()
    execution.checkpoint(tokens=0, subcalls=0)
    return [{"id": 1, "name": "ada"}, {"id": 2, "name": "grace"}]

def _schema(execution):
    execution.check()
    return "people(id INTEGER, name TEXT)"

def _bind(source, context=None):
    return ProviderRuntime(
        handlers={"query": _query, "schema": _schema},
        source_description="people(id INTEGER, name TEXT)",
    )

_registration = ProviderRegistration(
    manifest=SQLITE_PROVIDER_MANIFEST,
    effects={"query": SideEffect.READ, "schema": SideEffect.READ},
    binder=_bind,
)
_source = ProviderCatalog((_registration,)).bind(
    (ConfiguredSource("db", "sqlite"),)
).sources[0]
_service = ProviderService(_source)
_DBSVC_MARKER = "only visible in the DB-service interpreter"
`);
  const handle = dbsvc.globals.get("_service").handle;

  // -- untrusted "REPL" interpreter: never sees the real source. ---------------
  const repl = await loadWithDroste();

  // The bridge: an async JS forwarder into the OTHER interpreter, set as a
  // global the REPL's Python side calls synchronously via run_sync — same
  // shape as relay.ts's host_fetch and spike_topology.ts's _js_query.
  const bridgeCall = async (
    method: string,
    paramsJson: string,
  ): Promise<string> => {
    return handle(method, paramsJson) as string;
  };
  repl.globals.set("bridge_call", bridgeCall);

  const result = await repl.runPythonAsync(`
import json
from droste import CapabilityBroker, ConfiguredSource, ProviderCatalog, SideEffect
from droste.sources.bridge import BridgeProvider

bridge = BridgeProvider(bridge_call)
registration = bridge.registration(
    effects={"query": SideEffect.READ, "schema": SideEffect.READ},
)
registry = ProviderCatalog((registration,)).bind(
    (ConfiguredSource(bridge.source_id, bridge.manifest.provider_type),),
    default_source_id=bridge.source_id,
)
broker = CapabilityBroker(registry.capability_registrations())
db = registry.broker_globals(broker)["db"]
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
  assert(
    !markerLeaked,
    "DB-service interpreter globals must not leak into the REPL interpreter",
  );
  const sourceLeaked = await repl.runPythonAsync(
    `"_source" in dir() or "_service" in dir()`,
  );
  assert(
    !sourceLeaked,
    "the live provider runtime and ProviderService must not exist in the REPL interpreter",
  );

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

  // -- security: an operation absent from the immutable provider manifest is
  // rejected by the SERVICE even when called directly, not merely absent from
  // the BridgeProvider-composed sandbox namespace.
  const gated = await repl.runPythonAsync(`
import json
from pyodide.ffi import run_sync
raw = run_sync(bridge_call("invoke", json.dumps({
    "operation_id": "search",
    "args": ["anything"],
    "kwargs": {},
    "execution": {
        "version": 1,
        "call_id": "forged-call",
        "run_id": "forged-run",
        "parent_run_id": None,
        "deadline_remaining_ms": 1000,
        "reservation": {"tokens": 0, "subcalls": 0, "wall_ms": 1000, "depth": 0},
        "cancellation_requested": False,
    },
})))
json.dumps(json.loads(raw))
`);
  const gatedEnvelope = JSON.parse(gated);
  assertEquals(gatedEnvelope.ok, false);
  assertEquals(gatedEnvelope.error.type, "PermissionError");
});

Deno.test("bridge v2 pumps cancellation, checkpoints, terminal races, and loss across two Pyodide interpreters", async () => {
  const dbsvc = await loadWithDroste();
  await dbsvc.runPythonAsync(`
from droste import ConfiguredSource, ProviderCatalog, ProviderRegistration, ProviderRuntime, SideEffect
from droste.sources.bridge import ProviderService
from droste.sources.sql_local import SQLITE_PROVIDER_MANIFEST

def _query(execution, scenario):
    if scenario == "checkpoints":
        execution.checkpoint(tokens=1, subcalls=1)
        execution.checkpoint(tokens=1, subcalls=1)
        execution.checkpoint(tokens=3, subcalls=2)
    elif scenario in {"cancel", "loss", "cancel_terminal"}:
        execution.checkpoint(tokens=2, subcalls=1)
        if scenario != "cancel_terminal":
            execution.check()
    return [{"scenario": scenario}]

def _schema(execution):
    execution.check()
    return "scenarios(name TEXT)"

def _bind(source, context=None):
    return ProviderRuntime(
        handlers={"query": _query, "schema": _schema},
        source_description="scenarios(name TEXT)",
    )

_registration = ProviderRegistration(
    manifest=SQLITE_PROVIDER_MANIFEST,
    effects={"query": SideEffect.READ, "schema": SideEffect.READ},
    binder=_bind,
)
_source = ProviderCatalog((_registration,)).bind(
    (ConfiguredSource("db", "sqlite"),)
).sources[0]
_service = ProviderService(_source)
`);
  const unaryHandle = dbsvc.globals.get("_service").handle;
  let busy = false;
  const terminalFrames: Record<string, number> = {};

  const unaryCall = async (
    method: string,
    paramsJson: string,
  ): Promise<string> => unaryHandle(method, paramsJson) as string;

  const duplexCall = (method: string, paramsJson: string): any => {
    const payload = JSON.parse(paramsJson);
    const scenario = payload.args[0] as string;
    let providerCheckpointAcked = false;
    const pump = startProviderDuplex(async (emit) => {
      if (busy) throw new Error("test provider interpreter is busy");
      busy = true;
      dbsvc.globals.set("_duplex_method", method);
      dbsvc.globals.set("_duplex_params", paramsJson);
      const providerEmit = async (frameJson: string): Promise<string> => {
        if (scenario === "loss" && providerCheckpointAcked) {
          throw new Error("provider interpreter was terminated");
        }
        const ack = await emit(frameJson);
        if (JSON.parse(frameJson).kind === "checkpoint") {
          providerCheckpointAcked = true;
        }
        return ack;
      };
      dbsvc.globals.set("_duplex_emit", providerEmit);
      try {
        const raw = await dbsvc.runPythonAsync(
          `_service.handle_duplex(_duplex_method, _duplex_params, _duplex_emit)`,
        );
        const envelope = JSON.parse(raw);
        if (!envelope.ok) throw new Error(envelope.error.message);
      } finally {
        dbsvc.globals.delete("_duplex_method");
        dbsvc.globals.delete("_duplex_params");
        dbsvc.globals.delete("_duplex_emit");
        busy = false;
      }
    });
    let checkpointSeen = false;
    return {
      receive: async () => {
        const frameJson = await pump.receive();
        const frame = JSON.parse(frameJson);
        if (frame.kind === "terminal") {
          terminalFrames[scenario] = (terminalFrames[scenario] ?? 0) + 1;
        }
        if (frame.kind === "checkpoint") checkpointSeen = true;
        if (
          (scenario === "cancel" && checkpointSeen && frame.kind === "check") ||
          (scenario === "cancel_terminal" && frame.kind === "terminal")
        ) {
          assertEquals(pump.requestActiveCancellation(), true);
        }
        return frameJson;
      },
      send: (ackJson: string) => pump.send(ackJson),
      cancellation_requested: (id: string) => pump.cancellation_requested(id),
      requestActiveCancellation: () => pump.requestActiveCancellation(),
      close: () => pump.close(),
    };
  };

  const repl = await loadWithDroste();
  repl.globals.set("unary_call", unaryCall);
  repl.globals.set("duplex_call", duplexCall);
  const result = await repl.runPythonAsync(`
import json
from droste import (
    CapabilityAdmission, CapabilityBroker, CapabilityMetadata,
    CapabilityReservation, ConfiguredSource, ProviderCatalog, SideEffect,
)
from droste.sources.bridge import BridgeProvider

class Authority:
    def __init__(self):
        self.call_id = None
        self.checkpoints = []
        self.settlements = 0
    def admit(self, call):
        self.call_id = call.call_id
        return CapabilityAdmission(CapabilityReservation(10, 3, 10_000, 0))
    def checkpoint(self, call, cumulative):
        self.checkpoints.append(cumulative.to_dict())
        return cumulative
    def settle(self, call, result, error, checkpoint, *, attempted):
        self.settlements += 1
        return CapabilityMetadata()

bridge = BridgeProvider(unary_call, duplex_call=duplex_call)
registration = bridge.registration(effects={"query": SideEffect.READ, "schema": SideEffect.READ})
registry = ProviderCatalog((registration,)).bind((ConfiguredSource("db", "sqlite"),))
capability_id = registry.capability_registrations()[0].descriptor.capability_id
out = {}
for scenario in ("checkpoints", "cancel", "cancel_terminal", "loss"):
    authority = Authority()
    broker = CapabilityBroker(registry.capability_registrations(), attempt_authority=authority)
    envelope = broker.call(capability_id, scenario)
    out[scenario] = {
        "ok": envelope.ok,
        "error": envelope.error.code if envelope.error else None,
        "error_message": envelope.error.message if envelope.error else None,
        "checkpoints": authority.checkpoints,
        "settlements": authority.settlements,
        "late_cancel": broker.cancel(authority.call_id),
    }
json.dumps(out)
`);
  const parsed = JSON.parse(result);
  assertEquals(parsed.checkpoints.ok, true);
  assertEquals(parsed.checkpoints.checkpoints, [
    { tokens: 1, subcalls: 1 },
    { tokens: 3, subcalls: 2 },
  ]);
  assertEquals(parsed.cancel.error, "cancelled", JSON.stringify(parsed.cancel));
  assertEquals(parsed.cancel.checkpoints, [{ tokens: 2, subcalls: 1 }]);
  assertEquals(parsed.cancel_terminal.error, "cancelled");
  assertEquals(parsed.loss.error, "bridge.transport_lost");
  assertEquals(parsed.loss.checkpoints, [{ tokens: 2, subcalls: 1 }]);
  for (const scenario of Object.values(parsed) as any[]) {
    assertEquals(scenario.settlements, 1);
    assertEquals(scenario.late_cancel, false);
  }
  assertEquals(terminalFrames.checkpoints, 1);
  assertEquals(terminalFrames.cancel, 1);
  assertEquals(terminalFrames.cancel_terminal, 1);
  assertEquals(terminalFrames.loss ?? 0, 0);
});
