// Pyodide conformance for a manifest discovered from MCP tools/list. The stdio
// process remains in a trusted native host (covered by Python conformance);
// the untrusted interpreter receives only the existing generic provider bridge.
import { assert, assertEquals } from "jsr:@std/assert@1";
import { loadPyodide } from "../src/droste/substrates/_relay/deps.ts";

const SRC_DIR = new URL("../src", import.meta.url).pathname;
const quiet = { stdout: () => {}, stderr: () => {} };

async function loadWithDroste(): Promise<any> {
  const interp = await loadPyodide(quiet);
  await interp.loadPackage("sqlite3", {
    messageCallback: () => {},
    errorCallback: (message: string) =>
      console.error(`loadPackage(sqlite3): ${message}`),
  });
  interp.mountNodeFS("/app", SRC_DIR);
  await interp.runPythonAsync(`import sys; sys.path.insert(0, "/app")`);
  return interp;
}

Deno.test("MCP-discovered manifest uses the generic bridge beside SQLite", async () => {
  const provider = await loadWithDroste();
  await provider.runPythonAsync(`
import sqlite3
from droste import ConfiguredSource, ProviderCatalog, ProviderRuntime, SideEffect
from droste.capabilities import CapabilityOutcome
from droste.providers import ProviderManifest, ProviderRegistration
from droste.sources.bridge import ProviderService
from droste.sources.mcp_stdio import McpManifestPolicy, mcp_tools_to_manifest
from droste.sources.sql_local import sqlite_provider

_schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}
_manifest = mcp_tools_to_manifest(
    "reference_filesystem",
    [{
        "name": "ReadFile",
        "description": "Read one reference document.",
        "inputSchema": _schema,
        "outputSchema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    }],
    McpManifestPolicy(
        allowed_tools=("ReadFile",),
        bindings={"ReadFile": "read_file"},
        budget_classes={"ReadFile": "data.read"},
        max_descriptor_bytes=65536,
    ),
)

def _bind(source, context=None):
    del source, context
    def read_file(execution, *, path):
        execution.check()
        return CapabilityOutcome(result={"text": f"bridge fact from {path}"})
    return ProviderRuntime({"ReadFile": read_file})

_mcp_registration = ProviderRegistration(
    manifest=_manifest,
    effects={"ReadFile": SideEffect.READ},
    binder=_bind,
)
_connection = sqlite3.connect(":memory:", check_same_thread=False)
_connection.execute("CREATE TABLE facts(value TEXT)")
_connection.execute("INSERT INTO facts VALUES ('sqlite bridge fact')")
_connection.commit()
_registry = ProviderCatalog((sqlite_provider(), _mcp_registration)).bind(
    (
        ConfiguredSource("db", "sqlite"),
        ConfiguredSource("reference_docs", "reference_filesystem"),
    ),
    context=_connection,
)
_db_service = ProviderService(_registry.sources[0])
_docs_service = ProviderService(_registry.sources[1])
`);
  try {
    const dbHandle = provider.globals.get("_db_service").handle;
    const docsHandle = provider.globals.get("_docs_service").handle;
    const dbCall = async (method: string, params: string): Promise<string> =>
      dbHandle(method, params) as string;
    const docsCall = async (method: string, params: string): Promise<string> =>
      docsHandle(method, params) as string;

    const repl = await loadWithDroste();
    repl.globals.set("db_call", dbCall);
    repl.globals.set("docs_call", docsCall);
    const raw = await repl.runPythonAsync(`
import json
from droste import CapabilityBroker, ConfiguredSource, ProviderCatalog, SideEffect
from droste.sources.bridge import BridgeProvider

db_bridge = BridgeProvider(db_call)
docs_bridge = BridgeProvider(docs_call)
registry = ProviderCatalog((
    db_bridge.registration(effects={"query": SideEffect.READ, "schema": SideEffect.READ}),
    docs_bridge.registration(effects={"ReadFile": SideEffect.READ}),
)).bind((
    ConfiguredSource("db", "sqlite"),
    ConfiguredSource("reference_docs", "reference_filesystem"),
))
broker = CapabilityBroker(registry.capability_registrations())
generated = registry.broker_globals(broker)
payload = {
    "document": generated["reference_docs"].read_file(path="guide.md"),
    "rows": generated["db"].query("SELECT value FROM facts"),
    "bindings": {name: sorted(vars(value)) for name, value in generated.items()},
    "raw_operations": sorted(
        descriptor.capability_id.operation
        for descriptor in broker.describe().descriptors
    ),
    "prompt": registry.prompt_fragment(),
}
registry.close()
json.dumps(payload)
`);
    const result = JSON.parse(raw);
    assertEquals(result.document, { text: "bridge fact from guide.md" });
    assertEquals(result.rows, [{ value: "sqlite bridge fact" }]);
    assertEquals(result.bindings.reference_docs, ["read_file"]);
    assertEquals(result.bindings.db, ["get_schema", "query"]);
    assert(result.raw_operations.includes("ReadFile"));
    assert(result.prompt.includes("reference_docs.read_file"));
    assert(result.prompt.includes("db.query"));
    assert(!result.prompt.includes("MCP"));
  } finally {
    await provider.runPythonAsync(`_registry.close(); _registry.close()`);
  }
});
