// Real two-interpreter conformance for the first-party filesystem/text provider.
// The configured data root exists only in the trusted provider interpreter;
// generated-code globals in the REPL interpreter reach it only over the generic
// ProviderService/BridgeProvider path, alongside a real SQLite source.
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

Deno.test("filesystem_text shares the generic bridge with SQLite and keeps its root non-ambient", async () => {
  const provider = await loadWithDroste();
  await provider.runPythonAsync(`
import os
import sqlite3
from droste import ConfiguredSource, ProviderCatalog
from droste.sources.bridge import ProviderService
from droste.sources.filesystem_text import filesystem_text_provider
from droste.sources.sql_local import sqlite_provider

os.mkdir("/trusted_docs")
with open("/trusted_docs/secret.txt", "w", encoding="utf-8") as stream:
    stream.write("broker-only filesystem fact\\n")
with open("/trusted_docs/guide.md", "w", encoding="utf-8") as stream:
    stream.write("# Guide\\n\\nBridge evidence.\\n")

_connection = sqlite3.connect(":memory:", check_same_thread=False)
_connection.execute("CREATE TABLE facts(value TEXT)")
_connection.execute("INSERT INTO facts VALUES ('sqlite fact')")
_connection.commit()
_registry = ProviderCatalog((sqlite_provider(), filesystem_text_provider())).bind(
    (
        ConfiguredSource("db", "sqlite"),
        ConfiguredSource("docs", "filesystem_text", {
            "root": "/trusted_docs",
            "include": ["**/*.txt", "**/*.md"],
        }),
    ),
    context=_connection,
)
_db_service = ProviderService(_registry.sources[0])
_docs_service = ProviderService(_registry.sources[1])
`);
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
import os
from droste import CapabilityBroker, ConfiguredSource, ProviderCatalog, SideEffect
from droste.sources.bridge import BridgeProvider

db_bridge = BridgeProvider(db_call)
docs_bridge = BridgeProvider(docs_call)
catalog = ProviderCatalog((
    db_bridge.registration(effects={"query": SideEffect.READ, "schema": SideEffect.READ}),
    docs_bridge.registration(effects={
        "list": SideEffect.READ,
        "read": SideEffect.READ,
        "grep": SideEffect.READ,
        "search": SideEffect.READ,
        "stat": SideEffect.READ,
    }),
))
registry = catalog.bind((
    ConfiguredSource("db", "sqlite"),
    ConfiguredSource("docs", "filesystem_text"),
))
broker = CapabilityBroker(registry.capability_registrations())
generated = registry.broker_globals(broker)

ambient_exists = os.path.exists("/trusted_docs/secret.txt")
try:
    open("/trusted_docs/secret.txt", encoding="utf-8").read()
    ambient_opened = True
except FileNotFoundError:
    ambient_opened = False

document = generated["docs"].read("secret.txt")
rows = generated["db"].query("SELECT value FROM facts")
listing = generated["docs"].list_files(limit=10)
json.dumps({
    "ambient_exists": ambient_exists,
    "ambient_opened": ambient_opened,
    "text": document["text"],
    "evidence": document["evidence"],
    "rows": rows,
    "paths": [item["path"] for item in listing["items"]],
    "bindings": {name: sorted(vars(value)) for name, value in generated.items()},
    "accessors": sorted([list(item) for item in registry.accessor_manifest().namespaced]),
    "prompt": registry.prompt_fragment(),
    "provider_globals_leaked": any(name in globals() for name in (
        "_registry", "_docs_service", "_db_service", "_connection"
    )),
})
`);
  const result = JSON.parse(raw);
  assertEquals(result.ambient_exists, false);
  assertEquals(result.ambient_opened, false);
  assertEquals(result.text, "broker-only filesystem fact\n");
  assertEquals(result.evidence.source_id, "docs");
  assertEquals(result.evidence.path, "secret.txt");
  assertEquals(result.rows, [{ value: "sqlite fact" }]);
  assertEquals(result.paths, ["guide.md", "secret.txt"]);
  assertEquals(result.bindings.db, ["get_schema", "query"]);
  assertEquals(result.bindings.docs, [
    "grep",
    "list_files",
    "read",
    "search",
    "stat",
  ]);
  assert(
    result.accessors.some((item: string[]) => item.join(".") === "docs.read"),
  );
  assert(result.prompt.includes("docs.read"));
  assert(result.prompt.includes("db.query"));
  assert(!result.prompt.includes("/trusted_docs"));
  assertEquals(result.provider_globals_leaked, false);
});
