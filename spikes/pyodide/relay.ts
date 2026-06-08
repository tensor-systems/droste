// Deno+Pyodide relay — a drop-in for the native `recall-rlm-helper`:
// reads a HostRequest JSON from stdin, runs the Recall RLM in Pyodide, writes a
// HostResponse JSON to stdout. Deno holds the only ambient capabilities (narrow
// net to ModelRelay + read of the DB dir); the Pyodide sandbox has none.
//
//   echo '<request json>' | deno run --allow-net=api.modelrelay.ai \
//       --allow-read --allow-env relay.ts <sources.zip>
//
// (Spike: sources come from the staged zip. Production bundles them on disk.)
import { loadPyodide } from "npm:pyodide";
import { basename, dirname } from "node:path";

const sources = Deno.args[0]; // bundled Python sources DIR (prod) or a .zip (spike)
const request = JSON.parse(await new Response(Deno.stdin.readable).text());

// Fully resolve symlinks to a real on-disk path, returning the input unchanged
// if it cannot be resolved (e.g. the file does not exist yet).
function realPathOr(path: string): string {
  try {
    return Deno.realPathSync(path);
  } catch {
    return path;
  }
}

// Route Python stdout/stderr off the relay's stdout (which carries only the
// response JSON); silence the package loader's "Loading sqlite3" chatter too.
const py = await loadPyodide({ stdout: () => {}, stderr: () => {} });
await py.loadPackage("sqlite3", { messageCallback: () => {}, errorCallback: () => {} });
if (sources.endsWith(".zip")) {
  py.unpackArchive(await Deno.readFile(sources), "zip", { extractDir: "/app" });
} else {
  py.mountNodeFS("/app", sources); // bundled sources mounted into Pyodide's FS
}
await py.runPythonAsync(`import sys; sys.path.insert(0, "/app")`);

// Resolve symlinks before mounting. Pyodide's NODEFS mounts a host directory
// into its own VFS; if the DB file is a symlink to an absolute host path
// OUTSIDE that directory (e.g. the Recall→Cozy rename leaves
// Cozy/shadow.db -> RecallRLM/shadow.db), the in-VFS symlink target is
// unreachable and SQLite reports "unable to open database file". Mounting the
// resolved real directory (and opening the resolved file) sidesteps this.
// Falls back to the original path if it cannot be resolved (e.g. not yet on
// disk). The resolved directory must be in Deno's --allow-read (callers grant
// it; see RLMHelperRunner.denoArgs and the CLI relay invocation).
const realDbPath = realPathOr(request.db_path);
const dbDir = dirname(realDbPath);
py.mountNodeFS("/data", dbDir);
request.db_path = "/data/" + basename(realDbPath);
if (request.contacts_db_path && request.contacts_db_path !== "nil") {
  request.contacts_db_path = "/data/" + basename(realPathOr(request.contacts_db_path));
} else {
  delete request.contacts_db_path;
}

// Host-brokered ModelRelay transport (Pyodide has no network of its own).
py.globals.set("host_fetch", async (m: string, u: string, h: string, b: string) => {
  const r = await fetch(u, { method: m, headers: JSON.parse(h), body: b });
  const text = await r.text();
  // Surface HTTP errors instead of returning an error/empty body that the
  // Python client would blindly json.loads() into a cryptic JSONDecodeError.
  if (!r.ok) {
    throw new Error(`ModelRelay HTTP ${r.status} ${r.statusText}: ${text.slice(0, 1000)}`);
  }
  return text;
});
py.globals.set("request_json", JSON.stringify(request));

const out = await py.runPythonAsync(`
import json, io, contextlib
from pyodide_runtime import run_for_host_pyodide
# Capture the RLM's progress prints so stdout carries only the response JSON.
_buf = io.StringIO()
try:
    with contextlib.redirect_stdout(_buf):
        resp = run_for_host_pyodide(json.loads(request_json), host_fetch)
except Exception as e:
    resp = {"answer": None, "error": {"type": type(e).__name__, "message": str(e)}}
json.dumps(resp)
`);

// The ONLY thing written to stdout — the HostResponse JSON.
await Deno.stdout.write(new TextEncoder().encode(out + "\n"));
