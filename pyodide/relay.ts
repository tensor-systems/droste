// Deno+Pyodide relay — a drop-in for the native `recall-rlm-helper`:
// reads a HostRequest JSON from stdin, runs the Recall RLM in Pyodide, writes a
// HostResponse JSON to stdout. Deno holds the only ambient capabilities (narrow
// net to ModelRelay + read of the DB dir); the Pyodide sandbox has none.
//
//   echo '<request json>' | deno run --allow-net=api.modelrelay.ai \
//       --allow-read --allow-env relay.ts <sources.zip>
//
// (.zip path is a dev convenience; production bundles sources as a directory.)
// Pin pyodide: an unpinned `npm:pyodide` lets `deno cache` drift to a new major
// (e.g. 314.0.0) whose sqlite3 package/API differs, breaking the offline bundle
// ("No known package with name 'sqlite3'"). 0.29.4 is the version this relay is
// built and tested against (run_sync + mountNodeFS + loadPackage("sqlite3")).
import { loadPyodide } from "npm:pyodide@0.29.4";
import { basename, dirname } from "node:path";
import { streamResponses } from "./stream.ts";
import { isModelRelayResponsesCall, splitCredentials, stripAndInjectAuth } from "./broker.ts";
import { isRlmEvent } from "./events.ts";

const sources = Deno.args[0]; // bundled Python sources DIR (prod) or a .zip (dev)
const request = JSON.parse(await new Response(Deno.stdin.readable).text());

const _enc = new TextEncoder();

// Fully resolve symlinks to a real on-disk path, returning the input unchanged
// if it cannot be resolved (e.g. the file does not exist yet).
function realPathOr(path: string): string {
  try {
    return Deno.realPathSync(path);
  } catch {
    return path;
  }
}

// Stage the bundled Python sources into a Pyodide interpreter's /app — shared
// between the REPL interpreter and (in DB-service mode) the second, trusted
// interpreter, both of which import from droste/rcl_rlm.
async function mountSources(interp: Awaited<ReturnType<typeof loadPyodide>>): Promise<void> {
  if (sources.endsWith(".zip")) {
    interp.unpackArchive(await Deno.readFile(sources), "zip", { extractDir: "/app" });
  } else {
    interp.mountNodeFS("/app", sources);
  }
  await interp.runPythonAsync(`import sys; sys.path.insert(0, "/app")`);
}

// The ONLY thing ever written to stdout — the HostResponse JSON.
async function writeHostResponse(resp: unknown): Promise<void> {
  await Deno.stdout.write(_enc.encode(JSON.stringify(resp) + "\n"));
}

// A'-2 sandbox split (droste#3): move the DB out of the untrusted REPL
// interpreter entirely, into a second, trusted "DB service" interpreter that
// the REPL only ever reaches through a bridged RPC call (droste.sources.bridge).
// Opt-in, default off — mirrors the RLM_BRIDGE=legacy kill-switch shape from
// A'-1. Orthogonal to RLM_BRIDGE: that flag governs credential visibility
// (which payload the sandbox sees, whether host_fetch strips/injects auth);
// this one governs DB locality. All four combinations of the two flags are
// coherent and each is a useful debugging bisect.
const DB_SERVICE = Deno.env.get("RLM_DB_SERVICE") === "1";

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
const hasContactsField = request.contacts_db_path && request.contacts_db_path !== "nil";
const realContactsPath = hasContactsField ? realPathOr(request.contacts_db_path) : null;

// In DB-service mode, build the trusted interpreter FIRST, before the
// untrusted REPL interpreter even exists — a bad db_path (or any bootstrap
// failure) then costs one interpreter boot, not two, and can never surface as
// a confusing bridge-timeout-shaped error from the REPL side.
let bridgeCall: ((method: string, paramsJson: string) => Promise<string>) | null = null;
let bridgeMeta: { has_contacts: boolean; default_max_calls: number } | null = null;
if (DB_SERVICE) {
  try {
    const quiet = { stdout: () => {}, stderr: () => {} };
    const dbsvc = await loadPyodide(quiet);
    await dbsvc.loadPackage("sqlite3", { messageCallback: () => {}, errorCallback: () => {} });
    await mountSources(dbsvc);

    dbsvc.mountNodeFS("/data", dbDir);
    const svcDbPath = "/data/" + basename(realDbPath);
    let svcContactsPath: string | null = null;
    if (realContactsPath) {
      const contactsDir = dirname(realContactsPath);
      svcContactsPath = contactsDir === dbDir
        ? "/data/" + basename(realContactsPath)
        : (() => {
          dbsvc.mountNodeFS("/contacts", contactsDir);
          return "/contacts/" + basename(realContactsPath);
        })();
    }
    // Globals, not string interpolation into the Python template — these are
    // real filesystem paths, not attacker-controlled prompt text, but there's
    // no reason to string-build Python source out of them either.
    dbsvc.globals.set("svc_db_path", svcDbPath);
    dbsvc.globals.set("svc_contacts_db_path", svcContactsPath);
    const metaJson = await dbsvc.runPythonAsync(`
import json
from pyodide_runtime import build_db_service
_service, _meta = build_db_service(svc_db_path, svc_contacts_db_path)
json.dumps(_meta)
`);
    bridgeMeta = JSON.parse(metaJson);
    const handle = dbsvc.globals.get("_service").handle;
    // Async wrapper: the REPL interpreter's Python side calls this via
    // run_sync, which requires an awaitable (proven in
    // bridge_source_integration_test.ts). handle() itself is synchronous.
    bridgeCall = async (method: string, paramsJson: string) => handle(method, paramsJson) as string;
  } catch (e) {
    await writeHostResponse({
      answer: null,
      error: { type: "DBServiceError", message: String(e instanceof Error ? e.message : e) },
    });
    Deno.exit(0); // match the existing contract: exit 0, error in the JSON body
  }
}

// Route Python stdout off the relay's stdout (which carries only the response
// JSON); silence the package loader's "Loading sqlite3" chatter too. Forward the
// RLM's structured events (NDJSON lines on Python's stderr — progress plus the
// loop events from #2: iteration_start/code/output/subcall) to the relay's own
// stderr so the host can render real-time progress; drop other stderr noise.
const py = await loadPyodide({
  stdout: () => {},
  stderr: (msg: string) => {
    for (const line of msg.split("\n")) {
      if (isRlmEvent(line)) {
        try {
          Deno.stderr.writeSync(_enc.encode(line.trim() + "\n"));
        } catch {
          // best-effort event forwarding; never let it break the run
        }
      }
    }
  },
});
// Needed even in DB-service mode: run_for_host_pyodide's bridge branch imports
// droste.sources.bridge, and droste.sources.__init__ eagerly imports
// sql_local.py, which imports the stdlib sqlite3 module — the REPL
// interpreter never OPENS a database, but it still needs the package loaded
// just to import the client-side BridgeDataSource.
await py.loadPackage("sqlite3", { messageCallback: () => {}, errorCallback: () => {} });
await mountSources(py);

if (DB_SERVICE) {
  // The whole point of the split: the untrusted interpreter never sees the
  // DB directory at all, mounted or otherwise.
  delete request.db_path;
  delete request.contacts_db_path;
} else {
  py.mountNodeFS("/data", dbDir);
  request.db_path = "/data/" + basename(realDbPath);
  if (realContactsPath) {
    request.contacts_db_path = "/data/" + basename(realContactsPath);
  } else {
    delete request.contacts_db_path;
  }
}

// Emit a structured event line to the relay's own stderr — the same one-way
// channel the host already reads for progress (see the loadPyodide stderr hook).
// Best-effort; a failed write must never break the run.
function emitEvent(obj: unknown): void {
  try {
    Deno.stderr.writeSync(_enc.encode(JSON.stringify(obj) + "\n"));
  } catch {
    // ignore
  }
}

// Streaming is on by default; RLM_STREAM=0 forces the legacy unary path (a
// no-rebuild kill switch). --allow-env is already granted (see usage header).
const STREAM_ENABLED = Deno.env.get("RLM_STREAM") !== "0";

// A′ sandbox split (#3): the untrusted interpreter must never hold the ModelRelay
// credential. The broker strips it from the request the sandbox sees, holds it
// host-side, and injects the auth header on every ModelRelay call below.
// RLM_BRIDGE=legacy restores the pre-A′ behavior (credential visible to the
// sandbox) as a no-rebuild kill switch, mirroring RLM_STREAM=0.
const BRIDGE_LEGACY = Deno.env.get("RLM_BRIDGE") === "legacy";
// splitCredentials reads whatever's left on `request` at this point, so the
// db_path/contacts_db_path mutation above (deleted in DB-service mode,
// rewritten to /data/... otherwise) is already reflected in sandboxRequest —
// no separate strip needed for the legacy payload.
const { creds, sandboxRequest } = splitCredentials(request);

// Host-brokered ModelRelay transport (Pyodide has no network of its own).
// The last HTTP error status (0 = none) is captured structurally here, where
// `r.status` is available, and injected into the HostResponse error below — so
// callers (e.g. the Swift app) can branch on it (402 = out of balance) without
// parsing the human-readable message string. Fresh per process (one run/query).
let lastHttpErrorStatus = 0;
py.globals.set("host_fetch", async (m: string, u: string, h: string, b: string) => {
  const headers = JSON.parse(h);
  // A′: host auth is authoritative — drop whatever auth header the sandbox sent
  // and inject the held credential. Scoped to the exact `POST /api/v1/responses`
  // ModelRelay call (isModelRelayResponsesCall parses method + URL), so the
  // credential is a single-purpose LLM-transport key — sandbox code can't reuse
  // it against other ModelRelay endpoints, another host, or over plaintext.
  // Legacy mode leaves the sandbox-assembled header untouched.
  if (!BRIDGE_LEGACY && isModelRelayResponsesCall(m, u)) {
    stripAndInjectAuth(headers, creds);
  }
  // Ask ModelRelay to stream /responses so reasoning tokens reach the host live.
  // Pyodide still receives one complete response (see streamResponses), so the
  // sync RLM loop is unaffected.
  const wantsStream = STREAM_ENABLED && m === "POST" && u.endsWith("/responses");
  if (wantsStream) {
    headers["Accept"] = 'application/x-ndjson; profile="responses-stream/v2"';
  }
  const r = await fetch(u, { method: m, headers, body: b });
  // Surface HTTP errors instead of returning an error/empty body that the
  // Python client would blindly json.loads() into a cryptic JSONDecodeError.
  if (!r.ok) {
    lastHttpErrorStatus = r.status;
    const text = await r.text();
    throw new Error(`ModelRelay HTTP ${r.status} ${r.statusText}: ${text.slice(0, 1000)}`);
  }
  // Stream only when we asked for it AND the server actually returned ndjson;
  // otherwise fall back to the unary path — behavior identical to before.
  const contentType = r.headers.get("content-type") || "";
  const isNdjson = contentType.includes("ndjson") || contentType.includes("event-stream");
  if (wantsStream && isNdjson) {
    return await streamResponses(r, (chunk) => emitEvent({ type: "reasoning_delta", text: chunk }));
  }
  return await r.text();
});
// A′: the sandbox receives the credential-stripped request; legacy keeps the
// full request (credential included) for the pre-A′ in-sandbox auth path.
py.globals.set("request_json", JSON.stringify(BRIDGE_LEGACY ? request : sandboxRequest));
if (DB_SERVICE) {
  py.globals.set("bridge_call", bridgeCall);
}

// bridge_call/has_contacts/default_max_calls are only threaded through in
// DB-service mode; run_for_host_pyodide ignores them when bridge_call is None.
const bridgeArgs = DB_SERVICE
  ? `, bridge_call=bridge_call, has_contacts=${bridgeMeta!.has_contacts ? "True" : "False"}, default_max_calls=${bridgeMeta!.default_max_calls}`
  : "";

const out = await py.runPythonAsync(`
import json, io, contextlib
from pyodide_runtime import run_for_host_pyodide
# Capture the RLM's progress prints so stdout carries only the response JSON.
_buf = io.StringIO()
try:
    with contextlib.redirect_stdout(_buf):
        resp = run_for_host_pyodide(json.loads(request_json), host_fetch${bridgeArgs})
except Exception as e:
    resp = {"answer": None, "error": {"type": type(e).__name__, "message": str(e)}}
json.dumps(resp)
`);

// When the run failed due to an HTTP error, attach the captured status to the
// structured error so the host can distinguish out-of-balance (402) from other
// failures. Best-effort: if our own output isn't parseable JSON, fall back to
// it unchanged (the error's message still carries the detail).
let outText = out;
if (lastHttpErrorStatus !== 0) {
  try {
    const parsed = JSON.parse(out);
    if (parsed && typeof parsed.error === "object" && parsed.error !== null) {
      parsed.error.status = lastHttpErrorStatus;
      outText = JSON.stringify(parsed);
    }
  } catch {
    // Output left unchanged; status enrichment is best-effort, not load-bearing.
  }
}
await Deno.stdout.write(new TextEncoder().encode(outText + "\n"));
