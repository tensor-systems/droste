// Deno+Pyodide relay — a drop-in for a host's native runner helper:
// reads a HostRequest JSON from stdin, runs the RLM in Pyodide via a
// host-supplied Python adapter module, writes a HostResponse JSON to stdout.
// Deno holds the only ambient capabilities (narrow net to ModelRelay + trusted
// provider files); the Pyodide sandbox has none.
//
//   echo '<request json>' | DROSTE_RELAY_EVENT_FD=3 \
//       deno run --allow-net=api.modelrelay.ai --allow-read --allow-env \
//       relay.ts <sources.zip> <adapter_module> 3>events.ndjson
//
// (.zip path is a dev convenience; production bundles sources as a directory.)
// <adapter_module> is a Python module path, staged into <sources> alongside
// droste, that implements this relay's host-adapter contract — see
// pyodide/README.md ("Writing a host adapter") and examples/pyodide-host/ for
// droste's own minimal, no-third-party-deps example. This relay is itself
// droste-general: it knows nothing about any specific host's data layer or
// product wiring, only the adapter contract.
// The Pyodide runtime pin lives in deps.ts — the single bump site (#33).
import { loadPyodide } from "./deps.ts";
import {
  type ProviderDuplexSession,
  startProviderDuplex,
} from "./provider_duplex.ts";
import { basename, dirname } from "node:path";
import { streamResponses } from "./stream.ts";
import {
  isModelRelayResponsesCall,
  isRunnerCallback,
  splitCredentials,
  stripAndInjectAuth,
  stripAndInjectBearer,
} from "./broker.ts";
import { isRlmEvent } from "./events.ts";
import {
  type EventChannel,
  eventChannelFromEnvironment,
  RelayEventChannelError,
} from "./event_channel.ts";

const sources = Deno.args[0]; // bundled Python sources DIR (prod) or a .zip (dev)
const adapterModule = Deno.args[1]; // Python module implementing the host-adapter contract
// Trusted-channel value (Deno.args, set by the code that spawns this process —
// the same trust class as `sources`), but validated before ever reaching
// importlib: code-selection must never flow from the request body (the data
// plane), matching droste_runner's own adapter_module rule
// (droste_runner.run() rejects adapter_module from request files entirely).
if (!adapterModule || !/^[A-Za-z_][A-Za-z0-9_.]*$/.test(adapterModule)) {
  console.error(
    `usage: relay.ts <sources> <adapter_module>\n` +
      `<adapter_module> must be a dotted Python module path (got: ${
        JSON.stringify(adapterModule ?? null)
      })`,
  );
  Deno.exit(1);
}
const request = JSON.parse(await new Response(Deno.stdin.readable).text());
if (typeof request.run_id !== "string" || request.run_id.length === 0) {
  request.run_id = crypto.randomUUID();
}
// Relay-owned streaming telemetry is a child run: one sequence has one
// stamping owner, while parent_run_id correlates it with the engine record.
const relayRunId = crypto.randomUUID();
let relaySeq = 0;

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
// between the REPL interpreter and the trusted provider interpreter, both of
// which import from droste and the host's adapter package.
async function mountSources(
  interp: Awaited<ReturnType<typeof loadPyodide>>,
): Promise<void> {
  if (sources.endsWith(".zip")) {
    interp.unpackArchive(await Deno.readFile(sources), "zip", {
      extractDir: "/app",
    });
  } else {
    interp.mountNodeFS("/app", sources);
  }
  await interp.runPythonAsync(`import sys; sys.path.insert(0, "/app")`);
}

// The ONLY thing ever written to stdout — the HostResponse JSON.
async function writeHostResponse(resp: unknown): Promise<void> {
  await Deno.stdout.write(_enc.encode(JSON.stringify(resp) + "\n"));
}

function eventChannelFailureResponse(error: RelayEventChannelError): unknown {
  return {
    answer: null,
    error: {
      type: "RelayEventChannelError",
      code: error.code,
      message: error.message,
    },
  };
}

function writeEventChannelDiagnostic(error: RelayEventChannelError): void {
  try {
    console.error(`droste relay: event_channel_error code=${error.code}`);
  } catch {
    // fd2 diagnostics are best-effort and never replace the structured stdout
    // failure when the event channel itself is unavailable.
  }
}

async function terminateForEventChannel(
  error: RelayEventChannelError,
): Promise<never> {
  writeEventChannelDiagnostic(error);
  try {
    await writeHostResponse(eventChannelFailureResponse(error));
  } catch {
    Deno.exit(1);
  }
  Deno.exit(0);
}

let eventChannel: EventChannel;
try {
  eventChannel = eventChannelFromEnvironment();
} catch (error) {
  if (error instanceof RelayEventChannelError) {
    await terminateForEventChannel(error);
  }
  throw error;
}

// A'-2 sandbox split: keep host data out of the untrusted REPL interpreter.
// When a request names a database, a second trusted provider interpreter owns
// it and the REPL can reach it only through the brokered provider bridge.
// There is intentionally no direct-mount fallback: one egress path means the
// sandbox never gains ambient access to a database or its sibling files.
const hasDbPath = typeof request.db_path === "string" &&
  request.db_path.length > 0 && request.db_path !== "nil";

// Resolve symlinks before mounting. Pyodide's NODEFS mounts a host directory
// into its own VFS; if the DB file is a symlink to an absolute host path
// OUTSIDE that directory (e.g. an app-data rename that leaves
// NewApp/data.db -> OldApp/data.db), the in-VFS symlink target is
// unreachable and SQLite reports "unable to open database file". Mounting the
// resolved real directory (and opening the resolved file) sidesteps this.
// Falls back to the original path if it cannot be resolved (e.g. not yet on
// disk). The resolved directory must be in Deno's --allow-read (the code that
// spawns this relay grants it when building the deno invocation).
const realDbPath = hasDbPath ? realPathOr(request.db_path) : null;
const dbDir = realDbPath ? dirname(realDbPath) : null;
const hasContactsField = request.contacts_db_path &&
  request.contacts_db_path !== "nil";
const realContactsPath = hasContactsField
  ? realPathOr(request.contacts_db_path)
  : null;

// Build the trusted provider interpreter FIRST, before the untrusted REPL
// interpreter even exists. A bad db_path (or any bootstrap failure) then costs
// one interpreter boot, not two, and never becomes a bridge-timeout-shaped
// error from the REPL side.
let bridgeCall:
  | ((method: string, paramsJson: string) => Promise<string>)
  | null = null;
let duplexBridgeCall:
  | ((method: string, paramsJson: string) => ProviderDuplexSession)
  | null = null;
const activeDuplexSessions = new Set<ProviderDuplexSession>();
if (Deno.build.os !== "windows") {
  // A one-shot relay has at most one active provider interpreter call. SIGUSR1
  // is the trusted host's cooperative-cancellation ingress; SIGTERM/SIGKILL
  // retain their normal hard-stop semantics.
  Deno.addSignalListener("SIGUSR1", () => {
    for (const session of activeDuplexSessions) {
      session.requestActiveCancellation();
    }
  });
}
// Opaque to this relay — whatever the adapter's build_db_service() returns,
// ferried across to run_for_host_pyodide()'s meta= kwarg unexamined. Only the
// adapter (on both sides of the bridge) knows or cares what's in it. Kept as
// the raw JSON *string* the service interpreter produced, never
// JSON.parse()'d in Deno: round-tripping through a JS value would silently
// round any integer beyond Number.MAX_SAFE_INTEGER (e.g. a 64-bit record id
// or timestamp an adapter puts in meta) before the adapter ever sees it.
let bridgeMetaJson: string | null = null;
let closeProviderService: (() => Promise<void>) | null = null;
if (hasDbPath) {
  try {
    const quiet = { stdout: () => {}, stderr: () => {} };
    const dbsvc = await loadPyodide(quiet);
    await dbsvc.loadPackage("sqlite3", {
      messageCallback: () => {},
      errorCallback: () => {},
    });
    await mountSources(dbsvc);

    dbsvc.mountNodeFS("/data", dbDir!);
    const svcDbPath = "/data/" + basename(realDbPath!);
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
    // no reason to string-build Python source out of them either. Same for
    // adapter_module_name: validated above, but a global assignment keeps the
    // Python template free of any interpolated string regardless.
    dbsvc.globals.set("svc_db_path", svcDbPath);
    dbsvc.globals.set("svc_contacts_db_path", svcContactsPath);
    dbsvc.globals.set("adapter_module_name", adapterModule);
    const metaJson = await dbsvc.runPythonAsync(`
import importlib, json
_adapter = importlib.import_module(adapter_module_name)
# Pyodide gotcha: a JS \`null\` set via globals.set() crosses over as a JsProxy
# ("JsNull"), NOT Python's None — normalize here so every adapter gets a
# clean "str or None" contract regardless of whether it knows this quirk.
_contacts_db_path = svc_contacts_db_path if isinstance(svc_contacts_db_path, str) else None
try:
    _service, _meta = _adapter.build_db_service(svc_db_path, _contacts_db_path)
    _meta_json = json.dumps(_meta)
except BaseException:
    if "_service" in globals():
        _service.close()
    raise
_meta_json
`);
    bridgeMetaJson = metaJson;
    closeProviderService = async () => {
      await dbsvc.runPythonAsync(`_service.close()`);
    };
    const handle = dbsvc.globals.get("_service").handle;
    // Async wrapper: the REPL interpreter's Python side calls this via
    // run_sync, which requires an awaitable (proven in
    // bridge_source_integration_test.ts). handle() itself is synchronous.
    bridgeCall = async (method: string, paramsJson: string) =>
      handle(method, paramsJson) as string;
    // Explicit bridge-v2 selection. The pump carries at most one frame and
    // never re-enters the suspended REPL interpreter; Python pulls each frame,
    // applies it through the receiving broker, then acknowledges it.
    let duplexBusy = false;
    duplexBridgeCall = (method: string, paramsJson: string) => {
      const execution = JSON.parse(paramsJson)?.execution;
      const callId = typeof execution?.call_id === "string"
        ? execution.call_id
        : null;
      if (!callId) {
        throw new Error("duplex invoke requires execution.call_id");
      }
      const session = startProviderDuplex(async (emit) => {
        if (duplexBusy) {
          throw new Error(
            "provider interpreter already has an active duplex call",
          );
        }
        duplexBusy = true;
        dbsvc.globals.set("_duplex_method", method);
        dbsvc.globals.set("_duplex_params", paramsJson);
        dbsvc.globals.set("_duplex_emit", emit);
        try {
          // runPythonAsync supplies the JSPI suspender used when the remote
          // context waits for each host acknowledgement.
          const raw = await dbsvc.runPythonAsync(
            `_service.handle_duplex(_duplex_method, _duplex_params, _duplex_emit)`,
          );
          const envelope = JSON.parse(raw);
          if (!envelope?.ok) {
            throw new Error(
              `${envelope?.error?.type ?? "BridgeError"}: ${
                envelope?.error?.message ?? "unknown duplex bridge error"
              }`,
            );
          }
        } finally {
          dbsvc.globals.delete("_duplex_method");
          dbsvc.globals.delete("_duplex_params");
          dbsvc.globals.delete("_duplex_emit");
          duplexBusy = false;
        }
      }, callId);
      activeDuplexSessions.add(session);
      return {
        receive: () => session.receive(),
        send: (ackJson: string) => session.send(ackJson),
        cancellation_requested: (callId: string) =>
          session.cancellation_requested(callId),
        requestCancellation: (callId: string) =>
          session.requestCancellation(callId),
        requestActiveCancellation: () => session.requestActiveCancellation(),
        close: async () => {
          try {
            await session.close();
          } finally {
            activeDuplexSessions.delete(session);
          }
        },
      };
    };
  } catch (e) {
    let failure: unknown = e;
    if (closeProviderService !== null) {
      try {
        await closeProviderService();
      } catch (cleanupError) {
        failure = new AggregateError(
          [e, cleanupError],
          "provider setup and cleanup failed",
        );
      }
      closeProviderService = null;
    }
    await writeHostResponse({
      answer: null,
      error: {
        type: "DBServiceError",
        message: String(failure instanceof Error ? failure.message : failure),
      },
    });
    Deno.exit(0); // match the existing contract: exit 0, error in the JSON body
  }
}

// Route Python stdout off the relay's stdout (which carries only the response
// JSON); silence the package loader's "Loading sqlite3" chatter too. Forward the
// RLM's structured events (NDJSON lines on Python's stderr — progress plus the
// loop events from #1: iteration_start/code/output/subcall) to the dedicated
// host event channel; drop other stderr noise. fd2 remains diagnostic-only.
const py = await loadPyodide({
  stdout: () => {},
  // Install the event forwarder only after startup metadata is ready. Package
  // and interpreter bootstrap stderr is diagnostic noise, never Trace input.
  stderr: () => {},
});
// The client-side bridge import currently reaches sql_local.py through
// droste.sources.__init__, so the REPL needs the sqlite3 package to import the
// module. It never opens or mounts a database in this interpreter.
await py.loadPackage("sqlite3", {
  messageCallback: () => {},
  errorCallback: () => {},
});
await mountSources(py);

// Prepare the contract handshake (#33), but do not emit it yet. The first
// canonical RUN event emits startup immediately before itself. Preflight and
// pre-admission refusal have no Trace event surface, so their fd3 stays empty
// without teaching the relay a second copy of runner admission semantics.
// The engine version comes from the staged wheel's dist-info when present
// ("unknown" for raw source mounts); null protocol values remain a loud signal.
const startupEvent: Record<string, unknown> = JSON.parse(
  await py.runPythonAsync(`
import json
try:
    from importlib.metadata import version as _version
    _engine_version = _version("droste")
except Exception:
    _engine_version = "unknown"
try:
    from droste_runner.runner import RUNNER_PROTOCOL_VERSION as _runner_protocol
except Exception:
    _runner_protocol = None
try:
    from droste.providers import PROVIDER_PROTOCOL_VERSION as _provider_protocol
except Exception:
    _provider_protocol = None
json.dumps({
    "type": "startup",
    "engine_version": _engine_version,
    "runner_protocol": _runner_protocol,
    "provider_protocol": _provider_protocol,
})
  `),
);
py.setStderr({ batched: forwardPyodideStderr });

// The untrusted interpreter never receives host filesystem paths. For a
// provider-backed request those paths have already been consumed by the
// trusted interpreter; for a context-only request they have no meaning.
delete request.db_path;
delete request.contacts_db_path;

// Relay-owned events and forwarded Pyodide events share one physical writer.
// A channel failure is terminal: falling back to stderr would make event and
// diagnostic bytes ambiguous to the host.
function writeRelayEvent(obj: Record<string, unknown>): void {
  const event = {
    ...obj,
    run_id: relayRunId,
    parent_run_id: request.run_id,
    depth: 1,
    seq: ++relaySeq,
    timestamp: new Date().toISOString(),
    version: 2,
    persistence_class: "transient",
  };
  eventChannel.writeFrame(JSON.stringify(event));
}

let startupEmitted = false;

function emitStartupIfNeeded(): void {
  if (startupEmitted) return;
  writeRelayEvent(startupEvent);
  startupEmitted = true;
}

function forwardPyodideStderr(message: string): void {
  if (eventChannel.failure !== null) return;
  for (const line of message.split("\n")) {
    if (isRlmEvent(line)) {
      try {
        forwardEngineEvent(line.trim());
      } catch (error) {
        if (error instanceof RelayEventChannelError) return;
        throw error;
      }
    }
  }
}

function forwardEngineEvent(frame: string): void {
  emitStartupIfNeeded();
  eventChannel.writeFrame(frame);
}

function emitEvent(obj: unknown): void {
  if (obj === null || typeof obj !== "object" || Array.isArray(obj)) return;
  emitStartupIfNeeded();
  writeRelayEvent(obj as Record<string, unknown>);
}

// Streaming is on by default; RLM_STREAM=0 forces the legacy unary path (a
// no-rebuild kill switch). --allow-env is already granted (see usage header).
const STREAM_ENABLED = Deno.env.get("RLM_STREAM") !== "0";

// A′ sandbox split: the untrusted interpreter must never hold the ModelRelay
// credential. The broker strips it from the request the sandbox sees, holds it
// host-side, and injects the auth header on every ModelRelay call below.
// RLM_BRIDGE=legacy restores the pre-A′ behavior (credential visible to the
// sandbox) as a no-rebuild kill switch, mirroring RLM_STREAM=0.
const BRIDGE_LEGACY = Deno.env.get("RLM_BRIDGE") === "legacy";
// splitCredentials runs only after host filesystem paths have been removed,
// so sandboxRequest cannot carry a database locator into generated code.
const { creds, sandboxRequest } = splitCredentials(request);

// Host-brokered ModelRelay transport (Pyodide has no network of its own).
// The last HTTP error status (0 = none) is captured structurally here, where
// `r.status` is available, and injected into the HostResponse error below — so
// callers (e.g. the Swift app) can branch on it (402 = out of balance) without
// parsing the human-readable message string. Fresh per process (one run/query).
let lastHttpErrorStatus = 0;
py.globals.set(
  "host_fetch",
  async (m: string, u: string, h: string, b: string) => {
    if (eventChannel.failure !== null) throw eventChannel.failure;
    const headers = JSON.parse(h);
    // A′: host auth is authoritative — drop whatever auth header the sandbox sent
    // and inject the held credential. Scoped to the exact `POST /api/v1/responses`
    // ModelRelay call (isModelRelayResponsesCall parses method + URL), so the
    // credential is a single-purpose LLM-transport key — sandbox code can't reuse
    // it against other ModelRelay endpoints, another host, or over plaintext.
    // Legacy mode leaves the sandbox-assembled header untouched.
    if (!BRIDGE_LEGACY && isModelRelayResponsesCall(m, u)) {
      stripAndInjectAuth(headers, creds);
    } else if (
      !BRIDGE_LEGACY &&
      isRunnerCallback(m, u, [
        request.root_endpoint,
        request.subcall_endpoint,
        request.subcall_batch_endpoint,
        request.data_source_endpoint,
      ])
    ) {
      stripAndInjectBearer(headers, creds.runnerToken);
    }
    // Ask ModelRelay to stream /responses so reasoning tokens reach the host live.
    // Pyodide still receives one complete response (see streamResponses), so the
    // sync RLM loop is unaffected.
    const wantsStream = STREAM_ENABLED && m === "POST" &&
      u.endsWith("/responses");
    if (wantsStream) {
      headers["Accept"] = 'application/x-ndjson; profile="responses-stream/v2"';
    }
    const r = await fetch(u, { method: m, headers, body: b });
    // Surface HTTP errors instead of returning an error/empty body that the
    // Python client would blindly json.loads() into a cryptic JSONDecodeError.
    if (!r.ok) {
      lastHttpErrorStatus = r.status;
      const text = await r.text();
      throw new Error(
        `ModelRelay HTTP ${r.status} ${r.statusText}: ${text.slice(0, 1000)}`,
      );
    }
    // Stream only when we asked for it AND the server actually returned ndjson;
    // otherwise fall back to the unary path — behavior identical to before.
    const contentType = r.headers.get("content-type") || "";
    const isNdjson = contentType.includes("ndjson") ||
      contentType.includes("event-stream");
    if (wantsStream && isNdjson) {
      return await streamResponses(
        r,
        (chunk) => emitEvent({ type: "reasoning_delta", text: chunk }),
      );
    }
    return await r.text();
  },
);
// A′: the sandbox receives the credential-stripped request; legacy keeps the
// full request (credential included) for the pre-A′ in-sandbox auth path.
py.globals.set(
  "request_json",
  JSON.stringify(BRIDGE_LEGACY ? request : sandboxRequest),
);
py.globals.set("adapter_module_name", adapterModule);
// Always set, never conditionally interpolated. They are None/"null" only for
// a context-only request with no provider. relay.ts never inspects what's
// inside meta; it is opaque cargo between the two adapter calls and stays a raw
// JSON string here.
py.globals.set("bridge_call", bridgeCall);
py.globals.set("duplex_bridge_call", duplexBridgeCall);
py.globals.set("bridge_meta_json", bridgeMetaJson ?? "null");
// A callable, not a value snapshot: lastHttpErrorStatus is still 0 at this
// point (host_fetch, which sets it, hasn't run yet — it runs synchronously
// *inside* the runPythonAsync call below via run_sync). The Python template
// calls this after computing resp, to read the CURRENT value.
py.globals.set("get_last_http_error_status", () => lastHttpErrorStatus);

// The status enrichment (attaching the captured HTTP status, e.g. 402 = out
// of balance, to resp["error"]) happens INSIDE this same Python template,
// not via a JS-side JSON.parse(out)/JSON.stringify(parsed) pass afterward —
// that would reintroduce exactly the float64 precision loss the raw
// bridge_meta_json passthrough above exists to avoid, for ANY large integer
// anywhere in the adapter's response (not just meta). `out` is the final
// stdout payload as-is; nothing here re-parses it in JS.
let out: string | null = null;
try {
  out = await py.runPythonAsync(`
import importlib, json, io, contextlib, traceback
from droste_runner.protocol import (
    RUNNER_PROTOCOL_VERSION, build_exception_response, resolve_operation,
)
_adapter = importlib.import_module(adapter_module_name)
_meta = json.loads(bridge_meta_json)
_request = json.loads(request_json)
_operation = None
if _request.get("protocol_version") == RUNNER_PROTOCOL_VERSION:
    try:
        _operation = resolve_operation(_request)
    except ValueError:
        pass
# Pyodide gotcha: a JS \`null\` set via globals.set() crosses over as a JsProxy
# ("JsNull"), NOT Python's None — \`bridge_call is not None\` would not catch
# it, and the adapter would try to call it. Normalize explicitly here rather
# than trust every adapter to know this quirk.
_bridge_call = bridge_call if callable(bridge_call) else None
_duplex_bridge_call = duplex_bridge_call if callable(duplex_bridge_call) else None
# Capture the RLM's progress prints so stdout carries only the response JSON.
_buf = io.StringIO()
try:
    with contextlib.redirect_stdout(_buf):
        resp = _adapter.run_for_host_pyodide(
            _request, host_fetch, bridge_call=_bridge_call,
            duplex_bridge_call=_duplex_bridge_call, meta=_meta,
        )
except Exception as e:
    resp = build_exception_response(e, traceback.format_exc(), operation=_operation)
_status = get_last_http_error_status()
if _status and isinstance(resp.get("error"), dict):
    resp["error"]["status"] = _status
# Runner responses use the same deterministic UTF-8 representation as the
# published protocol conformance fixtures. Python remains the sole serializer
# so arbitrary-precision integer values never cross a JS number boundary.
json.dumps(resp, ensure_ascii=False, separators=(",", ":"))
`);
} catch (error) {
  if (eventChannel.failure === null) {
    throw error;
  }
} finally {
  const cleanupErrors: unknown[] = [];
  const sessions = [...activeDuplexSessions];
  for (const session of sessions) {
    try {
      await session.close();
    } catch (error) {
      cleanupErrors.push(error);
    } finally {
      activeDuplexSessions.delete(session);
    }
  }
  if (closeProviderService !== null) {
    try {
      await closeProviderService();
    } catch (error) {
      cleanupErrors.push(error);
    }
  }
  if (cleanupErrors.length > 0) {
    const detail = cleanupErrors.map((error) =>
      error instanceof Error ? `${error.name}: ${error.message}` : String(error)
    ).join("; ").replace(/\s+/g, " ").trim().slice(0, 1_000);
    console.error(
      `droste relay: provider cleanup failed (${cleanupErrors.length}): ${detail}`,
    );
  }
}

if (eventChannel.failure !== null) {
  await terminateForEventChannel(eventChannel.failure);
}

if (out === null) {
  throw new Error("relay response is unavailable");
}
await Deno.stdout.write(new TextEncoder().encode(out + "\n"));
