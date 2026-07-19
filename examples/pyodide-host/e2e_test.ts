// Droste-owned end-to-end proof that pyodide/relay.ts's host-adapter seam
// (adapter-agnostic relay split) actually works: spawns the REAL relay.ts as a
// subprocess — real Deno process boundary, real Pyodide interpreter, real
// dynamic `importlib.import_module` of an adapter module, real stdin/stdout
// HostRequest/HostResponse contract — with zero host-app dependency and
// zero real network (a local mock server stands in for ModelRelay). Runs
// unconditionally in droste's own CI: no sibling checkout, no skip.
//
// The two tests that used to prove this (db_service_integration_test.ts,
// broker_batch_integration_test.ts) required a sibling host-repo checkout and
// tested that host's own adapter in-process; they moved to that host's repo
// alongside its adapter module. This test proves the DROSTE side of the same
// contract — the part that must work for ANY adapter, not just one host's.
//
// Run: deno test --allow-run --allow-read --allow-write --allow-net=127.0.0.1 --allow-env examples/pyodide-host/e2e_test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { copy } from "jsr:@std/fs@1/copy";
import { spawn } from "node:child_process";
import { closeSync, openSync } from "node:fs";
import { dirname } from "node:path";
import type { Readable } from "node:stream";
import { isRlmEvent } from "../../src/droste/substrates/_relay/events.ts";

const HERE = new URL(".", import.meta.url).pathname;
const DROSTE_SRC = new URL("../../src", import.meta.url).pathname;
const RELAY_TS =
  new URL("../../src/droste/substrates/_relay/relay.ts", import.meta.url)
    .pathname;
const EVENT_CHANNEL_PROBE =
  new URL("../../pyodide/event_channel_probe.ts", import.meta.url).pathname;
const RUNNER_REFUSAL_FIXTURE = new URL(
  "../../src/droste/testing/fixtures/runner-v8-refusal.ndjson",
  import.meta.url,
);
const TRACE_LIFECYCLE_FIXTURE = new URL(
  "../../src/droste/testing/fixtures/trace-v4-lifecycle.ndjson",
  import.meta.url,
);
const TEST_BUDGET = {
  tokens: 500_000,
  subcalls: 50,
  depth: 1,
  wall_ms: 300_000,
  root_output_tokens: 4_096,
  subcall_output_tokens: 2_048,
};

const QUERY_CODE = [
  "```python",
  'rows = query("SELECT COUNT(*) AS n FROM widgets")',
  'answer["content"] = str(rows[0]["n"])',
  'answer["ready"] = True',
  "```",
].join("\n");

async function collectNodeStream(
  stream: AsyncIterable<Uint8Array>,
): Promise<string> {
  const decoder = new TextDecoder();
  let text = "";
  for await (const chunk of stream) {
    text += decoder.decode(chunk, { stream: true });
  }
  return text + decoder.decode();
}

function captureNodeStream(stream: Readable): {
  firstLine: Promise<string>;
  completed: Promise<string>;
} {
  const decoder = new TextDecoder();
  let text = "";
  let firstLineSettled = false;
  let resolveFirstLine!: (line: string) => void;
  let rejectFirstLine!: (error: Error) => void;
  const firstLine = new Promise<string>((resolve, reject) => {
    resolveFirstLine = resolve;
    rejectFirstLine = reject;
  });
  const completed = new Promise<string>((resolve, reject) => {
    stream.on("data", (chunk: Uint8Array) => {
      text += decoder.decode(chunk, { stream: true });
      const newline = text.indexOf("\n");
      if (!firstLineSettled && newline >= 0) {
        firstLineSettled = true;
        resolveFirstLine(text.slice(0, newline));
      }
    });
    stream.once("error", (error) => {
      if (!firstLineSettled) {
        firstLineSettled = true;
        rejectFirstLine(error);
      }
      reject(error);
    });
    stream.once("end", () => {
      text += decoder.decode();
      if (!firstLineSettled) {
        firstLineSettled = true;
        rejectFirstLine(new Error("stream ended before its first line"));
      }
      resolve(text);
    });
  });
  return { firstLine, completed };
}

async function waitBounded<T>(
  promise: Promise<T>,
  description: string,
): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(
          () => reject(new Error(`timed out waiting for ${description}`)),
          10_000,
        );
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

// A minimal ModelRelay stand-in: any POST to /responses gets one scripted
// reply. The default proves the answer came from a real query() round-trip
// through BridgeProvider -> ProviderService -> the real SQLite file.
function startMockModelRelay(code = QUERY_CODE): Promise<
  {
    port: number;
    rootPayloads: Record<string, unknown>[];
    shutdown: () => Promise<void>;
  }
> {
  const rootPayloads: Record<string, unknown>[] = [];
  const server = Deno.serve(
    // hostname must be explicit: Deno.serve defaults to 0.0.0.0, which the
    // documented least-privilege `--allow-net=127.0.0.1` permission rejects.
    { port: 0, hostname: "127.0.0.1", onListen: () => {} },
    async (req) => {
      if (
        req.method === "POST" &&
        new URL(req.url).pathname.endsWith("/responses")
      ) {
        rootPayloads.push(await req.json());
        return Response.json({
          output: [{
            type: "message",
            role: "assistant",
            content: [{ type: "text", text: code }],
          }],
        });
      }
      return new Response("not found", { status: 404 });
    },
  );
  return Promise.resolve({
    port: (server.addr as Deno.NetAddr).port,
    rootPayloads,
    shutdown: () => server.shutdown(),
  });
}

// A mock that always fails /responses with an HTTP 402 (out of balance) —
// proves relay.ts's status-enrichment (attaching the captured HTTP status to
// resp.error.status) happens without any JS-side re-parsing of the adapter's
// response, per the same adapter-agnostic-split finding the precision test above
// guards on the success path.
function startFailingMockModelRelay(): Promise<
  { port: number; shutdown: () => Promise<void> }
> {
  const server = Deno.serve(
    { port: 0, hostname: "127.0.0.1", onListen: () => {} },
    (req) => {
      if (
        req.method === "POST" &&
        new URL(req.url).pathname.endsWith("/responses")
      ) {
        return Response.json({ error: "insufficient balance" }, {
          status: 402,
        });
      }
      return new Response("not found", { status: 404 });
    },
  );
  return Promise.resolve({
    port: (server.addr as Deno.NetAddr).port,
    shutdown: () => server.shutdown(),
  });
}

function startGatedMockModelRelay(): Promise<{
  port: number;
  requestStarted: Promise<void>;
  release: () => void;
  shutdown: () => Promise<void>;
}> {
  let markRequestStarted!: () => void;
  const requestStarted = new Promise<void>((resolve) => {
    markRequestStarted = resolve;
  });
  let releaseResponse!: () => void;
  const responseGate = new Promise<void>((resolve) => {
    releaseResponse = resolve;
  });
  const server = Deno.serve(
    { port: 0, hostname: "127.0.0.1", onListen: () => {} },
    async (req) => {
      if (
        req.method === "POST" &&
        new URL(req.url).pathname.endsWith("/responses")
      ) {
        await req.json();
        markRequestStarted();
        await responseGate;
        return Response.json({
          output: [{
            type: "message",
            role: "assistant",
            content: [{ type: "text", text: QUERY_CODE }],
          }],
        });
      }
      return new Response("not found", { status: 404 });
    },
  );
  return Promise.resolve({
    port: (server.addr as Deno.NetAddr).port,
    requestStarted,
    release: releaseResponse,
    shutdown: () => server.shutdown(),
  });
}

async function buildTempSources(): Promise<string> {
  // Copy, not symlink: Pyodide's mountNodeFS does not reliably follow
  // symlinked entries inside the mounted directory (a symlinked .py file
  // mounts as present but unreadable — importlib reports ModuleNotFoundError).
  const dir = await Deno.makeTempDir({ prefix: "droste-pyodide-e2e-" });
  await copy(`${DROSTE_SRC}/droste`, `${dir}/droste`, { overwrite: true });
  await copy(`${DROSTE_SRC}/droste_runner`, `${dir}/droste_runner`, {
    overwrite: true,
  });
  await copy(
    `${HERE}pyodide_host_adapter.py`,
    `${dir}/pyodide_host_adapter.py`,
    { overwrite: true },
  );
  await copy(
    `${HERE}_meta_precision_adapter.py`,
    `${dir}/_meta_precision_adapter.py`,
    { overwrite: true },
  );
  await copy(
    `${HERE}_failing_adapter.py`,
    `${dir}/_failing_adapter.py`,
    { overwrite: true },
  );
  await copy(
    `${HERE}_close_failing_adapter.py`,
    `${dir}/_close_failing_adapter.py`,
    { overwrite: true },
  );
  await copy(
    `${HERE}_runner_protocol_adapter.py`,
    `${dir}/_runner_protocol_adapter.py`,
    { overwrite: true },
  );
  await copy(
    `${HERE}_large_event_adapter.py`,
    `${dir}/_large_event_adapter.py`,
    { overwrite: true },
  );
  return dir;
}

async function buildTempDb(): Promise<string> {
  const dir = await Deno.makeTempDir({ prefix: "droste-pyodide-e2e-db-" });
  const dbPath = `${dir}/widgets.db`;
  // Build the fixture DB with the sqlite3 CLI so this test needs no Python.
  const p = new Deno.Command("sqlite3", {
    args: [
      dbPath,
      "CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT NOT NULL);" +
      "INSERT INTO widgets (name) VALUES ('gizmo'), ('gadget'), ('widget');",
    ],
  });
  const { code, stderr } = await p.output();
  if (code !== 0) {
    throw new Error(
      `sqlite3 fixture build failed: ${new TextDecoder().decode(stderr)}`,
    );
  }
  return dbPath;
}

async function runRelayRaw(
  sourcesDir: string,
  request: Record<string, unknown>,
  port: number,
  env: Record<string, string>,
  adapterModule = "pyodide_host_adapter",
): Promise<{ lastLine: string; stderrText: string; eventText: string }> {
  const child = spawnRelayRaw(sourcesDir, request, port, env, adapterModule);
  const completion = probeCompletion(child);
  const stdoutPromise = collectNodeStream(child.stdout!);
  const stderrPromise = collectNodeStream(child.stderr!);
  const eventPromise = collectNodeStream(
    child.stdio[3] as AsyncIterable<Uint8Array>,
  );
  const [{ code, signal }, stdoutText, stderrText, eventText] = await Promise
    .all([
      completion,
      stdoutPromise,
      stderrPromise,
      eventPromise,
    ]);
  if (code !== 0) {
    throw new Error(
      `relay.ts exited ${
        code ?? signal
      }\nstdout: ${stdoutText}\nstderr: ${stderrText}`,
    );
  }
  const lines = stdoutText.trimEnd().split("\n");
  assertEquals(
    lines.length,
    1,
    `stdout must contain one HostResponse:\n${stdoutText}`,
  );
  JSON.parse(lines[0]);
  return { lastLine: lines[0], stderrText, eventText };
}

function spawnRelayRaw(
  sourcesDir: string,
  request: Record<string, unknown>,
  port: number,
  env: Record<string, string>,
  adapterModule = "pyodide_host_adapter",
  eventStdio: "pipe" | number = "pipe",
) {
  return spawnRelayProcess(
    "/bin/sh",
    [
      "-c",
      'DENO_EXTRA_STDIO_FDS="$DROSTE_RELAY_EVENT_FD" "$@"',
      "droste-relay",
      Deno.execPath(),
      ...relayDenoArgs(sourcesDir, port, adapterModule),
    ],
    request,
    env,
    eventStdio,
  );
}

function relayDenoArgs(
  sourcesDir: string,
  port: number,
  adapterModule: string,
): string[] {
  return [
    "run",
    `--allow-net=127.0.0.1:${port}`,
    // Unscoped read: this dev/CI run uses the ambient Deno cache (Pyodide's
    // wasm assets live there), unlike a production host's offline bundle
    // (own isolated DENO_DIR, narrowly --allow-read-scoped — the shape a
    // host's runner spawn args / the CLI relay invocation use).
    "--allow-read",
    "--allow-env",
    RELAY_TS,
    sourcesDir,
    adapterModule,
  ];
}

function spawnRelayWithoutDenoExtraStdio(
  sourcesDir: string,
  request: Record<string, unknown>,
  port: number,
  env: Record<string, string>,
) {
  // A native shell, not Deno's child_process shim, is the relay's immediate
  // parent. Explicitly remove the marker that Deno's outer test shim adds so
  // this proves an ordinary external launch fails closed when it is omitted.
  return spawnRelayProcess(
    "/bin/sh",
    [
      "-c",
      'unset DENO_EXTRA_STDIO_FDS; "$@"',
      "droste-relay",
      Deno.execPath(),
      ...relayDenoArgs(sourcesDir, port, "pyodide_host_adapter"),
    ],
    request,
    env,
    "pipe",
  );
}

function spawnRelayDirectForSignalTest(
  sourcesDir: string,
  request: Record<string, unknown>,
  port: number,
  env: Record<string, string>,
) {
  return spawnRelayProcess(
    Deno.execPath(),
    relayDenoArgs(sourcesDir, port, "pyodide_host_adapter"),
    request,
    { ...env, DENO_EXTRA_STDIO_FDS: "3" },
    "pipe",
  );
}

function spawnRelayProcess(
  command: string,
  args: string[],
  request: Record<string, unknown>,
  env: Record<string, string>,
  eventStdio: "pipe" | number,
) {
  const child = spawn(command, args, {
    env: {
      ...Deno.env.toObject(),
      ...env,
      DROSTE_RELAY_EVENT_FD: "3",
    },
    stdio: ["pipe", "pipe", "pipe", eventStdio],
  });
  child.stdin!.end(JSON.stringify(request));
  return child;
}

async function runRelay(
  sourcesDir: string,
  request: Record<string, unknown>,
  port: number,
  env: Record<string, string>,
): Promise<{
  resp: Record<string, unknown>;
  stderrText: string;
  eventText: string;
}> {
  const { lastLine, stderrText, eventText } = await runRelayRaw(
    sourcesDir,
    request,
    port,
    env,
  );
  return { resp: JSON.parse(lastLine), stderrText, eventText };
}

type ProbeResult = {
  code: number | null;
  signal: string | null;
  stdout: string;
  stderr: string;
  events: string;
};

function spawnEventChannelProbe(mode: "large" | "fail" | "cancel") {
  return spawn(
    Deno.execPath(),
    ["run", "--allow-env", "--allow-read", EVENT_CHANNEL_PROBE, mode],
    {
      env: { ...Deno.env.toObject(), DROSTE_RELAY_EVENT_FD: "3" },
      stdio: ["ignore", "pipe", "pipe", "pipe"],
    },
  );
}

function probeCompletion(child: ReturnType<typeof spawn>): Promise<{
  code: number | null;
  signal: string | null;
}> {
  return new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("close", (code, signal) => resolve({ code, signal }));
  });
}

async function runEventChannelProbe(
  mode: "large" | "fail",
): Promise<ProbeResult> {
  const child = spawnEventChannelProbe(mode);
  const [stdout, stderr, events, status] = await Promise.all([
    collectNodeStream(child.stdout!),
    collectNodeStream(child.stderr!),
    collectNodeStream(child.stdio[3] as AsyncIterable<Uint8Array>),
    probeCompletion(child),
  ]);
  return { ...status, stdout, stderr, events };
}

Deno.test("large response, diagnostic, and event lanes drain independently", async () => {
  const result = await runEventChannelProbe("large");
  assertEquals(result.code, 0);
  assertEquals(result.signal, null);
  assertEquals(result.stdout, '{"answer":"ok","error":null}\n');
  assert(!result.stderr.includes('"type":'));
  assert(!result.events.includes("diagnostic-"));
  const fixture = await Deno.readTextFile(TRACE_LIFECYCLE_FIXTURE);
  assertEquals(result.events, fixture.repeat(128));
  const frames = result.events.trimEnd().split("\n");
  assert(frames.every(isRlmEvent));
  assertEquals(JSON.parse(frames[0]).type, "startup");
  assert(result.stderr.length > 4_000_000);
  assert(result.events.length > 4_000_000);
});

Deno.test("process failure leaves a valid nonterminal event prefix", async () => {
  const result = await runEventChannelProbe("fail");
  assertEquals(result.code, 17);
  assertEquals(result.signal, null);
  assertEquals(result.stdout, "");
  assertEquals(result.stderr, "event-channel-probe-process-failure\n");
  const frames = result.events.trimEnd().split("\n");
  assertEquals(frames.length, 1);
  assert(isRlmEvent(frames[0]));
  assertEquals(JSON.parse(frames[0]).type, "startup");
});

Deno.test("cancellation leaves a valid nonterminal event prefix", async () => {
  const child = spawnEventChannelProbe("cancel");
  const stdout = collectNodeStream(child.stdout!);
  const events = collectNodeStream(
    child.stdio[3] as AsyncIterable<Uint8Array>,
  );
  const diagnostics = captureNodeStream(child.stderr!);
  const completion = probeCompletion(child);
  assertEquals(
    await waitBounded(diagnostics.firstLine, "probe readiness"),
    "event-channel-probe-ready",
  );
  assert(child.kill("SIGTERM"));
  const status = await completion;
  assertEquals(status.code, null);
  assertEquals(status.signal, "SIGTERM");
  assertEquals(await stdout, "");
  assertEquals(await diagnostics.completed, "event-channel-probe-ready\n");
  const frames = (await events).trimEnd().split("\n");
  assertEquals(frames.length, 1);
  assert(isRlmEvent(frames[0]));
  assertEquals(JSON.parse(frames[0]).type, "startup");
});

Deno.test("relay hard cancellation and process death leave only a nonterminal prefix", async () => {
  const sourcesDir = await buildTempSources();
  const dbPath = await buildTempDb();
  for (const signal of ["SIGTERM", "SIGKILL"] as const) {
    const server = await startGatedMockModelRelay();
    const child = spawnRelayDirectForSignalTest(
      sourcesDir,
      {
        question: "wait for hard stop",
        db_path: dbPath,
        root_model: "test-model",
        base_url: `http://127.0.0.1:${server.port}/api/v1`,
        api_key: "test-key",
        budget: TEST_BUDGET,
      },
      server.port,
      {},
    );
    const completion = probeCompletion(child);
    const stdout = collectNodeStream(child.stdout!);
    const diagnostics = collectNodeStream(child.stderr!);
    const eventCapture = captureNodeStream(child.stdio[3] as Readable);
    try {
      assertEquals(
        JSON.parse(
          await waitBounded(eventCapture.firstLine, `${signal} startup event`),
        ).type,
        "startup",
      );
      await waitBounded(server.requestStarted, `${signal} provider request`);
      assert(child.kill(signal));
      const status = await waitBounded(completion, `${signal} process exit`);
      assertEquals(status.code, null);
      assertEquals(status.signal, signal);
      assertEquals(await stdout, "");
      const diagnosticText = await diagnostics;
      assert(
        diagnosticText.trimEnd().split("\n").filter(Boolean).every((line) =>
          !isRlmEvent(line)
        ),
      );
      const eventText = await eventCapture.completed;
      const frames = eventText.trimEnd().split("\n");
      assert(frames.every(isRlmEvent));
      const events = frames.map((frame) => JSON.parse(frame));
      assertEquals(events[0].type, "startup");
      assert(events.every((event) => event.type !== "done"));
    } finally {
      if (child.exitCode === null && child.signalCode === null) {
        child.kill("SIGKILL");
        await completion.catch(() => {});
      }
      server.release();
      await server.shutdown();
    }
  }
});

async function runRelayEventChannelFailure(
  eventDescriptor: string | undefined,
): Promise<{ code: number; stdout: string; stderr: string }> {
  const env = Deno.env.toObject();
  if (eventDescriptor === undefined) {
    delete env.DROSTE_RELAY_EVENT_FD;
  } else {
    env.DROSTE_RELAY_EVENT_FD = eventDescriptor;
  }
  const cmd = new Deno.Command("sh", {
    args: [
      "-c",
      'exec 3>&-; exec "$@"',
      "droste-relay",
      "deno",
      "run",
      "--allow-read",
      "--allow-env",
      RELAY_TS,
      "/unused-before-event-channel-validation",
      "pyodide_host_adapter",
    ],
    env,
    stdin: "piped",
    stdout: "piped",
    stderr: "piped",
  });
  const child = cmd.spawn();
  const writer = child.stdin.getWriter();
  await writer.write(new TextEncoder().encode("{}"));
  await writer.close();
  const result = await child.output();
  return {
    code: result.code,
    stdout: new TextDecoder().decode(result.stdout),
    stderr: new TextDecoder().decode(result.stderr),
  };
}

Deno.test({
  name:
    "relay.ts fails closed when its event descriptor is missing or unavailable",
  fn: async () => {
    for (
      const [descriptor, expectedCode] of [
        [undefined, "missing_descriptor"],
        ["3", "descriptor_unavailable"],
      ] as const
    ) {
      const result = await runRelayEventChannelFailure(descriptor);
      assertEquals(result.code, 0);
      const lines = result.stdout.trimEnd().split("\n");
      assertEquals(lines.length, 1);
      assertEquals(JSON.parse(lines[0]), {
        answer: null,
        error: {
          type: "RelayEventChannelError",
          code: expectedCode,
          message: "dedicated relay event channel is unavailable",
        },
      });
      assertEquals(
        result.stderr,
        `droste relay: event_channel_error code=${expectedCode}\n`,
      );
    }
  },
});

Deno.test({
  name: "external shell parent must register inherited fd3 with Deno",
  fn: async () => {
    const child = spawnRelayWithoutDenoExtraStdio(
      "/unused-before-event-channel-validation",
      {},
      1,
      {},
    );
    const [status, stdout, stderr, events] = await Promise.all([
      probeCompletion(child),
      collectNodeStream(child.stdout!),
      collectNodeStream(child.stderr!),
      collectNodeStream(child.stdio[3] as AsyncIterable<Uint8Array>),
    ]);
    assertEquals(status, { code: 0, signal: null });
    assertEquals(JSON.parse(stdout), {
      answer: null,
      error: {
        type: "RelayEventChannelError",
        code: "descriptor_unavailable",
        message: "dedicated relay event channel is unavailable",
      },
    });
    assertEquals(
      stderr,
      "droste relay: event_channel_error code=descriptor_unavailable\n",
    );
    assertEquals(events, "");
  },
});

Deno.test({
  name: "relay.ts returns one structured failure when fd3 is read-only",
  fn: async () => {
    const sourcesDir = await buildTempSources();
    const descriptor = openSync(new URL(import.meta.url), "r");
    try {
      const child = spawnRelayRaw(
        sourcesDir,
        {},
        1,
        {},
        "_large_event_adapter",
        descriptor,
      );
      const [status, stdout, stderr] = await Promise.all([
        probeCompletion(child),
        collectNodeStream(child.stdout!),
        collectNodeStream(child.stderr!),
      ]);
      assertEquals(status, { code: 0, signal: null });
      assertEquals(stdout.trimEnd().split("\n").length, 1);
      assertEquals(JSON.parse(stdout), {
        answer: null,
        error: {
          type: "RelayEventChannelError",
          code: "write_failed",
          message: "dedicated relay event channel is unavailable",
        },
      });
      assertEquals(
        stderr,
        "droste relay: event_channel_error code=write_failed\n",
      );
    } finally {
      closeSync(descriptor);
    }
  },
});

Deno.test({
  name:
    "external shell parent preserves a large canonical fd3 event without lane pollution",
  fn: async () => {
    const { port, shutdown } = await startMockModelRelay();
    try {
      const sourcesDir = await buildTempSources();
      const { lastLine, stderrText, eventText } = await runRelayRaw(
        sourcesDir,
        {},
        port,
        {},
        "_large_event_adapter",
      );
      assertEquals(JSON.parse(lastLine), { answer: "ok", error: null });
      assertEquals(stderrText, "");
      const frames = eventText.trimEnd().split("\n");
      assertEquals(frames.length, 2);
      assert(frames.every(isRlmEvent));
      assertEquals(JSON.parse(frames[0]).type, "startup");
      const largeEvent = JSON.parse(frames[1]);
      assertEquals(largeEvent.type, "code");
      assertEquals(largeEvent.code.length, 1_048_576);
    } finally {
      await shutdown();
    }
  },
});

Deno.test({
  name:
    "relay.ts + pyodide_host_adapter.py: brokered provider, real subprocess + Pyodide + mocked ModelRelay",
  fn: async () => {
    const { port, rootPayloads, shutdown } = await startMockModelRelay();
    try {
      const sourcesDir = await buildTempSources();
      const dbPath = await buildTempDb();
      const request = {
        question: "how many widgets are there",
        db_path: dbPath,
        root_model: "test-model",
        root_reasoning_effort: "none",
        base_url: `http://127.0.0.1:${port}/api/v1`,
        api_key: "test-key",
        budget: TEST_BUDGET,
      };
      // Exercises build_db_service + the opaque meta blob + bridge_call, the
      // parts the adapter-agnostic split changed.
      const { resp, stderrText, eventText } = await runRelay(
        sourcesDir,
        request,
        port,
        {},
      );
      assertEquals(resp.error, null);
      assertEquals(resp.answer, "3"); // real COUNT(*) via the real bridged query()
      assertEquals(rootPayloads.length, 1);
      assertEquals(rootPayloads[0].reasoning_effort, "none");
      // Contract handshake (#33): the relay announces engine + protocol
      // versions on the dedicated event descriptor before doing any work.
      const eventFrames = eventText.trimEnd().split("\n");
      assert(
        eventFrames.length > 1 && eventFrames.every(isRlmEvent),
        `fd3 must contain only canonical events:\n${eventText}`,
      );
      const startup = JSON.parse(eventFrames[0]);
      assertEquals(startup.type, "startup");
      assert(
        !stderrText.split("\n").some((line) => line.trim().startsWith("{")),
        `fd2 must remain diagnostic-only:\n${stderrText}`,
      );
      assertEquals(startup.runner_protocol, 8);
      assertEquals(startup.provider_protocol, 4);
      assert(
        typeof startup.engine_version === "string" &&
          startup.engine_version.length > 0,
      );
    } finally {
      await shutdown();
    }
  },
});

Deno.test({
  name: "relay.ts preserves a successful response when provider cleanup fails",
  fn: async () => {
    const { port, shutdown } = await startMockModelRelay();
    try {
      const sourcesDir = await buildTempSources();
      const dbPath = await buildTempDb();
      const { lastLine, stderrText, eventText } = await runRelayRaw(
        sourcesDir,
        {
          question: "how many widgets are there",
          db_path: dbPath,
          root_model: "test-model",
          base_url: `http://127.0.0.1:${port}/api/v1`,
          api_key: "test-key",
          budget: TEST_BUDGET,
        },
        port,
        {},
        "_close_failing_adapter",
      );

      const response = JSON.parse(lastLine);
      assertEquals(response.error, null);
      assertEquals(response.answer, "3");
      const diagnostic = stderrText.split("\n").find((line) =>
        line.startsWith("droste relay: provider cleanup failed (1): ")
      );
      assert(
        diagnostic,
        `expected cleanup diagnostic on stderr:\n${stderrText}`,
      );
      assert(
        diagnostic.includes("intentional provider close failure"),
        `expected cleanup failure detail on stderr:\n${stderrText}`,
      );
      assert(
        diagnostic.length <= 1_100,
        "cleanup diagnostic must stay bounded",
      );
      assert(
        stderrText.trimEnd().split("\n").every((line) => !isRlmEvent(line)),
        "fd2 diagnostics must never validate as Trace events",
      );
      assert(
        eventText.trimEnd().split("\n").every(isRlmEvent),
        "fd3 must contain only canonical Trace events",
      );
    } finally {
      await shutdown();
    }
  },
});

Deno.test({
  name: "relay.ts exposes no ambient database directory to generated code",
  fn: async () => {
    const code = [
      "```python",
      "import os",
      'ambient = "blocked"',
      'for root, _, files in os.walk("/"):',
      '    if "host-secret-45.txt" in files:',
      '        with open(os.path.join(root, "host-secret-45.txt")) as f:',
      "            ambient = f.read()",
      "        break",
      'rows = query("SELECT COUNT(*) AS n FROM widgets")',
      'answer["content"] = ambient + ":" + str(rows[0]["n"])',
      'answer["ready"] = True',
      "```",
    ].join("\n");
    const { port, rootPayloads, shutdown } = await startMockModelRelay(code);
    try {
      const sourcesDir = await buildTempSources();
      const dbPath = await buildTempDb();
      const secretPath = `${dirname(dbPath)}/host-secret-45.txt`;
      await Deno.writeTextFile(secretPath, "exposed");
      assertEquals(await Deno.readTextFile(secretPath), "exposed");
      const request = {
        question: "how many widgets are there",
        db_path: dbPath,
        root_model: "test-model",
        base_url: `http://127.0.0.1:${port}/api/v1`,
        api_key: "test-key",
        budget: TEST_BUDGET,
      };
      const { resp } = await runRelay(sourcesDir, request, port, {});
      assertEquals(resp.error, null);
      // The VFS-wide search would find "exposed" under any mount point if the
      // database's parent reached the untrusted interpreter. The SQL result
      // still proves brokered provider access works in the same run.
      assertEquals(resp.answer, "blocked:3");
      assertEquals(rootPayloads.length, 1);
      assert(!("reasoning_effort" in rootPayloads[0]));
    } finally {
      await shutdown();
    }
  },
});

Deno.test({
  name:
    "relay.ts: meta blob survives the DB-service round trip at full int64 precision",
  fn: async () => {
    const { port, shutdown } = await startMockModelRelay();
    try {
      const sourcesDir = await buildTempSources();
      const dbPath = await buildTempDb();
      const request = {
        question: "how many widgets are there",
        db_path: dbPath,
        root_model: "test-model",
        base_url: `http://127.0.0.1:${port}/api/v1`,
        api_key: "test-key",
        budget: TEST_BUDGET,
      };
      const { lastLine: raw } = await runRelayRaw(
        sourcesDir,
        request,
        port,
        {},
        "_meta_precision_adapter",
      );
      // Deliberately NOT JSON.parse()'d: parsing in this test would itself
      // round the value back to a float64, masking the exact regression
      // this test exists to catch. A raw substring match on the literal
      // digits is the only way to prove relay.ts never routed `meta`
      // through a JS number anywhere along the path.
      assert(
        raw.includes('"received_meta_large_id":9223372036854775807'),
        `expected the exact int64 literal in the raw response, got:\n${raw}`,
      );
    } finally {
      await shutdown();
    }
  },
});

Deno.test({
  name:
    "relay.ts: adapter exceptions retain strict run/preflight protocol-v8 envelopes",
  fn: async () => {
    const { port, shutdown } = await startMockModelRelay();
    try {
      const sourcesDir = await buildTempSources();
      for (const operation of ["run", "preflight"] as const) {
        const { lastLine, eventText } = await runRelayRaw(
          sourcesDir,
          { protocol_version: 8, operation },
          port,
          {},
          "_failing_adapter",
        );
        const response = JSON.parse(lastLine);
        assertEquals(response.protocol_version, 8);
        assertEquals(response.operation, operation);
        assertEquals(response.status, "error");
        assertEquals(response.error.type, "RuntimeError");
        assertEquals(response.error.message, "adapter boom");
        assert(typeof response.error.traceback === "string");
        if (operation === "preflight") {
          assertEquals(eventText, "");
          assertEquals(Object.keys(response).sort(), [
            "error",
            "operation",
            "preflight",
            "protocol_version",
            "status",
          ]);
          assertEquals(response.preflight, null);
        } else {
          assert(!("preflight" in response));
          assert("answer" in response);
        }
      }
    } finally {
      await shutdown();
    }
  },
});

Deno.test({
  name: "relay.ts: exact canonical refusal and preflight leave fd3 empty",
  fn: async () => {
    const { port, shutdown } = await startMockModelRelay();
    try {
      const sourcesDir = await buildTempSources();
      const { lastLine, stderrText, eventText } = await runRelayRaw(
        sourcesDir,
        {},
        port,
        {},
        "_runner_protocol_adapter",
      );
      const response = JSON.parse(lastLine);
      assertEquals(response.status, "refusal");
      assertEquals(response.operation, null);
      assertEquals(response.error.code, "protocol_version_missing");
      assertEquals(
        `${lastLine}\n`,
        await Deno.readTextFile(RUNNER_REFUSAL_FIXTURE),
      );
      assertEquals(eventText, "");
      assertEquals(stderrText, "");

      const preflight = await runRelayRaw(
        sourcesDir,
        {
          protocol_version: 8,
          operation: "preflight",
          model: "test-model",
          budget: TEST_BUDGET,
        },
        port,
        {},
        "_runner_protocol_adapter",
      );
      const preflightResponse = JSON.parse(preflight.lastLine);
      assertEquals(preflightResponse.status, "success");
      assertEquals(preflightResponse.operation, "preflight");
      assertEquals(preflight.eventText, "");
      assertEquals(preflight.stderrText, "");
    } finally {
      await shutdown();
    }
  },
});

Deno.test({
  name:
    "relay.ts: HTTP error status attaches to error.status without a JS-side JSON round trip",
  fn: async () => {
    const { port, shutdown } = await startFailingMockModelRelay();
    try {
      const sourcesDir = await buildTempSources();
      const dbPath = await buildTempDb();
      const request = {
        question: "how many widgets are there",
        db_path: dbPath,
        root_model: "test-model",
        base_url: `http://127.0.0.1:${port}/api/v1`,
        api_key: "test-key",
        budget: TEST_BUDGET,
      };
      const { resp } = await runRelay(sourcesDir, request, port, {});
      assertEquals(
        resp.error && (resp.error as Record<string, unknown>).status,
        402,
      );
    } finally {
      await shutdown();
    }
  },
});
