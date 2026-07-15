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
import { dirname } from "node:path";

const HERE = new URL(".", import.meta.url).pathname;
const DROSTE_SRC = new URL("../../src", import.meta.url).pathname;
const RELAY_TS =
  new URL("../../src/droste/substrates/_relay/relay.ts", import.meta.url)
    .pathname;
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
): Promise<{ lastLine: string; stderrText: string }> {
  const cmd = new Deno.Command("deno", {
    args: [
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
    ],
    env: { ...Deno.env.toObject(), ...env },
    stdin: "piped",
    stdout: "piped",
    stderr: "piped",
  });
  const child = cmd.spawn();
  const writer = child.stdin.getWriter();
  await writer.write(new TextEncoder().encode(JSON.stringify(request)));
  await writer.close();
  const { code, stdout, stderr } = await child.output();

  const stdoutText = new TextDecoder().decode(stdout);
  const stderrText = new TextDecoder().decode(stderr);
  if (code !== 0) {
    throw new Error(
      `relay.ts exited ${code}\nstdout: ${stdoutText}\nstderr: ${stderrText}`,
    );
  }
  const lines = stdoutText.trim().split("\n").filter((l) =>
    l.trim().startsWith("{")
  );
  assert(lines.length > 0, `no JSON on stdout:\n${stdoutText}`);
  return { lastLine: lines[lines.length - 1], stderrText };
}

async function runRelay(
  sourcesDir: string,
  request: Record<string, unknown>,
  port: number,
  env: Record<string, string>,
): Promise<{ resp: Record<string, unknown>; stderrText: string }> {
  const { lastLine, stderrText } = await runRelayRaw(
    sourcesDir,
    request,
    port,
    env,
  );
  return { resp: JSON.parse(lastLine), stderrText };
}

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
      const { resp, stderrText } = await runRelay(
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
      // versions as a structured stderr event before doing any work.
      const startup = stderrText.split("\n")
        .filter((l) => l.trim().startsWith("{"))
        .map((l) => JSON.parse(l))
        .find((e) => e.type === "startup");
      assert(startup, `no startup handshake event on stderr:\n${stderrText}`);
      assertEquals(startup.runner_protocol, 6);
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
      const { lastLine, stderrText } = await runRelayRaw(
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
        raw.includes('"received_meta_large_id": 9223372036854775807'),
        `expected the exact int64 literal in the raw response, got:\n${raw}`,
      );
    } finally {
      await shutdown();
    }
  },
});

Deno.test({
  name:
    "relay.ts: adapter exceptions retain strict run/preflight protocol-v6 envelopes",
  fn: async () => {
    const { port, shutdown } = await startMockModelRelay();
    try {
      const sourcesDir = await buildTempSources();
      for (const operation of ["run", "preflight"] as const) {
        const { lastLine } = await runRelayRaw(
          sourcesDir,
          { protocol_version: 6, operation },
          port,
          {},
          "_failing_adapter",
        );
        const response = JSON.parse(lastLine);
        assertEquals(response.protocol_version, 6);
        assertEquals(response.operation, operation);
        assertEquals(response.status, "error");
        assertEquals(response.error.type, "RuntimeError");
        assertEquals(response.error.message, "adapter boom");
        assert(typeof response.error.traceback === "string");
        if (operation === "preflight") {
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
