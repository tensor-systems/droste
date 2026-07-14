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

// A minimal ModelRelay stand-in: any POST to /responses gets one scripted
// reply telling the RLM to query the real (bridged) SQL source and finish.
// Proves the answer actually came from a real query() round-trip through
// BridgeProvider -> ProviderService -> the real SQLite file, not a stub.
function startMockModelRelay(): Promise<
  { port: number; shutdown: () => Promise<void> }
> {
  const code = [
    "```python",
    'rows = query("SELECT COUNT(*) AS n FROM widgets")',
    'answer["content"] = str(rows[0]["n"])',
    'answer["ready"] = True',
    "```",
  ].join("\n");
  const server = Deno.serve(
    // hostname must be explicit: Deno.serve defaults to 0.0.0.0, which the
    // documented least-privilege `--allow-net=127.0.0.1` permission rejects.
    { port: 0, hostname: "127.0.0.1", onListen: () => {} },
    (req) => {
      if (
        req.method === "POST" &&
        new URL(req.url).pathname.endsWith("/responses")
      ) {
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
    "relay.ts + pyodide_host_adapter.py: DB-service (bridge) mode, real subprocess + Pyodide + mocked ModelRelay",
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
      // Pinned "on", not inherited: an ambient RLM_DB_SERVICE=0 in the
      // developer's or CI's environment would otherwise silently downgrade
      // this to the single-interpreter path (runRelay merges over
      // Deno.env.toObject()) — both modes return the same answer here, so
      // the test would keep passing while testing the wrong code path.
      // Exercises build_db_service + the opaque meta blob + bridge_call, the
      // parts the adapter-agnostic split changed.
      const { resp, stderrText } = await runRelay(sourcesDir, request, port, {
        RLM_DB_SERVICE: "1",
      });
      assertEquals(resp.error, null);
      assertEquals(resp.answer, "3"); // real COUNT(*) via the real bridged query()
      // The legacy-mode WARNING must NOT fire on the default path.
      assert(
        !stderrText.includes("RLM_DB_SERVICE=0"),
        `unexpected legacy-mode warning in DB-service mode:\n${stderrText}`,
      );
      // Contract handshake (#33): the relay announces engine + protocol
      // versions as a structured stderr event before doing any work.
      const startup = stderrText.split("\n")
        .filter((l) => l.trim().startsWith("{"))
        .map((l) => JSON.parse(l))
        .find((e) => e.type === "startup");
      assert(startup, `no startup handshake event on stderr:\n${stderrText}`);
      assertEquals(startup.runner_protocol, 4);
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
  name:
    "relay.ts + pyodide_host_adapter.py: single-interpreter (RLM_DB_SERVICE=0) mode, meta stays null",
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
      // build_db_service/bridge_call are never invoked on this path; the
      // adapter's run_for_host_pyodide must handle bridge_call=None,
      // meta=None cleanly (opens db_path directly, single interpreter).
      const { resp, stderrText } = await runRelay(sourcesDir, request, port, {
        RLM_DB_SERVICE: "0",
      });
      assertEquals(resp.error, null);
      assertEquals(resp.answer, "3");
      // Legacy mode must announce itself (#7): a WARNING progress event on
      // stderr, so this mode can never be enabled silently.
      assert(
        stderrText.includes("RLM_DB_SERVICE=0 (legacy mode)"),
        `expected the legacy-mode warning on stderr, got:\n${stderrText}`,
      );
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
        { RLM_DB_SERVICE: "1" },
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
