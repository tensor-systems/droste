// Tests for the RLM stderr event filter (#1). Run: deno test events_test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import {
  isRlmEvent,
  PERSISTENCE_BY_TYPE,
  RLM_EVENT_TYPES,
} from "../src/droste/substrates/_relay/events.ts";

const BODIES: Record<string, Record<string, unknown>> = {
  startup: { engine_version: "0.14.0" },
  progress: { status: "working" },
  iteration_start: { iteration: 1, remaining_tokens: 10 },
  llm_response: { iteration: 1, response: "reply" },
  code: { iteration: 1, code: "print(1)" },
  output: {
    iteration: 1,
    stdout: "ok",
    calls_made: 0,
    answer_ready: true,
    answer_content_chars: 2,
  },
  execution_error: { iteration: 1, error_type: "ValueError", message: "bad" },
  reasoning_delta: { text: "thinking" },
  subcall: {
    phase: "start",
    call_id: "call-1",
    operation: "llm_query",
    iteration: 1,
    reservation: { tokens: 10, subcalls: 1, wall_ms: 100, depth: 0 },
  },
  repair: { phase: "start", kind: "execution_error", iteration: 1 },
  extract: { phase: "start", iteration: 1 },
  result: { result: {} },
  replay: { result: {} },
  usage: {
    kind: "resolved",
    root: {
      input_tokens: 0,
      cache_read_tokens: 0,
      cache_creation_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
      requests: 0,
      successes: 0,
      complete: true,
    },
    subcall: {
      input_tokens: 0,
      cache_read_tokens: 0,
      cache_creation_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
      requests: 0,
      successes: 0,
      complete: true,
    },
    unattributed: { total_tokens: 0 },
    total_tokens: 0,
    wall_time_ms: 0,
  },
  usage_progress: {
    boundary: "root",
    kind: "resolved",
    root: {
      input_tokens: 0,
      cache_read_tokens: 0,
      cache_creation_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
      requests: 0,
      successes: 0,
      complete: true,
    },
    subcall: {
      input_tokens: 0,
      cache_read_tokens: 0,
      cache_creation_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
      requests: 0,
      successes: 0,
      complete: true,
    },
    unattributed: { total_tokens: 0 },
    total_tokens: 0,
  },
  budget: {
    kind: "snapshot",
    source: "test",
    configured: {},
    consumed: {},
    remaining: {},
  },
  policy: {
    contract_enforced: false,
    outcome: "not_enforced",
    violation_type: null,
  },
  capability: { outcome: {} },
  done: {
    status: "success",
    ready: true,
    extracted: false,
    iterations: 1,
    usage: {
      kind: "resolved",
      root: {
        input_tokens: 0,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
        output_tokens: 0,
        total_tokens: 0,
        requests: 0,
        successes: 0,
        complete: true,
      },
      subcall: {
        input_tokens: 0,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
        output_tokens: 0,
        total_tokens: 0,
        requests: 0,
        successes: 0,
        complete: true,
      },
      unattributed: { total_tokens: 0 },
      total_tokens: 0,
      wall_time_ms: 0,
    },
    budget: {
      kind: "snapshot",
      source: "test",
      configured: {},
      consumed: {},
      remaining: {},
    },
    policy: {
      contract_enforced: false,
      outcome: "not_enforced",
      violation_type: null,
    },
    retention: {},
    error: null,
    extract_error: null,
    recovered_error: null,
  },
};

function wire(
  type: string,
  body: Record<string, unknown>,
  persistence?: string,
): string {
  return JSON.stringify({
    type,
    run_id: "run-1",
    seq: 1,
    timestamp: "2026-07-14T00:00:00Z",
    version: 5,
    persistence_class: persistence ?? PERSISTENCE_BY_TYPE[type],
    depth: 0,
    ...body,
  });
}

Deno.test("forwards every emitted event type (with json.dumps spacing)", () => {
  for (const type of RLM_EVENT_TYPES) {
    assert(isRlmEvent(wire(type, BODIES[type])), `should forward ${type}`);
  }
});

Deno.test("carries the real payload for a code event (live code streaming)", () => {
  assert(
    isRlmEvent(wire("code", { iteration: 2, code: "print(get_stats())" })),
  );
});

Deno.test("drops non-events: loader chatter, stray prints, empty lines", () => {
  for (
    const noise of [
      "Loading sqlite3",
      "Loading sqlite3, package 1/1",
      "",
      "   ",
      "print output that isn't json",
      '{"type": "debug", "msg": "not ours"}',
      '{"no_type": true}',
      "{malformed json",
      '"a bare json string"',
      "[1,2,3]",
      "42",
    ]
  ) {
    assert(!isRlmEvent(noise), `should DROP: ${JSON.stringify(noise)}`);
  }
});

Deno.test("a type that is not a string is not an event", () => {
  assert(!isRlmEvent(`{"type": 5}`));
  assert(!isRlmEvent(`{"type": null}`));
});

Deno.test("rejects partial and falsely classified envelopes", () => {
  assert(!isRlmEvent(`{"type":"code"}`));
  assert(!isRlmEvent(wire("code", BODIES.code, "durable")));
  assert(
    !isRlmEvent(JSON.stringify({
      type: "code",
      run_id: "run-1",
      seq: 1,
      timestamp: "2026-07-14T00:00:00Z",
      version: 1,
      persistence_class: "configurable",
      depth: 0,
      ...BODIES.code,
    })),
  );
});

Deno.test("rejects malformed and unknown discriminated lifecycle bodies", () => {
  assert(
    !isRlmEvent(wire("subcall", {
      phase: "unknown",
      call_id: "c",
      operation: "llm_query",
      iteration: 1,
    })),
  );
  assert(
    !isRlmEvent(wire("subcall", {
      phase: "failure",
      call_id: "c",
      operation: "llm_query",
      iteration: 1,
      checkpoint: { tokens: 1, subcalls: 1 },
    })),
  );
  assert(
    !isRlmEvent(wire("subcall", {
      phase: "completion",
      call_id: "c",
      operation: "llm_query",
      iteration: 1,
      checkpoint: { tokens: 1, subcalls: 1 },
      secret: "not in the ABI",
    })),
  );
  assert(
    !isRlmEvent(wire("subcall", {
      phase: "start",
      call_id: "batch",
      operation: "llm_batch",
      iteration: 1,
      reservation: { tokens: 1, subcalls: 1, wall_ms: 1, depth: 0 },
    })),
  );
  assert(
    !isRlmEvent(wire("subcall", {
      phase: "start",
      call_id: "query",
      operation: "llm_query",
      iteration: 1,
      reservation: { tokens: 1, subcalls: 1, wall_ms: 1, depth: 0 },
      batch_count: 1,
    })),
  );
  assert(
    isRlmEvent(wire("subcall", {
      phase: "start",
      call_id: "empty-batch",
      operation: "llm_batch",
      iteration: 1,
      reservation: { tokens: 0, subcalls: 0, wall_ms: 1, depth: 0 },
      batch_count: 0,
    })),
  );
});

Deno.test("usage totals reconcile at top level and inside done", () => {
  const mismatchedUsage = { ...BODIES.usage, total_tokens: 1 };

  assert(!isRlmEvent(wire("usage", mismatchedUsage)));
  assert(
    !isRlmEvent(wire("done", {
      ...BODIES.done,
      usage: mismatchedUsage,
    })),
  );
});

Deno.test("usage requires cache classes and validates complete subsets", () => {
  const missingCache = structuredClone(BODIES.usage);
  delete (missingCache.root as Record<string, unknown>).cache_read_tokens;
  assert(!isRlmEvent(wire("usage", missingCache)));

  const invalidComplete = structuredClone(BODIES.usage);
  Object.assign(invalidComplete.root as Record<string, unknown>, {
    input_tokens: 1,
    cache_read_tokens: 1,
    cache_creation_tokens: 1,
    total_tokens: 1,
  });
  invalidComplete.total_tokens = 1;
  assert(!isRlmEvent(wire("usage", invalidComplete)));

  (invalidComplete.root as Record<string, unknown>).complete = false;
  invalidComplete.kind = "partial";
  assert(isRlmEvent(wire("usage", invalidComplete)));
  assert(isRlmEvent(wire("done", {
    ...BODIES.done,
    usage: invalidComplete,
  })));
});

Deno.test("successful output beginning ERROR remains an output event", () => {
  assert(isRlmEvent(wire("output", {
    iteration: 1,
    stdout: "ERROR: ordinary successful stdout",
    calls_made: 0,
    answer_ready: false,
    answer_content_chars: 0,
  })));
});

Deno.test("Python and relay accept the same execution golden NDJSON", async () => {
  const fixture = new URL(
    "../src/droste/testing/fixtures/trace-v5-execution.ndjson",
    import.meta.url,
  );
  const lines = (await Deno.readTextFile(fixture)).trim().split("\n");
  assertEquals(lines.length, 9);
  assert(lines.every(isRlmEvent));

  const events = lines.map((line) =>
    JSON.parse(line) as Record<string, unknown>
  );
  assertEquals(
    events.map((event) => event.type),
    [
      "llm_response",
      "code",
      "output",
      "llm_response",
      "code",
      "execution_error",
      "llm_response",
      "code",
      "output",
    ],
  );
  assertEquals(
    events.slice(0, 6).map((event) => event.iteration),
    [1, 1, 1, 2, 2, 2],
  );
  assertEquals(events[2].stdout, "ERROR: ordinary successful stdout\n");
  assertEquals(events[2].type, "output");
  assertEquals(events[5].error_type, "ValueError");
  assertEquals(events[6].depth, 1);
  assertEquals(events[6].parent_run_id, "golden-execution-root");
  assertEquals(events[6].seq, 1);
  assertEquals(
    events.filter((event) =>
      ["code", "output", "execution_error"].includes(String(event.type))
    ).length,
    6,
  );
});

Deno.test("Python and relay accept the same lifecycle golden NDJSON", async () => {
  const fixture = new URL(
    "../src/droste/testing/fixtures/trace-v5-lifecycle.ndjson",
    import.meta.url,
  );
  const lines = (await Deno.readTextFile(fixture)).trim().split("\n");
  assertEquals(lines.length, 67);
  assert(lines.every(isRlmEvent));

  const runs = new Map<string, Record<string, unknown>[]>();
  const closed = new Set<string>();
  let previousRunId: string | undefined;
  for (const line of lines) {
    const event = JSON.parse(line) as Record<string, unknown>;
    const runId = String(event.run_id);
    if (runId !== previousRunId) {
      assert(!closed.has(runId), `non-contiguous golden run ${runId}`);
      if (previousRunId !== undefined) closed.add(previousRunId);
      previousRunId = runId;
    }
    const events = runs.get(runId) ?? [];
    events.push(event);
    runs.set(runId, events);
  }
  assertEquals([...runs.keys()], [
    "golden-success",
    "golden-recovered",
    "golden-output-limit",
    "golden-extract-failed",
    "golden-cancelled",
  ]);

  const errorType = (value: unknown): unknown =>
    value !== null && typeof value === "object"
      ? (value as Record<string, unknown>).type
      : undefined;
  for (const events of runs.values()) {
    assertEquals(
      events.map((event) => event.seq),
      Array.from({ length: events.length }, (_, index) => index + 1),
    );
    assertEquals(events.at(-1)?.type, "done");
    assertEquals(events.filter((event) => event.type === "done").length, 1);

    const result = (events.find((event) =>
      event.type === "result"
    )?.result ?? {}) as Record<string, unknown>;
    const done = events.at(-1) ?? {};
    const usage = done.usage as Record<string, unknown>;
    const subcallUsage = usage.subcall as Record<string, unknown>;
    assertEquals(result.ready, done.ready);
    assertEquals(result.extracted, done.extracted);
    assertEquals(result.iterations, done.iterations);
    assertEquals(result.tokens_used, usage.total_tokens);
    assertEquals(result.subcalls, subcallUsage.requests);
    assertEquals(result.successful_subcalls, subcallUsage.successes);
    assertEquals(result.stdout_chars, done.stdout_chars);
    const scaffoldManifest = result.scaffold_manifest as Record<
      string,
      unknown
    >;
    assertEquals(scaffoldManifest.id, done.scaffold_manifest_id);
    assertEquals(
      scaffoldManifest.schema_version,
      done.scaffold_manifest_version,
    );
    for (const key of ["error", "extract_error", "recovered_error"]) {
      assertEquals(errorType(result[key]), errorType(done[key]));
    }
    for (const type of ["usage", "budget", "policy"]) {
      const emitted = events.find((event) =>
        event.type === type
      );
      assertEquals(
        Object.fromEntries(
          Object.entries(emitted ?? {}).filter(([key]) =>
            ![
              "run_id",
              "seq",
              "timestamp",
              "type",
              "version",
              "persistence_class",
              "parent_run_id",
              "depth",
            ].includes(key)
          ),
        ),
        done[type],
      );
    }
  }

  const lifecycle = lines.map((line) =>
    JSON.parse(line) as Record<string, unknown>
  );
  assert(lifecycle.some((event) => event.type === "execution_error"));
  assert(
    lifecycle.some((event) =>
      event.type === "subcall" && event.operation === "llm_batch" &&
      event.phase === "failure" && event.call_id === "batch-failed"
    ),
  );
  assert(
    lifecycle.some((event) => {
      if (event.type !== "subcall" || event.phase !== "failure") return false;
      const error = event.error as Record<string, unknown> | undefined;
      return error?.code === "cancelled" &&
        error.type === "CapabilityCancelled";
    }),
  );
  assertEquals(
    new Set(
      lifecycle.filter((event) => event.type === "repair").map((event) =>
        event.phase
      ),
    ),
    new Set(["start", "completion", "failure"]),
  );
  assertEquals(
    new Set(
      lifecycle.filter((event) => event.type === "extract").map((event) =>
        event.phase
      ),
    ),
    new Set(["start", "completion", "failure"]),
  );
  assertEquals(
    runs.get("golden-output-limit")?.some((event) => event.type === "output"),
    false,
  );
  assertEquals(runs.get("golden-cancelled")?.at(-1)?.status, "error");
  assertEquals(
    errorType(runs.get("golden-cancelled")?.at(-1)?.error),
    "RuntimeError",
  );
  assert(
    runs.get("golden-cancelled")?.some((event) =>
      event.type === "execution_error" &&
      event.error_type === "CapabilityCallError"
    ),
  );
  const startup = runs.get("golden-success")?.find((event) =>
    event.type === "startup"
  );
  const successfulDone = runs.get("golden-success")?.at(-1);
  assertEquals(
    startup?.scaffold_manifest_id,
    successfulDone?.scaffold_manifest_id,
  );
  assertEquals(
    startup?.scaffold_manifest_version,
    successfulDone?.scaffold_manifest_version,
  );
});

Deno.test("runner refusal fixture remains outside the event stream", async () => {
  const fixture = new URL(
    "../src/droste/testing/fixtures/runner-v9-refusal.ndjson",
    import.meta.url,
  );
  const bytes = await Deno.readTextFile(fixture);
  const refusal = JSON.parse(bytes) as Record<string, unknown>;
  assertEquals(refusal.status, "refusal");
  assertEquals(refusal.protocol_version, 9);
  assertEquals(refusal.run_id, null);
  assertEquals(refusal.run_record, null);
  assertEquals(
    (refusal.error as Record<string, unknown>).type,
    "protocol_version_missing",
  );
  assert(!isRlmEvent(bytes));
});

Deno.test("vocabulary matches the engine's emitters", () => {
  assertEquals(
    [...RLM_EVENT_TYPES].sort(),
    [
      "budget",
      "capability",
      "code",
      "done",
      "execution_error",
      "extract",
      "iteration_start",
      "llm_response",
      "output",
      "policy",
      "progress",
      "reasoning_delta",
      "repair",
      "replay",
      "result",
      "startup",
      "subcall",
      "usage",
      "usage_progress",
    ].sort(),
  );
});
