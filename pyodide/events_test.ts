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
    root: {},
    subcall: {},
    unattributed: {},
    total_tokens: 0,
    wall_time_ms: 0,
  },
  budget: { kind: "snapshot", source: "test", configured: {}, consumed: {}, remaining: {} },
  policy: { contract_enforced: false, outcome: "not_enforced", violation_type: null },
  capability: { outcome: {} },
  done: {
    status: "success",
    ready: true,
    extracted: false,
    iterations: 1,
    usage: {},
    budget: {},
    policy: {},
    retention: {},
    error: null,
    extract_error: null,
    recovered_error: null,
  },
};

function wire(type: string, body: Record<string, unknown>, persistence?: string): string {
  return JSON.stringify({
    type,
    run_id: "run-1",
    seq: 1,
    timestamp: "2026-07-14T00:00:00Z",
    version: 2,
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
  assert(isRlmEvent(wire("code", { iteration: 2, code: "print(get_stats())" })));
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
  assert(!isRlmEvent(JSON.stringify({
    type: "code",
    run_id: "run-1",
    seq: 1,
    timestamp: "2026-07-14T00:00:00Z",
    version: 1,
    persistence_class: "configurable",
    depth: 0,
    ...BODIES.code,
  })));
});

Deno.test("rejects malformed and unknown discriminated lifecycle bodies", () => {
  assert(!isRlmEvent(wire("subcall", {
    phase: "unknown",
    call_id: "c",
    operation: "llm_query",
    iteration: 1,
  })));
  assert(!isRlmEvent(wire("subcall", {
    phase: "failure",
    call_id: "c",
    operation: "llm_query",
    iteration: 1,
    checkpoint: { tokens: 1, subcalls: 1 },
  })));
  assert(!isRlmEvent(wire("subcall", {
    phase: "completion",
    call_id: "c",
    operation: "llm_query",
    iteration: 1,
    checkpoint: { tokens: 1, subcalls: 1 },
    secret: "not in the ABI",
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

Deno.test("Python and relay accept the same lifecycle golden NDJSON", async () => {
  const fixture = new URL("../tests/fixtures/trace-v2-lifecycle.ndjson", import.meta.url);
  const lines = (await Deno.readTextFile(fixture)).trim().split("\n");
  assertEquals(lines.length, 10);
  assert(lines.every(isRlmEvent));
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
    ].sort(),
  );
});
