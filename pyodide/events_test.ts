// Tests for the RLM stderr event filter (#1). Run: deno test events_test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import {
  isRlmEvent,
  PERSISTENCE_BY_TYPE,
  RLM_EVENT_TYPES,
} from "../src/droste/substrates/_relay/events.ts";

Deno.test("forwards every emitted event type (with json.dumps spacing)", () => {
  for (const type of RLM_EVENT_TYPES) {
    const event = JSON.stringify({
      type,
      run_id: "run-1",
      seq: 1,
      timestamp: "2026-07-14T00:00:00Z",
      version: 1,
      persistence_class: PERSISTENCE_BY_TYPE[type],
      depth: 0,
      iteration: 1,
    });
    assert(
      isRlmEvent(event),
      `should forward ${type}`,
    );
  }
});

Deno.test("carries the real payload for a code event (live code streaming)", () => {
  assert(
    isRlmEvent(
      JSON.stringify({
        type: "code",
        run_id: "run-1",
        seq: 2,
        timestamp: "2026-07-14T00:00:00Z",
        version: 1,
        persistence_class: "configurable",
        depth: 0,
        iteration: 2,
        code: "print(get_stats())",
      }),
    ),
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
      '{"type": "debug", "msg": "not ours"}', // unknown type
      '{"no_type": true}',
      "{malformed json",
      '"a bare json string"',
      "[1,2,3]", // array, not an object
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
  assert(!isRlmEvent(JSON.stringify({
    type: "code",
    run_id: "run-1",
    seq: 1,
    timestamp: "2026-07-14T00:00:00Z",
    version: 1,
    persistence_class: "durable",
  })));
});

Deno.test("vocabulary matches the engine's emitters", () => {
  // Guard against drift: these are exactly what droste/execution + loop emit,
  // plus the relay-side startup handshake (#33) and reasoning_delta.
  assertEquals(
    [...RLM_EVENT_TYPES].sort(),
    [
      "code",
      "budget",
      "capability",
      "done",
      "execution_error",
      "extract_error",
      "finalization_error",
      "iteration_start",
      "llm_response",
      "output",
      "progress",
      "policy",
      "repair",
      "result",
      "replay",
      "reasoning_delta",
      "startup",
      "usage",
    ].sort(),
  );
});
