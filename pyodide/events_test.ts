// Tests for the RLM stderr event filter (#2). Run: deno test events_test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { isRlmEvent, RLM_EVENT_TYPES } from "./events.ts";

Deno.test("forwards every emitted event type (with json.dumps spacing)", () => {
  for (const type of RLM_EVENT_TYPES) {
    assert(
      isRlmEvent(`{"type": "${type}", "iteration": 1}`),
      `should forward ${type}`,
    );
    assert(
      isRlmEvent(`{"type":"${type}"}`),
      `whitespace-insensitive for ${type}`,
    );
  }
});

Deno.test("carries the real payload for a code event (live code streaming)", () => {
  assert(
    isRlmEvent(
      `{"type": "code", "iteration": 2, "code": "print(get_stats())"}`,
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

Deno.test("vocabulary matches the engine's emitters", () => {
  // Guard against drift: these are exactly what droste/execution + loop emit,
  // plus relay-side reasoning_delta and the future subcall/done.
  assertEquals(
    [...RLM_EVENT_TYPES].sort(),
    [
      "code",
      "done",
      "extract_error",
      "iteration_start",
      "output",
      "progress",
      "reasoning_delta",
      "subcall",
    ].sort(),
  );
});
