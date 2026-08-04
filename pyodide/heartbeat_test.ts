// Liveness while a provider call is in flight.
//
// A host's no-progress watchdog cannot tell "blocked on the network" from
// "wedged in WASM" by silence alone, and was killing healthy runs whose subcall
// simply took a while. Batch calls make this worst: they post to
// `/responses/batch`, which does not match the `/responses` streaming suffix, so
// they emit no `reasoning_delta` either — nothing at all between the subcall's
// start and its completion.
//
// The heartbeat is emitted from the relay rather than the engine on purpose.
// Pyodide runs on this same thread, so a timer here cannot fire while generated
// code spins in WASM — which is exactly when a watchdog SHOULD fire. It ticks
// only while the event loop is free, i.e. only when the process is healthy.
//
// Run: deno test --allow-read --allow-env pyodide/heartbeat_test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";

const { isRlmEvent, PERSISTENCE_BY_TYPE, RLM_EVENT_TYPES } = await import(
  "../src/droste/substrates/_relay/events.ts"
);

function wire(body: Record<string, unknown>): string {
  return JSON.stringify({
    type: "heartbeat",
    version: 9,
    run_id: "run-1",
    seq: 4,
    timestamp: "2026-08-03T00:00:00Z",
    depth: 0,
    persistence_class: "transient",
    ...body,
  });
}

Deno.test("the relay forwards a heartbeat", () => {
  assert(isRlmEvent(wire({ elapsed_ms: 15_000 })));
  assert(isRlmEvent(wire({ elapsed_ms: 0 })));
});

Deno.test("a heartbeat carries liveness and nothing else", () => {
  // Content would make it retainable and turn a liveness ping into a privacy
  // question. The body is exactly one non-negative reading.
  assert(!isRlmEvent(wire({ elapsed_ms: 15_000, text: "leaked" })));
  assert(!isRlmEvent(wire({})));
  assert(!isRlmEvent(wire({ elapsed_ms: -1 })));
  assert(!isRlmEvent(wire({ elapsed_ms: "15000" })));
});

Deno.test("heartbeats are transient, so they never reach a run record", () => {
  assertEquals(PERSISTENCE_BY_TYPE.heartbeat, "transient");
  assert(RLM_EVENT_TYPES.has("heartbeat"));
});

Deno.test("the timer only ticks while the event loop is free", async () => {
  // The property the whole design rests on. Reproduced here with the same
  // shape the relay uses: an interval alongside awaited work.
  const ticks: number[] = [];
  const started = Date.now();
  const timer = setInterval(() => ticks.push(Date.now() - started), 20);
  try {
    // Awaiting yields, so the timer runs — a healthy provider wait.
    await new Promise((resolve) => setTimeout(resolve, 120));
    assert(
      ticks.length >= 3,
      `expected ticks while awaiting, got ${ticks.length}`,
    );

    // Blocking the thread is what a wedged Pyodide execution looks like from
    // here. No tick may land during it, which is what lets a watchdog still
    // (correctly) detect a wedge.
    const before = ticks.length;
    const spinUntil = Date.now() + 120;
    while (Date.now() < spinUntil) { /* occupy the single thread */ }
    assertEquals(
      ticks.length,
      before,
      "a blocked thread must not be able to report itself alive",
    );
  } finally {
    clearInterval(timer);
  }
});

// The shape a real subcall actually produced, captured from a live 75-second
// provider call. Pinned verbatim because the first implementation wrapped only
// `fetch` — which a streamed response resolves as soon as headers arrive — and
// so reported nothing for exactly the wait it exists to cover. The mechanism
// test above passed anyway, because it proved the timer worked rather than that
// it spanned the right window. Zero heartbeats in a 4428-delta live run is what
// actually caught it.
Deno.test("a heartbeat a live subcall produced is forwarded", () => {
  const observed = {
    type: "heartbeat",
    elapsed_ms: 15003,
    run_id: "ca4fd03d-5532-4255-9669-86165c4bb0a3",
    parent_run_id: "f6708912-b58d-4e3d-b0e9-a2c28d608635",
    depth: 1,
    seq: 2,
    timestamp: "2026-08-03T21:46:54.805Z",
    version: 9,
    persistence_class: "transient",
  };

  assert(isRlmEvent(JSON.stringify(observed)));
  // depth 1 is a subcall — the calls that run long and, when batched, stream
  // nothing at all. That is the case this whole change exists for.
  assertEquals(observed.depth, 1);
});
