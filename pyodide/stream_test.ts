// Hermetic tests for streamResponses — no Pyodide, no network. Verifies the
// ndjson /responses stream is reconstructed into the unary payload the RLM loop
// expects, and that text deltas are forwarded in order.
//
//   deno test pyodide/stream_test.ts

import { strict as assert } from "node:assert";
import { streamResponses } from "../src/droste/substrates/_relay/stream.ts";

function ndjson(events: unknown[]): Response {
  const body = events.map((e) => JSON.stringify(e) + "\n").join("");
  return new Response(body, {
    headers: { "content-type": "application/x-ndjson" },
  });
}

Deno.test("reconstructs output + usage from completion; forwards deltas in order", async () => {
  const deltas: string[] = [];
  const out = await streamResponses(
    ndjson([
      { type: "start", request_id: "x", model: "m", stream_mode: "text-delta" },
      { type: "update", delta: "Hello" },
      { type: "update", delta: " world" },
      {
        type: "completion",
        content: "Hello world",
        stop_reason: "end_turn",
        usage: {
          input_tokens: 1,
          cache_read_input_tokens: 1,
          cache_write_input_tokens: 0,
          output_tokens: 2,
          total_tokens: 3,
        },
      },
    ]),
    (c) => deltas.push(c),
  );
  const payload = JSON.parse(out);
  assert.equal(payload.output[0].type, "message");
  assert.equal(payload.output[0].role, "assistant");
  assert.equal(payload.output[0].content[0].text, "Hello world");
  assert.deepEqual(payload.usage, {
    input_tokens: 1,
    cache_read_input_tokens: 1,
    cache_write_input_tokens: 0,
    output_tokens: 2,
    total_tokens: 3,
  });
  assert.deepEqual(deltas, ["Hello", " world"]);
});

Deno.test("prefers completion.content over accumulated deltas (authoritative)", async () => {
  const out = await streamResponses(
    ndjson([
      { type: "update", delta: "par" },
      { type: "update", delta: "tial" },
      { type: "completion", content: "the full canonical answer", usage: {} },
    ]),
    () => {},
  );
  assert.equal(
    JSON.parse(out).output[0].content[0].text,
    "the full canonical answer",
  );
});

Deno.test("real wire shape: completion carries usage only, text from deltas", async () => {
  // Verified against live ModelRelay (gemini-3-flash-preview): the production
  // `completion` event has NO `content` — only `usage`. The full assistant text
  // is the concatenation of `update` deltas, so reconstruction relies on the
  // accumulated-delta fallback while still taking usage from `completion`.
  const deltas: string[] = [];
  const out = await streamResponses(
    ndjson([
      {
        type: "start",
        stream_mode: "text-delta",
        stream_version: "v2",
        model: "m",
      },
      { type: "update", delta: "Hello" },
      { type: "update", delta: " there" },
      {
        type: "completion",
        usage: { input_tokens: 8, output_tokens: 46, total_tokens: 54 },
      },
    ]),
    (c) => deltas.push(c),
  );
  const payload = JSON.parse(out);
  assert.equal(payload.output[0].content[0].text, "Hello there");
  assert.deepEqual(payload.usage, {
    input_tokens: 8,
    output_tokens: 46,
    total_tokens: 54,
  });
  assert.deepEqual(deltas, ["Hello", " there"]);
});

Deno.test("tolerates the content_delta/{delta:{content}} shape", async () => {
  const deltas: string[] = [];
  const out = await streamResponses(
    ndjson([
      { type: "content_delta", delta: { type: "text", content: "Hi" } },
      { type: "content_delta", delta: { type: "text", content: "!" } },
      {
        type: "completion",
        content: "Hi!",
        usage: { input_tokens: 0, output_tokens: 0, total_tokens: 0 },
      },
    ]),
    (c) => deltas.push(c),
  );
  assert.deepEqual(deltas, ["Hi", "!"]);
  assert.equal(JSON.parse(out).output[0].content[0].text, "Hi!");
});

Deno.test("ignores keepalive/non-JSON lines and unknown event types", async () => {
  const body = [
    "",
    "   ",
    "not json",
    JSON.stringify({ type: "ping" }),
    JSON.stringify({ type: "update", delta: "ok" }),
    JSON.stringify({ type: "completion", content: "ok", usage: {} }),
  ].join("\n");
  const r = new Response(body, {
    headers: { "content-type": "application/x-ndjson" },
  });
  const out = await streamResponses(r, () => {});
  assert.equal(JSON.parse(out).output[0].content[0].text, "ok");
});

Deno.test("a stream that ends without a terminal event throws (droste#43)", async () => {
  // A dropped connection / proxy timeout must never surface accumulated
  // deltas as a clean answer — the loop would execute truncated code.
  const deltas: string[] = [];
  await assert.rejects(
    () =>
      streamResponses(
        ndjson([
          { type: "start", stream_mode: "text-delta", model: "m" },
          { type: "update", delta: "partial answ" },
        ]),
        (c) => deltas.push(c),
      ),
    /ended without a completion event/,
  );
  // Deltas were still forwarded live before the drop was detected.
  assert.deepEqual(deltas, ["partial answ"]);
});

Deno.test("a mid-stream error event throws with the provider detail (droste#43)", async () => {
  await assert.rejects(
    () =>
      streamResponses(
        ndjson([
          { type: "update", delta: "some prefix" },
          {
            type: "error",
            code: "PROVIDER_UNAVAILABLE",
            message: "upstream provider failed",
            detail: "backend 502",
            status: 502,
          },
        ]),
        () => {},
      ),
    /ModelRelay stream error 502: upstream provider failed: backend 502/,
  );
});
