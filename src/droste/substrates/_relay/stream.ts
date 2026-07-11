// ndjson `/responses` stream parsing for the Pyodide relay.
//
// ModelRelay's streaming /responses (profile "responses-stream/v2") emits ndjson
// events: `start`, repeated `update` ({delta:"<chunk>"}), and a terminal
// `completion` ({content, usage}). This forwards each text delta via `onDelta`
// (the host renders the model's reasoning live) and reconstructs the SAME payload
// the non-streaming /responses returns — so the RLM loop, which calls the unary
// client and reads `output[].content[].text` + `usage`, is unaffected.
//
// Extracted from relay.ts so it is unit-testable without Pyodide or the network.

export async function streamResponses(
  r: Response,
  onDelta: (chunk: string) => void,
): Promise<string> {
  const reader = r.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let assembled = ""; // accumulated from deltas (live display + fallback text)
  let finalContent: string | null = null; // authoritative text from `completion`
  let usage: unknown = null;
  // The wire contract (responses-stream/v2) terminates every stream with
  // exactly one `completion` or `error` record. Anything else — a dropped
  // connection, a proxy timeout — leaves this false, and the accumulated
  // deltas are a TRUNCATED answer that must never be returned as success
  // (droste#43): the loop would execute partial generated code.
  let sawCompletion = false;

  const handleLine = (raw: string) => {
    const line = raw.trim();
    if (!line) return;
    let ev: Record<string, unknown>;
    try {
      ev = JSON.parse(line);
    } catch {
      return; // ignore keepalives / non-JSON noise
    }
    // v2 schema: {"type":"update","delta":"<chunk>"}. Also tolerate the
    // {"type":"content_delta","delta":{"type":"text","content":"<chunk>"}} shape.
    let chunk: string | null = null;
    if (ev.type === "update" && typeof ev.delta === "string") {
      chunk = ev.delta;
    } else if (
      ev.type === "content_delta" &&
      typeof ev.delta === "object" && ev.delta !== null &&
      (ev.delta as Record<string, unknown>).type === "text"
    ) {
      chunk = String((ev.delta as Record<string, unknown>).content ?? "");
    }
    if (chunk) {
      assembled += chunk;
      onDelta(chunk);
      return;
    }
    if (ev.type === "completion") {
      sawCompletion = true;
      if (typeof ev.content === "string") finalContent = ev.content;
      if (ev.usage) usage = ev.usage;
      return;
    }
    if (ev.type === "error") {
      // Mid-stream provider failure: surface it as the failure it is,
      // never keep accumulating toward a fake success.
      const status = typeof ev.status === "number" ? ev.status : 0;
      const message = typeof ev.message === "string" && ev.message
        ? ev.message
        : "provider stream error";
      const detail = typeof ev.detail === "string" && ev.detail
        ? `: ${ev.detail}`
        : "";
      throw new Error(`ModelRelay stream error ${status}: ${message}${detail}`);
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      handleLine(buf.slice(0, nl));
      buf = buf.slice(nl + 1);
    }
  }
  if (buf) handleLine(buf); // trailing line with no newline

  if (!sawCompletion) {
    throw new Error(
      `ModelRelay stream ended without a completion event (connection dropped ` +
        `mid-generation; ${assembled.length} chars of partial output discarded)`,
    );
  }

  const text = finalContent ?? assembled;
  const payload: Record<string, unknown> = {
    output: [{
      type: "message",
      role: "assistant",
      content: [{ type: "text", text }],
    }],
  };
  if (usage) payload.usage = usage; // {input_tokens, output_tokens, total_tokens}
  return JSON.stringify(payload);
}
