// events.ts — the structured RLM event vocabulary and the stderr forwarding
// filter (#2). The relay's Pyodide stderr carries the loop's NDJSON events
// (progress / iteration_start / code / output / subcall) plus package-loader
// chatter and stray prints; only real events may reach the host. Extracted from
// relay.ts so the filter is unit-testable (see events_test.ts).

// The events the engine + relay emit. The native subprocess runner uses the same
// set on stdout NDJSON — one vocabulary across substrates.
export const RLM_EVENT_TYPES = new Set<string>([
  "progress", // coarse human-readable status
  "iteration_start", // {iteration, max_iterations}
  "code", // {iteration, code} — the model's generated code, streamed for live UIs
  "output", // {iteration, stdout}
  "subcall", // {depth, seq, ...} (future)
  "reasoning_delta", // {text} — emitted relay-side from streamed /responses
  "done", // final HostResponse mirror (future)
]);

/**
 * True iff a stderr line is a forwardable RLM event: a JSON object whose `type`
 * is one we emit. Parsing (rather than substring matching) keeps loader chatter
 * and stray prints out and admits the whole vocabulary uniformly.
 */
export function isRlmEvent(line: string): boolean {
  const t = line.trim();
  if (!t.startsWith("{")) return false;
  try {
    const o = JSON.parse(t);
    return o !== null && typeof o === "object" && typeof o.type === "string" && RLM_EVENT_TYPES.has(o.type);
  } catch {
    return false;
  }
}
