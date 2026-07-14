// events.ts — the structured RLM event vocabulary and the stderr forwarding
// filter (#1). The relay's Pyodide stderr carries the loop's NDJSON events
// (progress / iteration_start / code / output) plus package-loader
// chatter and stray prints; only real events may reach the host. Extracted from
// relay.ts so the filter is unit-testable (see events_test.ts).

// The events the engine + relay emit. The native subprocess runner uses the same
// set on stdout NDJSON — one vocabulary across substrates.
export const RLM_EVENT_TYPES = new Set<string>([
  "startup", // {engine_version, runner_protocol, provider_protocol} — contract handshake (#33)
  "progress", // coarse human-readable status
  "iteration_start", // {iteration, max_iterations}
  "llm_response", // {iteration, response} — the root model's full reply (#35)
  "code", // {iteration, code} — the model's generated code, streamed for live UIs
  "output", // {iteration, stdout, calls_made, answer_ready, answer_content_chars}
  "execution_error", // {iteration, error_type, message} — a step failed; repair may follow (#35)
  "reasoning_delta", // {text} — emitted relay-side from streamed /responses
  "finalization_error", // {error_type, message} — terminal root finalization failed
  "extract_error", // {error_type, message} — post-exhaustion extract pass failed; answer is raw loop output
  "repair", // configurable repair details
  "result", // canonical unary-equivalent final result
  "replay", // configurable replay details
  "usage", // durable resolved accounting
  "budget", // durable budget facts
  "policy", // durable policy decisions
  "capability", // durable broker-owned capability outcome
  "done", // durable terminal result mirror
]);

export const PERSISTENCE_BY_TYPE: Readonly<Record<string, string>> = {
  startup: "transient",
  progress: "transient",
  reasoning_delta: "transient",
  iteration_start: "configurable",
  llm_response: "configurable",
  code: "configurable",
  output: "configurable",
  execution_error: "configurable",
  finalization_error: "configurable",
  extract_error: "configurable",
  repair: "configurable",
  result: "configurable",
  replay: "configurable",
  usage: "durable",
  budget: "durable",
  policy: "durable",
  capability: "durable",
  done: "durable",
};

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
    return o !== null && typeof o === "object" && !Array.isArray(o) &&
      typeof o.type === "string" && RLM_EVENT_TYPES.has(o.type) &&
      typeof o.run_id === "string" && o.run_id.length > 0 &&
      Number.isInteger(o.seq) && o.seq > 0 &&
      typeof o.timestamp === "string" &&
      /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/.test(o.timestamp) &&
      o.version === 1 && o.persistence_class === PERSISTENCE_BY_TYPE[o.type] &&
      Number.isInteger(o.depth) && o.depth >= 0 &&
      (o.depth === 0
        ? o.parent_run_id === undefined
        : typeof o.parent_run_id === "string" && o.parent_run_id.length > 0);
  } catch {
    return false;
  }
}
