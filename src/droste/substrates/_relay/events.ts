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
  "iteration_start", // {iteration, remaining_tokens}
  "llm_response", // {iteration, response} — the root model's full reply (#35)
  "code", // {iteration, code} — the model's generated code, streamed for live UIs
  "output", // {iteration, stdout, calls_made, answer_ready, answer_content_chars}
  "execution_error", // {iteration, error_type, message} — a step failed; repair may follow (#35)
  "reasoning_delta", // {text} — emitted relay-side from streamed /responses
  "subcall", // broker-correlated start/progress/completion/failure
  "repair", // discriminated repair start/completion/failure
  "extract", // discriminated extract start/completion/failure
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
  subcall: "configurable",
  repair: "configurable",
  extract: "configurable",
  result: "configurable",
  replay: "configurable",
  usage: "durable",
  budget: "durable",
  policy: "durable",
  capability: "durable",
  done: "durable",
};

const ENVELOPE_KEYS = new Set([
  "run_id",
  "seq",
  "timestamp",
  "type",
  "version",
  "persistence_class",
  "parent_run_id",
  "depth",
]);

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isInteger(value: unknown): value is number {
  return Number.isInteger(value);
}

function exactBody(
  body: Record<string, unknown>,
  required: readonly string[],
  optional: readonly string[] = [],
): boolean {
  const allowed = new Set([...required, ...optional]);
  return required.every((key) => Object.hasOwn(body, key)) &&
    Object.keys(body).every((key) => allowed.has(key));
}

function validError(value: unknown, key: "error" | "extract_error"): boolean {
  if (!isObject(value)) return false;
  const fields = key === "error" && Object.hasOwn(value, "code")
    ? ["code", "type"]
    : ["type", "message"];
  return exactBody(value, fields) && fields.every((field) => typeof value[field] === "string");
}

function validSubcallBody(body: Record<string, unknown>): boolean {
  if (
    !exactBody(
      body,
      ["phase", "call_id", "operation", "iteration"],
      ["reservation", "checkpoint", "batch_id", "batch_index", "batch_count", "error"],
    ) ||
    !["start", "progress", "completion", "failure"].includes(String(body.phase)) ||
    !["llm_query", "llm_batch", "llm_batch_with_errors"].includes(String(body.operation)) ||
    typeof body.call_id !== "string" || body.call_id.length === 0 ||
    !isInteger(body.iteration) || body.iteration < 1
  ) return false;
  const reservation = body.reservation;
  const checkpoint = body.checkpoint;
  const error = body.error;
  if (body.phase === "start") {
    if (!isObject(reservation) || checkpoint !== undefined || error !== undefined) return false;
    if (
      !exactBody(reservation, ["tokens", "subcalls", "wall_ms", "depth"]) ||
      !Object.values(reservation).every((value) => isInteger(value) && value >= 0)
    ) return false;
  } else {
    if (!isObject(checkpoint) || reservation !== undefined) return false;
    if (
      !exactBody(checkpoint, ["tokens", "subcalls"]) ||
      !Object.values(checkpoint).every((value) => isInteger(value) && value >= 0)
    ) return false;
    if (body.phase === "failure") {
      if (!validError(error, "error")) return false;
    } else if (error !== undefined) return false;
  }
  if (body.batch_id !== undefined && (typeof body.batch_id !== "string" || !body.batch_id)) {
    return false;
  }
  if ((body.batch_index !== undefined || body.batch_count !== undefined) && !body.batch_id) return false;
  if (body.batch_index !== undefined && (!isInteger(body.batch_index) || body.batch_index < 0)) {
    return false;
  }
  if (body.batch_count !== undefined && (!isInteger(body.batch_count) || body.batch_count < 1)) {
    return false;
  }
  return body.batch_index === undefined || body.batch_count === undefined ||
    body.batch_index < body.batch_count;
}

function validBody(type: string, body: Record<string, unknown>): boolean {
  const stringField = (key: string) => typeof body[key] === "string";
  const integerField = (key: string) => isInteger(body[key]);
  switch (type) {
    case "startup":
      return exactBody(
        body,
        ["engine_version"],
        ["runner_protocol", "provider_protocol", "scaffold_manifest_id", "scaffold_manifest_version"],
      ) && stringField("engine_version");
    case "progress":
      return exactBody(body, ["status"]) && stringField("status");
    case "iteration_start":
      return exactBody(body, ["iteration", "remaining_tokens"]) &&
        integerField("iteration") && integerField("remaining_tokens");
    case "llm_response":
      return exactBody(body, ["iteration", "response"]) &&
        integerField("iteration") && stringField("response");
    case "code":
      return exactBody(body, ["iteration", "code"]) && integerField("iteration") && stringField("code");
    case "output":
      return exactBody(
        body,
        ["iteration", "stdout", "calls_made", "answer_ready", "answer_content_chars"],
        ["stdout_chars"],
      ) && integerField("iteration") && stringField("stdout") && integerField("calls_made") &&
        typeof body.answer_ready === "boolean" && integerField("answer_content_chars") &&
        (body.stdout_chars === undefined || integerField("stdout_chars"));
    case "execution_error":
      return exactBody(body, ["iteration", "error_type", "message"]) &&
        integerField("iteration") && stringField("error_type") && stringField("message");
    case "reasoning_delta":
      return exactBody(body, ["text"]) && stringField("text");
    case "subcall":
      return validSubcallBody(body);
    case "repair": {
      if (
        !exactBody(body, ["phase", "kind", "iteration"], ["error"]) ||
        !["start", "completion", "failure"].includes(String(body.phase)) ||
        !["missing_code", "execution_error", "terminal"].includes(String(body.kind)) ||
        !integerField("iteration") || Number(body.iteration) < 1
      ) return false;
      return body.phase === "failure" ? validError(body.error, "error") : body.error === undefined;
    }
    case "extract":
      if (
        !exactBody(body, ["phase", "iteration"], ["extract_error"]) ||
        !["start", "completion", "failure"].includes(String(body.phase)) ||
        !integerField("iteration") || Number(body.iteration) < 1
      ) return false;
      return body.phase === "failure"
        ? validError(body.extract_error, "extract_error")
        : body.extract_error === undefined;
    case "result":
    case "replay":
      return exactBody(body, ["result"]) && isObject(body.result);
    case "usage":
      return exactBody(
        body,
        ["kind", "root", "subcall", "unattributed", "total_tokens", "wall_time_ms"],
      ) && body.kind === "resolved" && isObject(body.root) && isObject(body.subcall) &&
        isObject(body.unattributed) && integerField("total_tokens") && integerField("wall_time_ms");
    case "budget":
      if (!stringField("kind") || !stringField("source")) return false;
      return body.kind === "snapshot"
        ? exactBody(body, ["kind", "source", "configured", "consumed", "remaining"]) &&
          isObject(body.configured) && isObject(body.consumed) && isObject(body.remaining)
        : body.kind === "mutation" &&
          exactBody(body, ["kind", "source", "action", "resource", "amount"], ["call_id"]) &&
          stringField("action") && stringField("resource") && typeof body.amount === "number";
    case "policy":
      return exactBody(body, ["contract_enforced", "outcome", "violation_type"]) &&
        typeof body.contract_enforced === "boolean" && stringField("outcome") &&
        (body.violation_type === null || stringField("violation_type"));
    case "capability":
      return exactBody(body, ["outcome"]) && isObject(body.outcome);
    case "done":
      return exactBody(
        body,
        [
          "status", "ready", "extracted", "iterations", "usage", "budget", "policy", "retention",
          "error", "extract_error", "recovered_error",
        ],
        ["scaffold_manifest_id", "scaffold_manifest_version", "stdout_chars"],
      ) && stringField("status") && typeof body.ready === "boolean" &&
        typeof body.extracted === "boolean" && integerField("iterations") && isObject(body.usage) &&
        isObject(body.budget) && isObject(body.policy) && isObject(body.retention);
    default:
      return false;
  }
}

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
    if (!(o !== null && typeof o === "object" && !Array.isArray(o) &&
      typeof o.type === "string" && RLM_EVENT_TYPES.has(o.type) &&
      typeof o.run_id === "string" && o.run_id.length > 0 &&
      Number.isInteger(o.seq) && o.seq > 0 &&
      typeof o.timestamp === "string" &&
      /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$/.test(o.timestamp) &&
      o.version === 2 && o.persistence_class === PERSISTENCE_BY_TYPE[o.type] &&
      Number.isInteger(o.depth) && o.depth >= 0 &&
      (o.depth === 0
        ? o.parent_run_id === undefined
        : typeof o.parent_run_id === "string" && o.parent_run_id.length > 0))) return false;
    const body = Object.fromEntries(
      Object.entries(o).filter(([key]) => !ENVELOPE_KEYS.has(key)),
    );
    return validBody(o.type, body);
  } catch {
    return false;
  }
}
