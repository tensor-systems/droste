// broker.ts — host-side credential handling for the A′ sandbox split.
//
// In the pre-A′ (legacy) relay, the full request — including the ModelRelay
// api_key / customer_token — is set as a global INSIDE the untrusted Pyodide
// interpreter, and the sandbox's Python client assembles the auth header itself.
// That puts a live billing credential within reach of LLM-authored code.
//
// A′ moves the credential out: the broker (Deno host) strips it from the request
// the sandbox sees, holds it host-side, and injects the auth header on every
// ModelRelay call — overriding whatever the sandbox sent. The sandbox can no
// longer read the token or influence which credential is used.
//
// Pure + dependency-free so it is unit-testable without Pyodide or the network
// (see broker_test.ts), mirroring the stream.ts extraction.

export interface Credentials {
  authType: string; // "customer_token" | "api_key"
  apiKey?: string;
  customerToken?: string;
  runnerToken?: string;
}

// Header names the sandbox is not allowed to control. Compared case-insensitively.
const AUTH_HEADER_NAMES = ["authorization", "x-modelrelay-api-key"];

// The ONE origin the held credential may be injected into. Exact host + HTTPS —
// never a substring match: the sandbox supplies the URL, so `https://evil.com/
// ?x=api.modelrelay.ai` or plaintext `http://api.modelrelay.ai` must NOT qualify.
const MODELRELAY_HOSTNAME = "api.modelrelay.ai";

// The routes the credential is scoped to — the LLM transport, nothing else.
// Exact-match set, not a prefix: adding a path here is a deliberate widening
// of what the held credential can reach, so each one is enumerated by hand.
const MODELRELAY_RESPONSES_PATHS = new Set([
  "/api/v1/responses",
  "/api/v1/responses/batch",
]);

/**
 * True only for the exact `POST https://api.modelrelay.ai/api/v1/responses(/batch)`
 * calls — the sole requests the held credential may be attached to. The sandbox
 * controls the method and URL, so gating on origin alone is not enough: it would
 * leave the token usable against other ModelRelay endpoints (e.g. customer/billing
 * routes). Requires POST + HTTPS + exact host + exact path so the credential is a
 * single-purpose LLM-transport key, never a general ModelRelay credential.
 */
export function isModelRelayResponsesCall(
  method: string,
  url: string,
): boolean {
  if (method.toUpperCase() !== "POST") return false;
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return false;
  }
  return (
    parsed.protocol === "https:" &&
    parsed.hostname === MODELRELAY_HOSTNAME &&
    MODELRELAY_RESPONSES_PATHS.has(parsed.pathname)
  );
}

/** Exact-match short-lived hosted-runner callback endpoints. */
export function isRunnerCallback(
  method: string,
  url: string,
  endpoints: readonly unknown[],
): boolean {
  if (method.toUpperCase() !== "POST") return false;
  return endpoints.some((endpoint) =>
    typeof endpoint === "string" && endpoint.length > 0 && endpoint === url
  );
}

/**
 * Callback failures are protocol values only for exact runner model callbacks
 * with a JSON media type. Other HTTP failures retain the relay's fail-fast
 * transport behavior, including data-source callbacks and direct /responses.
 */
export function shouldReturnRunnerCallbackFailureBody(
  method: string,
  url: string,
  contentType: string,
  endpoints: readonly unknown[],
): boolean {
  return isRunnerCallbackJSONContentType(contentType) &&
    isRunnerCallback(method, url, endpoints);
}

const MIME_TOKEN = String.raw`[!#$%&'*+.^_\x60|~0-9A-Z-]+`;
const RUNNER_CALLBACK_JSON_CONTENT_TYPE = new RegExp(
  String
    .raw`^[ \t]*application\/(?:json|${MIME_TOKEN}\+json)[ \t]*(?:;[ \t]*charset[ \t]*=[ \t]*(?:utf-8|"utf-8")[ \t]*)?$`,
  "i",
);

/** The one media-type grammar accepted for trusted runner callback failures. */
export function isRunnerCallbackJSONContentType(contentType: string): boolean {
  if (contentType.length > 512) return false;
  return RUNNER_CALLBACK_JSON_CONTENT_TYPE.exec(contentType)?.[0] ===
    contentType;
}

const JSON_NUMBER = /-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/y;
const MAX_CALLBACK_JSON_DEPTH = 256;
const MAX_CALLBACK_JSON_FLOAT_CHARS = 256;
const MAX_SIGNED_INT64_DIGITS = "9223372036854775807";
const MIN_SIGNED_INT64_DIGITS = "9223372036854775808";
const RUNNER_USAGE_COUNTERS = new Set([
  "input_tokens",
  "output_tokens",
  "total_tokens",
  "cache_read_input_tokens",
  "cache_write_input_tokens",
  "cache_creation_input_tokens",
  "reasoning_tokens",
]);

type ParsedJSONValue =
  | { kind: "string"; text: string; value: string }
  | {
    kind: "number";
    text: string;
    integer: boolean;
    signedInt64: boolean;
  }
  | { kind: "object"; text: string; entries: Map<string, ParsedJSONValue> }
  | { kind: "array"; text: string; items: ParsedJSONValue[] }
  | { kind: "literal"; text: string };

function invalidCallbackJSON(): never {
  throw new Error("runner callback failure body is not valid JSON");
}

function normalizeHeldSecrets(heldSecrets: readonly unknown[]): string[] {
  const unique = new Set(
    heldSecrets.filter((value): value is string =>
      typeof value === "string" && value.length > 0
    ),
  );
  return [...unique].sort((left, right) => {
    const lengthOrder = right.length - left.length;
    if (lengthOrder !== 0) return lengthOrder;
    return left === right ? 0 : left < right ? -1 : 1;
  });
}

function heldSecretMarker(secrets: readonly string[]): string {
  return secrets.some((secret) => "[redacted]".includes(secret))
    ? ""
    : "[redacted]";
}

/** Replace the union of every match in the original string, never sequentially. */
function redactLiteralHeldSecretMatches(
  text: string,
  secrets: readonly string[],
): string {
  const matches: Array<{ start: number; end: number }> = [];
  for (const secret of secrets) {
    let searchFrom = 0;
    while (searchFrom <= text.length - secret.length) {
      const start = text.indexOf(secret, searchFrom);
      if (start < 0) break;
      matches.push({ start, end: start + secret.length });
      // Advance one code unit so overlapping occurrences join the same span.
      searchFrom = start + 1;
    }
  }
  if (matches.length === 0) return text;
  matches.sort((left, right) =>
    left.start - right.start || right.end - left.end
  );
  const merged: Array<{ start: number; end: number }> = [];
  for (const match of matches) {
    const previous = merged.at(-1);
    if (previous !== undefined && match.start <= previous.end) {
      previous.end = Math.max(previous.end, match.end);
    } else {
      merged.push({ ...match });
    }
  }
  const marker = heldSecretMarker(secrets);
  const parts: string[] = [];
  let offset = 0;
  for (const match of merged) {
    parts.push(text.slice(offset, match.start), marker);
    offset = match.end;
  }
  parts.push(text.slice(offset));
  const redacted = parts.join("");
  return secrets.some((secret) => redacted.includes(secret))
    ? marker
    : redacted;
}

class JSONHeldSecretRedactor {
  private offset = 0;

  constructor(
    private readonly source: string,
    private readonly secrets: readonly string[],
  ) {}

  parse(): ParsedJSONValue {
    const value = this.parseValue(0);
    const trailing = this.parseWhitespace();
    if (this.offset !== this.source.length) invalidCallbackJSON();
    return { ...value, text: value.text + trailing };
  }

  redact(): string {
    return this.parse().text;
  }

  private parseValue(depth: number): ParsedJSONValue {
    if (depth > MAX_CALLBACK_JSON_DEPTH) invalidCallbackJSON();
    const leading = this.parseWhitespace();
    const char = this.source[this.offset];
    let parsed: ParsedJSONValue;
    if (char === '"') {
      parsed = this.parseString();
    } else if (char === "{") {
      parsed = this.parseObject(depth + 1);
    } else if (char === "[") {
      parsed = this.parseArray(depth + 1);
    } else {
      for (const literal of ["true", "false", "null"] as const) {
        if (this.source.startsWith(literal, this.offset)) {
          this.offset += literal.length;
          return { kind: "literal", text: leading + literal };
        }
      }
      parsed = this.parseNumber();
    }
    return { ...parsed, text: leading + parsed.text };
  }

  private parseNumber(): ParsedJSONValue {
    JSON_NUMBER.lastIndex = this.offset;
    const number = JSON_NUMBER.exec(this.source);
    if (number === null) invalidCallbackJSON();
    this.offset = JSON_NUMBER.lastIndex;
    const raw = number[0];
    const integer = !/[.eE]/.test(raw);
    if (!integer) {
      // Mirror Python's bounded parse_float without rewriting the token.
      // Number() is validation-only; the original lexeme remains authoritative.
      if (
        raw.length > MAX_CALLBACK_JSON_FLOAT_CHARS ||
        !Number.isFinite(Number(raw))
      ) {
        invalidCallbackJSON();
      }
    }
    return {
      kind: "number",
      text: raw,
      integer,
      signedInt64: !integer || isSignedInt64(raw),
    };
  }

  private parseObject(depth: number): ParsedJSONValue {
    this.offset += 1;
    const parts = ["{", this.parseWhitespace()];
    const entries = new Map<string, ParsedJSONValue>();
    if (this.source[this.offset] === "}") {
      this.offset += 1;
      return { kind: "object", text: parts.join("") + "}", entries };
    }
    while (true) {
      if (this.source[this.offset] !== '"') invalidCallbackJSON();
      const key = this.parseString();
      if (entries.has(key.value)) invalidCallbackJSON();
      parts.push(key.text, this.parseWhitespace());
      if (this.source[this.offset] !== ":") invalidCallbackJSON();
      this.offset += 1;
      const value = this.parseValue(depth);
      entries.set(key.value, value);
      parts.push(":", value.text, this.parseWhitespace());
      const separator = this.source[this.offset];
      if (separator === "}") {
        this.offset += 1;
        parts.push("}");
        return { kind: "object", text: parts.join(""), entries };
      }
      if (separator !== ",") invalidCallbackJSON();
      this.offset += 1;
      parts.push(",", this.parseWhitespace());
    }
  }

  private parseArray(depth: number): ParsedJSONValue {
    this.offset += 1;
    const parts = ["[", this.parseWhitespace()];
    const items: ParsedJSONValue[] = [];
    if (this.source[this.offset] === "]") {
      this.offset += 1;
      return { kind: "array", text: parts.join("") + "]", items };
    }
    while (true) {
      const value = this.parseValue(depth);
      items.push(value);
      parts.push(value.text, this.parseWhitespace());
      const separator = this.source[this.offset];
      if (separator === "]") {
        this.offset += 1;
        parts.push("]");
        return { kind: "array", text: parts.join(""), items };
      }
      if (separator !== ",") invalidCallbackJSON();
      this.offset += 1;
      parts.push(",", this.parseWhitespace());
    }
  }

  private parseString(): Extract<ParsedJSONValue, { kind: "string" }> {
    const start = this.offset;
    this.offset += 1;
    while (this.offset < this.source.length) {
      const char = this.source[this.offset];
      if (char === '"') {
        this.offset += 1;
        const token = this.source.slice(start, this.offset);
        let decoded: string;
        try {
          decoded = JSON.parse(token);
        } catch {
          invalidCallbackJSON();
        }
        const redacted = redactLiteralHeldSecretMatches(
          decoded,
          this.secrets,
        );
        return {
          kind: "string",
          text: redacted === decoded ? token : JSON.stringify(redacted),
          value: decoded,
        };
      }
      if (char === "\\") {
        this.offset += 2;
        continue;
      }
      if (char.charCodeAt(0) < 0x20) invalidCallbackJSON();
      this.offset += 1;
    }
    return invalidCallbackJSON();
  }

  private parseWhitespace(): string {
    const start = this.offset;
    while (
      this.offset < this.source.length &&
      " \t\r\n".includes(this.source[this.offset])
    ) {
      this.offset += 1;
    }
    return this.source.slice(start, this.offset);
  }
}

function isSignedInt64(raw: string): boolean {
  const negative = raw.startsWith("-");
  const digits = negative ? raw.slice(1) : raw;
  if (digits.length < MAX_SIGNED_INT64_DIGITS.length) return true;
  if (digits.length > MAX_SIGNED_INT64_DIGITS.length) return false;
  return digits <=
    (negative ? MIN_SIGNED_INT64_DIGITS : MAX_SIGNED_INT64_DIGITS);
}

function containsOutOfRangeInteger(value: ParsedJSONValue): boolean {
  if (value.kind === "number") return value.integer && !value.signedInt64;
  if (value.kind === "object") {
    return [...value.entries.values()].some(containsOutOfRangeInteger);
  }
  if (value.kind === "array") {
    return value.items.some(containsOutOfRangeInteger);
  }
  return false;
}

function hasOutOfRangeIntegerOutsideUsageCounters(
  root: Extract<ParsedJSONValue, { kind: "object" }>,
): boolean {
  for (const [key, value] of root.entries) {
    if (key !== "usage") {
      if (containsOutOfRangeInteger(value)) return true;
      continue;
    }
    if (value.kind !== "object") return containsOutOfRangeInteger(value);
    for (const [usageKey, usageValue] of value.entries) {
      if (
        RUNNER_USAGE_COUNTERS.has(usageKey) &&
        usageValue.kind === "number" &&
        usageValue.integer &&
        !usageValue.signedInt64
      ) {
        continue;
      }
      if (containsOutOfRangeInteger(usageValue)) return true;
    }
  }
  return false;
}

function assertTrustedRunnerCallbackFailure(
  parsed: ParsedJSONValue,
): asserts parsed is Extract<ParsedJSONValue, { kind: "object" }> {
  if (parsed.kind !== "object") invalidCallbackJSON();
  const error = parsed.entries.get("error");
  const usage = parsed.entries.get("usage");
  if (
    error?.kind !== "string" || error.value !== "api_error" ||
    usage?.kind !== "object" ||
    hasOutOfRangeIntegerOutsideUsageCounters(parsed)
  ) {
    invalidCallbackJSON();
  }
}

/**
 * Prove one callback failure body has the native trusted-envelope schema while
 * preserving number lexemes and redacting only decoded strings. No JSON.parse
 * of the complete value is allowed at this JS precision boundary.
 */
export function validateAndRedactRunnerCallbackFailureBody(
  text: string,
  heldSecrets: readonly unknown[] = [],
): string {
  const secrets = normalizeHeldSecrets(heldSecrets);
  const parsed = new JSONHeldSecretRedactor(text, secrets).parse();
  assertTrustedRunnerCallbackFailure(parsed);
  // Redaction can change keys or the discriminator for pathological held
  // values. Re-parse the exact safe text so the body crossing into Python is
  // itself canonical, not merely canonical before credential removal.
  const safe = new JSONHeldSecretRedactor(parsed.text, []).parse();
  assertTrustedRunnerCallbackFailure(safe);
  return safe.text;
}

/**
 * Redact credentials from JSON string tokens while copying every number token
 * byte-for-byte. Malformed JSON throws before any response reaches Python.
 */
export function redactHeldSecrets(
  text: string,
  heldSecrets: readonly unknown[] = [],
): string {
  const secrets = normalizeHeldSecrets(heldSecrets);
  return new JSONHeldSecretRedactor(text, secrets).redact();
}

function redactLiteralHeldSecrets(
  text: string,
  heldSecrets: readonly unknown[],
): string {
  const secrets = normalizeHeldSecrets(heldSecrets);
  return redactLiteralHeldSecretMatches(text, secrets);
}

/** Bound relay diagnostics must never echo held or conventionally shaped secrets. */
export function redactRelayErrorText(
  text: string,
  heldSecrets: readonly unknown[] = [],
): string {
  let redacted = redactLiteralHeldSecrets(text, heldSecrets);
  redacted = redacted.replace(
    /bearer\s+[A-Za-z0-9._~+/=-]+/gi,
    "[redacted]",
  );
  redacted = redacted.replace(
    /\b(api[_-]?key|apikey|token|authorization|secret|password|key)\b(["'\s]*[:=]["'\s]*)[^\s"'&,;}]+/gi,
    (_match, name: string, separator: string) =>
      `${name}${separator}[redacted]`,
  );
  redacted = redacted.replace(/\bmr_sk_[A-Za-z0-9_-]{8,}/g, "[redacted]");
  redacted = redacted.replace(/\bsk-[A-Za-z0-9_-]{8,}/g, "[redacted]");
  // Conventional-pattern markers must not introduce a held value that was not
  // present in the source (for example, a pathological held value "redact").
  return redactLiteralHeldSecrets(redacted, heldSecrets);
}

/**
 * Pull secret credentials out of the request so they never become a sandbox
 * global. The normalized auth type is nonsecret routing metadata, so preserve
 * it for adapters that must distinguish customer-tier defaults from tierless
 * API-key requests without seeing either credential.
 */
export function splitCredentials(
  request: Record<string, unknown>,
): { creds: Credentials; sandboxRequest: Record<string, unknown> } {
  const { api_key, customer_token, token, auth_type, ...rest } = request;
  const normalizedAuthType = auth_type === "customer_token"
    ? "customer_token"
    : "api_key";
  return {
    creds: {
      authType: normalizedAuthType,
      apiKey: typeof api_key === "string" ? api_key : undefined,
      customerToken: typeof customer_token === "string"
        ? customer_token
        : undefined,
      runnerToken: typeof token === "string" ? token : undefined,
    },
    sandboxRequest: { ...rest, auth_type: normalizedAuthType },
  };
}

/** Make a short-lived bearer credential authoritative on an outbound call. */
export function stripAndInjectBearer(
  headers: Record<string, string>,
  token: string | undefined,
): Record<string, string> {
  for (const key of Object.keys(headers)) {
    if (AUTH_HEADER_NAMES.includes(key.toLowerCase())) delete headers[key];
  }
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

/**
 * The auth header the sandbox is NOT allowed to assemble. Customer token (PAYGO)
 * takes precedence over the dev API key — mirrors the old in-sandbox
 * BridgedLLMClient._auth_headers, but host-side. Empty when no credential is
 * present (e.g. a dev run with ambient auth).
 */
export function authHeader(creds: Credentials): Record<string, string> {
  if (creds.authType === "customer_token" && creds.customerToken) {
    return { Authorization: `Bearer ${creds.customerToken}` };
  }
  if (creds.apiKey) {
    return { "X-ModelRelay-Api-Key": creds.apiKey };
  }
  return {};
}

/**
 * Make host auth authoritative on an outbound header set: drop any auth header
 * the sandbox tried to set (any case), then inject the real one. Mutates and
 * returns `headers`. Whatever the sandbox sent for auth is irrelevant after this.
 */
export function stripAndInjectAuth(
  headers: Record<string, string>,
  creds: Credentials,
): Record<string, string> {
  for (const key of Object.keys(headers)) {
    if (AUTH_HEADER_NAMES.includes(key.toLowerCase())) delete headers[key];
  }
  Object.assign(headers, authHeader(creds));
  return headers;
}
