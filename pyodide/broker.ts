// broker.ts — host-side credential handling for the A′ sandbox split (#3).
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

/**
 * Pull credentials out of the request so they never become a sandbox global.
 * Returns the held creds and the request stripped of every credential field.
 */
export function splitCredentials(
  request: Record<string, unknown>,
): { creds: Credentials; sandboxRequest: Record<string, unknown> } {
  const { api_key, customer_token, auth_type, ...rest } = request;
  return {
    creds: {
      authType: typeof auth_type === "string" ? auth_type : "api_key",
      apiKey: typeof api_key === "string" ? api_key : undefined,
      customerToken: typeof customer_token === "string"
        ? customer_token
        : undefined,
    },
    sandboxRequest: rest,
  };
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
