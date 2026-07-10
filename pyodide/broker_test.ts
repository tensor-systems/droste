// Tests for the A′ credential broker. Pure functions — no Pyodide/network.
// Run: deno test --allow-none broker_test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import {
  authHeader,
  type Credentials,
  isModelRelayResponsesCall,
  splitCredentials,
  stripAndInjectAuth,
} from "./broker.ts";

Deno.test("isModelRelayResponsesCall: only POST https://api.modelrelay.ai/api/v1/responses qualifies", () => {
  // The one legitimate LLM-transport call.
  assert(
    isModelRelayResponsesCall(
      "POST",
      "https://api.modelrelay.ai/api/v1/responses",
    ),
  );
  assert(
    isModelRelayResponsesCall(
      "post",
      "https://api.modelrelay.ai/api/v1/responses",
    ),
  ); // case-insensitive method
  // Other ModelRelay endpoints must NOT get the credential (no general-purpose token).
  assert(
    !isModelRelayResponsesCall(
      "POST",
      "https://api.modelrelay.ai/api/v1/account/card-verification/confirm",
    ),
  );
  assert(
    !isModelRelayResponsesCall(
      "GET",
      "https://api.modelrelay.ai/api/v1/responses",
    ),
  ); // wrong method
  assert(
    !isModelRelayResponsesCall(
      "POST",
      "https://api.modelrelay.ai/api/v1/responses/extra",
    ),
  ); // not exact path
  // Plaintext, other hosts, substring/lookalike, subdomain — all rejected.
  assert(
    !isModelRelayResponsesCall(
      "POST",
      "http://api.modelrelay.ai/api/v1/responses",
    ),
  );
  assert(
    !isModelRelayResponsesCall(
      "POST",
      "https://evil.com/?x=api.modelrelay.ai/api/v1/responses",
    ),
  );
  assert(
    !isModelRelayResponsesCall(
      "POST",
      "https://api.modelrelay.ai.evil.com/api/v1/responses",
    ),
  );
  assert(
    !isModelRelayResponsesCall(
      "POST",
      "https://sub.api.modelrelay.ai/api/v1/responses",
    ),
  );
  // Garbage doesn't throw.
  assert(!isModelRelayResponsesCall("POST", "not a url"));
  assert(!isModelRelayResponsesCall("POST", ""));
});

Deno.test("isModelRelayResponsesCall: the batch endpoint is scoped in too, but only exactly", () => {
  assert(
    isModelRelayResponsesCall(
      "POST",
      "https://api.modelrelay.ai/api/v1/responses/batch",
    ),
  );
  assert(
    isModelRelayResponsesCall(
      "post",
      "https://api.modelrelay.ai/api/v1/responses/batch",
    ),
  );
  // Near-misses of the batch path must NOT qualify — still exact-match, not prefix.
  assert(
    !isModelRelayResponsesCall(
      "GET",
      "https://api.modelrelay.ai/api/v1/responses/batch",
    ),
  );
  assert(
    !isModelRelayResponsesCall(
      "POST",
      "https://api.modelrelay.ai/api/v1/responses/batch/extra",
    ),
  );
  assert(
    !isModelRelayResponsesCall(
      "POST",
      "https://api.modelrelay.ai/api/v1/responses/batches",
    ),
  );
  assert(
    !isModelRelayResponsesCall(
      "POST",
      "http://api.modelrelay.ai/api/v1/responses/batch",
    ),
  );
  assert(
    !isModelRelayResponsesCall(
      "POST",
      "https://evil.com/?x=api.modelrelay.ai/api/v1/responses/batch",
    ),
  );
});

Deno.test("splitCredentials removes secrets and preserves normalized auth type", () => {
  const req = {
    question: "who texts me most?",
    db_path: "/data/sample.db",
    api_key: "mr_sk_secret",
    customer_token: "ct_secret",
    auth_type: "customer_token",
    root_model: "gemini-3.5-flash",
  };
  const { creds, sandboxRequest } = splitCredentials(req);
  // The sandbox request keeps the real work and NONE of the credentials.
  assertEquals(sandboxRequest, {
    question: "who texts me most?",
    db_path: "/data/sample.db",
    root_model: "gemini-3.5-flash",
    auth_type: "customer_token",
  });
  assert(!("api_key" in sandboxRequest), "api_key must not reach the sandbox");
  assert(
    !("customer_token" in sandboxRequest),
    "customer_token must not reach the sandbox",
  );
  assertEquals(sandboxRequest.auth_type, "customer_token");
  // Held host-side.
  assertEquals(creds, {
    authType: "customer_token",
    apiKey: "mr_sk_secret",
    customerToken: "ct_secret",
  });
});

Deno.test("splitCredentials tolerates a request with no credentials", () => {
  const { creds, sandboxRequest } = splitCredentials({ question: "q" });
  assertEquals(sandboxRequest, { question: "q", auth_type: "api_key" });
  assertEquals(creds, {
    authType: "api_key",
    apiKey: undefined,
    customerToken: undefined,
  });
});

Deno.test("splitCredentials normalizes unknown auth types to api_key", () => {
  const { creds, sandboxRequest } = splitCredentials({
    question: "q",
    auth_type: "unexpected",
  });
  assertEquals(creds.authType, "api_key");
  assertEquals(sandboxRequest.auth_type, "api_key");
});

Deno.test("authHeader: customer token takes precedence over api key", () => {
  const creds: Credentials = {
    authType: "customer_token",
    apiKey: "mr_sk_x",
    customerToken: "ct_y",
  };
  assertEquals(authHeader(creds), { Authorization: "Bearer ct_y" });
});

Deno.test("authHeader: api key path uses the X-ModelRelay-Api-Key header, never bearer", () => {
  const creds: Credentials = { authType: "api_key", apiKey: "mr_sk_x" };
  assertEquals(authHeader(creds), { "X-ModelRelay-Api-Key": "mr_sk_x" });
});

Deno.test("authHeader: customer_token auth_type but token missing falls back to api key", () => {
  const creds: Credentials = { authType: "customer_token", apiKey: "mr_sk_x" };
  assertEquals(authHeader(creds), { "X-ModelRelay-Api-Key": "mr_sk_x" });
});

Deno.test("authHeader: empty when no credential present (ambient-auth dev run)", () => {
  assertEquals(authHeader({ authType: "api_key" }), {});
});

Deno.test("stripAndInjectAuth: host auth overrides whatever the sandbox tried to set", () => {
  // The sandbox attempts to smuggle its own auth (any case) + a legit content type.
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "authorization": "Bearer attacker_controlled",
    "X-ModelRelay-Api-Key": "sandbox_guess",
  };
  const out = stripAndInjectAuth(headers, {
    authType: "customer_token",
    customerToken: "ct_real",
  });
  assertEquals(out, {
    "Content-Type": "application/json",
    Authorization: "Bearer ct_real",
  });
  // No trace of the sandbox's attempts.
  assert(out["authorization"] === undefined);
  assert(!Object.values(out).includes("attacker_controlled"));
  assert(!Object.values(out).includes("sandbox_guess"));
});

Deno.test("stripAndInjectAuth: no credential means no auth header leaks through", () => {
  const headers: Record<string, string> = {
    "X-ModelRelay-Api-Key": "sandbox_guess",
  };
  const out = stripAndInjectAuth(headers, { authType: "api_key" });
  assertEquals(out, {});
});
