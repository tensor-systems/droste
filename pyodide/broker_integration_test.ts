// A′ credential-boundary E2E: with a real Pyodide interpreter + a mock
// ModelRelay, prove the two security properties the split exists for:
//   (1) the credential is invisible to sandbox (LLM-authored) code, and
//   (2) host auth is authoritative — anything the sandbox sends is overridden.
// Replicates relay.ts's wiring (splitCredentials + host_fetch auth injection)
// without the DB path, so it needs no sqlite3.
//
// Run: deno test --allow-read --allow-env --allow-ffi broker_integration_test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { loadPyodide } from "npm:pyodide@0.29.4";
import {
  isModelRelayResponsesCall,
  splitCredentials,
  stripAndInjectAuth,
} from "./broker.ts";

Deno.test("A′: credential is invisible to sandbox code and host auth is authoritative", async () => {
  const request = {
    question: "who texts me most?",
    api_key: "mr_sk_SECRET",
    customer_token: "ct_SECRET",
    auth_type: "customer_token",
  };
  const { creds, sandboxRequest } = splitCredentials(request);

  const py = await loadPyodide({ stdout: () => {}, stderr: () => {} });

  // Capture what the host would actually send to ModelRelay.
  let sentHeaders: Record<string, string> = {};
  py.globals.set(
    "host_fetch",
    (m: string, u: string, h: string, _b: string): string => {
      const headers = JSON.parse(h);
      if (isModelRelayResponsesCall(m, u)) stripAndInjectAuth(headers, creds); // relay.ts A′ path
      sentHeaders = headers;
      return JSON.stringify({
        output: [{
          type: "message",
          role: "assistant",
          content: [{ type: "text", text: "ok" }],
        }],
      });
    },
  );

  // The sandbox sees no secret credential. It retains only the normalized,
  // nonsecret auth type so product adapters can apply the right routing contract.
  py.globals.set("request_json", JSON.stringify(sandboxRequest));

  // (1) LLM-authored code cannot find the credential anywhere it can reach.
  const leak = await py.runPythonAsync(`
import json
req = json.loads(request_json)
found = []
for k in ("api_key", "customer_token"):
    if k in req:
        found.append(k)
# Also scan every value for the secret substrings, in case of nesting.
blob = json.dumps(req)
for secret in ("mr_sk_SECRET", "ct_SECRET"):
    if secret in blob:
        found.append(secret)
",".join(found)
`);
  assertEquals(
    leak,
    "",
    "no credential field or secret value may be reachable from the sandbox",
  );
  const visibleAuthType = await py.runPythonAsync(`
import json
json.loads(request_json)["auth_type"]
`);
  assertEquals(visibleAuthType, "customer_token");

  // (2) Generated code sends its OWN bogus auth; the host must override it.
  await py.runPythonAsync(`
import json
bogus = json.dumps({"Content-Type": "application/json", "Authorization": "Bearer ATTACKER"})
host_fetch("POST", "https://api.modelrelay.ai/api/v1/responses", bogus, "{}")
`);
  assertEquals(
    sentHeaders["Authorization"],
    "Bearer ct_SECRET",
    "host injects the real customer token",
  );
  assert(
    !Object.values(sentHeaders).includes("Bearer ATTACKER"),
    "the sandbox-supplied auth header must be discarded",
  );
});

Deno.test("A′ legacy kill switch keeps the pre-split behavior (credential in the request)", () => {
  // In legacy mode relay.ts sets request_json to the FULL request; assert that
  // splitCredentials is the only thing standing between the sandbox and the token,
  // so the kill switch (which bypasses it) is a real, understood fallback.
  const request = { question: "q", api_key: "mr_sk_X", auth_type: "api_key" };
  const legacyGlobal = JSON.stringify(request); // BRIDGE_LEGACY path
  assert(
    legacyGlobal.includes("mr_sk_X"),
    "legacy path intentionally exposes the credential",
  );
  const { sandboxRequest } = splitCredentials(request);
  assert(
    !JSON.stringify(sandboxRequest).includes("mr_sk_X"),
    "A′ path does not",
  );
});
