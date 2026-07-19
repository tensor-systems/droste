// Tests for the A′ credential broker. Pure functions — no Pyodide/network.
// Run: deno test --allow-none broker_test.ts
import { assert, assertEquals, assertThrows } from "jsr:@std/assert@1";
import {
  authHeader,
  type Credentials,
  isModelRelayResponsesCall,
  isRunnerCallback,
  isRunnerCallbackJSONContentType,
  redactHeldSecrets,
  redactRelayErrorText,
  shouldReturnRunnerCallbackFailureBody,
  splitCredentials,
  stripAndInjectAuth,
  stripAndInjectBearer,
  validateAndRedactRunnerCallbackFailureBody,
} from "../src/droste/substrates/_relay/broker.ts";

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
    subcall_concurrency: 3,
    token: "runner_secret",
  };
  const { creds, sandboxRequest } = splitCredentials(req);
  // The sandbox request keeps the real work and NONE of the credentials.
  assertEquals(sandboxRequest, {
    question: "who texts me most?",
    db_path: "/data/sample.db",
    root_model: "gemini-3.5-flash",
    subcall_concurrency: 3,
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
    runnerToken: "runner_secret",
  });
});

Deno.test("splitCredentials tolerates a request with no credentials", () => {
  const { creds, sandboxRequest } = splitCredentials({ question: "q" });
  assertEquals(sandboxRequest, { question: "q", auth_type: "api_key" });
  assertEquals(creds, {
    authType: "api_key",
    apiKey: undefined,
    customerToken: undefined,
    runnerToken: undefined,
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

Deno.test("runner callback token is stripped and scoped to exact endpoints", () => {
  const endpoint = "https://internal.example/api/v1/rlm/subcall/batch";
  assert(isRunnerCallback("POST", endpoint, [endpoint]));
  assert(!isRunnerCallback("GET", endpoint, [endpoint]));
  assert(!isRunnerCallback("POST", endpoint + "/extra", [endpoint]));

  const headers = stripAndInjectBearer(
    { authorization: "Bearer sandbox", "Content-Type": "application/json" },
    "runner_real",
  );
  assertEquals(headers, {
    Authorization: "Bearer runner_real",
    "Content-Type": "application/json",
  });
});

Deno.test("typed JSON failures are values only for exact model callbacks", () => {
  const root = "https://internal.example/api/v1/rlm/root";
  const subcall = "https://internal.example/api/v1/rlm/subcall";
  const batch = "https://internal.example/api/v1/rlm/subcall/batch";
  const callbacks = [root, subcall, batch];

  const validContentTypes = [
    "application/json",
    "application/problem+json",
    "Application/Vnd.ModelRelay.Error+Json; charset=UTF-8",
    'application/json; charset="utf-8"',
    'application/json; CHARSET = "UTF-8"',
  ];
  for (const contentType of validContentTypes) {
    assert(isRunnerCallbackJSONContentType(contentType));
    for (const endpoint of callbacks) {
      assert(
        shouldReturnRunnerCallbackFailureBody(
          "post",
          endpoint,
          contentType,
          callbacks,
        ),
      );
    }
  }

  for (
    const contentType of [
      "text/json",
      "text/plain+json",
      "application/+json",
      "application/problem json",
      "application/problem@json",
      "application/json garbage",
      "application/json;",
      "application/json; garbage",
      "application/json; charset",
      'application/json; charset="unterminated',
      "application/json; charset=iso-8859-1",
      'application/json; charset="iso-8859-1"',
      "application/json; charset=utf-8; charset=utf-8",
      "application/json; charset=utf-8; profile=callback",
      "application/json; profile=callback",
      "application/json, application/json",
      "application/json\n",
      "application/jsoñ",
      "",
    ]
  ) {
    assert(!isRunnerCallbackJSONContentType(contentType), contentType);
    assert(
      !shouldReturnRunnerCallbackFailureBody(
        "POST",
        subcall,
        contentType,
        callbacks,
      ),
    );
  }

  assert(
    !shouldReturnRunnerCallbackFailureBody(
      "POST",
      "https://api.modelrelay.ai/api/v1/responses",
      "application/json",
      callbacks,
    ),
  );
  assert(
    !shouldReturnRunnerCallbackFailureBody(
      "POST",
      "https://internal.example/api/v1/rlm/data-source",
      "application/json",
      callbacks,
    ),
  );
  assert(
    !shouldReturnRunnerCallbackFailureBody(
      "POST",
      subcall,
      "text/plain",
      callbacks,
    ),
  );
  assert(
    !shouldReturnRunnerCallbackFailureBody(
      "GET",
      subcall,
      "application/json",
      callbacks,
    ),
  );
  assert(
    !shouldReturnRunnerCallbackFailureBody(
      "POST",
      subcall + "/extra",
      "application/json",
      callbacks,
    ),
  );
});

Deno.test("canonical typed callback failures preserve raw counters and redact strings", () => {
  const raw = String.raw`{
    "error":"api_error",
    "message":"runner\/secret cl\u00e9\ud83d\ude00secret",
    "score":0.5,
    "usage":{
      "input_tokens":9999999999999999999999999999999999999999,
      "output_tokens":3.0,
      "total_tokens":9223372036854775807
    }
  }`;
  const safe = validateAndRedactRunnerCallbackFailureBody(raw, [
    "runner/secret",
    "clé😀secret",
  ]);
  assert(
    safe.includes("9999999999999999999999999999999999999999"),
  );
  assert(safe.includes("9223372036854775807"));
  assert(safe.includes('"score":0.5'));
  assert(safe.includes('"output_tokens":3.0'));
  assert(!safe.includes("runner/secret"));
  assert(!safe.includes("clé😀secret"));
  const parsed = JSON.parse(safe);
  assertEquals(parsed.error, "api_error");
  assertEquals(parsed.message, "[redacted] [redacted]");
});

Deno.test("noncanonical callback envelopes fail closed before redacted output", () => {
  const held = "runner/secret";
  const huge = "9".repeat(5_000);
  const longFloat = `0.${"0".repeat(5_000)}1`;
  for (
    const raw of [
      '{"error":{"type":"ProviderError"},"usage":{}}',
      '{"error":"other_error","usage":{}}',
      '{"error":"api_error"}',
      '{"error":"api_error","usage":null}',
      '{"error":"api_error","usage":[]}',
      '[{"error":"api_error","usage":{}}]',
      '{"error":"api_error","error":"api_error","usage":{}}',
      String.raw`{"\u0065rror":"api_error","error":"api_error","usage":{}}`,
      '{"error":"api_error","usage":{"input_tokens":1,"input_tokens":2}}',
      '{"error":"api_error","usage":{},"score":1e10000}',
      `{"error":"api_error","usage":{},"score":${longFloat}}`,
      `{"error":"api_error","usage":{},"attempt":${huge}}`,
      `{"error":"api_error","usage":{"provider_counter":${huge}}}`,
      `{"error":"api_error","usage":{"input_tokens":[${huge}]}}`,
      `{"error":"api_error","usage":{},"late":"${held}","nested":${
        "[".repeat(300)
      }0${"]".repeat(300)}}`,
    ]
  ) {
    const error = assertThrows(() =>
      validateAndRedactRunnerCallbackFailureBody(raw, [held])
    );
    assert(error instanceof Error);
    assertEquals(
      error.message,
      "runner callback failure body is not valid JSON",
    );
    assert(!error.message.includes(held));
  }
});

Deno.test("redacted callback output must remain a canonical unique-key envelope", () => {
  for (
    const [raw, held] of [
      ['{"error":"api_error","usage":{}}', ["api_error"]],
      ['{"error":"api_error","usage":{}}', ["error"]],
      [
        '{"error":"api_error","usage":{},"first":1,"second":2}',
        ["first", "second"],
      ],
    ] as const
  ) {
    const error = assertThrows(() =>
      validateAndRedactRunnerCallbackFailureBody(raw, held)
    );
    assert(error instanceof Error);
    assertEquals(
      error.message,
      "runner callback failure body is not valid JSON",
    );
    for (const secret of held) assert(!error.message.includes(secret));
  }
});

Deno.test("relay HTTP diagnostics redact shaped and held secrets idempotently", () => {
  const held = "opaque-runner-secret";
  const input =
    `Authorization: Bearer sk-abc123456789 api_key=AIzaSyFAKEKEY ${held}`;
  const redacted = redactRelayErrorText(input, [held]);
  assert(!redacted.includes("sk-abc123456789"));
  assert(!redacted.includes("AIzaSyFAKEKEY"));
  assert(!redacted.includes(held));
  assert(redacted.includes("[redacted]"));
  assertEquals(redactRelayErrorText(redacted, [held]), redacted);
});

Deno.test("held-secret redaction is order-independent for overlapping values", () => {
  for (
    const held of [
      ["abc", "abcdef"],
      ["abcdef", "abc"],
      ["abc", "abcdef", "abc", ""],
    ]
  ) {
    assertEquals(redactRelayErrorText("abcdef", held), "[redacted]");
    assertEquals(
      redactHeldSecrets('{"literal":"abcdef"}', held),
      '{"literal":"[redacted]"}',
    );
    assertEquals(
      redactHeldSecrets(String.raw`{"escaped":"\u0061bcdef"}`, held),
      '{"escaped":"[redacted]"}',
    );
  }

  for (
    const held of [
      ["abc", "bcdef"],
      ["bcdef", "abc"],
      ["bcdef", "abc", "bcdef"],
    ]
  ) {
    // The two matches overlap in the original text. Redact their union so
    // neither a prefix nor a suffix fragment crosses the boundary.
    assertEquals(redactRelayErrorText("abcdef", held), "[redacted]");
    assertEquals(
      redactHeldSecrets(String.raw`{"escaped":"\u0061bcdef"}`, held),
      '{"escaped":"[redacted]"}',
    );
  }

  assertEquals(redactRelayErrorText("ababa", ["aba"]), "[redacted]");
  assert(
    !redactRelayErrorText("Bearer sk-abc123456789", ["redact"]).includes(
      "redact",
    ),
  );
});

Deno.test("held-secret redaction preserves raw int64 JSON and escaped credentials", () => {
  const held = "runner/é😀secret";
  const raw = String
    .raw`{"usage":{"total_tokens":9223372036854775807},"late":"runner\/\u00e9\ud83d\ude00secret"}`;
  const redacted = redactHeldSecrets(raw, [held]);
  assertEquals(
    redacted,
    '{"usage":{"total_tokens":9223372036854775807},"late":"[redacted]"}',
  );
});

Deno.test("numeric-looking credentials redact only JSON strings, never number lexemes", () => {
  const held = "9223372036854775807";
  const raw =
    '{"usage":{"total_tokens":9223372036854775807},"echo":"9223372036854775807"}';
  assertEquals(
    redactHeldSecrets(raw, [held]),
    '{"usage":{"total_tokens":9223372036854775807},"echo":"[redacted]"}',
  );
});

Deno.test("malformed callback JSON fails closed without echoing a held secret", () => {
  const held = "runner/secret";
  for (
    const raw of [
      String.raw`{"late":"runner\/secret"`,
      String.raw`{"late":"runner\/secret}`,
      String.raw`{"late":"runner\/secret\x"}`,
      String.raw`{"ok":true runner/secret}`,
      `{"nested":${"[".repeat(300)}0${"]".repeat(300)}}`,
    ]
  ) {
    const error = assertThrows(() => redactHeldSecrets(raw, [held]));
    assert(error instanceof Error);
    assertEquals(
      error.message,
      "runner callback failure body is not valid JSON",
    );
    assert(!error.message.includes(held));
  }
});
