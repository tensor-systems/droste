// A′ + BridgedLLMClient.batch_responses_typed E2E (droste#16 fix): proves the
// batch path works end-to-end under a REAL Pyodide interpreter with an ASYNC
// fake host_fetch — broker_integration_test.ts's existing case uses a
// sync-string host_fetch, which never exercises the `run_sync(awaitable)`
// branch in `BridgedLLMClient._post` that the real Deno relay always takes
// (host_fetch there is `async (m,u,h,b) => ...`). This test:
//   (1) wires host_fetch exactly like relay.ts (isModelRelayResponsesCall +
//       stripAndInjectAuth over the /responses/batch path — the credential
//       scoping fix this PR makes), with the sandbox-side BridgedLLMClient
//       holding NO real credential (mirrors the stripped sandboxRequest),
//   2) scripts a /responses/batch-shaped reply with one success + one error
//      item + usage, and asserts BridgedLLMClient.batch_responses_typed
//      returns the typed BatchResponse llm_orchestrator.py actually reads
//      (.usage.total_input_tokens/.total_output_tokens,
//      .results[i].id/.status/.response/.error.code).
//
// Requires a sibling cozy checkout (RCL_RLM_SRC env override) for
// rcl_rlm.modelrelay.BatchResponse — skipped if absent, matching
// db_service_integration_test.ts's convention.
//
// Run: deno test --allow-read --allow-env --allow-ffi broker_batch_integration_test.ts
import { assertEquals } from "jsr:@std/assert@1";
import { loadPyodide } from "npm:pyodide@0.29.4";
import {
  isModelRelayResponsesCall,
  splitCredentials,
  stripAndInjectAuth,
} from "./broker.ts";

const DROSTE_SRC = new URL("../src", import.meta.url).pathname;
const RUNTIME_DIR = new URL(".", import.meta.url).pathname;
const RCL_RLM_SRC = Deno.env.get("RCL_RLM_SRC") ??
  new URL("../../cozy/tools/rcl-rlm/src", import.meta.url).pathname;

let rclRlmAvailable = false;
try {
  rclRlmAvailable = (await Deno.stat(RCL_RLM_SRC)).isDirectory;
} catch {
  rclRlmAvailable = false;
}

Deno.test({
  name:
    "A′ batch: BridgedLLMClient.batch_responses_typed over an async host_fetch, real broker scoping",
  ignore: !rclRlmAvailable,
  fn: async () => {
    const request = {
      question: "q",
      api_key: "mr_sk_REAL",
      auth_type: "api_key",
    };
    const { creds, sandboxRequest } = splitCredentials(request);
    assertEquals(
      sandboxRequest.api_key,
      undefined,
      "sandbox-side request must not carry the real key",
    );

    let sentHeaders: Record<string, string> = {};
    let sentUrl = "";
    // Not declared `async` (no internal `await`) but still returns a Promise —
    // this is the actual shape relay.ts's host_fetch has, and the point of this
    // test is to exercise BridgedLLMClient._post's `run_sync(awaitable)` branch.
    const hostFetch = (m: string, u: string, h: string, _b: string) => {
      const headers = JSON.parse(h);
      if (isModelRelayResponsesCall(m, u)) stripAndInjectAuth(headers, creds); // relay.ts's A′ path
      sentHeaders = headers;
      sentUrl = u;
      return Promise.resolve(JSON.stringify({
        id: "batch_1",
        results: [
          {
            id: "subcall_0",
            status: "success",
            response: {
              output: [{
                type: "message",
                role: "assistant",
                content: [{ type: "text", text: "ok" }],
              }],
            },
          },
          {
            id: "subcall_1",
            status: "error",
            error: {
              status: 429,
              message: "rate limited",
              code: "RATE_LIMITED",
            },
          },
        ],
        usage: {
          total_input_tokens: 42,
          total_output_tokens: 7,
          total_requests: 2,
          successful_requests: 1,
          failed_requests: 1,
        },
      }));
    };

    const py = await loadPyodide({ stdout: () => {}, stderr: () => {} });
    // rcl_rlm's __init__ eagerly imports message_database, which imports sqlite3.
    await py.loadPackage("sqlite3", {
      messageCallback: () => {},
      errorCallback: () => {},
    });
    py.mountNodeFS("/droste_src", DROSTE_SRC);
    py.mountNodeFS("/rcl_rlm_src", RCL_RLM_SRC);
    py.mountNodeFS("/pyodide_runtime", RUNTIME_DIR);
    await py.runPythonAsync(`
import sys
for p in ("/droste_src", "/rcl_rlm_src", "/pyodide_runtime"):
    if p not in sys.path:
        sys.path.insert(0, p)`);
    py.globals.set("host_fetch", hostFetch);
    // The sandbox-side client is built with NO real credential — exactly like
    // run_for_host_pyodide constructs it from the credential-stripped request.
    // If it succeeds, the auth came from the host's stripAndInjectAuth, not here.
    const out = await py.runPythonAsync(`
import json
from pyodide_runtime import BridgedLLMClient
client = BridgedLLMClient(host_fetch, api_key=None, customer_token=None)
requests = [
    {"id": "subcall_0", "model": "test-model", "input": [], "max_output_tokens": 512, "temperature": 0.0},
    {"id": "subcall_1", "model": "test-model", "input": [], "max_output_tokens": 512, "temperature": 0.0},
]
payload = client.batch_responses_typed(requests, options={"max_concurrent": 2})
json.dumps({
    "usage": {
        "total_input_tokens": payload.usage.total_input_tokens,
        "total_output_tokens": payload.usage.total_output_tokens,
    },
    "results": [
        {
            "id": r.id,
            "status": r.status,
            "has_response": r.response is not None,
            "error_code": r.error.code if r.error else None,
        }
        for r in payload.results
    ],
})
`);
    const parsed = JSON.parse(out);
    assertEquals(parsed.usage.total_input_tokens, 42);
    assertEquals(parsed.usage.total_output_tokens, 7);
    assertEquals(parsed.results, [
      {
        id: "subcall_0",
        status: "success",
        has_response: true,
        error_code: null,
      },
      {
        id: "subcall_1",
        status: "error",
        has_response: false,
        error_code: "RATE_LIMITED",
      },
    ]);

    // The real credential scoping fix this PR makes: the batch URL must have
    // gotten the injected host credential, not a blank sandbox-side header.
    assertEquals(sentUrl, "https://api.modelrelay.ai/api/v1/responses/batch");
    assertEquals(sentHeaders["X-ModelRelay-Api-Key"], "mr_sk_REAL");
  },
});
