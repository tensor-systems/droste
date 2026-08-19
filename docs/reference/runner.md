# Runner protocol v10

`python -m droste_runner` is a one-shot JSON worker for non-Python hosts. It
reads a request from `RLM_RUNNER_REQUEST_PATH` or its first argument and writes
one JSON response to stdout.

Every request must include `"protocol_version": 10`. Missing or mismatched
versions return a structured refusal before operation resolution, source
binding, credentials, or model dispatch.

## Operations

`operation` is `run` by default. `preflight` resolves the same prompt pack,
provider descriptors, rollout settings, budget, sandbox, and scaffold manifest
as a run without dispatching root or subcall inference. Dynamic providers may
still be acquired when their descriptors are needed, and are then closed.

A preflight response always has five fields:

```json
{
  "protocol_version": 10,
  "operation": "preflight",
  "status": "success",
  "preflight": {
    "scaffold_manifest": {},
    "manifest_id": "sha256:...",
    "compatible": true,
    "mismatches": []
  },
  "error": null
}
```

Checkpoint incompatibility returns the same outer shape with
`status="refusal"`, `preflight=null`, and a `scaffold_incompatible` error.

## Required run authorization

Run requests carry one exact seven-field budget object:

```json
{
  "tokens": 500000,
  "subcalls": 50,
  "depth": 1,
  "wall_ms": 300000,
  "max_iterations": 30,
  "root_output_tokens": 4096,
  "subcall_output_tokens": 2048
}
```

Unknown, missing, boolean, or out-of-range values fail before dispatch. Optional
rollout facts include model/source revisions, root and subcall sampling,
concurrency, seed, subcall input capacity, root reasoning effort, and scaffold
requirements. Conflicting duplicate evidence fails rather than being rewritten.

Native callbacks use `root_endpoint` and `subcall_endpoint`. The root client
uses Responses-style messages. Hosted subcalls may negotiate
`responses-stream/v2` NDJSON; reconstruction requires a terminal completion and
preserves its `stop_reason`.

## Responses and events

Run responses have one stable field set across success, refusal, and error. The
important result fields are `answer`, `answer_metadata`, `ready`, `extracted`,
`degradations`, typed errors, iteration and subcall counts, usage, prompt-pack
and scaffold identities, `stdout_chars`, `run_id`, and `run_record`.

Trajectory entries carry typed `execution_status` and `attempt_kind`; consumers
must not infer either from stdout text. Oversized stdout is rejected instead of
being silently truncated.

The native process runner can emit Trace ABI events to its configured sink. The
Deno relay uses a separate inherited event descriptor; see
[pyodide/README.md](../../pyodide/README.md). Protocol refusals happen before
run admission and emit no trace events.

Breaking request or response changes require a runner protocol bump. Current
migration guidance is in [UPGRADING.md](../../UPGRADING.md).
