# Scaffold manifest v1

A scaffold manifest is the content-addressed identity of everything that can
materially change a Droste rollout. It lets a trainer reject a checkpoint/run
mismatch before the first model request and join trainer-owned reward metadata
without putting task content in the engine trace.

`ScaffoldManifest.manifest_id` is `sha256:` plus the lowercase SHA-256 of the
UTF-8 JSON returned by `as_dict()`. JSON is serialized with sorted object keys,
compact separators, Unicode preserved, finite numbers only, and no `id` field.
Arrays retain order. The optional wire `id` is a claim over that content and is
verified when parsed.

## Closed schema

The top-level and owned nested objects are closed: unknown and missing fields,
wrong scalar types, invalid digests, duplicate/unsorted globals, and unsupported
schema versions fail. `root_sampling` and `subcall_sampling` are deliberately
open JSON objects because their provider-specific keys are themselves resolved
inference facts.

```json
{
  "schema_version": 1,
  "engine": {"version": "0.10.6", "source_revision": null},
  "abis": {
    "kernel": 1, "capability": 1, "trace": 1,
    "prompt_pack": 1, "provider": 3, "runner": 3
  },
  "prompt_pack": {
    "id": "droste.generic.full", "revision": "1.0.2",
    "profile": "full", "content_hash": "sha256:<64 lowercase hex>"
  },
  "capabilities": {
    "manifest_hash": "sha256:<64 lowercase hex>",
    "model_visible_globals": ["answer", "context", "llm_query"]
  },
  "contracts": {
    "terminal": "answer-ready-v1",
    "subcall_identity": "capability-call-parent-v1",
    "templates": {
      "refinement": "sha256:<64 lowercase hex>",
      "missing_code_repair": "sha256:<64 lowercase hex>",
      "error_repair": "sha256:<64 lowercase hex>",
      "extract_system": "sha256:<64 lowercase hex>",
      "extract_user": "sha256:<64 lowercase hex>"
    },
    "overrides": {
      "system_prompt": null, "system_prompt_additions": null,
      "user_prompt": null, "refinement_prompt": null
    }
  },
  "inference": {
    "root": {"id": "model", "revision": null},
    "subcall": {"id": "model", "revision": null},
    "root_sampling": {}, "subcall_sampling": {},
    "output_limits": {"root_tokens": 4096, "subcall_tokens": 2048},
    "concurrency": 1, "seed": null
  },
  "budget": {
    "tokens": 500000, "subcalls": 50, "depth": 1, "wall_ms": 300000,
    "root_output_tokens": 4096, "subcall_output_tokens": 2048
  },
  "sandbox": {
    "output_chars": 25000, "execution_timeout_ms": 5000,
    "capture_output_chars": 50000
  },
  "parent_child": {
    "trace_depth": "root-zero-child-increment",
    "identity": "capability-call-parent-v1"
  }
}
```

Model IDs are either non-empty strings or explicit `null` when the host keeps
the identity opaque. A revision cannot accompany a null model ID. Source and
model revisions are never guessed by the engine; the host supplies them through
`RolloutConfiguration`.

## Compatibility and storage

`ScaffoldRequirements` accepts an exact `manifest_id`, a partial nested
`required` object, or both. `require_scaffold_compatibility()` returns normally
or raises `ScaffoldCompatibilityError` containing typed path/expected/actual
mismatches. `run_rlm` performs this check before its first model call when
`RLMConfig.checkpoint_requirements` is set.

The full manifest is returned live as `RLMResult.scaffold_manifest` and in the
runner/CLI result. Durable default-retention terminal records store only its
content-free ID and schema version. Task, verifier, reward, split, and dataset
metadata remain trainer-owned; join them externally with
`OutcomeJoinKey(run_id, scaffold_manifest_id)`.
