# Scaffold manifest v3

A scaffold manifest is the content-addressed identity of the facts that can
materially change a run. It lets a host reject incompatible checkpoints before
inference and join external outcome metadata without putting task content in a
durable trace.

The closed manifest contains:

- engine version and source revision;
- kernel, capability, trace, prompt-pack, provider, and runner versions;
- prompt-pack identity and content hash;
- capability-manifest hash and generated globals;
- terminal, subcall-identity, template, and override contracts;
- root/subcall model identities, sampling, capacities, limits, concurrency,
  and seed;
- the complete budget and sandbox limits; and
- parent/child trace and call identity rules.

`manifest_id` is `sha256:` plus the SHA-256 of canonical UTF-8 JSON from
`as_dict()`. Object keys are sorted, separators are compact, Unicode is
preserved, numbers are finite, and array order is retained. A supplied wire
`id` is verified against the content.

New manifests use schema v3. `ScaffoldManifest.from_dict()` also reads stored
v1 and v2 manifests without rewriting their identity. Unknown fields, missing
fields, wrong scalar types, invalid digests, unsupported versions, and
duplicate or unsorted generated globals fail closed.

`ScaffoldRequirements` may specify a complete `manifest_id`, a partial nested
`required` object, or both. `require_scaffold_compatibility()` raises an ordered
`ScaffoldCompatibilityError` on mismatch. `preflight_rlm()` and runner
`operation="preflight"` use the same resolver as `run_rlm()` and return the
complete content-free identity without dispatching model calls.

The full manifest is returned in live results. Durable default retention stores
only its ID and schema version; task, reward, dataset, and split metadata remain
host-owned and join externally by run and manifest ID.
