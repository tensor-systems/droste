# Reproducible benchmarks

This directory is repository-local tooling, not part of the published `droste`
wheel. It defines the versioned contract for benchmark manifests, per-task run
artifacts, deterministic scoring, and aggregate reports.

## Zero-cost smoke run

From a clean checkout:

```bash
output="$(mktemp -d)/droste-benchmark-smoke"
uv run python -m benchmarks smoke --output "$output"
```

The smoke suite makes no network calls. It runs checked-in predictions through
the exact-match, numeric, and token-F1 scorers, writes one immutable JSON
artifact per task and arm, and prints the report derived from those artifacts.
Reusing the same output directory fails rather than overwriting evidence.

Render reports independently from an artifact directory:

```bash
uv run python -m benchmarks report \
  benchmarks/manifests/smoke-v1.json \
  "$output" \
  --markdown /tmp/droste-smoke.md \
  --json /tmp/droste-smoke.json
```

Validate a manifest without executing it:

```bash
uv run python -m benchmarks validate benchmarks/manifests/rlm-paper-v1.json
```

## RLM paper suite

`manifests/rlm-paper-v1.json` pins the target paper revision and names the
paper's S-NIAH, BrowseComp-Plus, OOLONG, OOLONG-Pairs, and LongBench-v2 CodeQA
task families. TAG-Bench is tracked separately in the same manifest as a
Droste-specific follow-on rather than being presented as part of the paper.

The datasets and live executors remain `planned`. A planned benchmark cannot be
run: it has no task path, and a blocked executor cannot silently degrade to a
fixture or another provider. Dataset adapters will promote each benchmark to
`ready` only after its source revision, split, scorer, and license are pinned.

## Live-run gate

Do not publish live OpenAI benchmark results until
[ModelRelay #1686](https://github.com/tensor-systems/modelrelay/issues/1686) is
merged, released, deployed, and verified end to end. Before that fix,
`reasoning_effort=none` can be silently dropped, so reported OpenAI cost and
latency would not match the manifest.

Infrastructure work—dataset adapters, scorers, artifact plumbing, and report
generation—can proceed while the gate is closed. The manifest records the gate
as data and all live arms use the `blocked` executor kind, so the runner fails
fast instead of producing mislabeled results.

## Artifact rules

- Every artifact records the exact manifest SHA-256.
- Artifact identity is benchmark + arm + task and must be unique.
- Token counts separate root and subcall input/output usage.
- Monetary values use integer micro-US dollars; floats are not used for money.
- Failures and timeouts are artifacts with typed status and an error, not
  discarded tasks.
- Reports reject mixed suite versions or artifacts from a changed manifest.
- Aggregate rows are sorted, and reports are deterministic from the same raw
  artifacts.

The machine-readable contracts are `schemas/suite-manifest-v1.schema.json` and
`schemas/run-artifact-v1.schema.json`. Runtime validation in `models.py`
remains the fail-fast source of truth used by the tooling.
