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

Repeat `--task-id <id>` to report a selected gate. The artifact directory must
then contain exactly every runnable arm for the selected tasks and no other run
artifacts. Without `--task-id`, reporting retains strict full-suite coverage.

Validate a manifest without executing it:

```bash
uv run python -m benchmarks validate benchmarks/manifests/rlm-paper-v1.json
```

## Pinned OOLONG data

Materialize the public 50-task, 131K-token `trec_coarse` validation slice used
by the first OOLONG arm:

```bash
uv run python -m benchmarks materialize-oolong \
  --output benchmarks/.data/oolong-trec-coarse-131k-v1
```

The materializer downloads rows 1050–1099 from the pinned public dataset
revision, verifies every task id and the SHA-256 of both shared contexts, and
writes a deterministic task file. The generated data is gitignored; it is not
part of the published wheel or source tree. Existing files are never
overwritten.

The `oolong_official` scorer implements the benchmark's published synth
evaluation: exact parsed answers, comparison-phrase matching, date matching,
and graded numeric credit of `0.75 ** absolute_error`.

## RLM paper suite

`manifests/rlm-paper-v1.json` pins the target paper revision and names the
paper's S-NIAH, BrowseComp-Plus, OOLONG, OOLONG-Pairs, and LongBench-v2 CodeQA
task families. TAG-Bench is tracked separately in the same manifest as a
Droste-specific follow-on rather than being presented as part of the paper.

The OOLONG 131K `trec_coarse` validation slice is `ready` and has published results: immutable artifacts, a price-snapshot provenance record, and regenerated reports live under `results/oolong-trec-coarse-131k-2026-07-17/` ([#81](https://github.com/tensor-systems/droste/issues/81)).
The other datasets remain `planned`. A planned benchmark cannot be run: it has
no task path, and a blocked executor cannot silently degrade to a fixture or
another provider. Dataset adapters promote each benchmark to `ready` only
after its public source revision, split, integrity checks, and scorer are
pinned.

## Live runs

The checked-in manifest pins one public live configuration (models, reasoning efforts, budgets, concurrency) for the OOLONG arms. Live runs require a new output directory, refuse to overwrite artifacts, snapshot the endpoint's public price table, and reject additions if that snapshot changes. Published reports regenerate offline from the committed artifacts:

```bash
uv run python -m benchmarks report benchmarks/manifests/rlm-paper-v1.json benchmarks/results/oolong-trec-coarse-131k-2026-07-17/artifacts --json /tmp/regen-check.json --markdown /tmp/regen-check.md
```

When enabling a public configuration, include `--max-cost-microusd <amount>` in
the run command. That optional integer micro-USD cap includes existing
artifacts in the output directory. Once actual cumulative cost reaches the cap,
or an observed same-arm cost projects the next run past it, execution stops
before dispatching more work without inventing a skipped artifact (artifact v1
has no skipped status).

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
