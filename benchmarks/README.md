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

## Pinned OOLONG-Pairs data

Materialize Appendix D.1's 20 OOLONG-Pairs questions against the paper-matched
32K-token context window at row 900:

```bash
uv run python -m benchmarks materialize-oolong-pairs \
  --output benchmarks/.data/oolong-pairs-32k-v1
```

The materializer verifies rows 900–943, their exact task metadata, and both
labeled and unlabeled context hashes. It uses labels only to derive answer keys,
cross-checks pairwise enumeration against cardinality/intersection counts, and
writes the original unlabeled row 900 context for model input. Generated task,
context, and provenance files are gitignored and never checked in.
The `oolong_pairs_f1` scorer parses, normalizes, and deduplicates ID pairs before
computing set precision, recall, and F1.

## RLM paper suite

`manifests/rlm-paper-v1.json` pins the target paper revision and names the
paper's S-NIAH, BrowseComp-Plus, OOLONG, OOLONG-Pairs, and LongBench-v2 CodeQA
task families. TAG-Bench is tracked separately in the same manifest as a
Droste-specific follow-on rather than being presented as part of the paper.

The OOLONG 131K `trec_coarse` validation slice is `ready` and has published results: immutable artifacts, a price-snapshot provenance record, and regenerated reports live under `results/oolong-trec-coarse-131k-2026-07-17/` ([#81](https://github.com/tensor-systems/droste/issues/81)).
OOLONG-Pairs is `ready` and has published results: immutable artifacts,
dataset and materializer provenance, and regenerated reports live under
[`results/oolong-pairs-32k-2026-07-17/`](results/oolong-pairs-32k-2026-07-17/).
Its three arms compare direct `gpt-5.6-sol`, direct `gpt-5.6-terra`, and Droste
with a `gpt-5.6-terra` root and `gpt-5.6-luna` subcalls.
The other datasets remain `planned`. A planned benchmark cannot be run: it has
no task path, and a blocked executor cannot silently degrade to a fixture or
another provider. Dataset adapters promote each benchmark to `ready` only
after its public source revision, split, integrity checks, and scorer are
pinned.

## Live runs

The checked-in manifest pins public live configurations (models, reasoning
efforts, budgets, concurrency) for both sets of OOLONG arms. Materializing or
validating the suite makes no model calls. Live runs require a new output
directory, refuse to overwrite artifacts, snapshot the endpoint's public price
table, and reject additions if that snapshot changes. The OOLONG report
regenerates offline from the committed artifacts:

```bash
uv run python -m benchmarks report benchmarks/manifests/rlm-paper-v0.5.0-oolong.json benchmarks/results/oolong-trec-coarse-131k-2026-07-17/artifacts --json /tmp/regen-check.json --markdown /tmp/regen-check.md
```

The OOLONG-Pairs report regenerates from the committed artifacts with all 20
task IDs selected:

```bash
task_args=()
for task_id in {1..20}; do task_args+=(--task-id "$task_id"); done
uv run python -m benchmarks report \
  benchmarks/manifests/rlm-paper-v0.3.0-oolong-pairs.json \
  benchmarks/results/oolong-pairs-32k-2026-07-17/artifacts \
  "${task_args[@]}" \
  --json /tmp/oolong-pairs-report.json \
  --markdown /tmp/oolong-pairs-report.md
cmp benchmarks/results/oolong-pairs-32k-2026-07-17/report.json \
  /tmp/oolong-pairs-report.json
cmp benchmarks/results/oolong-pairs-32k-2026-07-17/report.md \
  /tmp/oolong-pairs-report.md
```

These result-specific manifests are exact snapshots of the configurations that
produced the immutable artifacts. The current `rlm-paper-v1.json` manifest is
the additive union of both live benchmark configurations and therefore has a
different SHA-256; reports continue to reject artifacts whose recorded suite
version or manifest hash differs from the selected snapshot.

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

## Structured semantic guidance: exact replay

`llm_batch_json` semantic evidence includes the validator object's identity. If generated
benchmark code recreates a validator after an incomplete batch, the new call cannot clear the
old unresolved request even when its prompts and schema look identical. Repeated outer-loop
cell rewrites then accumulate unresolved requests until the policy reports that exact replay
cannot fit the remaining subcall budget. See
[droste#167](https://github.com/tensor-systems/droste/issues/167) for the engine-level
mechanism.

Benchmark-specific guidance that uses a validator must therefore put all one-time request
construction behind a persistent-globals guard, including chunks, prompts, contexts, schema,
validator, result slots, and attempt counters. It must handle incomplete results with a bounded
in-cell loop that replays the complete exact request using those same objects. Never rebuild the
validator and never retry only the failing subset as a new structured batch. State the full
worst-case arithmetic, including internal repair calls, and keep it below the arm's subcall
limit. If the bounded replay still fails, retain it as a typed benchmark failure rather than
aggregating partial values.
