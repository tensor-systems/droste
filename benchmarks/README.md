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

## Deterministic RULER S-NIAH data

Materialize the 50-task, 32,768-token noise S-NIAH configuration:

```bash
uv run python -m benchmarks materialize-sniah \
  --output benchmarks/.data/sniah-noise-words-32768-50-v1
```

This materializer performs no network or filesystem input. Top-level seed `42`
deterministically derives one recorded seed per task; each task draws an
adjective-noun key and value, repeats the fixed RULER noise sentence to the
largest count that fits the token budget, and inserts the needle at a uniformly
sampled noise-unit index. The generated `tasks.json` records the full context,
question and answer prefix, expected value, key/value, insertion index and
fractional depth, seed, token accounting, and provenance. Matching immutable
context files support the existing live runner. Existing files are never
overwritten.

The algorithm is an independent implementation of Hsieh et al. (2024),
[RULER: What's the Real Context Size of Your Long-Context Language
Models?](https://arxiv.org/abs/2404.06654), checked against NVIDIA/RULER commit
[`38da79d79519ef87aa46ae804f838e1eab7f86d7`](https://github.com/NVIDIA/RULER/tree/38da79d79519ef87aa46ae804f838e1eab7f86d7).
At that commit, `niah.py` has SHA-256
`e9cada0a7660d274fe73a1338a90a7087e17b630169f1aaf14a8d3221c6805b5`
and `constants.py` has SHA-256
`6296e901d495ec6200dc3f68993ea13d8282e3c0dbe1a8c47967f111105d1fde`.
The source repository is
[Apache-2.0 licensed](https://github.com/NVIDIA/RULER/blob/38da79d79519ef87aa46ae804f838e1eab7f86d7/LICENSE).
Droste fetches or redistributes no RULER corpus or generated examples; the
license check establishes the provenance of the reproduced published
algorithm and prompt methodology.

RULER accepts a caller-selected tokenizer and uses `wonderwords` adjective and
noun files. To make regeneration offline and dependency-free, generator v1
instead pins the in-repo word bank and `wordpunct-newline-v1` token counter.
The split names both choices. The RLM paper v3 specifies 50 S-NIAH tasks that
retrieve a phrase or number, but does not disambiguate its exact key/value
configuration. This arm therefore chooses word-pair keys and values as the
paper's phrase-shaped case. It reserves 128 output tokens and 256 model-template
tokens inside the 32,768-token budget.

The `exact_match` scorer accepts a normalized bare word-pair. The task includes
RULER's answer prefix in the live question to request that clean value; prose
around the value intentionally does not receive exact-match credit.

## RLM paper suite

`manifests/rlm-paper-v1.json` pins the target paper revision and names the
paper's S-NIAH, BrowseComp-Plus, OOLONG, OOLONG-Pairs, and LongBench-v2 CodeQA
task families. TAG-Bench is tracked separately in the same manifest as a
Droste-specific follow-on rather than being presented as part of the paper.

The OOLONG 131K `trec_coarse` validation slice is `ready` and has published
results: immutable artifacts, a price-snapshot provenance record, and
regenerated reports live under `results/oolong-trec-coarse-131k-2026-07-17/`
([#81](https://github.com/tensor-systems/droste/issues/81)).

The deterministic S-NIAH 32K split is `ready` and has published results:
immutable artifacts, generator provenance, and regenerated reports live under
[`results/sniah-32k-2026-07-17/`](results/sniah-32k-2026-07-17/). Its three
arms compare direct `gpt-5.6-sol`, direct `gpt-5.6-terra`, and Droste with a
`gpt-5.6-terra` root and `gpt-5.6-luna` subcalls. The other datasets remain
`planned`. A planned benchmark cannot be run because it has no
task path. Dataset adapters promote each benchmark to `ready` only after source
or generator provenance, split, integrity checks, and scorer are pinned.

## Live runs

The checked-in manifest pins public live configurations (models, reasoning
efforts, budgets, concurrency) for both OOLONG and S-NIAH. Materializing or
validating the suite makes no model calls. Live runs require an explicit run
command and a new output directory, refuse to overwrite artifacts, snapshot
the endpoint's public price table, and reject additions if that snapshot
changes. The published OOLONG report regenerates offline from the committed
artifacts:

```bash
uv run python -m benchmarks report benchmarks/manifests/rlm-paper-v1.json benchmarks/results/oolong-trec-coarse-131k-2026-07-17/artifacts --json /tmp/regen-check.json --markdown /tmp/regen-check.md
```

After materializing the ready task sets above, the published S-NIAH reports
regenerate from the committed artifacts with the 50 S-NIAH task ids selected:

```bash
task_args=()
for id in {000..049}; do task_args+=(--task-id "sniah-$id"); done
uv run python -m benchmarks report \
  benchmarks/manifests/rlm-paper-v1.json \
  benchmarks/results/sniah-32k-2026-07-17/artifacts \
  "${task_args[@]}" \
  --json /tmp/sniah-regen-check.json \
  --markdown /tmp/sniah-regen-check.md
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
