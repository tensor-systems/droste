# Reproducible benchmarks

This directory is repository-local tooling, not part of the published `droste`
wheel. It defines the versioned contract for benchmark manifests, per-task run
artifacts, deterministic scoring, and aggregate reports.

## Results

### OOLONG `trec_coarse`

The published OOLONG result is a slice from the
[RLM paper suite](manifests/rlm-paper-v1.json): 131K-token contexts, 50 tasks,
three arms, one repetition, run 2026-07-17
([report](results/oolong-trec-coarse-131k-2026-07-17/report.md) ·
[raw artifacts](results/oolong-trec-coarse-131k-2026-07-17/artifacts)).

| Arm | Root model | Subcall model | Mean score | Cost | Tokens |
|---|---|---|---:|---:|---:|
| direct-sol | gpt-5.6-sol | — | 0.6020 | $26.175585 | 4,763,877 |
| direct-terra | gpt-5.6-terra | — | 0.5668 | $12.471453 | 4,722,032 |
| droste-terra-luna | gpt-5.6-terra | gpt-5.6-luna | 0.6432 | $10.158295 | 3,531,293 |

The direct arms place the full context in a single model call. The droste arm
runs this engine with a mid-tier root model delegating to a cheaper subcall
model (root reasoning `medium`, subcall reasoning `none`). All 150 task–arm
runs completed; a failure or timeout would be retained as a typed artifact,
not dropped. All arms ran through the same OpenAI-compatible endpoint
(ModelRelay); costs are measured in integer micro-USD against the price
snapshot recorded with the run. The `oolong_official` scorer
([scoring.py](scoring.py)) implements the benchmark's published rule — exact,
comparison-phrase, and date matches score 1.0, and numeric answers earn graded
credit of 0.75^|error| — so the mean score is graded, not plain accuracy.

droste-terra-luna scored highest of the three arms — 0.6432, against 0.6020
for direct-sol and 0.5668 for direct-terra — while costing about 2.6× less
than the stronger direct baseline and using fewer total tokens. With one
50-task repetition, a paired bootstrap over the per-task scores in the
committed artifacts does not separate the three mean scores at 95%
confidence, so treat the score ranking as this run's observed result rather
than a statistically established ranking; the cost figure is a direct
measurement, not a sampled statistic, and isn't subject to that caveat.
Run-to-run variation comes from provider sampling (temperature is
not pinned; the endpoint default applies), from possible server-side model
changes behind pinned model ids, and from trajectory variance in the droste
arm. Per-arm prompts are fixed harness prompts committed in
[live.py](live.py), including benchmark-specific guidance for the droste arm;
budgets, limits, and concurrency are pinned in the
[manifest](manifests/rlm-paper-v1.json).

The task slice is materialized from the public dataset
([oolongbench/oolong-synth](https://huggingface.co/datasets/oolongbench/oolong-synth),
pinned revision `f0d59ea`, validation rows 1050–1099) with SHA-256
verification. The dataset card at that revision does not state a license, so
the tasks themselves are not redistributed here; the committed artifacts
contain only per-task predictions, gold labels, scores, and usage. This guide
has the [materialization command](#pinned-oolong-data), the
[offline report-regeneration command](#live-runs), and the live-run procedure
(new output directory, immutable artifacts, explicit cost cap).

### S-NIAH

The adjacent S-NIAH result is the single needle-in-a-haystack retrieval task
from the RULER methodology of
[Hsieh et al. (2024)](https://arxiv.org/abs/2404.06654). This
run used 32,768-token noise haystacks, word-pair keys and values, seed `42`, 50
tasks, three arms, and one repetition on 2026-07-17
([report](results/sniah-32k-2026-07-17/report.md) ·
[raw artifacts](results/sniah-32k-2026-07-17/artifacts) ·
[generator provenance](results/sniah-32k-2026-07-17/provenance/generator.json)).

| Arm | Root model | Subcall model | Exact-match accuracy | Cost | Tokens |
|---|---|---|---:|---:|---:|
| direct-sol-sniah | gpt-5.6-sol | — | 84% | $7.791160 | 1,556,242 |
| direct-terra-sniah | gpt-5.6-terra | — | 100% | $3.895694 | 1,556,249 |
| droste-terra-luna-sniah | gpt-5.6-terra | gpt-5.6-luna | 100% | $0.659419 | 190,253 |

The direct arms place the complete prompt in one model call. The droste arm
runs this engine with a `gpt-5.6-terra` root delegating to `gpt-5.6-luna`
subcalls (root reasoning `medium`, subcall reasoning `none`). All 150 task-arm
runs completed; failures and timeouts would remain typed artifacts rather than
being dropped. Costs are measured in integer micro-USD from the price snapshot
used by each run.

This benchmark is Droste's deterministic reproduction of RULER's published
algorithm and prompt methodology, checked against NVIDIA/RULER commit
[`38da79d79519ef87aa46ae804f838e1eab7f86d7`](https://github.com/NVIDIA/RULER/tree/38da79d79519ef87aa46ae804f838e1eab7f86d7).
The generator is [committed in this repository](sniah.py); it fetches and
redistributes no dataset or generated examples. There is consequently no
external dataset revision, dataset citation, or dataset-license section for
this result. The provenance record instead pins the generator hash, seed,
configuration, materialized-task hash, and RULER commit.

### LongBench-v2 CodeQA

This published result is code-repository-understanding multiple-choice QA over
real long-context codebases. The run used 20 tasks, three arms, and one
repetition on 2026-07-17
([report](results/longbench-v2-codeqa-20-2026-07-17/report.md) ·
[raw artifacts](results/longbench-v2-codeqa-20-2026-07-17/artifacts) ·
[provenance](results/longbench-v2-codeqa-20-2026-07-17/PROVENANCE.md)).

| Arm | Root model | Subcall model | Mean score | Successful | Cost | Tokens |
|---|---|---|---:|---:|---:|---:|
| direct-sol | gpt-5.6-sol | — | 0.7500 | 18/20 | $19.597640 | 3,910,423 |
| direct-terra | gpt-5.6-terra | — | 0.6500 | 17/20 | $9.096737 | 3,627,983 |
| droste-terra-luna | gpt-5.6-terra | gpt-5.6-luna | 0.6500 | 20/20 | $3.793057 | 1,348,775 |

The direct arms place the complete codebase context in one model call. The
droste arm runs this engine with a `gpt-5.6-terra` root delegating to
`gpt-5.6-luna` subcalls (root reasoning `medium`, subcall reasoning `none`).
All 60 scheduled task–arm attempts remain in the committed artifacts, including
the two unsuccessful direct-sol attempts and three unsuccessful direct-terra
attempts. Costs are measured in integer micro-USD from the price snapshot used
by each run.

droste-terra-luna tied direct-terra's 0.6500 mean score and trailed direct-sol's
0.7500 by 10 percentage points, while costing 2.4× less than direct-terra and
5.2× less than direct-sol. This is a mixed, cost-favorable result, not a clean
sweep. With one 20-task sample, the two-task score difference from direct-sol
is the observed result, not evidence of a population-level separation.

In [*Recursive Language Models* (Zhang, Kraska, and Khattab, 2025;
arXiv:2512.24601)](https://arxiv.org/abs/2512.24601), Table 1 evaluates CodeQA
across the full 23K–4.2M-token range: its GPT-5 direct baseline, with no
fine-tuning, scores 24.0%, far below this capped sample's 75.0% direct-sol
score, and several CodeQA entries are flagged for partial context-limit
failures. The contrast shows that this cost-bounded sample tests an easier
regime than the scale where the paper demonstrates the clearest gap between
direct and recursive approaches; this result therefore likely understates,
rather than contradicts, RLM's advantage on CodeQA-style tasks at the scale the
paper evaluates. [Issue #172](https://github.com/tensor-systems/droste/issues/172)
tracks a full-domain, larger-scale run.

The tasks come from
[`zai-org/LongBench-v2`](https://huggingface.co/datasets/zai-org/LongBench-v2/tree/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9),
Apache-2.0, at pinned revision
`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`, filtered to the 50-task
`Code Repository Understanding` domain. The published run is explicitly a
disclosed, cost-bounded 20-of-50 stratified subsample, not the complete domain:
8 short, 7 medium, and 5 long tasks, comprising 8 easy and 12 hard tasks. The
full-domain cost was disproportionate for this run—one pilot task alone cost
$3.30—so the harness fixes centered, evenly spaced selections within each
length/difficulty stratum before model outcomes are observed. The
[materializer](longbench_codeqa.py), manifest task hash, selection rule, and
offline report-regeneration command are public.

### OOLONG-Pairs

OOLONG-Pairs tests multi-hop pairwise reasoning over OOLONG-style
synthetic conversations: find every pair of users satisfying a relational
predicate, scored with set-based F1 over normalized, deduplicated pairs. The
run used one 32,768-token context, 20 tasks, three arms, and one repetition on
2026-07-17
([report](results/oolong-pairs-32k-2026-07-17/report.md) ·
[raw artifacts](results/oolong-pairs-32k-2026-07-17/artifacts) ·
[provenance](results/oolong-pairs-32k-2026-07-17/PROVENANCE.md)).

| Arm | Root model | Subcall model | Mean F1 | Successful | Cost | Tokens |
|---|---|---|---:|---:|---:|---:|
| direct-sol-pairs | gpt-5.6-sol | — | 0.000000 | 0/20 | $0.000000 | 0 |
| direct-terra-pairs | gpt-5.6-terra | — | 0.034057 | 14/20 | $2.497269 | 435,842 |
| droste-terra-luna-pairs | gpt-5.6-terra | gpt-5.6-luna | 0.801724 | 20/20 | $2.141592 | 767,642 |

The `$0.000000` recorded for direct-sol's HTTP 504 failures is a measurement
limit, not a zero-cost guarantee: because no response arrived, the harness
received no usage to bill, although provider generation may already have
started. By contrast, an HTTP 400 `context_limit` rejection occurs before
generation and is genuinely free.

The direct arms place the complete context in one model call. The Droste arm
uses a `gpt-5.6-terra` root with `gpt-5.6-luna` subcalls (root reasoning
`medium`, subcall reasoning `none`). Its design is deliberately hybrid:
deterministic Python parses and aggregates the records and exhaustively
enumerates user pairs, while Luna handles only the irreducible semantic
classification step. All 60 scheduled attempts remain in the committed
artifacts, including failures. Costs are measured in integer micro-USD from
the recorded model usage.

This is Droste's strongest result across the benchmark families evaluated
under [#166](https://github.com/tensor-systems/droste/issues/166). Direct-sol
could not complete a single task: all 20 attempts ended in legitimate HTTP 504
timeouts. Direct-terra completed 14/20, with six further 504 timeouts, but its
mean F1 of 0.034 was near zero—it essentially failed at the pairwise reasoning.
Droste completed all 20 tasks at 0.802 mean F1 for $2.14, less than
direct-terra's $2.50 despite succeeding on every task. No attempt recorded a
402 or 429. Exhaustive multi-hop reasoning over facts scattered across a long
context is where the paper's thesis about direct approaches structurally
failing is clearest in these runs.

The tasks are materialized from
[`oolongbench/oolong-synth`](https://huggingface.co/datasets/oolongbench/oolong-synth/tree/f0d59eaf0febf130664cfceb710436c8e3216b2b),
validation `trec_coarse` context-window row 900, at pinned revision
`f0d59eaf0febf130664cfceb710436c8e3216b2b`. The manifest pins the materialized
20-task file at SHA-256
`169a2aaddc8603128f672d32f9aa8a2e0565974d91b6468b7431654dd81bde40`.
The materializer documents the predicate semantics, and the scorer normalizes
unordered pairs before computing set precision, recall, and F1. The dataset
card at the pinned revision does not state a license, so dataset contexts and
tasks are not redistributed here.

This is a clean result, but its disclosed scope is small: `n=20`, with one
repetition. It establishes the outcome of this paired run rather than a
population-wide guarantee.

### BrowseComp-Plus

BrowseComp-Plus tests multi-hop factual retrieval over a large document
corpus. The published run used a deterministic 150-of-830 query sample, about
1,000 documents per query, three arms, and one repetition on 2026-07-18
([judge-augmented summary](results/browsecomp-plus-1k-2026-07-18/SUMMARY.md) ·
[regenerable exact-match report](results/browsecomp-plus-1k-2026-07-18/report.md) ·
[raw artifacts](results/browsecomp-plus-1k-2026-07-18/artifacts) ·
[provenance](results/browsecomp-plus-1k-2026-07-18/PROVENANCE.md)).

| Arm | Root model | Subcall model | Semantic judge | Exact match | Successful | Cost | Tokens |
|---|---|---|---:|---:|---:|---:|---:|
| direct-sol-browsecomp-plus-1k | gpt-5.6-sol | — | N/A (context exceeds window) | N/A | 0/150 | $0.000000 | 0 |
| direct-terra-browsecomp-plus-1k | gpt-5.6-terra | — | N/A (context exceeds window) | N/A | 0/150 | $0.000000 | 0 |
| droste-terra-luna-browsecomp-plus-1k | gpt-5.6-terra | gpt-5.6-luna | **0.9400** | 0.5600 | 148/150 | $24.537529 | 7,970,677 |

The selected contexts range from 24.1 MB to 44.4 MB, approximately
6.0M–11.1M tokens. That is 6–10× beyond available model context windows. The
direct arms must place the complete raw context in one call, so they cannot
structurally attempt these tasks at any cost. All 300 direct attempts were
rejected as `context_limit` before generation, producing zero model tokens and
genuinely zero cost. The harness did not truncate, summarize, or otherwise
weaken their input.

The Droste arm keeps the raw context external. Local Python searches and ranks
the 1,000 documents, preserves document IDs and excerpts for verification, and
uses bounded `gpt-5.6-luna` subcalls only on promising evidence. No single LLM
call receives the full context. This is the paper's core recursive-analysis
thesis at its most extreme: direct approaches do not merely underperform; the
problem is structurally outside their input regime, while Droste completes
148/150 tasks and reaches 0.9400 semantic-judge accuracy for $24.54. The two
tasks without predictions count as incorrect, so this is 141 correct answers
out of all 150 scheduled tasks. The paper reports 88.0%–91.3% on
BrowseComp-Plus; this run is 2.7 percentage points above the top of that range,
although it uses a 150-task sample rather than the paper's full evaluation.
The paper's range was judged by Qwen3-32B, while this result was judged by
`gpt-5.6-terra`; judge-model leniency could account for part of the observed
gap. Treat the comparison as directionally informative, not a strictly
controlled methodology match.

BrowseComp-Plus's official methodology uses an LLM judge for semantic
equivalence. A `gpt-5.6-terra` pass through ModelRelay applied its canonical
prompt to all 148 predictions for $0.292503. The complete judge responses are
in [judge-results.json](results/browsecomp-plus-1k-2026-07-18/judge-results.json),
and the exact-match score remains disclosed as a secondary metric. Exact match
marked 57 semantically accepted predictions wrong for differences such as
terminal punctuation, typographic apostrophes, abbreviations, and correct
additional detail.

The two unsuccessful Droste tasks, `229` and `794`, ended in legitimate HTTP
504 timeouts. They remain typed artifacts and contribute to the reported
150-task result: a 1.3% infrastructure-timeout rate on the only arm that could
attempt substantive work. There were no duplicate artifacts and no HTTP 402
failures.

The query and corpus sources are the MIT-licensed
[`Tevatron/browsecomp-plus`](https://huggingface.co/datasets/Tevatron/browsecomp-plus/tree/144cff8e35b5eaef7e526346aa60774a9deb941f)
and
[`Tevatron/browsecomp-plus-corpus`](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus/tree/b27b02bc3e45511b8b82a13e6f90ce761df726f6),
pinned independently at revisions `144cff8e35b5eaef7e526346aa60774a9deb941f`
and `b27b02bc3e45511b8b82a13e6f90ce761df726f6`. Query IDs are sampled from
stable lexicographic order with seed `166001`; required gold and evidence
documents are combined with independently seeded fillers, and documents shared
across tasks are stored once in a 77,928-document pool. The task materializer,
task-file hash, snapshot manifest, and offline regeneration command are public.

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

## Pinned LongBench-v2 CodeQA data

Materialize the public Code Repository Understanding domain's disclosed
20-of-50 cost-bounded subsample:

```bash
uv run python -m benchmarks materialize-longbench-codeqa \
  --output benchmarks/.data/longbench-v2-codeqa-20-v1
```

The materializer downloads the 50 rows in that domain from
`zai-org/LongBench-v2` at revision
`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`, verifies the pinned source
projection, and writes the fixed 20-task selection plus content-addressed
contexts. The dataset is Apache-2.0 licensed. Existing files are never
overwritten.

This is not a result over the complete 50-task domain. The cost-bounded sample
contains 8 short, 7 medium, and 5 long tasks, split into 8 easy and 12 hard
tasks. Within each length/difficulty stratum, the materializer sorts task IDs
and selects centered evenly spaced positions using `floor((2i+1)n/(2k))`.
The full-domain cost was disproportionate for this run; one pilot task alone
cost $3.30. The rule makes the reduced scope explicit and reproducible rather
than selecting tasks from model outcomes.

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

Published OOLONG-Pairs artifacts keep only canonical hashes
for predictions and references. Materialize the release-pinned predictions
before regenerating their report:

```bash
uv run python -m benchmarks materialize-oolong-pairs-predictions \
  --output benchmarks/.data/oolong-pairs-32k-2026-07-17-predictions
```

The command verifies the release tarball's pinned SHA-256 before extracting any
predictions and refuses to replace an existing output directory. References
remain deterministic outputs of the task materializer above; report generation
never fetches either source implicitly.

## Pinned BrowseComp-Plus data

Install the benchmark-only Parquet reader and materialize Droste's seeded
150-query, 1,000-document-per-query reading:

```bash
uv sync --group benchmarks
uv run python -m benchmarks materialize-browsecomp-plus \
  --output benchmarks/.data/browsecomp-plus-1k-seed-166001-v1
```

The pinned `rlm-paper-v1.json` manifest declares S-NIAH, BrowseComp-Plus,
OOLONG, OOLONG-Pairs, and LongBench-v2 CodeQA `ready`, so the four sibling task
sets must also be materialized before running or reporting BrowseComp-Plus,
even when only the BrowseComp-Plus task ids are selected:

```bash
uv run python -m benchmarks materialize-sniah \
  --output benchmarks/.data/sniah-noise-words-32768-50-v1
uv run python -m benchmarks materialize-oolong \
  --output benchmarks/.data/oolong-trec-coarse-131k-v1
uv run python -m benchmarks materialize-oolong-pairs \
  --output benchmarks/.data/oolong-pairs-32k-v1
uv run python -m benchmarks materialize-longbench-codeqa \
  --output benchmarks/.data/longbench-v2-codeqa-20-v1
```

The query and corpus revisions are pinned independently. The materializer uses
the benchmark's published Base64-then-XOR decoder, verifies decrypted required
documents against the plaintext corpus, and hashes the complete decrypted
selected-query content and plaintext corpus. Query IDs are sampled from stable
lexicographic order with seed `166001`; each query's filler-document sample has
an independently derived seed and includes the union of every gold and evidence
document.

Documents selected by more than one task are written once to an indexed JSONL
pool. Each task stores its exact 1,000 IDs in pinned corpus row order, and the
live harness assembles and hashes a context file on demand. Generated data and
assembled context caches remain under gitignored `benchmarks/.data/`.

The harness retains the deterministic `exact_match` scorer (case- and
whitespace-normalized) as a secondary diagnostic. The published headline uses
BrowseComp-Plus's official semantic-equivalence methodology: the standalone
[`browsecomp_judge.py`](browsecomp_judge.py) applies the canonical judge prompt
to the original predictions and writes a separate JSON result without changing
the run artifacts.

The pinned task contexts range from about 24.1 MB to 44.4 MB (median 32.3 MB),
or roughly 6.0M–11.1M tokens under a coarse four-bytes-per-token estimate. The
Droste arm keeps that content external to the root prompt and searches it in
Python. The direct arms necessarily inline it; their 12M-token authorization is
deliberately unusual and is not evidence that the configured provider accepts
an input that large. Before a paid pilot, preflight the provider's effective
context window. If it is smaller, direct-arm `context_limit` artifacts are the
honest result rather than silently truncating the benchmark.

## RLM paper suite

`manifests/rlm-paper-v1.json` pins the target paper revision and names the
paper's S-NIAH, BrowseComp-Plus, OOLONG, OOLONG-Pairs, and LongBench-v2 CodeQA
task families. TAG-Bench is tracked separately in the same manifest as a
Droste-specific follow-on rather than being presented as part of the paper.

The OOLONG 131K `trec_coarse` validation slice is `ready` and has published
results: immutable artifacts, a price-snapshot provenance record, and
regenerated reports live under
[`results/oolong-trec-coarse-131k-2026-07-17/`](results/oolong-trec-coarse-131k-2026-07-17/)
([#81](https://github.com/tensor-systems/droste/issues/81)).

The deterministic S-NIAH 32K split is `ready` and has published results:
immutable artifacts, generator provenance, and regenerated reports live under
[`results/sniah-32k-2026-07-17/`](results/sniah-32k-2026-07-17/). Its three
arms compare direct `gpt-5.6-sol`, direct `gpt-5.6-terra`, and Droste with a
`gpt-5.6-terra` root and `gpt-5.6-luna` subcalls.

LongBench-v2 CodeQA is also `ready` and has published results: immutable
artifacts, dataset and selection provenance, and regenerated reports live under
[`results/longbench-v2-codeqa-20-2026-07-17/`](results/longbench-v2-codeqa-20-2026-07-17/).
Its three arms compare direct `gpt-5.6-sol`, direct `gpt-5.6-terra`, and Droste
with a `gpt-5.6-terra` root and `gpt-5.6-luna` subcalls.

OOLONG-Pairs is `ready` and has published results: immutable artifacts,
dataset and materializer provenance, and regenerated reports live under
[`results/oolong-pairs-32k-2026-07-17/`](results/oolong-pairs-32k-2026-07-17/).
Its three arms compare direct `gpt-5.6-sol`, direct `gpt-5.6-terra`, and Droste
with a `gpt-5.6-terra` root and `gpt-5.6-luna` subcalls.

BrowseComp-Plus is `ready` and has published results: immutable artifacts,
dataset and sampling provenance, and regenerated reports live under
[`results/browsecomp-plus-1k-2026-07-18/`](results/browsecomp-plus-1k-2026-07-18/).
Its two direct arms cannot fit the 6.0M–11.1M-token contexts in a model window;
the Droste arm searches the external context locally and delegates bounded
evidence groups to `gpt-5.6-luna`.

TAG-Bench remains `planned`. A planned benchmark cannot be run because it has
no task path. Dataset adapters promote each benchmark to `ready` only after
source or generator provenance, split, integrity checks, and scorer are pinned.

## Live runs

The checked-in manifest pins public live configurations (models, reasoning
efforts, budgets, concurrency) for OOLONG, S-NIAH, LongBench-v2 CodeQA,
OOLONG-Pairs, and BrowseComp-Plus.
Materializing or validating the suite makes no model calls. Live runs require
an explicit run command and a new output directory, refuse to overwrite
artifacts, snapshot the endpoint's public price table, and reject additions if
that snapshot changes.

Each published artifact set retains the exact run-era manifest named by its
manifest SHA-256. The OOLONG report regenerates offline with its snapshot:

```bash
uv run python -m benchmarks report benchmarks/manifests/oolong-2026-07-17.json benchmarks/results/oolong-trec-coarse-131k-2026-07-17/artifacts --json /tmp/regen-check.json --markdown /tmp/regen-check.md
```

The pinned S-NIAH snapshot declares both OOLONG and S-NIAH `ready`, so both
task sets must be materialized before regenerating the S-NIAH report, even when
only the S-NIAH task ids are selected:

```bash
uv run python -m benchmarks materialize-oolong \
  --output benchmarks/.data/oolong-trec-coarse-131k-v1
uv run python -m benchmarks materialize-sniah \
  --output benchmarks/.data/sniah-noise-words-32768-50-v1
```

The published S-NIAH reports then regenerate from the committed artifacts with
the 50 S-NIAH task ids selected:

```bash
task_args=()
for id in {000..049}; do task_args+=(--task-id "sniah-$id"); done
uv run python -m benchmarks report \
  benchmarks/manifests/sniah-2026-07-17.json \
  benchmarks/results/sniah-32k-2026-07-17/artifacts \
  "${task_args[@]}" \
  --json /tmp/sniah-regen-check.json \
  --markdown /tmp/sniah-regen-check.md
```

The pinned LongBench-v2 CodeQA snapshot declares both OOLONG and LongBench-v2
CodeQA `ready`, so both task sets must be materialized before regenerating the
CodeQA report, even when only the CodeQA task ids are selected:

```bash
uv run python -m benchmarks materialize-oolong \
  --output benchmarks/.data/oolong-trec-coarse-131k-v1
uv run python -m benchmarks materialize-longbench-codeqa \
  --output benchmarks/.data/longbench-v2-codeqa-20-v1
```

The published CodeQA reports then regenerate from the committed artifacts with
all 20 CodeQA task IDs selected:

```bash
task_ids=(
  66ebd3ba5a08c7b9b35e0446 66ec3644821e116aacb1c312
  66ece545821e116aacb1dd77 66f1dac1821e116aacb27df1
  66f39ac5821e116aacb2da81 66f3ad93821e116aacb2e29f
  66f3c219821e116aacb2eb4e 66f3cb88821e116aacb2eeb9
  66f3e318821e116aacb2f9d1 66f530ce821e116aacb32f09
  66f908e3bb02136c067c4992 66fa3843bb02136c067c655d
  66fa542bbb02136c067c686d 66fa700bbb02136c067c6c06
  66fa788abb02136c067c6d75 66fa7c81bb02136c067c6e09
  66faa0f5bb02136c067c722c 66fcf80dbb02136c067c928e
  66fcfb5fbb02136c067c93ae 6708a096bb02136c067d1789
)
task_args=()
for task_id in "${task_ids[@]}"; do task_args+=(--task-id "$task_id"); done
uv run python -m benchmarks report \
  benchmarks/manifests/longbench-v2-codeqa-2026-07-17.json \
  benchmarks/results/longbench-v2-codeqa-20-2026-07-17/artifacts \
  "${task_args[@]}" \
  --json /tmp/longbench-codeqa-regen.json \
  --markdown /tmp/longbench-codeqa-regen.md
```

The pinned OOLONG-Pairs snapshot declares both OOLONG and OOLONG-Pairs `ready`,
so both benchmark task sets and the lean-artifact predictions must be
materialized before report regeneration:

```bash
uv run python -m benchmarks materialize-oolong \
  --output benchmarks/.data/oolong-trec-coarse-131k-v1
uv run python -m benchmarks materialize-oolong-pairs \
  --output benchmarks/.data/oolong-pairs-32k-v1
uv run python -m benchmarks materialize-oolong-pairs-predictions \
  --output benchmarks/.data/oolong-pairs-32k-2026-07-17-predictions
```

The OOLONG-Pairs report then regenerates from the committed artifacts with all
20 task IDs selected:

```bash
task_args=()
for task_id in {1..20}; do task_args+=(--task-id "$task_id"); done
uv run python -m benchmarks report \
  benchmarks/manifests/oolong-pairs-2026-07-17.json \
  benchmarks/results/oolong-pairs-32k-2026-07-17/artifacts \
  "${task_args[@]}" \
  --json /tmp/oolong-pairs-report.json \
  --markdown /tmp/oolong-pairs-report.md
cmp benchmarks/results/oolong-pairs-32k-2026-07-17/report.json \
  /tmp/oolong-pairs-report.json
cmp benchmarks/results/oolong-pairs-32k-2026-07-17/report.md \
  /tmp/oolong-pairs-report.md
```

The pinned BrowseComp-Plus snapshot declares OOLONG and BrowseComp-Plus
`ready`, so both task sets must be materialized before report regeneration:

```bash
uv sync --group benchmarks
uv run python -m benchmarks materialize-oolong \
  --output benchmarks/.data/oolong-trec-coarse-131k-v1
uv run python -m benchmarks materialize-browsecomp-plus \
  --output benchmarks/.data/browsecomp-plus-1k-seed-166001-v1
```

The plain BrowseComp-Plus exact-match report then regenerates from the committed
artifacts with all 150 selected task IDs. The judge-augmented `SUMMARY.md` is a
separate, labeled summary and is not an output of this command:

```bash
task_args=()
while IFS= read -r task_id; do
  task_args+=(--task-id "$task_id")
done < <(jq -r '.task_id' \
  benchmarks/results/browsecomp-plus-1k-2026-07-18/artifacts/*.json | sort -u)
uv run python -m benchmarks report \
  benchmarks/manifests/browsecomp-plus-1k-2026-07-18.json \
  benchmarks/results/browsecomp-plus-1k-2026-07-18/artifacts \
  "${task_args[@]}" \
  --json /tmp/browsecomp-plus-report.json \
  --markdown /tmp/browsecomp-plus-report.md
cmp benchmarks/results/browsecomp-plus-1k-2026-07-18/report.json \
  /tmp/browsecomp-plus-report.json
cmp benchmarks/results/browsecomp-plus-1k-2026-07-18/report.md \
  /tmp/browsecomp-plus-report.md
```

These result-specific manifests are exact snapshots of the configurations that
produced the immutable artifacts. The current `rlm-paper-v1.json` manifest is
the additive union of the live benchmark configurations and therefore has a
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
