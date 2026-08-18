# Benchmark results

The published suite compares direct long-context calls with Droste runs that
use separate root and subcall models. Scores and costs below come from runs on
July 17 and 18, 2026.

| Benchmark | Scope | Best direct score | Droste score | Droste cost |
|---|---|---:|---:|---:|
| OOLONG | 131K tokens, 50 tasks | 0.6020 | **0.6432** | $10.16 |
| S-NIAH | 32K tokens, 50 tasks | **1.00** | **1.00** | $0.66 |
| LongBench-v2 CodeQA | 20 task sample | **0.75** | 0.65 | $3.79 |
| OOLONG-Pairs | 32K tokens, 20 tasks | 0.034 | **0.80** | $2.14 |
| BrowseComp-Plus | 6.0M to 11.1M tokens, 150 tasks | Could not run | **0.9400** | $24.54 |

Droste was strongest when answers required aggregation across scattered or
oversized context. Direct calls remained competitive on smaller lookup tasks,
and the direct arm scored higher on the CodeQA sample.

These are individual runs. Provider sampling, model changes behind an
identifier, and trajectory variance can change the result. The OOLONG score
ordering was not statistically separated at 95 percent confidence. Costs are
recorded measurements from the selected price snapshot, not estimates of
future runs.

The repository's [benchmark guide](../benchmarks/README.md) contains the model
identifiers used by each historical run, scoring rules, manifests, raw
artifacts, reproduction commands, and complete caveats.
