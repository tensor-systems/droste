# LongBench-v2 CodeQA provenance

This result uses the `train` split of
[`zai-org/LongBench-v2`](https://huggingface.co/datasets/zai-org/LongBench-v2/tree/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9),
pinned at revision `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`. The dataset is
licensed under Apache-2.0. Rows are restricted to the exact domain
`Code Repository Understanding`, which contains 50 tasks at the pinned
revision.

The published result is a disclosed, cost-bounded 20-of-50 stratified
subsample, not a result over the full domain. Its allocation preserves every
length bucket and approximates the source difficulty mix: 8 short, 7 medium,
and 5 long tasks, comprising 8 easy and 12 hard tasks. Within each
length/difficulty stratum, task IDs are sorted and centered evenly spaced
positions are selected with `floor((2i+1)n/(2k))`. This makes the selection
fixed and reproducible rather than chosen from model outcomes.

The complete materialization and selection procedure is implemented in
[`benchmarks/longbench_codeqa.py`](../../longbench_codeqa.py) and summarized in
the [benchmark documentation](../../README.md). The suite manifest records the
materialized task-file SHA-256 as
`d796fbcf741fbfc516903afd929e1e5aa6e64ded85445acbf950e638303ab5f5`.
The pinned 50-row source projection has SHA-256
`de11e20892981c365442db20bd5f477254e275a002cc09357f84ba3b0afa2d35`.

The 60 JSON files under `artifacts/` are the immutable per-task records for 20
tasks across three arms. The adjacent reports are deterministic aggregations
of those artifacts; no dataset contexts are redistributed in this result
directory.
