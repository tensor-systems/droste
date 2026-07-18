# OOLONG-Pairs provenance

This result materializes the 20 OOLONG-Pairs questions from Appendix D.1 of
the Recursive Language Models paper against the 32,768-token `trec_coarse`
context at validation row 900 of
[`oolongbench/oolong-synth`](https://huggingface.co/datasets/oolongbench/oolong-synth/tree/f0d59eaf0febf130664cfceb710436c8e3216b2b).
The dataset is pinned at revision
`f0d59eaf0febf130664cfceb710436c8e3216b2b`. The suite manifest records the
materialized task-file SHA-256 as
`169a2aaddc8603128f672d32f9aa8a2e0565974d91b6468b7431654dd81bde40`.

The task materializer and predicate encoder are implemented and documented in
[`benchmarks/oolong_pairs.py`](../../oolong_pairs.py). Four semantic decisions
are explicit there: `X or Y` is inclusive OR; universal date constraints use
vacuous truth when a user has no instances of the constrained label;
asymmetric predicates accept either assignment of the two user roles; and
“exactly one” means a count equal to one, not at least one. Output pairs are
unordered and normalized to `(lower_user_id, higher_user_id)` before the
set-based precision, recall, and F1 calculation.

The Droste arm uses a hybrid design. Deterministic Python parses the context,
aggregates per-user histories, and exhaustively enumerates candidate pairs.
Luna subcalls perform only the irreducible semantic classification of context
instances; they do not perform the pair enumeration or aggregation.

The 60 lean JSON files under `artifacts/` are the immutable per-task records for
20 tasks across three arms. They retain canonical SHA-256 markers instead of
duplicating the full predictions and references. The
adjacent reports are deterministic aggregations of those artifacts; no dataset
context or materialized task file is redistributed in this result directory.
The exact suite manifest used for this run is preserved as
[`oolong-pairs-2026-07-17.json`](../../manifests/oolong-pairs-2026-07-17.json)
so report regeneration retains strict suite-version and manifest-hash checks
after later additive suite changes.

The full predictions are load-bearing verification data in the release asset
[`oolong-pairs-32k-2026-07-17.tar.gz`](https://github.com/tensor-systems/droste/releases/download/benchmark-data/oolong-pairs-32k-2026-07-17/oolong-pairs-32k-2026-07-17.tar.gz),
pinned by SHA-256
`6aa129b1df692948a8c2961bfe049cbf68353f0aebff8a26d63b968b1abaa89f`.
The report command verifies materialized predictions against the lean artifact
markers. References are regenerated locally by the existing, task-SHA-pinned
OOLONG-Pairs materializer and checked against their markers. The machine-readable
release and verification record is
[`provenance/evidence-bundle.json`](provenance/evidence-bundle.json).
