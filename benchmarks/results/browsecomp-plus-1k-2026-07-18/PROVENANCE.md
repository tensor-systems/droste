# BrowseComp-Plus provenance

This result uses the `test` split of
[`Tevatron/browsecomp-plus`](https://huggingface.co/datasets/Tevatron/browsecomp-plus/tree/144cff8e35b5eaef7e526346aa60774a9deb941f),
pinned at revision `144cff8e35b5eaef7e526346aa60774a9deb941f`, and the
`train` split of
[`Tevatron/browsecomp-plus-corpus`](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus/tree/b27b02bc3e45511b8b82a13e6f90ce761df726f6),
pinned at revision `b27b02bc3e45511b8b82a13e6f90ce761df726f6`. Both
datasets are MIT licensed.

The published result is a deterministic 150-of-830 query sample. Query IDs
are sampled from stable lexicographic order with seed `166001`. Each selected
query receives about 1,000 documents: every gold and evidence document plus a
filler sample drawn with an independently derived per-query seed. Documents
shared across tasks are stored once in a 77,928-document indexed pool. The
suite manifest pins the resulting task-file SHA-256 as
`73a84c0021d55d1f44559613bd04eb28a867ceddc9aed22192af96e8551dded1`.

The Droste arm keeps each raw task context external to the root prompt and
uses local Python to search it. It makes bounded Luna subcalls over promising
retrieved excerpts; it never loads the full external context into any single
LLM call. This is the recursive-analysis path the benchmark is designed to
test, rather than a larger direct prompt.

The selected contexts range from about 24.1 MB to 44.4 MB, approximately
6.0M-11.1M tokens. That is 6-10 times beyond available model context windows.
The two direct arms therefore cannot structurally attempt these tasks: all 300
direct artifacts are honest `context_limit` results with zero model tokens and
zero cost. This is a genuine incapacity of the direct approach at this scale,
not a harness bug, silent truncation, or budget artifact.

The 450 JSON files under `artifacts/` are the immutable per-task records for
150 tasks across three arms. The Droste arm completed 148 tasks; tasks `229`
and `794` ended in legitimate HTTP 504 timeouts. The adjacent reports are
deterministic aggregations of those artifacts. No dataset contexts, corpus
documents, or materialized task files are redistributed in this result
directory. The exact run-era manifest is preserved as
[`browsecomp-plus-1k-2026-07-18.json`](../../manifests/browsecomp-plus-1k-2026-07-18.json)
so report regeneration retains strict suite-version and manifest-hash checks.
