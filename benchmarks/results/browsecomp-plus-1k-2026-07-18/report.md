# BrowseComp-Plus 150-task result

Manifest SHA-256: `2bc0a0010299db07fb7366c76dd7b1fa5c734859da31332ca63a3e7b0a635cba`

| Arm | Official semantic-judge accuracy | Exact match (secondary) | Successful | Tokens | Run cost |
|---|---:|---:|---:|---:|---:|
| direct-sol-browsecomp-plus-1k | N/A (context exceeds window) | N/A | 0/150 | 0 | $0.000000 |
| direct-terra-browsecomp-plus-1k | N/A (context exceeds window) | N/A | 0/150 | 0 | $0.000000 |
| droste-terra-luna-browsecomp-plus-1k | **0.9400 (141/150)** | 0.5600 (84/150) | 148/150 | 7,970,677 | $24.537529 |

The headline score uses BrowseComp-Plus's canonical semantic-equivalence judge
prompt. `gpt-5.6-terra` judged all 148 predictions through ModelRelay; the two
HTTP 504 tasks with no prediction were counted incorrect without a judge call.
The judge pass cost $0.292503. Its complete responses are in
[`judge-results.json`](judge-results.json), and the standalone rescorer is
[`browsecomp_judge.py`](../../browsecomp_judge.py).

The deterministic exact-match result is retained as a disclosed secondary
metric. It rejected 57 answers that the semantic judge accepted, including
punctuation, typography, spelling variants, and correct answers expressed with
additional detail. The direct arms have no accuracy score: all 300 attempts
were guaranteed `context_limit` rejections before generation, not substantive
0% results.

The 0.9400 result is 2.7 percentage points above the top of the paper's reported
88.0%–91.3% BrowseComp-Plus range. This is a 150-task sample rather than the
paper's full evaluation, so that comparison is descriptive rather than a
like-for-like population estimate.
