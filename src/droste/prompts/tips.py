from __future__ import annotations

from typing import Literal

TipsProfile = Literal["full", "minimal", "none"]

# Content informed by the two RLM reference implementations:
# Design principles for the tips below (all original text):
# - orchestrator-not-solver framing; a batching budget; "just read it if a
#   search already pins the answer"; worked chunking examples.
# - explore first; iterate; "string matching finds WHERE, llm_query
#   understands WHAT."
# These are general RLM orchestration patterns.

_ORCHESTRATOR = (
    "You are an orchestrator, not a solver. Your own context window is small and the "
    "data may be huge: push every long-context operation — reading, summarizing, "
    "classifying, verifying, answering sub-questions — into llm_query / "
    "llm_query_batched instead of pulling raw text into your own message stream. "
    "Reserve your own tokens for deciding what to ask next, combining subcall "
    "outputs in code, and finalizing."
)

_WHERE_VS_WHAT = (
    "String matching finds WHERE things are; llm_query understands WHAT things mean. "
    "Use Python (regex, keywords, slicing) to locate and narrow, then llm_query to "
    "interpret. Do not answer semantic questions by counting keyword hits."
)

_EXPLORE_FIRST = (
    "EXPLORE FIRST. Inspect `context` before processing it: print its type, size, and "
    'a small sample, then plan. Do not set answer["ready"] = True on your first '
    "turn — use it to understand the data. Iterate in small steps and verify "
    "intermediate results before finalizing."
)

_WORKED_EXAMPLE = """Worked pattern — map-reduce over chunks with llm_query_batched:
```python
# context is one long string; split into ~100K-char chunks
chunk_size = 100_000
chunks = [context[i:i + chunk_size] for i in range(0, len(context), chunk_size)]
prompts = [
    f"List every product complaint in the text below, one per line.\\n\\n{chunk}"
    for chunk in chunks
]
partials = llm_query_batched(prompts)  # one subcall per chunk, run concurrently
final = llm_query(
    "Merge these partial lists into one deduplicated summary:\\n\\n" + "\\n".join(partials)
)
print(final)
```"""

_BATCH_BUDGET = (
    "Budget subcalls on two axes: each prompt handles roughly ~100K characters of "
    "input, and keep each llm_query_batched batch to at most ~20 prompts. Pack each "
    "prompt near capacity (many items or a whole document per prompt) so one call "
    "does a lot of work. Fat-prompt small batches are correct; tiny-prompt "
    "mega-batches are the anti-pattern. If the data exceeds both budgets at once, "
    "filter aggressively in Python first, or stage it: a cheap coarse pass to narrow "
    "candidates, then a targeted second pass over the survivors."
)

_BALANCE = (
    "Subcalls are a tool, not a ritual. If a regex or keyword search over `context` "
    "already pins the answer, or a short visible slice contains it, just read it "
    "directly. Reach for llm_query when the raw text will not fit in your own window "
    "or the question needs semantic interpretation."
)

_MINIMAL_CORE = (
    "You are an orchestrator, not a solver: push long-context reading, summarizing, "
    "and classifying into llm_query / llm_query_batched, and keep your own tokens "
    "for planning and combining results. String matching finds WHERE things are; "
    "llm_query understands WHAT things mean."
)

_MINIMAL_PRACTICE = (
    "Inspect `context` (type, size, sample) before processing; do not finalize on "
    "turn 1. Budget: ~100K characters per subcall prompt, at most ~20 prompts per "
    "batch — fat prompts, small batches. If a simple search already pins the answer, "
    "just read it directly."
)

TIPS_PROFILES: dict[str, list[str]] = {
    "full": [
        _ORCHESTRATOR,
        _WHERE_VS_WHAT,
        _EXPLORE_FIRST,
        _WORKED_EXAMPLE,
        _BATCH_BUDGET,
        _BALANCE,
    ],
    "minimal": [
        _MINIMAL_CORE,
        _MINIMAL_PRACTICE,
    ],
    "none": [],
}
