from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

AGGREGATE_REGEX = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX|ROUND)\s*\(", re.IGNORECASE)
LLM_CALL_REGEX = re.compile(
    r"\b(llm_query_batched|llm_query|batch_llm_query|llm_batch)\s*\(", re.IGNORECASE
)
LEN_SEARCH_REGEX = re.compile(r"\blen\s*\(\s*(search|get_recent)\s*\(", re.IGNORECASE)
NUMERIC_OUTPUT_REGEX = re.compile(r"^\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)(%)?\s*$")


@dataclass(frozen=True)
class PolicyHints:
    """Optional enforcement hints supplied by the caller."""

    semantic: bool = False
    count: bool = False
    numeric_output: bool = False

    @classmethod
    def from_tokens(cls, tokens: Iterable[str]) -> "PolicyHints":
        normalized = {token.strip().lower() for token in tokens if token and token.strip()}
        return cls(
            semantic="semantic" in normalized,
            count=bool({"count", "aggregate", "aggregation"} & normalized),
            numeric_output=bool({"numeric", "numeric_output", "number"} & normalized),
        )


def uses_llm_query(code: str) -> bool:
    return bool(LLM_CALL_REGEX.search(code))


def uses_sql_aggregate(code: str) -> bool:
    if "query(" not in code and "query (" not in code:
        return False
    return bool(AGGREGATE_REGEX.search(code))


def is_numeric_output(text: str) -> bool:
    if text is None:
        return False
    return bool(NUMERIC_OUTPUT_REGEX.match(str(text).strip()))


def _len_over_accessor_regex(data_accessors: Iterable[str]) -> re.Pattern[str]:
    """`len(<accessor>(...))` detector over the verbs actually bound in the
    sandbox — including host-declared extras — so the count contract stays
    enforced no matter what a source names its accessors (#10). Falls back
    to the static generic verbs when the caller supplies none. The optional
    `\\w+\\.` prefix also catches namespaced calls (`len(db.search(...))`)."""
    names = sorted({str(n) for n in data_accessors if n}) or ["search", "get_recent"]
    alternation = "|".join(re.escape(n) for n in names)
    return re.compile(rf"\blen\s*\(\s*(?:\w+\.)?({alternation})\s*\(", re.IGNORECASE)


def contract_violations(
    code: str,
    hints: PolicyHints | None,
    data_accessors: Iterable[str] = (),
) -> list[str]:
    if hints is None:
        return []

    violations: list[str] = []

    if hints.semantic and not uses_llm_query(code):
        violations.append(
            "Semantic question requires llm_query() or llm_query_batched(). "
            "Use search()/get_recent() to pre-filter, then call llm_query."
        )

    if hints.count:
        if not uses_sql_aggregate(code):
            violations.append(
                "Count/percentage question must use SQL aggregates via query(), "
                "e.g. SELECT COUNT(*). Do not compute counts with len() over accessor results."
            )
        elif _len_over_accessor_regex(data_accessors).search(code):
            violations.append(
                "Do not compute counts with len() over data-accessor results. "
                "Use SQL COUNT() in query()."
            )

    return violations
