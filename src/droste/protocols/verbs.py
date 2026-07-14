"""Generated-binding vocabulary shared by provider and policy projections."""

from __future__ import annotations

import builtins
import keyword
from dataclasses import dataclass

RESERVED_NAMES = frozenset(
    {
        "answer",
        "context",
        "llm_query",
        "llm_batch",
        "batch_llm_query",
        "llm_query_batched",
        "llm_batch_json",
        "llm_query_batched_json",
        "aggregate_json_counts",
    }
)


def validate_binding_name(value: object, *, subject: str) -> str:
    """Validate one generated Python name without provider-specific vocabulary."""

    name = str(value)
    if not name.isidentifier() or keyword.iskeyword(name) or name.startswith("_"):
        raise ValueError(f"{subject} binding {name!r} is not a public Python identifier")
    if name in RESERVED_NAMES:
        raise ValueError(f"{subject} binding {name!r} collides with a reserved global")
    if name in dir(builtins):
        raise ValueError(f"{subject} binding {name!r} shadows a Python builtin")
    return name


@dataclass(frozen=True)
class AccessorManifest:
    """Exact generated data bindings consumed by count-policy discovery."""

    flat: frozenset[str] = frozenset()
    namespaced: frozenset[tuple[str, str]] = frozenset()


EMPTY_ACCESSOR_MANIFEST = AccessorManifest()


__all__ = [
    "AccessorManifest",
    "EMPTY_ACCESSOR_MANIFEST",
    "RESERVED_NAMES",
    "validate_binding_name",
]
