"""Immutable planning metadata for subcall input capacity.

Reading optional client metadata is kept at the edge.  The value and merge
functions are pure so scaffold construction and compatibility checks never
need to inspect a transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import getattr_static
from typing import Any, Mapping

_CAPACITY_STATES = frozenset({"bounded", "unbounded", "unknown"})


@dataclass(frozen=True, slots=True)
class SubcallInputCapacity:
    """One resolved input-capacity state for subcall planning.

    ``bounded`` carries a positive effective token count. ``unbounded`` means
    the client deliberately imposes no input bound. ``unknown`` means neither
    the rollout nor the client established a value; it is never a guessed
    model context window.
    """

    state: str
    tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, str) or self.state not in _CAPACITY_STATES:
            raise ValueError("subcall input capacity state must be bounded, unbounded, or unknown")
        if self.state == "bounded":
            if isinstance(self.tokens, bool) or not isinstance(self.tokens, int) or self.tokens < 1:
                raise ValueError("bounded subcall input capacity tokens must be positive")
        elif self.tokens is not None:
            raise ValueError(f"{self.state} subcall input capacity must have null tokens")

    @classmethod
    def bounded(cls, tokens: int) -> "SubcallInputCapacity":
        return cls("bounded", tokens)

    @classmethod
    def unbounded(cls) -> "SubcallInputCapacity":
        return cls("unbounded")

    @classmethod
    def unknown(cls) -> "SubcallInputCapacity":
        return cls("unknown")

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state, "tokens": self.tokens}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SubcallInputCapacity":
        if not isinstance(value, Mapping):
            raise TypeError("subcall input capacity must be an object")
        missing = {"state", "tokens"} - value.keys()
        unknown = value.keys() - {"state", "tokens"}
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unknown " + ", ".join(sorted(unknown)))
            raise ValueError("subcall input capacity has " + "; ".join(details))
        return cls(value["state"], value["tokens"])


def reported_subcall_input_capacity(client: object) -> SubcallInputCapacity:
    """Read the optional companion protocol once at the execution edge.

    Protocol absence is unknown. Implementations return the immutable value
    directly; getter failures and invalid values propagate rather than being
    mistaken for legacy absence.
    """

    missing = object()
    if getattr_static(client, "input_token_capacity", missing) is missing:
        return SubcallInputCapacity.unknown()
    value = getattr(client, "input_token_capacity")
    if not isinstance(value, SubcallInputCapacity):
        raise TypeError("input_token_capacity must return SubcallInputCapacity")
    return value


def resolve_subcall_input_capacity(
    declared: SubcallInputCapacity,
    reported: SubcallInputCapacity,
) -> SubcallInputCapacity:
    """Resolve rollout and client facts, rejecting contradictory known values."""

    if declared.state == "unknown":
        return reported
    if reported.state == "unknown":
        return declared
    if declared != reported:
        raise ValueError(
            "rollout subcall input capacity does not match the subcall client: "
            f"{declared.as_dict()} != {reported.as_dict()}"
        )
    return declared


def render_subcall_input_capacity(capacity: SubcallInputCapacity) -> str:
    """Project one capacity value into stable model-facing prompt text."""

    if capacity.state == "bounded":
        return f"{capacity.tokens} tokens per call (bounded)"
    if capacity.state == "unbounded":
        return "unbounded (deliberate)"
    return "unknown (client and rollout did not report)"


__all__ = [
    "SubcallInputCapacity",
    "render_subcall_input_capacity",
    "reported_subcall_input_capacity",
    "resolve_subcall_input_capacity",
]
