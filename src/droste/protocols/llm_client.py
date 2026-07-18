from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

# Reserved for the loop-to-client handoff. It is never part of the canonical
# transcript and must be consumed or stripped before an API payload is built.
CACHE_ANCHOR_MARKER = "_droste_cache_anchor"


@dataclass
class TokenUsage:
    """Token usage from an LLM call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


def strip_cache_anchor_markers(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a shallow outbound copy without Droste's cache marker."""
    return [
        {key: value for key, value in message.items() if key != CACHE_ANCHOR_MARKER}
        for message in messages
    ]


def total_tokens_from_usage(usage: Any) -> int:
    """Best-effort extraction of total tokens from a usage object."""
    if usage is None:
        return 0
    for attr in ("total_tokens", "total"):
        if hasattr(usage, attr):
            value = getattr(usage, attr)
            if isinstance(value, int):
                return value
    if hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
        return int(getattr(usage, "prompt_tokens")) + int(getattr(usage, "completion_tokens"))
    if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
        return (
            int(getattr(usage, "input_tokens"))
            + int(getattr(usage, "cache_read_input_tokens", 0))
            + int(getattr(usage, "cache_creation_input_tokens", 0))
            + int(getattr(usage, "output_tokens"))
        )
    return 0


class LLMClient(Protocol):
    """Abstract LLM API client."""

    def responses_create(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
        return_usage: bool = False,
    ) -> str | tuple[str, TokenUsage]:
        """Create a response from a list of messages.

        ``temperature=None`` means "don't send the parameter" — modern models
        (gpt-5.x, opus-4.x) reject it outright, so implementations must only
        include it when explicitly set.
        """
        ...

    # NOTE deliberately no batch method: the core loop parallelizes subcalls
    # via SubcallClient.llm_batch, and how THAT is transported (one gateway
    # batch call vs bounded concurrent fan-out) is each client's own
    # implementation detail — see the typed-batch design in #21.

    def get_model_context_window(self, model: str) -> int | None:
        """Return context window size for model, if known."""
        ...
