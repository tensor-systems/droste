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
    exact: bool = False

    def __post_init__(self) -> None:
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cache_read_tokens",
            "cache_creation_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"token usage {name} must be a non-negative integer")
        if not isinstance(self.exact, bool):
            raise TypeError("token usage exact must be a bool")
        if self.exact and self.total_tokens < self.prompt_tokens + self.completion_tokens:
            raise ValueError("exact token usage total cannot be less than its parts")
        if self.exact and self.cache_read_tokens + self.cache_creation_tokens > self.prompt_tokens:
            raise ValueError("exact cache token breakdown cannot exceed prompt tokens")

    @classmethod
    def unavailable(cls) -> TokenUsage:
        """Return the explicit fact that trustworthy provider usage was absent."""

        return cls(0, 0, 0, exact=False)


def strip_cache_anchor_markers(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a shallow outbound copy without Droste's cache marker."""
    return [
        {key: value for key, value in message.items() if key != CACHE_ANCHOR_MARKER}
        for message in messages
    ]


class LLMUsageFailure(RuntimeError):
    """An output/parsing failure paired with completed provider usage."""

    def __init__(self, usage: TokenUsage, cause: Exception) -> None:
        if not isinstance(usage, TokenUsage):
            raise TypeError("LLM usage failure requires TokenUsage")
        if not isinstance(cause, Exception):
            raise TypeError("LLM usage failure cause must be an Exception")
        super().__init__(str(cause))
        self.usage = usage
        self.cause = cause


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
