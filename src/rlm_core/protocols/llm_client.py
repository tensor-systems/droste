from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class TokenUsage:
    """Token usage from an LLM call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


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
        return int(getattr(usage, "input_tokens")) + int(getattr(usage, "output_tokens"))
    return 0


class LLMClient(Protocol):
    """Abstract LLM API client."""

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        return_usage: bool = False,
    ) -> str | tuple[str, TokenUsage]:
        """Make a chat completion request."""
        ...

    def batch_responses(self, requests: list[dict[str, Any]]) -> list[str]:
        """Batch multiple requests for parallel processing."""
        ...

    def get_model_context_window(self, model: str) -> int | None:
        """Return context window size for model, if known."""
        ...
