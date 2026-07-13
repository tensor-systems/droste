from __future__ import annotations

from typing import Protocol


class SubcallClient(Protocol):
    """Interface for recursive sub-LLM calls.

    Implementations attached to an :class:`ExecutionContext` reserve
    ``calls_made`` before dispatch and increment ``successful_calls`` only
    after an item returns a usable text response.
    """

    def llm_query(self, prompt: str, context: str = "") -> str:
        """Single sub-LLM call."""
        ...

    def llm_batch(self, prompts: list[str], contexts: list[str] | None = None) -> list[str]:
        """Batch sub-LLM calls for parallel processing."""
        ...

    def llm_batch_with_errors(
        self,
        prompts: list[str],
        contexts: list[str] | None = None,
    ) -> tuple[list[str], list[dict[str, object]]]:
        """Batch sub-LLM calls with structured per-item errors."""
        ...
