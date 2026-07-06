from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..protocols.llm_client import LLMClient, TokenUsage


@dataclass
class MockResponse:
    text: str
    usage: TokenUsage


class MockLLMClient(LLMClient):
    def __init__(self, responses: list[MockResponse] | None = None) -> None:
        self._responses = responses or []
        self._call_count = 0

    def responses_create(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
        return_usage: bool = False,
    ) -> str | tuple[str, TokenUsage]:
        if self._call_count >= len(self._responses):
            raise RuntimeError("MockLLMClient: no more responses")
        response = self._responses[self._call_count]
        self._call_count += 1
        if return_usage:
            return response.text, response.usage
        return response.text

    def batch_responses(self, requests: list[dict[str, Any]]) -> list[str]:
        raise NotImplementedError("batch_responses not implemented in MockLLMClient")

    def get_model_context_window(self, model: str) -> int | None:
        return None
