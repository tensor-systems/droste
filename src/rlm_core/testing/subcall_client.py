from __future__ import annotations

from typing import Any

from ..protocols.subcall_client import SubcallClient


class MockSubcallClient(SubcallClient):
    def llm_query(self, prompt: str, context: str = "") -> str:
        return ""

    def llm_batch(self, prompts: list[str], contexts: list[str] | None = None) -> list[str]:
        return ["" for _ in prompts]

    def llm_batch_with_errors(
        self,
        prompts: list[str],
        contexts: list[str] | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        return self.llm_batch(prompts, contexts), []
