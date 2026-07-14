from __future__ import annotations

import threading
from typing import Any

from ..execution.context import ExecutionContext
from ..protocols.subcall_client import SubcallClient


class MockSubcallClient(SubcallClient):
    def __init__(self, *, context: ExecutionContext | None = None) -> None:
        self._context = context
        self._context_is_explicit = context is not None
        self._lock = threading.Lock()

    def bind_context(self, context: ExecutionContext) -> None:
        with self._lock:
            if not self._context_is_explicit:
                self._context = context

    def _account_attempts(self, count: int) -> None:
        with self._lock:
            context = self._context
            if context is None:
                return
            context.record_subcall_attempts(count)

    def llm_query(self, prompt: str, context: str = "") -> str:
        self._account_attempts(1)
        return ""

    def llm_batch(self, prompts: list[str], contexts: list[str] | None = None) -> list[str]:
        self._account_attempts(len(prompts))
        return ["" for _ in prompts]

    def llm_batch_with_errors(
        self,
        prompts: list[str],
        contexts: list[str] | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        return self.llm_batch(prompts, contexts), []
