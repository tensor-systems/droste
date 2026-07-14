from __future__ import annotations

from typing import Protocol


class SubcallClient(Protocol):
    """Interface for recursive sub-LLM calls.

    Implementations attached to an :class:`ExecutionContext` reserve
    ``calls_made`` before dispatch and increment ``successful_calls`` only
    after an item returns a usable text response.

    Implementations may also implement the separate optional
    :class:`SubcallOutputTokenLimitProvider` protocol. It stays separate so
    existing third-party subcall clients remain compatible.
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
        """Batch sub-LLM calls with structured per-item errors.

        Every error has an integer ``index`` and human-readable ``error``
        string. Implementations may add ``type`` and an additive ``details``
        object shaped by :class:`droste.BatchItemErrorDetails`; custom clients
        that return only the original fields remain compatible.
        """
        ...


class SubcallOutputTokenLimitProvider(Protocol):
    """Optional read-only output-limit metadata for a subcall client.

    A positive integer is the effective maximum output tokens for each call.
    ``None`` means the client is deliberately unbounded. Clients that do not
    implement this companion protocol have an unknown limit.
    """

    @property
    def output_token_limit(self) -> int | None:
        """Return the effective per-call output-token limit."""
        ...


class SubcallConcurrencyProvider(Protocol):
    """Optional read-only effective batch-concurrency metadata.

    Built-in clients implement this companion protocol so the engine can
    reject a rollout whose immutable provenance disagrees with the transport.
    Existing third-party subcall clients remain compatible without it.
    """

    @property
    def subcall_concurrency(self) -> int:
        """Return the effective maximum number of in-flight batch items."""
        ...
