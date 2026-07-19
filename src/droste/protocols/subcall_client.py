from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..exceptions import BatchItemError
from .llm_client import TokenUsage
from .subcall_capacity import SubcallInputCapacity


@dataclass(frozen=True, slots=True)
class SubcallQueryResult:
    """One query result paired with its per-request provider usage fact."""

    result: str
    usage: TokenUsage

    def __post_init__(self) -> None:
        if not isinstance(self.result, str):
            raise TypeError("subcall query result must be a string")
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("subcall query usage must be TokenUsage")


@dataclass(frozen=True, slots=True)
class SubcallBatchResult:
    """Ordered batch results, errors, and item-attributed provider usage."""

    results: tuple[str, ...]
    errors: tuple[dict[str, Any], ...]
    usage: tuple[TokenUsage, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "errors", tuple(dict(item) for item in self.errors))
        object.__setattr__(self, "usage", tuple(self.usage))
        if not all(isinstance(item, str) for item in self.results):
            raise TypeError("subcall batch results must be strings")
        if len(self.usage) != len(self.results):
            raise ValueError("subcall batch usage must align with results")
        if not all(isinstance(item, TokenUsage) for item in self.usage):
            raise TypeError("subcall batch usage must contain TokenUsage values")
        for item in self.errors:
            index = item.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("subcall batch errors require integer indexes")
            if index < 0 or index >= len(self.results):
                raise ValueError("subcall batch error index is out of range")


class SubcallBatchFailure(RuntimeError):
    """A fail-fast batch error carrying the provider usage collected so far.

    The capability broker consumes ``result.usage`` before re-raising
    ``cause``. Plain ``llm_batch`` callers still receive the original cause.
    """

    def __init__(self, result: SubcallBatchResult, cause: Exception) -> None:
        if not isinstance(result, SubcallBatchResult):
            raise TypeError("subcall batch failure result must be SubcallBatchResult")
        if not isinstance(cause, Exception):
            raise TypeError("subcall batch failure cause must be an Exception")
        super().__init__(str(cause))
        self.result = result
        self.cause = cause


def structured_subcall_errors(
    errors: tuple[Exception | None, ...],
) -> tuple[dict[str, Any], ...]:
    """Project ordered item exceptions into the public batch-error shape."""

    structured: list[dict[str, Any]] = []
    for index, error in enumerate(errors):
        if error is None:
            continue
        item: dict[str, Any] = {"index": index, "error": str(error)}
        if isinstance(error, BatchItemError):
            details = error.details.to_dict()
            if details:
                item["details"] = details
        structured.append(item)
    return tuple(structured)


def fail_fast_subcall_batch(
    results: tuple[str, ...],
    errors: tuple[Exception | None, ...],
    usage: tuple[TokenUsage, ...],
) -> SubcallBatchResult:
    """Build a typed batch result or preserve it beside the first item error."""

    if len(errors) != len(results):
        raise ValueError("subcall batch errors must align with results")
    result = SubcallBatchResult(results, structured_subcall_errors(errors), usage)
    for error in errors:
        if error is not None:
            raise SubcallBatchFailure(result, error) from None
    return result


class SubcallClient(Protocol):
    """Interface for recursive sub-LLM calls.

    Implementations attached to an :class:`ExecutionContext` reserve
    ``calls_made`` before dispatch and increment ``successful_calls`` only
    after an item returns a usable text response.

    Implementations may also implement the separate optional
    :class:`SubcallOutputTokenLimitProvider` and
    :class:`SubcallInputCapacityProvider` protocols. They stay separate so
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


class SubcallUsageProvider(Protocol):
    """Optional per-invocation usage companion for budget reconciliation.

    Implementations return usage in caller order. An item whose provider usage
    is missing or malformed carries ``TokenUsage.unavailable()``; consumers
    must then retain the conservative reservation instead of estimating from
    visible result text or a shared cumulative counter.
    """

    def llm_query_with_usage(self, prompt: str, context: str = "") -> SubcallQueryResult: ...

    def llm_batch_with_usage(
        self, prompts: list[str], contexts: list[str] | None = None
    ) -> SubcallBatchResult: ...

    def llm_batch_with_errors_and_usage(
        self, prompts: list[str], contexts: list[str] | None = None
    ) -> SubcallBatchResult: ...


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


class SubcallInputCapacityProvider(Protocol):
    """Optional read-only effective input-capacity metadata.

    The immutable value is the effective usable caller-payload bound across
    the complete adapter/transport/model path. ``unbounded`` is valid only
    when arbitrary payloads are guaranteed, for example through transparent
    chunking. Clients that do not implement this companion protocol have
    unknown capacity; the engine never substitutes a guessed context window.
    """

    @property
    def input_token_capacity(self) -> SubcallInputCapacity:
        """Return the effective per-call input-token capacity."""
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
