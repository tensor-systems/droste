from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

# Reserved for the loop-to-client handoff. It is never part of the canonical
# transcript and must be consumed or stripped before an API payload is built.
CACHE_ANCHOR_MARKER = "_droste_cache_anchor"


class UsageObservationBasis(StrEnum):
    """Fidelity of the billed usage categories observed for one provider call."""

    UNAVAILABLE = "unavailable"
    INCOMPLETE = "incomplete"
    ESTIMATED_CATEGORIES = "estimated_categories"
    EXACT = "exact"


def aggregate_observation_basis(
    left: UsageObservationBasis,
    right: UsageObservationBasis,
) -> UsageObservationBasis:
    """Return the lossless fidelity summary for two additive observations."""

    if not isinstance(left, UsageObservationBasis) or not isinstance(right, UsageObservationBasis):
        raise TypeError("usage observation basis must be UsageObservationBasis")
    if left is right:
        return left
    if UsageObservationBasis.INCOMPLETE in (left, right):
        return UsageObservationBasis.INCOMPLETE
    if UsageObservationBasis.UNAVAILABLE in (left, right):
        return UsageObservationBasis.INCOMPLETE
    return UsageObservationBasis.ESTIMATED_CATEGORIES


@dataclass(init=False)
class TokenUsage:
    """Token usage from one LLM call with one canonical observation basis.

    ``exact`` remains a derived property for settlement callers. The optional
    constructor argument of the same name is normalized immediately and is not
    stored, so older embedders do not create a second completeness authority.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0
    observation_basis: UsageObservationBasis = UsageObservationBasis.INCOMPLETE

    def __init__(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        exact: bool | None = None,
        *,
        reasoning_tokens: int = 0,
        observation_basis: UsageObservationBasis | None = None,
    ) -> None:
        if exact is not None and observation_basis is not None:
            raise TypeError("provide observation_basis or exact, not both")
        if exact is not None and not isinstance(exact, bool):
            raise TypeError("token usage exact must be a bool")
        if observation_basis is None:
            observation_basis = (
                UsageObservationBasis.EXACT if exact else UsageObservationBasis.INCOMPLETE
            )
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_creation_tokens = cache_creation_tokens
        self.reasoning_tokens = reasoning_tokens
        self.observation_basis = observation_basis
        self.__post_init__()

    def __post_init__(self) -> None:
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cache_read_tokens",
            "cache_creation_tokens",
            "reasoning_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"token usage {name} must be a non-negative integer")
        if not isinstance(self.observation_basis, UsageObservationBasis):
            raise TypeError("token usage observation_basis must be UsageObservationBasis")
        if self.observation_basis is UsageObservationBasis.UNAVAILABLE and any(
            (
                self.prompt_tokens,
                self.completion_tokens,
                self.total_tokens,
                self.cache_read_tokens,
                self.cache_creation_tokens,
                self.reasoning_tokens,
            )
        ):
            raise ValueError("unavailable token usage cannot carry observed counters")
        categories_complete = self.observation_basis in (
            UsageObservationBasis.EXACT,
            UsageObservationBasis.ESTIMATED_CATEGORIES,
        )
        if categories_complete and self.total_tokens < self.prompt_tokens + self.completion_tokens:
            raise ValueError("complete token usage total cannot be less than its parts")
        if (
            categories_complete
            and self.cache_read_tokens + self.cache_creation_tokens > self.prompt_tokens
        ):
            raise ValueError("complete cache token breakdown cannot exceed prompt tokens")
        if categories_complete and self.reasoning_tokens > self.completion_tokens:
            raise ValueError("complete reasoning tokens cannot exceed completion tokens")

    @property
    def exact(self) -> bool:
        """Whether all billed usage categories were observed exactly."""

        return self.observation_basis is UsageObservationBasis.EXACT

    @classmethod
    def unavailable(cls) -> TokenUsage:
        """Return the explicit fact that trustworthy provider usage was absent."""

        return cls(0, 0, 0, observation_basis=UsageObservationBasis.UNAVAILABLE)


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


def token_usage_from_mapping(
    value: Any,
    *,
    prompt_names: tuple[str, ...] = ("input_tokens",),
    completion_names: tuple[str, ...] = ("output_tokens",),
    cache_read_names: tuple[str, ...] = ("cache_read_input_tokens",),
    cache_creation_names: tuple[str, ...] = (
        "cache_write_input_tokens",
        "cache_creation_input_tokens",
    ),
    reasoning_names: tuple[str, ...] = ("reasoning_tokens",),
    observation_basis_name: str | None = None,
    require_reasoning: bool = False,
) -> TokenUsage:
    """Preserve independently valid provider counters from one usage mapping.

    A malformed or absent counter becomes zero only as a typed placeholder;
    ``observation_basis`` distinguishes that placeholder from a reported zero.
    Alias groups preserve the first valid value in name order but remain inexact when
    another present alias is malformed or disagrees. Cache counters are
    optional exact-zero breakdowns inside inclusive prompt usage; a malformed
    present cache field or a cache sum above the prompt total makes the tuple
    partial without discarding independently valid siblings.
    """

    if not isinstance(value, dict) or not value:
        return TokenUsage.unavailable()

    def counter(names: tuple[str, ...], *, optional: bool = False) -> tuple[int, bool]:
        present = [value[name] for name in names if name in value]
        if optional and not present:
            return 0, True
        valid = [
            item
            for item in present
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0
        ]
        if not valid:
            return 0, False
        selected = valid[0]
        complete = len(valid) == len(present) and all(item == selected for item in valid)
        return selected, complete

    def zero_if_present(names: tuple[str, ...]) -> bool:
        present = [value[name] for name in names if name in value]
        return not present or all(
            isinstance(item, int) and not isinstance(item, bool) and item == 0 for item in present
        )

    prompt, prompt_complete = counter(prompt_names)
    completion, completion_complete = counter(completion_names)
    total, total_complete = counter(("total_tokens",))
    cache_read, cache_read_complete = counter(cache_read_names, optional=True)
    cache_creation, cache_creation_complete = counter(cache_creation_names, optional=True)
    reasoning, reasoning_complete = counter(reasoning_names, optional=not require_reasoning)
    structurally_exact = (
        prompt_complete
        and completion_complete
        and total_complete
        and cache_read_complete
        and cache_creation_complete
        and reasoning_complete
        and total >= prompt + completion
        and cache_read + cache_creation <= prompt
        and reasoning <= completion
    )
    if observation_basis_name is None:
        observation_basis = (
            UsageObservationBasis.EXACT if structurally_exact else UsageObservationBasis.INCOMPLETE
        )
    else:
        raw_basis = value.get(observation_basis_name)
        try:
            requested_basis = UsageObservationBasis(raw_basis)
        except (TypeError, ValueError):
            requested_basis = UsageObservationBasis.INCOMPLETE
        if requested_basis is UsageObservationBasis.UNAVAILABLE:
            if all(
                zero_if_present(names)
                for names in (
                    prompt_names,
                    completion_names,
                    ("total_tokens",),
                    cache_read_names,
                    cache_creation_names,
                    reasoning_names,
                )
            ):
                return TokenUsage.unavailable()
            observation_basis = UsageObservationBasis.INCOMPLETE
        elif requested_basis is UsageObservationBasis.EXACT and not structurally_exact:
            observation_basis = UsageObservationBasis.INCOMPLETE
        elif (
            requested_basis is UsageObservationBasis.ESTIMATED_CATEGORIES and not structurally_exact
        ):
            observation_basis = UsageObservationBasis.INCOMPLETE
        else:
            observation_basis = requested_basis
    return TokenUsage(
        prompt,
        completion,
        total,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        reasoning_tokens=reasoning,
        observation_basis=observation_basis,
    )


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
