from __future__ import annotations

import pytest

from droste import TokenUsage, UsageObservationBasis
from droste.execution.context import create_execution_context
from droste.protocols.llm_client import aggregate_observation_basis


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            UsageObservationBasis.EXACT,
            UsageObservationBasis.EXACT,
            UsageObservationBasis.EXACT,
        ),
        (
            UsageObservationBasis.ESTIMATED_CATEGORIES,
            UsageObservationBasis.ESTIMATED_CATEGORIES,
            UsageObservationBasis.ESTIMATED_CATEGORIES,
        ),
        (
            UsageObservationBasis.EXACT,
            UsageObservationBasis.ESTIMATED_CATEGORIES,
            UsageObservationBasis.ESTIMATED_CATEGORIES,
        ),
        (
            UsageObservationBasis.EXACT,
            UsageObservationBasis.INCOMPLETE,
            UsageObservationBasis.INCOMPLETE,
        ),
        (
            UsageObservationBasis.EXACT,
            UsageObservationBasis.UNAVAILABLE,
            UsageObservationBasis.INCOMPLETE,
        ),
        (
            UsageObservationBasis.UNAVAILABLE,
            UsageObservationBasis.UNAVAILABLE,
            UsageObservationBasis.UNAVAILABLE,
        ),
    ],
)
def test_aggregate_observation_basis_preserves_weakest_fidelity(
    left: UsageObservationBasis,
    right: UsageObservationBasis,
    expected: UsageObservationBasis,
) -> None:
    assert aggregate_observation_basis(left, right) is expected
    assert aggregate_observation_basis(right, left) is expected


def test_token_usage_exact_is_derived_from_observation_basis() -> None:
    exact = TokenUsage(
        7,
        4,
        11,
        reasoning_tokens=3,
        observation_basis=UsageObservationBasis.EXACT,
    )
    estimated = TokenUsage(
        7,
        4,
        11,
        reasoning_tokens=3,
        observation_basis=UsageObservationBasis.ESTIMATED_CATEGORIES,
    )

    assert exact.exact
    assert not estimated.exact
    assert exact.core_complete
    assert estimated.core_complete


@pytest.mark.parametrize(
    "basis",
    [UsageObservationBasis.UNAVAILABLE, UsageObservationBasis.INCOMPLETE],
)
def test_token_usage_incomplete_core_counters_are_not_settlement_authority(
    basis: UsageObservationBasis,
) -> None:
    usage = (
        TokenUsage.unavailable()
        if basis is UsageObservationBasis.UNAVAILABLE
        else TokenUsage(7, 4, 11, observation_basis=basis)
    )

    assert not usage.core_complete


def test_token_usage_legacy_exact_input_normalizes_without_second_authority() -> None:
    usage = TokenUsage(7, 4, 11, exact=True)

    assert usage.observation_basis is UsageObservationBasis.EXACT
    assert usage.exact
    assert "exact=" not in repr(usage)


def test_token_usage_rejects_conflicting_completeness_inputs() -> None:
    with pytest.raises(TypeError, match="observation_basis or exact"):
        TokenUsage(
            7,
            4,
            11,
            exact=True,
            observation_basis=UsageObservationBasis.EXACT,
        )


@pytest.mark.parametrize(
    "usage",
    [
        TokenUsage.unavailable(),
        TokenUsage(7, 4, 11, observation_basis=UsageObservationBasis.INCOMPLETE),
        TokenUsage(
            7,
            4,
            11,
            reasoning_tokens=3,
            observation_basis=UsageObservationBasis.ESTIMATED_CATEGORIES,
        ),
    ],
)
def test_nonexact_observations_remain_valid_partial_evidence(usage: TokenUsage) -> None:
    assert not usage.exact


@pytest.mark.parametrize(
    "basis",
    [UsageObservationBasis.EXACT, UsageObservationBasis.ESTIMATED_CATEGORIES],
)
def test_complete_reasoning_category_must_fit_inside_completion(
    basis: UsageObservationBasis,
) -> None:
    with pytest.raises(ValueError, match="reasoning tokens cannot exceed completion"):
        TokenUsage(
            7,
            4,
            11,
            reasoning_tokens=5,
            observation_basis=basis,
        )


def test_execution_context_revalidates_mutated_reasoning_counter() -> None:
    usage = TokenUsage(7, 4, 11, exact=True)
    usage.reasoning_tokens = -1

    with pytest.raises(ValueError, match="reasoning_tokens must be a non-negative integer"):
        create_execution_context().record_root_usage(usage)


def test_execution_context_retains_internal_reasoning_totals() -> None:
    context = create_execution_context()

    context.record_root_usage(TokenUsage(7, 4, 11, reasoning_tokens=3, exact=True))
    context.record_subcall_usage(TokenUsage(5, 2, 7, reasoning_tokens=1, exact=True))

    assert context.stats.root_reasoning_tokens == 3
    assert context.stats.subcall_reasoning_tokens == 1
