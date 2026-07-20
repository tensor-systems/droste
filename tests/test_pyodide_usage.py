from __future__ import annotations

import json

import pytest

from droste import LLMUsageFailure, TokenUsage, UsageObservationBasis
from droste.substrates.pyodide import BridgedLLMClient, _token_usage


def _usage_mapping(
    *,
    input_tokens: object = 7,
    output_tokens: object = 3,
    total_tokens: object = 10,
    reasoning_tokens: object = 0,
    observation_basis: object = "exact",
    **extra: object,
) -> dict[str, object]:
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "observation_basis": observation_basis,
    }
    usage.update(extra)
    return usage


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (None, TokenUsage.unavailable()),
        ({}, TokenUsage.unavailable()),
        ({"input_tokens": 1, "output_tokens": 2}, TokenUsage(1, 2, 0)),
        ({"input_tokens": -1, "output_tokens": 2, "total_tokens": 3}, TokenUsage(0, 2, 3)),
        ({"input_tokens": 1, "output_tokens": 2, "total_tokens": 2}, TokenUsage(1, 2, 2)),
        ({"input_tokens": True, "output_tokens": 2, "total_tokens": 3}, TokenUsage(0, 2, 3)),
        ({"input_tokens": 1, "output_tokens": "bad", "total_tokens": 9}, TokenUsage(1, 0, 9)),
    ],
)
def test_token_usage_partial_or_malformed_preserves_independent_counts(
    payload: object, expected: TokenUsage
) -> None:
    assert _token_usage(payload) == expected
    assert _token_usage(payload).exact is False


@pytest.mark.parametrize(
    ("raw_basis", "expected_basis"),
    [
        ("exact", UsageObservationBasis.EXACT),
        ("estimated_categories", UsageObservationBasis.ESTIMATED_CATEGORIES),
        ("incomplete", UsageObservationBasis.INCOMPLETE),
        (None, UsageObservationBasis.INCOMPLETE),
        ("wrong", UsageObservationBasis.INCOMPLETE),
    ],
)
def test_token_usage_requires_canonical_modelrelay_observation_basis(
    raw_basis: object,
    expected_basis: UsageObservationBasis,
) -> None:
    mapping = _usage_mapping(observation_basis=raw_basis)
    if raw_basis is None:
        mapping.pop("observation_basis")

    usage = _token_usage(mapping)

    assert usage.observation_basis is expected_basis
    assert usage.reasoning_tokens == 0


@pytest.mark.parametrize("reasoning_tokens", [None, "0"])
def test_token_usage_complete_basis_requires_explicit_valid_reasoning(
    reasoning_tokens: object,
) -> None:
    mapping = _usage_mapping(reasoning_tokens=reasoning_tokens)
    if reasoning_tokens is None:
        mapping.pop("reasoning_tokens")

    usage = _token_usage(mapping)

    assert usage.observation_basis is UsageObservationBasis.INCOMPLETE


def test_token_usage_declared_exact_all_zero_categories_is_exact() -> None:
    usage = _token_usage(
        _usage_mapping(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cache_read_input_tokens=0,
            cache_write_input_tokens=0,
        )
    )

    assert usage.exact is True
    assert usage.total_tokens == 0


def test_bridged_client_preserves_provider_total_with_hidden_reasoning() -> None:
    def host_fetch(_method: str, _url: str, _headers: str, _body: str) -> str:
        return json.dumps(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "done"}],
                    }
                ],
                "usage": _usage_mapping(
                    input_tokens=2,
                    output_tokens=3,
                    total_tokens=19,
                    reasoning_tokens=3,
                ),
            }
        )

    text, usage = BridgedLLMClient(host_fetch).responses_create(
        [{"role": "user", "content": "q"}],
        model="model",
        return_usage=True,
    )

    assert text == "done"
    assert usage.exact is True
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (2, 3, 19)
    assert usage.reasoning_tokens == 3


def test_bridged_client_preserves_usage_when_output_parsing_fails() -> None:
    def host_fetch(_method: str, _url: str, _headers: str, _body: str) -> str:
        return json.dumps(
            {
                "output": [None],
                "usage": _usage_mapping(total_tokens=19),
            }
        )

    with pytest.raises(LLMUsageFailure) as raised:
        BridgedLLMClient(host_fetch).responses_create(
            [{"role": "user", "content": "q"}],
            model="model",
            return_usage=True,
        )

    assert raised.value.usage == TokenUsage(
        7,
        3,
        19,
        observation_basis=UsageObservationBasis.EXACT,
    )
    assert isinstance(raised.value.cause, AttributeError)


def test_bridged_client_preserves_partial_usage_when_output_parsing_fails() -> None:
    def host_fetch(_method: str, _url: str, _headers: str, _body: str) -> str:
        return json.dumps(
            {
                "output": [None],
                "usage": _usage_mapping(output_tokens="bad", total_tokens=19),
            }
        )

    with pytest.raises(LLMUsageFailure) as raised:
        BridgedLLMClient(host_fetch).responses_create(
            [{"role": "user", "content": "q"}],
            model="model",
            return_usage=True,
        )

    assert raised.value.usage == TokenUsage(7, 0, 19)
