from __future__ import annotations

import json

import pytest

from droste import LLMUsageFailure, TokenUsage
from droste.substrates.pyodide import BridgedLLMClient, _token_usage


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"input_tokens": 1, "output_tokens": 2},
        {"input_tokens": -1, "output_tokens": 2, "total_tokens": 3},
        {"input_tokens": 1, "output_tokens": 2, "total_tokens": 2},
        {"input_tokens": True, "output_tokens": 2, "total_tokens": 3},
    ],
)
def test_token_usage_missing_or_malformed_is_unavailable(payload: object) -> None:
    assert _token_usage(payload).exact is False


def test_token_usage_complete_all_zero_usage_is_exact() -> None:
    usage = _token_usage({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

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
                "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 19},
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


def test_bridged_client_preserves_usage_when_output_parsing_fails() -> None:
    def host_fetch(_method: str, _url: str, _headers: str, _body: str) -> str:
        return json.dumps(
            {
                "output": [None],
                "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 19},
            }
        )

    with pytest.raises(LLMUsageFailure) as raised:
        BridgedLLMClient(host_fetch).responses_create(
            [{"role": "user", "content": "q"}],
            model="model",
            return_usage=True,
        )

    assert raised.value.usage == TokenUsage(7, 3, 19, exact=True)
    assert isinstance(raised.value.cause, AttributeError)
