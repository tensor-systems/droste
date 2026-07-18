"""Terminal extraction path — the bounded synthesis call, and
the fix for it silently swallowing failures (found via a real user report:
raw debug `print()` output was shown as a final answer with zero trace of why
the synthesis call that should have replaced it never ran)."""

from typing import Any

from droste import AnthropicClient, RLMConfig
from droste.execution import create_execution_context
from droste.loop.rlm import _extract_final_answer
from droste.loop.trajectory import EXECUTION_STATUS_SUCCESS, IterationRecord
from droste.prompts import load_builtin_prompt_catalog, resolve_prompt_pack
from droste.protocols.llm_client import TokenUsage
from droste.testing import MockLLMClient, MockResponse

_DEBUG_CODE = """```python\nprint('debug output, iteration {n}')\n```"""


def _responses(n: int) -> list[MockResponse]:
    return [
        MockResponse(
            text=_DEBUG_CODE.format(n=i),
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        for i in range(n)
    ]


_TRAJECTORY = [
    IterationRecord(
        iteration=1,
        llm_input=[{"role": "user", "content": "test"}],
        llm_output=_DEBUG_CODE.format(n=1),
        code_executed="print('useful evidence')",
        execution_result="useful evidence",
        tokens_used=2,
        execution_status=EXECUTION_STATUS_SUCCESS,
    )
]


def _extract(responses: list[MockResponse]):
    pack = resolve_prompt_pack(
        model="",
        profile="full",
        engine_catalog=load_builtin_prompt_catalog(),
    ).pack
    return _extract_final_answer(
        "test",
        "partial draft",
        _TRAJECTORY,
        MockLLMClient(responses=responses),
        RLMConfig(),
        create_execution_context(),
        pack,
    )


def test_extract_fallback_failure_surfaces_error_not_silent():
    """No response left for the extract call -> MockLLMClient raises. Before
    this fix, _extract_final_answer swallowed that silently and the caller
    couldn't tell "extraction succeeded" from "extraction failed" — both
    looked like extracted=False with no error. Now the failure must be
    visible on the result, and the raw fallback must still be there (never
    lose the user's partial progress just because synthesis failed)."""
    text, error = _extract([])
    assert text == ""
    assert error is not None
    assert error.type == "RuntimeError"


def test_extract_fallback_empty_response_surfaces_as_error():
    """A blank/whitespace-only extraction response is just as much a failure
    as an exception — must not be silently treated as a valid (empty) answer."""
    responses = [
        MockResponse(
            text="   \n", usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        )
    ]
    text, error = _extract(responses)
    assert text == ""
    assert error is not None
    assert error.type == "EmptyExtraction"


def test_extract_fallback_unable_sentinel_is_not_success():
    responses = [
        MockResponse(
            text="unable to determine from the work so far",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    ]
    text, error = _extract(responses)
    assert text == ""
    assert error is not None
    assert error.type == "InsufficientEvidence"


def test_extract_fallback_decorated_unable_sentinel_is_not_success():
    responses = [
        MockResponse(
            text='"Unable to determine from the work so far."',
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    ]
    text, error = _extract(responses)
    assert text == ""
    assert error is not None
    assert error.type == "InsufficientEvidence"


def test_extract_fallback_markdown_unable_sentinel_is_not_success():
    responses = [
        MockResponse(
            text="**Unable to determine from the work so far.**",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    ]
    text, error = _extract(responses)
    assert text == ""
    assert error is not None
    assert error.type == "InsufficientEvidence"


def test_extract_fallback_unicode_ellipsis_unable_sentinel_is_not_success():
    responses = [
        MockResponse(
            text="Unable to determine from the work so far…",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    ]
    text, error = _extract(responses)
    assert text == ""
    assert error is not None
    assert error.type == "InsufficientEvidence"


def test_extract_fallback_success_has_no_error():
    """The success path must be unaffected by this fix: extract_error stays
    None, extracted is True, and the synthesized text (not raw stdout) wins."""
    responses = [
        MockResponse(
            text="A proper synthesized answer.",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    ]
    text, error = _extract(responses)
    assert error is None
    assert text == "A proper synthesized answer."


def test_extract_fallback_anthropic_system_uses_legacy_string_wire_form(monkeypatch):
    payloads: list[dict[str, Any]] = []
    client = AnthropicClient(model="claude-test", api_key="sk-ant-test")

    def complete(payload: dict[str, Any]) -> dict[str, Any]:
        payloads.append(payload)
        return {
            "content": [{"type": "text", "text": "A synthesized answer."}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(client._transport, "complete", complete)
    pack = resolve_prompt_pack(
        model="",
        profile="full",
        engine_catalog=load_builtin_prompt_catalog(),
    ).pack

    text, error = _extract_final_answer(
        "test",
        "partial draft",
        _TRAJECTORY,
        client,
        RLMConfig(root_model="claude-test"),
        create_execution_context(),
        pack,
    )

    assert error is None
    assert text == "A synthesized answer."
    assert isinstance(payloads[0]["system"], str)
    assert "cache_control" not in payloads[0]["system"]
