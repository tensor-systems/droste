"""Extract-fallback path — the post-exhaustion synthesis call, and
the fix for it silently swallowing failures (found via a real user report:
raw debug `print()` output was shown as a final answer with zero trace of why
the synthesis call that should have replaced it never ran)."""

from droste import RLMConfig, run_rlm
from droste.protocols.llm_client import TokenUsage
from droste.testing import MockEnvironment, MockLLMClient, MockResponse, MockSubcallClient

_DEBUG_CODE = """```python\nprint('debug output, iteration {n}')\n```"""


def _non_ready_responses(n: int) -> list[MockResponse]:
    return [
        MockResponse(
            text=_DEBUG_CODE.format(n=i),
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        for i in range(n)
    ]


def test_extract_fallback_failure_surfaces_error_not_silent():
    """No response left for the extract call -> MockLLMClient raises. Before
    this fix, _extract_final_answer swallowed that silently and the caller
    couldn't tell "extraction succeeded" from "extraction failed" — both
    looked like extracted=False with no error. Now the failure must be
    visible on the result, and the raw fallback must still be there (never
    lose the user's partial progress just because synthesis failed)."""
    mock_llm = MockLLMClient(responses=_non_ready_responses(2))  # none left for extraction
    result = run_rlm(
        question="test",
        environment=MockEnvironment(),
        root_llm=mock_llm,
        subcalls=MockSubcallClient(),
        config=RLMConfig(max_iterations=2),
    )
    assert not result.ready
    assert result.extracted is False
    assert result.extract_error is not None
    assert result.extract_error.type == "RuntimeError"
    assert "debug output, iteration 1" in result.answer  # raw fallback preserved


def test_extract_fallback_empty_response_surfaces_as_error():
    """A blank/whitespace-only extraction response is just as much a failure
    as an exception — must not be silently treated as a valid (empty) answer."""
    responses = _non_ready_responses(2) + [
        MockResponse(
            text="   \n", usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        )
    ]
    mock_llm = MockLLMClient(responses=responses)
    result = run_rlm(
        question="test",
        environment=MockEnvironment(),
        root_llm=mock_llm,
        subcalls=MockSubcallClient(),
        config=RLMConfig(max_iterations=2),
    )
    assert result.extracted is False
    assert result.extract_error is not None
    assert result.extract_error.type == "EmptyExtraction"


def test_extract_fallback_success_has_no_error():
    """The success path must be unaffected by this fix: extract_error stays
    None, extracted is True, and the synthesized text (not raw stdout) wins."""
    responses = _non_ready_responses(2) + [
        MockResponse(
            text="A proper synthesized answer.",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    ]
    mock_llm = MockLLMClient(responses=responses)
    result = run_rlm(
        question="test",
        environment=MockEnvironment(),
        root_llm=mock_llm,
        subcalls=MockSubcallClient(),
        config=RLMConfig(max_iterations=2),
    )
    assert result.extracted is True
    assert result.extract_error is None
    assert result.answer == "A proper synthesized answer."
