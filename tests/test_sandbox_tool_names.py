"""The sandbox must answer to llm_query_batched.

That is the canonical batch-tool name in the RLM paper's reference
implementation (alexzhang13/rlm) and in dspy.RLM; models primed on RLM
literature call it first. Our OOLONG bench showed models burning iterations
on NameError discovering the local names, so the alias is contract.
"""

from droste import RLMConfig, run_rlm
from droste.protocols.llm_client import TokenUsage
from droste.testing import MockEnvironment, MockLLMClient, MockResponse, MockSubcallClient


def test_llm_query_batched_is_callable_in_sandbox():
    mock_llm = MockLLMClient(
        responses=[
            MockResponse(
                text=(
                    "```python\n"
                    "results = llm_query_batched(['a', 'b'])\n"
                    "answer['content'] = f'got {len(results)}'\n"
                    "answer['ready'] = True\n"
                    "```"
                ),
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        ]
    )
    result = run_rlm(
        question="test",
        environment=MockEnvironment(),
        root_llm=mock_llm,
        subcalls=MockSubcallClient(),
        config=RLMConfig(max_iterations=2),
    )
    assert result.ready
    assert result.answer == "got 2"
    assert result.error is None


def test_policy_recognizes_llm_query_batched_as_llm_call():
    from droste.policy import LLM_CALL_REGEX

    assert LLM_CALL_REGEX.search("results = llm_query_batched(prompts)")
    assert not LLM_CALL_REGEX.search("results = my_llm_query_batched_helper")
