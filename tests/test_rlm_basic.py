from rlm_core import RLMConfig, run_rlm
from rlm_core.testing import MockEnvironment, MockLLMClient, MockResponse, MockSubcallClient
from rlm_core.protocols.llm_client import TokenUsage


def test_run_rlm_basic():
    mock_llm = MockLLMClient(
        responses=[
            MockResponse(
                text="""```python\nanswer['content'] = 'ok'\nanswer['ready'] = True\n```""",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        ]
    )
    mock_env = MockEnvironment()
    mock_subcalls = MockSubcallClient()

    result = run_rlm(
        question="test",
        environment=mock_env,
        root_llm=mock_llm,
        subcalls=mock_subcalls,
        config=RLMConfig(max_iterations=1),
    )

    assert result.ready
    assert result.answer == "ok"
