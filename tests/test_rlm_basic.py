from droste import RLMConfig, run_rlm
from droste.protocols.llm_client import TokenUsage
from droste.testing import MockEnvironment, MockLLMClient, MockResponse, MockSubcallClient


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
    assert result.extracted is False


def test_run_rlm_rebound_answer_dict_registers_ready():
    """Sandbox code that REBINDS `answer` (instead of mutating in place) must
    still terminate the loop — the loop reads the current binding after each
    execution, not the dict it captured up front. Regression: a ready rebound
    answer previously burned every remaining iteration (found via TAG-Bench)."""
    mock_llm = MockLLMClient(
        responses=[
            MockResponse(
                text="""```python\nanswer = {'content': 'rebound', 'ready': True}\n```""",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
            MockResponse(
                text="""```python\nanswer['ready'] = True\n```""",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        ]
    )
    result = run_rlm(
        question="test",
        environment=MockEnvironment(),
        root_llm=mock_llm,
        subcalls=MockSubcallClient(),
        config=RLMConfig(max_iterations=5),
    )
    assert result.ready
    assert result.answer == "rebound"
    assert result.iterations == 1  # terminated on the rebind, no burn


def test_run_rlm_non_dict_rebind_is_normalized_not_fatal():
    """`answer = "plain string"` must not break the loop's dict contract: the
    value is preserved as content, readiness is not guessed, and the next
    iteration can still set answer['ready'] on a dict."""
    mock_llm = MockLLMClient(
        responses=[
            MockResponse(
                text="""```python\nanswer = '42'\n```""",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
            MockResponse(
                text="""```python\nanswer['ready'] = True\n```""",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        ]
    )
    result = run_rlm(
        question="test",
        environment=MockEnvironment(),
        root_llm=mock_llm,
        subcalls=MockSubcallClient(),
        config=RLMConfig(max_iterations=5),
    )
    assert result.ready
    assert result.answer == "42"
    assert result.iterations == 2


def test_run_rlm_rebound_answer_in_repaired_code_registers_ready():
    """The repaired-code execution path (after an execution error) must also
    observe a rebound `answer` — it re-executes and runs the same contract
    checks, so a stale capture there burns iterations identically."""
    mock_llm = MockLLMClient(
        responses=[
            MockResponse(
                text="""```python\nraise ValueError('boom')\n```""",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
            MockResponse(
                text="""```python\nanswer = {'content': 'fixed', 'ready': True}\n```""",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
            MockResponse(
                text="""```python\nanswer['ready'] = True\n```""",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
        ]
    )
    result = run_rlm(
        question="test",
        environment=MockEnvironment(),
        root_llm=mock_llm,
        subcalls=MockSubcallClient(),
        config=RLMConfig(max_iterations=5),
    )
    assert result.ready
    assert result.answer == "fixed"
    assert result.iterations == 1  # repair happened within iteration 1
