import math

import pytest

from droste import RLMConfig, run_rlm
from droste.loop.step import copy_answer_metadata
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
    assert result.answer_metadata == {}


def test_non_contract_plain_response_is_returned_as_answer():
    response = "A direct answer without an executable code block."
    result = run_rlm(
        question="test",
        environment=MockEnvironment(),
        root_llm=MockLLMClient(
            responses=[
                MockResponse(
                    text=response,
                    usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )
            ]
        ),
        subcalls=MockSubcallClient(),
        config=RLMConfig(max_iterations=1, enforce_contract=False),
    )

    assert result.answer == response
    assert result.ready is False
    assert result.trajectory == []


def test_run_rlm_preserves_confirmed_json_answer_metadata():
    mock_llm = MockLLMClient(
        responses=[
            MockResponse(
                text=(
                    "```python\n"
                    "answer['content'] = 'ok'\n"
                    "answer['metadata'] = {\n"
                    "    'evidence_ids': ['result-1'],\n"
                    "    'artifact': {'kind': 'table', 'rows': 3},\n"
                    "    'confidence': 0.9,\n"
                    "}\n"
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
        config=RLMConfig(max_iterations=1),
    )

    assert result.ready
    assert result.answer_metadata == {
        "evidence_ids": ["result-1"],
        "artifact": {"kind": "table", "rows": 3},
        "confidence": 0.9,
    }


def test_run_rlm_repairs_non_json_answer_metadata():
    mock_llm = MockLLMClient(
        responses=[
            MockResponse(
                text=(
                    "```python\nanswer['content'] = 'draft'\n"
                    "answer['metadata'] = {'bad': {1}}\nanswer['ready'] = True\n```"
                ),
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
            MockResponse(
                text=(
                    "```python\nanswer['metadata'] = {'fixed': True}\nanswer['ready'] = True\n```"
                ),
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
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
    assert result.answer_metadata == {"fixed": True}
    assert result.iterations == 1


@pytest.mark.parametrize(
    "metadata",
    [
        ["not", "an", "object"],
        {"value": math.inf},
        {"value": 1 << 54},
    ],
)
def test_copy_answer_metadata_rejects_nonportable_values(metadata):
    with pytest.raises(ValueError):
        copy_answer_metadata({"metadata": metadata})


def test_copy_answer_metadata_rejects_cycles():
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="reference cycle"):
        copy_answer_metadata({"metadata": {"cycle": cycle}})


def test_copy_answer_metadata_rejects_non_string_object_keys():
    with pytest.raises(ValueError, match="non-string object key"):
        copy_answer_metadata({"metadata": {1: "value"}})


def test_copy_answer_metadata_rejects_excessive_depth():
    nested = []
    for _ in range(101):
        nested = [nested]
    with pytest.raises(ValueError, match="depth limit"):
        copy_answer_metadata({"metadata": {"nested": nested}})


def test_copy_answer_metadata_rejects_oversized_values():
    with pytest.raises(ValueError, match="65536-byte limit"):
        copy_answer_metadata({"metadata": {"value": "x" * 65_536}})


def test_copy_answer_metadata_bounds_shared_subtree_traversal():
    shared = [0]
    for _ in range(40):
        shared = [shared, shared]
    with pytest.raises(ValueError, match="node limit"):
        copy_answer_metadata({"metadata": {"shared": shared}})


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
    assert [entry.execution_status for entry in result.trajectory] == ["success"]


def test_failed_rebound_answer_cannot_confirm_through_noop_repair():
    """A failed block may rebind answer before raising. The exception path
    must adopt that content but revoke its readiness, so a successful no-op
    repair cannot make the failed block look confirmed."""
    mock_llm = MockLLMClient(
        responses=[
            MockResponse(
                text=(
                    "```python\n"
                    "answer = {'content': 'draft from failed block', 'ready': True}\n"
                    "raise ValueError('boom')\n"
                    "```"
                ),
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            ),
            MockResponse(
                text="""```python\npass\n```""",
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
        config=RLMConfig(max_iterations=3),
    )

    assert result.ready is True
    assert result.answer == "draft from failed block"
    assert result.iterations == 2
