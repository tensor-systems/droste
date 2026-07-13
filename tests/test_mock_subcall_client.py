from concurrent.futures import ThreadPoolExecutor

import pytest

from droste import RLMConfig, SubcallBudgetExceeded, run_rlm
from droste.execution.context import ExecutionContext, create_execution_context
from droste.protocols.llm_client import TokenUsage
from droste.testing import MockEnvironment, MockLLMClient, MockResponse, MockSubcallClient


def _run_one_mock_subcall(mock: MockSubcallClient, context: ExecutionContext) -> None:
    root = MockLLMClient(
        responses=[
            MockResponse(
                text=(
                    "```python\n"
                    "llm_query('prompt')\n"
                    "answer['content'] = 'ok'\n"
                    "answer['ready'] = True\n"
                    "```"
                ),
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        ]
    )
    result = run_rlm(
        "test",
        environment=MockEnvironment(),
        root_llm=root,
        subcalls=mock,
        config=RLMConfig(max_iterations=1),
        context=context,
    )

    assert result.ready is True
    assert result.answer == "ok"


def test_mock_without_explicit_context_rebinds_for_sequential_runs() -> None:
    mock = MockSubcallClient()
    first = create_execution_context(max_calls=1)
    second = create_execution_context(max_calls=1)

    _run_one_mock_subcall(mock, first)
    _run_one_mock_subcall(mock, second)

    assert first.stats.calls_made == 1
    assert second.stats.calls_made == 1
    assert first.stats.successful_calls == 0
    assert second.stats.successful_calls == 0


def test_mock_with_explicit_context_keeps_it_when_run_rlm_binds_another() -> None:
    explicit = create_execution_context(max_calls=1)
    run_context = create_execution_context(max_calls=1)
    mock = MockSubcallClient(context=explicit)

    _run_one_mock_subcall(mock, run_context)

    assert explicit.stats.calls_made == 1
    assert explicit.stats.successful_calls == 0
    assert run_context.stats.calls_made == 0
    assert run_context.stats.successful_calls == 0


def test_mock_budget_reservation_is_atomic_and_empty_outputs_are_unsuccessful() -> None:
    context = create_execution_context(max_calls=2)
    mock = MockSubcallClient(context=context)

    assert mock.llm_batch(["one", "two"]) == ["", ""]
    with pytest.raises(SubcallBudgetExceeded, match="max subcalls exceeded"):
        mock.llm_query("over budget")

    assert context.stats.calls_made == 2
    assert context.stats.successful_calls == 0


def test_mock_budget_check_and_increment_are_thread_safe() -> None:
    context = create_execution_context(max_calls=8)
    mock = MockSubcallClient(context=context)

    def call(_: int) -> bool:
        try:
            mock.llm_query("prompt")
        except SubcallBudgetExceeded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=32) as executor:
        accepted = list(executor.map(call, range(64)))

    assert sum(accepted) == 8
    assert context.stats.calls_made == 8
    assert context.stats.successful_calls == 0
