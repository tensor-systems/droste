"""Wall-clock deadline exhaustion routes to the terminal extract fallback.

A run that spent `budget.wall_ms` used to end through `early_result`, which
hands the host a fatal `error` alongside whatever partial answer existed.
Hosts treat `result.error` as fatal and discard the run — so every iteration
of real work was thrown away over a *time* verdict, which says nothing about
whether that work was any good. Deadline exhaustion now takes the same
terminal handoff as an exhausted iteration budget and comes back as a
best-effort answer.
"""

import time

from droste import Budget, RLMConfig, run_rlm
from droste.exceptions import RLMError
from droste.execution import create_execution_context
from droste.loop.rlm import _extract_final_answer, _is_deadline_error
from droste.loop.trajectory import EXECUTION_STATUS_SUCCESS, IterationRecord
from droste.prompts import load_builtin_prompt_catalog, resolve_prompt_pack
from droste.protocols.llm_client import TokenUsage
from droste.testing import MockEnvironment, MockLLMClient, MockResponse, MockSubcallClient

_TRAJECTORY = [
    IterationRecord(
        iteration=1,
        llm_input=[{"role": "user", "content": "test"}],
        llm_output="```python\nprint('useful evidence')\n```",
        code_executed="print('useful evidence')",
        execution_result="useful evidence",
        tokens_used=2,
        execution_status=EXECUTION_STATUS_SUCCESS,
    )
]


def _usage() -> TokenUsage:
    return TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, exact=True)


def _deadline_error(resource: str = "wall_ms") -> RLMError:
    return RLMError(
        type="BudgetExhausted",
        message="budget exhausted",
        details={"resource": resource, "requested": 1, "remaining": 0},
    )


def _pack():
    return resolve_prompt_pack(
        model="", profile="full", engine_catalog=load_builtin_prompt_catalog()
    ).pack


def test_is_deadline_error_discriminates_resource():
    """Only wall-clock exhaustion reroutes. Token exhaustion must stay fatal:
    there is no budget left to pay for the extract call it would trigger."""
    assert _is_deadline_error(_deadline_error()) is True
    assert _is_deadline_error(_deadline_error("tokens")) is False
    assert _is_deadline_error(_deadline_error("subcalls")) is False
    assert _is_deadline_error(RLMError(type="PolicyError", message="x")) is False
    assert _is_deadline_error(None) is False
    # Details absent entirely must neither crash nor match.
    assert _is_deadline_error(RLMError(type="BudgetExhausted", message="x")) is False


def test_extract_call_survives_the_deadline_it_is_recovering_from():
    """The crux. `call_root` reserves with through_deadline=True, which refuses
    once remaining.wall_ms hits 0 — so without the exemption the extract pass
    is unreachable in exactly the case it exists to serve."""
    context = create_execution_context(budget=Budget(wall_ms=1))
    time.sleep(0.01)  # guarantee the shared run deadline is spent
    assert context.ledger.snapshot().remaining.wall_ms == 0

    text, error = _extract_final_answer(
        "test",
        "partial draft",
        _TRAJECTORY,
        MockLLMClient(responses=[MockResponse(text="synthesized answer", usage=_usage())]),
        RLMConfig(),
        context,
        _pack(),
    )

    assert error is None
    assert text == "synthesized answer"


def test_deadline_exhaustion_yields_extracted_answer_not_fatal_error():
    """End to end: iteration 1 does real work and burns the deadline, so
    iteration 2's root call trips wall_ms. The run must return a best-effort
    answer with the deadline preserved as diagnostic provenance."""
    responses = [
        MockResponse(
            text=(
                "```python\n"
                "import time\n"
                "answer['content'] = 'partial finding'\n"
                "time.sleep(0.05)\n"
                "```"
            ),
            usage=_usage(),
        ),
        MockResponse(text="Best-effort synthesis of the partial finding.", usage=_usage()),
    ]

    result = run_rlm(
        question="test",
        environment=MockEnvironment(),
        root_llm=MockLLMClient(responses=responses),
        subcalls=MockSubcallClient(),
        config=RLMConfig(budget=Budget(wall_ms=20)),
    )

    assert result.error is None, "a time verdict must not surface as a fatal run error"
    assert result.extracted is True
    assert result.answer == "Best-effort synthesis of the partial finding."
    assert result.recovered_error is not None
    assert result.recovered_error.type == "BudgetExhausted"
    assert result.recovered_error.details["resource"] == "wall_ms"


def test_token_exhaustion_still_ends_the_run_fatally():
    """Guard the discrimination above at the run level: a token-exhausted run
    must NOT be rerouted into an extract pass it cannot afford."""
    tiny = Budget(tokens=0x40, root_output_tokens=0x40, subcall_output_tokens=0x40)
    result = run_rlm(
        question="test",
        environment=MockEnvironment(),
        root_llm=MockLLMClient(
            responses=[MockResponse(text="```python\npass\n```", usage=_usage())]
        ),
        subcalls=MockSubcallClient(),
        config=RLMConfig(budget=tiny),
    )

    assert result.error is not None
    assert result.error.type == "BudgetExhausted"
    assert result.error.details["resource"] == "tokens"
    assert result.extracted is False
