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

import pytest

from droste import Budget, RLMConfig, run_rlm
from droste.exceptions import RLMError
from droste.execution import create_execution_context
from droste.loop.rlm import (
    _EXTRACT_HOST_CONTEXT_CHARS,
    _extract_final_answer,
    _is_deadline_error,
)
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
    # Fatal, so nothing is "recovered" — the distinction Cozy's chip relies on.
    assert result.recovered_error is None


def test_recovered_budget_exhaustion_is_always_the_wall_clock():
    """Pins a cross-repo invariant. Cozy renders a recovered `BudgetExhausted`
    as "ran out of time", which is only honest while the wall clock is the sole
    budget resource that can reach a recoverable terminal handoff: token
    exhaustion stays fatal (above) and semantic budgets surface as PolicyError.

    If a future change lets another resource recover here, this fails — rather
    than Cozy silently telling users the wrong thing about why their answer is
    incomplete."""
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
        MockResponse(text="Best-effort synthesis.", usage=_usage()),
    ]
    result = run_rlm(
        question="test",
        environment=MockEnvironment(),
        root_llm=MockLLMClient(responses=responses),
        subcalls=MockSubcallClient(),
        config=RLMConfig(budget=Budget(wall_ms=20)),
    )

    assert result.recovered_error is not None
    assert result.recovered_error.type == "BudgetExhausted"
    assert result.recovered_error.details["resource"] == "wall_ms"


# --- The host knows about work the engine cannot see ---------------------
#
# Generated code that retrieves real data through a host accessor and then
# raises before printing leaves the engine with nothing to test: no draft, no
# successful step, no stdout. It concluded there was nothing to extract while
# the host was holding everything the answer needed. Observed on a real run
# that retrieved 100 message GUIDs and then returned a bare budget error.


def _raising_code() -> str:
    return '```python\nrows = query("SELECT 1")\nraise ValueError("boom")\n```'


def _run_with_probe(probe):
    # Uniform responses: the loop's exact consumption (initial attempt, missing
    # code repair, execution repair) is not what is under test here, and pinning
    # a response count would make this fail for reasons unrelated to the probe.
    responses = [MockResponse(text=_raising_code(), usage=_usage()) for _ in range(12)]
    return run_rlm(
        question="test",
        environment=MockEnvironment(),
        root_llm=MockLLMClient(responses=responses),
        subcalls=MockSubcallClient(),
        config=RLMConfig(budget=Budget(max_iterations=2), extractable_work_probe=probe),
    )


def test_engine_alone_finds_nothing_extractable_in_failed_steps() -> None:
    """Baseline: the engine's own test is unchanged, so a trajectory of pure
    errors with no draft and no stdout still yields no extraction."""
    result = _run_with_probe(None)

    assert result.extracted is False
    assert result.error is not None


def test_host_probe_unlocks_extraction_the_engine_would_have_skipped() -> None:
    """The host says its data layer retrieved something, so the one terminal
    extract call runs and the run returns an answer instead of a bare error.

    This is the case seen on a real 314k-message corpus: generated code called
    an accessor, recorded 100 message GUIDs, then raised before printing any of
    them — leaving the engine no draft, no successful step, and no stdout."""
    result = _run_with_probe(lambda: True)

    assert result.extracted is True
    assert result.error is None


def test_a_probe_saying_no_changes_nothing() -> None:
    result = _run_with_probe(lambda: False)

    assert result.extracted is False


def test_a_raising_probe_cannot_fail_the_run_it_is_recovering() -> None:
    def explode() -> bool:
        raise RuntimeError("host ledger unavailable")

    with pytest.warns(RuntimeWarning, match="extractable-work probe failed"):
        result = _run_with_probe(explode)

    # Treated as "no work" rather than propagating: a terminal recovery path
    # must never be able to end the run it exists to rescue.
    assert result.extracted is False


# --- What the extract pass is allowed to see -----------------------------
#
# Unlocking the extract call was necessary but not sufficient: the prompt is
# built from draft + code + stdout, so a run whose data exists only in the host
# ledger handed the extract pass failing code and nothing else, and it returned
# InsufficientEvidence. The host can now contribute what its accessors actually
# returned.


def _captured_extract_prompt(**config_kwargs):
    """Run to a terminal handoff and return the extract call's user prompt."""
    seen: list[str] = []

    class CapturingClient(MockLLMClient):
        def responses_create(self, messages, *args, **kwargs):  # type: ignore[override]
            seen.append(str(messages[-1].get("content", "")))
            return super().responses_create(messages, *args, **kwargs)

    responses = [MockResponse(text=_raising_code(), usage=_usage()) for _ in range(12)]
    run_rlm(
        question="test",
        environment=MockEnvironment(),
        root_llm=CapturingClient(responses=responses),
        subcalls=MockSubcallClient(),
        config=RLMConfig(budget=Budget(max_iterations=2), **config_kwargs),
    )
    return seen[-1] if seen else ""


def test_host_observations_reach_the_extract_prompt() -> None:
    prompt = _captured_extract_prompt(
        extractable_work_probe=lambda: True,
        extract_context_provider=lambda: (
            "[evidence-1] query(...) -> 84 rows\nSarah | dinner friday"
        ),
    )

    assert "Data retrieved during the run" in prompt
    assert "Sarah | dinner friday" in prompt


def test_observations_are_bounded_so_recovery_cannot_eat_its_own_budget() -> None:
    """The extract call is deliberately uncached, so every character is paid at
    full prefill and an unbounded host could spend the whole recovery call."""
    prompt = _captured_extract_prompt(
        extractable_work_probe=lambda: True,
        extract_context_provider=lambda: "x" * 50_000,
    )

    assert prompt.count("x") == _EXTRACT_HOST_CONTEXT_CHARS


def test_a_raising_context_provider_still_lets_extraction_run() -> None:
    def explode() -> str:
        raise RuntimeError("ledger unavailable")

    with pytest.warns(RuntimeWarning, match="extract-context provider failed"):
        prompt = _captured_extract_prompt(
            extractable_work_probe=lambda: True,
            extract_context_provider=explode,
        )

    # Degraded, not fatal: the extract call still happened, just without them.
    assert prompt
    assert "Data retrieved during the run" not in prompt


def test_a_run_the_engine_could_already_extract_is_left_untouched() -> None:
    """The narrow gate. A run whose draft or successful step the engine can see
    gets the prompt it got before this hook existed — same answer, same spend,
    same benchmark score. Only runs that produce nothing today change."""
    provider_calls: list[int] = []

    responses = [
        MockResponse(
            text="```python\nanswer['content'] = 'partial finding'\n```",
            usage=_usage(),
        ),
        MockResponse(text="Best-effort synthesis.", usage=_usage()),
    ]
    result = run_rlm(
        question="test",
        environment=MockEnvironment(),
        root_llm=MockLLMClient(responses=responses),
        subcalls=MockSubcallClient(),
        config=RLMConfig(
            budget=Budget(max_iterations=1),
            extractable_work_probe=lambda: True,
            extract_context_provider=lambda: provider_calls.append(1) or "should not appear",
        ),
    )

    assert result.extracted is True
    # The engine saw the draft, so the host was never consulted at all.
    assert provider_calls == []
