from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest

from droste import (
    Budget,
    BudgetExhausted,
    BudgetLedger,
    BudgetRequest,
    create_execution_context,
)
from droste.execution.budget import conservative_token_estimate
from droste.loop.step import call_root
from droste.protocols.llm_client import LLMUsageFailure, TokenUsage


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _budget(**overrides: int) -> Budget:
    values = {
        "tokens": 1_000,
        "subcalls": 10,
        "depth": 3,
        "wall_ms": 1_000,
        "root_output_tokens": 100,
        "subcall_output_tokens": 50,
    }
    values.update(overrides)
    return Budget(**values)


def test_budget_is_one_strict_immutable_value() -> None:
    budget = _budget()
    assert Budget.from_dict(budget.as_dict()) == budget
    with pytest.raises(FrozenInstanceError):
        budget.tokens = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="unknown extra"):
        Budget.from_dict({**budget.as_dict(), "extra": 1})


def test_vector_reservation_is_atomic_and_preserves_root_synthesis() -> None:
    events: list[dict] = []
    ledger = BudgetLedger(_budget(tokens=300, root_output_tokens=100), on_event=events.append)

    with pytest.raises(BudgetExhausted) as raised:
        ledger.reserve(
            "too-large",
            BudgetRequest(tokens=201, subcalls=2, wall_ms=100),
            preserve_tokens=100,
        )

    assert raised.value.resource == "tokens"
    snapshot = ledger.snapshot()
    assert snapshot.reserved == BudgetRequest()
    assert snapshot.consumed.tokens == 0
    assert events == [
        {
            "type": "budget",
            "kind": "mutation",
            "source": "budget_ledger",
            "action": "exhaust",
            "resource": "tokens",
            "amount": 201,
            "call_id": "too-large",
        }
    ]


def test_commit_charges_actual_and_refunds_unused_reservation() -> None:
    clock = Clock()
    events: list[dict] = []
    ledger = BudgetLedger(_budget(), clock=clock, on_event=events.append)
    ledger.reserve("call-1", BudgetRequest(tokens=200, subcalls=4, wall_ms=500))

    clock.now += 0.125
    actual = ledger.commit("call-1", BudgetRequest(tokens=80, subcalls=3))

    assert actual == BudgetRequest(tokens=80, subcalls=3, wall_ms=125)
    snapshot = ledger.snapshot()
    assert snapshot.consumed.tokens == 80
    assert snapshot.consumed.subcalls == 3
    assert snapshot.reserved == BudgetRequest()
    assert any(event["action"] == "commit" and event["resource"] == "tokens" for event in events)
    assert any(event["action"] == "refund" and event["resource"] == "tokens" for event in events)


def test_cumulative_checkpoints_are_idempotent_and_final_reconciliation_is_monotonic() -> None:
    events: list[dict] = []
    ledger = BudgetLedger(_budget(), on_event=events.append)
    ledger.reserve("call-1", BudgetRequest(tokens=200, subcalls=4))

    first = ledger.checkpoint("call-1", BudgetRequest(tokens=50, subcalls=1))
    repeated = ledger.checkpoint("call-1", BudgetRequest(tokens=50, subcalls=1))
    second = ledger.checkpoint("call-1", BudgetRequest(tokens=80, subcalls=2))

    assert first == repeated == BudgetRequest(tokens=50, subcalls=1)
    assert second == BudgetRequest(tokens=80, subcalls=2)
    assert ledger.snapshot().consumed.tokens == 80
    assert ledger.snapshot().reserved.tokens == 120
    committed = ledger.commit("call-1", BudgetRequest(tokens=60, subcalls=1))
    assert committed.tokens == 80
    assert committed.subcalls == 2
    assert ledger.snapshot().reserved == BudgetRequest()
    token_commits = [
        event["amount"]
        for event in events
        if event["action"] == "commit" and event["resource"] == "tokens"
    ]
    assert token_commits == [50, 30]


def test_checkpoint_regression_and_overflow_do_not_change_accounting() -> None:
    events: list[dict] = []
    ledger = BudgetLedger(_budget(), on_event=events.append)
    ledger.reserve("call-1", BudgetRequest(tokens=100, subcalls=2))
    ledger.checkpoint("call-1", BudgetRequest(tokens=40, subcalls=1))

    with pytest.raises(ValueError, match="cannot move backward"):
        ledger.checkpoint("call-1", BudgetRequest(tokens=39, subcalls=1))
    with pytest.raises(BudgetExhausted, match="tokens"):
        ledger.checkpoint("call-1", BudgetRequest(tokens=101, subcalls=1))

    snapshot = ledger.snapshot()
    assert snapshot.consumed.tokens == 40
    assert snapshot.reserved.tokens == 60
    assert any(
        event["action"] == "exhaust"
        and event["resource"] == "tokens"
        and event["call_id"] == "call-1"
        for event in events
    )


def test_concurrent_identical_checkpoints_commit_once() -> None:
    events: list[dict] = []
    ledger = BudgetLedger(_budget(), on_event=events.append)
    ledger.reserve("call-1", BudgetRequest(tokens=100, subcalls=1))

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(
            pool.map(
                lambda _: ledger.checkpoint("call-1", BudgetRequest(tokens=50, subcalls=1)),
                range(32),
            )
        )

    assert results == [BudgetRequest(tokens=50, subcalls=1)] * 32
    assert ledger.snapshot().consumed.tokens == 50
    assert [
        event for event in events if event["action"] == "commit" and event["resource"] == "tokens"
    ] == [
        {
            "type": "budget",
            "kind": "mutation",
            "source": "budget_ledger",
            "action": "commit",
            "resource": "tokens",
            "amount": 50,
            "call_id": "call-1",
        }
    ]


def test_overrun_settles_reservation_before_raising() -> None:
    ledger = BudgetLedger(_budget())
    ledger.reserve("call-1", BudgetRequest(tokens=20, subcalls=1))

    with pytest.raises(BudgetExhausted, match="tokens"):
        ledger.commit("call-1", BudgetRequest(tokens=21, subcalls=1))

    assert ledger.snapshot().reserved == BudgetRequest()
    assert ledger.snapshot().consumed.tokens == 20
    with pytest.raises(ValueError, match="unknown"):
        ledger.reservation("call-1")


def test_concurrent_reservations_cannot_overspend() -> None:
    ledger = BudgetLedger(_budget(tokens=500, subcalls=5, root_output_tokens=100))

    def attempt(index: int) -> bool:
        try:
            ledger.reserve(
                f"call-{index}",
                BudgetRequest(tokens=80, subcalls=1),
                preserve_tokens=100,
            )
            return True
        except BudgetExhausted:
            return False

    with ThreadPoolExecutor(max_workers=16) as pool:
        accepted = list(pool.map(attempt, range(20)))

    assert sum(accepted) == 5
    snapshot = ledger.snapshot()
    assert snapshot.reserved.tokens == 400
    assert snapshot.reserved.subcalls == 5


def test_shared_wall_deadline_does_not_serialize_concurrent_work() -> None:
    clock = Clock()
    ledger = BudgetLedger(_budget(), clock=clock)

    first = ledger.reserve("first", BudgetRequest(tokens=10), through_deadline=True)
    second = ledger.reserve("second", BudgetRequest(tokens=10), through_deadline=True)

    assert first.request.wall_ms == 1_000
    assert second.request.wall_ms == 1_000
    assert ledger.snapshot().remaining.wall_ms == 1_000

    clock.now += 1
    with pytest.raises(BudgetExhausted, match="wall_ms"):
        ledger.reserve("late", BudgetRequest(tokens=10), through_deadline=True)


def test_child_is_strict_subledger_and_returns_unused_allocation() -> None:
    clock = Clock()
    parent = BudgetLedger(_budget(), clock=clock)
    child = parent.child(
        "child-1",
        _budget(
            tokens=300,
            subcalls=3,
            depth=2,
            wall_ms=400,
            root_output_tokens=50,
            subcall_output_tokens=25,
        ),
    )
    child.reserve("work", BudgetRequest(tokens=100, subcalls=1, wall_ms=200))
    clock.now += 0.05
    child.commit("work", BudgetRequest(tokens=60, subcalls=1))
    child.close()

    snapshot = parent.snapshot()
    assert snapshot.consumed.tokens == 60
    assert snapshot.consumed.subcalls == 1
    assert snapshot.reserved == BudgetRequest()


def test_child_uses_its_own_deadline_within_the_shared_parent_deadline() -> None:
    clock = Clock()
    parent = BudgetLedger(_budget(wall_ms=1_000), clock=clock)
    child = parent.child("child", _budget(tokens=300, subcalls=3, depth=2, wall_ms=400))

    clock.now += 0.5
    child.close()

    assert parent.snapshot().reserved == BudgetRequest()


def test_child_deadline_cannot_exceed_parent_remaining_deadline() -> None:
    clock = Clock()
    parent = BudgetLedger(_budget(wall_ms=1_000), clock=clock)
    clock.now += 0.7

    with pytest.raises(BudgetExhausted, match="wall_ms"):
        parent.child("child", _budget(tokens=300, subcalls=3, depth=2, wall_ms=400))

    assert parent.snapshot().reserved == BudgetRequest()


def test_child_depth_is_a_decaying_allocation_not_global_depth() -> None:
    root = BudgetLedger(_budget(depth=3))
    child = root.child("child", _budget(tokens=300, subcalls=3, depth=2))
    grandchild = child.child(
        "grandchild",
        _budget(tokens=150, subcalls=1, depth=1, root_output_tokens=50),
    )
    great_grandchild = grandchild.child(
        "great-grandchild",
        _budget(
            tokens=50,
            subcalls=0,
            depth=0,
            root_output_tokens=25,
            subcall_output_tokens=25,
        ),
    )

    assert child.snapshot().current_depth == 1
    assert grandchild.snapshot().current_depth == 2
    assert great_grandchild.snapshot().current_depth == 3
    with pytest.raises(BudgetExhausted, match="depth"):
        great_grandchild.child(
            "too-deep",
            _budget(
                tokens=25,
                subcalls=0,
                depth=0,
                wall_ms=100,
                root_output_tokens=25,
                subcall_output_tokens=25,
            ),
        )

    great_grandchild.close()
    grandchild.close()
    child.close()


def test_event_sink_can_observe_ledger_without_deadlock() -> None:
    snapshots = []
    ledger: BudgetLedger

    def sink(_event: dict) -> None:
        snapshots.append(ledger.snapshot())

    ledger = BudgetLedger(_budget(), on_event=sink)
    ledger.reserve("call-1", BudgetRequest(tokens=10))
    assert snapshots


def test_root_process_control_exit_settles_before_propagating() -> None:
    class Stop(BaseException):
        pass

    class StoppingLLM:
        def responses_create(self, *args, **kwargs):
            raise Stop

    context = create_execution_context()
    with pytest.raises(Stop):
        call_root(
            StoppingLLM(),  # type: ignore[arg-type]
            [{"role": "user", "content": "question"}],
            model="model",
            context=context,
        )

    snapshot = context.ledger.snapshot()
    assert snapshot.reserved == BudgetRequest()
    assert snapshot.consumed.tokens > 0


def test_root_success_settles_exact_usage_instead_of_visible_bytes() -> None:
    class ExactLLM:
        def responses_create(self, *args, **kwargs):
            return "x" * 500, TokenUsage(2, 3, 17, exact=True)

    context = create_execution_context(budget=_budget())

    response, usage, error = call_root(
        ExactLLM(),  # type: ignore[arg-type]
        [{"role": "user", "content": "question"}],
        model="model",
        context=context,
    )

    assert response == "x" * 500
    assert usage.total_tokens == 17
    assert error is None
    assert context.ledger.snapshot().consumed.tokens == 17


def test_root_success_without_trusted_usage_consumes_full_reservation() -> None:
    messages = [{"role": "user", "content": "question"}]

    class MissingUsageLLM:
        def responses_create(self, *args, **kwargs):
            return "short", TokenUsage.unavailable()

    context = create_execution_context(budget=_budget())

    response, usage, error = call_root(
        MissingUsageLLM(),  # type: ignore[arg-type]
        messages,
        model="model",
        context=context,
    )

    assert response == "short"
    assert usage.exact is False
    assert error is None
    assert context.stats.root_successes == 1
    assert context.stats.root_usage_complete is False
    assert context.stats.resolved_usage(0).as_dict()["kind"] == "partial"
    expected = conservative_token_estimate(messages) + context.budget.root_output_tokens
    assert context.ledger.snapshot().consumed.tokens == expected


def test_root_exact_token_overrun_preserves_terminal_usage_and_success() -> None:
    class OverBudgetLLM:
        def responses_create(self, *args, **kwargs):
            return "completed", TokenUsage(7, 5, 300, exact=True)

    context = create_execution_context(budget=_budget(tokens=200, root_output_tokens=100))

    response, usage, error = call_root(
        OverBudgetLLM(),  # type: ignore[arg-type]
        [{"role": "user", "content": "question"}],
        model="model",
        context=context,
    )

    assert response == ""
    assert usage == TokenUsage(7, 5, 300, exact=True)
    assert error is not None and error.type == "BudgetExhausted"
    assert context.stats.root_requests == context.stats.root_successes == 1
    resolved = context.stats.resolved_usage(0).as_dict()
    assert resolved["kind"] == "resolved"
    assert resolved["root"]["total_tokens"] == 300
    assert resolved["root"]["complete"] is True
    assert context.ledger.snapshot().reserved == BudgetRequest()


def test_root_wall_overrun_preserves_exact_terminal_usage_and_success() -> None:
    clock = Clock()
    budget = _budget(wall_ms=1_000)
    context = create_execution_context(budget=budget)
    context.ledger = BudgetLedger(budget, clock=clock)

    class SlowLLM:
        def responses_create(self, *args, **kwargs):
            clock.now += 1.1
            return "completed", TokenUsage(2, 3, 17, exact=True)

    response, usage, error = call_root(
        SlowLLM(),  # type: ignore[arg-type]
        [{"role": "user", "content": "question"}],
        model="model",
        context=context,
    )

    assert response == ""
    assert usage == TokenUsage(2, 3, 17, exact=True)
    assert error is not None and error.type == "BudgetExhausted"
    assert context.stats.root_requests == context.stats.root_successes == 1
    resolved = context.stats.resolved_usage(1_100).as_dict()
    assert resolved["kind"] == "resolved"
    assert resolved["root"]["total_tokens"] == 17
    assert resolved["root"]["complete"] is True
    assert context.ledger.snapshot().reserved == BudgetRequest()


def test_root_malformed_output_preserves_usage_without_counting_success() -> None:
    usage = TokenUsage(7, 3, 19, exact=True)

    class MalformedOutputLLM:
        def responses_create(self, *args, **kwargs):
            raise LLMUsageFailure(usage, RuntimeError("missing root output"))

    context = create_execution_context(budget=_budget())
    response, reported_usage, error = call_root(
        MalformedOutputLLM(),  # type: ignore[arg-type]
        [{"role": "user", "content": "question"}],
        model="model",
        context=context,
    )

    assert response == "" and reported_usage == usage
    assert error is not None and error.type == "RuntimeError"
    assert error.message == "missing root output"
    assert context.stats.root_requests == 1
    assert context.stats.root_successes == 0
    assert context.stats.root_total_tokens == 19
    assert context.stats.root_usage_complete is True
    assert context.ledger.snapshot().consumed.tokens == 19
    assert context.ledger.snapshot().reserved == BudgetRequest()
