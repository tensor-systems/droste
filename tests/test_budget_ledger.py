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
from droste.loop.step import call_root


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
