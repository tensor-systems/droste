from __future__ import annotations

from threading import Event, Thread

import pytest

from droste.capabilities import (
    LLM_BATCH_CAPABILITY,
    LLM_QUERY_CAPABILITY,
    CapabilityBroker,
    CapabilityError,
    CapabilityMetadata,
    CapabilityStatus,
    subcall_registrations,
)
from droste.execution.broker_budget import BrokerBudget
from droste.execution.budget import Budget, BudgetLedger


class _Subcalls:
    output_token_limit = 100

    def __init__(self) -> None:
        self.calls = 0

    def llm_query(self, prompt: str, context: str = "") -> str:
        self.calls += 1
        return f"answer:{prompt}:{context}"

    def llm_batch(self, prompts: list[str], contexts=None) -> list[str]:
        self.calls += len(prompts)
        return [f"answer:{prompt}" for prompt in prompts]

    def llm_batch_with_errors(self, prompts: list[str], contexts=None):
        return self.llm_batch(prompts, contexts), []


def _broker(
    ledger: BudgetLedger,
    subcalls: _Subcalls,
    *,
    guard=None,
    annotator=None,
) -> CapabilityBroker:
    accounting = BrokerBudget(ledger)
    return CapabilityBroker(
        subcall_registrations(subcalls),
        run_id="run",
        guard=guard,
        annotator=annotator,
        attempt_authority=accounting,
    )


def _budget(*, subcalls: int = 4) -> Budget:
    return Budget(
        tokens=1_000,
        subcalls=subcalls,
        depth=1,
        wall_ms=30_000,
        root_output_tokens=100,
        subcall_output_tokens=100,
    )


def test_success_reconciles_one_reservation_and_attaches_budget_facts() -> None:
    ledger = BudgetLedger(_budget())
    client = _Subcalls()

    result = _broker(ledger, client).call(LLM_QUERY_CAPABILITY.capability_id, "question")

    assert result.ok is True
    assert result.result == "answer:question:"
    assert client.calls == 1
    snapshot = ledger.snapshot()
    assert snapshot.reserved.tokens == 0
    assert snapshot.reserved.subcalls == 0
    assert snapshot.consumed.subcalls == 1
    deltas = {metric.name: metric for metric in result.budget_delta}
    assert deltas["tokens"].unit == "tokens"
    assert deltas["subcalls"].value == 1


def test_batch_reservation_rejects_the_whole_vector_before_dispatch() -> None:
    events: list[dict] = []
    ledger = BudgetLedger(_budget(subcalls=2), on_event=events.append)
    client = _Subcalls()

    result = _broker(ledger, client).call(
        LLM_BATCH_CAPABILITY.capability_id,
        ["one", "two", "three"],
    )

    assert result.status is CapabilityStatus.DENIED
    assert result.error is not None and result.error.code == "budget_exhausted"
    assert client.calls == 0
    assert ledger.snapshot().consumed.subcalls == 0
    assert ledger.snapshot().reserved.subcalls == 0
    assert any(event["action"] == "exhaust" and event["resource"] == "subcalls" for event in events)


def test_successful_batch_reconciles_frozen_broker_result() -> None:
    ledger = BudgetLedger(_budget(subcalls=3))
    client = _Subcalls()

    result = _broker(ledger, client).call(
        LLM_BATCH_CAPABILITY.capability_id,
        ["one", "two", "three"],
    )

    assert result.ok is True
    snapshot = ledger.snapshot()
    assert snapshot.consumed.subcalls == 3
    assert snapshot.reserved.subcalls == 0
    assert snapshot.reserved.wall_ms == 0


def test_downstream_guard_denial_refunds_the_admission_reservation() -> None:
    ledger = BudgetLedger(_budget())
    client = _Subcalls()

    result = _broker(
        ledger,
        client,
        guard=lambda call: CapabilityError("not_allowed", "Denied", "no"),
    ).call(LLM_QUERY_CAPABILITY.capability_id, "question")

    assert result.status is CapabilityStatus.DENIED
    assert client.calls == 0
    snapshot = ledger.snapshot()
    assert snapshot.consumed.tokens == 0
    assert snapshot.consumed.subcalls == 0
    assert snapshot.reserved.tokens == 0


def test_handler_error_charges_the_authorized_attempt_without_leaking() -> None:
    class Failing(_Subcalls):
        def llm_query(self, prompt: str, context: str = "") -> str:
            self.calls += 1
            raise RuntimeError("provider failed")

    ledger = BudgetLedger(_budget())
    client = Failing()

    result = _broker(ledger, client).call(LLM_QUERY_CAPABILITY.capability_id, "question")

    assert result.status is CapabilityStatus.ERROR
    assert result.error is not None and result.error.type == "RuntimeError"
    snapshot = ledger.snapshot()
    assert snapshot.consumed.subcalls == 1
    assert snapshot.consumed.tokens > 100
    assert snapshot.reserved.tokens == 0


def test_concurrent_admission_cannot_oversubscribe_one_subcall() -> None:
    entered = Event()
    release = Event()

    class Blocking(_Subcalls):
        def llm_query(self, prompt: str, context: str = "") -> str:
            self.calls += 1
            entered.set()
            assert release.wait(timeout=2)
            return "done"

    ledger = BudgetLedger(_budget(subcalls=1))
    client = Blocking()
    broker = _broker(ledger, client)
    results = []
    first = Thread(
        target=lambda: results.append(broker.call(LLM_QUERY_CAPABILITY.capability_id, "first"))
    )
    first.start()
    assert entered.wait(timeout=2)

    second = broker.call(LLM_QUERY_CAPABILITY.capability_id, "second")
    release.set()
    first.join(timeout=2)

    assert second.status is CapabilityStatus.DENIED
    assert len(results) == 1 and results[0].ok is True
    assert ledger.snapshot().consumed.subcalls == 1
    assert ledger.snapshot().reserved.subcalls == 0


def test_downstream_annotator_failure_still_settles_before_error_envelope() -> None:
    ledger = BudgetLedger(_budget())
    client = _Subcalls()

    def fail_annotation(call, result, error) -> CapabilityMetadata:
        raise RuntimeError("annotation failed")

    result = _broker(ledger, client, annotator=fail_annotation).call(
        LLM_QUERY_CAPABILITY.capability_id,
        "question",
    )

    assert result.status is CapabilityStatus.ERROR
    assert result.error is not None and result.error.code == "annotator_error"
    snapshot = ledger.snapshot()
    assert snapshot.consumed.subcalls == 1
    assert snapshot.reserved.subcalls == 0


def test_observational_event_sink_failure_cannot_change_or_leak_an_attempt() -> None:
    def fail_event(_event) -> None:
        raise RuntimeError("sink unavailable")

    ledger = BudgetLedger(_budget(), on_event=fail_event)
    client = _Subcalls()

    with pytest.warns(RuntimeWarning, match="budget event sink failed"):
        result = _broker(ledger, client).call(
            LLM_QUERY_CAPABILITY.capability_id,
            "question",
        )

    assert result.ok is True
    assert client.calls == 1
    snapshot = ledger.snapshot()
    assert snapshot.consumed.subcalls == 1
    assert snapshot.reserved.tokens == 0
    assert snapshot.reserved.subcalls == 0
