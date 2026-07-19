from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Thread
from time import sleep

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
from droste.exceptions import BatchItemError, BatchItemErrorDetails
from droste.execution.broker_budget import BrokerBudget
from droste.execution.budget import (
    Budget,
    BudgetLedger,
    BudgetRequest,
    conservative_token_estimate,
)
from droste.execution.context import create_execution_context
from droste.protocols.llm_client import LLMUsageFailure, TokenUsage
from droste.protocols.subcall_client import (
    SubcallBatchFailure,
    SubcallBatchResult,
    SubcallQueryResult,
    fail_fast_subcall_batch,
)


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


class _ExactSubcalls(_Subcalls):
    def __init__(self, usage: list[TokenUsage]) -> None:
        super().__init__()
        self.usage = usage

    def llm_query_with_usage(self, prompt: str, context: str = "") -> SubcallQueryResult:
        return SubcallQueryResult(self.llm_query(prompt, context), self.usage[0])

    def llm_batch_with_usage(self, prompts, contexts=None) -> SubcallBatchResult:
        return SubcallBatchResult(
            tuple(self.llm_batch(prompts, contexts)), (), tuple(self.usage[: len(prompts)])
        )

    def llm_batch_with_errors_and_usage(self, prompts, contexts=None) -> SubcallBatchResult:
        return self.llm_batch_with_usage(prompts, contexts)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def _broker(
    ledger: BudgetLedger,
    subcalls: _Subcalls,
    *,
    guard=None,
    annotator=None,
    usage_callback=None,
    settlement_callback=None,
    broker_clock=None,
) -> CapabilityBroker:
    accounting = BrokerBudget(ledger, on_inference_settlement=settlement_callback)
    return CapabilityBroker(
        subcall_registrations(subcalls, usage_callback=usage_callback),
        run_id="run",
        guard=guard,
        annotator=annotator,
        attempt_authority=accounting,
        clock=broker_clock or ledger.clock,
    )


def _budget(*, subcalls: int = 4, tokens: int = 1_000) -> Budget:
    return Budget(
        tokens=tokens,
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
    assert deltas["token_settlement_fallback"].value == 1


def test_exact_success_settles_provider_total_including_hidden_reasoning() -> None:
    ledger = BudgetLedger(_budget())
    client = _ExactSubcalls([TokenUsage(7, 3, 41, exact=True)])

    result = _broker(ledger, client).call(LLM_QUERY_CAPABILITY.capability_id, "question")

    assert result.ok is True
    assert ledger.snapshot().consumed.tokens == 41
    deltas = {metric.name: metric for metric in result.budget_delta}
    assert deltas["tokens"].value == 41
    assert deltas["token_settlement_exact"].value == 1


def test_exact_query_overrun_reaches_ledger_with_actual_usage() -> None:
    context = create_execution_context(budget=_budget())
    client = _ExactSubcalls([TokenUsage(7, 3, 1_200, exact=True)])

    result = _broker(
        context.ledger,
        client,
        usage_callback=context.record_subcall_usage,
        settlement_callback=context.record_subcall_settlement,
    ).call(LLM_QUERY_CAPABILITY.capability_id, "question")

    assert result.status is CapabilityStatus.ERROR
    assert result.error is not None
    assert result.error.code == "settlement_error"
    assert result.error.type == "BudgetExhausted"
    assert "requested 1200" in result.error.message
    assert context.stats.subcall_total_tokens == 1_200
    assert context.stats.subcall_usage_complete is True
    resolved = context.stats.resolved_usage(0).as_dict()
    assert resolved["kind"] == "resolved"
    assert resolved["subcall"]["total_tokens"] == 1_200
    snapshot = context.ledger.snapshot()
    assert snapshot.reserved == BudgetRequest()
    assert snapshot.consumed.tokens < 1_200


def test_malformed_query_output_preserves_exact_usage_before_original_error() -> None:
    context = create_execution_context(budget=_budget())

    class MalformedQuery(_ExactSubcalls):
        def llm_query_with_usage(self, prompt: str, context: str = "") -> SubcallQueryResult:
            raise LLMUsageFailure(
                TokenUsage(7, 3, 41, exact=True),
                RuntimeError("missing subcall output"),
            )

    result = _broker(
        context.ledger,
        MalformedQuery([TokenUsage(7, 3, 41, exact=True)]),
        usage_callback=context.record_subcall_usage,
        settlement_callback=context.record_subcall_settlement,
    ).call(LLM_QUERY_CAPABILITY.capability_id, "question")

    assert result.status is CapabilityStatus.ERROR
    assert result.error is not None
    assert result.error.code == "handler_error"
    assert result.error.type == "RuntimeError"
    assert result.error.message == "missing subcall output"
    assert context.stats.subcall_total_tokens == 41
    assert context.stats.subcall_usage_complete is True
    assert context.ledger.snapshot().consumed.tokens == 41


def test_exact_batch_overrun_reaches_ledger_with_summed_actual_usage() -> None:
    context = create_execution_context(budget=_budget(subcalls=2))
    client = _ExactSubcalls(
        [
            TokenUsage(5, 2, 700, exact=True),
            TokenUsage(6, 3, 800, exact=True),
        ]
    )

    result = _broker(
        context.ledger,
        client,
        usage_callback=context.record_subcall_usage,
        settlement_callback=context.record_subcall_settlement,
    ).call(LLM_BATCH_CAPABILITY.capability_id, ["one", "two"])

    assert result.status is CapabilityStatus.ERROR
    assert result.error is not None
    assert result.error.code == "settlement_error"
    assert result.error.type == "BudgetExhausted"
    assert "requested 1500" in result.error.message
    assert context.stats.subcall_total_tokens == 1_500
    assert context.stats.subcall_usage_complete is True
    resolved = context.stats.resolved_usage(0).as_dict()
    assert resolved["kind"] == "resolved"
    assert resolved["subcall"]["total_tokens"] == 1_500
    snapshot = context.ledger.snapshot()
    assert snapshot.reserved == BudgetRequest()
    assert snapshot.consumed.tokens < 1_500


def test_exact_batch_sums_item_usage_and_refunds_for_near_budget_reuse() -> None:
    context = create_execution_context(budget=_budget(subcalls=3))
    client = _ExactSubcalls(
        [
            TokenUsage(5, 2, 7, cache_read_tokens=2, exact=True),
            TokenUsage(6, 3, 19, cache_creation_tokens=4, exact=True),
            TokenUsage(4, 1, 5, cache_read_tokens=1, exact=True),
        ]
    )
    broker = _broker(
        context.ledger,
        client,
        usage_callback=context.record_subcall_usage,
        settlement_callback=context.record_subcall_settlement,
    )

    first = broker.call(LLM_BATCH_CAPABILITY.capability_id, ["one", "two"])
    assert first.ok is True
    assert context.stats.subcall_cache_read_tokens == 2
    assert context.stats.subcall_cache_creation_tokens == 4

    second = broker.call(LLM_QUERY_CAPABILITY.capability_id, "three")

    assert first.ok is True and second.ok is True
    assert context.stats.subcall_cache_read_tokens == 4
    assert context.stats.subcall_cache_creation_tokens == 4
    assert context.ledger.snapshot().consumed.tokens == 33
    assert context.ledger.snapshot().consumed.subcalls == 3


def test_concurrent_exact_queries_serialize_usage_callback_and_reconcile_totals() -> None:
    count = 16
    context = create_execution_context(budget=_budget(subcalls=count, tokens=100_000))
    client = _ExactSubcalls([TokenUsage(7, 3, 41, exact=True)])
    state_lock = Lock()
    active_callbacks = 0
    max_active_callbacks = 0

    def record_usage(usage: TokenUsage) -> None:
        nonlocal active_callbacks, max_active_callbacks
        with state_lock:
            active_callbacks += 1
            max_active_callbacks = max(max_active_callbacks, active_callbacks)
            if active_callbacks > 1:
                raise AssertionError("usage callback ran concurrently")
        sleep(0.002)
        context.record_subcall_usage(usage)
        with state_lock:
            active_callbacks -= 1

    broker = _broker(
        context.ledger,
        client,
        usage_callback=record_usage,
        settlement_callback=context.record_subcall_settlement,
    )
    with ThreadPoolExecutor(max_workers=count) as pool:
        results = list(
            pool.map(
                lambda index: broker.call(
                    LLM_QUERY_CAPABILITY.capability_id,
                    f"question-{index}",
                ),
                range(count),
            )
        )

    assert all(result.ok for result in results)
    assert max_active_callbacks == 1
    assert context.stats.subcall_total_tokens == count * 41
    assert context.stats.total_tokens == count * 41
    snapshot = context.ledger.snapshot()
    assert snapshot.consumed.tokens == count * 41
    assert snapshot.consumed.subcalls == count
    assert snapshot.reserved == BudgetRequest()


def test_unavailable_item_usage_keeps_the_full_batch_reservation() -> None:
    ledger = BudgetLedger(_budget(subcalls=2))
    client = _ExactSubcalls([TokenUsage(5, 2, 7, exact=True), TokenUsage.unavailable()])

    result = _broker(ledger, client).call(LLM_BATCH_CAPABILITY.capability_id, ["one", "two"])

    assert result.status is CapabilityStatus.OK
    reserved = conservative_token_estimate({"args": [["one", "two"]], "kwargs": {}}) + 200
    assert ledger.snapshot().consumed.tokens == reserved
    assert result.result.items == ("answer:one", "answer:two")
    deltas = {metric.name: metric for metric in result.budget_delta}
    assert deltas["token_settlement_fallback"].value == 1


def test_partial_item_usage_preserves_known_counts_and_keeps_full_reservation() -> None:
    context = create_execution_context(budget=_budget(subcalls=1))
    client = _ExactSubcalls([TokenUsage(7, 0, 19)])

    result = _broker(
        context.ledger,
        client,
        usage_callback=context.record_subcall_usage,
        settlement_callback=context.record_subcall_settlement,
    ).call(LLM_QUERY_CAPABILITY.capability_id, "question")

    assert result.ok is True
    usage = context.stats.resolved_usage(0).as_dict()
    assert usage["kind"] == "partial"
    assert usage["subcall"]["input_tokens"] == 7
    assert usage["subcall"]["output_tokens"] == 0
    assert usage["subcall"]["total_tokens"] == 19
    reserved = conservative_token_estimate({"args": ["question"], "kwargs": {}}) + 100
    assert context.ledger.snapshot().consumed.tokens == reserved
    deltas = {metric.name: metric for metric in result.budget_delta}
    assert deltas["token_settlement_fallback"].value == 1


def test_failed_batch_preserves_known_sibling_usage_through_broker() -> None:
    context = create_execution_context(budget=_budget(subcalls=3))

    class PartialBatch(_ExactSubcalls):
        def llm_batch_with_usage(self, prompts, contexts=None) -> SubcallBatchResult:
            self.calls += len(prompts)
            context.record_subcall_attempts(len(prompts))
            context.record_subcall_successes(2)
            return fail_fast_subcall_batch(
                tuple("answer" if index != 1 else "" for index in range(len(prompts))),
                (None, RuntimeError("provider failed"), None),
                tuple(self.usage[: len(prompts)]),
            )

    client = PartialBatch(
        [
            TokenUsage(2, 1, 11, cache_read_tokens=1, exact=True),
            TokenUsage.unavailable(),
            TokenUsage(3, 2, 13, cache_creation_tokens=2, exact=True),
        ]
    )
    result = _broker(
        context.ledger,
        client,
        usage_callback=context.record_subcall_usage,
        settlement_callback=context.record_subcall_settlement,
    ).call(LLM_BATCH_CAPABILITY.capability_id, ["one", "two", "three"])

    assert result.status is CapabilityStatus.ERROR
    assert result.error is not None
    assert result.error.type == "RuntimeError" and result.error.message == "provider failed"
    assert context.stats.calls_made == 3
    assert context.stats.successful_calls == 2
    assert context.stats.subcall_total_tokens == 24
    assert context.stats.subcall_cache_read_tokens == 1
    assert context.stats.subcall_cache_creation_tokens == 2
    assert context.stats.subcall_usage_complete is False
    resolved = context.stats.resolved_usage(0).as_dict()
    assert resolved["kind"] == "partial"
    assert resolved["subcall"]["total_tokens"] == 24
    assert resolved["subcall"]["cache_read_tokens"] == 1
    assert resolved["subcall"]["cache_creation_tokens"] == 2


def test_fail_fast_batch_carries_all_public_errors_and_original_first_cause() -> None:
    first = BatchItemError(
        "first failed",
        BatchItemErrorDetails(request_id="req-1", retryable=True),
    )
    later = ValueError("later failed")

    with pytest.raises(SubcallBatchFailure) as failure:
        fail_fast_subcall_batch(
            ("", "ok", ""),
            (first, None, later),
            (
                TokenUsage.unavailable(),
                TokenUsage(1, 1, 2, exact=True),
                TokenUsage.unavailable(),
            ),
        )

    assert failure.value.cause is first
    assert failure.value.result.errors == (
        {
            "index": 0,
            "error": "first failed",
            "details": {"request_id": "req-1", "retryable": True},
        },
        {"index": 2, "error": "later failed"},
    )


def test_fail_fast_batch_rejects_misaligned_error_slots() -> None:
    with pytest.raises(ValueError, match="errors must align with results"):
        fail_fast_subcall_batch(
            ("one", "two"),
            (None,),
            (
                TokenUsage(1, 1, 2, exact=True),
                TokenUsage(1, 1, 2, exact=True),
            ),
        )


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


def test_ledger_wall_overrun_commit_still_reports_fallback_settlement() -> None:
    clock = _Clock()
    events: list[dict] = []
    ledger = BudgetLedger(_budget(), clock=clock, on_event=events.append)
    settlements: list[bool] = []

    class Slow(_Subcalls):
        def llm_query(self, prompt: str, context: str = "") -> str:
            self.calls += 1
            clock.value = 31.0
            return "late"

    result = _broker(
        ledger,
        Slow(),
        settlement_callback=settlements.append,
    ).call(LLM_QUERY_CAPABILITY.capability_id, "question")

    assert result.status is CapabilityStatus.ERROR
    assert result.error is not None and result.error.code == "settlement_error"
    assert settlements == [False]
    assert any(event["action"] == "exhaust" and event["resource"] == "wall_ms" for event in events)
    snapshot = ledger.snapshot()
    assert snapshot.reserved.tokens == 0
    assert snapshot.reserved.subcalls == 0


def test_controller_deadline_forces_fallback_usage_before_ledger_deadline() -> None:
    ledger_clock = _Clock()
    controller_clock = _Clock()
    ledger = BudgetLedger(_budget(), clock=ledger_clock)
    context = create_execution_context()

    class Late(_Subcalls):
        def llm_query(self, prompt: str, context: str = "") -> str:
            self.calls += 1
            controller_clock.value = 31.0
            return "late"

    result = _broker(
        ledger,
        Late(),
        settlement_callback=context.record_subcall_settlement,
        broker_clock=controller_clock,
    ).call(LLM_QUERY_CAPABILITY.capability_id, "question")

    assert result.status is CapabilityStatus.CANCELLED
    assert result.error is not None and result.error.code == "deadline_exceeded"
    assert ledger.snapshot().reserved.subcalls == 0
    usage = context.stats.resolved_usage(0).as_dict()
    assert usage["kind"] == "partial"
    assert usage["subcall"]["complete"] is False


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
