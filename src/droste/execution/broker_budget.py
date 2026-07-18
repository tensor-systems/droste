"""Capability-broker composition for the run-scoped budget ledger."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..capabilities import (
    CapabilityAdmission,
    CapabilityCall,
    CapabilityCheckpoint,
    CapabilityError,
    CapabilityErrorCode,
    CapabilityKind,
    CapabilityMetadata,
    CapabilityMetric,
    CapabilityReservation,
    thaw_value,
)
from .budget import (
    BudgetExhausted,
    BudgetLedger,
    BudgetRequest,
    conservative_token_estimate,
)

InferenceSettlementCallback = Callable[[bool], None]


def _batch_size(call: CapabilityCall) -> int:
    if call.capability_id.operation == "llm_query":
        return 1
    if call.capability_id.operation not in {"llm_batch", "llm_batch_with_errors"}:
        return 1
    prompts: Any = thaw_value(call.args[0] if call.args else call.kwargs.get("prompts", ()))
    return len(prompts) if isinstance(prompts, (list, tuple)) else 0


def _call_payload(call: CapabilityCall) -> dict[str, Any]:
    return {
        "args": [thaw_value(item) for item in call.args],
        "kwargs": {key: thaw_value(item) for key, item in call.kwargs.items()},
    }


def capability_budget_request(call: CapabilityCall, ledger: BudgetLedger) -> BudgetRequest:
    """Purely derive the maximum authorized spend for one broker call."""

    if call.capability_id.kind is not CapabilityKind.INFERENCE:
        return BudgetRequest()
    count = _batch_size(call)
    input_tokens = conservative_token_estimate(_call_payload(call))
    return BudgetRequest(
        tokens=input_tokens + count * ledger.budget.subcall_output_tokens,
        subcalls=count,
    )


def _actual_request(
    call: CapabilityCall,
    reserved: BudgetRequest,
    checkpoint: CapabilityCheckpoint,
) -> tuple[BudgetRequest, bool]:
    if call.capability_id.kind is not CapabilityKind.INFERENCE:
        return BudgetRequest(), False
    count = _batch_size(call)
    exact = count == 0 or checkpoint.subcalls == count
    if exact:
        return BudgetRequest(tokens=checkpoint.tokens, subcalls=count), True
    # Without a complete per-item checkpoint, a handler may have reached a
    # provider without returning trustworthy usage. Visible output cannot
    # account for hidden reasoning, so retain the full fail-closed reservation.
    return BudgetRequest(tokens=reserved.tokens, subcalls=count), False


def _reservation(request: BudgetRequest) -> CapabilityReservation:
    return CapabilityReservation(
        tokens=request.tokens,
        subcalls=request.subcalls,
        wall_ms=request.wall_ms,
        depth=request.depth,
    )


@dataclass(slots=True)
class BrokerBudget:
    """The one budget authority composed into a capability broker."""

    ledger: BudgetLedger
    on_inference_settlement: InferenceSettlementCallback | None = None

    def bind_inference_settlement_callback(self, callback: InferenceSettlementCallback) -> None:
        """Bind the run's mandatory usage-completeness accounting callback."""

        if not callable(callback):
            raise TypeError("inference settlement callback must be callable")
        if self.on_inference_settlement is not None and self.on_inference_settlement != callback:
            raise ValueError("inference settlement callback is already bound")
        self.on_inference_settlement = callback

    def _record_inference_settlement(self, call: CapabilityCall, exact: bool) -> None:
        if (
            call.capability_id.kind is CapabilityKind.INFERENCE
            and self.on_inference_settlement is not None
        ):
            self.on_inference_settlement(exact)

    def admit(self, call: CapabilityCall) -> CapabilityAdmission | CapabilityError:
        request = capability_budget_request(call, self.ledger)
        try:
            reservation = self.ledger.reserve(
                call.call_id,
                request,
                preserve_tokens=(
                    self.ledger.budget.root_output_tokens
                    if call.capability_id.kind is CapabilityKind.INFERENCE
                    else 0
                ),
                through_deadline=True,
            )
        except BudgetExhausted as exc:
            return CapabilityError("budget_exhausted", "BudgetExhausted", str(exc))
        return CapabilityAdmission(
            reservation=_reservation(reservation.request),
            deadline_monotonic=(
                reservation.started_at + reservation.request.wall_ms / 1000
                if reservation.request.wall_ms
                else None
            ),
        )

    def checkpoint(
        self, call: CapabilityCall, cumulative: CapabilityCheckpoint
    ) -> CapabilityCheckpoint:
        requested = BudgetRequest(tokens=cumulative.tokens, subcalls=cumulative.subcalls)
        reserved = self.ledger.reservation(call.call_id).request
        if requested.tokens > reserved.tokens or requested.subcalls > reserved.subcalls:
            # A provider usage report is an observed fact, not new authority.
            # Keep an overrun in the controller's terminal checkpoint without
            # incrementally committing it. Final settlement will pass the
            # actual total to BudgetLedger.commit(), which closes the
            # reservation and raises BudgetExhausted with the real amount.
            return cumulative
        committed = self.ledger.checkpoint(call.call_id, requested)
        return CapabilityCheckpoint(tokens=committed.tokens, subcalls=committed.subcalls)

    def settle(
        self,
        call: CapabilityCall,
        result: Any,
        error: CapabilityError | None,
        checkpoint: CapabilityCheckpoint,
        *,
        attempted: bool,
    ) -> CapabilityMetadata:
        del result
        if not attempted:
            self.ledger.release(call.call_id)
            return CapabilityMetadata()
        reserved = self.ledger.reservation(call.call_id).request
        actual, exact = _actual_request(call, reserved, checkpoint)
        if (
            call.capability_id.kind is CapabilityKind.INFERENCE
            and error is not None
            and error.code == CapabilityErrorCode.DEADLINE_EXCEEDED
        ):
            # The controller owns the finalization cutoff. If its deadline
            # wins, a late handler result cannot establish complete provider
            # usage even when the ledger's clock has not independently crossed
            # the reservation deadline yet.
            actual = reserved
            exact = False
        try:
            committed = self.ledger.commit(call.call_id, actual)
        except BudgetExhausted as exc:
            # commit() closes the reservation before surfacing an overrun.
            # Exact token usage remains a complete provider fact even though it
            # exceeded authorization; wall/fallback overruns remain partial.
            self._record_inference_settlement(
                call,
                exact and exc.resource == "tokens",
            )
            raise
        self._record_inference_settlement(call, exact)
        settlement_metric = (
            CapabilityMetric(
                "token_settlement_exact" if exact else "token_settlement_fallback",
                1,
                "count",
            )
            if call.capability_id.kind is CapabilityKind.INFERENCE
            else None
        )
        return CapabilityMetadata(
            budget_delta=tuple(
                CapabilityMetric(
                    name,
                    amount,
                    {
                        "tokens": "tokens",
                        "subcalls": "count",
                        "wall_ms": "milliseconds",
                    }[name],
                )
                for name, amount in (
                    ("tokens", committed.tokens),
                    ("subcalls", committed.subcalls),
                    ("wall_ms", committed.wall_ms),
                )
                if amount
            )
            + ((settlement_metric,) if settlement_metric is not None else ())
        )
