"""Capability-broker composition for the run-scoped budget ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..capabilities import (
    CapabilityAdmission,
    CapabilityCall,
    CapabilityCheckpoint,
    CapabilityError,
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
    result: Any,
    error: CapabilityError | None,
    reserved: BudgetRequest,
) -> BudgetRequest:
    if call.capability_id.kind is not CapabilityKind.INFERENCE:
        return BudgetRequest()
    count = _batch_size(call)
    if error is not None:
        # A handler attempt may have reached a provider without returning
        # trusted usage. Charge the complete reservation rather than guessing.
        return BudgetRequest(tokens=reserved.tokens, subcalls=count)
    input_tokens = conservative_token_estimate(_call_payload(call))
    return BudgetRequest(
        tokens=input_tokens
        + min(
            conservative_token_estimate(thaw_value(result)),
            max(0, reserved.tokens - input_tokens),
        ),
        subcalls=count,
    )


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
        committed = self.ledger.checkpoint(
            call.call_id,
            BudgetRequest(tokens=cumulative.tokens, subcalls=cumulative.subcalls),
        )
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
        if not attempted:
            self.ledger.release(call.call_id)
            return CapabilityMetadata()
        reserved = self.ledger.reservation(call.call_id).request
        inferred = _actual_request(call, result, error, reserved)
        actual = BudgetRequest(
            tokens=max(inferred.tokens, checkpoint.tokens),
            subcalls=max(inferred.subcalls, checkpoint.subcalls),
            depth=inferred.depth,
        )
        committed = self.ledger.commit(call.call_id, actual)
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
        )
