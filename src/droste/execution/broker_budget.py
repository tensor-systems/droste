"""Capability-broker composition for the run-scoped budget ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..capabilities import (
    CapabilityAnnotator,
    CapabilityCall,
    CapabilityError,
    CapabilityGuard,
    CapabilityKind,
    CapabilityMetadata,
    CapabilityMetric,
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


@dataclass(slots=True)
class BrokerBudget:
    """One composed guard/finalizer; no second enforcement path."""

    ledger: BudgetLedger
    guard_after_budget: CapabilityGuard | None = None
    annotator_after_budget: CapabilityAnnotator | None = None

    def guard(self, call: CapabilityCall) -> CapabilityError | None:
        request = capability_budget_request(call, self.ledger)
        try:
            self.ledger.reserve(
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
            return CapabilityError(
                "budget_exhausted",
                "BudgetExhausted",
                str(exc),
            )
        if self.guard_after_budget is None:
            return None
        try:
            denial = self.guard_after_budget(call)
            if denial is not None:
                self.ledger.release(call.call_id)
            return denial
        except BaseException:
            # The broker skips its annotator for guard failures, so this
            # composed lifecycle must release before propagating.
            self.ledger.release(call.call_id)
            raise

    def annotate(
        self,
        call: CapabilityCall,
        result: Any,
        error: CapabilityError | None,
    ) -> CapabilityMetadata:
        reserved = self.ledger.reservation(call.call_id).request
        metadata = CapabilityMetadata()
        downstream_error: BaseException | None = None
        if self.annotator_after_budget is not None:
            try:
                metadata = self.annotator_after_budget(call, result, error)
            except BaseException as exc:
                downstream_error = exc
        actual = self.ledger.commit(
            call.call_id,
            _actual_request(call, result, error, reserved),
        )
        budget_metadata = CapabilityMetadata(
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
                    ("tokens", actual.tokens),
                    ("subcalls", actual.subcalls),
                    ("wall_ms", actual.wall_ms),
                )
                if amount
            )
        )
        if downstream_error is not None:
            raise downstream_error
        return metadata.merged_with(budget_metadata)
