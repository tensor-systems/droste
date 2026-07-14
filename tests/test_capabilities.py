"""Broker ABI conformance and built-in environment egress tests (#9)."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from threading import Event, Thread

import pytest

from droste import RLMConfig, run_rlm
from droste.capabilities import (
    JSON_SCHEMA_2020_12,
    CapabilityAdmission,
    CapabilityBroker,
    CapabilityCall,
    CapabilityCallError,
    CapabilityCheckpoint,
    CapabilityDescriptor,
    CapabilityError,
    CapabilityErrorCode,
    CapabilityExecutionContext,
    CapabilityId,
    CapabilityKind,
    CapabilityManifest,
    CapabilityMetadata,
    CapabilityMetric,
    CapabilityOutcome,
    CapabilityRegistration,
    CapabilityReservation,
    CapabilityResult,
    CapabilityResultHandle,
    CapabilityStatus,
    EvidenceLocation,
    EvidenceRange,
    PaginationMode,
    ProviderOperation,
    ResultDelivery,
    SchemaSpec,
    SideEffect,
    broker_subcalls,
    generate_binding,
    thaw_value,
    validate_call,
)
from droste.environments import (
    EnvironmentConfig,
    RunnerEnvironment,
    create_environment,
    create_environment_context,
)
from droste.protocols.llm_client import TokenUsage
from droste.providers import ConfiguredSource, ProviderCatalog
from droste.testing import (
    MockEnvironment,
    MockLLMClient,
    MockResponse,
    fake_records_provider,
)


def _operation(
    operation_id: str,
    binding_name: str,
    delivery: ResultDelivery = ResultDelivery.INLINE,
) -> ProviderOperation:
    parameters = SchemaSpec(
        {"type": "object"}, JSON_SCHEMA_2020_12, f"test:{operation_id}/params@1"
    )
    result = SchemaSpec({}, JSON_SCHEMA_2020_12, f"test:{operation_id}/result@1")
    return ProviderOperation(
        operation_id,
        binding_name,
        f"Test {operation_id}.",
        parameters,
        result,
        PaginationMode.NONE,
        delivery,
        "test.call",
    )


QUERY = CapabilityDescriptor(
    CapabilityId(
        kind=CapabilityKind.DATA,
        provider_type="test",
        source_id="db",
        operation="query",
    ),
    operation=_operation("query", "query"),
    side_effect=SideEffect.READ,
    provider_revision="1",
    provider_digest="sha256:" + "0" * 64,
)
HANDLE_QUERY = CapabilityDescriptor(
    CapabilityId(
        kind=CapabilityKind.DATA,
        provider_type="test",
        source_id="db",
        operation="export",
    ),
    operation=_operation("export", "export", ResultDelivery.HANDLE),
    side_effect=SideEffect.READ,
    provider_revision="1",
    provider_digest="sha256:" + "2" * 64,
)


class _AttemptAuthority:
    def __init__(self, *, deadline: float | None = None) -> None:
        self.deadline = deadline
        self.checkpoints: list[CapabilityCheckpoint] = []
        self.settlements: list[tuple[bool, str | None, CapabilityCheckpoint]] = []

    def admit(self, call):
        return CapabilityAdmission(
            CapabilityReservation(tokens=100, subcalls=2, wall_ms=1_000),
            self.deadline,
        )

    def checkpoint(self, call, cumulative):
        self.checkpoints.append(cumulative)
        return cumulative

    def settle(self, call, result, error, checkpoint, *, attempted):
        self.settlements.append((attempted, error.code if error else None, checkpoint))
        return CapabilityMetadata()


def test_manifest_and_envelopes_are_immutable_values() -> None:
    original_arg = {"nested": [1]}
    original_result = {"rows": [{"x": 1}]}
    manifest = CapabilityManifest((QUERY,))
    call = CapabilityCall(
        capability_id=QUERY.capability_id,
        call_id="call-1",
        run_id="run-1",
        args=("SELECT 1",),
        kwargs={"options": original_arg},
    )
    result = CapabilityResult(
        call=call, ok=True, status=CapabilityStatus.OK, result=original_result
    )
    original_arg["nested"].append(2)
    original_result["rows"][0]["x"] = 99

    with pytest.raises(FrozenInstanceError):
        QUERY.capability_id.operation = "delete"  # type: ignore[misc]
    with pytest.raises(TypeError):
        call.kwargs["options"] = {}  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.ok = False  # type: ignore[misc]
    assert manifest.find(QUERY.capability_id) is QUERY
    assert call.to_dict()["params"]["kwargs"] == {"options": {"nested": [1]}}
    assert result.to_dict()["result"] == {"rows": [{"x": 1}]}


def test_context_is_frozen_and_reports_cumulative_checkpoints() -> None:
    authority = _AttemptAuthority()
    observed: list[CapabilityExecutionContext] = []

    def handler(context: CapabilityExecutionContext) -> str:
        observed.append(context)
        assert context.checkpoint(tokens=10, subcalls=1) == CapabilityCheckpoint(10, 1)
        assert context.checkpoint(tokens=10, subcalls=1) == CapabilityCheckpoint(10, 1)
        with pytest.raises(FrozenInstanceError):
            context.run_id = "changed"  # type: ignore[misc]
        return "ok"

    result = CapabilityBroker(
        (CapabilityRegistration(QUERY, handler),),
        run_id="run-1",
        parent_run_id="parent-1",
        attempt_authority=authority,
    ).dispatch(CapabilityCall(QUERY.capability_id, "call-1", "run-1", parent_run_id="parent-1"))

    assert result.ok is True
    assert observed[0].call_id == "call-1"
    assert observed[0].reservation == CapabilityReservation(100, 2, 1_000, 0)
    assert authority.checkpoints == [CapabilityCheckpoint(10, 1)]
    assert authority.settlements == [(True, None, CapabilityCheckpoint(10, 1))]


def test_deadline_before_handler_is_typed_and_releases_at_admission_boundary() -> None:
    authority = _AttemptAuthority(deadline=10.0)
    invoked = False

    def handler(_context):
        nonlocal invoked
        invoked = True

    result = CapabilityBroker(
        (CapabilityRegistration(QUERY, handler),),
        run_id="run-1",
        attempt_authority=authority,
        clock=lambda: 10.0,
    ).dispatch(CapabilityCall(QUERY.capability_id, "call-1", "run-1"))

    assert result.status is CapabilityStatus.CANCELLED
    assert result.error is not None
    assert result.error.code == CapabilityErrorCode.DEADLINE_EXCEEDED
    assert invoked is False
    assert authority.settlements == [
        (False, CapabilityErrorCode.DEADLINE_EXCEEDED, CapabilityCheckpoint())
    ]


def test_typed_provider_deadline_uses_cancelled_status() -> None:
    result = CapabilityBroker(
        (
            CapabilityRegistration(
                QUERY,
                lambda _context: CapabilityOutcome(
                    error=CapabilityError(
                        CapabilityErrorCode.DEADLINE_EXCEEDED,
                        "CapabilityDeadlineExceeded",
                        "remote deadline elapsed",
                    )
                ),
            ),
        )
    ).call(QUERY.capability_id)

    assert result.status is CapabilityStatus.CANCELLED
    assert result.error is not None
    assert result.error.code == CapabilityErrorCode.DEADLINE_EXCEEDED


def test_cancellation_during_handler_is_observed_cooperatively() -> None:
    authority = _AttemptAuthority()
    entered = Event()
    continue_handler = Event()

    def handler(context: CapabilityExecutionContext) -> str:
        context.checkpoint(tokens=20, subcalls=1)
        entered.set()
        assert continue_handler.wait(timeout=2)
        context.check()
        return "unreachable"

    broker = CapabilityBroker(
        (CapabilityRegistration(QUERY, handler),),
        run_id="run-1",
        attempt_authority=authority,
    )
    call = CapabilityCall(QUERY.capability_id, "call-1", "run-1")
    results = []
    worker = Thread(target=lambda: results.append(broker.dispatch(call)))
    worker.start()
    assert entered.wait(timeout=2)
    assert broker.cancel("call-1") is True
    continue_handler.set()
    worker.join(timeout=2)

    assert len(results) == 1
    assert results[0].status is CapabilityStatus.CANCELLED
    assert results[0].error is not None
    assert results[0].error.code == CapabilityErrorCode.CANCELLED
    assert authority.settlements == [
        (True, CapabilityErrorCode.CANCELLED, CapabilityCheckpoint(20, 1))
    ]
    assert broker.cancel("call-1") is False


def test_cancellation_after_admission_stops_before_handler_dispatch() -> None:
    authority = _AttemptAuthority()
    guarding = Event()
    continue_guard = Event()
    invoked = False

    def guard(_call):
        guarding.set()
        assert continue_guard.wait(timeout=2)
        return None

    def handler(_context):
        nonlocal invoked
        invoked = True

    broker = CapabilityBroker(
        (CapabilityRegistration(QUERY, handler),),
        run_id="run-1",
        guard=guard,
        attempt_authority=authority,
    )
    results = []
    worker = Thread(
        target=lambda: results.append(
            broker.dispatch(CapabilityCall(QUERY.capability_id, "call-1", "run-1"))
        )
    )
    worker.start()
    assert guarding.wait(timeout=2)
    assert broker.cancel("call-1") is True
    continue_guard.set()
    worker.join(timeout=2)

    assert invoked is False
    assert results[0].error is not None
    assert results[0].error.code == CapabilityErrorCode.CANCELLED
    assert authority.settlements == [(False, CapabilityErrorCode.CANCELLED, CapabilityCheckpoint())]


def test_cancellation_wins_the_finalization_race_before_settlement() -> None:
    authority = _AttemptAuthority()
    annotating = Event()
    continue_finalization = Event()

    def annotator(call, result, error):
        annotating.set()
        assert continue_finalization.wait(timeout=2)
        return CapabilityMetadata()

    broker = CapabilityBroker(
        (CapabilityRegistration(QUERY, lambda _context: "finished"),),
        run_id="run-1",
        annotator=annotator,
        attempt_authority=authority,
    )
    results = []
    worker = Thread(
        target=lambda: results.append(
            broker.dispatch(CapabilityCall(QUERY.capability_id, "call-1", "run-1"))
        )
    )
    worker.start()
    assert annotating.wait(timeout=2)
    assert broker.cancel("call-1") is True
    continue_finalization.set()
    worker.join(timeout=2)

    assert results[0].status is CapabilityStatus.CANCELLED
    assert results[0].result is None
    assert authority.settlements[0][1] == CapabilityErrorCode.CANCELLED


def test_process_control_from_guard_and_annotator_settles_before_propagating() -> None:
    class Stop(BaseException):
        pass

    guarded_authority = _AttemptAuthority()
    guarded = CapabilityBroker(
        (CapabilityRegistration(QUERY, lambda _context: "unreachable"),),
        guard=lambda _call: (_ for _ in ()).throw(Stop()),
        attempt_authority=guarded_authority,
    )
    with pytest.raises(Stop):
        guarded.call(QUERY.capability_id)
    assert guarded_authority.settlements == [
        (False, CapabilityErrorCode.GUARD_ERROR, CapabilityCheckpoint())
    ]

    annotated_authority = _AttemptAuthority()
    annotated = CapabilityBroker(
        (CapabilityRegistration(QUERY, lambda _context: "done"),),
        annotator=lambda call, result, error: (_ for _ in ()).throw(Stop()),
        attempt_authority=annotated_authority,
    )
    with pytest.raises(Stop):
        annotated.call(QUERY.capability_id)
    assert annotated_authority.settlements == [(True, None, CapabilityCheckpoint())]


def test_validation_and_dispatch_fail_closed_without_invoking_a_handler() -> None:
    invoked = False

    def handler(_execution) -> str:
        nonlocal invoked
        invoked = True
        return "unreachable"

    broker = CapabilityBroker((CapabilityRegistration(QUERY, handler),), run_id="run-1")
    unknown = CapabilityDescriptor(
        CapabilityId(
            kind=CapabilityKind.DATA,
            provider_type="test",
            source_id="db",
            operation="drop",
        ),
        operation=_operation("drop", "drop"),
        side_effect=SideEffect.EFFECTFUL,
        provider_revision="1",
        provider_digest="sha256:" + "1" * 64,
    )
    call = CapabilityCall(unknown.capability_id, "call-1", "run-1")

    error = validate_call(broker.describe(), call)
    result = broker.dispatch(call)

    assert isinstance(error, CapabilityError)
    assert error.code == CapabilityErrorCode.NOT_ALLOWED
    assert result.ok is False
    assert result.status is CapabilityStatus.INVALID
    assert result.error is not None
    assert result.error.code == CapabilityErrorCode.NOT_ALLOWED
    assert invoked is False


def test_one_result_shape_carries_identity_error_usage_budget_and_evidence() -> None:
    observed: list[CapabilityResult] = []

    def annotator(call: CapabilityCall, result: object, error: CapabilityError | None):
        assert call.run_id == "run-1"
        assert thaw_value(result) == {"rows": 1}
        assert error is None
        return CapabilityMetadata(
            usage=(CapabilityMetric("rows", 1),),
            budget_delta=(CapabilityMetric("capability_calls", 1),),
            evidence=(
                EvidenceLocation(
                    "db",
                    "rows/7",
                    revision="rev-1",
                    ranges=(EvidenceRange(line_start=7, line_end=8),),
                ),
            ),
        )

    broker = CapabilityBroker(
        (CapabilityRegistration(QUERY, lambda _execution: {"rows": 1}),),
        run_id="run-1",
        parent_run_id="parent-1",
        annotator=annotator,
        observer=observed.append,
    )
    result = broker.call(QUERY.capability_id)
    wire = result.to_dict()

    assert result.ok is True
    assert wire["capability_id"] == {
        "kind": "data",
        "provider_type": "test",
        "source_id": "db",
        "operation": "query",
    }
    assert wire["run_id"] == "run-1"
    assert wire["parent_run_id"] == "parent-1"
    assert wire["call_id"]
    assert wire["status"] == "ok"
    assert wire["usage"] == [{"name": "rows", "value": 1, "unit": None}]
    assert wire["budget_delta"][0]["name"] == "capability_calls"
    assert wire["evidence"] == [
        {
            "source_id": "db",
            "path": "rows/7",
            "revision": "rev-1",
            "ranges": [
                {
                    "byte_start": None,
                    "byte_end": None,
                    "line_start": 7,
                    "line_end": 8,
                    "section": None,
                }
            ],
        }
    ]
    assert json.loads(json.dumps(wire))["call_id"] == wire["call_id"]
    assert observed == [result]


def test_provider_outcomes_and_finalizer_metadata_share_one_envelope_path() -> None:
    finalized: list[tuple[str, str | None]] = []

    def finalizer(
        call: CapabilityCall, result: object, error: CapabilityError | None
    ) -> CapabilityMetadata:
        finalized.append((call.call_id, error.code if error else None))
        return CapabilityMetadata(
            usage=(CapabilityMetric("accounted_calls", 1, "call"),),
            budget_delta=(CapabilityMetric("remaining_calls", -1, "call"),),
            evidence=(EvidenceLocation("audit", f"audit:{call.call_id}"),),
        )

    provider_success = CapabilityOutcome(
        result={"rows": 1},
        metadata=CapabilityMetadata(
            usage=(CapabilityMetric("rows", 1, "row"),),
            evidence=(EvidenceLocation("db", "row-7"),),
        ),
    )
    success = CapabilityBroker(
        (CapabilityRegistration(QUERY, lambda _execution: provider_success),),
        annotator=finalizer,
    ).call(QUERY.capability_id)

    provider_error = CapabilityError(
        "sqlite.not_found",
        "RecordNotFound",
        "record 7 does not exist",
    )
    failed = CapabilityBroker(
        (
            CapabilityRegistration(
                QUERY,
                lambda _execution: CapabilityOutcome(
                    error=provider_error,
                    metadata=CapabilityMetadata(
                        usage=(CapabilityMetric("lookups", 1, "call"),),
                        evidence=(EvidenceLocation("db", "lookup-7"),),
                    ),
                ),
            ),
        ),
        annotator=finalizer,
    ).call(QUERY.capability_id)
    invalid = CapabilityBroker(
        (
            CapabilityRegistration(
                QUERY,
                lambda _execution: CapabilityOutcome(
                    result=object(),
                    metadata=CapabilityMetadata(
                        usage=(CapabilityMetric("provider_attempts", 1, "call"),)
                    ),
                ),
            ),
        ),
        annotator=finalizer,
    ).call(QUERY.capability_id)

    assert thaw_value(success.result) == {"rows": 1}
    assert [item.name for item in success.usage] == ["rows", "accounted_calls"]
    assert [item.source_id for item in success.evidence] == ["db", "audit"]
    assert failed.error is provider_error
    assert failed.error.code == "sqlite.not_found"
    assert [item.name for item in failed.usage] == ["lookups", "accounted_calls"]
    assert [item.source_id for item in failed.evidence] == ["db", "audit"]
    assert invalid.error is not None
    assert invalid.error.code == CapabilityErrorCode.INVALID_RESULT
    assert [item.name for item in invalid.usage] == [
        "provider_attempts",
        "accounted_calls",
    ]
    assert finalized == [
        (success.call.call_id, None),
        (failed.call.call_id, "sqlite.not_found"),
        (invalid.call.call_id, CapabilityErrorCode.INVALID_RESULT),
    ]

    raw_registration = CapabilityRegistration(QUERY, lambda _execution: "raw")
    assert isinstance(raw_registration.handler(None), CapabilityOutcome)


def test_provider_and_finalizer_singular_metadata_conflicts_fail_closed() -> None:
    provider_handle = CapabilityResultHandle("provider-result")
    result = CapabilityBroker(
        (
            CapabilityRegistration(
                QUERY,
                lambda _execution: CapabilityOutcome(
                    result="not-inlined",
                    metadata=CapabilityMetadata(result_handle=provider_handle),
                ),
            ),
        ),
        annotator=lambda call, value, error: CapabilityMetadata(
            result_handle=CapabilityResultHandle("different-result")
        ),
    ).call(QUERY.capability_id)

    assert result.status is CapabilityStatus.ERROR
    assert result.error is not None
    assert result.error.code == CapabilityErrorCode.ANNOTATOR_ERROR
    assert result.result_handle is provider_handle
    assert result.result is None


def test_trace_projection_is_json_safe_and_contains_no_replay_content() -> None:
    secret = "secret prompt and result"

    def fail(_execution, _prompt: str) -> None:
        raise RuntimeError(f"provider exposed {secret}")

    result = CapabilityBroker(
        (CapabilityRegistration(QUERY, fail),),
        run_id="run-1",
        annotator=lambda call, value, error: CapabilityMetadata(
            evidence=(EvidenceLocation("db", f"private/path/{secret}"),)
        ),
    ).call(QUERY.capability_id, secret)

    trace = result.to_trace_dict()
    encoded = json.dumps(trace)

    assert trace["capability_id"] == QUERY.capability_id.to_dict()
    assert trace["error"] == {"code": "handler_error", "type": "RuntimeError"}
    assert trace["evidence"] == {"count": 1}
    assert "params" not in trace
    assert "result" not in trace
    assert secret not in encoded
    assert "private/path" not in encoded
    assert json.loads(encoded) == trace

    handled = CapabilityBroker(
        (CapabilityRegistration(HANDLE_QUERY, lambda _execution: secret),),
        annotator=lambda call, value, error: CapabilityMetadata(
            result_handle=CapabilityResultHandle(
                handle=f"file:///private/{secret}",
                media_type="application/json",
                size_bytes=42,
            )
        ),
    ).call(HANDLE_QUERY.capability_id)
    handled_trace = handled.to_trace_dict()
    assert handled_trace["result_handle"] == {
        "present": True,
        "media_type": "application/json",
        "size_bytes": 42,
    }
    assert secret not in json.dumps(handled_trace)


def test_result_delivery_mode_is_enforced_by_the_broker() -> None:
    missing_handle = CapabilityBroker(
        (CapabilityRegistration(HANDLE_QUERY, lambda _execution: "inline"),)
    ).call(HANDLE_QUERY.capability_id)
    unexpected_handle = CapabilityBroker(
        (
            CapabilityRegistration(
                QUERY,
                lambda _execution: CapabilityOutcome(
                    metadata=CapabilityMetadata(result_handle=CapabilityResultHandle("unexpected"))
                ),
            ),
        )
    ).call(QUERY.capability_id)

    assert missing_handle.error is not None
    assert missing_handle.error.code == CapabilityErrorCode.INVALID_RESULT
    assert unexpected_handle.error is not None
    assert unexpected_handle.error.code == CapabilityErrorCode.INVALID_RESULT


def test_accounting_metadata_rejects_non_json_values_at_construction() -> None:
    with pytest.raises(ValueError, match="finite"):
        CapabilityMetric("tokens", float("nan"))
    with pytest.raises(ValueError, match="non-negative"):
        CapabilityResultHandle("result-1", size_bytes=-1)
    with pytest.raises(ValueError, match="lowercase ASCII media type"):
        CapabilityResultHandle("result-1", media_type="private/path secret")
    with pytest.raises(ValueError, match="lowercase ASCII"):
        CapabilityError("not found: /private/path", "NotFound", "private detail")
    with pytest.raises(ValueError, match="ASCII identifier"):
        CapabilityError("provider.not_found", "Not found: /private/path", "private detail")


def test_unexpected_exception_type_is_sanitized_before_exactly_once_finalization() -> None:
    finalized: list[str] = []

    class ProviderFailure(Exception):
        pass

    ProviderFailure.__name__ = "private/path secret"
    broker = CapabilityBroker(
        (
            CapabilityRegistration(
                QUERY, lambda _execution: (_ for _ in ()).throw(ProviderFailure("detail"))
            ),
        ),
        annotator=lambda call, result, error: (
            finalized.append(call.call_id) or CapabilityMetadata()
        ),
    )

    result = broker.call(QUERY.capability_id)

    assert result.error is not None
    assert result.error.type == "Exception"
    assert finalized == [result.call.call_id]
    assert "private/path" not in json.dumps(result.to_trace_dict())


def test_guard_denial_is_typed_and_does_not_call_handler() -> None:
    invoked = False

    def handler(_execution) -> None:
        nonlocal invoked
        invoked = True

    denial = CapabilityError(
        CapabilityErrorCode.POLICY_DENIED,
        "PolicyDenied",
        "read blocked by host policy",
    )
    broker = CapabilityBroker(
        (CapabilityRegistration(QUERY, handler),),
        guard=lambda call: denial,
    )

    result = broker.call(QUERY.capability_id)

    assert result.status is CapabilityStatus.DENIED
    assert result.error is denial
    assert invoked is False


def test_run_identity_and_hook_failures_stay_in_typed_envelopes() -> None:
    invoked = False

    def handler(_execution) -> dict[str, list[int]]:
        nonlocal invoked
        invoked = True
        return {"items": [1]}

    guarded = CapabilityBroker(
        (CapabilityRegistration(QUERY, handler),),
        run_id="run-1",
        guard=lambda call: (_ for _ in ()).throw(RuntimeError("guard offline")),
    )
    guard_result = guarded.call(QUERY.capability_id)
    assert guard_result.status is CapabilityStatus.DENIED
    assert guard_result.error is not None
    assert guard_result.error.code == CapabilityErrorCode.GUARD_ERROR
    assert invoked is False

    mismatched = CapabilityBroker(
        (CapabilityRegistration(QUERY, handler),), run_id="run-1"
    ).dispatch(CapabilityCall(QUERY.capability_id, "call-1", "run-2"))
    assert mismatched.status is CapabilityStatus.INVALID
    assert mismatched.error is not None
    assert mismatched.error.type == "RunIdentityMismatch"
    assert invoked is False

    observed: list[CapabilityResult] = []
    annotated = CapabilityBroker(
        (CapabilityRegistration(QUERY, handler),),
        run_id="run-1",
        annotator=lambda call, result, error: (_ for _ in ()).throw(RuntimeError("meter offline")),
        observer=observed.append,
    ).call(QUERY.capability_id)
    assert invoked is True
    assert annotated.status is CapabilityStatus.ERROR
    assert annotated.error is not None
    assert annotated.error.code == CapabilityErrorCode.ANNOTATOR_ERROR
    assert annotated.to_dict()["result"] == {"items": [1]}
    assert observed == [annotated]


def test_post_attempt_finalizer_runs_exactly_once_and_never_before_attempt() -> None:
    finalized: list[tuple[str, str | None]] = []

    def finalizer(
        call: CapabilityCall, result: object, error: CapabilityError | None
    ) -> CapabilityMetadata:
        finalized.append((call.call_id, error.code if error else None))
        return CapabilityMetadata()

    successful = CapabilityBroker(
        (CapabilityRegistration(QUERY, lambda _execution: "ok"),), annotator=finalizer
    ).call(QUERY.capability_id)
    failed = CapabilityBroker(
        (
            CapabilityRegistration(
                QUERY, lambda _execution: (_ for _ in ()).throw(RuntimeError("handler failed"))
            ),
        ),
        annotator=finalizer,
    ).call(QUERY.capability_id)
    invalid_result = CapabilityBroker(
        (CapabilityRegistration(QUERY, lambda _execution: object()),), annotator=finalizer
    ).call(QUERY.capability_id)

    class AttemptCancelled(BaseException):
        pass

    cancelled_broker = CapabilityBroker(
        (
            CapabilityRegistration(
                QUERY, lambda _execution: (_ for _ in ()).throw(AttemptCancelled())
            ),
        ),
        annotator=finalizer,
    )
    with pytest.raises(AttemptCancelled):
        cancelled_broker.call(QUERY.capability_id)

    attempted = finalized.copy()
    assert [code for _, code in attempted] == [
        None,
        CapabilityErrorCode.HANDLER_ERROR,
        CapabilityErrorCode.INVALID_RESULT,
        CapabilityErrorCode.HANDLER_ERROR,
    ]
    assert [call_id for call_id, _ in attempted[:3]] == [
        successful.call.call_id,
        failed.call.call_id,
        invalid_result.call.call_id,
    ]
    assert len({call_id for call_id, _ in attempted}) == 4

    denied = CapabilityError(CapabilityErrorCode.POLICY_DENIED, "PolicyDenied", "not allowed")
    denied_broker = CapabilityBroker(
        (CapabilityRegistration(QUERY, lambda _execution: "unreachable"),),
        guard=lambda call: denied,
        annotator=finalizer,
    )
    denied_broker.call(QUERY.capability_id)

    guard_error_broker = CapabilityBroker(
        (CapabilityRegistration(QUERY, lambda _execution: "unreachable"),),
        guard=lambda call: (_ for _ in ()).throw(RuntimeError("guard failed")),
        annotator=finalizer,
    )
    guard_error_broker.call(QUERY.capability_id)

    rejected_broker = CapabilityBroker(
        (CapabilityRegistration(QUERY, lambda _execution: "unreachable"),), annotator=finalizer
    )
    unknown_id = CapabilityId(kind=CapabilityKind.DATA, source_id="db", operation="unknown")
    rejected_broker.dispatch(CapabilityCall(unknown_id, "unknown-call", rejected_broker.run_id))
    rejected_broker.dispatch(CapabilityCall(QUERY.capability_id, "wrong-run-call", "wrong-run"))
    rejected_broker.call(QUERY.capability_id, object())

    assert finalized == attempted


def test_generated_binding_preserves_values_and_raises_typed_compatibility_error() -> None:
    ok_broker = CapabilityBroker(
        (CapabilityRegistration(QUERY, lambda _execution, value: value + 1),)
    )
    binding = generate_binding(ok_broker, QUERY, name="query")
    assert binding(1) == 2
    assert binding.__name__ == "query"

    failed_broker = CapabilityBroker(
        (
            CapabilityRegistration(
                QUERY, lambda _execution: (_ for _ in ()).throw(ValueError("bad sql"))
            ),
        )
    )
    with pytest.raises(CapabilityCallError) as caught:
        generate_binding(failed_broker, QUERY)()
    assert caught.value.error.code == CapabilityErrorCode.HANDLER_ERROR
    assert caught.value.error.type == "ValueError"

    with pytest.raises(CapabilityCallError) as invalid:
        binding(object())
    assert invalid.value.error.code == CapabilityErrorCode.INVALID_CALL

    with pytest.raises(CapabilityCallError) as non_finite:
        binding(float("nan"))
    assert non_finite.value.error.code == CapabilityErrorCode.INVALID_CALL

    non_finite_result = CapabilityBroker(
        (CapabilityRegistration(QUERY, lambda _execution: float("inf")),)
    ).call(QUERY.capability_id)
    assert non_finite_result.error is not None
    assert non_finite_result.error.code == CapabilityErrorCode.INVALID_RESULT


class RecordingSubcalls:
    def __init__(self) -> None:
        self.query_calls: list[tuple[str, str]] = []
        self.batch_calls: list[tuple[list[str], list[str] | None]] = []
        self.error_batch_calls: list[tuple[list[str], list[str] | None]] = []

    def llm_query(self, prompt: str, context: str = "") -> str:
        self.query_calls.append((prompt, context))
        return f"answer:{prompt}:{context}"

    def llm_batch(self, prompts: list[str], contexts: list[str] | None = None) -> list[str]:
        self.batch_calls.append((prompts, contexts))
        return [f"answer:{prompt}" for prompt in prompts]

    def llm_batch_with_errors(
        self, prompts: list[str], contexts: list[str] | None = None
    ) -> tuple[list[str], list[dict[str, object]]]:
        self.error_batch_calls.append((prompts, contexts))
        return [f'{{"value":"{prompt}"}}' for prompt in prompts], []


def test_brokered_subcalls_fail_fast_when_required_batch_contract_is_missing() -> None:
    class IncompleteSubcalls:
        def llm_query(self, prompt: str, context: str = "") -> str:
            return ""

        def llm_batch(self, prompts: list[str], contexts=None) -> list[str]:
            return []

    with pytest.raises(TypeError, match="llm_batch_with_errors"):
        broker_subcalls(
            IncompleteSubcalls(),  # type: ignore[arg-type]
            create_environment_context(EnvironmentConfig(kind="native")).ledger,
        )


def _runner(
    subcalls: RecordingSubcalls,
    *,
    observer: object | None = None,
) -> RunnerEnvironment:
    registry = ProviderCatalog((fake_records_provider(),)).bind(
        (ConfiguredSource("records", "fake_records"),),
        default_source_id="records",
    )
    return RunnerEnvironment(
        context={},
        registry=registry,
        subcalls=subcalls,
        max_output_chars=10_000,
        exec_timeout_ms=0,
        capability_run_id="run-1",
        capability_observer=observer,  # type: ignore[arg-type]
    )


def test_builtin_globals_are_generated_bindings_not_raw_bound_methods() -> None:
    subcalls = RecordingSubcalls()
    environment = _runner(subcalls)
    globals_ = environment.globals()

    for binding in (
        globals_["llm_query"],
        globals_["llm_batch"],
        globals_["search"],
        globals_["records"].search,
    ):
        assert callable(binding)
        assert getattr(binding, "__self__", None) is None

    assert globals_["llm_batch"] is globals_["batch_llm_query"]
    assert globals_["llm_batch"] is globals_["llm_query_batched"]
    environment.execute(
        "one = llm_query('p', 'c')\nmany = llm_batch(['a', 'b'])\nrows = search('alpha')"
    )
    assert subcalls.query_calls == [("p", "c")]
    assert subcalls.batch_calls == [(["a", "b"], None)]
    assert globals_["rows"]["items"] == [{"id": "1", "title": "alpha"}]

    manifest_keys = {
        item.capability_id.key for item in environment.capability_broker().describe().descriptors
    }
    assert ("inference", "subcall", None, "llm_query") in manifest_keys
    assert ("inference", "subcall", None, "llm_batch") in manifest_keys
    assert ("data", "fake_records", "records", "records.search") in manifest_keys


def test_structured_batch_is_one_atomic_broker_operation() -> None:
    subcalls = RecordingSubcalls()
    environment = _runner(subcalls)
    schema = {
        "type": "object",
        "required": ["value"],
        "properties": {"value": {"type": "string"}},
    }

    result = environment.globals()["llm_batch_json"](["a", "b"], schema)

    assert result["values"] == [{"value": "a"}, {"value": "b"}]
    assert subcalls.error_batch_calls == [(["a", "b"], None)]
    assert subcalls.batch_calls == []
    assert subcalls.query_calls == []


def test_runner_rejects_a_different_runtime_subcall_client() -> None:
    environment = _runner(RecordingSubcalls())

    with pytest.raises(ValueError, match="same client"):
        environment.sandbox_subcalls(
            RecordingSubcalls(),
            create_environment_context(EnvironmentConfig(kind="native")).ledger,
        )


def test_run_loop_keeps_canonical_query_and_batch_on_the_broker_path() -> None:
    subcalls = RecordingSubcalls()
    observed: list[CapabilityResult] = []
    environment = _runner(subcalls, observer=observed.append)
    root = MockLLMClient(
        responses=[
            MockResponse(
                text=(
                    "```python\n"
                    "one = llm_query('one')\n"
                    "many = llm_query_batched(['two', 'three'])\n"
                    "answer['content'] = one + ':' + ','.join(many)\n"
                    "answer['ready'] = True\n"
                    "```"
                ),
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        ]
    )

    result = run_rlm(
        "test broker routing",
        environment=environment,
        root_llm=root,
        subcalls=subcalls,
        config=RLMConfig(),
    )

    assert result.ready is True
    assert [item.call.capability_id.operation for item in observed] == [
        "llm_query",
        "llm_batch",
    ]
    assert subcalls.query_calls == [("one", "")]
    assert subcalls.batch_calls == [(["two", "three"], None)]
    assert subcalls.error_batch_calls == []


def test_run_loop_replaces_custom_environment_raw_subcall_globals() -> None:
    raw_called = False

    def raw_query(prompt: str, context: str = "") -> str:
        nonlocal raw_called
        raw_called = True
        return "bypass"

    environment = MockEnvironment(
        {
            "answer": {"content": "", "ready": False},
            "llm_query": raw_query,
        }
    )
    subcalls = RecordingSubcalls()
    root = MockLLMClient(
        responses=[
            MockResponse(
                text=(
                    "```python\n"
                    "answer['content'] = llm_query('brokered')\n"
                    "answer['ready'] = True\n"
                    "```"
                ),
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        ]
    )

    result = run_rlm(
        "test custom routing",
        environment=environment,
        root_llm=root,
        subcalls=subcalls,
        config=RLMConfig(),
    )

    assert result.answer == "answer:brokered:"
    assert subcalls.query_calls == [("brokered", "")]
    assert raw_called is False


@pytest.mark.parametrize(
    "config",
    [
        EnvironmentConfig(kind="native"),
        EnvironmentConfig(
            kind="pyodide",
            host_managed_timeout=True,
            host_managed_isolation=True,
        ),
    ],
)
def test_native_and_pyodide_publish_the_same_manifest_and_brokered_results(
    config: EnvironmentConfig,
) -> None:
    subcalls = RecordingSubcalls()
    registry = ProviderCatalog((fake_records_provider(),)).bind(
        (ConfiguredSource("records", "fake_records"),),
        default_source_id="records",
    )
    environment = create_environment(
        config,
        context={},
        registry=registry,
        subcalls=subcalls,
        execution_context=create_environment_context(config),
        capability_run_id="run-parity",
    )

    keys = tuple(
        item.capability_id.key
        for item in environment.capability_broker().describe().descriptors  # type: ignore[attr-defined]
    )
    expected = (
        ("inference", "subcall", None, "llm_query"),
        ("inference", "subcall", None, "llm_batch"),
        ("inference", "subcall", None, "llm_batch_with_errors"),
        ("data", "fake_records", "records", "records.search"),
        ("data", "fake_records", "records", "records.fetch"),
    )
    assert keys == expected
    assert environment.globals()["llm_query"]("p") == "answer:p:"
    assert environment.globals()["search"]("alpha")["items"] == [{"id": "1", "title": "alpha"}]
