"""Manifest-driven provider bridge conformance."""

from __future__ import annotations

import json
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

import pytest

from droste import (
    CapabilityAdmission,
    CapabilityBroker,
    CapabilityCheckpoint,
    CapabilityExecutionContext,
    CapabilityMetadata,
    CapabilityOutcome,
    CapabilityReservation,
    ConfiguredSource,
    ProviderCatalog,
    SideEffect,
)
from droste.sources.bridge import BridgeProvider, ProviderService
from droste.testing import LifecycleGate, fake_records_provider, run_while_blocked


def _service() -> ProviderService:
    source = (
        ProviderCatalog((fake_records_provider(),))
        .bind((ConfiguredSource("records", "fake_records"),))
        .sources[0]
    )
    return ProviderService(source)


def _service_with_fetch(handler) -> ProviderService:
    from droste.providers import BoundSource, ProviderRuntime

    source = (
        ProviderCatalog((fake_records_provider(),))
        .bind((ConfiguredSource("records", "fake_records"),))
        .sources[0]
    )
    handlers = dict(source.runtime.handlers)
    handlers["records.fetch"] = handler
    return ProviderService(
        BoundSource(source.source, source.registration, ProviderRuntime(handlers))
    )


def _execution() -> dict:
    return {
        "version": 1,
        "call_id": "call-1",
        "run_id": "run-1",
        "parent_run_id": None,
        "deadline_remaining_ms": 1_000,
        "reservation": {"tokens": 0, "subcalls": 0, "wall_ms": 1_000, "depth": 0},
        "cancellation_requested": False,
    }


def _remote_registry(*, effect: SideEffect = SideEffect.READ):
    service = _service()
    bridge = BridgeProvider(service.handle)
    registration = bridge.registration(
        effects={"records.search": effect, "records.fetch": SideEffect.READ}
    )
    return ProviderCatalog((registration,)).bind(
        (ConfiguredSource("records", "fake_records"),),
        default_source_id="records",
    )


class _ThreadDuplexSession:
    """Bounded test pump: provider and receiver never re-enter each other."""

    def __init__(
        self,
        service: ProviderService,
        method: str,
        payload: str,
        *,
        fail_after_receives: int | None = None,
        cancel_after_receives: int | None = None,
    ) -> None:
        self._frames: Queue[str | BaseException] = Queue(maxsize=1)
        self._acks: Queue[str | BaseException] = Queue(maxsize=1)
        self._closed = Event()
        self._receives = 0
        self._fail_after_receives = fail_after_receives
        self._cancel_after_receives = cancel_after_receives
        self._call_id = json.loads(payload)["execution"]["call_id"]

        def control(frame: str) -> str:
            if self._closed.is_set():
                raise ConnectionError("duplex session closed")
            self._frames.put(frame, timeout=2)
            ack = self._acks.get(timeout=2)
            if isinstance(ack, BaseException):
                raise ack
            return ack

        def run() -> None:
            try:
                response = json.loads(service.handle_duplex(method, payload, control))
                if not response.get("ok") and not self._closed.is_set():
                    self._frames.put(RuntimeError(response["error"]["message"]), timeout=2)
            except BaseException as exc:
                if not self._closed.is_set():
                    self._frames.put(exc, timeout=2)

        self._thread = Thread(target=run, daemon=True)
        self._thread.start()

    def receive(self) -> str:
        if self._fail_after_receives is not None and self._receives >= self._fail_after_receives:
            raise ConnectionError("injected provider transport loss")
        self._receives += 1
        try:
            frame = self._frames.get(timeout=2)
        except Empty as exc:
            raise TimeoutError("duplex provider produced no frame") from exc
        if isinstance(frame, BaseException):
            raise frame
        return frame

    def send(self, ack: str) -> None:
        self._acks.put(ack, timeout=2)

    def cancellation_requested(self, call_id: str) -> bool:
        return (
            call_id == self._call_id
            and self._cancel_after_receives is not None
            and self._receives >= self._cancel_after_receives
        )

    def close(self) -> None:
        self._closed.set()
        try:
            self._acks.put_nowait(ConnectionError("duplex session closed"))
        except Exception:
            pass
        self._thread.join(timeout=2)


def _duplex_registry(
    service: ProviderService,
    *,
    fail_after_receives: int | None = None,
    cancel_after_receives: int | None = None,
):
    bridge = BridgeProvider(
        service.handle,
        duplex_call=lambda method, payload: _ThreadDuplexSession(
            service,
            method,
            payload,
            fail_after_receives=fail_after_receives,
            cancel_after_receives=cancel_after_receives,
        ),
    )
    return ProviderCatalog(
        (
            bridge.registration(
                effects={"records.search": SideEffect.READ, "records.fetch": SideEffect.READ}
            ),
        )
    ).bind((ConfiguredSource("records", "fake_records"),))


class _CapturingAuthority:
    def __init__(self, *, deadline_monotonic: float | None = None) -> None:
        self.call = None
        self.deadline_monotonic = deadline_monotonic
        self.checkpoints: list[CapabilityCheckpoint] = []
        self.checkpointed = Event()
        self.settlements: list[tuple[Any, Any, CapabilityCheckpoint, bool]] = []

    def admit(self, call):
        self.call = call
        return CapabilityAdmission(
            CapabilityReservation(10, 3, 10_000, 0),
            deadline_monotonic=self.deadline_monotonic,
        )

    def checkpoint(self, call, cumulative):
        self.checkpoints.append(cumulative)
        self.checkpointed.set()
        return cumulative

    def settle(self, call, result, error, checkpoint, *, attempted):
        self.settlements.append((result, error, checkpoint, attempted))
        return CapabilityMetadata()


class _ScriptedDuplexSession:
    def __init__(self, frames: list[str], *, fail_terminal_ack: bool = False) -> None:
        self.frames = frames
        self.acks: list[dict[str, Any]] = []
        self.fail_terminal_ack = fail_terminal_ack

    def receive(self) -> str:
        if not self.frames:
            raise ConnectionError("scripted duplex EOF")
        return self.frames.pop(0)

    def send(self, ack: str) -> None:
        parsed = json.loads(ack)
        self.acks.append(parsed)
        if self.fail_terminal_ack:
            raise ConnectionError("terminal acknowledgement was lost")

    def cancellation_requested(self, call_id: str) -> bool:
        return False

    def close(self) -> None:
        pass


def _scripted_duplex_result(frame_factory, *, fail_terminal_ack: bool = False):
    service = _service()

    def start(_method, payload):
        call_id = json.loads(payload)["execution"]["call_id"]
        return _ScriptedDuplexSession(
            [
                json.dumps(frame) if not isinstance(frame, str) else frame
                for frame in frame_factory(call_id)
            ],
            fail_terminal_ack=fail_terminal_ack,
        )

    bridge = BridgeProvider(service.handle, duplex_call=start)
    registry = ProviderCatalog(
        (
            bridge.registration(
                effects={"records.search": SideEffect.READ, "records.fetch": SideEffect.READ}
            ),
        )
    ).bind((ConfiguredSource("records", "fake_records"),))
    return CapabilityBroker(registry.capability_registrations()).call(
        registry.capability_registrations()[1].descriptor.capability_id, "7"
    )


def _frame(call_id: str, seq: int, kind: str, *, checkpoint=None, outcome=None):
    return {
        "version": 2,
        "call_id": call_id,
        "seq": seq,
        "kind": kind,
        "checkpoint": checkpoint,
        "outcome": outcome,
    }


def test_bridge_round_trips_raw_operations_and_generic_bindings() -> None:
    registry = _remote_registry()
    from droste.capabilities import CapabilityBroker

    broker = CapabilityBroker(registry.capability_registrations())
    globals_ = registry.broker_globals(broker)

    assert globals_["search"]("alpha")["items"] == [{"id": "1", "title": "alpha"}]
    descriptor = broker.describe().descriptors[0]
    assert descriptor.capability_id.operation == "records.search"
    assert descriptor.operation.binding_name == "search"


def test_bridge_round_trips_execution_facts_and_cumulative_checkpoint() -> None:
    from droste.providers import BoundSource, ProviderRuntime

    source = (
        ProviderCatalog((fake_records_provider(),))
        .bind((ConfiguredSource("records", "fake_records"),))
        .sources[0]
    )
    observed = []
    handlers = dict(source.runtime.handlers)

    def fetch(execution, record_id):
        observed.append(execution)
        execution.checkpoint(tokens=3, subcalls=1)
        return {"id": record_id}

    handlers["records.fetch"] = fetch
    bridge = BridgeProvider(
        ProviderService(
            BoundSource(source.source, source.registration, ProviderRuntime(handlers))
        ).handle
    )
    registry = ProviderCatalog(
        (
            bridge.registration(
                effects={"records.search": SideEffect.READ, "records.fetch": SideEffect.READ}
            ),
        )
    ).bind((ConfiguredSource("records", "fake_records"),))

    checkpoints = []

    class CapturingAuthority:
        def admit(self, call):
            return CapabilityAdmission(CapabilityReservation(10, 2, 1_000, 0))

        def checkpoint(self, call, cumulative):
            checkpoints.append(cumulative)
            return cumulative

        def settle(self, call, result, error, checkpoint, *, attempted):
            return CapabilityMetadata()

    broker = CapabilityBroker(
        registry.capability_registrations(), attempt_authority=CapturingAuthority()
    )
    fetch_binding = registry.broker_globals(broker)["records"].fetch

    assert fetch_binding("7") == {"id": "7"}
    assert checkpoints == [CapabilityCheckpoint(3, 1)]
    assert observed[0].call_id
    assert observed[0].run_id == broker.run_id
    assert observed[0].reservation == CapabilityReservation(10, 2, 1_000, 0)


def test_service_denies_unknown_control_and_operation_names() -> None:
    service = _service()
    unknown_control = json.loads(service.handle("getattr", "{}"))
    unknown_operation = json.loads(
        service.handle(
            "invoke",
            json.dumps(
                {
                    "operation_id": "records.delete",
                    "args": [],
                    "kwargs": {},
                    "execution": _execution(),
                }
            ),
        )
    )

    assert unknown_control["ok"] is False
    assert unknown_control["error"] == {
        "type": "ValueError",
        "message": "unknown bridge method: 'getattr'",
    }
    assert unknown_operation["ok"] is False
    assert unknown_operation["error"]["type"] == "PermissionError"


def test_service_observes_dispatch_cancellation_before_remote_handler() -> None:
    payload = {
        "operation_id": "records.fetch",
        "args": ["1"],
        "kwargs": {},
        "execution": {**_execution(), "cancellation_requested": True},
    }

    envelope = json.loads(_service().handle("invoke", json.dumps(payload)))

    assert envelope["ok"] is True
    outcome = envelope["result"]
    assert outcome["kind"] == "capability_outcome"
    assert outcome["value"]["error"]["code"] == "cancelled"
    assert outcome["checkpoint"] == {"tokens": 0, "subcalls": 0}


def test_service_observes_dispatched_deadline() -> None:
    payload = {
        "operation_id": "records.fetch",
        "args": ["1"],
        "kwargs": {},
        "execution": {**_execution(), "deadline_remaining_ms": 0},
    }

    envelope = json.loads(_service().handle("invoke", json.dumps(payload)))

    assert envelope["ok"] is True
    outcome = envelope["result"]
    assert outcome["kind"] == "capability_outcome"
    assert outcome["value"]["error"]["code"] == "deadline_exceeded"


def test_bridge_rejects_checkpoint_beyond_receiving_reservation() -> None:
    service = _service()

    def overflow(method: str, payload: str) -> str:
        envelope = json.loads(service.handle(method, payload))
        if method == "invoke" and envelope.get("ok"):
            envelope["result"]["checkpoint"] = {"tokens": 1, "subcalls": 0}
        return json.dumps(envelope)

    bridge = BridgeProvider(overflow)
    registry = ProviderCatalog(
        (
            bridge.registration(
                effects={"records.search": SideEffect.READ, "records.fetch": SideEffect.READ}
            ),
        )
    ).bind((ConfiguredSource("records", "fake_records"),))
    from droste import CapabilityCallError

    fetch = registry.broker_globals(CapabilityBroker(registry.capability_registrations()))[
        "records"
    ].fetch
    with pytest.raises(CapabilityCallError) as exc_info:
        fetch("1")
    assert exc_info.value.error.code == "handler_error"
    assert "exceed reservation" in exc_info.value.error.message


def test_receiving_host_effects_are_authoritative() -> None:
    descriptor = (
        _remote_registry(effect=SideEffect.EFFECTFUL).capability_registrations()[0].descriptor
    )
    assert descriptor.side_effect is SideEffect.EFFECTFUL

    described = _service().describe()
    assert "effects" not in described
    assert "side_effect" not in json.dumps(described)


def test_bridge_verifies_manifest_digest_and_source_identity() -> None:
    service = _service()

    def tampered(method: str, payload: str) -> str:
        envelope = json.loads(service.handle(method, payload))
        if method == "describe":
            envelope["result"]["manifest"]["revision"] = "tampered"
        return json.dumps(envelope)

    with pytest.raises(ValueError, match="digest mismatch"):
        BridgeProvider(tampered)

    bridge = BridgeProvider(service.handle)
    registration = bridge.registration(
        effects={"records.search": SideEffect.READ, "records.fetch": SideEffect.READ}
    )
    with pytest.raises(ValueError, match="bound to source"):
        registration.bind(ConfiguredSource("other", "fake_records"))


def test_bridge_requires_exact_receiving_host_effect_map() -> None:
    bridge = BridgeProvider(_service().handle)
    with pytest.raises(ValueError, match="classify every"):
        bridge.registration(effects={"records.search": SideEffect.READ})
    with pytest.raises(ValueError, match="explicit"):
        bridge.registration(
            effects={
                "records.search": SideEffect.UNSPECIFIED,
                "records.fetch": SideEffect.READ,
            }
        )
    with pytest.raises(ValueError, match="classify every"):
        bridge.registration(
            effects={
                "records.search": SideEffect.READ,
                "records.fetch": SideEffect.READ,
                "records.delete": SideEffect.EFFECTFUL,
            }
        )


def test_bridge_fails_typed_json_serialization_instead_of_stringifying() -> None:
    source = (
        ProviderCatalog((fake_records_provider(),))
        .bind((ConfiguredSource("records", "fake_records"),))
        .sources[0]
    )
    handlers = dict(source.runtime.handlers)
    handlers["records.fetch"] = lambda _execution, record_id: b"not-json"
    from droste.providers import BoundSource, ProviderRuntime

    service = ProviderService(
        BoundSource(
            source.source,
            source.registration,
            ProviderRuntime(handlers, source.runtime.source_description),
        )
    )
    bridge = BridgeProvider(service.handle)
    registry = ProviderCatalog(
        (
            bridge.registration(
                effects={"records.search": SideEffect.READ, "records.fetch": SideEffect.READ}
            ),
        )
    ).bind((ConfiguredSource("records", "fake_records"),))
    from droste.capabilities import CapabilityBroker, CapabilityCallError, CapabilityErrorCode

    fetch = registry.broker_globals(CapabilityBroker(registry.capability_registrations()))[
        "records"
    ].fetch

    with pytest.raises(CapabilityCallError) as exc_info:
        fetch("1")
    assert exc_info.value.error.code == CapabilityErrorCode.INVALID_RESULT


def test_bridge_normalizes_handler_exceptions_as_typed_outcomes() -> None:
    from droste.capabilities import CapabilityBroker, CapabilityCallError, CapabilityErrorCode
    from droste.providers import BoundSource, ProviderRuntime

    source = (
        ProviderCatalog((fake_records_provider(),))
        .bind((ConfiguredSource("records", "fake_records"),))
        .sources[0]
    )
    handlers = dict(source.runtime.handlers)

    def fail(_execution, record_id: str) -> object:
        raise LookupError(f"missing {record_id}")

    handlers["records.fetch"] = fail
    bridge = BridgeProvider(
        ProviderService(
            BoundSource(source.source, source.registration, ProviderRuntime(handlers))
        ).handle
    )
    registry = ProviderCatalog(
        (
            bridge.registration(
                effects={"records.search": SideEffect.READ, "records.fetch": SideEffect.READ}
            ),
        )
    ).bind((ConfiguredSource("records", "fake_records"),))
    fetch = registry.broker_globals(CapabilityBroker(registry.capability_registrations()))[
        "records"
    ].fetch

    with pytest.raises(CapabilityCallError) as exc_info:
        fetch("404")
    assert exc_info.value.error.code == CapabilityErrorCode.HANDLER_ERROR
    assert exc_info.value.error.type == "LookupError"
    assert exc_info.value.error.message == "missing 404"


def test_bridge_preserves_capability_outcomes_without_provider_specific_errors() -> None:
    from droste.capabilities import (
        CapabilityBroker,
        CapabilityCallError,
        CapabilityError,
        CapabilityMetadata,
        CapabilityMetric,
        CapabilityOutcome,
    )
    from droste.providers import BoundSource, ProviderRuntime

    source = (
        ProviderCatalog((fake_records_provider(),))
        .bind((ConfiguredSource("records", "fake_records"),))
        .sources[0]
    )
    handlers = dict(source.runtime.handlers)
    handlers["records.fetch"] = lambda _execution, record_id: CapabilityOutcome(
        error=CapabilityError(
            "records.not_found", "RecordNotFound", f"record {record_id} was not found"
        ),
        metadata=CapabilityMetadata(usage=(CapabilityMetric("lookups", 1, "call"),)),
    )
    service = ProviderService(
        BoundSource(source.source, source.registration, ProviderRuntime(handlers))
    )
    bridge = BridgeProvider(service.handle)
    registry = ProviderCatalog(
        (
            bridge.registration(
                effects={
                    "records.search": SideEffect.READ,
                    "records.fetch": SideEffect.READ,
                }
            ),
        )
    ).bind((ConfiguredSource("records", "fake_records"),))
    broker = CapabilityBroker(registry.capability_registrations())
    fetch = registry.broker_globals(broker)["records"].fetch

    with pytest.raises(CapabilityCallError) as exc_info:
        fetch("missing")
    assert exc_info.value.error.code == "records.not_found"
    assert exc_info.value.result.usage == (CapabilityMetric("lookups", 1, "call"),)


def test_duplex_streams_ordered_idempotent_checkpoints_to_receiving_authority() -> None:
    def fetch(execution, record_id):
        execution.checkpoint(tokens=2, subcalls=1)
        execution.checkpoint(tokens=2, subcalls=1)
        execution.checkpoint(tokens=5, subcalls=2)
        return {"id": record_id}

    authority = _CapturingAuthority()
    registry = _duplex_registry(_service_with_fetch(fetch))
    broker = CapabilityBroker(
        registry.capability_registrations(),
        attempt_authority=authority,
    )
    result = broker.call(registry.capability_registrations()[1].descriptor.capability_id, "7")

    assert result.ok is True
    assert authority.checkpoints == [
        CapabilityCheckpoint(2, 1),
        CapabilityCheckpoint(5, 2),
    ]
    assert len(authority.settlements) == 1
    assert authority.settlements[0][1:] == (
        None,
        CapabilityCheckpoint(5, 2),
        True,
    )


def test_duplex_delivers_cancellation_after_remote_entry_and_settles_once() -> None:
    gate = LifecycleGate()

    def fetch(execution, record_id):
        execution.checkpoint(tokens=3, subcalls=1)
        gate.pause()
        execution.check()
        return {"id": record_id}

    authority = _CapturingAuthority()
    registry = _duplex_registry(_service_with_fetch(fetch))
    observed = []
    broker = CapabilityBroker(
        registry.capability_registrations(),
        attempt_authority=authority,
        observer=observed.append,
    )
    capability_id = registry.capability_registrations()[1].descriptor.capability_id

    def cancel() -> None:
        assert authority.call is not None
        assert authority.checkpointed.wait(timeout=2)
        assert broker.cancel(authority.call.call_id) is True

    try:
        result = run_while_blocked(
            lambda: broker.call(capability_id, "7"), gate=gate, while_blocked=cancel
        ).require_value()

        assert result.error is not None
        assert result.error.code == "cancelled"
        assert len(authority.settlements) == 1
        assert authority.settlements[0][2:] == (CapabilityCheckpoint(3, 1), True)
        assert len(observed) == 1
        assert broker.cancel(authority.call.call_id) is False
    finally:
        registry.close()


def test_duplex_cancellation_property_is_local_and_does_not_emit_a_frame() -> None:
    observed = []

    def fetch(execution, record_id):
        observed.append(execution.cancellation_requested)
        return {"id": record_id}

    payload = {
        "bridge_protocol": 2,
        "operation_id": "records.fetch",
        "args": ["7"],
        "kwargs": {},
        "execution": _execution(),
    }
    frame_kinds = []

    def control(frame_json):
        frame = json.loads(frame_json)
        frame_kinds.append(frame["kind"])
        return json.dumps(
            {
                "version": 2,
                "call_id": frame["call_id"],
                "seq": frame["seq"],
                "ok": True,
                "error": None,
            }
        )

    response = json.loads(
        _service_with_fetch(fetch).handle_duplex("invoke", json.dumps(payload), control)
    )

    assert response["ok"] is True
    assert observed == [False]
    assert frame_kinds == ["check", "terminal"]


def test_duplex_deadline_ack_propagates_and_settles_once() -> None:
    clock_values = iter((0.0, 0.0, 3.0))
    authority = _CapturingAuthority(deadline_monotonic=2.0)
    registry = _duplex_registry(_service())
    broker = CapabilityBroker(
        registry.capability_registrations(),
        attempt_authority=authority,
        clock=lambda: next(clock_values, 3.0),
    )
    result = broker.call(registry.capability_registrations()[1].descriptor.capability_id, "7")

    assert result.error is not None
    assert result.error.code == "deadline_exceeded"
    assert result.error.type == "CapabilityDeadlineExceeded"
    assert len(authority.settlements) == 1
    assert authority.settlements[0][2:] == (CapabilityCheckpoint(), True)
    assert broker.cancel(authority.call.call_id) is False


def test_duplex_host_cancellation_enters_broker_cutoff_by_call_id() -> None:
    observed = []

    def fetch(execution, record_id):
        execution.checkpoint(tokens=3, subcalls=1)
        try:
            execution.check()
        except Exception as exc:
            observed.append(type(exc).__name__)
            raise
        return {"id": record_id}

    authority = _CapturingAuthority()
    registry = _duplex_registry(_service_with_fetch(fetch), cancel_after_receives=3)
    broker = CapabilityBroker(registry.capability_registrations(), attempt_authority=authority)
    result = broker.call(registry.capability_registrations()[1].descriptor.capability_id, "7")

    assert result.error is not None and result.error.code == "cancelled"
    assert observed == ["CapabilityCancelled"]
    assert authority.checkpoints == [CapabilityCheckpoint(3, 1)]
    assert len(authority.settlements) == 1


def test_duplex_cancellation_on_checkpoint_retains_accepted_usage() -> None:
    observed = []

    def fetch(execution, record_id):
        try:
            execution.checkpoint(tokens=3, subcalls=1)
        except Exception as exc:
            observed.append(type(exc).__name__)
            raise
        return {"id": record_id}

    authority = _CapturingAuthority()
    registry = _duplex_registry(_service_with_fetch(fetch), cancel_after_receives=2)
    broker = CapabilityBroker(registry.capability_registrations(), attempt_authority=authority)
    result = broker.call(registry.capability_registrations()[1].descriptor.capability_id, "7")

    assert result.error is not None and result.error.code == "cancelled"
    assert observed == ["CapabilityCancelled"]
    assert authority.checkpoints == [CapabilityCheckpoint(3, 1)]
    assert len(authority.settlements) == 1
    assert authority.settlements[0][2:] == (CapabilityCheckpoint(3, 1), True)


def test_duplex_rejects_overspend_without_mutating_receiving_checkpoint() -> None:
    def fetch(execution, record_id):
        execution.checkpoint(tokens=11, subcalls=1)
        return {"id": record_id}

    authority = _CapturingAuthority()
    registry = _duplex_registry(_service_with_fetch(fetch))
    broker = CapabilityBroker(registry.capability_registrations(), attempt_authority=authority)
    result = broker.call(registry.capability_registrations()[1].descriptor.capability_id, "7")

    assert result.error is not None
    assert result.error.code == "bridge.protocol_error"
    assert authority.checkpoints == []
    assert len(authority.settlements) == 1
    assert authority.settlements[0][2] == CapabilityCheckpoint()


def test_duplex_transport_loss_after_checkpoint_is_typed_and_settles_once() -> None:
    def fetch(execution, record_id):
        execution.checkpoint(tokens=4, subcalls=1)
        execution.check()
        return {"id": record_id}

    authority = _CapturingAuthority()
    # Initial dispatch check and the handler checkpoint are both acknowledged;
    # the next receive simulates a killed remote provider interpreter.
    registry = _duplex_registry(_service_with_fetch(fetch), fail_after_receives=2)
    broker = CapabilityBroker(registry.capability_registrations(), attempt_authority=authority)
    result = broker.call(registry.capability_registrations()[1].descriptor.capability_id, "7")

    assert result.error is not None
    assert result.error.code == "bridge.transport_lost"
    assert authority.checkpoints == [CapabilityCheckpoint(4, 1)]
    assert len(authority.settlements) == 1
    assert authority.settlements[0][2:] == (CapabilityCheckpoint(4, 1), True)


def test_duplex_transport_loss_before_first_frame_has_no_checkpoint() -> None:
    authority = _CapturingAuthority()
    registry = _duplex_registry(_service(), fail_after_receives=0)
    broker = CapabilityBroker(registry.capability_registrations(), attempt_authority=authority)
    result = broker.call(registry.capability_registrations()[1].descriptor.capability_id, "7")

    assert result.error is not None
    assert result.error.code == "bridge.transport_lost"
    assert authority.checkpoints == []
    assert len(authority.settlements) == 1


def test_duplex_oversized_terminal_is_protocol_error_not_transport_loss() -> None:
    def fetch(execution, record_id):
        return {"id": record_id, "payload": "x" * (8 * 1024 * 1024)}

    authority = _CapturingAuthority()
    registry = _duplex_registry(_service_with_fetch(fetch))
    broker = CapabilityBroker(registry.capability_registrations(), attempt_authority=authority)
    result = broker.call(registry.capability_registrations()[1].descriptor.capability_id, "7")

    assert result.error is not None
    assert result.error.code == "bridge.protocol_error"
    assert result.error.type == "BridgeProtocolError"
    assert result.error.message == "duplex terminal exceeds 8 MiB"
    assert len(authority.settlements) == 1


def test_duplex_concurrent_calls_keep_checkpoint_identity_isolated() -> None:
    def fetch(execution, record_id):
        value = int(record_id)
        execution.checkpoint(tokens=value, subcalls=1)
        return {"id": record_id}

    authority = _CapturingAuthority()
    registry = _duplex_registry(_service_with_fetch(fetch))
    broker = CapabilityBroker(registry.capability_registrations(), attempt_authority=authority)
    capability_id = registry.capability_registrations()[1].descriptor.capability_id
    results = []
    workers = [
        Thread(target=lambda value=value: results.append(broker.call(capability_id, value)))
        for value in ("2", "5")
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)

    assert all(not worker.is_alive() for worker in workers)
    assert sorted(result.result.items[0][1] for result in results) == ["2", "5"]
    assert sorted(authority.checkpoints, key=lambda item: item.tokens) == [
        CapabilityCheckpoint(2, 1),
        CapabilityCheckpoint(5, 1),
    ]
    assert len(authority.settlements) == 2


@pytest.mark.parametrize(
    "frames",
    [
        lambda call_id: ["not-json"],
        lambda call_id: [_frame("wrong-call", 1, "check")],
        lambda call_id: [_frame(call_id, 2, "check")],
        lambda call_id: [
            _frame(call_id, 1, "check"),
            _frame(
                call_id,
                1,
                "checkpoint",
                checkpoint={"tokens": 1, "subcalls": 0},
            ),
        ],
    ],
)
def test_duplex_malformed_wrong_call_reordered_and_conflicting_frames_fail_closed(
    frames,
) -> None:
    result = _scripted_duplex_result(frames)

    assert result.error is not None
    assert result.error.code == "bridge.protocol_error"


def test_duplex_invalid_session_factory_is_protocol_error_not_transport_loss() -> None:
    service = _service()
    bridge = BridgeProvider(service.handle, duplex_call=lambda _method, _payload: object())
    registry = ProviderCatalog(
        (
            bridge.registration(
                effects={"records.search": SideEffect.READ, "records.fetch": SideEffect.READ}
            ),
        )
    ).bind((ConfiguredSource("records", "fake_records"),))
    result = CapabilityBroker(registry.capability_registrations()).call(
        registry.capability_registrations()[1].descriptor.capability_id, "7"
    )

    assert result.error is not None
    assert result.error.code == "bridge.protocol_error"


def test_duplex_request_construction_failure_is_protocol_error() -> None:
    service = _service()
    transport_started = []
    bridge = BridgeProvider(
        service.handle,
        duplex_call=lambda _method, _payload: transport_started.append(True),
    )
    registry = ProviderCatalog(
        (
            bridge.registration(
                effects={"records.search": SideEffect.READ, "records.fetch": SideEffect.READ}
            ),
        )
    ).bind((ConfiguredSource("records", "fake_records"),))
    context = CapabilityExecutionContext(
        call_id="call-1",
        run_id="run-1",
        parent_run_id=None,
        deadline_monotonic=None,
        reservation=CapabilityReservation(),
        _check=lambda: None,
        _checkpoint=lambda checkpoint: checkpoint,
        _is_cancelled=lambda: False,
    )

    result = registry.sources[0].runtime.handlers["records.fetch"](context, object())

    assert isinstance(result, CapabilityOutcome)
    assert result.error is not None
    assert result.error.code == "bridge.protocol_error"
    assert transport_started == []


def test_duplex_service_classifies_malformed_ack_as_protocol_error() -> None:
    payload = {
        "bridge_protocol": 2,
        "operation_id": "records.fetch",
        "args": ["7"],
        "kwargs": {},
        "execution": _execution(),
    }

    response = json.loads(
        _service().handle_duplex("invoke", json.dumps(payload), lambda _frame: "not-an-ack")
    )

    assert response["ok"] is False
    assert response["error"]["type"] == "BridgeProtocolError"


@pytest.mark.parametrize("seq", [True, 1.0, 0, -1])
def test_duplex_service_rejects_non_integer_positive_ack_sequence(seq) -> None:
    payload = {
        "bridge_protocol": 2,
        "operation_id": "records.fetch",
        "args": ["7"],
        "kwargs": {},
        "execution": _execution(),
    }

    def malformed_ack(frame_json):
        frame = json.loads(frame_json)
        return json.dumps(
            {
                "version": 2,
                "call_id": frame["call_id"],
                "seq": seq,
                "ok": True,
                "error": None,
            }
        )

    response = json.loads(_service().handle_duplex("invoke", json.dumps(payload), malformed_ack))

    assert response["ok"] is False
    assert response["error"]["type"] == "BridgeProtocolError"


def test_duplex_exact_duplicate_checkpoint_is_idempotent() -> None:
    def frames(call_id):
        checkpoint = _frame(
            call_id,
            1,
            "checkpoint",
            checkpoint={"tokens": 0, "subcalls": 0},
        )
        return [
            checkpoint,
            checkpoint,
            _frame(
                call_id,
                2,
                "terminal",
                checkpoint={"tokens": 0, "subcalls": 0},
                outcome={"kind": "value", "value": {"id": "7"}},
            ),
        ]

    result = _scripted_duplex_result(frames)

    assert result.ok is True


def test_duplex_valid_terminal_wins_when_only_its_ack_is_lost() -> None:
    result = _scripted_duplex_result(
        lambda call_id: [
            _frame(
                call_id,
                1,
                "terminal",
                checkpoint={"tokens": 0, "subcalls": 0},
                outcome={"kind": "value", "value": {"id": "7"}},
            )
        ],
        fail_terminal_ack=True,
    )

    assert result.ok is True


def test_unary_remains_the_default_when_duplex_is_not_selected() -> None:
    service = _service()
    calls = []

    def unary(method, payload):
        calls.append(method)
        return service.handle(method, payload)

    registry = ProviderCatalog(
        (
            BridgeProvider(unary).registration(
                effects={"records.search": SideEffect.READ, "records.fetch": SideEffect.READ}
            ),
        )
    ).bind((ConfiguredSource("records", "fake_records"),))
    result = CapabilityBroker(registry.capability_registrations()).call(
        registry.capability_registrations()[1].descriptor.capability_id, "7"
    )

    assert result.ok is True
    assert calls == ["describe", "invoke"]
