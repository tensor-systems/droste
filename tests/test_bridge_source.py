"""Manifest-driven provider bridge conformance."""

from __future__ import annotations

import json

import pytest

from droste import ConfiguredSource, ProviderCatalog, SideEffect
from droste.sources.bridge import BridgeProvider, ProviderService
from droste.testing import fake_records_provider


def _service() -> ProviderService:
    source = (
        ProviderCatalog((fake_records_provider(),))
        .bind((ConfiguredSource("records", "fake_records"),))
        .sources[0]
    )
    return ProviderService(source)


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


def test_bridge_round_trips_raw_operations_and_generic_bindings() -> None:
    registry = _remote_registry()
    from droste.capabilities import CapabilityBroker

    broker = CapabilityBroker(registry.capability_registrations())
    globals_ = registry.broker_globals(broker)

    assert globals_["search"]("alpha")["items"] == [{"id": "1", "title": "alpha"}]
    descriptor = broker.describe().descriptors[0]
    assert descriptor.capability_id.operation == "records.search"
    assert descriptor.operation.binding_name == "search"


def test_service_denies_unknown_control_and_operation_names() -> None:
    service = _service()
    unknown_control = json.loads(service.handle("getattr", "{}"))
    unknown_operation = json.loads(
        service.handle(
            "invoke",
            json.dumps({"operation_id": "records.delete", "args": [], "kwargs": {}}),
        )
    )

    assert unknown_control["ok"] is False
    assert unknown_control["error"] == {
        "type": "ValueError",
        "message": "unknown bridge method: 'getattr'",
    }
    assert unknown_operation["ok"] is False
    assert unknown_operation["error"]["type"] == "PermissionError"


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
    handlers["records.fetch"] = lambda record_id: b"not-json"
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

    def fail(record_id: str) -> object:
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
    handlers["records.fetch"] = lambda record_id: CapabilityOutcome(
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
