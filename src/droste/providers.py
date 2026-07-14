"""Descriptor-driven data providers, configured sources, and binding shell."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Any

from .capabilities import (
    CapabilityBroker,
    CapabilityDescriptor,
    CapabilityId,
    CapabilityKind,
    CapabilityRegistration,
    FrozenDict,
    PaginationMode,
    ProviderOperation,
    ResultDelivery,
    SchemaSpec,
    SideEffect,
    freeze_value,
    generate_binding,
    thaw_value,
)
from .protocols.verbs import RESERVED_NAMES, AccessorManifest, validate_binding_name

PROVIDER_PROTOCOL_VERSION = 3
_PROVIDER_TYPE_PATTERN = re.compile(r"[a-z][a-z0-9_.-]*\Z", re.ASCII)


@dataclass(frozen=True, slots=True)
class ProviderManifest:
    """Immutable source-agnostic snapshot published by one provider type."""

    provider_type: str
    revision: str
    operations: tuple[ProviderOperation, ...]
    protocol_version: int = PROVIDER_PROTOCOL_VERSION
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.protocol_version != PROVIDER_PROTOCOL_VERSION:
            raise RuntimeError(
                f"provider {self.provider_type!r} uses protocol {self.protocol_version}; "
                f"this engine speaks {PROVIDER_PROTOCOL_VERSION}"
            )
        if not isinstance(self.provider_type, str) or not _PROVIDER_TYPE_PATTERN.fullmatch(
            self.provider_type
        ):
            raise ValueError("provider_type must be a stable lowercase ASCII ID")
        if not isinstance(self.revision, str) or not self.revision:
            raise ValueError("provider manifest revision must not be empty")
        object.__setattr__(self, "operations", tuple(self.operations))
        if not self.operations:
            raise ValueError("provider manifest must declare at least one operation")
        if not all(isinstance(item, ProviderOperation) for item in self.operations):
            raise TypeError("provider manifest operations require ProviderOperation values")
        operation_ids = [item.operation_id for item in self.operations]
        binding_names = [item.binding_name for item in self.operations]
        for binding_name in binding_names:
            validate_binding_name(
                binding_name,
                subject=f"provider {self.provider_type!r} operation",
            )
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("provider manifest contains duplicate operation IDs")
        if len(binding_names) != len(set(binding_names)):
            raise ValueError("provider manifest contains duplicate binding names")
        payload = self._digest_payload()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        object.__setattr__(self, "digest", "sha256:" + hashlib.sha256(encoded).hexdigest())

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "provider_type": self.provider_type,
            "revision": self.revision,
            "operations": [item.to_dict() for item in self.operations],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._digest_payload(), "digest": self.digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProviderManifest:
        protocol_version = value.get("protocol_version")
        if isinstance(protocol_version, bool) or not isinstance(protocol_version, int):
            raise ValueError("provider manifest protocol_version must be an integer")
        operations: list[ProviderOperation] = []
        raw_operations = value.get("operations")
        if not isinstance(raw_operations, list):
            raise ValueError("provider manifest operations must be a list")
        for raw in raw_operations:
            if not isinstance(raw, Mapping):
                raise ValueError("provider manifest operation must be an object")

            def schema(name: str) -> SchemaSpec | None:
                item = raw.get(name)
                if item is None:
                    return None
                if not isinstance(item, Mapping):
                    raise ValueError(f"provider operation {name} must be an object")
                return SchemaSpec(
                    item.get("schema"),
                    str(item.get("dialect") or ""),
                    str(item.get("provenance") or ""),
                )

            parameters = schema("parameters")
            if parameters is None:
                raise ValueError("provider operation parameters are required")
            operations.append(
                ProviderOperation(
                    operation_id=str(raw.get("operation_id") or ""),
                    binding_name=str(raw.get("binding_name") or ""),
                    description=str(raw.get("description") or ""),
                    parameters=parameters,
                    result=schema("result"),
                    pagination=PaginationMode(str(raw.get("pagination") or "")),
                    delivery=ResultDelivery(str(raw.get("delivery") or "")),
                    budget_class=str(raw.get("budget_class") or ""),
                )
            )
        manifest = cls(
            provider_type=str(value.get("provider_type") or ""),
            revision=str(value.get("revision") or ""),
            operations=tuple(operations),
            protocol_version=protocol_version,
        )
        if value.get("digest") != manifest.digest:
            raise ValueError("provider manifest digest mismatch")
        return manifest


@dataclass(frozen=True, slots=True)
class ConfiguredSource:
    """Named configuration for one provider; contains no live implementation."""

    source_id: str
    provider_type: str
    config: Any = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_binding_name(self.source_id, subject="configured source")
        if not isinstance(self.provider_type, str) or not _PROVIDER_TYPE_PATTERN.fullmatch(
            self.provider_type
        ):
            raise ValueError("configured source provider_type must be a stable lowercase ASCII ID")
        frozen = freeze_value(self.config)
        if not isinstance(frozen, FrozenDict):
            raise TypeError("configured source config must be an object")
        object.__setattr__(self, "config", frozen)

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> ConfiguredSource:
        provider_type = str(spec.get("type") or "").strip()
        source_id = str(spec.get("name") or "").strip()
        if not provider_type or not source_id:
            raise ValueError("configured sources require non-empty type and name")
        config = {key: value for key, value in spec.items() if key not in {"type", "name"}}
        return cls(source_id=source_id, provider_type=provider_type, config=config)

    def config_dict(self) -> dict[str, Any]:
        return thaw_value(self.config)


@dataclass(frozen=True, slots=True)
class ProviderRuntime:
    """Live edge returned by a provider binder."""

    handlers: Mapping[str, Callable[..., Any]]
    source_description: str = ""
    stats: Callable[[], Mapping[str, Any]] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        copied = dict(self.handlers)
        if not all(isinstance(key, str) and callable(value) for key, value in copied.items()):
            raise TypeError("provider runtime handlers must map operation IDs to callables")
        object.__setattr__(self, "handlers", MappingProxyType(copied))
        if not isinstance(self.source_description, str):
            raise TypeError("provider source_description must be a string")
        if self.stats is not None and not callable(self.stats):
            raise TypeError("provider runtime stats must be callable")


ProviderBinder = Callable[[ConfiguredSource, Any], ProviderRuntime]


@dataclass(frozen=True, slots=True)
class ProviderRegistration:
    """Host-owned provider manifest, effect policy, and live binding function."""

    manifest: ProviderManifest
    effects: Mapping[str, SideEffect]
    binder: ProviderBinder = field(repr=False, compare=False)
    policy_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ProviderManifest):
            raise TypeError("provider registration requires a ProviderManifest")
        if not callable(self.binder):
            raise TypeError("provider binder must be callable")
        operation_ids = {item.operation_id for item in self.manifest.operations}
        effects = dict(self.effects)
        if set(effects) != operation_ids:
            raise ValueError("host effects must classify every provider operation exactly")
        if not all(
            isinstance(value, SideEffect) and value is not SideEffect.UNSPECIFIED
            for value in effects.values()
        ):
            raise ValueError("host effects must use explicit read/effectful classifications")
        object.__setattr__(self, "effects", MappingProxyType(effects))
        policies = {key: freeze_value(value) for key, value in self.policy_metadata.items()}
        if not set(policies).issubset(operation_ids):
            raise ValueError("policy metadata names an unknown provider operation")
        if not all(isinstance(value, FrozenDict) for value in policies.values()):
            raise TypeError("provider policy metadata values must be objects")
        object.__setattr__(self, "policy_metadata", MappingProxyType(policies))

    def bind(self, source: ConfiguredSource, context: Any = None) -> BoundSource:
        if source.provider_type != self.manifest.provider_type:
            raise ValueError("configured source provider_type does not match registration")
        runtime = self.binder(source, context)
        if not isinstance(runtime, ProviderRuntime):
            raise TypeError("provider binder must return ProviderRuntime")
        expected = {item.operation_id for item in self.manifest.operations}
        if set(runtime.handlers) != expected:
            raise ValueError("provider runtime handlers must exactly match its manifest")
        return BoundSource(source, self, runtime)


@dataclass(frozen=True, slots=True)
class BoundSource:
    source: ConfiguredSource
    registration: ProviderRegistration
    runtime: ProviderRuntime = field(repr=False, compare=False)

    def capability_registrations(self) -> tuple[CapabilityRegistration, ...]:
        manifest = self.registration.manifest
        return tuple(
            CapabilityRegistration(
                CapabilityDescriptor(
                    capability_id=CapabilityId(
                        kind=CapabilityKind.DATA,
                        provider_type=manifest.provider_type,
                        source_id=self.source.source_id,
                        operation=operation.operation_id,
                    ),
                    operation=operation,
                    side_effect=self.registration.effects[operation.operation_id],
                    provider_revision=manifest.revision,
                    provider_digest=manifest.digest,
                    policy_metadata=thaw_value(
                        self.registration.policy_metadata.get(
                            operation.operation_id, freeze_value({})
                        )
                    ),
                ),
                self.runtime.handlers[operation.operation_id],
            )
            for operation in manifest.operations
        )


@dataclass(frozen=True, slots=True)
class ProviderCatalog:
    registrations: tuple[ProviderRegistration, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "registrations", tuple(self.registrations))
        if not all(isinstance(item, ProviderRegistration) for item in self.registrations):
            raise TypeError("provider catalog requires ProviderRegistration values")
        provider_types = [item.manifest.provider_type for item in self.registrations]
        if len(provider_types) != len(set(provider_types)):
            raise ValueError("provider catalog contains duplicate provider types")

    def registration(self, provider_type: str) -> ProviderRegistration:
        for item in self.registrations:
            if item.manifest.provider_type == provider_type:
                return item
        raise ValueError(f"unknown provider type: {provider_type!r}")

    def bind(
        self,
        sources: tuple[ConfiguredSource, ...],
        *,
        context: Any = None,
        default_source_id: str | None = None,
    ) -> ProviderRegistry:
        return ProviderRegistry(
            tuple(self.registration(item.provider_type).bind(item, context) for item in sources),
            default_source_id=default_source_id,
        )


@dataclass(frozen=True, slots=True, init=False)
class ProviderRegistry:
    """Per-run immutable source bindings projected into broker and prompt views."""

    _sources: tuple[BoundSource, ...]
    _default_source_id: str | None
    _registrations: tuple[CapabilityRegistration, ...]

    def __init__(
        self,
        sources: tuple[BoundSource, ...],
        *,
        default_source_id: str | None = None,
    ) -> None:
        frozen_sources = tuple(sources)
        source_ids = [item.source.source_id for item in frozen_sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("duplicate configured source name")
        if any(source_id in RESERVED_NAMES for source_id in source_ids):
            raise ValueError("configured source name collides with a reserved global")
        if default_source_id is not None and default_source_id not in source_ids:
            raise ValueError("default source is not configured")
        registrations = tuple(
            registration
            for source in frozen_sources
            for registration in source.capability_registrations()
        )
        object.__setattr__(self, "_sources", frozen_sources)
        object.__setattr__(self, "_default_source_id", default_source_id)
        object.__setattr__(self, "_registrations", registrations)

    @property
    def sources(self) -> tuple[BoundSource, ...]:
        return self._sources

    def capability_registrations(self) -> tuple[CapabilityRegistration, ...]:
        return self._registrations

    def _descriptors(self, source_id: str) -> tuple[CapabilityDescriptor, ...]:
        return tuple(
            registration.descriptor
            for registration in self._registrations
            if registration.descriptor.capability_id.source_id == source_id
        )

    def broker_globals(self, broker: CapabilityBroker) -> dict[str, Any]:
        env: dict[str, Any] = {}
        source_ids = {item.source.source_id for item in self._sources}
        broker_manifest = broker.describe()
        for source in self._sources:
            source_id = source.source.source_id
            bindings: dict[str, Any] = {}
            for descriptor in self._descriptors(source_id):
                broker_descriptor = broker_manifest.find(descriptor.capability_id)
                if broker_descriptor is None:
                    raise ValueError(
                        f"broker manifest is missing source {source_id!r} operation "
                        f"{descriptor.capability_id.operation!r}"
                    )
                if broker_descriptor != descriptor:
                    raise ValueError(
                        f"broker descriptor drift for source {source_id!r} operation "
                        f"{descriptor.capability_id.operation!r}"
                    )
                name = validate_binding_name(
                    descriptor.operation.binding_name,
                    subject=f"operation on source {source_id!r}",
                )
                bindings[name] = generate_binding(broker, descriptor, name=name)
            env[source_id] = SimpleNamespace(**bindings)
            if self._default_source_id == source_id:
                for name, binding in bindings.items():
                    if name in source_ids:
                        raise ValueError(
                            f"flattened binding {name!r} would overwrite a source namespace"
                        )
                    env[name] = binding
        return env

    def accessor_manifest(self) -> AccessorManifest:
        flat: set[str] = set()
        namespaced: set[tuple[str, str]] = set()
        for source in self._sources:
            source_id = source.source.source_id
            names = {
                descriptor.operation.binding_name for descriptor in self._descriptors(source_id)
            }
            namespaced.update((source_id, name) for name in names)
            if self._default_source_id == source_id:
                flat.update(names)
        return AccessorManifest(frozenset(flat), frozenset(namespaced))

    def prompt_fragment(self) -> str:
        if not self._sources:
            return ""
        parts = ["## Data sources"]
        for source in self._sources:
            source_id = source.source.source_id
            descriptors = self._descriptors(source_id)
            provider_type = source.registration.manifest.provider_type
            parts.append(f"### `{source_id}` ({provider_type})")
            if source.runtime.source_description:
                parts.append(source.runtime.source_description)
            for descriptor in descriptors:
                operation = descriptor.operation
                properties = thaw_value(operation.parameters.schema).get("properties", {})
                parameters = ", ".join(properties) if isinstance(properties, dict) else ""
                parts.append(
                    f"- `{source_id}.{operation.binding_name}({parameters})`: "
                    f"{operation.description}"
                )
        parts.append(
            "Call only the operations listed for each source. Pull bounded data into "
            "Python variables, reduce it in code, and print only the result needed."
        )
        if self._default_source_id:
            parts.append(
                f"The default source `{self._default_source_id}` is also available unprefixed."
            )
        return "\n".join(parts)

    def stats(self) -> dict[str, dict[str, Any]]:
        return {
            source.source.source_id: dict(source.runtime.stats())
            for source in self._sources
            if source.runtime.stats is not None
        }


__all__ = [
    "BoundSource",
    "ConfiguredSource",
    "PROVIDER_PROTOCOL_VERSION",
    "ProviderBinder",
    "ProviderCatalog",
    "ProviderManifest",
    "ProviderRegistration",
    "ProviderRegistry",
    "ProviderRuntime",
]
