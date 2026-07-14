"""Typed capability values and the single sandbox egress mechanism.

The broker is deliberately transport-blind.  Registrations are trusted host
values; generated bindings are the compatibility surface exposed to model
written code.  Budget policy and tracing can observe the same immutable call
and result values without becoming alternate dispatch paths.
"""

from __future__ import annotations

import hashlib
import keyword
import math
import re
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Event, Lock
from time import monotonic
from types import MappingProxyType
from typing import Any, Protocol
from uuid import uuid4

from .protocols.subcall_client import SubcallClient

JSON_SCHEMA_2020_12 = "https://json-schema.org/draft/2020-12/schema"

_ERROR_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_.-]*\Z", re.ASCII)
_ERROR_TYPE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z", re.ASCII)
_MEDIA_TYPE_PATTERN = re.compile(
    r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*\Z", re.ASCII
)
_OPERATION_ID_PATTERN = re.compile(r"[a-z][a-z0-9_.:/-]*\Z", re.ASCII)


def _exception_type_name(exc: BaseException) -> str:
    """Return a durable-safe exception type without copying arbitrary content."""

    name = type(exc).__name__
    return name if _ERROR_TYPE_PATTERN.fullmatch(name) else "Exception"


class CapabilityKind(StrEnum):
    DATA = "data"
    INFERENCE = "inference"


class SideEffect(StrEnum):
    READ = "read"
    EFFECTFUL = "effectful"
    UNSPECIFIED = "unspecified"


class PaginationMode(StrEnum):
    NONE = "none"
    CURSOR = "cursor"


class ResultDelivery(StrEnum):
    INLINE = "inline"
    HANDLE = "handle"
    UNTYPED = "untyped"


class CapabilityStatus(StrEnum):
    OK = "ok"
    INVALID = "invalid"
    DENIED = "denied"
    ERROR = "error"
    CANCELLED = "cancelled"


class CapabilityErrorCode(StrEnum):
    INVALID_CALL = "invalid_call"
    NOT_ALLOWED = "not_allowed"
    POLICY_DENIED = "policy_denied"
    GUARD_ERROR = "guard_error"
    HANDLER_ERROR = "handler_error"
    INVALID_RESULT = "invalid_result"
    ANNOTATOR_ERROR = "annotator_error"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    SETTLEMENT_ERROR = "settlement_error"


@dataclass(frozen=True, slots=True)
class FrozenList:
    items: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class FrozenTuple:
    items: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class FrozenDict:
    items: tuple[tuple[str, Any], ...]


def freeze_value(value: Any, *, _seen: frozenset[int] = frozenset()) -> Any:
    """Snapshot the transport-neutral value subset into immutable values."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("capability numbers must be finite")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (FrozenList, FrozenTuple, FrozenDict)):
        return value
    if isinstance(value, (list, tuple, Mapping)):
        identity = id(value)
        if identity in _seen:
            raise ValueError("capability values must not contain cycles")
        seen = _seen | {identity}
        if isinstance(value, list):
            return FrozenList(tuple(freeze_value(item, _seen=seen) for item in value))
        if isinstance(value, tuple):
            return FrozenTuple(tuple(freeze_value(item, _seen=seen) for item in value))
        if not all(isinstance(key, str) for key in value):
            raise TypeError("capability object keys must be strings")
        return FrozenDict(
            tuple((key, freeze_value(item, _seen=seen)) for key, item in value.items())
        )
    raise TypeError(f"unsupported capability value type: {type(value).__name__}")


def thaw_value(value: Any) -> Any:
    """Return ordinary compatibility values from an immutable ABI snapshot."""

    if isinstance(value, FrozenList):
        return [thaw_value(item) for item in value.items]
    if isinstance(value, FrozenTuple):
        return tuple(thaw_value(item) for item in value.items)
    if isinstance(value, FrozenDict):
        return {key: thaw_value(item) for key, item in value.items}
    return value


@dataclass(frozen=True, slots=True)
class SchemaSpec:
    """One explicit JSON Schema document and its origin."""

    schema: Any
    dialect: str
    provenance: str

    def __post_init__(self) -> None:
        frozen = freeze_value(self.schema)
        if not isinstance(frozen, FrozenDict):
            raise TypeError("capability schema must be a JSON object")
        object.__setattr__(self, "schema", frozen)
        if not isinstance(self.dialect, str) or not self.dialect:
            raise ValueError("schema dialect must not be empty")
        if not isinstance(self.provenance, str) or not self.provenance:
            raise ValueError("schema provenance must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": thaw_value(self.schema),
            "dialect": self.dialect,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class ProviderOperation:
    """Source-agnostic operation metadata supplied by a provider manifest."""

    operation_id: str
    binding_name: str
    description: str
    parameters: SchemaSpec
    result: SchemaSpec | None
    pagination: PaginationMode
    delivery: ResultDelivery
    budget_class: str

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str) or not _OPERATION_ID_PATTERN.fullmatch(
            self.operation_id
        ):
            raise ValueError("provider operation_id must be a stable lowercase ASCII ID")
        if (
            not isinstance(self.binding_name, str)
            or not self.binding_name.isidentifier()
            or keyword.iskeyword(self.binding_name)
            or self.binding_name.startswith("_")
        ):
            raise ValueError("provider binding_name must be a public Python identifier")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("provider operation description must not be empty")
        if not isinstance(self.parameters, SchemaSpec):
            raise TypeError("provider operation parameters must be a SchemaSpec")
        if not isinstance(self.pagination, PaginationMode):
            raise TypeError("provider operation pagination must be a PaginationMode")
        if not isinstance(self.delivery, ResultDelivery):
            raise TypeError("provider operation delivery must be a ResultDelivery")
        if not isinstance(self.budget_class, str) or not _ERROR_CODE_PATTERN.fullmatch(
            self.budget_class
        ):
            raise ValueError("provider operation budget_class must be a stable lowercase ID")
        if self.delivery is ResultDelivery.UNTYPED:
            if self.result is not None:
                raise ValueError("untyped provider operations must not declare a result schema")
        elif not isinstance(self.result, SchemaSpec):
            raise TypeError("typed provider operations require a result SchemaSpec")
        if self.pagination is PaginationMode.CURSOR:
            params = thaw_value(self.parameters.schema)
            result = thaw_value(self.result.schema) if self.result is not None else {}
            param_properties = params.get("properties") if isinstance(params, dict) else None
            result_properties = result.get("properties") if isinstance(result, dict) else None
            if not isinstance(param_properties, dict) or "cursor" not in param_properties:
                raise ValueError("cursor pagination requires a cursor parameter schema")
            if not isinstance(result_properties, dict) or "next_cursor" not in result_properties:
                raise ValueError("cursor pagination requires a next_cursor result schema")

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "binding_name": self.binding_name,
            "description": self.description,
            "parameters": self.parameters.to_dict(),
            "result": self.result.to_dict() if self.result else None,
            "pagination": self.pagination.value,
            "delivery": self.delivery.value,
            "budget_class": self.budget_class,
        }


@dataclass(frozen=True, slots=True)
class CapabilityId:
    """Stable operation identity carried by calls and results.

    ``provider_type`` names a reusable implementation type; ``source_id`` names
    one configured source instance. Descriptor documentation and policy metadata
    deliberately do not participate in dispatch identity.
    """

    kind: CapabilityKind
    operation: str
    source_id: str | None = None
    provider_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CapabilityKind):
            raise TypeError("capability kind must be a CapabilityKind")
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("capability operation must not be empty")
        if self.provider_type is not None and (
            not isinstance(self.provider_type, str) or not self.provider_type
        ):
            raise ValueError("capability provider_type must be a non-empty string")
        if self.source_id is not None and (
            not isinstance(self.source_id, str) or not self.source_id
        ):
            raise ValueError("capability source_id must be a non-empty string")
        if self.kind is CapabilityKind.DATA and not self.source_id:
            raise ValueError("data capabilities require source_id")

    @property
    def key(self) -> tuple[str, str | None, str | None, str]:
        return (self.kind.value, self.provider_type, self.source_id, self.operation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "provider_type": self.provider_type,
            "source_id": self.source_id,
            "operation": self.operation,
        }


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """Manifest metadata for one stable capability identity."""

    capability_id: CapabilityId
    operation: ProviderOperation
    side_effect: SideEffect
    provider_revision: str
    provider_digest: str
    policy_metadata: Any = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.capability_id, CapabilityId):
            raise TypeError("capability descriptor requires a CapabilityId")
        if not isinstance(self.operation, ProviderOperation):
            raise TypeError("capability descriptor requires a ProviderOperation")
        if self.capability_id.operation != self.operation.operation_id:
            raise ValueError("capability identity and descriptor operation_id disagree")
        if not isinstance(self.side_effect, SideEffect):
            raise TypeError("capability side_effect must be a SideEffect")
        if self.side_effect is SideEffect.UNSPECIFIED:
            raise ValueError("capability side_effect must be classified by the host")
        if not isinstance(self.provider_revision, str) or not self.provider_revision:
            raise ValueError("capability provider_revision must not be empty")
        if not isinstance(self.provider_digest, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.provider_digest
        ):
            raise ValueError("capability provider_digest must be a sha256 digest")
        policy_metadata = freeze_value(self.policy_metadata)
        if not isinstance(policy_metadata, FrozenDict):
            raise TypeError("capability policy_metadata must be an object")
        object.__setattr__(self, "policy_metadata", policy_metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id.to_dict(),
            "operation": self.operation.to_dict(),
            "side_effect": self.side_effect.value,
            "provider_revision": self.provider_revision,
            "provider_digest": self.provider_digest,
            "policy_metadata": thaw_value(self.policy_metadata),
        }


@dataclass(frozen=True, slots=True)
class CapabilityManifest:
    """Immutable, exact allowlist used by validation and binding generation."""

    descriptors: tuple[CapabilityDescriptor, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptors", tuple(self.descriptors))
        keys = [descriptor.capability_id for descriptor in self.descriptors]
        if len(keys) != len(set(keys)):
            raise ValueError("capability manifest contains duplicate operations")

    def find(self, capability_id: CapabilityId) -> CapabilityDescriptor | None:
        return next(
            (
                descriptor
                for descriptor in self.descriptors
                if descriptor.capability_id == capability_id
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "capabilities": [item.to_dict() for item in self.descriptors]}


@dataclass(frozen=True, slots=True)
class CapabilityMetric:
    """Transport-neutral usage or budget delta supplied by a host annotator."""

    name: str
    value: int | float
    unit: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("capability metric name must not be empty")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("capability metric value must be a number")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("capability metric value must be finite")
        if self.unit is not None and not isinstance(self.unit, str):
            raise TypeError("capability metric unit must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "unit": self.unit}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityMetric:
        return cls(
            name=value.get("name"),  # type: ignore[arg-type]
            value=value.get("value"),  # type: ignore[arg-type]
            unit=value.get("unit"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class EvidenceRange:
    """Optional byte/line/section coordinates within one evidence path."""

    byte_start: int | None = None
    byte_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    section: str | None = None

    def __post_init__(self) -> None:
        for start_name, end_name in (("byte_start", "byte_end"), ("line_start", "line_end")):
            start = getattr(self, start_name)
            end = getattr(self, end_name)
            if (start is None) != (end is None):
                raise ValueError(f"evidence {start_name}/{end_name} must be supplied together")
            if start is not None and (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end < start
            ):
                raise ValueError(f"evidence {start_name}/{end_name} must be an ordered range")
        if self.section is not None and (not isinstance(self.section, str) or not self.section):
            raise ValueError("evidence section must be a non-empty string")
        if self.byte_start is None and self.line_start is None and self.section is None:
            raise ValueError("evidence range must include byte, line, or section coordinates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "section": self.section,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvidenceRange:
        return cls(
            byte_start=value.get("byte_start"),  # type: ignore[arg-type]
            byte_end=value.get("byte_end"),  # type: ignore[arg-type]
            line_start=value.get("line_start"),  # type: ignore[arg-type]
            line_end=value.get("line_end"),  # type: ignore[arg-type]
            section=value.get("section"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class EvidenceLocation:
    source_id: str
    path: str
    revision: str | None = None
    ranges: tuple[EvidenceRange, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("evidence source_id must not be empty")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("evidence path must not be empty")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision):
            raise ValueError("evidence revision must be a non-empty string")
        object.__setattr__(self, "ranges", tuple(self.ranges))
        if not all(isinstance(item, EvidenceRange) for item in self.ranges):
            raise TypeError("evidence ranges require EvidenceRange values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "path": self.path,
            "revision": self.revision,
            "ranges": [item.to_dict() for item in self.ranges],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvidenceLocation:
        raw_ranges = value.get("ranges", [])
        if not isinstance(raw_ranges, list) or not all(
            isinstance(item, Mapping) for item in raw_ranges
        ):
            raise TypeError("evidence ranges must be a list of objects")
        return cls(
            source_id=value.get("source_id"),  # type: ignore[arg-type]
            path=value.get("path"),  # type: ignore[arg-type]
            revision=value.get("revision"),  # type: ignore[arg-type]
            ranges=tuple(EvidenceRange.from_dict(item) for item in raw_ranges),
        )


@dataclass(frozen=True, slots=True)
class CapabilityResultHandle:
    """Reference to a bounded result retained outside the envelope."""

    handle: str
    media_type: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.handle, str) or not self.handle:
            raise ValueError("capability result handle must not be empty")
        if self.media_type is not None and (
            not isinstance(self.media_type, str)
            or not _MEDIA_TYPE_PATTERN.fullmatch(self.media_type)
        ):
            raise ValueError("capability result media_type must be a lowercase ASCII media type")
        if self.size_bytes is not None and (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("capability result size_bytes must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle": self.handle,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityResultHandle:
        return cls(
            handle=value.get("handle"),  # type: ignore[arg-type]
            media_type=value.get("media_type"),  # type: ignore[arg-type]
            size_bytes=value.get("size_bytes"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CapabilityError:
    code: str
    type: str
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        normalized_code = str(self.code) if isinstance(self.code, str) else ""
        if not _ERROR_CODE_PATTERN.fullmatch(normalized_code):
            raise ValueError("capability error code must match lowercase ASCII [a-z][a-z0-9_.-]*")
        object.__setattr__(self, "code", normalized_code)
        if not isinstance(self.type, str) or not _ERROR_TYPE_PATTERN.fullmatch(self.type):
            raise ValueError("capability error type must be an ASCII identifier or dotted name")
        if not isinstance(self.message, str):
            raise TypeError("capability error message must be a string")
        if not isinstance(self.retryable, bool):
            raise TypeError("capability error retryable must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "type": self.type,
            "message": self.message,
            "retryable": self.retryable,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityError:
        return cls(
            code=value.get("code"),  # type: ignore[arg-type]
            type=value.get("type"),  # type: ignore[arg-type]
            message=value.get("message"),  # type: ignore[arg-type]
            retryable=value.get("retryable", False),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CapabilityCall:
    capability_id: CapabilityId
    call_id: str
    run_id: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    parent_run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(freeze_value(item) for item in self.args))
        object.__setattr__(
            self,
            "kwargs",
            MappingProxyType({key: freeze_value(item) for key, item in self.kwargs.items()}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "capability_id": self.capability_id.to_dict(),
            "call_id": self.call_id,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "params": {
                "args": [thaw_value(item) for item in self.args],
                "kwargs": {key: thaw_value(item) for key, item in self.kwargs.items()},
            },
        }


@dataclass(frozen=True, slots=True)
class CapabilityReservation:
    """Immutable authorization facts for one admitted capability attempt.

    ``call_id`` is deliberately absent: the enclosing execution context's
    call ID is the sole reservation identity.
    """

    tokens: int = 0
    subcalls: int = 0
    wall_ms: int = 0
    depth: int = 0

    def __post_init__(self) -> None:
        for name in ("tokens", "subcalls", "wall_ms", "depth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"capability reservation {name} must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return {
            "tokens": self.tokens,
            "subcalls": self.subcalls,
            "wall_ms": self.wall_ms,
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityReservation:
        expected = {"tokens", "subcalls", "wall_ms", "depth"}
        if set(value) != expected:
            raise ValueError("capability reservation requires tokens, subcalls, wall_ms, depth")
        return cls(**{name: value[name] for name in expected})


@dataclass(frozen=True, slots=True)
class CapabilityCheckpoint:
    """Cumulative trusted-handler usage; repeated equal facts are idempotent."""

    tokens: int = 0
    subcalls: int = 0

    def __post_init__(self) -> None:
        for name in ("tokens", "subcalls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"capability checkpoint {name} must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return {"tokens": self.tokens, "subcalls": self.subcalls}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityCheckpoint:
        if set(value) != {"tokens", "subcalls"}:
            raise ValueError("capability checkpoint requires tokens and subcalls")
        return cls(tokens=value["tokens"], subcalls=value["subcalls"])


class CapabilityCancelled(RuntimeError):
    """Cooperative cancellation observed by a trusted handler."""


class CapabilityDeadlineExceeded(RuntimeError):
    """The caller-authorized monotonic deadline has elapsed."""


@dataclass(frozen=True, slots=True)
class CapabilityExecutionContext:
    """Frozen handler context over one broker-owned mutable attempt.

    Public fields are inert facts. The two methods are the only mechanisms a
    trusted handler receives: observe cancellation/deadline, and report a
    cumulative usage value. The handler has no ledger, trace, or callback
    registration authority.
    """

    call_id: str
    run_id: str
    parent_run_id: str | None
    deadline_monotonic: float | None
    reservation: CapabilityReservation
    _check: Callable[[], None] = field(repr=False, compare=False)
    _checkpoint: Callable[[CapabilityCheckpoint], CapabilityCheckpoint] = field(
        repr=False, compare=False
    )
    _is_cancelled: Callable[[], bool] = field(repr=False, compare=False)
    _clock: Callable[[], float] = field(default=monotonic, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id:
            raise ValueError("capability execution context requires call_id and run_id")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("capability execution context requires call_id and run_id")
        if self.parent_run_id is not None and (
            not isinstance(self.parent_run_id, str) or not self.parent_run_id
        ):
            raise ValueError("capability parent_run_id must be a non-empty string or None")
        if self.deadline_monotonic is not None and (
            isinstance(self.deadline_monotonic, bool)
            or not isinstance(self.deadline_monotonic, (int, float))
            or not math.isfinite(self.deadline_monotonic)
        ):
            raise ValueError("capability deadline must be finite")
        if not isinstance(self.reservation, CapabilityReservation):
            raise TypeError("capability execution context requires a reservation")
        if not all(
            callable(item)
            for item in (self._check, self._checkpoint, self._is_cancelled, self._clock)
        ):
            raise TypeError("capability execution context mechanisms must be callable")

    def check(self) -> None:
        self._check()

    def checkpoint(self, *, tokens: int = 0, subcalls: int = 0) -> CapabilityCheckpoint:
        return self._checkpoint(CapabilityCheckpoint(tokens=tokens, subcalls=subcalls))

    @property
    def cancellation_requested(self) -> bool:
        return self._is_cancelled()

    def to_transport_dict(self) -> dict[str, Any]:
        remaining_ms: int | None = None
        if self.deadline_monotonic is not None:
            remaining_ms = max(0, round((self.deadline_monotonic - self._clock()) * 1000))
        return {
            "version": 1,
            "call_id": self.call_id,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "deadline_remaining_ms": remaining_ms,
            "reservation": self.reservation.to_dict(),
            "cancellation_requested": self.cancellation_requested,
        }


@dataclass(frozen=True, slots=True)
class CapabilityMetadata:
    """Optional facts attached by accounting/evidence integrations."""

    usage: tuple[CapabilityMetric, ...] = ()
    budget_delta: tuple[CapabilityMetric, ...] = ()
    evidence: tuple[EvidenceLocation, ...] = ()
    result_handle: CapabilityResultHandle | None = None
    child_run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", tuple(self.usage))
        object.__setattr__(self, "budget_delta", tuple(self.budget_delta))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        _validate_metadata_values(
            self.usage,
            self.budget_delta,
            self.evidence,
            self.result_handle,
            self.child_run_id,
        )

    def merged_with(self, finalizer: CapabilityMetadata) -> CapabilityMetadata:
        """Combine provider facts with post-attempt facts without aggregation.

        Provider sequences come first and finalizer sequences are appended.
        Singular facts must agree when both sides supply them; conflicting facts
        fail rather than silently choosing an owner.
        """

        if not isinstance(finalizer, CapabilityMetadata):
            raise TypeError("capability metadata can only merge with CapabilityMetadata")

        def singular(name: str, provider: Any, finalized: Any) -> Any:
            if provider is not None and finalized is not None and provider != finalized:
                raise ValueError(f"conflicting capability metadata for {name}")
            return finalized if finalized is not None else provider

        return CapabilityMetadata(
            usage=self.usage + finalizer.usage,
            budget_delta=self.budget_delta + finalizer.budget_delta,
            evidence=self.evidence + finalizer.evidence,
            result_handle=singular("result_handle", self.result_handle, finalizer.result_handle),
            child_run_id=singular("child_run_id", self.child_run_id, finalizer.child_run_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "usage": [item.to_dict() for item in self.usage],
            "budget_delta": [item.to_dict() for item in self.budget_delta],
            "evidence": [item.to_dict() for item in self.evidence],
            "result_handle": self.result_handle.to_dict() if self.result_handle else None,
            "child_run_id": self.child_run_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityMetadata:
        def objects(name: str) -> list[Mapping[str, Any]]:
            raw = value.get(name, [])
            if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
                raise TypeError(f"capability metadata {name} must be a list of objects")
            return raw

        raw_handle = value.get("result_handle")
        if raw_handle is not None and not isinstance(raw_handle, Mapping):
            raise TypeError("capability metadata result_handle must be an object")
        return cls(
            usage=tuple(CapabilityMetric.from_dict(item) for item in objects("usage")),
            budget_delta=tuple(
                CapabilityMetric.from_dict(item) for item in objects("budget_delta")
            ),
            evidence=tuple(EvidenceLocation.from_dict(item) for item in objects("evidence")),
            result_handle=(
                CapabilityResultHandle.from_dict(raw_handle) if raw_handle is not None else None
            ),
            child_run_id=value.get("child_run_id"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CapabilityOutcome:
    """Normalized trusted-handler value: result or typed provider error + facts."""

    result: Any = None
    error: CapabilityError | None = None
    metadata: CapabilityMetadata = field(default_factory=CapabilityMetadata)

    def __post_init__(self) -> None:
        if self.error is not None and not isinstance(self.error, CapabilityError):
            raise TypeError("capability outcome error must be a CapabilityError")
        if not isinstance(self.metadata, CapabilityMetadata):
            raise TypeError("capability outcome metadata must be CapabilityMetadata")
        if self.error is not None and self.result is not None:
            raise ValueError("failed capability outcome cannot also contain a result")

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": thaw_value(freeze_value(self.result)),
            "error": self.error.to_dict() if self.error else None,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityOutcome:
        raw_error = value.get("error")
        raw_metadata = value.get("metadata", {})
        if raw_error is not None and not isinstance(raw_error, Mapping):
            raise TypeError("capability outcome error must be an object")
        if not isinstance(raw_metadata, Mapping):
            raise TypeError("capability outcome metadata must be an object")
        return cls(
            result=value.get("result"),
            error=CapabilityError.from_dict(raw_error) if raw_error is not None else None,
            metadata=CapabilityMetadata.from_dict(raw_metadata),
        )


def _validate_metadata_values(
    usage: tuple[CapabilityMetric, ...],
    budget_delta: tuple[CapabilityMetric, ...],
    evidence: tuple[EvidenceLocation, ...],
    result_handle: CapabilityResultHandle | None,
    child_run_id: str | None,
) -> None:
    if not all(isinstance(item, CapabilityMetric) for item in (*usage, *budget_delta)):
        raise TypeError("capability usage and budget_delta require CapabilityMetric values")
    if not all(isinstance(item, EvidenceLocation) for item in evidence):
        raise TypeError("capability evidence requires EvidenceLocation values")
    if result_handle is not None and not isinstance(result_handle, CapabilityResultHandle):
        raise TypeError("capability result_handle must be a CapabilityResultHandle")
    if child_run_id is not None and (not isinstance(child_run_id, str) or not child_run_id):
        raise ValueError("capability child_run_id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class CapabilityResult:
    """The one result envelope returned by every broker operation."""

    call: CapabilityCall
    ok: bool
    status: CapabilityStatus
    result: Any = None
    error: CapabilityError | None = None
    usage: tuple[CapabilityMetric, ...] = ()
    budget_delta: tuple[CapabilityMetric, ...] = ()
    evidence: tuple[EvidenceLocation, ...] = ()
    result_handle: CapabilityResultHandle | None = None
    child_run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "result", freeze_value(self.result))
        object.__setattr__(self, "usage", tuple(self.usage))
        object.__setattr__(self, "budget_delta", tuple(self.budget_delta))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        _validate_metadata_values(
            self.usage,
            self.budget_delta,
            self.evidence,
            self.result_handle,
            self.child_run_id,
        )
        if self.ok != (self.status is CapabilityStatus.OK):
            raise ValueError("capability result ok/status fields disagree")
        if self.ok and self.error is not None:
            raise ValueError("successful capability result cannot contain an error")
        if not self.ok and self.error is None:
            raise ValueError("failed capability result requires an error")
        if self.result is not None and self.result_handle is not None:
            raise ValueError("capability result cannot contain both a value and a handle")

    def to_dict(self) -> dict[str, Any]:
        payload = self.call.to_dict()
        payload.update(
            {
                "ok": self.ok,
                "status": self.status.value,
                "result": thaw_value(self.result),
                "result_handle": self.result_handle.to_dict() if self.result_handle else None,
                "error": self.error.to_dict() if self.error else None,
                "usage": [item.to_dict() for item in self.usage],
                "budget_delta": [item.to_dict() for item in self.budget_delta],
                "evidence": [item.to_dict() for item in self.evidence],
                "child_run_id": self.child_run_id,
            }
        )
        return payload

    def to_trace_dict(self) -> dict[str, Any]:
        """Return the content-free outcome projection safe for durable traces.

        Arguments, keyword arguments, result values, error messages, and other
        replay content are intentionally absent. Full envelopes may still be
        retained separately by an explicitly configured replay facility.
        """

        return {
            "version": 1,
            "capability_id": self.call.capability_id.to_dict(),
            "call_id": self.call.call_id,
            "run_id": self.call.run_id,
            "parent_run_id": self.call.parent_run_id,
            "child_run_id": self.child_run_id,
            "ok": self.ok,
            "status": self.status.value,
            "error": (
                {"code": self.error.code, "type": self.error.type}
                if self.error is not None
                else None
            ),
            "usage": [item.to_dict() for item in self.usage],
            "budget_delta": [item.to_dict() for item in self.budget_delta],
            "evidence": {"count": len(self.evidence)},
            "result_handle": (
                {
                    "present": True,
                    "media_type": self.result_handle.media_type,
                    "size_bytes": self.result_handle.size_bytes,
                }
                if self.result_handle
                else None
            ),
        }


CapabilityHandler = Callable[..., Any]
NormalizedCapabilityHandler = Callable[..., CapabilityOutcome]


def _normalize_handler(handler: CapabilityHandler) -> NormalizedCapabilityHandler:
    """Adapt raw trusted handlers once so dispatch sees one outcome convention."""

    def normalized(*args: Any, **kwargs: Any) -> CapabilityOutcome:
        value = handler(*args, **kwargs)
        return value if isinstance(value, CapabilityOutcome) else CapabilityOutcome(result=value)

    return normalized


@dataclass(frozen=True, slots=True)
class CapabilityRegistration:
    descriptor: CapabilityDescriptor
    handler: NormalizedCapabilityHandler = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not callable(self.handler):
            raise TypeError("capability handler must be callable")
        object.__setattr__(self, "handler", _normalize_handler(self.handler))


class CapabilityGuard(Protocol):
    """Future policy/budget seam. Return an error to deny, otherwise ``None``."""

    def __call__(self, call: CapabilityCall) -> CapabilityError | None: ...


class CapabilityAnnotator(Protocol):
    """Exactly-once post-attempt usage/budget/evidence finalizer.

    It runs once after a handler is attempted, including handler errors, invalid
    results, and propagated cancellation. It does not run for validation or
    guard exits where no handler was attempted. It cannot mutate the frozen
    payload. A finalizer failure changes a returned envelope to a typed error so
    accounting cannot silently disappear after an operation executed.
    """

    def __call__(
        self,
        call: CapabilityCall,
        result: Any,
        error: CapabilityError | None,
    ) -> CapabilityMetadata: ...


@dataclass(frozen=True, slots=True)
class CapabilityAdmission:
    """Facts returned by the one run-scoped admission authority."""

    reservation: CapabilityReservation = CapabilityReservation()
    deadline_monotonic: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reservation, CapabilityReservation):
            raise TypeError("capability admission requires a reservation")
        if self.deadline_monotonic is not None and (
            isinstance(self.deadline_monotonic, bool)
            or not isinstance(self.deadline_monotonic, (int, float))
            or not math.isfinite(self.deadline_monotonic)
        ):
            raise ValueError("capability admission deadline must be finite")


class CapabilityAttemptAuthority(Protocol):
    """Run-scoped admission/checkpoint/settlement authority used by the broker."""

    def admit(self, call: CapabilityCall) -> CapabilityAdmission | CapabilityError: ...

    def checkpoint(
        self, call: CapabilityCall, cumulative: CapabilityCheckpoint
    ) -> CapabilityCheckpoint: ...

    def settle(
        self,
        call: CapabilityCall,
        result: Any,
        error: CapabilityError | None,
        checkpoint: CapabilityCheckpoint,
        *,
        attempted: bool,
    ) -> CapabilityMetadata: ...


class CapabilityObserver(Protocol):
    """Trace seam. Observes an immutable result envelope, never dispatches."""

    def __call__(self, result: CapabilityResult) -> None: ...


def validate_call(
    manifest: CapabilityManifest, call: CapabilityCall
) -> CapabilityDescriptor | CapabilityError:
    """Purely resolve one call against the exact immutable allowlist."""

    if not call.call_id or not call.run_id:
        return CapabilityError(
            CapabilityErrorCode.INVALID_CALL,
            "InvalidCapabilityCall",
            "call_id and run_id must not be empty",
        )
    if not isinstance(call.args, tuple) or not isinstance(call.kwargs, Mapping):
        return CapabilityError(
            CapabilityErrorCode.INVALID_CALL,
            "InvalidCapabilityCall",
            "params must contain an args tuple and kwargs mapping",
        )
    if not all(isinstance(key, str) for key in call.kwargs):
        return CapabilityError(
            CapabilityErrorCode.INVALID_CALL,
            "InvalidCapabilityCall",
            "capability keyword names must be strings",
        )
    descriptor = manifest.find(call.capability_id)
    if descriptor is None:
        return CapabilityError(
            CapabilityErrorCode.NOT_ALLOWED,
            "CapabilityNotAllowed",
            f"capability operation is not allowed: {call.capability_id.key!r}",
        )
    return descriptor


class CapabilityCallError(RuntimeError):
    """Compatibility-binding exception backed by a typed broker error."""

    def __init__(self, result: CapabilityResult) -> None:
        if result.error is None:
            raise ValueError("CapabilityCallError requires a failed result")
        self.result = result
        self.error = result.error
        super().__init__(f"{result.error.type}: {result.error.message}")


class _CapabilityAttemptController:
    """Broker-owned mutable lifecycle behind one frozen handler context."""

    def __init__(
        self,
        call: CapabilityCall,
        admission: CapabilityAdmission,
        authority: CapabilityAttemptAuthority | None,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.call = call
        self._authority = authority
        self._clock = clock
        self._cancelled = Event()
        self._lock = Lock()
        self._transition_lock = Lock()
        self._checkpoint_value = CapabilityCheckpoint()
        self._finishing = False
        self._settled = False
        self.context = CapabilityExecutionContext(
            call_id=call.call_id,
            run_id=call.run_id,
            parent_run_id=call.parent_run_id,
            deadline_monotonic=admission.deadline_monotonic,
            reservation=admission.reservation,
            _check=self.check,
            _checkpoint=self.checkpoint,
            _is_cancelled=self._cancelled.is_set,
            _clock=clock,
        )

    def cancel(self) -> bool:
        with self._lock:
            if self._finishing or self._settled:
                return False
            self._cancelled.set()
            return True

    def _terminal_error_locked(self) -> CapabilityError | None:
        if self._cancelled.is_set():
            return CapabilityError(
                CapabilityErrorCode.CANCELLED,
                "CapabilityCancelled",
                "capability call was cancelled",
            )
        deadline = self.context.deadline_monotonic
        if deadline is not None and self._clock() >= deadline:
            return CapabilityError(
                CapabilityErrorCode.DEADLINE_EXCEEDED,
                "CapabilityDeadlineExceeded",
                "capability call deadline exceeded",
            )
        return None

    def check(self) -> None:
        with self._lock:
            if self._finishing or self._settled:
                raise RuntimeError("capability attempt is already finalizing")
            error = self._terminal_error_locked()
        if error is not None:
            if error.code == CapabilityErrorCode.CANCELLED:
                raise CapabilityCancelled(error.message)
            raise CapabilityDeadlineExceeded(error.message)

    def checkpoint(self, cumulative: CapabilityCheckpoint) -> CapabilityCheckpoint:
        if not isinstance(cumulative, CapabilityCheckpoint):
            raise TypeError("capability checkpoint requires a CapabilityCheckpoint")
        with self._transition_lock:
            with self._lock:
                if self._finishing or self._settled:
                    raise RuntimeError("capability attempt is already finalizing")
                error = self._terminal_error_locked()
                if error is not None:
                    if error.code == CapabilityErrorCode.CANCELLED:
                        raise CapabilityCancelled(error.message)
                    raise CapabilityDeadlineExceeded(error.message)
                previous = self._checkpoint_value
                if cumulative.tokens < previous.tokens or cumulative.subcalls < previous.subcalls:
                    raise ValueError("capability checkpoint cannot move backward")
                if cumulative.tokens > self.context.reservation.tokens:
                    raise ValueError("capability checkpoint tokens exceed reservation")
                if cumulative.subcalls > self.context.reservation.subcalls:
                    raise ValueError("capability checkpoint subcalls exceed reservation")
                if cumulative == previous:
                    return previous
            # Accounting may synchronously deliver observational events. Never
            # hold the attempt state lock while calling that external seam.
            if self._authority is not None:
                cumulative = self._authority.checkpoint(self.call, cumulative)
                if not isinstance(cumulative, CapabilityCheckpoint):
                    raise TypeError("attempt authority must return a CapabilityCheckpoint")
            with self._lock:
                self._checkpoint_value = cumulative
                error = self._terminal_error_locked()
            if error is not None:
                if error.code == CapabilityErrorCode.CANCELLED:
                    raise CapabilityCancelled(error.message)
                raise CapabilityDeadlineExceeded(error.message)
            return cumulative

    def begin_finish(self) -> CapabilityError | None:
        """Close the cancellation race and return its authoritative terminal fact."""

        with self._transition_lock:
            with self._lock:
                if self._finishing or self._settled:
                    raise RuntimeError("capability attempt was finalized more than once")
                error = self._terminal_error_locked()
                self._finishing = True
                return error

    def settle(
        self,
        result: Any,
        error: CapabilityError | None,
        *,
        attempted: bool,
    ) -> CapabilityMetadata:
        with self._lock:
            if not self._finishing or self._settled:
                raise RuntimeError("capability attempt settlement is out of order")
            checkpoint = self._checkpoint_value
        try:
            metadata = (
                self._authority.settle(
                    self.call,
                    result,
                    error,
                    checkpoint,
                    attempted=attempted,
                )
                if self._authority is not None
                else CapabilityMetadata()
            )
            if not isinstance(metadata, CapabilityMetadata):
                raise TypeError("attempt authority must return CapabilityMetadata")
            return metadata
        finally:
            with self._lock:
                self._settled = True


class CapabilityBroker:
    """Thin imperative shell around pure allowlist validation."""

    def __init__(
        self,
        registrations: tuple[CapabilityRegistration, ...],
        *,
        run_id: str | None = None,
        parent_run_id: str | None = None,
        guard: CapabilityGuard | None = None,
        annotator: CapabilityAnnotator | None = None,
        observer: CapabilityObserver | None = None,
        attempt_authority: CapabilityAttemptAuthority | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._manifest = CapabilityManifest(tuple(item.descriptor for item in registrations))
        self._handlers = MappingProxyType(
            {item.descriptor.capability_id: item.handler for item in registrations}
        )
        self._run_id = run_id or str(uuid4())
        self._parent_run_id = parent_run_id
        self._guard = guard
        self._annotator = annotator
        self._observer = observer
        self._attempt_authority = attempt_authority
        self._clock = clock
        self._active_attempts: dict[str, _CapabilityAttemptController] = {}
        self._active_lock = Lock()

    @property
    def run_id(self) -> str:
        return self._run_id

    def describe(self) -> CapabilityManifest:
        return self._manifest

    def cancel(self, call_id: str) -> bool:
        """Request cooperative cancellation for one currently admitted call."""

        with self._active_lock:
            attempt = self._active_attempts.get(call_id)
        return attempt.cancel() if attempt is not None else False

    def call(self, capability_id: CapabilityId, *args: Any, **kwargs: Any) -> CapabilityResult:
        call_id = str(uuid4())
        try:
            call = CapabilityCall(
                capability_id=capability_id,
                call_id=call_id,
                run_id=self._run_id,
                parent_run_id=self._parent_run_id,
                args=tuple(args),
                kwargs=kwargs,
            )
        except Exception as exc:
            call = CapabilityCall(
                capability_id=capability_id,
                call_id=call_id,
                run_id=self._run_id,
                parent_run_id=self._parent_run_id,
            )
            return self._finish(
                call,
                status=CapabilityStatus.INVALID,
                error=CapabilityError(
                    CapabilityErrorCode.INVALID_CALL,
                    _exception_type_name(exc),
                    str(exc),
                ),
                annotate=False,
            )
        return self.dispatch(call)

    def dispatch(self, call: CapabilityCall) -> CapabilityResult:
        if call.run_id != self._run_id:
            return self._finish(
                call,
                status=CapabilityStatus.INVALID,
                error=CapabilityError(
                    CapabilityErrorCode.INVALID_CALL,
                    "RunIdentityMismatch",
                    "capability call run_id does not match this broker",
                ),
                annotate=False,
            )
        validated = validate_call(self._manifest, call)
        if isinstance(validated, CapabilityError):
            return self._finish(
                call, status=CapabilityStatus.INVALID, error=validated, annotate=False
            )
        with self._active_lock:
            duplicate = call.call_id in self._active_attempts
        if duplicate:
            return self._finish(
                call,
                status=CapabilityStatus.INVALID,
                error=CapabilityError(
                    CapabilityErrorCode.INVALID_CALL,
                    "DuplicateCapabilityCall",
                    "call_id is already active",
                ),
                annotate=False,
            )

        try:
            admission = (
                self._attempt_authority.admit(call)
                if self._attempt_authority is not None
                else CapabilityAdmission()
            )
        except Exception as exc:
            return self._finish(
                call,
                status=CapabilityStatus.DENIED,
                error=CapabilityError(
                    CapabilityErrorCode.GUARD_ERROR,
                    _exception_type_name(exc),
                    str(exc),
                ),
                annotate=False,
            )
        if isinstance(admission, CapabilityError):
            return self._finish(
                call,
                status=CapabilityStatus.DENIED,
                error=admission,
                annotate=False,
            )
        if not isinstance(admission, CapabilityAdmission):
            return self._finish(
                call,
                status=CapabilityStatus.DENIED,
                error=CapabilityError(
                    CapabilityErrorCode.GUARD_ERROR,
                    "InvalidCapabilityAdmission",
                    "attempt authority must return CapabilityAdmission or CapabilityError",
                ),
                annotate=False,
            )
        attempt = _CapabilityAttemptController(
            call, admission, self._attempt_authority, clock=self._clock
        )
        with self._active_lock:
            raced = call.call_id in self._active_attempts
            if not raced:
                self._active_attempts[call.call_id] = attempt
        if raced:
            return self._finish(
                call,
                status=CapabilityStatus.INVALID,
                error=CapabilityError(
                    CapabilityErrorCode.INVALID_CALL,
                    "DuplicateCapabilityCall",
                    "call_id is already active",
                ),
                annotate=False,
                attempt=attempt,
                attempted=False,
                remove_active=False,
            )

        try:
            attempt.check()
        except (CapabilityCancelled, CapabilityDeadlineExceeded) as exc:
            return self._finish(
                call,
                status=CapabilityStatus.CANCELLED,
                error=self._cooperative_error(exc),
                annotate=False,
                attempt=attempt,
                attempted=False,
            )

        if self._guard is not None:
            try:
                denial = self._guard(call)
                if denial is not None and not isinstance(denial, CapabilityError):
                    raise TypeError("capability guard must return CapabilityError or None")
            except BaseException as exc:
                envelope = self._finish(
                    call,
                    status=CapabilityStatus.DENIED,
                    error=CapabilityError(
                        CapabilityErrorCode.GUARD_ERROR,
                        _exception_type_name(exc),
                        str(exc),
                    ),
                    annotate=False,
                    attempt=attempt,
                    attempted=False,
                )
                if not isinstance(exc, Exception):
                    raise
                return envelope
            if denial is not None:
                return self._finish(
                    call,
                    status=CapabilityStatus.DENIED,
                    error=denial,
                    annotate=False,
                    attempt=attempt,
                    attempted=False,
                )

        try:
            attempt.check()
        except (CapabilityCancelled, CapabilityDeadlineExceeded) as exc:
            return self._finish(
                call,
                status=CapabilityStatus.CANCELLED,
                error=self._cooperative_error(exc),
                annotate=False,
                attempt=attempt,
                attempted=False,
            )

        try:
            outcome = self._handlers[validated.capability_id](
                attempt.context,
                *(thaw_value(item) for item in call.args),
                **{key: thaw_value(item) for key, item in call.kwargs.items()},
            )
        except BaseException as exc:
            error = (
                self._cooperative_error(exc)
                if isinstance(exc, (CapabilityCancelled, CapabilityDeadlineExceeded))
                else CapabilityError(
                    CapabilityErrorCode.HANDLER_ERROR,
                    _exception_type_name(exc),
                    str(exc),
                )
            )
            envelope = self._finish(
                call,
                status=(
                    CapabilityStatus.CANCELLED
                    if isinstance(exc, (CapabilityCancelled, CapabilityDeadlineExceeded))
                    else CapabilityStatus.ERROR
                ),
                error=error,
                attempt=attempt,
                attempted=True,
            )
            if not isinstance(exc, Exception):
                # Cancellation and process-control exits keep their propagation
                # semantics, but the post-attempt finalizer still runs exactly once.
                raise
            return envelope
        if not isinstance(outcome, CapabilityOutcome):
            # Registrations normalize raw handlers, so reaching this branch
            # indicates a broken trusted adapter rather than a provider error.
            return self._finish(
                call,
                status=CapabilityStatus.ERROR,
                error=CapabilityError(
                    CapabilityErrorCode.HANDLER_ERROR,
                    "InvalidCapabilityOutcome",
                    "normalized handler did not return CapabilityOutcome",
                ),
                attempt=attempt,
                attempted=True,
            )
        if outcome.error is not None:
            if outcome.error.code == CapabilityErrorCode.CANCELLED:
                # A trusted transport/provider cancellation becomes a
                # broker-owned terminal fact before the finalization cutoff.
                attempt.cancel()
            return self._finish(
                call,
                status=(
                    CapabilityStatus.CANCELLED
                    if outcome.error.code
                    in {
                        CapabilityErrorCode.CANCELLED,
                        CapabilityErrorCode.DEADLINE_EXCEEDED,
                    }
                    else CapabilityStatus.ERROR
                ),
                error=outcome.error,
                provider_metadata=outcome.metadata,
                attempt=attempt,
                attempted=True,
            )
        try:
            frozen_result = freeze_value(outcome.result)
        except BaseException as exc:
            envelope = self._finish(
                call,
                status=CapabilityStatus.ERROR,
                error=CapabilityError(
                    CapabilityErrorCode.INVALID_RESULT,
                    _exception_type_name(exc),
                    str(exc),
                ),
                provider_metadata=outcome.metadata,
                attempt=attempt,
                attempted=True,
            )
            if not isinstance(exc, Exception):
                raise
            return envelope
        return self._finish(
            call,
            status=CapabilityStatus.OK,
            result=frozen_result,
            provider_metadata=outcome.metadata,
            attempt=attempt,
            attempted=True,
        )

    @staticmethod
    def _cooperative_error(exc: BaseException) -> CapabilityError:
        if isinstance(exc, CapabilityCancelled):
            return CapabilityError(
                CapabilityErrorCode.CANCELLED,
                "CapabilityCancelled",
                str(exc) or "capability call was cancelled",
            )
        return CapabilityError(
            CapabilityErrorCode.DEADLINE_EXCEEDED,
            "CapabilityDeadlineExceeded",
            str(exc) or "capability call deadline exceeded",
        )

    def _finish(
        self,
        call: CapabilityCall,
        *,
        status: CapabilityStatus,
        result: Any = None,
        error: CapabilityError | None = None,
        annotate: bool = True,
        provider_metadata: CapabilityMetadata | None = None,
        attempt: _CapabilityAttemptController | None = None,
        attempted: bool = False,
        remove_active: bool = True,
    ) -> CapabilityResult:
        metadata = provider_metadata or CapabilityMetadata()
        annotation_error: CapabilityError | None = None
        annotation_control: BaseException | None = None
        if attempt is not None:
            terminal_error = attempt.begin_finish()
            if terminal_error is not None:
                status = CapabilityStatus.CANCELLED
                error = terminal_error
                result = None
        if annotate and self._annotator is not None:
            try:
                finalized = self._annotator(call, result, error)
                if not isinstance(finalized, CapabilityMetadata):
                    raise TypeError("capability annotator must return CapabilityMetadata")
                metadata = metadata.merged_with(finalized)
            except BaseException as exc:
                annotation_error = CapabilityError(
                    CapabilityErrorCode.ANNOTATOR_ERROR,
                    _exception_type_name(exc),
                    str(exc),
                )
                if not isinstance(exc, Exception):
                    annotation_control = exc
        if attempt is not None:
            try:
                metadata = metadata.merged_with(attempt.settle(result, error, attempted=attempted))
            except Exception as exc:
                annotation_error = CapabilityError(
                    CapabilityErrorCode.SETTLEMENT_ERROR,
                    _exception_type_name(exc),
                    str(exc),
                )
            finally:
                if remove_active:
                    with self._active_lock:
                        self._active_attempts.pop(call.call_id, None)
        delivery_error: CapabilityError | None = None
        descriptor = self._manifest.find(call.capability_id)
        if status is CapabilityStatus.OK and annotation_error is None and descriptor is not None:
            delivery = descriptor.operation.delivery
            if delivery is ResultDelivery.HANDLE and metadata.result_handle is None:
                delivery_error = CapabilityError(
                    CapabilityErrorCode.INVALID_RESULT,
                    "ResultDeliveryMismatch",
                    "handle delivery requires a CapabilityResultHandle",
                )
            elif delivery is ResultDelivery.INLINE and metadata.result_handle is not None:
                delivery_error = CapabilityError(
                    CapabilityErrorCode.INVALID_RESULT,
                    "ResultDeliveryMismatch",
                    "inline delivery cannot return a CapabilityResultHandle",
                )
        delivery_override = annotation_error or delivery_error
        terminal_error = delivery_override or error
        envelope = CapabilityResult(
            call=call,
            ok=status is CapabilityStatus.OK and terminal_error is None,
            status=(CapabilityStatus.ERROR if delivery_override is not None else status),
            # An annotator-provided handle deliberately replaces the inline value.
            result=result if metadata.result_handle is None else None,
            error=terminal_error,
            usage=metadata.usage,
            budget_delta=metadata.budget_delta,
            evidence=metadata.evidence,
            result_handle=metadata.result_handle,
            child_run_id=metadata.child_run_id,
        )
        self.emit(envelope)
        if annotation_control is not None:
            raise annotation_control
        return envelope

    def emit(self, result: CapabilityResult) -> None:
        """Notify the optional trace sink without making it an authority path."""

        if self._observer is None:
            return
        try:
            self._observer(result)
        except Exception as exc:
            warnings.warn(
                f"capability observer failed: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def generate_binding(
    broker: CapabilityBroker,
    descriptor: CapabilityDescriptor,
    *,
    name: str | None = None,
) -> Callable[..., Any]:
    """Generate one model-facing function over the typed broker envelope."""

    def binding(*args: Any, **kwargs: Any) -> Any:
        envelope = broker.call(descriptor.capability_id, *args, **kwargs)
        if not envelope.ok:
            raise CapabilityCallError(envelope)
        return (
            thaw_value(envelope.result)
            if envelope.result_handle is None
            else envelope.result_handle
        )

    binding.__name__ = name or descriptor.operation.binding_name
    binding.__qualname__ = binding.__name__
    binding.__doc__ = descriptor.operation.description
    return binding


def _schema(schema: dict[str, Any], provenance: str) -> SchemaSpec:
    return SchemaSpec(schema, JSON_SCHEMA_2020_12, provenance)


_SUBCALL_REVISION = "1"
_SUBCALL_DIGEST = "sha256:" + hashlib.sha256(b"droste.subcall-provider.v1").hexdigest()
_QUERY_OPERATION = ProviderOperation(
    operation_id="llm_query",
    binding_name="llm_query",
    description="Ask one sub-LLM using an optional context string.",
    parameters=_schema(
        {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "context": {"type": "string", "default": ""},
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
        "droste:subcall/llm_query/parameters@1",
    ),
    result=_schema({"type": "string"}, "droste:subcall/llm_query/result@1"),
    pagination=PaginationMode.NONE,
    delivery=ResultDelivery.INLINE,
    budget_class="inference.query",
)
_BATCH_OPERATION = ProviderOperation(
    operation_id="llm_batch",
    binding_name="llm_batch",
    description="Ask a sub-LLM for an ordered batch of prompts atomically.",
    parameters=_schema(
        {
            "type": "object",
            "properties": {
                "prompts": {"type": "array", "items": {"type": "string"}},
                "contexts": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ]
                },
            },
            "required": ["prompts"],
            "additionalProperties": False,
        },
        "droste:subcall/llm_batch/parameters@1",
    ),
    result=_schema(
        {"type": "array", "items": {"type": "string"}},
        "droste:subcall/llm_batch/result@1",
    ),
    pagination=PaginationMode.NONE,
    delivery=ResultDelivery.INLINE,
    budget_class="inference.batch",
)
_BATCH_ERRORS_OPERATION = ProviderOperation(
    operation_id="llm_batch_with_errors",
    binding_name="llm_batch_with_errors",
    description="Internal ordered batch operation retaining typed per-item errors.",
    parameters=_BATCH_OPERATION.parameters,
    result=None,
    pagination=PaginationMode.NONE,
    delivery=ResultDelivery.UNTYPED,
    budget_class="inference.batch",
)


LLM_QUERY_CAPABILITY = CapabilityDescriptor(
    CapabilityId(
        kind=CapabilityKind.INFERENCE,
        provider_type="subcall",
        operation="llm_query",
    ),
    operation=_QUERY_OPERATION,
    side_effect=SideEffect.READ,
    provider_revision=_SUBCALL_REVISION,
    provider_digest=_SUBCALL_DIGEST,
)
LLM_BATCH_CAPABILITY = CapabilityDescriptor(
    CapabilityId(
        kind=CapabilityKind.INFERENCE,
        provider_type="subcall",
        operation="llm_batch",
    ),
    operation=_BATCH_OPERATION,
    side_effect=SideEffect.READ,
    provider_revision=_SUBCALL_REVISION,
    provider_digest=_SUBCALL_DIGEST,
)
LLM_BATCH_WITH_ERRORS_CAPABILITY = CapabilityDescriptor(
    CapabilityId(
        kind=CapabilityKind.INFERENCE,
        provider_type="subcall",
        operation="llm_batch_with_errors",
    ),
    operation=_BATCH_ERRORS_OPERATION,
    side_effect=SideEffect.READ,
    provider_revision=_SUBCALL_REVISION,
    provider_digest=_SUBCALL_DIGEST,
)

_MISSING_OUTPUT_TOKEN_LIMIT = object()


def _reported_output_token_limit(subcalls: SubcallClient) -> int | None | object:
    try:
        limit = getattr(subcalls, "output_token_limit")
    except Exception:
        return _MISSING_OUTPUT_TOKEN_LIMIT
    if limit is None:
        return None
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        return limit
    return _MISSING_OUTPUT_TOKEN_LIMIT


def subcall_registrations(subcalls: SubcallClient) -> tuple[CapabilityRegistration, ...]:
    llm_batch_with_errors = getattr(subcalls, "llm_batch_with_errors", None)
    if not callable(llm_batch_with_errors):
        raise TypeError("SubcallClient must implement callable llm_batch_with_errors")

    def llm_query(context: CapabilityExecutionContext, *args: Any, **kwargs: Any) -> Any:
        context.check()
        return subcalls.llm_query(*args, **kwargs)

    def llm_batch(context: CapabilityExecutionContext, *args: Any, **kwargs: Any) -> Any:
        context.check()
        return subcalls.llm_batch(*args, **kwargs)

    def batch_with_errors(context: CapabilityExecutionContext, *args: Any, **kwargs: Any) -> Any:
        context.check()
        return llm_batch_with_errors(*args, **kwargs)

    return (
        CapabilityRegistration(LLM_QUERY_CAPABILITY, llm_query),
        CapabilityRegistration(LLM_BATCH_CAPABILITY, llm_batch),
        CapabilityRegistration(LLM_BATCH_WITH_ERRORS_CAPABILITY, batch_with_errors),
    )


class BrokeredSubcallClient:
    """Subcall protocol adapter whose methods are generated broker bindings."""

    def __init__(
        self,
        broker: CapabilityBroker,
        *,
        metadata_source: SubcallClient | None = None,
    ) -> None:
        self.llm_query = generate_binding(broker, LLM_QUERY_CAPABILITY, name="llm_query")
        self.llm_batch = generate_binding(broker, LLM_BATCH_CAPABILITY, name="llm_batch")
        self.llm_batch_with_errors = generate_binding(
            broker,
            LLM_BATCH_WITH_ERRORS_CAPABILITY,
            name="llm_batch_with_errors",
        )
        self._output_token_limit = (
            _reported_output_token_limit(metadata_source)
            if metadata_source is not None
            else _MISSING_OUTPUT_TOKEN_LIMIT
        )

    @property
    def output_token_limit(self) -> int | None:
        """Forward optional planning metadata without exposing the raw client."""
        if self._output_token_limit is None:
            return None
        if isinstance(self._output_token_limit, int):
            return self._output_token_limit
        raise AttributeError("wrapped subcall client did not report an output token limit")


def broker_subcalls(subcalls: SubcallClient, ledger: Any) -> BrokeredSubcallClient:
    """Create the mandatory standalone broker path for a custom environment."""

    from .execution.broker_budget import BrokerBudget
    from .execution.budget import BudgetLedger

    if not isinstance(ledger, BudgetLedger):
        raise TypeError("broker_subcalls requires the run BudgetLedger")
    accounting = BrokerBudget(ledger)
    return BrokeredSubcallClient(
        CapabilityBroker(
            subcall_registrations(subcalls),
            attempt_authority=accounting,
        ),
        metadata_source=subcalls,
    )


__all__ = [
    "CapabilityAdmission",
    "CapabilityAnnotator",
    "CapabilityAttemptAuthority",
    "CapabilityBroker",
    "CapabilityCall",
    "CapabilityCallError",
    "CapabilityCancelled",
    "CapabilityCheckpoint",
    "CapabilityDescriptor",
    "CapabilityError",
    "CapabilityErrorCode",
    "CapabilityDeadlineExceeded",
    "CapabilityExecutionContext",
    "CapabilityGuard",
    "CapabilityId",
    "CapabilityKind",
    "CapabilityManifest",
    "CapabilityMetadata",
    "CapabilityMetric",
    "CapabilityObserver",
    "CapabilityOutcome",
    "CapabilityRegistration",
    "CapabilityReservation",
    "CapabilityResult",
    "CapabilityResultHandle",
    "CapabilityStatus",
    "BrokeredSubcallClient",
    "EvidenceLocation",
    "EvidenceRange",
    "FrozenDict",
    "FrozenList",
    "FrozenTuple",
    "JSON_SCHEMA_2020_12",
    "PaginationMode",
    "ProviderOperation",
    "ResultDelivery",
    "SchemaSpec",
    "SideEffect",
    "LLM_BATCH_CAPABILITY",
    "LLM_BATCH_WITH_ERRORS_CAPABILITY",
    "LLM_QUERY_CAPABILITY",
    "broker_subcalls",
    "generate_binding",
    "freeze_value",
    "thaw_value",
    "subcall_registrations",
    "validate_call",
]
