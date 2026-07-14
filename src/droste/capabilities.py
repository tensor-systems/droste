"""Typed capability values and the single sandbox egress mechanism.

The broker is deliberately transport-blind.  Registrations are trusted host
values; generated bindings are the compatibility surface exposed to model
written code.  Budget policy and tracing can observe the same immutable call
and result values without becoming alternate dispatch paths.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol
from uuid import uuid4

from .protocols.subcall_client import SubcallClient

_ERROR_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_.-]*\Z", re.ASCII)
_ERROR_TYPE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z", re.ASCII)
_MEDIA_TYPE_PATTERN = re.compile(
    r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*\Z", re.ASCII
)


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


class CapabilityStatus(StrEnum):
    OK = "ok"
    INVALID = "invalid"
    DENIED = "denied"
    ERROR = "error"


class CapabilityErrorCode(StrEnum):
    INVALID_CALL = "invalid_call"
    NOT_ALLOWED = "not_allowed"
    POLICY_DENIED = "policy_denied"
    GUARD_ERROR = "guard_error"
    HANDLER_ERROR = "handler_error"
    INVALID_RESULT = "invalid_result"
    ANNOTATOR_ERROR = "annotator_error"


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
    side_effect: SideEffect = SideEffect.UNSPECIFIED

    def __post_init__(self) -> None:
        if not isinstance(self.capability_id, CapabilityId):
            raise TypeError("capability descriptor requires a CapabilityId")
        if not isinstance(self.side_effect, SideEffect):
            raise TypeError("capability side_effect must be a SideEffect")

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id.to_dict(),
            "side_effect": self.side_effect.value,
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


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    kind: str
    ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("evidence kind must not be empty")
        if not isinstance(self.ref, str) or not self.ref:
            raise ValueError("evidence ref must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "ref": self.ref}


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
class CapabilityMetadata:
    """Optional facts attached by accounting/evidence integrations."""

    usage: tuple[CapabilityMetric, ...] = ()
    budget_delta: tuple[CapabilityMetric, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    result_handle: CapabilityResultHandle | None = None
    child_run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", tuple(self.usage))
        object.__setattr__(self, "budget_delta", tuple(self.budget_delta))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        _validate_metadata_values(
            self.usage,
            self.budget_delta,
            self.evidence_refs,
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
            evidence_refs=self.evidence_refs + finalizer.evidence_refs,
            result_handle=singular("result_handle", self.result_handle, finalizer.result_handle),
            child_run_id=singular("child_run_id", self.child_run_id, finalizer.child_run_id),
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


def _validate_metadata_values(
    usage: tuple[CapabilityMetric, ...],
    budget_delta: tuple[CapabilityMetric, ...],
    evidence_refs: tuple[EvidenceRef, ...],
    result_handle: CapabilityResultHandle | None,
    child_run_id: str | None,
) -> None:
    if not all(isinstance(item, CapabilityMetric) for item in (*usage, *budget_delta)):
        raise TypeError("capability usage and budget_delta require CapabilityMetric values")
    if not all(isinstance(item, EvidenceRef) for item in evidence_refs):
        raise TypeError("capability evidence_refs require EvidenceRef values")
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
    evidence_refs: tuple[EvidenceRef, ...] = ()
    result_handle: CapabilityResultHandle | None = None
    child_run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "result", freeze_value(self.result))
        object.__setattr__(self, "usage", tuple(self.usage))
        object.__setattr__(self, "budget_delta", tuple(self.budget_delta))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        _validate_metadata_values(
            self.usage,
            self.budget_delta,
            self.evidence_refs,
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
                "evidence_refs": [item.to_dict() for item in self.evidence_refs],
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
            "evidence": {"count": len(self.evidence_refs)},
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

    @property
    def run_id(self) -> str:
        return self._run_id

    def describe(self) -> CapabilityManifest:
        return self._manifest

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

        if self._guard is not None:
            try:
                denial = self._guard(call)
                if denial is not None and not isinstance(denial, CapabilityError):
                    raise TypeError("capability guard must return CapabilityError or None")
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
            if denial is not None:
                return self._finish(
                    call, status=CapabilityStatus.DENIED, error=denial, annotate=False
                )

        try:
            outcome = self._handlers[validated.capability_id](
                *(thaw_value(item) for item in call.args),
                **{key: thaw_value(item) for key, item in call.kwargs.items()},
            )
        except BaseException as exc:
            error = CapabilityError(
                CapabilityErrorCode.HANDLER_ERROR,
                _exception_type_name(exc),
                str(exc),
            )
            envelope = self._finish(call, status=CapabilityStatus.ERROR, error=error)
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
            )
        if outcome.error is not None:
            return self._finish(
                call,
                status=CapabilityStatus.ERROR,
                error=outcome.error,
                provider_metadata=outcome.metadata,
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
            )
            if not isinstance(exc, Exception):
                raise
            return envelope
        return self._finish(
            call,
            status=CapabilityStatus.OK,
            result=frozen_result,
            provider_metadata=outcome.metadata,
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
    ) -> CapabilityResult:
        metadata = provider_metadata or CapabilityMetadata()
        annotation_error: CapabilityError | None = None
        if annotate and self._annotator is not None:
            try:
                finalized = self._annotator(call, result, error)
                if not isinstance(finalized, CapabilityMetadata):
                    raise TypeError("capability annotator must return CapabilityMetadata")
                metadata = metadata.merged_with(finalized)
            except Exception as exc:
                annotation_error = CapabilityError(
                    CapabilityErrorCode.ANNOTATOR_ERROR,
                    _exception_type_name(exc),
                    str(exc),
                )
        envelope = CapabilityResult(
            call=call,
            ok=status is CapabilityStatus.OK and annotation_error is None,
            status=(CapabilityStatus.ERROR if annotation_error is not None else status),
            # An annotator-provided handle deliberately replaces the inline value.
            result=result if metadata.result_handle is None else None,
            error=annotation_error or error,
            usage=metadata.usage,
            budget_delta=metadata.budget_delta,
            evidence_refs=metadata.evidence_refs,
            result_handle=metadata.result_handle,
            child_run_id=metadata.child_run_id,
        )
        self.emit(envelope)
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

    binding.__name__ = name or descriptor.capability_id.operation
    binding.__qualname__ = binding.__name__
    binding.__doc__ = f"Brokered compatibility binding for {descriptor.capability_id.operation}."
    return binding


LLM_QUERY_CAPABILITY = CapabilityDescriptor(
    CapabilityId(
        kind=CapabilityKind.INFERENCE,
        provider_type="subcall",
        operation="llm_query",
    ),
    side_effect=SideEffect.READ,
)
LLM_BATCH_CAPABILITY = CapabilityDescriptor(
    CapabilityId(
        kind=CapabilityKind.INFERENCE,
        provider_type="subcall",
        operation="llm_batch",
    ),
    side_effect=SideEffect.READ,
)
LLM_BATCH_WITH_ERRORS_CAPABILITY = CapabilityDescriptor(
    CapabilityId(
        kind=CapabilityKind.INFERENCE,
        provider_type="subcall",
        operation="llm_batch_with_errors",
    ),
    side_effect=SideEffect.READ,
)


def subcall_registrations(subcalls: SubcallClient) -> tuple[CapabilityRegistration, ...]:
    llm_batch_with_errors = getattr(subcalls, "llm_batch_with_errors", None)
    if not callable(llm_batch_with_errors):
        raise TypeError("SubcallClient must implement callable llm_batch_with_errors")
    return (
        CapabilityRegistration(LLM_QUERY_CAPABILITY, subcalls.llm_query),
        CapabilityRegistration(LLM_BATCH_CAPABILITY, subcalls.llm_batch),
        CapabilityRegistration(LLM_BATCH_WITH_ERRORS_CAPABILITY, llm_batch_with_errors),
    )


class BrokeredSubcallClient:
    """Subcall protocol adapter whose methods are generated broker bindings."""

    def __init__(self, broker: CapabilityBroker) -> None:
        self.llm_query = generate_binding(broker, LLM_QUERY_CAPABILITY, name="llm_query")
        self.llm_batch = generate_binding(broker, LLM_BATCH_CAPABILITY, name="llm_batch")
        self.llm_batch_with_errors = generate_binding(
            broker,
            LLM_BATCH_WITH_ERRORS_CAPABILITY,
            name="llm_batch_with_errors",
        )


def broker_subcalls(subcalls: SubcallClient) -> BrokeredSubcallClient:
    """Create the mandatory standalone broker path for a custom environment."""

    return BrokeredSubcallClient(CapabilityBroker(subcall_registrations(subcalls)))


__all__ = [
    "CapabilityAnnotator",
    "CapabilityBroker",
    "CapabilityCall",
    "CapabilityCallError",
    "CapabilityDescriptor",
    "CapabilityError",
    "CapabilityErrorCode",
    "CapabilityGuard",
    "CapabilityId",
    "CapabilityKind",
    "CapabilityManifest",
    "CapabilityMetadata",
    "CapabilityMetric",
    "CapabilityObserver",
    "CapabilityOutcome",
    "CapabilityRegistration",
    "CapabilityResult",
    "CapabilityResultHandle",
    "CapabilityStatus",
    "BrokeredSubcallClient",
    "EvidenceRef",
    "FrozenDict",
    "FrozenList",
    "FrozenTuple",
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
