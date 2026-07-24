"""Versioned run/trace values and pure retention projections.

Droste resolves which immutable values belong in a terminal ``RunRecord``.
It deliberately does not write them anywhere: a host may persist the returned
record (or receive it through an injected callback) using its own I/O shell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from threading import Lock
from time import monotonic
from types import MappingProxyType
from typing import Any, Callable, Mapping
from uuid import uuid4

TRACE_ABI_VERSION = 5


class PersistenceClass(str, Enum):
    """Whether an event is required, host-selectable, or stream-only."""

    DURABLE = "durable"
    CONFIGURABLE = "configurable"
    TRANSIENT = "transient"


DURABLE_EVENT_TYPES = frozenset({"usage", "budget", "policy", "capability", "done"})
CONFIGURABLE_EVENT_TYPES = frozenset(
    {
        "iteration_start",
        "llm_response",
        "code",
        "output",
        "execution_error",
        "subcall",
        "extract",
        "repair",
        "result",
        "replay",
    }
)
TRANSIENT_EVENT_TYPES = frozenset({"startup", "progress", "reasoning_delta", "usage_progress"})

PERSISTENCE_BY_TYPE: Mapping[str, PersistenceClass] = MappingProxyType(
    {
        **dict.fromkeys(DURABLE_EVENT_TYPES, PersistenceClass.DURABLE),
        **dict.fromkeys(CONFIGURABLE_EVENT_TYPES, PersistenceClass.CONFIGURABLE),
        **dict.fromkeys(TRANSIENT_EVENT_TYPES, PersistenceClass.TRANSIENT),
    }
)

_NONE_TYPE = type(None)
EventFieldType = type | tuple[type, ...]
EventBodySchema = tuple[Mapping[str, EventFieldType], Mapping[str, EventFieldType]]

# One exhaustive v5 table. The first mapping is required fields; the second is
# optional fields. Nested broker/result values keep their own schema authority.
EVENT_BODY_SCHEMAS: Mapping[str, EventBodySchema] = MappingProxyType(
    {
        "startup": (
            {"engine_version": str},
            {
                "runner_protocol": (int, _NONE_TYPE),
                "provider_protocol": (int, _NONE_TYPE),
                "scaffold_manifest_id": str,
                "scaffold_manifest_version": int,
            },
        ),
        "progress": ({"status": str}, {}),
        "iteration_start": ({"iteration": int, "remaining_tokens": int}, {}),
        "llm_response": ({"iteration": int, "response": str}, {}),
        "code": ({"iteration": int, "code": str}, {}),
        "output": (
            {
                "iteration": int,
                "stdout": str,
                "calls_made": int,
                "answer_ready": bool,
                "answer_content_chars": int,
            },
            {"stdout_chars": int},
        ),
        "execution_error": (
            {"iteration": int, "error_type": str, "message": str},
            {},
        ),
        "reasoning_delta": ({"text": str}, {}),
        "subcall": (
            {
                "phase": str,
                "call_id": str,
                "operation": str,
                "iteration": int,
            },
            {
                "reservation": Mapping,
                "checkpoint": Mapping,
                "batch_count": int,
                "error": Mapping,
            },
        ),
        "repair": (
            {"phase": str, "kind": str, "iteration": int},
            {"error": Mapping},
        ),
        "extract": (
            {"phase": str, "iteration": int},
            {"extract_error": Mapping},
        ),
        "result": ({"result": Mapping}, {}),
        "replay": ({"result": Mapping}, {}),
        "usage_progress": (
            {
                "boundary": str,
                "kind": str,
                "root": Mapping,
                "subcall": Mapping,
                "unattributed": Mapping,
                "total_tokens": int,
            },
            {},
        ),
        "usage": (
            {
                "kind": str,
                "root": Mapping,
                "subcall": Mapping,
                "unattributed": Mapping,
                "total_tokens": int,
                "wall_time_ms": int,
            },
            {},
        ),
        "budget": (
            {"kind": str, "source": str},
            {
                "configured": Mapping,
                "consumed": Mapping,
                "remaining": Mapping,
                "action": str,
                "resource": str,
                "amount": (int, float),
                "call_id": str,
            },
        ),
        "policy": (
            {"contract_enforced": bool, "outcome": str, "violation_type": (str, _NONE_TYPE)},
            {},
        ),
        "capability": ({"outcome": Mapping}, {}),
        "done": (
            {
                "status": str,
                "ready": bool,
                "extracted": bool,
                "iterations": int,
                "usage": Mapping,
                "budget": Mapping,
                "policy": Mapping,
                "retention": Mapping,
                "error": (Mapping, _NONE_TYPE),
                "extract_error": (Mapping, _NONE_TYPE),
                "recovered_error": (Mapping, _NONE_TYPE),
            },
            {
                "scaffold_manifest_id": (str, _NONE_TYPE),
                "scaffold_manifest_version": (int, _NONE_TYPE),
                "stdout_chars": int,
            },
        ),
    }
)


def _matches_field_type(value: Any, expected: EventFieldType) -> bool:
    expected_types = expected if isinstance(expected, tuple) else (expected,)
    if int in expected_types and bool not in expected_types and isinstance(value, bool):
        return False
    return isinstance(value, expected_types)


def validate_event_body(event_type: str, body: Mapping[str, Any]) -> None:
    """Validate one body against the exhaustive Trace ABI v5 table."""
    try:
        required, optional = EVENT_BODY_SCHEMAS[event_type]
    except KeyError as exc:
        raise ValueError(f"event type {event_type!r} has no v5 body schema") from exc
    missing = required.keys() - body.keys()
    if missing:
        raise ValueError(f"event {event_type!r} missing body fields: " + ", ".join(sorted(missing)))
    unknown = body.keys() - required.keys() - optional.keys()
    if unknown:
        raise ValueError(
            f"event {event_type!r} has unknown body fields: " + ", ".join(sorted(unknown))
        )
    for key, value in body.items():
        expected = required.get(key, optional.get(key))
        assert expected is not None
        if not _matches_field_type(value, expected):
            raise TypeError(
                f"event {event_type!r} field {key!r} has invalid type {type(value).__name__}"
            )
    _validate_structured_body(event_type, body)


def _require_exact_mapping(
    value: Any, *, name: str, fields: Mapping[str, EventFieldType]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    missing = fields.keys() - value.keys()
    unknown = value.keys() - fields.keys()
    if missing or unknown:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            detail.append("unknown " + ", ".join(sorted(unknown)))
        raise ValueError(f"{name} has " + "; ".join(detail))
    for key, expected in fields.items():
        if not _matches_field_type(value[key], expected):
            raise TypeError(f"{name}.{key} has invalid type {type(value[key]).__name__}")
    return value


def _validate_structured_body(event_type: str, body: Mapping[str, Any]) -> None:
    if event_type == "subcall":
        phase = body["phase"]
        if phase not in {"start", "progress", "completion", "failure"}:
            raise ValueError("subcall phase is not recognized by Trace ABI v5")
        if body["operation"] not in {"llm_query", "llm_batch", "llm_batch_with_errors"}:
            raise ValueError("subcall operation is not recognized by Trace ABI v5")
        if not body["call_id"]:
            raise ValueError("subcall call_id must not be empty")
        if body["iteration"] < 1:
            raise ValueError("subcall iteration must be positive")
        reservation = body.get("reservation")
        checkpoint = body.get("checkpoint")
        error = body.get("error")
        if phase == "start":
            if reservation is None:
                raise ValueError("subcall start requires reservation")
            if checkpoint is not None or error is not None:
                raise ValueError("subcall start cannot carry checkpoint or error")
        elif phase == "progress":
            if checkpoint is None:
                raise ValueError("subcall progress requires checkpoint")
            if reservation is not None or error is not None:
                raise ValueError("subcall progress cannot carry reservation or error")
        elif phase == "completion":
            if checkpoint is None:
                raise ValueError("subcall completion requires checkpoint")
            if reservation is not None or error is not None:
                raise ValueError("subcall completion cannot carry reservation or error")
        else:
            if checkpoint is None or error is None:
                raise ValueError("subcall failure requires checkpoint and error")
            if reservation is not None:
                raise ValueError("subcall failure cannot carry reservation")
        if reservation is not None:
            values = _require_exact_mapping(
                reservation,
                name="subcall.reservation",
                fields={"tokens": int, "subcalls": int, "wall_ms": int, "depth": int},
            )
            if any(isinstance(value, bool) or value < 0 for value in values.values()):
                raise ValueError("subcall reservation values must be non-negative integers")
        if checkpoint is not None:
            values = _require_exact_mapping(
                checkpoint,
                name="subcall.checkpoint",
                fields={"tokens": int, "subcalls": int},
            )
            if any(isinstance(value, bool) or value < 0 for value in values.values()):
                raise ValueError("subcall checkpoint values must be non-negative integers")
        if error is not None:
            _require_exact_mapping(
                error,
                name="subcall.error",
                fields={"code": str, "type": str},
            )
        is_batch = body["operation"] in {"llm_batch", "llm_batch_with_errors"}
        if is_batch and "batch_count" not in body:
            raise ValueError("batch subcall requires batch_count")
        if not is_batch and "batch_count" in body:
            raise ValueError("unary subcall cannot carry batch_count")
        if "batch_count" in body and body["batch_count"] < 0:
            raise ValueError("subcall batch_count must be non-negative")
    elif event_type == "repair":
        if body["phase"] not in {"start", "completion", "failure"}:
            raise ValueError("repair phase is not recognized by Trace ABI v5")
        if body["kind"] not in {"missing_code", "execution_error", "terminal"}:
            raise ValueError("repair kind is not recognized by Trace ABI v5")
        if body["iteration"] < 1:
            raise ValueError("repair iteration must be positive")
        error = body.get("error")
        if body["phase"] == "failure":
            if error is None:
                raise ValueError("repair failure requires error")
            _require_exact_mapping(
                error,
                name="repair.error",
                fields={"type": str, "message": str},
            )
        elif error is not None:
            raise ValueError("repair start/completion cannot carry error")
    elif event_type == "extract":
        if body["phase"] not in {"start", "completion", "failure"}:
            raise ValueError("extract phase is not recognized by Trace ABI v5")
        if body["iteration"] < 1:
            raise ValueError("extract iteration must be positive")
        extract_error = body.get("extract_error")
        if body["phase"] == "failure":
            if extract_error is None:
                raise ValueError("extract failure requires extract_error")
            _require_exact_mapping(
                extract_error,
                name="extract.extract_error",
                fields={"type": str, "message": str},
            )
        elif extract_error is not None:
            raise ValueError("extract start/completion cannot carry extract_error")
    elif event_type in {"usage", "usage_progress"}:
        if event_type == "usage_progress" and body["boundary"] not in {"root", "subcall"}:
            raise ValueError("usage_progress boundary must be 'root' or 'subcall'")
        if body["kind"] not in {"resolved", "partial"}:
            raise ValueError(f"{event_type} kind must be 'resolved' or 'partial'")
        breakdown_fields: Mapping[str, EventFieldType] = {
            "input_tokens": int,
            "cache_read_tokens": int,
            "cache_creation_tokens": int,
            "output_tokens": int,
            "total_tokens": int,
            "requests": int,
            "successes": int,
            "complete": bool,
        }
        root = _require_exact_mapping(
            body["root"], name=f"{event_type}.root", fields=breakdown_fields
        )
        subcall = _require_exact_mapping(
            body["subcall"], name=f"{event_type}.subcall", fields=breakdown_fields
        )
        unattributed = _require_exact_mapping(
            body["unattributed"],
            name=f"{event_type}.unattributed",
            fields={"total_tokens": int},
        )
        counts = [
            *(root[field] for field in breakdown_fields if field != "complete"),
            *(subcall[field] for field in breakdown_fields if field != "complete"),
            unattributed["total_tokens"],
            body["total_tokens"],
        ]
        if event_type == "usage":
            counts.append(body["wall_time_ms"])
        if any(isinstance(value, bool) or value < 0 for value in counts):
            raise ValueError(f"{event_type} counts must be non-negative integers")
        if root["successes"] > root["requests"] or subcall["successes"] > subcall["requests"]:
            raise ValueError(f"{event_type} successes cannot exceed requests")
        if root["complete"] and (
            root["cache_read_tokens"] + root["cache_creation_tokens"] > root["input_tokens"]
        ):
            raise ValueError("complete root cache token breakdown cannot exceed input tokens")
        if subcall["complete"] and (
            subcall["cache_read_tokens"] + subcall["cache_creation_tokens"]
            > subcall["input_tokens"]
        ):
            raise ValueError("complete subcall cache token breakdown cannot exceed input tokens")
        expected_kind = "resolved" if root["complete"] and subcall["complete"] else "partial"
        if body["kind"] != expected_kind:
            raise ValueError(f"{event_type} kind must agree with scope completeness")
        if (
            root["total_tokens"] + subcall["total_tokens"] + unattributed["total_tokens"]
            != body["total_tokens"]
        ):
            raise ValueError(f"{event_type} token scopes must reconcile to total_tokens")
    elif event_type == "budget":
        if body["kind"] == "snapshot":
            required = {"configured", "consumed", "remaining"}
            if not required <= body.keys():
                raise ValueError("budget snapshot requires configured, consumed, and remaining")
            forbidden = {"action", "resource", "amount", "call_id"} & body.keys()
            if forbidden:
                raise ValueError("budget snapshot cannot carry mutation fields")
        elif body["kind"] == "mutation":
            required = {"action", "resource", "amount"}
            if not required <= body.keys():
                raise ValueError("budget mutation requires action, resource, and amount")
            if body["action"] not in {"reserve", "commit", "refund", "exhaust"}:
                raise ValueError("budget mutation action is not recognized by Trace ABI v5")
            if body["amount"] < 0:
                raise ValueError("budget mutation amount must be non-negative")
        else:
            raise ValueError("budget kind must be 'snapshot' or 'mutation'")
        if not body["source"]:
            raise ValueError("budget source must not be empty")
    elif event_type == "result":
        raw_result = body["result"]
        if not isinstance(raw_result, Mapping):
            raise TypeError("result.result must be an object")
        scaffold_manifest = raw_result.get("scaffold_manifest")
        stdout_chars = raw_result.get("stdout_chars", 0)
        result = _require_exact_mapping(
            {
                key: value
                for key, value in raw_result.items()
                if key not in {"scaffold_manifest", "stdout_chars"}
            },
            name="result.result",
            fields={
                "answer": str,
                "answer_metadata": Mapping,
                "ready": bool,
                "iterations": int,
                "tokens_used": int,
                "subcalls": int,
                "successful_subcalls": int,
                "extracted": bool,
                "error": (Mapping, _NONE_TYPE),
                "extract_error": (Mapping, _NONE_TYPE),
                "recovered_error": (Mapping, _NONE_TYPE),
                "prompt_pack": (Mapping, _NONE_TYPE),
            },
        )
        result_counts = (
            result["iterations"],
            result["tokens_used"],
            result["subcalls"],
            result["successful_subcalls"],
        )
        if any(value < 0 for value in result_counts):
            raise ValueError("result counts must be non-negative")
        if result["successful_subcalls"] > result["subcalls"]:
            raise ValueError("result successful_subcalls cannot exceed subcalls")
        if isinstance(stdout_chars, bool) or not isinstance(stdout_chars, int) or stdout_chars < 0:
            raise ValueError("result stdout_chars must be a non-negative integer")
        if scaffold_manifest is not None:
            if not isinstance(scaffold_manifest, Mapping):
                raise TypeError("result.result.scaffold_manifest must be an object or null")
            # Reuse the manifest's one closed-schema/canonical-ID parser rather
            # than maintaining a weaker second validator in the trace layer.
            from .manifest import ScaffoldManifest

            ScaffoldManifest.from_dict(scaffold_manifest)
    elif event_type == "policy":
        if body["outcome"] not in {"passed", "violated", "not_evaluated", "not_enforced"}:
            raise ValueError("policy outcome is not recognized by Trace ABI v5")
    elif event_type == "done":
        if body["status"] not in {"success", "error", "cancelled"}:
            raise ValueError("done status is not recognized by Trace ABI v5")
        if body["iterations"] < 0:
            raise ValueError("done iterations must be non-negative")
        stdout_chars = body.get("stdout_chars")
        if stdout_chars is not None and (
            isinstance(stdout_chars, bool) or not isinstance(stdout_chars, int) or stdout_chars < 0
        ):
            raise ValueError("done stdout_chars must be a non-negative integer")
        validate_event_body("usage", body["usage"])
        validate_event_body("budget", body["budget"])
        validate_event_body("policy", body["policy"])
        retention = _require_exact_mapping(
            body["retention"],
            name="done.retention",
            fields={
                "policy_id": str,
                "retain": list,
                "expires_at": (str, _NONE_TYPE),
                "host_managed_expiry": bool,
                "replay_retained": bool,
            },
        )
        if any(not isinstance(value, str) for value in retention["retain"]):
            raise TypeError("done.retention.retain values must be strings")
        if retention["replay_retained"] != ("replay" in retention["retain"]):
            raise ValueError("done retention replay facts disagree")
        for error_name in ("error", "extract_error", "recovered_error"):
            if body[error_name] is not None:
                _require_exact_mapping(
                    body[error_name],
                    name=f"done.{error_name}",
                    fields={"type": str},
                )


def persistence_class_for(event_type: str) -> PersistenceClass:
    """Classify every event explicitly; unknown vocabulary fails closed."""
    try:
        return PERSISTENCE_BY_TYPE[event_type]
    except KeyError as exc:
        raise ValueError(f"event type {event_type!r} has no persistence classification") from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("trace object keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, float) and not isfinite(value):
        raise TypeError("trace numbers must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"trace values must be JSON-compatible, got {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RunEvent:
    """One immutable event value in the append-only Trace ABI."""

    run_id: str
    seq: int
    timestamp: str
    type: str
    version: int = TRACE_ABI_VERSION
    persistence_class: PersistenceClass = PersistenceClass.CONFIGURABLE
    body: Mapping[str, Any] = field(default_factory=dict)
    parent_run_id: str | None = None
    depth: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("event run_id must not be empty")
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 1:
            raise ValueError("event seq must be positive")
        if not isinstance(self.timestamp, str):
            raise TypeError("event timestamp must be a string")
        try:
            timestamp = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("event timestamp must be an ISO-8601 timestamp") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise ValueError("event timestamp must be UTC")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("event version must be an integer")
        if self.version != TRACE_ABI_VERSION:
            raise ValueError(f"unsupported trace event version: {self.version}")
        if not isinstance(self.type, str) or not self.type:
            raise ValueError("event type must not be empty")
        if self.parent_run_id is not None and (
            not isinstance(self.parent_run_id, str) or not self.parent_run_id
        ):
            raise ValueError("event parent_run_id must be a non-empty string")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError("event depth must be non-negative")
        if self.depth == 0 and self.parent_run_id is not None:
            raise ValueError("root events cannot have parent_run_id")
        if self.depth > 0 and not self.parent_run_id:
            raise ValueError("child events require parent_run_id")
        expected = persistence_class_for(self.type)
        if self.persistence_class is not expected:
            raise ValueError(
                f"event {self.type!r} must be {expected.value}, not {self.persistence_class.value}"
            )
        frozen_body = _freeze_json(self.body)
        validate_event_body(self.type, _thaw_json(frozen_body))
        object.__setattr__(self, "body", frozen_body)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical flat Trace ABI wire value."""
        wire = _thaw_json(self.body)
        wire.update(
            {
                "run_id": self.run_id,
                "seq": self.seq,
                "timestamp": self.timestamp,
                "type": self.type,
                "version": self.version,
                "persistence_class": self.persistence_class.value,
            }
        )
        if self.parent_run_id is not None:
            wire["parent_run_id"] = self.parent_run_id
        wire["depth"] = self.depth
        return wire


_ENVELOPE_KEYS = frozenset(
    {
        "run_id",
        "seq",
        "timestamp",
        "type",
        "version",
        "persistence_class",
        "parent_run_id",
        "depth",
    }
)


def parse_event(value: RunEvent | Mapping[str, Any]) -> RunEvent:
    """The one strict parser for Trace ABI v5 event values."""
    if isinstance(value, RunEvent):
        return value
    required = {
        "run_id",
        "seq",
        "timestamp",
        "type",
        "version",
        "persistence_class",
        "depth",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError("trace event missing envelope fields: " + ", ".join(sorted(missing)))
    if not isinstance(value["run_id"], str):
        raise TypeError("trace event run_id must be a string")
    if isinstance(value["seq"], bool) or not isinstance(value["seq"], int):
        raise TypeError("trace event seq must be an integer")
    if not isinstance(value["timestamp"], str):
        raise TypeError("trace event timestamp must be a string")
    if not isinstance(value["type"], str):
        raise TypeError("trace event type must be a string")
    if isinstance(value["version"], bool) or not isinstance(value["version"], int):
        raise TypeError("trace event version must be an integer")
    if not isinstance(value["persistence_class"], str):
        raise TypeError("trace event persistence_class must be a string")
    if value.get("parent_run_id") is not None and not isinstance(value["parent_run_id"], str):
        raise TypeError("trace event parent_run_id must be a string")
    if isinstance(value["depth"], bool) or not isinstance(value["depth"], int):
        raise TypeError("trace event depth must be an integer")
    event_type = value["type"]
    persistence = PersistenceClass(value["persistence_class"])
    body = {key: item for key, item in value.items() if key not in _ENVELOPE_KEYS}
    return RunEvent(
        run_id=value["run_id"],
        seq=value["seq"],
        timestamp=value["timestamp"],
        type=event_type,
        version=value["version"],
        persistence_class=persistence,
        body=body,
        parent_run_id=(value["parent_run_id"] if value.get("parent_run_id") is not None else None),
        depth=value["depth"],
    )


@dataclass(frozen=True, slots=True)
class TraceRetentionPolicy:
    """Host selection for configurable content; durable facts are unconditional."""

    retain: frozenset[str] = frozenset()
    policy_id: str = "default-no-content"
    expires_at: str | None = None
    host_managed_expiry: bool = False

    def __post_init__(self) -> None:
        retain = frozenset(self.retain)
        if any(not isinstance(event_type, str) for event_type in retain):
            raise TypeError("retained event types must be strings")
        invalid = retain - CONFIGURABLE_EVENT_TYPES
        if invalid:
            raise ValueError(
                "only configurable event types may be selected for retention: "
                + ", ".join(sorted(invalid))
            )
        object.__setattr__(self, "retain", retain)
        if not isinstance(self.policy_id, str):
            raise TypeError("retention policy_id must be a string")
        if not self.policy_id:
            raise ValueError("retention policy_id must not be empty")
        if not isinstance(self.host_managed_expiry, bool):
            raise TypeError("host_managed_expiry must be a bool")
        if self.expires_at is not None:
            if not isinstance(self.expires_at, str):
                raise TypeError("retention expires_at must be a string")
            try:
                expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("retention expires_at must be an ISO-8601 timestamp") from exc
            if expiry.tzinfo is None:
                raise ValueError("retention expires_at must include a timezone")
            if not self.host_managed_expiry:
                raise ValueError("expires_at requires host_managed_expiry=True")

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "retain": sorted(self.retain),
            "expires_at": self.expires_at,
            "host_managed_expiry": self.host_managed_expiry,
        }


@dataclass(frozen=True, slots=True)
class DataUseAuthorization:
    """Independent host authorization; retention never grants training use."""

    training_allowed: bool = False
    authorization_ref: str | None = None
    purposes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.training_allowed, bool):
            raise TypeError("training_allowed must be a bool")
        purposes = frozenset(self.purposes)
        if any(not isinstance(purpose, str) or not purpose for purpose in purposes):
            raise TypeError("data-use purposes must be non-empty strings")
        object.__setattr__(self, "purposes", purposes)
        if self.authorization_ref is not None:
            if not isinstance(self.authorization_ref, str):
                raise TypeError("authorization_ref must be a string")
            if not self.authorization_ref:
                raise ValueError("authorization_ref must not be empty")
        if purposes and self.authorization_ref is None:
            raise ValueError("data-use purposes require authorization_ref")
        if "training" in purposes and not self.training_allowed:
            raise ValueError("purpose 'training' requires training_allowed=True")
        if self.training_allowed and (self.authorization_ref is None or "training" not in purposes):
            raise ValueError(
                "training_allowed=True requires authorization_ref and purpose 'training'"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "training_allowed": self.training_allowed,
            "authorization_ref": self.authorization_ref,
            "purposes": sorted(self.purposes),
        }


def select_retained_events(
    events: tuple[RunEvent, ...] | list[RunEvent], policy: TraceRetentionPolicy
) -> tuple[RunEvent, ...]:
    """Purely select durable plus explicitly allowed configurable values."""
    return tuple(
        event
        for event in events
        if event.persistence_class is PersistenceClass.DURABLE
        or (
            event.persistence_class is PersistenceClass.CONFIGURABLE and event.type in policy.retain
        )
    )


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Policy-resolved terminal record ready for a host persistence shell."""

    run_id: str
    events: tuple[RunEvent, ...]
    terminal: Mapping[str, Any]
    retention: TraceRetentionPolicy
    data_use: DataUseAuthorization
    parent_run_id: str | None = None
    depth: int = 0
    version: int = TRACE_ABI_VERSION

    def __post_init__(self) -> None:
        events = tuple(self.events)
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run record run_id must not be empty")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("run record version must be an integer")
        if self.version != TRACE_ABI_VERSION:
            raise ValueError(f"unsupported run record version: {self.version}")
        if self.parent_run_id is not None and (
            not isinstance(self.parent_run_id, str) or not self.parent_run_id
        ):
            raise ValueError("run record parent_run_id must be a non-empty string")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError("run record depth must be non-negative")
        if self.depth == 0 and self.parent_run_id is not None:
            raise ValueError("root run records cannot have parent_run_id")
        if self.depth > 0 and not self.parent_run_id:
            raise ValueError("child run records require parent_run_id")
        if not isinstance(self.retention, TraceRetentionPolicy):
            raise TypeError("run record retention must be TraceRetentionPolicy")
        if not isinstance(self.data_use, DataUseAuthorization):
            raise TypeError("run record data_use must be DataUseAuthorization")
        if any(event.run_id != self.run_id for event in events):
            raise ValueError("run record events must share its run_id")
        if any(event.parent_run_id != self.parent_run_id for event in events):
            raise ValueError("run record events must share its parent_run_id")
        if any(event.depth != self.depth for event in events):
            raise ValueError("run record events must share its depth")
        if any(left.seq >= right.seq for left, right in zip(events, events[1:])):
            raise ValueError("run record events must be in increasing seq order")
        if any(event.persistence_class is PersistenceClass.TRANSIENT for event in events):
            raise ValueError("run records cannot retain transient events")
        if any(
            event.persistence_class is PersistenceClass.CONFIGURABLE
            and event.type not in self.retention.retain
            for event in events
        ):
            raise ValueError("run record contains configurable events excluded by retention")
        if not events or events[-1].type != "done":
            raise ValueError("run record must end with a durable done event")
        if _thaw_json(events[-1].body) != _thaw_json(self.terminal):
            raise ValueError("run record terminal must match its done event")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "terminal", _freeze_json(self.terminal))

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "version": self.version,
            "run_id": self.run_id,
            "retention": self.retention.as_dict(),
            "data_use": self.data_use.as_dict(),
            "terminal": _thaw_json(self.terminal),
            "events": [parse_event(event).as_dict() for event in self.events],
        }
        if self.parent_run_id is not None:
            value["parent_run_id"] = self.parent_run_id
        value["depth"] = self.depth
        return value


Clock = Callable[[], datetime]
MonotonicClock = Callable[[], float]
RunRecordCallback = Callable[[RunRecord], None]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class TraceRecorder:
    """Small identity/ordering shell around the pure trace values."""

    run_id: str = field(default_factory=lambda: str(uuid4()))
    parent_run_id: str | None = None
    depth: int = 0
    retention: TraceRetentionPolicy = field(default_factory=TraceRetentionPolicy)
    data_use: DataUseAuthorization = field(default_factory=DataUseAuthorization)
    clock: Clock = utc_now
    monotonic_clock: MonotonicClock = monotonic
    _events: list[RunEvent] = field(default_factory=list, init=False, repr=False)
    _terminal_record: RunRecord | None = field(default=None, init=False, repr=False)
    _started_monotonic: float | None = field(default=None, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("trace run_id must not be empty")
        if self.parent_run_id is not None and (
            not isinstance(self.parent_run_id, str) or not self.parent_run_id
        ):
            raise ValueError("trace parent_run_id must be a non-empty string")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError("trace depth must be non-negative")
        if self.depth == 0 and self.parent_run_id is not None:
            raise ValueError("root traces cannot have parent_run_id")
        if self.depth > 0 and not self.parent_run_id:
            raise ValueError("child traces require parent_run_id")

    @property
    def events(self) -> tuple[RunEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def elapsed_ms(self) -> int:
        """Resolved wall duration from first event to this observation."""
        with self._lock:
            if self._started_monotonic is None:
                return 0
            return max(0, round((self.monotonic_clock() - self._started_monotonic) * 1000))

    def append(self, event: Mapping[str, Any]) -> RunEvent:
        with self._lock:
            return self._append_locked(event)

    def _append_locked(self, event: Mapping[str, Any]) -> RunEvent:
        if self._terminal_record is not None:
            raise RuntimeError("cannot append an event after the terminal record")
        event_type = event.get("type")
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("trace event has no type")
        pre_stamped = (event.keys() & _ENVELOPE_KEYS) - {"type"}
        if pre_stamped:
            raise ValueError(
                "TraceRecorder is the envelope authority; body supplied: "
                + ", ".join(sorted(pre_stamped))
            )
        if self._started_monotonic is None:
            self._started_monotonic = self.monotonic_clock()
        body = {key: value for key, value in event.items() if key != "type"}
        now = self.clock()
        if now.tzinfo is None:
            raise ValueError("trace clock must return a timezone-aware datetime")
        timestamp = now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        value = RunEvent(
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            depth=self.depth,
            seq=len(self._events) + 1,
            timestamp=timestamp,
            type=event_type,
            persistence_class=persistence_class_for(event_type),
            body=body,
        )
        self._events.append(value)
        return value

    def finish(self, terminal: Mapping[str, Any]) -> RunRecord:
        with self._lock:
            if self._terminal_record is not None:
                return self._terminal_record
            self._append_locked({**terminal, "type": "done"})
            record = RunRecord(
                run_id=self.run_id,
                parent_run_id=self.parent_run_id,
                depth=self.depth,
                events=select_retained_events(self._events, self.retention),
                terminal=terminal,
                retention=self.retention,
                data_use=self.data_use,
            )
            self._terminal_record = record
            return record
