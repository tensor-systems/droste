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

TRACE_ABI_VERSION = 1


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
        "finalization_error",
        "extract_error",
        "repair",
        "replay",
    }
)
TRANSIENT_EVENT_TYPES = frozenset({"startup", "progress", "reasoning_delta"})

PERSISTENCE_BY_TYPE: Mapping[str, PersistenceClass] = MappingProxyType(
    {
        **dict.fromkeys(DURABLE_EVENT_TYPES, PersistenceClass.DURABLE),
        **dict.fromkeys(CONFIGURABLE_EVENT_TYPES, PersistenceClass.CONFIGURABLE),
        **dict.fromkeys(TRANSIENT_EVENT_TYPES, PersistenceClass.TRANSIENT),
    }
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
    depth: int | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("event run_id must not be empty")
        if self.seq < 1:
            raise ValueError("event seq must be positive")
        if not self.timestamp:
            raise ValueError("event timestamp must not be empty")
        if self.version != TRACE_ABI_VERSION:
            raise ValueError(f"unsupported trace event version: {self.version}")
        if self.depth is not None and self.depth < 0:
            raise ValueError("event depth must be non-negative")
        expected = persistence_class_for(self.type)
        if self.persistence_class is not expected:
            raise ValueError(
                f"event {self.type!r} must be {expected.value}, not {self.persistence_class.value}"
            )
        object.__setattr__(self, "body", _freeze_json(self.body))

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
        if self.depth is not None:
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
    """The one strict parser for Trace ABI v1 event values."""
    if isinstance(value, RunEvent):
        return value
    required = {
        "run_id",
        "seq",
        "timestamp",
        "type",
        "version",
        "persistence_class",
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
    if value.get("depth") is not None and (
        isinstance(value["depth"], bool) or not isinstance(value["depth"], int)
    ):
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
        depth=value["depth"] if value.get("depth") is not None else None,
    )


@dataclass(frozen=True, slots=True)
class TraceRetentionPolicy:
    """Host selection for configurable content; durable facts are unconditional."""

    retain: frozenset[str] = frozenset()

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

    def as_dict(self) -> dict[str, Any]:
        return {"retain": sorted(self.retain)}


@dataclass(frozen=True, slots=True)
class DataUseAuthorization:
    """Independent host authorization; retention never grants training use."""

    training_allowed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.training_allowed, bool):
            raise TypeError("training_allowed must be a bool")

    def as_dict(self) -> dict[str, bool]:
        return {"training_allowed": self.training_allowed}


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
    depth: int | None = None
    version: int = TRACE_ABI_VERSION

    def __post_init__(self) -> None:
        events = tuple(self.events)
        if not self.run_id:
            raise ValueError("run record run_id must not be empty")
        if self.version != TRACE_ABI_VERSION:
            raise ValueError(f"unsupported run record version: {self.version}")
        if self.depth is not None and self.depth < 0:
            raise ValueError("run record depth must be non-negative")
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
        if self.depth is not None:
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
    depth: int | None = None
    retention: TraceRetentionPolicy = field(default_factory=TraceRetentionPolicy)
    data_use: DataUseAuthorization = field(default_factory=DataUseAuthorization)
    clock: Clock = utc_now
    monotonic_clock: MonotonicClock = monotonic
    _events: list[RunEvent] = field(default_factory=list, init=False, repr=False)
    _terminal_record: RunRecord | None = field(default=None, init=False, repr=False)
    _started_monotonic: float | None = field(default=None, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("trace run_id must not be empty")
        if self.depth is not None and self.depth < 0:
            raise ValueError("trace depth must be non-negative")

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
