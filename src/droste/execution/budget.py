"""Immutable compute budgets and the one run-scoped accounting ledger.

The budget is caller-owned data.  The ledger is the sole mutable identity that
reserves and reconciles that data across root requests, capability calls, and
future child runs.  It deliberately knows nothing about transports or models.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from math import isfinite
from threading import Lock, RLock
from time import monotonic
from typing import Any, Callable, Mapping

DEFAULT_TOKEN_BUDGET = 500_000
DEFAULT_SUBCALL_BUDGET = 50
DEFAULT_DEPTH_BUDGET = 1
DEFAULT_WALL_TIME_MS = 300_000
DEFAULT_ROOT_OUTPUT_TOKENS = 4_096
DEFAULT_SUBCALL_OUTPUT_TOKENS = 2_048

_RESERVABLE_RESOURCES = ("tokens", "subcalls", "wall_ms")


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    value = _non_negative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class Budget:
    """One fully resolved caller authorization vector.

    ``tokens`` and ``subcalls`` are consumable. ``wall_ms`` defines one shared
    run deadline, and ``depth`` is a nesting ceiling. The two output fields are
    per-request ceilings and remain separate facts within the same
    authorization value.
    """

    tokens: int = DEFAULT_TOKEN_BUDGET
    subcalls: int = DEFAULT_SUBCALL_BUDGET
    depth: int = DEFAULT_DEPTH_BUDGET
    wall_ms: int = DEFAULT_WALL_TIME_MS
    root_output_tokens: int = DEFAULT_ROOT_OUTPUT_TOKENS
    subcall_output_tokens: int = DEFAULT_SUBCALL_OUTPUT_TOKENS

    def __post_init__(self) -> None:
        _positive_int(self.tokens, "budget.tokens")
        _non_negative_int(self.subcalls, "budget.subcalls")
        _non_negative_int(self.depth, "budget.depth")
        _positive_int(self.wall_ms, "budget.wall_ms")
        _positive_int(self.root_output_tokens, "budget.root_output_tokens")
        _positive_int(self.subcall_output_tokens, "budget.subcall_output_tokens")
        if self.root_output_tokens > self.tokens:
            raise ValueError("budget.root_output_tokens cannot exceed budget.tokens")
        if self.subcall_output_tokens > self.tokens:
            raise ValueError("budget.subcall_output_tokens cannot exceed budget.tokens")

    def as_dict(self) -> dict[str, int]:
        return {
            "tokens": self.tokens,
            "subcalls": self.subcalls,
            "depth": self.depth,
            "wall_ms": self.wall_ms,
            "root_output_tokens": self.root_output_tokens,
            "subcall_output_tokens": self.subcall_output_tokens,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Budget:
        expected = {
            "tokens",
            "subcalls",
            "depth",
            "wall_ms",
            "root_output_tokens",
            "subcall_output_tokens",
        }
        missing = expected - value.keys()
        unknown = value.keys() - expected
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unknown " + ", ".join(sorted(unknown)))
            raise ValueError("budget has " + "; ".join(details))
        return cls(**{name: value[name] for name in expected})


@dataclass(frozen=True, slots=True)
class BudgetRequest:
    """Consumable reservation plus an optional child-depth increment."""

    tokens: int = 0
    subcalls: int = 0
    wall_ms: int = 0
    depth: int = 0

    def __post_init__(self) -> None:
        for name in ("tokens", "subcalls", "wall_ms", "depth"):
            _non_negative_int(getattr(self, name), f"budget request {name}")

    def as_dict(self) -> dict[str, int]:
        return {
            "tokens": self.tokens,
            "subcalls": self.subcalls,
            "wall_ms": self.wall_ms,
            "depth": self.depth,
        }


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    call_id: str
    request: BudgetRequest
    started_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id:
            raise ValueError("budget reservation call_id must not be empty")
        if not isinstance(self.request, BudgetRequest):
            raise TypeError("budget reservation requires a BudgetRequest")
        if (
            isinstance(self.started_at, bool)
            or not isinstance(self.started_at, (int, float))
            or not isfinite(self.started_at)
        ):
            raise ValueError("budget reservation started_at must be finite")


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    configured: Budget
    consumed: BudgetRequest
    reserved: BudgetRequest
    remaining: BudgetRequest
    current_depth: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "configured": self.configured.as_dict(),
            "consumed": self.consumed.as_dict(),
            "reserved": self.reserved.as_dict(),
            "remaining": self.remaining.as_dict(),
            "current_depth": self.current_depth,
        }


class BudgetExhausted(RuntimeError):
    """Typed rejection raised before authorized compute can be exceeded."""

    def __init__(self, resource: str, requested: int, remaining: int) -> None:
        self.resource = resource
        self.requested = requested
        self.remaining = remaining
        super().__init__(
            f"budget exhausted for {resource}: requested {requested}, remaining {remaining}"
        )


BudgetEventSink = Callable[[dict[str, Any]], None]
MonotonicClock = Callable[[], float]


@dataclass(slots=True)
class BudgetLedger:
    """The single lock-owning reservation and reconciliation authority."""

    budget: Budget
    depth: int = 0
    on_event: BudgetEventSink | None = None
    clock: MonotonicClock = monotonic
    _started_at: float = field(init=False, repr=False)
    _consumed_tokens: int = field(default=0, init=False, repr=False)
    _consumed_subcalls: int = field(default=0, init=False, repr=False)
    _reservations: dict[str, BudgetReservation] = field(
        default_factory=dict, init=False, repr=False
    )
    _checkpoints: dict[str, BudgetRequest] = field(default_factory=dict, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _parent: BudgetLedger | None = field(default=None, init=False, repr=False)
    _parent_call_id: str | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _event_journal: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _emitted_events: int = field(default=0, init=False, repr=False)
    _emit_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.budget, Budget):
            raise TypeError("BudgetLedger requires a Budget")
        _non_negative_int(self.depth, "ledger depth")
        self._started_at = self.clock()

    def _elapsed_ms_locked(self) -> int:
        return max(0, round((self.clock() - self._started_at) * 1000))

    def _reserved_locked(self) -> BudgetRequest:
        values = tuple(
            (
                item.request,
                self._checkpoints.get(item.call_id, BudgetRequest()),
            )
            for item in self._reservations.values()
        )
        return BudgetRequest(
            tokens=sum(request.tokens - checkpoint.tokens for request, checkpoint in values),
            subcalls=sum(request.subcalls - checkpoint.subcalls for request, checkpoint in values),
            # Wall time is one shared run deadline, not additive capacity.
            wall_ms=max((request.wall_ms for request, _ in values), default=0),
            depth=max((request.depth for request, _ in values), default=0),
        )

    def _remaining_locked(self) -> BudgetRequest:
        reserved = self._reserved_locked()
        elapsed = min(self._elapsed_ms_locked(), self.budget.wall_ms)
        return BudgetRequest(
            tokens=max(0, self.budget.tokens - self._consumed_tokens - reserved.tokens),
            subcalls=max(0, self.budget.subcalls - self._consumed_subcalls - reserved.subcalls),
            wall_ms=max(0, self.budget.wall_ms - elapsed),
            depth=self.budget.depth,
        )

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> BudgetSnapshot:
        reserved = self._reserved_locked()
        elapsed = min(self._elapsed_ms_locked(), self.budget.wall_ms)
        return BudgetSnapshot(
            configured=self.budget,
            consumed=BudgetRequest(
                tokens=self._consumed_tokens,
                subcalls=self._consumed_subcalls,
                wall_ms=elapsed,
                depth=0,
            ),
            reserved=reserved,
            remaining=self._remaining_locked(),
            current_depth=self.depth,
        )

    def reservation(self, call_id: str) -> BudgetReservation:
        """Return the immutable allocation facts for one live call."""

        with self._lock:
            try:
                return self._reservations[call_id]
            except KeyError as exc:
                raise ValueError(f"unknown budget reservation call_id: {call_id}") from exc

    def reserve(
        self,
        call_id: str,
        request: BudgetRequest,
        *,
        preserve_tokens: int = 0,
        through_deadline: bool = False,
    ) -> BudgetReservation:
        """Atomically reserve the entire vector or reject without mutation."""

        if not isinstance(call_id, str) or not call_id:
            raise ValueError("budget reservation call_id must not be empty")
        if not isinstance(request, BudgetRequest):
            raise TypeError("budget reserve requires a BudgetRequest")
        _non_negative_int(preserve_tokens, "preserve_tokens")
        if not isinstance(through_deadline, bool):
            raise TypeError("through_deadline must be a bool")
        exhaustion: BudgetExhausted | None = None
        reservation: BudgetReservation | None = None
        with self._lock:
            if self._closed:
                raise RuntimeError("budget ledger is closed")
            if call_id in self._reservations:
                raise ValueError(f"duplicate budget reservation call_id: {call_id}")
            remaining = self._remaining_locked()
            required_wall_ms = request.wall_ms
            if through_deadline:
                request = BudgetRequest(
                    tokens=request.tokens,
                    subcalls=request.subcalls,
                    wall_ms=remaining.wall_ms,
                    depth=request.depth,
                )
            checks = (
                ("tokens", request.tokens, max(0, remaining.tokens - preserve_tokens)),
                ("subcalls", request.subcalls, remaining.subcalls),
                ("wall_ms", request.wall_ms, remaining.wall_ms),
                ("depth", request.depth, remaining.depth),
            )
            failed = next((item for item in checks if item[1] > item[2]), None)
            if failed is None and through_deadline and required_wall_ms > remaining.wall_ms:
                failed = ("wall_ms", required_wall_ms, remaining.wall_ms)
            if failed is None and through_deadline and remaining.wall_ms == 0:
                failed = ("wall_ms", 1, 0)
            if failed is not None:
                resource, requested, available = failed
                self._queue_event_locked("exhaust", resource, requested, call_id)
                exhaustion = BudgetExhausted(resource, requested, available)
            else:
                started_at = self.clock()
                reservation = BudgetReservation(
                    call_id=call_id,
                    request=request,
                    started_at=started_at,
                )
                self._reservations[call_id] = reservation
                self._checkpoints[call_id] = BudgetRequest()
                for resource in (*_RESERVABLE_RESOURCES, "depth"):
                    amount = getattr(request, resource)
                    if amount:
                        self._queue_event_locked("reserve", resource, amount, call_id)
        self._drain_events()
        if exhaustion is not None:
            raise exhaustion
        assert reservation is not None
        return reservation

    def checkpoint(self, call_id: str, cumulative: BudgetRequest) -> BudgetRequest:
        """Commit a cumulative in-flight usage fact idempotently.

        Handlers report only token and subcall progress. Wall time remains a
        broker-measured fact and child depth is settled by the child ledger.
        Repeating the same cumulative value is a no-op; moving any dimension
        backward or beyond the reservation fails without changing accounting.
        """

        if not isinstance(cumulative, BudgetRequest):
            raise TypeError("budget checkpoint requires a BudgetRequest")
        if cumulative.wall_ms or cumulative.depth:
            raise ValueError("budget checkpoints may report only tokens and subcalls")
        exhaustion: BudgetExhausted | None = None
        with self._lock:
            reservation = self._reservations.get(call_id)
            if reservation is None:
                raise ValueError(f"unknown budget reservation call_id: {call_id}")
            previous = self._checkpoints[call_id]
            for resource in ("tokens", "subcalls"):
                value = getattr(cumulative, resource)
                prior = getattr(previous, resource)
                authorized = getattr(reservation.request, resource)
                if value < prior:
                    raise ValueError(
                        f"budget checkpoint {resource} cannot move backward: {value} < {prior}"
                    )
                if value > authorized:
                    self._queue_event_locked("exhaust", resource, value - authorized, call_id)
                    exhaustion = BudgetExhausted(resource, value, authorized)
                    break
            if exhaustion is not None:
                delta = BudgetRequest()
            else:
                delta = BudgetRequest(
                    tokens=cumulative.tokens - previous.tokens,
                    subcalls=cumulative.subcalls - previous.subcalls,
                )
            if exhaustion is None and delta == BudgetRequest():
                return previous
            if exhaustion is None:
                self._checkpoints[call_id] = cumulative
                self._consumed_tokens += delta.tokens
                self._consumed_subcalls += delta.subcalls
                for resource in ("tokens", "subcalls"):
                    amount = getattr(delta, resource)
                    if amount:
                        self._queue_event_locked("commit", resource, amount, call_id)
        self._drain_events()
        if exhaustion is not None:
            raise exhaustion
        return cumulative

    def commit(self, call_id: str, actual: BudgetRequest) -> BudgetRequest:
        """Commit actual spend and refund every unused reserved unit."""

        if not isinstance(actual, BudgetRequest):
            raise TypeError("budget commit requires a BudgetRequest")
        exhaustion: BudgetExhausted | None = None
        with self._lock:
            reservation = self._reservations.get(call_id)
            if reservation is None:
                raise ValueError(f"unknown budget reservation call_id: {call_id}")
            checkpoint = self._checkpoints[call_id]
            elapsed = max(0, round((self.clock() - reservation.started_at) * 1000))
            actual = BudgetRequest(
                tokens=actual.tokens,
                subcalls=actual.subcalls,
                wall_ms=elapsed if reservation.request.wall_ms else 0,
                depth=actual.depth,
            )
            overrun = next(
                (
                    resource
                    for resource in (*_RESERVABLE_RESOURCES, "depth")
                    if getattr(actual, resource) > getattr(reservation.request, resource)
                ),
                None,
            )
            # The checkpoint is an already-committed fact. A stale final
            # estimate can never retract it; reconciliation takes the
            # component-wise maximum and still closes the reservation.
            actual = BudgetRequest(
                tokens=max(actual.tokens, checkpoint.tokens),
                subcalls=max(actual.subcalls, checkpoint.subcalls),
                wall_ms=actual.wall_ms,
                depth=actual.depth,
            )
            if overrun is not None:
                requested = getattr(actual, overrun)
                authorized = getattr(reservation.request, overrun)
                self._queue_event_locked("exhaust", overrun, requested - authorized, call_id)
                exhaustion = BudgetExhausted(overrun, requested, authorized)
                actual = BudgetRequest(
                    tokens=min(actual.tokens, reservation.request.tokens),
                    subcalls=min(actual.subcalls, reservation.request.subcalls),
                    wall_ms=min(actual.wall_ms, reservation.request.wall_ms),
                    depth=min(actual.depth, reservation.request.depth),
                )
            del self._reservations[call_id]
            del self._checkpoints[call_id]
            self._consumed_tokens += actual.tokens - checkpoint.tokens
            self._consumed_subcalls += actual.subcalls - checkpoint.subcalls
            for resource in (*_RESERVABLE_RESOURCES, "depth"):
                consumed = getattr(actual, resource) - (
                    getattr(checkpoint, resource) if resource in {"tokens", "subcalls"} else 0
                )
                refunded = getattr(reservation.request, resource) - consumed
                if resource in {"tokens", "subcalls"}:
                    refunded -= getattr(checkpoint, resource)
                if consumed:
                    self._queue_event_locked("commit", resource, consumed, call_id)
                if refunded:
                    self._queue_event_locked("refund", resource, refunded, call_id)
        self._drain_events()
        if exhaustion is not None:
            raise exhaustion
        return actual

    def release(self, call_id: str) -> BudgetRequest:
        """Return a reservation that never reached dispatch."""

        with self._lock:
            reservation = self._reservations.pop(call_id, None)
            if reservation is None:
                raise ValueError(f"unknown budget reservation call_id: {call_id}")
            checkpoint = self._checkpoints.pop(call_id)
            for resource in (*_RESERVABLE_RESOURCES, "depth"):
                amount = getattr(reservation.request, resource) - (
                    getattr(checkpoint, resource) if resource in {"tokens", "subcalls"} else 0
                )
                if amount:
                    self._queue_event_locked("refund", resource, amount, call_id)
        self._drain_events()
        return reservation.request

    def child(self, call_id: str, budget: Budget) -> BudgetLedger:
        """Carve a strict child sub-ledger from one parent reservation."""

        if not isinstance(budget, Budget):
            raise TypeError("child ledger requires a Budget")
        available_child_depth = max(0, self.budget.depth - 1)
        if budget.depth > available_child_depth:
            raise BudgetExhausted(
                "depth",
                budget.depth,
                available_child_depth,
            )
        self.reserve(
            call_id,
            BudgetRequest(
                tokens=budget.tokens,
                subcalls=budget.subcalls,
                wall_ms=budget.wall_ms,
                depth=1,
            ),
            preserve_tokens=self.budget.root_output_tokens,
            through_deadline=True,
        )
        child = BudgetLedger(
            budget=budget,
            depth=self.depth + 1,
            on_event=self.on_event,
            clock=self.clock,
        )
        child._parent = self
        child._parent_call_id = call_id
        return child

    def close(self) -> None:
        """Reconcile a child into its parent exactly once."""

        with self._lock:
            if self._closed:
                return
            if self._reservations:
                raise RuntimeError("cannot close a budget ledger with active reservations")
            self._closed = True
            snapshot = self._snapshot_locked()
            parent = self._parent
            parent_call_id = self._parent_call_id
        if parent is not None and parent_call_id is not None:
            parent.commit(
                parent_call_id,
                BudgetRequest(
                    tokens=snapshot.consumed.tokens,
                    subcalls=snapshot.consumed.subcalls,
                    wall_ms=snapshot.consumed.wall_ms,
                    depth=1,
                ),
            )

    def _queue_event_locked(self, action: str, resource: str, amount: int, call_id: str) -> None:
        self._event_journal.append(
            {
                "type": "budget",
                "kind": "mutation",
                "source": "budget_ledger",
                "action": action,
                "resource": resource,
                "amount": amount,
                "call_id": call_id,
            }
        )

    def _drain_events(self) -> None:
        """Emit the ledger journal in mutation order without holding its lock."""

        if self.on_event is None:
            return
        with self._emit_lock:
            while True:
                with self._lock:
                    if self._emitted_events >= len(self._event_journal):
                        return
                    event = dict(self._event_journal[self._emitted_events])
                    # Mark before calling out. A re-entrant sink may reserve
                    # more work, but cannot emit this fact twice.
                    self._emitted_events += 1
                try:
                    self.on_event(event)
                except Exception as exc:
                    warnings.warn(
                        f"budget event sink failed: {type(exc).__name__}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


def conservative_token_estimate(value: Any) -> int:
    """Tokenizer-independent upper bound for JSON-compatible request content.

    A tokenizer cannot produce more tokens than the number of UTF-8 bytes in
    the serialized value. This intentionally trades utilization for a stable,
    dependency-free fail-closed bound.
    """

    encoded = json.dumps(
        _plain_json(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(encoded)
