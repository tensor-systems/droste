"""Deterministic, transport-neutral lifecycle test helpers.

These helpers deliberately describe test facts and synchronization points. They
do not provide a production lifecycle abstraction or choose retry/cancellation
policy for a caller.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any, Generic, TypeVar

from ..capabilities import (
    CapabilityAdmission,
    CapabilityCall,
    CapabilityCheckpoint,
    CapabilityError,
    CapabilityMetadata,
    CapabilityReservation,
)

T = TypeVar("T")

DEFAULT_LIFECYCLE_TIMEOUT = 5.0


class _GateTimeout(AssertionError):
    pass


class LifecycleGate:
    """One explicit rendezvous for a deterministic blocked-operation test."""

    def __init__(self, *, timeout: float = DEFAULT_LIFECYCLE_TIMEOUT) -> None:
        self._entered = Event()
        self._release = Event()
        self._timeout = timeout

    def pause(self) -> None:
        """Signal entry and block until the test explicitly releases the gate."""

        self.arrive()
        if not self._release.wait(self._timeout):
            raise _GateTimeout("lifecycle gate was not released")

    def arrive(self) -> None:
        """Signal the rendezvous without blocking an observational callback."""

        self._entered.set()

    def wait_until_paused(self) -> None:
        """Fail with a useful error when the operation never reaches the gate."""

        if not self._entered.wait(self._timeout):
            raise _GateTimeout("operation did not reach lifecycle gate")

    def release(self) -> None:
        self._release.set()


@dataclass(frozen=True, slots=True)
class ThreadOutcome(Generic[T]):
    """The immutable terminal fact produced by one test operation."""

    value: T | None = None
    error: BaseException | None = None

    def require_value(self) -> T:
        if self.error is not None:
            raise AssertionError("operation raised unexpectedly") from self.error
        return self.value  # type: ignore[return-value]


def run_while_blocked(
    operation: Callable[[], T],
    *,
    gate: LifecycleGate,
    while_blocked: Callable[[], None],
    timeout: float = DEFAULT_LIFECYCLE_TIMEOUT,
) -> ThreadOutcome[T]:
    """Run an operation, act at its explicit barrier, and require bounded exit."""

    values: list[T] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            values.append(operation())
        except BaseException as exc:
            errors.append(exc)

    worker = Thread(target=run, name="droste-lifecycle-scenario", daemon=True)
    worker.start()
    coordination_error: BaseException | None = None
    try:
        gate.wait_until_paused()
        while_blocked()
    except BaseException as exc:
        coordination_error = exc
    finally:
        gate.release()
        worker.join(timeout)
    if worker.is_alive():
        raise AssertionError("lifecycle operation did not terminate within its bound") from (
            coordination_error
        )
    if isinstance(coordination_error, _GateTimeout) and errors:
        raise AssertionError("operation failed before reaching its lifecycle gate") from errors[0]
    if coordination_error is not None:
        raise coordination_error
    if len(values) + len(errors) != 1:
        raise AssertionError("lifecycle operation did not produce exactly one terminal outcome")
    return ThreadOutcome(
        value=values[0] if values else None,
        error=errors[0] if errors else None,
    )


@dataclass(frozen=True, slots=True)
class Settlement:
    """One immutable attempt-settlement observation."""

    call_id: str
    error_code: str | None
    checkpoint: CapabilityCheckpoint
    attempted: bool


class RecordingAttemptAuthority:
    """A thread-safe recording authority for transport conformance tests."""

    def __init__(
        self,
        *,
        reservation: CapabilityReservation = CapabilityReservation(),
        deadline_monotonic: float | None = None,
    ) -> None:
        self._admission = CapabilityAdmission(reservation, deadline_monotonic)
        self._lock = Lock()
        self._admissions: list[str] = []
        self._active: set[str] = set()
        self._checkpoints: list[tuple[str, CapabilityCheckpoint]] = []
        self._settlements: list[Settlement] = []

    @property
    def admissions(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._admissions)

    @property
    def active_calls(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._active)

    @property
    def checkpoints(self) -> tuple[tuple[str, CapabilityCheckpoint], ...]:
        with self._lock:
            return tuple(self._checkpoints)

    @property
    def settlements(self) -> tuple[Settlement, ...]:
        with self._lock:
            return tuple(self._settlements)

    def admit(self, call: CapabilityCall) -> CapabilityAdmission:
        with self._lock:
            self._admissions.append(call.call_id)
            self._active.add(call.call_id)
        return self._admission

    def checkpoint(
        self, call: CapabilityCall, cumulative: CapabilityCheckpoint
    ) -> CapabilityCheckpoint:
        with self._lock:
            self._checkpoints.append((call.call_id, cumulative))
        return cumulative

    def settle(
        self,
        call: CapabilityCall,
        result: Any,
        error: CapabilityError | None,
        checkpoint: CapabilityCheckpoint,
        *,
        attempted: bool,
    ) -> CapabilityMetadata:
        del result
        settlement = Settlement(call.call_id, error.code if error else None, checkpoint, attempted)
        with self._lock:
            self._settlements.append(settlement)
            self._active.discard(call.call_id)
        return CapabilityMetadata()

    def require_single_settlement(self, call_id: str) -> Settlement:
        matches = tuple(item for item in self.settlements if item.call_id == call_id)
        if len(matches) != 1:
            raise AssertionError(
                f"capability {call_id!r} settled {len(matches)} times instead of exactly once"
            )
        return matches[0]


def require_unknown_completion(error: CapabilityError, *, attempts: int) -> None:
    """Assert the transport's ambiguous-completion invariant without naming it."""

    if attempts != 1:
        raise AssertionError(f"unknown completion was attempted {attempts} times")
    if error.retryable:
        raise AssertionError("unknown completion must not be advertised as retryable")


def require_ordered_terminal_events(events: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Validate contiguous live delivery and return its substrate-neutral order."""

    if not events:
        raise AssertionError("event stream must not be empty")
    sequences = tuple(event.get("seq") for event in events)
    expected = tuple(range(1, len(events) + 1))
    if sequences != expected:
        raise AssertionError(f"event sequence is not contiguous: {sequences!r}")
    types = tuple(str(event.get("type")) for event in events)
    if types.count("result") != 1 or types.count("done") != 1:
        raise AssertionError("event stream must contain exactly one result and one done event")
    if types[-1] != "done" or types.index("result") > types.index("done"):
        raise AssertionError("result must precede the terminal done event")
    return types


__all__ = [
    "DEFAULT_LIFECYCLE_TIMEOUT",
    "LifecycleGate",
    "RecordingAttemptAuthority",
    "Settlement",
    "ThreadOutcome",
    "require_ordered_terminal_events",
    "require_unknown_completion",
    "run_while_blocked",
]
