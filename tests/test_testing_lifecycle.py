"""Published lifecycle scenario helpers remain deterministic and policy-free."""

from __future__ import annotations

from threading import Event

import pytest

from droste import (
    CapabilityCall,
    CapabilityCheckpoint,
    CapabilityError,
    CapabilityId,
    CapabilityKind,
)
from droste.testing import (
    LifecycleGate,
    RecordingAttemptAuthority,
    require_ordered_terminal_events,
    require_unknown_completion,
    run_while_blocked,
)


def _call(call_id: str = "call-1") -> CapabilityCall:
    return CapabilityCall(
        CapabilityId(CapabilityKind.DATA, "read", source_id="source"),
        call_id,
        "run-1",
    )


def test_blocked_scenario_uses_an_explicit_barrier_and_preserves_its_value() -> None:
    gate = LifecycleGate()
    actions: list[str] = []

    def operation() -> str:
        actions.append("entered")
        gate.pause()
        actions.append("released")
        return "complete"

    outcome = run_while_blocked(
        operation,
        gate=gate,
        while_blocked=lambda: actions.append("while-blocked"),
    )

    assert outcome.require_value() == "complete"
    assert actions == ["entered", "while-blocked", "released"]


def test_blocked_scenario_preserves_the_operation_error_as_a_terminal_fact() -> None:
    gate = LifecycleGate()
    failure = RuntimeError("failed after release")

    def operation() -> None:
        gate.pause()
        raise failure

    outcome = run_while_blocked(operation, gate=gate, while_blocked=lambda: None)

    assert outcome.error is failure
    with pytest.raises(AssertionError, match="raised unexpectedly") as raised:
        outcome.require_value()
    assert raised.value.__cause__ is failure


def test_blocked_scenario_surfaces_failure_before_the_rendezvous() -> None:
    gate = LifecycleGate()
    failure = RuntimeError("failed before arrival")

    with pytest.raises(AssertionError, match="failed before reaching") as raised:
        run_while_blocked(
            lambda: (_ for _ in ()).throw(failure),
            gate=gate,
            while_blocked=lambda: None,
            timeout=0.01,
        )

    assert raised.value.__cause__ is failure


def test_blocked_scenario_runs_timeout_cleanup_before_reporting_a_stuck_operation() -> None:
    gate = LifecycleGate()
    stopped = Event()
    cleanup_calls: list[str] = []

    def operation() -> None:
        gate.pause()
        stopped.wait()

    def cleanup() -> None:
        cleanup_calls.append("cleanup")
        stopped.set()

    with pytest.raises(AssertionError, match="exceeded its bound"):
        run_while_blocked(
            operation,
            gate=gate,
            while_blocked=lambda: None,
            timeout=0.01,
            on_timeout=cleanup,
        )

    assert cleanup_calls == ["cleanup"]


def test_recording_authority_exposes_snapshots_and_detects_duplicate_settlement() -> None:
    authority = RecordingAttemptAuthority()
    call = _call()
    checkpoint = CapabilityCheckpoint(tokens=2, subcalls=1)
    error = CapabilityError("cancelled", "Cancelled", "cancelled")

    authority.admit(call)
    authority.checkpoint(call, checkpoint)
    authority.settle(call, None, error, checkpoint, attempted=True)

    assert authority.admissions == ("call-1",)
    assert authority.active_calls == frozenset()
    assert authority.checkpoints == (("call-1", checkpoint),)
    assert authority.require_single_settlement("call-1").error_code == "cancelled"

    authority.settle(call, None, error, checkpoint, attempted=True)
    with pytest.raises(AssertionError, match="settled 2 times"):
        authority.require_single_settlement("call-1")


def test_unknown_completion_and_terminal_event_assertions_are_value_checks() -> None:
    unknown = CapabilityError("transport_error", "TransportError", "completion unknown")
    require_unknown_completion(unknown, attempts=1)
    assert require_ordered_terminal_events(
        (
            {"seq": 1, "type": "startup"},
            {"seq": 2, "type": "result"},
            {"seq": 3, "type": "done"},
        )
    ) == ("startup", "result", "done")

    with pytest.raises(AssertionError, match="attempted 2 times"):
        require_unknown_completion(unknown, attempts=2)
    with pytest.raises(AssertionError, match="not contiguous"):
        require_ordered_terminal_events(({"seq": 2, "type": "result"}, {"seq": 3, "type": "done"}))
    with pytest.raises(AssertionError, match="must not be empty"):
        require_ordered_terminal_events(())
