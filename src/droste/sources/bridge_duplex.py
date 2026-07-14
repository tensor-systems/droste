"""Strict bridge-v2 transport state for remote provider invocations."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any, Protocol

from ..capabilities import (
    CapabilityCancelled,
    CapabilityCheckpoint,
    CapabilityDeadlineExceeded,
    CapabilityError,
    CapabilityErrorCode,
    CapabilityExecutionContext,
    CapabilityOutcome,
)

BRIDGE_PROTOCOL_VERSION = 2
MAX_FRAME_BYTES = 8 * 1024 * 1024
TRANSPORT_LOST = "bridge.transport_lost"
PROTOCOL_ERROR = "bridge.protocol_error"

DuplexControl = Callable[[str], Any]


class DuplexBridgeSession(Protocol):
    """One host-owned message pump for a single bridge-v2 invocation."""

    def receive(self) -> Any: ...

    def send(self, message_json: str) -> Any: ...

    def cancellation_requested(self, call_id: str) -> Any: ...

    def close(self) -> Any: ...


DuplexBridgeCall = Callable[[str, str], DuplexBridgeSession]


class BridgeTransportLost(RuntimeError):
    """The selected duplex transport disappeared before terminal delivery."""


class BridgeProtocolError(RuntimeError):
    """A duplex peer violated the strict bridge v2 message contract."""


@dataclass(frozen=True, slots=True)
class _DuplexEvent:
    call_id: str
    seq: int
    kind: str
    checkpoint: CapabilityCheckpoint | None = None
    outcome: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id:
            raise BridgeProtocolError("duplex event requires call_id")
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 1:
            raise BridgeProtocolError("duplex event seq must be a positive integer")
        if self.kind not in {"check", "checkpoint", "terminal"}:
            raise BridgeProtocolError("duplex event has an unknown kind")
        if self.kind == "check" and (self.checkpoint is not None or self.outcome is not None):
            raise BridgeProtocolError("duplex check event shape is invalid")
        if self.kind == "checkpoint" and (self.checkpoint is None or self.outcome is not None):
            raise BridgeProtocolError("duplex checkpoint event shape is invalid")
        if self.kind == "terminal" and (
            self.checkpoint is None or not isinstance(self.outcome, Mapping)
        ):
            raise BridgeProtocolError("duplex terminal event shape is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": BRIDGE_PROTOCOL_VERSION,
            "call_id": self.call_id,
            "seq": self.seq,
            "kind": self.kind,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "outcome": dict(self.outcome) if isinstance(self.outcome, Mapping) else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> _DuplexEvent:
        if set(value) != {"version", "call_id", "seq", "kind", "checkpoint", "outcome"}:
            raise BridgeProtocolError("duplex event requires the exact version 2 fields")
        if value.get("version") != BRIDGE_PROTOCOL_VERSION:
            raise BridgeProtocolError("duplex event version mismatch")
        raw_checkpoint = value.get("checkpoint")
        if raw_checkpoint is not None and not isinstance(raw_checkpoint, Mapping):
            raise BridgeProtocolError("duplex checkpoint must be an object or null")
        try:
            checkpoint = (
                CapabilityCheckpoint.from_dict(raw_checkpoint)
                if isinstance(raw_checkpoint, Mapping)
                else None
            )
        except (TypeError, ValueError) as exc:
            raise BridgeProtocolError(str(exc)) from exc
        raw_outcome = value.get("outcome")
        if raw_outcome is not None and not isinstance(raw_outcome, Mapping):
            raise BridgeProtocolError("duplex outcome must be an object or null")
        return cls(
            call_id=value.get("call_id"),  # type: ignore[arg-type]
            seq=value.get("seq"),  # type: ignore[arg-type]
            kind=value.get("kind"),  # type: ignore[arg-type]
            checkpoint=checkpoint,
            outcome=dict(raw_outcome) if isinstance(raw_outcome, Mapping) else None,
        )


@dataclass(frozen=True, slots=True)
class _DuplexAck:
    call_id: str
    seq: int
    error: CapabilityError | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id:
            raise BridgeProtocolError("duplex acknowledgement requires call_id")
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 1:
            raise BridgeProtocolError("duplex acknowledgement seq must be a positive integer")
        if self.error is not None and not isinstance(self.error, CapabilityError):
            raise BridgeProtocolError(
                "duplex acknowledgement error must be a CapabilityError or null"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": BRIDGE_PROTOCOL_VERSION,
            "call_id": self.call_id,
            "seq": self.seq,
            "ok": self.error is None,
            "error": self.error.to_dict() if self.error else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> _DuplexAck:
        if set(value) != {"version", "call_id", "seq", "ok", "error"}:
            raise BridgeProtocolError("duplex acknowledgement requires exact version 2 fields")
        if value.get("version") != BRIDGE_PROTOCOL_VERSION:
            raise BridgeProtocolError("duplex acknowledgement version mismatch")
        ok = value.get("ok")
        if not isinstance(ok, bool):
            raise BridgeProtocolError("duplex acknowledgement ok must be a bool")
        raw_error = value.get("error")
        if raw_error is not None and not isinstance(raw_error, Mapping):
            raise BridgeProtocolError("duplex acknowledgement error must be an object or null")
        try:
            error = CapabilityError.from_dict(raw_error) if isinstance(raw_error, Mapping) else None
        except (TypeError, ValueError) as exc:
            raise BridgeProtocolError(str(exc)) from exc
        if ok == (error is not None):
            raise BridgeProtocolError("duplex acknowledgement ok/error fields disagree")
        return cls(
            call_id=value.get("call_id"),  # type: ignore[arg-type]
            seq=value.get("seq"),  # type: ignore[arg-type]
            error=error,
        )


def await_value(value: Any) -> Any:
    if hasattr(value, "__await__"):
        from pyodide.ffi import run_sync

        return run_sync(value)
    return value


def _bridge_error(code: str, error_type: str, message: str) -> CapabilityOutcome:
    return CapabilityOutcome(error=CapabilityError(code, error_type, message[:1000]))


def _encode_event(event: _DuplexEvent) -> str:
    event_json = json.dumps(event.to_dict())
    if len(event_json.encode("utf-8")) > MAX_FRAME_BYTES:
        raise BridgeProtocolError("duplex frame exceeds 8 MiB")
    return event_json


def _send_control(control: DuplexControl, event: _DuplexEvent) -> _DuplexAck:
    event_json = _encode_event(event)
    try:
        raw = await_value(control(event_json))
    except BaseException as exc:
        if not isinstance(exc, Exception):
            raise
        raise BridgeTransportLost(str(exc)) from exc
    try:
        if not isinstance(raw, str):
            raise BridgeProtocolError("duplex acknowledgement must be JSON text")
        if len(raw.encode("utf-8")) > MAX_FRAME_BYTES:
            raise BridgeProtocolError("duplex acknowledgement exceeds 8 MiB")
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise BridgeProtocolError("duplex acknowledgement must be a JSON object")
        ack = _DuplexAck.from_dict(payload)
    except (TypeError, ValueError, BridgeProtocolError) as exc:
        raise BridgeProtocolError(str(exc)) from exc
    if ack.call_id != event.call_id or ack.seq != event.seq:
        raise BridgeProtocolError("duplex acknowledgement identity mismatch")
    return ack


def _decode_operation_result(result: Mapping[str, Any]) -> Any:
    kind = result.get("kind")
    value = result.get("value")
    if kind == "value":
        return value
    if kind == "capability_outcome" and isinstance(value, Mapping):
        return CapabilityOutcome.from_dict(value)
    raise BridgeProtocolError("bridge invoke response has an unknown result kind")


def duplex_terminal(result: Mapping[str, Any], call_id: str, last_seq: int) -> dict[str, Any]:
    return {
        **result,
        "bridge_protocol": BRIDGE_PROTOCOL_VERSION,
        "call_id": call_id,
        "last_seq": last_seq,
    }


def deliver_terminal(result: Mapping[str, Any], control: DuplexControl) -> dict[str, Any]:
    """Deliver one terminal frame, compacting an oversized provider result."""

    raw_checkpoint = result.get("checkpoint")
    if not isinstance(raw_checkpoint, Mapping):
        raise BridgeProtocolError("duplex terminal requires a checkpoint")
    terminal = _DuplexEvent(
        call_id=result["call_id"],
        seq=result["last_seq"] + 1,
        kind="terminal",
        checkpoint=CapabilityCheckpoint.from_dict(raw_checkpoint),
        outcome={"kind": result.get("kind"), "value": result.get("value")},
    )
    try:
        _encode_event(terminal)
    except BridgeProtocolError:
        # A provider result that cannot fit in one bounded frame is a
        # deterministic protocol rejection, not transport loss. Send
        # a compact terminal so the receiver can settle it normally.
        terminal = _DuplexEvent(
            call_id=terminal.call_id,
            seq=terminal.seq,
            kind="terminal",
            checkpoint=terminal.checkpoint,
            outcome={
                "kind": "capability_outcome",
                "value": _bridge_error(
                    PROTOCOL_ERROR,
                    "BridgeProtocolError",
                    "duplex terminal exceeds 8 MiB",
                ).to_dict(),
            },
        )
    ack = _send_control(control, terminal)
    if ack.error is not None:
        raise BridgeProtocolError(ack.error.message)
    return {
        "bridge_protocol": BRIDGE_PROTOCOL_VERSION,
        "call_id": terminal.call_id,
        "terminal_seq": terminal.seq,
    }


class _DuplexReceiver:
    """Strict ordered reducer from bridge events into one receiving context."""

    def __init__(self, execution: CapabilityExecutionContext) -> None:
        self._execution = execution
        self._lock = Lock()
        self._last_event: _DuplexEvent | None = None
        self._last_ack: _DuplexAck | None = None
        self._checkpoint = CapabilityCheckpoint()
        self._control_error: CapabilityError | None = None
        self._last_terminal: Any = None

    def accept(
        self,
        event: _DuplexEvent,
        *,
        cancellation_requested: bool = False,
    ) -> tuple[_DuplexAck, bool, Any]:
        with self._lock:
            if event.call_id != self._execution.call_id:
                error = CapabilityError(
                    PROTOCOL_ERROR,
                    "BridgeProtocolError",
                    "duplex event call_id does not match the active call",
                )
                ack = _DuplexAck(event.call_id, event.seq, error)
                self._control_error = error
                is_terminal = False
                terminal = None
            elif self._last_event is not None and event == self._last_event:
                assert self._last_ack is not None
                ack = self._last_ack
                is_terminal = event.kind == "terminal"
                terminal = self._last_terminal if is_terminal else None
            elif event.seq != (self._last_event.seq + 1 if self._last_event else 1):
                error = CapabilityError(
                    PROTOCOL_ERROR,
                    "BridgeProtocolError",
                    "duplex events must be delivered in order",
                )
                ack = _DuplexAck(event.call_id, event.seq, error)
                self._control_error = error
                is_terminal = False
                terminal = None
            else:
                ack, is_terminal, terminal = self._apply(
                    event, cancellation_requested=cancellation_requested
                )
                self._last_event = event
                self._last_ack = ack
                if is_terminal:
                    self._last_terminal = terminal
        return ack, is_terminal, terminal

    def _apply(
        self,
        event: _DuplexEvent,
        *,
        cancellation_requested: bool,
    ) -> tuple[_DuplexAck, bool, Any]:
        is_terminal = event.kind == "terminal"
        terminal: Any | None = None
        try:
            if event.kind == "check":
                if cancellation_requested:
                    raise CapabilityCancelled("capability call was cancelled")
                self._execution.check()
            elif event.kind == "checkpoint":
                assert event.checkpoint is not None
                self._checkpoint = self._execution.checkpoint(
                    tokens=event.checkpoint.tokens,
                    subcalls=event.checkpoint.subcalls,
                )
                if cancellation_requested:
                    raise CapabilityCancelled("capability call was cancelled")
            else:
                assert event.checkpoint is not None
                if event.checkpoint != self._checkpoint:
                    self._checkpoint = self._execution.checkpoint(
                        tokens=event.checkpoint.tokens,
                        subcalls=event.checkpoint.subcalls,
                    )
                if cancellation_requested:
                    raise CapabilityCancelled("capability call was cancelled")
                self._execution.check()
                assert isinstance(event.outcome, Mapping)
                terminal = _decode_operation_result(event.outcome)
        except CapabilityCancelled as exc:
            error = CapabilityError(CapabilityErrorCode.CANCELLED, type(exc).__name__, str(exc))
        except CapabilityDeadlineExceeded as exc:
            error = CapabilityError(
                CapabilityErrorCode.DEADLINE_EXCEEDED, type(exc).__name__, str(exc)
            )
        except Exception as exc:
            error = CapabilityError(PROTOCOL_ERROR, "BridgeProtocolError", str(exc))
        else:
            error = None
        if error is not None:
            self._control_error = error
            if event.kind == "terminal":
                terminal = CapabilityOutcome(error=error)
                # The terminal was consumed and reduced to the receiver's
                # authoritative cancellation/protocol fact. Ack its delivery;
                # the error already belongs in the broker outcome.
                return _DuplexAck(event.call_id, event.seq), True, terminal
        return _DuplexAck(event.call_id, event.seq, error), is_terminal, terminal

    def loss(self, message: str) -> CapabilityOutcome:
        if self._control_error is not None:
            return CapabilityOutcome(error=self._control_error)
        return _bridge_error(TRANSPORT_LOST, "BridgeTransportLost", message)


def request_operation(
    duplex_call: DuplexBridgeCall,
    operation_id: str,
    execution: CapabilityExecutionContext,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    """Run one receiving-side duplex operation through its message pump."""

    receiver = _DuplexReceiver(execution)
    try:
        params_json = json.dumps(
            {
                "bridge_protocol": BRIDGE_PROTOCOL_VERSION,
                "operation_id": operation_id,
                "args": list(args),
                "kwargs": dict(kwargs),
                "execution": execution.to_transport_dict(),
            }
        )
    except (TypeError, ValueError) as exc:
        return _bridge_error(PROTOCOL_ERROR, "BridgeProtocolError", str(exc))

    try:
        session = duplex_call("invoke", params_json)
        if not all(
            callable(getattr(session, name, None))
            for name in ("receive", "send", "cancellation_requested", "close")
        ):
            raise BridgeProtocolError(
                "duplex_call must return a receive/send/cancellation_requested/close session"
            )
    except BridgeProtocolError as exc:
        return _bridge_error(PROTOCOL_ERROR, "BridgeProtocolError", str(exc))
    except BaseException as exc:
        if not isinstance(exc, Exception):
            raise
        return _bridge_error(TRANSPORT_LOST, "BridgeTransportLost", str(exc))

    try:
        while True:
            try:
                raw = await_value(session.receive())
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    raise
                return receiver.loss(str(exc))
            try:
                if not isinstance(raw, str):
                    raise BridgeProtocolError("duplex frame must be JSON text")
                if len(raw.encode("utf-8")) > MAX_FRAME_BYTES:
                    raise BridgeProtocolError("duplex frame exceeds 8 MiB")
                frame_value = json.loads(raw)
                if not isinstance(frame_value, Mapping):
                    raise BridgeProtocolError("duplex frame must be a JSON object")
                frame = _DuplexEvent.from_dict(frame_value)
            except (TypeError, ValueError, BridgeProtocolError) as exc:
                return _bridge_error(PROTOCOL_ERROR, "BridgeProtocolError", str(exc))

            try:
                requested = await_value(session.cancellation_requested(execution.call_id))
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    raise
                return receiver.loss(str(exc))
            if not isinstance(requested, bool):
                return _bridge_error(
                    PROTOCOL_ERROR,
                    "BridgeProtocolError",
                    "duplex cancellation_requested must return a bool",
                )
            ack, is_terminal, terminal = receiver.accept(frame, cancellation_requested=requested)
            try:
                await_value(session.send(json.dumps(ack.to_dict())))
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    raise
                if is_terminal:
                    return terminal
                return receiver.loss(str(exc))
            if ack.error is not None and ack.error.code == PROTOCOL_ERROR:
                return CapabilityOutcome(error=ack.error)
            if is_terminal:
                return terminal
    except (CapabilityCancelled, CapabilityDeadlineExceeded):
        raise
    except (TypeError, ValueError, BridgeProtocolError) as exc:
        return _bridge_error(PROTOCOL_ERROR, "BridgeProtocolError", str(exc))
    finally:
        try:
            await_value(session.close())
        except Exception:
            pass


def execution_context(
    local: CapabilityExecutionContext,
    local_checkpoint: Callable[[], CapabilityCheckpoint],
    control: DuplexControl,
) -> tuple[
    CapabilityExecutionContext,
    Callable[[], CapabilityCheckpoint],
    Callable[[], int],
]:
    """Wrap local execution facts with bridge-v2 control mechanisms."""

    transition = Lock()
    remote_cancellation = Event()
    seq = [0]

    def send(kind: str, checkpoint: CapabilityCheckpoint | None = None) -> None:
        next_seq = seq[0] + 1
        event = _DuplexEvent(local.call_id, next_seq, kind, checkpoint)
        ack = _send_control(control, event)
        # An identity-valid acknowledgement consumes the frame even when it
        # carries cancellation, deadline, or protocol control.
        seq[0] = next_seq
        if ack.error is not None:
            if ack.error.code == CapabilityErrorCode.CANCELLED:
                remote_cancellation.set()
                raise CapabilityCancelled(ack.error.message)
            if ack.error.code == CapabilityErrorCode.DEADLINE_EXCEEDED:
                raise CapabilityDeadlineExceeded(ack.error.message)
            raise BridgeProtocolError(ack.error.message)

    def check() -> None:
        with transition:
            local.check()
            send("check")

    def checkpoint(next_value: CapabilityCheckpoint) -> CapabilityCheckpoint:
        with transition:
            local.check()
            try:
                send("checkpoint", next_value)
            except CapabilityCancelled:
                # A cancelled checkpoint acknowledgement means the receiver
                # accepted the cumulative usage before surfacing cancellation.
                # Mirror that accepted fact locally so the terminal repeats it.
                local.checkpoint(tokens=next_value.tokens, subcalls=next_value.subcalls)
                raise
            return local.checkpoint(tokens=next_value.tokens, subcalls=next_value.subcalls)

    def is_cancelled() -> bool:
        # This property is a local, non-throwing observation. Live remote
        # cancellation is accepted at the explicit check/checkpoint boundary,
        # which records the fact here before raising to the handler.
        return local.cancellation_requested or remote_cancellation.is_set()

    context = CapabilityExecutionContext(
        call_id=local.call_id,
        run_id=local.run_id,
        parent_run_id=local.parent_run_id,
        deadline_monotonic=local.deadline_monotonic,
        reservation=local.reservation,
        _check=check,
        _checkpoint=checkpoint,
        _is_cancelled=is_cancelled,
    )

    def last_seq() -> int:
        with transition:
            return seq[0]

    return context, local_checkpoint, last_seq


__all__ = [
    "BRIDGE_PROTOCOL_VERSION",
    "PROTOCOL_ERROR",
    "TRANSPORT_LOST",
    "BridgeProtocolError",
    "BridgeTransportLost",
    "DuplexBridgeCall",
    "DuplexBridgeSession",
    "DuplexControl",
    "await_value",
    "deliver_terminal",
    "duplex_terminal",
    "execution_context",
    "request_operation",
]
