"""Transport-neutral provider bridge with a fixed dispatch boundary."""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping
from threading import Event, Lock
from time import monotonic
from typing import Any

from ..capabilities import (
    CapabilityCancelled,
    CapabilityCheckpoint,
    CapabilityDeadlineExceeded,
    CapabilityError,
    CapabilityErrorCode,
    CapabilityExecutionContext,
    CapabilityOutcome,
    CapabilityReservation,
    SideEffect,
    freeze_value,
    thaw_value,
)
from ..providers import (
    BoundSource,
    ConfiguredSource,
    ProviderManifest,
    ProviderRegistration,
    ProviderRuntime,
)
from . import bridge_duplex as _duplex
from .bridge_duplex import (
    BridgeProtocolError,
    BridgeTransportLost,
    DuplexBridgeCall,
    DuplexBridgeSession,
    DuplexControl,
)

BridgeCall = Callable[[str, str], Any]


class ProviderService:
    """Trusted half of a bridge for one already-bound configured source.

    The transport publishes immutable provider metadata but deliberately does
    not publish effects or host policy. Those remain host-owned inputs on the
    receiving side and cannot be spoofed by a transport annotation.
    """

    def __init__(self, source: BoundSource) -> None:
        if not isinstance(source, BoundSource):
            raise TypeError("provider service requires a BoundSource")
        self._source = source

    def close(self) -> None:
        """Release the bound runtime owned by this trusted service."""

        self._source.close()

    def describe(self) -> dict[str, Any]:
        return {
            "source_id": self._source.source.source_id,
            "manifest": self._source.registration.manifest.to_dict(),
            "source_description": self._source.runtime.source_description,
        }

    def handle(self, method: str, params_json: str) -> str:
        """Dispatch only ``describe`` or a manifest-declared operation."""

        try:
            result = self._dispatch(method, params_json)
            return json.dumps({"ok": True, "result": result})
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )

    def handle_duplex(
        self,
        method: str,
        params_json: str,
        control: DuplexControl,
    ) -> str:
        """Dispatch one explicit bridge-v2 invocation with live control/progress."""

        try:
            if method != "invoke":
                raise ValueError(f"unknown duplex bridge method: {method!r}")
            if not callable(control):
                raise TypeError("duplex bridge requires a control channel")
            payload = json.loads(params_json) if params_json else {}
            if not isinstance(payload, dict):
                raise ValueError("duplex bridge params must be a JSON object")
            result = self._dispatch_invoke(payload, control=control)
            return json.dumps(
                {
                    "ok": True,
                    "result": _duplex.deliver_terminal(result, control),
                }
            )
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )

    def _dispatch(self, method: str, params_json: str) -> Any:
        payload = json.loads(params_json) if params_json else {}
        if not isinstance(payload, dict):
            raise ValueError("bridge params must be a JSON object")
        if method == "describe":
            if payload:
                raise ValueError("describe takes no parameters")
            return self.describe()
        if method != "invoke":
            raise ValueError(f"unknown bridge method: {method!r}")

        return self._dispatch_invoke(payload)

    def _dispatch_invoke(
        self,
        payload: Mapping[str, Any],
        *,
        control: DuplexControl | None = None,
    ) -> dict[str, Any]:
        expected = {"operation_id", "args", "kwargs", "execution"}
        if control is not None:
            expected.add("bridge_protocol")
        if set(payload) != expected:
            raise ValueError("bridge invoke requires operation_id, args, kwargs, and execution")
        if (
            control is not None
            and payload.get("bridge_protocol") != _duplex.BRIDGE_PROTOCOL_VERSION
        ):
            raise BridgeProtocolError("duplex invoke requires bridge_protocol 2")
        operation_id = payload.get("operation_id")
        args = payload.get("args", [])
        kwargs = payload.get("kwargs", {})
        execution_payload = payload.get("execution")
        if not isinstance(operation_id, str):
            raise ValueError("bridge invoke requires operation_id")
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise ValueError("bridge invoke requires an args list and kwargs object")
        if not isinstance(execution_payload, Mapping):
            raise ValueError("bridge invoke requires an execution object")
        handler = self._source.runtime.handlers.get(operation_id)
        if handler is None:
            raise PermissionError(f"operation {operation_id!r} is not in the provider manifest")
        if control is None:
            execution, checkpoint = _execution_context(execution_payload)

            def last_seq() -> int:
                return 0
        else:
            local, local_checkpoint = _execution_context(execution_payload)
            execution, checkpoint, last_seq = _duplex.execution_context(
                local, local_checkpoint, control
            )
        try:
            execution.check()
            result = handler(execution, *args, **kwargs)
        except Exception as exc:
            if isinstance(exc, CapabilityCancelled):
                code = CapabilityErrorCode.CANCELLED
            elif isinstance(exc, CapabilityDeadlineExceeded):
                code = CapabilityErrorCode.DEADLINE_EXCEEDED
            elif isinstance(exc, BridgeTransportLost):
                code = _duplex.TRANSPORT_LOST
            elif isinstance(exc, BridgeProtocolError):
                code = _duplex.PROTOCOL_ERROR
            else:
                code = CapabilityErrorCode.HANDLER_ERROR
            response = {
                "kind": "capability_outcome",
                "value": CapabilityOutcome(
                    error=CapabilityError(
                        code,
                        _exception_type(exc),
                        str(exc),
                    )
                ).to_dict(),
                "checkpoint": checkpoint().to_dict(),
            }
            return (
                _duplex.duplex_terminal(response, execution.call_id, last_seq())
                if control
                else response
            )
        if isinstance(result, CapabilityOutcome):
            try:
                value = result.to_dict()
            except (TypeError, ValueError) as exc:
                value = CapabilityOutcome(
                    error=CapabilityError(
                        CapabilityErrorCode.INVALID_RESULT,
                        _exception_type(exc),
                        str(exc),
                    ),
                    metadata=result.metadata,
                ).to_dict()
            response = {
                "kind": "capability_outcome",
                "value": value,
                "checkpoint": checkpoint().to_dict(),
            }
            return (
                _duplex.duplex_terminal(response, execution.call_id, last_seq())
                if control
                else response
            )
        try:
            portable = thaw_value(freeze_value(result))
        except (TypeError, ValueError) as exc:
            response = {
                "kind": "capability_outcome",
                "value": CapabilityOutcome(
                    error=CapabilityError(
                        CapabilityErrorCode.INVALID_RESULT,
                        _exception_type(exc),
                        str(exc),
                    )
                ).to_dict(),
                "checkpoint": checkpoint().to_dict(),
            }
            return (
                _duplex.duplex_terminal(response, execution.call_id, last_seq())
                if control
                else response
            )
        response = {
            "kind": "value",
            "value": portable,
            "checkpoint": checkpoint().to_dict(),
        }
        return (
            _duplex.duplex_terminal(response, execution.call_id, last_seq())
            if control
            else response
        )


class BridgeProvider:
    """Receiving half that projects one remote manifest into a registration.

    Construction performs the ``describe`` call immediately. Pyodide callers
    whose bridge is async must therefore construct this value from a
    ``runPythonAsync``/JSPI-capable stack, as the bundled host adapter does.
    """

    def __init__(
        self,
        bridge_call: BridgeCall,
        *,
        duplex_call: DuplexBridgeCall | None = None,
    ) -> None:
        if not callable(bridge_call):
            raise TypeError("bridge_call must be callable")
        if duplex_call is not None and not callable(duplex_call):
            raise TypeError("duplex_call must be callable or None")
        self._call = bridge_call
        self._duplex_call = duplex_call
        described = self._request("describe")
        if not isinstance(described, dict):
            raise ValueError("bridge describe response must be an object")
        manifest = described.get("manifest")
        if not isinstance(manifest, Mapping):
            raise ValueError("bridge describe response requires a manifest")
        self.manifest = ProviderManifest.from_dict(manifest)
        self.source_id = str(described.get("source_id") or "")
        if not self.source_id:
            raise ValueError("bridge describe response requires source_id")
        self.source_description = str(described.get("source_description") or "")

    def registration(
        self,
        *,
        effects: Mapping[str, SideEffect],
        policy_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> ProviderRegistration:
        """Bind remote handlers using authoritative receiving-host policy."""

        return ProviderRegistration(
            manifest=self.manifest,
            effects=effects,
            binder=self._bind,
            policy_metadata=policy_metadata or {},
        )

    def _bind(self, source: ConfiguredSource, context: Any = None) -> ProviderRuntime:
        del context
        if source.source_id != self.source_id:
            raise ValueError(
                f"bridge is bound to source {self.source_id!r}, not {source.source_id!r}"
            )
        return ProviderRuntime(
            handlers={
                operation.operation_id: functools.partial(
                    self._request_operation, operation.operation_id
                )
                for operation in self.manifest.operations
            },
            source_description=self.source_description,
        )

    def _request_operation(
        self,
        operation_id: str,
        execution: CapabilityExecutionContext,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(execution, CapabilityExecutionContext):
            raise TypeError("bridge provider requires a CapabilityExecutionContext")
        if self._duplex_call is not None:
            return self._request_operation_duplex(operation_id, execution, args, kwargs)
        result = self._request(
            "invoke",
            operation_id=operation_id,
            args=list(args),
            kwargs=kwargs,
            execution=execution.to_transport_dict(),
        )
        if not isinstance(result, Mapping):
            raise RuntimeError("bridge invoke response must be an object")
        raw_checkpoint = result.get("checkpoint")
        if not isinstance(raw_checkpoint, Mapping):
            raise RuntimeError("bridge invoke response requires a checkpoint")
        checkpoint = CapabilityCheckpoint.from_dict(raw_checkpoint)
        execution.checkpoint(tokens=checkpoint.tokens, subcalls=checkpoint.subcalls)
        kind = result.get("kind")
        value = result.get("value")
        if kind == "value":
            return value
        if kind == "capability_outcome" and isinstance(value, Mapping):
            return CapabilityOutcome.from_dict(value)
        raise RuntimeError("bridge invoke response has an unknown result kind")

    def _request_operation_duplex(
        self,
        operation_id: str,
        execution: CapabilityExecutionContext,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        assert self._duplex_call is not None
        return _duplex.request_operation(
            self._duplex_call,
            operation_id,
            execution,
            args,
            kwargs,
        )

    def _request(self, method: str, **payload: Any) -> Any:
        raw = _duplex.await_value(self._call(method, json.dumps(payload)))
        try:
            envelope = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"bridge call {method!r} returned a non-JSON response: {raw!r}"
            ) from exc
        if not isinstance(envelope, dict) or not envelope.get("ok"):
            error = envelope.get("error") if isinstance(envelope, dict) else None
            error = error if isinstance(error, dict) else {}
            raise RuntimeError(
                f"{error.get('type', 'BridgeError')}: "
                f"{error.get('message', 'unknown bridge error')}"
            )
        return envelope.get("result")


__all__ = [
    "BridgeCall",
    "BridgeProtocolError",
    "BridgeProvider",
    "BridgeTransportLost",
    "DuplexBridgeCall",
    "DuplexBridgeSession",
    "DuplexControl",
    "ProviderService",
]


def _exception_type(exc: Exception) -> str:
    """Return a capability-safe exception class name for the typed envelope."""

    name = type(exc).__name__
    return name if name.isidentifier() and name.isascii() else "Exception"


def _execution_context(
    value: Mapping[str, Any],
) -> tuple[CapabilityExecutionContext, Callable[[], CapabilityCheckpoint]]:
    """Reconstruct portable invocation facts with a local cumulative collector."""

    expected = {
        "version",
        "call_id",
        "run_id",
        "parent_run_id",
        "deadline_remaining_ms",
        "reservation",
        "cancellation_requested",
    }
    if set(value) != expected or value.get("version") != 1:
        raise ValueError("bridge execution requires the exact version 1 fields")
    reservation_value = value.get("reservation")
    if not isinstance(reservation_value, Mapping):
        raise ValueError("bridge execution reservation must be an object")
    reservation = CapabilityReservation.from_dict(reservation_value)
    remaining = value.get("deadline_remaining_ms")
    if remaining is not None and (
        isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 0
    ):
        raise ValueError("bridge execution deadline_remaining_ms must be non-negative or null")
    cancelled = value.get("cancellation_requested")
    if not isinstance(cancelled, bool):
        raise ValueError("bridge execution cancellation_requested must be a bool")
    cancellation = Event()
    if cancelled:
        cancellation.set()
    lock = Lock()
    cumulative = [CapabilityCheckpoint()]

    def check() -> None:
        if cancellation.is_set():
            raise CapabilityCancelled("capability call was cancelled")
        if deadline is not None and monotonic() >= deadline:
            raise CapabilityDeadlineExceeded("capability call deadline exceeded")

    def checkpoint(next_value: CapabilityCheckpoint) -> CapabilityCheckpoint:
        check()
        with lock:
            previous = cumulative[0]
            if next_value.tokens < previous.tokens or next_value.subcalls < previous.subcalls:
                raise ValueError("bridge checkpoint cannot move backward")
            if next_value.tokens > reservation.tokens:
                raise ValueError("bridge checkpoint tokens exceed reservation")
            if next_value.subcalls > reservation.subcalls:
                raise ValueError("bridge checkpoint subcalls exceed reservation")
            cumulative[0] = next_value
            return next_value

    deadline = monotonic() + remaining / 1000 if remaining is not None else None
    context = CapabilityExecutionContext(
        call_id=value.get("call_id"),  # type: ignore[arg-type]
        run_id=value.get("run_id"),  # type: ignore[arg-type]
        parent_run_id=value.get("parent_run_id"),  # type: ignore[arg-type]
        deadline_monotonic=deadline,
        reservation=reservation,
        _check=check,
        _checkpoint=checkpoint,
        _is_cancelled=cancellation.is_set,
    )

    def current() -> CapabilityCheckpoint:
        with lock:
            return cumulative[0]

    return context, current
