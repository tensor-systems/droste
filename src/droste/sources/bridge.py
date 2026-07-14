"""Transport-neutral provider bridge with a fixed dispatch boundary."""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping
from typing import Any

from ..capabilities import (
    CapabilityError,
    CapabilityErrorCode,
    CapabilityOutcome,
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

        operation_id = payload.get("operation_id")
        args = payload.get("args", [])
        kwargs = payload.get("kwargs", {})
        if not isinstance(operation_id, str):
            raise ValueError("bridge invoke requires operation_id")
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise ValueError("bridge invoke requires an args list and kwargs object")
        handler = self._source.runtime.handlers.get(operation_id)
        if handler is None:
            raise PermissionError(f"operation {operation_id!r} is not in the provider manifest")
        try:
            result = handler(*args, **kwargs)
        except Exception as exc:
            return {
                "kind": "capability_outcome",
                "value": CapabilityOutcome(
                    error=CapabilityError(
                        CapabilityErrorCode.HANDLER_ERROR,
                        _exception_type(exc),
                        str(exc),
                    )
                ).to_dict(),
            }
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
            return {"kind": "capability_outcome", "value": value}
        try:
            portable = thaw_value(freeze_value(result))
        except (TypeError, ValueError) as exc:
            return {
                "kind": "capability_outcome",
                "value": CapabilityOutcome(
                    error=CapabilityError(
                        CapabilityErrorCode.INVALID_RESULT,
                        _exception_type(exc),
                        str(exc),
                    )
                ).to_dict(),
            }
        return {"kind": "value", "value": portable}


class BridgeProvider:
    """Receiving half that projects one remote manifest into a registration."""

    def __init__(self, bridge_call: BridgeCall) -> None:
        if not callable(bridge_call):
            raise TypeError("bridge_call must be callable")
        self._call = bridge_call
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

    def _request_operation(self, operation_id: str, *args: Any, **kwargs: Any) -> Any:
        result = self._request("invoke", operation_id=operation_id, args=list(args), kwargs=kwargs)
        if not isinstance(result, Mapping):
            raise RuntimeError("bridge invoke response must be an object")
        kind = result.get("kind")
        value = result.get("value")
        if kind == "value":
            return value
        if kind == "capability_outcome" and isinstance(value, Mapping):
            return CapabilityOutcome.from_dict(value)
        raise RuntimeError("bridge invoke response has an unknown result kind")

    def _request(self, method: str, **payload: Any) -> Any:
        raw = self._call(method, json.dumps(payload))
        if hasattr(raw, "__await__"):
            from pyodide.ffi import run_sync

            raw = run_sync(raw)
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


__all__ = ["BridgeCall", "BridgeProvider", "ProviderService"]


def _exception_type(exc: Exception) -> str:
    """Return a capability-safe exception class name for the typed envelope."""

    name = type(exc).__name__
    return name if name.isidentifier() and name.isascii() else "Exception"
