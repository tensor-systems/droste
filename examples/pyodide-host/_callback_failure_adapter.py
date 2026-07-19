"""E2E probe proving typed callback failures cross the relay as raw JSON."""

from __future__ import annotations

import json
from typing import Any


class ProviderError(RuntimeError):
    """Probe exception used to surface one canonical callback failure."""


def _post(host_fetch: Any, endpoint: str) -> Any:
    raw = host_fetch(
        "POST",
        endpoint,
        json.dumps({"Content-Type": "application/json"}),
        "{}",
    )
    if hasattr(raw, "__await__"):
        from pyodide.ffi import run_sync

        raw = run_sync(raw)
    return json.loads(raw)


def run_for_host_pyodide(
    request: dict[str, Any],
    host_fetch: Any,
    bridge_call: Any = None,
    duplex_bridge_call: Any = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del bridge_call, duplex_bridge_call, meta
    endpoint_key = request.get("callback_endpoint_key", "subcall_endpoint")
    if endpoint_key not in {
        "root_endpoint",
        "subcall_endpoint",
        "subcall_batch_endpoint",
    }:
        raise ValueError("unsupported callback endpoint key")
    endpoint = request[endpoint_key]
    if request.get("recover_transport"):
        try:
            _post(host_fetch, endpoint)
        except Exception:
            pass
        _post(host_fetch, endpoint)
        return {
            "status": "error",
            "error": {
                "type": "RuntimeError",
                "message": "local validation failed after transport recovery",
            },
        }

    body = _post(host_fetch, endpoint)
    if request.get("surface_callback_error"):
        raise ProviderError(str(body.get("message") or "provider failed"))
    return {
        "status": "error",
        "error": {
            "type": "RuntimeError",
            "message": "local validation failed after callback was handled",
        },
        "received_usage": body["usage"],
        "received_late_secret": body["late_secret"],
        "received_unicode_secret": body["unicode_secret"],
    }
