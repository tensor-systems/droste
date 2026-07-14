"""Versioned droste_runner request/response envelope helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

RUNNER_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class RootResponseMetadata:
    """Metadata returned with the most recent root HTTP response."""

    provider: str = ""
    response_id: str = ""
    stop_reason: str = ""
    model: str = ""


def _error_payload(error: Any) -> dict[str, Any] | None:
    if error is None:
        return None
    return {
        "type": error.type,
        "message": error.message,
        "code": error.code,
        "details": error.details,
    }


def build_response(
    *,
    result: Any | None = None,
    error: dict[str, Any] | None = None,
    metadata: RootResponseMetadata | None = None,
    requested_model: str = "",
    data_source_requests: int | None = None,
) -> dict[str, Any]:
    """Build both refusal and completed-run envelopes from one field list."""

    response: dict[str, Any] = {
        "answer": "" if result is None else result.answer,
        "ready": False if result is None else result.ready,
        "iterations": 0 if result is None else result.iterations,
        "tokens_used": 0 if result is None else result.tokens_used,
        "subcalls": 0 if result is None else result.sub_calls_made,
        "extracted": False if result is None else bool(getattr(result, "extracted", False)),
        "extract_error": (
            None if result is None else _error_payload(getattr(result, "extract_error", None))
        ),
        "recovered_error": (
            None if result is None else _error_payload(getattr(result, "recovered_error", None))
        ),
        "trajectory": (
            []
            if result is None
            else [
                {
                    "iteration": entry.iteration,
                    "llm_input": json.dumps(entry.llm_input),
                    "llm_output": entry.llm_output,
                    "code_executed": entry.code_executed,
                    "execution_result": entry.execution_result,
                    "execution_status": entry.execution_status,
                    "tokens_used": entry.tokens_used,
                }
                for entry in result.trajectory
            ]
        ),
        "protocol_version": RUNNER_PROTOCOL_VERSION,
        "error": error if result is None else _error_payload(result.error),
    }
    if result is None:
        return response

    root_metadata = metadata or RootResponseMetadata()
    response.update(
        {
            "answer_metadata": getattr(result, "answer_metadata", {}),
            "successful_subcalls": int(getattr(result, "sub_calls_succeeded", 0)),
            "provider": root_metadata.provider,
            "response_id": root_metadata.response_id,
            "stop_reason": root_metadata.stop_reason,
            "model": root_metadata.model or requested_model,
        }
    )
    if data_source_requests is not None:
        response["data_source_requests"] = data_source_requests
    return response


def _protocol_error_response(requested: object, error_type: str) -> dict[str, Any]:
    """The versioned refusal: minimal envelope, structured error, no work done."""
    if error_type == "protocol_version_missing":
        message = (
            "request has no protocol_version; this engine speaks "
            f'{RUNNER_PROTOCOL_VERSION} — add "protocol_version": '
            f"{RUNNER_PROTOCOL_VERSION} to the request"
        )
    else:
        message = (
            f"request speaks runner protocol {requested!r}; "
            f"this engine speaks {RUNNER_PROTOCOL_VERSION}"
        )
    return build_response(
        error={
            "type": error_type,
            "message": message,
            "code": error_type,
            "details": {"requested": requested, "supported": RUNNER_PROTOCOL_VERSION},
        }
    )


def _check_protocol_version(request: dict[str, Any]) -> dict[str, Any] | None:
    """Return a refusal response for a missing/mismatched version, else None."""
    raw = request.get("protocol_version")
    if raw is None or raw == "":
        return _protocol_error_response(None, "protocol_version_missing")
    if isinstance(raw, bool) or not isinstance(raw, int):
        return _protocol_error_response(raw, "protocol_version_mismatch")
    if raw != RUNNER_PROTOCOL_VERSION:
        return _protocol_error_response(raw, "protocol_version_mismatch")
    return None


def build_exception_response(exc: Exception, traceback_text: str) -> dict[str, Any]:
    """Build the version-stamped worker exception envelope."""
    return {
        "protocol_version": RUNNER_PROTOCOL_VERSION,
        "error": {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback_text,
        },
    }
