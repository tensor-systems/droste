"""Versioned droste_runner request/response envelope helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from droste.execution.manifest import ScaffoldCompatibilityError
from droste.execution.report import project_result
from droste.loop.rlm import RLMPreflight

RUNNER_PROTOCOL_VERSION = 7


class RunnerOperation(str, Enum):
    """The explicit operation selected at the runner boundary."""

    RUN = "run"
    PREFLIGHT = "preflight"


def resolve_operation(request: dict[str, Any]) -> RunnerOperation:
    """Resolve the additive operation field; absence preserves normal execution."""

    raw = request.get("operation", RunnerOperation.RUN.value)
    if not isinstance(raw, str):
        raise ValueError("request.operation must be 'run' or 'preflight'")
    try:
        return RunnerOperation(raw)
    except ValueError as exc:
        raise ValueError("request.operation must be 'run' or 'preflight'") from exc


@dataclass(frozen=True)
class RootResponseMetadata:
    """Metadata returned with the most recent root HTTP response."""

    provider: str = ""
    response_id: str = ""
    stop_reason: str = ""
    model: str = ""


def build_response(
    *,
    result: Any | None = None,
    error: dict[str, Any] | None = None,
    metadata: RootResponseMetadata | None = None,
    requested_model: str = "",
    data_source_requests: int | None = None,
    status: str | None = None,
    operation: RunnerOperation | None = RunnerOperation.RUN,
) -> dict[str, Any]:
    """Build both refusal and completed-run envelopes from one field list."""
    root_metadata = metadata or RootResponseMetadata()
    response: dict[str, Any] = {
        "answer": "",
        "answer_metadata": {},
        "ready": False,
        "iterations": 0,
        "tokens_used": 0,
        "subcalls": 0,
        "successful_subcalls": 0,
        "extracted": False,
        "error": error,
        "extract_error": None,
        "recovered_error": None,
        "prompt_pack": None,
        "scaffold_manifest": None,
        "stdout_chars": 0,
        "run_record": None,
        "run_id": None,
        "status": status or "error",
        "operation": operation.value if operation is not None else None,
        "protocol_version": RUNNER_PROTOCOL_VERSION,
        "provider": root_metadata.provider,
        "response_id": root_metadata.response_id,
        "stop_reason": root_metadata.stop_reason,
        "model": root_metadata.model or requested_model,
        "data_source_requests": data_source_requests,
    }
    if result is not None:
        response.update(project_result(result, include_trajectory=False))
        record = getattr(result, "run_record", None)
        response["run_id"] = record.run_id if record is not None else None
        response["status"] = (
            str(record.terminal["status"])
            if record is not None
            else (
                status
                or (
                    "success"
                    if result.error is None and (result.ready or result.extracted)
                    else "error"
                )
            )
        )
    return response


def build_preflight_response(
    *,
    result: RLMPreflight | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the content-free, versioned response for scaffold preflight."""

    if (result is None) == (error is None):
        raise ValueError("preflight response requires exactly one of result or error")
    return {
        "protocol_version": RUNNER_PROTOCOL_VERSION,
        "operation": RunnerOperation.PREFLIGHT.value,
        "status": "success" if result is not None else "refusal",
        "preflight": result.as_dict() if result is not None else None,
        "error": error,
    }


def build_scaffold_refusal(exc: ScaffoldCompatibilityError) -> dict[str, Any]:
    """Project the public compatibility error without requiring prose parsing."""

    return build_preflight_response(
        error={
            "type": "ScaffoldCompatibilityError",
            "code": "scaffold_incompatible",
            "message": str(exc),
            "details": {"mismatches": [item.as_dict() for item in exc.mismatches]},
        }
    )


def build_operation_refusal(
    message: str,
    *,
    operation: RunnerOperation | None = None,
    code: str = "operation_invalid",
) -> dict[str, Any]:
    """Reject an invalid operation before request-specific work starts."""

    return {
        "protocol_version": RUNNER_PROTOCOL_VERSION,
        "operation": operation.value if operation is not None else None,
        "status": "refusal",
        "preflight": None,
        "error": {
            "type": "RunnerOperationError",
            "code": code,
            "message": message,
            "details": {"supported": [item.value for item in RunnerOperation]},
        },
    }


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
        status="refusal",
        operation=None,
        error={
            "type": error_type,
            "message": message,
            "code": error_type,
            "details": {"requested": requested, "supported": RUNNER_PROTOCOL_VERSION},
        },
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


def build_exception_response(
    exc: Exception,
    traceback_text: str,
    *,
    operation: RunnerOperation | None = None,
) -> dict[str, Any]:
    """Build the version-stamped worker exception envelope."""
    error = {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": traceback_text,
    }
    if operation is RunnerOperation.PREFLIGHT:
        return {
            "protocol_version": RUNNER_PROTOCOL_VERSION,
            "operation": RunnerOperation.PREFLIGHT.value,
            "status": "error",
            "preflight": None,
            "error": error,
        }
    return build_response(status="error", operation=operation, error=error)
