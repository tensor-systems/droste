"""Pure projections shared by the CLI, runner, and terminal Trace ABI."""

from __future__ import annotations

from typing import Any


def error_payload(error: Any, *, include_details: bool = True) -> dict[str, Any] | None:
    if error is None:
        return None
    value: dict[str, Any] = {
        "type": error.type,
        "message": error.message,
    }
    if include_details:
        value["code"] = error.code
        value["details"] = error.details
    return value


def trajectory_payload(result: Any) -> list[dict[str, Any]]:
    return [
        {
            "iteration": entry.iteration,
            "llm_input": [dict(message) for message in entry.llm_input],
            "llm_output": entry.llm_output,
            "code_executed": entry.code_executed,
            "execution_result": entry.execution_result,
            "execution_status": entry.execution_status,
            "attempt_kind": entry.attempt_kind,
            "stdout_chars": entry.stdout_chars,
            "tokens_used": entry.tokens_used,
        }
        for entry in result.trajectory
    ]


def project_result(
    result: Any,
    *,
    include_trajectory: bool = True,
    include_error_details: bool = True,
) -> dict[str, Any]:
    """One result projection shared by completed-run shells."""
    run_record = getattr(result, "run_record", None)
    scaffold_manifest = getattr(result, "scaffold_manifest", None)
    value = {
        "answer": result.answer,
        "answer_metadata": getattr(result, "answer_metadata", {}),
        "ready": result.ready,
        "iterations": result.iterations,
        "tokens_used": result.tokens_used,
        "subcalls": result.sub_calls_made,
        "successful_subcalls": int(getattr(result, "sub_calls_succeeded", 0)),
        "extracted": bool(getattr(result, "extracted", False)),
        "error": error_payload(result.error, include_details=include_error_details),
        "extract_error": error_payload(
            getattr(result, "extract_error", None), include_details=include_error_details
        ),
        "recovered_error": error_payload(
            getattr(result, "recovered_error", None), include_details=include_error_details
        ),
        "prompt_pack": (
            result.prompt_pack.as_dict() if getattr(result, "prompt_pack", None) else None
        ),
        "run_record": run_record.as_dict() if run_record is not None else None,
        "scaffold_manifest": (
            {
                **scaffold_manifest.as_dict(),
                "id": scaffold_manifest.manifest_id,
            }
            if scaffold_manifest is not None
            else None
        ),
        "stdout_chars": int(getattr(result, "stdout_chars", 0)),
    }
    if include_trajectory:
        value["trajectory"] = trajectory_payload(result)
    return value
