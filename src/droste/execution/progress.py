"""The structured RLM event vocabulary, its typed builders, and sinks (#35).

One vocabulary, one channel: every event the engine emits is built here and
validated against ``EVENT_TYPES`` at emission. The relay's forwarding filter
(``droste/substrates/_relay/events.ts``) carries the same set, pinned in
lockstep by a parity test — an event type added on either side without the
other fails a test instead of being silently dropped by the filter.

Emission is opt-in: the core loop performs no I/O of its own. Entry points
attach sinks explicitly — ``droste_runner`` writes NDJSON to its original
stderr, while Pyodide writes to its original stderr for the Deno relay to
validate and forward onto the dedicated host event descriptor. The CLI's
``--trace`` renders events through the pure ``render_verbose`` projection.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable

from .trace import parse_event

ProgressCallback = Callable[[str], None]
# Structured loop events (iteration_start / code / output / …) for "watch it
# think" UIs and NDJSON streaming (#1). ProgressCallback remains a compatibility
# view; progress also travels through this one structured event stream.
EventCallback = Callable[[dict[str, Any]], None]

# The engine + relay event vocabulary. The relay-side copy lives in
# droste/substrates/_relay/events.ts (its stderr forwarding filter); the two
# sets are asserted equal by tests/test_event_vocabulary.py.
EVENT_TYPES = frozenset(
    {
        "startup",  # relay handshake {engine_version, runner_protocol, provider_protocol}
        "progress",  # coarse human-readable status {status}
        "iteration_start",  # {iteration, remaining_tokens}
        "llm_response",  # {iteration, response} — the root model's full reply
        "code",  # {iteration, code} — the code that is about to execute
        "output",  # {iteration, stdout, calls_made, answer_ready, answer_content_chars}
        "execution_error",  # {iteration, error_type, message} — a step failed; repair may follow
        "reasoning_delta",  # relay-side {text}, from streamed /responses
        "subcall",  # broker-correlated subcall lifecycle facts
        "repair",  # discriminated repair lifecycle facts
        "extract",  # discriminated terminal extraction lifecycle facts
        "result",  # canonical unary-equivalent final result (without trajectory)
        "replay",  # configurable replay input/output details
        "usage",  # durable resolved token/call accounting
        "budget",  # durable configured/consumed budget facts
        "policy",  # durable policy decision facts
        "capability",  # durable broker-owned capability outcome value
        "done",  # durable terminal result mirror
    }
)


# --- typed event builders (pure) --------------------------------------------


def progress_event(status: str) -> dict[str, Any]:
    return {"type": "progress", "status": status}


def iteration_start_event(iteration: int, remaining_tokens: int) -> dict[str, Any]:
    return {
        "type": "iteration_start",
        "iteration": iteration,
        "remaining_tokens": remaining_tokens,
    }


def llm_response_event(iteration: int, response: str) -> dict[str, Any]:
    return {"type": "llm_response", "iteration": iteration, "response": response}


def code_event(iteration: int, code: str) -> dict[str, Any]:
    return {"type": "code", "iteration": iteration, "code": code}


def output_event(
    iteration: int,
    stdout: str,
    *,
    calls_made: int,
    answer_ready: bool,
    answer_content_chars: int,
    stdout_chars: int | None = None,
) -> dict[str, Any]:
    return {
        "type": "output",
        "iteration": iteration,
        "stdout": stdout,
        "calls_made": calls_made,
        "answer_ready": answer_ready,
        "answer_content_chars": answer_content_chars,
        "stdout_chars": len(stdout) if stdout_chars is None else stdout_chars,
    }


def execution_error_event(iteration: int, error_type: str, message: str) -> dict[str, Any]:
    return {
        "type": "execution_error",
        "iteration": iteration,
        "error_type": error_type,
        "message": message,
    }


def repair_event(
    iteration: int,
    kind: str,
    phase: str,
    *,
    error_type: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": "repair",
        "phase": phase,
        "kind": kind,
        "iteration": iteration,
    }
    if error_type is not None or message is not None:
        if error_type is None or message is None:
            raise ValueError("repair errors require both error_type and message")
        value["error"] = {"type": error_type, "message": message}
    return value


def extract_event(
    iteration: int,
    phase: str,
    *,
    error_type: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": "extract",
        "phase": phase,
        "iteration": iteration,
    }
    if error_type is not None or message is not None:
        if error_type is None or message is None:
            raise ValueError("extract errors require both error_type and message")
        value["extract_error"] = {"type": error_type, "message": message}
    return value


# --- sinks -------------------------------------------------------------------


def emit_event(event: dict[str, Any]) -> None:
    """The stderr NDJSON sink: one structured event per line.

    Original Python stderr is the ingress on the Pyodide substrate; the Deno
    relay validates these frames and forwards them to its dedicated host event
    descriptor. The native subprocess runner uses original stderr directly.
    Attached EXPLICITLY by entry points; the core loop emits nothing when no
    sink is configured (#35).
    """
    # Model-authored code executes under a sandbox stderr redirect. Lifecycle
    # events may be emitted from inside that execution, so the host event lane
    # must use the process's original stderr rather than becoming sandbox
    # output. Embedders that need another destination inject another sink.
    if sys.__stderr__ is None:
        raise RuntimeError("original stderr is unavailable for the host event lane")
    print(json.dumps(event, ensure_ascii=True), file=sys.__stderr__, flush=True)


def emit_progress(status: str) -> None:
    """The stderr sink for coarse, human-readable progress events."""
    emit_event(progress_event(status))


# --- verbose rendering (pure) -------------------------------------------------


def render_verbose(event: dict[str, Any]) -> str | None:
    """Project one structured event to the human 'full loop trace' line the
    core used to print directly under ``verbose`` — now a pure function a
    shell applies (the CLI's ``--trace`` sink). Returns None for events the
    trace view does not show."""
    event = parse_event(event).as_dict()
    etype = event.get("type")
    if etype == "progress":
        status = str(event.get("status") or "")
        if "Generating code" in status:
            bar = "=" * 60
            return f"\n{bar}\n{status}\n{bar}"
        return f"\n{status}"
    if etype == "llm_response":
        return f"\nLLM Response:\n{event.get('response', '')}"
    if etype == "code":
        return f"\nExecuting code:\n{event.get('code', '')}"
    if etype == "output":
        parts = []
        stdout = str(event.get("stdout") or "")
        if stdout:
            parts.append(f"\nOutput:\n{stdout}")
        parts.append(f"\nSub-calls made: {event.get('calls_made', 0)}")
        parts.append(f"answer['ready'] = {event.get('answer_ready')}")
        content_chars = int(event.get("answer_content_chars") or 0)
        if content_chars:
            parts.append(f"answer['content'] length: {content_chars} chars")
        return "\n".join(parts)
    if etype == "execution_error":
        return f"\nExecution error: {event.get('message', '')}"
    if etype == "repair" and event.get("phase") == "failure":
        error = event.get("error") or {}
        return f"\n{event.get('kind')} repair failed: {error.get('type')}: {error.get('message')}"
    if etype == "extract" and event.get("phase") == "failure":
        error = event.get("extract_error") or {}
        return f"\nExtract fallback failed: {error.get('type')}: {error.get('message')}"
    return None
