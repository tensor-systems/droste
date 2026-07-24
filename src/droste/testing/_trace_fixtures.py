"""Canonical construction for released Trace ABI fixture bytes."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from ..execution.progress import (
    code_event,
    execution_error_event,
    llm_response_event,
    output_event,
)
from ..execution.trace import RunEvent, TraceRecorder


def _ndjson(events: tuple[RunEvent, ...]) -> bytes:
    return b"".join(
        json.dumps(event.as_dict(), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        + b"\n"
        for event in events
    )


def build_trace_v5_execution_ndjson() -> bytes:
    """Build the deterministic code/output/error projection corpus."""

    started_at = datetime(2026, 7, 16, tzinfo=timezone.utc)
    timestamps = iter(started_at + timedelta(seconds=offset) for offset in range(9))
    root = TraceRecorder(run_id="golden-execution-root", clock=lambda: next(timestamps))
    root.append(llm_response_event(1, "```python\nprint('first iteration')\n```"))
    root.append(code_event(1, "print('first iteration')"))
    root.append(
        output_event(
            1,
            "ERROR: ordinary successful stdout\n",
            calls_made=0,
            answer_ready=False,
            answer_content_chars=0,
        )
    )
    root.append(llm_response_event(2, "```python\nraise ValueError('synthetic failure')\n```"))
    root.append(code_event(2, "raise ValueError('synthetic failure')"))
    root.append(execution_error_event(2, "ValueError", "synthetic execution failure"))

    child = TraceRecorder(
        run_id="golden-execution-child",
        parent_run_id=root.run_id,
        depth=1,
        clock=lambda: next(timestamps),
    )
    child.append(llm_response_event(1, "```python\nprint('child run')\n```"))
    child.append(code_event(1, "print('child run')"))
    child.append(
        output_event(
            1,
            "child output\n",
            calls_made=0,
            answer_ready=False,
            answer_content_chars=0,
        )
    )
    return _ndjson((*root.events, *child.events))
