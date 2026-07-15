"""Test-only adapter emitting one large value through the canonical Trace ABI."""

from droste.execution.progress import emit_event
from droste.execution.trace import TraceRecorder


def run_for_host_pyodide(
    request,
    host_fetch,
    bridge_call=None,
    duplex_bridge_call=None,
    meta=None,
):
    event = TraceRecorder(run_id="large-relay-event").append(
        {
            "type": "code",
            "iteration": 1,
            "code": "x" * 1_048_576,
        }
    )
    emit_event(event.as_dict())
    return {"answer": "ok", "error": None}
