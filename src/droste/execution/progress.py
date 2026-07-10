from __future__ import annotations

import json
import sys
from typing import Any, Callable

ProgressCallback = Callable[[str], None]
# Structured loop events (iteration_start / code / output / …) for "watch it
# think" UIs and NDJSON streaming (#1). Separate from ProgressCallback: progress
# is a human-readable status string; an event is a typed dict.
EventCallback = Callable[[dict[str, Any]], None]


def emit_event(event: dict[str, Any]) -> None:
    """Emit one structured event as an NDJSON line on stderr.

    stderr is the event lane on the Pyodide substrate — stdout is reserved for
    the final HostResponse JSON (see pyodide/relay.ts). The native subprocess
    runner uses the same event objects on stdout NDJSON.
    """
    print(json.dumps(event, ensure_ascii=True), file=sys.stderr, flush=True)


def emit_progress(status: str) -> None:
    """Emit a coarse, human-readable progress event to stderr."""
    emit_event({"type": "progress", "status": status})
