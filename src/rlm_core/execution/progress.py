from __future__ import annotations

import json
import sys
from typing import Callable


ProgressCallback = Callable[[str], None]


def emit_progress(status: str) -> None:
    """Emit a structured progress event to stderr."""
    event = {"type": "progress", "status": status}
    print(json.dumps(event, ensure_ascii=True), file=sys.stderr, flush=True)
