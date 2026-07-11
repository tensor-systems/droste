from __future__ import annotations

from dataclasses import dataclass

from .progress import EventCallback, ProgressCallback

DEFAULT_MAX_OUTPUT_CHARS = 25000
DEFAULT_MAX_DEPTH = 1
# Raised 10 -> 50 and 5 -> 20: an explore-first orchestration
# budget needs room — long-context tasks routinely use 10+ subcalls, and
# 5 iterations forecloses that before it starts.
DEFAULT_MAX_CALLS = 50
DEFAULT_MAX_ITERATIONS = 20


@dataclass(frozen=True)
class ExecutionConfig:
    """Immutable configuration for RLM execution."""

    max_depth: int = DEFAULT_MAX_DEPTH
    max_calls: int = DEFAULT_MAX_CALLS
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    # Legacy switch retained for constructor compatibility; the core no
    # longer reads it (#35) — verbose views are rendered by shells from
    # structured events (droste.execution.progress.render_verbose).
    verbose: bool = False
    on_progress: ProgressCallback | None = None
    # Structured loop events (#1). None means NO emission (#35): entry points
    # attach a sink explicitly (droste_runner/relay: the stderr NDJSON sink
    # droste.execution.progress.emit_event). Independent of on_progress.
    on_event: EventCallback | None = None
