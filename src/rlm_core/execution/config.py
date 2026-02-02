from __future__ import annotations

from dataclasses import dataclass

from .progress import ProgressCallback


DEFAULT_MAX_OUTPUT_CHARS = 8192
DEFAULT_MAX_DEPTH = 1
DEFAULT_MAX_CALLS = 10
DEFAULT_MAX_ITERATIONS = 5


@dataclass(frozen=True)
class ExecutionConfig:
    """Immutable configuration for RLM execution."""
    max_depth: int = DEFAULT_MAX_DEPTH
    max_calls: int = DEFAULT_MAX_CALLS
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    verbose: bool = False
    on_progress: ProgressCallback | None = None
