from __future__ import annotations

from dataclasses import dataclass

from .progress import ProgressCallback


DEFAULT_MAX_OUTPUT_CHARS = 25000
DEFAULT_MAX_DEPTH = 1
# Raised 10 -> 50 and 5 -> 20 (issue #21): an explore-first orchestration
# budget needs room — dspy.RLM defaults to 20 iterations / 50 subcalls and
# actually used ~13.5 subcalls/task on OOLONG-131k; 5 iterations forecloses
# that before it starts.
DEFAULT_MAX_CALLS = 50
DEFAULT_MAX_ITERATIONS = 20


@dataclass(frozen=True)
class ExecutionConfig:
    """Immutable configuration for RLM execution."""
    max_depth: int = DEFAULT_MAX_DEPTH
    max_calls: int = DEFAULT_MAX_CALLS
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    verbose: bool = False
    on_progress: ProgressCallback | None = None
