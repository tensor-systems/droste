from __future__ import annotations

from dataclasses import dataclass

EXECUTION_STATUS_SUCCESS = "success"
EXECUTION_STATUS_ERROR = "error"


@dataclass
class IterationRecord:
    """Record of a single iteration."""

    iteration: int
    # The message list sent to the root LLM, snapshotted at record time —
    # structured data, not a repr string; boundaries that need text serialize
    # it themselves (the runner emits it as a JSON string on the wire).
    llm_input: list[dict[str, str]]
    llm_output: str
    code_executed: str
    execution_result: str
    tokens_used: int
    # Additive structured status: execution_result remains the exact feedback
    # text for compatibility and must never be parsed to recover this state.
    execution_status: str = EXECUTION_STATUS_SUCCESS
