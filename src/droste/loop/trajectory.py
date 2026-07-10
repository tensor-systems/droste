from __future__ import annotations

from dataclasses import dataclass


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
