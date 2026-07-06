from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IterationRecord:
    """Record of a single iteration."""

    iteration: int
    llm_input: str
    llm_output: str
    code_executed: str
    execution_result: str
    tokens_used: int
