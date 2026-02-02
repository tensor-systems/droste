from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypedDict


class EnvCapabilities(TypedDict):
    """Capabilities and limits for the RLM environment."""
    tools_in_root: bool
    max_output_chars: int


@dataclass
class ExecutionResult:
    """Result from environment execution."""
    stdout: str
    stderr: str
    timed_out: bool
    exit_code: int
    files_written: list[str]


class RLMEnvironment(Protocol):
    """Abstract REPL environment interface."""

    def capabilities(self) -> EnvCapabilities:
        """Return environment capabilities and limits."""
        ...

    def globals(self) -> dict[str, Any]:
        """Return mutable globals dict used for code execution."""
        ...

    def prompt_fragment(self) -> str:
        """Return prompt fragment describing available data sources and functions."""
        ...

    def execute(self, code: str) -> ExecutionResult:
        """Execute code within the environment."""
        ...

    def close(self) -> None:
        """Release environment resources."""
        ...
