from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from .verbs import AccessorManifest


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
    """Abstract REPL environment interface.

    An environment that composes data sources should ALSO implement
    ``AccessorManifestEnvironment`` (optional, checked structurally at
    runtime) so the count contract enforces its actual accessor names.
    """

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


class AccessorManifestEnvironment(Protocol):
    """Optional companion protocol to ``RLMEnvironment`` (#31).

    Kept separate so existing environments type-check unchanged: the loop
    probes for the method at runtime. An environment composing data sources
    should forward its registry's ``accessor_manifest()`` (as
    RunnerEnvironment does) so the count contract's len() check enforces the
    environment's actual accessor names. Without it — or with an empty
    manifest — the policy layer falls back to its static generic verbs,
    which do NOT cover custom accessor names.
    """

    def accessor_manifest(self) -> AccessorManifest:
        """Report the data accessors bound into globals()."""
        ...
