from __future__ import annotations

from typing import Any

from ..protocols.environment import EnvCapabilities, ExecutionResult, RLMEnvironment
from ..protocols.verbs import EMPTY_ACCESSOR_MANIFEST, AccessorManifest


class MockEnvironment(RLMEnvironment):
    def __init__(self, globals_dict: dict[str, Any] | None = None) -> None:
        self._globals = globals_dict or {"answer": {"content": "", "ready": False}}
        self._calls: list[str] = []

    def capabilities(self) -> EnvCapabilities:
        return {"tools_in_root": False, "max_output_chars": 25000}

    def globals(self) -> dict[str, Any]:
        return self._globals

    def accessor_manifest(self) -> AccessorManifest:
        return EMPTY_ACCESSOR_MANIFEST

    def prompt_fragment(self) -> str:
        return ""

    def execute(self, code: str) -> ExecutionResult:
        self._calls.append(code)
        exec(code, {}, self._globals)
        return ExecutionResult(stdout="", stderr="", timed_out=False, exit_code=0, files_written=[])

    def close(self) -> None:
        return None

    @property
    def calls(self) -> list[str]:
        return list(self._calls)
