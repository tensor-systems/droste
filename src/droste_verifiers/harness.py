"""Verifiers v1 harness that runs Droste through the interception server."""

from __future__ import annotations

import json
import re
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from packaging.version import InvalidVersion, Version
from pydantic import Field, model_validator
from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import metric
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

RESULT_PATH = ".droste-verifiers-result.json"
_SAFE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*\Z", re.ASCII)


def _droste_version() -> str:
    try:
        return version("droste")
    except PackageNotFoundError:
        return "0.10.6"


def _program_source(droste_version: str) -> str:
    """A pinned PEP 723 program; Verifiers content-addresses and caches it."""

    return f"""# /// script
# requires-python = ">=3.11"
# dependencies = ["droste=={droste_version}"]
# ///
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

from droste_cli.main import main

buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    exit_code = main(sys.argv[1:])
output = buffer.getvalue()
sys.stdout.write(output)
lines = [line for line in output.splitlines() if line.strip()]
if lines:
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        payload = {{"harness_error": "droste did not emit a JSON result"}}
else:
    payload = {{"harness_error": "droste emitted no result"}}
Path({RESULT_PATH!r}).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
raise SystemExit(exit_code)
"""


class DrosteHarnessConfig(HarnessConfig):
    droste_version: str = Field(default_factory=_droste_version)
    """Exact Droste package version installed inside the runtime."""

    depth: Literal[0, 1] = 1
    """Depth 0 disables semantic subcalls; depth 1 enables current flat subcalls."""

    data_paths: list[str] = Field(default_factory=lambda: ["."])
    """Runtime-local files/directories exposed through Droste's symbolic context handle."""

    prompt_profile: str = "full"
    root_revision: str | None = None
    subcall_revision: str | None = None
    max_subcalls: int = Field(50, ge=0)
    token_budget: int = Field(500_000, ge=1)
    wall_ms: int = Field(300_000, ge=1)
    root_output_tokens: int = Field(4_096, ge=1)
    subcall_output_tokens: int = Field(2_048, ge=1)
    max_bytes: int = Field(50_000_000, ge=1)
    max_file_bytes: int = Field(2_000_000, ge=1)
    seed: int | None = None

    @model_validator(mode="after")
    def validate_harness(self) -> "DrosteHarnessConfig":
        if not _SAFE_VERSION.fullmatch(self.droste_version):
            raise ValueError("droste_version must be a safe PEP 440 version token")
        try:
            Version(self.droste_version)
        except InvalidVersion as exc:
            raise ValueError("droste_version must be PEP 440 compatible") from exc
        if not self.data_paths or any(not path for path in self.data_paths):
            raise ValueError("data_paths must contain at least one non-empty runtime path")
        if self.disabled_tools:
            raise ValueError("Droste has a fixed brokered capability surface")
        return self


class DrosteHarness(Harness[DrosteHarnessConfig]):
    """Run one recursive-analysis rollout while Verifiers intercepts every model call."""

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(
            _program_source(self.config.droste_version), self.config.resolved_env
        )

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        if mcp_urls:
            raise ValueError("the Droste harness does not yet project Verifiers MCP tools")
        system, prompt = self.resolve_prompt(trace.task.data)
        if system is not None:
            raise ValueError("the Droste harness cannot emit a separate task system prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("the Droste harness requires one non-empty string task prompt")

        sampling = ctx.sampling.model_dump(mode="json", exclude_none=True)
        rollout = {
            "root_revision": self.config.root_revision,
            "subcall_model": ctx.model,
            "subcall_revision": self.config.subcall_revision,
            "root_sampling": sampling,
            "subcall_sampling": sampling,
            "concurrency": 1,
            "seed": self.config.seed,
            "runner_protocol": None,
            "source_revision": None,
        }
        subcalls = 0 if self.config.depth == 0 else self.config.max_subcalls
        args = [
            f"--model={ctx.model}",
            f"--subcall-model={ctx.model}",
            f"--base-url={endpoint}",
            f"--prompt-profile={self.config.prompt_profile}",
            "--rollout-config=" + json.dumps(rollout, sort_keys=True, separators=(",", ":")),
            f"--budget-tokens={self.config.token_budget}",
            f"--budget-subcalls={subcalls}",
            f"--budget-depth={self.config.depth}",
            f"--budget-wall-ms={self.config.wall_ms}",
            f"--root-output-tokens={self.config.root_output_tokens}",
            f"--subcall-output-tokens={self.config.subcall_output_tokens}",
            f"--max-bytes={self.config.max_bytes}",
            f"--max-file-bytes={self.config.max_file_bytes}",
            "--json",
            "--quiet",
            prompt,
            *self.config.data_paths,
        ]
        program = await runtime.prepare_uv_script(
            _program_source(self.config.droste_version), self.config.resolved_env
        )
        env = {**self.config.resolved_env, "OPENAI_API_KEY": secret}
        return await runtime.run_program(program + args, env)

    @metric
    async def droste(self, trace: Trace, runtime: Runtime) -> dict[str, float]:
        try:
            raw = await runtime.read(RESULT_PATH)
            result = json.loads(raw)
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        fields = {
            "iterations": result.get("iterations"),
            "tokens": result.get("tokens_used"),
            "subcalls": result.get("subcalls"),
            "successful_subcalls": result.get("successful_subcalls"),
            "stdout_chars": result.get("stdout_chars"),
        }
        metrics = {
            f"droste_{name}": float(value)
            for name, value in fields.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        metrics["droste_ready"] = float(result.get("ready") is True)
        metrics["droste_extracted"] = float(result.get("extracted") is True)
        metrics["droste_depth"] = float(self.config.depth)
        return metrics
