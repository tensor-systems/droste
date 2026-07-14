"""Generic native in-process RLM environment.

This module lives in droste rather than droste_runner because execution,
registry globals, output bounds, and native signal timeouts are substrate
concerns, not HTTP-runner concerns.
"""

from __future__ import annotations

import contextlib
import io
import json
import signal
from typing import Any

from ..capabilities import (
    BrokeredSubcallClient,
    CapabilityAnnotator,
    CapabilityBroker,
    CapabilityGuard,
    CapabilityObserver,
    subcall_registrations,
)
from ..execution.budget import BudgetLedger
from ..protocols.environment import EnvCapabilities, ExecutionResult, RLMEnvironment
from ..protocols.subcall_client import SubcallClient
from ..protocols.verbs import EMPTY_ACCESSOR_MANIFEST, AccessorManifest
from ..providers import ProviderRegistry
from ..structured import aggregate_json_counts, bind_structured_batch


class OutputBuffer(io.StringIO):
    def __init__(self, max_chars: int) -> None:
        super().__init__()
        self._max_chars = max(0, int(max_chars or 0))
        self._size = 0

    def write(self, text: str) -> int:
        if not text:
            return 0
        if self._max_chars > 0:
            new_size = self._size + len(text)
            if new_size > self._max_chars:
                raise RuntimeError(
                    f"Sandbox output exceeded {self._max_chars} characters (attempted {new_size})."
                )
            self._size = new_size
        return super().write(text)


CONTEXT_PREVIEW_CHARS = 400
CONTEXT_PREVIEW_MAX_FILES = 20


def _safe_preview(text: str, limit: int = CONTEXT_PREVIEW_CHARS) -> str:
    """Head of `text`, truncated and defused so it cannot break its fenced block."""
    head = text[:limit]
    if len(text) > limit:
        head += "..."
    return head.replace("```", "'''")


def _safe_label(text: str, limit: int = 200) -> str:
    """File path/name for prompt inclusion: control chars and newlines
    stripped, then JSON-quoted so attacker-controlled names cannot inject
    prompt instructions outside a fence."""
    cleaned = "".join(ch for ch in text if ch.isprintable())[:limit]
    return json.dumps(cleaned, ensure_ascii=True)


def _describe_files_context(files: list[Any]) -> str:
    lines = [f"`context` is a dict with {len(files)} file(s) in context['files']:"]
    total_text = 0
    for entry in files[:CONTEXT_PREVIEW_MAX_FILES]:
        if not isinstance(entry, dict):
            lines.append(f"- (non-dict entry of type {type(entry).__name__})")
            continue
        path = _safe_label(str(entry.get("path") or entry.get("name") or "(unnamed)"))
        text = entry.get("text")
        text_len = len(text) if isinstance(text, str) else 0
        total_text += text_len
        lines.append(f"- {path} (text: {text_len:,} chars)")
    if len(files) > CONTEXT_PREVIEW_MAX_FILES:
        lines.append(f"- ... and {len(files) - CONTEXT_PREVIEW_MAX_FILES} more file(s)")
        for entry in files[CONTEXT_PREVIEW_MAX_FILES:]:
            text = entry.get("text") if isinstance(entry, dict) else None
            if isinstance(text, str):
                total_text += len(text)
    lines.append(f"Total attached text: {total_text:,} characters.")
    return "\n".join(lines)


def describe_context(context: Any) -> str:
    """Describe the `context` variable for the system prompt: type, total size,
    and a short escaped head preview. Dict-of-files contexts get a
    shape summary (file count, per-file path + text length) instead of a raw
    dump."""
    if context is None:
        return "`context` is None (no context payload was provided)."
    if isinstance(context, str):
        return (
            f"`context` is a str of {len(context):,} characters. "
            f"Preview (first {CONTEXT_PREVIEW_CHARS} chars):\n"
            f"```\n{_safe_preview(context)}\n```"
        )
    if isinstance(context, dict) and isinstance(context.get("files"), list):
        return _describe_files_context(context["files"])
    try:
        serialized = json.dumps(context, ensure_ascii=True, default=str)
    except Exception:
        serialized = str(context)
    shape = f"a {type(context).__name__}"
    if isinstance(context, (list, tuple)):
        shape += f" of {len(context)} item(s)"
    elif isinstance(context, dict):
        shape += f" with {len(context)} key(s)"
    return (
        f"`context` is {shape}, {len(serialized):,} characters when JSON-serialized. "
        f"Preview (first {CONTEXT_PREVIEW_CHARS} chars):\n"
        f"```\n{_safe_preview(serialized)}\n```"
    )


class RunnerEnvironment(RLMEnvironment):
    def __init__(
        self,
        *,
        context: Any,
        registry: ProviderRegistry | None,
        subcalls: SubcallClient,
        max_output_chars: int,
        exec_timeout_ms: int,
        budget_ledger: BudgetLedger | None = None,
        capability_run_id: str | None = None,
        capability_parent_run_id: str | None = None,
        capability_guard: CapabilityGuard | None = None,
        capability_annotator: CapabilityAnnotator | None = None,
        capability_observer: CapabilityObserver | None = None,
    ) -> None:
        self._context = context
        self._registry = registry
        self._subcalls = subcalls
        self._max_output_chars = max_output_chars
        self._exec_timeout_ms = exec_timeout_ms
        self._budget_ledger = budget_ledger
        registrations = list(subcall_registrations(subcalls))
        if registry is not None:
            registrations.extend(registry.capability_registrations())
        self._broker = CapabilityBroker(
            tuple(registrations),
            run_id=capability_run_id,
            parent_run_id=capability_parent_run_id,
            guard=capability_guard,
            annotator=capability_annotator,
            observer=capability_observer,
        )
        self._sandbox_subcalls = BrokeredSubcallClient(
            self._broker,
            metadata_source=subcalls,
        )
        llm_query = self._sandbox_subcalls.llm_query
        llm_batch = self._sandbox_subcalls.llm_batch
        self._globals: dict[str, Any] = {
            "answer": {"content": "", "ready": False},
            "context": context,
            "llm_query": llm_query,
            "llm_batch": llm_batch,
            "batch_llm_query": llm_batch,
            "llm_query_batched": llm_batch,
        }
        structured_batch = bind_structured_batch(self._sandbox_subcalls)
        self._globals["llm_batch_json"] = structured_batch
        self._globals["llm_query_batched_json"] = structured_batch
        self._globals["aggregate_json_counts"] = aggregate_json_counts
        if registry is not None:
            # Namespaced (e.g. db.query / vault.search) + default-flattened globals.
            self._globals.update(registry.broker_globals(self._broker))

    def capability_broker(self) -> CapabilityBroker:
        """Trusted host access to the manifest and typed result envelopes."""

        return self._broker

    def sandbox_subcalls(self, subcalls: SubcallClient, ledger: BudgetLedger) -> SubcallClient:
        """Broker-backed client used by loop-installed structured helpers."""

        if self._budget_ledger is not None and ledger is not self._budget_ledger:
            raise ValueError("run_rlm must use the BudgetLedger brokered by RunnerEnvironment")
        if subcalls is not self._subcalls:
            raise ValueError(
                "run_rlm subcalls must be the same client brokered by RunnerEnvironment"
            )
        return self._sandbox_subcalls

    def capabilities(self) -> EnvCapabilities:
        return {
            "tools_in_root": False,
            "max_output_chars": self._max_output_chars,
        }

    def globals(self) -> dict[str, Any]:
        return self._globals

    def accessor_manifest(self) -> AccessorManifest:
        # Explicit accessor inventory for the count contract's len() check —
        # the loop reads this instead of sniffing markers out of globals().
        if self._registry is None:
            return EMPTY_ACCESSOR_MANIFEST
        return self._registry.accessor_manifest()

    def prompt_fragment(self) -> str:
        parts: list[str] = []
        parts.append(
            "Context is available in a Python variable named `context`. "
            "If it contains files, expect context['files'] entries with path, name, mime, size, and optional text."
        )
        # Size + preview signal: without it the model reasonably
        # assumes the context fits in its own window and prints/counts in
        # Python instead of subcalling. Showing the variable's type,
        # length, and a short preview is the cue that keeps it subcalling.
        description = describe_context(self._context)
        if description:
            parts.append(description)
        parts.append(
            "Each llm_query / llm_query_batched subcall can handle roughly ~100k tokens; "
            "size chunks accordingly."
        )
        if self._registry is not None:
            fragment = self._registry.prompt_fragment()
            if fragment:
                parts.append(fragment)
        return "\n".join(parts)

    def execute(self, code: str) -> ExecutionResult:
        stdout_buf = OutputBuffer(self._max_output_chars)
        stderr_buf = io.StringIO()
        timed_out = False
        exit_code = 0

        def _handle_timeout(signum: int, frame: Any) -> None:
            raise TimeoutError("execution timed out")

        # Native in-process execution uses SIGALRM. New Pyodide/WASM hosts are
        # selected through create_environment and use PyodideEnvironment, but
        # keep the availability guard for compatibility with older hosts that
        # still construct RunnerEnvironment directly under Pyodide.
        use_signal_timeout = bool(
            self._exec_timeout_ms and self._exec_timeout_ms > 0 and hasattr(signal, "setitimer")
        )
        old_handler = None
        if use_signal_timeout:
            old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, self._exec_timeout_ms / 1000.0)

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(compile(code, "<rlm>", "exec"), self._globals)
        except TimeoutError:
            timed_out = True
            exit_code = 124
            raise
        finally:
            if use_signal_timeout:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

        return ExecutionResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            timed_out=timed_out,
            exit_code=exit_code,
            files_written=[],
        )

    def close(self) -> None:
        return
