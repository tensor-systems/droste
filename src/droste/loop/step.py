"""The per-iteration core of the RLM loop.

One step = policy pre-check -> execute -> output budget -> event emit ->
answer refresh -> ready-time policy checks. ``run_rlm`` calls the same
``execute_step`` for a first attempt and for repaired code, so the two paths
cannot drift. Message building, iteration recording, and ``RLMResult``
assembly are pure functions, unit-testable without an LLM.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from ..exceptions import BatchLLMError, PolicyError, RLMError, SandboxError
from ..execution.config import (
    DEFAULT_MAX_CALLS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_OUTPUT_CHARS,
)
from ..execution.context import ExecutionContext
from ..execution.progress import execution_error_event, output_event
from ..policy import PolicyHints, contract_violations, ready_violations
from ..protocols.environment import ExecutionResult, RLMEnvironment
from ..protocols.llm_client import LLMClient, total_tokens_from_usage
from .trajectory import EXECUTION_STATUS_ERROR, EXECUTION_STATUS_SUCCESS, IterationRecord

# Fed back instead of an empty string when executed code prints nothing, so the
# model learns that only stdout is visible.
EMPTY_OUTPUT_NUDGE = "(no output - did you forget to print?)"

# Structured answer metadata crosses process and language boundaries as JSON.
MAX_ANSWER_METADATA_BYTES = 64 * 1024
MAX_ANSWER_METADATA_NODES = 10_000
MAX_ANSWER_METADATA_DEPTH = 100
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1


@dataclass
class RLMConfig:
    """Configuration for RLM execution."""

    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_depth: int = DEFAULT_MAX_DEPTH
    max_calls: int = DEFAULT_MAX_CALLS
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    tips_profile: str = "full"
    verbose: bool = False
    root_model: str | None = None
    on_progress: Any | None = None
    enforce_contract: bool = True
    policy_hints: PolicyHints | None = None


@dataclass
class RLMResult:
    """Result from RLM execution."""

    answer: str
    ready: bool
    iterations: int
    tokens_used: int
    sub_calls_made: int
    trajectory: list[IterationRecord]
    error: RLMError | None = None
    # True when the answer came from the post-exhaustion extract pass rather
    # than the model marking answer['ready'] — hosts may surface it as a
    # best-effort answer but must not present it as confirmed.
    extracted: bool = False
    # Set when max_iterations was exhausted, extraction was attempted, and it
    # failed (or returned empty) — `answer` is then raw loop output (the last
    # printed stdout), NOT prose a host should present verbatim as an answer.
    # None whenever extraction wasn't attempted or succeeded (extracted=True).
    extract_error: RLMError | None = None
    # The terminal step error superseded by a successful extract fallback.
    # This is diagnostic provenance, not a fatal run error: hosts should
    # present the answer as best-effort using `extracted`, while telemetry and
    # benchmarks can still distinguish policy, budget, and execution recovery.
    recovered_error: RLMError | None = None
    # Append-only result surface: keep positional construction of every older
    # field stable while exposing successful semantic evidence separately.
    sub_calls_succeeded: int = 0
    # Domain-neutral structured output submitted alongside a confirmed answer.
    # The loop validates and defensively copies this JSON object. Unconfirmed
    # and extracted answers carry no metadata because the text-only extraction
    # pass cannot verify that a partial metadata draft still supports its output.
    answer_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepOutcome:
    """Result of executing one code block against the environment."""

    output: str
    answer: dict[str, Any]
    error: RLMError | None = None
    # The original exception behind ``error`` — repair-message building needs
    # its type (PolicyError feedback differs from plain execution errors).
    exception: Exception | None = None
    answer_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def execution_status(self) -> str:
        """Structured execution state derived from the typed outcome, not stdout."""
        return EXECUTION_STATUS_ERROR if self.error is not None else EXECUTION_STATUS_SUCCESS


def _execution_output(result: ExecutionResult | str) -> str:
    if isinstance(result, ExecutionResult):
        return result.stdout
    return str(result)


def _feedback_output(output: str) -> str:
    """What the model sees as the execution result. Empty stdout becomes a
    nudge instead of silence; the raw output is kept separately so an empty
    run never leaks the nudge text into `_best_answer`."""
    return output if output else EMPTY_OUTPUT_NUDGE


def _error_feedback(exec_error: Exception) -> str:
    if isinstance(exec_error, PolicyError):
        return (
            f"{exec_error}\n\n"
            "Your accumulated answer['content'] was kept, but answer['ready'] was "
            "reset. Address the policy violation above, then set "
            'answer["ready"] = True again.'
        )
    return f"That code failed with this error:\n{exec_error}\n\nFix the code and try again."


def _resolved_output(answer: dict[str, Any], last_output: str) -> str:
    if answer.get("content"):
        return str(answer.get("content", ""))
    return last_output


def _best_answer(
    answer: dict[str, Any],
    last_output: str,
    last_response: str,
    last_execution_status: str | None,
) -> str:
    if answer.get("content"):
        return str(answer.get("content", ""))
    if last_output and last_execution_status == EXECUTION_STATUS_SUCCESS:
        return last_output
    if last_response:
        return last_response
    return ""


def _validate_json_value(
    value: Any,
    *,
    path: str,
    ancestors: set[int],
    nodes: list[int],
    bytes_seen: list[int],
    depth: int,
) -> None:
    """Validate the strict JSON value subset accepted on public result surfaces."""
    nodes[0] += 1
    if nodes[0] > MAX_ANSWER_METADATA_NODES:
        raise ValueError(f"answer['metadata'] exceeds the {MAX_ANSWER_METADATA_NODES}-node limit")
    if depth > MAX_ANSWER_METADATA_DEPTH:
        raise ValueError(
            f"answer['metadata'] exceeds the {MAX_ANSWER_METADATA_DEPTH}-level depth limit"
        )

    def add_bytes(count: int) -> None:
        bytes_seen[0] += count
        if bytes_seen[0] > MAX_ANSWER_METADATA_BYTES:
            raise ValueError(
                f"answer['metadata'] exceeds the {MAX_ANSWER_METADATA_BYTES}-byte limit"
            )

    def add_json_string(value: str) -> None:
        # Every character occupies at least one UTF-8 byte. Reject huge strings
        # in O(1) before the exact escape-aware scan below.
        if len(value) + 2 > MAX_ANSWER_METADATA_BYTES - bytes_seen[0]:
            add_bytes(MAX_ANSWER_METADATA_BYTES + 1)
        add_bytes(_json_string_byte_size(value))

    if value is None:
        add_bytes(4)
        return
    if type(value) is bool:
        add_bytes(4 if value else 5)
        return
    if type(value) is str:
        add_json_string(value)
        return
    if type(value) is int:
        if abs(value) > MAX_SAFE_JSON_INTEGER:
            raise ValueError(f"{path} contains an integer outside the JSON safe range")
        add_bytes(len(str(value)))
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        add_bytes(len(json.dumps(value)))
        return
    if type(value) not in (dict, list):
        raise ValueError(f"{path} contains unsupported type {type(value).__name__}")

    identity = id(value)
    if identity in ancestors:
        raise ValueError(f"{path} contains a reference cycle")
    add_bytes(2 + max(0, len(value) - 1) + (len(value) if type(value) is dict else 0))
    ancestors.add(identity)
    try:
        if type(value) is list:
            for index, item in enumerate(value):
                _validate_json_value(
                    item,
                    path=f"{path}[{index}]",
                    ancestors=ancestors,
                    nodes=nodes,
                    bytes_seen=bytes_seen,
                    depth=depth + 1,
                )
            return
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} contains a non-string object key")
            add_json_string(key)
            _validate_json_value(
                item,
                path=f"{path}.{key}",
                ancestors=ancestors,
                nodes=nodes,
                bytes_seen=bytes_seen,
                depth=depth + 1,
            )
    finally:
        ancestors.remove(identity)


def _json_string_byte_size(value: str) -> int:
    """Exact UTF-8 byte size of ``json.dumps(value, ensure_ascii=False)``.

    Counting without materializing an encoded copy lets validation reject a
    huge sandbox string before whole-object serialization begins.
    """
    size = 2  # surrounding quotes
    short_escapes = {8, 9, 10, 12, 13}
    for char in value:
        codepoint = ord(char)
        if char in ('"', "\\") or codepoint in short_escapes:
            size += 2
        elif codepoint < 0x20:
            size += 6
        elif 0xD800 <= codepoint <= 0xDFFF:
            raise ValueError("answer['metadata'] contains an unpaired Unicode surrogate")
        elif codepoint <= 0x7F:
            size += 1
        elif codepoint <= 0x7FF:
            size += 2
        elif codepoint <= 0xFFFF:
            size += 3
        else:
            size += 4
    return size


def copy_answer_metadata(answer: dict[str, Any]) -> dict[str, Any]:
    """Return a validated, detached copy of ``answer['metadata']``.

    A reserved object prevents arbitrary sandbox working state from leaking
    into host responses, while the byte limit bounds runner envelopes.
    """
    raw = answer.get("metadata")
    if raw is None:
        return {}
    if type(raw) is not dict:
        raise ValueError("answer['metadata'] must be a JSON object")
    try:
        _validate_json_value(
            raw,
            path="answer['metadata']",
            ancestors=set(),
            nodes=[0],
            bytes_seen=[0],
            depth=0,
        )
        encoded = json.dumps(raw, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (RecursionError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("answer['metadata']"):
            raise
        raise ValueError(f"answer['metadata'] is not valid JSON: {exc}") from exc
    if len(encoded.encode("utf-8")) > MAX_ANSWER_METADATA_BYTES:
        raise ValueError(f"answer['metadata'] exceeds the {MAX_ANSWER_METADATA_BYTES}-byte limit")
    return json.loads(encoded)


def _enforce_output_budget(output: str, max_chars: int) -> None:
    if max_chars <= 0:
        return
    if len(output) > max_chars:
        raise SandboxError(
            f"Sandbox output exceeded {max_chars} characters (attempted {len(output)}). "
            "Summarize or aggregate results instead of printing raw rows."
        )


def _refresh_answer(env_globals: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    """Re-read the sandbox's current `answer` binding after an execution.

    Sandbox code may REBIND `answer` (answer = {...}) instead of mutating it in
    place; the loop must observe the current binding, not the object it
    captured before the loop — otherwise a ready answer is invisible and every
    remaining iteration burns. Non-dict rebinds (answer = "42") are normalized
    back to the dict contract, preserving the value as content but never
    guessing readiness.
    """
    rebound = env_globals.get("answer")
    if rebound is answer:
        return answer
    if isinstance(rebound, dict):
        return rebound
    normalized = {
        "content": "" if rebound is None else str(rebound),
        "ready": False,
    }
    env_globals["answer"] = normalized
    return normalized


def build_initial_messages(system_prompt: str, user_content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_refinement_messages(
    messages: list[dict[str, str]],
    *,
    template: str,
    code: str,
    answer_content: Any,
    last_output: str,
) -> list[dict[str, str]]:
    refinement_content = template.format(
        current_content=answer_content,
        last_output=_feedback_output(last_output),
    )
    return messages + [
        {"role": "assistant", "content": f"```python\n{code}\n```"},
        {"role": "user", "content": refinement_content},
    ]


def build_missing_code_repair_messages(
    messages: list[dict[str, str]], response: str
) -> list[dict[str, str]]:
    return messages + [
        {"role": "assistant", "content": response},
        {
            "role": "user",
            "content": "Your response must include a single ```python code block. Return only code.",
        },
    ]


def build_error_repair_messages(
    messages: list[dict[str, str]], code: str, exec_error: Exception
) -> list[dict[str, str]]:
    return messages + [
        {"role": "assistant", "content": f"```python\n{code}\n```"},
        {"role": "user", "content": _error_feedback(exec_error)},
    ]


def call_root(
    root_llm: LLMClient,
    messages: list[dict[str, str]],
    *,
    model: str,
    context: ExecutionContext,
) -> tuple[str, Any, RLMError | None]:
    """One root-LLM call with token accounting.

    Returns ``(response, usage, None)`` on success or ``("", None, error)`` —
    a root failure always ends the run, so the caller finalizes immediately.
    """
    try:
        response, usage = root_llm.responses_create(messages, model=model, return_usage=True)
    except Exception as exc:
        return "", None, RLMError(type=exc.__class__.__name__, message=str(exc))
    context.stats.total_tokens += total_tokens_from_usage(usage)
    return response, usage, None


def execute_step(
    code: str,
    *,
    iteration: int,
    environment: RLMEnvironment,
    env_globals: dict[str, Any],
    answer: dict[str, Any],
    cfg: RLMConfig,
    context: ExecutionContext,
    data_accessor_names: set[str],
    namespaced_accessor_pairs: set[tuple[str, str]],
) -> StepOutcome:
    """Execute one code block: the identical path for a first attempt and for
    repaired code. Policy pre-check, execute, output budget, event emit,
    answer refresh, ready-time policy checks."""
    try:
        if cfg.enforce_contract:
            violations = contract_violations(
                code,
                cfg.policy_hints,
                data_accessor_names,
                namespaced_accessor_pairs,
            )
            if violations:
                raise PolicyError("Policy violation: " + " | ".join(violations))

        output = environment.execute(code)
        last_output = _execution_output(output)
        # Enforce the output budget BEFORE emitting: an over-budget print
        # must raise here (error path), never stream its full contents to
        # the event channel — the guard gates the event, not just the loop.
        _enforce_output_budget(last_output, cfg.max_output_chars)
        answer = _refresh_answer(env_globals, answer)

        hints = cfg.policy_hints if cfg.enforce_contract else None
        violations = ready_violations(
            hints,
            answer_ready=bool(answer.get("ready")),
            successful_calls=context.stats.successful_calls,
            resolved_output=_resolved_output(answer, last_output),
        )
        answer_metadata: dict[str, Any] = {}
        if answer.get("ready"):
            try:
                answer_metadata = copy_answer_metadata(answer)
            except ValueError as exc:
                violations.append(str(exc))
        if violations:
            # Revoke readiness BEFORE emitting the output event, so the
            # published answer_ready is the post-gate truth — never a state
            # the policy gate is about to reject (codex review). The generic
            # softening in the handler below is then a no-op for this path.
            answer["ready"] = False

        # The output event carries the post-refresh, post-gate iteration
        # state the old verbose prints used to leak on a side channel (#35)
        # — a trace view is now a pure projection of it (render_verbose).
        context.emit_event(
            output_event(
                iteration,
                last_output,
                calls_made=context.stats.calls_made,
                answer_ready=bool(answer.get("ready")),
                answer_content_chars=len(str(answer.get("content") or "")),
            )
        )
        if violations:
            raise PolicyError("Policy violation: " + " | ".join(violations))
    except Exception as exec_error:
        details = None
        if isinstance(exec_error, BatchLLMError):
            details = {"errors": exec_error.errors}
        # Failed code cannot submit a confirmed answer, even if it set ready
        # before raising or rebound the answer dict entirely. Keep accumulated
        # content for repair/extraction, but always revoke readiness. Policy
        # errors use the same state rule and receive specialized feedback from
        # `_error_feedback`.
        answer = _refresh_answer(env_globals, answer)
        answer["ready"] = False
        context.emit_event(
            execution_error_event(iteration, exec_error.__class__.__name__, str(exec_error))
        )
        return StepOutcome(
            output=f"ERROR: {exec_error}",
            answer=answer,
            error=RLMError(
                type=exec_error.__class__.__name__,
                message=str(exec_error),
                code=code,
                details=details,
            ),
            exception=exec_error,
        )

    return StepOutcome(
        output=last_output,
        answer=answer,
        answer_metadata=answer_metadata,
    )


def record_iteration(
    *,
    iteration: int,
    messages: list[dict[str, str]],
    response: str,
    code: str,
    output: str,
    usage: Any,
    execution_status: str = EXECUTION_STATUS_SUCCESS,
) -> IterationRecord:
    """The one place iteration records are built: a structured message-list
    snapshot (deep-copied — the live list keeps growing), nudge-normalized
    execution output."""
    return IterationRecord(
        iteration=iteration,
        llm_input=[dict(message) for message in messages],
        llm_output=response,
        code_executed=code,
        execution_result=_feedback_output(output),
        tokens_used=total_tokens_from_usage(usage),
        execution_status=execution_status,
    )


def finalize(
    *,
    answer_text: str,
    answer: dict[str, Any],
    iterations: int,
    context: ExecutionContext,
    trajectory: list[IterationRecord],
    error: RLMError | None = None,
    extracted: bool = False,
    extract_error: RLMError | None = None,
    recovered_error: RLMError | None = None,
    answer_metadata: dict[str, Any] | None = None,
) -> RLMResult:
    """The one RLMResult construction site."""
    return RLMResult(
        answer=answer_text,
        ready=bool(answer.get("ready")),
        iterations=iterations,
        tokens_used=context.stats.total_tokens,
        sub_calls_made=context.stats.calls_made,
        trajectory=trajectory,
        sub_calls_succeeded=context.stats.successful_calls,
        error=error,
        extracted=extracted,
        extract_error=extract_error,
        recovered_error=recovered_error,
        answer_metadata=answer_metadata if answer.get("ready") and answer_metadata else {},
    )
