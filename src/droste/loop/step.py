"""The per-iteration core of the RLM loop.

One step = policy pre-check -> execute -> output budget -> event emit ->
answer refresh -> ready-time policy checks. ``run_rlm`` calls the same
``execute_step`` for a first attempt and for repaired code, so the two paths
cannot drift. Message building, iteration recording, and ``RLMResult``
assembly are pure functions, unit-testable without an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from .trajectory import IterationRecord

# Fed back instead of an empty string when executed code prints nothing, so the
# model learns that only stdout is visible.
EMPTY_OUTPUT_NUDGE = "(no output - did you forget to print?)"


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


@dataclass
class StepOutcome:
    """Result of executing one code block against the environment."""

    output: str
    answer: dict[str, Any]
    error: RLMError | None = None
    # The original exception behind ``error`` — repair-message building needs
    # its type (PolicyError feedback differs from plain execution errors).
    exception: Exception | None = None


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


def _best_answer(answer: dict[str, Any], last_output: str, last_response: str) -> str:
    if answer.get("content"):
        return str(answer.get("content", ""))
    if last_output and not last_output.startswith("ERROR:"):
        return last_output
    if last_response:
        return last_response
    return ""


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
                code, cfg.policy_hints, data_accessor_names, namespaced_accessor_pairs
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
        # The output event carries the post-refresh iteration state the old
        # verbose prints used to leak on a side channel (#35) — a trace view
        # is now a pure projection of this event (render_verbose).
        context.emit_event(
            output_event(
                iteration,
                last_output,
                calls_made=context.stats.calls_made,
                answer_ready=bool(answer.get("ready")),
                answer_content_chars=len(str(answer.get("content") or "")),
            )
        )

        hints = cfg.policy_hints if cfg.enforce_contract else None
        violations = ready_violations(
            hints,
            answer_ready=bool(answer.get("ready")),
            calls_made=context.stats.calls_made,
            resolved_output=_resolved_output(answer, last_output),
        )
        if violations:
            raise PolicyError("Policy violation: " + " | ".join(violations))
    except Exception as exec_error:
        details = None
        if isinstance(exec_error, BatchLLMError):
            details = {"errors": exec_error.errors}
        if isinstance(exec_error, PolicyError):
            # Softened: revoke readiness so the gate still gates, but keep
            # the accumulated content — the violation is fed back as
            # guidance, not punished with a wiped draft.
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

    return StepOutcome(output=last_output, answer=answer)


def record_iteration(
    *,
    iteration: int,
    messages: list[dict[str, str]],
    response: str,
    code: str,
    output: str,
    usage: Any,
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
) -> RLMResult:
    """The one RLMResult construction site."""
    return RLMResult(
        answer=answer_text,
        ready=bool(answer.get("ready")),
        iterations=iterations,
        tokens_used=context.stats.total_tokens,
        sub_calls_made=context.stats.calls_made,
        trajectory=trajectory,
        error=error,
        extracted=extracted,
        extract_error=extract_error,
    )
