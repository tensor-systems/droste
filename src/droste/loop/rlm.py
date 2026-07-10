from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from ..exceptions import BatchLLMError, PolicyError, RLMError, SandboxError
from ..execution.config import (
    DEFAULT_MAX_CALLS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_OUTPUT_CHARS,
)
from ..execution.context import ExecutionContext, create_execution_context
from ..policy import PolicyHints, contract_violations, is_numeric_output
from ..prompts.builder import SystemPromptBuilder
from ..protocols.environment import ExecutionResult, RLMEnvironment
from ..protocols.llm_client import LLMClient, total_tokens_from_usage
from ..protocols.subcall_client import SubcallClient
from .code_extractor import extract_code_block
from .trajectory import IterationRecord

DEFAULT_USER_PROMPT_TEMPLATE = "Question: {question}"

DEFAULT_REFINEMENT_PROMPT_TEMPLATE = """Your current accumulated answer:
```
{current_content}
```

Last execution output:
```
{last_output}
```

Continue refining. When done, set `answer[\"ready\"] = True`."""

# Fed back instead of an empty string when executed code prints nothing, so the
# model learns that only stdout is visible.
EMPTY_OUTPUT_NUDGE = "(no output - did you forget to print?)"

# Compact per-iteration truncation budgets for the extract-fallback trajectory
# summary.
_EXTRACT_CODE_CHARS = 1000
_EXTRACT_OUTPUT_CHARS = 1500
_EXTRACT_SUMMARY_CHARS = 60000

# Sentinel used in extract summaries for iterations that printed nothing; the
# conversational nudge shown to the in-loop model must not read as real output.
_EXTRACT_EMPTY_OUTPUT = "<empty stdout>"

EXTRACT_FALLBACK_SYSTEM_PROMPT = (
    "An iterative Python-REPL session ran out of turns before submitting a final "
    "answer. Below are the question, the draft answer so far, and a compact "
    "trajectory of the code executed and its output. Produce the final answer "
    "using ONLY what the trajectory supports - do not guess, extrapolate, or "
    "fabricate. Reply with the answer text only - no code, no preamble. If the "
    "answer cannot be determined from the trajectory, reply exactly: "
    "unable to determine from the work so far"
)


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


ProgressCallback = Any


def _execution_output(result: ExecutionResult | str) -> str:
    if isinstance(result, ExecutionResult):
        return result.stdout
    return str(result)


def _feedback_output(output: str) -> str:
    """What the model sees as the execution result. Empty stdout becomes a
    nudge instead of silence; the raw output is kept separately so an empty
    run never leaks the nudge text into `_best_answer`."""
    return output if output else EMPTY_OUTPUT_NUDGE


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... (truncated, {len(text):,} chars total)"


def _trajectory_summary(
    question: str,
    draft: str,
    trajectory: list[IterationRecord],
) -> str:
    parts = [f"Question: {question}"]
    if draft:
        parts.append(f"Draft answer so far:\n{_truncate(draft, _EXTRACT_OUTPUT_CHARS)}")
    for entry in trajectory:
        raw_output = entry.execution_result
        if not raw_output or raw_output == EMPTY_OUTPUT_NUDGE:
            output = _EXTRACT_EMPTY_OUTPUT
        else:
            output = _truncate(raw_output, _EXTRACT_OUTPUT_CHARS)
        parts.append(
            f"--- Iteration {entry.iteration} ---\n"
            f"Code:\n{_truncate(entry.code_executed, _EXTRACT_CODE_CHARS)}\n"
            f"Output:\n{output}"
        )
    summary = "\n\n".join(parts)
    if len(summary) > _EXTRACT_SUMMARY_CHARS:
        # Keep the most recent work; late iterations carry the conclusions.
        summary = "(earlier trajectory truncated)\n" + summary[-_EXTRACT_SUMMARY_CHARS:]
    return summary


def _extract_final_answer(
    question: str,
    draft: str,
    trajectory: list[IterationRecord],
    root_llm: LLMClient,
    cfg: "RLMConfig",
    context: ExecutionContext,
) -> tuple[str, RLMError | None]:
    """One extract pass:
    when the loop exhausts max_iterations without answer['ready'], make a
    single root-LLM call over a compact trajectory summary so the run returns
    the best answer learnable from the work done, instead of scraps.

    Returns (text, None) on success. On failure returns ("", RLMError(...)) —
    the caller falls back to the raw-loop-output answer already computed via
    _best_answer, but the error is no longer swallowed: it's surfaced via
    RLMResult.extract_error so a host can tell "extraction ran and produced
    this" apart from "extraction failed, this is raw debug output," instead
    of both cases looking identical."""
    try:
        messages = [
            {"role": "system", "content": EXTRACT_FALLBACK_SYSTEM_PROMPT},
            {"role": "user", "content": _trajectory_summary(question, draft, trajectory)},
        ]
        response, usage = root_llm.responses_create(
            messages,
            model=cfg.root_model or "",
            return_usage=True,
        )
        context.stats.total_tokens += total_tokens_from_usage(usage)
        text = str(response).strip()
        if not text:
            return "", RLMError(type="EmptyExtraction", message="extract call returned empty text")
        return text, None
    except Exception as exc:
        return "", RLMError(type=exc.__class__.__name__, message=str(exc))


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


def _apply_batch_error_guard(subcalls: SubcallClient, env_globals: dict[str, Any]) -> None:
    if hasattr(subcalls, "llm_batch_with_errors"):

        def _wrapped_batch(prompts: list[str], contexts: list[str] | None = None) -> list[str]:
            results, errors = subcalls.llm_batch_with_errors(prompts, contexts)
            if errors:
                raise BatchLLMError(
                    f"batch_llm_query failed for {len(errors)}/{len(prompts)} items",
                    errors,
                )
            return results

        env_globals["llm_batch"] = _wrapped_batch
        env_globals["batch_llm_query"] = _wrapped_batch
        env_globals["llm_query_batched"] = _wrapped_batch


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


def run_rlm(
    question: str,
    *,
    environment: RLMEnvironment,
    root_llm: LLMClient,
    subcalls: SubcallClient,
    config: RLMConfig | None = None,
    system_prompt: str | None = None,
    system_prompt_additions: str | None = None,
    conversation_context: str | None = None,
    user_prompt_template: str | None = None,
    refinement_prompt_template: str | None = None,
    on_progress: ProgressCallback | None = None,
    context: ExecutionContext | None = None,
) -> RLMResult:
    cfg = config or RLMConfig()
    if context is None:
        context = create_execution_context(
            max_depth=cfg.max_depth,
            max_calls=cfg.max_calls,
            max_iterations=cfg.max_iterations,
            max_output_chars=cfg.max_output_chars,
            verbose=cfg.verbose,
            on_progress=on_progress or cfg.on_progress,
        )

    env_globals = environment.globals()
    answer = env_globals.get("answer")
    if not isinstance(answer, dict):
        answer = {"content": "", "ready": False}
        env_globals["answer"] = answer

    env_globals.setdefault("llm_query", subcalls.llm_query)
    env_globals.setdefault("llm_batch", subcalls.llm_batch)
    env_globals.setdefault("batch_llm_query", subcalls.llm_batch)
    # llm_query_batched is the name models primed on RLM conventions reach
    # for first, so the sandbox must answer to it.
    env_globals.setdefault("llm_query_batched", subcalls.llm_batch)
    _apply_batch_error_guard(subcalls, env_globals)

    # The data-accessor names actually bound in this sandbox — flattened
    # default-source verbs plus every namespaced source's verbs, including
    # host-declared extras — so the count contract's len()-over-accessor
    # check enforces against whatever THIS environment exposes (#10), not a
    # hardcoded verb list.
    _llm_names = {"llm_query", "llm_batch", "batch_llm_query", "llm_query_batched"}
    data_accessor_names: set[str] = set()
    for _key, _value in env_globals.items():
        if _key in _llm_names:
            continue
        if callable(_value):
            data_accessor_names.add(_key)
        elif isinstance(_value, SimpleNamespace):
            data_accessor_names.update(k for k, v in vars(_value).items() if callable(v))

    if system_prompt is None:
        prompt_additions = environment.prompt_fragment()
        if system_prompt_additions:
            prompt_additions = (
                f"{prompt_additions}\n\n{system_prompt_additions}"
                if prompt_additions
                else system_prompt_additions
            )
        builder = SystemPromptBuilder().with_tips(cfg.tips_profile).with_additions(prompt_additions)
        system_prompt = builder.build()

    user_prompt_template = user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE
    refinement_prompt_template = refinement_prompt_template or DEFAULT_REFINEMENT_PROMPT_TEMPLATE

    user_content = user_prompt_template.format(question=question)
    if conversation_context:
        user_content = f"{user_content}\n\nConversation Context:\n{conversation_context}"

    trajectory: list[IterationRecord] = []
    iterations = 0
    last_output = ""
    last_response = ""
    error: RLMError | None = None

    messages: list[dict[str, str]] = []
    code = ""

    try:
        while not answer.get("ready") and iterations < cfg.max_iterations:
            iterations += 1
            context.emit_event(
                {
                    "type": "iteration_start",
                    "iteration": iterations,
                    "max_iterations": cfg.max_iterations,
                }
            )

            if iterations == 1:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
            else:
                refinement_content = refinement_prompt_template.format(
                    current_content=answer.get("content", ""),
                    last_output=_feedback_output(last_output),
                )
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                messages.append({"role": "user", "content": refinement_content})

            context.emit_progress(
                f"Iteration {iterations}/{cfg.max_iterations}: Generating code..."
            )
            if cfg.verbose:
                print(f"\n{'=' * 60}", file=sys.stderr)
                print(
                    f"Iteration {iterations}/{cfg.max_iterations}: Generating code...",
                    file=sys.stderr,
                )
                print(f"{'=' * 60}", file=sys.stderr)

            try:
                response, usage = root_llm.responses_create(
                    messages,
                    model=cfg.root_model or "",
                    return_usage=True,
                )
                context.stats.total_tokens += total_tokens_from_usage(usage)
                last_response = response
                if cfg.verbose:
                    print(f"\nLLM Response:\n{response}", file=sys.stderr)
            except Exception as exc:
                error = RLMError(type=exc.__class__.__name__, message=str(exc))
                final_answer = _best_answer(answer, last_output, last_response)
                return RLMResult(
                    answer=final_answer,
                    ready=bool(answer.get("ready")),
                    iterations=iterations,
                    tokens_used=context.stats.total_tokens,
                    sub_calls_made=context.stats.calls_made,
                    trajectory=trajectory,
                    error=error,
                )

            code = extract_code_block(response, "python")
            if not code:
                if cfg.enforce_contract:
                    if cfg.verbose:
                        print(
                            "\nNo code block found, retrying with contract enforcement",
                            file=sys.stderr,
                        )
                    repair_messages = messages + [
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": "Your response must include a single ```python code block. Return only code.",
                        },
                    ]
                    try:
                        repair_response, repair_usage = root_llm.responses_create(
                            repair_messages,
                            model=cfg.root_model or "",
                            return_usage=True,
                        )
                        context.stats.total_tokens += total_tokens_from_usage(repair_usage)
                        last_response = repair_response
                    except Exception as repair_exc:
                        error = RLMError(
                            type=repair_exc.__class__.__name__, message=str(repair_exc)
                        )
                        final_answer = _best_answer(answer, last_output, last_response)
                        return RLMResult(
                            answer=final_answer,
                            ready=bool(answer.get("ready")),
                            iterations=iterations,
                            tokens_used=context.stats.total_tokens,
                            sub_calls_made=context.stats.calls_made,
                            trajectory=trajectory,
                            error=error,
                        )

                    code = extract_code_block(repair_response, "python")
                    if not code:
                        error = RLMError(
                            type="PolicyError",
                            message="Response missing python code block.",
                        )
                        final_answer = _best_answer(answer, last_output, last_response)
                        return RLMResult(
                            answer=final_answer,
                            ready=bool(answer.get("ready")),
                            iterations=iterations,
                            tokens_used=context.stats.total_tokens,
                            sub_calls_made=context.stats.calls_made,
                            trajectory=trajectory,
                            error=error,
                        )
                    response = repair_response
                    messages = repair_messages
                else:
                    if cfg.verbose:
                        print(
                            "\nNo code block found, returning response as answer", file=sys.stderr
                        )
                    final_answer = _best_answer(answer, last_output, last_response) or response
                    return RLMResult(
                        answer=final_answer,
                        ready=bool(answer.get("ready")),
                        iterations=iterations,
                        tokens_used=context.stats.total_tokens,
                        sub_calls_made=context.stats.calls_made,
                        trajectory=trajectory,
                        error=None,
                    )

            context.emit_progress(f"Iteration {iterations}/{cfg.max_iterations}: Executing...")
            context.emit_event({"type": "code", "iteration": iterations, "code": code})
            if cfg.verbose:
                print(f"\nExecuting code:\n{code}", file=sys.stderr)

            try:
                if cfg.enforce_contract:
                    violations = contract_violations(code, cfg.policy_hints, data_accessor_names)
                    if violations:
                        raise PolicyError("Policy violation: " + " | ".join(violations))

                output = environment.execute(code)
                last_output = _execution_output(output)
                # Enforce the output budget BEFORE emitting: an over-budget print
                # must raise here (error path), never stream its full contents to
                # the event channel — the guard gates the event, not just the loop.
                _enforce_output_budget(last_output, cfg.max_output_chars)
                context.emit_event(
                    {"type": "output", "iteration": iterations, "stdout": last_output}
                )
                answer = _refresh_answer(env_globals, answer)
                if (
                    cfg.enforce_contract
                    and cfg.policy_hints is not None
                    and cfg.policy_hints.semantic
                    and answer.get("ready")
                    and context.stats.calls_made == 0
                ):
                    raise PolicyError(
                        "Policy violation: semantic question must call llm_query/batch_llm_query at least once."
                    )
                if (
                    cfg.enforce_contract
                    and cfg.policy_hints is not None
                    and cfg.policy_hints.numeric_output
                    and answer.get("ready")
                ):
                    candidate = _resolved_output(answer, last_output)
                    if not is_numeric_output(candidate):
                        raise PolicyError(
                            "Policy violation: output must be a single number (optionally with %)."
                        )
                error = None
                if cfg.verbose:
                    if last_output and not last_output.startswith("ERROR:"):
                        print(f"\nOutput:\n{last_output}", file=sys.stderr)
                    print(f"\nSub-calls made: {context.stats.calls_made}", file=sys.stderr)
                    print(f"answer['ready'] = {answer.get('ready')}", file=sys.stderr)
                    if answer.get("content"):
                        print(
                            f"answer['content'] length: {len(str(answer.get('content')))} chars",
                            file=sys.stderr,
                        )
                trajectory.append(
                    IterationRecord(
                        iteration=iterations,
                        llm_input=str(messages),
                        llm_output=response,
                        code_executed=code,
                        execution_result=_feedback_output(last_output),
                        tokens_used=total_tokens_from_usage(usage),
                    )
                )
            except Exception as exec_error:
                last_output = f"ERROR: {exec_error}"
                details = None
                if isinstance(exec_error, BatchLLMError):
                    details = {"errors": exec_error.errors}
                if isinstance(exec_error, PolicyError):
                    # Softened: revoke readiness so the gate still
                    # gates, but keep the accumulated content — the violation
                    # is fed back as guidance, not punished with a wiped draft.
                    answer["ready"] = False
                error = RLMError(
                    type=exec_error.__class__.__name__,
                    message=str(exec_error),
                    code=code,
                    details=details,
                )
                if cfg.verbose:
                    print(f"\nExecution error: {exec_error}", file=sys.stderr)

                context.emit_progress(
                    f"Iteration {iterations}/{cfg.max_iterations}: Retrying with error feedback..."
                )
                if cfg.verbose:
                    print("\nRetrying with error feedback...", file=sys.stderr)
                repair_messages = messages + [
                    {"role": "assistant", "content": f"```python\n{code}\n```"},
                    {
                        "role": "user",
                        "content": _error_feedback(exec_error),
                    },
                ]
                try:
                    repair_response, repair_usage = root_llm.responses_create(
                        repair_messages,
                        model=cfg.root_model or "",
                        return_usage=True,
                    )
                    context.stats.total_tokens += total_tokens_from_usage(repair_usage)
                    last_response = repair_response
                except Exception as repair_exc:
                    error = RLMError(type=repair_exc.__class__.__name__, message=str(repair_exc))
                    final_answer = _best_answer(answer, last_output, last_response)
                    return RLMResult(
                        answer=final_answer,
                        ready=bool(answer.get("ready")),
                        iterations=iterations,
                        tokens_used=context.stats.total_tokens,
                        sub_calls_made=context.stats.calls_made,
                        trajectory=trajectory,
                        error=error,
                    )

                repaired_code = extract_code_block(repair_response, "python")
                if repaired_code:
                    context.emit_progress(
                        f"Iteration {iterations}/{cfg.max_iterations}: Executing repaired code..."
                    )
                    # The repaired code is what actually runs this iteration — emit
                    # it too, or event consumers see only the failed first attempt
                    # and miss the code/output that produced the answer.
                    context.emit_event(
                        {"type": "code", "iteration": iterations, "code": repaired_code}
                    )
                    try:
                        if cfg.enforce_contract:
                            violations = contract_violations(
                                repaired_code, cfg.policy_hints, data_accessor_names
                            )
                            if violations:
                                raise PolicyError("Policy violation: " + " | ".join(violations))

                        output = environment.execute(repaired_code)
                        last_output = _execution_output(output)
                        _enforce_output_budget(last_output, cfg.max_output_chars)
                        context.emit_event(
                            {"type": "output", "iteration": iterations, "stdout": last_output}
                        )
                        answer = _refresh_answer(env_globals, answer)
                        if (
                            cfg.enforce_contract
                            and cfg.policy_hints is not None
                            and cfg.policy_hints.semantic
                            and answer.get("ready")
                            and context.stats.calls_made == 0
                        ):
                            raise PolicyError(
                                "Policy violation: semantic question must call llm_query/batch_llm_query at least once."
                            )
                        if (
                            cfg.enforce_contract
                            and cfg.policy_hints is not None
                            and cfg.policy_hints.numeric_output
                            and answer.get("ready")
                        ):
                            candidate = _resolved_output(answer, last_output)
                            if not is_numeric_output(candidate):
                                raise PolicyError(
                                    "Policy violation: output must be a single number (optionally with %)."
                                )
                        error = None
                        code = repaired_code
                        trajectory.append(
                            IterationRecord(
                                iteration=iterations,
                                llm_input=str(repair_messages),
                                llm_output=repair_response,
                                code_executed=repaired_code,
                                execution_result=_feedback_output(last_output),
                                tokens_used=total_tokens_from_usage(repair_usage),
                            )
                        )
                    except Exception as repair_exec_error:
                        last_output = f"ERROR: {repair_exec_error}"
                        details = None
                        if isinstance(repair_exec_error, BatchLLMError):
                            details = {"errors": repair_exec_error.errors}
                        if isinstance(repair_exec_error, PolicyError):
                            # Softened: see the main handler above.
                            answer["ready"] = False
                        error = RLMError(
                            type=repair_exec_error.__class__.__name__,
                            message=str(repair_exec_error),
                            code=repaired_code,
                            details=details,
                        )
                        if cfg.verbose:
                            print(f"\nRepair execution error: {repair_exec_error}", file=sys.stderr)

        # A run that ends with a PolicyError outstanding must not present the
        # gated content as its answer. The draft stays intact in-loop (the
        # model keeps it across repair attempts), but the final result
        # withholds it, surfacing it under error.details for debugging.
        policy_outstanding = error is not None and error.type == "PolicyError"
        if policy_outstanding:
            withheld = str(answer.get("content") or "")
            if withheld:
                details = dict(error.details or {})
                details["withheld_content"] = withheld
                error.details = details
            final_answer = ""
        else:
            final_answer = _best_answer(answer, last_output, last_response)

        # Extract fallback: the loop exhausted its iteration budget
        # without answer['ready']. Reaching here means the root client survived
        # every prior call (root failures return early above), so one more
        # extract call is affordable; a trajectory must exist or there is
        # nothing to extract from.
        was_extracted = False
        extract_error: RLMError | None = None
        if not answer.get("ready") and iterations >= cfg.max_iterations and trajectory:
            context.emit_progress("Max iterations reached: extracting best final answer...")
            draft = "" if policy_outstanding else str(answer.get("content") or "")
            extracted, extract_error = _extract_final_answer(
                question, draft, trajectory, root_llm, cfg, context
            )
            if extracted:
                final_answer = extracted
                was_extracted = True
            elif extract_error is not None:
                # Don't swallow this silently (the bug being fixed here):
                # final_answer stays the raw-loop-output fallback from
                # _best_answer above, but the failure is now visible to hosts
                # via RLMResult.extract_error, not indistinguishable from a
                # clean extraction.
                context.emit_event(
                    {
                        "type": "extract_error",
                        "error_type": extract_error.type,
                        "message": extract_error.message,
                    }
                )
                if cfg.verbose:
                    print(
                        f"\nExtract fallback failed: {extract_error.type}: {extract_error.message}",
                        file=sys.stderr,
                    )

        if not final_answer:
            if error:
                final_answer = f"Error: {error.message}"
            else:
                final_answer = "No output produced."

        if cfg.verbose:
            print(f"\nFinal iterations: {iterations}", file=sys.stderr)
            print(f"answer['ready']: {answer.get('ready')}", file=sys.stderr)

        return RLMResult(
            answer=final_answer,
            ready=bool(answer.get("ready")),
            iterations=iterations,
            tokens_used=context.stats.total_tokens,
            sub_calls_made=context.stats.calls_made,
            trajectory=trajectory,
            error=error,
            extracted=was_extracted,
            extract_error=extract_error,
        )
    finally:
        environment.close()
