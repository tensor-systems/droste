from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any

from ..exceptions import BatchLLMError, PolicyError, RLMError, SandboxError
from ..execution.context import ExecutionContext, create_execution_context
from ..execution.config import (
    DEFAULT_MAX_CALLS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_OUTPUT_CHARS,
)
from ..protocols.environment import ExecutionResult, RLMEnvironment
from ..protocols.llm_client import LLMClient, total_tokens_from_usage
from ..protocols.subcall_client import SubcallClient
from ..prompts.builder import SystemPromptBuilder
from ..policy import PolicyHints, contract_violations, is_numeric_output
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


ProgressCallback = Any


def _execution_output(result: ExecutionResult | str) -> str:
    if isinstance(result, ExecutionResult):
        return result.stdout
    return str(result)


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


def _enforce_output_budget(output: str, max_chars: int) -> None:
    if max_chars <= 0:
        return
    if len(output) > max_chars:
        raise SandboxError(
            f"Sandbox output exceeded {max_chars} characters (attempted {len(output)}). "
            "Summarize or aggregate results instead of printing raw rows."
        )


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
    _apply_batch_error_guard(subcalls, env_globals)

    if system_prompt is None:
        prompt_additions = environment.prompt_fragment()
        if system_prompt_additions:
            prompt_additions = f"{prompt_additions}\n\n{system_prompt_additions}" if prompt_additions else system_prompt_additions
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

            if iterations == 1:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
            else:
                refinement_content = refinement_prompt_template.format(
                    current_content=answer.get("content", ""),
                    last_output=last_output,
                )
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                messages.append({"role": "user", "content": refinement_content})

            context.emit_progress(f"Iteration {iterations}/{cfg.max_iterations}: Generating code...")
            if cfg.verbose:
                print(f"\n{'='*60}", file=sys.stderr)
                print(f"Iteration {iterations}/{cfg.max_iterations}: Generating code...", file=sys.stderr)
                print(f"{'='*60}", file=sys.stderr)

            try:
                response, usage = root_llm.chat_completion(
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
                        print("\nNo code block found, retrying with contract enforcement", file=sys.stderr)
                    repair_messages = messages + [
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": "Your response must include a single ```python code block. Return only code.",
                        },
                    ]
                    try:
                        repair_response, repair_usage = root_llm.chat_completion(
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
                        print("\nNo code block found, returning response as answer", file=sys.stderr)
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
            if cfg.verbose:
                print(f"\nExecuting code:\n{code}", file=sys.stderr)

            try:
                if cfg.enforce_contract:
                    violations = contract_violations(code, cfg.policy_hints)
                    if violations:
                        raise PolicyError("Policy violation: " + " | ".join(violations))

                output = environment.execute(code)
                last_output = _execution_output(output)
                _enforce_output_budget(last_output, cfg.max_output_chars)
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
                        execution_result=last_output,
                        tokens_used=total_tokens_from_usage(usage),
                    )
                )
            except Exception as exec_error:
                last_output = f"ERROR: {exec_error}"
                details = None
                if isinstance(exec_error, BatchLLMError):
                    details = {"errors": exec_error.errors}
                if isinstance(exec_error, PolicyError):
                    answer["ready"] = False
                    answer["content"] = ""
                error = RLMError(type=exec_error.__class__.__name__, message=str(exec_error), code=code, details=details)
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
                        "content": f"That code failed with this error:\n{exec_error}\n\nFix the code and try again.",
                    },
                ]
                try:
                    repair_response, repair_usage = root_llm.chat_completion(
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
                    try:
                        if cfg.enforce_contract:
                            violations = contract_violations(repaired_code, cfg.policy_hints)
                            if violations:
                                raise PolicyError("Policy violation: " + " | ".join(violations))

                        output = environment.execute(repaired_code)
                        last_output = _execution_output(output)
                        _enforce_output_budget(last_output, cfg.max_output_chars)
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
                                execution_result=last_output,
                                tokens_used=total_tokens_from_usage(repair_usage),
                            )
                        )
                    except Exception as repair_exec_error:
                        last_output = f"ERROR: {repair_exec_error}"
                        details = None
                        if isinstance(repair_exec_error, BatchLLMError):
                            details = {"errors": repair_exec_error.errors}
                        if isinstance(repair_exec_error, PolicyError):
                            answer["ready"] = False
                            answer["content"] = ""
                        error = RLMError(
                            type=repair_exec_error.__class__.__name__,
                            message=str(repair_exec_error),
                            code=repaired_code,
                            details=details,
                        )
                        if cfg.verbose:
                            print(f"\nRepair execution error: {repair_exec_error}", file=sys.stderr)

        final_answer = _best_answer(answer, last_output, last_response)
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
        )
    finally:
        environment.close()
