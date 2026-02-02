from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..exceptions import RLMError
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


def _truncate_output(output: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(output) <= max_chars:
        return output
    return output[:max_chars]


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

            try:
                response, usage = root_llm.chat_completion(
                    messages,
                    model=cfg.root_model or "",
                    return_usage=True,
                )
                context.stats.total_tokens += total_tokens_from_usage(usage)
            except Exception as exc:
                error = RLMError(type=exc.__class__.__name__, message=str(exc))
                return RLMResult(
                    answer=str(answer.get("content", "")),
                    ready=bool(answer.get("ready")),
                    iterations=iterations,
                    tokens_used=context.stats.total_tokens,
                    sub_calls_made=context.stats.calls_made,
                    trajectory=trajectory,
                    error=error,
                )

            code = extract_code_block(response, "python")
            if not code:
                error = RLMError(type="MissingCodeBlock", message="No python code block returned")
                return RLMResult(
                    answer=str(answer.get("content", "")),
                    ready=bool(answer.get("ready")),
                    iterations=iterations,
                    tokens_used=context.stats.total_tokens,
                    sub_calls_made=context.stats.calls_made,
                    trajectory=trajectory,
                    error=error,
                )

            context.emit_progress(f"Iteration {iterations}/{cfg.max_iterations}: Executing...")

            try:
                output = environment.execute(code)
                last_output = _truncate_output(_execution_output(output), cfg.max_output_chars)
                error = None
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
                error = RLMError(type=exec_error.__class__.__name__, message=str(exec_error), code=code)

                context.emit_progress(
                    f"Iteration {iterations}/{cfg.max_iterations}: Retrying with error feedback..."
                )
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
                except Exception as repair_exc:
                    error = RLMError(type=repair_exc.__class__.__name__, message=str(repair_exc))
                    return RLMResult(
                        answer=str(answer.get("content", "")),
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
                        output = environment.execute(repaired_code)
                        last_output = _truncate_output(_execution_output(output), cfg.max_output_chars)
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
                        error = RLMError(
                            type=repair_exec_error.__class__.__name__,
                            message=str(repair_exec_error),
                            code=repaired_code,
                        )

        if error is not None:
            return RLMResult(
                answer=str(answer.get("content", "")),
                ready=bool(answer.get("ready")),
                iterations=iterations,
                tokens_used=context.stats.total_tokens,
                sub_calls_made=context.stats.calls_made,
                trajectory=trajectory,
                error=error,
            )

        if not answer.get("ready"):
            error = RLMError(type="AnswerNotReady", message="answer['ready'] was never set")
            return RLMResult(
                answer=str(answer.get("content", "")),
                ready=False,
                iterations=iterations,
                tokens_used=context.stats.total_tokens,
                sub_calls_made=context.stats.calls_made,
                trajectory=trajectory,
                error=error,
            )

        return RLMResult(
            answer=str(answer.get("content", "")),
            ready=True,
            iterations=iterations,
            tokens_used=context.stats.total_tokens,
            sub_calls_made=context.stats.calls_made,
            trajectory=trajectory,
            error=None,
        )
    finally:
        environment.close()
