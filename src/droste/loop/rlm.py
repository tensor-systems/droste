from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..exceptions import PolicyError, RLMError, SubcallBudgetExceeded
from ..execution.context import ExecutionContext, create_execution_context
from ..execution.progress import (
    EventCallback,
    extract_error_event,
    finalization_error_event,
    iteration_start_event,
    llm_response_event,
)
from ..execution.progress import (
    code_event as build_code_event,
)
from ..prompts.pack import (
    CODE_OUTPUT_CONTRACT,
    DEFAULT_PROMPT_PROFILE,
    EXTRACT_OUTPUT_CONTRACT,
    PromptPack,
    PromptPackCatalog,
    PromptSlots,
    load_builtin_prompt_catalog,
    render_prompt_template,
    render_system_prompt,
    resolve_prompt_pack,
)
from ..protocols.environment import RLMEnvironment
from ..protocols.llm_client import LLMClient, total_tokens_from_usage
from ..protocols.subcall_client import SubcallClient
from ..protocols.verbs import EMPTY_ACCESSOR_MANIFEST, AccessorManifest
from ..structured import _StructuredBatchEvidence, aggregate_json_counts, bind_structured_batch
from .code_extractor import extract_code_block
from .step import (
    EMPTY_OUTPUT_NUDGE,
    RLMConfig,
    RLMResult,
    _best_answer,
    build_error_repair_messages,
    build_initial_messages,
    build_missing_code_repair_messages,
    build_refinement_messages,
    call_root,
    error_repair_history,
    execute_step,
    finalize,
    record_iteration,
)
from .trajectory import ExecutionStatus, IterationRecord

__all__ = ["RLMConfig", "RLMResult", "run_rlm", "EMPTY_OUTPUT_NUDGE"]

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

# Compact per-iteration truncation budgets for the extract-fallback trajectory
# summary.
_EXTRACT_CODE_CHARS = 1000
_EXTRACT_OUTPUT_CHARS = 1500
_EXTRACT_SUMMARY_CHARS = 60000

# Sentinel used in extract summaries for iterations that printed nothing; the
# conversational nudge shown to the in-loop model must not read as real output.
_EXTRACT_EMPTY_OUTPUT = "<empty stdout>"
_EXTRACT_UNABLE = "unable to determine from the work so far"


ProgressCallback = Any


class _SubcallGate:
    """Run-scoped subcall bindings that can be revoked before delegation."""

    def __init__(self, subcalls: SubcallClient) -> None:
        self._subcalls = subcalls
        self._blocked = False

    def _check(self) -> None:
        if self._blocked:
            raise SubcallBudgetExceeded("subcalls are disabled during terminal finalization")

    def llm_query(self, prompt: str, context: str = "") -> str:
        self._check()
        return self._subcalls.llm_query(prompt, context)

    def llm_batch(self, prompts: list[str], contexts: list[str] | None = None) -> list[str]:
        self._check()
        return self._subcalls.llm_batch(prompts, contexts)

    def llm_batch_with_errors(
        self,
        prompts: list[str],
        contexts: list[str] | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        self._check()
        batch_with_errors = getattr(self._subcalls, "llm_batch_with_errors", None)
        if callable(batch_with_errors):
            return batch_with_errors(prompts, contexts)
        return self._subcalls.llm_batch(prompts, contexts), []

    def block(self) -> None:
        """Permanently revoke this run's model-visible subcall bindings."""
        self._blocked = True


def _accessor_manifest(environment: RLMEnvironment) -> AccessorManifest:
    """Data-accessor names for the count contract's len() check (#10).

    Explicit data, not sniffing: an environment that composes data sources
    (e.g. one wrapping a DataSourceRegistry) reports them via an optional
    ``accessor_manifest()`` method. Environments without one yield an empty
    manifest, and the policy layer falls back to its static generic verbs."""
    manifest_fn = getattr(environment, "accessor_manifest", None)
    if callable(manifest_fn):
        manifest = manifest_fn()
        if isinstance(manifest, AccessorManifest):
            return manifest
    return EMPTY_ACCESSOR_MANIFEST


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... (truncated, {len(text):,} chars total)"


def _is_unable_extraction(text: str, sentinel: str = _EXTRACT_UNABLE) -> bool:
    """Recognize the model's no-evidence sentinel with harmless decoration."""
    decoration = "\"'`“”‘’*_"

    def normalize(value: str) -> str:
        normalized = value.casefold().strip().strip(decoration)
        return normalized.rstrip(".!?…").rstrip().strip(decoration)

    return normalize(text) == normalize(sentinel)


def _budget_prompt(cfg: "RLMConfig") -> str:
    return (
        "## Authorized compute\n"
        f"iterations={cfg.max_iterations}; subcalls={cfg.max_calls}; "
        f"depth={cfg.max_depth}; output_chars_per_iteration={cfg.max_output_chars}"
    )


def _refinement_history(answer_content: Any, last_output: str) -> str:
    return (
        "Current accumulated answer:\n"
        f"```\n{answer_content}\n```\n\n"
        "Last execution output:\n"
        f"```\n{last_output if last_output else EMPTY_OUTPUT_NUDGE}\n```"
    )


def _trajectory_summary(
    draft: str,
    trajectory: list[IterationRecord],
) -> str:
    parts: list[str] = []
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
            f"Status: {entry.execution_status}\n"
            f"Code:\n{_truncate(entry.code_executed, _EXTRACT_CODE_CHARS)}\n"
            f"Output:\n{output}"
        )
    summary = "\n\n".join(parts)
    if len(summary) > _EXTRACT_SUMMARY_CHARS:
        # Keep the most recent work; late iterations carry the conclusions.
        summary = "(earlier trajectory truncated)\n" + summary[-_EXTRACT_SUMMARY_CHARS:]
    return summary


def _has_extractable_work(answer: dict[str, Any], has_successful_step: bool) -> bool:
    """Whether partial work contains evidence worth a terminal extract call.

    Failed attempts are retained for provenance, but a trajectory made only of
    errors is not evidence. Extraction may recover a retained draft or inspect
    any successfully executed step (including one with empty stdout whose code
    accumulated useful intermediate state).
    """
    if str(answer.get("content") or "").strip():
        return True
    return has_successful_step


def _terminal_semantic_budget_error(
    evidence: _StructuredBatchEvidence | None,
    context: ExecutionContext,
) -> RLMError | None:
    """Return a fail-closed policy error when exact recovery is impossible."""
    if evidence is None or context.max_calls < 0:
        return None
    required = evidence.minimum_exact_retry_calls
    remaining = max(context.max_calls - context.stats.calls_made, 0)
    if required <= remaining:
        return None
    return RLMError(
        type="PolicyError",
        message=(
            "Policy violation: incomplete structured semantic batch evidence "
            "cannot be cleared within the subcall budget: clearing all "
            f"{evidence.unresolved_batches} unresolved recorded exact request(s) "
            f"requires at least {required} call(s), but only {remaining} remain."
        ),
        details={
            "reason": "semantic_exact_retry_budget_exhausted",
            "required_subcalls": required,
            "remaining_subcalls": remaining,
            "unresolved_batches": evidence.unresolved_batches,
            "unresolved_items": evidence.unresolved_items,
        },
    )


def _extract_final_answer(
    question: str,
    draft: str,
    trajectory: list[IterationRecord],
    root_llm: LLMClient,
    cfg: "RLMConfig",
    context: ExecutionContext,
    prompt_pack: PromptPack,
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
        history = (
            _trajectory_summary(draft, trajectory)
            + f"\n\nUnable sentinel: {prompt_pack.unable_sentinel}"
        )
        slots = PromptSlots(
            question=question,
            history=history,
            output_contract=(
                f"{EXTRACT_OUTPUT_CONTRACT} If the answer is unsupported, reply exactly: "
                f"{prompt_pack.unable_sentinel}"
            ),
        )
        messages = [
            {
                "role": "system",
                "content": render_prompt_template(prompt_pack.templates.extract_system, slots),
            },
            {
                "role": "user",
                "content": render_prompt_template(prompt_pack.templates.extract_user, slots),
            },
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
        if _is_unable_extraction(text, prompt_pack.unable_sentinel):
            return "", RLMError(type="InsufficientEvidence", message=text)
        return text, None
    except Exception as exc:
        return "", RLMError(type=exc.__class__.__name__, message=str(exc))


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
    # All params after * are keyword-only, so placement is API-neutral;
    # appended last anyway to keep the signature append-only.
    on_event: EventCallback | None = None,
    prompt_pack: PromptPack | None = None,
    consumer_prompt_catalog: PromptPackCatalog | None = None,
) -> RLMResult:
    cfg = config or RLMConfig()
    requested_profile = cfg.prompt_profile or cfg.tips_profile or DEFAULT_PROMPT_PROFILE
    resolved_prompt_pack = resolve_prompt_pack(
        model=cfg.root_model or "",
        profile=requested_profile,
        caller_pack=prompt_pack,
        consumer_catalog=consumer_prompt_catalog,
        engine_catalog=load_builtin_prompt_catalog(),
    )
    if cfg.enforce_contract is None:
        cfg = replace(
            cfg,
            enforce_contract=resolved_prompt_pack.pack.policy_defaults.enforce_contract,
        )
    prompt_pack_record = resolved_prompt_pack.record()
    if context is None:
        context = create_execution_context(
            max_depth=cfg.max_depth,
            max_calls=cfg.max_calls,
            max_iterations=cfg.max_iterations,
            max_output_chars=cfg.max_output_chars,
            verbose=cfg.verbose,
            on_progress=on_progress or cfg.on_progress,
            # None -> NO emission (#35). Embedders that want the NDJSON
            # stderr stream attach droste.execution.progress.emit_event.
            on_event=on_event,
        )
    bind_context = getattr(subcalls, "bind_context", None)
    if callable(bind_context):
        bind_context(context)

    env_globals = environment.globals()
    # The environment contract owns one brokered subcall surface. The host's
    # raw client remains trusted loop state for context binding/accounting.
    sandbox_subcalls = environment.sandbox_subcalls(subcalls)
    answer = env_globals.get("answer")
    if not isinstance(answer, dict):
        answer = {"content": "", "ready": False}
        env_globals["answer"] = answer

    semantic_evidence = (
        _StructuredBatchEvidence()
        if cfg.enforce_contract and cfg.policy_hints is not None and cfg.policy_hints.semantic
        else None
    )
    # The semantic finalization gate wraps the brokered adapter, never the raw
    # trusted client, so revocable saved bindings cannot become an egress bypass.
    subcall_gate = _SubcallGate(sandbox_subcalls)
    model_subcalls: SubcallClient = (
        subcall_gate if semantic_evidence is not None else sandbox_subcalls
    )
    env_globals["llm_query"] = model_subcalls.llm_query
    env_globals["llm_batch"] = model_subcalls.llm_batch
    env_globals["batch_llm_query"] = model_subcalls.llm_batch
    # llm_query_batched is the name models primed on RLM conventions reach
    # for first, so the sandbox must answer to it.
    env_globals["llm_query_batched"] = model_subcalls.llm_batch

    structured_batch = bind_structured_batch(model_subcalls, semantic_evidence)
    # These two aliases are one core helper, so bind Droste's version even when
    # an environment pre-populated a custom helper during setup. Semantic runs
    # use the evidence-tracked form selected above.
    env_globals["llm_batch_json"] = structured_batch
    env_globals["llm_query_batched_json"] = structured_batch
    env_globals["aggregate_json_counts"] = aggregate_json_counts

    manifest = _accessor_manifest(environment)
    data_accessor_names = set(manifest.flat)
    namespaced_accessor_pairs = set(manifest.namespaced)

    if system_prompt is None:
        prompt_additions = environment.prompt_fragment()
        if system_prompt_additions:
            prompt_additions = (
                f"{prompt_additions}\n\n{system_prompt_additions}"
                if prompt_additions
                else system_prompt_additions
            )
        system_prompt = render_system_prompt(
            resolved_prompt_pack.pack,
            PromptSlots(
                capabilities=prompt_additions,
                budget=_budget_prompt(cfg),
                question=question,
                output_contract=CODE_OUTPUT_CONTRACT,
            ),
        )

    if user_prompt_template is not None:
        user_content = user_prompt_template.format(question=question)
        if conversation_context:
            user_content = f"{user_content}\n\nConversation Context:\n{conversation_context}"
    else:
        user_content = render_prompt_template(
            resolved_prompt_pack.pack.templates.user,
            PromptSlots(
                question=question,
                history=(
                    f"Conversation Context:\n{conversation_context}" if conversation_context else ""
                ),
                output_contract=CODE_OUTPUT_CONTRACT,
            ),
        )

    trajectory: list[IterationRecord] = []
    has_successful_step = False
    iterations = 0
    last_output = ""
    last_response = ""
    last_execution_status: ExecutionStatus | None = None
    error: RLMError | None = None
    answer_metadata: dict[str, Any] = {}
    terminal_budget_handoff = False
    finalization_base_messages: list[dict[str, str]] = []
    finalization_base_code = ""

    messages: list[dict[str, str]] = []
    code = ""

    def step_kwargs() -> dict[str, Any]:
        return dict(
            iteration=iterations,
            environment=environment,
            env_globals=env_globals,
            answer=answer,
            cfg=cfg,
            context=context,
            data_accessor_names=data_accessor_names,
            namespaced_accessor_pairs=namespaced_accessor_pairs,
            semantic_evidence=semantic_evidence,
        )

    def early_result(run_error: RLMError | None) -> RLMResult:
        return finalize(
            answer_text=_best_answer(answer, last_output, last_response, last_execution_status),
            answer=answer,
            iterations=iterations,
            context=context,
            trajectory=trajectory,
            error=run_error,
            answer_metadata=answer_metadata,
            prompt_pack=prompt_pack_record,
        )

    try:
        while not answer.get("ready") and iterations < cfg.max_iterations:
            iterations += 1
            context.emit_event(iteration_start_event(iterations, cfg.max_iterations))

            if iterations == 1:
                messages = build_initial_messages(system_prompt, user_content)
            else:
                rendered_refinement = None
                if refinement_prompt_template is None:
                    rendered_refinement = render_prompt_template(
                        resolved_prompt_pack.pack.templates.refinement,
                        PromptSlots(
                            question=question,
                            history=_refinement_history(answer.get("content", ""), last_output),
                            output_contract=CODE_OUTPUT_CONTRACT,
                        ),
                    )
                messages = build_refinement_messages(
                    messages,
                    template=(refinement_prompt_template or DEFAULT_REFINEMENT_PROMPT_TEMPLATE),
                    code=code,
                    answer_content=answer.get("content", ""),
                    last_output=last_output,
                    rendered_prompt=rendered_refinement,
                )

            context.emit_progress(
                f"Iteration {iterations}/{cfg.max_iterations}: Generating code..."
            )

            response, usage, root_error = call_root(
                root_llm, messages, model=cfg.root_model or "", context=context
            )
            if root_error is not None:
                return early_result(root_error)
            last_response = response
            context.emit_event(llm_response_event(iterations, response))

            code = extract_code_block(response, "python")
            if not code:
                if cfg.enforce_contract:
                    context.emit_progress("No code block found, retrying with contract enforcement")
                    missing_code_prompt = render_prompt_template(
                        resolved_prompt_pack.pack.templates.missing_code_repair,
                        PromptSlots(output_contract=CODE_OUTPUT_CONTRACT),
                    )
                    repair_messages = build_missing_code_repair_messages(
                        messages, response, repair_prompt=missing_code_prompt
                    )
                    repair_response, repair_usage, root_error = call_root(
                        root_llm, repair_messages, model=cfg.root_model or "", context=context
                    )
                    if root_error is not None:
                        return early_result(root_error)
                    last_response = repair_response
                    context.emit_event(llm_response_event(iterations, repair_response))

                    code = extract_code_block(repair_response, "python")
                    if not code:
                        return early_result(
                            RLMError(
                                type="PolicyError",
                                message="Response missing python code block.",
                            )
                        )
                    response = repair_response
                    messages = repair_messages
                    usage = repair_usage
                else:
                    context.emit_progress("No code block found, returning response as answer")
                    final_answer = _best_answer(
                        answer, last_output, last_response, last_execution_status
                    )
                    return finalize(
                        answer_text=final_answer,
                        answer=answer,
                        iterations=iterations,
                        context=context,
                        trajectory=trajectory,
                        error=None,
                        answer_metadata=answer_metadata,
                        prompt_pack=prompt_pack_record,
                    )

            context.emit_progress(f"Iteration {iterations}/{cfg.max_iterations}: Executing...")
            context.emit_event(build_code_event(iterations, code))

            finalization_base_messages = messages
            finalization_base_code = code
            outcome = execute_step(code, **step_kwargs())
            answer = outcome.answer
            last_output = outcome.output
            last_execution_status = outcome.execution_status
            error = outcome.error
            answer_metadata = outcome.answer_metadata
            if outcome.error is None:
                has_successful_step = True
                trajectory.append(
                    record_iteration(
                        iteration=iterations,
                        messages=messages,
                        response=response,
                        code=code,
                        outcome=outcome,
                        usage=usage,
                    )
                )
                terminal_error = _terminal_semantic_budget_error(semantic_evidence, context)
                if terminal_error is not None:
                    error = terminal_error
                    terminal_budget_handoff = True
                    break
                continue

            failed_record = record_iteration(
                iteration=iterations,
                messages=messages,
                response=response,
                code=code,
                outcome=outcome,
                usage=usage,
            )

            terminal_error = _terminal_semantic_budget_error(semantic_evidence, context)
            if terminal_error is not None:
                trajectory.append(failed_record)
                error = terminal_error
                terminal_budget_handoff = True
                break

            context.emit_progress(
                f"Iteration {iterations}/{cfg.max_iterations}: Retrying with error feedback..."
            )
            assert outcome.exception is not None
            error_repair_prompt = render_prompt_template(
                resolved_prompt_pack.pack.templates.error_repair,
                PromptSlots(
                    question=question,
                    history=error_repair_history(outcome.exception),
                    output_contract=CODE_OUTPUT_CONTRACT,
                ),
            )
            repair_messages = build_error_repair_messages(
                messages,
                code,
                outcome.exception,
                repair_prompt=error_repair_prompt,
            )
            repair_response, repair_usage, root_error = call_root(
                root_llm, repair_messages, model=cfg.root_model or "", context=context
            )
            if root_error is not None:
                trajectory.append(failed_record)
                return early_result(root_error)
            last_response = repair_response
            context.emit_event(llm_response_event(iterations, repair_response))

            repaired_code = extract_code_block(repair_response, "python")
            if repaired_code:
                context.emit_progress(
                    f"Iteration {iterations}/{cfg.max_iterations}: Executing repaired code..."
                )
                # The repaired code is what actually runs this iteration — emit
                # it too, or event consumers see only the failed first attempt
                # and miss the code/output that produced the answer.
                context.emit_event(build_code_event(iterations, repaired_code))
                finalization_base_messages = repair_messages
                finalization_base_code = repaired_code
                outcome = execute_step(repaired_code, **step_kwargs())
                answer = outcome.answer
                last_output = outcome.output
                last_execution_status = outcome.execution_status
                error = outcome.error
                answer_metadata = outcome.answer_metadata
                if outcome.error is None:
                    code = repaired_code
                    has_successful_step = True
                else:
                    # Keep the attempt that produced the retained draft as
                    # well as the failed repair that ended the iteration.
                    trajectory.append(failed_record)
                trajectory.append(
                    record_iteration(
                        iteration=iterations,
                        messages=repair_messages,
                        response=repair_response,
                        code=repaired_code,
                        outcome=outcome,
                        usage=repair_usage,
                    )
                )
            else:
                trajectory.append(failed_record)

            terminal_error = _terminal_semantic_budget_error(semantic_evidence, context)
            if terminal_error is not None:
                error = terminal_error
                terminal_budget_handoff = True
                break

        # A fail-closed budget handoff can strand useful values in the
        # persistent REPL even when no answer draft was retained. Give that
        # state one root-generated code attempt to populate answer['content'].
        # There is deliberately no missing-code or execution repair, and every
        # model-visible subcall callable (including saved aliases) is revoked
        # before the code runs.
        if terminal_budget_handoff:
            subcall_gate.block()

        if (
            terminal_budget_handoff
            and error is not None
            and not str(answer.get("content") or "").strip()
        ):
            context.emit_progress(
                "Exact semantic retry cannot fit: finalizing from persistent state..."
            )
            terminal_history = f"type={error.type}\nmessage={error.message}"
            finalization_prompt = render_prompt_template(
                resolved_prompt_pack.pack.templates.error_repair,
                PromptSlots(
                    question=question,
                    history=terminal_history,
                    output_contract=CODE_OUTPUT_CONTRACT,
                ),
            )
            finalization_messages = build_error_repair_messages(
                finalization_base_messages,
                finalization_base_code,
                PolicyError(error.message),
                repair_prompt=finalization_prompt,
            )
            finalization_response, finalization_usage, finalization_root_error = call_root(
                root_llm,
                finalization_messages,
                model=cfg.root_model or "",
                context=context,
            )
            if finalization_root_error is None:
                last_response = finalization_response
                context.emit_event(llm_response_event(iterations, finalization_response))
                finalization_code = extract_code_block(finalization_response, "python")
                if finalization_code:
                    context.emit_event(build_code_event(iterations, finalization_code))
                    finalization_outcome = execute_step(finalization_code, **step_kwargs())
                    answer = finalization_outcome.answer
                    last_output = finalization_outcome.output
                    last_execution_status = finalization_outcome.execution_status
                    answer_metadata = finalization_outcome.answer_metadata
                    trajectory.append(
                        record_iteration(
                            iteration=iterations,
                            messages=finalization_messages,
                            response=finalization_response,
                            code=finalization_code,
                            outcome=finalization_outcome,
                            usage=finalization_usage,
                        )
                    )
            else:
                context.emit_event(
                    finalization_error_event(
                        finalization_root_error.type,
                        finalization_root_error.message,
                    )
                )

        # If extraction cannot recover an outstanding PolicyError, do not
        # present the gated draft as a normal answer. A successful extraction
        # below may use it as evidence, but remains explicitly unconfirmed and
        # preserves the violation in `recovered_error`.
        policy_outstanding = error is not None and error.type == "PolicyError"
        withheld_content = ""
        if policy_outstanding:
            withheld_content = str(answer.get("content") or "")
            final_answer = ""
        else:
            final_answer = _best_answer(answer, last_output, last_response, last_execution_status)

        # Extract fallback: the loop exhausted its iteration budget or reached
        # a fail-closed terminal handoff without answer['ready']. All main-loop
        # root failures return early. A failed terminal finalization is the
        # sole exception: its event is emitted above while the original
        # terminal error remains authoritative.
        # Failed terminal attempts are trajectory evidence too: they can mutate
        # answer['content'] before raising, and their code/error explain how
        # trustworthy that draft is.
        was_extracted = False
        extract_error: RLMError | None = None
        recovered_error: RLMError | None = None
        if (
            not answer.get("ready")
            and (iterations >= cfg.max_iterations or terminal_budget_handoff)
            and trajectory
            and _has_extractable_work(answer, has_successful_step)
        ):
            context.emit_progress("Loop ended unconfirmed: extracting best final answer...")
            draft = str(answer.get("content") or "")
            extracted, extract_error = _extract_final_answer(
                question,
                draft,
                trajectory,
                root_llm,
                cfg,
                context,
                resolved_prompt_pack.pack,
            )
            if extracted:
                final_answer = extracted
                was_extracted = True
                # The extract pass is the bounded terminal recovery for the
                # failed step. Hosts treat result.error as fatal, so leaving
                # the superseded execution/policy error set would still make
                # them discard the recovered answer. The failed attempt stays
                # available in the trajectory and recovered_error for typed
                # diagnostics.
                recovered_error = error
                error = None
            elif extract_error is not None:
                # Don't swallow this silently (the bug being fixed here):
                # final_answer stays the raw-loop-output fallback from
                # _best_answer above, but the failure is now visible to hosts
                # via RLMResult.extract_error, not indistinguishable from a
                # clean extraction.
                context.emit_event(extract_error_event(extract_error.type, extract_error.message))

        if policy_outstanding and not was_extracted and error is not None and withheld_content:
            details = dict(error.details or {})
            details["withheld_content"] = withheld_content
            error.details = details

        if not final_answer:
            if error:
                final_answer = f"Error: {error.message}"
            else:
                final_answer = "No output produced."

        return finalize(
            answer_text=final_answer,
            answer=answer,
            iterations=iterations,
            context=context,
            trajectory=trajectory,
            error=error,
            extracted=was_extracted,
            extract_error=extract_error,
            recovered_error=recovered_error,
            answer_metadata=answer_metadata,
            prompt_pack=prompt_pack_record,
        )
    finally:
        environment.close()
