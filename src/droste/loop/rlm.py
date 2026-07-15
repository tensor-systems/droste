from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping

from ..capabilities import CapabilityManifest
from ..exceptions import PolicyError, RLMError
from ..execution.budget import BudgetExhausted
from ..execution.config import validate_subcall_concurrency
from ..execution.context import ExecutionContext, create_execution_context
from ..execution.manifest import (
    EngineIdentity,
    ScaffoldManifest,
    build_scaffold_manifest,
    require_scaffold_compatibility,
)
from ..execution.progress import (
    EventCallback,
    extract_event,
    iteration_start_event,
    llm_response_event,
    repair_event,
)
from ..execution.progress import (
    code_event as build_code_event,
)
from ..execution.trace import DataUseAuthorization, TraceRetentionPolicy
from ..prompts.pack import (
    CODE_OUTPUT_CONTRACT,
    DEFAULT_PROMPT_PROFILE,
    EXTRACT_OUTPUT_CONTRACT,
    PromptPack,
    PromptPackCatalog,
    PromptPackRecord,
    PromptSlots,
    ResolvedPromptPack,
    load_builtin_prompt_catalog,
    render_prompt_template,
    render_system_prompt,
    resolve_prompt_pack,
)
from ..protocols.environment import RLMEnvironment
from ..protocols.llm_client import LLMClient
from ..protocols.subcall_capacity import (
    SubcallInputCapacity,
    render_subcall_input_capacity,
    reported_subcall_input_capacity,
    resolve_subcall_input_capacity,
)
from ..protocols.subcall_client import SubcallClient
from ..protocols.verbs import EMPTY_ACCESSOR_MANIFEST, AccessorManifest
from ..providers import PROVIDER_PROTOCOL_VERSION
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

__all__ = [
    "RLMConfig",
    "RLMPreflight",
    "RLMResult",
    "RLM_PREFLIGHT_SCHEMA_VERSION",
    "preflight_rlm",
    "run_rlm",
    "EMPTY_OUTPUT_NUDGE",
]

RLM_PREFLIGHT_SCHEMA_VERSION = 1

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
_UNKNOWN_OUTPUT_TOKEN_LIMIT = object()
_UNKNOWN_SUBCALL_CONCURRENCY = object()


def _raise_with_environment_cleanup(
    environment: RLMEnvironment,
    primary: BaseException,
    message: str,
) -> None:
    """Close after a failed operation without replacing its primary error."""

    try:
        environment.close()
    except BaseException as cleanup_error:
        raise BaseExceptionGroup(message, [primary, cleanup_error]) from None


def _warn_environment_cleanup(cleanup_error: BaseException) -> None:
    """Surface post-result cleanup failure without discarding the result."""

    detail = " ".join(str(cleanup_error).split())[:1_000]
    warnings.warn(
        "RLM result preserved after environment cleanup failed: "
        f"{type(cleanup_error).__name__}: {detail}",
        RuntimeWarning,
        stacklevel=3,
    )


ProgressCallback = Any


@dataclass(frozen=True, slots=True)
class RLMPreflight:
    """Content-free result of resolving and checking one effective scaffold."""

    scaffold_manifest: ScaffoldManifest
    schema_version: int = RLM_PREFLIGHT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RLM_PREFLIGHT_SCHEMA_VERSION:
            raise ValueError(f"unsupported RLM preflight version: {self.schema_version}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scaffold_manifest": self.scaffold_manifest.as_wire_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RLMPreflight":
        """Strictly parse the public preflight value."""

        if not isinstance(value, Mapping):
            raise TypeError("RLM preflight must be an object")
        missing = {"schema_version", "scaffold_manifest"} - value.keys()
        unknown = value.keys() - {"schema_version", "scaffold_manifest"}
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unknown " + ", ".join(sorted(unknown)))
            raise ValueError("RLM preflight has " + "; ".join(details))
        schema_version = value["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise TypeError("RLM preflight schema_version must be an integer")
        manifest = value["scaffold_manifest"]
        if not isinstance(manifest, Mapping):
            raise TypeError("RLM preflight scaffold_manifest must be an object")
        return cls(
            scaffold_manifest=ScaffoldManifest.from_dict(manifest),
            schema_version=schema_version,
        )


@dataclass(frozen=True, slots=True)
class _ResolvedRLMScaffold:
    """Immutable facts shared by preflight and execution after environment snapshotting."""

    config: RLMConfig
    prompt_pack: ResolvedPromptPack
    prompt_pack_record: PromptPackRecord
    scaffold_manifest: ScaffoldManifest


def _prepare_rlm_scaffold(
    *,
    environment: RLMEnvironment,
    config: RLMConfig | None,
    system_prompt: str | None,
    system_prompt_additions: str | None,
    user_prompt_template: str | None,
    refinement_prompt_template: str | None,
    prompt_pack: PromptPack | None,
    consumer_prompt_catalog: PromptPackCatalog | None,
) -> tuple[_ResolvedRLMScaffold, dict[str, Any]]:
    """Resolve the one scaffold authority used by both preflight and execution."""

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

    env_globals = environment.globals()
    scaffold_manifest = build_scaffold_manifest(
        engine=_engine_identity(cfg.rollout.source_revision),
        prompt_pack=resolved_prompt_pack.pack,
        capability_manifest=_capability_manifest(environment),
        provider_protocol=PROVIDER_PROTOCOL_VERSION,
        model_visible_globals=tuple(env_globals),
        root_model=cfg.root_model,
        rollout=cfg.rollout,
        budget=cfg.budget,
        sandbox=cfg.sandbox,
        system_prompt_override=system_prompt,
        system_prompt_additions=system_prompt_additions,
        user_prompt_override=user_prompt_template,
        refinement_prompt_override=refinement_prompt_template,
    )
    require_scaffold_compatibility(scaffold_manifest, cfg.checkpoint_requirements)
    return (
        _ResolvedRLMScaffold(
            config=cfg,
            prompt_pack=resolved_prompt_pack,
            prompt_pack_record=resolved_prompt_pack.record(),
            scaffold_manifest=scaffold_manifest,
        ),
        env_globals,
    )


def preflight_rlm(
    *,
    environment: RLMEnvironment,
    config: RLMConfig | None = None,
    system_prompt: str | None = None,
    system_prompt_additions: str | None = None,
    user_prompt_template: str | None = None,
    refinement_prompt_template: str | None = None,
    prompt_pack: PromptPack | None = None,
    consumer_prompt_catalog: PromptPackCatalog | None = None,
    subcalls: SubcallClient | None = None,
) -> RLMPreflight:
    """Resolve and check a run scaffold without model or provider execution."""

    try:
        cfg = config or RLMConfig()
        reported_capacity = _snapshot_subcall_input_capacity(environment, subcalls)
        cfg = _with_resolved_subcall_input_capacity(cfg, reported_capacity)
        resolved, _ = _prepare_rlm_scaffold(
            environment=environment,
            config=cfg,
            system_prompt=system_prompt,
            system_prompt_additions=system_prompt_additions,
            user_prompt_template=user_prompt_template,
            refinement_prompt_template=refinement_prompt_template,
            prompt_pack=prompt_pack,
            consumer_prompt_catalog=consumer_prompt_catalog,
        )
        return RLMPreflight(resolved.scaffold_manifest)
    finally:
        environment.close()


def _engine_identity(source_revision: str | None) -> EngineIdentity:
    """I/O edge for installed-package identity; manifest composition stays pure."""

    try:
        engine_version = version("droste")
    except PackageNotFoundError:
        engine_version = "source"
    return EngineIdentity(engine_version, source_revision)


class _SubcallGate:
    """Run-scoped subcall bindings that can be revoked before delegation."""

    def __init__(self, subcalls: SubcallClient) -> None:
        self._subcalls = subcalls
        self._blocked = False

    def _check(self) -> None:
        if self._blocked:
            raise BudgetExhausted("subcalls", 1, 0)

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

    @property
    def output_token_limit(self) -> int | None:
        """Forward optional planning metadata without changing the base protocol."""
        return getattr(self._subcalls, "output_token_limit")

    @property
    def input_token_capacity(self) -> SubcallInputCapacity:
        """Forward optional planning metadata without changing the base protocol."""
        return getattr(self._subcalls, "input_token_capacity")

    def block(self) -> None:
        """Permanently revoke this run's model-visible subcall bindings."""
        self._blocked = True


def _validate_existing_context_trace(cfg: RLMConfig, context: ExecutionContext) -> None:
    """An injected context owns trace identity/policy; reject explicit conflicts.

    Contexts carry shared accounting and may already contain events, so changing
    their recorder at run start would splice two identities into one sequence.
    Default RLMConfig trace values mean "use the context's value".
    """
    conflicts: list[str] = []
    if cfg.budget != context.budget:
        conflicts.append("budget")
    if cfg.sandbox != context.sandbox:
        conflicts.append("sandbox")
    if cfg.run_id is not None and cfg.run_id != context.trace.run_id:
        conflicts.append("run_id")
    if cfg.parent_run_id is not None and cfg.parent_run_id != context.trace.parent_run_id:
        conflicts.append("parent_run_id")
    if cfg.trace_depth is not None and cfg.trace_depth != context.trace.depth:
        conflicts.append("trace_depth")
    if (
        cfg.trace_retention != TraceRetentionPolicy()
        and cfg.trace_retention != context.trace.retention
    ):
        conflicts.append("trace_retention")
    if cfg.data_use != DataUseAuthorization() and cfg.data_use != context.trace.data_use:
        conflicts.append("data_use")
    if cfg.on_run_record is not None and cfg.on_run_record is not context.config.on_run_record:
        conflicts.append("on_run_record")
    if conflicts:
        raise ValueError(
            "an injected ExecutionContext owns trace settings; conflicting RLMConfig fields: "
            + ", ".join(conflicts)
        )


def _accessor_manifest(environment: RLMEnvironment) -> AccessorManifest:
    """Data-accessor names for the count contract's len() check (#10).

    Explicit data, not sniffing: an environment that composes data sources
    (e.g. one wrapping a ProviderRegistry) reports them via an optional
    ``accessor_manifest()`` method. Environments without one yield an empty
    manifest, and the policy layer otherwise sees no data accessors."""
    manifest_fn = getattr(environment, "accessor_manifest", None)
    if callable(manifest_fn):
        manifest = manifest_fn()
        if isinstance(manifest, AccessorManifest):
            return manifest
    return EMPTY_ACCESSOR_MANIFEST


def _capability_manifest(environment: RLMEnvironment) -> CapabilityManifest:
    """Read the broker's immutable allowlist without making it a protocol requirement."""

    broker_fn = getattr(environment, "capability_broker", None)
    if not callable(broker_fn):
        return CapabilityManifest()
    broker = broker_fn()
    describe = getattr(broker, "describe", None)
    manifest = describe() if callable(describe) else None
    if not isinstance(manifest, CapabilityManifest):
        raise TypeError("environment capability_broker().describe() must return CapabilityManifest")
    return manifest


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


def _reported_output_token_limit(subcalls: SubcallClient) -> int | None | object:
    try:
        limit = getattr(subcalls, "output_token_limit")
    except Exception:
        return _UNKNOWN_OUTPUT_TOKEN_LIMIT
    if limit is None:
        return None
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        return limit
    return _UNKNOWN_OUTPUT_TOKEN_LIMIT


def _reported_subcall_concurrency(subcalls: SubcallClient) -> int | object:
    """Read optional effective concurrency without making it a base protocol field."""

    try:
        concurrency = getattr(subcalls, "subcall_concurrency")
    except AttributeError:
        return _UNKNOWN_SUBCALL_CONCURRENCY
    return validate_subcall_concurrency(concurrency)


def _with_resolved_subcall_input_capacity(
    cfg: "RLMConfig", reported: SubcallInputCapacity
) -> "RLMConfig":
    resolved = resolve_subcall_input_capacity(
        cfg.rollout.subcall_input_capacity,
        reported,
    )
    if resolved == cfg.rollout.subcall_input_capacity:
        return cfg
    return replace(cfg, rollout=replace(cfg.rollout, subcall_input_capacity=resolved))


def _snapshot_subcall_input_capacity(
    environment: RLMEnvironment,
    subcalls: SubcallClient | None,
) -> SubcallInputCapacity:
    """Read one host-owned capacity snapshot before scaffold resolution."""

    environment_snapshot = getattr(environment, "subcall_input_capacity", None)
    if callable(environment_snapshot):
        value = environment_snapshot()
        if not isinstance(value, SubcallInputCapacity):
            raise TypeError("environment subcall_input_capacity must return SubcallInputCapacity")
        return value
    if subcalls is None:
        return SubcallInputCapacity.unknown()
    return reported_subcall_input_capacity(subcalls)


def _budget_prompt(cfg: "RLMConfig", subcalls: SubcallClient) -> str:
    limit = _reported_output_token_limit(subcalls)
    if limit is _UNKNOWN_OUTPUT_TOKEN_LIMIT:
        rendered_limit = "unknown (client did not report)"
    elif limit is None:
        rendered_limit = "unbounded (deliberate)"
    else:
        rendered_limit = f"{limit} (bounded)"
    rendered_input_capacity = render_subcall_input_capacity(cfg.rollout.subcall_input_capacity)
    return (
        "## Authorized compute\n"
        f"tokens={cfg.budget.tokens}; subcalls={cfg.budget.subcalls}; "
        f"depth={cfg.budget.depth}; wall_ms={cfg.budget.wall_ms}; "
        f"root_output_tokens_per_call={cfg.budget.root_output_tokens}; "
        f"subcall_output_tokens_per_call={cfg.budget.subcall_output_tokens}\n"
        f"client_reported_subcall_output_limit={rendered_limit}\n"
        f"subcall_input_capacity={rendered_input_capacity}\n"
        "Input capacity guides prompt/context chunking only; it does not increase "
        "the per-call subcall output-token limit.\n"
        f"Sandbox output_chars_per_iteration={cfg.sandbox.output_chars}."
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
    if evidence is None:
        return None
    required = evidence.minimum_exact_retry_calls
    remaining = context.ledger.snapshot().remaining.subcalls
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
    when a terminal budget handoff occurs without answer['ready'], make one
    root-LLM call over a compact trajectory summary so the run returns
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
        response, _usage, root_error = call_root(
            root_llm,
            messages,
            model=cfg.root_model or "",
            context=context,
        )
        if root_error is not None:
            return "", root_error
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
    try:
        cfg = config or RLMConfig()
        reported_concurrency = _reported_subcall_concurrency(subcalls)
        if (
            reported_concurrency is not _UNKNOWN_SUBCALL_CONCURRENCY
            and reported_concurrency != cfg.rollout.concurrency
        ):
            raise ValueError(
                "rollout concurrency does not match the subcall client: "
                f"{cfg.rollout.concurrency} != {reported_concurrency}"
            )
        if context is None:
            context = create_execution_context(
                budget=cfg.budget,
                sandbox=cfg.sandbox,
                verbose=cfg.verbose,
                on_progress=on_progress or cfg.on_progress,
                # None -> NO emission (#35). Embedders that want the NDJSON
                # stderr stream attach droste.execution.progress.emit_event.
                on_event=on_event,
                on_run_record=cfg.on_run_record,
                run_id=cfg.run_id,
                parent_run_id=cfg.parent_run_id,
                trace_depth=cfg.trace_depth if cfg.trace_depth is not None else 0,
                trace_retention=cfg.trace_retention,
                data_use=cfg.data_use,
            )
        else:
            _validate_existing_context_trace(cfg, context)
        bind_context = getattr(subcalls, "bind_context", None)
        if callable(bind_context):
            bind_context(context)

        # The environment contract owns one brokered subcall surface. The host's
        # raw client remains trusted loop state for context binding/accounting.
        sandbox_subcalls = environment.sandbox_subcalls(subcalls, context.ledger)
        reported_capacity = reported_subcall_input_capacity(sandbox_subcalls)
        cfg = _with_resolved_subcall_input_capacity(cfg, reported_capacity)
        resolved_scaffold, env_globals = _prepare_rlm_scaffold(
            environment=environment,
            config=cfg,
            system_prompt=system_prompt,
            system_prompt_additions=system_prompt_additions,
            user_prompt_template=user_prompt_template,
            refinement_prompt_template=refinement_prompt_template,
            prompt_pack=prompt_pack,
            consumer_prompt_catalog=consumer_prompt_catalog,
        )
        cfg = resolved_scaffold.config
        resolved_prompt_pack = resolved_scaffold.prompt_pack
        prompt_pack_record = resolved_scaffold.prompt_pack_record
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
                    budget=_budget_prompt(cfg, model_subcalls),
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
                        f"Conversation Context:\n{conversation_context}"
                        if conversation_context
                        else ""
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
        scaffold_manifest: ScaffoldManifest | None = resolved_scaffold.scaffold_manifest
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
                config=cfg,
                scaffold_manifest=scaffold_manifest,
            )

    except BaseException as exc:
        _raise_with_environment_cleanup(
            environment,
            exc,
            "RLM setup and environment cleanup failed",
        )
        raise
    primary_error: BaseException | None = None
    try:
        context.emit_event(
            {
                "type": "startup",
                "engine_version": scaffold_manifest.body["engine"]["version"],
                "runner_protocol": cfg.rollout.runner_protocol,
                "provider_protocol": scaffold_manifest.body["abis"]["provider"],
                "scaffold_manifest_id": scaffold_manifest.manifest_id,
                "scaffold_manifest_version": scaffold_manifest.schema_version,
            }
        )
        while not answer.get("ready"):
            iterations += 1
            context.begin_iteration(iterations)
            context.emit_event(
                iteration_start_event(iterations, context.ledger.snapshot().remaining.tokens)
            )

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

            context.emit_progress(f"Iteration {iterations}: Generating code...")

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
                    context.emit_event(repair_event(iterations, "missing_code", "start"))
                    repair_response, repair_usage, root_error = call_root(
                        root_llm, repair_messages, model=cfg.root_model or "", context=context
                    )
                    if root_error is not None:
                        context.emit_event(
                            repair_event(
                                iterations,
                                "missing_code",
                                "failure",
                                error_type=root_error.type,
                                message=root_error.message,
                            )
                        )
                        return early_result(root_error)
                    last_response = repair_response
                    context.emit_event(llm_response_event(iterations, repair_response))

                    code = extract_code_block(repair_response, "python")
                    if not code:
                        missing_code_error = RLMError(
                            type="PolicyError",
                            message="Response missing python code block.",
                        )
                        context.emit_event(
                            repair_event(
                                iterations,
                                "missing_code",
                                "failure",
                                error_type=missing_code_error.type,
                                message=missing_code_error.message,
                            )
                        )
                        return early_result(missing_code_error)
                    context.emit_event(repair_event(iterations, "missing_code", "completion"))
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
                        config=cfg,
                        scaffold_manifest=scaffold_manifest,
                    )

            context.emit_progress(f"Iteration {iterations}: Executing...")
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

            context.emit_progress(f"Iteration {iterations}: Retrying with error feedback...")
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
            context.emit_event(
                repair_event(
                    iterations,
                    "execution_error",
                    "start",
                )
            )
            repair_response, repair_usage, root_error = call_root(
                root_llm, repair_messages, model=cfg.root_model or "", context=context
            )
            if root_error is not None:
                context.emit_event(
                    repair_event(
                        iterations,
                        "execution_error",
                        "failure",
                        error_type=root_error.type,
                        message=root_error.message,
                    )
                )
                trajectory.append(failed_record)
                return early_result(root_error)
            last_response = repair_response
            context.emit_event(llm_response_event(iterations, repair_response))

            repaired_code = extract_code_block(repair_response, "python")
            if repaired_code:
                context.emit_progress(f"Iteration {iterations}: Executing repaired code...")
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
                    context.emit_event(repair_event(iterations, "execution_error", "completion"))
                else:
                    context.emit_event(
                        repair_event(
                            iterations,
                            "execution_error",
                            "failure",
                            error_type=outcome.error.type,
                            message=outcome.error.message,
                        )
                    )
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
                        attempt_kind="repair",
                    )
                )
            else:
                context.emit_event(
                    repair_event(
                        iterations,
                        "execution_error",
                        "failure",
                        error_type="PolicyError",
                        message="Repair response missing python code block.",
                    )
                )
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
            context.emit_event(repair_event(iterations, "terminal", "start"))
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
                            attempt_kind="terminal",
                        )
                    )
                    if finalization_outcome.error is None:
                        context.emit_event(repair_event(iterations, "terminal", "completion"))
                    else:
                        context.emit_event(
                            repair_event(
                                iterations,
                                "terminal",
                                "failure",
                                error_type=finalization_outcome.error.type,
                                message=finalization_outcome.error.message,
                            )
                        )
                else:
                    context.emit_event(
                        repair_event(
                            iterations,
                            "terminal",
                            "failure",
                            error_type="PolicyError",
                            message="Terminal repair response missing python code block.",
                        )
                    )
            else:
                context.emit_event(
                    repair_event(
                        iterations,
                        "terminal",
                        "failure",
                        error_type=finalization_root_error.type,
                        message=finalization_root_error.message,
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
            and terminal_budget_handoff
            and trajectory
            and _has_extractable_work(answer, has_successful_step)
        ):
            context.emit_progress("Loop ended unconfirmed: extracting best final answer...")
            context.emit_event(extract_event(iterations, "start"))
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
                context.emit_event(extract_event(iterations, "completion"))
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
                context.emit_event(
                    extract_event(
                        iterations,
                        "failure",
                        error_type=extract_error.type,
                        message=extract_error.message,
                    )
                )

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
            config=cfg,
            scaffold_manifest=scaffold_manifest,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            environment.close()
        except BaseException as cleanup_error:
            if primary_error is None:
                _warn_environment_cleanup(cleanup_error)
            else:
                raise BaseExceptionGroup(
                    "RLM execution and environment cleanup failed",
                    [primary_error, cleanup_error],
                ) from None
