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
from uuid import uuid4

from ..exceptions import PolicyError, RLMError, SandboxError
from ..execution.budget import (
    Budget,
    BudgetExhausted,
    BudgetRequest,
    conservative_token_estimate,
)
from ..execution.config import SandboxLimits
from ..execution.context import ExecutionContext
from ..execution.manifest import (
    RolloutConfiguration,
    ScaffoldManifest,
    ScaffoldRequirements,
)
from ..execution.progress import execution_error_event, output_event
from ..execution.report import project_result
from ..execution.trace import (
    DataUseAuthorization,
    RunRecord,
    RunRecordCallback,
    TraceRetentionPolicy,
)
from ..policy import PolicyHints, contract_violations, ready_violations
from ..prompts.pack import PromptPackRecord
from ..protocols.environment import ExecutionResult, RLMEnvironment
from ..protocols.llm_client import (
    CACHE_ANCHOR_MARKER,
    LLMClient,
    LLMUsageFailure,
    TokenUsage,
    total_tokens_from_usage,
)
from ..structured import _StructuredBatchEvidence
from .trajectory import (
    EXECUTION_STATUS_ERROR,
    EXECUTION_STATUS_SUCCESS,
    AttemptKind,
    ExecutionStatus,
    IterationRecord,
)

# Fed back instead of an empty string when executed code prints nothing, so the
# model learns that only stdout is visible.
EMPTY_OUTPUT_NUDGE = "(no output - did you forget to print?)"

# Live root calls keep two completed iterations byte-for-byte. Once an older
# refinement leaves that tail, its frozen projection retains at most this many
# characters of stdout, including the pack-authored elision marker.
LIVE_TRANSCRIPT_VERBATIM_ITERATIONS = 2
HISTORICAL_STDOUT_CHARS = 2_000

# Structured answer metadata crosses process and language boundaries as JSON.
MAX_ANSWER_METADATA_BYTES = 64 * 1024
MAX_ANSWER_METADATA_NODES = 10_000
MAX_ANSWER_METADATA_DEPTH = 100
MAX_SAFE_JSON_INTEGER = (1 << 53) - 1


@dataclass
class RLMConfig:
    """Configuration for RLM execution."""

    budget: Budget = field(default_factory=Budget)
    sandbox: SandboxLimits = field(default_factory=SandboxLimits)
    # ``tips_profile`` remains the compatibility spelling; ``prompt_profile``
    # selects a complete pack when set and otherwise inherits it.
    tips_profile: str = "full"
    prompt_profile: str | None = None
    verbose: bool = False
    root_model: str | None = None
    on_progress: Any | None = None
    # None delegates to the resolved pack's immutable policy default.
    enforce_contract: bool | None = None
    policy_hints: PolicyHints | None = None
    trace_retention: TraceRetentionPolicy = field(default_factory=TraceRetentionPolicy)
    data_use: DataUseAuthorization = field(default_factory=DataUseAuthorization)
    run_id: str | None = None
    parent_run_id: str | None = None
    trace_depth: int | None = None
    on_run_record: RunRecordCallback | None = None
    rollout: RolloutConfiguration = field(default_factory=RolloutConfiguration)
    checkpoint_requirements: ScaffoldRequirements | None = None


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
    # Set when terminal extraction was attempted and failed (or returned
    # empty) — `answer` is then raw loop output (the last
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
    # Identity and provenance of the one immutable pack resolved at run start.
    prompt_pack: PromptPackRecord | None = None
    # Policy-resolved terminal Trace ABI value.
    run_record: RunRecord | None = None
    # Content-free, model-facing ABI identity resolved before the first request.
    scaffold_manifest: ScaffoldManifest | None = None
    stdout_chars: int = 0


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
    # Exact length of stdout returned by the environment. Error feedback is
    # deliberately excluded; an executor that raises before returning stdout
    # contributes zero rather than a guessed partial length.
    stdout_chars: int = 0

    @property
    def execution_status(self) -> ExecutionStatus:
        """Structured execution state derived from the typed outcome, not stdout."""
        return EXECUTION_STATUS_ERROR if self.error is not None else EXECUTION_STATUS_SUCCESS


@dataclass(frozen=True, slots=True)
class TranscriptWindowEntry:
    """Frozen outbound replacement for one completed iteration's refinement.

    The replacement is created when the iteration finishes, using only state
    known through that iteration. It can therefore move into the stable region
    on any later call without changing bytes.
    """

    iteration: int
    message_index: int
    elided_content: str

    def __post_init__(self) -> None:
        if self.iteration < 1:
            raise ValueError("transcript window iteration must be positive")
        if self.message_index < 0:
            raise ValueError("transcript window message_index must be non-negative")
        if not isinstance(self.elided_content, str):
            raise TypeError("transcript window elided_content must be a string")


def _execution_output(result: ExecutionResult | str) -> str:
    if isinstance(result, ExecutionResult):
        return result.stdout
    return str(result)


def _feedback_output(output: str) -> str:
    """What the model sees as the execution result. Empty stdout becomes a
    nudge instead of silence; the raw output is kept separately so an empty
    run never leaks the nudge text into `_best_answer`."""
    return output if output else EMPTY_OUTPUT_NUDGE


def elide_historical_stdout(output: str, *, placeholder: str) -> str:
    """Return a deterministic head/tail window for historical stdout."""
    if len(output) <= HISTORICAL_STDOUT_CHARS:
        return output
    retained = max(0, HISTORICAL_STDOUT_CHARS - len(placeholder))
    head_chars = (retained + 1) // 2
    tail_chars = retained - head_chars
    tail = output[-tail_chars:] if tail_chars else ""
    return output[:head_chars] + placeholder + tail


def project_live_transcript(
    messages: list[dict[str, str]],
    window_entries: tuple[TranscriptWindowEntry, ...] = (),
) -> tuple[list[dict[str, Any]], int | None]:
    """Build the outbound-only live transcript and its stable frontier.

    The last two completed iterations remain verbatim. Older refinements use
    their already-frozen replacement. The canonical messages and entry values
    are never mutated or aliased by the returned dictionaries.
    """
    outbound: list[dict[str, Any]] = [dict(message) for message in messages]
    stable_count = max(0, len(window_entries) - LIVE_TRANSCRIPT_VERBATIM_ITERATIONS)
    if stable_count == 0:
        return outbound, None

    stable_entries = window_entries[:stable_count]
    previous_iteration = 0
    previous_index = -1
    for entry in stable_entries:
        if entry.iteration <= previous_iteration or entry.message_index <= previous_index:
            raise ValueError("transcript window entries must be strictly ordered")
        if entry.message_index >= len(outbound):
            raise ValueError("transcript window message_index is outside the transcript")
        outbound[entry.message_index]["content"] = entry.elided_content
        previous_iteration = entry.iteration
        previous_index = entry.message_index
    return outbound, stable_entries[-1].message_index


def error_repair_history(exec_error: Exception) -> str:
    """Describe a failed step once for legacy and prompt-pack repair paths."""
    facts = [f"type={exec_error.__class__.__name__}", f"message={exec_error}"]
    if isinstance(exec_error, PolicyError):
        facts.append(
            "Your accumulated answer['content'] was kept, but answer['ready'] was "
            "reset. Address the policy violation above, then set "
            'answer["ready"] = True again.'
        )
    else:
        facts.append("Fix the code and try again.")
    return "\n".join(facts)


def _resolved_output(answer: dict[str, Any], last_output: str) -> str:
    if answer.get("content"):
        return str(answer.get("content", ""))
    return last_output


def _best_answer(
    answer: dict[str, Any],
    last_output: str,
    last_response: str,
    last_execution_status: ExecutionStatus | None,
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
    rendered_prompt: str | None = None,
) -> list[dict[str, str]]:
    refinement_content = rendered_prompt
    if refinement_content is None:
        refinement_content = template.format(
            current_content=answer_content,
            last_output=_feedback_output(last_output),
        )
    return messages + [
        {"role": "assistant", "content": f"```python\n{code}\n```"},
        {"role": "user", "content": refinement_content},
    ]


def build_missing_code_repair_messages(
    messages: list[dict[str, str]], response: str, *, repair_prompt: str | None = None
) -> list[dict[str, str]]:
    return messages + [
        {"role": "assistant", "content": response},
        {
            "role": "user",
            "content": repair_prompt
            or "Your response must include a single ```python code block. Return only code.",
        },
    ]


def build_error_repair_messages(
    messages: list[dict[str, str]],
    code: str,
    exec_error: Exception,
    *,
    repair_prompt: str | None = None,
) -> list[dict[str, str]]:
    return messages + [
        {"role": "assistant", "content": f"```python\n{code}\n```"},
        {"role": "user", "content": repair_prompt or error_repair_history(exec_error)},
    ]


def call_root(
    root_llm: LLMClient,
    messages: list[dict[str, str]],
    *,
    model: str,
    context: ExecutionContext,
    cache_anchors: tuple[int, ...] | None = (0, -1),
    transcript_window: tuple[TranscriptWindowEntry, ...] = (),
) -> tuple[str, Any, RLMError | None]:
    """One root-LLM call with token accounting.

    Returns ``(response, usage, None)`` on success or ``("", None, error)`` —
    a root failure always ends the run, so the caller finalizes immediately.
    """
    outbound_messages, frontier = project_live_transcript(messages, transcript_window)
    call_id = "root:" + str(uuid4())
    input_estimate = conservative_token_estimate(outbound_messages)
    request = BudgetRequest(
        tokens=input_estimate + context.budget.root_output_tokens,
    )
    try:
        context.ledger.reserve(call_id, request, through_deadline=True)
    except BudgetExhausted as exc:
        return (
            "",
            None,
            RLMError(
                type="BudgetExhausted",
                message=str(exc),
                details={
                    "resource": exc.resource,
                    "requested": exc.requested,
                    "remaining": exc.remaining,
                },
            ),
        )
    context.record_root_attempt()
    if cache_anchors is not None:
        # Before elision, preserve the system + tail policy from prompt caching.
        # Once a stable region exists, anchor its frontier instead: the verbatim
        # tail is rewritten by the next iteration and cannot be reused.
        if frontier is not None:
            cache_anchors = (0, frontier)
        message_count = len(outbound_messages)
        for anchor in cache_anchors:
            index = anchor if anchor >= 0 else message_count + anchor
            if 0 <= index < message_count:
                outbound_messages[index][CACHE_ANCHOR_MARKER] = True

    def settle_completed_usage(usage: TokenUsage, *, success: bool) -> RLMError | None:
        actual = BudgetRequest(tokens=usage.total_tokens if usage.exact else request.tokens)
        accounting_error: RLMError | None = None
        try:
            if success:
                context.record_root_success(usage)
            else:
                context.record_root_usage(usage)
        except Exception as exc:
            accounting_error = RLMError(type=exc.__class__.__name__, message=str(exc))
        try:
            context.ledger.commit(call_id, actual)
        except BudgetExhausted as exc:
            return RLMError(type="BudgetExhausted", message=str(exc))
        return accounting_error

    try:
        response, usage = root_llm.responses_create(
            outbound_messages,
            model=model,
            max_tokens=context.budget.root_output_tokens,
            return_usage=True,
        )
    except LLMUsageFailure as exc:
        accounting_error = settle_completed_usage(exc.usage, success=False)
        if accounting_error is not None:
            return "", exc.usage, accounting_error
        return (
            "",
            exc.usage,
            RLMError(type=exc.cause.__class__.__name__, message=str(exc.cause)),
        )
    except BaseException as exc:
        context.record_root_usage_unavailable()
        try:
            context.ledger.commit(call_id, BudgetRequest(tokens=request.tokens))
        except BudgetExhausted as budget_exc:
            if not isinstance(exc, Exception):
                raise
            return "", None, RLMError(type="BudgetExhausted", message=str(budget_exc))
        if not isinstance(exc, Exception):
            raise
        return "", None, RLMError(type=exc.__class__.__name__, message=str(exc))
    if not isinstance(usage, TokenUsage):
        usage = TokenUsage.unavailable()
    # A completed provider response remains a usage/success fact even when
    # final ledger reconciliation later rejects a token or wall-time overrun.
    # Record it before commit so the terminal trace never reports zero usage
    # for work the provider completed and billed.
    accounting_error = settle_completed_usage(usage, success=True)
    if accounting_error is not None:
        return "", usage, accounting_error
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
    semantic_evidence: _StructuredBatchEvidence | None = None,
) -> StepOutcome:
    """Execute one code block: the identical path for a first attempt and for
    repaired code. Policy pre-check, execute, output budget, event emit,
    answer refresh, ready-time policy checks."""
    # Direct execute_step callers have no resolved pack; None retains the
    # historical strict default. run_rlm resolves None before reaching here.
    enforce_contract = cfg.enforce_contract is not False
    stdout_chars = 0
    try:
        if enforce_contract:
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
        stdout_chars = len(last_output)
        # Enforce the output budget BEFORE emitting: an over-budget print
        # must raise here (error path), never stream its full contents to
        # the event channel — the guard gates the event, not just the loop.
        _enforce_output_budget(last_output, cfg.sandbox.output_chars)
        answer = _refresh_answer(env_globals, answer)

        hints = cfg.policy_hints if enforce_contract else None
        violations = ready_violations(
            hints,
            answer_ready=bool(answer.get("ready")),
            successful_calls=context.stats.successful_calls,
            resolved_output=_resolved_output(answer, last_output),
            unresolved_semantic_batches=(
                semantic_evidence.unresolved_batches if semantic_evidence is not None else 0
            ),
            unresolved_semantic_items=(
                semantic_evidence.unresolved_items if semantic_evidence is not None else 0
            ),
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
        # Failed code cannot submit a confirmed answer, even if it set ready
        # before raising or rebound the answer dict entirely. Keep accumulated
        # content for repair/extraction, but always revoke readiness. Policy
        # errors use the same state rule and receive specialized feedback from
        # `error_repair_history`.
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
            ),
            exception=exec_error,
            stdout_chars=stdout_chars,
        )

    return StepOutcome(
        output=last_output,
        answer=answer,
        answer_metadata=answer_metadata,
        stdout_chars=stdout_chars,
    )


def record_iteration(
    *,
    iteration: int,
    messages: list[dict[str, str]],
    response: str,
    code: str,
    outcome: StepOutcome,
    usage: Any,
    attempt_kind: AttemptKind = "initial",
) -> IterationRecord:
    """The one place iteration records are built: a structured message-list
    snapshot (deep-copied — the live list keeps growing), nudge-normalized
    execution output. Taking the typed outcome keeps its output and status
    atomic instead of accepting an independently assembled pair."""
    return IterationRecord(
        iteration=iteration,
        llm_input=[dict(message) for message in messages],
        llm_output=response,
        code_executed=code,
        execution_result=_feedback_output(outcome.output),
        tokens_used=total_tokens_from_usage(usage),
        execution_status=outcome.execution_status,
        attempt_kind=attempt_kind,
        stdout_chars=outcome.stdout_chars,
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
    prompt_pack: PromptPackRecord | None = None,
    config: RLMConfig | None = None,
    scaffold_manifest: ScaffoldManifest | None = None,
) -> RLMResult:
    """The one RLMResult and terminal RunRecord construction site."""
    result = RLMResult(
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
        prompt_pack=prompt_pack,
        scaffold_manifest=scaffold_manifest,
        stdout_chars=sum(entry.stdout_chars for entry in trajectory),
    )
    cfg = config or RLMConfig(budget=context.budget, sandbox=context.sandbox)
    usage = context.stats.resolved_usage(context.trace.elapsed_ms()).as_dict()
    snapshot = context.ledger.snapshot()
    budget = {
        "kind": "snapshot",
        "source": "budget_ledger",
        "configured": snapshot.configured.as_dict(),
        "consumed": snapshot.consumed.as_dict(),
        "remaining": snapshot.remaining.as_dict(),
    }
    observed_errors = (result.error, result.extract_error, result.recovered_error)
    policy_violation = next(
        (value for value in observed_errors if value is not None and value.type == "PolicyError"),
        None,
    )
    if not cfg.enforce_contract:
        policy_outcome = "not_enforced"
    elif policy_violation is not None:
        policy_outcome = "violated"
    elif any(value is not None for value in observed_errors):
        policy_outcome = "not_evaluated"
    else:
        policy_outcome = "passed"
    policy = {
        "contract_enforced": bool(cfg.enforce_contract),
        "outcome": policy_outcome,
        "violation_type": policy_violation.type if policy_violation is not None else None,
    }
    context.emit_event({"type": "usage", **usage})
    context.emit_event({"type": "budget", **budget})
    context.emit_event({"type": "policy", **policy})
    # The live stream always receives one canonical unary-equivalent result.
    # Retention controls storage only; it must never suppress live delivery.
    canonical_result = project_result(result, include_trajectory=False)
    canonical_result.pop("run_record")
    context.emit_event({"type": "result", "result": canonical_result})
    # Full replay/trajectory content is retained and delivered only when a host
    # explicitly selects it.
    if "replay" in context.trace.retention.retain:
        replay = project_result(result, include_trajectory=True)
        replay.pop("run_record")
        context.emit_event({"type": "replay", "result": replay})

    def terminal_error(value: RLMError | None) -> dict[str, Any] | None:
        if value is None:
            return None
        # RLMError.code is executed source code, not a stable machine code.
        # Terminal errors therefore carry type only; all content stays in the
        # configurable replay value.
        return {"type": value.type}

    terminal = {
        "status": (
            "success" if result.error is None and (result.ready or result.extracted) else "error"
        ),
        "ready": result.ready,
        "extracted": result.extracted,
        "iterations": result.iterations,
        "usage": usage,
        "budget": budget,
        "policy": policy,
        "retention": {
            **context.trace.retention.as_dict(),
            "replay_retained": "replay" in context.trace.retention.retain,
        },
        "error": terminal_error(result.error),
        "extract_error": terminal_error(result.extract_error),
        "recovered_error": terminal_error(result.recovered_error),
        "scaffold_manifest_id": (
            scaffold_manifest.manifest_id if scaffold_manifest is not None else None
        ),
        "scaffold_manifest_version": (
            scaffold_manifest.schema_version if scaffold_manifest is not None else None
        ),
        "stdout_chars": result.stdout_chars,
    }
    result.run_record = context.finish_trace(terminal)
    return result
