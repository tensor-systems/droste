"""Unit tests for the pure per-iteration core split out of run_rlm (#30):
message builders, ready-time policy checks, iteration recording, and the
single RLMResult construction site — no LLM or sandbox required."""

from __future__ import annotations

import json
from dataclasses import asdict

from droste.exceptions import PolicyError, RLMError
from droste.execution.context import create_execution_context
from droste.loop.step import (
    EMPTY_OUTPUT_NUDGE,
    StepOutcome,
    TranscriptWindowEntry,
    build_error_repair_messages,
    build_initial_messages,
    build_missing_code_repair_messages,
    build_refinement_messages,
    call_root,
    elide_historical_stdout,
    finalize,
    project_live_transcript,
    record_iteration,
)
from droste.loop.trajectory import IterationRecord
from droste.policy import PolicyHints, contract_violations, ready_violations
from droste.protocols.llm_client import (
    CACHE_ANCHOR_MARKER,
    TokenUsage,
    strip_cache_anchor_markers,
)


def test_ready_violations_requires_hints_and_readiness() -> None:
    hints = PolicyHints(semantic=True, numeric_output=True)
    assert ready_violations(None, answer_ready=True, successful_calls=0, resolved_output="x") == []
    assert (
        ready_violations(hints, answer_ready=False, successful_calls=0, resolved_output="x") == []
    )


def test_ready_violations_semantic_requires_a_subcall() -> None:
    hints = PolicyHints(semantic=True)
    violations = ready_violations(
        hints, answer_ready=True, successful_calls=0, resolved_output="42"
    )
    assert violations == [
        "semantic question must complete at least one successful llm_query/batch_llm_query subcall."
    ]
    assert (
        ready_violations(hints, answer_ready=True, successful_calls=1, resolved_output="42") == []
    )


def test_ready_violations_semantic_rejects_incomplete_structured_evidence() -> None:
    violations = ready_violations(
        PolicyHints(semantic=True),
        answer_ready=True,
        successful_calls=3,
        resolved_output="partial",
        unresolved_semantic_batches=2,
        unresolved_semantic_items=3,
    )

    assert violations == [
        "incomplete structured semantic batch evidence remains unresolved "
        "(3 failed item(s) across 2 batch request(s)); rerun each exact request "
        "successfully before confirming the answer."
    ]


def test_semantic_contract_allows_inspection_before_ready_gate() -> None:
    hints = PolicyHints(semantic=True)
    assert contract_violations("print(context['files'][0]['text'][:100])", hints) == []


def test_ready_violations_numeric_output_gate() -> None:
    hints = PolicyHints(numeric_output=True)
    ok = ready_violations(hints, answer_ready=True, successful_calls=0, resolved_output="12.5%")
    assert ok == []
    bad = ready_violations(hints, answer_ready=True, successful_calls=0, resolved_output="about 12")
    assert bad == ["output must be a single number (optionally with %)."]


def test_ready_violations_reports_both_when_both_trip() -> None:
    hints = PolicyHints(semantic=True, numeric_output=True)
    violations = ready_violations(
        hints, answer_ready=True, successful_calls=0, resolved_output="prose"
    )
    assert len(violations) == 2


def test_build_initial_messages_shape() -> None:
    messages = build_initial_messages("SYS", "Question: q")
    assert messages == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Question: q"},
    ]


def test_build_refinement_messages_appends_without_mutating() -> None:
    base = build_initial_messages("SYS", "Q")
    out = build_refinement_messages(
        base,
        template="answer: {current_content} / output: {last_output}",
        code="print(1)",
        answer_content="draft",
        last_output="1",
    )
    assert len(base) == 2  # input untouched
    assert out[2] == {"role": "assistant", "content": "```python\nprint(1)\n```"}
    assert out[3]["content"] == "answer: draft / output: 1"


def test_build_refinement_messages_nudges_on_empty_output() -> None:
    out = build_refinement_messages(
        [],
        template="{current_content}|{last_output}",
        code="pass",
        answer_content="",
        last_output="",
    )
    assert EMPTY_OUTPUT_NUDGE in out[1]["content"]


def test_build_missing_code_repair_messages() -> None:
    out = build_missing_code_repair_messages([{"role": "user", "content": "Q"}], "prose reply")
    assert out[1] == {"role": "assistant", "content": "prose reply"}
    assert "```python code block" in out[2]["content"]


def test_build_error_repair_messages_policy_vs_plain_error() -> None:
    policy = build_error_repair_messages([], "c", PolicyError("Policy violation: x"))
    assert "answer['content'] was kept" in policy[1]["content"]
    plain = build_error_repair_messages([], "c", ValueError("boom"))
    assert "Fix the code and try again." in plain[1]["content"]
    assert "boom" in plain[1]["content"]


def test_strip_cache_anchor_markers_returns_clean_shallow_copy() -> None:
    messages = [
        {"role": "system", "content": "SYS", CACHE_ANCHOR_MARKER: True},
        {"role": "user", "content": "Q"},
    ]

    outbound = strip_cache_anchor_markers(messages)

    assert outbound == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Q"},
    ]
    assert CACHE_ANCHOR_MARKER in messages[0]
    assert outbound is not messages
    assert all(clean is not original for clean, original in zip(outbound, messages, strict=True))


def test_call_root_cache_anchors_never_alias_canonical_or_iteration_record() -> None:
    class CapturingLLM:
        def __init__(self) -> None:
            self.calls = []

        def responses_create(self, messages, **_kwargs):
            self.calls.append(messages)
            return "response", TokenUsage(1, 1, 2)

    client = CapturingLLM()
    context = create_execution_context()
    messages = build_initial_messages("SYS", "Q")
    canonical_bytes = json.dumps(messages, sort_keys=True)

    _response, usage, error = call_root(
        client,
        messages,
        model="model",
        context=context,  # type: ignore[arg-type]
    )
    assert error is None
    assert [CACHE_ANCHOR_MARKER in message for message in client.calls[0]] == [True, True]
    assert json.dumps(messages, sort_keys=True) == canonical_bytes

    record = record_iteration(
        iteration=1,
        messages=messages,
        response="response",
        code="pass",
        outcome=StepOutcome(output="", answer={}),
        usage=usage,
    )
    record_bytes = json.dumps(asdict(record), sort_keys=True)
    later_messages = build_refinement_messages(
        messages,
        template="{current_content}|{last_output}",
        code="pass",
        answer_content="draft",
        last_output="",
    )
    _response, _usage, error = call_root(
        client,
        later_messages,
        model="model",
        context=context,  # type: ignore[arg-type]
    )
    assert error is None
    assert json.dumps(asdict(record), sort_keys=True) == record_bytes
    assert all(CACHE_ANCHOR_MARKER not in message for message in record.llm_input)


def _window_fixture(count: int) -> tuple[list[dict[str, str]], tuple[TranscriptWindowEntry, ...]]:
    messages = build_initial_messages("SYS", "Q")
    entries: list[TranscriptWindowEntry] = []
    for iteration in range(1, count + 1):
        messages += [
            {"role": "assistant", "content": f"code-{iteration}"},
            {"role": "user", "content": f"verbatim-{iteration}"},
        ]
        entries.append(
            TranscriptWindowEntry(
                iteration=iteration,
                message_index=len(messages) - 1,
                elided_content=f"elided-{iteration}",
            )
        )
    return messages, tuple(entries)


def test_live_transcript_elision_is_append_only_and_frozen() -> None:
    messages, entries = _window_fixture(10)
    earlier, earlier_frontier = project_live_transcript(messages[:8], entries[:3])
    later, later_frontier = project_live_transcript(messages, entries)

    assert earlier_frontier == entries[0].message_index
    assert later_frontier == entries[-3].message_index
    assert later[: earlier_frontier + 1] == earlier[: earlier_frontier + 1]
    assert earlier[entries[0].message_index]["content"] == "elided-1"
    assert later[entries[0].message_index]["content"] == "elided-1"
    assert later[entries[-2].message_index]["content"] == "verbatim-9"
    assert later[entries[-1].message_index]["content"] == "verbatim-10"


def test_live_transcript_projection_and_frontier_anchor_do_not_alias_canonical() -> None:
    class CapturingLLM:
        def __init__(self) -> None:
            self.calls = []

        def responses_create(self, messages, **_kwargs):
            self.calls.append(messages)
            return "response", TokenUsage(1, 1, 2)

    messages, entries = _window_fixture(5)
    canonical_bytes = json.dumps(messages, sort_keys=True)
    context = create_execution_context()
    client = CapturingLLM()

    _response, _usage, error = call_root(
        client,
        messages,
        model="model",
        context=context,  # type: ignore[arg-type]
        transcript_window=entries,
    )

    assert error is None
    frontier = entries[-3].message_index
    anchored = [
        index for index, message in enumerate(client.calls[0]) if CACHE_ANCHOR_MARKER in message
    ]
    assert anchored == [0, frontier]
    assert frontier < entries[-2].message_index
    assert json.dumps(messages, sort_keys=True) == canonical_bytes
    assert all(CACHE_ANCHOR_MARKER not in message for message in messages)
    client.calls[0][frontier]["content"] = "mutated outbound"
    assert json.dumps(messages, sort_keys=True) == canonical_bytes


def test_historical_stdout_elision_is_bounded_and_deterministic() -> None:
    output = "a" * 1_500 + "b" * 1_500
    placeholder = "<elided>"

    projected = elide_historical_stdout(output, placeholder=placeholder)

    assert len(projected) == 2_000
    assert projected == elide_historical_stdout(output, placeholder=placeholder)
    assert projected.startswith("a")
    assert placeholder in projected
    assert projected.endswith("b")


def test_record_iteration_snapshots_messages() -> None:
    messages = [{"role": "user", "content": "Q"}]
    record = record_iteration(
        iteration=1,
        messages=messages,
        response="R",
        code="print(1)",
        outcome=StepOutcome(output="1", answer={}, stdout_chars=1),
        usage=None,
    )
    messages.append({"role": "user", "content": "later turn"})
    messages[0]["content"] = "mutated"
    assert record.llm_input == [{"role": "user", "content": "Q"}]
    assert record.execution_result == "1"
    assert record.execution_status == "success"
    assert record.stdout_chars == 1


def test_record_iteration_normalizes_empty_output_to_nudge() -> None:
    record = record_iteration(
        iteration=2,
        messages=[],
        response="R",
        code="pass",
        outcome=StepOutcome(output="", answer={}),
        usage=None,
    )
    assert record.execution_result == EMPTY_OUTPUT_NUDGE


def test_record_iteration_keeps_error_text_and_status_separate() -> None:
    record = record_iteration(
        iteration=2,
        messages=[],
        response="R",
        code="raise ValueError('boom')",
        outcome=StepOutcome(
            output="ERROR: boom",
            answer={},
            error=RLMError(type="ValueError", message="boom"),
        ),
        usage=None,
    )
    assert record.execution_result == "ERROR: boom"
    assert record.execution_status == "error"
    assert record.stdout_chars == 0


def test_iteration_record_positional_construction_remains_compatible() -> None:
    record = IterationRecord(1, [], "R", "pass", "legacy output", 2)
    assert record.execution_result == "legacy output"
    assert record.execution_status == "error"


def test_finalize_reads_stats_and_readiness() -> None:
    context = create_execution_context()
    context.stats.total_tokens = 7
    context.stats.calls_made = 3
    context.stats.successful_calls = 2
    result = finalize(
        answer_text="42",
        answer={"content": "42", "ready": True},
        iterations=2,
        context=context,
        trajectory=[],
        answer_metadata={"source": "result-1"},
    )
    assert result.answer == "42"
    assert result.ready is True
    assert result.tokens_used == 7
    assert result.sub_calls_made == 3
    assert result.sub_calls_succeeded == 2
    assert result.error is None and result.extracted is False
    assert result.answer_metadata == {"source": "result-1"}
