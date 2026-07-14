from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from droste import (
    DataUseAuthorization,
    PolicyHints,
    RLMConfig,
    TraceRetentionPolicy,
    create_execution_context,
    parse_event,
    run_rlm,
)
from droste.execution.trace import PersistenceClass, TraceRecorder
from droste.protocols.llm_client import TokenUsage
from droste.testing import MockEnvironment, MockLLMClient, MockResponse, MockSubcallClient


def _ready_reply(marker: str = "ok") -> MockResponse:
    return MockResponse(
        text=(
            "```python\n"
            f"print({marker!r})\n"
            f"answer['content'] = {marker!r}\n"
            "answer['ready'] = True\n"
            "```"
        ),
        usage=TokenUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
    )


def test_envelope_orders_parent_child_values_and_freezes_bodies() -> None:
    now = datetime(2026, 7, 14, tzinfo=timezone.utc)
    ticks = iter((now, now + timedelta(milliseconds=1)))
    recorder = TraceRecorder(
        run_id="child",
        parent_run_id="parent",
        depth=2,
        clock=lambda: next(ticks),
    )

    first = recorder.append({"type": "progress", "status": "starting"})
    second = recorder.append({"type": "capability", "outcome": {"ok": True}})

    assert [first.seq, second.seq] == [1, 2]
    assert datetime.fromisoformat(first.timestamp) < datetime.fromisoformat(second.timestamp)
    assert second.parent_run_id == "parent"
    assert second.depth == 2
    assert second.persistence_class is PersistenceClass.DURABLE
    with pytest.raises(TypeError):
        second.body["outcome"]["ok"] = False  # type: ignore[index]


def test_concurrent_append_assigns_one_unique_monotonic_sequence() -> None:
    recorder = TraceRecorder(run_id="concurrent")
    with ThreadPoolExecutor(max_workers=16) as pool:
        events = list(
            pool.map(
                lambda index: recorder.append({"type": "progress", "index": index}),
                range(200),
            )
        )

    assert sorted(event.seq for event in events) == list(range(1, 201))
    assert [event.seq for event in recorder.events] == list(range(1, 201))


def test_run_record_allows_repeated_durable_budget_mutations() -> None:
    recorder = TraceRecorder(run_id="budget-ledger")
    recorder.append({"type": "budget", "action": "reserve", "amount": 2})
    recorder.append({"type": "budget", "action": "commit", "amount": 1})
    recorder.append({"type": "budget", "action": "release", "amount": 1})
    record = recorder.finish({"status": "success"})

    assert [event.body["action"] for event in record.events if event.type == "budget"] == [
        "reserve",
        "commit",
        "release",
    ]


def test_parser_requires_v1_envelope_and_rejects_false_classification() -> None:
    with pytest.raises(ValueError, match="missing envelope fields"):
        parse_event({"type": "code", "iteration": 1, "code": "print(1)"})

    parsed = parse_event(
        {
            "type": "code",
            "run_id": "run",
            "seq": 1,
            "timestamp": "2026-07-14T00:00:00Z",
            "version": 1,
            "persistence_class": "configurable",
            "iteration": 1,
            "code": "print(1)",
        }
    )
    assert parsed.body["code"] == "print(1)"

    with pytest.raises(ValueError, match="must be configurable"):
        parse_event(
            {
                "type": "code",
                "run_id": "run",
                "seq": 1,
                "timestamp": "2026-07-14T00:00:00Z",
                "version": 1,
                "persistence_class": "durable",
                "code": "print(1)",
            }
        )


def test_trace_values_reject_non_string_object_keys() -> None:
    recorder = TraceRecorder(run_id="strict-json")
    with pytest.raises(TypeError, match="keys must be strings"):
        recorder.append({"type": "capability", "outcome": {1: "collision", "1": "other"}})

    with pytest.raises(ValueError, match="envelope authority"):
        recorder.append({"type": "progress", "run_id": "second-owner"})


def test_default_retention_cannot_smuggle_code_output_or_answer_into_done() -> None:
    secret = "PRIVATE_MARKER_13"
    result = run_rlm(
        question="q",
        environment=MockEnvironment(),
        root_llm=MockLLMClient([_ready_reply(secret)]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(max_iterations=1),
    )

    assert result.run_record is not None
    wire = result.run_record.as_dict()
    assert {event["type"] for event in wire["events"]} == {
        "usage",
        "budget",
        "policy",
        "done",
    }
    assert secret not in json.dumps(wire)
    assert "trajectory" not in wire["terminal"]
    assert "answer" not in wire["terminal"]


def test_configurable_replay_is_retained_only_when_selected() -> None:
    result = run_rlm(
        question="q",
        environment=MockEnvironment(),
        root_llm=MockLLMClient([_ready_reply("selected")]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(
            max_iterations=1,
            trace_retention=TraceRetentionPolicy(frozenset({"code", "output", "replay"})),
        ),
    )

    assert result.run_record is not None
    wire = result.run_record.as_dict()
    types = [event["type"] for event in wire["events"]]
    assert "code" in types
    assert "output" in types
    assert "replay" in types
    assert "selected" in json.dumps(wire)


def test_repair_attempts_are_first_class_configurable_events() -> None:
    missing_code = MockResponse(
        text="I forgot the code block.",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    result = run_rlm(
        question="q",
        environment=MockEnvironment(),
        root_llm=MockLLMClient([missing_code, _ready_reply()]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(
            max_iterations=1,
            enforce_contract=True,
            trace_retention=TraceRetentionPolicy(frozenset({"repair"})),
        ),
    )

    assert result.run_record is not None
    repair = next(event for event in result.run_record.events if event.type == "repair")
    assert repair.body["reason"] == "missing_code"
    assert repair.persistence_class is PersistenceClass.CONFIGURABLE


def test_terminal_reconciles_usage_and_training_permission_is_independent() -> None:
    result = run_rlm(
        question="q",
        environment=MockEnvironment(),
        root_llm=MockLLMClient([_ready_reply()]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(
            max_iterations=1,
            trace_retention=TraceRetentionPolicy(frozenset({"replay"})),
        ),
    )

    assert result.run_record is not None
    record = result.run_record
    usage = next(event for event in record.events if event.type == "usage")
    assert (
        usage.body["tokens"]["total"]
        == result.tokens_used
        == record.terminal["usage"]["tokens"]["total"]
    )
    assert record.terminal["retention"]["replay_retained"] is True
    assert record.data_use == DataUseAuthorization(training_allowed=False)
    assert record.retention.retain == frozenset({"replay"})


def test_host_record_sink_receives_explicit_training_authorization_without_content() -> None:
    records = []
    result = run_rlm(
        question="q",
        environment=MockEnvironment(),
        root_llm=MockLLMClient([_ready_reply()]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(
            max_iterations=1,
            data_use=DataUseAuthorization(training_allowed=True),
            on_run_record=records.append,
        ),
    )

    assert records == [result.run_record]
    assert records[0].data_use.training_allowed is True
    assert records[0].retention.retain == frozenset()
    assert all(event.persistence_class is PersistenceClass.DURABLE for event in records[0].events)


def test_non_policy_failure_is_not_recorded_as_policy_violation() -> None:
    result = run_rlm(
        question="q",
        environment=MockEnvironment(),
        root_llm=MockLLMClient([]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(max_iterations=1, enforce_contract=True),
    )

    assert result.error is not None and result.error.type != "PolicyError"
    assert result.run_record is not None
    policy = next(event for event in result.run_record.events if event.type == "policy")
    assert policy.body["outcome"] == "not_evaluated"
    assert policy.body["violation_type"] is None


def test_withheld_policy_content_cannot_enter_durable_terminal_errors() -> None:
    secret = "WITHHELD_PRIVATE_CONTENT"
    violating = MockResponse(
        text=(f"```python\nanswer['content'] = {secret!r}\nanswer['ready'] = True\n```"),
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    result = run_rlm(
        question="q",
        environment=MockEnvironment(),
        root_llm=MockLLMClient([violating, violating]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(
            max_iterations=1,
            enforce_contract=True,
            policy_hints=PolicyHints(numeric_output=True),
        ),
    )

    assert result.error is not None and result.error.type == "PolicyError"
    assert result.error.details is not None
    assert result.error.details["withheld_content"] == secret
    assert result.run_record is not None
    assert secret not in json.dumps(result.run_record.as_dict())
    assert result.run_record.terminal["error"] == {"type": "PolicyError"}


def test_recovered_policy_violation_remains_a_durable_policy_fact() -> None:
    violating = MockResponse(
        text=("```python\nanswer['content'] = 'draft'\nanswer['ready'] = True\n```"),
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    extracted = MockResponse(
        text="Recovered from the retained evidence.",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    result = run_rlm(
        question="q",
        environment=MockEnvironment(),
        root_llm=MockLLMClient([violating, violating, extracted]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(
            max_iterations=1,
            enforce_contract=True,
            policy_hints=PolicyHints(numeric_output=True),
        ),
    )

    assert result.error is None
    assert result.recovered_error is not None
    assert result.recovered_error.type == "PolicyError"
    assert result.run_record is not None
    policy = next(event for event in result.run_record.events if event.type == "policy")
    assert policy.body["outcome"] == "violated"


def test_injected_context_owns_trace_settings_and_conflicts_fail() -> None:
    context = create_execution_context(
        run_id="context-run",
        trace_retention=TraceRetentionPolicy(frozenset({"code"})),
    )
    with pytest.raises(ValueError, match="run_id, trace_retention"):
        run_rlm(
            question="q",
            environment=MockEnvironment(),
            root_llm=MockLLMClient([_ready_reply()]),
            subcalls=MockSubcallClient(),
            config=RLMConfig(
                max_iterations=1,
                run_id="config-run",
                trace_retention=TraceRetentionPolicy(frozenset({"output"})),
            ),
            context=context,
        )
