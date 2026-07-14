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


def _capability_outcome() -> dict[str, object]:
    return {
        "version": 1,
        "capability_id": {
            "kind": "inference",
            "provider_type": None,
            "source_id": None,
            "operation": "query",
        },
        "call_id": "call-1",
        "run_id": "child",
        "parent_run_id": "parent",
        "child_run_id": None,
        "ok": True,
        "status": "ok",
        "error": None,
        "usage": [],
        "budget_delta": [],
        "evidence": {"count": 0},
        "result_handle": None,
    }


def _terminal() -> dict[str, object]:
    usage = {
        "kind": "resolved",
        "root": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
            "successes": 0,
        },
        "subcall": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
            "successes": 0,
        },
        "unattributed": {"total_tokens": 0},
        "total_tokens": 0,
        "wall_time_ms": 0,
    }
    budget = {
        "kind": "snapshot",
        "source": "test",
        "configured": {},
        "consumed": {},
        "remaining": {},
    }
    policy = {
        "contract_enforced": True,
        "outcome": "passed",
        "violation_type": None,
    }
    return {
        "status": "success",
        "ready": True,
        "extracted": False,
        "iterations": 0,
        "usage": usage,
        "budget": budget,
        "policy": policy,
        "retention": {
            "policy_id": "default-no-content",
            "retain": [],
            "expires_at": None,
            "host_managed_expiry": False,
            "replay_retained": False,
        },
        "error": None,
        "extract_error": None,
        "recovered_error": None,
    }


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
    second = recorder.append({"type": "capability", "outcome": _capability_outcome()})

    assert [first.seq, second.seq] == [1, 2]
    assert datetime.fromisoformat(first.timestamp) < datetime.fromisoformat(second.timestamp)
    assert second.parent_run_id == "parent"
    assert second.depth == 2
    assert second.persistence_class is PersistenceClass.DURABLE
    with pytest.raises(TypeError):
        second.body["outcome"]["ok"] = False  # type: ignore[index]

    with pytest.raises(ValueError, match="child traces require parent_run_id"):
        TraceRecorder(run_id="orphan", depth=1)
    with pytest.raises(ValueError, match="root traces cannot have parent_run_id"):
        TraceRecorder(run_id="false-root", parent_run_id="parent", depth=0)


def test_concurrent_append_assigns_one_unique_monotonic_sequence() -> None:
    recorder = TraceRecorder(run_id="concurrent")
    with ThreadPoolExecutor(max_workers=16) as pool:
        events = list(
            pool.map(
                lambda index: recorder.append({"type": "progress", "status": str(index)}),
                range(200),
            )
        )

    assert sorted(event.seq for event in events) == list(range(1, 201))
    assert [event.seq for event in recorder.events] == list(range(1, 201))


def test_run_record_allows_repeated_durable_budget_mutations() -> None:
    recorder = TraceRecorder(run_id="budget-ledger")
    recorder.append(
        {
            "type": "budget",
            "kind": "mutation",
            "source": "test",
            "action": "reserve",
            "resource": "subcalls",
            "amount": 2,
        }
    )
    recorder.append(
        {
            "type": "budget",
            "kind": "mutation",
            "source": "test",
            "action": "commit",
            "resource": "subcalls",
            "amount": 1,
        }
    )
    recorder.append(
        {
            "type": "budget",
            "kind": "mutation",
            "source": "test",
            "action": "refund",
            "resource": "subcalls",
            "amount": 1,
        }
    )
    record = recorder.finish(_terminal())

    assert [event.body["action"] for event in record.events if event.type == "budget"] == [
        "reserve",
        "commit",
        "refund",
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
            "depth": 0,
            "iteration": 1,
            "code": "print(1)",
        }
    )
    assert parsed.body["code"] == "print(1)"

    startup = parse_event(
        {
            "type": "startup",
            "run_id": "run",
            "seq": 2,
            "timestamp": "2026-07-14T00:00:00Z",
            "version": 1,
            "persistence_class": "transient",
            "depth": 0,
            "engine_version": "0.10.6",
            "runner_protocol": 2,
            "provider_protocol": 3,
        }
    )
    assert startup.body["provider_protocol"] == 3

    with pytest.raises(ValueError, match="must be configurable"):
        parse_event(
            {
                "type": "code",
                "run_id": "run",
                "seq": 1,
                "timestamp": "2026-07-14T00:00:00Z",
                "version": 1,
                "persistence_class": "durable",
                "depth": 0,
                "code": "print(1)",
            }
        )

    with pytest.raises(ValueError, match="timestamp must be UTC"):
        parse_event(
            {
                "type": "progress",
                "run_id": "run",
                "seq": 1,
                "timestamp": "2026-07-14T01:00:00+01:00",
                "version": 1,
                "persistence_class": "transient",
                "depth": 0,
                "status": "working",
            }
        )


def test_event_bodies_reject_missing_unknown_and_wrong_primitive_fields() -> None:
    recorder = TraceRecorder(run_id="strict-body")
    with pytest.raises(ValueError, match="missing body fields: status"):
        recorder.append({"type": "progress"})
    with pytest.raises(ValueError, match="unknown body fields: detail"):
        recorder.append({"type": "progress", "status": "working", "detail": "private"})
    with pytest.raises(TypeError, match="iteration.*invalid type"):
        recorder.append({"type": "iteration_start", "iteration": True, "max_iterations": 2})
    with pytest.raises(ValueError, match="unknown body fields: details"):
        recorder.append(
            {
                "type": "budget",
                "kind": "mutation",
                "source": "test",
                "action": "reserve",
                "resource": "subcalls",
                "amount": 1,
                "details": {"message": "private provider detail"},
            }
        )

    terminal = _terminal()
    terminal["error"] = {"type": "ProviderError", "message": "private provider detail"}
    with pytest.raises(ValueError, match="done.error has unknown message"):
        recorder.finish(terminal)


def test_trace_values_reject_non_string_object_keys() -> None:
    recorder = TraceRecorder(run_id="strict-json")
    with pytest.raises(TypeError, match="keys must be strings"):
        recorder.append({"type": "capability", "outcome": {1: "collision", "1": "other"}})

    with pytest.raises(ValueError, match="envelope authority"):
        recorder.append({"type": "progress", "run_id": "second-owner"})


def test_default_retention_cannot_smuggle_code_output_or_answer_into_done() -> None:
    secret = "PRIVATE_MARKER_13"
    events: list[dict[str, object]] = []
    result = run_rlm(
        question="q",
        environment=MockEnvironment(),
        root_llm=MockLLMClient([_ready_reply(secret)]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(max_iterations=1),
        on_event=events.append,
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
    live_result = next(event for event in events if event["type"] == "result")
    assert live_result["result"]["answer"] == secret  # type: ignore[index]
    assert live_result["depth"] == 0
    assert result.run_record.depth == 0


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
        usage.body["total_tokens"] == result.tokens_used == record.terminal["usage"]["total_tokens"]
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
            data_use=DataUseAuthorization(
                training_allowed=True,
                authorization_ref="consent://trace-training/1",
                purposes=frozenset({"training"}),
            ),
            on_run_record=records.append,
        ),
    )

    assert records == [result.run_record]
    assert records[0].data_use.training_allowed is True
    assert records[0].retention.retain == frozenset()
    assert all(event.persistence_class is PersistenceClass.DURABLE for event in records[0].events)


def test_retention_expiry_and_training_authorization_are_explicit_governance_facts() -> None:
    with pytest.raises(ValueError, match="requires host_managed_expiry"):
        TraceRetentionPolicy(expires_at="2026-10-14T00:00:00Z")
    with pytest.raises(ValueError, match="purpose 'training'"):
        DataUseAuthorization(training_allowed=True, authorization_ref="consent://trace/1")
    with pytest.raises(ValueError, match="requires training_allowed=True"):
        DataUseAuthorization(
            training_allowed=False,
            authorization_ref="consent://trace/1",
            purposes=frozenset({"training"}),
        )

    retention = TraceRetentionPolicy(
        retain=frozenset({"replay"}),
        policy_id="local-training-v1",
        expires_at="2026-10-14T00:00:00Z",
        host_managed_expiry=True,
    )
    authorization = DataUseAuthorization(
        training_allowed=True,
        authorization_ref="consent://trace/1",
        purposes=frozenset({"training"}),
    )

    assert retention.as_dict() == {
        "policy_id": "local-training-v1",
        "retain": ["replay"],
        "expires_at": "2026-10-14T00:00:00Z",
        "host_managed_expiry": True,
    }
    assert authorization.as_dict() == {
        "training_allowed": True,
        "authorization_ref": "consent://trace/1",
        "purposes": ["training"],
    }


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
