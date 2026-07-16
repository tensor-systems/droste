from __future__ import annotations

import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event, Thread

import pytest

from droste import (
    Budget,
    BudgetRequest,
    DataUseAuthorization,
    RLMConfig,
    RunEvent,
    RunRecord,
    ScaffoldManifest,
    TraceRetentionPolicy,
    create_execution_context,
    parse_event,
    run_rlm,
)
from droste.capabilities import broker_subcalls
from droste.exceptions import RLMError
from droste.execution.trace import PersistenceClass, TraceRecorder, select_retained_events
from droste.loop.step import finalize
from droste.protocols.llm_client import TokenUsage
from droste.testing import (
    MockEnvironment,
    MockLLMClient,
    MockResponse,
    MockSubcallClient,
    runner_v6_refusal_ndjson,
    trace_v2_execution_ndjson,
    trace_v2_lifecycle_ndjson,
)
from droste.testing._trace_fixtures import build_trace_v2_execution_ndjson
from droste_runner.run import run as run_worker


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


def test_live_delivery_preserves_recorded_order_across_concurrent_emitters() -> None:
    first_delivering = Event()
    second_attempted = Event()
    second_delivered = Event()
    delivered: list[int] = []
    raced: list[bool] = []

    def sink(event: dict[str, object]) -> None:
        seq = event["seq"]
        assert isinstance(seq, int)
        if seq == 1:
            first_delivering.set()
            assert second_attempted.wait(timeout=2)
            raced.append(second_delivered.wait(timeout=0.1))
        delivered.append(seq)
        if seq == 2:
            second_delivered.set()

    context = create_execution_context(run_id="live-order", on_event=sink)
    first = Thread(target=lambda: context.emit_event({"type": "progress", "status": "one"}))
    first.start()
    assert first_delivering.wait(timeout=2)

    def emit_second() -> None:
        second_attempted.set()
        context.emit_event({"type": "progress", "status": "two"})

    second = Thread(target=emit_second)
    second.start()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive() and not second.is_alive()
    assert raced == [False]
    assert delivered == [1, 2]
    assert [event.seq for event in context.trace.events] == [1, 2]


def test_concurrent_subcalls_share_broker_identity_and_trace_order() -> None:
    events: list[dict[str, object]] = []
    context = create_execution_context(
        budget=Budget(tokens=100_000, subcalls=20),
        on_event=events.append,
    )
    context.begin_iteration(3)
    subcalls = broker_subcalls(
        MockSubcallClient(context=context),
        context.ledger,
        attempt_observer=context.observe_capability_attempt,
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert (
            list(pool.map(subcalls.llm_query, [f"prompt-{index}" for index in range(8)]))
            == [""] * 8
        )

    lifecycle = [event for event in events if event["type"] == "subcall"]
    assert len(lifecycle) == 16
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    by_call: dict[str, list[dict[str, object]]] = {}
    for event in lifecycle:
        by_call.setdefault(str(event["call_id"]), []).append(event)
    assert len(by_call) == 8
    assert all(
        [event["phase"] for event in pair] == ["start", "completion"] for pair in by_call.values()
    )
    assert all(event["iteration"] == 3 and event["depth"] == 0 for event in lifecycle)


def test_batch_fanout_uses_call_identity_without_body_sequence() -> None:
    events: list[dict[str, object]] = []
    context = create_execution_context(
        budget=Budget(tokens=100_000, subcalls=10),
        on_event=events.append,
    )
    context.begin_iteration(2)
    subcalls = broker_subcalls(
        MockSubcallClient(context=context),
        context.ledger,
        attempt_observer=context.observe_capability_attempt,
    )

    assert subcalls.llm_batch(["a", "b", "c"]) == ["", "", ""]

    lifecycle = [event for event in events if event["type"] == "subcall"]
    assert [event["phase"] for event in lifecycle] == ["start", "completion"]
    assert lifecycle[0]["call_id"] == lifecycle[1]["call_id"]
    assert lifecycle[0]["batch_count"] == lifecycle[1]["batch_count"] == 3
    assert all("batch_id" not in event and "batch_index" not in event for event in lifecycle)
    assert all(set(event).isdisjoint({"prompt", "context", "result"}) for event in lifecycle)


def test_empty_batch_still_reports_its_explicit_zero_count() -> None:
    events: list[dict[str, object]] = []
    context = create_execution_context(
        budget=Budget(tokens=100_000, subcalls=10),
        on_event=events.append,
    )
    context.begin_iteration(1)
    subcalls = broker_subcalls(
        MockSubcallClient(context=context),
        context.ledger,
        attempt_observer=context.observe_capability_attempt,
    )

    assert subcalls.llm_batch([]) == []

    lifecycle = [event for event in events if event["type"] == "subcall"]
    assert [event["phase"] for event in lifecycle] == ["start", "completion"]
    assert [event["batch_count"] for event in lifecycle] == [0, 0]


def test_direct_subcall_outside_a_run_does_not_invent_an_iteration() -> None:
    events: list[dict[str, object]] = []
    context = create_execution_context(
        budget=Budget(tokens=100_000, subcalls=10),
        on_event=events.append,
    )
    subcalls = broker_subcalls(
        MockSubcallClient(context=context),
        context.ledger,
        attempt_observer=context.observe_capability_attempt,
    )

    assert subcalls.llm_query("prompt") == ""

    assert all(event["type"] != "subcall" for event in events)


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


def test_parser_requires_v2_envelope_and_rejects_false_classification() -> None:
    with pytest.raises(ValueError, match="missing envelope fields"):
        parse_event({"type": "code", "iteration": 1, "code": "print(1)"})

    parsed = parse_event(
        {
            "type": "code",
            "run_id": "run",
            "seq": 1,
            "timestamp": "2026-07-14T00:00:00Z",
            "version": 2,
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
            "version": 2,
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
                "version": 2,
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
                "version": 2,
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
        recorder.append({"type": "iteration_start", "iteration": True, "remaining_tokens": 2})
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


def test_lifecycle_bodies_are_strict_discriminated_values() -> None:
    recorder = TraceRecorder(run_id="lifecycle")
    with pytest.raises(ValueError, match="subcall start requires reservation"):
        recorder.append(
            {
                "type": "subcall",
                "phase": "start",
                "call_id": "call-1",
                "operation": "llm_query",
                "iteration": 1,
            }
        )
    with pytest.raises(ValueError, match="subcall failure requires checkpoint and error"):
        recorder.append(
            {
                "type": "subcall",
                "phase": "failure",
                "call_id": "call-1",
                "operation": "llm_query",
                "iteration": 1,
                "checkpoint": {"tokens": 1, "subcalls": 1},
            }
        )
    with pytest.raises(ValueError, match="unknown body fields: batch_id"):
        recorder.append(
            {
                "type": "subcall",
                "phase": "start",
                "call_id": "call-1",
                "operation": "llm_batch",
                "iteration": 1,
                "reservation": {"tokens": 1, "subcalls": 1, "wall_ms": 1, "depth": 0},
                "batch_id": "redundant-call-identity",
            }
        )
    with pytest.raises(ValueError, match="batch subcall requires batch_count"):
        recorder.append(
            {
                "type": "subcall",
                "phase": "start",
                "call_id": "call-1",
                "operation": "llm_batch",
                "iteration": 1,
                "reservation": {"tokens": 1, "subcalls": 1, "wall_ms": 1, "depth": 0},
            }
        )
    with pytest.raises(ValueError, match="unary subcall cannot carry batch_count"):
        recorder.append(
            {
                "type": "subcall",
                "phase": "start",
                "call_id": "call-1",
                "operation": "llm_query",
                "iteration": 1,
                "reservation": {"tokens": 1, "subcalls": 1, "wall_ms": 1, "depth": 0},
                "batch_count": 1,
            }
        )
    with pytest.raises(TypeError, match="field 'iteration' has invalid type bool"):
        recorder.append(
            {
                "type": "subcall",
                "phase": "start",
                "call_id": "call-1",
                "operation": "llm_query",
                "iteration": True,
                "reservation": {"tokens": 1, "subcalls": 1, "wall_ms": 1, "depth": 0},
            }
        )
    with pytest.raises(TypeError, match="field 'batch_count' has invalid type bool"):
        recorder.append(
            {
                "type": "subcall",
                "phase": "start",
                "call_id": "call-1",
                "operation": "llm_batch",
                "iteration": 1,
                "reservation": {"tokens": 1, "subcalls": 1, "wall_ms": 1, "depth": 0},
                "batch_count": True,
            }
        )
    with pytest.raises(TypeError, match="field 'iteration' has invalid type bool"):
        recorder.append(
            {
                "type": "repair",
                "phase": "start",
                "kind": "execution_error",
                "iteration": True,
            }
        )
    with pytest.raises(TypeError, match="field 'iteration' has invalid type bool"):
        recorder.append({"type": "extract", "phase": "start", "iteration": True})
    with pytest.raises(ValueError, match="unknown body fields: detail"):
        recorder.append(
            {
                "type": "repair",
                "phase": "start",
                "kind": "execution_error",
                "iteration": 1,
                "detail": "private",
            }
        )
    with pytest.raises(ValueError, match="extract failure requires extract_error"):
        recorder.append({"type": "extract", "phase": "failure", "iteration": 1})


def test_trace_abi_v1_is_rejected_instead_of_silently_reinterpreted() -> None:
    with pytest.raises(ValueError, match="unsupported trace event version: 1"):
        parse_event(
            {
                "type": "progress",
                "run_id": "old",
                "seq": 1,
                "timestamp": "2026-07-14T00:00:00Z",
                "version": 1,
                "persistence_class": "transient",
                "depth": 0,
                "status": "legacy",
            }
        )


def _trace_v2_golden_runs() -> dict[str, list[RunEvent]]:
    runs: dict[str, list[RunEvent]] = {}
    previous_run_id: str | None = None
    closed: set[str] = set()
    for line in trace_v2_lifecycle_ndjson().decode("utf-8").splitlines():
        event = parse_event(json.loads(line))
        if event.run_id != previous_run_id:
            if event.run_id in closed:
                raise AssertionError(f"non-contiguous golden run {event.run_id}")
            if previous_run_id is not None:
                closed.add(previous_run_id)
            previous_run_id = event.run_id
        runs.setdefault(event.run_id, []).append(event)
    return runs


def test_trace_v2_execution_fixture_is_canonical_and_exact() -> None:
    fixture = trace_v2_execution_ndjson()
    assert fixture == build_trace_v2_execution_ndjson()
    events = [parse_event(json.loads(line)) for line in fixture.decode("utf-8").splitlines()]

    root = events[:6]
    child = events[6:]
    assert [event.type for event in root] == [
        "llm_response",
        "code",
        "output",
        "llm_response",
        "code",
        "execution_error",
    ]
    assert [event.body["iteration"] for event in root] == [1, 1, 1, 2, 2, 2]
    assert [event.seq for event in root] == [1, 2, 3, 4, 5, 6]
    assert all(event.depth == 0 and event.parent_run_id is None for event in root)
    assert root[2].body["stdout"].startswith("ERROR:")
    assert root[2].type == "output"
    assert root[5].body["error_type"] == "ValueError"

    assert [event.type for event in child] == ["llm_response", "code", "output"]
    assert [event.seq for event in child] == [1, 2, 3]
    assert all(event.depth == 1 for event in child)
    assert all(event.parent_run_id == root[0].run_id for event in child)
    projected = [event for event in events if event.type in {"code", "output", "execution_error"}]
    assert len(projected) == 6


def _error_type(value: object) -> object:
    return value.get("type") if isinstance(value, Mapping) else None


def test_trace_v2_lifecycle_golden_ndjson_is_strict_ordered_and_terminal() -> None:
    runs = _trace_v2_golden_runs()
    assert set(runs) == {
        "golden-success",
        "golden-recovered",
        "golden-output-limit",
        "golden-extract-failed",
        "golden-cancelled",
    }

    for run_id, events in runs.items():
        assert [event.seq for event in events] == list(range(1, len(events) + 1))
        assert events[-1].type == "done"
        assert sum(event.type == "done" for event in events) == 1

        result_event = next(event for event in events if event.type == "result")
        result = result_event.body["result"]
        done = events[-1].body
        assert result["ready"] == done["ready"]
        assert result["extracted"] == done["extracted"]
        assert result["iterations"] == done["iterations"]
        assert result["tokens_used"] == done["usage"]["total_tokens"]
        assert result["subcalls"] == done["usage"]["subcall"]["requests"]
        assert result["successful_subcalls"] == done["usage"]["subcall"]["successes"]
        assert result["stdout_chars"] == done["stdout_chars"]
        scaffold_manifest = ScaffoldManifest.from_dict(result["scaffold_manifest"])
        assert scaffold_manifest.manifest_id == done["scaffold_manifest_id"]
        assert scaffold_manifest.schema_version == done["scaffold_manifest_version"] == 2
        assert scaffold_manifest.body["abis"] == {
            "capability": 1,
            "kernel": 1,
            "prompt_pack": 1,
            "provider": 4,
            "runner": 6,
            "trace": 2,
        }
        assert dict(scaffold_manifest.body["budget"]) == dict(done["budget"]["configured"])
        assert (
            f"sha256:{result['prompt_pack']['content_sha256']}"
            == scaffold_manifest.body["prompt_pack"]["content_hash"]
        )
        for key in ("error", "extract_error", "recovered_error"):
            assert _error_type(result[key]) == _error_type(done[key])

        for event_type in ("usage", "budget", "policy"):
            emitted = next(event for event in events if event.type == event_type)
            assert dict(emitted.body) == dict(done[event_type])

        budget = done["budget"]
        configured = Budget.from_dict(budget["configured"])
        consumed = BudgetRequest(**budget["consumed"])
        remaining = BudgetRequest(**budget["remaining"])
        assert remaining.tokens == configured.tokens - consumed.tokens
        assert remaining.subcalls == configured.subcalls - consumed.subcalls
        assert remaining.wall_ms == configured.wall_ms - consumed.wall_ms
        assert consumed.depth == 0
        assert remaining.depth == configured.depth

        reservation_wall_ms = [
            event.body["reservation"]["wall_ms"]
            for event in events
            if event.type == "subcall" and event.body["phase"] == "start"
        ]
        assert reservation_wall_ms == sorted(reservation_wall_ms, reverse=True)
        assert all(value >= remaining.wall_ms for value in reservation_wall_ms)

        raw_retention = done["retention"]
        retention = TraceRetentionPolicy(
            retain=frozenset(raw_retention["retain"]),
            policy_id=raw_retention["policy_id"],
            expires_at=raw_retention["expires_at"],
            host_managed_expiry=raw_retention["host_managed_expiry"],
        )
        record = RunRecord(
            run_id=run_id,
            events=select_retained_events(events, retention),
            terminal=done,
            retention=retention,
            data_use=DataUseAuthorization(),
        )
        assert record.events[-1].body == record.terminal

    startup = runs["golden-success"][0].body
    terminal = runs["golden-success"][-1].body
    successful_result = next(
        event.body["result"] for event in runs["golden-success"] if event.type == "result"
    )
    assert startup["scaffold_manifest_id"] == terminal["scaffold_manifest_id"]
    assert startup["scaffold_manifest_version"] == terminal["scaffold_manifest_version"]
    assert startup["runner_protocol"] == successful_result["scaffold_manifest"]["abis"]["runner"]


def test_trace_v2_golden_corpus_covers_each_discriminated_lifecycle() -> None:
    events = [event for run in _trace_v2_golden_runs().values() for event in run]
    subcalls = [event for event in events if event.type == "subcall"]
    repairs = [event for event in events if event.type == "repair"]
    extracts = [event for event in events if event.type == "extract"]

    assert {(event.body["operation"], event.body["phase"]) for event in subcalls} >= {
        ("llm_query", "completion"),
        ("llm_query", "failure"),
        ("llm_batch", "failure"),
    }
    assert any(
        event.body.get("error") == {"code": "handler_error", "type": "RuntimeError"}
        and event.body["call_id"] == "batch-failed"
        and event.body["checkpoint"] == {"tokens": 0, "subcalls": 0}
        for event in subcalls
    )
    assert any(
        event.body.get("error") == {"code": "cancelled", "type": "CapabilityCancelled"}
        and event.body["checkpoint"] == {"tokens": 0, "subcalls": 0}
        for event in subcalls
    )
    assert {event.body["phase"] for event in repairs} == {"start", "completion", "failure"}
    assert {event.body["phase"] for event in extracts} == {"start", "completion", "failure"}

    output_limit = _trace_v2_golden_runs()["golden-output-limit"]
    assert not any(event.type == "output" for event in output_limit)
    assert any(
        event.type == "execution_error" and event.body["error_type"] == "SandboxError"
        for event in output_limit
    )
    assert output_limit[-1].body["error"] == {"type": "RuntimeError"}
    assert output_limit[-1].body["budget"]["consumed"]["tokens"] == 416
    assert output_limit[-1].body["usage"]["root"]["successes"] == 1
    assert output_limit[-1].body["usage"]["root"]["requests"] == 2
    output_manifest = next(
        event.body["result"]["scaffold_manifest"]
        for event in output_limit
        if event.type == "result"
    )
    assert output_manifest["sandbox"] == {
        "capture_output_chars": 4097,
        "execution_timeout_ms": 0,
        "output_chars": 4096,
    }
    output_error = next(
        event.body["message"] for event in output_limit if event.type == "execution_error"
    )
    assert f"exceeded {output_manifest['sandbox']['output_chars']} characters" in output_error
    cancelled = _trace_v2_golden_runs()["golden-cancelled"]
    assert cancelled[-1].body["status"] == "error"
    assert any(
        event.type == "execution_error" and event.body["error_type"] == "CapabilityCallError"
        for event in cancelled
    )
    assert cancelled[-1].body["error"] == {"type": "RuntimeError"}
    assert cancelled[-1].body["budget"]["consumed"]["tokens"] == 372
    assert cancelled[-1].body["usage"]["root"]["successes"] == 1
    assert cancelled[-1].body["usage"]["root"]["requests"] == 2

    extract_failed = _trace_v2_golden_runs()["golden-extract-failed"]
    result = next(event.body["result"] for event in extract_failed if event.type == "result")
    assert result["answer"] == f"Error: {result['error']['message']}"
    assert result["error"]["details"]["withheld_content"] == "retained evidence"


def test_runner_refusal_fixture_is_the_exact_pre_admission_response() -> None:
    fixture = json.loads(runner_v6_refusal_ndjson())
    assert fixture == run_worker({})
    assert fixture["status"] == "refusal"
    assert fixture["run_record"] is None
    assert fixture["run_id"] is None


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
        config=RLMConfig(),
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
            enforce_contract=True,
            trace_retention=TraceRetentionPolicy(frozenset({"repair"})),
        ),
    )

    assert result.run_record is not None
    repair = next(event for event in result.run_record.events if event.type == "repair")
    assert repair.body == {
        "phase": "start",
        "kind": "missing_code",
        "iteration": 1,
    }
    assert repair.persistence_class is PersistenceClass.CONFIGURABLE


def test_terminal_reconciles_usage_and_training_permission_is_independent() -> None:
    result = run_rlm(
        question="q",
        environment=MockEnvironment(),
        root_llm=MockLLMClient([_ready_reply()]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(
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
        config=RLMConfig(enforce_contract=True),
    )

    assert result.error is not None and result.error.type != "PolicyError"
    assert result.run_record is not None
    policy = next(event for event in result.run_record.events if event.type == "policy")
    assert policy.body["outcome"] == "not_evaluated"
    assert policy.body["violation_type"] is None


def test_withheld_policy_content_cannot_enter_durable_terminal_errors() -> None:
    secret = "WITHHELD_PRIVATE_CONTENT"
    context = create_execution_context()
    error = RLMError(
        type="PolicyError",
        message="withheld",
        details={"withheld_content": secret},
    )
    result = finalize(
        answer_text="Error: policy violation",
        answer={"content": secret, "ready": False},
        iterations=1,
        context=context,
        trajectory=[],
        error=error,
        config=RLMConfig(enforce_contract=True),
    )

    assert result.error is not None and result.error.type == "PolicyError"
    assert result.error.details is not None
    assert result.error.details["withheld_content"] == secret
    assert result.run_record is not None
    assert secret not in json.dumps(result.run_record.as_dict())
    assert result.run_record.terminal["error"] == {"type": "PolicyError"}


def test_recovered_policy_violation_remains_a_durable_policy_fact() -> None:
    result = finalize(
        answer_text="Recovered from the retained evidence.",
        answer={"content": "draft", "ready": False},
        iterations=1,
        context=create_execution_context(),
        trajectory=[],
        extracted=True,
        recovered_error=RLMError(type="PolicyError", message="recovered"),
        config=RLMConfig(enforce_contract=True),
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
                run_id="config-run",
                trace_retention=TraceRetentionPolicy(frozenset({"output"})),
            ),
            context=context,
        )
