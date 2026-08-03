"""Answer-state checkpoints (Trace ABI v8).

All answer-critical state used to live only in the loop's memory until the
terminal result. A host that lost the process — a watchdog kill, a crashed
substrate — lost every draft with it. The `checkpoint` event publishes that
state as it moves, so a kill becomes a render rather than a recovery.

The engine stays agnostic about what a host considers answer-critical:
`payload` is opaque, carried across without inspection or schema checks, and
nothing about a checkpoint may ever fail a run.
"""

from __future__ import annotations

import pytest

from droste import RLMConfig, RunEvent, TraceRetentionPolicy, run_rlm
from droste.execution.progress import EVENT_TYPES, checkpoint_event
from droste.execution.trace import (
    TRACE_ABI_VERSION,
    PersistenceClass,
    TraceRecorder,
    persistence_class_for,
    select_retained_events,
)
from droste.protocols.llm_client import TokenUsage
from droste.testing import MockEnvironment, MockLLMClient, MockResponse, MockSubcallClient


def _reply(code: str) -> MockResponse:
    return MockResponse(
        text=f"```python\n{code}\n```",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, exact=True),
    )


def _draft_reply(content: str) -> MockResponse:
    return _reply(f"answer['content'] = {content!r}")


def _ready_reply(content: str) -> MockResponse:
    return _reply(f"answer['content'] = {content!r}\nanswer['ready'] = True")


def _run(responses: list[MockResponse], **config_kwargs: object) -> tuple[object, list[dict]]:
    events: list[dict] = []
    result = run_rlm(
        question="q",
        environment=MockEnvironment(),
        root_llm=MockLLMClient(responses=responses),
        subcalls=MockSubcallClient(),
        config=RLMConfig(**config_kwargs),  # type: ignore[arg-type]
        on_event=events.append,
    )
    return result, [event for event in events if event["type"] == "checkpoint"]


def _valid_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "type": "checkpoint",
        "iteration": 1,
        "checkpoint_seq": 1,
        "draft": "partial answer",
        "draft_chars": len("partial answer"),
        "ready": False,
        "payload": None,
    }
    body.update(overrides)
    return body


# --- Layer 1: the wire contract ----------------------------------------------


def test_trace_abi_is_version_eight_and_knows_checkpoint() -> None:
    assert TRACE_ABI_VERSION == 8
    assert "checkpoint" in EVENT_TYPES


def test_checkpoint_is_retention_gated_content_not_a_durable_fact() -> None:
    """A checkpoint carries the draft itself, so it must be selectable by the
    same retention machinery as every other content-bearing event — never
    durable, which would persist message-derived content unconditionally."""
    assert persistence_class_for("checkpoint") is PersistenceClass.CONFIGURABLE

    recorder = TraceRecorder(run_id="retention")
    event = recorder.append(_valid_body())
    assert select_retained_events((event,), TraceRetentionPolicy()) == ()
    retained = select_retained_events(
        (event,), TraceRetentionPolicy(retain=frozenset({"checkpoint"}))
    )
    assert retained == (event,)


def test_checkpoint_envelope_is_strict() -> None:
    recorder = TraceRecorder(run_id="strict-checkpoint")

    with pytest.raises(ValueError, match="draft_chars must equal the draft length"):
        recorder.append(_valid_body(draft_chars=3))
    with pytest.raises(ValueError, match="checkpoint_seq must be positive"):
        recorder.append(_valid_body(checkpoint_seq=0))
    with pytest.raises(ValueError, match="checkpoint iteration must be positive"):
        recorder.append(_valid_body(iteration=0))
    with pytest.raises(TypeError, match="ready.*invalid type"):
        recorder.append(_valid_body(ready="no"))
    with pytest.raises(ValueError, match="missing body fields: draft"):
        body = _valid_body()
        del body["draft"]
        recorder.append(body)
    with pytest.raises(ValueError, match="unknown body fields: evidence"):
        recorder.append(_valid_body(evidence=[]))


def test_checkpoint_seq_strictly_increases_within_a_run() -> None:
    recorder = TraceRecorder(run_id="ordered-checkpoints")
    recorder.append(_valid_body(checkpoint_seq=1))
    recorder.append(_valid_body(checkpoint_seq=4))

    with pytest.raises(ValueError, match="strictly increase"):
        recorder.append(_valid_body(checkpoint_seq=4))
    with pytest.raises(ValueError, match="strictly increase"):
        recorder.append(_valid_body(checkpoint_seq=2))

    # A different run owns its own ordinals.
    other = TraceRecorder(run_id="other-run")
    assert other.append(_valid_body(checkpoint_seq=1)).body["checkpoint_seq"] == 1


def test_checkpoint_payload_is_carried_but_never_schema_checked() -> None:
    """The envelope pins the container; the contents are the host's business.
    Anything narrower would make droste learn what a host puts in there."""
    payload = {
        "anything": [{"nested": 1}, None, "text"],
        "the_engine": {"does": {"not": {"care": True}}},
    }
    recorder = TraceRecorder(run_id="opaque-payload")
    event = recorder.append(_valid_body(payload=payload))
    assert event.as_dict()["payload"] == payload

    assert recorder.append(_valid_body(checkpoint_seq=2, payload=None)).body["payload"] is None
    with pytest.raises(TypeError, match="payload.*invalid type"):
        recorder.append(_valid_body(checkpoint_seq=3, payload=["not", "an", "object"]))


def test_checkpoint_event_builder_derives_draft_chars() -> None:
    assert checkpoint_event(2, 3, "abcd", ready=True, payload={"k": 1}) == {
        "type": "checkpoint",
        "iteration": 2,
        "checkpoint_seq": 3,
        "draft": "abcd",
        "draft_chars": 4,
        "ready": True,
        "payload": {"k": 1},
    }


# --- Layer 2: the loop emission point and the payload hook --------------------


def test_checkpoint_follows_each_executed_step_whose_draft_moved() -> None:
    result, checkpoints = _run([_draft_reply("first draft"), _ready_reply("final answer")])

    assert result.ready
    assert [event["checkpoint_seq"] for event in checkpoints] == [1, 2]
    assert [event["draft"] for event in checkpoints] == ["first draft", "final answer"]
    assert [event["draft_chars"] for event in checkpoints] == [11, 12]
    assert [event["ready"] for event in checkpoints] == [False, True]
    assert [event["iteration"] for event in checkpoints] == [1, 2]
    assert all(event["payload"] is None for event in checkpoints)
    assert all(event["version"] == 8 for event in checkpoints)
    assert all(event["persistence_class"] == "configurable" for event in checkpoints)


def test_repaired_code_checkpoints_its_own_draft() -> None:
    """The repaired attempt is what actually ran; its draft must be published
    too, or a kill right after a repair renders the pre-repair answer."""
    _, checkpoints = _run(
        [
            _reply("answer['content'] = 'salvageable'\nraise ValueError('boom')"),
            _ready_reply("repaired answer"),
        ]
    )

    assert [event["draft"] for event in checkpoints] == ["salvageable", "repaired answer"]
    assert [event["checkpoint_seq"] for event in checkpoints] == [1, 2]
    assert [event["ready"] for event in checkpoints] == [False, True]


def test_unmoved_draft_without_a_payload_emits_nothing() -> None:
    _, checkpoints = _run([_reply("print('no draft')"), _ready_reply("done")])

    assert [event["draft"] for event in checkpoints] == ["done"]


def test_payload_alone_is_reason_enough_to_checkpoint() -> None:
    """A host whose own state moved needs a checkpoint even when the draft
    stood still — the draft is not the only answer-critical state."""
    payloads = iter([{"host": 1}, {"host": 2}])
    _, checkpoints = _run(
        [_reply("print('no draft')"), _ready_reply("done")],
        checkpoint_payload_provider=lambda: next(payloads),
    )

    assert [event["draft"] for event in checkpoints] == ["", "done"]
    assert [event["payload"] for event in checkpoints] == [{"host": 1}, {"host": 2}]


def test_payload_provider_value_reaches_the_wire_unexamined() -> None:
    payload = {"opaque": [{"to": "droste"}, 7, None], "totals": {"a": 1}}
    _, checkpoints = _run(
        [_ready_reply("answer")],
        checkpoint_payload_provider=lambda: payload,
    )

    assert [event["payload"] for event in checkpoints] == [payload]


def test_raising_payload_provider_reports_and_still_checkpoints() -> None:
    def provider() -> dict[str, object]:
        raise RuntimeError("host ledger exploded")

    with pytest.warns(RuntimeWarning, match="checkpoint payload provider failed"):
        result, checkpoints = _run(
            [_ready_reply("answer")],
            checkpoint_payload_provider=provider,
        )

    assert result.ready
    assert result.answer == "answer"
    assert [event["payload"] for event in checkpoints] == [None]
    assert [event["draft"] for event in checkpoints] == ["answer"]


def test_unrepresentable_payload_drops_the_checkpoint_not_the_run() -> None:
    """A payload droste cannot put on the wire is still the host's mistake to
    fix, never a reason to lose a completed run."""
    with pytest.warns(RuntimeWarning, match="checkpoint dropped"):
        result, checkpoints = _run(
            [_ready_reply("answer")],
            checkpoint_payload_provider=lambda: object(),
        )

    assert result.ready
    assert result.answer == "answer"
    assert result.error is None
    assert checkpoints == []


def test_checkpoints_reach_a_run_record_only_when_retained() -> None:
    result, checkpoints = _run([_ready_reply("answer")])
    assert checkpoints
    assert all(event.type != "checkpoint" for event in result.run_record.events)

    retained, _ = _run(
        [_ready_reply("answer")],
        trace_retention=TraceRetentionPolicy(
            retain=frozenset({"checkpoint"}), policy_id="checkpoint-retained"
        ),
    )
    checkpoint_events: list[RunEvent] = [
        event for event in retained.run_record.events if event.type == "checkpoint"
    ]
    assert [event.body["draft"] for event in checkpoint_events] == ["answer"]
