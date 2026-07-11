"""One event vocabulary, one channel (#35): the Python EVENT_TYPES set is the
canonical vocabulary, pinned against the relay's events.ts forwarding filter;
emission is validated against it; and a bare run_rlm performs zero stderr
writes — entry points attach sinks explicitly."""

from __future__ import annotations

import re

import pytest

from droste import RLMConfig, run_rlm
from droste.execution.context import create_execution_context
from droste.execution.progress import (
    EVENT_TYPES,
    output_event,
    progress_event,
    render_verbose,
)
from droste.protocols.llm_client import TokenUsage
from droste.substrates import relay_dir
from droste.testing import MockEnvironment, MockLLMClient, MockResponse, MockSubcallClient

READY_REPLY = MockResponse(
    text="""```python\nprint('hi')\nanswer['content'] = 'ok'\nanswer['ready'] = True\n```""",
    usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
)


def test_python_vocabulary_matches_relay_events_ts() -> None:
    # The standard cross-language-constant pin: adding an event type on
    # either side without the other fails here instead of the relay filter
    # silently dropping the new event.
    ts = (relay_dir() / "events.ts").read_text()
    match = re.search(r"new Set<string>\(\[(.*?)\]\)", ts, re.S)
    assert match, "events.ts RLM_EVENT_TYPES set literal not found"
    ts_names = set(re.findall(r'"([a-z_]+)"', match.group(1)))
    assert ts_names == EVENT_TYPES


def test_unknown_event_type_fails_loudly() -> None:
    context = create_execution_context()
    with pytest.raises(ValueError, match="unknown RLM event type 'brand_new'"):
        context.emit_event({"type": "brand_new"})


def test_bare_run_rlm_writes_nothing_to_stderr(capfd) -> None:
    # FCIS acceptance (#35): with no sinks attached, the engine performs no
    # emission I/O of its own — even with the legacy verbose flag set.
    # (Print-free sandbox code: MockEnvironment execs in-process, so a
    # sandbox print would hit the real stdout — that is the mock's I/O, not
    # the engine's.)
    silent_reply = MockResponse(
        text="""```python\nanswer['content'] = 'ok'\nanswer['ready'] = True\n```""",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    result = run_rlm(
        question="q",
        environment=MockEnvironment(),
        root_llm=MockLLMClient(responses=[silent_reply]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(max_iterations=1, verbose=True),
    )
    assert result.ready
    captured = capfd.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_run_rlm_event_stream_through_attached_sink() -> None:
    events: list[dict] = []
    result = run_rlm(
        question="q",
        environment=MockEnvironment(),
        root_llm=MockLLMClient(responses=[READY_REPLY]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(max_iterations=1),
        on_event=events.append,
    )
    assert result.ready
    assert [e["type"] for e in events] == [
        "iteration_start",
        "llm_response",
        "code",
        "output",
    ]
    assert all(e["type"] in EVENT_TYPES for e in events)
    output = events[-1]
    assert output["calls_made"] == 0
    assert output["answer_ready"] is True
    assert output["answer_content_chars"] == len("ok")


def test_render_verbose_projects_the_trace_view() -> None:
    banner = render_verbose(progress_event("Iteration 1/5: Generating code..."))
    assert banner is not None and "=" * 60 in banner and "Iteration 1/5" in banner

    rendered = render_verbose(
        output_event(1, "42\n", calls_made=3, answer_ready=True, answer_content_chars=2)
    )
    assert rendered is not None
    assert "Output:\n42" in rendered
    assert "Sub-calls made: 3" in rendered
    assert "answer['ready'] = True" in rendered
    assert "answer['content'] length: 2 chars" in rendered

    # Events the trace view does not show project to None.
    assert render_verbose({"type": "startup", "engine_version": "x"}) is None
    assert render_verbose({"type": "reasoning_delta", "text": "t"}) is None


def test_output_event_reports_post_gate_readiness() -> None:
    # A model-set ready answer that the ready-time policy gate rejects must
    # never be published as ready (codex review): the output event carries
    # the post-gate state, and the execution_error event follows it. The
    # violation must be a READY-TIME one (numeric_output over non-numeric
    # content) — a semantic hint would trip the pre-exec contract check
    # before any output event exists (codex review on the first cut).
    from droste.loop.step import execute_step
    from droste.policy import PolicyHints

    events: list[dict] = []
    context = create_execution_context(on_event=events.append)
    env = MockEnvironment()
    outcome = execute_step(
        "answer['content'] = 'not a number'\nanswer['ready'] = True",
        iteration=1,
        environment=env,
        env_globals=env.globals(),
        answer=env.globals()["answer"],
        cfg=RLMConfig(policy_hints=PolicyHints(numeric_output=True)),
        context=context,
        data_accessor_names=set(),
        namespaced_accessor_pairs=set(),
    )
    assert outcome.error is not None and outcome.error.type == "PolicyError"
    assert [e["type"] for e in events] == ["output", "execution_error"]
    assert events[0]["answer_ready"] is False
