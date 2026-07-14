"""Regression: the runner must report subcalls from the same execution context
the subcall client increments.

run() builds an ExecutionContext for HTTPSubcallClient but historically did not
pass it to run_rlm, which then created its own fresh context — so the response
reported subcalls=0 regardless of how many subcalls actually ran.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from droste import Budget
from droste.execution.context import create_execution_context
from droste_runner.runner import HTTPSubcallClient, run

ROOT_REPLY = (
    "I'll query the model once and finish.\n"
    "```python\n"
    "hint = llm_query('classify: is this spam?')\n"
    "answer['content'] = 'label: ' + hint\n"
    "answer['metadata'] = {'evidence_ids': ['classification-1']}\n"
    "answer['ready'] = True\n"
    "```\n"
)


def _budget(**overrides: int) -> dict[str, int]:
    return Budget(**overrides).as_dict()


def test_http_subcall_reports_only_an_explicit_output_limit() -> None:
    context = create_execution_context()
    kwargs = {
        "endpoint": "https://example.invalid/subcall",
        "token": "t",
        "session": "s",
        "session_index": 0,
        "context": context,
    }

    assert not hasattr(HTTPSubcallClient(**kwargs), "output_token_limit")
    assert HTTPSubcallClient(**kwargs, max_output_tokens=512).output_token_limit == 512


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path == "/root":
            body = {"result": ROOT_REPLY, "usage": {"input_tokens": 1, "output_tokens": 1}}
        else:
            body = {
                "result": "spam",
                "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            }
        raw = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        pass


def test_run_reports_actual_subcall_count() -> None:
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        response = run(
            {
                "protocol_version": 3,
                "model": "test-model",
                "question": "is it spam?",
                "budget": _budget(subcalls=10, depth=2),
                "token": "test-token",
                "root_endpoint": f"{base}/root",
                "subcall_endpoint": f"{base}/subcall",
                "trace_policy_id": "local-training-v1",
                "trace_expires_at": "2026-10-14T00:00:00Z",
                "trace_host_managed_expiry": True,
                "training_allowed": True,
                "data_use_authorization_ref": "consent://trace/1",
                "data_use_purposes": ["training"],
                "root_model_revision": "root-rev",
                "subcall_model": "leaf-model",
                "subcall_model_revision": "leaf-rev",
                "root_sampling": {"temperature": 0.25},
                "subcall_sampling": {"temperature": 0},
                "subcall_concurrency": 2,
                "seed": 17,
                "source_revision": "commit-a",
            }
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert response["error"] is None
    assert response["ready"] is True
    assert response["answer"] == "label: spam"
    assert response["subcalls"] == 1
    assert response["successful_subcalls"] == 1
    assert response["extracted"] is False
    assert response["recovered_error"] is None
    assert response["answer_metadata"] == {"evidence_ids": ["classification-1"]}
    assert "stdout_truncations" not in response
    manifest = response["scaffold_manifest"]
    assert manifest["inference"] == {
        "root": {"id": "test-model", "revision": "root-rev"},
        "subcall": {"id": "leaf-model", "revision": "leaf-rev"},
        "root_sampling": {"temperature": 0.25},
        "subcall_sampling": {"temperature": 0},
        "output_limits": {"root_tokens": 4096, "subcall_tokens": 2048},
        "concurrency": 2,
        "seed": 17,
    }
    assert manifest["abis"]["runner"] == 3
    assert manifest["engine"]["source_revision"] == "commit-a"
    assert manifest["id"].startswith("sha256:")
    assert "trajectory" not in response
    usage = response["run_record"]["terminal"]["usage"]
    assert usage["root"] == {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "requests": 1,
        "successes": 1,
    }
    assert usage["subcall"] == {
        "input_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
        "requests": 1,
        "successes": 1,
    }
    assert usage["total_tokens"] == 7
    capability = next(
        event for event in response["run_record"]["events"] if event["type"] == "capability"
    )
    assert capability["outcome"]["run_id"] == response["run_id"]
    assert capability["outcome"]["capability_id"]["operation"] == "llm_query"
    assert "params" not in capability["outcome"]
    assert "result" not in capability["outcome"]
    assert response["run_record"]["retention"] == {
        "policy_id": "local-training-v1",
        "retain": [],
        "expires_at": "2026-10-14T00:00:00Z",
        "host_managed_expiry": True,
    }
    assert response["run_record"]["data_use"] == {
        "training_allowed": True,
        "authorization_ref": "consent://trace/1",
        "purposes": ["training"],
    }


def test_runner_trajectory_adds_status_without_rewriting_result(monkeypatch) -> None:
    from importlib import import_module

    import droste_runner.runner as runner_module
    from droste.loop.step import RLMResult
    from droste.loop.trajectory import IterationRecord

    execution_result = "ERROR: legitimate application output\n"

    def fake_run_rlm(*args: Any, **kwargs: Any) -> RLMResult:
        return RLMResult(
            answer=execution_result,
            ready=True,
            iterations=1,
            tokens_used=2,
            sub_calls_made=0,
            trajectory=[
                IterationRecord(
                    iteration=1,
                    llm_input=[{"role": "user", "content": "q"}],
                    llm_output="```python\nprint('x')\n```",
                    code_executed="print('x')",
                    execution_result=execution_result,
                    tokens_used=2,
                    execution_status="success",
                )
            ],
        )

    monkeypatch.setattr(import_module("droste_runner.run"), "run_rlm", fake_run_rlm)
    response = runner_module.run(
        {
            "protocol_version": 3,
            "model": "test-model",
            "question": "q",
            "budget": _budget(),
            "token": "unused",
            "root_endpoint": "http://127.0.0.1:1/root",
            "subcall_endpoint": "http://127.0.0.1:1/subcall",
        }
    )

    assert response["answer"] == execution_result
    assert "trajectory" not in response


def _client() -> tuple[HTTPSubcallClient, Any]:
    context = create_execution_context()
    client = HTTPSubcallClient(
        endpoint="http://unused.invalid",
        token="t",
        session="",
        session_index=0,
        context=context,
    )
    client._request = lambda payload: "ok"  # type: ignore[method-assign]
    return client, context


def test_http_client_reports_each_issued_call_without_owning_budget_policy() -> None:
    client, context = _client()
    assert client.llm_query("a") == "ok"
    assert client.llm_query("b") == "ok"
    assert context.stats.calls_made == 2
    assert context.stats.successful_calls == 2


def test_failed_request_is_attempted_but_not_successful() -> None:
    client, context = _client()

    def fail(_payload: dict[str, Any]) -> str:
        raise RuntimeError("request failed")

    client._request = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="request failed"):
        client.llm_query("a")

    assert context.stats.calls_made == 1
    assert context.stats.successful_calls == 0


def test_concurrent_batch_counts_each_issued_call() -> None:
    client, context = _client()
    results = client.llm_batch([f"p{i}" for i in range(20)])
    assert results == ["ok"] * 20
    assert context.stats.calls_made == 20
    assert context.stats.successful_calls == 20


# --- llm_batch / llm_batch_with_errors share one bounded fan-out (#34).


def test_llm_batch_with_errors_bounded_concurrency() -> None:
    client, _ = _client()
    lock = threading.Lock()
    active = 0
    peak = 0

    def _request(payload: dict[str, Any]) -> str:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.02)
            return "ok"
        finally:
            with lock:
                active -= 1

    client._request = _request  # type: ignore[method-assign]
    results, errors = client.llm_batch_with_errors([f"p{i}" for i in range(30)])
    assert results == ["ok"] * 30
    assert errors == []
    assert peak <= 5


def test_llm_batch_with_errors_enforces_prompt_cap() -> None:
    client, _ = _client()
    with pytest.raises(ValueError, match="exceeds max 50"):
        client.llm_batch_with_errors(["p"] * 51)


def test_llm_batch_with_errors_orders_errors_by_index() -> None:
    client, _ = _client()

    def _request(payload: dict[str, Any]) -> str:
        prompt = payload["prompt"]
        if prompt == "p1":
            # Finish after p3's failure so completion order != index order.
            time.sleep(0.05)
            raise RuntimeError("boom p1")
        if prompt == "p3":
            raise RuntimeError("boom p3")
        return "ok"

    client._request = _request  # type: ignore[method-assign]
    results, errors = client.llm_batch_with_errors([f"p{i}" for i in range(5)])
    assert results == ["ok", "", "ok", "", "ok"]
    assert [e["index"] for e in errors] == [1, 3]
    assert "boom p1" in str(errors[0]["error"])
    assert "boom p3" in str(errors[1]["error"])


def test_llm_batch_raises_lowest_index_error_unwrapped() -> None:
    client, _ = _client()

    class _Boom(RuntimeError):
        pass

    def _request(payload: dict[str, Any]) -> str:
        prompt = payload["prompt"]
        if prompt in ("p2", "p4"):
            raise _Boom(f"boom {prompt}")
        return "ok"

    client._request = _request  # type: ignore[method-assign]
    with pytest.raises(_Boom, match="boom p2"):
        client.llm_batch([f"p{i}" for i in range(5)])


# --- Subcall cost controls: the resolved budget always supplies the output
# ceiling; optional model controls are included only when set.


class _CapturingHandler(BaseHTTPRequestHandler):
    subcall_payloads: list[dict[str, Any]] = []
    root_payloads: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length)
        if self.path == "/root":
            type(self).root_payloads.append(json.loads(raw_body.decode("utf-8")))
            body = {"result": ROOT_REPLY, "usage": {"input_tokens": 1, "output_tokens": 1}}
        else:
            type(self).subcall_payloads.append(json.loads(raw_body.decode("utf-8")))
            body = {"result": "spam"}
        raw = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:
        pass


def _run_with_capture(request_extra: dict[str, Any]) -> list[dict[str, Any]]:
    _CapturingHandler.subcall_payloads = []
    _CapturingHandler.root_payloads = []
    server = HTTPServer(("127.0.0.1", 0), _CapturingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        request: dict[str, Any] = {
            "protocol_version": 3,
            "model": "test-model",
            "question": "is it spam?",
            "budget": _budget(subcalls=10, depth=2),
            "token": "test-token",
            "root_endpoint": f"{base}/root",
            "subcall_endpoint": f"{base}/subcall",
        }
        request.update(request_extra)
        response = run(request)
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert response["error"] is None
    assert response["subcalls"] == 1
    return _CapturingHandler.subcall_payloads


def test_subcall_payload_includes_cost_controls_when_set() -> None:
    payloads = _run_with_capture(
        {
            "budget": _budget(subcalls=10, depth=2, subcall_output_tokens=1024),
            "subcall_model": "grok-4-fast-non-reasoning",
            "subcall_reasoning_effort": "none",
        }
    )
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["max_output_tokens"] == 1024
    assert payload["model"] == "grok-4-fast-non-reasoning"
    assert payload["reasoning_effort"] == "none"
    system_prompt = _CapturingHandler.root_payloads[0]["messages"][0]["content"]
    assert "subcall_output_tokens_per_call=1024" in system_prompt


def test_subcall_payload_uses_budget_ceiling_and_omits_optional_controls() -> None:
    payloads = _run_with_capture({})
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["max_output_tokens"] == 2048
    assert "model" not in payload
    assert "reasoning_effort" not in payload
    system_prompt = _CapturingHandler.root_payloads[0]["messages"][0]["content"]
    assert "subcall_output_tokens_per_call=2048" in system_prompt


def test_zero_subcall_output_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="budget.subcall_output_tokens must be positive"):
        budget = _budget()
        budget["subcall_output_tokens"] = 0
        run(
            {
                "protocol_version": 3,
                "model": "m",
                "question": "q",
                "budget": budget,
                "token": "t",
                "root_endpoint": "http://127.0.0.1:1/root",
                "subcall_endpoint": "http://127.0.0.1:1/subcall",
            }
        )


def test_wrapper_call_enforces_allowed_hosts():
    # The wrapper's allowed_hosts is an enforced allowlist, not prompt
    # decoration: a request-supplied base_url outside it must be refused
    # before any connection is attempted (pre-publish read-through).
    from droste_runner.runner import WrapperTransport

    wrapper = WrapperTransport(
        {
            "base_url": "http://evil.internal:9",
            "token": "t",
            "allowed_hosts": ["partner.example.com"],
        }
    )
    import pytest as _pytest

    with _pytest.raises(ValueError, match="not in allowed_hosts"):
        wrapper._call("/search", {"query": "q"})


def test_wrapper_rejects_redirect_to_disallowed_host():
    # SSRF-via-redirect: a call starting at an allowed host must still refuse
    # a 30x to a host outside the allowlist (codex review).
    from droste_runner.runner import _allowlist_opener

    opener = _allowlist_opener({"partner.example.com"})
    handler = next(h for h in opener.handlers if hasattr(h, "redirect_request"))
    import pytest as _pytest

    class _Req:
        pass

    with _pytest.raises(ValueError, match="not in allowed_hosts"):
        handler.redirect_request(
            _Req(), None, 302, "Found", {}, "http://169.254.169.254/latest/meta-data"
        )


def test_wrapper_malformed_allowed_hosts_fails_closed():
    # A present-but-malformed allowed_hosts (string / empty list) is a config
    # error, not an allow-all — fail closed (codex review).
    import pytest as _pytest

    from droste_runner.runner import WrapperTransport

    for bad in ("partner.example.com", [], ["", "  "]):
        w = WrapperTransport({"base_url": "http://h/x", "token": "t", "allowed_hosts": bad})
        with _pytest.raises(ValueError, match="allowed_hosts"):
            w._call("/search", {"query": "q"})

    # Absent key still means allow-all (no raise for the host check; will fail
    # later at connection instead).
    w = WrapperTransport({"base_url": "http://127.0.0.1:9/x", "token": "t"})
    with _pytest.raises(ValueError, match="request failed"):
        w._call("/search", {"query": "q"})
