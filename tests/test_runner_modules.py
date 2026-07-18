"""Focused boundaries for the split droste_runner package (#11)."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from typing import Any

from droste.environments import RunnerEnvironment
from droste.loop.trajectory import IterationRecord
from droste.protocols.llm_client import CACHE_ANCHOR_MARKER, TokenUsage
from droste_runner import runner
from droste_runner.http_clients import HTTPSubcallClient, RootLLMClient
from droste_runner.protocol import (
    RootResponseMetadata,
    RunnerOperation,
    _protocol_error_response,
    build_exception_response,
    build_response,
)
from droste_runner.sources import WrapperTransport, build_provider_registry


def test_legacy_runner_facade_reexports_canonical_implementations() -> None:
    run_module = importlib.import_module("droste_runner.run")

    assert runner.RunnerEnvironment is RunnerEnvironment
    assert runner.HTTPSubcallClient is HTTPSubcallClient
    assert runner.RootLLMClient is RootLLMClient
    assert runner.WrapperTransport is WrapperTransport
    assert runner.build_provider_registry is build_provider_registry
    assert runner.run is run_module.run
    assert runner.run_worker_request is run_module.run_worker_request
    assert not hasattr(runner, "run_rlm")


def test_focused_runner_modules_import_independently() -> None:
    for module_name in (
        "droste.environments.inprocess",
        "droste_runner.environment",
        "droste_runner.http_clients",
        "droste_runner.sources",
        "droste_runner.protocol",
        "droste_runner.run",
    ):
        assert importlib.import_module(module_name).__name__ == module_name


def test_cli_does_not_import_runner_package_for_environment() -> None:
    cli_main = importlib.import_module("droste_cli.main")

    source = cli_main.__loader__.get_source(cli_main.__name__)
    assert source is not None
    assert "droste_runner" not in source


def test_one_response_builder_shapes_refusal_and_success() -> None:
    refusal = _protocol_error_response(None, "protocol_version_missing")
    assert refusal["protocol_version"] == runner.RUNNER_PROTOCOL_VERSION
    assert refusal["error"]["type"] == "protocol_version_missing"
    assert "trajectory" not in refusal
    assert refusal["status"] == "refusal"

    result = SimpleNamespace(
        answer="ok",
        answer_metadata={"evidence": "result-1"},
        ready=True,
        iterations=1,
        tokens_used=2,
        sub_calls_made=1,
        sub_calls_succeeded=1,
        extracted=False,
        extract_error=None,
        recovered_error=None,
        trajectory=[
            IterationRecord(
                iteration=1,
                llm_input=[{"role": "user", "content": "q"}],
                llm_output="response",
                code_executed="print('ok')",
                execution_result="ok",
                tokens_used=2,
                execution_status="success",
            )
        ],
        error=None,
    )
    response = build_response(
        result=result,
        metadata=RootResponseMetadata(
            provider="provider-a",
            response_id="response-1",
            stop_reason="stop",
            model="resolved-model",
        ),
        requested_model="requested-model",
        data_source_requests=3,
    )

    assert response["answer"] == "ok"
    assert response["answer_metadata"] == {"evidence": "result-1"}
    assert "trajectory" not in response
    assert response["status"] == "success"
    assert response["provider"] == "provider-a"
    assert response["response_id"] == "response-1"
    assert response["model"] == "resolved-model"
    assert response["data_source_requests"] == 3
    run_exception = build_exception_response(
        RuntimeError("boom"), "traceback", operation=RunnerOperation.RUN
    )
    assert set(refusal) == set(response) == set(run_exception)

    preflight_exception = build_exception_response(
        RuntimeError("boom"), "traceback", operation=RunnerOperation.PREFLIGHT
    )
    assert set(preflight_exception) == {
        "protocol_version",
        "operation",
        "status",
        "preflight",
        "error",
    }
    assert preflight_exception["operation"] == "preflight"
    assert preflight_exception["status"] == "error"
    assert preflight_exception["preflight"] is None
    assert preflight_exception["error"] == {
        "type": "RuntimeError",
        "message": "boom",
        "traceback": "traceback",
    }


def test_root_client_includes_explicit_reasoning_effort(monkeypatch) -> None:
    import droste_runner.http_clients as clients

    captured: dict[str, Any] = {}

    class Response:
        def read(self) -> bytes:
            return json.dumps({"result": "ok"}).encode()

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

    def urlopen(request, **kwargs):
        captured.update(json.loads(request.data.decode("utf-8")))
        return Response()

    monkeypatch.setattr(clients.urllib.request, "urlopen", urlopen)
    client = RootLLMClient(
        endpoint="https://example.invalid/root",
        token="t",
        default_model="requested-model",
        provider=None,
        max_output_tokens=100,
        temperature=None,
        stop=None,
        session="session",
        session_index=0,
        reasoning_effort="none",
    )

    assert (
        client.responses_create(
            [{"role": "user", "content": "q", CACHE_ANCHOR_MARKER: True}], model=""
        )
        == "ok"
    )
    assert captured["reasoning_effort"] == "none"
    assert CACHE_ANCHOR_MARKER not in json.dumps(captured)


def test_catalog_aware_worker_entrypoint_owns_exception_attribution() -> None:
    outcome = runner.run_worker_request(
        {
            "protocol_version": runner.RUNNER_PROTOCOL_VERSION,
            "operation": "preflight",
        },
        provider_catalog=runner.ProviderCatalog(()),
    )

    assert outcome.exit_code == 1
    assert set(outcome.response) == {
        "protocol_version",
        "operation",
        "status",
        "preflight",
        "error",
    }
    assert outcome.response["operation"] == "preflight"
    assert outcome.response["error"]["type"] == "ValueError"


def test_root_client_collects_response_metadata_as_one_record(monkeypatch) -> None:
    import droste_runner.http_clients as clients

    class Response:
        def read(self) -> bytes:
            return json.dumps(
                {
                    "result": "ok",
                    "usage": {"input_tokens": 2, "output_tokens": 3},
                    "provider": "provider-a",
                    "response_id": "response-1",
                    "stop_reason": "stop",
                    "model": "resolved-model",
                }
            ).encode()

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

    monkeypatch.setattr(clients.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    client = RootLLMClient(
        endpoint="https://example.invalid/root",
        token="t",
        default_model="requested-model",
        provider=None,
        max_output_tokens=0,
        temperature=None,
        stop=None,
        session="session",
        session_index=0,
    )

    text, usage = client.responses_create([], model="", return_usage=True)

    assert text == "ok"
    assert usage == TokenUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5)
    assert client.response_metadata == RootResponseMetadata(
        provider="provider-a",
        response_id="response-1",
        stop_reason="stop",
        model="resolved-model",
    )
    # Legacy accessors remain readable while hosts migrate to the record.
    assert client.last_response_id == "response-1"
