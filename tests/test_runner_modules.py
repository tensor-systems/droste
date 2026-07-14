"""Focused boundaries for the split droste_runner package (#11)."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from typing import Any

from droste.environments import RunnerEnvironment
from droste.loop.trajectory import IterationRecord
from droste.protocols.llm_client import TokenUsage
from droste_runner import runner
from droste_runner.http_clients import HTTPSubcallClient, RootLLMClient
from droste_runner.protocol import (
    RootResponseMetadata,
    _protocol_error_response,
    build_exception_response,
    build_response,
)
from droste_runner.sources import WrapperV1DataSource, build_data_sources


def test_legacy_runner_facade_reexports_canonical_implementations() -> None:
    run_module = importlib.import_module("droste_runner.run")

    assert runner.RunnerEnvironment is RunnerEnvironment
    assert runner.HTTPSubcallClient is HTTPSubcallClient
    assert runner.RootLLMClient is RootLLMClient
    assert runner.WrapperV1DataSource is WrapperV1DataSource
    assert runner.build_data_sources is build_data_sources
    assert runner.run is run_module.run
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
    exception = build_exception_response(RuntimeError("boom"), "traceback")
    assert set(refusal) == set(response) == set(exception)


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
