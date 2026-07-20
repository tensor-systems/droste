"""Exact callback failure usage must survive transport into budget settlement."""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from droste import CapabilityCallError, LLMUsageFailure, TokenUsage
from droste.capabilities import broker_subcalls
from droste.execution.budget import Budget, BudgetRequest, conservative_token_estimate
from droste.execution.context import create_execution_context
from droste_runner.http_clients import (
    HTTPSubcallClient,
    RootLLMClient,
    _parse_json_float,
    _parse_json_integer,
)

_EXACT_USAGE = {
    "input_tokens": 2,
    "output_tokens": 3,
    "total_tokens": 5,
    "reasoning_tokens": 0,
    "observation_basis": "exact",
}


@contextmanager
def _callback_server(
    responses: list[tuple[int, str, bytes | dict[str, object]]],
    *,
    requests: list[tuple[str, object]] | None = None,
) -> Iterator[str]:
    pending = list(responses)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request_body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if requests is not None:
                requests.append((self.path, json.loads(request_body)))
            status, content_type, value = pending.pop(0)
            body = value if isinstance(value, bytes) else json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/callback"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@contextmanager
def _handler_server(handler: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/callback"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _subcall_client(
    endpoint: str,
    *,
    context=None,
    max_output_tokens: int = 20,
    max_parallel: int = 5,
):
    return HTTPSubcallClient(
        endpoint=endpoint,
        token="runner-token",
        session="session",
        session_index=0,
        context=context or create_execution_context(),
        max_output_tokens=max_output_tokens,
        max_parallel=max_parallel,
    )


def _root_client(endpoint: str) -> RootLLMClient:
    return RootLLMClient(
        endpoint=endpoint,
        token="runner-token",
        default_model="test-model",
        provider=None,
        max_output_tokens=20,
        temperature=None,
        stop=None,
        session="session",
        session_index=0,
    )


def _failure(usage: object) -> dict[str, object]:
    return {
        "error": "api_error",
        "code": "PROVIDER_ERROR",
        "message": "provider failed",
        "usage": usage,
    }


def _raw_failure_with_input_lexeme(
    input_tokens: str,
    *,
    output_tokens: str = "3",
    total_tokens: str = "5",
) -> bytes:
    return (
        '{"error":"api_error","code":"PROVIDER_ERROR","message":"provider failed",'
        f'"usage":{{"input_tokens":{input_tokens},"output_tokens":{output_tokens},'
        f'"total_tokens":{total_tokens},"reasoning_tokens":0,'
        '"observation_basis":"exact"}}'
    ).encode("ascii")


def _raw_success_with_input_lexeme(input_tokens: str) -> bytes:
    return (
        '{"result":"answer","usage":{"input_tokens":'
        + input_tokens
        + ',"output_tokens":3,"total_tokens":5,"reasoning_tokens":0,'
        '"observation_basis":"exact"}}'
    ).encode("ascii")


def _raw_ndjson_completion_with_input_lexeme(input_tokens: str) -> bytes:
    return (
        '{"type":"update","delta":"answer"}\n'
        '{"type":"completion","usage":{"input_tokens":'
        + input_tokens
        + ',"output_tokens":3,"total_tokens":5,"reasoning_tokens":0,'
        '"observation_basis":"exact"}}\n'
    ).encode("ascii")


def _raw_ndjson_error_with_input_lexeme(input_tokens: str) -> bytes:
    return (
        '{"type":"error","code":"UPSTREAM_FAILURE","message":"provider failed",'
        '"usage":{"input_tokens":' + input_tokens + ',"output_tokens":3,"total_tokens":5,'
        '"reasoning_tokens":0,"observation_basis":"exact"}}\n'
    ).encode("ascii")


def _assert_plain_callback_failure(
    body: bytes | dict[str, object],
    *,
    content_type: str = "application/json",
) -> None:
    with _callback_server([(502, content_type, body)]) as endpoint:
        with pytest.raises(RuntimeError) as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert type(raised.value) is RuntimeError


def test_json_http_failure_preserves_exact_usage_and_plain_api_unwraps() -> None:
    response = (502, "application/json", _failure(_EXACT_USAGE))
    with _callback_server([response, response]) as endpoint:
        client = _subcall_client(endpoint)
        with pytest.raises(LLMUsageFailure, match="HTTP 502") as raised:
            client.llm_query_with_usage("question")
        assert raised.value.usage == TokenUsage(2, 3, 5, exact=True)

        with pytest.raises(RuntimeError, match="HTTP 502") as plain:
            client.llm_query("question")
        assert type(plain.value) is RuntimeError


def test_json_http_failure_distinguishes_exact_zero_from_unavailable() -> None:
    zero = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "observation_basis": "exact",
    }
    with _callback_server([(500, "application/json", _failure(zero))]) as endpoint:
        with pytest.raises(LLMUsageFailure) as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert raised.value.usage == TokenUsage(0, 0, 0, exact=True)


@pytest.mark.parametrize(
    "content_type",
    [
        "application/problem+json",
        "Application/Vnd.ModelRelay.Error+Json; charset=UTF-8",
        'application/json; charset="utf-8"',
        'application/json; CHARSET="UTF-8"',
    ],
)
def test_json_suffix_media_type_preserves_exact_usage(content_type: str) -> None:
    with _callback_server([(502, content_type, _failure(_EXACT_USAGE))]) as endpoint:
        with pytest.raises(LLMUsageFailure) as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert raised.value.usage == TokenUsage(2, 3, 5, exact=True)


def test_json_http_failure_preserves_exact_int64_usage_for_root_and_subcall() -> None:
    maximum = 2**63 - 1
    usage = {
        "input_tokens": maximum,
        "output_tokens": 0,
        "total_tokens": maximum,
        "reasoning_tokens": 0,
        "observation_basis": "exact",
    }
    response = (502, "application/json", _failure(usage))
    with _callback_server([response, response]) as endpoint:
        with pytest.raises(LLMUsageFailure) as raised_subcall:
            _subcall_client(endpoint).llm_query_with_usage("question")
        with pytest.raises(LLMUsageFailure) as raised_root:
            _root_client(endpoint).responses_create(
                [{"role": "user", "content": "question"}],
                "test-model",
                return_usage=True,
            )
    expected = TokenUsage(maximum, 0, maximum, exact=True)
    assert raised_subcall.value.usage == expected
    assert raised_root.value.usage == expected


def test_json_http_failure_rejects_out_of_int64_range_for_root_and_subcall() -> None:
    maximum = 2**63 - 1
    usage = {"input_tokens": maximum + 1, "output_tokens": 3, "total_tokens": maximum}
    response = (502, "application/json", _failure(usage))
    with _callback_server([response, response]) as endpoint:
        with pytest.raises(LLMUsageFailure) as raised_subcall:
            _subcall_client(endpoint).llm_query_with_usage("question")
        with pytest.raises(LLMUsageFailure) as raised_root:
            _root_client(endpoint).responses_create(
                [{"role": "user", "content": "question"}],
                "test-model",
                return_usage=True,
            )
    expected = TokenUsage(0, 3, maximum, exact=False)
    assert raised_subcall.value.usage == expected
    assert raised_root.value.usage == expected


@pytest.mark.parametrize(
    ("lexeme", "expected"),
    [
        ("9223372036854775807", 2**63 - 1),
        ("-9223372036854775808", -(2**63)),
        ("9223372036854775808", None),
        ("-9223372036854775809", None),
    ],
)
def test_json_integer_parser_enforces_signed_int64_lexically(
    lexeme: str,
    expected: int | None,
) -> None:
    parsed = _parse_json_integer(lexeme)
    if expected is None:
        assert not isinstance(parsed, int)
    else:
        assert parsed == expected


@pytest.mark.parametrize("lexeme", ["", "+1", "01", "-01", "--1", "1.0"])
def test_json_integer_parser_rejects_non_integer_lexemes(lexeme: str) -> None:
    with pytest.raises(ValueError, match="invalid JSON integer lexeme"):
        _parse_json_integer(lexeme)


@pytest.mark.parametrize(
    ("lexeme", "expected"),
    [("0.5", 0.5), ("-1.25e2", -125.0), ("1e308", 1e308)],
)
def test_json_float_parser_accepts_bounded_finite_values(
    lexeme: str,
    expected: float,
) -> None:
    assert _parse_json_float(lexeme) == expected


@pytest.mark.parametrize(
    "lexeme",
    ["1e10000", "-1e10000", "0." + "0" * 5_000 + "1"],
)
def test_json_float_parser_rejects_nonfinite_or_resource_abusive_values(lexeme: str) -> None:
    with pytest.raises(ValueError):
        _parse_json_float(lexeme)


@pytest.mark.parametrize("digits", [4_300, 5_000])
@pytest.mark.parametrize("negative", [False, True])
def test_long_json_integer_usage_is_partial_without_global_int_limit_changes(
    digits: int,
    negative: bool,
) -> None:
    configured_limit = sys.get_int_max_str_digits()
    lexeme = ("-" if negative else "") + "9" * digits
    body = _raw_failure_with_input_lexeme(lexeme)
    with _callback_server([(502, "application/json", body)]) as endpoint:
        with pytest.raises(LLMUsageFailure) as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert raised.value.usage == TokenUsage(0, 3, 5, exact=False)
    assert sys.get_int_max_str_digits() == configured_limit


def test_long_json_integer_success_retains_root_and_subcall_results() -> None:
    body = _raw_success_with_input_lexeme("9" * 5_000)
    response = (200, "application/json", body)
    with _callback_server([response, response]) as endpoint:
        subcall = _subcall_client(endpoint).llm_query_with_usage("question")
        root_result, root_usage = _root_client(endpoint).responses_create(
            [{"role": "user", "content": "question"}],
            "test-model",
            return_usage=True,
        )
    expected = TokenUsage(0, 3, 5, exact=False)
    assert subcall.result == "answer"
    assert subcall.usage == expected
    assert root_result == "answer"
    assert root_usage == expected


def test_finite_json_floats_preserve_results_but_not_usage_counter_authority() -> None:
    body = (
        b'{"result":"answer","score":0.5,"usage":{"input_tokens":2.0,'
        b'"output_tokens":3,"total_tokens":5}}'
    )
    response = (200, "application/json", body)
    with _callback_server([response, response]) as endpoint:
        subcall = _subcall_client(endpoint).llm_query_with_usage("question")
        root_result, root_usage = _root_client(endpoint).responses_create(
            [{"role": "user", "content": "question"}],
            "test-model",
            return_usage=True,
        )
    expected = TokenUsage(0, 3, 5, exact=False)
    assert subcall.result == "answer"
    assert subcall.usage == expected
    assert root_result == "answer"
    assert root_usage == expected


@pytest.mark.parametrize(
    "body",
    [
        (
            b'{"result":"answer","score":1e10000,"usage":{"input_tokens":2,'
            b'"output_tokens":3,"total_tokens":5}}'
        ),
        (
            b'{"result":"answer","usage":{"input_tokens":1e10000,'
            b'"output_tokens":3,"total_tokens":5}}'
        ),
    ],
)
def test_exponent_overflowing_float_rejects_unary_success_envelopes(body: bytes) -> None:
    response = (200, "application/json", body)
    with _callback_server([response, response]) as endpoint:
        with pytest.raises(RuntimeError, match="malformed JSON") as subcall:
            _subcall_client(endpoint).llm_query_with_usage("question")
        with pytest.raises(RuntimeError, match="malformed JSON") as root:
            _root_client(endpoint).responses_create(
                [{"role": "user", "content": "question"}],
                "test-model",
                return_usage=True,
            )
    assert type(subcall.value) is RuntimeError
    assert type(root.value) is RuntimeError


def test_long_json_integer_ndjson_completion_retains_result_and_partial_usage() -> None:
    body = _raw_ndjson_completion_with_input_lexeme("9" * 5_000)
    with _callback_server([(200, "application/x-ndjson", body)]) as endpoint:
        result = _subcall_client(endpoint).llm_query_with_usage("question")
    assert result.result == "answer"
    assert result.usage == TokenUsage(0, 3, 5, exact=False)


def test_long_json_integer_ndjson_error_preserves_partial_usage() -> None:
    body = _raw_ndjson_error_with_input_lexeme("9" * 5_000)
    with _callback_server([(200, "application/x-ndjson", body)]) as endpoint:
        with pytest.raises(LLMUsageFailure, match="UPSTREAM_FAILURE") as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert raised.value.usage == TokenUsage(0, 3, 5, exact=False)


def test_finite_json_floats_are_valid_in_ndjson_but_usage_floats_are_partial() -> None:
    body = (
        b'{"type":"update","delta":"answer","progress":0.5}\n'
        b'{"type":"completion","score":1.25,"usage":{"input_tokens":2.0,'
        b'"output_tokens":3,"total_tokens":5}}\n'
    )
    with _callback_server([(200, "application/x-ndjson", body)]) as endpoint:
        result = _subcall_client(endpoint).llm_query_with_usage("question")
    assert result.result == "answer"
    assert result.usage == TokenUsage(0, 3, 5, exact=False)


@pytest.mark.parametrize(
    "body",
    [
        b'{"type":"update","delta":"answer","progress":1e10000}\n',
        (
            b'{"type":"completion","content":"answer","score":1e10000,'
            b'"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}\n'
        ),
        (
            b'{"type":"error","code":"UPSTREAM_FAILURE","retry_after":1e10000,'
            b'"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}\n'
        ),
    ],
)
def test_exponent_overflowing_float_rejects_every_ndjson_event(body: bytes) -> None:
    with _callback_server([(200, "application/x-ndjson", body)]) as endpoint:
        with pytest.raises(RuntimeError, match="malformed stream data") as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert type(raised.value) is RuntimeError


def test_long_json_integer_outside_usage_rejects_unary_success_envelopes() -> None:
    body = (
        b'{"result":"answer","attempt":'
        + b"9" * 5_000
        + b',"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5,'
        b'"reasoning_tokens":0,"observation_basis":"exact"}}'
    )
    response = (200, "application/json", body)
    with _callback_server([response, response]) as endpoint:
        with pytest.raises(LLMUsageFailure, match="recognized usage counter") as subcall:
            _subcall_client(endpoint).llm_query_with_usage("question")
        with pytest.raises(LLMUsageFailure, match="recognized usage counter") as root:
            _root_client(endpoint).responses_create(
                [{"role": "user", "content": "question"}],
                "test-model",
                return_usage=True,
            )
    expected = TokenUsage(2, 3, 5, exact=True)
    assert subcall.value.usage == expected
    assert root.value.usage == expected


def test_long_json_integer_outside_usage_rejects_ndjson_completion() -> None:
    body = (
        b'{"type":"completion","content":"answer","attempt":'
        + b"9" * 5_000
        + b',"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5,'
        b'"reasoning_tokens":0,"observation_basis":"exact"}}\n'
    )
    with _callback_server([(200, "application/x-ndjson", body)]) as endpoint:
        with pytest.raises(LLMUsageFailure, match="recognized usage counter") as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert raised.value.usage == TokenUsage(2, 3, 5, exact=True)


def test_long_json_integer_outside_usage_rejects_ndjson_update() -> None:
    body = b'{"type":"update","attempt":' + b"9" * 5_000 + b"}\n"
    with _callback_server([(200, "application/x-ndjson", body)]) as endpoint:
        with pytest.raises(RuntimeError, match="recognized usage counter") as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert type(raised.value) is RuntimeError


@pytest.mark.parametrize("streaming", [False, True])
def test_long_json_integer_success_keeps_full_reservation(streaming: bool) -> None:
    prompt = "question"
    subcall_output_tokens = 20
    reservation = (
        conservative_token_estimate({"args": [prompt], "kwargs": {}}) + subcall_output_tokens
    )
    budget = Budget(
        tokens=200,
        subcalls=1,
        root_output_tokens=5,
        subcall_output_tokens=subcall_output_tokens,
    )
    context = create_execution_context(budget=budget)
    if streaming:
        content_type = "application/x-ndjson"
        body = _raw_ndjson_completion_with_input_lexeme("9" * 5_000)
    else:
        content_type = "application/json"
        body = _raw_success_with_input_lexeme("9" * 5_000)
    with _callback_server([(200, content_type, body)]) as endpoint:
        brokered = broker_subcalls(
            _subcall_client(endpoint, context=context),
            context.ledger,
            usage_callback=context.record_subcall_usage,
            settlement_callback=context.record_subcall_settlement,
        )
        assert brokered.llm_query(prompt) == "answer"

    assert context.ledger.snapshot().consumed.tokens == reservation
    assert context.stats.subcall_total_tokens == 5
    assert context.stats.subcall_usage_complete is False


def test_long_json_integer_ndjson_error_keeps_full_reservation() -> None:
    prompt = "question"
    subcall_output_tokens = 20
    reservation = (
        conservative_token_estimate({"args": [prompt], "kwargs": {}}) + subcall_output_tokens
    )
    budget = Budget(
        tokens=200,
        subcalls=1,
        root_output_tokens=5,
        subcall_output_tokens=subcall_output_tokens,
    )
    context = create_execution_context(budget=budget)
    body = _raw_ndjson_error_with_input_lexeme("9" * 5_000)
    with _callback_server([(200, "application/x-ndjson", body)]) as endpoint:
        brokered = broker_subcalls(
            _subcall_client(endpoint, context=context),
            context.ledger,
            usage_callback=context.record_subcall_usage,
            settlement_callback=context.record_subcall_settlement,
        )
        with pytest.raises(CapabilityCallError, match="UPSTREAM_FAILURE"):
            brokered.llm_query(prompt)

    assert context.ledger.snapshot().consumed.tokens == reservation
    assert context.stats.subcall_total_tokens == 5
    assert context.stats.subcall_usage_complete is False


@pytest.mark.parametrize("event_type", ["completion", "error"])
def test_exponent_overflowing_float_ndjson_keeps_full_reservation(event_type: str) -> None:
    prompt = "question"
    subcall_output_tokens = 20
    reservation = (
        conservative_token_estimate({"args": [prompt], "kwargs": {}}) + subcall_output_tokens
    )
    budget = Budget(
        tokens=200,
        subcalls=1,
        root_output_tokens=5,
        subcall_output_tokens=subcall_output_tokens,
    )
    context = create_execution_context(budget=budget)
    body = (
        f'{{"type":"{event_type}","content":"answer","score":1e10000,'
        '"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}\n'
    ).encode("ascii")
    with _callback_server([(200, "application/x-ndjson", body)]) as endpoint:
        brokered = broker_subcalls(
            _subcall_client(endpoint, context=context),
            context.ledger,
            usage_callback=context.record_subcall_usage,
            settlement_callback=context.record_subcall_settlement,
        )
        with pytest.raises(CapabilityCallError, match="malformed stream data"):
            brokered.llm_query(prompt)

    assert context.ledger.snapshot().consumed.tokens == reservation
    assert context.stats.subcall_total_tokens == 0
    assert context.stats.subcall_usage_complete is False


@pytest.mark.parametrize(
    "lexeme",
    [
        "-9223372036854775808",
        "-9223372036854775809",
    ],
)
def test_minimum_and_below_int64_usage_are_invalid_partial_counters(lexeme: str) -> None:
    body = _raw_failure_with_input_lexeme(lexeme)
    with _callback_server([(502, "application/json", body)]) as endpoint:
        with pytest.raises(LLMUsageFailure) as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert raised.value.usage == TokenUsage(0, 3, 5, exact=False)


@pytest.mark.parametrize(
    "body",
    [
        (
            b'{"error":"api_error","status_code":'
            + b"9" * 5_000
            + b',"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}'
        ),
        (
            b'{"error":'
            + b"9" * 5_000
            + b',"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}'
        ),
        (
            b'{"error":"api_error","failure":{"attempt":-'
            + b"9" * 5_000
            + b'},"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}'
        ),
        (
            b'{"error":"api_error","usage":{"input_tokens":2,"output_tokens":3,'
            b'"total_tokens":5,"provider_counter":' + b"9" * 5_000 + b"}}"
        ),
    ],
)
def test_invalid_integer_outside_known_usage_counters_rejects_envelope(body: bytes) -> None:
    _assert_plain_callback_failure(body)


@pytest.mark.parametrize(
    ("malformed", "expected"),
    [
        (
            {"input_tokens": 2, "output_tokens": 3, "total_tokens": "five"},
            TokenUsage(2, 3, 0, exact=False),
        ),
        (
            {"input_tokens": 2**63 - 1, "output_tokens": 1, "total_tokens": 2**63 - 1},
            TokenUsage(2**63 - 1, 1, 2**63 - 1, exact=False),
        ),
        (
            {
                "input_tokens": 1,
                "output_tokens": 0,
                "total_tokens": 1,
                "cache_read_input_tokens": 1,
                "cache_write_input_tokens": 1,
            },
            TokenUsage(
                1,
                0,
                1,
                cache_read_tokens=1,
                cache_creation_tokens=1,
                exact=False,
            ),
        ),
    ],
)
def test_json_http_failure_preserves_partial_usage_without_marking_it_exact(
    malformed: dict[str, object],
    expected: TokenUsage,
) -> None:
    with _callback_server([(502, "application/json", _failure(malformed))]) as endpoint:
        with pytest.raises(LLMUsageFailure) as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert raised.value.usage == expected


def test_json_http_failure_without_usage_remains_a_transport_error() -> None:
    body = {"error": "api_error", "code": "PROVIDER_ERROR", "message": "provider failed"}
    with _callback_server([(502, "application/json", body)]) as endpoint:
        with pytest.raises(RuntimeError) as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert type(raised.value) is RuntimeError


@pytest.mark.parametrize(
    "content_type",
    [
        "text/json",
        "text/plain+json",
        "application/octet-stream",
        "application/x-ndjson",
        "application/+json",
        "application/problem json",
        "application/problem@json",
        "application/json garbage",
        "application/json;",
        "application/json; charset",
        'application/json; charset="unterminated',
        "application/json; charset=iso-8859-1",
        'application/json; charset="iso-8859-1"',
        "application/json; charset=utf-8; charset=utf-8",
        "application/json; charset=utf-8; profile=callback",
        "application/json; profile=callback",
        "application/json, application/json",
        "application/jsoñ",
        "",
    ],
)
def test_non_json_media_type_never_supplies_failure_usage(content_type: str) -> None:
    _assert_plain_callback_failure(_failure(_EXACT_USAGE), content_type=content_type)


@pytest.mark.parametrize("content_types", [(), ("application/json", "application/json")])
def test_missing_or_repeated_json_media_type_never_supplies_failure_usage(
    content_types: tuple[str, ...],
) -> None:
    body = json.dumps(_failure(_EXACT_USAGE)).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(502)
            for content_type in content_types:
                self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    with _handler_server(Handler) as endpoint:
        with pytest.raises(RuntimeError) as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert type(raised.value) is RuntimeError


@pytest.mark.parametrize(
    "body",
    [
        json.dumps(_failure(_EXACT_USAGE)).encode("utf-16"),
        b"\xef\xbb\xbf" + json.dumps(_failure(_EXACT_USAGE)).encode("utf-8"),
        b'{"error":"api_error","usage":{"input_tokens":2,"output_tokens":3,"total_tokens":NaN}}',
        b'{"error":"api_error","usage":{"input_tokens":2,"output_tokens":3,"total_tokens":Infinity}}',
        b'{"error":"api_error","usage":{"input_tokens":2,"output_tokens":3,"total_tokens":-Infinity}}',
        b'{"error":"api_error","error":"api_error","usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}',
        b'{"error":"api_error","usage":{"input_tokens":1,"output_tokens":0,"total_tokens":1},"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}',
        b'{"error":"api_error","usage":{"input_tokens":1,"input_tokens":2,"output_tokens":3,"total_tokens":5}}',
        b'{"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}',
        b'{"error":{"type":"ProviderError"},"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}',
        b'{"error":"other_error","usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}',
        b'{"error":"api_error","usage":[2,3,5]}',
        b'[{"error":"api_error","usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}]',
        b'{"error":"api_error","usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}',
    ],
)
def test_noncanonical_callback_json_never_supplies_failure_usage(body: bytes) -> None:
    _assert_plain_callback_failure(body)


@pytest.mark.parametrize(
    "body",
    [
        (
            b'{"error":"api_error","retry_after":1e10000,"usage":'
            b'{"input_tokens":2,"output_tokens":3,"total_tokens":5}}'
        ),
        (
            b'{"error":"api_error","usage":{"input_tokens":1e10000,'
            b'"output_tokens":3,"total_tokens":5}}'
        ),
    ],
)
def test_exponent_overflowing_float_never_supplies_failure_usage(body: bytes) -> None:
    response = (502, "application/json", body)
    with _callback_server([response, response]) as endpoint:
        with pytest.raises(RuntimeError, match="HTTP 502") as subcall:
            _subcall_client(endpoint).llm_query_with_usage("question")
        with pytest.raises(RuntimeError, match="HTTP 502") as root:
            _root_client(endpoint).responses_create(
                [{"role": "user", "content": "question"}],
                "test-model",
                return_usage=True,
            )
    assert type(subcall.value) is RuntimeError
    assert type(root.value) is RuntimeError


def test_utf8_bom_is_rejected_by_success_and_ndjson_callback_parsers() -> None:
    bom = b"\xef\xbb\xbf"
    unary = bom + json.dumps({"result": "answer", "usage": _EXACT_USAGE}).encode("utf-8")
    stream = bom + json.dumps(
        {"type": "completion", "content": "answer", "usage": _EXACT_USAGE}
    ).encode("utf-8")
    responses = [
        (200, "application/json", unary),
        (200, "application/json", unary),
        (200, "application/x-ndjson", stream + b"\n"),
    ]
    with _callback_server(responses) as endpoint:
        with pytest.raises(RuntimeError) as subcall:
            _subcall_client(endpoint).llm_query_with_usage("question")
        with pytest.raises(RuntimeError) as root:
            _root_client(endpoint).responses_create(
                [{"role": "user", "content": "question"}],
                "test-model",
                return_usage=True,
            )
        with pytest.raises(RuntimeError) as ndjson:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert type(subcall.value) is RuntimeError
    assert type(root.value) is RuntimeError
    assert type(ndjson.value) is RuntimeError


def test_overly_deep_callback_json_never_supplies_failure_usage() -> None:
    body = (
        b'{"error":"api_error","usage":{"input_tokens":2,"output_tokens":3,'
        b'"total_tokens":5},"extra":' + b"[" * 2_000 + b"0" + b"]" * 2_000 + b"}"
    )
    _assert_plain_callback_failure(body)


def test_oversized_json_http_failure_is_not_parsed_for_usage() -> None:
    body = json.dumps(_failure(_EXACT_USAGE)).encode("utf-8") + b" " * (65 * 1024)
    with _callback_server([(502, "application/json", body)]) as endpoint:
        with pytest.raises(RuntimeError) as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert type(raised.value) is RuntimeError
    assert len(str(raised.value)) < 400


def test_short_close_before_declared_content_length_keeps_usage_conservative() -> None:
    body = json.dumps(_failure(_EXACT_USAGE)).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body) + 16))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, *_args: object) -> None:
            pass

    with _handler_server(Handler) as endpoint:
        with pytest.raises(RuntimeError) as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert type(raised.value) is RuntimeError


@pytest.mark.parametrize("framing", ["close", "chunked"])
def test_eof_delimited_error_body_preserves_exact_usage(framing: str) -> None:
    body = json.dumps(_failure(_EXACT_USAGE)).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            if framing == "chunked":
                self.send_header("Transfer-Encoding", "ChUnKeD")
            else:
                self.send_header("Connection", "close")
            self.end_headers()
            if framing == "chunked":
                self.wfile.write(f"{len(body):x}\r\n".encode("ascii"))
                self.wfile.write(body + b"\r\n0\r\n\r\n")
            else:
                self.wfile.write(body)
                self.close_connection = True
            self.wfile.flush()

        def log_message(self, *_args: object) -> None:
            pass

    with _handler_server(Handler) as endpoint:
        with pytest.raises(LLMUsageFailure) as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert raised.value.usage == TokenUsage(2, 3, 5, exact=True)


@pytest.mark.parametrize(
    ("transfer_encodings", "include_content_length", "chunked_wire"),
    [
        (("identity",), False, False),
        (("gzip",), False, False),
        (("chunked, chunked",), False, False),
        (("gzip, chunked",), False, False),
        (("chunked", "chunked"), False, True),
        (("chunked",), True, True),
        (("chunk ed",), False, False),
        (("chunked ",), False, False),
        (("chunked;",), False, False),
    ],
)
def test_noncanonical_transfer_encoding_keeps_usage_conservative(
    transfer_encodings: tuple[str, ...],
    include_content_length: bool,
    chunked_wire: bool,
) -> None:
    body = json.dumps(_failure(_EXACT_USAGE)).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            for transfer_encoding in transfer_encodings:
                self.send_header("Transfer-Encoding", transfer_encoding)
            if include_content_length:
                self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            if chunked_wire:
                self.wfile.write(f"{len(body):x}\r\n".encode("ascii"))
                self.wfile.write(body + b"\r\n0\r\n\r\n")
            else:
                self.wfile.write(body)
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, *_args: object) -> None:
            pass

    with _handler_server(Handler) as endpoint:
        with pytest.raises(RuntimeError) as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert type(raised.value) is RuntimeError


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (_EXACT_USAGE, TokenUsage(2, 3, 5, exact=True)),
        (
            {"input_tokens": 2, "output_tokens": 3, "total_tokens": None},
            TokenUsage(2, 3, 0, exact=False),
        ),
        (
            {
                "input_tokens": 2**63 - 1,
                "output_tokens": 0,
                "total_tokens": 2**63 - 1,
                "reasoning_tokens": 0,
                "observation_basis": "exact",
            },
            TokenUsage(2**63 - 1, 0, 2**63 - 1, exact=True),
        ),
        (
            {"input_tokens": 2**63, "output_tokens": 3, "total_tokens": 2**63 - 1},
            TokenUsage(0, 3, 2**63 - 1, exact=False),
        ),
    ],
)
def test_ndjson_error_event_preserves_structured_usage(
    usage: object,
    expected: TokenUsage,
) -> None:
    event = {
        "type": "error",
        "code": "UPSTREAM_FAILURE",
        "message": "provider failed",
        "usage": usage,
    }
    body = json.dumps(event).encode("utf-8") + b"\n"
    with _callback_server([(200, "application/x-ndjson", body)]) as endpoint:
        with pytest.raises(LLMUsageFailure, match="UPSTREAM_FAILURE") as raised:
            _subcall_client(endpoint).llm_query_with_usage("question")
    assert raised.value.usage == expected


def test_ndjson_error_message_is_bounded_and_redacted() -> None:
    secret = "sk-abc123456789"
    event = {
        "type": "error",
        "code": "token=supersecret" + "x" * 200,
        "message": f"Authorization: Bearer {secret} " + "m" * 1_000,
    }
    body = json.dumps(event).encode("utf-8") + b"\n"
    with _callback_server([(200, "application/x-ndjson", body)]) as endpoint:
        with pytest.raises(RuntimeError) as raised:
            _subcall_client(endpoint).llm_query("question")
    message = str(raised.value)
    assert secret not in message
    assert "supersecret" not in message
    assert "[redacted]" in message
    assert len(message) < 700


def test_native_batch_uses_strict_unary_fanout_for_json_and_ndjson_items() -> None:
    requests: list[tuple[str, object]] = []
    responses = [
        (200, "application/json", {"result": "first", "usage": _EXACT_USAGE}),
        (
            200,
            "application/x-ndjson",
            _raw_ndjson_completion_with_input_lexeme("2"),
        ),
    ]
    with _callback_server(responses, requests=requests) as endpoint:
        value = _subcall_client(endpoint, max_parallel=1).llm_batch_with_usage(["one", "two"])

    assert value.results == ("first", "answer")
    assert value.errors == ()
    assert value.usage == (
        TokenUsage(2, 3, 5, exact=True),
        TokenUsage(2, 3, 5, exact=True),
    )
    assert [path for path, _body in requests] == ["/callback", "/callback"]
    assert [body["prompt"] for _path, body in requests if isinstance(body, dict)] == [
        "one",
        "two",
    ]
    assert all(isinstance(body, dict) and "prompts" not in body for _path, body in requests)


@pytest.mark.parametrize(
    ("status", "content_type", "body", "expected_usage"),
    [
        (
            502,
            "application/json",
            _failure(_EXACT_USAGE),
            TokenUsage(2, 3, 5, exact=True),
        ),
        (
            502,
            "application/json",
            _failure({"input_tokens": 2, "output_tokens": "bad", "total_tokens": 5}),
            TokenUsage(2, 0, 5, exact=False),
        ),
        (
            502,
            "application/json",
            {"error": "proxy_error", "usage": _EXACT_USAGE},
            TokenUsage.unavailable(),
        ),
        (
            502,
            "application/json",
            b"<html>proxy failure</html>",
            TokenUsage.unavailable(),
        ),
        (
            502,
            "text/plain",
            _failure(_EXACT_USAGE),
            TokenUsage.unavailable(),
        ),
        (
            200,
            "application/x-ndjson",
            _raw_ndjson_error_with_input_lexeme("2"),
            TokenUsage(2, 3, 5, exact=True),
        ),
    ],
    ids=[
        "typed-exact",
        "typed-partial",
        "wrong-discriminator",
        "non-json-body",
        "non-json-media",
        "ndjson-error",
    ],
)
def test_native_batch_item_failures_preserve_only_strict_usage_evidence(
    status: int,
    content_type: str,
    body: bytes | dict[str, object],
    expected_usage: TokenUsage,
) -> None:
    with _callback_server([(status, content_type, body)]) as endpoint:
        value = _subcall_client(
            endpoint,
            max_parallel=1,
        ).llm_batch_with_errors_and_usage(["bad"])

    assert value.results == ("",)
    assert value.usage == (expected_usage,)
    assert len(value.errors) == 1
    assert value.errors[0]["index"] == 0


def test_exact_native_batch_failure_usage_refunds_for_a_near_tail_call() -> None:
    prompts = ["bad", "ok"]
    later_prompt = "later"
    subcall_output_tokens = 20
    root_output_tokens = 5
    batch_reservation = (
        conservative_token_estimate({"args": [prompts], "kwargs": {}})
        + len(prompts) * subcall_output_tokens
    )
    budget = Budget(
        tokens=root_output_tokens + batch_reservation,
        subcalls=3,
        root_output_tokens=root_output_tokens,
        subcall_output_tokens=subcall_output_tokens,
    )
    context = create_execution_context(budget=budget)
    responses = [
        (502, "application/json", _failure(_EXACT_USAGE)),
        (200, "application/json", {"result": "batch sibling", "usage": _EXACT_USAGE}),
        (200, "application/json", {"result": "later answer", "usage": _EXACT_USAGE}),
    ]
    with _callback_server(responses) as endpoint:
        brokered = broker_subcalls(
            _subcall_client(
                endpoint,
                context=context,
                max_output_tokens=subcall_output_tokens,
                max_parallel=1,
            ),
            context.ledger,
            usage_callback=context.record_subcall_usage,
            settlement_callback=context.record_subcall_settlement,
        )
        with pytest.raises(CapabilityCallError, match="HTTP 502"):
            brokered.llm_batch(prompts)
        assert brokered.llm_query(later_prompt) == "later answer"

    snapshot = context.ledger.snapshot()
    assert snapshot.consumed.tokens == 15
    assert snapshot.consumed.subcalls == 3
    assert snapshot.reserved == BudgetRequest()
    assert context.stats.subcall_total_tokens == 15
    assert context.stats.subcall_usage_complete is True


def test_partial_native_batch_failure_usage_keeps_full_batch_reservation() -> None:
    prompts = ["bad"]
    subcall_output_tokens = 20
    reservation = (
        conservative_token_estimate({"args": [prompts], "kwargs": {}}) + subcall_output_tokens
    )
    budget = Budget(
        tokens=200,
        subcalls=1,
        root_output_tokens=5,
        subcall_output_tokens=subcall_output_tokens,
    )
    context = create_execution_context(budget=budget)
    body = _failure({"input_tokens": 2, "output_tokens": "bad", "total_tokens": 5})
    with _callback_server([(502, "application/json", body)]) as endpoint:
        brokered = broker_subcalls(
            _subcall_client(
                endpoint,
                context=context,
                max_output_tokens=subcall_output_tokens,
                max_parallel=1,
            ),
            context.ledger,
            usage_callback=context.record_subcall_usage,
            settlement_callback=context.record_subcall_settlement,
        )
        with pytest.raises(CapabilityCallError, match="HTTP 502"):
            brokered.llm_batch(prompts)

    assert context.ledger.snapshot().consumed.tokens == reservation
    assert context.stats.subcall_total_tokens == 5
    assert context.stats.subcall_usage_complete is False


def test_root_http_failure_preserves_usage_only_for_usage_aware_call() -> None:
    response = (502, "application/json", _failure(_EXACT_USAGE))
    messages = [{"role": "user", "content": "question"}]
    with _callback_server([response, response]) as endpoint:
        client = _root_client(endpoint)
        with pytest.raises(LLMUsageFailure, match="HTTP 502") as raised:
            client.responses_create(messages, "test-model", return_usage=True)
        assert raised.value.usage == TokenUsage(2, 3, 5, exact=True)

        with pytest.raises(RuntimeError, match="HTTP 502") as plain:
            client.responses_create(messages, "test-model", return_usage=False)
        assert type(plain.value) is RuntimeError


def test_exact_failure_usage_refunds_capacity_for_a_near_tail_call() -> None:
    prompt = "later"
    subcall_output_tokens = 20
    root_output_tokens = 5
    reservation = conservative_token_estimate({"args": [prompt], "kwargs": {}}) + 20
    budget = Budget(
        tokens=root_output_tokens + reservation + _EXACT_USAGE["total_tokens"],
        subcalls=2,
        root_output_tokens=root_output_tokens,
        subcall_output_tokens=subcall_output_tokens,
    )
    context = create_execution_context(budget=budget)
    failed = (502, "application/json", _failure(_EXACT_USAGE))
    succeeded = (
        200,
        "application/json",
        {"result": "answer", "usage": _EXACT_USAGE},
    )
    with _callback_server([failed, succeeded]) as endpoint:
        client = _subcall_client(
            endpoint,
            context=context,
            max_output_tokens=subcall_output_tokens,
        )
        brokered = broker_subcalls(
            client,
            context.ledger,
            usage_callback=context.record_subcall_usage,
            settlement_callback=context.record_subcall_settlement,
        )
        with pytest.raises(CapabilityCallError, match="HTTP 502"):
            brokered.llm_query(prompt)
        assert brokered.llm_query(prompt) == "answer"

    snapshot = context.ledger.snapshot()
    assert snapshot.consumed.tokens == 10
    assert snapshot.consumed.subcalls == 2
    assert snapshot.reserved == BudgetRequest()
    assert context.stats.subcall_total_tokens == 10
    assert context.stats.subcall_usage_complete is True


@pytest.mark.parametrize(
    "failure_body",
    [
        {"error": "api_error", "code": "PROVIDER_ERROR", "message": "no usage"},
        _failure({"input_tokens": 2, "output_tokens": 3, "total_tokens": "five"}),
        _failure({"input_tokens": 2**63, "output_tokens": 3, "total_tokens": 2**63 - 1}),
        _raw_failure_with_input_lexeme("9" * 5_000),
        (
            b'{"error":"api_error","retry_after":1e10000,"usage":'
            b'{"input_tokens":2,"output_tokens":3,"total_tokens":5}}'
        ),
        b'{"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}\x80',
    ],
)
def test_untrustworthy_failure_usage_keeps_full_reservation(
    failure_body: bytes | dict[str, object],
) -> None:
    prompt = "question"
    subcall_output_tokens = 20
    reservation = (
        conservative_token_estimate({"args": [prompt], "kwargs": {}}) + subcall_output_tokens
    )
    budget = Budget(
        tokens=200,
        subcalls=1,
        root_output_tokens=5,
        subcall_output_tokens=subcall_output_tokens,
    )
    context = create_execution_context(budget=budget)
    with _callback_server([(502, "application/json", failure_body)]) as endpoint:
        brokered = broker_subcalls(
            _subcall_client(endpoint, context=context),
            context.ledger,
            usage_callback=context.record_subcall_usage,
            settlement_callback=context.record_subcall_settlement,
        )
        with pytest.raises(CapabilityCallError, match="HTTP 502"):
            brokered.llm_query(prompt)

    assert context.ledger.snapshot().consumed.tokens == reservation
    assert context.stats.subcall_usage_complete is False
