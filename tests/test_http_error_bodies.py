"""HTTP errors from the runner's clients must carry the response body.

Bare "HTTP 502: Bad Gateway" destroyed the server's actual explanation during
a 2026-07 provider incident (a circuit-breaker rejection was
indistinguishable from an upstream provider error).
"""

from __future__ import annotations

import io
import json
import threading
import urllib.error
from email.message import Message
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from droste.clients.errors import read_http_error_body
from droste.execution.context import create_execution_context
from droste_runner.runner import HTTPSubcallClient, _http_error_excerpt


def _http_error(
    body: bytes,
    code: int = 502,
    headers: Message | None = None,
) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://test.invalid",
        code=code,
        msg="Bad Gateway",
        hdrs=headers,
        fp=io.BytesIO(body),
    )


def _headers(
    *content_lengths: str,
    transfer_encoding: str | tuple[str, ...] | None = None,
) -> Message:
    headers = Message()
    for value in content_lengths:
        headers.add_header("Content-Length", value)
    transfer_encodings = (
        transfer_encoding if isinstance(transfer_encoding, tuple) else (transfer_encoding,)
    )
    for value in transfer_encodings:
        if value is not None:
            headers.add_header("Transfer-Encoding", value)
    return headers


def test_excerpt_returns_normalized_body():
    exc = _http_error(b'provider "google-ai-studio" is temporarily\ncircuit-broken until T')
    assert (
        _http_error_excerpt(exc)
        == 'provider "google-ai-studio" is temporarily circuit-broken until T'
    )


def test_excerpt_truncates_long_bodies():
    exc = _http_error(b"x" * 1000)
    out = _http_error_excerpt(exc)
    assert len(out) == 303 and out.endswith("...")


def test_excerpt_degrades_to_empty_on_unreadable_body():
    exc = urllib.error.HTTPError("http://test.invalid", 502, "Bad Gateway", None, None)
    assert _http_error_excerpt(exc) == ""


def test_llm_query_error_includes_server_body():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = b'no healthy provider offers model "gemini-3.5-flash"'
            self.send_response(503)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = HTTPSubcallClient(
            endpoint=f"http://127.0.0.1:{server.server_address[1]}/subcall",
            token="t",
            session="",
            session_index=0,
            context=create_execution_context(),
        )
        with pytest.raises(RuntimeError, match=r"HTTP 503: no healthy provider offers model"):
            client.llm_query("hi")
    finally:
        server.shutdown()


def test_llm_query_streams_and_reassembles_ndjson():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            assert self.headers.get("Accept") == (
                'application/x-ndjson; profile="responses-stream/v2"'
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            events = [
                {"type": "start"},
                {"type": "keepalive"},
                {"type": "reasoning_delta", "reasoning_delta": "thinking"},
                {"type": "update", "delta": "slow "},
                {"type": "update", "delta": "model"},
                {"type": "completion"},
            ]
            for event in events:
                self.wfile.write(json.dumps(event).encode() + b"\n")
                self.wfile.flush()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = HTTPSubcallClient(
            endpoint=f"http://127.0.0.1:{server.server_address[1]}/subcall",
            token="t",
            session="",
            session_index=0,
            context=create_execution_context(),
        )
        assert client.llm_query("hi") == "slow model"
    finally:
        server.shutdown()


def test_llm_query_rejects_truncated_ndjson_stream():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.wfile.write(b'{"type":"update","delta":"partial"}\n')

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = HTTPSubcallClient(
            endpoint=f"http://127.0.0.1:{server.server_address[1]}/subcall",
            token="t",
            session="",
            session_index=0,
            context=create_execution_context(),
        )
        with pytest.raises(RuntimeError, match="without a completion event"):
            client.llm_query("hi")
    finally:
        server.shutdown()


def test_llm_query_preserves_explicitly_empty_completion_content():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.wfile.write(b'{"type":"update","delta":"superseded"}\n')
            self.wfile.write(b'{"type":"completion","content":""}\n')

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = HTTPSubcallClient(
            endpoint=f"http://127.0.0.1:{server.server_address[1]}/subcall",
            token="t",
            session="",
            session_index=0,
            context=create_execution_context(),
        )
        assert client.llm_query("hi") == ""
    finally:
        server.shutdown()


def test_llm_query_surfaces_ndjson_error_event():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.wfile.write(
                b'{"type":"error","code":"UPSTREAM_TIMEOUT","message":"provider stalled"}\n'
            )

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = HTTPSubcallClient(
            endpoint=f"http://127.0.0.1:{server.server_address[1]}/subcall",
            token="t",
            session="",
            session_index=0,
            context=create_execution_context(),
        )
        with pytest.raises(RuntimeError, match=r"UPSTREAM_TIMEOUT.*provider stalled"):
            client.llm_query("hi")
    finally:
        server.shutdown()


def test_excerpt_bounds_the_read():
    class SlowEndlessBody:
        """read(n) returns n bytes; read() with no arg would never finish."""

        def read(self, n=-1):
            if n is None or n < 0:
                raise AssertionError("unbounded read() must not be used")
            return b"x" * n

    exc = urllib.error.HTTPError("http://test.invalid", 502, "Bad Gateway", None, None)
    exc.read = SlowEndlessBody().read  # type: ignore[method-assign]
    out = _http_error_excerpt(exc)
    assert out.endswith("...") and len(out) == 303


def test_excerpt_redacts_secrets():
    body = b'{"error":"upstream said Authorization: Bearer sk-abc123456789 failed", "api_key": "AIzaSyFAKE-KEY", "detail":"key=supersecret&x=1"}'
    exc = _http_error(body)
    out = _http_error_excerpt(exc)
    assert "sk-abc123456789" not in out
    assert "AIzaSyFAKE-KEY" not in out
    assert "supersecret" not in out
    assert "[redacted]" in out


def test_error_body_shorter_than_declared_length_is_incomplete() -> None:
    body = b'{"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}'
    captured = read_http_error_body(_http_error(body, headers=_headers(str(len(body) + 10))))
    assert captured.body == body
    assert captured.complete is False


@pytest.mark.parametrize(
    "headers",
    [
        _headers("-1"),
        _headers("+1"),
        _headers("1.0"),
        _headers(""),
        _headers("١"),
        _headers("not-a-number"),
        _headers("9" * 21),
        _headers("9" * 100_000),
        _headers(",".join(["5"] * 9)),
        _headers("5", "6"),
        _headers("5, 6"),
        _headers("5", transfer_encoding="chunked"),
        _headers(transfer_encoding="identity"),
        _headers(transfer_encoding="gzip"),
        _headers(transfer_encoding="chunked, chunked"),
        _headers(transfer_encoding="gzip, chunked"),
        _headers(transfer_encoding=("chunked", "chunked")),
        _headers(transfer_encoding=" chunked"),
        _headers(transfer_encoding="chunked "),
        _headers(transfer_encoding="chunk ed"),
        _headers(transfer_encoding="chunked;"),
    ],
)
def test_error_body_invalid_or_conflicting_framing_is_incomplete(headers: Message) -> None:
    captured = read_http_error_body(_http_error(b"12345", headers=headers))
    assert captured.body == b"12345"
    assert captured.complete is False


def test_error_body_identical_lengths_and_no_length_eof_are_complete() -> None:
    duplicate = read_http_error_body(_http_error(b"12345", headers=_headers("5", "5")))
    no_length = read_http_error_body(_http_error(b"12345"))
    chunked = read_http_error_body(
        _http_error(b"12345", headers=_headers(transfer_encoding="ChUnKeD"))
    )
    assert duplicate.complete is True
    assert no_length.complete is True
    assert chunked.complete is True
