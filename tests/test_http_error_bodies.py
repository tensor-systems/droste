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
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from droste.execution.context import create_execution_context
from droste_runner.runner import HTTPSubcallClient, _http_error_excerpt


def _http_error(body: bytes, code: int = 502) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://test.invalid", code=code, msg="Bad Gateway", hdrs=None, fp=io.BytesIO(body)
    )


def test_excerpt_returns_normalized_body():
    exc = _http_error(b'provider "google-ai-studio" is temporarily\ncircuit-broken until T')
    assert _http_error_excerpt(exc) == 'provider "google-ai-studio" is temporarily circuit-broken until T'


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
            max_calls=5,
            max_depth=5,
            context=create_execution_context(max_calls=5, max_depth=5),
        )
        with pytest.raises(RuntimeError, match=r'HTTP 503: no healthy provider offers model'):
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
