"""Shared HTTP-error surfacing for the engine's built-in clients.

Bare "HTTP 502: Bad Gateway" errors destroy the server's actual explanation
(e.g. a circuit-breaker rejection vs a provider error), which has cost real
diagnosis time. These helpers were born in droste_runner (a 2026-07 incident where an oversized upstream error body leaked verbatim) and
moved here so the BYOK OpenAI-compatible client and the ModelRelay runner
clients share one bounded-read + redaction implementation.
"""

from __future__ import annotations

import http.client
import urllib.error
from dataclasses import dataclass

from ..redaction import redact_secrets

_MAX_CONTENT_LENGTH_FIELDS = 8
_MAX_CONTENT_LENGTH_FIELD_CHARS = 128
_MAX_CONTENT_LENGTH_DIGITS = 20
_CHUNKED_TRANSFER_ENCODING = "chunked"


@dataclass(frozen=True, slots=True)
class HTTPErrorBody:
    """One bounded HTTP error body read, reusable for parsing and display."""

    body: bytes
    complete: bool


def _declared_content_length(exc: "urllib.error.HTTPError") -> tuple[int | None, bool]:
    try:
        headers = getattr(exc, "headers", None) or getattr(exc, "hdrs", None)
        if headers is None:
            return None, True
        get_all = getattr(headers, "get_all", None)
        if callable(get_all):
            raw_values = get_all("Content-Length") or []
            transfer_encodings = get_all("Transfer-Encoding") or []
        else:
            raw = headers.get("Content-Length")
            raw_values = [] if raw is None else [raw]
            transfer = headers.get("Transfer-Encoding")
            transfer_encodings = [] if transfer is None else [transfer]
        if not isinstance(raw_values, (list, tuple)) or not isinstance(
            transfer_encodings, (list, tuple)
        ):
            return None, False
        if len(raw_values) > _MAX_CONTENT_LENGTH_FIELDS:
            return None, False
        if raw_values and transfer_encodings:
            return None, False
        if transfer_encodings:
            if len(transfer_encodings) != 1:
                return None, False
            transfer_encoding = transfer_encodings[0]
            if (
                not isinstance(transfer_encoding, str)
                or len(transfer_encoding) != len(_CHUNKED_TRANSFER_ENCODING)
                or not transfer_encoding.isascii()
                or transfer_encoding.lower() != _CHUNKED_TRANSFER_ENCODING
            ):
                return None, False
            return None, True
        if not raw_values:
            return None, True
        lengths: list[int] = []
        for raw_value in raw_values:
            if not isinstance(raw_value, str) or len(raw_value) > _MAX_CONTENT_LENGTH_FIELD_CHARS:
                return None, False
            for raw_part in raw_value.split(","):
                if len(lengths) >= _MAX_CONTENT_LENGTH_FIELDS:
                    return None, False
                part = raw_part.strip()
                if (
                    not part
                    or len(part) > _MAX_CONTENT_LENGTH_DIGITS
                    or not part.isascii()
                    or not part.isdigit()
                ):
                    return None, False
                lengths.append(int(part))
        if not lengths or any(length != lengths[0] for length in lengths[1:]):
            return None, False
        return lengths[0], True
    except Exception:
        return None, False


def read_http_error_body(exc: "urllib.error.HTTPError", limit: int = 64 * 1024) -> HTTPErrorBody:
    """Read an HTTP error body once, bounded, and report whether it was complete."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("HTTP error body limit must be a positive integer")
    declared_length, framing_valid = _declared_content_length(exc)
    read_complete = True
    try:
        raw = exc.read(limit + 1)
    except http.client.IncompleteRead as error:
        raw = error.partial[: limit + 1]
        read_complete = False
    except Exception:
        return HTTPErrorBody(b"", complete=False)
    if not isinstance(raw, bytes):
        return HTTPErrorBody(b"", complete=False)
    complete = (
        read_complete
        and framing_valid
        and len(raw) <= limit
        and (declared_length is None or len(raw) == declared_length)
    )
    return HTTPErrorBody(raw[:limit], complete=complete)


def http_error_body_excerpt(body: bytes, limit: int = 300) -> str:
    """Return a short, redacted excerpt from an already-read response body."""

    if not body:
        return ""
    text = body.decode("utf-8", errors="replace").strip()
    text = " ".join(text.split())
    text = redact_secrets(text)
    if len(text) > limit:
        text = text[:limit] + "..."
    return text


def http_error_excerpt(exc: "urllib.error.HTTPError", limit: int = 300) -> str:
    """Return a short, redacted response-body excerpt for an HTTP error.

    The read is byte-bounded (a chunked/stalled error body must not hang the
    client or bypass response-size limits), the excerpt is redacted (it flows
    into exception text and from there into the repair prompt shown to the
    root LLM), and any failure degrades to the empty string — never raise.
    """
    captured = read_http_error_body(exc, 4 * limit)
    return http_error_body_excerpt(captured.body, limit)
