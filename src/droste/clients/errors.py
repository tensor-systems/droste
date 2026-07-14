"""Shared HTTP-error surfacing for the engine's built-in clients.

Bare "HTTP 502: Bad Gateway" errors destroy the server's actual explanation
(e.g. a circuit-breaker rejection vs a provider error), which has cost real
diagnosis time. These helpers were born in droste_runner (a 2026-07 incident where an oversized upstream error body leaked verbatim) and
moved here so the BYOK OpenAI-compatible client and the ModelRelay runner
clients share one bounded-read + redaction implementation.
"""

from __future__ import annotations

import urllib.error

from ..redaction import redact_secrets


def http_error_excerpt(exc: "urllib.error.HTTPError", limit: int = 300) -> str:
    """Return a short, redacted response-body excerpt for an HTTP error.

    The read is byte-bounded (a chunked/stalled error body must not hang the
    client or bypass response-size limits), the excerpt is redacted (it flows
    into exception text and from there into the repair prompt shown to the
    root LLM), and any failure degrades to the empty string — never raise.
    """
    try:
        body = exc.read(4 * limit)
    except Exception:
        return ""
    if not body:
        return ""
    text = body.decode("utf-8", errors="replace").strip()
    text = " ".join(text.split())
    text = redact_secrets(text)
    if len(text) > limit:
        text = text[:limit] + "..."
    return text
