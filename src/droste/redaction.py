"""Dependency-neutral secret redaction shared by public and client values."""

from __future__ import annotations

import re

_SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(
        r"""(?i)\b(api[_-]?key|apikey|token|authorization|secret|password|key)\b(["\'\s]*[:=]["\'\s]*)[^\s"\'&,;}]+"""
    ),
    re.compile(r"\bmr_sk_[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
)


def redact_secrets(text: str) -> str:
    """Replace recognized credential forms without changing safe text.

    Reapplying the function to its own output is stable, which lets already
    sanitized values cross multiple construction boundaries without accumulating
    additional markers.
    """

    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(
            lambda match: (
                (
                    match.group(1) + match.group(2)
                    if match.lastindex and match.lastindex >= 2
                    else ""
                )
                + "[redacted]"
            ),
            text,
        )
    return text
