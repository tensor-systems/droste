"""One User-Agent for every HTTP request the engine makes (#49).

urllib's default (``Python-urllib/3.x``) is a canonical bot-fight signature:
WAF edges — including ModelRelay's — block it outright (Cloudflare error
1010), so a fresh install failed before ever reaching auth. Every
``urllib.request.Request`` the engine builds sends this instead: identify
honestly, and the block never applies.
"""

from __future__ import annotations


def _engine_version() -> str:
    try:
        from importlib.metadata import version

        return version("droste")
    except Exception:
        return "unknown"


USER_AGENT = f"droste/{_engine_version()}"
