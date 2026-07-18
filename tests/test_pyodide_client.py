"""Pyodide substrate root-client wire behavior."""

from __future__ import annotations

import json

from droste.protocols.llm_client import CACHE_ANCHOR_MARKER
from droste.substrates.pyodide import BridgedLLMClient


def test_bridged_llm_client_strips_cache_anchor_marker_from_payload() -> None:
    captured: list[dict] = []

    def host_fetch(_method: str, _url: str, _headers: str, body: str) -> str:
        captured.append(json.loads(body))
        return json.dumps(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "ok"}],
                    }
                ]
            }
        )

    messages = [{"role": "user", "content": "q", CACHE_ANCHOR_MARKER: True}]
    client = BridgedLLMClient(host_fetch, api_key="mr_sk_test")

    assert client.responses_create(messages, model="root-model") == "ok"
    assert CACHE_ANCHOR_MARKER not in json.dumps(captured)
    assert CACHE_ANCHOR_MARKER in messages[0]
