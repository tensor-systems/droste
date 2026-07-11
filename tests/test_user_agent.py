"""Every HTTP request the engine builds must identify itself (#49):
urllib's default Python-urllib UA is a bot signature that WAF edges
(including ModelRelay's — Cloudflare error 1010) block before auth, so a
fresh install failed on its first real call."""

from __future__ import annotations

from droste.clients.useragent import USER_AGENT


def test_user_agent_names_engine_and_version() -> None:
    assert USER_AGENT.startswith("droste/")
    assert USER_AGENT != "droste/"  # version or the literal "unknown"


def test_modelrelay_request_carries_the_user_agent() -> None:
    from droste.clients.modelrelay import _ResponsesTransport

    req = _ResponsesTransport(
        base_url="https://example.invalid/api/v1", api_key="mr_sk_test", timeout=1, label="t"
    )._request({"model": "m"})
    assert req.get_header("User-agent") == USER_AGENT


def test_openai_compat_request_carries_the_user_agent(monkeypatch) -> None:
    import droste.clients.openai_compat as oc

    captured = {}

    class _Resp:
        def read(self):
            return b'{"choices": []}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=0):
        captured["ua"] = req.get_header("User-agent")
        return _Resp()

    monkeypatch.setattr(oc.urllib.request, "urlopen", _fake_urlopen)
    oc._ChatCompletionsTransport(
        base_url="https://example.invalid/v1", api_key="k", timeout=1, label="t"
    ).complete({"model": "m"})
    assert captured["ua"] == USER_AGENT


def test_anthropic_request_carries_the_user_agent() -> None:
    from droste.clients.anthropic import _MessagesTransport

    req = _MessagesTransport(
        base_url="https://example.invalid", api_key="sk-ant-test", timeout=1, label="t"
    )._request({"model": "m"})
    assert req.get_header("User-agent") == USER_AGENT


def test_runner_subcall_request_carries_the_user_agent(monkeypatch) -> None:
    import urllib.request

    from droste.execution.context import create_execution_context
    from droste_runner.runner import HTTPSubcallClient

    captured = {}

    class _Resp:
        def read(self):
            return b'{"result": "ok"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=0):
        captured["ua"] = req.get_header("User-agent")
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    client = HTTPSubcallClient(
        endpoint="https://example.invalid/subcall",
        token="t",
        session="",
        session_index=0,
        max_calls=5,
        max_depth=2,
        context=create_execution_context(max_calls=5, max_depth=2),
    )
    assert client.llm_query("p") == "ok"
    assert captured["ua"] == USER_AGENT
