from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from typing import Any

import pytest

from droste import (
    CapabilityBroker,
    CapabilityCall,
    CapabilityCallError,
    CapabilityId,
    CapabilityKind,
    ConfiguredSource,
    ProviderRegistry,
)
from droste.sources import (
    McpHttpHost,
    McpSecretRequest,
    open_mcp_http_source,
)
from droste.sources._mcp_http_transport import (
    McpHttpSession,
    _Endpoint,
    _HttpResponse,
    _SseDisconnected,
)
from droste.sources._mcp_stdio_transport import McpTransportError
from droste.testing import (
    LifecycleGate,
    RecordingAttemptAuthority,
    require_unknown_completion,
    run_while_blocked,
)

TOOLS = [
    {
        "name": "ReadFile",
        "description": "Read one allowlisted file.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    }
]


def _source(
    *,
    source_id: str = "remote_docs",
    endpoint: str = "https://mcp.example.test/mcp",
    auth: Mapping[str, Any] | None = None,
    **extra: Any,
) -> ConfiguredSource:
    return ConfiguredSource(
        source_id,
        "reference_filesystem",
        {
            "endpoint": endpoint,
            "allowed_endpoints": [endpoint],
            "tenant_id": "tenant-a",
            "auth": dict(auth or {"type": "none"}),
            "allowed_tools": ["ReadFile"],
            "bindings": {"ReadFile": "read_file"},
            "effects": {"ReadFile": "read"},
            "budget_classes": {"ReadFile": "data.read"},
            "policy_metadata": {"ReadFile": {"read_only": True}},
            "source_description": "Remote reference documents.",
            **extra,
        },
    )


class _Server:
    def __init__(self) -> None:
        self.session = "session-1"
        self.initializations = 0
        self.calls = 0
        self.expire_once = False
        self.redirect = False
        self.expected_token: str | None = None
        self.requests: list[tuple[str, str, Mapping[str, str], bytes | None]] = []

    def exchange(
        self,
        session: McpHttpSession,
        method: str,
        endpoint: _Endpoint,
        headers: Mapping[str, str],
        body: bytes | None,
        **_: Any,
    ) -> _HttpResponse:
        del session
        self.requests.append((method, endpoint.url, dict(headers), body))
        if self.redirect:
            return _HttpResponse(302, {"location": "https://169.254.169.254/latest"}, b"")
        if self.expected_token is not None and headers.get("Authorization") != (
            f"Bearer {self.expected_token}"
        ):
            return _HttpResponse(401, {}, b"")
        if method == "DELETE":
            return _HttpResponse(204, {}, b"")
        assert body is not None
        payload = json.loads(body)
        method_name = payload["method"]
        if method_name == "notifications/initialized":
            return _HttpResponse(202, {}, b"")
        if method_name == "initialize":
            self.initializations += 1
            response = {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fixture", "version": "1"},
                },
            }
            return _json_response(response, {"mcp-session-id": self.session})
        if headers.get("MCP-Session-Id") != self.session:
            return _HttpResponse(404, {}, b"")
        if method_name == "tools/list":
            return _json_response(
                {"jsonrpc": "2.0", "id": payload["id"], "result": {"tools": TOOLS}}
            )
        assert method_name == "tools/call"
        if self.expire_once:
            self.expire_once = False
            self.session = "session-2"
            return _HttpResponse(404, {}, b"")
        self.calls += 1
        path = payload["params"]["arguments"]["path"]
        return _json_response(
            {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {
                    "content": [{"type": "text", "text": f"raw:{path}"}],
                    "structuredContent": {"text": f"value:{path}"},
                },
            }
        )


def _json_response(value: Any, headers: Mapping[str, str] | None = None) -> _HttpResponse:
    return _HttpResponse(
        200,
        {"content-type": "application/json", **dict(headers or {})},
        json.dumps(value, separators=(",", ":")).encode(),
    )


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> _Server:
    value = _Server()

    def exchange(session: McpHttpSession, *args: Any, **kwargs: Any) -> _HttpResponse:
        return value.exchange(session, *args, **kwargs)

    monkeypatch.setattr(McpHttpSession, "_exchange", exchange)
    return value


def test_http_source_uses_same_manifest_broker_and_content_free_trace(
    server: _Server,
) -> None:
    bound = open_mcp_http_source(_source(), McpHttpHost())
    durable: list[dict[str, Any]] = []
    registry = ProviderRegistry((bound,))
    broker = CapabilityBroker(
        registry.capability_registrations(),
        observer=lambda item: durable.append(item.to_trace_dict()),
    )
    generated = registry.broker_globals(broker)["remote_docs"]
    try:
        assert generated.read_file(path="private.txt") == {"text": "value:private.txt"}
        manifest = bound.registration.manifest
        assert manifest.operations[0].operation_id == "ReadFile"
        assert manifest.operations[0].binding_name == "read_file"
        assert "MCP" not in registry.prompt_fragment()
        encoded = json.dumps(durable)
        assert "private.txt" not in encoded
        assert "raw:" not in encoded
        stats = bound.runtime.stats()
        assert stats["transport"] == "streamable_http"
        assert stats["requests_made"] == 1
    finally:
        registry.close()


def test_session_expiry_reconnects_and_revalidates_frozen_descriptor(server: _Server) -> None:
    bound = open_mcp_http_source(_source(), McpHttpHost())
    registry = ProviderRegistry((bound,))
    broker = CapabilityBroker(registry.capability_registrations())
    generated = registry.broker_globals(broker)["remote_docs"]
    server.expire_once = True
    try:
        assert generated.read_file(path="guide.md") == {"text": "value:guide.md"}
        assert server.initializations == 2
        assert bound.runtime.stats()["reconnects"] == 1
    finally:
        registry.close()


def test_sse_disconnect_resumes_with_get_last_event_id_and_bounded_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server()
    interrupted = False
    interrupted_request_id = 0
    resumed_headers: list[Mapping[str, str]] = []

    def exchange(
        session: McpHttpSession,
        method: str,
        endpoint: _Endpoint,
        headers: Mapping[str, str],
        body: bytes | None,
        **kwargs: Any,
    ) -> _HttpResponse:
        nonlocal interrupted, interrupted_request_id
        if body and json.loads(body).get("method") == "tools/call" and not interrupted:
            interrupted = True
            interrupted_request_id = json.loads(body)["id"]
            raise _SseDisconnected("event-7", 0)
        if method == "GET":
            resumed_headers.append(dict(headers))
            payload = {
                "jsonrpc": "2.0",
                "id": interrupted_request_id,
                "result": {
                    "content": [{"type": "text", "text": "raw"}],
                    "structuredContent": {"text": "resumed"},
                },
            }
            body_bytes = f"id: event-8\ndata: {json.dumps(payload)}\n\n".encode()
            return _HttpResponse(200, {"content-type": "text/event-stream"}, body_bytes)
        return server.exchange(session, method, endpoint, headers, body, **kwargs)

    monkeypatch.setattr(McpHttpSession, "_exchange", exchange)
    bound = open_mcp_http_source(_source(backoff_ms=10), McpHttpHost())
    registry = ProviderRegistry((bound,))
    broker = CapabilityBroker(registry.capability_registrations())
    try:
        assert registry.broker_globals(broker)["remote_docs"].read_file(path="guide") == {
            "text": "resumed"
        }
        assert resumed_headers[0]["Last-Event-ID"] == "event-7"
        assert resumed_headers[0]["MCP-Session-Id"] == "session-1"
    finally:
        registry.close()


def test_sse_resume_session_expiry_never_replays_accepted_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server()
    tool_posts = 0

    def exchange(
        session: McpHttpSession,
        method: str,
        endpoint: _Endpoint,
        headers: Mapping[str, str],
        body: bytes | None,
        **kwargs: Any,
    ) -> _HttpResponse:
        nonlocal tool_posts
        if body and json.loads(body).get("method") == "tools/call":
            tool_posts += 1
            raise _SseDisconnected("accepted-1", 0)
        if method == "GET":
            return _HttpResponse(404, {}, b"")
        return server.exchange(session, method, endpoint, headers, body, **kwargs)

    monkeypatch.setattr(McpHttpSession, "_exchange", exchange)
    bound = open_mcp_http_source(_source(effects={"ReadFile": "effectful"}), McpHttpHost())
    registry = ProviderRegistry((bound,))
    broker = CapabilityBroker(registry.capability_registrations())
    try:
        with pytest.raises(CapabilityCallError) as captured:
            registry.broker_globals(broker)["remote_docs"].read_file(path="effectful-risk")
        assert captured.value.error.code == "mcp.transport_error"
        assert "accepted request" in captured.value.error.message
        require_unknown_completion(captured.value.error, attempts=tool_posts)
        assert server.initializations == 1
    finally:
        registry.close()


def test_redirects_and_non_exact_endpoints_fail_closed(
    server: _Server,
) -> None:
    server.redirect = True
    with pytest.raises(McpTransportError, match="redirects are forbidden"):
        open_mcp_http_source(_source(), McpHttpHost())
    with pytest.raises(ValueError, match="exactly allowlisted"):
        open_mcp_http_source(
            ConfiguredSource(
                "remote_docs",
                "reference_filesystem",
                {
                    **_source().config_dict(),
                    "allowed_endpoints": ["https://other.example.test/mcp"],
                },
            ),
            McpHttpHost(),
        )


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "0.0.0.0",
        "::1",
        "fe80::1",
    ],
)
def test_dns_private_rebinding_and_metadata_addresses_fail_closed(
    server: _Server, address: str
) -> None:
    bound = open_mcp_http_source(_source(), McpHttpHost())
    session = bound.runtime.close_callback.__self__
    session._host = McpHttpHost(resolver=lambda host, port: (address,))
    with pytest.raises(McpTransportError, match="forbidden network"):
        session._resolve(session._endpoint)
    bound.close()


def test_explicit_private_network_is_still_pinned_to_named_cidr(server: _Server) -> None:
    bound = open_mcp_http_source(
        _source(),
        McpHttpHost(
            resolver=lambda host, port: ("10.8.1.4",),
            allowed_networks=("10.8.0.0/16",),
        ),
    )
    session = bound.runtime.close_callback.__self__
    assert session._resolve(session._endpoint) == ("10.8.1.4",)
    session._host = McpHttpHost(
        resolver=lambda host, port: ("10.9.1.4",), allowed_networks=("10.8.0.0/16",)
    )
    with pytest.raises(McpTransportError, match="forbidden network"):
        session._resolve(session._endpoint)
    bound.close()


def test_hung_dns_is_time_bounded_and_cannot_multiply_resolver_threads(
    server: _Server,
) -> None:
    bound = open_mcp_http_source(_source(), McpHttpHost())
    session = bound.runtime.close_callback.__self__
    started = threading.Event()
    release = threading.Event()

    def stalled(host: str, port: int) -> tuple[str, ...]:
        del host, port
        started.set()
        release.wait(1)
        return ("203.0.113.10",)

    session._host = McpHttpHost(resolver=stalled)
    began = time.monotonic()
    with pytest.raises(McpTransportError, match="DNS resolution timed out"):
        session._resolve_bounded(session._endpoint, execution=None, timeout_ms=30)
    assert time.monotonic() - began < 0.2
    assert started.is_set()
    with pytest.raises(McpTransportError, match="already busy"):
        session._resolve_bounded(session._endpoint, execution=None, timeout_ms=30)
    release.set()
    deadline = time.monotonic() + 1
    while session._dns_lock.locked() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not session._dns_lock.locked()
    bound.close()


def test_close_uses_one_total_close_deadline_without_reacquiring_auth(server: _Server) -> None:
    server.expected_token = "token"
    bound = open_mcp_http_source(
        _source(
            auth={"type": "bearer", "token_ref": "token"},
            close_timeout_ms=30,
        ),
        McpHttpHost(resolve_secret=lambda request: "token"),
    )
    session = bound.runtime.close_callback.__self__
    session._auth_lock.acquire()
    began = time.monotonic()
    try:
        bound.close()
    finally:
        session._auth_lock.release()
    assert time.monotonic() - began < 0.2
    assert session._tokens is None
    assert session._last_access_token is None


def test_hung_secret_resolution_is_bounded_and_cannot_multiply_workers(server: _Server) -> None:
    server.expected_token = "token"
    bound = open_mcp_http_source(
        _source(
            auth={"type": "bearer", "token_ref": "token"},
            request_timeout_ms=100,
        ),
        McpHttpHost(resolve_secret=lambda request: "token"),
    )
    session = bound.runtime.close_callback.__self__
    started = threading.Event()
    release = threading.Event()

    def stalled(request: McpSecretRequest) -> str:
        del request
        started.set()
        release.wait(1)
        return "token"

    session._host = McpHttpHost(resolve_secret=stalled)
    began = time.monotonic()
    with pytest.raises(McpTransportError, match="secret resolution timed out"):
        session._access_token()
    assert time.monotonic() - began < 0.3
    assert started.is_set()
    with pytest.raises(McpTransportError, match="already busy"):
        session._access_token()
    release.set()
    deadline = time.monotonic() + 1
    while session._secret_lock.locked() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not session._secret_lock.locked()
    bound.close()


def test_secret_resolution_is_tenant_and_source_scoped(server: _Server) -> None:
    observed: list[McpSecretRequest] = []

    def resolve(request: McpSecretRequest) -> str:
        observed.append(request)
        return f"token-for-{request.source_id}"

    for source_id in ("docs_a", "docs_b"):
        server.expected_token = f"token-for-{source_id}"
        bound = open_mcp_http_source(
            _source(source_id=source_id, auth={"type": "bearer", "token_ref": "mcp/token"}),
            McpHttpHost(resolve_secret=resolve),
        )
        bound.close()
    assert observed
    assert {(item.tenant_id, item.source_id, item.reference) for item in observed} == {
        ("tenant-a", "docs_a", "mcp/token"),
        ("tenant-a", "docs_b", "mcp/token"),
    }
    serialized = json.dumps(
        [request[3].decode() if request[3] is not None else None for request in server.requests]
    )
    assert "token-for-" not in serialized


def test_debug_payloads_are_opt_in_bounded_and_exclude_credentials(server: _Server) -> None:
    debug: list[tuple[str, bytes]] = []
    server.expected_token = "top-secret"
    bound = open_mcp_http_source(
        _source(
            auth={"type": "bearer", "token_ref": "token"},
            max_debug_payload_bytes=32,
        ),
        McpHttpHost(
            resolve_secret=lambda request: "top-secret",
            debug_sink=lambda direction, payload: debug.append((direction, payload)),
        ),
    )
    try:
        registry = ProviderRegistry((bound,))
        broker = CapabilityBroker(registry.capability_registrations())
        registry.broker_globals(broker)["remote_docs"].read_file(path="sensitive")
        assert debug
        assert all(len(payload) <= 32 for _, payload in debug)
        assert "top-secret" not in repr(debug)
    finally:
        bound.close()


def test_oauth_discovery_acquisition_and_401_refresh_keep_secrets_host_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_url = "https://mcp.example.test/mcp"
    metadata_url = "https://mcp.example.test/.well-known/oauth-protected-resource/mcp"
    issuer = "https://auth.example.test/"
    discovery_url = "https://auth.example.test/.well-known/oauth-authorization-server"
    token_url = "https://auth.example.test/token"
    secret_requests: list[McpSecretRequest] = []
    token_requests: list[bytes] = []
    token_executions: list[object | None] = []
    token_count = 0
    reject_call_once = True
    mcp = _Server()

    def resolve(request: McpSecretRequest) -> str:
        secret_requests.append(request)
        return {"oauth/client-id": "client", "oauth/client-secret": "secret"}[request.reference]

    def exchange(
        session: McpHttpSession,
        method: str,
        endpoint: _Endpoint,
        headers: Mapping[str, str],
        body: bytes | None,
        **kwargs: Any,
    ) -> _HttpResponse:
        nonlocal token_count, reject_call_once
        if endpoint.url == metadata_url:
            return _json_response({"resource": mcp_url, "authorization_servers": [issuer]})
        if endpoint.url == discovery_url:
            return _json_response(
                {
                    "issuer": issuer,
                    "token_endpoint": token_url,
                    "grant_types_supported": ["client_credentials", "refresh_token"],
                }
            )
        if endpoint.url == token_url:
            assert method == "POST"
            assert headers["Authorization"].startswith("Basic ")
            assert body is not None
            token_requests.append(body)
            token_executions.append(kwargs.get("execution"))
            token_count += 1
            return _json_response(
                {
                    "access_token": f"access-{token_count}",
                    "token_type": "Bearer",
                    "expires_in": 300,
                }
            )
        assert endpoint.url == mcp_url
        if body and json.loads(body).get("method") == "tools/call" and reject_call_once:
            reject_call_once = False
            return _HttpResponse(401, {}, b"")
        mcp.expected_token = f"access-{token_count}"
        return mcp.exchange(session, method, endpoint, headers, body, **kwargs)

    monkeypatch.setattr(McpHttpSession, "_exchange", exchange)
    source = _source(
        auth={
            "type": "oauth_client_credentials",
            "client_id_ref": "oauth/client-id",
            "client_secret_ref": "oauth/client-secret",
            "resource_metadata_url": metadata_url,
            "authorization_server": issuer,
            "token_endpoints": [token_url],
            "scopes": ["files.read"],
        }
    )
    bound = open_mcp_http_source(source, McpHttpHost(resolve_secret=resolve))
    registry = ProviderRegistry((bound,))
    broker = CapabilityBroker(registry.capability_registrations())
    try:
        result = registry.broker_globals(broker)["remote_docs"].read_file(path="guide.md")
        assert result == {"text": "value:guide.md"}
        assert token_count == 2
        assert token_executions[0] is None
        assert token_executions[1] is not None
        assert bound.runtime.stats()["auth_refreshes"] == 2
        assert all(
            b"resource=https%3A%2F%2Fmcp.example.test%2Fmcp" in item for item in token_requests
        )
        assert {item.reference for item in secret_requests} == {
            "oauth/client-id",
            "oauth/client-secret",
        }
        encoded_source = json.dumps(source.config_dict())
        assert "access-" not in encoded_source
        assert '"client_secret":' not in encoded_source
    finally:
        registry.close()


def test_oauth_discovery_shares_one_total_startup_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_url = "https://mcp.example.test/mcp"
    metadata_url = "https://mcp.example.test/.well-known/oauth-protected-resource/mcp"
    issuer = "https://auth.example.test/"
    token_url = "https://auth.example.test/token"

    def exchange(
        session: McpHttpSession,
        method: str,
        endpoint: _Endpoint,
        headers: Mapping[str, str],
        body: bytes | None,
        **kwargs: Any,
    ) -> _HttpResponse:
        del session, method, headers, body, kwargs
        if endpoint.url == metadata_url:
            time.sleep(0.12)
            return _json_response({"resource": mcp_url, "authorization_servers": [issuer]})
        raise AssertionError("startup continued after its total deadline")

    monkeypatch.setattr(McpHttpSession, "_exchange", exchange)
    began = time.monotonic()
    with pytest.raises(McpTransportError, match="authentication timed out"):
        open_mcp_http_source(
            _source(
                startup_timeout_ms=100,
                auth={
                    "type": "oauth_client_credentials",
                    "client_id_ref": "oauth/client-id",
                    "client_secret_ref": "oauth/client-secret",
                    "resource_metadata_url": metadata_url,
                    "authorization_server": issuer,
                    "token_endpoints": [token_url],
                },
            ),
            McpHttpHost(resolve_secret=lambda request: "unused"),
        )
    assert time.monotonic() - began < 0.3


def test_close_prevents_in_flight_oauth_token_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_url = "https://mcp.example.test/mcp"
    metadata_url = "https://mcp.example.test/.well-known/oauth-protected-resource/mcp"
    issuer = "https://auth.example.test/"
    discovery_url = "https://auth.example.test/.well-known/oauth-authorization-server"
    token_url = "https://auth.example.test/token"
    mcp = _Server()
    token_count = 0
    block_refresh = False
    refresh_started = threading.Event()
    release_refresh = threading.Event()

    def exchange(
        session: McpHttpSession,
        method: str,
        endpoint: _Endpoint,
        headers: Mapping[str, str],
        body: bytes | None,
        **kwargs: Any,
    ) -> _HttpResponse:
        nonlocal token_count
        if endpoint.url == metadata_url:
            return _json_response({"resource": mcp_url, "authorization_servers": [issuer]})
        if endpoint.url == discovery_url:
            return _json_response(
                {
                    "issuer": issuer,
                    "token_endpoint": token_url,
                    "grant_types_supported": ["client_credentials"],
                }
            )
        if endpoint.url == token_url:
            if block_refresh:
                refresh_started.set()
                release_refresh.wait(1)
            token_count += 1
            return _json_response({"access_token": f"access-{token_count}", "token_type": "Bearer"})
        mcp.expected_token = f"access-{token_count}"
        return mcp.exchange(session, method, endpoint, headers, body, **kwargs)

    monkeypatch.setattr(McpHttpSession, "_exchange", exchange)
    bound = open_mcp_http_source(
        _source(
            auth={
                "type": "oauth_client_credentials",
                "client_id_ref": "oauth/client-id",
                "client_secret_ref": "oauth/client-secret",
                "resource_metadata_url": metadata_url,
                "authorization_server": issuer,
                "token_endpoints": [token_url],
            }
        ),
        McpHttpHost(resolve_secret=lambda request: "credential"),
    )
    session = bound.runtime.close_callback.__self__
    block_refresh = True
    outcomes: list[BaseException] = []

    def refresh() -> None:
        try:
            session._access_token(force_refresh=True)
        except BaseException as exc:
            outcomes.append(exc)

    worker = threading.Thread(target=refresh)
    worker.start()
    assert refresh_started.wait(1)
    bound.close()
    release_refresh.set()
    worker.join(2)
    assert not worker.is_alive()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], McpTransportError)
    assert "closed" in str(outcomes[0])
    assert session._tokens is None
    assert session._last_access_token is None


def test_safe_retries_share_one_total_request_deadline(
    server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound = open_mcp_http_source(
        _source(request_timeout_ms=100, reconnect_attempts=3, backoff_ms=10),
        McpHttpHost(),
    )
    session = bound.runtime.close_callback.__self__

    def unavailable(*args: Any, **kwargs: Any) -> _HttpResponse:
        del args, kwargs
        time.sleep(0.06)
        return _HttpResponse(503, {}, b"")

    monkeypatch.setattr(McpHttpSession, "_exchange", unavailable)
    began = time.monotonic()
    with pytest.raises(McpTransportError, match="timed out"):
        session._post_message(
            {"jsonrpc": "2.0", "method": "notifications/test"},
            execution=None,
            timeout_ms=100,
            session_id="session-1",
            safe_retry=True,
        )
    assert time.monotonic() - began < 0.2


def test_cancellation_and_attempt_finalization_happen_once(
    server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = LifecycleGate()

    def slow_exchange(
        session: McpHttpSession,
        method: str,
        endpoint: _Endpoint,
        headers: Mapping[str, str],
        body: bytes | None,
        *,
        execution=None,
        **kwargs: Any,
    ) -> _HttpResponse:
        if body and json.loads(body).get("method") == "tools/call":
            gate.pause()
            assert execution is not None
            execution.check()
            raise AssertionError("cancelled execution unexpectedly continued")
        return server.exchange(session, method, endpoint, headers, body, **kwargs)

    monkeypatch.setattr(McpHttpSession, "_exchange", slow_exchange)
    bound = open_mcp_http_source(_source(), McpHttpHost())
    registry = ProviderRegistry((bound,))
    authority = RecordingAttemptAuthority()

    broker = CapabilityBroker(
        registry.capability_registrations(), run_id="run", attempt_authority=authority
    )
    call = CapabilityCall(
        CapabilityId(
            CapabilityKind.DATA,
            "ReadFile",
            source_id="remote_docs",
            provider_type="reference_filesystem",
        ),
        "call-1",
        "run",
        kwargs={"path": "slow"},
    )

    def cancel() -> None:
        assert broker.cancel("call-1") is True

    try:
        outcome = run_while_blocked(
            lambda: broker.dispatch(call),
            gate=gate,
            while_blocked=cancel,
            on_timeout=registry.close,
        ).require_value()
        assert outcome.error.code == "cancelled"
        settlement = authority.require_single_settlement("call-1")
        assert settlement.attempted is True
        assert settlement.error_code == "cancelled"
        assert authority.active_calls == frozenset()
        assert bound.runtime.stats()["calls"] == 0
        assert bound.runtime.stats()["failures"] == 1
        assert bound.runtime.stats()["requests_made"] == 1
    finally:
        registry.close()
