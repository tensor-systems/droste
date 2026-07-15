"""Production MCP Streamable HTTP transport owned by the trusted host.

The transport deliberately has no dependency on provider descriptors.  It owns
only remote protocol state, OAuth credentials, network policy, and bounded I/O.
"""

from __future__ import annotations

import base64
import http.client
import ipaddress
import json
import socket
import ssl
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import monotonic, sleep
from types import MappingProxyType
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

from ..capabilities import CapabilityExecutionContext
from ._mcp_stdio_transport import (
    MCP_PROTOCOL_VERSION,
    McpProtocolError,
    McpRemoteError,
    McpTransportError,
    _client_version,
    _strict_json_loads,
)

_TRANSIENT_STATUSES = frozenset({429, 502, 503, 504})
_MAX_SESSION_ID_BYTES = 1024
_MAX_SECRET_BYTES = 65_536


@dataclass(frozen=True, slots=True)
class McpSecretRequest:
    """Tenant/source-scoped request made only to the trusted host resolver."""

    tenant_id: str
    source_id: str
    reference: str


SecretResolver = Callable[[McpSecretRequest], str]
DebugSink = Callable[[str, bytes], None]
DnsResolver = Callable[[str, int], Sequence[str]]


def _system_resolver(host: str, port: int) -> tuple[str, ...]:
    values = {
        item[4][0]
        for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        if item[0] in {socket.AF_INET, socket.AF_INET6}
    }
    return tuple(sorted(values))


@dataclass(frozen=True, slots=True)
class McpHttpHost:
    """Live trusted-host dependencies; never serialized into a source spec.

    Private networks remain denied unless the host explicitly names CIDRs.  A
    custom resolver is useful for split-horizon/VPC DNS, but resolved addresses
    still pass the same IP policy and are pinned through TLS connection setup.
    """

    resolve_secret: SecretResolver | None = None
    allowed_networks: tuple[str, ...] = ()
    ssl_context: ssl.SSLContext | None = None
    resolver: DnsResolver = _system_resolver
    debug_sink: DebugSink | None = None

    def __post_init__(self) -> None:
        if self.resolve_secret is not None and not callable(self.resolve_secret):
            raise TypeError("MCP secret resolver must be callable")
        if not callable(self.resolver):
            raise TypeError("MCP DNS resolver must be callable")
        if self.debug_sink is not None and not callable(self.debug_sink):
            raise TypeError("MCP debug sink must be callable")
        networks: list[str] = []
        for raw in self.allowed_networks:
            if not isinstance(raw, str):
                raise TypeError("MCP allowed networks must be CIDR strings")
            networks.append(str(ipaddress.ip_network(raw, strict=True)))
        object.__setattr__(self, "allowed_networks", tuple(networks))


@dataclass(frozen=True, slots=True)
class _Endpoint:
    url: str
    host: str
    port: int
    path: str


def canonical_https_url(value: Any, *, field: str) -> _Endpoint:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"MCP {field} must be a non-empty HTTPS URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"MCP {field} must be an HTTPS URL without credentials, query, or fragment"
        )
    try:
        host = parsed.hostname.encode("idna").decode("ascii").lower()
        port = parsed.port or 443
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"MCP {field} has an invalid host or port") from exc
    if host.endswith("."):
        raise ValueError(f"MCP {field} host must not use a trailing dot")
    path = parsed.path or "/"
    authority = f"[{host}]" if ":" in host else host
    if parsed.port is not None:
        authority = f"{authority}:{port}"
    canonical = f"https://{authority}{parsed.path}"
    if value != canonical:
        raise ValueError(f"MCP {field} must use its exact canonical HTTPS form")
    return _Endpoint(canonical, host, port, path)


@dataclass(frozen=True, slots=True)
class McpOAuthConfig:
    client_id_ref: str
    client_secret_ref: str
    resource_metadata_url: str
    authorization_server: str
    token_endpoints: tuple[str, ...]
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class McpHttpTransportConfig:
    endpoint: str
    allowed_endpoints: tuple[str, ...]
    tenant_id: str
    source_id: str
    auth_type: str
    token_ref: str | None
    oauth: McpOAuthConfig | None
    startup_timeout_ms: int
    request_timeout_ms: int
    close_timeout_ms: int
    max_frame_bytes: int
    max_tool_pages: int
    max_tools: int
    max_in_flight: int
    reconnect_attempts: int
    backoff_ms: int
    max_debug_payload_bytes: int


@dataclass(slots=True)
class _HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class _SseDisconnected(Exception):
    def __init__(self, last_event_id: str, retry_ms: int | None) -> None:
        self.last_event_id = last_event_id
        self.retry_ms = retry_ms
        super().__init__("MCP SSE stream disconnected before its response")


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        endpoint: _Endpoint,
        address: str,
        *,
        context: ssl.SSLContext,
        timeout: float,
    ) -> None:
        super().__init__(endpoint.host, endpoint.port, context=context, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        raw = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        peer = ipaddress.ip_address(self.sock.getpeername()[0])
        if peer != ipaddress.ip_address(self._address):
            self.sock.close()
            raise McpTransportError("MCP TLS peer did not match the pinned DNS address")


class _OAuthTokens:
    __slots__ = ("access", "refresh", "expires_at")

    def __init__(self, access: str, refresh: str | None, expires_at: float) -> None:
        self.access = access
        self.refresh = refresh
        self.expires_at = expires_at


class McpHttpSession:
    """One tenant/source-isolated Streamable HTTP session."""

    def __init__(self, config: McpHttpTransportConfig, host: McpHttpHost) -> None:
        self._config = config
        self._host = host
        self._endpoint = canonical_https_url(config.endpoint, field="endpoint")
        allowed = {
            canonical_https_url(item, field="allowed_endpoints entry").url
            for item in config.allowed_endpoints
        }
        if self._endpoint.url not in allowed:
            raise McpTransportError("MCP endpoint is not in the exact host allowlist")
        self._allowed_urls = frozenset(allowed)
        self._allowed_networks = tuple(
            ipaddress.ip_network(item, strict=True) for item in host.allowed_networks
        )
        self._ssl_context = host.ssl_context or ssl.create_default_context()
        self._state_lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._auth_lock = threading.Lock()
        self._closed = False
        self._session_id: str | None = None
        self._generation = 0
        self._next_id = 1
        self._in_flight = 0
        self._calls = 0
        self._failures = 0
        self._latency_ms = 0.0
        self._bytes_received = 0
        self._reconnects = 0
        self._auth_refreshes = 0
        self._tokens: _OAuthTokens | None = None
        self._token_endpoint: str | None = None
        self._frozen_tools: bytes | None = None
        self._startup_expires = monotonic() + config.startup_timeout_ms / 1000
        try:
            self._initialize(self._startup_remaining_ms())
        except BaseException:
            self.close()
            raise

    def _initialize(self, timeout_ms: int) -> None:
        result, headers, _ = self._rpc_once(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "droste", "version": _client_version()},
            },
            timeout_ms=timeout_ms,
            session_id=None,
            safe_retry=True,
        )
        if not isinstance(result, dict):
            raise McpProtocolError("MCP initialize result must be an object")
        if result.get("protocolVersion") != MCP_PROTOCOL_VERSION:
            raise McpProtocolError(
                f"MCP server did not negotiate the supported protocol version {MCP_PROTOCOL_VERSION}"
            )
        capabilities = result.get("capabilities")
        if not isinstance(capabilities, dict) or not isinstance(capabilities.get("tools"), dict):
            raise McpProtocolError("MCP server must advertise the tools capability")
        raw_session = headers.get("mcp-session-id")
        if raw_session is not None:
            try:
                encoded = raw_session.encode("ascii", errors="strict")
            except UnicodeEncodeError as exc:
                raise McpProtocolError("MCP session id must be bounded visible ASCII") from exc
            if (
                not encoded
                or len(encoded) > _MAX_SESSION_ID_BYTES
                or any(item < 0x21 or item > 0x7E for item in encoded)
            ):
                raise McpProtocolError("MCP session id must be bounded visible ASCII")
        with self._state_lock:
            self._session_id = raw_session
            self._generation += 1
        self._notify("notifications/initialized", {}, timeout_ms=timeout_ms)

    def list_tools(self) -> tuple[dict[str, Any], ...]:
        tools = self._list_tools(
            timeout_ms=self._startup_remaining_ms(), allow_session_restart=True
        )
        frozen = self._canonical_bytes(tools)
        with self._state_lock:
            if self._frozen_tools is None:
                self._frozen_tools = frozen
            elif self._frozen_tools != frozen:
                raise McpProtocolError("MCP tools/list changed across session reconnection")
        return tools

    def _list_tools(
        self, *, timeout_ms: int, allow_session_restart: bool
    ) -> tuple[dict[str, Any], ...]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        seen: set[str] = set()
        for _ in range(self._config.max_tool_pages):
            params = {} if cursor is None else {"cursor": cursor}
            result = self._rpc(
                "tools/list",
                params,
                timeout_ms=timeout_ms,
                safe_retry=True,
                allow_session_restart=allow_session_restart,
            )
            if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
                raise McpProtocolError("MCP tools/list result requires a tools array")
            for item in result["tools"]:
                if not isinstance(item, dict):
                    raise McpProtocolError("MCP tools/list entries must be objects")
                tools.append(item)
                if len(tools) > self._config.max_tools:
                    raise McpProtocolError("MCP tools/list exceeded the configured tool bound")
            raw_cursor = result.get("nextCursor")
            if raw_cursor is None:
                return tuple(tools)
            if not isinstance(raw_cursor, str):
                raise McpProtocolError("MCP tools/list nextCursor must be a string")
            if raw_cursor in seen:
                raise McpProtocolError("MCP tools/list repeated a pagination cursor")
            seen.add(raw_cursor)
            cursor = raw_cursor
        raise McpProtocolError("MCP tools/list exceeded the configured page bound")

    def call_tool(
        self,
        execution: CapabilityExecutionContext,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], float, int]:
        started = monotonic()
        try:
            result, response_bytes = self._rpc_with_size(
                "tools/call", {"name": name, "arguments": arguments}, execution=execution
            )
        except BaseException:
            with self._state_lock:
                self._failures += 1
            raise
        elapsed = (monotonic() - started) * 1000
        with self._state_lock:
            self._calls += 1
            self._latency_ms += elapsed
        if not isinstance(result, dict):
            raise McpProtocolError("MCP tools/call result must be an object")
        return result, elapsed, response_bytes

    def _rpc(
        self,
        method: str,
        params: dict[str, Any],
        *,
        execution: CapabilityExecutionContext | None = None,
        timeout_ms: int | None = None,
        safe_retry: bool = False,
        allow_session_restart: bool = True,
    ) -> Any:
        result, _ = self._rpc_with_size(
            method,
            params,
            execution=execution,
            timeout_ms=timeout_ms,
            safe_retry=safe_retry,
            allow_session_restart=allow_session_restart,
        )
        return result

    def _rpc_with_size(
        self,
        method: str,
        params: dict[str, Any],
        *,
        execution: CapabilityExecutionContext | None = None,
        timeout_ms: int | None = None,
        safe_retry: bool = False,
        allow_session_restart: bool = True,
    ) -> tuple[Any, int]:
        with self._state_lock:
            self._raise_open_locked()
            if self._in_flight >= self._config.max_in_flight:
                raise McpTransportError("MCP session reached its in-flight request bound")
            self._in_flight += 1
            generation = self._generation
            session_id = self._session_id
        try:
            try:
                result, _, size = self._rpc_once(
                    method,
                    params,
                    execution=execution,
                    timeout_ms=timeout_ms or self._config.request_timeout_ms,
                    session_id=session_id,
                    safe_retry=safe_retry,
                )
                return result, size
            except _SessionExpired:
                if not allow_session_restart:
                    raise McpTransportError(
                        "MCP session expired again during bounded reconnection"
                    ) from None
                self._restart_session(generation, execution)
                with self._state_lock:
                    session_id = self._session_id
                result, _, size = self._rpc_once(
                    method,
                    params,
                    execution=execution,
                    timeout_ms=timeout_ms or self._config.request_timeout_ms,
                    session_id=session_id,
                    safe_retry=False,
                )
                return result, size
        finally:
            with self._state_lock:
                self._in_flight -= 1

    def _restart_session(
        self, generation: int, execution: CapabilityExecutionContext | None
    ) -> None:
        with self._session_lock:
            if execution is not None:
                execution.check()
            with self._state_lock:
                if self._generation != generation:
                    return
                self._session_id = None
            self._initialize(self._config.request_timeout_ms)
            tools = self._list_tools(
                timeout_ms=self._config.request_timeout_ms,
                allow_session_restart=False,
            )
            frozen = self._canonical_bytes(tools)
            with self._state_lock:
                if self._frozen_tools is not None and frozen != self._frozen_tools:
                    raise McpProtocolError("MCP tools/list changed across session reconnection")
                self._reconnects += 1

    def _rpc_once(
        self,
        method: str,
        params: dict[str, Any],
        *,
        execution: CapabilityExecutionContext | None = None,
        timeout_ms: int,
        session_id: str | None,
        safe_retry: bool,
    ) -> tuple[Any, Mapping[str, str], int]:
        with self._state_lock:
            request_id = self._next_id
            self._next_id += 1
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        response = self._post_message(
            message,
            execution=execution,
            timeout_ms=timeout_ms,
            session_id=session_id,
            safe_retry=safe_retry,
        )
        result = self._decode_rpc_response(response, request_id)
        return result, response.headers, len(response.body)

    def _notify(self, method: str, params: dict[str, Any], *, timeout_ms: int) -> None:
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        with self._state_lock:
            session_id = self._session_id
        response = self._post_message(
            message,
            execution=None,
            timeout_ms=timeout_ms,
            session_id=session_id,
            safe_retry=True,
        )
        if response.status != 202 or response.body:
            raise McpProtocolError("MCP notification requires an empty HTTP 202 response")

    def _post_message(
        self,
        message: dict[str, Any],
        *,
        execution: CapabilityExecutionContext | None,
        timeout_ms: int,
        session_id: str | None,
        safe_retry: bool,
    ) -> _HttpResponse:
        body = self._canonical_bytes(message)
        if len(body) > self._config.max_frame_bytes:
            raise McpProtocolError("MCP request exceeds the configured frame bound")
        self._debug("request", body)
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
        }
        if session_id is not None:
            headers["MCP-Session-Id"] = session_id
        max_transient_attempts = self._config.reconnect_attempts if safe_retry else 0
        transient_attempt = 0
        refreshed = False
        refresh_next = False
        while True:
            token = self._access_token(force_refresh=refresh_next, execution=execution)
            refresh_next = False
            request_headers = dict(headers)
            if token is not None:
                request_headers["Authorization"] = f"Bearer {token}"
            try:
                response = self._exchange(
                    "POST",
                    self._endpoint,
                    request_headers,
                    body,
                    execution=execution,
                    timeout_ms=timeout_ms,
                )
            except _SseDisconnected as interrupted:
                response = self._resume_sse(
                    interrupted,
                    execution=execution,
                    timeout_ms=timeout_ms,
                    session_id=session_id,
                )
            if 300 <= response.status < 400:
                raise McpTransportError("MCP HTTP redirects are forbidden")
            if response.status == 401 and self._config.auth_type == "oauth" and not refreshed:
                refreshed = True
                refresh_next = True
                self._clear_tokens(execution)
                continue
            if response.status == 404 and session_id is not None:
                raise _SessionExpired
            if (
                response.status in _TRANSIENT_STATUSES
                and transient_attempt < max_transient_attempts
            ):
                self._backoff(transient_attempt, execution)
                transient_attempt += 1
                continue
            if response.status not in {200, 202}:
                raise McpTransportError(f"MCP HTTP request failed with status {response.status}")
            with self._state_lock:
                self._bytes_received += len(response.body)
            self._debug("response", response.body)
            return response

    def _resume_sse(
        self,
        interrupted: _SseDisconnected,
        *,
        execution: CapabilityExecutionContext | None,
        timeout_ms: int,
        session_id: str | None,
    ) -> _HttpResponse:
        expires = monotonic() + timeout_ms / 1000
        cursor = interrupted
        auth_refreshed = False
        refresh_next = False
        for attempt in range(self._config.reconnect_attempts + 1):
            wait_ms = (
                cursor.retry_ms
                if cursor.retry_ms is not None
                else (self._config.backoff_ms * (2**attempt))
            )
            wait_expires = min(expires, monotonic() + min(wait_ms, 60_000) / 1000)
            while monotonic() < wait_expires:
                if execution is not None:
                    execution.check()
                sleep(min(0.01, max(0, wait_expires - monotonic())))
            remaining_ms = max(1, int((expires - monotonic()) * 1000))
            if monotonic() >= expires:
                raise McpTransportError("MCP SSE reconnection timed out")
            headers = {
                "Accept": "text/event-stream",
                "Last-Event-ID": cursor.last_event_id,
                "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            }
            if session_id is not None:
                headers["MCP-Session-Id"] = session_id
            token = self._access_token(force_refresh=refresh_next, execution=execution)
            refresh_next = False
            if token is not None:
                headers["Authorization"] = f"Bearer {token}"
            try:
                response = self._exchange(
                    "GET",
                    self._endpoint,
                    headers,
                    None,
                    execution=execution,
                    timeout_ms=remaining_ms,
                )
            except _SseDisconnected as next_cursor:
                cursor = next_cursor
                continue
            if response.status == 401 and self._config.auth_type == "oauth" and not auth_refreshed:
                self._clear_tokens(execution)
                auth_refreshed = True
                refresh_next = True
                continue
            if response.status == 404 and session_id is not None:
                # The original POST was already accepted and produced a
                # resumable stream. Replaying it after a new session could
                # duplicate an effectful tool invocation.
                raise McpTransportError("MCP session expired while resuming an accepted request")
            if 300 <= response.status < 400:
                raise McpTransportError("MCP HTTP redirects are forbidden")
            if response.status != 200:
                raise McpTransportError(
                    f"MCP SSE reconnection failed with status {response.status}"
                )
            with self._state_lock:
                self._reconnects += 1
            return response
        raise McpTransportError("MCP SSE reconnection bound exhausted")

    def _exchange(
        self,
        method: str,
        endpoint: _Endpoint,
        headers: Mapping[str, str],
        body: bytes | None,
        *,
        execution: CapabilityExecutionContext | None,
        timeout_ms: int,
    ) -> _HttpResponse:
        result: list[_HttpResponse] = []
        failure: list[BaseException] = []
        complete = threading.Event()
        connection: list[_PinnedHTTPSConnection] = []

        def run() -> None:
            try:
                addresses = self._resolve(endpoint)
                # One pinned address per exchange.  Trying another after an
                # ambiguous POST failure could duplicate an effectful tool.
                conn = _PinnedHTTPSConnection(
                    endpoint,
                    addresses[0],
                    context=self._ssl_context,
                    timeout=max(0.1, timeout_ms / 1000),
                )
                connection[:] = [conn]
                conn.request(method, endpoint.path, body=body, headers=dict(headers))
                response = conn.getresponse()
                response_headers: dict[str, str] = {}
                security_headers = {
                    "content-type",
                    "location",
                    "mcp-session-id",
                    "www-authenticate",
                }
                for key, value in response.getheaders():
                    lowered = key.lower()
                    if (
                        lowered in security_headers
                        and lowered in response_headers
                        and response_headers[lowered] != value
                    ):
                        raise McpProtocolError("MCP HTTP response repeated a security header")
                    response_headers.setdefault(lowered, value)
                media_type = (
                    response_headers.get("content-type", "").split(";", 1)[0].strip().lower()
                )
                payload = (
                    self._read_sse_response(response)
                    if media_type == "text/event-stream"
                    else response.read(self._config.max_frame_bytes + 1)
                )
                if len(payload) > self._config.max_frame_bytes:
                    raise McpProtocolError("MCP HTTP response exceeds the frame bound")
                result.append(_HttpResponse(response.status, response_headers, payload))
                conn.close()
            except BaseException as exc:
                if connection:
                    connection[0].close()
                failure.append(exc)
            finally:
                complete.set()

        worker = threading.Thread(target=run, name="droste-mcp-http", daemon=True)
        worker.start()
        expires = monotonic() + timeout_ms / 1000
        try:
            while not complete.wait(0.01):
                if execution is not None:
                    execution.check()
                if monotonic() >= expires:
                    raise McpTransportError("MCP HTTP request timed out")
        except BaseException:
            if connection:
                connection[0].close()
            complete.wait(self._config.close_timeout_ms / 1000)
            raise
        if failure:
            error = failure[0]
            if isinstance(error, (McpProtocolError, McpTransportError, _SseDisconnected)):
                raise error
            raise McpTransportError(f"MCP HTTPS exchange failed: {type(error).__name__}") from error
        return result[0]

    def _read_sse_response(self, response: http.client.HTTPResponse) -> bytes:
        chunks: list[bytes] = []
        total = 0
        data: list[str] = []
        last_event_id: str | None = None
        retry_ms: int | None = None
        while True:
            line = response.readline(self._config.max_frame_bytes + 1)
            if not line:
                if last_event_id is not None:
                    raise _SseDisconnected(last_event_id, retry_ms)
                return b"".join(chunks)
            total += len(line)
            if total > self._config.max_frame_bytes:
                raise McpProtocolError("MCP SSE response exceeds the frame bound")
            chunks.append(line)
            try:
                text = line.rstrip(b"\r\n").decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise McpProtocolError("MCP SSE response was not UTF-8") from exc
            if not text:
                encoded_data = "\n".join(data)
                data.clear()
                if encoded_data:
                    message = self._decode_json(encoded_data.encode("utf-8"))
                    if isinstance(message, dict) and "id" in message and "method" not in message:
                        return b"".join(chunks)
                continue
            if text.startswith(":"):
                continue
            field, _, value = text.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data.append(value)
            elif field == "id":
                if (
                    not value
                    or len(value.encode("utf-8")) > _MAX_SESSION_ID_BYTES
                    or "\x00" in value
                    or "\r" in value
                    or "\n" in value
                ):
                    raise McpProtocolError("MCP SSE event id is invalid or excessive")
                last_event_id = value
            elif field == "retry":
                if not value.isdigit() or int(value) > 60_000:
                    raise McpProtocolError("MCP SSE retry value is invalid or excessive")
                retry_ms = int(value)

    def _resolve(self, endpoint: _Endpoint) -> tuple[str, ...]:
        try:
            raw = tuple(self._host.resolver(endpoint.host, endpoint.port))
        except Exception as exc:
            raise McpTransportError("MCP endpoint DNS resolution failed") from exc
        if not raw:
            raise McpTransportError("MCP endpoint DNS resolution returned no addresses")
        addresses: list[str] = []
        for item in raw:
            try:
                address = ipaddress.ip_address(item)
            except ValueError as exc:
                raise McpTransportError("MCP DNS resolver returned an invalid IP address") from exc
            if not address.is_global and not any(
                address in network for network in self._allowed_networks
            ):
                raise McpTransportError("MCP endpoint resolved to a forbidden network address")
            addresses.append(str(address))
        return tuple(sorted(set(addresses)))

    def _decode_rpc_response(self, response: _HttpResponse, request_id: int) -> Any:
        media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type == "application/json":
            messages = (self._decode_json(response.body),)
        elif media_type == "text/event-stream":
            messages = self._decode_sse(response.body)
        else:
            raise McpProtocolError("MCP request response has an unsupported content type")
        matched: dict[str, Any] | None = None
        for message in messages:
            if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                raise McpProtocolError("MCP HTTP message must be a JSON-RPC 2.0 object")
            if "method" in message:
                if "id" in message:
                    raise McpProtocolError(
                        "MCP server requests are unsupported by this provider transport"
                    )
                # Notifications are advisory.  Descriptor changes do not
                # mutate the frozen per-run manifest and progress is not
                # provider budget usage.
                continue
            if message.get("id") != request_id:
                raise McpProtocolError("MCP response id does not match the active request")
            if matched is not None:
                raise McpProtocolError("MCP request received more than one response")
            matched = message
        if matched is None:
            raise McpTransportError("MCP SSE stream ended before its response")
        has_result = "result" in matched
        has_error = "error" in matched
        if has_result == has_error:
            raise McpProtocolError("MCP response requires exactly one of result or error")
        if has_error:
            error = matched["error"]
            if not isinstance(error, dict):
                raise McpProtocolError("MCP JSON-RPC error must be an object")
            code = error.get("code")
            message = error.get("message")
            if isinstance(code, bool) or not isinstance(code, int) or not isinstance(message, str):
                raise McpProtocolError("MCP JSON-RPC error requires integer code and message")
            encoded = message.encode("utf-8")
            bounded = message if len(encoded) <= 1024 else encoded[:1024].decode("utf-8", "ignore")
            raise McpRemoteError(code, bounded)
        return matched["result"]

    @staticmethod
    def _decode_json(body: bytes) -> Any:
        try:
            return _strict_json_loads(body.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise McpProtocolError("MCP HTTP response contained invalid JSON") from exc

    def _decode_sse(self, body: bytes) -> tuple[Any, ...]:
        try:
            text = body.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise McpProtocolError("MCP SSE response was not UTF-8") from exc
        messages: list[Any] = []
        data: list[str] = []
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            if not line:
                if data:
                    messages.append(self._decode_json("\n".join(data).encode("utf-8")))
                    data.clear()
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data.append(value)
            elif field == "retry":
                if not value.isdigit() or int(value) > 60_000:
                    raise McpProtocolError("MCP SSE retry value is invalid or excessive")
        if data:
            messages.append(self._decode_json("\n".join(data).encode("utf-8")))
        return tuple(messages)

    def _access_token(
        self,
        *,
        force_refresh: bool = False,
        execution: CapabilityExecutionContext | None = None,
    ) -> str | None:
        if self._config.auth_type == "none":
            return None
        if self._config.auth_type == "bearer":
            assert self._config.token_ref is not None
            return self._secret(self._config.token_ref, execution)
        self._acquire_auth_lock(execution)
        try:
            now = monotonic()
            if (
                not force_refresh
                and self._tokens is not None
                and now < self._tokens.expires_at - 30
            ):
                return self._tokens.access
            oauth = self._config.oauth
            assert oauth is not None
            if self._token_endpoint is None:
                self._token_endpoint = self._discover_token_endpoint(oauth, execution)
            token = self._request_oauth_token(oauth, self._token_endpoint, self._tokens, execution)
            self._tokens = token
            self._auth_refreshes += 1
            return token.access
        finally:
            self._auth_lock.release()

    def _discover_token_endpoint(
        self,
        oauth: McpOAuthConfig,
        execution: CapabilityExecutionContext | None,
    ) -> str:
        metadata_endpoint = canonical_https_url(
            oauth.resource_metadata_url, field="OAuth resource metadata URL"
        )
        response = self._exchange(
            "GET",
            metadata_endpoint,
            {"Accept": "application/json"},
            None,
            execution=execution,
            timeout_ms=self._config.startup_timeout_ms,
        )
        if response.status != 200:
            raise McpTransportError("MCP OAuth protected-resource discovery failed")
        resource = self._json_mapping(response.body, "protected-resource metadata")
        if resource.get("resource") != self._endpoint.url:
            raise McpProtocolError("MCP OAuth metadata resource does not match the exact endpoint")
        servers = resource.get("authorization_servers")
        if not isinstance(servers, list) or oauth.authorization_server not in servers:
            raise McpProtocolError(
                "MCP OAuth metadata omitted the allowlisted authorization server"
            )
        issuer = canonical_https_url(oauth.authorization_server, field="OAuth authorization server")
        base_path = issuer.path.rstrip("/")
        candidates = (
            f"https://{issuer.host}{'' if issuer.port == 443 else f':{issuer.port}'}"
            f"/.well-known/oauth-authorization-server{base_path if base_path != '' else ''}",
            f"https://{issuer.host}{'' if issuer.port == 443 else f':{issuer.port}'}"
            f"/.well-known/openid-configuration{base_path if base_path != '' else ''}",
            f"{issuer.url.rstrip('/')}/.well-known/openid-configuration",
        )
        allowed_tokens = {
            canonical_https_url(item, field="OAuth token endpoint allowlist entry").url
            for item in oauth.token_endpoints
        }
        for candidate in dict.fromkeys(candidates):
            endpoint = canonical_https_url(candidate, field="OAuth discovery URL")
            discovered = self._exchange(
                "GET",
                endpoint,
                {"Accept": "application/json"},
                None,
                execution=execution,
                timeout_ms=self._config.startup_timeout_ms,
            )
            if discovered.status == 404:
                continue
            if discovered.status != 200:
                raise McpTransportError("MCP OAuth authorization-server discovery failed")
            metadata = self._json_mapping(discovered.body, "authorization-server metadata")
            if metadata.get("issuer") != issuer.url:
                raise McpProtocolError("MCP OAuth issuer metadata did not match the allowlist")
            token_endpoint = metadata.get("token_endpoint")
            if not isinstance(token_endpoint, str) or token_endpoint not in allowed_tokens:
                raise McpProtocolError("MCP OAuth token endpoint is not exactly allowlisted")
            grants = metadata.get("grant_types_supported", ["authorization_code"])
            if not isinstance(grants, list) or "client_credentials" not in grants:
                raise McpProtocolError("MCP OAuth server does not support client_credentials")
            return token_endpoint
        raise McpProtocolError("MCP OAuth authorization-server metadata was not found")

    def _request_oauth_token(
        self,
        oauth: McpOAuthConfig,
        token_url: str,
        prior: _OAuthTokens | None,
        execution: CapabilityExecutionContext | None,
    ) -> _OAuthTokens:
        client_id = self._secret(oauth.client_id_ref, execution)
        client_secret = self._secret(oauth.client_secret_ref, execution)
        if prior is not None and prior.refresh:
            values = {
                "grant_type": "refresh_token",
                "refresh_token": prior.refresh,
                "resource": self._endpoint.url,
            }
        else:
            values = {"grant_type": "client_credentials", "resource": self._endpoint.url}
            if oauth.scopes:
                values["scope"] = " ".join(oauth.scopes)
        basic = base64.b64encode(
            f"{quote(client_id, safe='')}:{quote(client_secret, safe='')}".encode()
        ).decode()
        endpoint = canonical_https_url(token_url, field="OAuth token endpoint")
        response = self._exchange(
            "POST",
            endpoint,
            {
                "Accept": "application/json",
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            urlencode(values).encode("ascii"),
            execution=execution,
            timeout_ms=self._config.request_timeout_ms,
        )
        if response.status != 200:
            raise McpTransportError("MCP OAuth token request failed")
        payload = self._json_mapping(response.body, "OAuth token response")
        access = payload.get("access_token")
        token_type = payload.get("token_type")
        expires_in = payload.get("expires_in", 300)
        refresh = payload.get("refresh_token")
        if (
            not isinstance(access, str)
            or not access
            or len(access.encode()) > _MAX_SECRET_BYTES
            or not isinstance(token_type, str)
            or token_type.lower() != "bearer"
            or isinstance(expires_in, bool)
            or not isinstance(expires_in, (int, float))
            or not 1 <= expires_in <= 86_400
            or (refresh is not None and (not isinstance(refresh, str) or not refresh))
        ):
            raise McpProtocolError("MCP OAuth token response is invalid")
        return _OAuthTokens(access, refresh, monotonic() + float(expires_in))

    def _secret(
        self,
        reference: str,
        execution: CapabilityExecutionContext | None,
    ) -> str:
        resolver = self._host.resolve_secret
        if resolver is None:
            raise McpTransportError("MCP authentication requires a trusted-host secret resolver")
        if execution is not None:
            execution.check()
        values: list[str] = []
        failures: list[BaseException] = []
        complete = threading.Event()

        def resolve() -> None:
            try:
                values.append(
                    resolver(
                        McpSecretRequest(
                            self._config.tenant_id,
                            self._config.source_id,
                            reference,
                        )
                    )
                )
            except BaseException as exc:
                failures.append(exc)
            finally:
                complete.set()

        threading.Thread(target=resolve, name="droste-mcp-secret", daemon=True).start()
        expires = monotonic() + self._config.request_timeout_ms / 1000
        while not complete.wait(0.01):
            if execution is not None:
                execution.check()
            if monotonic() >= expires:
                raise McpTransportError("MCP trusted-host secret resolution timed out")
        if failures:
            raise McpTransportError("MCP trusted-host secret resolution failed") from failures[0]
        value = values[0]
        if (
            not isinstance(value, str)
            or not value
            or len(value.encode()) > _MAX_SECRET_BYTES
            or "\r" in value
            or "\n" in value
        ):
            raise McpTransportError("MCP trusted-host secret resolver returned an invalid value")
        return value

    def _acquire_auth_lock(self, execution: CapabilityExecutionContext | None) -> None:
        while not self._auth_lock.acquire(timeout=0.01):
            if execution is not None:
                execution.check()

    @staticmethod
    def _json_mapping(body: bytes, label: str) -> dict[str, Any]:
        value = McpHttpSession._decode_json(body)
        if not isinstance(value, dict):
            raise McpProtocolError(f"MCP {label} must be a JSON object")
        return value

    def _backoff(self, attempt: int, execution: CapabilityExecutionContext | None) -> None:
        duration = min(5.0, self._config.backoff_ms / 1000 * (2**attempt))
        expires = monotonic() + duration
        while monotonic() < expires:
            if execution is not None:
                execution.check()
            sleep(min(0.01, max(0, expires - monotonic())))

    def _debug(self, direction: str, payload: bytes) -> None:
        limit = self._config.max_debug_payload_bytes
        sink = self._host.debug_sink
        if not limit or sink is None:
            return
        try:
            sink(direction, payload[:limit])
        except Exception:
            pass

    def _startup_remaining_ms(self) -> int:
        remaining = self._startup_expires - monotonic()
        if remaining <= 0:
            raise McpTransportError("MCP acquisition timed out")
        return max(1, int(remaining * 1000))

    @staticmethod
    def _canonical_bytes(value: Any) -> bytes:
        try:
            return json.dumps(
                value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise McpProtocolError("MCP message is not finite JSON") from exc

    def _clear_tokens(self, execution: CapabilityExecutionContext | None = None) -> None:
        self._acquire_auth_lock(execution)
        try:
            self._tokens = None
        finally:
            self._auth_lock.release()

    def _raise_open_locked(self) -> None:
        if self._closed:
            raise McpTransportError("MCP HTTP session is closed")

    def stats(self) -> Mapping[str, int | float | str]:
        with self._state_lock:
            return MappingProxyType(
                {
                    "transport": "streamable_http",
                    "calls": self._calls,
                    "requests_made": self._calls + self._failures,
                    "failures": self._failures,
                    "latency_ms": round(self._latency_ms, 3),
                    "bytes_received": self._bytes_received,
                    "reconnects": self._reconnects,
                    "auth_refreshes": self._auth_refreshes,
                }
            )

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            session_id = self._session_id
            self._session_id = None
        if session_id is not None:
            try:
                headers = {
                    "Accept": "application/json",
                    "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
                    "MCP-Session-Id": session_id,
                }
                token = self._access_token(execution=None)
                if token is not None:
                    headers["Authorization"] = f"Bearer {token}"
                response = self._exchange(
                    "DELETE",
                    self._endpoint,
                    headers,
                    None,
                    execution=None,
                    timeout_ms=self._config.close_timeout_ms,
                )
                if response.status not in {200, 202, 204, 404, 405}:
                    raise McpTransportError("MCP session close failed")
            finally:
                self._tokens = None


class _SessionExpired(Exception):
    pass


__all__ = [
    "McpHttpHost",
    "McpHttpSession",
    "McpHttpTransportConfig",
    "McpOAuthConfig",
    "McpSecretRequest",
    "canonical_https_url",
]
