"""Runner-owned remote source transport and declarative source construction."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from droste.capabilities import (
    JSON_SCHEMA_2020_12,
    PaginationMode,
    ProviderOperation,
    ResultDelivery,
    SchemaSpec,
    SideEffect,
)
from droste.clients.errors import http_error_excerpt
from droste.clients.useragent import USER_AGENT
from droste.providers import (
    ConfiguredSource,
    ProviderCatalog,
    ProviderManifest,
    ProviderRegistration,
    ProviderRegistry,
    ProviderRuntime,
)

_http_error_excerpt = http_error_excerpt


def _require_allowed_host(url: str, allowed: set[str]) -> None:
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    if host not in allowed:
        raise ValueError(f"data_source host {host!r} is not in allowed_hosts")


def _allowlist_opener(allowed: set[str]) -> Any:
    """A urllib opener that re-checks every redirect target against the
    allowlist, closing the SSRF-via-redirect hole (a 30x from an allowed
    host to an internal one)."""

    class _Guard(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
            _require_allowed_host(newurl, allowed)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    return urllib.request.build_opener(_Guard)


class WrapperTransport:
    def __init__(self, config: dict[str, Any] | None) -> None:
        self._config = config or {}
        self._requests_made = 0

    @property
    def requests_made(self) -> int:
        return self._requests_made

    def _limits(self) -> dict[str, Any]:
        limits = self._config.get("limits")
        if isinstance(limits, dict):
            return limits
        return {}

    def _int_limit(self, value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        return None

    def _check_request_budget(self) -> None:
        self._requests_made += 1
        max_requests = self._int_limit(self._limits().get("max_requests"))
        if max_requests is not None and self._requests_made > max_requests:
            raise ValueError("data_source max_requests exceeded")

    def _allowed_hosts(self) -> set[str] | None:
        # Absent key = allow all (the deployer's explicit opt-out). Present but
        # malformed (a bare string, an empty list, all-blank entries) reads as
        # an intended restriction typed wrong — fail closed rather than
        # silently allowing every host.
        if not isinstance(self._config, dict) or "allowed_hosts" not in self._config:
            return None
        raw = self._config.get("allowed_hosts")
        if not isinstance(raw, list):
            raise ValueError("allowed_hosts must be a list of hostnames")
        hosts = {str(h).lower() for h in raw if str(h).strip()}
        if not hosts:
            raise ValueError("allowed_hosts is present but empty — remove it to allow all hosts")
        return hosts

    def _timeout_seconds(self) -> float | None:
        timeout_ms = self._int_limit(self._limits().get("timeout_ms"))
        if timeout_ms is None or timeout_ms < 0:
            return None
        return timeout_ms / 1000.0

    def _read_response(self, resp: Any) -> bytes:
        max_bytes = self._int_limit(self._limits().get("max_response_bytes"))
        if max_bytes is None:
            return resp.read()
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("data_source max_response_bytes exceeded")
        return data

    def _call(self, path: str, payload: dict[str, Any]) -> Any:
        if not isinstance(self._config, dict):
            raise ValueError("data_source is not configured")
        base_url = self._config.get("base_url")
        token = self._config.get("token")
        if not base_url or not token:
            raise ValueError("data_source missing base_url or token")
        # Host allowlist: when the host configures allowed_hosts, the request
        # target must resolve to an allowed hostname — checked here AND on
        # every redirect hop (see _allowlist_opener), so a 30x from an allowed
        # endpoint to an internal host can't slip through. Without this the
        # wrapper is an SSRF primitive in hosted deployments. No allowlist
        # configured = the deployer accepts arbitrary hosts (explicit choice).
        allowed = self._allowed_hosts()
        if allowed is not None:
            _require_allowed_host(str(base_url), allowed)
        self._check_request_budget()
        url = str(base_url).rstrip("/") + "/" + path.lstrip("/")
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": "Bearer " + str(token),
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        timeout = self._timeout_seconds()
        opener = _allowlist_opener(allowed) if allowed is not None else urllib.request
        try:
            if timeout is None:
                resp = opener.urlopen(req)
            else:
                resp = opener.urlopen(req, timeout=timeout)
            with resp:
                raw = self._read_response(resp)
        except urllib.error.HTTPError as exc:
            excerpt = _http_error_excerpt(exc)
            detail = f": {excerpt}" if excerpt else ""
            raise ValueError(f"data_source HTTP error {getattr(exc, 'code', 0)}{detail}") from exc
        except Exception as exc:
            raise ValueError(f"data_source request failed: {exc}") from exc
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError("data_source response must be JSON") from exc

    def search(self, query: str, filters: Any = None, page: Any = None) -> Any:
        payload: dict[str, Any] = {"query": query}
        if filters is not None:
            payload["filters"] = filters
        if page is not None:
            payload["page"] = page
        return self._call("/search", payload)

    def get(self, id: str) -> Any:
        return self._call("/get", {"id": id})

    def content(self, id: str, format: str = "text", max_bytes: int | None = None) -> Any:
        payload: dict[str, Any] = {"id": id}
        if format is not None:
            payload["format"] = format
        if max_bytes is not None:
            payload["max_bytes"] = max_bytes
        return self._call("/content", payload)


def _schema(value: dict[str, Any], name: str) -> SchemaSpec:
    return SchemaSpec(value, JSON_SCHEMA_2020_12, f"droste:provider/wrapper_v1/{name}@1")


WRAPPER_PROVIDER_MANIFEST = ProviderManifest(
    provider_type="wrapper_v1",
    revision="1",
    operations=(
        ProviderOperation(
            "search",
            "search",
            "Search the remote source using provider-defined filters.",
            _schema(
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "filters": {},
                        "page": {},
                    },
                    "required": ["query"],
                },
                "search/parameters",
            ),
            None,
            PaginationMode.NONE,
            ResultDelivery.UNTYPED,
            "data.search",
        ),
        ProviderOperation(
            "get",
            "get",
            "Fetch one remote item by provider-defined ID.",
            _schema(
                {"type": "object", "properties": {"id": {}}, "required": ["id"]},
                "get/parameters",
            ),
            None,
            PaginationMode.NONE,
            ResultDelivery.UNTYPED,
            "data.read",
        ),
        ProviderOperation(
            "content",
            "content",
            "Fetch bounded content for one remote item.",
            _schema(
                {
                    "type": "object",
                    "properties": {
                        "id": {},
                        "format": {"type": "string"},
                        "max_bytes": {"type": ["integer", "null"]},
                    },
                    "required": ["id"],
                },
                "content/parameters",
            ),
            None,
            PaginationMode.NONE,
            ResultDelivery.UNTYPED,
            "data.read",
        ),
        ProviderOperation(
            "stats",
            "get_stats",
            "Return request counts for this configured remote source.",
            _schema({"type": "object", "properties": {}}, "stats/parameters"),
            None,
            PaginationMode.NONE,
            ResultDelivery.UNTYPED,
            "data.metadata",
        ),
    ),
)


def _bind_wrapper(source: ConfiguredSource, context: Any = None) -> ProviderRuntime:
    config = source.config_dict()
    transport = WrapperTransport(config)
    parts = ["Remote wrapper_v1 provider over HTTP."]
    allowed_hosts = config.get("allowed_hosts")
    if isinstance(allowed_hosts, list) and allowed_hosts:
        parts.append("Allowed hosts: " + ", ".join(str(item) for item in allowed_hosts))

    def contextual(handler):
        def invoke(execution, *args, **kwargs):
            execution.check()
            result = handler(*args, **kwargs)
            execution.check()
            return result

        return invoke

    return ProviderRuntime(
        handlers={
            "search": contextual(transport.search),
            "get": contextual(transport.get),
            "content": contextual(transport.content),
            "stats": contextual(lambda: {"requests_made": transport.requests_made}),
        },
        source_description=" ".join(parts),
        stats=lambda: {"requests_made": transport.requests_made},
    )


def wrapper_provider() -> ProviderRegistration:
    return ProviderRegistration(
        WRAPPER_PROVIDER_MANIFEST,
        effects={
            operation.operation_id: SideEffect.READ
            for operation in WRAPPER_PROVIDER_MANIFEST.operations
        },
        binder=_bind_wrapper,
    )


def default_provider_catalog() -> ProviderCatalog:
    return ProviderCatalog((wrapper_provider(),))


def build_provider_registry(
    request: dict[str, Any],
    *,
    catalog: ProviderCatalog,
    context: Any = None,
) -> ProviderRegistry | None:
    """Bind explicit configured sources through an explicit host catalog."""

    if "data_source" in request:
        raise ValueError("legacy data_source is removed; use data_sources")
    raw = request.get("data_sources", [])
    if not isinstance(raw, list):
        raise ValueError("data_sources must be a list")
    sources: list[ConfiguredSource] = []
    for spec in raw:
        if not isinstance(spec, dict):
            raise ValueError("each data_sources entry must be an object")
        sources.append(ConfiguredSource.from_spec(spec))
    if not sources:
        return None
    default_source = request.get("default_source")
    return catalog.bind(
        tuple(sources),
        context=context,
        default_source_id=str(default_source) if default_source is not None else None,
    )
