"""Runner-owned remote source transport and declarative source construction."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from droste.clients.errors import http_error_excerpt
from droste.clients.useragent import USER_AGENT
from droste.sources.registration import (
    SOURCE_PROTOCOL_VERSION,
    register_source_type,
    source_factory,
)
from droste.sources.registration import (
    SourceFactory as SourceFactory,
)
from droste.sources.registration import (
    _reset_source_types as _registration_reset_source_types,
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


class DataSourceWrapper:
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


class WrapperV1DataSource:
    """`DataSource` adapter over the remote wrapper_v1 HTTP transport.

    Lets the remote search/get/content partner API participate in the
    `DataSourceRegistry` like any in-process source instead of being a parallel
    special-case. Request budgets and the `allowed_hosts` allowlist are
    enforced per-request by the wrapped `DataSourceWrapper` (no allowlist
    configured = the deployer accepts arbitrary hosts). `content` is exposed
    as an extra method (the registry binds it via its `hasattr` path); it is
    not a first-class capability.
    """

    def __init__(self, config: dict[str, Any] | None, *, name: str = "wrapper") -> None:
        self._config = config or {}
        self._name = name or "wrapper"
        self._wrapper = DataSourceWrapper(self._config)

    def name(self) -> str:
        return self._name

    def capabilities(self) -> dict[str, bool]:
        return {
            "sql": False,
            "search": True,
            "get": True,
            "recent": False,
            "schema": True,
            "stats": True,
        }

    def get_schema(self) -> str:
        parts = ["Remote wrapper_v1 source — search(query)/get(id)/content(id) over HTTP."]
        allowed_hosts = self._config.get("allowed_hosts")
        if isinstance(allowed_hosts, list) and allowed_hosts:
            parts.append("Allowed hosts: " + ", ".join(str(h) for h in allowed_hosts if h))
        limits = self._config.get("limits")
        if isinstance(limits, dict) and limits:
            parts.append("Limits: " + json.dumps(limits, ensure_ascii=True))
        return " ".join(parts)

    def get_stats(self) -> dict[str, Any]:
        return {"requests_made": self._wrapper.requests_made}

    @property
    def requests_made(self) -> int:
        return self._wrapper.requests_made

    def search(self, query: str, filters: Any = None, page: Any = None) -> Any:
        return self._wrapper.search(query, filters, page)

    def get(self, id: str) -> Any:
        return self._wrapper.get(id)

    def content(self, id: str, format: str = "text", max_bytes: int | None = None) -> Any:
        return self._wrapper.content(id, format, max_bytes)


# --- Option C: build-time source-type registration (source unification)
#
# Consumers register factories for their own source types at *startup* — never
# from the request. The machinery lives in droste.sources.registration (#32)
# so non-runner embedders can use it too; this module re-exports the names
# and registers the runner's own built-in wrapper_v1 type through it.


def _wrapper_v1_factory(spec: dict[str, Any], ctx: Any = None) -> Any:
    name = str(spec.get("name") or "").strip()
    return WrapperV1DataSource(spec, name=name or "wrapper")


def _register_builtin_source_types() -> None:
    # The runner's built-in remote source registers through the same
    # mechanism as everyone else (#32) instead of a _build_one_source
    # carve-out; attempts to re-register the type fail like any duplicate.
    # Import order must not subvert that (codex review): a consumer that
    # registered its own 'wrapper_v1' BEFORE this module was imported gets a
    # loud failure here, never a silent factory swap.
    existing = source_factory("wrapper_v1")
    if existing is _wrapper_v1_factory:
        return
    if existing is not None:
        raise ValueError(
            "source type 'wrapper_v1' is built in to droste_runner and cannot "
            "be replaced by a pre-registered factory"
        )
    register_source_type("wrapper_v1", _wrapper_v1_factory, protocol=SOURCE_PROTOCOL_VERSION)


_register_builtin_source_types()


def _reset_source_types() -> None:
    """Test hook: clear registered source-type factories (builtins stay)."""
    _registration_reset_source_types()
    _register_builtin_source_types()


def _build_one_source(spec: dict[str, Any], ctx: Any = None) -> Any:
    stype = str(spec.get("type") or "").strip()
    factory = source_factory(stype)
    if factory is not None:
        source = factory(spec, ctx)
        if source is None:
            raise ValueError(f"factory for source type {stype!r} returned no source")
        return source
    if stype in ("sql", "fs"):
        # Minimal-engine contract (Option C): in-process sources carry DB drivers
        # and path/SQL policy, which live with the consumer that owns the data
        # boundary — registered via register_source_type() at startup, never
        # constructed here.
        raise ValueError(
            f"data source type '{stype}' has no registered factory; the consumer's "
            "runner entrypoint must call register_source_type() at startup "
            "(the base runner only builds remote wrapper_v1 sources)"
        )
    raise ValueError(f"unknown data source type: {stype!r}")


def build_data_sources(request: dict[str, Any], ctx: Any = None) -> tuple[list[Any], str | None]:
    """Resolve a request's data sources into `DataSource` objects + a default name.

    Accepts the new `data_sources` list shape, and the legacy singular
    `data_source` wrapper_v1 config as sugar for a one-element list. Non-built-in
    types dispatch to factories registered via `register_source_type`; `ctx` is
    passed through to each factory as the host-supplied edge context.
    """
    default_name = request.get("default_source")
    if default_name is not None:
        default_name = str(default_name)
    sources: list[Any] = []

    raw = request.get("data_sources")
    if raw is not None:
        # Present-but-malformed must fail loudly, not silently fall through to
        # the legacy singular path (which would ignore the caller's intent).
        if not isinstance(raw, list):
            raise ValueError("data_sources must be a list")
        for spec in raw:
            if not isinstance(spec, dict):
                raise ValueError("each data_sources entry must be an object")
            sources.append(_build_one_source(spec, ctx))
        return sources, default_name

    legacy = request.get("data_source")
    if legacy is not None:
        if not isinstance(legacy, dict):
            raise ValueError("data_source must be an object")
        sources.append(WrapperV1DataSource(legacy, name="wrapper"))
        if default_name is None:
            default_name = "wrapper"
    return sources, default_name
