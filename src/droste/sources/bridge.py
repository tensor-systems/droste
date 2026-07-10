"""Bridge-backed `DataSource`: proxy a real data source across a trust
boundary between two interpreter contexts (A'-2 sandbox split).

Two halves, kept in one file so the wire contract can't drift between them:

  DataSourceService - server half. Runs alongside the real `DataSource` in the
                       trusted context. Dispatches `(method, params_json)` to a
                       fixed method allowlist and returns a JSON envelope.
  BridgeDataSource   - client half. Runs in the untrusted context. Implements
                       the `DataSource` protocol by forwarding every call over
                       an injected synchronous `bridge_call(method, params_json)
                       -> response_json`.

Neither half knows what's really behind the bridge (iMessage, SQL, anything);
the host wires a `bridge_call` implementation (e.g. a Deno JSON-RPC transport
between two Pyodide interpreters) and supplies the real `DataSource` to
`DataSourceService` on the trusted side.

Security note: generated code in the untrusted context can always reach the
raw `bridge_call` directly (e.g. via `db.query.__self__._call`), bypassing
`BridgeDataSource` entirely. `DataSourceService.handle` is therefore the real
boundary — it dispatches only against a hardcoded method allowlist gated by
the wrapped source's own `capabilities()` / `hasattr()`, never generic
`getattr(source, method)` on the caller-supplied method name.
"""

from __future__ import annotations

import functools
import json
from typing import Any, Callable

from ..protocols.data_source import DataSource

# Core protocol verbs, gated by the corresponding capabilities() flag (mirrors
# registry.py's own gating, so a method the registry would never bind into the
# sandbox can't be reached by calling the bridge directly either).
_CORE_METHODS: dict[str, str] = {
    "search": "search",
    "query": "sql",
    "get": "get",
    "get_recent": "recent",
    "get_schema": "schema",
    "get_stats": "stats",
}

# Optional verbs, gated by hasattr(source, name) — mirrors registry.py's own
# hasattr checks for these (no capabilities() flag governs them). Host-specific
# extras that registry.py doesn't know about (e.g. a host's own retrieved-IDs
# tracking) go through DataSourceService's extra_methods= instead of this
# tuple — this one is droste's own sandbox-verb list, not a place for a
# specific host's own conventions.
_OPTIONAL_METHODS: tuple[str, ...] = (
    "find",
    "content",
    "get_messages",
    "get_chats",
    "get_chat_messages",
    "sample",
)


class DataSourceService:
    """Server half: dispatches bridge calls to a real `DataSource`."""

    def __init__(self, source: DataSource, *, extra_methods: tuple[str, ...] = ()) -> None:
        self._source = source
        self._caps: dict[str, bool] = dict(source.capabilities())
        # extra_methods: host-specific optional verbs beyond droste's own
        # _OPTIONAL_METHODS — e.g. a host's DataSource may track retrieved-
        # record IDs for a citations feature via a get_retrieved_guids()-shaped
        # method that isn't part of the DataSource Protocol at all and means
        # nothing to droste. Gated identically (hasattr on the wrapped source)
        # and folded into the same describe()/dispatch machinery, so
        # BridgeDataSource needs no changes to pick them up — it already binds
        # whatever describe() reports in optional_methods.
        self._optional_names: tuple[str, ...] = _OPTIONAL_METHODS + tuple(extra_methods)
        self._optional: set[str] = {name for name in self._optional_names if hasattr(source, name)}

    def describe(self) -> dict[str, Any]:
        schema = self._source.get_schema() if self._caps.get("schema") else None
        return {
            "name": self._source.name(),
            "capabilities": self._caps,
            "schema": schema,
            "optional_methods": sorted(self._optional),
        }

    def handle(self, method: str, params_json: str) -> str:
        """Dispatch one bridge call; never raises — errors come back in the envelope."""
        try:
            result = self._dispatch(method, params_json)
            return json.dumps({"ok": True, "result": result}, default=str)
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
            )

    def _dispatch(self, method: str, params_json: str) -> Any:
        payload = json.loads(params_json) if params_json else {}
        if not isinstance(payload, dict):
            raise ValueError("bridge params must be a JSON object")
        args = payload.get("args") or []
        kwargs = payload.get("kwargs") or {}
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise ValueError("bridge params must have an 'args' list and a 'kwargs' object")

        if method == "describe":
            return self.describe()

        gate = _CORE_METHODS.get(method)
        if gate is not None:
            if not self._caps.get(gate):
                raise PermissionError(f"method {method!r} is not enabled by this data source")
            return getattr(self._source, method)(*args, **kwargs)

        if method in self._optional_names:
            if method not in self._optional:
                raise PermissionError(f"method {method!r} is not implemented by this data source")
            return getattr(self._source, method)(*args, **kwargs)

        raise ValueError(f"unknown bridge method: {method!r}")


BridgeCall = Callable[[str, str], Any]

# Core verbs whose real-world signatures vary by source beyond what the
# DataSource Protocol declares — e.g. WrapperV1DataSource.search(query,
# filters=None, page=None) has a `page` the Protocol doesn't, and lacks
# limit/sender/chat/days/table that other search()-capable sources use. A
# fixed BridgeDataSource signature can't proxy that faithfully in either
# direction: a concrete default would get forwarded on every call even when
# the caller omitted it (silently overriding the wrapped source's own
# default), and a missing param can't be forwarded at all. These are instead
# bound dynamically (see __init__), the same as the optional verbs below,
# forwarding *args/**kwargs verbatim — byte-identical, argument-wise, to
# calling the wrapped source directly, for any signature.
_DYNAMIC_CORE_METHODS: tuple[str, ...] = ("search", "query", "get", "get_recent")


class BridgeDataSource:
    """Client half: a `DataSource` that proxies every call over `bridge_call`."""

    def __init__(self, bridge_call: BridgeCall, *, name: str) -> None:
        self._call = bridge_call
        self._name = name
        described = self._request("describe")
        self._caps: dict[str, bool] = dict(described.get("capabilities") or {})
        self._schema: str = described.get("schema") or ""
        self._optional: set[str] = set(described.get("optional_methods") or [])
        # Both the capability-gated core verbs and the hasattr-gated optional
        # verbs are bound as INSTANCE attributes, and only when the remote
        # side actually reports them — never defined at class level. A
        # class-level definition would make e.g. hasattr(bridged,
        # "get_chats") true for every BridgeDataSource regardless of what the
        # wrapped source actually supports (registry.py gates optional verbs
        # by hasattr), so a model calling it would get a runtime bridge error
        # instead of the method never being offered in the first place.
        for method in _DYNAMIC_CORE_METHODS:
            if self._caps.get(_CORE_METHODS[method]):
                setattr(self, method, functools.partial(self._request, method))
        for method in self._optional:
            setattr(self, method, functools.partial(self._request, method))

    def _request(self, method: str, *args: Any, **kwargs: Any) -> Any:
        raw = self._call(method, json.dumps({"args": list(args), "kwargs": kwargs}))
        # Awaitable tolerance: under Pyodide the bridge call may be async
        # (mirrors BridgedLLMClient._post in droste.substrates.pyodide); block the
        # sync RLM loop on it via run_sync. A plain string (native tests, or a
        # synchronous in-process loopback) is used as-is.
        if hasattr(raw, "__await__"):
            from pyodide.ffi import run_sync

            raw = run_sync(raw)
        try:
            envelope = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"bridge call {method!r} returned a non-JSON response: {raw!r}"
            ) from exc
        if not isinstance(envelope, dict) or not envelope.get("ok"):
            error = envelope.get("error") if isinstance(envelope, dict) else None
            error = error if isinstance(error, dict) else {}
            err_type = error.get("type", "BridgeError")
            err_message = error.get("message", "unknown bridge error")
            # The validator/OperationalError message is model-facing feedback the
            # loop iterates on — preserve type + message instead of an opaque
            # "bridge call failed".
            raise RuntimeError(f"{err_type}: {err_message}")
        return envelope["result"]

    # -- DataSource protocol -------------------------------------------------

    def name(self) -> str:
        return self._name

    def capabilities(self) -> dict[str, bool]:
        return dict(self._caps)

    def get_schema(self) -> str:
        return self._schema

    def get_stats(self) -> dict[str, Any]:
        return self._request("get_stats")

    # search/query/get/get_recent are bound dynamically in __init__ (see
    # _DYNAMIC_CORE_METHODS) when the wrapped source enables them — not
    # defined here, so calling one that isn't enabled fails the same way
    # (AttributeError) as it would on any other DataSource that simply
    # doesn't implement it (e.g. LocalSqlDataSource has no search/get/
    # get_recent at all).
