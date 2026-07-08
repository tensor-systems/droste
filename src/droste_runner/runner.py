from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from typing import Any, Callable

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from droste.clients.errors import http_error_excerpt, redact_secrets  # type: ignore
from droste.execution.config import DEFAULT_MAX_CALLS, DEFAULT_MAX_ITERATIONS  # type: ignore
from droste.loop.rlm import RLMConfig, run_rlm  # type: ignore
from droste.protocols.environment import (  # type: ignore
    EnvCapabilities,
    ExecutionResult,
    RLMEnvironment,
)
from droste.protocols.llm_client import TokenUsage  # type: ignore
from droste.protocols.subcall_client import SubcallClient  # type: ignore
from droste.registry import DataSourceRegistry  # type: ignore


class OutputBuffer(io.StringIO):
    def __init__(self, max_chars: int) -> None:
        super().__init__()
        self._max_chars = max(0, int(max_chars or 0))
        self._size = 0

    def write(self, text: str) -> int:
        if not text:
            return 0
        if self._max_chars > 0:
            new_size = self._size + len(text)
            if new_size > self._max_chars:
                raise RuntimeError(
                    f"Sandbox output exceeded {self._max_chars} characters (attempted {new_size})."
                )
            self._size = new_size
        return super().write(text)


CONTEXT_PREVIEW_CHARS = 400
CONTEXT_PREVIEW_MAX_FILES = 20


def _safe_preview(text: str, limit: int = CONTEXT_PREVIEW_CHARS) -> str:
    """Head of `text`, truncated and defused so it cannot break its fenced block."""
    head = text[:limit]
    if len(text) > limit:
        head += "..."
    return head.replace("```", "'''")


def _safe_label(text: str, limit: int = 200) -> str:
    """File path/name for prompt inclusion: control chars and newlines
    stripped, then JSON-quoted so attacker-controlled names cannot inject
    prompt instructions outside a fence."""
    cleaned = "".join(ch for ch in text if ch.isprintable())[:limit]
    return json.dumps(cleaned, ensure_ascii=True)


def _describe_files_context(files: list[Any]) -> str:
    lines = [f"`context` is a dict with {len(files)} file(s) in context['files']:"]
    total_text = 0
    for entry in files[:CONTEXT_PREVIEW_MAX_FILES]:
        if not isinstance(entry, dict):
            lines.append(f"- (non-dict entry of type {type(entry).__name__})")
            continue
        path = _safe_label(str(entry.get("path") or entry.get("name") or "(unnamed)"))
        text = entry.get("text")
        text_len = len(text) if isinstance(text, str) else 0
        total_text += text_len
        lines.append(f"- {path} (text: {text_len:,} chars)")
    if len(files) > CONTEXT_PREVIEW_MAX_FILES:
        lines.append(f"- ... and {len(files) - CONTEXT_PREVIEW_MAX_FILES} more file(s)")
        for entry in files[CONTEXT_PREVIEW_MAX_FILES:]:
            text = entry.get("text") if isinstance(entry, dict) else None
            if isinstance(text, str):
                total_text += len(text)
    lines.append(f"Total attached text: {total_text:,} characters.")
    return "\n".join(lines)


def describe_context(context: Any) -> str:
    """Describe the `context` variable for the system prompt: type, total size,
    and a short escaped head preview (issue #20). Dict-of-files contexts get a
    shape summary (file count, per-file path + text length) instead of a raw
    dump."""
    if context is None:
        return "`context` is None (no context payload was provided)."
    if isinstance(context, str):
        return (
            f"`context` is a str of {len(context):,} characters. "
            f"Preview (first {CONTEXT_PREVIEW_CHARS} chars):\n"
            f"```\n{_safe_preview(context)}\n```"
        )
    if isinstance(context, dict) and isinstance(context.get("files"), list):
        return _describe_files_context(context["files"])
    try:
        serialized = json.dumps(context, ensure_ascii=True, default=str)
    except Exception:
        serialized = str(context)
    shape = f"a {type(context).__name__}"
    if isinstance(context, (list, tuple)):
        shape += f" of {len(context)} item(s)"
    elif isinstance(context, dict):
        shape += f" with {len(context)} key(s)"
    return (
        f"`context` is {shape}, {len(serialized):,} characters when JSON-serialized. "
        f"Preview (first {CONTEXT_PREVIEW_CHARS} chars):\n"
        f"```\n{_safe_preview(serialized)}\n```"
    )


class RunnerEnvironment(RLMEnvironment):
    def __init__(
        self,
        *,
        context: Any,
        registry: DataSourceRegistry | None,
        subcalls: SubcallClient,
        max_output_chars: int,
        exec_timeout_ms: int,
    ) -> None:
        self._context = context
        self._registry = registry
        self._subcalls = subcalls
        self._max_output_chars = max_output_chars
        self._exec_timeout_ms = exec_timeout_ms
        self._globals: dict[str, Any] = {
            "answer": {"content": "", "ready": False},
            "context": context,
            "llm_query": subcalls.llm_query,
            "llm_batch": subcalls.llm_batch,
            "batch_llm_query": subcalls.llm_batch,
            "llm_query_batched": subcalls.llm_batch,
        }
        if registry is not None:
            # Namespaced (e.g. db.query / vault.search) + default-flattened globals.
            self._globals.update(registry.globals())

    def capabilities(self) -> EnvCapabilities:
        return {
            "tools_in_root": False,
            "max_output_chars": self._max_output_chars,
        }

    def globals(self) -> dict[str, Any]:
        return self._globals

    def prompt_fragment(self) -> str:
        parts: list[str] = []
        parts.append(
            "Context is available in a Python variable named `context`. "
            "If it contains files, expect context['files'] entries with path, name, mime, size, and optional text."
        )
        # Size + preview signal (issue #20): without it the model reasonably
        # assumes the context fits in its own window and prints/counts in
        # Python instead of subcalling. Showing the variable's type,
        # length, and a short preview is the cue that keeps it subcalling.
        description = describe_context(self._context)
        if description:
            parts.append(description)
        parts.append(
            "Each llm_query / llm_query_batched subcall can handle roughly ~100k tokens; "
            "size chunks accordingly."
        )
        if self._registry is not None:
            fragment = self._registry.prompt_fragment()
            if fragment:
                parts.append(fragment)
        return "\n".join(parts)

    def execute(self, code: str) -> ExecutionResult:
        stdout_buf = OutputBuffer(self._max_output_chars)
        stderr_buf = io.StringIO()
        timed_out = False
        exit_code = 0

        def _handle_timeout(signum: int, frame: Any) -> None:
            raise TimeoutError("execution timed out")

        # The in-process SIGALRM timer is native-CPython only. Under Pyodide/WASM
        # (the v1 substrate) signals are unavailable, so the per-exec timeout is
        # enforced by the host instead (Deno wall-clock kill of the run).
        use_signal_timeout = bool(
            self._exec_timeout_ms and self._exec_timeout_ms > 0 and hasattr(signal, "setitimer")
        )
        old_handler = None
        if use_signal_timeout:
            old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, self._exec_timeout_ms / 1000.0)

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(compile(code, "<rlm>", "exec"), self._globals)
        except TimeoutError:
            timed_out = True
            exit_code = 124
            raise
        finally:
            if use_signal_timeout:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

        return ExecutionResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            timed_out=timed_out,
            exit_code=exit_code,
            files_written=[],
        )

    def close(self) -> None:
        return


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
            headers={"Authorization": "Bearer " + str(token), "Content-Type": "application/json"},
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


# --- Option C: build-time source-type registration (unified-data-sources §7.2)
#
# Consumers register factories for their own source types at *startup* — never
# from the request. The request stays declarative ({type, name, ...} — no module
# paths, no code); the set of runnable types is fixed by the deployment's own
# entrypoint, so there is no request-controlled import path.

# Version of the DataSource/registration contract. A consumer built against a
# different contract fails at registration (startup), not subtly at request time.
SOURCE_PROTOCOL_VERSION = 1

# factory(config, ctx) -> DataSource. `config` is the request's declarative
# per-source entry; `ctx` is the host-supplied edge context (e.g. a live
# read-only DB handle) threaded through build_data_sources — see §7.3.
SourceFactory = Callable[[dict[str, Any], Any], Any]

_source_factories: dict[str, SourceFactory] = {}


def register_source_type(
    stype: str,
    factory: SourceFactory,
    *,
    protocol: int = SOURCE_PROTOCOL_VERSION,
) -> None:
    """Register a factory for a data source type (process-global, at startup)."""
    if protocol != SOURCE_PROTOCOL_VERSION:
        raise RuntimeError(
            f"source type {stype!r} was registered against protocol {protocol}; "
            f"this engine speaks protocol {SOURCE_PROTOCOL_VERSION}"
        )
    key = str(stype or "").strip()
    if not key:
        raise ValueError("source type must be a non-empty string")
    if key == "wrapper_v1":
        raise ValueError("source type 'wrapper_v1' is built in and cannot be re-registered")
    if key in _source_factories:
        raise ValueError(f"source type {key!r} is already registered")
    if not callable(factory):
        raise TypeError("factory must be callable")
    _source_factories[key] = factory


def _reset_source_types() -> None:
    """Test hook: clear registered source-type factories."""
    _source_factories.clear()


def _build_one_source(spec: dict[str, Any], ctx: Any = None) -> Any:
    stype = str(spec.get("type") or "").strip()
    name = str(spec.get("name") or "").strip()
    if stype == "wrapper_v1":
        return WrapperV1DataSource(spec, name=name or "wrapper")
    factory = _source_factories.get(stype)
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


# The bounded-read + redaction HTTP-error helpers moved to droste.clients.errors
# so the BYOK OpenAI-compatible client shares them (#27). Aliased here because
# this module's callers (and its tests) know them by the underscored names.
_redact_secrets = redact_secrets
_http_error_excerpt = http_error_excerpt


class HTTPSubcallClient(SubcallClient):
    def __init__(
        self,
        *,
        endpoint: str,
        token: str,
        session: str,
        session_index: int,
        max_calls: int,
        max_depth: int,
        context: Any,
        max_output_tokens: int = 0,
        model: str = "",
        reasoning_effort: str = "",
    ) -> None:
        self._endpoint = endpoint
        self._token = token
        self._session = session
        self._session_index = int(session_index or 0)
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._max_calls = int(max_calls)
        self._max_depth = int(max_depth)
        self._context = context
        self._depth = threading.local()
        # Subcall cost controls (#25): included in each subcall payload when
        # set; omitted when unset so the server owns the defaults (bounded
        # output + no thinking).
        self._max_output_tokens = int(max_output_tokens or 0)
        self._model = str(model or "")
        self._reasoning_effort = str(reasoning_effort or "")

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def _depth_get(self) -> int:
        return getattr(self._depth, "value", 0)

    def _depth_set(self, value: int) -> None:
        self._depth.value = value

    def _increment_calls(self) -> None:
        # Check-then-increment under the lock: the count is the reported
        # subcall total, so a rejected over-limit attempt must not inflate it,
        # and concurrent llm_batch threads must not race the check.
        with self._seq_lock:
            if self._max_calls >= 0 and self._context.stats.calls_made >= self._max_calls:
                raise RuntimeError("max subcalls exceeded")
            self._context.stats.calls_made += 1

    def _request(self, payload: dict[str, Any]) -> str:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Authorization": "Bearer " + self._token, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status = getattr(exc, "code", 0)
            excerpt = _http_error_excerpt(exc)
            detail = f": {excerpt}" if excerpt else f": {exc}"
            raise RuntimeError(f"llm_query failed with HTTP {status}{detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"llm_query failed: {exc}") from exc
        data = json.loads(raw)
        result = data.get("result")
        if not isinstance(result, str):
            raise RuntimeError("missing subcall result")
        return result

    def llm_query(self, prompt: str, context: str = "") -> str:
        if context:
            prompt = f"{context}\n\n{prompt}"
        auto_depth = True
        depth = self._depth_get() + 1
        if auto_depth:
            self._depth_set(depth)
        try:
            if self._max_depth >= 0 and depth > self._max_depth:
                raise RuntimeError("max depth exceeded")
            self._increment_calls()
            payload: dict[str, Any] = {
                "prompt": prompt,
                "depth": depth,
                "seq": self._next_seq(),
                "session": self._session,
                "session_index": self._session_index,
            }
            if self._max_output_tokens > 0:
                payload["max_output_tokens"] = self._max_output_tokens
            if self._model:
                payload["model"] = self._model
            if self._reasoning_effort:
                payload["reasoning_effort"] = self._reasoning_effort
            return self._request(payload)
        finally:
            if auto_depth:
                self._depth_set(depth - 1)

    def llm_batch(self, prompts: list[str], contexts: list[str] | None = None) -> list[str]:
        if contexts is None:
            contexts = [""] * len(prompts)
        if len(contexts) != len(prompts):
            raise ValueError("contexts length must match prompts length")
        results: list[str] = [""] * len(prompts)
        if not prompts:
            return results
        if len(prompts) > 50:
            raise ValueError("llm_batch prompt count exceeds max 50")
        max_parallel = 5

        def _run_one(idx: int, prompt: str, ctx: str) -> str:
            if idx > 0:
                time.sleep(0.05)
            return self.llm_query(prompt, ctx)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        errors: list[Exception | None] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {}
            for idx, (prompt, ctx) in enumerate(zip(prompts, contexts)):
                future = executor.submit(_run_one, idx, prompt, ctx)
                futures[future] = idx
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    errors[idx] = exc
        for err in errors:
            if err is not None:
                raise err
        return results

    def llm_batch_with_errors(
        self,
        prompts: list[str],
        contexts: list[str] | None = None,
    ) -> tuple[list[str], list[dict[str, object]]]:
        if contexts is None:
            contexts = [""] * len(prompts)
        if len(contexts) != len(prompts):
            raise ValueError("contexts length must match prompts length")
        results: list[str] = [""] * len(prompts)
        errors: list[dict[str, object]] = []
        if not prompts:
            return results, errors

        def _run_one(idx: int, prompt: str, ctx: str) -> None:
            try:
                results[idx] = self.llm_query(prompt, ctx)
            except Exception as exc:
                errors.append({"index": idx, "error": str(exc)})

        threads = []
        for idx, (prompt, ctx) in enumerate(zip(prompts, contexts)):
            t = threading.Thread(
                target=lambda i=idx, p=prompt, c=ctx: _run_one(i, p, c), daemon=True
            )
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results, errors


class RootLLMClient:
    def __init__(
        self,
        *,
        endpoint: str,
        token: str,
        default_model: str,
        provider: str | None,
        max_output_tokens: int,
        temperature: float | None,
        stop: list[str] | None,
        session: str,
        session_index: int,
    ) -> None:
        self._endpoint = endpoint
        self._token = token
        self._default_model = default_model
        self._provider = provider
        self._max_output_tokens = int(max_output_tokens or 0)
        self._temperature = temperature
        self._stop = stop or []
        self._session = session
        self._session_index = int(session_index or 0)
        self.last_provider = ""
        self.last_response_id = ""
        self.last_stop_reason = ""
        self.last_model = ""

    def responses_create(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
        return_usage: bool = False,
    ) -> str | tuple[str, TokenUsage]:
        resolved_model = model or self._default_model
        if not resolved_model:
            raise ValueError("model is required")
        max_output_tokens = self._max_output_tokens or int(max_tokens or 0)
        # Only send temperature when someone actually set it — modern models
        # (gpt-5.x, opus-4.x) reject the parameter outright, so a synthetic
        # 0.0 default breaks the root call for no benefit.
        temp = self._temperature if self._temperature is not None else temperature
        payload: dict[str, Any] = {
            "messages": messages,
            "model": resolved_model,
            "max_output_tokens": max_output_tokens,
            "stop": self._stop,
            "session": self._session,
            "session_index": self._session_index,
        }
        if temp is not None:
            payload["temperature"] = temp
        if self._provider:
            payload["provider"] = self._provider
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Authorization": "Bearer " + self._token, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status = getattr(exc, "code", 0)
            excerpt = _http_error_excerpt(exc)
            detail = f": {excerpt}" if excerpt else f": {exc}"
            raise RuntimeError(f"root llm failed with HTTP {status}{detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"root llm failed: {exc}") from exc
        data = json.loads(raw)
        result = data.get("result")
        if not isinstance(result, str):
            raise RuntimeError("missing root result")
        self.last_provider = str(data.get("provider") or "")
        self.last_response_id = str(data.get("response_id") or "")
        self.last_stop_reason = str(data.get("stop_reason") or "")
        self.last_model = str(data.get("model") or "")
        if return_usage:
            usage_payload = data.get("usage", {}) if isinstance(data, dict) else {}
            input_tokens = int(usage_payload.get("input_tokens", 0) or 0)
            output_tokens = int(usage_payload.get("output_tokens", 0) or 0)
            total_tokens = usage_payload.get("total_tokens")
            if total_tokens is None:
                total_tokens = input_tokens + output_tokens
            usage = TokenUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=int(total_tokens or 0),
            )
            return result, usage
        return result


def _read_request(path: str | None = None) -> dict[str, Any]:
    path = path or os.environ.get("RLM_RUNNER_REQUEST_PATH")
    if not path and len(sys.argv) > 1:
        path = sys.argv[1]
    if not path:
        raise RuntimeError("RLM_RUNNER_REQUEST_PATH is required")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_context(payload: dict[str, Any]) -> Any:
    if "context_path" in payload and payload["context_path"]:
        with open(payload["context_path"], "r", encoding="utf-8") as handle:
            return json.load(handle)
    return payload.get("context")


def _run_adapter(request: dict[str, Any]) -> dict[str, Any]:
    adapter_module = str(request.get("adapter_module") or "").strip()
    if not adapter_module:
        raise RuntimeError("adapter_module is required for adapter runs")
    module = importlib.import_module(adapter_module)
    run_fn = getattr(module, "run", None)
    if not callable(run_fn):
        raise RuntimeError(f"adapter module {adapter_module} missing run(request) function")
    result = run_fn(request)
    if not isinstance(result, dict):
        raise RuntimeError(
            f"adapter module {adapter_module} returned {type(result)}; expected dict"
        )
    return result


def run(request: dict[str, Any], *, source_ctx: Any = None) -> dict[str, Any]:
    adapter_module = request.get("adapter_module")
    if isinstance(adapter_module, str) and adapter_module.strip():
        return _run_adapter(request)

    context = _build_context(request)
    # source_ctx is the host-supplied edge context for registered source
    # factories (unified-data-sources §7.3): in-process hosts pass live
    # handles; subprocess hosts pass whatever their entrypoint assembled.
    sources, default_source = build_data_sources(request, source_ctx)
    registry = DataSourceRegistry(sources, default_source_name=default_source) if sources else None

    # Omitted budgets fall back to the core loop defaults instead of the old
    # runner-local 1 iteration / 0 subcalls (0 made the FIRST llm_query raise
    # "max subcalls exceeded"). An EXPLICIT value is always honored — including
    # max_subcalls=0, which deliberately forbids subcalls. Hosts that care
    # about cost must pass explicit budgets in the request.
    def _budget(key: str, default: int) -> int:
        raw = request.get(key)
        if raw is None or raw == "":
            return default
        return int(raw)

    max_iterations = _budget("max_iterations", DEFAULT_MAX_ITERATIONS)
    max_depth = int(request.get("max_depth") or 1)
    max_subcalls = _budget("max_subcalls", DEFAULT_MAX_CALLS)
    max_output_chars = int(request.get("max_output_chars") or 0)
    exec_timeout_ms = int(request.get("exec_timeout_ms") or 0)

    token = str(request.get("token") or "")
    root_endpoint = str(request.get("root_endpoint") or "")
    subcall_endpoint = str(request.get("subcall_endpoint") or "")
    session = str(request.get("session") or "")
    session_index = int(request.get("session_index") or 0)

    from droste.execution.context import create_execution_context  # type: ignore

    exec_context = create_execution_context(
        max_depth=max_depth,
        max_calls=max_subcalls,
        max_iterations=max_iterations,
        max_output_chars=max_output_chars,
        verbose=False,
    )

    model = str(request.get("model") or "")
    provider = str(request.get("provider") or "") or None
    max_output_tokens = int(request.get("max_output_tokens") or 0)
    temperature = request.get("temperature")
    stop = request.get("stop")

    if not token or not root_endpoint or not subcall_endpoint:
        raise RuntimeError("missing endpoints or token")
    root_client = RootLLMClient(
        endpoint=root_endpoint,
        token=token,
        default_model=model,
        provider=provider,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        stop=stop,
        session=session,
        session_index=session_index,
    )
    # Subcall cost controls (#25): optional per-run overrides forwarded in
    # every subcall payload; unset values are omitted so the server applies
    # its defaults (bounded output + no thinking for subcalls). An explicit
    # zero/negative budget is rejected: a subcall cannot answer in 0 tokens,
    # and silently treating it as unset would mask a caller bug.
    raw_subcall_max = request.get("subcall_max_output_tokens")
    if raw_subcall_max in (None, ""):
        subcall_max_output_tokens = 0
    else:
        subcall_max_output_tokens = int(raw_subcall_max)
        if subcall_max_output_tokens <= 0:
            raise ValueError("subcall_max_output_tokens must be positive when set")
    subcall_model = str(request.get("subcall_model") or "")
    subcall_reasoning_effort = str(request.get("subcall_reasoning_effort") or "")

    subcalls = HTTPSubcallClient(
        endpoint=subcall_endpoint,
        token=token,
        session=session,
        session_index=session_index,
        max_calls=max_subcalls,
        max_depth=max_depth,
        context=exec_context,
        max_output_tokens=subcall_max_output_tokens,
        model=subcall_model,
        reasoning_effort=subcall_reasoning_effort,
    )

    environment = RunnerEnvironment(
        context=context,
        registry=registry,
        subcalls=subcalls,
        max_output_chars=max_output_chars,
        exec_timeout_ms=exec_timeout_ms,
    )

    config = RLMConfig(
        max_iterations=max_iterations,
        max_depth=max_depth,
        max_calls=max_subcalls,
        max_output_chars=max_output_chars,
        root_model=str(request.get("model") or ""),
        verbose=False,
    )

    system_prompt_raw = request.get("system_prompt")
    system_prompt = None
    if isinstance(system_prompt_raw, str) and system_prompt_raw.strip():
        system_prompt = system_prompt_raw
    system_prompt_additions = str(request.get("system_prompt_additions") or "")

    result = run_rlm(
        str(request.get("question") or ""),
        environment=environment,
        root_llm=root_client,
        subcalls=subcalls,
        config=config,
        system_prompt=system_prompt,
        system_prompt_additions=system_prompt_additions,
        conversation_context=str(request.get("conversation_context") or ""),
        # The subcall client increments exec_context.stats.calls_made; without
        # sharing it, run_rlm creates its own context and reports subcalls=0
        # no matter how many subcalls actually ran.
        context=exec_context,
    )

    response: dict[str, Any] = {
        "answer": result.answer,
        "ready": result.ready,
        "iterations": result.iterations,
        "tokens_used": result.tokens_used,
        "subcalls": result.sub_calls_made,
        "extracted": bool(getattr(result, "extracted", False)),
        "extract_error": None,
        "trajectory": [
            {
                "iteration": entry.iteration,
                "llm_input": entry.llm_input,
                "llm_output": entry.llm_output,
                "code_executed": entry.code_executed,
                "execution_result": entry.execution_result,
                "tokens_used": entry.tokens_used,
            }
            for entry in result.trajectory
        ],
        "error": None,
        "provider": root_client.last_provider,
        "response_id": root_client.last_response_id,
        "stop_reason": root_client.last_stop_reason,
        "model": root_client.last_model or str(request.get("model") or ""),
    }
    wrapper_sources = [s for s in sources if isinstance(s, WrapperV1DataSource)]
    if wrapper_sources:
        response["data_source_requests"] = sum(s.requests_made for s in wrapper_sources)
    if result.error:
        response["error"] = {
            "type": result.error.type,
            "message": result.error.message,
            "code": result.error.code,
            "details": result.error.details,
        }
    extract_error = getattr(result, "extract_error", None)
    if extract_error:
        response["extract_error"] = {
            "type": extract_error.type,
            "message": extract_error.message,
            "code": extract_error.code,
            "details": extract_error.details,
        }
    return response


def main() -> None:
    request = _read_request()
    # The request file is the untrusted boundary (hosted runners are fed one by
    # the parent process). A request must never name code to import: source
    # types come from register_source_type() in the entrypoint (Option C), and
    # the in-process adapter seam is reserved for trusted callers of run().
    if str(request.get("adapter_module") or "").strip():
        raise RuntimeError(
            "adapter_module is not accepted from the request file; register "
            "source types via register_source_type() in the runner entrypoint"
        )
    response = run(request)
    sys.stdout.write(json.dumps(response, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        payload = {
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=True))
        sys.exit(1)
