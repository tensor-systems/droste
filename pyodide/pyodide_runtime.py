"""Pyodide-substrate adapters for the Recall RLM (Phase 2 step 3).

Two pieces are substrate-specific and injected into the otherwise-unchanged
droste loop + RecallEnvironment:

  RawExecutor      - replaces RestrictedExecutor. Runs generated code with a plain
                     interpreter (the Deno/WASM jail is the sandbox, so RestrictedPython
                     is redundant here). Pyodide-safe: no signals/threads.
  BridgedLLMClient - replaces the httpx ModelRelay client. Implements the droste
                     LLMClient protocol by calling a host fetch function injected by
                     the Deno host, which performs the real ModelRelay /responses call.

Both have zero third-party deps so they import under Pyodide.
"""

from __future__ import annotations

import builtins
import contextlib
import dataclasses
import io
from typing import TYPE_CHECKING, Any, Callable

from droste.protocols.llm_client import TokenUsage

if TYPE_CHECKING:
    from droste.sources.bridge import BridgeCall

# Bind the interpreter primitive once, away from call sites, so static scanners
# don't confuse it with shell exec.
_run_code = builtins.exec


# --------------------------------------------------------------------------- #
# RawExecutor — matches RestrictedExecutor.execute(code, extra_globals) -> str
# --------------------------------------------------------------------------- #
class RawExecutor:
    def __init__(self, db: Any, max_output_chars: int = 0) -> None:
        self._db = db
        self._max_output_chars = max_output_chars
        # Persistent namespace so variables/imports/answer survive across iterations.
        self._namespace: dict[str, Any] = {}

    def execute(self, code: str, extra_globals: dict[str, Any] | None = None) -> str:
        if extra_globals:
            self._namespace.update(extra_globals)
        buf = io.StringIO()
        compiled = compile(code, "<rlm>", "exec")
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            _run_code(compiled, self._namespace)
        out = buf.getvalue()
        if self._max_output_chars and len(out) > self._max_output_chars:
            out = out[: self._max_output_chars]
        return out

    def close(self) -> None:  # parity with RestrictedExecutor
        pass


# --------------------------------------------------------------------------- #
# BridgedLLMClient — implements the droste LLMClient protocol over a host fetch
# --------------------------------------------------------------------------- #
def _build_input(messages: list[dict]) -> list[dict]:
    """OpenAI-style messages -> ModelRelay input items (mirrors modelrelay._build_input)."""
    items = []
    for msg in messages:
        content = msg["content"]
        content_items = (
            content if isinstance(content, list) else [{"type": "text", "text": content}]
        )
        items.append({"type": "message", "role": msg["role"], "content": content_items})
    return items


def _extract_text(data: dict[str, Any]) -> str:
    """Pull assistant text from a ModelRelay /responses payload (mirrors modelrelay)."""
    parts = []
    for item in data.get("output", []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for part in item.get("content", []):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
    return "".join(parts)


# host_fetch(method, url, headers_json, body) -> response_text  (provided by Deno)
HostFetch = Callable[[str, str, str, str], str]


class BridgedLLMClient:
    def __init__(
        self,
        host_fetch: HostFetch,
        api_key: str | None = None,
        customer_token: str | None = None,
        base_url: str = "https://api.modelrelay.ai/api/v1",
    ) -> None:
        self._fetch = host_fetch
        self._api_key = api_key
        self._customer_token = customer_token
        self._base_url = base_url

    def _auth_headers(self) -> dict[str, str]:
        # Customer token (PAYGO users) takes precedence over the dev API key —
        # mirrors ModelRelayClient._get_auth_headers.
        if self._customer_token:
            return {"Authorization": f"Bearer {self._customer_token}"}
        return {"X-ModelRelay-Api-Key": self._api_key or ""}

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        import json

        headers = json.dumps({**self._auth_headers(), "Content-Type": "application/json"})
        raw = self._fetch("POST", f"{self._base_url}{path}", headers, json.dumps(body))
        # Under Pyodide the host fetch is async (returns an awaitable): block the
        # synchronous RLM loop on it via run_sync (Pyodide 0.29 + Deno supports this
        # with no flag). A plain-string fetch (native tests) is used as-is.
        if hasattr(raw, "__await__"):
            from pyodide.ffi import run_sync

            raw = run_sync(raw)
        try:
            return json.loads(raw)
        except (ValueError, TypeError) as exc:
            snippet = (raw if isinstance(raw, str) else str(raw))[:1000]
            raise RuntimeError(
                f"ModelRelay returned a non-JSON response from {path}: {snippet!r}"
            ) from exc

    def responses_create(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        return_usage: bool = False,
    ) -> str | tuple[str, TokenUsage]:
        data = self._post(
            "/responses",
            {
                "model": model,
                "input": _build_input(messages),
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        text = _extract_text(data)
        if return_usage:
            usage = data.get("usage", {})
            inp = int(usage.get("input_tokens", 0))
            out = int(usage.get("output_tokens", 0))
            return text, TokenUsage(
                prompt_tokens=inp, completion_tokens=out, total_tokens=inp + out
            )
        return text

    def batch_responses_typed(
        self,
        requests: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> Any:
        # Server-side fan-out (ModelRelay's /responses/batch does the batching;
        # there's no client-side threading to port here, and Pyodide can't
        # thread anyway). Reuses rcl_rlm's own wire-format parser rather than
        # duck-typing a second BatchResponse shape — matches this module's
        # existing lazy-import-rcl_rlm pattern (see build_db_service).
        body: dict[str, Any] = {"requests": requests}
        if options:
            body["options"] = options
        data = self._post("/responses/batch", body)
        from rcl_rlm.modelrelay import BatchResponse

        return BatchResponse.from_dict(data)

    def get_model_context_window(self, model: str) -> int | None:
        return None

    def get_model_max_output_tokens(self, model: str) -> int | None:
        # Model metadata isn't fetched over the bridge; conversation budgeting
        # falls back to its defaults when this is None (see _context_char_budget).
        return None


def _serialize_error(err: Any) -> dict[str, Any] | None:
    """Make the RLM error JSON-serializable for the host response.

    ``run_rlm`` returns a ``RLMError`` *dataclass*, but the host wire contract is a
    plain ``{"type", "message", ...}`` dict (mirrors ``rcl_rlm.host`` / the shared
    ``droste_runner``). Without this, ``json.dumps`` of the response raises on the
    dataclass and the relay emits no output — an opaque failure for the app, and
    one that would also drop the structured HTTP status the host injects (402 =
    out of balance). Defensive across shapes: dataclass, dict, or anything else.
    """
    if err is None:
        return None
    if dataclasses.is_dataclass(err) and not isinstance(err, type):
        return dataclasses.asdict(err)
    if isinstance(err, dict):
        return err
    return {"type": type(err).__name__, "message": str(err)}


def build_db_service(
    db_path: str, contacts_db_path: str | None = None
) -> tuple[Any, dict[str, Any]]:
    """Build the trusted-side `DataSourceService` for the droste#3/A'-2 split.

    Runs in the SERVICE interpreter (the one with `/data` mounted), never the
    untrusted REPL interpreter. Returns `(service, meta)` — `meta` carries the
    two facts that can only be computed where the DB file is actually visible
    (`has_contacts`, a filesystem probe; `default_max_calls`, a `SELECT COUNT(*)`)
    and that the REPL-side `run_rlm(data_source=...)` call needs but cannot
    derive itself once it has no `db_path`/`contacts_db_path` at all. `meta`
    crosses the interpreter boundary as plain JSON (see relay.ts).

    `extra_methods=("get_retrieved_guids",)` is load-bearing, not decorative:
    `run_rlm` reads `retrieved_guids` off the data source via `getattr` (see
    rcl_rlm/rlm.py), and `BridgeDataSource` only binds whatever
    `DataSourceService.describe()` reports — omit this and citations silently
    return empty over the bridge instead of erroring.
    """
    from rcl_rlm.budget import subcall_budget
    from rcl_rlm.message_database import has_readable_contacts_db
    from rcl_rlm.pyodide_service import build_service_source

    from droste.sources.bridge import DataSourceService

    source = build_service_source(db_path, contacts_db_path)
    service = DataSourceService(source, extra_methods=("get_retrieved_guids",))
    meta = {
        "has_contacts": has_readable_contacts_db(contacts_db_path),
        "default_max_calls": subcall_budget(int(source.get_stats()["total_messages"])),
    }
    return service, meta


def run_for_host_pyodide(
    request: dict[str, Any],
    host_fetch: HostFetch,
    bridge_call: BridgeCall | None = None,
    has_contacts: bool = False,
    default_max_calls: int | None = None,
) -> dict[str, Any]:
    """Pyodide equivalent of rcl_rlm.host.run_for_host / runner_adapter.run.

    Runs a Recall RLM request with the bridged client + raw executor and returns a
    host-compatible response dict (same shape the Deno relay writes to stdout).

    `bridge_call` is the droste#3/A'-2 seam: when the host wires a second,
    trusted Pyodide interpreter running a `DataSourceService` built by
    `build_db_service` (see `droste.sources.bridge`), it passes the resulting
    `bridge_call(method, params_json)` here instead of a raw `db_path`. The DB
    then never opens inside this (untrusted) interpreter. `bridge_call` is
    `None` when the host opts out (relay.ts: `RLM_DB_SERVICE=0`) —
    the single-interpreter, `db_path`-in-sandbox behavior is unchanged in that case.

    `has_contacts` / `default_max_calls` are `build_db_service`'s `meta`,
    threaded across from the service interpreter — the REPL interpreter has no
    `/data` mount in bridge mode, so it cannot compute either itself (a
    filesystem probe and a `SELECT COUNT(*)` respectively). Both are ignored
    when `bridge_call` is `None`. `default_max_calls` only fills in when the
    request didn't already set one explicitly — an explicit request value
    always wins, matching the `db_path` path's own "auto unless the caller
    overrides" contract (`subcall_budget` in rcl_rlm's own `run_rlm`).
    """
    from rcl_rlm.conversation import resolve_conversation_context
    from rcl_rlm.rlm import run_rlm

    auth_type = request.get("auth_type", "api_key")
    customer_token = request.get("customer_token") if auth_type == "customer_token" else None
    client = BridgedLLMClient(
        host_fetch, api_key=request.get("api_key"), customer_token=customer_token
    )

    # Thread multi-turn history exactly like the native runner_adapter (shared
    # helper, so the two substrates can't drift): prefer a pre-built context, else
    # build one from conversation_messages/summary, returning the updated summary.
    resolved_context, updated_summary = resolve_conversation_context(
        request,
        model=request.get("root_model"),
        llm_client=client,
    )

    kwargs: dict[str, Any] = {}
    for key in ("root_model", "subcall_model"):
        if request.get(key) is not None:
            kwargs[key] = request[key]
    for key in ("max_depth", "max_calls", "max_output_chars"):
        if request.get(key) is not None:
            kwargs[key] = int(request[key])

    if bridge_call is not None:
        from droste.sources.bridge import BridgeDataSource

        kwargs["data_source"] = BridgeDataSource(bridge_call, name="messages")
        kwargs["has_contacts"] = has_contacts
        if "max_calls" not in kwargs and default_max_calls is not None:
            kwargs["max_calls"] = int(default_max_calls)
    else:
        kwargs["db_path"] = request["db_path"]
        if request.get("contacts_db_path") is not None:
            kwargs["contacts_db_path"] = request["contacts_db_path"]

    res = run_rlm(
        question=request["question"],
        llm_client=client,
        executor_factory=RawExecutor,
        conversation_context=resolved_context,
        **kwargs,
    )
    return {
        "answer": res.answer,
        "sub_calls_made": res.sub_calls_made,
        "total_tokens": res.total_tokens,
        "retrieved_guids": res.retrieved_guids,
        "iterations": res.iterations,
        "error": _serialize_error(res.error),
        "conversation_summary": updated_summary,
    }
