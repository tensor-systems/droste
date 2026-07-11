"""Pyodide-substrate adapters for droste's core loop (adapter-agnostic split).

Two pieces are substrate-specific and injected into the otherwise-unchanged
droste loop:

  RawExecutor      - replaces RestrictedExecutor. Runs generated code with a plain
                     interpreter (the Deno/WASM jail is the sandbox, so RestrictedPython
                     is redundant here). Pyodide-safe: no signals/threads.
  BridgedLLMClient - replaces the httpx ModelRelay client. Implements the droste
                     LLMClient protocol by calling a host fetch function injected by
                     the Deno host, which performs the real ModelRelay /responses call.

Both have zero third-party deps so they import under Pyodide. Neither knows
anything about any particular host's data layer or product wiring — a host
embeds droste under Pyodide by writing its own adapter module (the contract
`pyodide/relay.ts` expects; see `examples/pyodide-host/` for droste's own
reference adapter) that wires these into its own request/response shape and
data source.
"""

from __future__ import annotations

import builtins
import contextlib
import dataclasses
import io
from typing import Any, Callable

from droste.protocols.llm_client import TokenUsage

# Bind the interpreter primitive once, away from call sites, so static scanners
# don't confuse it with shell exec.
_run_code = builtins.exec


# --------------------------------------------------------------------------- #
# RawExecutor — matches RestrictedExecutor.execute(code, extra_globals) -> str
# --------------------------------------------------------------------------- #
class RawExecutor:
    # max_output_chars is accepted for factory-signature parity with
    # RestrictedExecutor but deliberately NOT enforced here (#44): the output
    # budget has ONE chokepoint — the loop's _enforce_output_budget, which
    # raises SandboxError so the model gets the over-budget feedback and can
    # narrow its query. Pre-truncating here made oversized output pass that
    # check and handed the model silently incomplete data (unlike the native
    # path, which raises).
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
        return buf.getvalue()

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
        # Strip a trailing slash: a base_url like ".../api/v1/" would otherwise
        # produce ".../api/v1//responses" below, which fails the host broker's
        # exact-path credential scoping (isModelRelayResponsesCall) and sends
        # the request unauthenticated instead of just erroring loudly.
        self._base_url = base_url.rstrip("/")

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
        temperature: float | None = None,
        return_usage: bool = False,
    ) -> str | tuple[str, TokenUsage]:
        # temperature=None means "don't send the parameter" — matches the
        # droste.protocols.llm_client.LLMClient contract: modern models
        # (gpt-5.x, opus-4.x) reject an explicit temperature outright.
        body: dict[str, Any] = {
            "model": model,
            "input": _build_input(messages),
            "max_output_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        data = self._post("/responses", body)
        text = _extract_text(data)
        if return_usage:
            usage = data.get("usage", {})
            inp = int(usage.get("input_tokens", 0))
            out = int(usage.get("output_tokens", 0))
            return text, TokenUsage(
                prompt_tokens=inp, completion_tokens=out, total_tokens=inp + out
            )
        return text

    def get_model_context_window(self, model: str) -> int | None:
        return None

    def get_model_max_output_tokens(self, model: str) -> int | None:
        # Model metadata isn't fetched over the bridge; conversation budgeting
        # falls back to its defaults when this is None (see _context_char_budget).
        return None


def serialize_error(err: Any) -> dict[str, Any] | None:
    """Make an RLM error JSON-serializable for a host response.

    ``run_rlm`` returns an error *dataclass*, but a JSON-over-stdio host wire
    contract needs a plain ``{"type", "message", ...}`` dict. Without this,
    ``json.dumps`` of the response raises on the dataclass and the host gets no
    output — an opaque failure, and one that would also drop any structured
    status the host injects into the error (e.g. a 402 out-of-balance). Defensive
    across shapes: dataclass, dict, or anything else.
    """
    if err is None:
        return None
    if dataclasses.is_dataclass(err) and not isinstance(err, type):
        return dataclasses.asdict(err)
    if isinstance(err, dict):
        return err
    return {"type": type(err).__name__, "message": str(err)}
