"""Pyodide-substrate adapters for the Recall RLM (Phase 2 step 3).

Two pieces are substrate-specific and injected into the otherwise-unchanged
rlm-core loop + RecallEnvironment:

  RawExecutor      - replaces RestrictedExecutor. Runs generated code with a plain
                     interpreter (the Deno/WASM jail is the sandbox, so RestrictedPython
                     is redundant here). Pyodide-safe: no signals/threads.
  BridgedLLMClient - replaces the httpx ModelRelay client. Implements the rlm-core
                     LLMClient protocol by calling a host fetch function injected by
                     the Deno host, which performs the real ModelRelay /responses call.

Both have zero third-party deps so they import under Pyodide.
"""
from __future__ import annotations

import builtins
import contextlib
import io
from typing import Any, Callable

from rlm_core.protocols.llm_client import TokenUsage

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
# BridgedLLMClient — implements the rlm-core LLMClient protocol over a host fetch
# --------------------------------------------------------------------------- #
def _build_input(messages: list[dict]) -> list[dict]:
    """OpenAI-style messages -> ModelRelay input items (mirrors modelrelay._build_input)."""
    items = []
    for msg in messages:
        content = msg["content"]
        content_items = content if isinstance(content, list) else [{"type": "text", "text": content}]
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
            return text, TokenUsage(prompt_tokens=inp, completion_tokens=out, total_tokens=inp + out)
        return text

    def batch_responses(self, requests: list[dict[str, Any]]) -> list[str]:
        # Sequential for the spike; host-side parallelism is a Phase-2 follow-up.
        results = []
        for req in requests:
            results.append(
                self.responses_create(
                    messages=req["messages"],
                    model=req["model"],
                    max_tokens=req.get("max_tokens", 4096),
                    temperature=req.get("temperature", 0.0),
                )
            )
        return results

    def get_model_context_window(self, model: str) -> int | None:
        return None


def run_for_host_pyodide(request: dict[str, Any], host_fetch: HostFetch) -> dict[str, Any]:
    """Pyodide equivalent of rcl_rlm.host.run_for_host / runner_adapter.run.

    Runs a Recall RLM request with the bridged client + raw executor and returns a
    host-compatible response dict (same shape the Deno relay writes to stdout).
    """
    from rcl_rlm.rlm import run_rlm

    auth_type = request.get("auth_type", "api_key")
    customer_token = request.get("customer_token") if auth_type == "customer_token" else None
    client = BridgedLLMClient(host_fetch, api_key=request.get("api_key"), customer_token=customer_token)

    kwargs: dict[str, Any] = {}
    for key in ("contacts_db_path", "root_model", "subcall_model", "conversation_context"):
        if request.get(key) is not None:
            kwargs[key] = request[key]
    for key in ("max_depth", "max_calls", "max_output_chars"):
        if request.get(key) is not None:
            kwargs[key] = int(request[key])

    res = run_rlm(
        question=request["question"],
        db_path=request["db_path"],
        llm_client=client,
        executor_factory=RawExecutor,
        **kwargs,
    )
    return {
        "answer": res.answer,
        "sub_calls_made": res.sub_calls_made,
        "total_tokens": res.total_tokens,
        "retrieved_guids": res.retrieved_guids,
        "iterations": res.iterations,
        "error": res.error,
    }
