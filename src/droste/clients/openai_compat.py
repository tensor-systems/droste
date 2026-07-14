"""BYOK client for any OpenAI-compatible chat-completions endpoint.

The engine's HTTP clients in droste_runner speak ModelRelay's root/subcall
protocol only. This module lets an OSS user run the loop against any endpoint
that speaks the OpenAI chat-completions shape (OpenAI, OpenRouter, Google's
OpenAI-compat endpoint, vLLM, Ollama, ...) with just base_url + api_key +
model — no ModelRelay required.

Two classes, one per protocol:

- ``OpenAICompatClient`` implements ``LLMClient`` for the root loop
  (``responses_create`` with ``return_usage``).
- ``OpenAICompatSubcallClient`` implements ``SubcallClient`` (``llm_query`` /
  ``llm_batch`` / ``llm_batch_with_errors``) with bounded concurrency, a
  bounded per-subcall output budget (default 2048 tokens), and call/token
  accounting into the shared ``ExecutionContext`` — mirroring
  ``HTTPSubcallClient``'s counting semantics (count only issued calls, honor
  ``max_calls``, thread-safe).

Config resolution: explicit constructor args win, then ``OPENAI_API_KEY`` /
``OPENAI_BASE_URL``, then the OpenAI default base URL. An empty api_key is
allowed (local endpoints like Ollama/vLLM need none); the Authorization
header is simply omitted.

Honest note on reasoning/thinking control: ``reasoning_effort`` and
``extra_body`` are passed through as-is. Server-side thinking control (e.g.
disabling Gemini thinking per subcall) is a gateway capability — on
ModelRelay these knobs are enforced server-side; BYOK gets whatever the raw
endpoint honors (providers do not reliably honor client-side thinking disables).

Dependency-free by design: urllib only, like the rest of the engine.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ..exceptions import SubcallBudgetExceeded
from ..execution.context import ExecutionContext
from ..protocols.llm_client import TokenUsage
from ..protocols.subcall_client import SubcallClient
from .errors import http_error_excerpt
from .useragent import USER_AGENT

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS = 2048
DEFAULT_MAX_PARALLEL = 5
MAX_BATCH_PROMPTS = 50
DEFAULT_TIMEOUT_SECONDS = 120.0


def resolve_base_url(base_url: str | None = None) -> str:
    resolved = base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    return str(resolved).rstrip("/")


def resolve_api_key(api_key: str | None = None) -> str:
    if api_key is not None:
        return api_key
    return os.environ.get("OPENAI_API_KEY", "")


def _message_content(data: Any, *, label: str) -> str:
    """Extract choices[0].message.content, tolerating list-of-parts content
    (some compat servers return [{"type": "text", "text": ...}, ...])."""
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError(f"{label} response missing choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"{label} response missing choices[0].message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        ]
        return "".join(parts)
    if content is None:
        # An endpoint that answered with tool_calls (rather than text) is not
        # something this client supports — it sends no tools — so surface it
        # instead of silently returning an empty string.
        if message.get("tool_calls"):
            raise RuntimeError(
                f"{label} response returned tool_calls but no text content; "
                "the OpenAI-compatible client does not request or support tools"
            )
        return ""
    raise RuntimeError(f"{label} response has non-text content of type {type(content).__name__}")


def _usage_from(data: Any) -> TokenUsage:
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total = usage.get("total_tokens")
    if total is None:
        total = prompt + completion
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=int(total or 0),
    )


class _ChatCompletionsTransport:
    """POST /chat/completions with bounded, redacted error surfacing."""

    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str | None,
        timeout: float,
        label: str,
    ) -> None:
        self._url = resolve_base_url(base_url) + "/chat/completions"
        self._api_key = resolve_api_key(api_key)
        self._timeout = float(timeout)
        self._label = label

    @property
    def url(self) -> str:
        return self._url

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        req = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status = getattr(exc, "code", 0)
            excerpt = http_error_excerpt(exc)
            detail = f": {excerpt}" if excerpt else f": {exc}"
            raise RuntimeError(f"{self._label} failed with HTTP {status}{detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"{self._label} failed: {exc}") from exc
        try:
            data = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"{self._label} returned non-JSON response") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"{self._label} returned a non-object JSON response")
        return data

    def stream(self, payload: dict[str, Any], on_delta: Any) -> dict[str, Any]:
        """POST with ``stream: true``; assemble SSE chunks into the same
        response shape ``complete`` returns, invoking ``on_delta(text)`` per
        content fragment as it arrives.

        ``stream_options.include_usage`` is requested; endpoints that honor it
        (OpenAI, vLLM, Ollama, Google's compat endpoint) report usage in the
        final chunk. If the endpoint sends none, usage is absent from the
        assembled response — the caller's usage parser treats that as zeros
        rather than failing the run.
        """
        payload = dict(payload)
        payload["stream"] = True
        payload.setdefault("stream_options", {"include_usage": True})
        try:
            return self._stream_once(payload, on_delta)
        except RuntimeError as exc:
            # Some compat endpoints (older vLLM, thin proxies) reject unknown
            # stream_options with a 400; usage-in-stream is best-effort, so
            # retry once without it rather than failing verbose streaming.
            if "HTTP 400" in str(exc) and "stream_options" in payload:
                retry = dict(payload)
                retry.pop("stream_options", None)
                return self._stream_once(retry, on_delta)
            raise

    def _stream_once(self, payload: dict[str, Any], on_delta: Any) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": USER_AGENT,
        }
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        req = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
        parts: list[str] = []
        finish_reason = ""
        response_id = ""
        model = ""
        usage: dict[str, Any] | None = None
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue  # tolerate keep-alive/partial noise
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("error"):
                        # Mid-stream provider error: fail loudly, never return
                        # partial text as a successful response.
                        raise RuntimeError(
                            f"{self._label} streamed an error: {json.dumps(chunk['error'])[:500]}"
                        )
                    response_id = str(chunk.get("id") or response_id)
                    model = str(chunk.get("model") or model)
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        if text:
                            parts.append(text)
                            on_delta(text)
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])
        except urllib.error.HTTPError as exc:
            status = getattr(exc, "code", 0)
            excerpt = http_error_excerpt(exc)
            detail = f": {excerpt}" if excerpt else f": {exc}"
            raise RuntimeError(f"{self._label} failed with HTTP {status}{detail}") from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"{self._label} failed: {exc}") from exc
        assembled: dict[str, Any] = {
            "id": response_id,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(parts)},
                    "finish_reason": finish_reason or "stop",
                }
            ],
        }
        if usage is not None:
            assembled["usage"] = usage
        return assembled


def _token_param_for(param_state: dict[str, str], model: str) -> str:
    return param_state.get(model, "max_tokens")


def _complete_with_token_param_migration(
    transport: "_ChatCompletionsTransport",
    payload: dict[str, Any],
    param_state: dict[str, str],
    model: str,
) -> dict[str, Any]:
    """POST via ``transport`` applying the modern-OpenAI token-param rule.

    Modern OpenAI models 400 on ``max_tokens`` ("use max_completion_tokens");
    the wider compat ecosystem (Google, Ollama, vLLM) speaks ``max_tokens``.
    On that specific 400 the param migrates and ``param_state`` remembers it
    **per model** (one client can serve mixed models across concurrent
    subcall workers; a modern model must not poison max_tokens-only models
    on the same endpoint). ``extra_body`` wins: a caller-set max_completion_tokens
    survives. The retry decision keys off the PAYLOAD, not the state —
    concurrent batch workers race the state flip.
    """
    try:
        return transport.complete(payload)
    except RuntimeError as exc:
        if "max_completion_tokens" in str(exc) and "max_tokens" in payload:
            param_state[model] = "max_completion_tokens"
            moved = payload.pop("max_tokens")
            payload.setdefault("max_completion_tokens", moved)
            return transport.complete(payload)
        raise


class OpenAICompatClient:
    """Root ``LLMClient`` over an OpenAI-compatible chat-completions endpoint.

    ``model`` set here is the default; a non-empty ``model`` argument to
    ``responses_create`` (e.g. ``RLMConfig.root_model``) wins. Constructor
    ``temperature``/``stop``/``max_output_tokens`` override the per-call
    arguments when set, matching ``RootLLMClient``'s precedence. ``extra_body``
    entries are merged into every request payload last (they win), providing a
    passthrough for provider-specific params (``reasoning_effort``,
    ``reasoning``, ...).

    ``on_delta`` (optional) streams the response: content fragments are
    delivered to the callback as they generate (SSE) and the call still
    returns the full assembled text — the ``LLMClient`` protocol is
    unchanged. Used by the CLI's ``--verbose`` to show code as it is
    written.
    """

    def __init__(
        self,
        *,
        model: str = "",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        max_output_tokens: int = 0,
        extra_body: dict[str, Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        on_delta: Any | None = None,
    ) -> None:
        self._transport = _ChatCompletionsTransport(
            base_url=base_url, api_key=api_key, timeout=timeout, label="root llm"
        )
        self._model = str(model or "")
        self._temperature = temperature
        self._stop = list(stop) if stop else []
        self._max_output_tokens = int(max_output_tokens or 0)
        self._extra_body = dict(extra_body) if extra_body else {}
        self._on_delta = on_delta
        self._token_param: dict[str, str] = {}  # per-model migration memory
        # Parity with RootLLMClient's response metadata surface.
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
        resolved_model = model or self._model
        if not resolved_model:
            raise ValueError("model is required")
        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
        }
        # Only send temperature when explicitly set — modern models
        # (gpt-5.x, opus-4.x) reject the parameter outright.
        temp = self._temperature if self._temperature is not None else temperature
        if temp is not None:
            payload["temperature"] = temp
        max_output_tokens = self._max_output_tokens or int(max_tokens or 0)
        if max_output_tokens > 0:
            payload[_token_param_for(self._token_param, resolved_model)] = max_output_tokens
        if self._stop:
            payload["stop"] = self._stop
        payload.update(self._extra_body)
        # Streaming is a concrete-client concern, not a protocol one: with an
        # on_delta callback the request streams (SSE) and fragments surface as
        # they generate; the assembled response is shape-identical, so the
        # loop never knows the difference.
        if self._on_delta is not None:
            try:
                data = self._transport.stream(payload, self._on_delta)
            except RuntimeError as exc:
                # Same token-param migration as the non-streaming path (see
                # _complete_with_token_param_migration).
                if "max_completion_tokens" in str(exc) and "max_tokens" in payload:
                    self._token_param[resolved_model] = "max_completion_tokens"
                    moved = payload.pop("max_tokens")
                    payload.setdefault("max_completion_tokens", moved)
                    data = self._transport.stream(payload, self._on_delta)
                else:
                    raise
        else:
            data = _complete_with_token_param_migration(
                self._transport, payload, self._token_param, resolved_model
            )
        result = _message_content(data, label="root llm")
        choice = data["choices"][0]
        self.last_provider = str(data.get("provider") or "")
        self.last_response_id = str(data.get("id") or "")
        self.last_stop_reason = str(choice.get("finish_reason") or "")
        self.last_model = str(data.get("model") or resolved_model)
        if return_usage:
            return result, _usage_from(data)
        return result

    def get_model_context_window(self, model: str) -> int | None:
        return None


class OpenAICompatSubcallClient(SubcallClient):
    """``SubcallClient`` over an OpenAI-compatible chat-completions endpoint.

    Mirrors ``HTTPSubcallClient``'s accounting semantics: check-then-increment
    of ``context.stats.calls_made`` under a lock (a rejected over-limit attempt
    must not inflate the count, and concurrent llm_batch threads must not race
    the check), per-thread depth tracking against ``max_depth``, bounded batch
    concurrency. Unlike the ModelRelay transport — where the server owns token
    accounting — subcall token usage here is read from the response's standard
    usage block and added to ``context.stats.total_tokens``, so
    ``RLMResult.tokens_used`` covers root + subcall tokens.

    ``max_output_tokens`` bounds every subcall's output (cost control; default
    2048). Pass 0 to leave the endpoint's default unbounded — deliberate
    opt-out, not the default.
    """

    def __init__(
        self,
        *,
        model: str,
        context: ExecutionContext,
        base_url: str | None = None,
        api_key: str | None = None,
        max_calls: int | None = None,
        max_depth: int | None = None,
        max_output_tokens: int = DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS,
        temperature: float | None = None,
        reasoning_effort: str = "",
        extra_body: dict[str, Any] | None = None,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        if max_output_tokens < 0:
            raise ValueError("max_output_tokens must be >= 0 (0 disables the bound)")
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        self._transport = _ChatCompletionsTransport(
            base_url=base_url, api_key=api_key, timeout=timeout, label="llm_query"
        )
        self._model = str(model)
        self._context = context
        self._max_calls = int(context.max_calls if max_calls is None else max_calls)
        self._max_depth = int(context.max_depth if max_depth is None else max_depth)
        self._max_output_tokens = int(max_output_tokens)
        self._temperature = temperature
        self._reasoning_effort = str(reasoning_effort or "")
        self._extra_body = dict(extra_body) if extra_body else {}
        self._token_param: dict[str, str] = {}  # per-model migration memory
        self._max_parallel = int(max_parallel)
        self._lock = threading.Lock()
        self._depth = threading.local()

    @property
    def output_token_limit(self) -> int | None:
        """Effective maximum output tokens for each subcall, or no limit."""
        return self._max_output_tokens or None

    def _depth_get(self) -> int:
        return getattr(self._depth, "value", 0)

    def _depth_set(self, value: int) -> None:
        self._depth.value = value

    def _increment_calls(self) -> None:
        with self._lock:
            if self._max_calls >= 0 and self._context.stats.calls_made >= self._max_calls:
                raise SubcallBudgetExceeded("max subcalls exceeded")
            self._context.record_subcall_attempts()

    def _account_usage(self, usage: TokenUsage) -> None:
        with self._lock:
            self._context.record_subcall_usage(usage)

    def _increment_successful_calls(self) -> None:
        with self._lock:
            self._context.record_subcall_successes()

    def llm_query(self, prompt: str, context: str = "") -> str:
        if context:
            prompt = f"{context}\n\n{prompt}"
        depth = self._depth_get() + 1
        self._depth_set(depth)
        try:
            if self._max_depth >= 0 and depth > self._max_depth:
                raise RuntimeError("max depth exceeded")
            self._increment_calls()
            payload: dict[str, Any] = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if self._max_output_tokens > 0:
                payload[_token_param_for(self._token_param, self._model)] = self._max_output_tokens
            if self._temperature is not None:
                payload["temperature"] = self._temperature
            if self._reasoning_effort:
                payload["reasoning_effort"] = self._reasoning_effort
            payload.update(self._extra_body)
            data = _complete_with_token_param_migration(
                self._transport, payload, self._token_param, self._model
            )
            result = _message_content(data, label="llm_query")
            self._account_usage(_usage_from(data))
            if result.strip():
                self._increment_successful_calls()
            return result
        finally:
            self._depth_set(depth - 1)

    def llm_batch(self, prompts: list[str], contexts: list[str] | None = None) -> list[str]:
        results, errors = self._run_batch(prompts, contexts)
        for err in errors:
            if err is not None:
                raise err
        return results

    def llm_batch_with_errors(
        self,
        prompts: list[str],
        contexts: list[str] | None = None,
    ) -> tuple[list[str], list[dict[str, object]]]:
        results, errors = self._run_batch(prompts, contexts)
        structured = [
            {
                "index": idx,
                "error": str(err),
                **({"type": "budget_exhausted"} if isinstance(err, SubcallBudgetExceeded) else {}),
            }
            for idx, err in enumerate(errors)
            if err is not None
        ]
        return results, structured

    def _run_batch(
        self, prompts: list[str], contexts: list[str] | None
    ) -> tuple[list[str], list[Exception | None]]:
        if contexts is None:
            contexts = [""] * len(prompts)
        if len(contexts) != len(prompts):
            raise ValueError("contexts length must match prompts length")
        results: list[str] = [""] * len(prompts)
        errors: list[Exception | None] = [None] * len(prompts)
        if not prompts:
            return results, errors
        if len(prompts) > MAX_BATCH_PROMPTS:
            raise ValueError(f"llm_batch prompt count exceeds max {MAX_BATCH_PROMPTS}")

        def _run_one(idx: int, prompt: str, ctx: str) -> str:
            if idx > 0:
                time.sleep(0.05)
            return self.llm_query(prompt, ctx)

        with ThreadPoolExecutor(max_workers=self._max_parallel) as executor:
            futures = {
                executor.submit(_run_one, idx, prompt, ctx): idx
                for idx, (prompt, ctx) in enumerate(zip(prompts, contexts))
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    errors[idx] = exc
        return results, errors
