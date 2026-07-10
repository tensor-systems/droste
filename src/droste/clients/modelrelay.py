"""Native ModelRelay clients for the logged-in path.

`droste login` stores a ModelRelay API key; these clients run the loop against
ModelRelay's native `POST /responses` — the same endpoint the platform's hosted
runner ultimately talks to — with no OpenAI-compat shim in between.

Two classes, one per protocol, mirroring the BYOK pair in ``openai_compat``:

- ``ModelRelayClient`` implements ``LLMClient`` for the root loop
  (``responses_create`` with ``return_usage``); streams NDJSON when an
  ``on_delta`` callback is supplied (the CLI's ``--verbose``).
- ``ModelRelaySubcallClient`` implements ``SubcallClient`` with the same
  accounting semantics as ``OpenAICompatSubcallClient``: check-then-increment
  of ``context.stats.calls_made`` under a lock, per-thread depth tracking,
  bounded batch concurrency, and subcall token usage added to
  ``context.stats.total_tokens``.

Subcall economics: subcalls default to ``reasoning_effort="none"`` and a
bounded output budget — the same defaults ModelRelay applies server-side
for its hosted runner (an unbounded thinking subcall can burn tens of
thousands of hidden reasoning tokens to answer in a few words). Pass
``reasoning_effort``/``max_output_tokens`` explicitly to override.

Dependency-free by design: urllib only, like the rest of the engine.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ..execution.context import ExecutionContext
from ..protocols.llm_client import TokenUsage
from ..protocols.subcall_client import SubcallClient
from .errors import http_error_excerpt
from .openai_compat import (
    DEFAULT_MAX_PARALLEL,
    DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_BATCH_PROMPTS,
)

DEFAULT_MODELRELAY_BASE_URL = "https://api.modelrelay.ai/api/v1"
_STREAM_ACCEPT = 'application/x-ndjson; profile="responses-stream/v2"'


def _input_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat-style ``{role, content}`` messages -> /responses input items."""
    items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            parts = content
        else:
            parts = [{"type": "text", "text": str(content or "")}]
        items.append({"type": "message", "role": role, "content": parts})
    return items


def _output_text(data: Any) -> str:
    """Concatenate the text parts of the response's output messages."""
    output = data.get("output") if isinstance(data, dict) else None
    if not isinstance(output, list):
        raise RuntimeError("modelrelay response missing output")
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
    return "".join(parts)


def _usage_from(data: Any) -> TokenUsage:
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    prompt = int(usage.get("input_tokens") or 0)
    completion = int(usage.get("output_tokens") or 0)
    total = usage.get("total_tokens")
    if total is None:
        total = prompt + completion
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=int(total or 0),
    )


class _ResponsesTransport:
    """POST {base}/responses with bounded, redacted error surfacing."""

    def __init__(self, *, base_url: str | None, api_key: str, timeout: float, label: str) -> None:
        base = str(base_url or DEFAULT_MODELRELAY_BASE_URL).rstrip("/")
        self._url = base + "/responses"
        self._api_key = str(api_key or "")
        self._timeout = float(timeout)
        self._label = label

    @property
    def url(self) -> str:
        return self._url

    def _request(self, payload: dict[str, Any], *, accept: str | None = None) -> Any:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if accept:
            headers["Accept"] = accept
        if self._api_key:
            # The API separates API keys from OAuth bearer tokens: mr_sk_*
            # keys go in X-ModelRelay-Api-Key; Authorization: Bearer
            # mr_sk_... is rejected outright.
            headers["X-ModelRelay-Api-Key"] = self._api_key
        return urllib.request.Request(self._url, data=body, headers=headers, method="POST")

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = self._request(payload)
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
        """POST with the responses-stream/v2 Accept header; assemble the
        NDJSON events into the same response shape ``complete`` returns,
        invoking ``on_delta(text)`` per text delta as it arrives."""
        req = self._request(payload, accept=_STREAM_ACCEPT)
        parts: list[str] = []
        completion: dict[str, Any] = {}
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                content_type = str(resp.headers.get("Content-Type") or "")
                if "json" in content_type and "ndjson" not in content_type:
                    # Endpoint ignored the stream Accept and answered plainly.
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                    text = _output_text(data)
                    if text:
                        on_delta(text)
                    return data
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[len("data:") :].strip()
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue  # tolerate keep-alive/partial noise
                    if not isinstance(event, dict):
                        continue
                    event_type = event.get("type")
                    if event_type == "error":
                        # Mid-stream error: fail loudly, never return partial
                        # text as a successful response.
                        message = str(event.get("message") or "stream error")
                        code = event.get("code")
                        detail = f" ({code})" if code else ""
                        raise RuntimeError(
                            f"{self._label} streamed an error{detail}: {message[:500]}"
                        )
                    if event_type == "update":
                        delta = event.get("delta")
                        if delta:
                            parts.append(str(delta))
                            on_delta(str(delta))
                    elif event_type == "completion":
                        completion = event
        except urllib.error.HTTPError as exc:
            status = getattr(exc, "code", 0)
            excerpt = http_error_excerpt(exc)
            detail = f": {excerpt}" if excerpt else f": {exc}"
            raise RuntimeError(f"{self._label} failed with HTTP {status}{detail}") from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"{self._label} failed: {exc}") from exc
        if not completion:
            # A dropped connection or truncated stream must never surface
            # partial generated code as a successful response.
            raise RuntimeError(f"{self._label} stream ended without a completion event")
        text = str(completion.get("content") or "") or "".join(parts)
        assembled: dict[str, Any] = {
            "id": str(completion.get("request_id") or ""),
            "model": str(completion.get("model") or payload.get("model") or ""),
            "provider": str(completion.get("provider") or ""),
            "stop_reason": str(completion.get("stop_reason") or "stop"),
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                }
            ],
        }
        if isinstance(completion.get("usage"), dict):
            assembled["usage"] = completion["usage"]
        return assembled


class ModelRelayClient:
    """Root ``LLMClient`` over ModelRelay's native ``/responses`` endpoint.

    Same contract as ``OpenAICompatClient``: constructor
    ``temperature``/``stop``/``max_output_tokens`` override per-call
    arguments; ``on_delta`` (optional) streams text fragments as they
    generate while the call still returns the full assembled text.
    """

    def __init__(
        self,
        *,
        model: str = "",
        base_url: str | None = None,
        api_key: str = "",
        temperature: float | None = None,
        stop: list[str] | None = None,
        max_output_tokens: int = 0,
        reasoning_effort: str = "",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        on_delta: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required (run `droste login`)")
        self._transport = _ResponsesTransport(
            base_url=base_url, api_key=api_key, timeout=timeout, label="root llm"
        )
        self._model = str(model or "")
        self._temperature = temperature
        self._stop = list(stop) if stop else []
        self._max_output_tokens = int(max_output_tokens or 0)
        self._reasoning_effort = str(reasoning_effort or "")
        self._on_delta = on_delta
        # Parity with RootLLMClient's response metadata surface.
        self.last_provider = ""
        self.last_response_id = ""
        self.last_stop_reason = ""
        self.last_model = ""

    def _payload(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float | None,
    ) -> dict[str, Any]:
        resolved_model = model or self._model
        if not resolved_model:
            raise ValueError("model is required")
        payload: dict[str, Any] = {
            "model": resolved_model,
            "input": _input_items(messages),
        }
        max_output_tokens = self._max_output_tokens or int(max_tokens or 0)
        if max_output_tokens > 0:
            payload["max_output_tokens"] = max_output_tokens
        temp = self._temperature if self._temperature is not None else temperature
        if temp is not None:
            payload["temperature"] = temp
        if self._stop:
            payload["stop"] = self._stop
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        return payload

    def responses_create(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
        return_usage: bool = False,
    ) -> str | tuple[str, TokenUsage]:
        payload = self._payload(messages, model, max_tokens, temperature)
        if self._on_delta is not None:
            data = self._transport.stream(payload, self._on_delta)
        else:
            data = self._transport.complete(payload)
        result = _output_text(data)
        self.last_provider = str(data.get("provider") or "")
        self.last_response_id = str(data.get("id") or "")
        self.last_stop_reason = str(data.get("stop_reason") or "")
        self.last_model = str(data.get("model") or payload["model"])
        if return_usage:
            return result, _usage_from(data)
        return result

    def batch_responses(self, requests: list[dict[str, Any]]) -> list[str]:
        results: list[str] = []
        for request in requests:
            raw_max = request.get("max_tokens")
            # Preserve an explicit 0 (opt-out); only a missing value defaults.
            max_tokens = 4096 if raw_max in (None, "") else int(raw_max)
            response = self.responses_create(
                request.get("messages") or [],
                model=str(request.get("model") or ""),
                max_tokens=max_tokens,
                temperature=(
                    float(request["temperature"])
                    if request.get("temperature") is not None
                    else None
                ),
            )
            results.append(str(response))
        return results

    def get_model_context_window(self, model: str) -> int | None:
        return None


class ModelRelaySubcallClient(SubcallClient):
    """``SubcallClient`` over ModelRelay's native ``/responses`` endpoint.

    Accounting mirrors ``OpenAICompatSubcallClient`` (and the hosted
    ``HTTPSubcallClient``): check-then-increment of
    ``context.stats.calls_made`` under a lock, per-thread depth tracking
    against ``max_depth``, bounded batch concurrency, and subcall usage
    added to ``context.stats.total_tokens``.

    Cost defaults match ModelRelay's server-side subcall defaults: output
    bounded at ``DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS`` and
    ``reasoning_effort="none"``. Both are explicit overrides, not silent.
    """

    def __init__(
        self,
        *,
        model: str,
        context: ExecutionContext,
        base_url: str | None = None,
        api_key: str = "",
        max_calls: int | None = None,
        max_depth: int | None = None,
        max_output_tokens: int = DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS,
        temperature: float | None = None,
        reasoning_effort: str = "none",
        max_parallel: int = DEFAULT_MAX_PARALLEL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        if not api_key:
            raise ValueError("api_key is required (run `droste login`)")
        if max_output_tokens < 0:
            raise ValueError("max_output_tokens must be >= 0 (0 disables the bound)")
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        self._transport = _ResponsesTransport(
            base_url=base_url, api_key=api_key, timeout=timeout, label="llm_query"
        )
        self._model = str(model)
        self._context = context
        self._max_calls = int(context.max_calls if max_calls is None else max_calls)
        self._max_depth = int(context.max_depth if max_depth is None else max_depth)
        self._max_output_tokens = int(max_output_tokens)
        self._temperature = temperature
        self._reasoning_effort = str(reasoning_effort or "")
        self._max_parallel = int(max_parallel)
        self._lock = threading.Lock()
        self._depth = threading.local()

    def _depth_get(self) -> int:
        return getattr(self._depth, "value", 0)

    def _depth_set(self, value: int) -> None:
        self._depth.value = value

    def _increment_calls(self) -> None:
        with self._lock:
            if self._max_calls >= 0 and self._context.stats.calls_made >= self._max_calls:
                raise RuntimeError("max subcalls exceeded")
            self._context.stats.calls_made += 1

    def _account_usage(self, usage: TokenUsage) -> None:
        with self._lock:
            self._context.stats.total_tokens += usage.total_tokens

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
                "input": _input_items([{"role": "user", "content": prompt}]),
            }
            if self._max_output_tokens > 0:
                payload["max_output_tokens"] = self._max_output_tokens
            if self._temperature is not None:
                payload["temperature"] = self._temperature
            if self._reasoning_effort:
                payload["reasoning_effort"] = self._reasoning_effort
            data = self._transport.complete(payload)
            result = _output_text(data)
            self._account_usage(_usage_from(data))
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
            {"index": idx, "error": str(err)} for idx, err in enumerate(errors) if err is not None
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
