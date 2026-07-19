"""BYOK client for Anthropic's native Messages API.

Anthropic's API is not OpenAI-compatible (its compat layer is documented as a
testing shim), so first-class Claude support gets its own client pair speaking
``POST /v1/messages`` directly:

- ``AnthropicClient`` implements ``LLMClient`` for the root loop
  (``responses_create`` with ``return_usage``), including ``on_delta`` SSE
  streaming for the CLI's --verbose view.
- ``AnthropicSubcallClient`` implements ``SubcallClient`` with the same
  reporting semantics as ``OpenAICompatSubcallClient``: bounded batch
  concurrency and typed per-request usage returned to the capability broker,
  which owns stats accounting and authorization.

API notes encoded here:
- headers are ``x-api-key`` + ``anthropic-version`` (not Authorization);
- ``max_tokens`` is REQUIRED by the API — a positive value is always sent;
- the system prompt is a top-level ``system`` field, so ``role: system``
  messages are lifted out of the message list;
- ``stop`` maps to ``stop_sequences``;
- temperature is only sent when explicitly configured (same rule as the
  OpenAI-compat client — never inject a synthetic default);
- thinking control is passthrough-only via ``extra_body`` (e.g.
  ``{"thinking": {...}}``).

Config resolution: explicit constructor args win, then ``ANTHROPIC_API_KEY`` /
``ANTHROPIC_BASE_URL``, then the public default base URL.

Dependency-free by design: urllib only, like the rest of the engine.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from ..execution.budget import DEFAULT_SUBCALL_OUTPUT_TOKENS
from ..execution.config import DEFAULT_SUBCALL_CONCURRENCY, validate_subcall_concurrency
from ..execution.context import ExecutionContext
from ..protocols.llm_client import (
    CACHE_ANCHOR_MARKER,
    LLMUsageFailure,
    TokenUsage,
    strip_cache_anchor_markers,
)
from ..protocols.subcall_client import (
    SubcallBatchFailure,
    SubcallBatchResult,
    SubcallClient,
    SubcallQueryResult,
    fail_fast_subcall_batch,
    structured_subcall_errors,
)
from .errors import http_error_excerpt
from .useragent import USER_AGENT

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MAX_TOKENS = 4096
DEFAULT_MAX_PARALLEL = DEFAULT_SUBCALL_CONCURRENCY
MAX_BATCH_PROMPTS = 50
DEFAULT_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class _AnthropicBatchResult:
    results: tuple[str, ...]
    errors: tuple[Exception | None, ...]
    usage: tuple[TokenUsage, ...]


#: Anthropic key prefix — the fact the CLI's provider detection keys off.
ANTHROPIC_KEY_PREFIX = "sk-ant-"

logger = logging.getLogger(__name__)


def resolve_anthropic_base_url(base_url: str | None = None) -> str:
    resolved = base_url or os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_ANTHROPIC_BASE_URL
    return str(resolved).rstrip("/")


def resolve_anthropic_api_key(api_key: str | None = None) -> str:
    if api_key is not None:
        return api_key
    return os.environ.get("ANTHROPIC_API_KEY", "")


def _clean_content_blocks(content: Any) -> Any:
    """Shallow-copy content blocks and remove caller-side cache controls."""
    if not isinstance(content, list):
        return content
    return [
        {key: value for key, value in block.items() if key != "cache_control"}
        if isinstance(block, dict)
        else block
        for block in content
    ]


def _mark_content(content: Any) -> tuple[Any, bool]:
    """Attach one ephemeral breakpoint to content when its shape permits."""
    content = _clean_content_blocks(content)
    if isinstance(content, str) and content:
        return (
            [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            True,
        )
    if isinstance(content, list):
        for index in range(len(content) - 1, -1, -1):
            block = content[index]
            if isinstance(block, dict):
                content[index] = {
                    **block,
                    "cache_control": {"type": "ephemeral"},
                }
                return content, True
    logger.warning(
        "Anthropic cache anchor was not applied: unsupported or empty content shape (%s)",
        type(content).__name__,
    )
    return content, False


def _system_blocks(content: Any) -> list[Any]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        return content
    return []


def _lift_system(
    messages: list[dict[str, Any]],
) -> tuple[str | list[Any], list[dict[str, Any]]]:
    """Split ``role: system`` messages out into the top-level ``system`` field.

    The runner passes the system prompt as ``messages[0]``; Anthropic rejects
    a system role inside ``messages``. Multiple system messages concatenate.
    """
    system_contents: list[Any] = []
    system_uses_blocks = False
    rest: list[dict[str, Any]] = []
    cache_controls = 0
    clean_messages = strip_cache_anchor_markers(messages)
    for message, outbound in zip(messages, clean_messages, strict=True):
        anchored = bool(message.get(CACHE_ANCHOR_MARKER))
        content = outbound.get("content")
        marked = False
        if anchored and cache_controls < 4:
            content, marked = _mark_content(content)
            cache_controls += int(marked)
        else:
            content = _clean_content_blocks(content)
        outbound["content"] = content
        if str(message.get("role", "")).lower() == "system":
            system_contents.append(content)
            system_uses_blocks = system_uses_blocks or marked or isinstance(content, list)
        else:
            rest.append(outbound)
    if not system_uses_blocks:
        return "\n\n".join(
            content for content in system_contents if isinstance(content, str) and content
        ), rest

    system_blocks: list[Any] = []
    for content in system_contents:
        blocks = _system_blocks(content)
        if blocks:
            if system_blocks:
                system_blocks.append({"type": "text", "text": "\n\n"})
            system_blocks.extend(blocks)
    return system_blocks, rest


def _text_from_content(data: Any, *, label: str) -> str:
    """Concatenate the text blocks of a Messages API response."""
    content = data.get("content") if isinstance(data, dict) else None
    if not isinstance(content, list):
        raise RuntimeError(f"{label} response missing content blocks")
    parts = [
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts)


def _usage_from(data: Any) -> TokenUsage:
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return TokenUsage.unavailable()

    def count(name: str, *, optional: bool = False) -> tuple[int, bool]:
        if optional and name not in usage:
            return 0, True
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value, True
        return 0, False

    ordinary_input, ordinary_complete = count("input_tokens")
    cache_creation, cache_creation_complete = count("cache_creation_input_tokens", optional=True)
    cache_read, cache_read_complete = count("cache_read_input_tokens", optional=True)
    completion, completion_complete = count("output_tokens")
    prompt = ordinary_input + cache_creation + cache_read
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        exact=(
            ordinary_complete
            and cache_creation_complete
            and cache_read_complete
            and completion_complete
        ),
    )


class _MessagesTransport:
    """POST /v1/messages with bounded, redacted error surfacing."""

    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str | None,
        timeout: float,
        label: str,
    ) -> None:
        self._url = resolve_anthropic_base_url(base_url) + "/v1/messages"
        self._api_key = resolve_anthropic_api_key(api_key)
        self._timeout = float(timeout)
        self._label = label

    @property
    def url(self) -> str:
        return self._url

    def _request(
        self, payload: dict[str, Any], *, accept: str | None = None
    ) -> urllib.request.Request:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "User-Agent": USER_AGENT,
        }
        if accept:
            headers["Accept"] = accept
        if self._api_key:
            headers["x-api-key"] = self._api_key
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
        """POST with ``stream: true``; assemble the Messages SSE events into
        the same response shape ``complete`` returns, invoking
        ``on_delta(text)`` per text fragment as it arrives.

        Event handling: ``message_start`` carries id/model/input usage,
        ``content_block_delta`` carries ``text_delta`` fragments,
        ``message_delta`` carries stop_reason + output usage, ``error``
        events raise (never return partial text as success), ``ping`` and
        unknown events are ignored.
        """
        payload = dict(payload)
        payload["stream"] = True
        req = self._request(payload, accept="text/event-stream")
        parts: list[str] = []
        response_id = ""
        model = ""
        stop_reason = ""
        input_usage: dict[str, Any] | None = None
        output_tokens: Any = None
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    try:
                        event = json.loads(data_str)
                    except Exception:
                        continue  # tolerate keep-alive/partial noise
                    if not isinstance(event, dict):
                        continue
                    etype = event.get("type")
                    if etype == "error":
                        # Mid-stream provider error: fail loudly, never return
                        # partial text as a successful response.
                        raise RuntimeError(
                            f"{self._label} streamed an error: "
                            f"{json.dumps(event.get('error'))[:500]}"
                        )
                    if etype == "message_start":
                        message = event.get("message") or {}
                        response_id = str(message.get("id") or response_id)
                        model = str(message.get("model") or model)
                        usage = message.get("usage")
                        input_usage = dict(usage) if isinstance(usage, dict) else None
                        if input_usage is not None:
                            # Anthropic's message_start output count is only a
                            # preliminary value. Terminal output usage belongs
                            # exclusively to message_delta.
                            input_usage.pop("output_tokens", None)
                    elif etype == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            text = delta.get("text")
                            if text:
                                parts.append(text)
                                on_delta(text)
                    elif etype == "message_delta":
                        delta = event.get("delta") or {}
                        if delta.get("stop_reason"):
                            stop_reason = str(delta["stop_reason"])
                        usage = event.get("usage")
                        if isinstance(usage, dict) and "output_tokens" in usage:
                            output_tokens = usage["output_tokens"]
                    elif etype == "message_stop":
                        break
        except urllib.error.HTTPError as exc:
            status = getattr(exc, "code", 0)
            excerpt = http_error_excerpt(exc)
            detail = f": {excerpt}" if excerpt else f": {exc}"
            raise RuntimeError(f"{self._label} failed with HTTP {status}{detail}") from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"{self._label} failed: {exc}") from exc
        resolved_usage = dict(input_usage) if input_usage is not None else {}
        if output_tokens is not None:
            resolved_usage["output_tokens"] = output_tokens
        return {
            "id": response_id,
            "model": model,
            "content": [{"type": "text", "text": "".join(parts)}],
            "stop_reason": stop_reason or "end_turn",
            "usage": resolved_usage,
        }


class AnthropicClient:
    """Root ``LLMClient`` over Anthropic's native Messages API.

    Mirrors ``OpenAICompatClient``'s surface: ``model`` set here is the
    default, a non-empty per-call ``model`` wins; constructor
    ``temperature``/``stop``/``max_output_tokens`` override per-call
    arguments; ``extra_body`` merges into every payload last (thinking
    control passes through here). ``on_delta`` streams text fragments as
    they generate while the call still returns the assembled text.

    ``max_tokens`` is required by the API: when neither the constructor nor
    the call provides a positive value, ``DEFAULT_ANTHROPIC_MAX_TOKENS``
    is sent.
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
        self._transport = _MessagesTransport(
            base_url=base_url, api_key=api_key, timeout=timeout, label="root llm"
        )
        self._model = str(model or "")
        self._temperature = temperature
        self._stop = list(stop) if stop else []
        self._max_output_tokens = int(max_output_tokens or 0)
        self._extra_body = dict(extra_body) if extra_body else {}
        self._on_delta = on_delta
        # Parity with the other root clients' response metadata surface.
        self.last_provider = "anthropic"
        self.last_response_id = ""
        self.last_stop_reason = ""
        self.last_model = ""

    def responses_create(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS,
        temperature: float | None = None,
        return_usage: bool = False,
    ) -> str | tuple[str, TokenUsage]:
        resolved_model = model or self._model
        if not resolved_model:
            raise ValueError("model is required")
        system, rest = _lift_system(messages)
        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": rest,
            # Required by the API — always positive.
            "max_tokens": self._max_output_tokens
            or int(max_tokens or 0)
            or DEFAULT_ANTHROPIC_MAX_TOKENS,
        }
        if system:
            payload["system"] = system
        # Only send temperature when explicitly set — never a synthetic default.
        temp = self._temperature if self._temperature is not None else temperature
        if temp is not None:
            payload["temperature"] = temp
        if self._stop:
            payload["stop_sequences"] = self._stop
        payload.update(self._extra_body)
        if self._on_delta is not None:
            data = self._transport.stream(payload, self._on_delta)
        else:
            data = self._transport.complete(payload)
        usage = _usage_from(data)
        try:
            result = _text_from_content(data, label="root llm")
        except Exception as exc:
            if return_usage:
                raise LLMUsageFailure(usage, exc) from None
            raise
        self.last_response_id = str(data.get("id") or "")
        self.last_stop_reason = str(data.get("stop_reason") or "")
        self.last_model = str(data.get("model") or resolved_model)
        if return_usage:
            return result, usage
        return result

    def get_model_context_window(self, model: str) -> int | None:
        return None


class AnthropicSubcallClient(SubcallClient):
    """``SubcallClient`` over Anthropic's native Messages API.

    Reports issued calls under a lock and bounds batch concurrency. Provider
    usage is returned to the capability broker, which owns stats accounting
    and budget authorization.

    ``max_output_tokens`` bounds every subcall's output (cost control;
    default 2048). The Messages API requires ``max_tokens``, so 0 is not a
    valid opt-out here — a positive bound is always sent.
    """

    def __init__(
        self,
        *,
        model: str,
        context: ExecutionContext,
        base_url: str | None = None,
        api_key: str | None = None,
        max_output_tokens: int = DEFAULT_SUBCALL_OUTPUT_TOKENS,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be > 0 (the Messages API requires max_tokens)")
        resolved_concurrency = validate_subcall_concurrency(max_parallel)
        self._transport = _MessagesTransport(
            base_url=base_url, api_key=api_key, timeout=timeout, label="llm_query"
        )
        self._model = str(model)
        self._context = context
        self._max_output_tokens = int(max_output_tokens)
        self._temperature = temperature
        self._extra_body = dict(extra_body) if extra_body else {}
        self._max_parallel = resolved_concurrency
        self._lock = threading.Lock()

    @property
    def output_token_limit(self) -> int | None:
        """Effective maximum output tokens for each subcall."""
        return self._max_output_tokens

    @property
    def subcall_concurrency(self) -> int:
        """Effective maximum number of in-flight batch items."""
        return self._max_parallel

    def _increment_calls(self) -> None:
        with self._lock:
            self._context.record_subcall_attempts()

    def _increment_successful_calls(self) -> None:
        with self._lock:
            self._context.record_subcall_successes()

    def llm_query(self, prompt: str, context: str = "") -> str:
        try:
            return self.llm_query_with_usage(prompt, context).result
        except LLMUsageFailure as exc:
            raise exc.cause from exc

    def llm_query_with_usage(self, prompt: str, context: str = "") -> SubcallQueryResult:
        if context:
            prompt = f"{context}\n\n{prompt}"
        self._increment_calls()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_output_tokens,
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        payload.update(self._extra_body)
        data = self._transport.complete(payload)
        usage = _usage_from(data)
        try:
            result = _text_from_content(data, label="llm_query")
        except Exception as exc:
            raise LLMUsageFailure(usage, exc) from None
        if result.strip():
            self._increment_successful_calls()
        return SubcallQueryResult(result, usage)

    def llm_batch(self, prompts: list[str], contexts: list[str] | None = None) -> list[str]:
        try:
            return list(self.llm_batch_with_usage(prompts, contexts).results)
        except SubcallBatchFailure as exc:
            raise exc.cause from exc

    def llm_batch_with_usage(
        self, prompts: list[str], contexts: list[str] | None = None
    ) -> SubcallBatchResult:
        value = self._run_batch(prompts, contexts)
        return fail_fast_subcall_batch(value.results, value.errors, value.usage)

    def llm_batch_with_errors(
        self,
        prompts: list[str],
        contexts: list[str] | None = None,
    ) -> tuple[list[str], list[dict[str, object]]]:
        try:
            value = self.llm_batch_with_errors_and_usage(prompts, contexts)
        except SubcallBatchFailure as exc:
            raise exc.cause from exc
        return list(value.results), [dict(item) for item in value.errors]

    def llm_batch_with_errors_and_usage(
        self,
        prompts: list[str],
        contexts: list[str] | None = None,
    ) -> SubcallBatchResult:
        value = self._run_batch(prompts, contexts)
        return SubcallBatchResult(
            value.results,
            structured_subcall_errors(value.errors),
            value.usage,
        )

    def _run_batch(self, prompts: list[str], contexts: list[str] | None) -> _AnthropicBatchResult:
        if contexts is None:
            contexts = [""] * len(prompts)
        if len(contexts) != len(prompts):
            raise ValueError("contexts length must match prompts length")
        results: list[str] = [""] * len(prompts)
        errors: list[Exception | None] = [None] * len(prompts)
        usage = [TokenUsage.unavailable() for _ in prompts]
        if not prompts:
            return _AnthropicBatchResult(tuple(results), tuple(errors), tuple(usage))
        if len(prompts) > MAX_BATCH_PROMPTS:
            raise ValueError(f"llm_batch prompt count exceeds max {MAX_BATCH_PROMPTS}")

        def _run_one(idx: int, prompt: str, ctx: str) -> SubcallQueryResult:
            if idx > 0:
                time.sleep(0.05)
            return self.llm_query_with_usage(prompt, ctx)

        with ThreadPoolExecutor(max_workers=self._max_parallel) as executor:
            futures = {
                executor.submit(_run_one, idx, prompt, ctx): idx
                for idx, (prompt, ctx) in enumerate(zip(prompts, contexts))
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    value = future.result()
                    results[idx] = value.result
                    usage[idx] = value.usage
                except LLMUsageFailure as exc:
                    usage[idx] = exc.usage
                    errors[idx] = exc.cause
                except Exception as exc:
                    errors[idx] = exc
        return _AnthropicBatchResult(tuple(results), tuple(errors), tuple(usage))
