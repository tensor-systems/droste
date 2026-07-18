"""HTTP-backed root and subcall clients for droste_runner."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from droste.clients.errors import http_error_excerpt, redact_secrets
from droste.clients.useragent import USER_AGENT
from droste.execution.config import DEFAULT_SUBCALL_CONCURRENCY, validate_subcall_concurrency
from droste.protocols.llm_client import LLMUsageFailure, TokenUsage, strip_cache_anchor_markers
from droste.protocols.subcall_capacity import SubcallInputCapacity
from droste.protocols.subcall_client import (
    SubcallBatchFailure,
    SubcallBatchResult,
    SubcallClient,
    SubcallQueryResult,
    fail_fast_subcall_batch,
    structured_subcall_errors,
)

from .protocol import RootResponseMetadata

# The bounded-read + redaction HTTP-error helpers moved to droste.clients.errors
# so the BYOK OpenAI-compatible client shares them. Aliased here because
# this module's callers (and its tests) know them by the underscored names.
_redact_secrets = redact_secrets
_http_error_excerpt = http_error_excerpt

_SUBCALL_STREAM_ACCEPT = 'application/x-ndjson; profile="responses-stream/v2"'


def _token_usage(payload: object) -> TokenUsage:
    if not isinstance(payload, dict):
        return TokenUsage.unavailable()

    def token_count(name: str) -> int | None:
        value = payload.get(name)
        return (
            value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
        )

    input_tokens = token_count("input_tokens")
    output_tokens = token_count("output_tokens")
    total_tokens = token_count("total_tokens")
    if (
        input_tokens is None
        or output_tokens is None
        or total_tokens is None
        or total_tokens < input_tokens + output_tokens
    ):
        return TokenUsage.unavailable()
    return TokenUsage(input_tokens, output_tokens, total_tokens, exact=True)


@dataclass(frozen=True, slots=True)
class _HTTPBatchResult:
    results: tuple[str, ...]
    errors: tuple[Exception | None, ...]
    usage: tuple[TokenUsage, ...]


class HTTPSubcallClient(SubcallClient):
    def __init__(
        self,
        *,
        endpoint: str,
        token: str,
        session: str,
        session_index: int,
        context: Any,
        max_output_tokens: int = 0,
        model: str = "",
        reasoning_effort: str = "",
        max_parallel: int = DEFAULT_SUBCALL_CONCURRENCY,
        input_capacity: SubcallInputCapacity | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._token = token
        self._session = session
        self._session_index = int(session_index or 0)
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._context = context
        # Subcall cost controls: included in each subcall payload when
        # set; omitted when unset so the server owns the defaults (bounded
        # output + no thinking).
        self._max_output_tokens = int(max_output_tokens or 0)
        self._model = str(model or "")
        self._reasoning_effort = str(reasoning_effort or "")
        self._max_parallel = validate_subcall_concurrency(max_parallel)
        self._input_capacity = input_capacity or SubcallInputCapacity.unknown()

    @property
    def subcall_concurrency(self) -> int:
        """Effective maximum number of in-flight batch items."""
        return self._max_parallel

    @property
    def output_token_limit(self) -> int:
        """Explicit per-call limit, when the runner controls it."""
        if self._max_output_tokens <= 0:
            raise AttributeError("the callback owns the output token default")
        return self._max_output_tokens

    @property
    def input_token_capacity(self) -> SubcallInputCapacity:
        """Effective input capacity when the runner request declares it."""
        return self._input_capacity

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def _increment_calls(self) -> None:
        # Check-then-increment under the lock: the count is the reported
        # subcall total, so a rejected over-limit attempt must not inflate it,
        # and concurrent llm_batch threads must not race the check.
        with self._seq_lock:
            self._context.record_subcall_attempts()

    def _increment_successful_calls(self) -> None:
        with self._seq_lock:
            self._context.record_subcall_successes()

    def _record_usage(self, payload: object) -> TokenUsage:
        return _token_usage(payload)

    def _request(self, payload: dict[str, Any]) -> SubcallQueryResult:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={
                "Authorization": "Bearer " + self._token,
                "Accept": _SUBCALL_STREAM_ACCEPT,
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                content_type = str(resp.headers.get("Content-Type") or "")
                if "application/x-ndjson" not in content_type.lower():
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                    usage = self._record_usage(
                        data.get("usage") if isinstance(data, dict) else None
                    )
                    result = data.get("result") if isinstance(data, dict) else None
                    if not isinstance(result, str):
                        raise LLMUsageFailure(usage, RuntimeError("missing subcall result"))
                    return SubcallQueryResult(result, usage)

                parts: list[str] = []
                completed = False
                completion_has_content = False
                completion_content = ""
                completion_usage: object = None
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("llm_query returned malformed stream data") from exc
                    if not isinstance(event, dict):
                        raise RuntimeError("llm_query returned a non-object stream event")
                    event_type = event.get("type")
                    if event_type == "error":
                        message = str(event.get("message") or "stream error")
                        code = str(event.get("code") or "")
                        detail = f" ({code})" if code else ""
                        raise RuntimeError(f"llm_query streamed an error{detail}: {message[:500]}")
                    if event_type == "update":
                        delta = event.get("delta")
                        if delta is not None:
                            parts.append(str(delta))
                    elif event_type == "completion":
                        completed = True
                        completion_usage = event.get("usage")
                        if isinstance(event.get("content"), str):
                            completion_has_content = True
                            completion_content = event["content"]
                if not completed:
                    raise RuntimeError("llm_query stream ended without a completion event")
                usage = self._record_usage(completion_usage)
                return SubcallQueryResult(
                    completion_content if completion_has_content else "".join(parts),
                    usage,
                )
        except urllib.error.HTTPError as exc:
            status = getattr(exc, "code", 0)
            excerpt = _http_error_excerpt(exc)
            detail = f": {excerpt}" if excerpt else f": {exc}"
            raise RuntimeError(f"llm_query failed with HTTP {status}{detail}") from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"llm_query failed: {exc}") from exc

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
            "prompt": prompt,
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
        value = self._request(payload)
        if value.result.strip():
            self._increment_successful_calls()
        return value

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

    def _run_batch(self, prompts: list[str], contexts: list[str] | None) -> _HTTPBatchResult:
        if contexts is None:
            contexts = [""] * len(prompts)
        if len(contexts) != len(prompts):
            raise ValueError("contexts length must match prompts length")
        results: list[str] = [""] * len(prompts)
        errors: list[Exception | None] = [None] * len(prompts)
        usage = [TokenUsage.unavailable() for _ in prompts]
        if not prompts:
            return _HTTPBatchResult(tuple(results), tuple(errors), tuple(usage))
        if len(prompts) > 50:
            raise ValueError("llm_batch prompt count exceeds max 50")

        def _run_one(idx: int, prompt: str, ctx: str) -> SubcallQueryResult:
            if idx > 0:
                time.sleep(0.05)
            return self.llm_query_with_usage(prompt, ctx)

        from concurrent.futures import ThreadPoolExecutor, as_completed

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
        return _HTTPBatchResult(tuple(results), tuple(errors), tuple(usage))


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
        reasoning_effort: str = "",
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
        self._reasoning_effort = str(reasoning_effort or "")
        self.response_metadata = RootResponseMetadata()

    @property
    def last_provider(self) -> str:
        return self.response_metadata.provider

    @property
    def last_response_id(self) -> str:
        return self.response_metadata.response_id

    @property
    def last_stop_reason(self) -> str:
        return self.response_metadata.stop_reason

    @property
    def last_model(self) -> str:
        return self.response_metadata.model

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
            "messages": strip_cache_anchor_markers(messages),
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
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={
                "Authorization": "Bearer " + self._token,
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
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
        usage = _token_usage(data.get("usage") if isinstance(data, dict) else None)
        result = data.get("result") if isinstance(data, dict) else None
        if not isinstance(result, str):
            error = RuntimeError("missing root result")
            if return_usage:
                raise LLMUsageFailure(usage, error) from None
            raise error
        self.response_metadata = RootResponseMetadata(
            provider=str(data.get("provider") or ""),
            response_id=str(data.get("response_id") or ""),
            stop_reason=str(data.get("stop_reason") or ""),
            model=str(data.get("model") or ""),
        )
        if return_usage:
            return result, usage
        return result
