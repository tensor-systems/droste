"""Native ModelRelay clients for the logged-in path.

`droste login` stores a ModelRelay API key; these clients run the loop against
ModelRelay's native `POST /responses` — the same endpoint the platform's hosted
runner ultimately talks to — with no OpenAI-compat shim in between.

Two classes, one per protocol, mirroring the BYOK pair in ``openai_compat``:

- ``ModelRelayClient`` implements ``LLMClient`` for the root loop
  (``responses_create`` with ``return_usage``); streams NDJSON when an
  ``on_delta`` callback is supplied (the CLI's ``--verbose``).
- ``ModelRelaySubcallClient`` implements ``SubcallClient`` with the same
  reporting semantics as ``OpenAICompatSubcallClient``: bounded batch
  concurrency and typed per-request usage returned to the capability broker,
  which owns stats accounting and authorization.

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
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn

from ..exceptions import BatchItemError, BatchItemErrorDetails
from ..execution.budget import DEFAULT_SUBCALL_OUTPUT_TOKENS
from ..execution.config import validate_subcall_concurrency
from ..execution.context import ExecutionContext
from ..protocols.llm_client import (
    LLMUsageFailure,
    TokenUsage,
    strip_cache_anchor_markers,
    token_usage_from_mapping,
)
from ..protocols.subcall_client import (
    SubcallBatchFailure,
    SubcallBatchResult,
    SubcallClient,
    SubcallQueryResult,
    fail_fast_subcall_batch,
    structured_subcall_errors,
)
from .errors import http_error_excerpt, redact_secrets
from .openai_compat import (
    DEFAULT_MAX_PARALLEL,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_BATCH_PROMPTS,
)
from .useragent import USER_AGENT

DEFAULT_MODELRELAY_BASE_URL = "https://api.modelrelay.ai/api/v1"
_STREAM_ACCEPT = 'application/x-ndjson; profile="responses-stream/v2"'


@dataclass(frozen=True, slots=True)
class _NativeBatchResult:
    results: tuple[str, ...]
    errors: tuple[Exception | None, ...]
    usage: tuple[TokenUsage, ...]


def _batch_detail_string(value: Any) -> str | None:
    """Accept one string field; the public value owns redaction and bounds."""

    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _first_batch_detail_string(*values: Any) -> str | None:
    for value in values:
        if accepted := _batch_detail_string(value):
            return accepted
    return None


def _batch_item_error_details(
    batch: dict[str, Any],
    item: dict[str, Any],
    error: dict[str, Any],
) -> BatchItemErrorDetails:
    """Project a provider error onto the explicit public scalar allowlist."""

    raw_status = error.get("status_code", error.get("status"))
    status_code = (
        raw_status
        if isinstance(raw_status, int)
        and not isinstance(raw_status, bool)
        and 100 <= raw_status <= 599
        else None
    )
    retryable = error.get("retryable")
    if not isinstance(retryable, bool):
        retryable = None
    return BatchItemErrorDetails(
        request_id=_first_batch_detail_string(error.get("request_id"), item.get("request_id")),
        batch_id=_first_batch_detail_string(
            error.get("batch_id"), batch.get("batch_id"), batch.get("id")
        ),
        item_id=_first_batch_detail_string(error.get("item_id"), item.get("id")),
        layer=_batch_detail_string(error.get("layer")),
        cause=_batch_detail_string(error.get("cause")),
        status_code=status_code,
        code=_batch_detail_string(error.get("code")),
        retryable=retryable,
    )


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
    return token_usage_from_mapping(usage)


class _ResponsesTransport:
    """POST {base}/responses with bounded, redacted error surfacing."""

    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str,
        timeout: float,
        label: str,
        on_dispatch: Callable[[], None] | None = None,
    ) -> None:
        base = str(base_url or DEFAULT_MODELRELAY_BASE_URL).rstrip("/")
        self._base_url = base
        self._url = base + "/responses"
        self._api_key = str(api_key or "")
        self._timeout = float(timeout)
        self._label = label
        self._on_dispatch = on_dispatch

    @property
    def url(self) -> str:
        return self._url

    def _request(
        self,
        payload: dict[str, Any],
        *,
        accept: str | None = None,
        url: str | None = None,
    ) -> Any:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if accept:
            headers["Accept"] = accept
        if self._api_key:
            # The API separates API keys from OAuth bearer tokens: mr_sk_*
            # keys go in X-ModelRelay-Api-Key; Authorization: Bearer
            # mr_sk_... is rejected outright.
            headers["X-ModelRelay-Api-Key"] = self._api_key
        return urllib.request.Request(url or self._url, data=body, headers=headers, method="POST")

    def complete(
        self,
        payload: dict[str, Any],
        *,
        path: str = "/responses",
        label: str | None = None,
    ) -> dict[str, Any]:
        request_label = label or self._label
        req = self._request(payload, url=self._base_url + path)
        if self._on_dispatch is not None:
            self._on_dispatch()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status = getattr(exc, "code", 0)
            excerpt = http_error_excerpt(exc)
            detail = f": {excerpt}" if excerpt else f": {exc}"
            raise RuntimeError(f"{request_label} failed with HTTP {status}{detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"{request_label} failed: {exc}") from exc
        try:
            data = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"{request_label} returned non-JSON response") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"{request_label} returned a non-object JSON response")
        return data

    def stream(self, payload: dict[str, Any], on_delta: Any) -> dict[str, Any]:
        """POST with the responses-stream/v2 Accept header; assemble the
        NDJSON events into the same response shape ``complete`` returns,
        invoking ``on_delta(text)`` per text delta as it arrives."""
        req = self._request(payload, accept=_STREAM_ACCEPT)
        if self._on_dispatch is not None:
            self._on_dispatch()
        parts: list[str] = []
        completion: dict[str, Any] = {}
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                content_type = str(resp.headers.get("Content-Type") or "")
                if "json" in content_type and "ndjson" not in content_type:
                    # Endpoint ignored the stream Accept and answered plainly.
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                    if not isinstance(data, dict):
                        raise RuntimeError(f"{self._label} returned a non-object JSON response")
                    try:
                        text = _output_text(data)
                    except RuntimeError:
                        # Let the client account any returned usage before it
                        # surfaces the malformed-output error.
                        return data
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

    ``root_requests_issued`` is the cumulative number of HTTP root requests
    dispatched by this client. It includes streamed and non-streamed requests
    that later fail, but excludes failures while validating or serializing a
    request before dispatch.
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
        self._accounting_lock = threading.Lock()
        self._total_usage = TokenUsage(0, 0, 0, exact=True)
        self._root_requests_issued = 0
        self._transport = _ResponsesTransport(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            label="root llm",
            on_dispatch=self._account_root_request,
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

    @property
    def total_usage(self) -> TokenUsage:
        with self._accounting_lock:
            return TokenUsage(
                prompt_tokens=self._total_usage.prompt_tokens,
                completion_tokens=self._total_usage.completion_tokens,
                total_tokens=self._total_usage.total_tokens,
                cache_read_tokens=self._total_usage.cache_read_tokens,
                cache_creation_tokens=self._total_usage.cache_creation_tokens,
                exact=self._total_usage.exact,
            )

    @property
    def root_requests_issued(self) -> int:
        """Number of HTTP root requests dispatched by this client."""
        with self._accounting_lock:
            return self._root_requests_issued

    def _account_root_request(self) -> None:
        with self._accounting_lock:
            self._root_requests_issued += 1

    def _account_usage(self, usage: TokenUsage) -> None:
        with self._accounting_lock:
            self._total_usage = TokenUsage(
                prompt_tokens=self._total_usage.prompt_tokens + usage.prompt_tokens,
                completion_tokens=self._total_usage.completion_tokens + usage.completion_tokens,
                total_tokens=self._total_usage.total_tokens + usage.total_tokens,
                cache_read_tokens=self._total_usage.cache_read_tokens + usage.cache_read_tokens,
                cache_creation_tokens=(
                    self._total_usage.cache_creation_tokens + usage.cache_creation_tokens
                ),
                exact=self._total_usage.exact and usage.exact,
            )

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
            "input": _input_items(strip_cache_anchor_markers(messages)),
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
        # Account a provider-completed response before parsing its output. A
        # malformed output is still billable and must remain visible to hosts.
        usage = _usage_from(data)
        self._account_usage(usage)
        self.last_provider = str(data.get("provider") or "")
        self.last_response_id = str(data.get("id") or "")
        self.last_stop_reason = str(data.get("stop_reason") or "")
        self.last_model = str(data.get("model") or payload["model"])
        try:
            result = _output_text(data)
        except Exception as exc:
            if return_usage:
                raise LLMUsageFailure(usage, exc) from None
            raise
        if return_usage:
            return result, usage
        return result

    def get_model_context_window(self, model: str) -> int | None:
        return None


class ModelRelaySubcallClient(SubcallClient):
    """``SubcallClient`` over ModelRelay's native ``/responses`` endpoint.

    Reporting mirrors ``OpenAICompatSubcallClient`` (and the hosted
    ``HTTPSubcallClient``): issued calls are recorded locally while provider
    usage is returned to the capability broker for stats and budget accounting.
    Batch concurrency stays bounded.

    Cost defaults match ModelRelay's server-side subcall defaults: output
    bounded at ``DEFAULT_SUBCALL_OUTPUT_TOKENS`` and
    ``reasoning_effort="none"``. Both are explicit overrides, not silent.
    """

    def __init__(
        self,
        *,
        model: str,
        context: ExecutionContext,
        base_url: str | None = None,
        api_key: str = "",
        max_output_tokens: int = DEFAULT_SUBCALL_OUTPUT_TOKENS,
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
        resolved_concurrency = validate_subcall_concurrency(max_parallel)
        self._transport = _ResponsesTransport(
            base_url=base_url, api_key=api_key, timeout=timeout, label="llm_query"
        )
        self._model = str(model)
        self._context = context
        self._max_output_tokens = int(max_output_tokens)
        self._temperature = temperature
        self._reasoning_effort = str(reasoning_effort or "")
        self._max_parallel = resolved_concurrency
        self._lock = threading.Lock()
        self._total_usage = TokenUsage(0, 0, 0, exact=True)

    @property
    def output_token_limit(self) -> int | None:
        """Effective maximum output tokens for each subcall, or no limit."""
        return self._max_output_tokens or None

    @property
    def subcall_concurrency(self) -> int:
        """Effective maximum number of in-flight batch items."""
        return self._max_parallel

    @property
    def total_usage(self) -> TokenUsage:
        with self._lock:
            return TokenUsage(
                prompt_tokens=self._total_usage.prompt_tokens,
                completion_tokens=self._total_usage.completion_tokens,
                total_tokens=self._total_usage.total_tokens,
                cache_read_tokens=self._total_usage.cache_read_tokens,
                cache_creation_tokens=self._total_usage.cache_creation_tokens,
                exact=self._total_usage.exact,
            )

    def _increment_calls(self, count: int = 1) -> None:
        with self._lock:
            if count < 0:
                raise ValueError("subcall count must be non-negative")
            self._context.record_subcall_attempts(count)

    def _account_usage(self, usage: TokenUsage) -> None:
        with self._lock:
            self._total_usage = TokenUsage(
                prompt_tokens=self._total_usage.prompt_tokens + usage.prompt_tokens,
                completion_tokens=self._total_usage.completion_tokens + usage.completion_tokens,
                total_tokens=self._total_usage.total_tokens + usage.total_tokens,
                cache_read_tokens=self._total_usage.cache_read_tokens + usage.cache_read_tokens,
                cache_creation_tokens=(
                    self._total_usage.cache_creation_tokens + usage.cache_creation_tokens
                ),
                exact=self._total_usage.exact and usage.exact,
            )

    def _increment_successful_calls(self, count: int = 1) -> None:
        with self._lock:
            self._context.record_subcall_successes(count)

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
            "input": _input_items([{"role": "user", "content": prompt}]),
        }
        if self._max_output_tokens > 0:
            payload["max_output_tokens"] = self._max_output_tokens
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        data = self._transport.complete(payload)
        usage = _usage_from(data)
        self._account_usage(usage)
        try:
            result = _output_text(data)
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
        result = self._run_batch(prompts, contexts)
        return fail_fast_subcall_batch(result.results, result.errors, result.usage)

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

    def _run_batch(self, prompts: list[str], contexts: list[str] | None) -> _NativeBatchResult:
        if contexts is None:
            contexts = [""] * len(prompts)
        if len(contexts) != len(prompts):
            raise ValueError("contexts length must match prompts length")
        results: list[str] = [""] * len(prompts)
        errors: list[Exception | None] = [None] * len(prompts)
        usage = [TokenUsage.unavailable() for _ in prompts]
        if not prompts:
            return _NativeBatchResult(tuple(results), tuple(errors), tuple(usage))
        if len(prompts) > MAX_BATCH_PROMPTS:
            raise ValueError(f"llm_batch prompt count exceeds max {MAX_BATCH_PROMPTS}")

        self._increment_calls(len(prompts))
        requests: list[dict[str, Any]] = []
        for idx, (prompt, ctx) in enumerate(zip(prompts, contexts)):
            if ctx:
                prompt = f"{ctx}\n\n{prompt}"
            item: dict[str, Any] = {
                "id": str(idx),
                "model": self._model,
                "input": _input_items([{"role": "user", "content": prompt}]),
            }
            if self._max_output_tokens > 0:
                item["max_output_tokens"] = self._max_output_tokens
            if self._temperature is not None:
                item["temperature"] = self._temperature
            if self._reasoning_effort:
                item["reasoning_effort"] = self._reasoning_effort
            requests.append(item)

        data = self._transport.complete(
            {
                "requests": requests,
                "options": {"max_concurrent": self._max_parallel, "fail_fast": False},
            },
            path="/responses/batch",
            label="llm_batch",
        )

        def fail_with_partial_usage(error: Exception) -> NoReturn:
            raise SubcallBatchFailure(
                SubcallBatchResult(
                    tuple(results),
                    structured_subcall_errors(tuple(errors)),
                    tuple(usage),
                ),
                error,
            ) from None

        raw_results = data.get("results")
        if not isinstance(raw_results, list):
            fail_with_partial_usage(RuntimeError("llm_batch response missing results"))

        seen: set[int] = set()
        for raw in raw_results:
            if not isinstance(raw, dict):
                fail_with_partial_usage(RuntimeError("llm_batch returned a non-object result"))
            try:
                idx = int(raw.get("id"))
            except (TypeError, ValueError) as exc:
                error = RuntimeError("llm_batch returned an invalid result id")
                error.__cause__ = exc
                fail_with_partial_usage(error)
            if idx < 0 or idx >= len(prompts) or idx in seen:
                fail_with_partial_usage(
                    RuntimeError(f"llm_batch returned unexpected result id {idx}")
                )
            seen.add(idx)
            status = str(raw.get("status") or "")
            if status == "success":
                response = raw.get("response")
                if not isinstance(response, dict):
                    fail_with_partial_usage(RuntimeError(f"llm_batch item {idx} missing response"))
                usage[idx] = _usage_from(response)
                self._account_usage(usage[idx])
                try:
                    results[idx] = _output_text(response)
                except Exception as exc:
                    errors[idx] = exc
                    continue
                if results[idx].strip():
                    self._increment_successful_calls()
            elif status == "error":
                error = raw.get("error")
                if not isinstance(error, dict):
                    fail_with_partial_usage(RuntimeError(f"llm_batch item {idx} missing error"))
                code = str(error.get("code") or "")
                message = redact_secrets(str(error.get("message") or "batch item failed"))[:500]
                detail = f" ({code})" if code else ""
                errors[idx] = BatchItemError(
                    f"llm_batch item {idx} failed{detail}: {message}",
                    _batch_item_error_details(data, raw, error),
                )
            else:
                fail_with_partial_usage(
                    RuntimeError(f"llm_batch item {idx} has invalid status {status!r}")
                )

        if len(seen) != len(prompts):
            missing = sorted(set(range(len(prompts))) - seen)
            fail_with_partial_usage(
                RuntimeError(f"llm_batch response missing result ids {missing}")
            )
        return _NativeBatchResult(tuple(results), tuple(errors), tuple(usage))
