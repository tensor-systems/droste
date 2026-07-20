"""HTTP-backed root and subcall clients for droste_runner."""

from __future__ import annotations

import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from droste.clients.errors import (
    HTTPErrorBody,
    http_error_body_excerpt,
    http_error_excerpt,
    read_http_error_body,
    redact_secrets,
)
from droste.clients.useragent import USER_AGENT
from droste.execution.config import DEFAULT_SUBCALL_CONCURRENCY, validate_subcall_concurrency
from droste.protocols.llm_client import (
    LLMUsageFailure,
    TokenUsage,
    strip_cache_anchor_markers,
    token_usage_from_mapping,
)
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
_MAX_SIGNED_INT64 = 2**63 - 1
_MIN_SIGNED_INT64 = -(2**63)
_MIN_SIGNED_INT64_DIGITS = "9223372036854775808"
_MAX_SIGNED_INT64_DIGITS = "9223372036854775807"
_MAX_CONTENT_TYPE_FIELD_CHARS = 512
_MAX_CALLBACK_JSON_DEPTH = 256
_MAX_CALLBACK_JSON_FLOAT_CHARS = 256
_HTTP_TOKEN = r"[!#$%&'*+.^_`|~0-9a-z-]+"
_JSON_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)", re.ASCII)
_JSON_CONTENT_TYPE = re.compile(
    rf"[ \t]*application/(?:json|{_HTTP_TOKEN}\+json)[ \t]*"
    r'(?:;[ \t]*charset[ \t]*=[ \t]*(?:utf-8|"utf-8")[ \t]*)?',
    re.ASCII | re.IGNORECASE,
)
_RUNNER_USAGE_COUNTERS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_write_input_tokens",
        "cache_creation_input_tokens",
        "reasoning_tokens",
    }
)


@dataclass(frozen=True, slots=True)
class _InvalidJSONInteger:
    """A syntactically valid JSON integer outside the signed-int64 contract."""


_INVALID_JSON_INTEGER = _InvalidJSONInteger()


def _token_usage(payload: object) -> TokenUsage:
    if isinstance(payload, dict):
        payload = {
            key: None
            if key in _RUNNER_USAGE_COUNTERS
            and isinstance(value, int)
            and not isinstance(value, bool)
            and not (_MIN_SIGNED_INT64 <= value <= _MAX_SIGNED_INT64)
            else value
            for key, value in payload.items()
        }
    return token_usage_from_mapping(
        payload,
        observation_basis_name="observation_basis",
        require_reasoning=True,
    )


def _has_json_media_type(exc: urllib.error.HTTPError) -> bool:
    """Return whether one response Content-Type declares a JSON media type."""

    try:
        headers = getattr(exc, "headers", None) or getattr(exc, "hdrs", None)
        if headers is None:
            return False
        get_all = getattr(headers, "get_all", None)
        if callable(get_all):
            raw_values = get_all("Content-Type") or []
        else:
            raw = headers.get("Content-Type")
            raw_values = [] if raw is None else [raw]
        if not isinstance(raw_values, (list, tuple)) or len(raw_values) != 1:
            return False
        raw_value = raw_values[0]
        if (
            not isinstance(raw_value, str)
            or len(raw_value) > _MAX_CONTENT_TYPE_FIELD_CHARS
            or not raw_value.isascii()
        ):
            return False
        return _JSON_CONTENT_TYPE.fullmatch(raw_value) is not None
    except Exception:
        return False


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _parse_json_integer(value: str) -> int | _InvalidJSONInteger:
    if _JSON_INTEGER.fullmatch(value) is None:
        raise ValueError("invalid JSON integer lexeme")
    negative = value.startswith("-")
    digits = value[1:] if negative else value
    if len(digits) > len(_MAX_SIGNED_INT64_DIGITS):
        return _INVALID_JSON_INTEGER
    if len(digits) == len(_MAX_SIGNED_INT64_DIGITS):
        limit = _MIN_SIGNED_INT64_DIGITS if negative else _MAX_SIGNED_INT64_DIGITS
        if digits > limit:
            return _INVALID_JSON_INTEGER
    # Bounded to 19 digits above; this never reaches Python's configurable
    # unbounded-integer conversion limit.
    return int(value)


def _parse_json_float(value: str) -> float:
    # parse_constant does not see exponent overflow: Python's default decoder
    # turns a valid JSON lexeme such as 1e10000 into infinity. Bound the work
    # before conversion, then reject every non-finite binary result.
    if len(value) > _MAX_CALLBACK_JSON_FLOAT_CHARS:
        raise ValueError("JSON float lexeme exceeds maximum length")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON float must be finite")
    return parsed


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _load_callback_json(body: bytes) -> object:
    """Load any runner callback JSON under the one strict wire contract."""

    text = body.decode("utf-8", errors="strict")
    if text.startswith("\ufeff"):
        raise ValueError("callback JSON must not contain a byte-order mark")
    value = json.loads(
        text,
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
        parse_float=_parse_json_float,
        parse_int=_parse_json_integer,
    )
    pending: list[tuple[object, int]] = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > _MAX_CALLBACK_JSON_DEPTH:
            raise ValueError("callback JSON exceeds maximum depth")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
    return value


def _contains_invalid_json_integer(value: object) -> bool:
    pending = [value]
    while pending:
        item = pending.pop()
        if item is _INVALID_JSON_INTEGER:
            return True
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return False


def _has_invalid_integer_outside_usage_counters(payload: dict[str, Any]) -> bool:
    for key, value in payload.items():
        if key != "usage" and _contains_invalid_json_integer(value):
            return True
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return _contains_invalid_json_integer(usage)
    for key, value in usage.items():
        if value is _INVALID_JSON_INTEGER and key in _RUNNER_USAGE_COUNTERS:
            continue
        if _contains_invalid_json_integer(value):
            return True
    return False


def _load_callback_object_with_usage(
    body: bytes,
    operation: str,
) -> tuple[dict[str, Any], TokenUsage]:
    try:
        value = _load_callback_json(body)
    except (UnicodeError, ValueError, RecursionError, OverflowError) as exc:
        raise RuntimeError(f"{operation} returned malformed JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{operation} returned a non-object JSON envelope")
    usage = _token_usage(value.get("usage"))
    if _has_invalid_integer_outside_usage_counters(value):
        raise LLMUsageFailure(
            usage,
            RuntimeError(
                f"{operation} returned an out-of-range integer outside a recognized usage counter"
            ),
        )
    return value, usage


def _callback_http_failure(
    exc: urllib.error.HTTPError,
    operation: str,
) -> tuple[RuntimeError, TokenUsage | None]:
    """Build one bounded callback error and preserve its structured usage field."""

    captured: HTTPErrorBody = read_http_error_body(exc)
    status = getattr(exc, "code", 0)
    excerpt = http_error_body_excerpt(captured.body)
    detail = f": {excerpt}" if excerpt else f": {exc}"
    cause = RuntimeError(f"{operation} failed with HTTP {status}{detail}")
    if not captured.complete or not captured.body or not _has_json_media_type(exc):
        return cause, None
    try:
        payload = _load_callback_json(captured.body)
    except (UnicodeError, ValueError, RecursionError, OverflowError):
        return cause, None
    if (
        not isinstance(payload, dict)
        or payload.get("error") != "api_error"
        or not isinstance(payload.get("usage"), dict)
        or _has_invalid_integer_outside_usage_counters(payload)
    ):
        return cause, None
    return cause, _token_usage(payload["usage"])


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
                    data, usage = _load_callback_object_with_usage(
                        resp.read(),
                        "llm_query",
                    )
                    result = data.get("result")
                    if not isinstance(result, str):
                        raise LLMUsageFailure(usage, RuntimeError("missing subcall result"))
                    return SubcallQueryResult(result, usage)

                parts: list[str] = []
                completed = False
                completion_has_content = False
                completion_content = ""
                completion_usage: object = None
                for raw_line in resp:
                    if not raw_line.strip(b" \t\r\n"):
                        continue
                    try:
                        event = _load_callback_json(raw_line)
                    except (UnicodeError, ValueError, RecursionError, OverflowError) as exc:
                        raise RuntimeError("llm_query returned malformed stream data") from exc
                    if not isinstance(event, dict):
                        raise RuntimeError("llm_query returned a non-object stream event")
                    event_type = event.get("type")
                    if _has_invalid_integer_outside_usage_counters(event):
                        cause = RuntimeError(
                            "llm_query stream event contained an out-of-range integer "
                            "outside a recognized usage counter"
                        )
                        if event_type in ("error", "completion") and isinstance(
                            event.get("usage"), dict
                        ):
                            raise LLMUsageFailure(self._record_usage(event["usage"]), cause)
                        raise cause
                    if event_type == "error":
                        message = _redact_secrets(str(event.get("message") or "stream error"))[:500]
                        code = _redact_secrets(str(event.get("code") or ""))[:100]
                        detail = f" ({code})" if code else ""
                        cause = RuntimeError(f"llm_query streamed an error{detail}: {message}")
                        if "usage" in event:
                            raise LLMUsageFailure(self._record_usage(event["usage"]), cause)
                        raise cause
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
            cause, usage = _callback_http_failure(exc, "llm_query")
            if usage is not None:
                raise LLMUsageFailure(usage, cause) from None
            raise cause from exc
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
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            cause, usage = _callback_http_failure(exc, "root llm")
            if return_usage and usage is not None:
                raise LLMUsageFailure(usage, cause) from None
            raise cause from exc
        except Exception as exc:
            raise RuntimeError(f"root llm failed: {exc}") from exc
        try:
            data, usage = _load_callback_object_with_usage(raw, "root llm")
        except LLMUsageFailure as exc:
            if return_usage:
                raise
            raise exc.cause from exc
        result = data.get("result")
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
