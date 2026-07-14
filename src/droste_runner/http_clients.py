"""HTTP-backed root and subcall clients for droste_runner."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from droste.clients.errors import http_error_excerpt, redact_secrets
from droste.clients.useragent import USER_AGENT
from droste.exceptions import SubcallBudgetExceeded
from droste.protocols.llm_client import TokenUsage
from droste.protocols.subcall_client import SubcallClient

from .protocol import RootResponseMetadata

# The bounded-read + redaction HTTP-error helpers moved to droste.clients.errors
# so the BYOK OpenAI-compatible client shares them. Aliased here because
# this module's callers (and its tests) know them by the underscored names.
_redact_secrets = redact_secrets
_http_error_excerpt = http_error_excerpt

_SUBCALL_STREAM_ACCEPT = 'application/x-ndjson; profile="responses-stream/v2"'


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
        # Subcall cost controls: included in each subcall payload when
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
                raise SubcallBudgetExceeded("max subcalls exceeded")
            self._context.stats.calls_made += 1

    def _increment_successful_calls(self) -> None:
        with self._seq_lock:
            self._context.stats.successful_calls += 1

    def _request(self, payload: dict[str, Any]) -> str:
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
                    result = data.get("result")
                    if not isinstance(result, str):
                        raise RuntimeError("missing subcall result")
                    return result

                parts: list[str] = []
                completed = False
                completion_has_content = False
                completion_content = ""
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
                        if isinstance(event.get("content"), str):
                            completion_has_content = True
                            completion_content = event["content"]
                if not completed:
                    raise RuntimeError("llm_query stream ended without a completion event")
                return completion_content if completion_has_content else "".join(parts)
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
            result = self._request(payload)
            if result.strip():
                self._increment_successful_calls()
            return result
        finally:
            if auto_depth:
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
        if len(prompts) > 50:
            raise ValueError("llm_batch prompt count exceeds max 50")
        max_parallel = 5

        def _run_one(idx: int, prompt: str, ctx: str) -> str:
            if idx > 0:
                time.sleep(0.05)
            return self.llm_query(prompt, ctx)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
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
        result = data.get("result")
        if not isinstance(result, str):
            raise RuntimeError("missing root result")
        self.response_metadata = RootResponseMetadata(
            provider=str(data.get("provider") or ""),
            response_id=str(data.get("response_id") or ""),
            stop_reason=str(data.get("stop_reason") or ""),
            model=str(data.get("model") or ""),
        )
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
