from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from typing import Any

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rlm_core.loop.rlm import RLMConfig, run_rlm  # type: ignore
from rlm_core.protocols.environment import EnvCapabilities, ExecutionResult, RLMEnvironment  # type: ignore
from rlm_core.protocols.llm_client import TokenUsage  # type: ignore
from rlm_core.protocols.subcall_client import SubcallClient  # type: ignore


class OutputBuffer(io.StringIO):
    def __init__(self, max_chars: int) -> None:
        super().__init__()
        self._max_chars = max(0, int(max_chars or 0))
        self._size = 0

    def write(self, text: str) -> int:
        if not text:
            return 0
        if self._max_chars > 0:
            new_size = self._size + len(text)
            if new_size > self._max_chars:
                raise RuntimeError(
                    f"Sandbox output exceeded {self._max_chars} characters (attempted {new_size})."
                )
            self._size = new_size
        return super().write(text)


class RunnerEnvironment(RLMEnvironment):
    def __init__(
        self,
        *,
        context: Any,
        data_source: dict[str, Any] | None,
        subcalls: SubcallClient,
        max_output_chars: int,
        exec_timeout_ms: int,
    ) -> None:
        self._context = context
        self._data_source = data_source
        self._subcalls = subcalls
        self._max_output_chars = max_output_chars
        self._exec_timeout_ms = exec_timeout_ms
        self._globals: dict[str, Any] = {
            "answer": {"content": "", "ready": False},
            "context": context,
            "llm_query": subcalls.llm_query,
            "llm_batch": subcalls.llm_batch,
            "batch_llm_query": subcalls.llm_batch,
        }
        if data_source is not None:
            self._globals["data_source"] = data_source

    def capabilities(self) -> EnvCapabilities:
        return {
            "tools_in_root": False,
            "max_output_chars": self._max_output_chars,
        }

    def globals(self) -> dict[str, Any]:
        return self._globals

    def prompt_fragment(self) -> str:
        parts: list[str] = []
        parts.append(
            "Context is available in a Python variable named `context`. "
            "If it contains files, expect context['files'] entries with path, name, mime, size, and optional text."
        )
        if self._data_source:
            parts.append("Data source: wrapper_v1 (use data_source_search/get/content helpers).")
            allowed_hosts = self._data_source.get("allowed_hosts")
            if isinstance(allowed_hosts, list) and allowed_hosts:
                parts.append("Allowed hosts: " + ", ".join(str(h) for h in allowed_hosts if h))
            limits = self._data_source.get("limits")
            if isinstance(limits, dict) and limits:
                parts.append("Wrapper limits: " + json.dumps(limits, ensure_ascii=True))
        return "\n".join(parts)

    def execute(self, code: str) -> ExecutionResult:
        stdout_buf = OutputBuffer(self._max_output_chars)
        stderr_buf = io.StringIO()
        timed_out = False
        exit_code = 0

        def _handle_timeout(signum: int, frame: Any) -> None:
            raise TimeoutError("execution timed out")

        old_handler = None
        if self._exec_timeout_ms and self._exec_timeout_ms > 0:
            old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, self._exec_timeout_ms / 1000.0)

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(compile(code, "<rlm>", "exec"), self._globals)
        except TimeoutError:
            timed_out = True
            exit_code = 124
            raise
        finally:
            if old_handler is not None:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

        return ExecutionResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            timed_out=timed_out,
            exit_code=exit_code,
            files_written=[],
        )

    def close(self) -> None:
        return


class DataSourceWrapper:
    def __init__(self, config: dict[str, Any] | None) -> None:
        self._config = config or {}
        self._requests_made = 0

    @property
    def requests_made(self) -> int:
        return self._requests_made

    def _limits(self) -> dict[str, Any]:
        limits = self._config.get("limits")
        if isinstance(limits, dict):
            return limits
        return {}

    def _int_limit(self, value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        return None

    def _check_request_budget(self) -> None:
        self._requests_made += 1
        max_requests = self._int_limit(self._limits().get("max_requests"))
        if max_requests is not None and self._requests_made > max_requests:
            raise ValueError("data_source max_requests exceeded")

    def _timeout_seconds(self) -> float | None:
        timeout_ms = self._int_limit(self._limits().get("timeout_ms"))
        if timeout_ms is None or timeout_ms < 0:
            return None
        return timeout_ms / 1000.0

    def _read_response(self, resp: Any) -> bytes:
        max_bytes = self._int_limit(self._limits().get("max_response_bytes"))
        if max_bytes is None:
            return resp.read()
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("data_source max_response_bytes exceeded")
        return data

    def _call(self, path: str, payload: dict[str, Any]) -> Any:
        if not isinstance(self._config, dict):
            raise ValueError("data_source is not configured")
        base_url = self._config.get("base_url")
        token = self._config.get("token")
        if not base_url or not token:
            raise ValueError("data_source missing base_url or token")
        self._check_request_budget()
        url = str(base_url).rstrip("/") + "/" + path.lstrip("/")
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Authorization": "Bearer " + str(token), "Content-Type": "application/json"},
            method="POST",
        )
        timeout = self._timeout_seconds()
        try:
            if timeout is None:
                resp = urllib.request.urlopen(req)
            else:
                resp = urllib.request.urlopen(req, timeout=timeout)
            with resp:
                raw = self._read_response(resp)
        except urllib.error.HTTPError as exc:
            raise ValueError(f"data_source HTTP error {getattr(exc, 'code', 0)}") from exc
        except Exception as exc:
            raise ValueError(f"data_source request failed: {exc}") from exc
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError("data_source response must be JSON") from exc

    def search(self, query: str, filters: Any = None, page: Any = None) -> Any:
        payload: dict[str, Any] = {"query": query}
        if filters is not None:
            payload["filters"] = filters
        if page is not None:
            payload["page"] = page
        return self._call("/search", payload)

    def get(self, id: str) -> Any:
        return self._call("/get", {"id": id})

    def content(self, id: str, format: str = "text", max_bytes: int | None = None) -> Any:
        payload: dict[str, Any] = {"id": id}
        if format is not None:
            payload["format"] = format
        if max_bytes is not None:
            payload["max_bytes"] = max_bytes
        return self._call("/content", payload)


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

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def _depth_get(self) -> int:
        return getattr(self._depth, "value", 0)

    def _depth_set(self, value: int) -> None:
        self._depth.value = value

    def _increment_calls(self) -> None:
        self._context.stats.calls_made += 1
        if self._max_calls >= 0 and self._context.stats.calls_made > self._max_calls:
            raise RuntimeError("max subcalls exceeded")

    def _request(self, payload: dict[str, Any]) -> str:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Authorization": "Bearer " + self._token, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status = getattr(exc, "code", 0)
            raise RuntimeError(f"llm_query failed with HTTP {status}: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"llm_query failed: {exc}") from exc
        data = json.loads(raw)
        result = data.get("result")
        if not isinstance(result, str):
            raise RuntimeError("missing subcall result")
        return result

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
            payload = {
                "prompt": prompt,
                "depth": depth,
                "seq": self._next_seq(),
                "session": self._session,
                "session_index": self._session_index,
            }
            return self._request(payload)
        finally:
            if auto_depth:
                self._depth_set(depth - 1)

    def llm_batch(self, prompts: list[str], contexts: list[str] | None = None) -> list[str]:
        if contexts is None:
            contexts = [""] * len(prompts)
        if len(contexts) != len(prompts):
            raise ValueError("contexts length must match prompts length")
        results: list[str] = [""] * len(prompts)
        if not prompts:
            return results
        if len(prompts) > 50:
            raise ValueError("llm_batch prompt count exceeds max 50")
        max_parallel = 5

        def _run_one(idx: int, prompt: str, ctx: str) -> str:
            if idx > 0:
                time.sleep(0.05)
            return self.llm_query(prompt, ctx)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        errors: list[Exception | None] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {}
            for idx, (prompt, ctx) in enumerate(zip(prompts, contexts)):
                future = executor.submit(_run_one, idx, prompt, ctx)
                futures[future] = idx
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    errors[idx] = exc
        for err in errors:
            if err is not None:
                raise err
        return results

    def llm_batch_with_errors(
        self,
        prompts: list[str],
        contexts: list[str] | None = None,
    ) -> tuple[list[str], list[dict[str, object]]]:
        if contexts is None:
            contexts = [""] * len(prompts)
        if len(contexts) != len(prompts):
            raise ValueError("contexts length must match prompts length")
        results: list[str] = [""] * len(prompts)
        errors: list[dict[str, object]] = []
        if not prompts:
            return results, errors
        def _run_one(idx: int, prompt: str, ctx: str) -> None:
            try:
                results[idx] = self.llm_query(prompt, ctx)
            except Exception as exc:
                errors.append({"index": idx, "error": str(exc)})
        threads = []
        for idx, (prompt, ctx) in enumerate(zip(prompts, contexts)):
            t = threading.Thread(target=lambda i=idx, p=prompt, c=ctx: _run_one(i, p, c), daemon=True)
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
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
        self.last_provider = ""
        self.last_response_id = ""
        self.last_stop_reason = ""
        self.last_model = ""

    def responses_create(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        return_usage: bool = False,
    ) -> str | tuple[str, TokenUsage]:
        resolved_model = model or self._default_model
        if not resolved_model:
            raise ValueError("model is required")
        max_output_tokens = self._max_output_tokens or int(max_tokens or 0)
        temp = self._temperature if self._temperature is not None else temperature
        payload: dict[str, Any] = {
            "messages": messages,
            "model": resolved_model,
            "max_output_tokens": max_output_tokens,
            "temperature": temp,
            "stop": self._stop,
            "session": self._session,
            "session_index": self._session_index,
        }
        if self._provider:
            payload["provider"] = self._provider
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Authorization": "Bearer " + self._token, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            status = getattr(exc, "code", 0)
            raise RuntimeError(f"root llm failed with HTTP {status}: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"root llm failed: {exc}") from exc
        data = json.loads(raw)
        result = data.get("result")
        if not isinstance(result, str):
            raise RuntimeError("missing root result")
        self.last_provider = str(data.get("provider") or "")
        self.last_response_id = str(data.get("response_id") or "")
        self.last_stop_reason = str(data.get("stop_reason") or "")
        self.last_model = str(data.get("model") or "")
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


def _read_request(path: str | None = None) -> dict[str, Any]:
    path = path or os.environ.get("RLM_RUNNER_REQUEST_PATH")
    if not path and len(sys.argv) > 1:
        path = sys.argv[1]
    if not path:
        raise RuntimeError("RLM_RUNNER_REQUEST_PATH is required")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_context(payload: dict[str, Any]) -> Any:
    if "context_path" in payload and payload["context_path"]:
        with open(payload["context_path"], "r", encoding="utf-8") as handle:
            return json.load(handle)
    return payload.get("context")


def run(request: dict[str, Any]) -> dict[str, Any]:
    context = _build_context(request)
    data_source = request.get("data_source")

    max_iterations = int(request.get("max_iterations") or 1)
    max_depth = int(request.get("max_depth") or 1)
    max_subcalls = int(request.get("max_subcalls") or 0)
    max_output_chars = int(request.get("max_output_chars") or 0)
    exec_timeout_ms = int(request.get("exec_timeout_ms") or 0)

    token = str(request.get("token") or "")
    root_endpoint = str(request.get("root_endpoint") or "")
    subcall_endpoint = str(request.get("subcall_endpoint") or "")
    session = str(request.get("session") or "")
    session_index = int(request.get("session_index") or 0)

    if not token or not root_endpoint or not subcall_endpoint:
        raise RuntimeError("missing endpoints or token")

    root_client = RootLLMClient(
        endpoint=root_endpoint,
        token=token,
        default_model=str(request.get("model") or ""),
        provider=str(request.get("provider") or "") or None,
        max_output_tokens=int(request.get("max_output_tokens") or 0),
        temperature=request.get("temperature"),
        stop=request.get("stop"),
        session=session,
        session_index=session_index,
    )

    from rlm_core.execution.context import create_execution_context  # type: ignore

    exec_context = create_execution_context(
        max_depth=max_depth,
        max_calls=max_subcalls,
        max_iterations=max_iterations,
        max_output_chars=max_output_chars,
        verbose=False,
    )

    subcalls = HTTPSubcallClient(
        endpoint=subcall_endpoint,
        token=token,
        session=session,
        session_index=session_index,
        max_calls=max_subcalls,
        max_depth=max_depth,
        context=exec_context,
    )

    environment = RunnerEnvironment(
        context=context,
        data_source=data_source,
        subcalls=subcalls,
        max_output_chars=max_output_chars,
        exec_timeout_ms=exec_timeout_ms,
    )

    if data_source is not None:
        wrapper = DataSourceWrapper(data_source)
        environment.globals()["data_source_search"] = wrapper.search
        environment.globals()["data_source_get"] = wrapper.get
        environment.globals()["data_source_content"] = wrapper.content
    else:
        wrapper = None

    config = RLMConfig(
        max_iterations=max_iterations,
        max_depth=max_depth,
        max_calls=max_subcalls,
        max_output_chars=max_output_chars,
        root_model=str(request.get("model") or ""),
        verbose=False,
    )

    system_prompt_raw = request.get("system_prompt")
    system_prompt = None
    if isinstance(system_prompt_raw, str) and system_prompt_raw.strip():
        system_prompt = system_prompt_raw
    system_prompt_additions = str(request.get("system_prompt_additions") or "")

    result = run_rlm(
        str(request.get("question") or ""),
        environment=environment,
        root_llm=root_client,
        subcalls=subcalls,
        config=config,
        system_prompt=system_prompt,
        system_prompt_additions=system_prompt_additions,
        conversation_context=str(request.get("conversation_context") or ""),
    )

    response: dict[str, Any] = {
        "answer": result.answer,
        "ready": result.ready,
        "iterations": result.iterations,
        "tokens_used": result.tokens_used,
        "subcalls": result.sub_calls_made,
        "trajectory": [
            {
                "iteration": entry.iteration,
                "llm_input": entry.llm_input,
                "llm_output": entry.llm_output,
                "code_executed": entry.code_executed,
                "execution_result": entry.execution_result,
                "tokens_used": entry.tokens_used,
            }
            for entry in result.trajectory
        ],
        "error": None,
        "provider": root_client.last_provider,
        "response_id": root_client.last_response_id,
        "stop_reason": root_client.last_stop_reason,
        "model": root_client.last_model or str(request.get("model") or ""),
    }
    if wrapper is not None:
        response["data_source_requests"] = wrapper.requests_made
    if result.error:
        response["error"] = {
            "type": result.error.type,
            "message": result.error.message,
            "code": result.error.code,
            "details": result.error.details,
        }
    return response


def main() -> None:
    request = _read_request()
    response = run(request)
    sys.stdout.write(json.dumps(response, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        payload = {
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=True))
        sys.exit(1)
