"""Runner orchestration: request wiring, adapter dispatch, and process entrypoint."""

from __future__ import annotations

import importlib
import json
import os
import sys
from typing import Any

from droste.environments import EnvironmentConfig, create_environment, create_environment_context
from droste.execution.config import DEFAULT_MAX_CALLS, DEFAULT_MAX_ITERATIONS
from droste.loop.rlm import RLMConfig, run_rlm
from droste.registry import DataSourceRegistry

from .http_clients import HTTPSubcallClient, RootLLMClient
from .protocol import RUNNER_PROTOCOL_VERSION, _check_protocol_version, build_response
from .sources import WrapperV1DataSource, build_data_sources


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


def _run_adapter(request: dict[str, Any]) -> dict[str, Any]:
    adapter_module = str(request.get("adapter_module") or "").strip()
    if not adapter_module:
        raise RuntimeError("adapter_module is required for adapter runs")
    module = importlib.import_module(adapter_module)
    run_fn = getattr(module, "run", None)
    if not callable(run_fn):
        raise RuntimeError(f"adapter module {adapter_module} missing run(request) function")
    result = run_fn(request)
    if not isinstance(result, dict):
        raise RuntimeError(
            f"adapter module {adapter_module} returned {type(result)}; expected dict"
        )
    return result


def run(request: dict[str, Any], *, source_ctx: Any = None) -> dict[str, Any]:
    refusal = _check_protocol_version(request)
    if refusal is not None:
        return refusal

    return _run_valid_request(request, source_ctx=source_ctx)


def _run_valid_request(request: dict[str, Any], *, source_ctx: Any = None) -> dict[str, Any]:
    """Run an already version-checked request."""

    adapter_module = request.get("adapter_module")
    if isinstance(adapter_module, str) and adapter_module.strip():
        # Adapters own their response shape; the envelope version is stamped
        # only when the adapter didn't claim one itself.
        response = _run_adapter(request)
        response.setdefault("protocol_version", RUNNER_PROTOCOL_VERSION)
        return response

    context = _build_context(request)
    # source_ctx is the host-supplied edge context for registered source
    # factories (source unification): in-process hosts pass live
    # handles; subprocess hosts pass whatever their entrypoint assembled.
    sources, default_source = build_data_sources(request, source_ctx)
    registry = DataSourceRegistry(sources, default_source_name=default_source) if sources else None

    # Omitted budgets fall back to the core loop defaults instead of the old
    # runner-local 1 iteration / 0 subcalls (0 made the FIRST llm_query raise
    # "max subcalls exceeded"). An EXPLICIT value is always honored — including
    # max_subcalls=0, which deliberately forbids subcalls. Hosts that care
    # about cost must pass explicit budgets in the request.
    def _budget(key: str, default: int) -> int:
        raw = request.get(key)
        if raw is None or raw == "":
            return default
        return int(raw)

    max_iterations = _budget("max_iterations", DEFAULT_MAX_ITERATIONS)
    max_depth = int(request.get("max_depth") or 1)
    max_subcalls = _budget("max_subcalls", DEFAULT_MAX_CALLS)
    max_output_chars = int(request.get("max_output_chars") or 0)
    exec_timeout_ms = int(request.get("exec_timeout_ms") or 0)

    token = str(request.get("token") or "")
    root_endpoint = str(request.get("root_endpoint") or "")
    subcall_endpoint = str(request.get("subcall_endpoint") or "")
    session = str(request.get("session") or "")
    session_index = int(request.get("session_index") or 0)

    from droste.execution.progress import emit_event  # type: ignore
    from droste.execution.trace import (  # type: ignore
        DataUseAuthorization,
        TraceRetentionPolicy,
    )

    environment_config = EnvironmentConfig(
        kind="native",
        max_depth=max_depth,
        max_calls=max_subcalls,
        max_iterations=max_iterations,
        max_output_chars=max_output_chars,
        exec_timeout_ms=exec_timeout_ms,
    )
    exec_context = create_environment_context(
        environment_config,
        verbose=False,
        # NDJSON on stderr is the runner's event contract — the relay's
        # forwarding filter and native hosts read it there. Attached
        # explicitly now that a bare engine call emits nothing (#35).
        on_event=emit_event,
        run_id=str(request.get("run_id") or "") or None,
        parent_run_id=str(request.get("parent_run_id") or "") or None,
        trace_depth=(int(request["trace_depth"]) if request.get("trace_depth") is not None else 0),
        trace_retention=TraceRetentionPolicy(
            retain=frozenset(str(value) for value in (request.get("retain_trace") or [])),
            policy_id=str(request.get("trace_policy_id") or "default-no-content"),
            expires_at=(
                str(request["trace_expires_at"])
                if request.get("trace_expires_at") is not None
                else None
            ),
            host_managed_expiry=request.get("trace_host_managed_expiry") is True,
        ),
        data_use=DataUseAuthorization(
            training_allowed=request.get("training_allowed") is True,
            authorization_ref=(
                str(request["data_use_authorization_ref"])
                if request.get("data_use_authorization_ref") is not None
                else None
            ),
            purposes=frozenset(str(value) for value in (request.get("data_use_purposes") or [])),
        ),
    )

    model = str(request.get("model") or "")
    provider = str(request.get("provider") or "") or None
    max_output_tokens = int(request.get("max_output_tokens") or 0)
    temperature = request.get("temperature")
    stop = request.get("stop")

    if not token or not root_endpoint or not subcall_endpoint:
        raise RuntimeError("missing endpoints or token")
    root_client = RootLLMClient(
        endpoint=root_endpoint,
        token=token,
        default_model=model,
        provider=provider,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        stop=stop,
        session=session,
        session_index=session_index,
    )
    # Subcall cost controls: optional per-run overrides forwarded in
    # every subcall payload; unset values are omitted so the server applies
    # its defaults (bounded output + no thinking for subcalls). An explicit
    # zero/negative budget is rejected: a subcall cannot answer in 0 tokens,
    # and silently treating it as unset would mask a caller bug.
    raw_subcall_max = request.get("subcall_max_output_tokens")
    if raw_subcall_max in (None, ""):
        subcall_max_output_tokens = 0
    else:
        subcall_max_output_tokens = int(raw_subcall_max)
        if subcall_max_output_tokens <= 0:
            raise ValueError("subcall_max_output_tokens must be positive when set")
    subcall_model = str(request.get("subcall_model") or "")
    subcall_reasoning_effort = str(request.get("subcall_reasoning_effort") or "")

    subcalls = HTTPSubcallClient(
        endpoint=subcall_endpoint,
        token=token,
        session=session,
        session_index=session_index,
        max_calls=environment_config.max_calls,
        max_depth=environment_config.max_depth,
        context=exec_context,
        max_output_tokens=subcall_max_output_tokens,
        model=subcall_model,
        reasoning_effort=subcall_reasoning_effort,
    )

    environment = create_environment(
        environment_config,
        context=context,
        registry=registry,
        subcalls=subcalls,
        capability_run_id=exec_context.trace.run_id,
        capability_parent_run_id=exec_context.trace.parent_run_id,
        capability_observer=exec_context.observe_capability,
    )

    config = RLMConfig(
        max_iterations=environment_config.max_iterations,
        max_depth=environment_config.max_depth,
        max_calls=environment_config.max_calls,
        max_output_chars=environment_config.max_output_chars,
        root_model=str(request.get("model") or ""),
        prompt_profile=str(request.get("prompt_profile") or "") or None,
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
        # The subcall client increments exec_context.stats.calls_made; without
        # sharing it, run_rlm creates its own context and reports subcalls=0
        # no matter how many subcalls actually ran.
        context=exec_context,
    )

    wrapper_sources = [s for s in sources if isinstance(s, WrapperV1DataSource)]
    data_source_requests = (
        sum(source.requests_made for source in wrapper_sources) if wrapper_sources else None
    )
    return build_response(
        result=result,
        metadata=root_client.response_metadata,
        requested_model=str(request.get("model") or ""),
        data_source_requests=data_source_requests,
    )


def main() -> None:
    request = _read_request()
    # The version gate runs FIRST — before any other request-field check —
    # so a bad envelope always gets the structured versioned refusal, never
    # a generic error from a later validation the caller can't attribute.
    refusal = _check_protocol_version(request)
    if refusal is not None:
        sys.stdout.write(json.dumps(refusal, ensure_ascii=True))
        return
    # The request file is the untrusted boundary (hosted runners are fed one by
    # the parent process). A request must never name code to import: source
    # types come from register_source_type() in the entrypoint (Option C), and
    # the in-process adapter seam is reserved for trusted callers of run().
    if str(request.get("adapter_module") or "").strip():
        raise RuntimeError(
            "adapter_module is not accepted from the request file; register "
            "source types via register_source_type() in the runner entrypoint"
        )
    response = _run_valid_request(request)
    sys.stdout.write(json.dumps(response, ensure_ascii=True))
