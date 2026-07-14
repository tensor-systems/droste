"""Runner orchestration: request wiring, adapter dispatch, and process entrypoint."""

from __future__ import annotations

import importlib
import json
import os
import sys
from typing import Any

from droste.environments import EnvironmentConfig, create_environment, create_environment_context
from droste.execution.budget import Budget
from droste.execution.config import DEFAULT_SUBCALL_CONCURRENCY, SandboxLimits
from droste.execution.manifest import RolloutConfiguration, ScaffoldRequirements
from droste.loop.rlm import RLMConfig, run_rlm
from droste.providers import ProviderCatalog

from .http_clients import HTTPSubcallClient, RootLLMClient
from .protocol import RUNNER_PROTOCOL_VERSION, _check_protocol_version, build_response
from .sources import build_provider_registry, default_provider_catalog


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


def _optional_object(request: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Read an optional JSON object without treating malformed input as absent."""

    value = request.get(name)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"request.{name} must be an object")
    return value


def _optional_text(request: dict[str, Any], name: str) -> str | None:
    value = request.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"request.{name} must be a non-empty string or null")
    return value


def _optional_integer(
    request: dict[str, Any], name: str, *, default: int | None = None, positive: bool = False
) -> int | None:
    value = request.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"request.{name} must be an integer or null")
    if positive and value < 1:
        raise ValueError(f"request.{name} must be positive")
    return value


def _checkpoint_requirements(
    request: dict[str, Any],
) -> ScaffoldRequirements | None:
    value = _optional_object(request, "checkpoint_scaffold_requirements")
    if value is None:
        return None
    unknown = value.keys() - {"manifest_id", "required"}
    if unknown:
        raise ValueError(
            "request.checkpoint_scaffold_requirements has unknown fields: "
            + ", ".join(sorted(unknown))
        )
    required = value.get("required", {})
    if not isinstance(required, dict):
        raise ValueError("request.checkpoint_scaffold_requirements.required must be an object")
    return ScaffoldRequirements(
        manifest_id=value.get("manifest_id"),
        required=required,
    )


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


def run(
    request: dict[str, Any],
    *,
    source_ctx: Any = None,
    provider_catalog: ProviderCatalog | None = None,
) -> dict[str, Any]:
    refusal = _check_protocol_version(request)
    if refusal is not None:
        return refusal

    return _run_valid_request(
        request,
        source_ctx=source_ctx,
        provider_catalog=provider_catalog or default_provider_catalog(),
    )


def _run_valid_request(
    request: dict[str, Any],
    *,
    source_ctx: Any = None,
    provider_catalog: ProviderCatalog,
) -> dict[str, Any]:
    """Run an already version-checked request."""

    adapter_module = request.get("adapter_module")
    if isinstance(adapter_module, str) and adapter_module.strip():
        # Adapters own their response shape; the envelope version is stamped
        # only when the adapter didn't claim one itself.
        response = _run_adapter(request)
        response.setdefault("protocol_version", RUNNER_PROTOCOL_VERSION)
        return response

    context = _build_context(request)
    # source_ctx is a host-supplied live edge passed only to explicit provider
    # binders. The request cannot name Python code or mutate the catalog.
    registry = build_provider_registry(
        request,
        catalog=provider_catalog,
        context=source_ctx,
    )

    raw_budget = request.get("budget")
    if not isinstance(raw_budget, dict):
        raise ValueError("request.budget must be a complete budget object")
    budget = Budget.from_dict(raw_budget)
    raw_sandbox = request.get("sandbox", {})
    if not isinstance(raw_sandbox, dict):
        raise ValueError("request.sandbox must be an object")
    sandbox_fields = {"output_chars", "execution_timeout_ms", "capture_output_chars"}
    unknown_sandbox = raw_sandbox.keys() - sandbox_fields
    if unknown_sandbox:
        raise ValueError(
            "request.sandbox has unknown fields: " + ", ".join(sorted(unknown_sandbox))
        )
    sandbox = SandboxLimits(**raw_sandbox)

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
        budget=budget,
        sandbox=sandbox,
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
    temperature = request.get("temperature")
    stop = request.get("stop")

    if not token or not root_endpoint or not subcall_endpoint:
        raise RuntimeError("missing endpoints or token")
    root_client = RootLLMClient(
        endpoint=root_endpoint,
        token=token,
        default_model=model,
        provider=provider,
        max_output_tokens=budget.root_output_tokens,
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
    subcall_model = str(request.get("subcall_model") or "")
    subcall_reasoning_effort = str(request.get("subcall_reasoning_effort") or "")
    root_sampling = _optional_object(request, "root_sampling")
    subcall_sampling = _optional_object(request, "subcall_sampling")
    checkpoint_requirements = _checkpoint_requirements(request)
    subcall_concurrency = _optional_integer(
        request,
        "subcall_concurrency",
        default=DEFAULT_SUBCALL_CONCURRENCY,
        positive=True,
    )
    assert subcall_concurrency is not None

    subcalls = HTTPSubcallClient(
        endpoint=subcall_endpoint,
        token=token,
        session=session,
        session_index=session_index,
        context=exec_context,
        max_output_tokens=budget.subcall_output_tokens,
        model=subcall_model,
        reasoning_effort=subcall_reasoning_effort,
        max_parallel=subcall_concurrency,
    )

    environment = create_environment(
        environment_config,
        context=context,
        registry=registry,
        subcalls=subcalls,
        execution_context=exec_context,
        capability_run_id=exec_context.trace.run_id,
        capability_parent_run_id=exec_context.trace.parent_run_id,
        capability_observer=exec_context.observe_capability,
    )

    config = RLMConfig(
        budget=budget,
        sandbox=sandbox,
        root_model=str(request.get("model") or ""),
        prompt_profile=str(request.get("prompt_profile") or "") or None,
        verbose=False,
        rollout=RolloutConfiguration(
            root_revision=_optional_text(request, "root_model_revision"),
            subcall_model=subcall_model or model,
            subcall_revision=_optional_text(request, "subcall_model_revision"),
            root_sampling=(
                root_sampling
                if root_sampling is not None
                else {"temperature": temperature, "stop": stop}
            ),
            subcall_sampling=(
                subcall_sampling
                if subcall_sampling is not None
                else {"reasoning_effort": subcall_reasoning_effort or None}
            ),
            concurrency=subcall_concurrency,
            seed=_optional_integer(request, "seed"),
            runner_protocol=RUNNER_PROTOCOL_VERSION,
            source_revision=_optional_text(request, "source_revision"),
        ),
        checkpoint_requirements=checkpoint_requirements,
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

    provider_stats = registry.stats() if registry is not None else {}
    request_counts = [
        int(stats["requests_made"]) for stats in provider_stats.values() if "requests_made" in stats
    ]
    data_source_requests = sum(request_counts) if request_counts else None
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
    # types come only from the host's explicit ProviderCatalog, and the
    # in-process adapter seam is reserved for trusted callers of run().
    if str(request.get("adapter_module") or "").strip():
        raise RuntimeError(
            "adapter_module is not accepted from the request file; pass an "
            "explicit ProviderCatalog to run() from a trusted host"
        )
    response = _run_valid_request(
        request,
        provider_catalog=default_provider_catalog(),
    )
    sys.stdout.write(json.dumps(response, ensure_ascii=True))
