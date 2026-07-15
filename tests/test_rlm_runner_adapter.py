import os
import sys
import types
from importlib import import_module

from droste import Budget, ProviderCatalog
from droste.sources.sql_local import sqlite_provider
from droste_runner import runner


def test_rlm_runner_adapter_delegates() -> None:
    module_name = "tests_dummy_rlm_adapter"
    module = types.ModuleType(module_name)

    def run(req: dict) -> dict:
        return {"answer": req.get("answer", "ok")}

    module.run = run  # type: ignore[attr-defined]
    sys.modules[module_name] = module

    try:
        result = runner.run(
            {"adapter_module": module_name, "answer": "hello", "protocol_version": 4}
        )
    finally:
        sys.modules.pop(module_name, None)

    assert result["answer"] == "hello"
    # The runner stamps the envelope version on adapter responses that
    # didn't claim one themselves (#16).
    assert result["protocol_version"] == runner.RUNNER_PROTOCOL_VERSION


def test_adapter_claimed_protocol_version_is_not_overwritten() -> None:
    module_name = "tests_dummy_rlm_adapter_versioned"
    module = types.ModuleType(module_name)

    def run(req: dict) -> dict:
        return {"answer": "ok", "protocol_version": 99}

    module.run = run  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    try:
        result = runner.run({"adapter_module": module_name, "protocol_version": 4})
    finally:
        sys.modules.pop(module_name, None)
    assert result["protocol_version"] == 99


# --- runner protocol version gate (#16) --------------------------------------
# protocol_version is REQUIRED (no-real-users decision, 2026-07-10): every
# request must be self-describing; there is no implicit legacy default.


def test_missing_protocol_version_refused_with_structured_error() -> None:
    result = runner.run({"question": "q"})
    assert result["error"]["type"] == "protocol_version_missing"
    assert result["error"]["details"] == {
        "requested": None,
        "supported": runner.RUNNER_PROTOCOL_VERSION,
    }
    # The message tells the caller exactly what to add.
    assert f'"protocol_version": {runner.RUNNER_PROTOCOL_VERSION}' in result["error"]["message"]
    assert result["ready"] is False
    assert result["protocol_version"] == runner.RUNNER_PROTOCOL_VERSION
    assert result["prompt_pack"] is None


def test_explicit_current_protocol_version_accepted() -> None:
    import pytest

    # Proceeds past the gate — and fails only on the missing endpoints this
    # minimal request never provided.
    with pytest.raises(RuntimeError, match="missing endpoints"):
        runner.run(
            {
                "question": "q",
                "protocol_version": runner.RUNNER_PROTOCOL_VERSION,
                "budget": Budget().as_dict(),
            }
        )


def test_future_protocol_version_rejected_with_structured_error() -> None:
    result = runner.run({"question": "q", "protocol_version": 99})
    assert result["error"]["type"] == "protocol_version_mismatch"
    assert result["error"]["details"] == {
        "requested": 99,
        "supported": runner.RUNNER_PROTOCOL_VERSION,
    }
    assert "99" in result["error"]["message"]
    assert str(runner.RUNNER_PROTOCOL_VERSION) in result["error"]["message"]
    assert result["ready"] is False
    assert result["protocol_version"] == runner.RUNNER_PROTOCOL_VERSION


def test_unparseable_protocol_version_rejected() -> None:
    result = runner.run({"question": "q", "protocol_version": "banana"})
    assert result["error"]["type"] == "protocol_version_mismatch"


def test_non_integer_protocol_versions_are_not_coerced() -> None:
    # A strict gate must not int()-coerce: JSON 1.9 or true would become 1
    # and slip through (codex review). Booleans are ints in Python — reject
    # them explicitly.
    for bad in (1.9, True, 1.0, "1"):
        result = runner.run({"question": "q", "protocol_version": bad})
        assert result["error"]["type"] == "protocol_version_mismatch", bad


def test_main_version_gate_precedes_adapter_module_rejection(monkeypatch, capsys) -> None:
    # A request file with BOTH a bad envelope and a prohibited adapter_module
    # must get the versioned refusal, not the generic adapter_module error —
    # the version gate is the first check on the untrusted request boundary
    # (codex review).
    run_module = import_module("droste_runner.run")
    monkeypatch.setattr(run_module, "_read_request", lambda: {"adapter_module": "evil.module"})
    runner.main()
    out = capsys.readouterr().out
    import json

    payload = json.loads(out)
    assert payload["error"]["type"] == "protocol_version_missing"
    assert payload["operation"] is None
    assert payload["protocol_version"] == runner.RUNNER_PROTOCOL_VERSION


def test_worker_exception_envelope_is_version_stamped(tmp_path) -> None:
    # `python -m droste_runner` on a valid-version request that fails later
    # (missing endpoints) emits the exception envelope — which must carry
    # protocol_version like every other response (codex review).
    import json
    import subprocess
    import sys as _sys

    req = tmp_path / "request.json"
    req.write_text(
        json.dumps(
            {
                "protocol_version": runner.RUNNER_PROTOCOL_VERSION,
                "question": "q",
                "budget": Budget().as_dict(),
            }
        )
    )
    proc = subprocess.run(
        [_sys.executable, "-m", "droste_runner"],
        cwd=tmp_path,
        env={**os.environ, "RLM_RUNNER_REQUEST_PATH": str(req)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["protocol_version"] == runner.RUNNER_PROTOCOL_VERSION
    assert payload["operation"] == "run"
    assert payload["error"]["type"] == "RuntimeError"
    assert "missing endpoints" in payload["error"]["message"]
    assert "traceback" in payload["error"]


def test_worker_preflight_validation_error_retains_active_operation(tmp_path) -> None:
    import json
    import subprocess
    import sys as _sys

    req = tmp_path / "preflight-request.json"
    req.write_text(
        json.dumps(
            {
                "protocol_version": runner.RUNNER_PROTOCOL_VERSION,
                "operation": "preflight",
            }
        )
    )
    proc = subprocess.run(
        [_sys.executable, "-m", "droste_runner"],
        cwd=tmp_path,
        env={**os.environ, "RLM_RUNNER_REQUEST_PATH": str(req)},
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert set(payload) == {
        "protocol_version",
        "operation",
        "status",
        "preflight",
        "error",
    }
    assert payload["status"] == "error"
    assert payload["operation"] == "preflight"
    assert payload["preflight"] is None
    assert payload["error"]["type"] == "ValueError"
    assert "budget" in payload["error"]["message"]
    assert "traceback" in payload["error"]


def test_version_gate_runs_before_adapter_dispatch() -> None:
    # A bad envelope must be refused even when the request names an adapter —
    # the version gate precedes everything, and the adapter must not run on a
    # request it may misread.
    module_name = "tests_dummy_rlm_adapter_never_called"
    module = types.ModuleType(module_name)
    calls: list[dict] = []

    def run(req: dict) -> dict:
        calls.append(req)
        return {"answer": "should not happen"}

    module.run = run  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    try:
        result = runner.run({"adapter_module": module_name, "protocol_version": 1})
    finally:
        sys.modules.pop(module_name, None)
    assert result["error"]["type"] == "protocol_version_mismatch"
    assert calls == []


def _manifest_request(**overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
        "protocol_version": runner.RUNNER_PROTOCOL_VERSION,
        "model": "root-model",
        "question": "q",
        "budget": Budget().as_dict(),
        "token": "unused",
        "root_endpoint": "http://127.0.0.1:1/root",
        "subcall_endpoint": "http://127.0.0.1:1/subcall",
    }
    request.update(overrides)
    return request


def test_runner_preflight_is_content_free_and_constructs_no_http_clients(monkeypatch) -> None:
    import json
    import urllib.request

    run_module = import_module("droste_runner.run")

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight constructed an HTTP model client")

    monkeypatch.setattr(run_module, "RootLLMClient", forbidden)
    monkeypatch.setattr(run_module, "HTTPSubcallClient", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    marker = "PRIVATE_REQUEST_CONTENT_MUST_NOT_APPEAR"
    response = runner.run(
        {
            "protocol_version": runner.RUNNER_PROTOCOL_VERSION,
            "operation": "preflight",
            "model": "root-model",
            "root_reasoning_effort": "none",
            "question": marker,
            "context": {"text": marker},
            "conversation_context": marker,
            "system_prompt": marker,
            "budget": Budget().as_dict(),
            "data_sources": [
                {
                    "type": "wrapper_v1",
                    "name": "remote",
                    "base_url": "https://example.com",
                    "token": "unused",
                }
            ],
        }
    )

    assert set(response) == {"protocol_version", "operation", "status", "preflight", "error"}
    assert response["operation"] == "preflight"
    assert response["status"] == "success"
    assert response["error"] is None
    assert response["preflight"]["schema_version"] == 1
    manifest = response["preflight"]["scaffold_manifest"]
    assert manifest["id"].startswith("sha256:")
    assert manifest["inference"]["root_sampling"]["reasoning_effort"] == "none"
    assert marker not in json.dumps(response)


def test_root_reasoning_effort_is_one_typed_scaffold_fact() -> None:
    import pytest

    accepted = runner.run(
        _manifest_request(
            operation="preflight",
            root_reasoning_effort="none",
            root_sampling={"reasoning_effort": "none"},
        )
    )
    assert (
        accepted["preflight"]["scaffold_manifest"]["inference"]["root_sampling"]["reasoning_effort"]
        == "none"
    )

    with pytest.raises(ValueError, match="root_reasoning_effort must be a non-empty string"):
        runner.run(_manifest_request(operation="preflight", root_reasoning_effort=7))

    with pytest.raises(ValueError, match="root_sampling.reasoning_effort must match"):
        runner.run(
            _manifest_request(
                operation="preflight",
                root_reasoning_effort="none",
                root_sampling={"reasoning_effort": "high"},
            )
        )


def test_runner_preflight_returns_typed_scaffold_refusal_without_dispatch(monkeypatch) -> None:
    run_module = import_module("droste_runner.run")

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight constructed an HTTP model client")

    monkeypatch.setattr(run_module, "RootLLMClient", forbidden)
    monkeypatch.setattr(run_module, "HTTPSubcallClient", forbidden)
    response = runner.run(
        {
            "protocol_version": runner.RUNNER_PROTOCOL_VERSION,
            "operation": "preflight",
            "model": "root-model",
            "budget": Budget().as_dict(),
            "checkpoint_scaffold_requirements": {
                "required": {"prompt_pack": {"revision": "not-this-revision"}}
            },
        }
    )

    assert response["status"] == "refusal"
    assert response["preflight"] is None
    assert response["error"]["type"] == "ScaffoldCompatibilityError"
    assert response["error"]["code"] == "scaffold_incompatible"
    assert response["error"]["details"]["mismatches"] == [
        {
            "path": "prompt_pack.revision",
            "expected": "not-this-revision",
            "actual": "1.0.2",
        }
    ]


def test_runner_operation_defaults_to_run_and_invalid_values_are_typed() -> None:
    import pytest

    with pytest.raises(RuntimeError, match="missing endpoints"):
        runner.run(
            {
                "protocol_version": runner.RUNNER_PROTOCOL_VERSION,
                "budget": Budget().as_dict(),
            }
        )

    refusal = runner.run(
        {
            "protocol_version": runner.RUNNER_PROTOCOL_VERSION,
            "operation": "inspect",
        }
    )
    assert refusal["status"] == "refusal"
    assert refusal["error"]["code"] == "operation_invalid"


def test_runner_preflight_and_run_reject_the_same_unresolved_model_identity() -> None:
    import pytest

    common = {
        "protocol_version": runner.RUNNER_PROTOCOL_VERSION,
        "budget": Budget().as_dict(),
        "model": "",
        "subcall_model": "",
    }
    with pytest.raises(ValueError, match="rollout subcall_model"):
        runner.run({**common, "operation": "preflight"})
    with pytest.raises(ValueError, match="rollout subcall_model"):
        runner.run(
            {
                **common,
                "operation": "run",
                "token": "unused",
                "root_endpoint": "https://example.com/root",
                "subcall_endpoint": "https://example.com/subcall",
            }
        )


def test_v3_preflight_intent_is_refused_before_any_runner_resolution(monkeypatch) -> None:
    run_module = import_module("droste_runner.run")

    def forbidden(*args, **kwargs):
        raise AssertionError("version gate allowed preflight work")

    monkeypatch.setattr(run_module, "default_provider_catalog", forbidden)
    monkeypatch.setattr(run_module, "RootLLMClient", forbidden)
    monkeypatch.setattr(run_module, "HTTPSubcallClient", forbidden)
    refusal = runner.run(
        {
            "protocol_version": 3,
            "operation": "preflight",
            "token": "must-not-be-read",
            "root_endpoint": "https://example.com/root",
            "subcall_endpoint": "https://example.com/subcall",
        }
    )

    assert refusal["status"] == "refusal"
    assert refusal["operation"] is None
    assert refusal["error"]["code"] == "protocol_version_mismatch"
    assert refusal["error"]["details"] == {"requested": 3, "supported": 4}


def test_preflight_never_binds_or_connects_configured_providers(monkeypatch) -> None:
    import sqlite3

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight opened a provider connection")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    response = runner.run(
        {
            "protocol_version": runner.RUNNER_PROTOCOL_VERSION,
            "operation": "preflight",
            "model": "root-model",
            "budget": Budget().as_dict(),
            "data_sources": [
                {
                    "type": "sqlite",
                    "name": "db",
                    "sqlite_path": "/path/that/must/not/be-opened.db",
                }
            ],
            "default_source": "db",
        },
        provider_catalog=ProviderCatalog((sqlite_provider(),)),
    )

    assert response["status"] == "success"
    assert response["preflight"]["scaffold_manifest"]["capabilities"]["model_visible_globals"] == [
        "aggregate_json_counts",
        "answer",
        "batch_llm_query",
        "context",
        "db",
        "get_schema",
        "llm_batch",
        "llm_batch_json",
        "llm_query",
        "llm_query_batched",
        "llm_query_batched_json",
        "query",
    ]


def test_preflight_rejects_adapter_without_dispatching_it() -> None:
    module_name = "tests_dummy_preflight_adapter_never_called"
    module = types.ModuleType(module_name)
    calls: list[dict] = []

    def run(req: dict) -> dict:
        calls.append(req)
        return {"answer": "must not happen"}

    module.run = run  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    try:
        refusal = runner.run(
            {
                "protocol_version": runner.RUNNER_PROTOCOL_VERSION,
                "operation": "preflight",
                "adapter_module": module_name,
            }
        )
    finally:
        sys.modules.pop(module_name, None)

    assert calls == []
    assert refusal["operation"] == "preflight"
    assert refusal["status"] == "refusal"
    assert refusal["error"]["code"] == "adapter_unsupported"


def test_manifest_objects_refuse_malformed_runner_values() -> None:
    import pytest

    for name in (
        "root_sampling",
        "subcall_sampling",
        "checkpoint_scaffold_requirements",
    ):
        with pytest.raises(ValueError, match=rf"request\.{name} must be an object"):
            runner.run(_manifest_request(**{name: []}))


def test_checkpoint_requirements_refuse_unknown_and_malformed_fields() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown fields: host_metadata"):
        runner.run(_manifest_request(checkpoint_scaffold_requirements={"host_metadata": {}}))
    with pytest.raises(ValueError, match="required must be an object"):
        runner.run(_manifest_request(checkpoint_scaffold_requirements={"required": []}))


def test_resolved_rollout_identity_fields_are_not_coerced() -> None:
    import pytest

    for name in ("root_model_revision", "subcall_model_revision", "source_revision"):
        with pytest.raises(ValueError, match=rf"request\.{name}"):
            runner.run(_manifest_request(**{name: 7}))
    for name, value in (("seed", 1.5), ("subcall_concurrency", True)):
        with pytest.raises(ValueError, match=rf"request\.{name}"):
            runner.run(_manifest_request(**{name: value}))
