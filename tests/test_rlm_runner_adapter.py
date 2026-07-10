import sys
import types

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
            {"adapter_module": module_name, "answer": "hello", "protocol_version": 1}
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
        result = runner.run({"adapter_module": module_name, "protocol_version": 1})
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
    assert '"protocol_version": 1' in result["error"]["message"]
    assert result["ready"] is False
    assert result["protocol_version"] == runner.RUNNER_PROTOCOL_VERSION


def test_explicit_current_protocol_version_accepted() -> None:
    import pytest

    # Proceeds past the gate — and fails only on the missing endpoints this
    # minimal request never provided.
    with pytest.raises(RuntimeError, match="missing endpoints"):
        runner.run({"question": "q", "protocol_version": runner.RUNNER_PROTOCOL_VERSION})


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
        result = runner.run({"adapter_module": module_name, "protocol_version": 2})
    finally:
        sys.modules.pop(module_name, None)
    assert result["error"]["type"] == "protocol_version_mismatch"
    assert calls == []
