from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from droste import (
    Budget,
    EnvironmentConfig,
    RLMConfig,
    RolloutConfiguration,
    ScaffoldManifest,
    SubcallInputCapacity,
    create_environment,
    create_environment_context,
    preflight_rlm,
    run_rlm,
)
from droste.capabilities import broker_subcalls
from droste.execution.budget import BudgetLedger
from droste.loop.rlm import _SubcallGate
from droste.protocols.llm_client import TokenUsage
from droste.protocols.subcall_capacity import resolve_subcall_input_capacity
from droste.testing import MockEnvironment, MockLLMClient, MockResponse, MockSubcallClient
from droste_runner.runner import RUNNER_PROTOCOL_VERSION
from droste_runner.runner import run as run_worker


class ReportingSubcalls(MockSubcallClient):
    def __init__(self, capacity: SubcallInputCapacity) -> None:
        super().__init__()
        self._capacity = capacity

    @property
    def input_token_capacity(self) -> SubcallInputCapacity:
        return self._capacity


class RecordingRoot(MockLLMClient):
    def __init__(self) -> None:
        super().__init__(
            [
                MockResponse(
                    "```python\nanswer['content'] = 'ok'\nanswer['ready'] = True\n```",
                    TokenUsage(1, 1, 2, exact=True),
                )
            ]
        )
        self.messages: list[list[dict[str, Any]]] = []

    def responses_create(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
        return_usage: bool = False,
    ) -> str | tuple[str, TokenUsage]:
        self.messages.append(messages)
        return super().responses_create(
            messages,
            model,
            max_tokens=max_tokens,
            temperature=temperature,
            return_usage=return_usage,
        )


def _run_with(subcalls: MockSubcallClient, config: RLMConfig | None = None):
    root = RecordingRoot()
    result = run_rlm(
        "q",
        environment=MockEnvironment(),
        root_llm=root,
        subcalls=subcalls,
        config=config,
    )
    return root, result


def test_capacity_value_is_frozen_strict_and_round_trips() -> None:
    capacity = SubcallInputCapacity.bounded(128_000)

    assert SubcallInputCapacity.from_dict(capacity.as_dict()) == capacity
    assert SubcallInputCapacity.unbounded().as_dict() == {
        "state": "unbounded",
        "tokens": None,
    }
    assert SubcallInputCapacity.unknown().as_dict() == {
        "state": "unknown",
        "tokens": None,
    }
    with pytest.raises(FrozenInstanceError):
        capacity.tokens = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="positive"):
        SubcallInputCapacity.bounded(0)
    with pytest.raises(ValueError, match="state must be"):
        SubcallInputCapacity([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown extra"):
        SubcallInputCapacity.from_dict({"state": "unknown", "tokens": None, "extra": 1})


@pytest.mark.parametrize(
    ("subcalls", "expected", "prompt_text"),
    [
        (
            ReportingSubcalls(SubcallInputCapacity.bounded(128_000)),
            SubcallInputCapacity.bounded(128_000),
            "128000 tokens per call (bounded)",
        ),
        (
            ReportingSubcalls(SubcallInputCapacity.unbounded()),
            SubcallInputCapacity.unbounded(),
            "unbounded (deliberate)",
        ),
        (
            MockSubcallClient(),
            SubcallInputCapacity.unknown(),
            "unknown (client and rollout did not report)",
        ),
    ],
)
def test_run_resolves_capacity_into_prompt_and_scaffold(
    subcalls: MockSubcallClient,
    expected: SubcallInputCapacity,
    prompt_text: str,
) -> None:
    root, result = _run_with(subcalls)

    assert result.scaffold_manifest is not None
    assert result.scaffold_manifest.body["inference"]["input_capacity"] == {
        "subcall": expected.as_dict()
    }
    system_prompt = str(root.messages[0][0]["content"])
    assert f"subcall_input_capacity={prompt_text}" in system_prompt
    assert "100k tokens" not in system_prompt.casefold()


def test_invalid_or_conflicting_client_capacity_fails_before_inference() -> None:
    class InvalidSubcalls(MockSubcallClient):
        @property
        def input_token_capacity(self) -> object:
            return 0

    invalid_root = RecordingRoot()
    with pytest.raises(TypeError, match="must return SubcallInputCapacity"):
        run_rlm(
            "q",
            environment=MockEnvironment(),
            root_llm=invalid_root,
            subcalls=InvalidSubcalls(),
        )
    assert invalid_root._call_count == 0

    mismatch_root = RecordingRoot()
    with pytest.raises(ValueError, match="does not match the subcall client"):
        run_rlm(
            "q",
            environment=MockEnvironment(),
            root_llm=mismatch_root,
            subcalls=ReportingSubcalls(SubcallInputCapacity.bounded(64_000)),
            config=RLMConfig(
                rollout=RolloutConfiguration(
                    subcall_input_capacity=SubcallInputCapacity.bounded(128_000)
                )
            ),
        )
    assert mismatch_root._call_count == 0


def test_client_capacity_getter_is_read_exactly_once_and_errors_propagate() -> None:
    class CountingSubcalls(MockSubcallClient):
        reads = 0

        @property
        def input_token_capacity(self) -> SubcallInputCapacity:
            self.reads += 1
            return SubcallInputCapacity.bounded(48_000)

    counting = CountingSubcalls()
    _run_with(counting)
    assert counting.reads == 1

    class BrokenSubcalls(MockSubcallClient):
        @property
        def input_token_capacity(self) -> SubcallInputCapacity:
            raise AttributeError("adapter metadata failed")

    root = RecordingRoot()
    with pytest.raises(AttributeError, match="adapter metadata failed"):
        run_rlm(
            "q",
            environment=MockEnvironment(),
            root_llm=root,
            subcalls=BrokenSubcalls(),
        )
    assert root._call_count == 0


def test_brokered_environment_reuses_its_construction_snapshot() -> None:
    class CountingSubcalls(MockSubcallClient):
        reads = 0

        @property
        def input_token_capacity(self) -> SubcallInputCapacity:
            self.reads += 1
            return SubcallInputCapacity.bounded(48_000)

    subcalls = CountingSubcalls()
    environment_config = EnvironmentConfig(kind="native", budget=Budget())
    execution_context = create_environment_context(environment_config)
    environment = create_environment(
        environment_config,
        context={},
        registry=None,
        subcalls=subcalls,
        execution_context=execution_context,
    )
    assert subcalls.reads == 1

    result = run_rlm(
        "q",
        environment=environment,
        root_llm=RecordingRoot(),
        subcalls=subcalls,
        context=execution_context,
    )

    assert result.ready
    assert subcalls.reads == 1


@pytest.mark.parametrize(
    ("declared", "reported", "expected"),
    [
        ("unknown", "unknown", "unknown"),
        ("unknown", "bounded", "bounded"),
        ("unknown", "unbounded", "unbounded"),
        ("bounded", "unknown", "bounded"),
        ("bounded", "bounded", "bounded"),
        ("unbounded", "unknown", "unbounded"),
        ("unbounded", "unbounded", "unbounded"),
    ],
)
def test_capacity_resolution_matrix(
    declared: str,
    reported: str,
    expected: str,
) -> None:
    values = {
        "unknown": SubcallInputCapacity.unknown(),
        "bounded": SubcallInputCapacity.bounded(64_000),
        "unbounded": SubcallInputCapacity.unbounded(),
    }

    assert resolve_subcall_input_capacity(values[declared], values[reported]) == values[expected]


@pytest.mark.parametrize(
    ("declared", "reported"),
    [
        (SubcallInputCapacity.bounded(64_000), SubcallInputCapacity.unbounded()),
        (SubcallInputCapacity.unbounded(), SubcallInputCapacity.bounded(64_000)),
        (SubcallInputCapacity.bounded(64_000), SubcallInputCapacity.bounded(32_000)),
    ],
)
def test_capacity_resolution_rejects_conflicting_known_values(
    declared: SubcallInputCapacity,
    reported: SubcallInputCapacity,
) -> None:
    with pytest.raises(ValueError, match="does not match"):
        resolve_subcall_input_capacity(declared, reported)


def test_declared_capacity_fills_unknown_client_metadata() -> None:
    declared = SubcallInputCapacity.bounded(32_000)
    _, result = _run_with(
        MockSubcallClient(),
        RLMConfig(
            rollout=RolloutConfiguration(subcall_input_capacity=declared),
        ),
    )

    assert result.scaffold_manifest is not None
    assert result.scaffold_manifest.body["inference"]["input_capacity"] == {
        "subcall": declared.as_dict()
    }


def test_broker_and_run_gate_forward_optional_capacity_without_owning_it() -> None:
    capacity = SubcallInputCapacity.bounded(96_000)
    raw = ReportingSubcalls(capacity)
    brokered = broker_subcalls(
        raw,
        BudgetLedger(Budget()),
        usage_callback=lambda _usage: None,
        settlement_callback=lambda _exact: None,
    )
    gated = _SubcallGate(brokered)

    assert brokered.input_token_capacity == capacity
    assert gated.input_token_capacity == capacity

    unknown = broker_subcalls(
        MockSubcallClient(),
        BudgetLedger(Budget()),
        usage_callback=lambda _usage: None,
        settlement_callback=lambda _exact: None,
    )
    assert unknown.input_token_capacity == SubcallInputCapacity.unknown()


def test_preflight_and_execution_resolve_the_same_capacity_without_dispatch() -> None:
    capacity = SubcallInputCapacity.bounded(72_000)
    subcalls = ReportingSubcalls(capacity)
    config = RLMConfig()
    preflight = preflight_rlm(
        environment=MockEnvironment(),
        config=config,
        subcalls=subcalls,
    )

    _, result = _run_with(ReportingSubcalls(capacity), config)

    assert result.scaffold_manifest is not None
    assert preflight.scaffold_manifest == result.scaffold_manifest
    assert preflight.scaffold_manifest.body["inference"]["input_capacity"] == {
        "subcall": capacity.as_dict()
    }


def test_runner_preflight_records_declared_capacity() -> None:
    response = run_worker(
        {
            "protocol_version": RUNNER_PROTOCOL_VERSION,
            "operation": "preflight",
            "model": "root-model",
            "budget": Budget().as_dict(),
            "subcall_input_capacity": SubcallInputCapacity.bounded(200_000).as_dict(),
        }
    )

    assert response["status"] == "success"
    assert response["preflight"]["scaffold_manifest"]["inference"]["input_capacity"] == {
        "subcall": {"state": "bounded", "tokens": 200_000}
    }


def test_runner_v5_is_refused_before_trace_v5_can_be_ignored() -> None:
    response = run_worker(
        {
            "protocol_version": 5,
            "operation": "preflight",
            "model": "root-model",
            "budget": Budget().as_dict(),
            "subcall_input_capacity": SubcallInputCapacity.bounded(200_000).as_dict(),
        }
    )

    assert response["status"] == "refusal"
    assert response["error"]["code"] == "protocol_version_mismatch"
    assert response["error"]["details"] == {"requested": 5, "supported": 9}


def test_scaffold_v1_round_trip_remains_explicitly_supported() -> None:
    path = Path(__file__).parent / "fixtures" / "scaffold_manifest_v1.json"
    legacy = json.loads(path.read_text(encoding="utf-8"))

    parsed = ScaffoldManifest.from_dict(legacy)

    assert parsed.schema_version == 1
    assert parsed.as_dict() == legacy
    assert (
        parsed.manifest_id
        == "sha256:57beba12b22dc080c562586606781abc07644c3bd8b20333fc98000d7319da26"
    )


def test_scaffold_v2_requires_one_strict_input_capacity_field() -> None:
    _, result = _run_with(MockSubcallClient())
    assert result.scaffold_manifest is not None
    missing = result.scaffold_manifest.as_dict()
    missing["inference"].pop("input_capacity")
    with pytest.raises(ValueError, match="missing input_capacity"):
        ScaffoldManifest.from_dict(missing)

    extra = result.scaffold_manifest.as_dict()
    extra["inference"]["input_capacity"]["extra"] = None
    with pytest.raises(ValueError, match="unknown extra"):
        ScaffoldManifest.from_dict(extra)

    invalid = result.scaffold_manifest.as_dict()
    invalid["inference"]["input_capacity"]["subcall"] = {
        "state": "bounded",
        "tokens": 0,
    }
    with pytest.raises(ValueError, match="positive"):
        ScaffoldManifest.from_dict(invalid)


def test_rollout_capacity_wire_round_trip_and_legacy_default() -> None:
    rollout = RolloutConfiguration(
        subcall_input_capacity=SubcallInputCapacity.unbounded(),
    )

    assert RolloutConfiguration.from_dict(rollout.as_dict()) == rollout
    legacy = rollout.as_dict()
    legacy.pop("subcall_input_capacity")
    assert (
        RolloutConfiguration.from_dict(legacy).subcall_input_capacity
        == SubcallInputCapacity.unknown()
    )
