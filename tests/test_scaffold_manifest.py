from __future__ import annotations

from dataclasses import replace

import pytest

from droste import (
    DEFAULT_SUBCALL_CONCURRENCY,
    Budget,
    EngineIdentity,
    RLMConfig,
    RLMPreflight,
    RolloutConfiguration,
    SandboxLimits,
    ScaffoldCompatibilityError,
    ScaffoldManifest,
    ScaffoldRequirements,
    preflight_rlm,
    run_rlm,
)
from droste.capabilities import (
    JSON_SCHEMA_2020_12,
    CapabilityDescriptor,
    CapabilityId,
    CapabilityKind,
    CapabilityManifest,
    PaginationMode,
    ProviderOperation,
    ResultDelivery,
    SchemaSpec,
    SideEffect,
)
from droste.environments import EnvironmentConfig, create_environment, create_environment_context
from droste.execution.manifest import build_scaffold_manifest
from droste.prompts import load_builtin_prompt_catalog, resolve_prompt_pack
from droste.protocols.llm_client import TokenUsage
from droste.testing import MockEnvironment, MockLLMClient, MockResponse, MockSubcallClient


def _pack():
    return resolve_prompt_pack(
        model="root-model",
        engine_catalog=load_builtin_prompt_catalog(),
    ).pack


def _manifest(**overrides):
    values = {
        "engine": EngineIdentity("1.2.3", "commit-a"),
        "prompt_pack": _pack(),
        "capability_manifest": CapabilityManifest(),
        "provider_protocol": 3,
        "model_visible_globals": ("llm_query", "context", "answer"),
        "root_model": "root-model",
        "rollout": RolloutConfiguration(
            root_revision="root-rev",
            subcall_model="leaf-model",
            subcall_revision="leaf-rev",
            root_sampling={"temperature": 0.2, "stop": ["END"]},
            subcall_sampling={"temperature": 0},
            concurrency=4,
            seed=7,
            runner_protocol=3,
            source_revision="commit-a",
        ),
        "budget": Budget(),
        "sandbox": SandboxLimits(),
    }
    values.update(overrides)
    return build_scaffold_manifest(**values)


def _ready_response(answer: str = "ok") -> MockResponse:
    return MockResponse(
        text=f"```python\nanswer['content'] = {answer!r}\nanswer['ready'] = True\n```",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def test_equivalent_manifests_are_byte_stable_and_ignore_mapping_order() -> None:
    left = _manifest()
    reordered = replace(
        left,
        body={key: left.body[key] for key in reversed(tuple(left.body))},
    )

    assert left.as_dict() == reordered.as_dict()
    assert left.manifest_id == reordered.manifest_id
    assert "output_truncation" not in left.body["contracts"]


def test_manifest_from_dict_is_strict_and_verifies_claimed_id() -> None:
    manifest = _manifest()
    wire = {**manifest.as_dict(), "id": manifest.manifest_id}

    assert ScaffoldManifest.from_dict(wire) == manifest

    missing = dict(wire)
    missing.pop("contracts")
    with pytest.raises(ValueError, match="missing contracts"):
        ScaffoldManifest.from_dict(missing)

    with pytest.raises(ValueError, match="unknown host_metadata"):
        ScaffoldManifest.from_dict({**wire, "host_metadata": {}})

    wrong_type = {**wire, "abis": {**wire["abis"], "kernel": "1"}}
    with pytest.raises(TypeError, match="abis.kernel"):
        ScaffoldManifest.from_dict(wrong_type)

    with pytest.raises(ValueError, match="id does not match"):
        ScaffoldManifest.from_dict({**wire, "id": "sha256:" + "0" * 64})


def test_host_opaque_model_identity_is_explicit_null_not_empty_string() -> None:
    manifest = _manifest(root_model=None, rollout=RolloutConfiguration())

    assert manifest.body["inference"]["root"] == {"id": None, "revision": None}
    assert manifest.body["inference"]["subcall"] == {"id": None, "revision": None}

    wire = manifest.as_dict()
    wire["inference"]["root"]["id"] = ""
    with pytest.raises(ValueError, match="inference.root.id"):
        ScaffoldManifest.from_dict(wire)


def test_rollout_concurrency_has_an_explicit_compatibility_default() -> None:
    assert RolloutConfiguration().concurrency == DEFAULT_SUBCALL_CONCURRENCY == 5


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2"])
def test_rollout_concurrency_rejects_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="subcall concurrency"):
        RolloutConfiguration(concurrency=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "change",
    [
        {"root_model": "other-root"},
        {"budget": Budget(subcalls=49)},
        {"sandbox": SandboxLimits(output_chars=24_999)},
        {"system_prompt_override": "different static scaffold"},
        {
            "rollout": RolloutConfiguration(
                root_revision="root-rev-2",
                subcall_model="leaf-model",
            )
        },
    ],
)
def test_material_scaffold_changes_change_manifest_id(change: dict[str, object]) -> None:
    assert _manifest(**change).manifest_id != _manifest().manifest_id


def test_prompt_content_change_changes_manifest_without_revision_change() -> None:
    pack = _pack()
    changed = replace(pack, tips=(*pack.tips, "A new strategy tip."))

    assert _manifest(prompt_pack=changed).manifest_id != _manifest().manifest_id


def test_capability_change_changes_manifest() -> None:
    parameters = SchemaSpec({"type": "object"}, JSON_SCHEMA_2020_12, "test:inspect/parameters@1")
    operation = ProviderOperation(
        "inspect",
        "inspect",
        "Inspect test data.",
        parameters,
        None,
        PaginationMode.NONE,
        ResultDelivery.UNTYPED,
        "test.read",
    )
    descriptor = CapabilityDescriptor(
        CapabilityId(
            kind=CapabilityKind.DATA,
            operation="inspect",
            source_id="source",
            provider_type="test",
        ),
        operation,
        SideEffect.READ,
        "1",
        "sha256:" + "1" * 64,
    )

    assert (
        _manifest(capability_manifest=CapabilityManifest((descriptor,))).manifest_id
        != _manifest().manifest_id
    )


def test_partial_compatibility_mismatch_is_typed_and_actionable() -> None:
    manifest = _manifest()
    requirements = ScaffoldRequirements(
        required={"inference": {"root": {"revision": "expected-revision"}}}
    )

    with pytest.raises(ScaffoldCompatibilityError) as exc_info:
        from droste import require_scaffold_compatibility

        require_scaffold_compatibility(manifest, requirements)

    assert exc_info.value.mismatches[0].path == "inference.root.revision"
    assert exc_info.value.mismatches[0].actual == "root-rev"


def test_preflight_value_round_trips_strictly() -> None:
    value = RLMPreflight(_manifest())

    assert RLMPreflight.from_dict(value.as_dict()) == value
    with pytest.raises(ValueError, match="unknown host_metadata"):
        RLMPreflight.from_dict({**value.as_dict(), "host_metadata": {}})


def test_preflight_closes_environment_once_on_success_and_mismatch() -> None:
    class CloseTrackingEnvironment(MockEnvironment):
        def __init__(self) -> None:
            super().__init__()
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    success_environment = CloseTrackingEnvironment()
    preflight_rlm(environment=success_environment, config=RLMConfig(root_model="root-model"))
    assert success_environment.close_count == 1

    mismatch_environment = CloseTrackingEnvironment()
    with pytest.raises(ScaffoldCompatibilityError):
        preflight_rlm(
            environment=mismatch_environment,
            config=RLMConfig(
                root_model="root-model",
                checkpoint_requirements=ScaffoldRequirements(
                    required={"prompt_pack": {"revision": "not-this-revision"}}
                ),
            ),
        )
    assert mismatch_environment.close_count == 1


def test_checkpoint_manifest_id_requires_lowercase_sha256_hex() -> None:
    for invalid in (
        "sha256:" + "g" * 64,
        "sha256:" + "A" * 64,
        "sha256:" + "0" * 63,
    ):
        with pytest.raises(ValueError, match="sha256 digest"):
            ScaffoldRequirements(manifest_id=invalid)


def test_run_rejects_incompatible_checkpoint_before_first_llm_request() -> None:
    root = MockLLMClient([_ready_response()])

    with pytest.raises(ScaffoldCompatibilityError):
        run_rlm(
            "question",
            environment=MockEnvironment(),
            root_llm=root,
            subcalls=MockSubcallClient(),
            config=RLMConfig(
                root_model="root-model",
                checkpoint_requirements=ScaffoldRequirements(
                    required={"prompt_pack": {"revision": "not-this-revision"}}
                ),
            ),
        )

    assert root._call_count == 0


def test_run_rejects_builtin_concurrency_drift_before_first_llm_request() -> None:
    class ReportingSubcalls(MockSubcallClient):
        @property
        def subcall_concurrency(self) -> int:
            return 2

    root = MockLLMClient([_ready_response()])

    with pytest.raises(ValueError, match="rollout concurrency does not match"):
        run_rlm(
            "question",
            environment=MockEnvironment(),
            root_llm=root,
            subcalls=ReportingSubcalls(),
            config=RLMConfig(
                root_model="root-model",
                rollout=RolloutConfiguration(concurrency=3),
            ),
        )

    assert root._call_count == 0


def test_run_exposes_manifest_but_durable_terminal_keeps_identity_only() -> None:
    marker = "PRIVATE_QUESTION_MARKER_101"
    result = run_rlm(
        marker,
        environment=MockEnvironment(),
        root_llm=MockLLMClient([_ready_response()]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(root_model="root-model"),
    )

    assert result.scaffold_manifest is not None
    assert result.run_record is not None
    terminal = dict(result.run_record.terminal)
    assert terminal["scaffold_manifest_id"] == result.scaffold_manifest.manifest_id
    assert terminal["scaffold_manifest_version"] == 2
    assert marker not in str(result.scaffold_manifest.as_dict())
    assert marker not in str(result.run_record.as_dict())


def test_native_and_pyodide_preflight_matches_execution_without_subcalls() -> None:
    budget = Budget()
    sandbox = SandboxLimits(execution_timeout_ms=0)
    rollout = RolloutConfiguration(
        root_revision="root-rev",
        subcall_model="leaf-model",
        subcall_revision="leaf-rev",
        root_sampling={"temperature": 0.1},
        subcall_sampling={"temperature": 0},
        concurrency=2,
        seed=17,
        source_revision="commit-a",
    )
    manifests = []
    for kind in ("native", "pyodide"):
        config = EnvironmentConfig(
            kind=kind,
            budget=budget,
            sandbox=sandbox,
            host_managed_timeout=kind == "pyodide",
            host_managed_isolation=kind == "pyodide",
        )
        preflight_context = create_environment_context(config)

        class CountingSubcalls(MockSubcallClient):
            calls = 0

            def llm_query(self, prompt: str, context: str = "") -> str:
                self.calls += 1
                return super().llm_query(prompt, context)

            def llm_batch(self, prompts: list[str], contexts: list[str] | None = None) -> list[str]:
                self.calls += 1
                return super().llm_batch(prompts, contexts)

        preflight_subcalls = CountingSubcalls()
        preflight_environment = create_environment(
            config,
            context={"private": "PREVIEW_MUST_NOT_APPEAR"},
            registry=None,
            subcalls=preflight_subcalls,
            execution_context=preflight_context,
        )
        preflight = preflight_rlm(
            environment=preflight_environment,
            config=RLMConfig(
                budget=budget,
                sandbox=sandbox,
                root_model="root-model",
                rollout=rollout,
            ),
        )
        assert preflight_subcalls.calls == 0
        assert "PREVIEW_MUST_NOT_APPEAR" not in str(preflight.as_dict())

        execution_context = create_environment_context(config)
        execution_subcalls = MockSubcallClient()
        execution_environment = create_environment(
            config,
            context={"records": []},
            registry=None,
            subcalls=execution_subcalls,
            execution_context=execution_context,
        )
        result = run_rlm(
            "q",
            environment=execution_environment,
            root_llm=MockLLMClient([_ready_response()]),
            subcalls=execution_subcalls,
            config=RLMConfig(
                budget=budget,
                sandbox=sandbox,
                root_model="root-model",
                rollout=rollout,
            ),
            context=execution_context,
        )
        assert result.scaffold_manifest is not None
        assert preflight.scaffold_manifest == result.scaffold_manifest
        manifests.append(preflight.scaffold_manifest)

    assert manifests[0].manifest_id == manifests[1].manifest_id
    assert manifests[0].as_dict() == manifests[1].as_dict()
