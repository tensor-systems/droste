from __future__ import annotations

import copy
import subprocess
import sys
import tomllib
from dataclasses import FrozenInstanceError, replace
from importlib.resources import files
from typing import Any

import pytest

from droste import PolicyHints, RLMConfig, run_rlm
from droste.exceptions import PolicyError
from droste.loop.step import error_repair_history
from droste.prompts import (
    PROMPT_SLOT_NAMES,
    PromptPackBinding,
    PromptPackCatalog,
    PromptPackError,
    PromptPolicyDefaults,
    PromptSlots,
    load_builtin_prompt_catalog,
    load_prompt_pack,
    parse_prompt_pack,
    render_prompt_template,
    resolve_prompt_pack,
)
from droste.prompts.pack import CODE_OUTPUT_CONTRACT, load_prompt_pack_resource
from droste.protocols.llm_client import TokenUsage
from droste.testing import MockEnvironment, MockLLMClient, MockResponse, MockSubcallClient


def _artifact_data() -> dict[str, Any]:
    resource = files("droste.prompts").joinpath("packs", "generic-full-v1.toml")
    return tomllib.loads(resource.read_text(encoding="utf-8"))


def _response(text: str) -> MockResponse:
    return MockResponse(
        text=text,
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


class RecordingLLMClient(MockLLMClient):
    def __init__(self, responses: list[MockResponse]) -> None:
        super().__init__(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def responses_create(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
        return_usage: bool = False,
    ) -> str | tuple[str, TokenUsage]:
        self.calls.append([dict(message) for message in messages])
        return super().responses_create(
            messages,
            model,
            max_tokens=max_tokens,
            temperature=temperature,
            return_usage=return_usage,
        )


class ReportingSubcalls(MockSubcallClient):
    def __init__(self, output_token_limit: int | None, *, successful: bool = False) -> None:
        super().__init__()
        self._output_token_limit = output_token_limit
        self._successful = successful

    @property
    def output_token_limit(self) -> int | None:
        return self._output_token_limit

    def llm_query(self, prompt: str, context: str = "") -> str:
        result = super().llm_query(prompt, context)
        if self._successful and self._context is not None:
            self._context.stats.successful_calls += 1
            return "ok"
        return result


class UnknownCustomSubcalls(MockSubcallClient):
    """A compatible custom client that does not opt into limit metadata."""


def _root_system_prompt(
    subcalls: MockSubcallClient,
    *,
    semantic: bool = False,
) -> str:
    code = "value = llm_query('q')\n" if semantic else ""
    code += "answer['content'] = 'ok'\nanswer['ready'] = True"
    llm = RecordingLLMClient([_response(f"```python\n{code}\n```")])
    run_rlm(
        "q",
        environment=MockEnvironment(),
        root_llm=llm,
        subcalls=subcalls,
        config=RLMConfig(
            max_iterations=1,
            max_calls=7,
            max_depth=2,
            max_output_chars=99,
            policy_hints=PolicyHints(semantic=True) if semantic else None,
        ),
    )
    return str(llm.calls[0][0]["content"])


@pytest.mark.parametrize(
    ("subcalls", "rendered_limit"),
    [
        (ReportingSubcalls(512), "512 (bounded)"),
        (ReportingSubcalls(None), "unbounded (deliberate)"),
        (UnknownCustomSubcalls(), "unknown (client did not report)"),
    ],
)
def test_root_authorized_compute_renders_output_limit_states_exactly(
    subcalls: MockSubcallClient,
    rendered_limit: str,
) -> None:
    expected = (
        "## Authorized compute\n"
        "iterations=1; subcalls=7; depth=2; output_chars_per_iteration=99\n"
        f"subcall_output_tokens_per_call={rendered_limit}"
    )

    system_prompt = _root_system_prompt(subcalls)
    actual = (
        "## Authorized compute\n"
        + system_prompt.split("## Authorized compute\n", 1)[1].split("\n\n## Tips", 1)[0]
    )

    assert actual == expected
    assert system_prompt.count("subcall_output_tokens_per_call=") == 1


def test_semantic_subcall_gate_forwards_output_limit_to_root_prompt() -> None:
    system_prompt = _root_system_prompt(ReportingSubcalls(768, successful=True), semantic=True)

    assert "subcall_output_tokens_per_call=768 (bounded)" in system_prompt


def test_builtin_catalog_loads_complete_immutable_profiles_from_resources() -> None:
    catalog = load_builtin_prompt_catalog()

    assert catalog is load_builtin_prompt_catalog()
    assert tuple(binding.profile for binding in catalog.bindings) == ("full", "minimal", "none")
    assert all(binding.model_family == "generic" for binding in catalog.bindings)
    assert all(binding.pack.schema_version == 1 for binding in catalog.bindings)
    assert all(binding.pack.provenance.source == "droste" for binding in catalog.bindings)
    assert len(catalog.bindings[0].pack.tips) > len(catalog.bindings[1].pack.tips)
    assert catalog.bindings[2].pack.tips == ()
    assert len({binding.pack.templates for binding in catalog.bindings}) == 1
    assert len({binding.pack.policy_defaults for binding in catalog.bindings}) == 1
    assert len({binding.pack.unable_sentinel for binding in catalog.bindings}) == 1
    with pytest.raises(FrozenInstanceError):
        catalog.bindings[0].pack.revision = "mutated"  # type: ignore[misc]


def test_importing_droste_does_not_materialize_legacy_prompt_projections() -> None:
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import droste; "
            "assert 'droste.prompts.base' not in sys.modules; "
            "assert 'droste.prompts.tips' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert check.returncode == 0, check.stderr


def test_policy_error_history_preserves_guidance_for_subclasses() -> None:
    class ConsumerPolicyError(PolicyError):
        pass

    history = error_repair_history(ConsumerPolicyError("consumer policy"))

    assert "type=ConsumerPolicyError" in history
    assert "answer['content'] was kept" in history
    assert 'set answer["ready"] = True again' in history


def test_installed_package_resource_loader_rejects_path_traversal() -> None:
    assert load_prompt_pack_resource("generic-full-v1.toml").pack_id == "droste.generic.full"
    with pytest.raises(PromptPackError, match="plain .toml name"):
        load_prompt_pack_resource("../generic-full-v1.toml")


def test_caller_file_loader_has_one_io_boundary(tmp_path: Any) -> None:
    artifact = tmp_path / "caller.toml"
    artifact.write_text(
        files("droste.prompts")
        .joinpath("packs", "generic-minimal-v1.toml")
        .read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert load_prompt_pack(artifact).pack_id == "droste.generic.minimal"

    broken = tmp_path / "broken.toml"
    broken.write_text("not = [valid", encoding="utf-8")
    with pytest.raises(PromptPackError, match="cannot load prompt pack"):
        load_prompt_pack(broken)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update(schema_version=2), "schema_version must be 1"),
        (
            lambda data: data["policy_defaults"].update(enforce_contract="yes"),
            "enforce_contract must be a boolean",
        ),
        (
            lambda data: data["provenance"].update(benchmark="bench-v1"),
            "benchmark and score must be provided together",
        ),
        (
            lambda data: data["templates"].update(
                system=data["templates"]["system"] + "\n{unknown_slot}"
            ),
            "uses unknown slot {unknown_slot}",
        ),
        (
            lambda data: data["templates"].update(user="Question without a slot"),
            "missing required slots",
        ),
    ],
)
def test_parse_prompt_pack_fails_closed_before_run(mutate: Any, message: str) -> None:
    data = copy.deepcopy(_artifact_data())
    mutate(data)
    with pytest.raises(PromptPackError, match=message):
        parse_prompt_pack(data, source="test-pack")


def test_stable_slot_contract_renders_values_without_interpreting_their_braces() -> None:
    assert PROMPT_SLOT_NAMES == {
        "capabilities",
        "budget",
        "question",
        "history",
        "output_contract",
    }
    rendered = render_prompt_template(
        "Q={question}; H={history}",
        PromptSlots(question="why {not_a_slot}?", history='result={"ok": true}'),
    )
    assert rendered == 'Q=why {not_a_slot}?; H=result={"ok": true}'


def test_catalog_rejects_duplicate_selectors_and_mismatched_profiles() -> None:
    pack = load_prompt_pack_resource("generic-full-v1.toml")
    binding = PromptPackBinding("OpenAI", "FULL", pack)
    assert (binding.model_family, binding.profile) == ("openai", "full")
    with pytest.raises(PromptPackError, match="duplicate selectors"):
        PromptPackCatalog((binding, binding))
    with pytest.raises(PromptPackError, match="does not match pack profile"):
        PromptPackBinding("openai", "minimal", pack)


def test_catalog_and_pack_copy_caller_owned_sequences() -> None:
    pack = load_prompt_pack_resource("generic-none-v1.toml")
    source_tips = ["one"]
    copied_pack = replace(pack, tips=source_tips)  # type: ignore[arg-type]
    source_tips.append("two")
    assert copied_pack.tips == ("one",)

    binding = PromptPackBinding("generic", "none", pack)
    source_bindings = [binding]
    catalog = PromptPackCatalog(source_bindings)  # type: ignore[arg-type]
    source_bindings.clear()
    assert catalog.bindings == (binding,)


def test_resolution_is_deterministic_across_every_fallback_tier() -> None:
    builtins = load_builtin_prompt_catalog()
    full = builtins.bindings[0].pack
    consumer_family = replace(full, pack_id="consumer.openai")
    consumer_generic = replace(full, pack_id="consumer.generic")
    engine_family = replace(full, pack_id="engine.anthropic")
    caller = replace(full, pack_id="caller.explicit")
    consumers = PromptPackCatalog(
        (
            PromptPackBinding("openai", "full", consumer_family),
            PromptPackBinding("generic", "full", consumer_generic),
        )
    )
    engine = PromptPackCatalog(
        builtins.bindings + (PromptPackBinding("anthropic", "full", engine_family),)
    )

    resolved = resolve_prompt_pack(
        model="gpt-5", profile="full", consumer_catalog=consumers, engine_catalog=engine
    )
    assert (resolved.pack.pack_id, resolved.tier) == (
        "consumer.openai",
        "consumer_model_family",
    )

    resolved = resolve_prompt_pack(
        model="custom-model",
        profile="full",
        consumer_catalog=consumers,
        engine_catalog=engine,
    )
    assert (resolved.pack.pack_id, resolved.tier) == (
        "consumer.generic",
        "consumer_generic",
    )

    resolved = resolve_prompt_pack(model="claude-opus-4-8", profile="full", engine_catalog=engine)
    assert (resolved.pack.pack_id, resolved.tier) == (
        "engine.anthropic",
        "engine_model_family",
    )

    resolved = resolve_prompt_pack(
        model="unknown-model", profile="not-defined", engine_catalog=engine
    )
    assert (resolved.pack.pack_id, resolved.pack.profile, resolved.tier) == (
        "droste.generic.full",
        "full",
        "generic",
    )

    resolved = resolve_prompt_pack(
        model="gpt-5",
        profile="full",
        caller_pack=caller,
        consumer_catalog=consumers,
        engine_catalog=engine,
    )
    assert (resolved.pack.pack_id, resolved.tier) == ("caller.explicit", "caller")
    assert resolve_prompt_pack(
        model="gpt-5", profile="full", consumer_catalog=consumers, engine_catalog=engine
    ) == resolve_prompt_pack(
        model="gpt-5", profile="full", consumer_catalog=consumers, engine_catalog=engine
    )


def test_directly_constructed_caller_pack_is_validated_before_resolution() -> None:
    base = load_prompt_pack_resource("generic-full-v1.toml")
    invalid = replace(base, templates=replace(base.templates, user="no slots"))
    with pytest.raises(PromptPackError, match="missing required slots"):
        resolve_prompt_pack(
            model="gpt-5",
            caller_pack=invalid,
            engine_catalog=load_builtin_prompt_catalog(),
        )


def test_run_uses_one_caller_pack_and_records_its_provenance() -> None:
    base = load_prompt_pack_resource("generic-none-v1.toml")
    templates = replace(
        base.templates,
        system="CALLER PACK\n{output_contract}\n{capabilities}\n{budget}",
        user="CUSTOM QUESTION={question}\n{history}",
    )
    caller = replace(base, pack_id="consumer.custom", revision="7", templates=templates)
    llm = RecordingLLMClient(
        [_response("```python\nanswer['content'] = 'ok'\nanswer['ready'] = True\n```")]
    )

    result = run_rlm(
        "q",
        environment=MockEnvironment(),
        root_llm=llm,
        subcalls=MockSubcallClient(),
        config=RLMConfig(max_iterations=1, root_model="gpt-5"),
        prompt_pack=caller,
    )

    assert llm.calls[0][0]["content"].startswith("CALLER PACK")
    assert llm.calls[0][1]["content"] == "CUSTOM QUESTION=q"
    assert result.prompt_pack is not None
    assert result.prompt_pack.as_dict() == {
        "id": "consumer.custom",
        "revision": "7",
        "profile": "none",
        "resolution_tier": "caller",
        "model_family": "openai",
        "provenance_source": "droste",
        "provenance_benchmark": None,
        "provenance_score": None,
    }


def test_pack_policy_default_applies_only_when_config_does_not_override_it() -> None:
    base = load_prompt_pack_resource("generic-none-v1.toml")
    permissive = replace(
        base,
        pack_id="caller.permissive",
        policy_defaults=PromptPolicyDefaults(enforce_contract=False),
    )
    prose = "plain response without code"
    result = run_rlm(
        "q",
        environment=MockEnvironment(),
        root_llm=MockLLMClient([_response(prose)]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(max_iterations=1),
        prompt_pack=permissive,
    )
    assert result.answer == prose
    assert result.error is None

    strict = run_rlm(
        "q",
        environment=MockEnvironment(),
        root_llm=MockLLMClient([_response(prose)]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(max_iterations=1, enforce_contract=True),
        prompt_pack=permissive,
    )
    assert strict.error is not None


def test_generic_pack_preserves_current_default_contract_and_tips() -> None:
    full = load_prompt_pack_resource("generic-full-v1.toml")
    system = render_prompt_template(
        full.templates.system,
        PromptSlots(
            capabilities="context is available",
            budget="iterations=20",
            output_contract=CODE_OUTPUT_CONTRACT,
        ),
    )
    normalized = " ".join(system.split())
    assert "validator(value, index)" in normalized
    assert "raise ValueError to reject that value and request repair" in normalized
    assert "context is available" in system
    assert "iterations=20" in system
    assert any("orchestrator, not a solver" in tip for tip in full.tips)
