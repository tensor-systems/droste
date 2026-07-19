from __future__ import annotations

from droste import (
    Budget,
    ConfiguredSource,
    EnvironmentConfig,
    ProviderCatalog,
    RLMConfig,
    RLMSkillCatalog,
    create_environment,
    create_environment_context,
    load_builtin_skill_catalog,
    parse_rlm_skill,
    rlm_skills_provider,
    run_rlm,
)
from droste.protocols.llm_client import TokenUsage
from droste.testing import MockEnvironment, MockLLMClient, MockResponse, MockSubcallClient


def _response(code: str) -> MockResponse:
    return MockResponse(
        text=f"```python\n{code}\n```",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, exact=True),
    )


def test_builtin_skills_are_immutable_content_addressed_artifacts() -> None:
    catalog = load_builtin_skill_catalog()

    assert isinstance(catalog, RLMSkillCatalog)
    assert [skill.skill_id for skill in catalog.skills] == [
        "droste.chunking",
        "droste.decomposition.example",
    ]
    assert all(skill.content_hash.startswith("sha256:") for skill in catalog.skills)
    assert parse_rlm_skill(
        "+++\n"
        "schema_version=1\n"
        'id="x"\nrevision="1"\nsummary="s"\nmodel_families=["qwen"]\n'
        '[provenance]\nsource="test"\n'
        "+++\nbody"
    ).model_families == ("qwen",)


def test_model_specific_skill_selection_never_becomes_universal() -> None:
    generic, example = load_builtin_skill_catalog().skills
    qwen = parse_rlm_skill(
        "+++\n"
        "schema_version=1\n"
        'id="qwen.decompose"\nrevision="1"\nsummary="q"\nmodel_families=["qwen"]\n'
        '[provenance]\nsource="test"\n'
        "+++\nqwen-only body"
    )
    catalog = RLMSkillCatalog((generic, example, qwen))

    assert qwen in catalog.list(model_family="qwen")
    assert qwen not in catalog.list(model_family="openai")
    assert qwen in catalog.list()


def test_skills_are_loaded_mid_run_through_the_capability_broker() -> None:
    budget = Budget()
    subcalls = MockSubcallClient()
    context = create_environment_context(EnvironmentConfig(kind="native", budget=budget))
    registration = rlm_skills_provider(load_builtin_skill_catalog())
    registry = ProviderCatalog((registration,)).bind((ConfiguredSource("skills", "rlm_skills"),))
    environment = create_environment(
        EnvironmentConfig(kind="native", budget=budget),
        context={"records": [1, 2, 3]},
        registry=registry,
        subcalls=subcalls,
        execution_context=context,
        capability_run_id=context.trace.run_id,
        capability_observer=context.observe_capability,
    )
    llm = MockLLMClient(
        [
            _response(
                "items = skills.available()\n"
                "chosen = skills.load('droste.chunking', '1.0.0')\n"
                "answer['content'] = chosen['id']\n"
                "answer['ready'] = True"
            )
        ]
    )

    result = run_rlm(
        "load a strategy",
        environment=environment,
        root_llm=llm,
        subcalls=subcalls,
        config=RLMConfig(budget=budget, root_model="model"),
        context=context,
    )

    assert result.answer == "droste.chunking"
    assert result.sub_calls_made == 0
    assert result.run_record is not None
    capability_events = [event for event in result.run_record.events if event.type == "capability"]
    assert [event.body["outcome"]["capability_id"]["operation"] for event in capability_events] == [
        "skills.list",
        "skills.load",
    ]
    assert all(event.body["outcome"]["status"] == "ok" for event in capability_events)


def test_prompt_profiles_swap_strategy_without_changing_runtime_contracts() -> None:
    manifests = []
    for profile in ("full", "minimal", "none"):
        result = run_rlm(
            "q",
            environment=MockEnvironment(),
            root_llm=MockLLMClient([_response("answer['content']='ok'; answer['ready']=True")]),
            subcalls=MockSubcallClient(),
            config=RLMConfig(root_model="model", prompt_profile=profile),
        )
        assert result.scaffold_manifest is not None
        manifests.append(result.scaffold_manifest.body)

    assert len({body["prompt_pack"]["content_hash"] for body in manifests}) == 3
    assert len({body["capabilities"]["manifest_hash"] for body in manifests}) == 1
    assert len({body["contracts"]["terminal"] for body in manifests}) == 1
    assert len({tuple(body["capabilities"]["model_visible_globals"]) for body in manifests}) == 1
