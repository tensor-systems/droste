"""Count-policy accessor discovery comes from provider descriptors."""

from __future__ import annotations

from droste import ConfiguredSource, ProviderCatalog
from droste.capabilities import CapabilityBroker
from droste.policy import PolicyHints, contract_violations
from droste.testing import MockSubcallClient, fake_records_provider
from droste_runner.runner import RunnerEnvironment

COUNT = PolicyHints(count=True)


def _registry(default_source_id: str | None = "records"):
    return ProviderCatalog((fake_records_provider(),)).bind(
        (ConfiguredSource("records", "fake_records"),),
        default_source_id=default_source_id,
    )


def test_len_over_descriptor_derived_flat_accessor_is_rejected() -> None:
    manifest = _registry().accessor_manifest()
    code = 'query("SELECT COUNT(*) FROM t")\nprint(len(search("alpha")))'
    assert contract_violations(code, COUNT, manifest.flat, manifest.namespaced)


def test_len_over_descriptor_derived_namespaced_accessor_is_rejected() -> None:
    manifest = _registry(default_source_id=None).accessor_manifest()
    code = 'query("SELECT COUNT(*) FROM t")\nprint(len(records.search("alpha")))'
    assert contract_violations(code, COUNT, manifest.flat, manifest.namespaced)


def test_plain_variables_and_arbitrary_receivers_are_not_accessors() -> None:
    manifest = _registry().accessor_manifest()
    assert (
        contract_violations(
            'rows = query("SELECT COUNT(*) FROM t")\nprint(len(rows))',
            COUNT,
            manifest.flat,
            manifest.namespaced,
        )
        == []
    )
    assert (
        contract_violations(
            'query("SELECT COUNT(*) FROM t")\nprint(len(row.get("items", [])))',
            COUNT,
            manifest.flat,
            manifest.namespaced,
        )
        == []
    )


def test_no_descriptor_accessors_means_no_fixed_verb_fallback() -> None:
    code = 'query("SELECT COUNT(*) FROM t")\nprint(len(search("x")))'
    assert contract_violations(code, COUNT) == []


def test_registry_namespace_contains_only_generated_bindings() -> None:
    registry = _registry()
    broker = CapabilityBroker(registry.capability_registrations())
    namespace = registry.broker_globals(broker)["records"]
    assert set(vars(namespace)) == {"search", "fetch"}
    assert not any(name.startswith("_droste") for name in vars(namespace))


def test_runner_environment_reports_provider_accessor_manifest() -> None:
    environment = RunnerEnvironment(
        context=None,
        registry=_registry(),
        subcalls=MockSubcallClient(),
        max_output_chars=10_000,
        exec_timeout_ms=0,
    )
    manifest = environment.accessor_manifest()
    assert manifest.flat == frozenset({"search", "fetch"})
    assert manifest.namespaced == frozenset({("records", "search"), ("records", "fetch")})

    empty = RunnerEnvironment(
        context=None,
        registry=None,
        subcalls=MockSubcallClient(),
        max_output_chars=10_000,
        exec_timeout_ms=0,
    ).accessor_manifest()
    assert not empty.flat and not empty.namespaced
