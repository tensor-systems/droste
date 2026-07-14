"""Count-contract enforcement over dynamic accessor names (#10).

The len()-over-accessor check must enforce against whatever verbs the
environment actually binds — including host-declared extras — not a
hardcoded list that went stale the day domain verbs moved out of the
engine (codex review on the #10 domain-blind change).
"""

from __future__ import annotations

from droste.capabilities import CapabilityBroker
from droste.policy import PolicyHints, contract_violations
from droste.registry import DataSourceRegistry

COUNT = PolicyHints(count=True)


def _broker_globals(registry: DataSourceRegistry):
    broker = CapabilityBroker(registry.capability_registrations())
    return registry.broker_globals(broker)


def test_len_over_declared_extra_is_rejected() -> None:
    # The aggregate alone satisfies uses_sql_aggregate; the len() over a
    # host extra must still trip the contract.
    code = 'query("SELECT COUNT(*) FROM t")\nprint(len(get_messages()))'
    violations = contract_violations(code, COUNT, data_accessors=("get_messages", "search"))
    assert violations, "len(get_messages()) must violate the count contract"
    assert "len()" in violations[0]


def test_len_over_namespaced_accessor_is_rejected() -> None:
    code = 'query("SELECT COUNT(*) FROM t")\nprint(len(db.search("x")))'
    violations = contract_violations(code, COUNT, namespaced_accessors=(("db", "search"),))
    assert violations


def test_len_over_plain_variable_is_fine() -> None:
    # len() over a local variable (or a non-accessor call) is legitimate
    # Python, not contract circumvention.
    code = 'rows = query("SELECT COUNT(*) FROM t")\nprint(len(rows))'
    assert contract_violations(code, COUNT, data_accessors=("get_messages",)) == []


def test_len_over_arbitrary_receiver_is_not_flagged() -> None:
    # A source exposing a verb named `get` must not make ordinary dict code
    # trip the contract (codex review): only the source's OWN namespace
    # qualifies, and unqualified matching applies only to flattened verbs.
    code = 'query("SELECT COUNT(*) FROM t")\nprint(len(row.get("items", [])))'
    assert (
        contract_violations(
            code,
            COUNT,
            data_accessors=("search",),
            namespaced_accessors=(("db", "get"),),
        )
        == []
    )
    # But the same verb under its real namespace still trips.
    code2 = 'query("SELECT COUNT(*) FROM t")\nprint(len(db.get("id1")))'
    assert contract_violations(code2, COUNT, namespaced_accessors=(("db", "get"),))


def test_accessor_matching_is_case_sensitive() -> None:
    # Python identifiers are case-sensitive: len(FETCH(...)) calls a distinct
    # local function, not the source's fetch accessor (codex review).
    code = 'query("SELECT COUNT(*) FROM t")\nprint(len(FETCH("x")))'
    assert contract_violations(code, COUNT, data_accessors=("fetch",)) == []


def test_flattened_extra_may_not_overwrite_another_sources_namespace() -> None:
    # A default source's flattened verb must never replace env["<other
    # source>"] (codex review).
    import pytest

    from droste.registry import DataSourceRegistry
    from droste.testing import MockDataSource

    class Archive(MockDataSource):
        def name(self):
            return "archive"

    class Default(MockDataSource):
        extra_methods = ("archive",)

        def archive(self):
            return []

    with pytest.raises(ValueError, match="overwrite a registered source"):
        _broker_globals(DataSourceRegistry([Archive(), Default()], default_source_name="mock"))


def test_static_fallback_when_no_accessors_supplied() -> None:
    # Callers that pass no accessor names keep the historical generic check.
    code = 'query("SELECT COUNT(*) FROM t")\nprint(len(search("x")))'
    assert contract_violations(code, COUNT)


def test_registry_accessor_manifest_is_explicit_data() -> None:
    # Accessor discovery reads the registry's own manifest (#31) — no
    # namespace marker, no callable-identity sniffing. Flat names come only
    # from the default source; namespaced pairs cover every bound verb,
    # including declared extras.
    from droste.registry import DataSourceRegistry
    from droste.testing import MockDataSource

    class Other(MockDataSource):
        extra_methods = ("get_threads",)

        def name(self):
            return "other"

        def get_threads(self):
            return []

    manifest = DataSourceRegistry(
        [MockDataSource(), Other()], default_source_name="mock"
    ).accessor_manifest()
    assert "search" in manifest.flat and "query" in manifest.flat
    assert ("mock", "search") in manifest.namespaced
    assert ("other", "get_threads") in manifest.namespaced
    # Non-default source's verbs are never flat.
    assert "get_threads" not in manifest.flat


def test_registry_namespaces_carry_no_marker_attributes() -> None:
    # The provenance marker died with #31: sandbox namespaces expose only
    # the source's verbs, nothing droste-internal to sniff.
    from droste.registry import DataSourceRegistry
    from droste.testing import MockDataSource

    env = _broker_globals(DataSourceRegistry([MockDataSource()], default_source_name="mock"))
    assert not any(attr.startswith("_droste") for attr in vars(env["mock"]))


def test_runner_environment_reports_registry_manifest() -> None:
    # The environment is the seam run_rlm reads the manifest through; without
    # a registry it reports the empty manifest (static policy fallback).
    from droste.registry import DataSourceRegistry
    from droste.testing import MockDataSource, MockSubcallClient
    from droste_runner.runner import RunnerEnvironment

    def _env(registry):
        return RunnerEnvironment(
            context=None,
            registry=registry,
            subcalls=MockSubcallClient(),
            max_output_chars=10000,
            exec_timeout_ms=0,
        )

    manifest = _env(
        DataSourceRegistry([MockDataSource()], default_source_name="mock")
    ).accessor_manifest()
    assert "search" in manifest.flat and ("mock", "query") in manifest.namespaced

    empty = _env(None).accessor_manifest()
    assert not empty.flat and not empty.namespaced
