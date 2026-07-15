"""Provider manifest, configured-source, and registry conformance."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import FrozenInstanceError, replace

import pytest

from droste import (
    ConfiguredSource,
    EnvironmentConfig,
    ProviderCatalog,
    ProviderManifest,
    SideEffect,
    create_environment,
    create_environment_context,
)
from droste.sources.sql_local import sqlite_provider
from droste.testing import MockSubcallClient, fake_records_provider
from droste_runner.sources import (
    build_opened_provider_registry,
    build_provider_registry,
    default_provider_catalog,
    wrapper_provider,
)


def _database(tmp_path) -> str:
    path = tmp_path / "records.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE entries (id INTEGER, title TEXT)")
    conn.execute("INSERT INTO entries VALUES (1, 'sqlite row')")
    conn.commit()
    conn.close()
    return str(path)


def test_manifest_round_trip_digest_and_immutable_snapshot() -> None:
    manifest = fake_records_provider().manifest
    wire = manifest.to_dict()

    assert ProviderManifest.from_dict(json.loads(json.dumps(wire))) == manifest
    assert manifest.digest.startswith("sha256:")
    assert manifest.operations[0].operation_id == "records.search"
    assert manifest.operations[0].binding_name == "search"
    assert manifest.operations[0].parameters.dialect.endswith("2020-12/schema")
    assert manifest.operations[0].parameters.provenance
    with pytest.raises(FrozenInstanceError):
        manifest.revision = "2"  # type: ignore[misc]

    wire["revision"] = "tampered"
    with pytest.raises(ValueError, match="digest mismatch"):
        ProviderManifest.from_dict(wire)


def test_manifest_rejects_missing_schema_provenance_and_unsafe_binding() -> None:
    wire = fake_records_provider().manifest.to_dict()
    wire["operations"][0]["parameters"]["provenance"] = ""
    with pytest.raises(ValueError, match="provenance"):
        ProviderManifest.from_dict(wire)

    wire = fake_records_provider().manifest.to_dict()
    wire["operations"][0]["binding_name"] = "len"
    wire.pop("digest")
    with pytest.raises(ValueError, match="builtin"):
        ProviderManifest.from_dict(wire)


def test_configured_sources_and_catalog_are_explicit_and_fail_closed() -> None:
    source = ConfiguredSource.from_spec({"type": "fake_records", "name": "records", "page_size": 2})
    assert source.config_dict() == {"page_size": 2}
    with pytest.raises(FrozenInstanceError):
        source.source_id = "other"  # type: ignore[misc]
    with pytest.raises(ValueError, match="unknown provider"):
        ProviderCatalog((fake_records_provider(),)).bind((ConfiguredSource("db", "unknown"),))
    with pytest.raises(ValueError, match="duplicate provider"):
        ProviderCatalog((fake_records_provider(), fake_records_provider()))
    with pytest.raises(ValueError, match="duplicate configured"):
        ProviderCatalog((fake_records_provider(),)).bind((source, source))
    with pytest.raises(ValueError, match="default source"):
        ProviderCatalog((fake_records_provider(),)).bind((source,), default_source_id="missing")


def test_one_provider_registration_binds_multiple_named_sources() -> None:
    registry = ProviderCatalog((fake_records_provider(),)).bind(
        (
            ConfiguredSource("primary", "fake_records", {"page_size": 1}),
            ConfiguredSource("secondary", "fake_records", {"page_size": 2}),
        ),
        default_source_id="primary",
    )
    from droste.capabilities import CapabilityBroker

    globals_ = registry.broker_globals(CapabilityBroker(registry.capability_registrations()))
    assert globals_["primary"].fetch("1") == {"id": "1", "title": "alpha"}
    assert globals_["secondary"].fetch("1") == {"id": "1", "title": "alpha"}
    assert len(registry.capability_registrations()) == 4
    assert "SQL" not in registry.prompt_fragment()


def test_flattened_binding_cannot_overwrite_another_source_namespace() -> None:
    registry = ProviderCatalog((fake_records_provider(),)).bind(
        (
            ConfiguredSource("records", "fake_records"),
            ConfiguredSource("search", "fake_records"),
        ),
        default_source_id="records",
    )
    from droste.capabilities import CapabilityBroker

    with pytest.raises(ValueError, match="would overwrite a source namespace"):
        registry.broker_globals(CapabilityBroker(registry.capability_registrations()))


@pytest.mark.parametrize(
    "environment_config",
    [
        EnvironmentConfig(kind="native"),
        EnvironmentConfig(
            kind="pyodide",
            host_managed_timeout=True,
            host_managed_isolation=True,
        ),
    ],
)
def test_mixed_sql_and_non_sql_sources_have_identical_generic_projections(
    tmp_path, environment_config: EnvironmentConfig
) -> None:
    catalog = ProviderCatalog((sqlite_provider(), fake_records_provider()))
    registry = catalog.bind(
        (
            ConfiguredSource("db", "sqlite", {"sqlite_path": _database(tmp_path)}),
            ConfiguredSource("records", "fake_records"),
        ),
        default_source_id="records",
    )
    environment = create_environment(
        environment_config,
        context={},
        registry=registry,
        subcalls=MockSubcallClient(),
        execution_context=create_environment_context(environment_config),
    )
    globals_ = environment.globals()

    assert globals_["records"].search("alpha")["items"][0]["title"] == "alpha"
    assert globals_["search"]("alpha")["next_cursor"] is None
    assert globals_["db"].query("SELECT title FROM entries") == [{"title": "sqlite row"}]
    assert "records.search(query, cursor, limit)" in registry.prompt_fragment()
    assert "db.get_schema()" in registry.prompt_fragment()
    assert registry.accessor_manifest().flat == frozenset({"search", "fetch"})

    descriptors = environment.capability_broker().describe().descriptors
    raw_to_binding = {
        item.capability_id.operation: item.operation.binding_name
        for item in descriptors
        if item.capability_id.source_id is not None
    }
    assert raw_to_binding == {
        "query": "query",
        "schema": "get_schema",
        "records.search": "search",
        "records.fetch": "fetch",
    }


def test_descriptor_metadata_changes_do_not_change_capability_identity() -> None:
    registration = fake_records_provider()
    source = ConfiguredSource("records", "fake_records")
    first = ProviderCatalog((registration,)).bind((source,)).capability_registrations()[0]
    changed = type(registration)(
        manifest=registration.manifest,
        effects={"records.search": SideEffect.EFFECTFUL, "records.fetch": SideEffect.READ},
        binder=registration.binder,
        policy_metadata={"records.search": {"host_override": True}},
    )
    second = ProviderCatalog((changed,)).bind((source,)).capability_registrations()[0]

    assert first.descriptor.capability_id == second.descriptor.capability_id
    assert first.descriptor.side_effect is SideEffect.READ
    assert second.descriptor.side_effect is SideEffect.EFFECTFUL
    assert second.descriptor.to_dict()["policy_metadata"] == {"host_override": True}

    revised_manifest = replace(
        registration.manifest,
        revision="2",
        operations=(
            replace(
                registration.manifest.operations[0],
                description="Revised search documentation.",
                parameters=replace(
                    registration.manifest.operations[0].parameters,
                    provenance="droste:testing/fake/search/parameters@2",
                ),
            ),
            registration.manifest.operations[1],
        ),
    )
    revised = type(registration)(
        manifest=revised_manifest,
        effects=registration.effects,
        binder=registration.binder,
    )
    third = ProviderCatalog((revised,)).bind((source,)).capability_registrations()[0]

    assert third.descriptor.capability_id == first.descriptor.capability_id
    assert third.descriptor.provider_revision == "2"
    assert third.descriptor.provider_digest != first.descriptor.provider_digest
    assert third.descriptor.operation.description == "Revised search documentation."


def test_runner_binds_only_explicit_configured_sources() -> None:
    registry = build_provider_registry(
        {
            "data_sources": [
                {
                    "type": "wrapper_v1",
                    "name": "remote",
                    "base_url": "https://example.com",
                    "token": "secret",
                }
            ],
            "default_source": "remote",
        },
        catalog=default_provider_catalog(),
    )
    assert registry is not None
    assert registry.sources[0].registration.manifest == wrapper_provider().manifest
    with pytest.raises(ValueError, match="legacy data_source"):
        build_provider_registry(
            {"data_source": {"base_url": "https://example.com"}},
            catalog=default_provider_catalog(),
        )


def test_runner_trusted_source_opener_acquires_dynamic_manifest_for_run_or_preflight() -> None:
    observed: list[tuple[ConfiguredSource, object]] = []
    context = object()

    def opener(source: ConfiguredSource, source_context: object):
        observed.append((source, source_context))
        return fake_records_provider().bind(source)

    request = {
        "data_sources": [{"type": "fake_records", "name": "records"}],
        "default_source": "records",
    }
    registry = build_opened_provider_registry(request, opener=opener, context=context)
    assert registry is not None
    try:
        assert observed == [(ConfiguredSource("records", "fake_records"), context)]
        assert registry.sources[0].registration.manifest == fake_records_provider().manifest
    finally:
        registry.close()

    preflight = build_opened_provider_registry(
        request,
        opener=opener,
        context=context,
        dispatch_enabled=False,
    )
    assert preflight is not None
    try:
        with pytest.raises(RuntimeError, match="preflight cannot dispatch"):
            preflight.sources[0].runtime.handlers["records.search"]()
    finally:
        preflight.close()
