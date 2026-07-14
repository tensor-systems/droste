# Provider manifests

Droste providers separate immutable description from live implementation.
There is no engine-wide data verb table and no process-global registry.

## Values and ownership

- `ProviderOperation` describes one source-agnostic operation: stable raw
  `operation_id`, Python `binding_name`, description, parameter and result
  `SchemaSpec`, pagination mode, delivery mode, and budget class.
- `ProviderManifest` groups operations for a `provider_type`, revision, and
  provider protocol. Its digest is the SHA-256 of the canonical manifest
  payload. Parsed manifests must reproduce the advertised digest.
- `ConfiguredSource` is only a source name, provider type, and frozen config.
  It contains no handlers.
- `ProviderRegistration` is host-owned. It pairs a manifest with an exact
  read/effectful classification, optional policy metadata, and a binder that
  creates a `ProviderRuntime` at the edge.
- `ProviderCatalog` is an explicit set of registrations. Binding configured
  sources creates one immutable `ProviderRegistry` for a run.

The registry derives broker registrations, Python bindings, prompt text, and
the policy accessor manifest from the same operation descriptors. It snapshots
those descriptors for the run. Changing a description, schema, revision,
digest, effect, or policy value does not change the stable `CapabilityId`
`(kind, provider_type, source_id, operation_id)`.

## Schemas, pagination, and delivery

Every parameter schema and typed result schema names its dialect and
provenance. Droste does not guess a dialect or claim a provider-authored schema
as engine-authored. Cursor pagination requires a `cursor` parameter property
and `next_cursor` result property.

Delivery is explicit:

- `inline` requires a typed result schema and returns a frozen inline value;
- `handle` requires a typed result schema and a `CapabilityResultHandle`;
- `untyped` declares no result schema and permits provider-defined inline data.

`budget_class` is stable classification data for the budget subsystem; the
provider layer does not implement a ledger.

## Host policy and bridges

The host must classify every operation as `read` or `effectful`. An incomplete
map or `unspecified` value fails before binding. Policy metadata is also
host-owned and frozen into the per-run descriptor.

`ProviderService` exposes one already-bound source across the JSON bridge. It
publishes the verified manifest, source identity, and source description, then
dispatches only raw operation IDs present in the bound runtime. It never
publishes authoritative effects or policy. The receiving host constructs a
`BridgeProvider`, supplies its own exact effects and policy, and binds the
advertised source through its explicit catalog.

## Example

```python
from droste import ConfiguredSource, ProviderCatalog
from droste.sources.sql_local import sqlite_provider

catalog = ProviderCatalog((sqlite_provider(),))
registry = catalog.bind(
    (
        ConfiguredSource(
            source_id="db",
            provider_type="sqlite",
            config={"sqlite_path": "local.db"},
        ),
    ),
    default_source_id="db",
)
```

SQLite's raw operations are `query` and `schema`; their Python bindings are
`query` and `get_schema`. That distinction lets transports keep stable operation
identity without making Python naming the provider protocol.

## Evidence

Capability metadata uses `EvidenceLocation(source_id, path, revision, ranges)`.
Each `EvidenceRange` may carry ordered byte coordinates, line coordinates, a
section identifier, or a combination. Durable trace projections retain only
the evidence count; full locations are replay content governed by the host's
retention policy.
