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
  sources creates one live `ProviderRegistry` whose descriptor snapshot is
  immutable for the run.

Binding is also the resource-acquisition boundary. A binder may attach an
optional `ProviderRuntime.close_callback`; resource-free providers omit it.
`ProviderRuntime.close()`, `BoundSource.close()`, and `ProviderRegistry.close()`
are idempotent and concurrency-safe. A registry closes every acquired runtime
once in reverse bind order, continues after individual close failures, and
cleans up earlier bindings if a later bind or registry validation fails. Live
resources are never assigned to a finalizer or process-global registry.

One `ProviderRuntime` object is one ownership token and cannot be installed for
two configured sources. Providers that share an underlying pool must return a
separate runtime lease for each bind and reconcile the shared pool inside those
lease callbacks. The registry does not guess sharing from handler or callback
identity.

The host owns a registry returned by `ProviderCatalog.bind()`. Passing it to
`create_environment()` transfers ownership immediately: construction failure
closes it, and a successful native or Pyodide environment closes it on every
run exit. A host that keeps a registry outside an environment must call
`registry.close()` explicitly. Closure begins only after broker dispatch is
quiescent; it is not a provider-cancellation mechanism. Runtime `stats`
callbacks must remain content-free and callable after close so hosts can report
final counters without reopening a resource.

If cleanup fails after `run_rlm()` has already computed a result, the result is
preserved and Python emits a bounded `RuntimeWarning`; the relay likewise emits
a bounded stderr diagnostic without replacing its response. If execution and
cleanup fail together, Python raises a `BaseExceptionGroup` containing both
causes. Direct `close()` calls still raise their cached cleanup failure.

Constructing `ProviderService` transfers one bound source to the trusted bridge
service for that service's lifetime. That host closes the service after the
last invocation and does not also transfer the supplying registry to an
environment. The runtime's once-only gate still makes outer failure cleanup of
the registry safe.

The registry derives broker registrations, Python bindings, prompt text, and
the policy accessor manifest from the same operation descriptors. It snapshots
those descriptors for the run. Changing a description, schema, revision,
digest, effect, or policy value does not change the stable `CapabilityId`
`(kind, provider_type, source_id, operation_id)`.

Every `ProviderRuntime` handler is context-first. Its first argument is a frozen
`CapabilityExecutionContext`; operation parameters follow unchanged. Generated
model bindings never expose this trusted argument. Providers use `check()` and
cumulative `checkpoint()` only—they cannot mutate the run ledger or trace.

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

Bridge `invoke` requests require a versioned `execution` object containing
call/run identity, remaining deadline, reservation facts, and the cancellation
snapshot. Every successful transport response includes a cumulative
`{tokens, subcalls}` checkpoint, including typed provider-error outcomes. The
receiving broker validates and applies that value before its final settlement.
The trusted bridge host must close the service after the last invocation.
Transport shutdown and runtime
resource ownership are separate: a duplex session closes per call, while the
provider runtime closes once for the bound source lifetime.

Unary invocation remains the default provider-protocol-4 transport. A host that
needs live cancellation or accounting explicitly supplies a bridge-v2 duplex
session to `BridgeProvider`. Each invocation gets one bounded message pump keyed
by the existing `call_id`; it carries ordered `check`, cumulative `checkpoint`,
and exactly one `terminal` frame. The receiver applies checkpoints through the
same `CapabilityExecutionContext` and acknowledges only after ledger acceptance.
The remote handler still receives only that context. The reference pump permits
one queued frame and rejects serialized frames larger than 8 MiB.

The pump is pull-based: the receiving interpreter returns from each transport
yield to reduce a frame, then sends one acknowledgement. Hosts must not call
back into a suspended Pyodide interpreter. A trusted host cancellation request
is sampled immediately before the receiver reduces each frame; providers cannot
set it. Once sampled, the receiver returns a cancelled acknowledgement and the
broker records that cancellation before its finalization cutoff. The terminal
sample is the transport cutoff, so a request arriving after it is late. Exact
duplicate frames are idempotent, while reordered, conflicting, wrong-call, or
over-reservation frames fail closed.

If the remote provider interpreter or transport disappears before a valid
terminal frame, the receiving broker produces `bridge.transport_lost` and runs
its ordinary exactly-once settlement. Previously acknowledged checkpoints stay
committed. For inference attempts, the budget authority conservatively charges
the remaining reservation because unacknowledged remote spend is unknowable.
Killing the receiving broker itself is outside this in-process contract and
requires host-level durable run reconciliation.

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
# Pass registry to create_environment(), or close it explicitly in a finally block.
```

SQLite's raw operations are `query` and `schema`; their Python bindings are
`query` and `get_schema`. That distinction lets transports keep stable operation
identity without making Python naming the provider protocol.

## Local filesystem/text provider

`filesystem_text_provider()` is the dependency-free, read-only provider for a
host-selected local directory. `filesystem_text` is the reusable provider type;
each `ConfiguredSource` supplies a source name and an absolute trusted `root`:

```python
from droste import ConfiguredSource, ProviderCatalog
from droste.sources import filesystem_text_provider

registry = ProviderCatalog((filesystem_text_provider(),)).bind(
    (
        ConfiguredSource(
            source_id="docs",
            provider_type="filesystem_text",
            config={
                "root": "/data/docs",
                "include": ["**/*.md", "**/*.txt", "**/*.log"],
                "exclude": [".git/**"],
                "enrichers": ["markdown"],
            },
        ),
    )
)
# Pass registry to create_environment(), or close it explicitly in a finally block.
```

Its raw operations are `list`, `read`, `grep`, `search`, and `stat`; the Python
binding for `list` is `list_files` because provider bindings may not shadow a
Python builtin. `grep` is case-sensitive literal matching, while `search` is an
index-free case-insensitive all-terms scan. Markdown section reads are an
optional removable enrichment. The five base operations work without an index
or non-stdlib dependency.

Paths are source-relative POSIX values. Include and exclude globs are
case-sensitive; `*`, `?`, and character classes stay within one segment, while
`**` must occupy a complete segment and spans zero or more segments. Exclusion
always wins, including for an explicit `read`, `stat`, or scan path. Directory
walks, file reads, lines, result pages, depth, and entry counts all have
configuration bounds. `max_result_bytes` limits the serialized inline result,
including JSON escaping rather than only the raw file bytes. Text is strict UTF-8 with no NUL bytes; binary files and
special files return typed unsupported outcomes rather than content.
The optional positive bounds are `max_read_bytes`, `max_scan_bytes`,
`max_result_bytes`, `max_scan_entries`, `max_results`, `max_line_bytes`, and
`max_depth`; invalid or internally inconsistent values fail at bind.

`list_files`, `grep`, and `search` return versioned, self-contained cursors, so
the provider keeps no cursor registry. Each cursor is authenticated by and
valid only for the bound runtime that issued it. A cursor contains no path
authority: on every use the provider re-applies path policy and revalidates the
immutable file inventory. Addition, removal, replacement, or revision drift
returns retryable `filesystem.changed` instead of silently duplicating or
skipping results. `read` may also receive the opaque revision returned by
`stat`.

The trusted runtime pins the configured root as a directory descriptor and
opens every path component relative to an already-open directory with
`O_NOFOLLOW`. It never uses string-prefix checks, `resolve()`, `os.walk`, or a
symlink-following fallback. Platforms missing the required POSIX `dir_fd`,
descriptor-listing, `O_DIRECTORY`, `O_NOFOLLOW`, `O_NONBLOCK`, or `pread`
primitives fail while binding the source. The configured absolute root is
never placed in descriptors, prompt text, results, evidence, or errors.

Native in-process execution still runs arbitrary Python and is not an OS
security boundary. Root non-ambient access requires the host to place the
provider on the trusted side of a process or Pyodide/WASM boundary. The generic
`ProviderService`/`BridgeProvider` transport preserves the same typed result,
error, usage, cursor, and evidence values; the filesystem provider adds no
transport-specific path.

## Evidence

Capability metadata uses `EvidenceLocation(source_id, path, revision, ranges)`.
Each `EvidenceRange` may carry ordered byte coordinates, line coordinates, a
section identifier, or a combination. Durable trace projections retain only
the evidence count; full locations are replay content governed by the host's
retention policy.

Filesystem byte ranges are zero-based and half-open. Line coordinates are
one-based and inclusive. Evidence paths are source-relative POSIX paths and
revisions are opaque stat-derived digests; neither exposes the configured host
root.

## MCP transports

`open_mcp_stdio_source()` acquires one host-configured local MCP process and
maps its complete paginated `tools/list` snapshot into the same immutable
provider values. It returns a lifecycle-owned `BoundSource`, which may be
combined directly with in-process bound sources in `ProviderRegistry`. MCP is
not a provider type, prompt vocabulary, binding namespace, policy authority, or
trace path. See the [local MCP stdio contract and spike report](mcp-stdio.md).

`open_mcp_http_source()` performs the same acquisition transaction over MCP
Streamable HTTP. The trusted host owns exact endpoints, tenant-scoped secret
resolution, OAuth, DNS/IP policy, and sessions; none of those values enters the
manifest or generated bindings. Cross-language hosts can implement
`McpToolTransport` and call `bind_mcp_transport_source()` so the same pure
`mcp_tools_to_manifest()` projection remains authoritative. See the
[Streamable HTTP contract](mcp-http.md).
