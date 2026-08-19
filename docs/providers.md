# Providers (protocol v4)

Providers expose trusted data operations to generated code without exposing
transport clients, credentials, or host paths. The same immutable description
drives the broker allowlist, prompt text, Python bindings, and policy metadata.

## Values and ownership

- `ProviderManifest` contains a revision, digest, and ordered
  `ProviderOperation` values.
- `ConfiguredSource` is frozen configuration for one named source.
- `ProviderRegistration` pairs a manifest with host-owned side-effect policy
  and a binder.
- `ProviderRuntime` owns the live handlers and optional close callback.
- `BoundSource` is one acquired runtime plus its registration.
- `ProviderRegistry` combines bound sources for one run and closes them in
  reverse acquisition order.

An operation has separate raw `operation_id` and Python `binding_name` values,
explicit parameter/result schema dialect and provenance, pagination and result
delivery modes, and a budget class. The host classifies every operation as
`read` or `effectful`; transports cannot grant effects.

Trusted handlers use the context-first signature:

```python
def handler(execution: CapabilityExecutionContext, *args, **kwargs):
    execution.check()
    return CapabilityOutcome(result={"value": "..."})
```

Passing a registry to `create_environment()` transfers ownership. Otherwise,
close it explicitly. Partial binds and failed environment construction close
already-acquired runtimes deterministically.

## Built-in providers

The SQLite provider opens databases read-only, enables `query_only`, and admits
only bounded single-statement `SELECT` queries. The filesystem-text provider
offers bounded `list`, `read`, `grep`, `search`, and `stat` operations under one
pinned root. It does not follow symlinks and revalidates path and cursor policy
on every request.

These policies constrain the provider operation; they do not isolate the native
Python process. Use a separate process or the Pyodide bridge when source paths
must be non-ambient.

## MCP over local stdio

`open_mcp_stdio_source()` launches one absolute, host-allowlisted executable,
negotiates MCP `2025-11-25`, freezes the complete paginated `tools/list`
snapshot, and returns a `BoundSource`. Generated code sees ordinary provider
bindings, not MCP vocabulary or process configuration.

The host must provide an explicit working directory and environment, executable
and tool allowlists, Python binding names, read classifications, budget classes,
and optional policy metadata. The transport currently admits read-only tools.

Supported result mapping prefers `structuredContent`; otherwise it returns the
bounded content-block JSON without fetching links. Cancellation terminates the
session after sending `notifications/cancelled`. There is no reconnect because
a replacement process would be a different session and an effect may already
have occurred.

MCP tasks, resources as first-class evidence, live tool refresh, and arbitrary
cross-dialect schema validation are not supported. The integration test uses
the pinned official filesystem MCP server and routes its tool through the same
broker as SQLite.

## Cross-interpreter bridge

`ProviderService` and `BridgeProvider` carry a verified manifest and operation
calls across an interpreter boundary. The receiving host supplies effects and
policy, rejects unknown operations, and validates cumulative checkpoints before
the broker performs its sole final settlement. Bridge protocol v2 adds a bounded
duplex pump for mid-call checkpoints and cancellation acknowledgement; unary
invocation remains available for hosts that do not select it.
