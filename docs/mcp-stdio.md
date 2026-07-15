# MCP stdio provider transport

Droste can acquire a local MCP server as a data provider without exposing MCP
to generated code. The trusted host launches the server, freezes one complete
`tools/list` snapshot into a `ProviderManifest`, and returns one owned
`BoundSource`. From that point onward, prompt text, Python bindings, policy,
budgets, traces, and result envelopes use the ordinary broker path.

This is deliberately a local stdio transport. Streamable HTTP, OAuth, remote
sessions, redirects, and SSRF controls belong to [#92](https://github.com/tensor-systems/droste/issues/92).
MCP tasks, live tool refresh, and full cross-dialect JSON Schema validation
belong to [#91](https://github.com/tensor-systems/droste/issues/91).

The base Droste wheel remains dependency-free. The transport is a small
spec-shaped JSON-RPC shell rather than a second capability implementation. It
owns bounds and request IDs needed for Droste cancellation; the stable MCP
Python SDK's public stdio client currently inherits ambient environment values,
does not bound newline framing, and does not expose request IDs for explicit
`notifications/cancelled` delivery.

## Host configuration

Launch and policy authority are explicit. Droste executes an absolute
allowlisted program directly—never through a shell—and supplies only the
declared working directory and environment object. The tool allowlist, Python binding names,
read/effectful classification, budget classes, and policy metadata are also
host facts. MCP annotations do not override them.
The current local spike accepts only tools the host classifies as `read`; it
does not expose the reference server's write/edit operations.

```python
import os

from droste import ConfiguredSource, ProviderRegistry
from droste.sources import open_mcp_stdio_source

source = ConfiguredSource(
    source_id="reference_docs",
    provider_type="reference_filesystem",  # logical type, not transport
    config={
        "command": "/absolute/path/to/node",
        "args": [
            "/absolute/pinned/server-filesystem/dist/index.js",
            "/absolute/data/docs",
        ],
        "env": {},
        "cwd": "/absolute/data/docs",
        "allowed_executables": ["/absolute/path/to/node"],
        "allowed_tools": ["read_text_file"],
        "bindings": {"read_text_file": "read_text_file"},
        "effects": {"read_text_file": "read"},
        "budget_classes": {"read_text_file": "data.read"},
        "policy_metadata": {"read_text_file": {"read_only": True}},
        "source_description": "Read-only reference documents.",
    },
)

bound = open_mcp_stdio_source(source)
registry = ProviderRegistry((bound,))
try:
    # Pass registry to create_environment(), or project it into a broker.
    ...
finally:
    registry.close()
```

`open_mcp_stdio_source()` is an acquisition transaction rather than a static
provider factory because the manifest does not exist until after process
startup, MCP initialization, and paginated discovery. Any failure during that
transaction reaps the process. A successful runtime owns its session until the
registry or environment closes it.

## Descriptor and result mapping

- The supported MCP protocol is exactly `2025-11-25`.
- Every `tools/list` page is read once. Duplicate names, repeated cursors, and
  independently configured page, tool, frame, and aggregate descriptor bounds
  fail before binding. Initialization and discovery share one startup deadline.
- MCP tool names remain the exact, case-sensitive raw `operation_id`. The host
  supplies a separate valid Python `binding_name`; collisions fail.
- Input and output schemas are retained exactly. Their explicit `$schema` is
  the dialect; absence means JSON Schema 2020-12 as required by MCP. Provenance
  identifies the frozen MCP snapshot and schema role.
- A declared `outputSchema` is typed inline delivery. Without one, the result
  is explicitly untyped. A tool requiring MCP task execution is rejected.
- `structuredContent` is preferred exactly. Content-only results retain the
  exact bounded content-block JSON in `{"content": [...], "losses": []}`;
  links are never fetched. `_meta` and duplicate compatibility text do not
  enter the returned structured value. A typed count records compatibility
  content blocks ignored in favor of structured data.
- Discovery pagination is not operation pagination. MCP tool descriptors do
  not declare cursor semantics, so the mapped operation remains
  `PaginationMode.NONE`; full schema/cursor compatibility stays in #91.
- `isError`, JSON-RPC errors, protocol failures, transport loss, and oversized
  normalized results have distinct stable provider error codes.

MCP resource URIs are not promoted to Droste `EvidenceLocation` values. They
lack Droste's source revision and byte/line/section coordinates, so claiming
equivalence would manufacture evidence. Resource blocks remain available in
the untyped result; first-class evidence mapping is an explicit gap.

## Cancellation, accounting, traces, and lifecycle

The stdio session polls the broker's `CapabilityExecutionContext` while a tool
call is being written and while it is in flight. Stdin frames use bounded
nonblocking writes so a server that stops reading cannot defeat cancellation.
Cancellation or deadline expiry emits MCP
`notifications/cancelled`, then makes the whole session terminal and performs a
bounded stdin → process-group `TERM` → process-group `KILL` shutdown. There is
no automatic reconnect or retry: a new process would be a different session,
and an effect may already have happened.

MCP progress is not compute usage and cannot mutate the budget ledger. Warm
call latency and response bytes are ordinary provider usage metrics. Raw
requests, results, process configuration, stderr, and error content never enter
the durable capability trace projection; the broker still owns the one
content-free `CapabilityResult.to_trace_dict()` path.

Native hosts own the stdio process. Pyodide/WASM code does not gain ambient
process authority; a trusted host places the resulting `BoundSource` behind
the existing `ProviderService`/`BridgeProvider` boundary. The Pyodide
conformance test exercises that generic path beside SQLite without an
MCP-specific sandbox API.

## Reference spike and measured gap

The conformance fixture pins the official
[`@modelcontextprotocol/server-filesystem`](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem)
at `2026.7.10` in a committed npm lockfile and exposes only
`read_text_file`. CI installs it with `npm ci --ignore-scripts`; it is never a
Droste runtime dependency.

On 2026-07-15, an Apple Silicon machine running macOS 26.5.2, Node 26.5.0,
Python 3.13, and 100 repeated reads measured:

| Path | Median | p95 |
| --- | ---: | ---: |
| Official MCP filesystem server, warm stdio call | 0.228 ms | 0.407 ms |
| First-party in-process filesystem provider | 0.093 ms | 0.144 ms |

Cold MCP acquisition (process launch, initialize, and discovery) was 112.720 ms;
warm median transport overhead was 0.135 ms. These are one-machine spike
measurements, not a performance guarantee. Reproduce them with:

```bash
uv run python benchmarks/mcp_stdio_latency.py \
  --node "$(command -v node)" \
  --server tests/reference_mcp/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js \
  --iterations 100
```

The remaining impedance gaps are schema validation across arbitrary dialects,
task results, refresh between sessions after `listChanged`, MCP progress as a
bounded diagnostic, and evidence coordinates. They stay outside the spike so
the transport does not become a second provider or budget architecture.

The implementation follows the current primary contracts: the
[`2025-11-25` tools specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools),
[`stdio` transport](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports),
[`lifecycle`](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle),
and [`cancellation`](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/cancellation).
