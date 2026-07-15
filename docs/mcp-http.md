# MCP Streamable HTTP provider transport

Droste can acquire a remote MCP server through a trusted host and project its
tools through the same immutable provider and capability ABI used by in-process
and stdio sources. Generated code receives only descriptor-generated Python
bindings. It never receives the MCP endpoint, authorization configuration,
resolved secrets, session ID, JSON-RPC envelope, or raw response payload.

The implementation targets MCP protocol `2025-11-25` and supports JSON and SSE
responses, event-ID resumption with bounded reconnect/backoff, stateful session
replacement, paginated `tools/list`, cancellation, deadlines, and explicit
session deletion. A replacement session must publish the byte-identical frozen
tool snapshot or the source fails closed.

## Native host API

```python
from droste import ConfiguredSource
from droste.sources import McpHttpHost, McpSecretRequest, open_mcp_http_source

source = ConfiguredSource(
    "documents",
    "company_documents",
    {
        "endpoint": "https://mcp.example.com/mcp",
        "allowed_endpoints": ["https://mcp.example.com/mcp"],
        "tenant_id": "tenant-42",
        "auth": {"type": "bearer", "token_ref": "secrets/mcp/documents"},
        "allowed_tools": ["documents.search", "documents.read"],
        "bindings": {
            "documents.search": "search",
            "documents.read": "read",
        },
        "effects": {
            "documents.search": "read",
            "documents.read": "read",
        },
        "budget_classes": {
            "documents.search": "data.search",
            "documents.read": "data.read",
        },
        "policy_metadata": {
            "documents.search": {"read_only": True},
            "documents.read": {"read_only": True},
        },
        "source_description": "Company documents available to this tenant.",
    },
)


def resolve_secret(request: McpSecretRequest) -> str:
    # Authorize the exact (tenant_id, source_id, reference) tuple here.
    return secret_store.read(request.tenant_id, request.source_id, request.reference)


bound = open_mcp_http_source(source, McpHttpHost(resolve_secret=resolve_secret))
```

The source configuration contains references, not credential values. The
resolver is live host state and is neither frozen into `ConfiguredSource` nor
included in descriptors. Every resolution carries the tenant and source IDs;
hosts must authorize that complete tuple rather than treating a reference as a
process-global secret name.

`auth.type` supports:

- `none`: no other auth fields;
- `bearer`: one `token_ref` resolved by the trusted host;
- `oauth_client_credentials`: `client_id_ref`, `client_secret_ref`, exact
  `resource_metadata_url`, exact `authorization_server`, a non-empty exact
  `token_endpoints` allowlist, and optional `scopes`.

OAuth performs protected-resource and authorization-server discovery, verifies
the exact resource, issuer, and token endpoint, includes the RFC 8707 `resource`
parameter, caches short-lived access tokens, and reacquires or refreshes after
expiry or one `401`. Access and refresh tokens remain only in the live session.
Interactive authorization-code consent is a host responsibility; after consent,
the host can expose the resulting access token through a tenant-scoped bearer
reference.

## Network and resource policy

Every URL must be canonical HTTPS without userinfo, query, or fragment. The MCP
endpoint must appear in `allowed_endpoints` as an exact URL. Redirects are never
followed. OAuth resource, issuer, and token locations are independently exact.

DNS is resolved for every exchange. Every answer is checked, then one approved
address is pinned through socket connection and TLS hostname verification.
Loopback, private, link-local, multicast, reserved, unspecified, and metadata
addresses fail closed by default. A trusted VPC host may opt into exact CIDRs
through `McpHttpHost(allowed_networks=("10.8.0.0/16",))`; DNS answers outside
those CIDRs still fail. This is authority and should be scoped as narrowly as
the deployment permits.

Defaults and accepted configuration keys:

| Key | Default / rule |
| --- | --- |
| `startup_timeout_ms` | 15,000; 100..120,000 |
| `request_timeout_ms` | 30,000; 100..300,000 |
| `close_timeout_ms` | 2,000; 10..30,000 |
| `max_frame_bytes` | 1 MiB; 1 KiB..16 MiB |
| `max_descriptor_bytes` | 1 MiB; 1 KiB..16 MiB |
| `max_result_bytes` | 256 KiB; 256 B..8 MiB |
| `max_tool_pages` | 32; 1..256 |
| `max_tools` | 256; 1..4,096 |
| `max_in_flight` | 64; 1..4,096 |
| `reconnect_attempts` | 2; 0..5 |
| `backoff_ms` | 100; 10..5,000 |
| `max_debug_payload_bytes` | 0 (disabled); maximum 4 KiB |

Ordinary stats and durable capability traces contain only normalized counts,
latencies, sizes, status, and immutable capability identity. Raw MCP payload
debugging is disabled by default. Enabling a bounded `max_debug_payload_bytes`
also requires a trusted `debug_sink`; only request/response bodies are passed,
never HTTP headers, endpoint, authorization, or session values. Raw debug data
is content-bearing and must not be routed into durable default traces.
`requests_made` counts every attempted tool dispatch, including failed and
cancelled attempts; `calls` and `failures` retain the success/failure split.

## Cross-language and runner hosts

`bind_mcp_transport_source(source, transport, policy)` is the connector-neutral
seam. `McpToolTransport` supplies `list_tools`, `call_tool`, `stats`, and
`close`; `McpBindingPolicy` supplies the host's allowlist, bindings, effects,
budget classes, and policy. This lets a trusted Deno, Go, or service process own
HTTP/OAuth and connect through `ProviderService`/`BridgeProvider` without
copying the provider schema or exposing MCP-specific sandbox bindings.

Trusted `droste_runner` callers can pass
`source_opener(ConfiguredSource, context) -> BoundSource` to `run()` or
`run_worker_request()`. The request cannot select this hook. Both `run` and
`preflight` acquire dynamic sources through it, so preflight records the exact
remote-discovered immutable capability shape and the runner closes the source
on every terminal path.

The transport follows the primary MCP
[`Streamable HTTP`](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports),
[`authorization`](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization),
[`lifecycle`](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle),
and [`tools`](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
contracts.
