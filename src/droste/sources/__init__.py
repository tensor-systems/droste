"""Provider implementations and transport adapters shipped with Droste."""

from typing import TYPE_CHECKING, Any

from .bridge import (
    BridgeCall,
    BridgeProtocolError,
    BridgeProvider,
    BridgeTransportLost,
    DuplexBridgeCall,
    DuplexBridgeSession,
    ProviderService,
)
from .filesystem_text import (
    FILESYSTEM_TEXT_PROVIDER_MANIFEST,
    FilesystemTextConfig,
    filesystem_text_provider,
)
from .mcp_stdio import (
    MCP_PROTOCOL_VERSION,
    McpBindingPolicy,
    McpConfigurationError,
    McpDescriptorError,
    McpManifestPolicy,
    McpToolTransport,
    bind_mcp_transport_source,
    mcp_tools_to_manifest,
    normalize_mcp_tool_result,
    open_mcp_stdio_source,
)
from .sql_local import (
    DEFAULT_LOCAL_SQL_POLICY,
    SQLITE_PROVIDER_MANIFEST,
    LocalSqlPolicy,
    LocalSqlRuntime,
    SqlPolicyError,
    sqlite_provider,
    validate_local_sql,
)

if TYPE_CHECKING:
    from .mcp_http import McpHttpHost, McpSecretRequest, open_mcp_http_source


def __getattr__(name: str) -> Any:
    if name in {"McpHttpHost", "McpSecretRequest", "open_mcp_http_source"}:
        from . import mcp_http

        value = getattr(mcp_http, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BridgeCall",
    "BridgeProtocolError",
    "BridgeProvider",
    "BridgeTransportLost",
    "DuplexBridgeCall",
    "DuplexBridgeSession",
    "ProviderService",
    "FILESYSTEM_TEXT_PROVIDER_MANIFEST",
    "FilesystemTextConfig",
    "filesystem_text_provider",
    "MCP_PROTOCOL_VERSION",
    "McpBindingPolicy",
    "McpConfigurationError",
    "McpDescriptorError",
    "McpHttpHost",
    "McpManifestPolicy",
    "McpSecretRequest",
    "McpToolTransport",
    "bind_mcp_transport_source",
    "mcp_tools_to_manifest",
    "normalize_mcp_tool_result",
    "open_mcp_http_source",
    "open_mcp_stdio_source",
    "DEFAULT_LOCAL_SQL_POLICY",
    "SQLITE_PROVIDER_MANIFEST",
    "LocalSqlRuntime",
    "LocalSqlPolicy",
    "SqlPolicyError",
    "sqlite_provider",
    "validate_local_sql",
]
