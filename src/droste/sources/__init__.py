"""Provider implementations and transport adapters shipped with Droste."""

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
    McpConfigurationError,
    McpDescriptorError,
    McpManifestPolicy,
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
    "McpConfigurationError",
    "McpDescriptorError",
    "McpManifestPolicy",
    "mcp_tools_to_manifest",
    "normalize_mcp_tool_result",
    "open_mcp_stdio_source",
    "DEFAULT_LOCAL_SQL_POLICY",
    "SQLITE_PROVIDER_MANIFEST",
    "LocalSqlRuntime",
    "LocalSqlPolicy",
    "SqlPolicyError",
    "sqlite_provider",
    "validate_local_sql",
]
