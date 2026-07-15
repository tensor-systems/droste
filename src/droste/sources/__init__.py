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
    "DEFAULT_LOCAL_SQL_POLICY",
    "SQLITE_PROVIDER_MANIFEST",
    "LocalSqlRuntime",
    "LocalSqlPolicy",
    "SqlPolicyError",
    "sqlite_provider",
    "validate_local_sql",
]
