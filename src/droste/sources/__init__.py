"""Concrete local-mode data sources shipped with the engine.

The engine's base runner only builds remote ``wrapper_v1`` sources; everything
in this package is opt-in. Most modules are registered by the consumer's
entrypoint via :func:`register_source_type` (Option C, the source-unification
design; ``registration.py``, re-exported by ``droste_runner``) — see each
module's ``register()`` helper. ``bridge.py`` is the exception: it's built
directly by a trusted host process handing a concrete transport into an
interpreter it created, not from a declarative request, so it has no
``register()``/factory.
"""

from .bridge import BridgeDataSource, DataSourceService
from .registration import (
    SOURCE_PROTOCOL_VERSION,
    SourceFactory,
    register_source_type,
    source_factory,
)
from .sql_local import (
    DEFAULT_LOCAL_SQL_POLICY,
    LocalSqlDataSource,
    LocalSqlPolicy,
    SqlPolicyError,
    local_sql_source_factory,
    register,
    validate_local_sql,
)

__all__ = [
    "BridgeDataSource",
    "DataSourceService",
    "SOURCE_PROTOCOL_VERSION",
    "SourceFactory",
    "register_source_type",
    "source_factory",
    "DEFAULT_LOCAL_SQL_POLICY",
    "LocalSqlDataSource",
    "LocalSqlPolicy",
    "SqlPolicyError",
    "local_sql_source_factory",
    "register",
    "validate_local_sql",
]
