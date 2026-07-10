"""Concrete local-mode data sources shipped with the engine.

The engine's base runner only builds remote ``wrapper_v1`` sources; everything
in this package is opt-in. Most modules are registered by the consumer's
entrypoint via :func:`droste_runner.runner.register_source_type` (Option C,
the source-unification design) — see each module's ``register()`` helper.
``bridge.py`` is the exception: it's built directly by a trusted host process
handing a concrete transport into an interpreter it created, not from a
declarative request, so it has no ``register()``/factory.
"""

from .bridge import BridgeDataSource, DataSourceService
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
    "DEFAULT_LOCAL_SQL_POLICY",
    "LocalSqlDataSource",
    "LocalSqlPolicy",
    "SqlPolicyError",
    "local_sql_source_factory",
    "register",
    "validate_local_sql",
]
