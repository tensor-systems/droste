"""Concrete local-mode data sources shipped with the engine.

The engine's base runner only builds remote ``wrapper_v1`` sources; everything
in this package is opt-in and must be registered by the consumer's entrypoint
via :func:`droste_runner.runner.register_source_type` (Option C,
unified-data-sources §7.2). See each module's ``register()`` helper.
"""

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
    "DEFAULT_LOCAL_SQL_POLICY",
    "LocalSqlDataSource",
    "LocalSqlPolicy",
    "SqlPolicyError",
    "local_sql_source_factory",
    "register",
    "validate_local_sql",
]
