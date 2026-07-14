"""Compatibility facade for the formerly monolithic runner module.

Canonical implementations now live in focused modules. Existing imports from
droste_runner.runner remain valid during migration.
"""

from droste.environments.inprocess import (
    OutputBuffer,
    RunnerEnvironment,
    describe_context,
)
from droste.sources.registration import (
    SOURCE_PROTOCOL_VERSION,
    SourceFactory,
    register_source_type,
)

from .http_clients import (
    HTTPSubcallClient,
    RootLLMClient,
)
from .http_clients import (
    _http_error_excerpt as _http_error_excerpt,
)
from .http_clients import (
    _redact_secrets as _redact_secrets,
)
from .protocol import (
    RUNNER_PROTOCOL_VERSION,
    RootResponseMetadata,
    build_response,
)
from .protocol import (
    _check_protocol_version as _check_protocol_version,
)
from .protocol import (
    _protocol_error_response as _protocol_error_response,
)
from .run import (
    _build_context as _build_context,
)
from .run import (
    _read_request as _read_request,
)
from .run import (
    _run_adapter as _run_adapter,
)
from .run import (
    main,
    run,
)
from .sources import (
    DataSourceWrapper as DataSourceWrapper,
)
from .sources import (
    WrapperV1DataSource,
    build_data_sources,
)
from .sources import (
    _allowlist_opener as _allowlist_opener,
)
from .sources import (
    _build_one_source as _build_one_source,
)
from .sources import (
    _register_builtin_source_types as _register_builtin_source_types,
)
from .sources import (
    _require_allowed_host as _require_allowed_host,
)
from .sources import (
    _reset_source_types as _reset_source_types,
)

__all__ = [
    "HTTPSubcallClient",
    "OutputBuffer",
    "RUNNER_PROTOCOL_VERSION",
    "RootLLMClient",
    "RootResponseMetadata",
    "RunnerEnvironment",
    "SOURCE_PROTOCOL_VERSION",
    "SourceFactory",
    "WrapperV1DataSource",
    "build_data_sources",
    "build_response",
    "describe_context",
    "main",
    "register_source_type",
    "run",
]
