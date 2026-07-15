"""Convenience facade for runner hosts."""

from droste.environments.inprocess import OutputBuffer, RunnerEnvironment, describe_context
from droste.providers import PROVIDER_PROTOCOL_VERSION, ProviderCatalog

from .http_clients import HTTPSubcallClient, RootLLMClient
from .http_clients import _http_error_excerpt as _http_error_excerpt
from .http_clients import _redact_secrets as _redact_secrets
from .protocol import RUNNER_PROTOCOL_VERSION, RootResponseMetadata, RunnerOperation, build_response
from .protocol import _check_protocol_version as _check_protocol_version
from .protocol import _protocol_error_response as _protocol_error_response
from .run import WorkerOutcome, main, run, run_worker_request
from .run import _build_context as _build_context
from .run import _read_request as _read_request
from .run import _run_adapter as _run_adapter
from .sources import (
    WRAPPER_PROVIDER_MANIFEST,
    SourceOpener,
    WrapperTransport,
    build_opened_provider_registry,
    build_provider_registry,
    default_provider_catalog,
    wrapper_provider,
)
from .sources import _allowlist_opener as _allowlist_opener
from .sources import _require_allowed_host as _require_allowed_host

__all__ = [
    "HTTPSubcallClient",
    "OutputBuffer",
    "PROVIDER_PROTOCOL_VERSION",
    "ProviderCatalog",
    "RUNNER_PROTOCOL_VERSION",
    "RootLLMClient",
    "RootResponseMetadata",
    "RunnerOperation",
    "RunnerEnvironment",
    "SourceOpener",
    "WRAPPER_PROVIDER_MANIFEST",
    "WrapperTransport",
    "WorkerOutcome",
    "build_opened_provider_registry",
    "build_provider_registry",
    "build_response",
    "default_provider_catalog",
    "describe_context",
    "main",
    "run",
    "run_worker_request",
    "wrapper_provider",
]
