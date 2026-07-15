"""RLM runner for HTTP-backed root/subcall orchestration (with optional adapters)."""

from .protocol import RUNNER_PROTOCOL_VERSION, RunnerOperation
from .run import WorkerOutcome, main, run, run_worker_request
from .sources import (
    SourceOpener,
    WrapperTransport,
    build_opened_provider_registry,
    build_provider_registry,
    default_provider_catalog,
    wrapper_provider,
)

__all__ = [
    "RUNNER_PROTOCOL_VERSION",
    "RunnerOperation",
    "SourceOpener",
    "WrapperTransport",
    "WorkerOutcome",
    "build_opened_provider_registry",
    "build_provider_registry",
    "default_provider_catalog",
    "main",
    "run",
    "run_worker_request",
    "wrapper_provider",
]
