"""RLM runner for HTTP-backed root/subcall orchestration (with optional adapters)."""

from .protocol import RUNNER_PROTOCOL_VERSION, RunnerOperation
from .run import main, run
from .sources import (
    WrapperTransport,
    build_provider_registry,
    default_provider_catalog,
    wrapper_provider,
)

__all__ = [
    "RUNNER_PROTOCOL_VERSION",
    "RunnerOperation",
    "WrapperTransport",
    "build_provider_registry",
    "default_provider_catalog",
    "main",
    "run",
    "wrapper_provider",
]
