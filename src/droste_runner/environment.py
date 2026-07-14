"""Compatibility imports for the native in-process environment."""

from droste.environments.inprocess import (
    CONTEXT_PREVIEW_CHARS,
    CONTEXT_PREVIEW_MAX_FILES,
    OutputBuffer,
    RunnerEnvironment,
    describe_context,
)

__all__ = [
    "CONTEXT_PREVIEW_CHARS",
    "CONTEXT_PREVIEW_MAX_FILES",
    "OutputBuffer",
    "RunnerEnvironment",
    "describe_context",
]
