"""Compatibility imports for the native in-process environment."""

from droste.environments.inprocess import (
    CONTEXT_PREVIEW_CHARS,
    CONTEXT_PREVIEW_MAX_FILES,
    OutputBuffer,
    RunnerEnvironment,
    describe_context,
)
from droste.environments.inprocess import (
    _describe_files_context as _describe_files_context,
)
from droste.environments.inprocess import (
    _safe_label as _safe_label,
)
from droste.environments.inprocess import (
    _safe_preview as _safe_preview,
)

__all__ = [
    "CONTEXT_PREVIEW_CHARS",
    "CONTEXT_PREVIEW_MAX_FILES",
    "OutputBuffer",
    "RunnerEnvironment",
    "describe_context",
]
