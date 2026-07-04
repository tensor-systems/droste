"""Built-in LLM clients shipped with the engine."""

from .errors import http_error_excerpt, redact_secrets
from .openai_compat import (
    DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS,
    OpenAICompatClient,
    OpenAICompatSubcallClient,
)

__all__ = [
    "http_error_excerpt",
    "redact_secrets",
    "DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS",
    "OpenAICompatClient",
    "OpenAICompatSubcallClient",
]
