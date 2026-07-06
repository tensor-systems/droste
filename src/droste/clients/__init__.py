"""Built-in LLM clients shipped with the engine."""

from .anthropic import AnthropicClient, AnthropicSubcallClient
from .errors import http_error_excerpt, redact_secrets
from .modelrelay import ModelRelayClient, ModelRelaySubcallClient
from .openai_compat import (
    DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS,
    OpenAICompatClient,
    OpenAICompatSubcallClient,
)

__all__ = [
    "http_error_excerpt",
    "redact_secrets",
    "AnthropicClient",
    "AnthropicSubcallClient",
    "ModelRelayClient",
    "ModelRelaySubcallClient",
    "DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS",
    "OpenAICompatClient",
    "OpenAICompatSubcallClient",
]
