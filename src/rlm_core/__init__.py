"""Core RLM loop and protocol definitions."""

from .loop.rlm import run_rlm, RLMConfig, RLMResult
from .loop.code_extractor import extract_code_block
from .execution.config import (
    DEFAULT_MAX_CALLS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_OUTPUT_CHARS,
    ExecutionConfig,
)
from .execution.stats import ExecutionStats
from .execution.context import ExecutionContext, create_execution_context
from .execution.progress import ProgressCallback, emit_progress
from .exceptions import RLMError, PolicyError
from .policy import PolicyHints
from .protocols.environment import RLMEnvironment, EnvCapabilities, ExecutionResult
from .protocols.data_source import DataSource, SearchResult, DataSourceCapabilities
from .protocols.llm_client import LLMClient, TokenUsage
from .protocols.subcall_client import SubcallClient
from .prompts.builder import SystemPromptBuilder
from .registry import DataSourceRegistry

__all__ = [
    "run_rlm",
    "RLMConfig",
    "RLMResult",
    "extract_code_block",
    "DEFAULT_MAX_CALLS",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "ExecutionConfig",
    "ExecutionStats",
    "ExecutionContext",
    "create_execution_context",
    "ProgressCallback",
    "emit_progress",
    "RLMError",
    "PolicyError",
    "PolicyHints",
    "RLMEnvironment",
    "EnvCapabilities",
    "ExecutionResult",
    "DataSource",
    "SearchResult",
    "DataSourceCapabilities",
    "LLMClient",
    "TokenUsage",
    "SubcallClient",
    "SystemPromptBuilder",
    "DataSourceRegistry",
]
