"""Core RLM loop and protocol definitions."""

from .capabilities import (
    CapabilityAnnotator,
    CapabilityBroker,
    CapabilityCall,
    CapabilityCallError,
    CapabilityDescriptor,
    CapabilityError,
    CapabilityErrorCode,
    CapabilityGuard,
    CapabilityId,
    CapabilityKind,
    CapabilityManifest,
    CapabilityMetadata,
    CapabilityMetric,
    CapabilityObserver,
    CapabilityOutcome,
    CapabilityRegistration,
    CapabilityResult,
    CapabilityResultHandle,
    CapabilityStatus,
    EvidenceRef,
    SideEffect,
    generate_binding,
    validate_call,
)
from .clients.anthropic import AnthropicClient, AnthropicSubcallClient
from .clients.modelrelay import ModelRelayClient, ModelRelaySubcallClient
from .clients.openai_compat import OpenAICompatClient, OpenAICompatSubcallClient
from .environments import EnvironmentConfig, create_environment, create_environment_context
from .exceptions import (
    BatchItemError,
    BatchItemErrorDetails,
    PolicyError,
    RLMError,
    SubcallBudgetExceeded,
)
from .execution.config import (
    DEFAULT_MAX_CALLS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_OUTPUT_CHARS,
    ExecutionConfig,
)
from .execution.context import ExecutionContext, create_execution_context
from .execution.progress import ProgressCallback, emit_progress
from .execution.stats import ExecutionStats
from .loop.code_extractor import extract_code_block
from .loop.rlm import RLMConfig, RLMResult, run_rlm
from .policy import PolicyHints
from .prompts.builder import SystemPromptBuilder
from .prompts.pack import (
    PromptPack,
    PromptPackBinding,
    PromptPackCatalog,
    PromptPackError,
    PromptPackRecord,
    load_prompt_pack,
    resolve_prompt_pack,
)
from .protocols.data_source import DataSource, DataSourceCapabilities, SearchResult
from .protocols.environment import EnvCapabilities, ExecutionResult, RLMEnvironment
from .protocols.llm_client import LLMClient, TokenUsage
from .protocols.subcall_client import SubcallClient, SubcallOutputTokenLimitProvider
from .registry import DataSourceRegistry
from .structured import aggregate_json_counts, structured_batch, validate_json

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
    "EnvironmentConfig",
    "create_environment",
    "create_environment_context",
    "ExecutionContext",
    "create_execution_context",
    "ProgressCallback",
    "emit_progress",
    "RLMError",
    "BatchItemError",
    "BatchItemErrorDetails",
    "PolicyError",
    "SubcallBudgetExceeded",
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
    "SubcallOutputTokenLimitProvider",
    "SystemPromptBuilder",
    "PromptPack",
    "PromptPackBinding",
    "PromptPackCatalog",
    "PromptPackError",
    "PromptPackRecord",
    "load_prompt_pack",
    "resolve_prompt_pack",
    "DataSourceRegistry",
    "AnthropicClient",
    "AnthropicSubcallClient",
    "ModelRelayClient",
    "ModelRelaySubcallClient",
    "OpenAICompatClient",
    "OpenAICompatSubcallClient",
    "structured_batch",
    "validate_json",
    "aggregate_json_counts",
    "CapabilityBroker",
    "CapabilityCall",
    "CapabilityCallError",
    "CapabilityDescriptor",
    "CapabilityError",
    "CapabilityErrorCode",
    "CapabilityGuard",
    "CapabilityId",
    "CapabilityKind",
    "CapabilityManifest",
    "CapabilityMetadata",
    "CapabilityMetric",
    "CapabilityObserver",
    "CapabilityOutcome",
    "CapabilityRegistration",
    "CapabilityResult",
    "CapabilityResultHandle",
    "CapabilityStatus",
    "CapabilityAnnotator",
    "EvidenceRef",
    "SideEffect",
    "generate_binding",
    "validate_call",
]
