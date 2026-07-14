from typing import Literal

from .builder import SystemPromptBuilder
from .pack import (
    PROMPT_PACK_SCHEMA_VERSION,
    PROMPT_SLOT_NAMES,
    PromptPack,
    PromptPackBinding,
    PromptPackCatalog,
    PromptPackError,
    PromptPackProvenance,
    PromptPackRecord,
    PromptPolicyDefaults,
    PromptSlots,
    PromptTemplates,
    ResolvedPromptPack,
    infer_model_family,
    load_builtin_prompt_catalog,
    load_prompt_pack,
    parse_prompt_pack,
    render_prompt_template,
    render_system_prompt,
    resolve_prompt_pack,
)

TipsProfile = Literal["full", "minimal", "none"]


def __getattr__(name: str) -> object:
    """Materialize legacy data projections only when callers request them."""
    if name == "BASE_SYSTEM_PROMPT":
        from .base import BASE_SYSTEM_PROMPT

        return BASE_SYSTEM_PROMPT
    if name == "TIPS_PROFILES":
        from .tips import TIPS_PROFILES

        return TIPS_PROFILES
    raise AttributeError(name)


__all__ = [
    "BASE_SYSTEM_PROMPT",
    "SystemPromptBuilder",
    "PROMPT_PACK_SCHEMA_VERSION",
    "PROMPT_SLOT_NAMES",
    "PromptPack",
    "PromptPackBinding",
    "PromptPackCatalog",
    "PromptPackError",
    "PromptPackProvenance",
    "PromptPackRecord",
    "PromptPolicyDefaults",
    "PromptSlots",
    "PromptTemplates",
    "ResolvedPromptPack",
    "infer_model_family",
    "load_builtin_prompt_catalog",
    "load_prompt_pack",
    "parse_prompt_pack",
    "render_prompt_template",
    "render_system_prompt",
    "resolve_prompt_pack",
    "TIPS_PROFILES",
    "TipsProfile",
]
