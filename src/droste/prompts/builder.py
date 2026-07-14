from __future__ import annotations

from dataclasses import dataclass, field


def _default_base() -> str:
    from .base import BASE_SYSTEM_PROMPT

    return BASE_SYSTEM_PROMPT


@dataclass
class SystemPromptBuilder:
    """Build a system prompt from base instructions, schema, and tips."""

    base: str = field(default_factory=_default_base)
    schema: str | None = None
    tips_profile: str | None = None
    additions: str | None = None

    def base_instructions(self) -> str:
        return self.base

    def with_schema(self, schema: str | None) -> "SystemPromptBuilder":
        self.schema = schema
        return self

    def with_tips(self, profile: str | None) -> "SystemPromptBuilder":
        self.tips_profile = profile
        return self

    def with_additions(self, additions: str | None) -> "SystemPromptBuilder":
        self.additions = additions
        return self

    def build(self) -> str:
        parts: list[str] = [self.base]
        if self.schema:
            parts.append("\n## Schema\n" + self.schema)
        tips = ""
        if self.tips_profile:
            from .tips import TIPS_PROFILES

            tips_list = TIPS_PROFILES.get(self.tips_profile, TIPS_PROFILES.get("full", []))
            tips = "\n\n".join(tips_list)
        if tips:
            parts.append("\n## Tips\n" + tips)
        if self.additions:
            parts.append(self.additions)
        return "\n\n".join(part for part in parts if part)
