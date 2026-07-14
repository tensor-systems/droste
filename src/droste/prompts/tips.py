"""Legacy tips-profile mapping backed by versioned prompt-pack artifacts."""

from __future__ import annotations

from typing import Literal

from .pack import load_builtin_prompt_pack

TipsProfile = Literal["full", "minimal", "none"]

# Keep the historical mutable-list value shape for import compatibility. The
# execution path resolves immutable tuple-valued PromptPack objects instead.
TIPS_PROFILES: dict[str, list[str]] = {
    profile: list(load_builtin_prompt_pack(profile).tips) for profile in ("full", "minimal", "none")
}
