"""Legacy base-prompt constant backed by the versioned generic pack."""

from __future__ import annotations

from .pack import (
    CODE_OUTPUT_CONTRACT,
    PromptSlots,
    load_builtin_prompt_pack,
    render_prompt_template,
)

_GENERIC_FULL = load_builtin_prompt_pack("full")

# Compatibility for callers importing ``BASE_SYSTEM_PROMPT``. The engine uses
# the resolved pack directly; this value is a projection of the same artifact,
# not a second Python-authored prompt.
BASE_SYSTEM_PROMPT = render_prompt_template(
    _GENERIC_FULL.templates.system,
    PromptSlots(output_contract=CODE_OUTPUT_CONTRACT),
)
