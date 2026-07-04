from __future__ import annotations

import re


def extract_code_block(response: str, block_type: str = "python") -> str | None:
    """Extract a fenced code block from the response."""
    pattern = rf"```{block_type}\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
