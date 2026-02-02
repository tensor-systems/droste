from __future__ import annotations

from typing import Literal

TipsProfile = Literal["full", "minimal", "none"]

TIPS_PROFILES: dict[str, list[str]] = {
    "full": [],
    "minimal": [],
    "none": [],
}
