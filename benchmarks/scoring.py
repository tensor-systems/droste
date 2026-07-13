from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .models import ScorerKind

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _normalized_text(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _tokens(value: Any) -> list[str]:
    return _TOKEN_RE.findall(_normalized_text(value))


def exact_match(prediction: Any, reference: Any) -> float:
    return float(_normalized_text(prediction) == _normalized_text(reference))


def numeric_score(prediction: Any, reference: Any, *, tolerance: float = 0.0) -> float:
    try:
        predicted = float(prediction)
        expected = float(reference)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(predicted) or not math.isfinite(expected):
        return 0.0
    return float(math.isclose(predicted, expected, rel_tol=0.0, abs_tol=tolerance))


def token_f1(prediction: Any, reference: Any) -> float:
    predicted = Counter(_tokens(prediction))
    expected = Counter(_tokens(reference))
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def score(
    kind: ScorerKind,
    prediction: Any,
    reference: Any,
    *,
    tolerance: float = 0.0,
) -> float:
    if kind == "exact_match":
        return exact_match(prediction, reference)
    if kind == "numeric":
        return numeric_score(prediction, reference, tolerance=tolerance)
    if kind == "token_f1":
        return token_f1(prediction, reference)
    if kind == "oolong_official":
        raise ValueError("oolong_official requires the pinned upstream scorer adapter")
    raise ValueError(f"unsupported scorer: {kind}")
