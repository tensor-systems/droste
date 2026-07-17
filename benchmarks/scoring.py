from __future__ import annotations

import ast
import math
import re
from collections import Counter
from datetime import datetime
from typing import Any

from dateutil import parser as date_parser

from .models import ScorerKind

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_OOLONG_PAIR_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


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


def _normalized_pair(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first <= second else (second, first)


def _oolong_prediction_pairs(prediction: Any) -> set[tuple[int, int]]:
    if isinstance(prediction, dict) and "answer" in prediction:
        prediction = prediction["answer"]
    return {
        _normalized_pair(int(match.group(1)), int(match.group(2)))
        for match in _OOLONG_PAIR_RE.finditer(str(prediction))
    }


def _oolong_reference_pairs(reference: Any) -> set[tuple[int, int]]:
    if isinstance(reference, dict):
        reference = reference.get("pairs")
    if not isinstance(reference, list):
        raise ValueError("oolong_pairs_f1 reference must be a list of pairs")
    pairs: set[tuple[int, int]] = set()
    for index, pair in enumerate(reference):
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(isinstance(item, int) and not isinstance(item, bool) for item in pair)
        ):
            raise ValueError(f"oolong_pairs_f1 reference pair {index} must contain two integers")
        pairs.add(_normalized_pair(pair[0], pair[1]))
    return pairs


def oolong_pairs_f1(prediction: Any, reference: Any) -> float:
    """Score normalized, deduplicated OOLONG-Pairs output as a set of ID pairs."""

    predicted = _oolong_prediction_pairs(prediction)
    expected = _oolong_reference_pairs(reference)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = len(predicted & expected)
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


_OOLONG_COMPARISONS = ("more common", "less common", "same frequency")


def _oolong_text(prediction: Any) -> str:
    if isinstance(prediction, dict) and "answer" in prediction:
        prediction = prediction["answer"]
    return str(prediction)


def _oolong_candidate(prediction: Any) -> str:
    """Parse the candidate using OOLONG-synth's published answer convention."""

    text = _oolong_text(prediction)
    if ":" not in text:
        return text if len(text) < 20 else text.split()[-1]
    candidate = text.rsplit(":", 1)[-1].strip().replace("*", "")
    candidate = candidate.replace("[", "").replace("]", "")
    if len(candidate) >= 20:
        lowered = candidate.casefold()
        for comparison in _OOLONG_COMPARISONS:
            if comparison in lowered:
                return comparison
    return candidate


def _oolong_reference(reference: Any) -> tuple[Any, str]:
    if not isinstance(reference, dict):
        raise ValueError("oolong_official reference must contain answer and answer_type")
    raw_answer = reference.get("answer")
    answer_type = reference.get("answer_type")
    if not isinstance(raw_answer, str) or not isinstance(answer_type, str):
        raise ValueError("oolong_official reference must contain string answer and answer_type")
    try:
        if "datetime" in raw_answer:
            gold: Any = datetime.strptime(raw_answer, "[datetime.date(%Y, %m, %d)]")
        else:
            parsed = ast.literal_eval(raw_answer)
            if not isinstance(parsed, list) or not parsed:
                raise ValueError("answer is not a non-empty list")
            gold = parsed[0]
    except (SyntaxError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid oolong_official answer: {raw_answer!r}") from exc
    return gold, answer_type


def oolong_official(prediction: Any, reference: Any) -> float:
    """Score one OOLONG-synth response with the benchmark's published rule."""

    gold, answer_type = _oolong_reference(reference)
    candidate = _oolong_candidate(prediction)
    if str(candidate) == str(gold):
        return 1.0
    if candidate in _OOLONG_COMPARISONS and candidate in str(gold):
        return 1.0
    if answer_type == "ANSWER_TYPE.NUMERIC":
        try:
            return 0.75 ** abs(int(gold) - int(candidate))
        except (TypeError, ValueError):
            return 0.0
    if answer_type == "ANSWER_TYPE.DATE":
        try:
            return float(date_parser.parse(candidate) == gold)
        except (TypeError, ValueError, OverflowError):
            return 0.0
    return 0.0


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
        return oolong_official(prediction, reference)
    if kind == "oolong_pairs_f1":
        return oolong_pairs_f1(prediction, reference)
    raise ValueError(f"unsupported scorer: {kind}")
