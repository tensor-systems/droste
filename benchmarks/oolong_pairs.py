"""Materialize and evaluate the OOLONG-Pairs tasks from Appendix D.1.

"X or Y" in a predicate means the user has >=1 instance labeled X OR labeled Y (inclusive or).
Date constraints (e.g. "all instances that are a human being for both users must be after January 6, 2023") use vacuous truth: if a user has zero instances of the constrained label, the constraint holds trivially (does not disqualify the pair) — standard logical convention for "all X satisfy P" when there are no X.
For asymmetric predicates (tasks 11-20, "one user has property A, the other has property B"), a pair (u1, u2) matches if EITHER role assignment satisfies it: (u1 has A and u2 has B) OR (u1 has B and u2 has A).
"exactly one instance with X" means count of instances labeled X for that user equals exactly 1 (not >=1).
Pairs are unordered for output purposes: always emit (lower_user_id, higher_user_id), never both orderings, never (id, id).
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import urllib.request
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from datetime import date, datetime
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, TypeAlias

from . import oolong
from .models import reject_json_constant
from .runner import BenchmarkRunError

DATASET_ID = "oolongbench/oolong-synth"
DATASET_REVISION = "f0d59eaf0febf130664cfceb710436c8e3216b2b"
DATASET_CONFIG = "default"
DATASET_SPLIT = "validation"
ROW_OFFSET = 900
ROW_COUNT = 44
DATASET_CATEGORY = "trec_coarse"
CONTEXT_LENGTH = 32768

PREDICTIONS_RELEASE_TAG = "benchmark-data/oolong-pairs-32k-2026-07-17"
PREDICTIONS_ASSET_URL = (
    "https://github.com/tensor-systems/droste/releases/download/"
    "benchmark-data/oolong-pairs-32k-2026-07-17/"
    "oolong-pairs-32k-2026-07-17.tar.gz"
)
PREDICTIONS_ASSET_SHA256 = "6aa129b1df692948a8c2961bfe049cbf68353f0aebff8a26d63b968b1abaa89f"
PREDICTIONS_CACHE_DIR = (
    Path(__file__).resolve().parent / ".data" / "oolong-pairs-32k-2026-07-17-predictions"
)
PREDICTIONS_PATH = PREDICTIONS_CACHE_DIR / "predictions.json"
PREDICTIONS_MATERIALIZE_COMMAND = (
    "python -m benchmarks materialize-oolong-pairs-predictions "
    "--output benchmarks/.data/oolong-pairs-32k-2026-07-17-predictions"
)
_EXPECTED_PREDICTION_COUNT = 60

PAPER_REVISION = "arXiv:2512.24601v3"
PAPER_APPENDIX = "D.1"
SELECTED_ROW_INDEX = ROW_OFFSET
EXPECTED_INSTANCE_COUNT = 787
EXPECTED_USER_COUNT = 231

_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    f"?dataset=oolongbench%2Foolong-synth&config={DATASET_CONFIG}"
    f"&split={DATASET_SPLIT}&offset={ROW_OFFSET}&length={ROW_COUNT}"
    f"&revision={DATASET_REVISION}"
)
_EXPECTED_TASK_IDS = tuple(str(task_id) for task_id in range(15000200, 15000244))
# Rows 920-924 are the five user-group questions sharing context window 0;
# every other row in the pinned slice belongs to the counting group.
_EXPECTED_TASK_GROUPS = ("counting",) * 20 + ("user",) * 5 + ("counting",) * 19
_EXPECTED_CONTEXT_WINDOW_IDS = (0,) * 25 + (1,) * 19
_EXPECTED_CONTEXT_HASHES = (
    "4d761ff162fc69b1f1a182a46bfb56b5704f1e8d622b4dc645ed5ead865ce619",
    "64e3a042daf4bf66fbbb8b59d0a008179d5d4ce4a1554d7bcc11eba7f2204150",
)
_EXPECTED_LABELED_CONTEXT_HASHES = (
    "005ba514b8c60b4dddc6e5ba95020a451ec7cc30fcd879d4c69d8ae08527f5d5",
    "f7f119dbe6a8b2613218ceb487a8b7823e0de49dbae63b4b9a6dd8a252fed0e8",
)

Label: TypeAlias = Literal[
    "description and abstract concept",
    "entity",
    "human being",
    "numeric value",
    "location",
    "abbreviation",
]
Instance: TypeAlias = tuple[Label, date]
Comparison: TypeAlias = Literal["at_least", "exactly"]
DateRelation: TypeAlias = Literal["after", "before"]
PredicateMode: TypeAlias = Literal["symmetric", "asymmetric"]

LABELS = frozenset(
    {
        "description and abstract concept",
        "entity",
        "human being",
        "numeric value",
        "location",
        "abbreviation",
    }
)


@dataclass(frozen=True)
class CountConstraint:
    label: Label
    comparison: Comparison
    count: int


@dataclass(frozen=True)
class DateConstraint:
    label: Label
    relation: DateRelation
    cutoff: date


@dataclass(frozen=True)
class RoleSpec:
    all_of: tuple[CountConstraint, ...] = ()
    any_of: tuple[CountConstraint, ...] = ()
    dates: tuple[DateConstraint, ...] = ()


@dataclass(frozen=True)
class PredicateSpec:
    task_id: int
    question: str
    mode: PredicateMode
    role_a: RoleSpec
    role_b: RoleSpec | None = None


@dataclass(frozen=True)
class MaterializedOolongPairs:
    tasks_path: Path
    provenance_path: Path
    tasks_sha256: str
    task_count: int
    context_count: int


@dataclass(frozen=True)
class MaterializedOolongPairsPredictions:
    predictions_path: Path
    asset_sha256: str
    prediction_count: int


def _at_least(label: Label, count: int = 1) -> CountConstraint:
    return CountConstraint(label, "at_least", count)


def _exactly(label: Label, count: int = 1) -> CountConstraint:
    return CountConstraint(label, "exactly", count)


def _date(label: Label, relation: DateRelation, year: int, month: int, day: int) -> DateConstraint:
    return DateConstraint(label, relation, date(year, month, day))


def _symmetric(
    task_id: int,
    question: str,
    first_label: Label,
    second_label: Label,
    *dates: DateConstraint,
) -> PredicateSpec:
    return PredicateSpec(
        task_id,
        question,
        "symmetric",
        RoleSpec(
            any_of=(_at_least(first_label), _at_least(second_label)),
            dates=dates,
        ),
    )


def _asymmetric(
    task_id: int,
    question: str,
    role_a: tuple[CountConstraint, ...],
    role_b: tuple[CountConstraint, ...],
) -> PredicateSpec:
    return PredicateSpec(
        task_id,
        question,
        "asymmetric",
        RoleSpec(all_of=role_a),
        RoleSpec(all_of=role_b),
    )


TASKS: tuple[PredicateSpec, ...] = (
    _symmetric(
        1,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a numeric value or location. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        "numeric value",
        "location",
    ),
    _symmetric(
        2,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with an entity or human being. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        "entity",
        "human being",
    ),
    _symmetric(
        3,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a description and abstract concept or abbreviation. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        "description and abstract concept",
        "abbreviation",
    ),
    _symmetric(
        4,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a human being or location, and all instances that are a human being for both users must be after January 6, 2023. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        "human being",
        "location",
        _date("human being", "after", 2023, 1, 6),
    ),
    _symmetric(
        5,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with an entity or numeric value, and all instances that are an entity for both users must be before March 15, 2023. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        "entity",
        "numeric value",
        _date("entity", "before", 2023, 3, 15),
    ),
    _symmetric(
        6,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a location or abbreviation. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        "location",
        "abbreviation",
    ),
    _symmetric(
        7,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a description and abstract concept or numeric value, and all instances that are a numeric value for both users must be after February 1, 2023. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        "description and abstract concept",
        "numeric value",
        _date("numeric value", "after", 2023, 2, 1),
    ),
    _symmetric(
        8,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a human being or description and abstract concept. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        "human being",
        "description and abstract concept",
    ),
    _symmetric(
        9,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with an entity or location, and all instances that are a location for both users must be after April 10, 2023. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        "entity",
        "location",
        _date("location", "after", 2023, 4, 10),
    ),
    _symmetric(
        10,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a numeric value or abbreviation, and all instances that are an abbreviation for both users must be before May 20, 2023. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        "numeric value",
        "abbreviation",
        _date("abbreviation", "before", 2023, 5, 20),
    ),
    _asymmetric(
        11,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with entity and one with abbreviation, and the other user has exactly one instance with entity. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        (_at_least("entity"), _at_least("abbreviation")),
        (_exactly("entity"),),
    ),
    _asymmetric(
        12,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least two instances with numeric value, and the other user has at least one instance with location and at least one instance with human being. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        (_at_least("numeric value", 2),),
        (_at_least("location"), _at_least("human being")),
    ),
    _asymmetric(
        13,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has exactly one instance with description and abstract concept, and the other user has at least one instance with abbreviation and at least one instance with entity. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        (_exactly("description and abstract concept"),),
        (_at_least("abbreviation"), _at_least("entity")),
    ),
    _asymmetric(
        14,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with human being and at least one instance with numeric value, and the other user has exactly two instances with location. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        (_at_least("human being"), _at_least("numeric value")),
        (_exactly("location", 2),),
    ),
    _asymmetric(
        15,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with entity, at least one instance with location, and at least one instance with abbreviation, and the other user has exactly one instance with numeric value. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        (_at_least("entity"), _at_least("location"), _at_least("abbreviation")),
        (_exactly("numeric value"),),
    ),
    _asymmetric(
        16,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with description and abstract concept and at least one instance with human being, and the other user has at least two instances with entity and exactly one instance with abbreviation. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        (_at_least("description and abstract concept"), _at_least("human being")),
        (_at_least("entity", 2), _exactly("abbreviation")),
    ),
    _asymmetric(
        17,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has exactly one instance with numeric value, and the other user has at least one instance with location and at least one instance with description and abstract concept. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        (_exactly("numeric value"),),
        (_at_least("location"), _at_least("description and abstract concept")),
    ),
    _asymmetric(
        18,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with abbreviation and exactly one instance with human being, and the other user has at least one instance with entity and at least one instance with numeric value. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        (_at_least("abbreviation"), _exactly("human being")),
        (_at_least("entity"), _at_least("numeric value")),
    ),
    _asymmetric(
        19,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least two instances with location and at least one instance with entity, and the other user has exactly one instance with description and abstract concept and exactly one instance with abbreviation. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        (_at_least("location", 2), _at_least("entity")),
        (_exactly("description and abstract concept"), _exactly("abbreviation")),
    ),
    _asymmetric(
        20,
        "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) such that one user has at least one instance with numeric value and at least one instance with human being, and the other user has at least one instance with location, at least one instance with entity, and exactly one instance with abbreviation. Each of the questions can be labelled as one of the labels (the data does not provide the labels, you need to figure out the label from the semantics of the question): description and abstract concept, entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the format (user_id_1, user_id_2), separated by newlines.",
        (_at_least("numeric value"), _at_least("human being")),
        (_at_least("location"), _at_least("entity"), _exactly("abbreviation")),
    ),
)


def _count_matches(constraint: CountConstraint, instances: Collection[Instance]) -> bool:
    count = sum(label == constraint.label for label, _ in instances)
    if constraint.comparison == "exactly":
        return count == constraint.count
    return count >= constraint.count


def _date_matches(constraint: DateConstraint, instances: Collection[Instance]) -> bool:
    relevant_dates = (
        instance_date for label, instance_date in instances if label == constraint.label
    )
    if constraint.relation == "after":
        return all(instance_date > constraint.cutoff for instance_date in relevant_dates)
    return all(instance_date < constraint.cutoff for instance_date in relevant_dates)


def _role_matches(role: RoleSpec, instances: Collection[Instance]) -> bool:
    return (
        all(_count_matches(constraint, instances) for constraint in role.all_of)
        and (not role.any_of or any(_count_matches(item, instances) for item in role.any_of))
        and all(_date_matches(constraint, instances) for constraint in role.dates)
    )


def evaluate_predicate(
    spec: PredicateSpec,
    first_user_instances: Collection[Instance],
    second_user_instances: Collection[Instance],
) -> bool:
    """Return whether two distinct users' instance collections satisfy a task predicate."""

    if spec.mode == "symmetric":
        return _role_matches(spec.role_a, first_user_instances) and _role_matches(
            spec.role_a, second_user_instances
        )
    if spec.role_b is None:
        raise ValueError("asymmetric predicate is missing role_b")
    return (
        _role_matches(spec.role_a, first_user_instances)
        and _role_matches(spec.role_b, second_user_instances)
    ) or (
        _role_matches(spec.role_b, first_user_instances)
        and _role_matches(spec.role_a, second_user_instances)
    )


def predicate_to_dict(spec: PredicateSpec) -> dict[str, Any]:
    def count_to_dict(value: CountConstraint) -> dict[str, Any]:
        return {"comparison": value.comparison, "count": value.count, "label": value.label}

    def date_to_dict(value: DateConstraint) -> dict[str, Any]:
        return {
            "cutoff": value.cutoff.isoformat(),
            "label": value.label,
            "relation": value.relation,
        }

    def role_to_dict(value: RoleSpec) -> dict[str, Any]:
        return {
            "all_of": [count_to_dict(item) for item in value.all_of],
            "any_of": [count_to_dict(item) for item in value.any_of],
            "date_constraints": [date_to_dict(item) for item in value.dates],
        }

    result: dict[str, Any] = {"mode": spec.mode, "role_a": role_to_dict(spec.role_a)}
    if spec.role_b is not None:
        result["role_b"] = role_to_dict(spec.role_b)
    return result


_LABELED_LINE_RE = re.compile(
    r"^Date: (?P<date>[A-Z][a-z]{2} \d{2}, \d{4}) \|\| "
    r"User: (?P<user_id>\d+) \|\| Instance: .* \|\| Label: (?P<label>.+)$"
)


def parse_labeled_context(context: str) -> dict[int, list[Instance]]:
    """Parse labeled OOLONG context lines into per-user label/date histories."""

    users: defaultdict[int, list[Instance]] = defaultdict(list)
    for line_number, line in enumerate(context.splitlines(), start=1):
        match = _LABELED_LINE_RE.fullmatch(line)
        if match is None:
            if line.startswith("Date:"):
                raise BenchmarkRunError(f"malformed labeled OOLONG line {line_number}")
            continue
        raw_label = match.group("label")
        if raw_label not in LABELS:
            raise BenchmarkRunError(
                f"labeled OOLONG line {line_number} has unknown label {raw_label!r}"
            )
        try:
            instance_date = datetime.strptime(match.group("date"), "%b %d, %Y").date()
        except ValueError as exc:
            raise BenchmarkRunError(f"labeled OOLONG line {line_number} has invalid date") from exc
        users[int(match.group("user_id"))].append((raw_label, instance_date))
    return dict(users)


def _download_rows() -> bytes:
    request = urllib.request.Request(_ROWS_URL, headers={"User-Agent": "droste-benchmarks/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except Exception as exc:
        raise BenchmarkRunError(f"failed to download pinned OOLONG-Pairs rows: {exc}") from exc


def _download_predictions_asset() -> bytes:
    request = urllib.request.Request(
        PREDICTIONS_ASSET_URL, headers={"User-Agent": "droste-benchmarks/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except Exception as exc:
        raise BenchmarkRunError(
            f"failed to download pinned OOLONG-Pairs predictions: {exc}"
        ) from exc


def materialize_oolong_pairs_predictions(
    output_dir: Path,
    *,
    fetch: Callable[[], bytes] = _download_predictions_asset,
) -> MaterializedOolongPairsPredictions:
    """Materialize release-pinned model predictions used by lean artifacts."""

    if output_dir.exists():
        raise BenchmarkRunError(f"refusing to overwrite materialized directory: {output_dir}")
    asset = fetch()
    asset_sha256 = hashlib.sha256(asset).hexdigest()
    if asset_sha256 != PREDICTIONS_ASSET_SHA256:
        raise BenchmarkRunError(
            f"OOLONG-Pairs predictions asset has SHA-256 {asset_sha256}; "
            f"expected {PREDICTIONS_ASSET_SHA256}"
        )

    predictions: dict[str, Any] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(asset), mode="r:gz") as archive:
            artifact_members = [
                member
                for member in archive.getmembers()
                if member.isfile()
                and PurePosixPath(member.name).parent.name == "artifacts"
                and PurePosixPath(member.name).suffix == ".json"
                and not PurePosixPath(member.name).name.startswith("._")
            ]
            for member in sorted(artifact_members, key=lambda item: item.name):
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise BenchmarkRunError(
                        f"could not read OOLONG-Pairs artifact {member.name} from release asset"
                    )
                try:
                    artifact = json.loads(extracted.read(), parse_constant=reject_json_constant)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise BenchmarkRunError(
                        f"OOLONG-Pairs release artifact {member.name} is invalid JSON: {exc}"
                    ) from exc
                if not isinstance(artifact, dict) or "prediction" not in artifact:
                    raise BenchmarkRunError(
                        f"OOLONG-Pairs release artifact {member.name} has no inline prediction"
                    )
                arm_id = artifact.get("arm_id")
                task_id = artifact.get("task_id")
                if not isinstance(arm_id, str) or not arm_id or not isinstance(task_id, str):
                    raise BenchmarkRunError(
                        f"OOLONG-Pairs release artifact {member.name} has invalid identity"
                    )
                key = f"{arm_id}--{task_id}"
                if key in predictions:
                    raise BenchmarkRunError(
                        f"OOLONG-Pairs release asset has duplicate prediction {key}"
                    )
                predictions[key] = artifact["prediction"]
    except (tarfile.TarError, OSError) as exc:
        raise BenchmarkRunError(
            f"OOLONG-Pairs predictions asset is not a valid tarball: {exc}"
        ) from exc

    if len(predictions) != _EXPECTED_PREDICTION_COUNT:
        raise BenchmarkRunError(
            f"OOLONG-Pairs release asset contains {len(predictions)} predictions; "
            f"expected {_EXPECTED_PREDICTION_COUNT}"
        )
    predictions_path = output_dir / "predictions.json"
    oolong._write_exclusive(predictions_path, oolong._encode_json(predictions))
    return MaterializedOolongPairsPredictions(
        predictions_path=predictions_path,
        asset_sha256=asset_sha256,
        prediction_count=len(predictions),
    )


def _validated_labeled_rows(
    raw: bytes,
) -> list[tuple[int, dict[str, Any], str, str, str, str]]:
    try:
        payload = json.loads(raw, parse_constant=reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise BenchmarkRunError(
            f"pinned OOLONG-Pairs endpoint returned invalid JSON: {exc}"
        ) from exc
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        count = len(rows) if isinstance(rows, list) else 0
        raise BenchmarkRunError(f"expected {ROW_COUNT} pinned OOLONG-Pairs rows; received {count}")

    result: list[tuple[int, dict[str, Any], str, str, str, str]] = []
    for position, item in enumerate(rows):
        row_index = ROW_OFFSET + position
        if not isinstance(item, dict) or item.get("row_idx") != row_index:
            raise BenchmarkRunError(f"OOLONG-Pairs row {position} does not have index {row_index}")
        row = item.get("row")
        if not isinstance(row, dict):
            raise BenchmarkRunError(f"OOLONG-Pairs row {row_index} is not an object")
        task_id = str(row.get("id") or "")
        if task_id != _EXPECTED_TASK_IDS[position]:
            raise BenchmarkRunError(
                f"OOLONG-Pairs row {row_index} has task id {task_id!r}; "
                f"expected {_EXPECTED_TASK_IDS[position]!r}"
            )
        if row.get("dataset") != DATASET_CATEGORY:
            raise BenchmarkRunError(f"OOLONG-Pairs task {task_id} is not {DATASET_CATEGORY}")
        if row.get("context_len") != CONTEXT_LENGTH:
            raise BenchmarkRunError(f"OOLONG-Pairs task {task_id} is not {CONTEXT_LENGTH} tokens")
        if row.get("task_group") != _EXPECTED_TASK_GROUPS[position]:
            raise BenchmarkRunError(
                f"OOLONG-Pairs task {task_id} has task group {row.get('task_group')!r}; "
                f"expected {_EXPECTED_TASK_GROUPS[position]!r}"
            )
        if row.get("context_window_id") != _EXPECTED_CONTEXT_WINDOW_IDS[position]:
            raise BenchmarkRunError(
                f"OOLONG-Pairs task {task_id} has context window "
                f"{row.get('context_window_id')!r}; "
                f"expected {_EXPECTED_CONTEXT_WINDOW_IDS[position]}"
            )
        for field in ("question", "answer", "answer_type", "context_window_text"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise BenchmarkRunError(f"OOLONG-Pairs task {task_id} has invalid {field}")
        context = row["context_window_text"]
        context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
        expected_context_hash = _EXPECTED_CONTEXT_HASHES[0 if position < 25 else 1]
        if context_hash != expected_context_hash:
            raise BenchmarkRunError(
                f"OOLONG-Pairs task {task_id} context has SHA-256 {context_hash}; "
                f"expected {expected_context_hash}"
            )
        labeled_context = row.get("context_window_text_with_labels")
        if not isinstance(labeled_context, str) or not labeled_context:
            raise BenchmarkRunError(
                f"OOLONG row {row_index} has invalid context_window_text_with_labels"
            )
        labeled_hash = hashlib.sha256(labeled_context.encode("utf-8")).hexdigest()
        expected_hash = _EXPECTED_LABELED_CONTEXT_HASHES[0 if position < 25 else 1]
        if labeled_hash != expected_hash:
            raise BenchmarkRunError(
                f"OOLONG row {row_index} labeled context has SHA-256 {labeled_hash}; "
                f"expected {expected_hash}"
            )
        result.append((row_index, row, context, context_hash, labeled_context, labeled_hash))
    return result


def _cardinality_pair_count(spec: PredicateSpec, users: dict[int, list[Instance]]) -> int:
    role_a_users = {
        user_id for user_id, instances in users.items() if _role_matches(spec.role_a, instances)
    }
    if spec.mode == "symmetric":
        return len(role_a_users) * (len(role_a_users) - 1) // 2
    if spec.role_b is None:
        raise ValueError("asymmetric predicate is missing role_b")
    role_b_users = {
        user_id for user_id, instances in users.items() if _role_matches(spec.role_b, instances)
    }
    intersection_count = len(role_a_users & role_b_users)
    return (
        len(role_a_users) * len(role_b_users)
        - intersection_count
        - intersection_count * (intersection_count - 1) // 2
    )


def materialize_oolong_pairs(
    output_dir: Path,
    *,
    fetch: Callable[[], bytes] = _download_rows,
) -> MaterializedOolongPairs:
    """Materialize the 20 paper tasks against pinned 32K context-window row 900."""

    validated = _validated_labeled_rows(fetch())
    row_index, row, context, context_hash, labeled_context, labeled_hash = validated[0]
    if row_index != SELECTED_ROW_INDEX:
        raise BenchmarkRunError(
            f"selected OOLONG-Pairs row is {row_index}; expected {SELECTED_ROW_INDEX}"
        )
    users = parse_labeled_context(labeled_context)
    instance_count = sum(len(instances) for instances in users.values())
    if instance_count != EXPECTED_INSTANCE_COUNT:
        raise BenchmarkRunError(
            f"selected OOLONG-Pairs context has {instance_count} instances; "
            f"expected {EXPECTED_INSTANCE_COUNT}"
        )
    if len(users) != EXPECTED_USER_COUNT:
        raise BenchmarkRunError(
            f"selected OOLONG-Pairs context has {len(users)} users; expected {EXPECTED_USER_COUNT}"
        )

    user_pairs = tuple(combinations(sorted(users), 2))
    context_path = f"contexts/{context_hash}.txt"
    tasks: list[dict[str, Any]] = []
    for spec in TASKS:
        answer_key = [
            [first_user, second_user]
            for first_user, second_user in user_pairs
            if evaluate_predicate(spec, users[first_user], users[second_user])
        ]
        cardinality_pair_count = _cardinality_pair_count(spec, users)
        if len(answer_key) != cardinality_pair_count:
            raise BenchmarkRunError(
                f"OOLONG-Pairs task {spec.task_id} pairwise count {len(answer_key)} "
                f"does not match cardinality count {cardinality_pair_count}"
            )
        tasks.append(
            {
                "context_len": CONTEXT_LENGTH,
                "context_path": context_path,
                "context_sha256": context_hash,
                "context_window_id": row.get("context_window_id"),
                "dataset": DATASET_CATEGORY,
                "dataset_revision": DATASET_REVISION,
                "expected_pair_count": len(answer_key),
                "id": str(spec.task_id),
                "predicate": predicate_to_dict(spec),
                "question": spec.question,
                "reference": answer_key,
                "row_idx": row_index,
                "source": {
                    "appendix": PAPER_APPENDIX,
                    "labeled_context_sha256": labeled_hash,
                    "paper_revision": PAPER_REVISION,
                },
            }
        )

    tasks_bytes = oolong._encode_json(tasks)
    tasks_hash = hashlib.sha256(tasks_bytes).hexdigest()
    provenance = {
        "context_sha256": context_hash,
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": DATASET_SPLIT,
        "instance_count": instance_count,
        "labeled_context_sha256": labeled_hash,
        "paper_appendix": PAPER_APPENDIX,
        "paper_revision": PAPER_REVISION,
        "selected_context_row": SELECTED_ROW_INDEX,
        "source_rows_verified": f"{ROW_OFFSET}-{ROW_OFFSET + ROW_COUNT - 1}",
        "task_count": len(tasks),
        "tasks_sha256": tasks_hash,
        "user_count": len(users),
    }
    provenance_bytes = oolong._encode_json(provenance)
    tasks_path = output_dir / "tasks.json"
    provenance_path = output_dir / "provenance.json"
    context_target = output_dir / context_path
    targets = (tasks_path, provenance_path, context_target)
    existing = next((path for path in targets if path.exists()), None)
    if existing is not None:
        raise BenchmarkRunError(f"refusing to overwrite materialized file: {existing}")
    oolong._write_exclusive(context_target, context.encode("utf-8"))
    oolong._write_exclusive(provenance_path, provenance_bytes)
    oolong._write_exclusive(tasks_path, tasks_bytes)
    return MaterializedOolongPairs(
        tasks_path=tasks_path,
        provenance_path=provenance_path,
        tasks_sha256=tasks_hash,
        task_count=len(tasks),
        context_count=1,
    )
