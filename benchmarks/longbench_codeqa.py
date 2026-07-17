from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .models import reject_json_constant
from .runner import BenchmarkRunError

DATASET_ID = "zai-org/LongBench-v2"
DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"
DOMAIN_FILTER = "Code Repository Understanding"
ROW_COUNT = 50

_FILTER_PARAMS = {
    "dataset": DATASET_ID,
    "config": DATASET_CONFIG,
    "split": DATASET_SPLIT,
    "where": f"\"domain\" = '{DOMAIN_FILTER}'",
    "offset": "0",
    "length": "100",
    "revision": DATASET_REVISION,
}
_FILTER_URL = "https://datasets-server.huggingface.co/filter?" + urllib.parse.urlencode(
    _FILTER_PARAMS
)
_EXPECTED_FILTERED_ROWS_SHA256 = "de11e20892981c365442db20bd5f477254e275a002cc09357f84ba3b0afa2d35"


@dataclass(frozen=True)
class MaterializedLongBenchCodeQA:
    tasks_path: Path
    tasks_sha256: str
    task_count: int
    context_count: int


def _download_rows() -> bytes:
    request = urllib.request.Request(
        _FILTER_URL,
        headers={"User-Agent": "droste-benchmarks/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read()
    except Exception as exc:
        raise BenchmarkRunError(f"failed to download pinned LongBench-v2 rows: {exc}") from exc


def _encode_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise BenchmarkRunError(f"refusing to overwrite materialized file: {path}") from exc
    finally:
        Path(temporary).unlink(missing_ok=True)


def _validated_rows(raw: bytes) -> list[tuple[int, dict[str, Any]]]:
    try:
        payload = json.loads(raw, parse_constant=reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise BenchmarkRunError(
            f"pinned LongBench-v2 endpoint returned invalid JSON: {exc}"
        ) from exc
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        count = len(rows) if isinstance(rows, list) else 0
        raise BenchmarkRunError(f"expected {ROW_COUNT} pinned LongBench-v2 rows; received {count}")
    if payload.get("num_rows_total") != ROW_COUNT or payload.get("partial") is not False:
        raise BenchmarkRunError("pinned LongBench-v2 filter response is incomplete")

    rows_sha256 = hashlib.sha256(_encode_json(rows)).hexdigest()
    if rows_sha256 != _EXPECTED_FILTERED_ROWS_SHA256:
        raise BenchmarkRunError(
            f"filtered LongBench-v2 rows have SHA-256 {rows_sha256}; "
            f"expected {_EXPECTED_FILTERED_ROWS_SHA256}"
        )

    required_fields = (
        "_id",
        "domain",
        "sub_domain",
        "difficulty",
        "length",
        "question",
        "choice_A",
        "choice_B",
        "choice_C",
        "choice_D",
        "answer",
        "context",
    )
    validated: list[tuple[int, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    previous_index = -1
    for position, item in enumerate(rows):
        if not isinstance(item, dict):
            raise BenchmarkRunError(f"LongBench-v2 filtered row {position} is not an object")
        row_index = item.get("row_idx")
        row = item.get("row")
        if (
            not isinstance(row_index, int)
            or isinstance(row_index, bool)
            or row_index <= previous_index
        ):
            raise BenchmarkRunError(
                f"LongBench-v2 filtered row {position} has an invalid source index"
            )
        if not isinstance(row, dict):
            raise BenchmarkRunError(f"LongBench-v2 row {row_index} is not an object")
        for field in required_fields:
            if not isinstance(row.get(field), str) or not row[field]:
                raise BenchmarkRunError(f"LongBench-v2 row {row_index} has invalid {field}")
        task_id = row["_id"]
        if task_id in seen_ids:
            raise BenchmarkRunError(f"LongBench-v2 task id {task_id!r} is duplicated")
        if row["domain"] != DOMAIN_FILTER:
            raise BenchmarkRunError(
                f"LongBench-v2 task {task_id} is not in domain {DOMAIN_FILTER!r}"
            )
        if row["answer"] not in {"A", "B", "C", "D"}:
            raise BenchmarkRunError(f"LongBench-v2 task {task_id} has invalid answer")
        if row["difficulty"] not in {"easy", "hard"}:
            raise BenchmarkRunError(f"LongBench-v2 task {task_id} has invalid difficulty")
        if row["length"] not in {"short", "medium", "long"}:
            raise BenchmarkRunError(f"LongBench-v2 task {task_id} has invalid length")
        seen_ids.add(task_id)
        previous_index = row_index
        validated.append((row_index, row))
    return validated


def _question_with_choices(row: dict[str, Any]) -> str:
    return (
        f"{row['question']}\n\n"
        f"A. {row['choice_A']}\n"
        f"B. {row['choice_B']}\n"
        f"C. {row['choice_C']}\n"
        f"D. {row['choice_D']}\n\n"
        "Answer with exactly one letter: A, B, C, or D."
    )


def materialize_longbench_codeqa(
    output_dir: Path,
    *,
    fetch: Callable[[], bytes] = _download_rows,
) -> MaterializedLongBenchCodeQA:
    """Materialize the pinned LongBench-v2 code-repository multiple-choice tasks."""

    validated = _validated_rows(fetch())
    tasks: list[dict[str, Any]] = []
    contexts: dict[str, bytes] = {}
    for row_index, row in validated:
        context = row["context"].encode("utf-8")
        context_hash = hashlib.sha256(context).hexdigest()
        contexts.setdefault(context_hash, context)
        tasks.append(
            {
                "answer_type": "multiple_choice",
                "benchmark": "longbench-v2-codeqa",
                "choices": {
                    "A": row["choice_A"],
                    "B": row["choice_B"],
                    "C": row["choice_C"],
                    "D": row["choice_D"],
                },
                "context_path": f"contexts/{context_hash}.txt",
                "context_sha256": context_hash,
                "dataset_revision": DATASET_REVISION,
                "difficulty": row["difficulty"],
                "domain": row["domain"],
                "id": row["_id"],
                "length": row["length"],
                "question": _question_with_choices(row),
                "reference": row["answer"],
                "row_idx": row_index,
                "source_question": row["question"],
                "sub_domain": row["sub_domain"],
            }
        )

    tasks_bytes = _encode_json(tasks)
    tasks_path = output_dir / "tasks.json"
    targets = [tasks_path, *(output_dir / "contexts" / f"{key}.txt" for key in contexts)]
    existing = next((path for path in targets if path.exists()), None)
    if existing is not None:
        raise BenchmarkRunError(f"refusing to overwrite materialized file: {existing}")
    for context_hash, content in sorted(contexts.items()):
        _write_exclusive(output_dir / "contexts" / f"{context_hash}.txt", content)
    _write_exclusive(tasks_path, tasks_bytes)
    return MaterializedLongBenchCodeQA(
        tasks_path=tasks_path,
        tasks_sha256=hashlib.sha256(tasks_bytes).hexdigest(),
        task_count=len(tasks),
        context_count=len(contexts),
    )
