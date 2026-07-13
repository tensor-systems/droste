from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .models import reject_json_constant
from .runner import BenchmarkRunError

DATASET_ID = "oolongbench/oolong-synth"
DATASET_REVISION = "f0d59eaf0febf130664cfceb710436c8e3216b2b"
DATASET_CONFIG = "default"
DATASET_SPLIT = "validation"
ROW_OFFSET = 1050
ROW_COUNT = 50
DATASET_CATEGORY = "trec_coarse"
CONTEXT_LENGTH = 131072

_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    f"?dataset=oolongbench%2Foolong-synth&config={DATASET_CONFIG}"
    f"&split={DATASET_SPLIT}&offset={ROW_OFFSET}&length={ROW_COUNT}"
    f"&revision={DATASET_REVISION}"
)
_EXPECTED_TASK_IDS = (
    "17000208",
    "17000209",
    "17000210",
    "17000211",
    "17000212",
    "17000213",
    "17000214",
    "17000215",
    "17000216",
    "17000217",
    "17000218",
    "17000219",
    "17000220",
    "17000221",
    "17000222",
    "17000223",
    "17000224",
    "17000225",
    "17000226",
    "17000227",
    "17000228",
    "17000229",
    "17000230",
    "17000231",
    "17000232",
    "17000206",
    "17000207",
    "17000233",
    "17000234",
    "17000235",
    "17000236",
    "17000237",
    "17000238",
    "17000239",
    "17000240",
    "17000241",
    "17000242",
    "17000243",
    "17000244",
    "17000245",
    "17000246",
    "17000247",
    "17000248",
    "17000249",
    "17000250",
    "17000251",
    "17000252",
    "17000253",
    "17000254",
    "17000255",
)
_EXPECTED_CONTEXT_HASHES = (
    "7813cfac8178a89cc21ac25602e5c1d3dbdbd4ad9a1cbfe3726281516d27f969",
    "fa1d459561929df8005cdd5d43e6d0dae6a0c3c076d60bc65ad814f560f1362e",
)


@dataclass(frozen=True)
class MaterializedOolong:
    tasks_path: Path
    tasks_sha256: str
    task_count: int
    context_count: int


def _download_rows() -> bytes:
    request = urllib.request.Request(_ROWS_URL, headers={"User-Agent": "droste-benchmarks/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except Exception as exc:
        raise BenchmarkRunError(f"failed to download pinned OOLONG rows: {exc}") from exc


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


def _validated_rows(raw: bytes) -> list[tuple[int, dict[str, Any], str, str]]:
    try:
        payload = json.loads(raw, parse_constant=reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise BenchmarkRunError(f"pinned OOLONG endpoint returned invalid JSON: {exc}") from exc
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != ROW_COUNT:
        count = len(rows) if isinstance(rows, list) else 0
        raise BenchmarkRunError(f"expected {ROW_COUNT} pinned OOLONG rows; received {count}")

    validated: list[tuple[int, dict[str, Any], str, str]] = []
    for position, item in enumerate(rows):
        expected_index = ROW_OFFSET + position
        if not isinstance(item, dict) or item.get("row_idx") != expected_index:
            raise BenchmarkRunError(f"OOLONG row {position} does not have index {expected_index}")
        row = item.get("row")
        if not isinstance(row, dict):
            raise BenchmarkRunError(f"OOLONG row {expected_index} is not an object")
        task_id = str(row.get("id") or "")
        if task_id != _EXPECTED_TASK_IDS[position]:
            raise BenchmarkRunError(
                f"OOLONG row {expected_index} has task id {task_id!r}; "
                f"expected {_EXPECTED_TASK_IDS[position]!r}"
            )
        if row.get("dataset") != DATASET_CATEGORY:
            raise BenchmarkRunError(f"OOLONG task {task_id} is not {DATASET_CATEGORY}")
        if row.get("context_len") != CONTEXT_LENGTH:
            raise BenchmarkRunError(f"OOLONG task {task_id} is not {CONTEXT_LENGTH} tokens")
        for field in ("question", "answer", "answer_type", "context_window_text"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise BenchmarkRunError(f"OOLONG task {task_id} has invalid {field}")
        context = row["context_window_text"]
        context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
        expected_hash = _EXPECTED_CONTEXT_HASHES[0 if position < 25 else 1]
        if context_hash != expected_hash:
            raise BenchmarkRunError(
                f"OOLONG task {task_id} context has SHA-256 {context_hash}; "
                f"expected {expected_hash}"
            )
        validated.append((expected_index, row, context, context_hash))
    return validated


def materialize_oolong(
    output_dir: Path,
    *,
    fetch: Callable[[], bytes] = _download_rows,
) -> MaterializedOolong:
    """Materialize the pinned public 131K trec_coarse validation slice."""

    validated = _validated_rows(fetch())
    tasks: list[dict[str, Any]] = []
    contexts: dict[str, bytes] = {}
    for row_index, row, context, context_hash in validated:
        contexts.setdefault(context_hash, context.encode("utf-8"))
        tasks.append(
            {
                "answer_type": row["answer_type"],
                "context_len": CONTEXT_LENGTH,
                "context_path": f"contexts/{context_hash}.txt",
                "context_sha256": context_hash,
                "context_window_id": row.get("context_window_id"),
                "dataset": DATASET_CATEGORY,
                "dataset_revision": DATASET_REVISION,
                "id": str(row["id"]),
                "input_subset": row.get("input_subset"),
                "question": row["question"],
                "reference": {
                    "answer": row["answer"],
                    "answer_type": row["answer_type"],
                },
                "row_idx": row_index,
                "task": row.get("task"),
                "task_group": row.get("task_group"),
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
    return MaterializedOolong(
        tasks_path=tasks_path,
        tasks_sha256=hashlib.sha256(tasks_bytes).hexdigest(),
        task_count=len(tasks),
        context_count=len(contexts),
    )
