from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, cast

from .models import ArmSpec, BenchmarkSpec, RunArtifact, SuiteManifest, reject_json_constant
from .scoring import score


class BenchmarkRunError(RuntimeError):
    """The requested benchmark run cannot be executed faithfully."""


_TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(), parse_constant=reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise BenchmarkRunError(f"invalid JSON in {path}: {exc}") from exc


def _resolve(manifest: SuiteManifest, relative: str) -> Path:
    path = (manifest.source_path.parent / relative).resolve()
    benchmark_root = manifest.source_path.parent.parent.resolve()
    if not path.is_relative_to(benchmark_root):
        raise BenchmarkRunError(f"manifest path escapes the benchmark root: {relative}")
    return path


def load_tasks(manifest: SuiteManifest, benchmark: BenchmarkSpec) -> tuple[dict[str, Any], ...]:
    if benchmark.tasks_path is None:
        raise BenchmarkRunError(f"benchmark {benchmark.benchmark_id} has no runnable tasks")
    path = _resolve(manifest, benchmark.tasks_path)
    if benchmark.tasks_sha256 is not None:
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != benchmark.tasks_sha256:
            raise BenchmarkRunError(
                f"tasks for {benchmark.benchmark_id} have SHA-256 {actual_sha256}; "
                f"expected {benchmark.tasks_sha256}"
            )
    value = _load_json(path)
    if not isinstance(value, list) or not value:
        raise BenchmarkRunError(f"tasks for {benchmark.benchmark_id} must be a non-empty array")
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise BenchmarkRunError(f"task {index} for {benchmark.benchmark_id} must be an object")
        task = cast(dict[str, Any], item)
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise BenchmarkRunError(f"task {index} for {benchmark.benchmark_id} has no id")
        if not _TASK_ID_RE.fullmatch(task_id) or "--" in task_id:
            raise BenchmarkRunError(
                f"task id {task_id!r} for {benchmark.benchmark_id} is not a safe artifact id"
            )
        if task_id in seen:
            raise BenchmarkRunError(f"duplicate task id {task_id} in {benchmark.benchmark_id}")
        if "reference" not in task:
            raise BenchmarkRunError(f"task {task_id} has no reference")
        task_tolerance(task, benchmark)
        seen.add(task_id)
        tasks.append(task)
    return tuple(tasks)


def task_tolerance(task: dict[str, Any], benchmark: BenchmarkSpec) -> float:
    task_id = task.get("id", "<unknown>")
    tolerance = task.get("tolerance", 0.0)
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise BenchmarkRunError(f"task {task_id} tolerance must be numeric")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise BenchmarkRunError(f"task {task_id} tolerance must be finite and non-negative")
    if "tolerance" in task and benchmark.scorer != "numeric":
        raise BenchmarkRunError(
            f"task {task_id} declares tolerance for non-numeric scorer {benchmark.scorer}"
        )
    return tolerance


def _load_predictions(manifest: SuiteManifest, arm: ArmSpec) -> dict[str, Any]:
    if arm.predictions_path is None:
        raise BenchmarkRunError(f"arm {arm.arm_id} has no fixture predictions")
    value = _load_json(_resolve(manifest, arm.predictions_path))
    if not isinstance(value, dict):
        raise BenchmarkRunError(f"predictions for {arm.arm_id} must be an object")
    return cast(dict[str, Any], value)


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
    except ValueError as exc:
        raise BenchmarkRunError(f"artifact contains a non-JSON value: {exc}") from exc
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise BenchmarkRunError(f"refusing to overwrite existing artifact: {path}") from exc
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_fixture_suite(manifest: SuiteManifest, output_dir: Path) -> tuple[RunArtifact, ...]:
    """Execute a deterministic fixture suite.

    This path validates manifest loading, task/scorer wiring, artifact identity,
    and reporting without making provider calls or spending money.
    """

    ready = tuple(item for item in manifest.benchmarks if item.status == "ready")
    fixture_arms = tuple(item for item in manifest.arms if item.executor == "fixture")
    if not ready:
        raise BenchmarkRunError("manifest has no ready benchmarks")
    if not fixture_arms:
        blocked = "; ".join(
            f"{item.arm_id}: {item.blocker}" for item in manifest.arms if item.blocker
        )
        raise BenchmarkRunError(
            f"manifest has no fixture arms; live execution is blocked: {blocked}"
        )
    artifacts: list[RunArtifact] = []
    for benchmark in ready:
        tasks = load_tasks(manifest, benchmark)
        for arm in fixture_arms:
            predictions = _load_predictions(manifest, arm)
            benchmark_predictions = predictions.get(benchmark.benchmark_id)
            if not isinstance(benchmark_predictions, dict):
                raise BenchmarkRunError(
                    f"arm {arm.arm_id} has no predictions for {benchmark.benchmark_id}"
                )
            extra = sorted(set(benchmark_predictions) - {task["id"] for task in tasks})
            if extra:
                raise BenchmarkRunError(
                    f"arm {arm.arm_id} has unknown tasks for {benchmark.benchmark_id}: "
                    + ", ".join(extra)
                )
            for task in tasks:
                task_id = cast(str, task["id"])
                if task_id not in benchmark_predictions:
                    raise BenchmarkRunError(
                        f"arm {arm.arm_id} is missing {benchmark.benchmark_id}/{task_id}"
                    )
                prediction = benchmark_predictions[task_id]
                tolerance = task_tolerance(task, benchmark)
                artifact = RunArtifact(
                    suite_id=manifest.suite_id,
                    suite_version=manifest.suite_version,
                    manifest_sha256=manifest.sha256,
                    benchmark_id=benchmark.benchmark_id,
                    task_id=task_id,
                    arm_id=arm.arm_id,
                    status="ok",
                    metric=benchmark.scorer,
                    score=score(
                        benchmark.scorer,
                        prediction,
                        task["reference"],
                        tolerance=tolerance,
                    ),
                    prediction=prediction,
                    reference=task["reference"],
                )
                artifacts.append(artifact)
    artifacts.sort(key=lambda artifact: artifact.artifact_id)
    paths = [output_dir / f"{artifact.artifact_id}.json" for artifact in artifacts]
    if len(set(paths)) != len(paths):
        raise BenchmarkRunError("artifact identities are not unique")
    existing = [path for path in paths if path.exists()]
    if existing:
        raise BenchmarkRunError(f"refusing to overwrite existing artifact: {existing[0]}")
    for artifact, path in zip(artifacts, paths, strict=True):
        _write_json_exclusive(path, artifact.to_dict())
    return tuple(artifacts)
