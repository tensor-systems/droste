from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ArtifactError, RunArtifact, SuiteManifest, reject_json_constant
from .runner import BenchmarkRunError, load_tasks, task_tolerance
from .scoring import score


class ReportError(RuntimeError):
    """Artifacts cannot be combined into a trustworthy report."""


@dataclass(frozen=True)
class Aggregate:
    benchmark_id: str
    arm_id: str
    metric: str
    attempted: int
    successful: int
    mean_score: float | None
    total_tokens: int
    cost_microusd: int
    wall_time_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "arm_id": self.arm_id,
            "metric": self.metric,
            "attempted": self.attempted,
            "successful": self.successful,
            "mean_score": self.mean_score,
            "total_tokens": self.total_tokens,
            "cost_microusd": self.cost_microusd,
            "wall_time_ms": self.wall_time_ms,
        }


def load_artifacts(directory: Path, manifest: SuiteManifest) -> tuple[RunArtifact, ...]:
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ReportError(f"no JSON artifacts found in {directory}")
    artifacts: list[RunArtifact] = []
    seen: set[str] = set()
    benchmarks = {benchmark.benchmark_id: benchmark for benchmark in manifest.benchmarks}
    arms = {arm.arm_id: arm for arm in manifest.arms}
    tasks: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for benchmark in manifest.benchmarks:
            if benchmark.status != "ready":
                continue
            for task in load_tasks(manifest, benchmark):
                tasks[(benchmark.benchmark_id, task["id"])] = task
    except BenchmarkRunError as exc:
        raise ReportError(f"cannot load declared tasks: {exc}") from exc
    for path in paths:
        try:
            artifact = RunArtifact.from_dict(
                json.loads(path.read_text(), parse_constant=reject_json_constant)
            )
        except (json.JSONDecodeError, ArtifactError, ValueError) as exc:
            raise ReportError(f"invalid artifact {path}: {exc}") from exc
        if (
            artifact.suite_id != manifest.suite_id
            or artifact.suite_version != manifest.suite_version
        ):
            raise ReportError(f"artifact {path} belongs to a different suite version")
        if artifact.manifest_sha256 != manifest.sha256:
            raise ReportError(f"artifact {path} was produced by a different manifest")
        benchmark = benchmarks.get(artifact.benchmark_id)
        if benchmark is None:
            raise ReportError(f"artifact {path} names undeclared benchmark {artifact.benchmark_id}")
        arm = arms.get(artifact.arm_id)
        if arm is None:
            raise ReportError(f"artifact {path} names undeclared arm {artifact.arm_id}")
        if arm.model is None:
            if any((artifact.provider, artifact.root_model, artifact.subcall_model)):
                raise ReportError(f"artifact {path} records a model for a model-free arm")
        elif (
            artifact.provider != arm.model.provider
            or artifact.root_model != arm.model.root_model
            or artifact.subcall_model != arm.model.subcall_model
        ):
            raise ReportError(f"artifact {path} model identity does not match the manifest arm")
        if artifact.metric != benchmark.scorer:
            raise ReportError(
                f"artifact {path} uses metric {artifact.metric}; "
                f"manifest declares {benchmark.scorer}"
            )
        task = tasks.get((artifact.benchmark_id, artifact.task_id))
        if task is None:
            raise ReportError(
                f"artifact {path} names undeclared task {artifact.benchmark_id}/{artifact.task_id}"
            )
        if artifact.reference != task["reference"] or type(artifact.reference) is not type(
            task["reference"]
        ):
            raise ReportError(f"artifact {path} reference does not match the declared task")
        if artifact.status == "ok":
            expected_score = score(
                benchmark.scorer,
                artifact.prediction,
                task["reference"],
                tolerance=task_tolerance(task, benchmark),
            )
            if artifact.score is None or not math.isclose(
                artifact.score, expected_score, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ReportError(f"artifact {path} score does not match its prediction")
        if artifact.artifact_id in seen:
            raise ReportError(f"duplicate artifact identity: {artifact.artifact_id}")
        seen.add(artifact.artifact_id)
        artifacts.append(artifact)
    runnable_arms = tuple(arm for arm in manifest.arms if arm.executor != "blocked")
    expected: set[str] = set()
    for benchmark in manifest.benchmarks:
        if benchmark.status != "ready":
            continue
        for arm in runnable_arms:
            for benchmark_id, task_id in tasks:
                if benchmark_id == benchmark.benchmark_id:
                    expected.add(f"{benchmark_id}--{arm.arm_id}--{task_id}")
    actual = {artifact.artifact_id for artifact in artifacts}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ReportError(f"artifact set is incomplete; missing: {', '.join(missing)}")
    if extra:
        raise ReportError(f"artifact set contains undeclared runs: {', '.join(extra)}")
    return tuple(artifacts)


def aggregate(artifacts: tuple[RunArtifact, ...]) -> tuple[Aggregate, ...]:
    groups: dict[tuple[str, str, str], list[RunArtifact]] = defaultdict(list)
    for artifact in artifacts:
        groups[(artifact.benchmark_id, artifact.arm_id, artifact.metric)].append(artifact)
    rows: list[Aggregate] = []
    for (benchmark_id, arm_id, metric), items in sorted(groups.items()):
        scores = [item.score for item in items if item.status == "ok" and item.score is not None]
        rows.append(
            Aggregate(
                benchmark_id=benchmark_id,
                arm_id=arm_id,
                metric=metric,
                attempted=len(items),
                successful=len(scores),
                mean_score=sum(scores) / len(items) if items else None,
                total_tokens=sum(item.usage.total_tokens for item in items),
                cost_microusd=sum(item.cost_microusd for item in items),
                wall_time_ms=sum(item.wall_time_ms for item in items),
            )
        )
    return tuple(rows)


def summary_dict(manifest: SuiteManifest, rows: tuple[Aggregate, ...]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite_id": manifest.suite_id,
        "suite_version": manifest.suite_version,
        "manifest_sha256": manifest.sha256,
        "aggregates": [row.to_dict() for row in rows],
    }


def render_markdown(manifest: SuiteManifest, rows: tuple[Aggregate, ...]) -> str:
    lines = [
        f"# {manifest.suite_id} {manifest.suite_version}",
        "",
        f"Manifest SHA-256: `{manifest.sha256}`",
        "",
        "| Benchmark | Arm | Metric | Score | Successful | Tokens | Cost (USD) | Wall time |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        score_text = "—" if row.mean_score is None else f"{row.mean_score:.4f}"
        lines.append(
            f"| {row.benchmark_id} | {row.arm_id} | {row.metric} | {score_text} | "
            f"{row.successful}/{row.attempted} | {row.total_tokens} | "
            f"${row.cost_microusd / 1_000_000:.6f} | {row.wall_time_ms / 1000:.3f}s |"
        )
    return "\n".join(lines) + "\n"
