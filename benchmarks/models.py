from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

SCHEMA_VERSION = 1

ScorerKind: TypeAlias = Literal["exact_match", "numeric", "token_f1", "oolong_official"]
ExecutorKind: TypeAlias = Literal["fixture", "modelrelay", "blocked"]
RunStatus: TypeAlias = Literal["ok", "error", "timeout", "context_limit", "refusal"]

_SCORERS = frozenset({"exact_match", "numeric", "token_f1", "oolong_official"})
_EXECUTORS = frozenset({"fixture", "modelrelay", "blocked"})
_METHODS = frozenset({"droste", "direct-model", "fixture"})
_STATUSES = frozenset({"ok", "error", "timeout", "context_limit", "refusal"})
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class ManifestError(ValueError):
    """A benchmark manifest violates the versioned contract."""


class ArtifactError(ValueError):
    """A run artifact violates the versioned contract."""


def reject_json_constant(value: str) -> None:
    """Reject Python's non-standard NaN and Infinity JSON extensions."""

    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _expect_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must be an object")
    return cast(dict[str, Any], value)


def _expect_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ManifestError(f"{path} must be an array")
    return value


def _expect_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{path} must be a non-empty string")
    return value


def _expect_int(value: Any, path: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ManifestError(f"{path} must be an integer >= {minimum}")
    return value


def _expect_id(value: Any, path: str) -> str:
    identifier = _expect_str(value, path)
    if not _ID_RE.fullmatch(identifier) or "--" in identifier:
        raise ManifestError(
            f"{path} must be a lowercase filesystem-safe id without the '--' delimiter"
        )
    return identifier


def _reject_unknown(data: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ManifestError(f"{path} has unknown fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class PaperReference:
    title: str
    revision: str
    url: str

    @classmethod
    def from_dict(cls, value: Any) -> PaperReference:
        data = _expect_mapping(value, "paper")
        _reject_unknown(data, {"title", "revision", "url"}, "paper")
        return cls(
            title=_expect_str(data.get("title"), "paper.title"),
            revision=_expect_str(data.get("revision"), "paper.revision"),
            url=_expect_str(data.get("url"), "paper.url"),
        )


@dataclass(frozen=True)
class LiveRunBlocker:
    issue: str
    reason: str

    @classmethod
    def from_dict(cls, value: Any, index: int) -> LiveRunBlocker:
        path = f"live_run.blockers[{index}]"
        data = _expect_mapping(value, path)
        _reject_unknown(data, {"issue", "reason"}, path)
        return cls(
            issue=_expect_str(data.get("issue"), f"{path}.issue"),
            reason=_expect_str(data.get("reason"), f"{path}.reason"),
        )


@dataclass(frozen=True)
class LiveRunPolicy:
    enabled: bool
    blockers: tuple[LiveRunBlocker, ...]

    @classmethod
    def from_dict(cls, value: Any) -> LiveRunPolicy:
        data = _expect_mapping(value, "live_run")
        _reject_unknown(data, {"enabled", "blockers"}, "live_run")
        enabled = data.get("enabled")
        if not isinstance(enabled, bool):
            raise ManifestError("live_run.enabled must be a boolean")
        blockers = tuple(
            LiveRunBlocker.from_dict(item, index)
            for index, item in enumerate(_expect_list(data.get("blockers"), "live_run.blockers"))
        )
        if enabled and blockers:
            raise ManifestError("live_run.blockers must be empty when live runs are enabled")
        if not enabled and not blockers:
            raise ManifestError("disabled live runs must name at least one blocker")
        return cls(enabled=enabled, blockers=blockers)


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    dataset: str
    dataset_version: str
    split: str
    scorer: ScorerKind
    tasks_path: str | None
    tasks_sha256: str | None
    phase: int
    status: Literal["ready", "planned"]

    @classmethod
    def from_dict(cls, value: Any, index: int) -> BenchmarkSpec:
        path = f"benchmarks[{index}]"
        data = _expect_mapping(value, path)
        _reject_unknown(
            data,
            {
                "id",
                "dataset",
                "dataset_version",
                "split",
                "scorer",
                "tasks_path",
                "tasks_sha256",
                "phase",
                "status",
            },
            path,
        )
        scorer = _expect_str(data.get("scorer"), f"{path}.scorer")
        if scorer not in _SCORERS:
            raise ManifestError(f"{path}.scorer is unsupported: {scorer}")
        status = _expect_str(data.get("status"), f"{path}.status")
        if status not in {"ready", "planned"}:
            raise ManifestError(f"{path}.status must be ready or planned")
        tasks_path = data.get("tasks_path")
        if tasks_path is not None:
            tasks_path = _expect_str(tasks_path, f"{path}.tasks_path")
        if status == "ready" and tasks_path is None:
            raise ManifestError(f"{path}.tasks_path is required when status is ready")
        tasks_sha256 = data.get("tasks_sha256")
        if tasks_sha256 is not None:
            tasks_sha256 = _expect_str(tasks_sha256, f"{path}.tasks_sha256")
            if not _SHA256_RE.fullmatch(tasks_sha256):
                raise ManifestError(f"{path}.tasks_sha256 must be a lowercase SHA-256 digest")
        if tasks_sha256 is not None and tasks_path is None:
            raise ManifestError(f"{path}.tasks_sha256 requires tasks_path")
        return cls(
            benchmark_id=_expect_id(data.get("id"), f"{path}.id"),
            dataset=_expect_str(data.get("dataset"), f"{path}.dataset"),
            dataset_version=_expect_str(data.get("dataset_version"), f"{path}.dataset_version"),
            split=_expect_str(data.get("split"), f"{path}.split"),
            scorer=cast(ScorerKind, scorer),
            tasks_path=tasks_path,
            tasks_sha256=tasks_sha256,
            phase=_expect_int(data.get("phase"), f"{path}.phase", minimum=1),
            status=cast(Literal["ready", "planned"], status),
        )


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    root_model: str
    subcall_model: str | None
    root_reasoning_effort: str | None
    subcall_reasoning_effort: str | None
    temperature: float | None

    @classmethod
    def from_dict(cls, value: Any, path: str) -> ModelConfig:
        data = _expect_mapping(value, path)
        _reject_unknown(
            data,
            {
                "provider",
                "root_model",
                "subcall_model",
                "root_reasoning_effort",
                "subcall_reasoning_effort",
                "temperature",
            },
            path,
        )
        subcall_model = data.get("subcall_model")
        if subcall_model is not None:
            subcall_model = _expect_str(subcall_model, f"{path}.subcall_model")
        root_reasoning_effort = data.get("root_reasoning_effort")
        if root_reasoning_effort is not None:
            root_reasoning_effort = _expect_str(
                root_reasoning_effort, f"{path}.root_reasoning_effort"
            )
        subcall_reasoning_effort = data.get("subcall_reasoning_effort")
        if subcall_reasoning_effort is not None:
            subcall_reasoning_effort = _expect_str(
                subcall_reasoning_effort, f"{path}.subcall_reasoning_effort"
            )
        temperature = data.get("temperature")
        if temperature is not None and (
            not isinstance(temperature, (int, float)) or isinstance(temperature, bool)
        ):
            raise ManifestError(f"{path}.temperature must be numeric or null")
        if temperature is not None and not math.isfinite(float(temperature)):
            raise ManifestError(f"{path}.temperature must be finite")
        if subcall_model is None and subcall_reasoning_effort is not None:
            raise ManifestError(f"{path} cannot configure subcall settings without a subcall_model")
        return cls(
            provider=_expect_str(data.get("provider"), f"{path}.provider"),
            root_model=_expect_str(data.get("root_model"), f"{path}.root_model"),
            subcall_model=subcall_model,
            root_reasoning_effort=root_reasoning_effort,
            subcall_reasoning_effort=subcall_reasoning_effort,
            temperature=float(temperature) if temperature is not None else None,
        )


@dataclass(frozen=True)
class ExecutionLimits:
    tokens: int
    subcalls: int
    depth: int
    wall_ms: int
    root_output_tokens: int
    subcall_output_tokens: int
    concurrency: int

    @classmethod
    def from_dict(cls, value: Any, path: str) -> ExecutionLimits:
        data = _expect_mapping(value, path)
        _reject_unknown(
            data,
            {
                "tokens", "subcalls", "depth", "wall_ms",
                "root_output_tokens", "subcall_output_tokens", "concurrency",
            },
            path,
        )
        return cls(
            tokens=_expect_int(data.get("tokens"), f"{path}.tokens", minimum=1),
            subcalls=_expect_int(data.get("subcalls"), f"{path}.subcalls", minimum=0),
            depth=_expect_int(data.get("depth"), f"{path}.depth", minimum=0),
            wall_ms=_expect_int(data.get("wall_ms"), f"{path}.wall_ms", minimum=1),
            root_output_tokens=_expect_int(
                data.get("root_output_tokens"), f"{path}.root_output_tokens", minimum=1
            ),
            subcall_output_tokens=_expect_int(
                data.get("subcall_output_tokens"),
                f"{path}.subcall_output_tokens",
                minimum=1,
            ),
            concurrency=_expect_int(data.get("concurrency"), f"{path}.concurrency", minimum=1),
        )


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    method: str
    executor: ExecutorKind
    predictions_path: str | None
    blocker: str | None
    model: ModelConfig | None
    limits: ExecutionLimits | None

    @classmethod
    def from_dict(cls, value: Any, index: int) -> ArmSpec:
        path = f"arms[{index}]"
        data = _expect_mapping(value, path)
        _reject_unknown(
            data,
            {"id", "method", "executor", "predictions_path", "blocker", "model", "limits"},
            path,
        )
        executor = _expect_str(data.get("executor"), f"{path}.executor")
        if executor not in _EXECUTORS:
            raise ManifestError(f"{path}.executor is unsupported: {executor}")
        method = _expect_str(data.get("method"), f"{path}.method")
        if method not in _METHODS:
            raise ManifestError(f"{path}.method is unsupported: {method}")
        predictions_path = data.get("predictions_path")
        blocker = data.get("blocker")
        if predictions_path is not None:
            predictions_path = _expect_str(predictions_path, f"{path}.predictions_path")
        if blocker is not None:
            blocker = _expect_str(blocker, f"{path}.blocker")
        if executor == "fixture" and predictions_path is None:
            raise ManifestError(f"{path}.predictions_path is required for fixture executors")
        if executor == "blocked" and blocker is None:
            raise ManifestError(f"{path}.blocker is required for blocked executors")
        model = ModelConfig.from_dict(data["model"], f"{path}.model") if "model" in data else None
        limits = (
            ExecutionLimits.from_dict(data["limits"], f"{path}.limits")
            if "limits" in data
            else None
        )
        if executor == "fixture" and (
            blocker is not None or model is not None or limits is not None
        ):
            raise ManifestError(f"{path} fixture executors must only declare predictions_path")
        if executor == "blocked" and predictions_path is not None:
            raise ManifestError(f"{path} blocked executors must not declare predictions_path")
        if executor == "blocked" and (model is None or limits is None):
            raise ManifestError(f"{path} blocked live executors require model and limits")
        if executor == "modelrelay" and predictions_path is not None:
            raise ManifestError(f"{path} modelrelay executors must not declare predictions_path")
        if executor == "modelrelay" and blocker is not None:
            raise ManifestError(f"{path} modelrelay executors must not declare a blocker")
        if executor == "modelrelay" and (model is None or limits is None):
            raise ManifestError(f"{path} modelrelay executors require model and limits")
        return cls(
            arm_id=_expect_id(data.get("id"), f"{path}.id"),
            method=method,
            executor=cast(ExecutorKind, executor),
            predictions_path=predictions_path,
            blocker=blocker,
            model=model,
            limits=limits,
        )


@dataclass(frozen=True)
class SuiteManifest:
    schema_version: int
    suite_id: str
    suite_version: str
    paper: PaperReference | None
    live_run: LiveRunPolicy
    benchmarks: tuple[BenchmarkSpec, ...]
    arms: tuple[ArmSpec, ...]
    source_path: Path
    sha256: str

    @classmethod
    def from_dict(cls, value: Any, *, source_path: Path, sha256: str) -> SuiteManifest:
        data = _expect_mapping(value, "manifest")
        _reject_unknown(
            data,
            {
                "schema_version",
                "suite_id",
                "suite_version",
                "paper",
                "live_run",
                "benchmarks",
                "arms",
            },
            "manifest",
        )
        schema_version = _expect_int(data.get("schema_version"), "schema_version", minimum=1)
        if schema_version != SCHEMA_VERSION:
            raise ManifestError(
                f"unsupported schema_version {schema_version}; expected {SCHEMA_VERSION}"
            )
        paper_value = data.get("paper")
        paper = PaperReference.from_dict(paper_value) if paper_value is not None else None
        benchmarks = tuple(
            BenchmarkSpec.from_dict(item, index)
            for index, item in enumerate(_expect_list(data.get("benchmarks"), "benchmarks"))
        )
        arms = tuple(
            ArmSpec.from_dict(item, index)
            for index, item in enumerate(_expect_list(data.get("arms"), "arms"))
        )
        if not benchmarks:
            raise ManifestError("benchmarks must not be empty")
        if not arms:
            raise ManifestError("arms must not be empty")
        benchmark_ids = [item.benchmark_id for item in benchmarks]
        arm_ids = [item.arm_id for item in arms]
        if len(set(benchmark_ids)) != len(benchmark_ids):
            raise ManifestError("benchmark ids must be unique")
        if len(set(arm_ids)) != len(arm_ids):
            raise ManifestError("arm ids must be unique")
        return cls(
            schema_version=schema_version,
            suite_id=_expect_id(data.get("suite_id"), "suite_id"),
            suite_version=_expect_str(data.get("suite_version"), "suite_version"),
            paper=paper,
            live_run=LiveRunPolicy.from_dict(data.get("live_run")),
            benchmarks=benchmarks,
            arms=arms,
            source_path=source_path,
            sha256=sha256,
        )


def load_manifest(path: Path) -> SuiteManifest:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, parse_constant=reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ManifestError(f"invalid JSON in {path}: {exc}") from exc
    return SuiteManifest.from_dict(
        value,
        source_path=path.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


@dataclass(frozen=True)
class Usage:
    root_input_tokens: int = 0
    root_output_tokens: int = 0
    subcall_input_tokens: int = 0
    subcall_output_tokens: int = 0

    def __post_init__(self) -> None:
        for name in (
            "root_input_tokens",
            "root_output_tokens",
            "subcall_input_tokens",
            "subcall_output_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ArtifactError(f"usage.{name} must be a non-negative integer")

    @property
    def total_tokens(self) -> int:
        return (
            self.root_input_tokens
            + self.root_output_tokens
            + self.subcall_input_tokens
            + self.subcall_output_tokens
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "root_input_tokens": self.root_input_tokens,
            "root_output_tokens": self.root_output_tokens,
            "subcall_input_tokens": self.subcall_input_tokens,
            "subcall_output_tokens": self.subcall_output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class RunArtifact:
    suite_id: str
    suite_version: str
    manifest_sha256: str
    benchmark_id: str
    task_id: str
    arm_id: str
    status: RunStatus
    metric: ScorerKind
    score: float | None
    prediction: Any
    reference: Any
    usage: Usage = Usage()
    cost_microusd: int = 0
    wall_time_ms: int = 0
    iterations: int = 0
    subcalls: int = 0
    error: str | None = None
    provider: str | None = None
    root_model: str | None = None
    subcall_model: str | None = None
    price_table_version: str | None = None
    started_at: str | None = None
    droste_commit: str = "fixture"

    def __post_init__(self) -> None:
        for name in ("suite_id", "benchmark_id", "task_id", "arm_id"):
            value = getattr(self, name)
            if not _ID_RE.fullmatch(value) or "--" in value:
                raise ArtifactError(
                    f"{name} must be a lowercase filesystem-safe id without the '--' delimiter"
                )
        if not _SHA256_RE.fullmatch(self.manifest_sha256):
            raise ArtifactError("manifest_sha256 must be a lowercase SHA-256 digest")
        if self.status not in _STATUSES:
            raise ArtifactError(f"unsupported run status: {self.status}")
        if self.metric not in _SCORERS:
            raise ArtifactError(f"unsupported metric: {self.metric}")
        if self.status == "ok" and self.score is None:
            raise ArtifactError("successful artifacts must have a score")
        if self.status == "ok" and self.error is not None:
            raise ArtifactError("successful artifacts must not have an error")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ArtifactError("score must be between 0 and 1")
        if self.status != "ok" and not self.error:
            raise ArtifactError("failed artifacts must have an error")
        if self.status != "ok" and self.score is not None:
            raise ArtifactError("failed artifacts must not have a score")
        for name in ("cost_microusd", "wall_time_ms", "iterations", "subcalls"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ArtifactError(f"{name} must be a non-negative integer")
        if self.cost_microusd and not self.price_table_version:
            raise ArtifactError("paid artifacts must name price_table_version")
        if (self.provider is None) != (self.root_model is None):
            raise ArtifactError("provider and root_model must be recorded together")

    @property
    def artifact_id(self) -> str:
        return f"{self.benchmark_id}--{self.arm_id}--{self.task_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "manifest_sha256": self.manifest_sha256,
            "benchmark_id": self.benchmark_id,
            "task_id": self.task_id,
            "arm_id": self.arm_id,
            "status": self.status,
            "metric": self.metric,
            "score": self.score,
            "prediction": self.prediction,
            "reference": self.reference,
            "usage": self.usage.to_dict(),
            "cost_microusd": self.cost_microusd,
            "wall_time_ms": self.wall_time_ms,
            "iterations": self.iterations,
            "subcalls": self.subcalls,
            "error": self.error,
            "provider": self.provider,
            "root_model": self.root_model,
            "subcall_model": self.subcall_model,
            "price_table_version": self.price_table_version,
            "started_at": self.started_at,
            "droste_commit": self.droste_commit,
        }

    @classmethod
    def from_dict(cls, value: Any) -> RunArtifact:
        if not isinstance(value, dict):
            raise ArtifactError("artifact must be an object")
        data = cast(dict[str, Any], value)
        expected_fields = {
            "schema_version",
            "suite_id",
            "suite_version",
            "manifest_sha256",
            "benchmark_id",
            "task_id",
            "arm_id",
            "status",
            "metric",
            "score",
            "prediction",
            "reference",
            "usage",
            "cost_microusd",
            "wall_time_ms",
            "iterations",
            "subcalls",
            "error",
            "provider",
            "root_model",
            "subcall_model",
            "price_table_version",
            "started_at",
            "droste_commit",
        }
        unknown = sorted(set(data) - expected_fields)
        if unknown:
            raise ArtifactError(f"artifact has unknown fields: {', '.join(unknown)}")
        missing = sorted(expected_fields - set(data))
        if missing:
            raise ArtifactError(f"artifact is missing fields: {', '.join(missing)}")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ArtifactError(f"artifact schema_version must be {SCHEMA_VERSION}")
        usage_value = data.get("usage")
        if not isinstance(usage_value, dict):
            raise ArtifactError("usage must be an object")
        expected_usage_fields = {
            "root_input_tokens",
            "root_output_tokens",
            "subcall_input_tokens",
            "subcall_output_tokens",
            "total_tokens",
        }
        unknown_usage = sorted(set(usage_value) - expected_usage_fields)
        if unknown_usage:
            raise ArtifactError(f"usage has unknown fields: {', '.join(unknown_usage)}")

        def required_string(name: str) -> str:
            value = data.get(name)
            if not isinstance(value, str) or not value:
                raise ArtifactError(f"{name} must be a non-empty string")
            return value

        def nonnegative_int(name: str, source: dict[str, Any] = data) -> int:
            value = source.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ArtifactError(f"{name} must be a non-negative integer")
            return value

        status = required_string("status")
        metric = required_string("metric")
        score = data.get("score")
        if score is not None and (not isinstance(score, (int, float)) or isinstance(score, bool)):
            raise ArtifactError("score must be numeric or null")

        def optional_string(name: str) -> str | None:
            value = data.get(name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ArtifactError(f"{name} must be a non-empty string or null")
            return value

        usage = Usage(
            root_input_tokens=nonnegative_int("root_input_tokens", usage_value),
            root_output_tokens=nonnegative_int("root_output_tokens", usage_value),
            subcall_input_tokens=nonnegative_int("subcall_input_tokens", usage_value),
            subcall_output_tokens=nonnegative_int("subcall_output_tokens", usage_value),
        )
        if nonnegative_int("total_tokens", usage_value) != usage.total_tokens:
            raise ArtifactError("usage.total_tokens does not match the token components")
        return cls(
            suite_id=required_string("suite_id"),
            suite_version=required_string("suite_version"),
            manifest_sha256=required_string("manifest_sha256"),
            benchmark_id=required_string("benchmark_id"),
            task_id=required_string("task_id"),
            arm_id=required_string("arm_id"),
            status=cast(RunStatus, status),
            metric=cast(ScorerKind, metric),
            score=float(score) if score is not None else None,
            prediction=data.get("prediction"),
            reference=data.get("reference"),
            usage=usage,
            cost_microusd=nonnegative_int("cost_microusd"),
            wall_time_ms=nonnegative_int("wall_time_ms"),
            iterations=nonnegative_int("iterations"),
            subcalls=nonnegative_int("subcalls"),
            error=optional_string("error"),
            provider=optional_string("provider"),
            root_model=optional_string("root_model"),
            subcall_model=optional_string("subcall_model"),
            price_table_version=optional_string("price_table_version"),
            started_at=optional_string("started_at"),
            droste_commit=required_string("droste_commit"),
        )
