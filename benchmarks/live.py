from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Callable, Iterator

from droste import ModelRelayClient, ModelRelaySubcallClient, RLMConfig, run_rlm
from droste.execution.context import create_execution_context
from droste.policy import PolicyHints
from droste_cli.credentials import CredentialsError, load_credentials
from droste_runner.runner import RunnerEnvironment

from .models import ArmSpec, BenchmarkSpec, RunArtifact, RunStatus, SuiteManifest, Usage
from .runner import (
    BenchmarkRunError,
    _write_json_exclusive,
    load_tasks,
    task_tolerance,
)
from .scoring import score

_PRICING_URL = "https://api.modelrelay.ai/api/v1/pricing"


@dataclass(frozen=True)
class ModelPrice:
    model_id: str
    provider: str
    input_cost_per_million_cents: int
    output_cost_per_million_cents: int


@dataclass(frozen=True)
class PricingSnapshot:
    version: str
    platform_fee_percent: int
    prices: dict[str, ModelPrice]
    payload: dict[str, Any]


class _LiveRunFailure(RuntimeError):
    """A paid live attempt that failed after producing accounting evidence."""

    def __init__(
        self,
        message: str,
        *,
        prediction: Any = None,
        usage: Usage = Usage(),
        iterations: int = 0,
        subcalls: int = 0,
        status: RunStatus | None = None,
    ) -> None:
        super().__init__(message)
        self.prediction = prediction
        self.usage = usage
        self.iterations = iterations
        self.subcalls = subcalls
        self.status = status


def _fetch_pricing(model_ids: set[str]) -> PricingSnapshot:
    request = urllib.request.Request(
        _PRICING_URL, headers={"User-Agent": "droste-benchmarks/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
        data = json.loads(raw)
    except Exception as exc:
        raise BenchmarkRunError(f"cannot load ModelRelay pricing: {exc}") from exc
    models = data.get("models") if isinstance(data, dict) else None
    fee = data.get("platform_fee_percent") if isinstance(data, dict) else None
    if (
        not isinstance(models, list)
        or not isinstance(fee, int)
        or isinstance(fee, bool)
        or fee < 0
    ):
        raise BenchmarkRunError("ModelRelay pricing response has an invalid shape")
    selected: list[dict[str, Any]] = []
    prices: dict[str, ModelPrice] = {}
    for item in models:
        if not isinstance(item, dict) or item.get("model_id") not in model_ids:
            continue
        model_id = str(item["model_id"])
        input_cents = item.get("input_cost_per_million_cents")
        output_cents = item.get("output_cost_per_million_cents")
        provider = item.get("provider")
        if (
            not isinstance(input_cents, int)
            or isinstance(input_cents, bool)
            or input_cents < 0
            or not isinstance(output_cents, int)
            or isinstance(output_cents, bool)
            or output_cents < 0
            or not isinstance(provider, str)
            or not provider
        ):
            raise BenchmarkRunError(f"ModelRelay pricing for {model_id} is incomplete")
        price = ModelPrice(model_id, provider, input_cents, output_cents)
        prices[model_id] = price
        selected.append(
            {
                "model_id": model_id,
                "provider": provider,
                "input_cost_per_million_cents": input_cents,
                "output_cost_per_million_cents": output_cents,
            }
        )
    missing = sorted(model_ids - set(prices))
    if missing:
        raise BenchmarkRunError(f"ModelRelay pricing is missing models: {', '.join(missing)}")
    payload = {
        "source": _PRICING_URL,
        "platform_fee_percent": fee,
        "models": sorted(selected, key=lambda item: item["model_id"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    version = "modelrelay-pricing-sha256:" + hashlib.sha256(encoded).hexdigest()
    return PricingSnapshot(version, fee, prices, payload)


def _cost_microusd(usage: Usage, arm: ArmSpec, pricing: PricingSnapshot) -> int:
    assert arm.model is not None
    root = pricing.prices[arm.model.root_model]
    numerator = (
        usage.root_input_tokens * root.input_cost_per_million_cents
        + usage.root_output_tokens * root.output_cost_per_million_cents
    )
    if arm.model.subcall_model is not None:
        subcall = pricing.prices[arm.model.subcall_model]
        numerator += (
            usage.subcall_input_tokens * subcall.input_cost_per_million_cents
            + usage.subcall_output_tokens * subcall.output_cost_per_million_cents
        )
    base_cost = Decimal(numerator) / Decimal(100)
    total_cost = base_cost * Decimal(100 + pricing.platform_fee_percent) / Decimal(100)
    return int(total_cost.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _worktree_identity() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
        if not status:
            return commit
        digest = hashlib.sha256()
        digest.update(
            subprocess.run(
                ["git", "diff", "--binary", "HEAD"], capture_output=True, check=True
            ).stdout
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            capture_output=True,
            check=True,
        ).stdout.split(b"\0")
        for raw_path in sorted(item for item in untracked if item):
            path = Path(raw_path.decode("utf-8", errors="surrogateescape"))
            digest.update(str(path).encode())
            if path.is_file():
                digest.update(path.read_bytes())
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}+worktree.{digest.hexdigest()[:16]}"


def _context_path(
    manifest: SuiteManifest, benchmark: BenchmarkSpec, task: dict[str, Any]
) -> Path:
    raw = task.get("context_path")
    if not isinstance(raw, str) or not raw:
        raise BenchmarkRunError(f"task {task.get('id')} has no context_path")
    if benchmark.tasks_path is None:
        raise BenchmarkRunError(f"benchmark {benchmark.benchmark_id} has no task path")
    tasks_parent = (manifest.source_path.parent / benchmark.tasks_path).resolve().parent
    path = (tasks_parent / raw).resolve()
    benchmark_root = manifest.source_path.parent.parent.resolve()
    if not path.is_relative_to(benchmark_root):
        raise BenchmarkRunError(f"task context path escapes the benchmark root: {raw}")
    content = path.read_bytes()
    expected = task.get("context_sha256")
    actual = hashlib.sha256(content).hexdigest()
    if not isinstance(expected, str) or actual != expected:
        raise BenchmarkRunError(
            f"task {task.get('id')} context has SHA-256 {actual}; expected {expected}"
        )
    return path


@contextmanager
def _task_timeout(seconds: int) -> Iterator[None]:
    if not hasattr(signal, "setitimer"):
        yield
        return

    def _raise_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError(f"benchmark task exceeded {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    old_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, old_handler)


def _status_for_exception(exc: Exception) -> RunStatus:
    if isinstance(exc, _LiveRunFailure) and exc.status is not None:
        return exc.status
    text = str(exc).casefold()
    if isinstance(exc, TimeoutError) or "timeout" in text or "timed out" in text:
        return "timeout"
    if "context" in text and ("limit" in text or "window" in text or "length" in text):
        return "context_limit"
    if "refusal" in text or "refused" in text or "safety" in text:
        return "refusal"
    return "error"


def _direct_run(
    task: dict[str, Any], context_text: str, arm: ArmSpec, api_key: str, base_url: str
) -> tuple[str, Usage, int, int]:
    assert arm.model is not None and arm.limits is not None
    client = ModelRelayClient(
        model=arm.model.root_model,
        base_url=base_url,
        api_key=api_key,
        temperature=arm.model.temperature,
        max_output_tokens=arm.model.max_root_output_tokens,
        reasoning_effort=arm.model.root_reasoning_effort or "",
        timeout=arm.limits.timeout_seconds,
    )
    prediction = client.responses_create(
        [
            {
                "role": "system",
                "content": (
                    "Answer the question exactly from the supplied benchmark data. "
                    "Do not estimate. End with the answer format requested by the question."
                ),
            },
            {
                "role": "user",
                "content": f"{context_text}\n\nQuestion: {task['question']}",
            },
        ],
        model=arm.model.root_model,
    )
    root = client.total_usage
    return (
        str(prediction),
        Usage(root.prompt_tokens, root.completion_tokens, 0, 0),
        1,
        0,
    )


def _droste_run(
    task: dict[str, Any],
    context_path: Path,
    context_text: str,
    arm: ArmSpec,
    api_key: str,
    base_url: str,
    diagnostic_path: Path,
) -> tuple[str, Usage, int, int]:
    assert arm.model is not None and arm.limits is not None
    if arm.model.subcall_model is None or arm.model.max_subcall_output_tokens is None:
        raise BenchmarkRunError(f"arm {arm.arm_id} has no subcall configuration")
    execution_context = create_execution_context(
        max_calls=arm.limits.max_subcalls,
        max_iterations=arm.limits.max_iterations,
    )
    root = ModelRelayClient(
        model=arm.model.root_model,
        base_url=base_url,
        api_key=api_key,
        temperature=arm.model.temperature,
        max_output_tokens=arm.model.max_root_output_tokens,
        reasoning_effort=arm.model.root_reasoning_effort or "",
        timeout=arm.limits.timeout_seconds,
    )
    subcalls = ModelRelaySubcallClient(
        model=arm.model.subcall_model,
        context=execution_context,
        base_url=base_url,
        api_key=api_key,
        max_calls=arm.limits.max_subcalls,
        max_output_tokens=arm.model.max_subcall_output_tokens,
        temperature=arm.model.temperature,
        reasoning_effort=arm.model.subcall_reasoning_effort or "",
        max_parallel=arm.limits.concurrency,
        timeout=arm.limits.timeout_seconds,
    )
    environment = RunnerEnvironment(
        context={
            "files": [
                {
                    "mime": "text/plain",
                    "name": context_path.name,
                    "path": str(context_path),
                    "size": len(context_text.encode("utf-8")),
                    "text": context_text,
                }
            ]
        },
        registry=None,
        subcalls=subcalls,
        max_output_chars=100_000,
        exec_timeout_ms=0,
    )
    semantic = task.get("answer_type") != "ANSWER_TYPE.USER"
    if semantic:
        benchmark_guidance = (
            "For this OOLONG aggregation task, the requested labels are latent semantic "
            "classes: records do not contain literal label fields. Classify every record "
            "by the expected answer type of its Instance question. Divide record lines into "
            "multiple non-overlapping chunks of roughly 100 records, and use "
            "llm_query_batched. Each subcall must receive the records, the question, and "
            "allowed labels; request a JSON count for every allowed label and require the "
            "counts to sum to the chunk size. Validate each response before aggregating "
            "counts in Python. Retry malformed responses only while the call budget permits. "
            "Do not send the entire "
            "dataset to one subcall, use 250+ record chunks, or ask one subcall for the final "
            "aggregate. The semantic policy requires at least one real llm_query call before "
            "the answer is confirmed; inspection and local aggregation blocks may run without "
            "a subcall."
        )
    else:
        benchmark_guidance = (
            "This OOLONG task asks for an exact statistic over explicit Date, User, and "
            "Instance fields. Parse the record lines and aggregate the requested field in "
            "Python; the caller intentionally did not set the semantic-subcall policy hint."
        )
    try:
        result = run_rlm(
            str(task["question"]),
            environment=environment,
            root_llm=root,
            subcalls=subcalls,
            config=RLMConfig(
                max_iterations=arm.limits.max_iterations,
                max_calls=arm.limits.max_subcalls,
                root_model=arm.model.root_model,
                policy_hints=PolicyHints(semantic=semantic),
            ),
            system_prompt_additions=benchmark_guidance,
            context=execution_context,
        )
    except Exception as exc:
        raise _LiveRunFailure(
            str(exc),
            usage=_combined_usage(root, subcalls),
            subcalls=execution_context.stats.calls_made,
            status=_status_for_exception(exc),
        ) from exc
    usage = _combined_usage(root, subcalls)
    try:
        _write_json_exclusive(
            diagnostic_path,
            {
                "answer": result.answer,
                "ready": result.ready,
                "extracted": result.extracted,
                "error": asdict(result.error) if result.error is not None else None,
                "extract_error": asdict(result.extract_error)
                if result.extract_error is not None
                else None,
                "recovered_error": asdict(result.recovered_error)
                if result.recovered_error is not None
                else None,
                "trajectory": [asdict(item) for item in result.trajectory],
            },
        )
    except Exception as exc:
        raise _LiveRunFailure(
            f"cannot write diagnostic trajectory: {exc}",
            prediction=result.answer or None,
            usage=usage,
            iterations=result.iterations,
            subcalls=result.sub_calls_made,
        ) from exc
    if result.error is not None:
        raise _LiveRunFailure(
            f"{result.error.type}: {result.error.message}",
            prediction=result.answer or None,
            usage=usage,
            iterations=result.iterations,
            subcalls=result.sub_calls_made,
        )
    if result.extracted:
        recovered = result.recovered_error
        detail = f"; recovered {recovered.type}: {recovered.message}" if recovered else ""
        raise _LiveRunFailure(
            f"unconfirmed extracted answer{detail}",
            prediction=result.answer or None,
            usage=usage,
            iterations=result.iterations,
            subcalls=result.sub_calls_made,
        )
    if not result.ready:
        raise _LiveRunFailure(
            "Droste run ended without a confirmed answer",
            prediction=result.answer or None,
            usage=usage,
            iterations=result.iterations,
            subcalls=result.sub_calls_made,
        )
    return result.answer, usage, result.iterations, result.sub_calls_made


def _combined_usage(
    root: ModelRelayClient, subcalls: ModelRelaySubcallClient
) -> Usage:
    root_usage = root.total_usage
    subcall_usage = subcalls.total_usage
    return Usage(
        root_usage.prompt_tokens,
        root_usage.completion_tokens,
        subcall_usage.prompt_tokens,
        subcall_usage.completion_tokens,
    )


def run_modelrelay_suite(
    manifest: SuiteManifest,
    output_dir: Path,
    *,
    benchmark_id: str,
    arm_ids: set[str],
    task_ids: set[str] | None = None,
    limit: int = 0,
    progress: Callable[[str], None] | None = None,
) -> tuple[RunArtifact, ...]:
    if not manifest.live_run.enabled:
        raise BenchmarkRunError("manifest live runs are disabled")
    benchmark = next(
        (item for item in manifest.benchmarks if item.benchmark_id == benchmark_id), None
    )
    if benchmark is None or benchmark.status != "ready":
        raise BenchmarkRunError(f"benchmark {benchmark_id} is not ready")
    arms = [arm for arm in manifest.arms if arm.arm_id in arm_ids]
    missing_arms = sorted(arm_ids - {arm.arm_id for arm in arms})
    if missing_arms:
        raise BenchmarkRunError(f"unknown arms: {', '.join(missing_arms)}")
    if not arms or any(arm.executor != "modelrelay" for arm in arms):
        raise BenchmarkRunError("selected arms must use the modelrelay executor")
    try:
        credentials = load_credentials()
    except CredentialsError as exc:
        raise BenchmarkRunError(str(exc)) from exc
    if credentials is None or credentials.provider != "modelrelay":
        raise BenchmarkRunError("ModelRelay credentials required; run `droste login`")

    tasks = list(load_tasks(manifest, benchmark))
    if task_ids is not None:
        available = {str(task["id"]) for task in tasks}
        missing_tasks = sorted(task_ids - available)
        if missing_tasks:
            raise BenchmarkRunError(f"unknown task ids: {', '.join(missing_tasks)}")
        tasks = [task for task in tasks if task["id"] in task_ids]
    if limit > 0:
        tasks = tasks[:limit]
    if not tasks:
        raise BenchmarkRunError("no tasks selected")

    model_ids = {
        model_id
        for arm in arms
        if arm.model is not None
        for model_id in (arm.model.root_model, arm.model.subcall_model)
        if model_id is not None
    }
    planned_paths = [
        output_dir / f"{benchmark_id}--{arm.arm_id}--{task['id']}.json"
        for task in tasks
        for arm in arms
    ]
    existing = next((path for path in planned_paths if path.exists()), None)
    if existing is not None:
        raise BenchmarkRunError(f"refusing to overwrite existing artifact: {existing}")
    pricing = _fetch_pricing(model_ids)
    provenance_path = output_dir / "_provenance" / "modelrelay-pricing.json"
    if provenance_path.exists():
        try:
            recorded_pricing = json.loads(provenance_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkRunError(f"invalid pricing provenance: {exc}") from exc
        if (
            not isinstance(recorded_pricing, dict)
            or recorded_pricing.get("price_table_version") != pricing.version
        ):
            raise BenchmarkRunError(
                "current pricing differs from the immutable output-directory snapshot; "
                "use a new output directory"
            )
    else:
        _write_json_exclusive(
            provenance_path,
            {"price_table_version": pricing.version, **pricing.payload},
        )
    commit = _worktree_identity()
    artifacts: list[RunArtifact] = []

    for task in tasks:
        context_path = _context_path(manifest, benchmark, task)
        context_text = context_path.read_text(encoding="utf-8")
        for arm in arms:
            assert arm.model is not None and arm.limits is not None
            if progress:
                progress(f"starting {benchmark_id}/{task['id']} arm={arm.arm_id}")
            started = datetime.now(UTC)
            start = time.monotonic()
            prediction: Any = None
            usage = Usage()
            iterations = 0
            subcalls_count = 0
            error: str | None = None
            status = "ok"
            try:
                with _task_timeout(arm.limits.timeout_seconds):
                    if arm.method == "direct-model":
                        prediction, usage, iterations, subcalls_count = _direct_run(
                            task,
                            context_text,
                            arm,
                            credentials.api_key,
                            credentials.base_url,
                        )
                    elif arm.method == "droste":
                        prediction, usage, iterations, subcalls_count = _droste_run(
                            task,
                            context_path,
                            context_text,
                            arm,
                            credentials.api_key,
                            credentials.base_url,
                            output_dir
                            / "_diagnostics"
                            / f"{benchmark_id}--{arm.arm_id}--{task['id']}.json",
                        )
                    else:
                        raise BenchmarkRunError(f"unsupported live method: {arm.method}")
            except Exception as exc:
                if isinstance(exc, _LiveRunFailure):
                    prediction = exc.prediction
                    usage = exc.usage
                    iterations = exc.iterations
                    subcalls_count = exc.subcalls
                status = _status_for_exception(exc)
                error = str(exc)[:1000]
            elapsed_ms = round((time.monotonic() - start) * 1000)
            artifact = RunArtifact(
                suite_id=manifest.suite_id,
                suite_version=manifest.suite_version,
                manifest_sha256=manifest.sha256,
                benchmark_id=benchmark_id,
                task_id=str(task["id"]),
                arm_id=arm.arm_id,
                status=status,
                metric=benchmark.scorer,
                score=(
                    score(
                        benchmark.scorer,
                        prediction,
                        task["reference"],
                        tolerance=task_tolerance(task, benchmark),
                    )
                    if status == "ok"
                    else None
                ),
                prediction=prediction,
                reference=task["reference"],
                usage=usage,
                cost_microusd=_cost_microusd(usage, arm, pricing),
                wall_time_ms=elapsed_ms,
                iterations=iterations,
                subcalls=subcalls_count,
                error=error,
                provider=arm.model.provider,
                root_model=arm.model.root_model,
                subcall_model=arm.model.subcall_model,
                price_table_version=pricing.version,
                started_at=started.isoformat().replace("+00:00", "Z"),
                droste_commit=commit,
            )
            path = output_dir / f"{artifact.artifact_id}.json"
            _write_json_exclusive(path, artifact.to_dict())
            artifacts.append(artifact)
            if progress:
                dollars = artifact.cost_microusd / 1_000_000
                progress(
                    f"finished {artifact.artifact_id} status={artifact.status} "
                    f"score={artifact.score} cost=${dollars:.6f} wall={elapsed_ms / 1000:.1f}s"
                )
    return tuple(artifacts)
