from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.models import ArtifactError, ManifestError, RunArtifact, load_manifest
from benchmarks.report import ReportError, aggregate, load_artifacts, render_markdown
from benchmarks.runner import BenchmarkRunError, run_fixture_suite
from benchmarks.scoring import exact_match, numeric_score, token_f1

ROOT = Path(__file__).resolve().parents[1]
SMOKE_MANIFEST = ROOT / "benchmarks" / "manifests" / "smoke-v1.json"
PAPER_MANIFEST = ROOT / "benchmarks" / "manifests" / "rlm-paper-v1.json"


def test_manifest_pins_paper_tasks_and_blocks_live_runs() -> None:
    manifest = load_manifest(PAPER_MANIFEST)

    assert manifest.paper is not None
    assert manifest.paper.revision == "arXiv:2512.24601v3"
    assert {benchmark.benchmark_id for benchmark in manifest.benchmarks} == {
        "s-niah",
        "browsecomp-plus-1k",
        "oolong",
        "oolong-pairs",
        "longbench-v2-codeqa",
        "tag-bench",
    }
    assert not manifest.live_run.enabled
    assert [blocker.issue for blocker in manifest.live_run.blockers] == [
        "https://github.com/tensor-systems/modelrelay/issues/1686"
    ]
    assert all(arm.executor == "blocked" for arm in manifest.arms)


def test_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    value = json.loads(SMOKE_MANIFEST.read_text())
    value["typo"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value))

    with pytest.raises(ManifestError, match="unknown fields: typo"):
        load_manifest(path)


def test_manifest_rejects_non_standard_numeric_constants(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        SMOKE_MANIFEST.read_text().replace('"schema_version": 1', '"schema_version": NaN')
    )

    with pytest.raises(ManifestError, match="non-standard JSON numeric constant: NaN"):
        load_manifest(path)


def test_manifest_rejects_artifact_delimiter_in_ids(tmp_path: Path) -> None:
    value = json.loads(SMOKE_MANIFEST.read_text())
    value["arms"][0]["id"] = "fixture--droste"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value))

    with pytest.raises(ManifestError, match="without the '--' delimiter"):
        load_manifest(path)


@pytest.mark.parametrize("model", [{}, [], None])
def test_manifest_validates_any_declared_model_value(tmp_path: Path, model: object) -> None:
    value = json.loads(SMOKE_MANIFEST.read_text())
    value["arms"][0]["model"] = model
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value))

    with pytest.raises(ManifestError, match=r"arms\[0\].model must be an object|provider"):
        load_manifest(path)


def test_scorers_are_deterministic_and_normalized() -> None:
    assert exact_match("  ALPHA\n", "alpha") == 1.0
    assert numeric_score("3.145", 3.14, tolerance=0.01) == 1.0
    assert numeric_score(float("nan"), float("nan")) == 0.0
    assert token_f1("alpha alpha gamma", "alpha beta") == pytest.approx(0.4)
    assert token_f1("", "") == 1.0


def test_smoke_suite_writes_artifacts_and_deterministic_report(tmp_path: Path) -> None:
    manifest = load_manifest(SMOKE_MANIFEST)
    output = tmp_path / "artifacts"

    artifacts = run_fixture_suite(manifest, output)
    loaded = load_artifacts(output, manifest)
    report = render_markdown(manifest, aggregate(loaded))

    assert len(artifacts) == 10
    assert len(list(output.glob("*.json"))) == 10
    assert artifacts == loaded
    assert "| smoke-exact | fixture-droste | exact_match | 1.0000 | 2/2 |" in report
    assert "| smoke-exact | fixture-direct | exact_match | 0.5000 | 2/2 |" in report
    assert report == render_markdown(manifest, aggregate(tuple(reversed(loaded))))


def test_smoke_suite_refuses_to_overwrite_artifacts(tmp_path: Path) -> None:
    manifest = load_manifest(SMOKE_MANIFEST)
    output = tmp_path / "artifacts"
    run_fixture_suite(manifest, output)

    with pytest.raises(BenchmarkRunError, match="refusing to overwrite"):
        run_fixture_suite(manifest, output)


def test_smoke_suite_rejects_non_standard_prediction_constants(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmarks"
    manifests = benchmark_root / "manifests"
    fixtures = benchmark_root / "fixtures" / "smoke"
    manifests.mkdir(parents=True)
    fixtures.mkdir(parents=True)
    manifest_value = json.loads(SMOKE_MANIFEST.read_text())
    for benchmark in manifest_value["benchmarks"]:
        source = ROOT / "benchmarks" / "fixtures" / "smoke" / Path(benchmark["tasks_path"]).name
        (fixtures / source.name).write_text(source.read_text())
        benchmark["tasks_path"] = f"../fixtures/smoke/{source.name}"
    for arm in manifest_value["arms"]:
        prediction_path = fixtures / Path(arm["predictions_path"]).name
        prediction_path.write_text('{"smoke-exact": {"match": NaN}}')
        arm["predictions_path"] = f"../fixtures/smoke/{prediction_path.name}"
    manifest_path = manifests / "smoke.json"
    manifest_path.write_text(json.dumps(manifest_value))

    with pytest.raises(BenchmarkRunError, match="non-standard JSON numeric constant: NaN"):
        run_fixture_suite(load_manifest(manifest_path), tmp_path / "artifacts")
    assert not (tmp_path / "artifacts").exists()


def test_report_rejects_artifacts_from_another_manifest(tmp_path: Path) -> None:
    manifest = load_manifest(SMOKE_MANIFEST)
    output = tmp_path / "artifacts"
    run_fixture_suite(manifest, output)
    path = next(output.glob("*.json"))
    value = json.loads(path.read_text())
    value["manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(value))

    with pytest.raises(ReportError, match="different manifest"):
        load_artifacts(output, manifest)


def test_report_rejects_incomplete_artifact_sets(tmp_path: Path) -> None:
    manifest = load_manifest(SMOKE_MANIFEST)
    output = tmp_path / "artifacts"
    run_fixture_suite(manifest, output)
    next(output.glob("*.json")).unlink()

    with pytest.raises(ReportError, match="artifact set is incomplete"):
        load_artifacts(output, manifest)


def test_report_rejects_metric_not_declared_by_manifest(tmp_path: Path) -> None:
    manifest = load_manifest(SMOKE_MANIFEST)
    output = tmp_path / "artifacts"
    run_fixture_suite(manifest, output)
    path = next(output.glob("smoke-exact*.json"))
    value = json.loads(path.read_text())
    value["metric"] = "token_f1"
    path.write_text(json.dumps(value))

    with pytest.raises(ReportError, match="manifest declares exact_match"):
        load_artifacts(output, manifest)


def test_report_rejects_reference_not_declared_by_task(tmp_path: Path) -> None:
    manifest = load_manifest(SMOKE_MANIFEST)
    output = tmp_path / "artifacts"
    run_fixture_suite(manifest, output)
    path = next(output.glob("smoke-exact--fixture-droste--mismatch.json"))
    value = json.loads(path.read_text())
    value["reference"] = "not Beta"
    path.write_text(json.dumps(value))

    with pytest.raises(ReportError, match="reference does not match"):
        load_artifacts(output, manifest)


def test_report_recomputes_deterministic_scores(tmp_path: Path) -> None:
    manifest = load_manifest(SMOKE_MANIFEST)
    output = tmp_path / "artifacts"
    run_fixture_suite(manifest, output)
    path = next(output.glob("smoke-exact--fixture-direct--mismatch.json"))
    value = json.loads(path.read_text())
    value["score"] = 1.0
    path.write_text(json.dumps(value))

    with pytest.raises(ReportError, match="score does not match"):
        load_artifacts(output, manifest)


def test_report_rejects_model_identity_on_fixture_arm(tmp_path: Path) -> None:
    manifest = load_manifest(SMOKE_MANIFEST)
    output = tmp_path / "artifacts"
    run_fixture_suite(manifest, output)
    path = next(output.glob("*.json"))
    value = json.loads(path.read_text())
    value["provider"] = "modelrelay"
    value["root_model"] = "openai/gpt-5.6"
    path.write_text(json.dumps(value))

    with pytest.raises(ReportError, match="model-free arm"):
        load_artifacts(output, manifest)


@pytest.mark.parametrize(
    ("tolerance", "message"),
    [(-1, "finite and non-negative"), (float("inf"), "non-standard JSON numeric constant")],
)
def test_invalid_tolerance_fails_before_writing_artifacts(
    tmp_path: Path, tolerance: float, message: str
) -> None:
    benchmark_root = tmp_path / "benchmarks"
    manifests = benchmark_root / "manifests"
    manifests.mkdir(parents=True)
    (benchmark_root / "tasks.json").write_text(
        json.dumps([{"id": "task", "reference": 1, "tolerance": tolerance}])
    )
    (benchmark_root / "predictions.json").write_text(json.dumps({"numeric": {"task": 1}}))
    manifest_path = manifests / "invalid-tolerance.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": "invalid-tolerance",
                "suite_version": "1",
                "live_run": {"enabled": True, "blockers": []},
                "benchmarks": [
                    {
                        "id": "numeric",
                        "dataset": "fixture",
                        "dataset_version": "1",
                        "split": "smoke",
                        "scorer": "numeric",
                        "tasks_path": "../tasks.json",
                        "phase": 1,
                        "status": "ready",
                    }
                ],
                "arms": [
                    {
                        "id": "fixture",
                        "method": "fixture",
                        "executor": "fixture",
                        "predictions_path": "../predictions.json",
                    }
                ],
            }
        )
    )
    output = tmp_path / "artifacts"

    with pytest.raises(BenchmarkRunError, match=message):
        run_fixture_suite(load_manifest(manifest_path), output)
    assert not output.exists()


def test_failed_artifact_requires_an_error() -> None:
    with pytest.raises(ArtifactError, match="failed artifacts must have an error"):
        RunArtifact(
            suite_id="suite",
            suite_version="1",
            manifest_sha256="a" * 64,
            benchmark_id="benchmark",
            task_id="task",
            arm_id="arm",
            status="timeout",
            metric="exact_match",
            score=None,
            prediction=None,
            reference="answer",
        )


@pytest.mark.parametrize(
    ("status", "score", "error", "message"),
    [
        ("ok", 1.0, "contradiction", "successful artifacts must not have an error"),
        ("error", 1.0, "failed", "failed artifacts must not have a score"),
    ],
)
def test_artifact_rejects_contradictory_status_fields(
    status: str, score: float, error: str, message: str
) -> None:
    with pytest.raises(ArtifactError, match=message):
        RunArtifact(
            suite_id="suite",
            suite_version="1",
            manifest_sha256="a" * 64,
            benchmark_id="benchmark",
            task_id="task",
            arm_id="arm",
            status=status,  # type: ignore[arg-type]
            metric="exact_match",
            score=score,
            prediction="answer",
            reference="answer",
            error=error,
        )
