from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from benchmarks.cli import main as benchmark_main
from benchmarks.live import (
    _OOLONG_PAIRS_GUIDANCE,
    _OOLONG_SEMANTIC_GUIDANCE,
    ModelPrice,
    PricingSnapshot,
    _budget_stop_reason,
    _context_path,
    _cost_microusd,
    _direct_run,
    _LiveRunFailure,
    _status_for_exception,
)
from benchmarks.models import ArtifactError, ManifestError, RunArtifact, Usage, load_manifest
from benchmarks.oolong import materialize_oolong
from benchmarks.oolong_pairs import TASKS as OOLONG_PAIR_TASKS
from benchmarks.oolong_pairs import evaluate_predicate, parse_labeled_context
from benchmarks.report import ReportError, aggregate, load_artifacts, render_markdown
from benchmarks.runner import BenchmarkRunError, run_fixture_suite
from benchmarks.scoring import (
    exact_match,
    numeric_score,
    oolong_official,
    oolong_pairs_f1,
    token_f1,
)
from droste.protocols.llm_client import TokenUsage

ROOT = Path(__file__).resolve().parents[1]
SMOKE_MANIFEST = ROOT / "benchmarks" / "manifests" / "smoke-v1.json"
PAPER_MANIFEST = ROOT / "benchmarks" / "manifests" / "rlm-paper-v1.json"


def test_manifest_pins_paper_tasks_and_enables_published_live_runs() -> None:
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
    assert manifest.live_run.enabled
    assert not manifest.live_run.blockers
    assert {arm.arm_id for arm in manifest.arms} == {
        "direct-sol",
        "direct-terra",
        "droste-terra-luna",
        "direct-sol-pairs",
        "direct-terra-pairs",
        "droste-terra-luna-pairs",
    }
    assert all(arm.executor == "modelrelay" for arm in manifest.arms)
    oolong = next(item for item in manifest.benchmarks if item.benchmark_id == "oolong")
    assert oolong.status == "ready"
    assert oolong.tasks_sha256 == "28abaefcbcba1d843a384115a1217ea0017201b13074c95de6feb313c40c8da4"
    oolong_pairs = next(item for item in manifest.benchmarks if item.benchmark_id == "oolong-pairs")
    assert oolong_pairs.status == "ready"
    assert oolong_pairs.scorer == "oolong_pairs_f1"
    assert oolong_pairs.tasks_sha256 == (
        "169a2aaddc8603128f672d32f9aa8a2e0565974d91b6468b7431654dd81bde40"
    )


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


def test_manifest_rejects_unknown_execution_method(tmp_path: Path) -> None:
    value = json.loads(SMOKE_MANIFEST.read_text())
    value["arms"][0]["method"] = "typo"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value))

    with pytest.raises(ManifestError, match="method is unsupported: typo"):
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


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        (
            "Label: human being",
            {"answer": "['human being']", "answer_type": "ANSWER_TYPE.LABEL"},
            1.0,
        ),
        (
            "Answer: human being is more common than abbreviation",
            {
                "answer": "['human being is more common than abbreviation']",
                "answer_type": "ANSWER_TYPE.COMPARISON",
            },
            1.0,
        ),
        (
            "Answer: 38",
            {"answer": "[40]", "answer_type": "ANSWER_TYPE.NUMERIC"},
            0.75**2,
        ),
        (
            "Answer: unknown",
            {"answer": "[40]", "answer_type": "ANSWER_TYPE.NUMERIC"},
            0.0,
        ),
        (
            "Date: January 5, 2024",
            {
                "answer": "[datetime.date(2024, 1, 5)]",
                "answer_type": "ANSWER_TYPE.DATE",
            },
            1.0,
        ),
    ],
)
def test_oolong_official_scorer(prediction: object, reference: object, expected: float) -> None:
    assert oolong_official(prediction, reference) == pytest.approx(expected)


def test_oolong_official_rejects_incomplete_reference() -> None:
    with pytest.raises(ValueError, match="answer and answer_type"):
        oolong_official("Answer: 1", {"answer": "[1]"})


def test_oolong_pairs_f1_parses_normalizes_and_deduplicates_pairs() -> None:
    prediction = "(2,1)\n(1, 2)\n (3,4)\nmalformed (5; 6)\n(8, 9)"
    reference = [[1, 2], [3, 4], [6, 7]]

    assert oolong_pairs_f1(prediction, reference) == pytest.approx(2 / 3)


def test_oolong_pairs_f1_defines_empty_set_edges() -> None:
    assert oolong_pairs_f1("no pairs", []) == 1.0
    assert oolong_pairs_f1("no pairs", [[1, 2]]) == 0.0
    assert oolong_pairs_f1("(1, 2)", []) == 0.0


def test_oolong_pairs_f1_rejects_malformed_reference() -> None:
    with pytest.raises(ValueError, match="pair 0 must contain two integers"):
        oolong_pairs_f1("(1, 2)", [[1, "2"]])


def test_oolong_pairs_predicates_cover_recorded_semantics() -> None:
    task_by_id = {task.task_id: task for task in OOLONG_PAIR_TASKS}

    assert evaluate_predicate(
        task_by_id[1],
        [("numeric value", date(2024, 1, 1))],
        [("location", date(2024, 1, 2))],
    )

    exact_entity_user = [("entity", date(2024, 1, 1))]
    entity_and_abbreviation_user = [
        ("entity", date(2024, 1, 1)),
        ("entity", date(2024, 1, 2)),
        ("abbreviation", date(2024, 1, 3)),
    ]
    assert evaluate_predicate(task_by_id[11], exact_entity_user, entity_and_abbreviation_user)
    assert not evaluate_predicate(
        task_by_id[11], entity_and_abbreviation_user, entity_and_abbreviation_user
    )

    human_after_cutoff = [("human being", date(2023, 1, 7))]
    location_with_no_human = [("location", date(2022, 1, 1))]
    assert evaluate_predicate(task_by_id[4], human_after_cutoff, location_with_no_human)
    assert not evaluate_predicate(
        task_by_id[4],
        [("human being", date(2023, 1, 6))],
        location_with_no_human,
    )


def test_oolong_pairs_task_structure_and_labeled_context_parser() -> None:
    assert [task.task_id for task in OOLONG_PAIR_TASKS] == list(range(1, 21))
    assert {task.task_id for task in OOLONG_PAIR_TASKS if task.role_a.dates} == {4, 5, 7, 9, 10}
    parsed = parse_labeled_context(
        "Header\n\n"
        "Date: Sep 06, 2023 || User: 14512 || Instance: What is a tonne ? "
        "|| Label: description and abstract concept\n"
        "Date: Jan 14, 2024 || User: 14512 || Instance: Who was it? "
        "|| Label: human being\n"
        "Date: Jun 21, 2023 || User: 16295 || Instance: Where is it? "
        "|| Label: location\n"
    )

    assert parsed == {
        14512: [
            ("description and abstract concept", date(2023, 9, 6)),
            ("human being", date(2024, 1, 14)),
        ],
        16295: [("location", date(2023, 6, 21))],
    }


def test_live_cost_uses_integer_microusd_and_snapshotted_fee() -> None:
    manifest = load_manifest(PAPER_MANIFEST)
    arm = next(item for item in manifest.arms if item.method == "droste")
    assert arm.model is not None
    assert arm.model.subcall_model is not None
    root_price = ModelPrice(arm.model.root_model, "test-provider", 100, 600)
    subcall_price = ModelPrice(arm.model.subcall_model, "test-provider", 200, 1200)
    pricing = PricingSnapshot(
        "test",
        5,
        {
            root_price.model_id: root_price,
            subcall_price.model_id: subcall_price,
        },
        {},
    )
    usage = Usage(10, 2, 20, 3)

    assert _cost_microusd(usage, arm, pricing) == 103


def test_live_cost_budget_stops_on_actual_cap_or_projected_next_arm() -> None:
    assert _budget_stop_reason(100, None, 50) is None
    assert "budget reached" in str(_budget_stop_reason(100, 100, None))
    assert "estimated next-arm cost" in str(_budget_stop_reason(80, 100, 21))
    assert _budget_stop_reason(80, 100, 20) is None


def test_oolong_semantic_guidance_bounds_auditable_per_record_classification() -> None:
    guidance = _OOLONG_SEMANTIC_GUIDANCE

    assert (
        "if 'oolong_result' not in globals():\n"
        "    oolong_chunk_size = min(100, max(1, len(records)))\n"
        "    oolong_chunks = [\n"
        "        records[start:start + oolong_chunk_size]\n"
        "        for start in range(0, len(records), oolong_chunk_size)\n"
        "    ]" in guidance
    )
    assert (
        "                    'required': ['i', 'label'],\n"
        "                    'properties': {\n"
        "                        'i': {'type': 'integer'},\n"
        "                        'label': {\n"
        "                            'type': 'string',\n"
        "                            'enum': ['A', 'D', 'E', 'H', 'L', 'N'],\n"
        "                        },\n"
        "                    },\n"
        "                    'additionalProperties': False," in guidance
    )
    assert (
        "                'minItems': 1,\n                'maxItems': oolong_chunk_size," in guidance
    )
    assert (
        "        'A = an abbreviation or its expansion; '\n"
        "        'D = a description or abstract concept such as a definition, manner, or "
        "reason; '\n"
        "        'E = a thing such as an animal, product, event, language, disease, or "
        "term; '\n"
        "        'H = a person, group, or human title; '\n"
        "        'L = a place; '\n"
        "        'N = a count, date, measure, code, order, or other number. '" in guidance
    )
    assert (
        "            + 'Return exactly one object per record, using the record number as i. "
        "'\n"
        "            + f'The i values must cover 1 through {len(oolong_chunk)} exactly "
        "once.\\n\\n'" in guidance
    )
    assert (
        "    def validate_labels(value, index):\n"
        "        expected = set(range(1, len(oolong_chunks[index]) + 1))\n"
        "        indices = [item['i'] for item in value['labels']]" in guidance
    )
    assert (
        "        missing = sorted(expected - set(indices))\n"
        "        duplicates = sorted(duplicates)\n"
        "        extra = sorted(set(indices) - expected)\n"
        "        if missing or duplicates or extra:" in guidance
    )
    assert (
        "repair_rounds_by_attempt = (2, 0)\n"
        "attempt = 0\n"
        "while (\n"
        "    (oolong_result is None or oolong_result['errors'])\n"
        "    and attempt < len(repair_rounds_by_attempt)\n"
        "):" in guidance
    )
    assert (
        "    oolong_result = llm_batch_json(oolong_prompts, oolong_schema, "
        "max_repair_attempts=repair_rounds_by_attempt[attempt], "
        "validator=validate_labels)\n"
        "    attempt += 1\n"
        "result = oolong_result\n"
        "if result['errors']:\n"
        "    raise RuntimeError('classification failed')" in guidance
    )
    assert (
        "Never retry a subset, reconstruct any of those objects, or call a subcall helper "
        "again after this loop." in guidance
    )
    assert (
        "chunk_label_strings = []\n"
        "for chunk_index, value in enumerate(result['values']):\n"
        "    labels_by_index = {item['i']: item['label'] for item in value['labels']}\n"
        "    chunk_label_strings.append(\n"
        "        ''.join(\n"
        "            labels_by_index[item_index]\n"
        "            for item_index in range(1, len(oolong_chunks[chunk_index]) + 1)\n"
        "        )\n"
        "    )\n"
        "flat_labels = ''.join(chunk_label_strings)" in guidance
    )
    assert (
        "if len(flat_labels) != len(records):\n"
        "    raise RuntimeError('classification length mismatch')" in guidance
    )
    assert (
        "code_to_label = {'A': 'abbreviation', 'D': 'description and abstract concept', "
        "'E': 'entity', 'H': 'human being', 'L': 'location', 'N': 'numeric value'}"
    ) in guidance
    assert "code_counts = {code: flat_labels.count(code) for code in code_to_label}" in guidance
    assert (
        "label_counts = {label: code_counts[code] for code, label in code_to_label.items()}"
    ) in guidance


def test_oolong_pairs_guidance_uses_bounded_exact_replay_then_local_pairs() -> None:
    guidance = _OOLONG_PAIRS_GUIDANCE
    prose = " ".join(guidance.split())

    assert "exactly 787 records for 231 users" in prose
    assert "if 'oolong_pairs_results' not in globals():" in guidance
    assert "oolong_pairs_records[start:start + 12]" in guidance
    assert "oolong_pairs_chunk_prompts[start:start + 20]" in guidance
    assert "oolong_pairs_make_validator" in guidance
    assert "mapping every displayed" in guidance
    assert "record number to one A/D/E/H/L/N code" in prose
    assert "set(labels) != expected_keys" in guidance
    assert "expected one numbered allowed label per record" in guidance
    assert "oolong_pairs_attempts[batch_index] < 2" in guidance
    assert "max_repair_attempts=0" in guidance
    assert "same prompts list" in prose
    assert "Never retry only the failed prompt indices" in prose
    assert "66 x 2 = 132 subcalls" in prose
    assert "leaving 18 of the arm's 150-call limit" in prose
    assert "all date constraints" in prose
    assert "vacuously true" in prose
    assert "asymmetric predicates accept either role assignment" in prose
    assert "answer['content'] = '\\n'.join" in guidance
    assert "do not print it" in guidance
    assert "set-builder notation" in guidance
    for index, code in enumerate(
        re.findall(r"```python\n(.*?)```", guidance, flags=re.DOTALL), start=1
    ):
        compile(code, f"<oolong-pairs-guidance-{index}>", "exec")


def test_direct_late_failure_preserves_accounted_usage_and_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.live as live

    manifest = load_manifest(PAPER_MANIFEST)
    arm = next(item for item in manifest.arms if item.method == "direct-model")

    class LateFailureClient:
        def __init__(self, **kwargs: object) -> None:
            self.total_usage = TokenUsage(120, 30, 150)

        def responses_create(self, *args: object, **kwargs: object) -> str:
            raise RuntimeError("response output was malformed")

    monkeypatch.setattr(live, "ModelRelayClient", LateFailureClient)

    with pytest.raises(_LiveRunFailure) as caught:
        _direct_run(
            {"question": "question"},
            "context",
            arm,
            "opaque-test-key",
            "https://example.invalid",
        )

    failure = caught.value
    assert failure.usage == Usage(root_input_tokens=120, root_output_tokens=30)
    assert failure.iterations == 1
    assert arm.model is not None
    price = ModelPrice(arm.model.root_model, "stub", 100, 600)
    pricing = PricingSnapshot("test", 0, {price.model_id: price}, {})
    assert _cost_microusd(failure.usage, arm, pricing) > 0


def test_paid_failure_preserves_typed_timeout_status() -> None:
    failure = _LiveRunFailure(
        "benchmark task exceeded 10s",
        usage=Usage(root_input_tokens=10),
        status="timeout",
    )

    assert _status_for_exception(failure) == "timeout"
    assert failure.usage.root_input_tokens == 10


def test_http_504_is_a_typed_error_retained_in_the_score_denominator() -> None:
    assert _status_for_exception(RuntimeError("root llm failed with HTTP 504")) == "error"

    common = {
        "suite_id": "suite",
        "suite_version": "1",
        "manifest_sha256": "a" * 64,
        "benchmark_id": "benchmark",
        "arm_id": "arm",
        "metric": "exact_match",
        "reference": "answer",
    }
    artifacts = (
        RunArtifact(
            **common,
            task_id="failed",
            status="error",
            score=None,
            prediction=None,
            error="root llm failed with HTTP 504",
        ),
        RunArtifact(
            **common,
            task_id="passed",
            status="ok",
            score=1.0,
            prediction="answer",
        ),
    )

    row = aggregate(artifacts)[0]
    assert row.attempted == 2
    assert row.successful == 1
    assert row.mean_score == 0.5


def test_context_path_is_relative_to_selected_benchmark(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmarks"
    manifest_path = benchmark_root / "manifests" / "suite.json"
    selected_dir = benchmark_root / "selected"
    context = selected_dir / "contexts" / "shared.txt"
    context.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}")
    (selected_dir / "tasks.json").write_text("[]")
    context.write_text("selected context")

    manifest = load_manifest(PAPER_MANIFEST)
    selected = next(item for item in manifest.benchmarks if item.benchmark_id == "oolong")
    selected = replace(selected, tasks_path="../selected/tasks.json")
    manifest = replace(manifest, source_path=manifest_path)
    task = {
        "id": "task",
        "context_path": "contexts/shared.txt",
        "context_sha256": hashlib.sha256(context.read_bytes()).hexdigest(),
    }

    assert _context_path(manifest, selected, task) == context


def test_materialize_oolong_validates_and_deduplicates_contexts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import benchmarks.oolong as module

    first = "first public context"
    second = "second public context"
    task_ids = tuple(str(9000 + index) for index in range(26))
    monkeypatch.setattr(module, "ROW_OFFSET", 100)
    monkeypatch.setattr(module, "ROW_COUNT", 26)
    monkeypatch.setattr(module, "CONTEXT_LENGTH", 128)
    monkeypatch.setattr(module, "_EXPECTED_TASK_IDS", task_ids)
    monkeypatch.setattr(
        module,
        "_EXPECTED_CONTEXT_HASHES",
        (
            hashlib.sha256(first.encode()).hexdigest(),
            hashlib.sha256(second.encode()).hexdigest(),
        ),
    )
    rows = []
    for position, task_id in enumerate(task_ids):
        rows.append(
            {
                "row_idx": 100 + position,
                "row": {
                    "id": int(task_id),
                    "dataset": "trec_coarse",
                    "context_len": 128,
                    "question": f"question {position}",
                    "answer": "[1]",
                    "answer_type": "ANSWER_TYPE.NUMERIC",
                    "context_window_text": first if position < 25 else second,
                    "context_window_id": position // 25,
                    "task": "count",
                    "task_group": "counting",
                    "input_subset": False,
                },
            }
        )

    result = materialize_oolong(
        tmp_path / "oolong", fetch=lambda: json.dumps({"rows": rows}).encode()
    )

    tasks = json.loads(result.tasks_path.read_text())
    assert result.task_count == 26
    assert result.context_count == 2
    assert result.tasks_sha256 == hashlib.sha256(result.tasks_path.read_bytes()).hexdigest()
    assert len(list((tmp_path / "oolong" / "contexts").glob("*.txt"))) == 2
    assert tasks[0]["reference"] == {
        "answer": "[1]",
        "answer_type": "ANSWER_TYPE.NUMERIC",
    }
    with pytest.raises(BenchmarkRunError, match="refusing to overwrite"):
        materialize_oolong(tmp_path / "oolong", fetch=lambda: json.dumps({"rows": rows}).encode())


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


def test_report_cli_aggregates_an_exact_paired_task_subset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = load_manifest(SMOKE_MANIFEST)
    output = tmp_path / "artifacts"
    run_fixture_suite(manifest, output)
    for path in output.glob("*.json"):
        if not path.name.endswith("--normalization.json"):
            path.unlink()

    assert (
        benchmark_main(
            [
                "report",
                str(SMOKE_MANIFEST),
                str(output),
                "--task-id",
                "normalization",
            ]
        )
        == 0
    )

    report = capsys.readouterr().out
    assert "| smoke-exact | fixture-droste | exact_match | 1.0000 | 1/1 |" in report
    assert "| smoke-exact | fixture-direct | exact_match | 1.0000 | 1/1 |" in report
    assert "smoke-numeric" not in report
    assert "smoke-f1" not in report
    with pytest.raises(ReportError, match="artifact set is incomplete"):
        load_artifacts(output, manifest)


def test_report_selected_subset_still_requires_every_selected_arm(tmp_path: Path) -> None:
    manifest = load_manifest(SMOKE_MANIFEST)
    output = tmp_path / "artifacts"
    run_fixture_suite(manifest, output)
    for path in output.glob("*.json"):
        if not path.name.endswith("--normalization.json"):
            path.unlink()
    (output / "smoke-exact--fixture-direct--normalization.json").unlink()

    with pytest.raises(
        ReportError,
        match="artifact set is incomplete; missing: smoke-exact--fixture-direct--normalization",
    ):
        load_artifacts(output, manifest, task_ids=["normalization"])


@pytest.mark.parametrize(
    ("task_ids", "message"),
    [(["unknown"], "unknown task ids: unknown"), (["integer", "integer"], "duplicate task ids")],
)
def test_report_task_selection_rejects_unknown_and_duplicate_ids(
    tmp_path: Path, task_ids: list[str], message: str
) -> None:
    manifest = load_manifest(SMOKE_MANIFEST)
    output = tmp_path / "artifacts"
    run_fixture_suite(manifest, output)

    with pytest.raises(ReportError, match=message):
        load_artifacts(output, manifest, task_ids=task_ids)


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
