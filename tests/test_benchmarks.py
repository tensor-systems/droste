from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from benchmarks.cli import main as benchmark_main
from benchmarks.live import (
    _OOLONG_PAIRS_GUIDANCE,
    _OOLONG_SEMANTIC_GUIDANCE,
    _SNIAH_GUIDANCE,
    ModelPrice,
    PricingSnapshot,
    _budget_stop_reason,
    _context_path,
    _cost_microusd,
    _direct_run,
    _LiveRunFailure,
    _policy_for_task,
    _status_for_exception,
)
from benchmarks.longbench_codeqa import materialize_longbench_codeqa
from benchmarks.models import (
    ArtifactError,
    ManifestError,
    RunArtifact,
    RunStatus,
    SuiteManifest,
    Usage,
    canonical_json_sha256,
    load_manifest,
    to_lean_artifact,
)
from benchmarks.oolong import materialize_oolong
from benchmarks.oolong_pairs import (
    PREDICTIONS_ASSET_SHA256,
    evaluate_predicate,
    materialize_oolong_pairs_predictions,
    parse_labeled_context,
)
from benchmarks.oolong_pairs import TASKS as OOLONG_PAIR_TASKS
from benchmarks.report import ReportError, aggregate, load_artifacts, render_markdown
from benchmarks.runner import BenchmarkRunError, run_fixture_suite
from benchmarks.scoring import (
    exact_match,
    numeric_score,
    oolong_official,
    oolong_pairs_f1,
    token_f1,
)
from benchmarks.sniah import (
    NOISE_SENTENCE,
    RULER_COMMIT,
    generate_task,
    materialize_sniah,
)
from droste.protocols.llm_client import TokenUsage

ROOT = Path(__file__).resolve().parents[1]
SMOKE_MANIFEST = ROOT / "benchmarks" / "manifests" / "smoke-v1.json"
PAPER_MANIFEST = ROOT / "benchmarks" / "manifests" / "rlm-paper-v1.json"
OOLONG_PAIRS_MANIFEST = ROOT / "benchmarks" / "manifests" / "oolong-pairs-2026-07-17.json"
OOLONG_PAIRS_ARTIFACTS = (
    ROOT / "benchmarks" / "results" / "oolong-pairs-32k-2026-07-17" / "artifacts"
)
OOLONG_PAIRS_FULL_VALUES_FIXTURE = (
    ROOT / "tests" / "fixtures" / "oolong-pairs-droste-terra-luna-pairs--19-values.json"
)


def test_manifest_pins_paper_tasks_and_published_live_arms() -> None:
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
        "direct-sol-sniah",
        "direct-terra-sniah",
        "droste-terra-luna-sniah",
        "direct-sol-pairs",
        "direct-terra-pairs",
        "droste-terra-luna-pairs",
    }
    assert all(arm.executor == "modelrelay" for arm in manifest.arms)
    sniah = next(item for item in manifest.benchmarks if item.benchmark_id == "s-niah")
    assert sniah.status == "ready"
    assert sniah.dataset_version == (
        "droste-native-generator-v1/ruler-38da79d79519ef87aa46ae804f838e1eab7f86d7"
    )
    assert sniah.tasks_sha256 == (
        "62b1e267fedc723349b4233b5d8929b6ffa1a822155056ec96e20dbc90ae2990"
    )
    oolong = next(item for item in manifest.benchmarks if item.benchmark_id == "oolong")
    assert oolong.status == "ready"
    assert oolong.tasks_sha256 == "28abaefcbcba1d843a384115a1217ea0017201b13074c95de6feb313c40c8da4"
    codeqa = next(
        item for item in manifest.benchmarks if item.benchmark_id == "longbench-v2-codeqa"
    )
    assert codeqa.dataset == "zai-org/LongBench-v2"
    assert codeqa.dataset_version == "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
    assert codeqa.split == (
        "train/domain=Code Repository Understanding/20-of-50-cost-bounded-subsample"
    )
    assert codeqa.status == "ready"
    assert codeqa.scorer == "exact_match"
    assert codeqa.tasks_sha256 == (
        "d796fbcf741fbfc516903afd929e1e5aa6e64ded85445acbf950e638303ab5f5"
    )
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
    assert exact_match("  QUIET-ANCHOR\n", "quiet-anchor") == 1.0
    assert exact_match("The answer is quiet-anchor", "quiet-anchor") == 0.0
    assert numeric_score("3.145", 3.14, tolerance=0.01) == 1.0
    assert numeric_score(float("nan"), float("nan")) == 0.0
    assert token_f1("alpha alpha gamma", "alpha beta") == pytest.approx(0.4)
    assert token_f1("", "") == 1.0


@pytest.mark.parametrize(
    "prediction",
    ["A", " a ", "(A)", "A.", "A)", "Answer: A", "answer: (a)"],
)
def test_exact_match_accepts_bounded_multiple_choice_variations(prediction: str) -> None:
    assert exact_match(prediction, "A") == 1.0


@pytest.mark.parametrize(
    "prediction",
    ["B", "The answer is A", "A because the code says so", "choice A", "alpha"],
)
def test_exact_match_rejects_wrong_or_unbounded_multiple_choice_text(prediction: str) -> None:
    assert exact_match(prediction, "A") == 0.0


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


def test_sniah_guidance_requires_exact_lexical_retrieval() -> None:
    assert "exact adjective-noun key" in _SNIAH_GUIDANCE
    assert "Parse the word-pair after 'is:'" in _SNIAH_GUIDANCE
    assert "return exactly that bare word-pair" in _SNIAH_GUIDANCE
    assert "no trailing punctuation or period" in _SNIAH_GUIDANCE
    assert "other extra characters" in _SNIAH_GUIDANCE
    assert "no semantic classification or model subcall is needed" in _SNIAH_GUIDANCE
    assert _policy_for_task("s-niah", {}) == (False, _SNIAH_GUIDANCE)


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


def test_generate_sniah_mirrors_noise_prompt_and_controlled_position() -> None:
    task = generate_task(
        context_length=1024,
        seed=7,
        task_id="sniah-test",
        depth=0.5,
    )

    assert task["ruler_commit"] == RULER_COMMIT
    assert task["haystack_type"] == "noise"
    assert task["needle_type"] == "words"
    assert task["needle"] == (
        f"One of the special magic words for {task['needle_key']} is: {task['needle_value']}."
    )
    assert task["context"].count(NOISE_SENTENCE) == task["haystack_repetitions"]
    assert task["context"].count(task["needle"]) == 1
    assert task["needle_index"] == task["haystack_repetitions"] // 2
    assert task["position"] == 0.5
    assert task["query"] in task["question"]
    assert task["answer_prefix"] in task["question"]
    assert task["reference"] == task["needle_value"]
    assert task["total_tokens"] <= task["context_length"]


def test_materialize_sniah_is_byte_identical_on_regeneration(tmp_path: Path) -> None:
    first = materialize_sniah(tmp_path / "first", context_length=2048, task_count=5, seed=42)
    second = materialize_sniah(tmp_path / "second", context_length=2048, task_count=5, seed=42)

    assert first.tasks_sha256 == second.tasks_sha256
    assert first.tasks_path.read_bytes() == second.tasks_path.read_bytes()
    assert first.task_count == second.task_count == 5
    assert first.context_count == second.context_count == 5
    first_contexts = {
        path.name: path.read_bytes() for path in (tmp_path / "first" / "contexts").iterdir()
    }
    second_contexts = {
        path.name: path.read_bytes() for path in (tmp_path / "second" / "contexts").iterdir()
    }
    assert first_contexts == second_contexts
    with pytest.raises(BenchmarkRunError, match="refusing to overwrite"):
        materialize_sniah(tmp_path / "first", context_length=2048, task_count=5, seed=42)


def test_default_sniah_generation_matches_manifest_hash(tmp_path: Path) -> None:
    result = materialize_sniah(tmp_path / "configured")
    manifest = load_manifest(PAPER_MANIFEST)
    benchmark = next(item for item in manifest.benchmarks if item.benchmark_id == "s-niah")

    assert result.task_count == result.context_count == 50
    assert result.tasks_sha256 == benchmark.tasks_sha256


def test_materialize_sniah_cli_reports_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "sniah"

    assert (
        benchmark_main(
            [
                "materialize-sniah",
                "--output",
                str(output),
                "--context-length",
                "1024",
                "--task-count",
                "2",
                "--seed",
                "9",
            ]
        )
        == 0
    )
    assert "materialized 2 tasks and 2 contexts; tasks SHA-256:" in capsys.readouterr().out


def test_materialize_longbench_codeqa_verifies_and_projects_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import benchmarks.longbench_codeqa as module

    rows = []
    for position in range(2):
        rows.append(
            {
                "row_idx": 7 + position * 3,
                "row": {
                    "_id": f"task-{position}",
                    "domain": "Code Repository Understanding",
                    "sub_domain": "Code repo QA",
                    "difficulty": "easy" if position == 0 else "hard",
                    "length": "short" if position == 0 else "long",
                    "question": f"Which behavior is correct for function {position}?",
                    "choice_A": "first behavior",
                    "choice_B": "second behavior",
                    "choice_C": "third behavior",
                    "choice_D": "fourth behavior",
                    "answer": "B",
                    "context": f"repository context {position}",
                },
            }
        )
    payload = {
        "num_rows_total": 2,
        "partial": False,
        "rows": rows,
    }
    monkeypatch.setattr(module, "ROW_COUNT", 2)
    monkeypatch.setattr(
        module,
        "SUBSAMPLE_COUNTS",
        {("short", "easy"): 1, ("long", "hard"): 1},
    )
    monkeypatch.setattr(
        module,
        "_EXPECTED_FILTERED_ROWS_SHA256",
        hashlib.sha256(module._encode_json(rows)).hexdigest(),
    )

    result = materialize_longbench_codeqa(
        tmp_path / "longbench-codeqa",
        fetch=lambda: json.dumps(payload).encode(),
    )

    tasks = json.loads(result.tasks_path.read_text())
    assert result.task_count == 2
    assert result.context_count == 2
    assert result.tasks_sha256 == hashlib.sha256(result.tasks_path.read_bytes()).hexdigest()
    assert tasks[0]["reference"] == "B"
    assert tasks[0]["choices"] == {
        "A": "first behavior",
        "B": "second behavior",
        "C": "third behavior",
        "D": "fourth behavior",
    }
    assert tasks[0]["question"].endswith("Answer with exactly one letter: A, B, C, or D.")
    context_path = result.tasks_path.parent / tasks[0]["context_path"]
    assert context_path.read_text() == "repository context 0"
    assert hashlib.sha256(context_path.read_bytes()).hexdigest() == tasks[0]["context_sha256"]
    with pytest.raises(BenchmarkRunError, match="refusing to overwrite"):
        materialize_longbench_codeqa(
            tmp_path / "longbench-codeqa",
            fetch=lambda: json.dumps(payload).encode(),
        )


def test_longbench_codeqa_subsample_uses_centered_even_spacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.longbench_codeqa as module

    validated = [
        (
            position,
            {
                "_id": f"task-{position}",
                "difficulty": "easy",
                "length": "short",
            },
        )
        for position in range(6)
    ]
    monkeypatch.setattr(module, "SUBSAMPLE_COUNTS", {("short", "easy"): 4})

    selected = module._subsample_rows(validated)

    assert [row["_id"] for _, row in selected] == ["task-0", "task-2", "task-3", "task-5"]


def test_materialize_longbench_codeqa_rejects_unverified_rows_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import benchmarks.longbench_codeqa as module

    row = {
        "row_idx": 7,
        "row": {
            "_id": "task",
            "domain": "Code Repository Understanding",
            "sub_domain": "Code repo QA",
            "difficulty": "easy",
            "length": "short",
            "question": "Question?",
            "choice_A": "A",
            "choice_B": "B",
            "choice_C": "C",
            "choice_D": "D",
            "answer": "A",
            "context": "context",
        },
    }
    payload = {"num_rows_total": 1, "partial": False, "rows": [row]}
    monkeypatch.setattr(module, "ROW_COUNT", 1)
    output = tmp_path / "longbench-codeqa"

    with pytest.raises(BenchmarkRunError, match="filtered LongBench-v2 rows have SHA-256"):
        materialize_longbench_codeqa(output, fetch=lambda: json.dumps(payload).encode())

    assert not output.exists()


def test_materialize_oolong_pairs_predictions_verifies_and_extracts_release_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import benchmarks.oolong_pairs as module

    buffer = io.BytesIO()
    artifacts = (
        ("direct--1", "direct", "1", ""),
        ("droste--2", "droste", "2", [[1, 2]]),
    )
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        metadata = b"not JSON"
        metadata_member = tarfile.TarInfo("oolong-pairs-32k-2026-07-17/artifacts/._ignored.json")
        metadata_member.size = len(metadata)
        archive.addfile(metadata_member, io.BytesIO(metadata))
        for filename, arm_id, task_id, prediction in artifacts:
            encoded = json.dumps(
                {"arm_id": arm_id, "task_id": task_id, "prediction": prediction}
            ).encode()
            member = tarfile.TarInfo(f"oolong-pairs-32k-2026-07-17/artifacts/{filename}.json")
            member.size = len(encoded)
            archive.addfile(member, io.BytesIO(encoded))
    asset = buffer.getvalue()
    monkeypatch.setattr(module, "PREDICTIONS_ASSET_SHA256", hashlib.sha256(asset).hexdigest())
    monkeypatch.setattr(module, "_EXPECTED_PREDICTION_COUNT", 2)

    output = tmp_path / "predictions"
    result = materialize_oolong_pairs_predictions(output, fetch=lambda: asset)

    assert result.prediction_count == 2
    assert json.loads(result.predictions_path.read_text()) == {
        "direct--1": "",
        "droste--2": [[1, 2]],
    }
    with pytest.raises(BenchmarkRunError, match="refusing to overwrite materialized directory"):
        materialize_oolong_pairs_predictions(output, fetch=lambda: asset)


def test_materialize_oolong_pairs_predictions_rejects_wrong_asset_hash(
    tmp_path: Path,
) -> None:
    output = tmp_path / "predictions"

    with pytest.raises(BenchmarkRunError, match="predictions asset has SHA-256"):
        materialize_oolong_pairs_predictions(output, fetch=lambda: b"not the pinned asset")
    assert not output.exists()


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


def test_report_missing_other_ready_tasks_names_materializer(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmarks"
    manifest_path = benchmark_root / "manifests" / "suite.json"
    sniah_tasks = benchmark_root / ".data" / "sniah-noise-words-32768-50-v1" / "tasks.json"
    sniah_tasks.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}")
    sniah_tasks.write_text('[{"id": "normalization", "reference": "Beta"}]')

    manifest = load_manifest(SMOKE_MANIFEST)
    sniah = replace(
        manifest.benchmarks[0],
        benchmark_id="s-niah",
        tasks_path="../.data/sniah-noise-words-32768-50-v1/tasks.json",
    )
    oolong = replace(
        manifest.benchmarks[1],
        benchmark_id="oolong",
        tasks_path="../.data/oolong-trec-coarse-131k-v1/tasks.json",
    )
    manifest = replace(manifest, benchmarks=(sniah, oolong), source_path=manifest_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "selected-sniah-artifact.json").write_text("{}")

    with pytest.raises(ReportError) as caught:
        load_artifacts(artifacts, manifest, task_ids=["normalization"])

    assert str(caught.value) == (
        "cannot load declared tasks for benchmark 'oolong': run "
        "`python -m benchmarks materialize-oolong "
        "--output benchmarks/.data/oolong-trec-coarse-131k-v1` first"
    )
    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_report_missing_longbench_codeqa_tasks_names_materializer(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmarks"
    manifest_path = benchmark_root / "manifests" / "suite.json"
    oolong_tasks = benchmark_root / ".data" / "oolong-trec-coarse-131k-v1" / "tasks.json"
    oolong_tasks.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}")
    oolong_tasks.write_text('[{"id": "normalization", "reference": "Beta"}]')

    manifest = load_manifest(SMOKE_MANIFEST)
    oolong = replace(
        manifest.benchmarks[0],
        benchmark_id="oolong",
        tasks_path="../.data/oolong-trec-coarse-131k-v1/tasks.json",
    )
    longbench_codeqa = replace(
        manifest.benchmarks[1],
        benchmark_id="longbench-v2-codeqa",
        tasks_path="../.data/longbench-v2-codeqa-20-v1/tasks.json",
    )
    manifest = replace(
        manifest,
        benchmarks=(oolong, longbench_codeqa),
        source_path=manifest_path,
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "selected-longbench-codeqa-artifact.json").write_text("{}")

    with pytest.raises(ReportError) as caught:
        load_artifacts(artifacts, manifest, task_ids=["normalization"])

    assert str(caught.value) == (
        "cannot load declared tasks for benchmark 'longbench-v2-codeqa': run "
        "`python -m benchmarks materialize-longbench-codeqa "
        "--output benchmarks/.data/longbench-v2-codeqa-20-v1` first"
    )
    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_report_missing_oolong_pairs_tasks_names_materializer(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmarks"
    manifest_path = benchmark_root / "manifests" / "suite.json"
    oolong_tasks = benchmark_root / ".data" / "oolong-trec-coarse-131k-v1" / "tasks.json"
    oolong_tasks.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}")
    oolong_tasks.write_text('[{"id": "normalization", "reference": "Beta"}]')

    manifest = load_manifest(SMOKE_MANIFEST)
    oolong = replace(
        manifest.benchmarks[0],
        benchmark_id="oolong",
        tasks_path="../.data/oolong-trec-coarse-131k-v1/tasks.json",
    )
    oolong_pairs = replace(
        manifest.benchmarks[1],
        benchmark_id="oolong-pairs",
        tasks_path="../.data/oolong-pairs-32k-v1/tasks.json",
    )
    manifest = replace(
        manifest,
        benchmarks=(oolong, oolong_pairs),
        source_path=manifest_path,
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "selected-oolong-pairs-artifact.json").write_text("{}")

    with pytest.raises(ReportError) as caught:
        load_artifacts(artifacts, manifest, task_ids=["normalization"])

    assert str(caught.value) == (
        "cannot load declared tasks for benchmark 'oolong-pairs': run "
        "`python -m benchmarks materialize-oolong-pairs "
        "--output benchmarks/.data/oolong-pairs-32k-v1` first"
    )
    assert isinstance(caught.value.__cause__, FileNotFoundError)


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


def _make_one_smoke_artifact_lean(output: Path) -> tuple[str, object]:
    path = next(output.glob("smoke-exact--fixture-direct--mismatch.json"))
    full = RunArtifact.from_dict(json.loads(path.read_text()))
    path.write_text(json.dumps(to_lean_artifact(full).to_dict()))
    return "fixture-direct--mismatch", full.prediction


def _write_single_oolong_pairs_lean_artifact(
    output: Path, *, status: RunStatus = "ok"
) -> tuple[SuiteManifest, Path]:
    manifest = load_manifest(SMOKE_MANIFEST)
    manifest = replace(
        manifest,
        benchmarks=(replace(manifest.benchmarks[0], benchmark_id="oolong-pairs"),),
        arms=(manifest.arms[0],),
    )
    prediction = "  ALPHA  " if status == "ok" else None
    full = RunArtifact(
        suite_id=manifest.suite_id,
        suite_version=manifest.suite_version,
        manifest_sha256=manifest.sha256,
        benchmark_id="oolong-pairs",
        task_id="normalization",
        arm_id="fixture-droste",
        status=status,
        metric="exact_match",
        score=1.0 if status == "ok" else None,
        prediction=prediction,
        reference="Alpha",
        error=None if status == "ok" else "provider failed",
    )
    output.mkdir()
    artifact_path = output / f"{full.artifact_id}.json"
    artifact_path.write_text(json.dumps(to_lean_artifact(full).to_dict()))
    return manifest, artifact_path


def test_report_verifies_oolong_pairs_lean_artifact_with_scoped_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import benchmarks.report as report_module

    output = tmp_path / "artifacts"
    manifest, _ = _write_single_oolong_pairs_lean_artifact(output)
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(json.dumps({"fixture-droste--normalization": "  ALPHA  "}))
    monkeypatch.setattr(report_module, "PREDICTIONS_PATH", predictions_path)

    assert len(load_artifacts(output, manifest, task_ids=["normalization"])) == 1


def test_report_rejects_cross_benchmark_lean_prediction_lookup(tmp_path: Path) -> None:
    manifest = load_manifest(SMOKE_MANIFEST)
    output = tmp_path / "artifacts"
    run_fixture_suite(manifest, output)
    _make_one_smoke_artifact_lean(output)

    with pytest.raises(
        ReportError,
        match="lean artifacts for benchmark 'smoke-exact' have no registered predictions",
    ):
        load_artifacts(output, manifest)


def test_report_rejects_tampered_failed_lean_prediction_hash(tmp_path: Path) -> None:
    output = tmp_path / "artifacts"
    manifest, artifact_path = _write_single_oolong_pairs_lean_artifact(output, status="error")
    artifact = json.loads(artifact_path.read_text())
    artifact["prediction_sha256"] = "0" * 64
    artifact_path.write_text(json.dumps(artifact))

    with pytest.raises(ReportError, match="prediction does not match canonical JSON null"):
        load_artifacts(output, manifest, task_ids=["normalization"])


def test_report_fails_closed_when_lean_predictions_are_not_materialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import benchmarks.report as report_module

    output = tmp_path / "artifacts"
    manifest, _ = _write_single_oolong_pairs_lean_artifact(output)
    monkeypatch.setattr(report_module, "PREDICTIONS_PATH", tmp_path / "missing.json")

    with pytest.raises(
        ReportError,
        match="materialize-oolong-pairs-predictions --output",
    ):
        load_artifacts(output, manifest, task_ids=["normalization"])


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


def test_artifact_accepts_falsy_inline_values_and_hash_only_values() -> None:
    common = {
        "suite_id": "suite",
        "suite_version": "1",
        "manifest_sha256": "a" * 64,
        "benchmark_id": "benchmark",
        "task_id": "task",
        "arm_id": "arm",
        "status": "ok",
        "metric": "exact_match",
        "score": 1.0,
    }

    inline = RunArtifact(**common, prediction="", reference=None)
    lean = RunArtifact(
        **common,
        prediction_sha256=canonical_json_sha256(""),
        reference_sha256=canonical_json_sha256(None),
    )

    assert inline.has_inline_prediction
    assert inline.has_inline_reference
    assert not lean.has_inline_prediction
    assert not lean.has_inline_reference
    assert "prediction" not in lean.to_dict()
    assert "reference" not in lean.to_dict()
    assert RunArtifact.from_dict(lean.to_dict()) == lean


def test_real_oolong_pairs_lean_artifact_is_reproducible_from_full_values() -> None:
    values = json.loads(OOLONG_PAIRS_FULL_VALUES_FIXTURE.read_text())
    manifest = load_manifest(OOLONG_PAIRS_MANIFEST)
    benchmark = next(item for item in manifest.benchmarks if item.benchmark_id == "oolong-pairs")
    assert values["prediction_asset_sha256"] == PREDICTIONS_ASSET_SHA256
    assert values["tasks_sha256"] == benchmark.tasks_sha256

    artifact_path = OOLONG_PAIRS_ARTIFACTS / (
        f"oolong-pairs--{values['arm_id']}--{values['task_id']}.json"
    )
    committed = RunArtifact.from_dict(json.loads(artifact_path.read_text()))
    full = replace(
        committed,
        prediction=values["prediction"],
        reference=values["reference"],
        prediction_sha256=None,
        reference_sha256=None,
    )

    assert to_lean_artifact(full) == committed


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "prediction": "value",
                "prediction_sha256": "a" * 64,
                "reference": "answer",
            },
            "never both",
        ),
        (
            {"prediction_sha256": "a" * 64},
            "reference must be inline or represented",
        ),
    ],
)
def test_artifact_rejects_invalid_inline_hash_exclusivity(
    overrides: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "suite_id": "suite",
        "suite_version": "1",
        "manifest_sha256": "a" * 64,
        "benchmark_id": "benchmark",
        "task_id": "task",
        "arm_id": "arm",
        "status": "ok",
        "metric": "exact_match",
        "score": 1.0,
    }
    values.update(overrides)

    with pytest.raises(ArtifactError, match=message):
        RunArtifact(**values)  # type: ignore[arg-type]


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
