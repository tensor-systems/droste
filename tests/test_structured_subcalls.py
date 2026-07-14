from __future__ import annotations

import json
from typing import Any

import pytest

from droste import SubcallBudgetExceeded, aggregate_json_counts, structured_batch, validate_json
from droste.execution.context import ExecutionContext, create_execution_context
from droste.prompts.base import BASE_SYSTEM_PROMPT
from droste.structured import _StructuredBatchEvidence

SCHEMA = {
    "type": "object",
    "required": ["count"],
    "properties": {"count": {"type": "integer", "minimum": 0}},
    "additionalProperties": False,
}


def test_base_prompt_documents_structured_validator_call_contract() -> None:
    normalized_prompt = " ".join(BASE_SYSTEM_PROMPT.split())

    assert "validator(value, index)" in normalized_prompt
    assert "index is the original prompt index" in normalized_prompt
    assert "raise ValueError to reject that value and request repair" in normalized_prompt


class ScriptedSubcalls:
    def __init__(
        self,
        batches: list[tuple[list[str], list[dict[str, object]]]],
        *,
        max_calls: int = 50,
    ) -> None:
        self.context: ExecutionContext = create_execution_context(max_calls=max_calls)
        self.batches = list(batches)
        self.prompts: list[list[str]] = []

    def llm_query(self, prompt: str, context: str = "") -> str:
        raise NotImplementedError

    def llm_batch(self, prompts: list[str], contexts: list[str] | None = None) -> list[str]:
        values, errors = self.llm_batch_with_errors(prompts, contexts)
        if errors:
            raise RuntimeError(str(errors[0]["error"]))
        return values

    def llm_batch_with_errors(
        self, prompts: list[str], contexts: list[str] | None = None
    ) -> tuple[list[str], list[dict[str, object]]]:
        count = len(prompts)
        stats = self.context.stats
        if stats.calls_made + count > self.context.max_calls:
            raise SubcallBudgetExceeded("max subcalls exceeded")
        stats.calls_made += count
        self.prompts.append(list(prompts))
        values, errors = self.batches.pop(0)
        failed = {int(item["index"]) for item in errors}
        stats.successful_calls += count - len(failed)
        stats.total_tokens += (count - len(failed)) * 10
        return values, errors


def test_structured_batch_valid_output_preserves_order_and_usage() -> None:
    client = ScriptedSubcalls([(['{"count":2}', '{"count":1}'], [])])

    result = structured_batch(client, ["a", "b"], SCHEMA)

    assert result == {
        "values": [{"count": 2}, {"count": 1}],
        "errors": [],
        "attempts": [1, 1],
        "repairs_made": 0,
    }
    assert client.context.stats.calls_made == 2
    assert client.context.stats.successful_calls == 2
    assert client.context.stats.total_tokens == 20


def test_structured_evidence_requires_an_exact_validated_retry() -> None:
    evidence = _StructuredBatchEvidence()
    prompts = ["classify a", "classify b"]
    contexts = ["record a", "record b"]

    def validator(value: Any, index: int) -> None:
        return None

    evidence.record(
        prompts=prompts,
        schema=SCHEMA,
        contexts=contexts,
        validator=validator,
        errors=[{"index": 1, "type": "validation_error", "error": "bad"}],
    )
    evidence.record(
        prompts=["different task"],
        schema=SCHEMA,
        contexts=["different record"],
        validator=validator,
        errors=[],
    )
    evidence.record(
        prompts=prompts,
        schema=SCHEMA,
        contexts=contexts,
        validator=lambda value, index: None,
        errors=[],
    )

    assert evidence.unresolved_batches == 1
    assert evidence.unresolved_items == 1

    evidence.record(
        prompts=prompts,
        schema=SCHEMA,
        contexts=contexts,
        validator=validator,
        errors=[],
    )
    assert evidence.unresolved_batches == 0
    assert evidence.unresolved_items == 0
    assert evidence.minimum_exact_retry_calls == 0


def test_structured_evidence_retry_cost_counts_each_recorded_exact_request() -> None:
    evidence = _StructuredBatchEvidence()

    def first_validator(value: Any, index: int) -> None:
        return None

    def second_validator(value: Any, index: int) -> None:
        return None

    evidence.record(
        prompts=["a", "b", "c"],
        schema=SCHEMA,
        contexts=None,
        validator=first_validator,
        errors=[{"index": 2, "type": "provider_error", "error": "temporary"}],
    )
    evidence.record(
        prompts=["a", "b", "c"],
        schema=SCHEMA,
        contexts=None,
        validator=second_validator,
        errors=[{"index": 0, "type": "validation_error", "error": "bad"}],
    )

    # Validator identity participates in exactness. These otherwise-identical
    # calls are two recorded requests, each requiring its full three-item batch.
    assert evidence.unresolved_batches == 2
    assert evidence.unresolved_items == 2
    assert evidence.minimum_exact_retry_calls == 6


def test_structured_batch_repairs_only_malformed_items_once() -> None:
    client = ScriptedSubcalls(
        [
            (['{"count":3}', "not json", '{"count":1}'], []),
            (['{"count":2}'], []),
        ]
    )

    result = structured_batch(client, ["a", "b", "c"], SCHEMA, max_repair_attempts=1)

    assert result["values"] == [{"count": 3}, {"count": 2}, {"count": 1}]
    assert result["errors"] == []
    assert result["attempts"] == [1, 2, 1]
    assert result["repairs_made"] == 1
    assert len(client.prompts) == 2
    assert len(client.prompts[1]) == 1
    assert "Original task:\nb" in client.prompts[1][0]
    assert client.context.stats.calls_made == 4
    assert client.context.stats.successful_calls == 4


def test_structured_batch_keeps_provider_errors_typed_and_does_not_retry_them() -> None:
    client = ScriptedSubcalls(
        [
            (
                ["", "bad", '{"count":4}'],
                [{"index": 0, "type": "provider_error", "error": "provider unavailable"}],
            ),
            (['{"count":2}'], []),
        ]
    )

    result = structured_batch(client, ["provider", "malformed", "valid"], SCHEMA)

    assert result["values"] == [None, {"count": 2}, {"count": 4}]
    assert result["errors"] == [
        {
            "index": 0,
            "type": "provider_error",
            "error": "provider unavailable",
            "attempts": 1,
        }
    ]
    assert result["attempts"] == [1, 2, 1]
    assert client.context.stats.calls_made == 4
    assert client.context.stats.successful_calls == 3


def test_structured_batch_reports_permanent_malformed_and_atomic_budget_exhaustion() -> None:
    malformed = ScriptedSubcalls([(["bad"], []), (["still bad"], [])])
    result = structured_batch(malformed, ["a"], SCHEMA, max_repair_attempts=1)
    assert result["values"] == [None]
    assert result["errors"][0]["type"] == "validation_error"
    assert result["errors"][0]["attempts"] == 2

    exhausted = ScriptedSubcalls([(["bad", "bad"], [])], max_calls=2)
    result = structured_batch(exhausted, ["a", "b"], SCHEMA, max_repair_attempts=1)
    assert [item["type"] for item in result["errors"]] == [
        "budget_exhausted",
        "budget_exhausted",
    ]
    assert exhausted.context.stats.calls_made == 2
    assert len(exhausted.prompts) == 1


def test_structured_batch_caller_validator_participates_in_bounded_repair() -> None:
    client = ScriptedSubcalls([(['{"count":1}'], []), (['{"count":2}'], [])])

    def require_two(value: Any, index: int) -> None:
        if value["count"] != 2:
            raise ValueError(f"item {index} count must equal 2")

    result = structured_batch(client, ["a"], SCHEMA, validator=require_two)

    assert result["values"] == [{"count": 2}]
    assert result["errors"] == []
    assert "count must equal 2" in client.prompts[1][0]


def test_structured_batch_attributes_validator_errors_without_repairing_them() -> None:
    client = ScriptedSubcalls([(['{"count":1}', '{"count":2}', '{"count":3}'], [])])

    def fail_middle(value: Any, index: int) -> None:
        if index == 1:
            raise KeyError("validator bug")

    result = structured_batch(client, ["a", "b", "c"], SCHEMA, validator=fail_middle)

    assert result == {
        "values": [{"count": 1}, None, {"count": 3}],
        "errors": [
            {
                "index": 1,
                "type": "validator_error",
                "error": "'validator bug'",
                "attempts": 1,
            }
        ],
        "attempts": [1, 1, 1],
        "repairs_made": 0,
    }
    assert len(client.prompts) == 1
    assert client.context.stats.calls_made == 3


@pytest.mark.parametrize(
    ("value", "schema"),
    [
        (True, {"const": 1}),
        (False, {"enum": [0]}),
        ({"nested": [True]}, {"const": {"nested": [1]}}),
        ([{"nested": False}], {"enum": [[{"nested": 0}]]}),
    ],
    ids=["const-top-level", "enum-top-level", "const-nested", "enum-nested"],
)
def test_validate_json_const_and_enum_distinguish_booleans_from_numbers(
    value: Any, schema: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        validate_json(value, schema)


@pytest.mark.parametrize(
    ("value", "schema"),
    [
        (1, {"const": 1.0}),
        (1.0, {"enum": [1]}),
        ({"nested": [1]}, {"const": {"nested": [1.0]}}),
        ([{"nested": 1.0}], {"enum": [[{"nested": 1}]]}),
    ],
    ids=["const-top-level", "enum-top-level", "const-nested", "enum-nested"],
)
def test_validate_json_const_and_enum_preserve_numeric_equivalence(
    value: Any, schema: dict[str, Any]
) -> None:
    validate_json(value, schema)


def test_validate_json_const_objects_require_matching_keys() -> None:
    with pytest.raises(ValueError, match="declared const"):
        validate_json({"actual": 1}, {"const": {"expected": 1}})


@pytest.mark.parametrize(
    ("schema", "invalid"),
    [
        ({"type": "number"}, "1e999"),
        ({"type": "object"}, '{"nested":[{"value":1e999}]}'),
    ],
    ids=["top-level", "nested"],
)
def test_structured_batch_rejects_non_finite_values_without_repair(
    schema: dict[str, Any], invalid: str
) -> None:
    client = ScriptedSubcalls([([invalid], [])])

    result = structured_batch(client, ["a"], schema, max_repair_attempts=0)

    assert result["values"] == [None]
    assert result["errors"][0]["type"] == "validation_error"
    assert "finite JSON number" in result["errors"][0]["error"]
    assert result["attempts"] == [1]
    assert result["repairs_made"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    ("schema", "invalid", "repaired", "expected"),
    [
        ({"type": "number"}, "1e999", "1", 1),
        (
            {"type": "object"},
            '{"nested":[{"value":1e999}]}',
            '{"nested":[{"value":1}]}',
            {"nested": [{"value": 1}]},
        ),
    ],
    ids=["top-level", "nested"],
)
def test_structured_batch_repairs_non_finite_values_once(
    schema: dict[str, Any], invalid: str, repaired: str, expected: Any
) -> None:
    client = ScriptedSubcalls([([invalid], []), ([repaired], [])])

    result = structured_batch(client, ["a"], schema, max_repair_attempts=1)

    assert result["values"] == [expected]
    assert result["errors"] == []
    assert result["attempts"] == [2]
    assert result["repairs_made"] == 1
    assert "finite JSON number" in client.prompts[1][0]
    json.dumps(result, allow_nan=False)


def test_structured_batch_rejects_unsupported_nested_schema_before_spending() -> None:
    client = ScriptedSubcalls([])
    schema = {"type": "object", "properties": {"x": {"pattern": "x"}}}

    with pytest.raises(ValueError, match="unsupported JSON schema keywords: pattern"):
        structured_batch(client, ["a"], schema)

    assert client.context.stats.calls_made == 0


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "unsupported"},
        {"type": ["object", 1]},
        {"minimum": True},
        {"maximum": "10"},
        {"minItems": True},
        {"maxItems": -1},
        {"required": "value"},
        {"required": ["value", "value"]},
        {"enum": {}},
        {"properties": []},
        {"properties": {1: {"type": "string"}}},
        {"additionalProperties": "yes"},
        {"items": []},
        {"properties": {"nested": {"minimum": False}}},
        {"additionalProperties": {"type": "unsupported"}},
        {"items": {"maxItems": True}},
    ],
)
def test_structured_batch_rejects_invalid_schema_leaf_shapes_without_spending(
    schema: dict[str, Any],
) -> None:
    client = ScriptedSubcalls([])

    with pytest.raises(ValueError):
        structured_batch(client, ["a"], schema)

    assert client.context.stats.calls_made == 0


def test_structured_batch_rejects_unsupported_type_before_spending() -> None:
    client = ScriptedSubcalls([])

    with pytest.raises(ValueError, match="unsupported names: impossible"):
        structured_batch(client, ["a"], {"type": "impossible"})

    assert client.context.stats.calls_made == 0


def test_structured_batch_uses_errors_to_disambiguate_valid_json_null() -> None:
    client = ScriptedSubcalls(
        [
            (
                ["null", ""],
                [{"index": 1, "type": "provider_error", "error": "unavailable"}],
            )
        ]
    )

    result = structured_batch(client, ["valid null", "failed"], {"type": "null"})

    assert result["values"] == [None, None]
    assert result["errors"] == [
        {
            "index": 1,
            "type": "provider_error",
            "error": "unavailable",
            "attempts": 1,
        }
    ]
    assert {item["index"] for item in result["errors"]} == {1}


def test_structured_batch_supports_exact_legacy_budget_runtime_error() -> None:
    class LegacyBudgetClient(ScriptedSubcalls):
        def llm_batch_with_errors(self, prompts, contexts=None):
            raise RuntimeError("max subcalls exceeded")

    result = structured_batch(LegacyBudgetClient([]), ["a"], SCHEMA)
    assert result["errors"][0]["type"] == "budget_exhausted"

    class SimilarProviderErrorClient(ScriptedSubcalls):
        def llm_batch_with_errors(self, prompts, contexts=None):
            raise RuntimeError("provider reported max subcalls exceeded downstream")

    provider = structured_batch(SimilarProviderErrorClient([]), ["a"], SCHEMA)
    assert provider["errors"][0]["type"] == "provider_error"


def test_subcall_budget_exception_preserves_runtime_error_compatibility() -> None:
    assert issubclass(SubcallBudgetExceeded, RuntimeError)


def test_aggregate_json_counts_is_exact_and_refuses_inconsistent_chunks() -> None:
    labels = ["entity", "location"]
    values = [
        {"counts": {"entity": 2, "location": 1}},
        {"counts": {"entity": 1, "location": 1}},
    ]
    assert aggregate_json_counts(values, labels, [3, 2]) == {"entity": 3, "location": 2}

    with pytest.raises(ValueError, match="do not sum to chunk size"):
        aggregate_json_counts(values, labels, [4, 2])
