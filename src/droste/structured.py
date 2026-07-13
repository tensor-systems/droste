from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .exceptions import SubcallBudgetExceeded
from .protocols.subcall_client import SubcallClient

_SCHEMA_KEYWORDS = {
    "type",
    "enum",
    "const",
    "required",
    "properties",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
}
_JSON_TYPES = frozenset({"null", "boolean", "integer", "number", "string", "array", "object"})


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_schema_definition(schema: Mapping[str, Any], *, path: str = "schema") -> None:
    if not isinstance(schema, Mapping):
        raise ValueError(f"{path} must be an object")
    unknown = sorted(set(schema) - _SCHEMA_KEYWORDS)
    if unknown:
        raise ValueError(f"unsupported JSON schema keywords: {', '.join(unknown)}")

    expected = schema.get("type")
    if "type" in schema:
        expected_types = [expected] if isinstance(expected, str) else expected
        if (
            not isinstance(expected_types, list)
            or not expected_types
            or not all(isinstance(item, str) for item in expected_types)
        ):
            raise ValueError(f"{path}.type must be a string or non-empty string array")
        unsupported = sorted(set(expected_types) - _JSON_TYPES)
        if unsupported:
            raise ValueError(f"{path}.type has unsupported names: {', '.join(unsupported)}")

    required = schema.get("required")
    if "required" in schema:
        if (
            not isinstance(required, list)
            or not all(isinstance(item, str) for item in required)
            or len(set(required)) != len(required)
        ):
            raise ValueError(f"{path}.required must be a unique string array")

    enum = schema.get("enum")
    if "enum" in schema and not isinstance(enum, list):
        raise ValueError(f"{path}.enum must be an array")

    properties = schema.get("properties")
    if "properties" in schema:
        if not isinstance(properties, Mapping):
            raise ValueError(f"{path}.properties must be an object")
        for key, child in properties.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}.properties keys must be strings")
            _validate_schema_definition(child, path=f"{path}.properties.{key}")

    additional = schema.get("additionalProperties")
    if "additionalProperties" in schema and not isinstance(additional, (bool, Mapping)):
        raise ValueError(f"{path}.additionalProperties must be a boolean or schema")
    if isinstance(additional, Mapping):
        _validate_schema_definition(additional, path=f"{path}.additionalProperties")

    items = schema.get("items")
    if "items" in schema:
        if not isinstance(items, Mapping):
            raise ValueError(f"{path}.items must be a schema object")
        _validate_schema_definition(items, path=f"{path}.items")

    for keyword in ("minimum", "maximum"):
        bound = schema.get(keyword)
        if keyword in schema:
            if not _is_number(bound) or (isinstance(bound, float) and not math.isfinite(bound)):
                raise ValueError(f"{path}.{keyword} must be a finite number, not a boolean")

    for keyword in ("minItems", "maxItems"):
        bound = schema.get(keyword)
        if keyword in schema and (
            not isinstance(bound, int) or isinstance(bound, bool) or bound < 0
        ):
            raise ValueError(f"{path}.{keyword} must be a non-negative integer")

    try:
        json.dumps(schema, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must contain only JSON-serializable values") from exc


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    raise ValueError(f"unsupported JSON schema type: {expected}")


def _validate_finite_json_numbers(value: Any, *, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be a finite JSON number")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json_numbers(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json_numbers(item, path=f"{path}[{index}]")


def _json_equal(left: Any, right: Any) -> bool:
    """Compare values using JSON Schema's JSON-value equality semantics."""

    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if _is_number(left) or _is_number(right):
        return _is_number(left) and _is_number(right) and left == right
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(_json_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or len(left) != len(right):
            return False
        if not all(isinstance(key, str) for key in left) or not all(
            isinstance(key, str) for key in right
        ):
            return False
        return all(key in right and _json_equal(value, right[key]) for key, value in left.items())
    return type(left) is type(right) and left == right


def _validate_json_value(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    expected = schema.get("type")
    if expected is not None:
        expected_types = [expected] if isinstance(expected, str) else expected
        if not any(_json_type_matches(value, item) for item in expected_types):
            raise ValueError(f"{path} must have type {' or '.join(expected_types)}")

    if "const" in schema and not _json_equal(value, schema["const"]):
        raise ValueError(f"{path} must equal the declared const value")
    if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
        raise ValueError(f"{path} must be one of the declared enum values")

    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [item for item in required if item not in value]
        if missing:
            raise ValueError(f"{path} is missing required properties: {', '.join(missing)}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key in properties:
                _validate_json_value(item, properties[key], path=item_path)
            elif additional is False:
                raise ValueError(f"{path} has unexpected property {key!r}")
            elif isinstance(additional, Mapping):
                _validate_json_value(item, additional, path=item_path)

    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        maximum_items = schema.get("maxItems")
        if minimum_items is not None and len(value) < minimum_items:
            raise ValueError(f"{path} must contain at least {minimum_items} items")
        if maximum_items is not None and len(value) > maximum_items:
            raise ValueError(f"{path} must contain at most {maximum_items} items")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate_json_value(item, item_schema, path=f"{path}[{index}]")

    if _is_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} must be <= {schema['maximum']}")


def validate_json(value: Any, schema: Mapping[str, Any], *, path: str = "$") -> None:
    """Validate a JSON value against Droste's deterministic schema subset.

    Supported keywords are ``type``, ``enum``, ``const``, ``required``,
    ``properties``, ``additionalProperties``, ``items``, ``minItems``,
    ``maxItems``, ``minimum``, and ``maximum``. Unsupported keywords fail
    closed so callers never mistake partial validation for full JSON Schema.
    """

    _validate_schema_definition(schema)
    _validate_finite_json_numbers(value, path=path)
    _validate_json_value(value, schema, path=path)


def _parse_json_output(text: str) -> Any:
    stripped = str(text).strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].strip() in {"```", "```json", "```JSON"}:
            stripped = "\n".join(lines[1:-1]).strip()
    return json.loads(
        stripped,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON numeric constant: {value}")
        ),
    )


def _repair_prompt(prompt: str, raw: str, schema: Mapping[str, Any], error: str) -> str:
    encoded_schema = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return (
        "Your previous response was malformed or did not match the required JSON schema. "
        "Return only corrected JSON, with no markdown or explanation.\n"
        f"Validation error: {error}\nRequired schema: {encoded_schema}\n"
        f"Original task:\n{prompt}\nPrevious response:\n{str(raw)[:4000]}"
    )


def structured_batch(
    subcalls: SubcallClient,
    prompts: list[str],
    schema: Mapping[str, Any],
    contexts: list[str] | None = None,
    *,
    max_repair_attempts: int = 1,
    validator: Callable[[Any, int], None] | None = None,
) -> dict[str, Any]:
    """Run an ordered JSON batch with malformed-only bounded repair.

    Provider/transport failures are never retried as parse failures. Each
    batch attempt delegates to ``llm_batch_with_errors``; native batch clients
    therefore retain one wire request per attempt and their atomic call-budget
    reservation. Returned values stay in caller order. Both a valid JSON
    ``null`` and a failed slot appear as ``None`` in ``values``; ``errors`` is
    authoritative for deciding which indices failed, keeping the sandbox
    result fully JSON-serializable. Caller validators should raise
    ``ValueError`` for output rejection that is eligible for repair. Other
    exceptions are reported per item as non-retried ``validator_error`` entries.
    """

    if not isinstance(max_repair_attempts, int) or isinstance(max_repair_attempts, bool):
        raise ValueError("max_repair_attempts must be an integer")
    if max_repair_attempts < 0:
        raise ValueError("max_repair_attempts must be >= 0")
    _validate_schema_definition(schema)
    if contexts is not None and len(contexts) != len(prompts):
        raise ValueError("contexts length must match prompts length")

    values: list[Any | None] = [None] * len(prompts)
    attempts = [0] * len(prompts)
    errors_by_index: dict[int, dict[str, Any]] = {}

    def call(indices: list[int], call_prompts: list[str], call_contexts: list[str] | None) -> None:
        try:
            raw_values, raw_errors = subcalls.llm_batch_with_errors(call_prompts, call_contexts)
        except Exception as exc:
            # Third-party clients predating SubcallBudgetExceeded commonly
            # used this exact RuntimeError. Keep that narrow compatibility
            # fallback while every Droste-owned client raises the typed form.
            legacy_budget_error = type(exc) is RuntimeError and str(exc) == "max subcalls exceeded"
            kind = (
                "budget_exhausted"
                if isinstance(exc, SubcallBudgetExceeded) or legacy_budget_error
                else "provider_error"
            )
            for index in indices:
                attempts[index] += 1
                errors_by_index[index] = {
                    "index": index,
                    "type": kind,
                    "error": str(exc),
                    "attempts": attempts[index],
                }
            return
        if len(raw_values) != len(indices):
            raise RuntimeError("subcall client returned the wrong number of batch results")
        provider_errors = {int(item["index"]): item for item in raw_errors}
        for local_index, index in enumerate(indices):
            attempts[index] += 1
            if local_index in provider_errors:
                item = provider_errors[local_index]
                errors_by_index[index] = {
                    **item,
                    "index": index,
                    "type": str(item.get("type") or "provider_error"),
                    "attempts": attempts[index],
                }
                continue
            raw = raw_values[local_index]
            try:
                parsed = _parse_json_output(raw)
                validate_json(parsed, schema)
            except (json.JSONDecodeError, ValueError) as exc:
                errors_by_index[index] = {
                    "index": index,
                    "type": "validation_error",
                    "error": str(exc),
                    "attempts": attempts[index],
                    "raw": str(raw)[:1000],
                }
                continue
            if validator is not None:
                try:
                    validator(parsed, index)
                except ValueError as exc:
                    errors_by_index[index] = {
                        "index": index,
                        "type": "validation_error",
                        "error": str(exc),
                        "attempts": attempts[index],
                        "raw": str(raw)[:1000],
                    }
                    continue
                except Exception as exc:
                    errors_by_index[index] = {
                        "index": index,
                        "type": "validator_error",
                        "error": str(exc),
                        "attempts": attempts[index],
                    }
                    continue
            values[index] = parsed
            errors_by_index.pop(index, None)

    initial_indices = list(range(len(prompts)))
    call(initial_indices, prompts, contexts)
    repairs_made = 0
    for _ in range(max_repair_attempts):
        repair_indices = [
            index
            for index, item in sorted(errors_by_index.items())
            if item["type"] == "validation_error"
        ]
        if not repair_indices:
            break
        repair_prompts = [
            _repair_prompt(
                prompts[index],
                errors_by_index[index].get("raw", ""),
                schema,
                errors_by_index[index]["error"],
            )
            for index in repair_indices
        ]
        repair_contexts = (
            [contexts[index] for index in repair_indices] if contexts is not None else None
        )
        repairs_made += len(repair_indices)
        call(repair_indices, repair_prompts, repair_contexts)

    return {
        "values": values,
        "errors": [errors_by_index[index] for index in sorted(errors_by_index)],
        "attempts": attempts,
        "repairs_made": repairs_made,
    }


def bind_structured_batch(subcalls: SubcallClient) -> Callable[..., dict[str, Any]]:
    def llm_batch_json(
        prompts: list[str],
        schema: Mapping[str, Any],
        contexts: list[str] | None = None,
        max_repair_attempts: int = 1,
        validator: Callable[[Any, int], None] | None = None,
    ) -> dict[str, Any]:
        return structured_batch(
            subcalls,
            prompts,
            schema,
            contexts,
            max_repair_attempts=max_repair_attempts,
            validator=validator,
        )

    return llm_batch_json


def aggregate_json_counts(
    values: Sequence[Any], labels: Sequence[str], chunk_sizes: Sequence[int]
) -> dict[str, int]:
    """Validate and sum compact ``{"counts": ...}`` chunk classifications."""

    if len(values) != len(chunk_sizes):
        raise ValueError("values length must match chunk_sizes length")
    normalized_labels = list(labels)
    if not normalized_labels or len(set(normalized_labels)) != len(normalized_labels):
        raise ValueError("labels must be a non-empty unique sequence")
    totals = {label: 0 for label in normalized_labels}
    expected_keys = set(normalized_labels)
    for index, (value, chunk_size) in enumerate(zip(values, chunk_sizes, strict=True)):
        if not isinstance(value, Mapping) or not isinstance(value.get("counts"), Mapping):
            raise ValueError(f"chunk {index} must contain a counts object")
        counts = value["counts"]
        if set(counts) != expected_keys:
            raise ValueError(f"chunk {index} counts must contain exactly the declared labels")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 0:
            raise ValueError(f"chunk {index} size must be a non-negative integer")
        for label, count in counts.items():
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"chunk {index} count for {label!r} must be non-negative integer")
        if sum(counts.values()) != chunk_size:
            raise ValueError(f"chunk {index} counts do not sum to chunk size {chunk_size}")
        for label in normalized_labels:
            totals[label] += counts[label]
    return totals
