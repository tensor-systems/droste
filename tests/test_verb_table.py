"""Provider operation vocabulary is descriptor data, not an engine verb table."""

from __future__ import annotations

import pytest

from droste import (
    JSON_SCHEMA_2020_12,
    PaginationMode,
    ProviderManifest,
    ProviderOperation,
    ResultDelivery,
    SchemaSpec,
)
from droste.testing import FAKE_RECORDS_MANIFEST


def _schema(value, provenance="test:schema@1") -> SchemaSpec:
    return SchemaSpec(value, JSON_SCHEMA_2020_12, provenance)


def test_provider_can_use_domain_specific_raw_ids_without_engine_changes() -> None:
    operation = FAKE_RECORDS_MANIFEST.operations[0]
    assert operation.operation_id == "records.search"
    assert operation.binding_name == "search"
    assert operation.pagination is PaginationMode.CURSOR
    assert operation.delivery is ResultDelivery.INLINE


def test_cursor_pagination_requires_both_cursor_schemas() -> None:
    with pytest.raises(ValueError, match="cursor parameter"):
        ProviderOperation(
            "items.list",
            "list_items",
            "List items.",
            _schema({"type": "object", "properties": {}}),
            _schema({"type": "object", "properties": {"next_cursor": {"type": "string"}}}),
            PaginationMode.CURSOR,
            ResultDelivery.INLINE,
            "data.list",
        )
    with pytest.raises(ValueError, match="next_cursor"):
        ProviderOperation(
            "items.list",
            "list_items",
            "List items.",
            _schema({"type": "object", "properties": {"cursor": {"type": "string"}}}),
            _schema({"type": "object", "properties": {"items": {"type": "array"}}}),
            PaginationMode.CURSOR,
            ResultDelivery.INLINE,
            "data.list",
        )


def test_typed_and_untyped_delivery_are_explicit() -> None:
    with pytest.raises(ValueError, match="must not declare"):
        ProviderOperation(
            "opaque.read",
            "read_opaque",
            "Read opaque data.",
            _schema({"type": "object"}),
            _schema({"type": "object"}),
            PaginationMode.NONE,
            ResultDelivery.UNTYPED,
            "data.read",
        )
    with pytest.raises(TypeError, match="require a result"):
        ProviderOperation(
            "typed.read",
            "read_typed",
            "Read typed data.",
            _schema({"type": "object"}),
            None,
            PaginationMode.NONE,
            ResultDelivery.HANDLE,
            "data.read",
        )


def test_manifest_rejects_duplicate_raw_or_python_names() -> None:
    operation = FAKE_RECORDS_MANIFEST.operations[0]
    with pytest.raises(ValueError, match="duplicate operation"):
        ProviderManifest("duplicate", "1", (operation, operation))
    second = ProviderOperation(
        "records.other",
        operation.binding_name,
        "Another search.",
        operation.parameters,
        operation.result,
        operation.pagination,
        operation.delivery,
        operation.budget_class,
    )
    with pytest.raises(ValueError, match="duplicate binding"):
        ProviderManifest("duplicate", "1", (operation, second))
