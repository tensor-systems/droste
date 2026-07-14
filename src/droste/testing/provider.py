"""Small non-SQL provider used by conformance tests and examples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..capabilities import (
    JSON_SCHEMA_2020_12,
    PaginationMode,
    ProviderOperation,
    ResultDelivery,
    SchemaSpec,
    SideEffect,
)
from ..providers import (
    ConfiguredSource,
    ProviderManifest,
    ProviderRegistration,
    ProviderRuntime,
)


def _schema(value: dict[str, Any], suffix: str) -> SchemaSpec:
    return SchemaSpec(value, JSON_SCHEMA_2020_12, f"droste:testing/fake/{suffix}@1")


FAKE_RECORDS_MANIFEST = ProviderManifest(
    provider_type="fake_records",
    revision="1",
    operations=(
        ProviderOperation(
            operation_id="records.search",
            binding_name="search",
            description="Search fake records with cursor pagination.",
            parameters=_schema(
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "cursor": {"type": ["string", "null"]},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "search/parameters",
            ),
            result=_schema(
                {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "items": {"type": "object"}},
                        "next_cursor": {"type": ["string", "null"]},
                    },
                    "required": ["items", "next_cursor"],
                },
                "search/result",
            ),
            pagination=PaginationMode.CURSOR,
            delivery=ResultDelivery.INLINE,
            budget_class="data.search",
        ),
        ProviderOperation(
            operation_id="records.fetch",
            binding_name="fetch",
            description="Fetch one fake record by ID.",
            parameters=_schema(
                {
                    "type": "object",
                    "properties": {"record_id": {"type": "string"}},
                    "required": ["record_id"],
                    "additionalProperties": False,
                },
                "fetch/parameters",
            ),
            result=_schema(
                {"anyOf": [{"type": "object"}, {"type": "null"}]},
                "fetch/result",
            ),
            pagination=PaginationMode.NONE,
            delivery=ResultDelivery.INLINE,
            budget_class="data.read",
        ),
    ),
)


def fake_records_provider(
    records: Mapping[str, Mapping[str, Any]] | None = None,
) -> ProviderRegistration:
    snapshot = {
        str(key): {str(field): value for field, value in record.items()}
        for key, record in (records or {"1": {"title": "alpha"}}).items()
    }

    def bind(source: ConfiguredSource, context: Any = None) -> ProviderRuntime:
        del context
        page_size = int(source.config_dict().get("page_size", 10))

        def search(
            query: str,
            cursor: str | None = None,
            limit: int | None = None,
        ) -> dict[str, Any]:
            start = int(cursor or 0)
            size = min(limit or page_size, page_size)
            matches = [
                {"id": key, **record}
                for key, record in snapshot.items()
                if query.lower() in str(record).lower()
            ]
            end = min(start + size, len(matches))
            return {
                "items": matches[start:end],
                "next_cursor": str(end) if end < len(matches) else None,
            }

        def fetch(record_id: str) -> dict[str, Any] | None:
            record = snapshot.get(record_id)
            return None if record is None else {"id": record_id, **record}

        return ProviderRuntime(
            handlers={"records.search": search, "records.fetch": fetch},
            source_description="In-memory fake records for provider conformance tests.",
        )

    return ProviderRegistration(
        manifest=FAKE_RECORDS_MANIFEST,
        effects={"records.search": SideEffect.READ, "records.fetch": SideEffect.READ},
        binder=bind,
    )


__all__ = ["FAKE_RECORDS_MANIFEST", "fake_records_provider"]
