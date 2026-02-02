from __future__ import annotations

from typing import Any

from ..protocols.data_source import DataSource, DataSourceCapabilities, SearchResult


class MockDataSource(DataSource):
    def __init__(
        self,
        *,
        schema: str = "",
        stats: dict[str, Any] | None = None,
        query_results: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._schema = schema
        self._stats = stats or {}
        self._query_results = query_results or {}

    def name(self) -> str:
        return "mock"

    def capabilities(self) -> DataSourceCapabilities:
        return {
            "sql": True,
            "search": True,
            "get": True,
            "recent": True,
            "schema": True,
            "stats": True,
        }

    def get_schema(self) -> str:
        return self._schema

    def get_stats(self) -> dict[str, Any]:
        return dict(self._stats)

    def search(self, query: str, limit: int = 50, filters: dict[str, Any] | None = None) -> list[SearchResult]:
        return []

    def query(self, sql: str) -> list[dict[str, Any]]:
        for key, value in self._query_results.items():
            if key in sql:
                return value
        return []

    def get(self, id: str) -> dict[str, Any] | None:
        return None

    def get_recent(self, days: int = 7, limit: int = 100) -> list[dict[str, Any]]:
        return []
