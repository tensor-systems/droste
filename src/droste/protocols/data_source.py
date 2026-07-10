from __future__ import annotations

from typing import Any, Protocol, TypedDict


class SearchResult(TypedDict, total=False):
    """Single search result."""

    id: str
    snippet: str
    metadata: dict[str, Any]
    score: float | None


class DataSourceCapabilities(TypedDict):
    """Capabilities for a data source."""

    sql: bool
    search: bool
    get: bool
    recent: bool
    schema: bool
    stats: bool


class DataSource(Protocol):
    """Abstract data access layer.

    The protocol is domain-blind: core verbs only, nothing that encodes any
    particular host's data shape. A host whose source has domain-specific
    verbs beyond these (a chat archive's bulk accessors, a citations
    feature's ID tracking, ...) declares them in an ``extra_methods``
    attribute — a tuple of method names — and the registry and the bridge's
    ``DataSourceService`` expose exactly those callables to the sandbox, on
    top of the capability-gated core verbs below.
    """

    def name(self) -> str:
        """Unique name for namespacing in the environment."""
        ...

    def capabilities(self) -> DataSourceCapabilities:
        """Return data source capabilities."""
        ...

    def get_schema(self) -> str:
        """Return schema description for system prompt."""
        ...

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about the data source."""
        ...

    def search(
        self,
        query: str,
        limit: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Full-text search. Domain-specific narrowing goes in ``filters``
        (or in a concrete source's own extra kwargs — the registry and
        bridge forward call arguments verbatim, so an implementation may
        accept more than the protocol declares)."""
        ...

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Direct SQL query."""
        ...

    def get(self, id: str) -> dict[str, Any] | None:
        """Get single record by ID."""
        ...

    def get_recent(self, days: int = 7, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent records."""
        ...

    def sample(self, n: int = 1000) -> list[dict[str, Any]]:
        """Get random sample (optional)."""
        ...
