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
    """Abstract data access layer."""

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
        sender: str | None = None,
        chat: str | None = None,
        days: int | None = None,
        table: str = "messages",
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Full-text search (supports chat-archive-style args plus filters)."""
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

    def get_messages(self, limit: int | None = 10000) -> list[dict[str, Any]]:
        """Get messages in bulk (optional)."""
        ...

    def get_chats(self) -> list[dict[str, Any]]:
        """Get list of chats (optional)."""
        ...

    def get_chat_messages(self, chat_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        """Get messages from a chat (optional)."""
        ...

    def sample(self, n: int = 1000) -> list[dict[str, Any]]:
        """Get random sample (optional)."""
        ...
