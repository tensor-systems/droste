from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .protocols.data_source import DataSource

# Base globals the runner owns; a data source may not shadow them.
RESERVED_NAMES = frozenset(
    {"answer", "context", "llm_query", "llm_batch", "batch_llm_query", "llm_query_batched"}
)


class DataSourceRegistry:
    """Registry for composing data sources into environment globals."""

    def __init__(
        self,
        sources: list[DataSource],
        *,
        default_source_name: str | None = None,
    ) -> None:
        self._sources = sources
        self._default_source_name = default_source_name

    def globals(self) -> dict[str, Any]:
        env: dict[str, Any] = {}
        seen: set[str] = set()
        for source in self._sources:
            name = source.name()
            if name in RESERVED_NAMES:
                raise ValueError(f"data source name {name!r} is reserved")
            if name in seen:
                raise ValueError(f"duplicate data source name: {name!r}")
            seen.add(name)

            ns: dict[str, Any] = {}
            caps = source.capabilities()

            if caps.get("search"):
                ns["search"] = source.search
            if hasattr(source, "find"):
                ns["find"] = source.find
            if caps.get("sql"):
                ns["query"] = source.query
            if caps.get("get"):
                ns["get"] = source.get
            if caps.get("recent"):
                ns["get_recent"] = source.get_recent
            if caps.get("schema"):
                ns["get_schema"] = source.get_schema
            if caps.get("stats"):
                ns["get_stats"] = source.get_stats

            if hasattr(source, "content"):
                ns["content"] = source.content
            if hasattr(source, "get_messages"):
                ns["get_messages"] = source.get_messages
            if hasattr(source, "get_chats"):
                ns["get_chats"] = source.get_chats
            if hasattr(source, "get_chat_messages"):
                ns["get_chat_messages"] = source.get_chat_messages
            if hasattr(source, "sample"):
                ns["sample"] = source.sample

            # Expose an attribute-accessible namespace so the model can write
            # `db.query(...)` (a dict would force `db["query"](...)`). The verbs
            # return Python values into the sandbox — they are not tool calls.
            env[name] = SimpleNamespace(**ns)

            if self._default_source_name == name:
                for key, value in ns.items():
                    env[key] = value

        if self._default_source_name is not None and self._default_source_name not in seen:
            raise ValueError(
                f"default_source {self._default_source_name!r} is not a defined source"
            )

        return env

    def prompt_fragment(self) -> str:
        parts: list[str] = ["## Data Sources"]
        for source in self._sources:
            parts.append(f"- {source.name()}:\n{source.get_schema()}")
        parts.append("## Working with data sources")
        parts.append(
            "These are accessors into data that may be far larger than your context "
            "window. Call them to pull data into Python variables, then reduce, filter, "
            "and compute over those variables in code — do not try to read everything "
            "into the prompt. Variables persist across iterations, so build up state: "
            'e.g. `rows = db.query("SELECT ...")`, then process `rows` in Python, then '
            "fan out over chunks with `llm_batch`. Only print the reduced result you need."
        )
        parts.append(
            'Call them namespaced: `db.query("SELECT ...")`, `vault.search("...")`.'
        )
        if self._default_source_name:
            parts.append(
                f"The default source '{self._default_source_name}' is also available "
                "unprefixed (e.g. `query(...)`, `search(...)`)."
            )
        return "\n".join(parts)
