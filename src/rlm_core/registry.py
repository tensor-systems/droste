from __future__ import annotations

from typing import Any

from .protocols.data_source import DataSource


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
        for source in self._sources:
            name = source.name()
            ns: dict[str, Any] = {}
            caps = source.capabilities()

            if caps.get("search"):
                ns["search"] = source.search
            if caps.get("sql"):
                ns["query"] = source.query
            if caps.get("get"):
                ns["get"] = source.get
            if caps.get("recent"):
                ns["get_recent"] = source.get_recent
            if caps.get("stats"):
                ns["get_stats"] = source.get_stats

            env[name] = ns

            if self._default_source_name == name:
                for key, value in ns.items():
                    env[key] = value

        return env

    def prompt_fragment(self) -> str:
        parts: list[str] = ["## Data Sources"]
        for source in self._sources:
            parts.append(f"- {source.name()}:\n{source.get_schema()}")
        parts.append("## Functions")
        parts.append("Use namespaced calls like gmail.search(), messages.query().")
        if self._default_source_name:
            parts.append(
                f"Default source '{self._default_source_name}' is also available as query(), search(), get_recent()."
            )
        return "\n".join(parts)
