from __future__ import annotations

import builtins
from types import SimpleNamespace
from typing import Any

from .protocols.data_source import DataSource

# Base globals the runner owns; a data source may not shadow them.
RESERVED_NAMES = frozenset(
    {"answer", "context", "llm_query", "llm_batch", "batch_llm_query", "llm_query_batched"}
)

# The engine's own verb vocabulary — capability-gated core verbs plus the
# generic hasattr-gated optionals. extra_methods may not reuse ANY of these,
# enabled or not: the bridge's dispatch checks core names before extras, so
# an extra that shadows a disabled core verb would work in-process but be
# rejected across the bridge — the same source must behave identically on
# every transport.
CORE_VERB_NAMES = frozenset(
    {
        "search",
        "query",
        "get",
        "get_recent",
        "get_schema",
        "get_stats",
        "find",
        "content",
        "sample",
    }
)

# Every name an extra_methods declaration may NOT use, shared by the registry
# and the bridge so a source config fails identically on every transport:
# engine verbs, runner-reserved globals, and the protocol/bridge machinery
# names (describe is a bridge wire method; name/capabilities/extra_methods
# are protocol surface a bridged client also binds as attributes).
EXTRA_METHOD_DISALLOWED = frozenset(
    CORE_VERB_NAMES | RESERVED_NAMES | {"name", "capabilities", "describe", "extra_methods"}
)


# A default source's verbs are flattened into the sandbox's execution
# globals, where a name like `len` or `print` would shadow the Python
# builtin for every line of generated code.
_BUILTIN_NAMES = frozenset(dir(builtins))


def validate_extra_method_name(extra: object, source_name: str) -> str:
    """Shared extras-name validation (registry + bridge). Returns the name."""
    extra_name = str(extra)
    if extra_name.startswith("_"):
        raise ValueError(
            f"extra method {extra_name!r} on source {source_name!r} may not begin "
            "with an underscore (private/machinery attributes are not sandbox verbs)"
        )
    if extra_name in EXTRA_METHOD_DISALLOWED:
        raise ValueError(
            f"extra method {extra_name!r} on source {source_name!r} collides with an "
            "engine verb, reserved global, or protocol attribute (core verbs may not "
            "be re-declared as extras, even when their capability is disabled)"
        )
    if extra_name in _BUILTIN_NAMES:
        raise ValueError(
            f"extra method {extra_name!r} on source {source_name!r} shadows a Python "
            "builtin — a flattened default-source verb by that name would hijack "
            "ordinary generated code"
        )
    return extra_name


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
            if hasattr(source, "sample"):
                ns["sample"] = source.sample

            # Host extras (#10): the engine is domain-blind, so any verbs
            # beyond the core set are declared by the source itself via an
            # `extra_methods` attribute (a tuple of method names) — the same
            # convention DataSourceService uses across the bridge, and what
            # BridgeDataSource re-exposes from the service's describe().
            for extra in tuple(getattr(source, "extra_methods", ()) or ()):
                extra_name = validate_extra_method_name(extra, name)
                fn = getattr(source, extra_name, None)
                if not callable(fn):
                    raise ValueError(
                        f"extra method {extra_name!r} on source {name!r} is not a callable"
                    )
                ns[extra_name] = fn

            # Expose an attribute-accessible namespace so the model can write
            # `db.query(...)` (a dict would force `db["query"](...)`). The verbs
            # return Python values into the sandbox — they are not tool calls.
            namespace = SimpleNamespace(**ns)
            # Provenance marker: the count contract's accessor discovery
            # (loop/rlm._collect_data_accessors) must classify only REAL
            # source namespaces, not any SimpleNamespace a custom
            # environment happens to expose (e.g. a `utils` helper bag).
            namespace._droste_data_source = True  # noqa: SLF001 - own marker
            env[name] = namespace

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
        parts.append('Call them namespaced: `db.query("SELECT ...")`, `vault.search("...")`.')
        if self._default_source_name:
            parts.append(
                f"The default source '{self._default_source_name}' is also available "
                "unprefixed (e.g. `query(...)`, `search(...)`)."
            )
        return "\n".join(parts)
