from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .protocols.data_source import DataSource
from .protocols.verbs import (
    RESERVED_NAMES,
    VERB_SPECS,
    AccessorManifest,
    validate_extra_method_name,
)

__all__ = [
    "DataSourceRegistry",
    "RESERVED_NAMES",
    "validate_extra_method_name",
]


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

    def _bound_verbs(self, source: DataSource) -> dict[str, Any]:
        """The verbs this source binds into the sandbox: one fold over the
        protocols-level verb table (capability- or hasattr-gated), plus the
        source's own validated extras (#10)."""
        name = source.name()
        ns: dict[str, Any] = {}
        caps = source.capabilities()

        for spec in VERB_SPECS:
            if spec.capability is not None:
                if caps.get(spec.capability):
                    ns[spec.name] = getattr(source, spec.name)
            elif hasattr(source, spec.name):
                ns[spec.name] = getattr(source, spec.name)

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

        return ns

    def globals(self) -> dict[str, Any]:
        env: dict[str, Any] = {}
        seen: set[str] = set()
        # All source names up front: a flattened default-source verb (core or
        # extra) must never overwrite another source's namespace object —
        # e.g. a default source with extra_methods=("archive",) next to a
        # source named "archive" would silently replace env["archive"] with
        # a bound method.
        all_source_names = {s.name() for s in self._sources}
        for source in self._sources:
            name = source.name()
            if name in RESERVED_NAMES:
                raise ValueError(f"data source name {name!r} is reserved")
            if name in seen:
                raise ValueError(f"duplicate data source name: {name!r}")
            seen.add(name)

            ns = self._bound_verbs(source)

            # Expose an attribute-accessible namespace so the model can write
            # `db.query(...)` (a dict would force `db["query"](...)`). The verbs
            # return Python values into the sandbox — they are not tool calls.
            env[name] = SimpleNamespace(**ns)

            if self._default_source_name == name:
                for key, value in ns.items():
                    if key in all_source_names:
                        raise ValueError(
                            f"flattened verb {key!r} of default source {name!r} would "
                            "overwrite a registered source's namespace"
                        )
                    env[key] = value

        if self._default_source_name is not None and self._default_source_name not in seen:
            raise ValueError(
                f"default_source {self._default_source_name!r} is not a defined source"
            )

        return env

    def accessor_manifest(self) -> AccessorManifest:
        """Explicit inventory of the data accessors globals() binds, for the
        count contract's len() check — replaces the old namespace provenance
        marker that policy discovery had to sniff back out of env globals."""
        flat: set[str] = set()
        namespaced: set[tuple[str, str]] = set()
        for source in self._sources:
            name = source.name()
            verbs = self._bound_verbs(source)
            namespaced.update((name, verb) for verb in verbs)
            if self._default_source_name == name:
                flat.update(verbs)
        return AccessorManifest(flat=frozenset(flat), namespaced=frozenset(namespaced))

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
