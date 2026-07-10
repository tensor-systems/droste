"""Typed source-type registration (#32).

Consumers register factories for their own source types at *startup* — never
from the request. The request stays declarative ({type, name, ...} — no module
paths, no code); the set of runnable types is fixed by the deployment's own
entrypoint, so there is no request-controlled import path.

Lives at the droste level so any embedder (Pyodide adapters, in-process
hosts) can use typed source registration without importing ``droste_runner``;
``droste_runner`` re-exports the old names.
"""

from __future__ import annotations

from typing import Any, Callable

# Version of the DataSource/registration contract. A consumer built against a
# different contract fails at registration (startup), not subtly at request time.
# v2 (#10): the engine no longer auto-binds domain verbs (get_messages/
# get_chats/get_chat_messages) by hasattr — a source declares them via
# `extra_methods` — and search() lost its chat-archive kwargs. A protocol-1
# source registering against this engine must fail loudly at startup rather
# than silently losing its accessors at runtime.
SOURCE_PROTOCOL_VERSION = 2

# factory(config, ctx) -> DataSource. `config` is the request's declarative
# per-source entry; `ctx` is the host-supplied edge context (e.g. a live
# read-only DB handle) threaded through build_data_sources — see §7.3.
SourceFactory = Callable[[dict[str, Any], Any], Any]

_source_factories: dict[str, SourceFactory] = {}


def register_source_type(
    stype: str,
    factory: SourceFactory,
    *,
    protocol: int,
) -> None:
    """Register a factory for a data source type (process-global, at startup).

    ``protocol`` is REQUIRED and must be the literal version the registrant
    was written against — a default would let a stale extension silently
    self-certify as current, defeating the startup compatibility check
    (codex review on the v2 bump). In-tree sources pass the imported
    constant because they are co-versioned with the engine; external
    registrants must pass the literal they implement.
    """
    if protocol != SOURCE_PROTOCOL_VERSION:
        raise RuntimeError(
            f"source type {stype!r} was registered against protocol {protocol}; "
            f"this engine speaks protocol {SOURCE_PROTOCOL_VERSION}"
        )
    key = str(stype or "").strip()
    if not key:
        raise ValueError("source type must be a non-empty string")
    if key in _source_factories:
        raise ValueError(f"source type {key!r} is already registered")
    if not callable(factory):
        raise TypeError("factory must be callable")
    _source_factories[key] = factory


def source_factory(stype: str) -> SourceFactory | None:
    """Look up the registered factory for a source type, if any."""
    return _source_factories.get(str(stype))


def _reset_source_types() -> None:
    """Test hook: clear registered source-type factories."""
    _source_factories.clear()
