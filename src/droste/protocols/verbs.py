"""The engine's verb vocabulary — one declarative table (#31).

The verb -> gate mapping used to be encoded four ways (registry if-chain,
registry name set, bridge service allowlist, bridge client denylist),
synchronized by comments. This table is the single source: the registry's
namespace binding, the bridge service's dispatch allowlist, the bridge
client's proxy binding and defense-in-depth denylist, and extras-name
validation are all folds over it. Adding a core verb is one row here.

This is also the seed of the capability manifest promised by
docs/design/principles.md §2 and the vocabulary the #9 broker allowlist
starts from.
"""

from __future__ import annotations

import builtins
import keyword
from dataclasses import dataclass


@dataclass(frozen=True)
class VerbSpec:
    """One sandbox verb of the engine's own (domain-blind) vocabulary."""

    name: str
    # The capabilities() flag that enables the verb; None means the verb is
    # optional and gated by hasattr(source, name) instead.
    capability: str | None = None
    # True when real-world signatures vary by source beyond what the
    # DataSource Protocol declares (extra kwargs like a wrapper's `page`).
    # The bridge client binds these as *args/**kwargs proxies instead of
    # fixed-signature methods, so every call forwards argument-identically.
    dynamic_signature: bool = False


VERB_SPECS: tuple[VerbSpec, ...] = (
    VerbSpec("search", capability="search", dynamic_signature=True),
    VerbSpec("query", capability="sql", dynamic_signature=True),
    VerbSpec("get", capability="get", dynamic_signature=True),
    VerbSpec("get_recent", capability="recent", dynamic_signature=True),
    VerbSpec("get_schema", capability="schema"),
    VerbSpec("get_stats", capability="stats"),
    VerbSpec("find"),
    VerbSpec("content"),
    VerbSpec("sample"),
)

# The engine's own verb names. extra_methods may not reuse ANY of these,
# enabled or not: the bridge's dispatch checks core names before extras, so
# an extra that shadows a disabled core verb would work in-process but be
# rejected across the bridge — the same source must behave identically on
# every transport.
CORE_VERB_NAMES = frozenset(spec.name for spec in VERB_SPECS)

# verb name -> capabilities() flag, for the capability-gated rows.
CAPABILITY_GATED_VERBS: dict[str, str] = {
    spec.name: spec.capability for spec in VERB_SPECS if spec.capability is not None
}

# Verbs gated by hasattr(source, name) — no capabilities() flag governs them.
HASATTR_GATED_VERBS: tuple[str, ...] = tuple(
    spec.name for spec in VERB_SPECS if spec.capability is None
)

# Capability-gated verbs the bridge client binds dynamically (see VerbSpec).
DYNAMIC_SIGNATURE_VERBS: tuple[str, ...] = tuple(
    spec.name for spec in VERB_SPECS if spec.dynamic_signature
)

# Base globals the runner owns; a data source may not shadow them.
RESERVED_NAMES = frozenset(
    {"answer", "context", "llm_query", "llm_batch", "batch_llm_query", "llm_query_batched"}
)

# Protocol/bridge machinery surface a bridged client also binds as attributes
# (describe is a bridge wire method; name/capabilities/extra_methods are
# DataSource protocol surface).
PROTOCOL_ATTRIBUTE_NAMES = frozenset({"name", "capabilities", "describe", "extra_methods"})

# Every name an extra_methods declaration may NOT use, shared by the registry
# and the bridge so a source config fails identically on every transport.
EXTRA_METHOD_DISALLOWED = CORE_VERB_NAMES | RESERVED_NAMES | PROTOCOL_ATTRIBUTE_NAMES

# Names a bridge service may never advertise as optional methods to a client:
# everything extras validation forbids EXCEPT the hasattr-gated verbs, which
# are legitimately advertised optionals. (Wider than the client's historical
# hand-copied denylist — it also covers the runner-reserved globals.)
UNSAFE_BRIDGE_OPTIONAL_NAMES = EXTRA_METHOD_DISALLOWED - frozenset(HASATTR_GATED_VERBS)

# A default source's verbs are flattened into the sandbox's execution
# globals, where a name like `len` or `print` would shadow the Python
# builtin for every line of generated code.
_BUILTIN_NAMES = frozenset(dir(builtins))


def validate_extra_method_name(extra: object, source_name: str) -> str:
    """Shared extras-name validation (registry + bridge). Returns the name."""
    extra_name = str(extra)
    if not extra_name.isidentifier() or keyword.iskeyword(extra_name):
        raise ValueError(
            f"extra method {extra_name!r} on source {source_name!r} is not a valid "
            "Python identifier — generated code could never call it"
        )
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


@dataclass(frozen=True)
class AccessorManifest:
    """Explicit data-accessor inventory for the count contract's len() check.

    Produced by DataSourceRegistry.accessor_manifest() and surfaced through an
    environment's optional ``accessor_manifest()`` method; replaces the old
    namespace provenance marker + callable-identity sniffing. ``flat`` holds
    the default source's unprefixed verb names; ``namespaced`` holds
    (source_name, verb) pairs. An empty manifest keeps the policy layer's
    static generic-verb fallback.
    """

    flat: frozenset[str] = frozenset()
    namespaced: frozenset[tuple[str, str]] = frozenset()


EMPTY_ACCESSOR_MANIFEST = AccessorManifest()
