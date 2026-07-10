"""The protocols-level verb table (#31): registry binding, bridge service
allowlist, bridge client binding, and extras validation are all folds over
VERB_SPECS — adding a verb is one row, and the transports cannot drift."""

from __future__ import annotations

from droste.protocols.verbs import (
    CAPABILITY_GATED_VERBS,
    CORE_VERB_NAMES,
    DYNAMIC_SIGNATURE_VERBS,
    EXTRA_METHOD_DISALLOWED,
    HASATTR_GATED_VERBS,
    RESERVED_NAMES,
    UNSAFE_BRIDGE_OPTIONAL_NAMES,
    VERB_SPECS,
)
from droste.registry import DataSourceRegistry
from droste.sources.bridge import DataSourceService
from droste.testing import MockDataSource


def test_table_partitions_core_verbs_by_gate() -> None:
    assert set(CAPABILITY_GATED_VERBS) | set(HASATTR_GATED_VERBS) == CORE_VERB_NAMES
    assert not set(CAPABILITY_GATED_VERBS) & set(HASATTR_GATED_VERBS)
    assert len({spec.name for spec in VERB_SPECS}) == len(VERB_SPECS)


def test_dynamic_signature_verbs_are_capability_gated() -> None:
    # The bridge client resolves each dynamic verb's gate via
    # CAPABILITY_GATED_VERBS[name]; a hasattr-gated row marked dynamic would
    # KeyError at client construction.
    assert set(DYNAMIC_SIGNATURE_VERBS) <= set(CAPABILITY_GATED_VERBS)


def test_extras_denylist_covers_the_whole_vocabulary() -> None:
    assert CORE_VERB_NAMES <= EXTRA_METHOD_DISALLOWED
    assert RESERVED_NAMES <= EXTRA_METHOD_DISALLOWED
    # The client-side denylist must not reject legitimately advertised
    # hasattr-gated optionals, but must cover reserved globals (the drift
    # the old hand-copied inline list had).
    assert not set(HASATTR_GATED_VERBS) & UNSAFE_BRIDGE_OPTIONAL_NAMES
    assert RESERVED_NAMES <= UNSAFE_BRIDGE_OPTIONAL_NAMES


def test_registry_and_bridge_service_expose_the_same_verbs() -> None:
    # Transport parity by construction: the verbs the registry binds into a
    # sandbox namespace are exactly the methods the bridge service reports
    # (enabled core verbs + implemented optionals) for the same source.
    class WithExtra(MockDataSource):
        extra_methods = ("get_threads",)

        def get_threads(self):
            return []

    source = WithExtra()
    registry_verbs = set(vars(DataSourceRegistry([source]).globals()["mock"]))

    described = DataSourceService(source).describe()
    service_verbs = {
        name for name, cap in CAPABILITY_GATED_VERBS.items() if described["capabilities"].get(cap)
    } | set(described["optional_methods"])

    assert registry_verbs == service_verbs
