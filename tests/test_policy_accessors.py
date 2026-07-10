"""Count-contract enforcement over dynamic accessor names (#10).

The len()-over-accessor check must enforce against whatever verbs the
environment actually binds — including host-declared extras — not a
hardcoded list that went stale the day domain verbs moved out of the
engine (codex review on the #10 domain-blind change).
"""

from __future__ import annotations

from droste.policy import PolicyHints, contract_violations

COUNT = PolicyHints(count=True)


def test_len_over_declared_extra_is_rejected() -> None:
    # The aggregate alone satisfies uses_sql_aggregate; the len() over a
    # host extra must still trip the contract.
    code = 'query("SELECT COUNT(*) FROM t")\nprint(len(get_messages()))'
    violations = contract_violations(code, COUNT, data_accessors=("get_messages", "search"))
    assert violations, "len(get_messages()) must violate the count contract"
    assert "len()" in violations[0]


def test_len_over_namespaced_accessor_is_rejected() -> None:
    code = 'query("SELECT COUNT(*) FROM t")\nprint(len(db.search("x")))'
    violations = contract_violations(code, COUNT, namespaced_accessors=(("db", "search"),))
    assert violations


def test_len_over_plain_variable_is_fine() -> None:
    # len() over a local variable (or a non-accessor call) is legitimate
    # Python, not contract circumvention.
    code = 'rows = query("SELECT COUNT(*) FROM t")\nprint(len(rows))'
    assert contract_violations(code, COUNT, data_accessors=("get_messages",)) == []


def test_len_over_arbitrary_receiver_is_not_flagged() -> None:
    # A source exposing a verb named `get` must not make ordinary dict code
    # trip the contract (codex review): only the source's OWN namespace
    # qualifies, and unqualified matching applies only to flattened verbs.
    code = 'query("SELECT COUNT(*) FROM t")\nprint(len(row.get("items", [])))'
    assert (
        contract_violations(
            code,
            COUNT,
            data_accessors=("search",),
            namespaced_accessors=(("db", "get"),),
        )
        == []
    )
    # But the same verb under its real namespace still trips.
    code2 = 'query("SELECT COUNT(*) FROM t")\nprint(len(db.get("id1")))'
    assert contract_violations(code2, COUNT, namespaced_accessors=(("db", "get"),))


def test_static_fallback_when_no_accessors_supplied() -> None:
    # Callers that pass no accessor names keep the historical generic check.
    code = 'query("SELECT COUNT(*) FROM t")\nprint(len(search("x")))'
    assert contract_violations(code, COUNT)
