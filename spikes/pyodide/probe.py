"""Shared fidelity routine — run identically under native CPython and Pyodide.

Compares what the RLM data layer (MessageDatabase, verbatim) returns from each
runtime against the same SQLite file, so any Pyodide-vs-native drift (SQLite
version skew, datetime/NULL/blob serialization) shows up as a digest mismatch.
No message content is emitted — only counts, column names, and a set-digest.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _digest(rows: list[dict[str, Any]]) -> str:
    # Set-equality digest: sort serialized rows so unordered queries don't
    # cause false mismatches, while any value/type difference still does.
    serialized = sorted(
        json.dumps(r, sort_keys=True, default=str, ensure_ascii=True) for r in rows
    )
    blob = "\n".join(serialized).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def run(db_path: str, contacts_path: str | None) -> dict[str, Any]:
    from rcl_rlm.message_database import MessageDatabase

    db = MessageDatabase(db_path=db_path, contacts_db_path=contacts_path)

    out: dict[str, Any] = {}
    import sqlite3 as _sqlite3

    out["sqlite_version"] = _sqlite3.sqlite_version

    count = db.query("SELECT COUNT(*) AS n FROM messages")
    out["message_count"] = count[0]["n"]

    # get_messages exercises the messages_with_contacts view + serialization.
    rows = db.get_messages(limit=500)
    out["get_messages_rows"] = len(rows)
    out["get_messages_columns"] = sorted(rows[0].keys()) if rows else []
    out["get_messages_digest"] = _digest(rows)

    # A raw query() through the contacts view (the LLM-authored-SQL path).
    view_rows = db.query(
        "SELECT * FROM messages_with_contacts ORDER BY ROWID LIMIT 300"
    )
    out["view_rows"] = len(view_rows)
    out["view_digest"] = _digest(view_rows)

    return out
