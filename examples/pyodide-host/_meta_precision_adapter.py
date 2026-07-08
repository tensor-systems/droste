"""Test fixture, not a reference adapter: proves relay.ts never round-trips
`meta` through a JS value (which would silently round any integer beyond
Number.MAX_SAFE_INTEGER — e.g. a 64-bit record id or timestamp a real
adapter might put in meta). See e2e_test.ts's meta-precision test.

Otherwise identical to pyodide_host_adapter.py; build_db_service returns an
oversized int in meta, and run_for_host_pyodide echoes it back verbatim in
the response so the test can assert on the exact digits in the raw response
text (parsing the response as JSON in the test would itself round the value
back — see the test's own comment).
"""

from __future__ import annotations

from typing import Any

import pyodide_host_adapter
from pyodide_host_adapter import _sql_source

from droste.sources.bridge import DataSourceService

# 2**63 - 1: the largest int64, well beyond Number.MAX_SAFE_INTEGER
# (2**53 - 1). A JSON.parse/JSON.stringify round trip in Deno would corrupt
# this; json.loads/json.dumps in Python (arbitrary-precision ints) will not.
LARGE_INT = 9223372036854775807


def build_db_service(db_path: str, contacts_db_path: str | None = None) -> tuple[DataSourceService, dict[str, Any]]:
    source = _sql_source(db_path)
    service = DataSourceService(source)
    return service, {"large_id": LARGE_INT}


def run_for_host_pyodide(
    request: dict[str, Any],
    host_fetch: Any,
    bridge_call: Any = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resp = pyodide_host_adapter.run_for_host_pyodide(request, host_fetch, bridge_call=bridge_call, meta=meta)
    resp["received_meta_large_id"] = (meta or {}).get("large_id")
    return resp
