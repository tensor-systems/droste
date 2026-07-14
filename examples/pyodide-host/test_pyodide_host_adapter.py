"""Fast native (no Pyodide) sanity test for pyodide_host_adapter.py.

BridgedLLMClient._post only branches into `run_sync` when the host_fetch
result has `__await__` — a plain synchronous fake, as used here, exercises
the same code path a real Pyodide interpreter would run, minus the
interpreter itself. The real Deno+Pyodide subprocess path is covered
separately by e2e_test.ts (spawns the actual relay.ts).
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from pyodide_host_adapter import build_db_service, run_for_host_pyodide  # noqa: E402

from droste.sources.bridge import ProviderService  # noqa: E402


def _fake_host_fetch_answering(content: str, captured_headers: dict | None = None):
    def host_fetch(method: str, url: str, headers_json: str, body: str) -> str:
        if captured_headers is not None:
            captured_headers.update(json.loads(headers_json))
        code = f"```python\nanswer['content'] = {content!r}\nanswer['ready'] = True\n```"
        return json.dumps(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": code}],
                    }
                ]
            }
        )

    return host_fetch


class TestPyodideHostAdapter(unittest.TestCase):
    def setUp(self) -> None:
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.remove, self.db_path)
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        conn.executemany("INSERT INTO widgets (name) VALUES (?)", [("gizmo",), ("gadget",)])
        conn.commit()
        conn.close()

    def test_single_interpreter_mode_no_bridge_call(self) -> None:
        request = {"question": "q", "db_path": self.db_path, "root_model": "test-model"}
        resp = run_for_host_pyodide(request, _fake_host_fetch_answering("two widgets"))
        self.assertIsNone(resp["error"])
        self.assertEqual(resp["answer"], "two widgets")
        self.assertFalse(resp["extracted"])
        self.assertIsNone(resp["extract_error"])

    def test_db_service_bridge_mode(self) -> None:
        """build_db_service + a real bridge_call round-trip, exactly like the
        A'-2 split: the "untrusted" side here never opens db_path itself."""
        service, meta = build_db_service(self.db_path)
        self.assertIsInstance(service, ProviderService)
        self.assertEqual(meta, {})
        self.assertEqual(service.describe()["source_id"], "db")
        self.assertEqual(service.describe()["manifest"]["provider_type"], "sqlite")

        def bridge_call(method: str, params_json: str) -> str:
            return service.handle(method, params_json)

        request = {"question": "q", "root_model": "test-model"}
        resp = run_for_host_pyodide(
            request,
            _fake_host_fetch_answering("bridged answer"),
            bridge_call=bridge_call,
            meta=meta,
        )
        self.assertIsNone(resp["error"])
        self.assertEqual(resp["answer"], "bridged answer")

    def test_customer_token_ignored_unless_auth_type_says_so(self) -> None:
        # Regression: a request carrying BOTH api_key and customer_token
        # (e.g. RLM_BRIDGE=legacy passes the full request through) must
        # authenticate with the api_key header when auth_type is "api_key"
        # (or absent) — BridgedLLMClient._auth_headers prefers a customer
        # token (bearer) whenever one is passed, so an unconditional
        # pass-through would silently pick the wrong credential.
        headers: dict = {}
        request = {
            "question": "q",
            "db_path": self.db_path,
            "root_model": "test-model",
            "api_key": "mr_sk_REAL",
            "customer_token": "ct_SHOULD_NOT_BE_USED",
            "auth_type": "api_key",
        }
        run_for_host_pyodide(request, _fake_host_fetch_answering("ok", headers))
        self.assertEqual(headers.get("X-ModelRelay-Api-Key"), "mr_sk_REAL")
        self.assertNotIn("Authorization", headers)

    def test_customer_token_used_when_auth_type_matches(self) -> None:
        headers: dict = {}
        request = {
            "question": "q",
            "db_path": self.db_path,
            "root_model": "test-model",
            "api_key": "mr_sk_UNUSED",
            "customer_token": "ct_REAL",
            "auth_type": "customer_token",
        }
        run_for_host_pyodide(request, _fake_host_fetch_answering("ok", headers))
        self.assertEqual(headers.get("Authorization"), "Bearer ct_REAL")

    def test_serialize_error_on_root_failure(self) -> None:
        def failing_host_fetch(method: str, url: str, headers_json: str, body: str) -> str:
            raise RuntimeError("boom")

        request = {"question": "q", "db_path": self.db_path, "root_model": "test-model"}
        resp = run_for_host_pyodide(request, failing_host_fetch)
        # run_rlm surfaces a JsException-shaped RLMError on a root LLM
        # exception; the crux under test is that it's JSON-serializable, not
        # its exact type/message.
        json.dumps(resp)
        self.assertIsNotNone(resp["error"])


if __name__ == "__main__":
    unittest.main()
