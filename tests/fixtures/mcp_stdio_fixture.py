"""Hermetic MCP 2025-11-25 stdio fixture used by provider tests."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

WRITE_LOCK = threading.Lock()
CANCELLED: dict[int, threading.Event] = {}
ROOT = Path(os.environ.get("MCP_TEST_ROOT", "."))
MODE = os.environ.get("MCP_FIXTURE_MODE", "normal")


def send(value: dict[str, Any]) -> None:
    with WRITE_LOCK:
        sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        value["required"] = required
    return value


TOOLS = [
    {
        "name": "ReadFile",
        "description": "Read one UTF-8 fixture file.",
        "inputSchema": schema({"path": {"type": "string"}}, ["path"]),
        "outputSchema": schema({"text": {"type": "string"}}, ["text"]),
        "annotations": {"readOnlyHint": False},
    },
    {
        "name": "content.blocks",
        "description": "Return bounded content blocks.",
        "inputSchema": schema({}),
    },
    {
        "name": "environment",
        "description": "Report fixture environment names.",
        "inputSchema": schema({}),
        "outputSchema": schema({"names": {"type": "array", "items": {"type": "string"}}}),
    },
    {
        "name": "fail",
        "description": "Return an ordinary MCP tool error.",
        "inputSchema": schema({}),
    },
    {
        "name": "slow-read",
        "description": "Wait until cancelled before reading.",
        "inputSchema": schema({}),
        "outputSchema": schema({"done": {"type": "boolean"}}),
    },
    {
        "name": "reserved.params",
        "description": "Echo names reserved by the host handler implementation.",
        "inputSchema": schema(
            {"execution": {"type": "string"}, "_operation": {"type": "string"}},
            ["execution", "_operation"],
        ),
        "outputSchema": schema(
            {"execution": {"type": "string"}, "_operation": {"type": "string"}},
            ["execution", "_operation"],
        ),
    },
    {
        "name": "write_file",
        "description": "A deliberately disallowed effectful tool.",
        "inputSchema": schema({"path": {"type": "string"}}, ["path"]),
    },
]


def result(request_id: int, value: Any) -> None:
    send({"jsonrpc": "2.0", "id": request_id, "result": value})


def call_tool(request_id: int, name: str, arguments: dict[str, Any]) -> None:
    if name == "slow-read":
        cancelled = CANCELLED.setdefault(request_id, threading.Event())

        def wait() -> None:
            sys.stderr.write("slow call started\n")
            sys.stderr.flush()
            cancelled.wait()

        threading.Thread(target=wait, daemon=True).start()
        return
    if name == "ReadFile":
        path = ROOT / str(arguments.get("path", ""))
        text = path.read_text(encoding="utf-8")
        result(
            request_id,
            {
                "content": [{"type": "text", "text": json.dumps({"text": text})}],
                "structuredContent": {"text": text},
                "isError": False,
                "_meta": {"private": "ignored"},
            },
        )
        return
    if name == "environment":
        result(
            request_id,
            {
                "content": [],
                "structuredContent": {"names": sorted(os.environ)},
                "isError": False,
            },
        )
        return
    if name == "content.blocks":
        result(
            request_id,
            {
                "content": [
                    {"type": "text", "text": "alpha"},
                    {
                        "type": "resource_link",
                        "uri": "file:///fixture/guide.md",
                        "name": "guide",
                        "mimeType": "text/markdown",
                    },
                    {
                        "type": "image",
                        "data": "AA==",
                        "mimeType": "image/png",
                    },
                ],
                "isError": False,
            },
        )
        return
    if name == "reserved.params":
        result(
            request_id,
            {
                "content": [],
                "structuredContent": {
                    "execution": arguments.get("execution"),
                    "_operation": arguments.get("_operation"),
                },
                "isError": False,
            },
        )
        return
    if name == "fail":
        result(
            request_id,
            {"content": [{"type": "text", "text": "fixture refusal"}], "isError": True},
        )
        return
    send(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": "unknown fixture tool"},
        }
    )


def main() -> None:
    if MODE == "ignore-close":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child: subprocess.Popen[bytes] | None = None
    if MODE == "orphan-child":
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import signal,time; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "time.sleep(60)"
                ),
            ],
        )
        (ROOT / "child.pid").write_text(str(child.pid), encoding="ascii")
    sys.stderr.write("fixture ready\n")
    if MODE == "stderr-flood":
        # Deliberately exceed the test bound while remaining below typical pipe
        # and BufferedReader chunk sizes; the transport must enforce promptly.
        sys.stderr.write("x" * 2048)
    sys.stderr.flush()
    pending_server_list: int | None = None
    for line in sys.stdin:
        if MODE == "invalid-json":
            sys.stdout.write("not json\n")
            sys.stdout.flush()
            MODE_LOCAL = "normal"
            del MODE_LOCAL
            return
        if MODE == "partial-eof":
            sys.stdout.write('{"jsonrpc":"2.0"')
            sys.stdout.flush()
            return
        message = json.loads(line)
        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") or {}
        if method == "initialize":
            if MODE == "init-timeout":
                continue
            if MODE == "oversized-frame":
                result(request_id, {"padding": "x" * 2048})
                continue
            if MODE == "slow-startup":
                time.sleep(0.08)
            version = "2025-06-18" if MODE == "old-version" else "2025-11-25"
            response_id = request_id + 1 if MODE == "unknown-id" else request_id
            if MODE == "result-and-error":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": response_id,
                        "result": {},
                        "error": {"code": -32603, "message": "ambiguous"},
                    }
                )
                continue
            result(
                response_id,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "fixture", "version": "1"},
                },
            )
        elif method == "notifications/initialized":
            if MODE == "server-request":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": "server-1",
                        "method": "roots/list",
                        "params": {},
                    }
                )
            else:
                send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
        elif method == "tools/list":
            if MODE == "hang-list":
                time.sleep(10)
            if MODE == "slow-startup":
                time.sleep(0.08)
            cursor = params.get("cursor")
            if (
                MODE == "server-request"
                and cursor is None
                and pending_server_list is None
                and not (ROOT / "server-response.json").exists()
            ):
                # Force discovery to contend with the client response. The
                # response must not be discarded merely because the ordinary
                # request writer was active when the server request arrived.
                pending_server_list = request_id
                continue
            if MODE == "cursor-loop":
                result(request_id, {"tools": TOOLS[:1], "nextCursor": "again"})
            elif MODE == "empty-cursor" and cursor is None:
                result(request_id, {"tools": TOOLS[:3], "nextCursor": ""})
            elif MODE == "empty-cursor" and cursor == "":
                result(request_id, {"tools": TOOLS[3:]})
            elif cursor is None:
                result(request_id, {"tools": TOOLS[:3], "nextCursor": "page-2"})
            elif cursor == "page-2":
                result(request_id, {"tools": TOOLS[3:]})
                if MODE == "stop-reading":
                    sys.stderr.write("fixture stopped reading\n")
                    sys.stderr.flush()
                    time.sleep(60)
            else:
                result(request_id, {"tools": []})
        elif method == "tools/call":
            call_tool(request_id, params.get("name"), params.get("arguments") or {})
        elif method == "notifications/cancelled":
            cancelled_id = params.get("requestId")
            CANCELLED.setdefault(cancelled_id, threading.Event()).set()
            if MODE == "init-timeout" and cancelled_id == 1:
                (ROOT / "initialize-cancelled").write_text("yes", encoding="ascii")
        elif MODE == "server-request" and method is None and request_id == "server-1":
            (ROOT / "server-response.json").write_text(
                json.dumps(message, separators=(",", ":")), encoding="utf-8"
            )
            if pending_server_list is not None:
                result(
                    pending_server_list,
                    {"tools": TOOLS[:3], "nextCursor": "page-2"},
                )
                pending_server_list = None
        elif request_id is not None:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "method not found"},
                }
            )
    if MODE == "ignore-close":
        while True:
            time.sleep(60)


if __name__ == "__main__":
    main()
