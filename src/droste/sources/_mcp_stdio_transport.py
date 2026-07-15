"""Bounded MCP JSON-RPC over a host-launched local stdio process.

This is intentionally only a transport shell.  Provider descriptors and result
normalization live in :mod:`droste.sources.mcp_stdio` so they remain pure and
testable without a process.
"""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from time import monotonic, sleep
from typing import Any

from ..capabilities import (
    CapabilityCancelled,
    CapabilityDeadlineExceeded,
    CapabilityExecutionContext,
)

MCP_PROTOCOL_VERSION = "2025-11-25"


def _client_version() -> str:
    try:
        return version("droste")
    except PackageNotFoundError:
        return "0+unknown"


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _strict_json_loads(value: str) -> Any:
    return json.loads(
        value,
        object_pairs_hook=_json_object,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"invalid JSON number: {constant}")
        ),
    )


class McpTransportError(RuntimeError):
    """The local MCP process or JSON-RPC stream failed."""


class McpProtocolError(RuntimeError):
    """The peer violated the supported MCP protocol subset."""


class McpRemoteError(RuntimeError):
    """A JSON-RPC request returned an error response."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.remote_message = message
        super().__init__(f"MCP JSON-RPC error {code}: {message}")


@dataclass(slots=True)
class _Pending:
    ready: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None
    failure: BaseException | None = None


class McpStdioSession:
    """One initialized, lifecycle-owned MCP stdio session.

    The reader owns framing.  Callers may issue concurrent requests; request
    identity and writes are serialized, while each caller waits on its own
    response and continues polling the broker execution context for
    cancellation/deadline facts.
    """

    def __init__(
        self,
        *,
        command: str,
        args: tuple[str, ...],
        env: dict[str, str],
        cwd: str,
        startup_timeout_ms: int,
        close_timeout_ms: int,
        max_frame_bytes: int,
        max_stderr_bytes: int,
        max_tool_pages: int,
        max_tools: int,
        max_in_flight: int,
    ) -> None:
        self._close_timeout = close_timeout_ms / 1000
        self._startup_expires = monotonic() + startup_timeout_ms / 1000
        self._max_frame_bytes = max_frame_bytes
        self._max_stderr_bytes = max_stderr_bytes
        self._max_tool_pages = max_tool_pages
        self._max_tools = max_tools
        self._max_in_flight = max_in_flight
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._next_id = 1
        self._closed = False
        self._close_complete = threading.Event()
        self._close_failure: BaseException | None = None
        self._failure: BaseException | None = None
        self._stderr_bytes = 0
        self._calls = 0
        self._failures = 0
        self._latency_ms = 0.0
        self._bytes_received = 0
        try:
            self._process = subprocess.Popen(
                (command, *args),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            raise McpTransportError(f"could not launch configured MCP executable: {exc}") from exc
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        os.set_blocking(self._process.stdin.fileno(), False)
        self._reader = threading.Thread(target=self._read_stdout, name="droste-mcp-stdout")
        self._stderr_reader = threading.Thread(target=self._read_stderr, name="droste-mcp-stderr")
        self._reader.start()
        self._stderr_reader.start()
        try:
            remaining_ms = self._startup_remaining_ms()
            initialized = self._request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "droste", "version": _client_version()},
                },
                timeout_ms=remaining_ms,
                cancellable=False,
            )
            self._validate_initialized(initialized)
            self._notify("notifications/initialized", {}, expires=self._startup_expires)
        except BaseException:
            self.close()
            raise

    def _validate_initialized(self, result: Any) -> None:
        if not isinstance(result, dict):
            raise McpProtocolError("MCP initialize result must be an object")
        if result.get("protocolVersion") != MCP_PROTOCOL_VERSION:
            raise McpProtocolError(
                "MCP server did not negotiate the supported protocol version "
                f"{MCP_PROTOCOL_VERSION}"
            )
        capabilities = result.get("capabilities")
        if not isinstance(capabilities, dict) or not isinstance(capabilities.get("tools"), dict):
            raise McpProtocolError("MCP server must advertise the tools capability")

    def list_tools(self) -> tuple[dict[str, Any], ...]:
        """Read every tools/list page once into one frozen caller snapshot."""

        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        expires = self._startup_expires
        for _ in range(self._max_tool_pages):
            params = {} if cursor is None else {"cursor": cursor}
            remaining_ms = max(1, int((expires - monotonic()) * 1000))
            if monotonic() >= expires:
                self.close()
                raise McpTransportError("MCP tools/list discovery timed out")
            result = self._request("tools/list", params, timeout_ms=remaining_ms)
            if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
                raise McpProtocolError("MCP tools/list result requires a tools array")
            for raw_tool in result["tools"]:
                if not isinstance(raw_tool, dict):
                    raise McpProtocolError("MCP tools/list entries must be objects")
                tools.append(raw_tool)
                if len(tools) > self._max_tools:
                    raise McpProtocolError("MCP tools/list exceeded the configured tool bound")
            raw_cursor = result.get("nextCursor")
            if raw_cursor is None:
                return tuple(tools)
            if not isinstance(raw_cursor, str):
                raise McpProtocolError("MCP tools/list nextCursor must be a string")
            if raw_cursor in seen_cursors:
                raise McpProtocolError("MCP tools/list repeated a pagination cursor")
            seen_cursors.add(raw_cursor)
            cursor = raw_cursor
        raise McpProtocolError("MCP tools/list exceeded the configured page bound")

    def call_tool(
        self,
        execution: CapabilityExecutionContext,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], float, int]:
        started = monotonic()
        result, response_bytes = self._request_with_size(
            "tools/call",
            {"name": name, "arguments": arguments},
            execution=execution,
        )
        elapsed_ms = (monotonic() - started) * 1000
        with self._state_lock:
            self._calls += 1
            self._latency_ms += elapsed_ms
        if not isinstance(result, dict):
            raise McpProtocolError("MCP tools/call result must be an object")
        return result, elapsed_ms, response_bytes

    def stats(self) -> dict[str, int | float]:
        with self._state_lock:
            return {
                "calls": self._calls,
                "failures": self._failures,
                "latency_ms": round(self._latency_ms, 3),
                "bytes_received": self._bytes_received,
                "stderr_bytes": self._stderr_bytes,
            }

    def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        execution: CapabilityExecutionContext | None = None,
        timeout_ms: int | None = None,
        cancellable: bool = True,
    ) -> Any:
        result, _ = self._request_with_size(
            method,
            params,
            execution=execution,
            timeout_ms=timeout_ms,
            cancellable=cancellable,
        )
        return result

    def _request_with_size(
        self,
        method: str,
        params: dict[str, Any],
        *,
        execution: CapabilityExecutionContext | None = None,
        timeout_ms: int | None = None,
        cancellable: bool = True,
    ) -> tuple[Any, int]:
        with self._state_lock:
            self._raise_if_unavailable_locked()
            if len(self._pending) >= self._max_in_flight:
                raise McpTransportError("MCP session reached its in-flight request bound")
            request_id = self._next_id
            self._next_id += 1
            pending = _Pending()
            self._pending[request_id] = pending
        try:
            expires = None if timeout_ms is None else monotonic() + timeout_ms / 1000
            try:
                self._write(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": method,
                        "params": params,
                    },
                    execution=execution,
                    expires=expires,
                )
            except (CapabilityCancelled, CapabilityDeadlineExceeded, McpTransportError):
                self.close()
                raise
            while not pending.ready.wait(0.01):
                if execution is not None:
                    try:
                        execution.check()
                    except BaseException:
                        self._cancel_and_close(request_id, "broker cancellation")
                        raise
                if expires is not None and monotonic() >= expires:
                    if cancellable:
                        self._cancel_and_close(request_id, "client timeout")
                    else:
                        self.close()
                    raise McpTransportError(f"MCP {method} request timed out")
            if pending.failure is not None:
                raise pending.failure
            response = pending.response
            if not isinstance(response, dict):
                raise McpProtocolError("MCP request completed without a response object")
            if "error" in response:
                error = response["error"]
                if not isinstance(error, dict):
                    raise McpProtocolError("MCP JSON-RPC error must be an object")
                code = error.get("code")
                message = error.get("message")
                if (
                    isinstance(code, bool)
                    or not isinstance(code, int)
                    or not isinstance(message, str)
                ):
                    raise McpProtocolError("MCP JSON-RPC error requires integer code and message")
                raise McpRemoteError(code, self._bounded_message(message))
            if "result" not in response:
                raise McpProtocolError("MCP JSON-RPC response requires result or error")
            encoded_size = len(
                json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            return response["result"], encoded_size
        except BaseException:
            with self._state_lock:
                self._failures += 1
            raise
        finally:
            with self._state_lock:
                self._pending.pop(request_id, None)

    @staticmethod
    def _bounded_message(message: str) -> str:
        encoded = message.encode("utf-8")
        if len(encoded) <= 1024:
            return message
        return encoded[:1024].decode("utf-8", errors="ignore") + "…"

    def _startup_remaining_ms(self) -> int:
        remaining = self._startup_expires - monotonic()
        if remaining <= 0:
            raise McpTransportError("MCP acquisition timed out")
        return max(1, int(remaining * 1000))

    def _notify(
        self,
        method: str,
        params: dict[str, Any],
        *,
        expires: float | None = None,
    ) -> None:
        self._write(
            {"jsonrpc": "2.0", "method": method, "params": params},
            expires=expires if expires is not None else monotonic() + self._close_timeout,
        )

    def _cancel_and_close(self, request_id: int, reason: str) -> None:
        try:
            self._notify("notifications/cancelled", {"requestId": request_id, "reason": reason})
        except Exception:
            pass
        self.close()

    def _write(
        self,
        message: dict[str, Any],
        *,
        execution: CapabilityExecutionContext | None = None,
        expires: float | None = None,
        wait_for_lock: bool = True,
    ) -> None:
        try:
            encoded = json.dumps(
                message, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise McpProtocolError(f"MCP request is not finite JSON: {exc}") from exc
        if len(encoded) > self._max_frame_bytes:
            raise McpProtocolError("MCP request exceeds the configured frame bound")
        if wait_for_lock:
            self._acquire_write_lock(execution, expires)
        elif not self._write_lock.acquire(blocking=False):
            raise McpTransportError("MCP standard-input writer is busy")
        try:
            with self._state_lock:
                self._raise_if_unavailable_locked()
            assert self._process.stdin is not None
            descriptor = self._process.stdin.fileno()
            frame = memoryview(encoded + b"\n")
            written = 0
            while written < len(frame):
                if execution is not None:
                    execution.check()
                remaining = None if expires is None else expires - monotonic()
                if remaining is not None and remaining <= 0:
                    raise McpTransportError("MCP standard-input write timed out")
                timeout = 0.01 if remaining is None else min(0.01, remaining)
                try:
                    _, writable, _ = select.select([], [descriptor], [], timeout)
                except (OSError, ValueError) as exc:
                    failure = McpTransportError("MCP process closed its standard input")
                    self._fail(failure)
                    raise failure from exc
                if not writable:
                    continue
                with self._state_lock:
                    self._raise_if_unavailable_locked()
                try:
                    count = os.write(descriptor, frame[written:])
                    if count == 0:
                        raise OSError("zero-byte MCP standard-input write")
                    written += count
                except BlockingIOError:
                    continue
                except OSError as exc:
                    failure = McpTransportError("MCP process closed its standard input")
                    self._fail(failure)
                    raise failure from exc
        finally:
            self._write_lock.release()

    def _acquire_write_lock(
        self,
        execution: CapabilityExecutionContext | None,
        expires: float | None,
    ) -> None:
        while not self._write_lock.acquire(timeout=0.01):
            if execution is not None:
                execution.check()
            if expires is not None and monotonic() >= expires:
                raise McpTransportError("MCP standard-input write timed out")

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        try:
            while True:
                line = self._process.stdout.readline(self._max_frame_bytes + 2)
                if not line:
                    if not self._closed:
                        self._fail(McpTransportError("MCP process closed standard output"))
                    return
                if len(line) > self._max_frame_bytes + 1:
                    self._fail(McpProtocolError("MCP response exceeds the configured frame bound"))
                    return
                if not line.endswith(b"\n"):
                    self._fail(McpProtocolError("MCP process closed standard output mid-frame"))
                    return
                with self._state_lock:
                    self._bytes_received += len(line)
                self._accept_line(line[:-1])
        except BaseException as exc:
            self._fail(McpTransportError(f"MCP stdout reader failed: {type(exc).__name__}"))

    def _accept_line(self, line: bytes) -> None:
        try:
            decoded = line.decode("utf-8", errors="strict")
            message = _strict_json_loads(decoded)
        except (UnicodeDecodeError, ValueError) as exc:
            self._fail(McpProtocolError(f"MCP stdout contained invalid JSON: {type(exc).__name__}"))
            return
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            self._fail(McpProtocolError("MCP stdout message must be a JSON-RPC 2.0 object"))
            return
        request_id = message.get("id")
        method = message.get("method")
        if method is not None and request_id is not None:
            self._respond_method_not_found(request_id)
            return
        if method is not None:
            # Notifications are advisory here. tools/list is frozen for the run;
            # progress units are not Droste budget units and cannot mutate its ledger.
            return
        if isinstance(request_id, bool) or not isinstance(request_id, int):
            self._fail(McpProtocolError("MCP response id must match an integer client request"))
            return
        has_result = "result" in message
        has_error = "error" in message
        if has_result == has_error:
            self._fail(
                McpProtocolError("MCP JSON-RPC response requires exactly one of result or error")
            )
            return
        with self._state_lock:
            pending = self._pending.get(request_id)
            if pending is None:
                failure = McpProtocolError("MCP response id does not match an active request")
            elif pending.ready.is_set():
                failure = McpProtocolError("MCP request received more than one response")
            else:
                pending.response = message
                pending.ready.set()
                return
        self._fail(failure)

    def _respond_method_not_found(self, request_id: Any) -> None:
        try:
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "Method not found"},
                },
                expires=monotonic() + min(self._close_timeout, 0.01),
                wait_for_lock=False,
            )
        except Exception:
            # The reader's transport failure path will wake active callers.
            pass

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        descriptor = self._process.stderr.fileno()
        while True:
            try:
                # BufferedReader.read(size) may wait for all ``size`` bytes on
                # a live pipe, delaying enforcement of a smaller total bound.
                chunk = os.read(descriptor, 4096)
            except OSError:
                return
            if not chunk:
                return
            with self._state_lock:
                self._stderr_bytes += len(chunk)
                stderr_exceeded = self._stderr_bytes > self._max_stderr_bytes
            if stderr_exceeded:
                self._fail(McpProtocolError("MCP stderr exceeded the configured byte bound"))
                return

    def _raise_if_unavailable_locked(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise McpTransportError("MCP stdio session is closed")

    def _fail(self, failure: BaseException) -> None:
        with self._state_lock:
            first_failure = self._failure is None and not self._closed
            if first_failure:
                self._failure = failure
            pending = tuple(self._pending.values())
            stored = self._failure or failure
            for item in pending:
                item.failure = stored
                item.ready.set()
        if first_failure:
            threading.Thread(target=self.close, name="droste-mcp-failure-close").start()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                first = False
            else:
                first = True
                self._closed = True
                pending = tuple(self._pending.values())
                for item in pending:
                    if item.failure is None:
                        item.failure = McpTransportError("MCP stdio session closed")
                    item.ready.set()
        if not first:
            self._close_complete.wait()
            if self._close_failure is not None:
                raise self._close_failure
            return
        try:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.close()
            except OSError:
                pass
            try:
                self._process.wait(timeout=self._close_timeout)
            except subprocess.TimeoutExpired:
                self._signal_process_group(signal.SIGTERM)
                try:
                    self._process.wait(timeout=self._close_timeout)
                except subprocess.TimeoutExpired:
                    self._signal_process_group(signal.SIGKILL)
                    self._process.wait(timeout=self._close_timeout)
            self._reap_process_group_descendants()
            self._reader.join(timeout=self._close_timeout)
            self._stderr_reader.join(timeout=self._close_timeout)
        except BaseException as exc:
            self._close_failure = exc
            raise
        finally:
            self._close_complete.set()

    def _signal_process_group(self, sig: signal.Signals) -> None:
        try:
            os.killpg(self._process.pid, sig)
        except ProcessLookupError:
            return

    def _process_group_exists(self) -> bool:
        try:
            os.killpg(self._process.pid, 0)
        except ProcessLookupError:
            return False
        return True

    def _reap_process_group_descendants(self) -> None:
        """Terminate children that kept the server's stdout/stderr pipes alive."""

        if not self._process_group_exists():
            return
        self._signal_process_group(signal.SIGTERM)
        expires = monotonic() + self._close_timeout
        while self._process_group_exists() and monotonic() < expires:
            sleep(min(0.01, self._close_timeout))
        if self._process_group_exists():
            self._signal_process_group(signal.SIGKILL)


__all__ = [
    "MCP_PROTOCOL_VERSION",
    "McpProtocolError",
    "McpRemoteError",
    "McpStdioSession",
    "McpTransportError",
]
