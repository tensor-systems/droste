from __future__ import annotations

import json
import os
import select
import shutil
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

from droste import (
    CapabilityAdmission,
    CapabilityBroker,
    CapabilityCall,
    CapabilityCallError,
    CapabilityCheckpoint,
    CapabilityId,
    CapabilityKind,
    CapabilityMetadata,
    CapabilityOutcome,
    CapabilityReservation,
    CapabilityResult,
    ConfiguredSource,
    EnvironmentConfig,
    ProviderRegistration,
    ProviderRegistry,
    ProviderRuntime,
    RLMConfig,
    SideEffect,
    create_environment,
    create_environment_context,
    run_rlm,
)
from droste.protocols.llm_client import TokenUsage
from droste.sources._mcp_stdio_transport import McpStdioSession
from droste.sources.mcp_stdio import (
    MCP_PROTOCOL_VERSION,
    McpConfigurationError,
    McpDescriptorError,
    McpManifestPolicy,
    mcp_tools_to_manifest,
    normalize_mcp_tool_result,
    open_mcp_stdio_source,
)
from droste.sources.sql_local import sqlite_provider
from droste.testing import (
    LifecycleGate,
    MockLLMClient,
    MockResponse,
    MockSubcallClient,
    RecordingAttemptAuthority,
    require_unknown_completion,
    run_while_blocked,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mcp_stdio_fixture.py"
REFERENCE_SERVER = (
    Path(__file__).parent
    / "reference_mcp"
    / "node_modules"
    / "@modelcontextprotocol"
    / "server-filesystem"
    / "dist"
    / "index.js"
)


def _source(
    tmp_path: Path,
    *,
    tools: tuple[str, ...] = ("ReadFile",),
    mode: str = "normal",
    max_result_bytes: int = 262_144,
    close_timeout_ms: int = 2_000,
    startup_timeout_ms: int = 10_000,
    max_frame_bytes: int = 1_048_576,
    max_descriptor_bytes: int = 1_048_576,
    max_stderr_bytes: int = 65_536,
    max_in_flight: int = 64,
) -> ConfiguredSource:
    bindings = {
        "ReadFile": "read_file",
        "content.blocks": "content_blocks",
        "environment": "environment",
        "fail": "fail",
        "slow-read": "slow_read",
        "reserved.params": "reserved_params",
    }
    return ConfiguredSource(
        "remote_docs",
        "reference_filesystem",
        {
            "command": os.path.realpath(sys.executable),
            "args": [str(FIXTURE)],
            "env": {"MCP_TEST_ROOT": str(tmp_path), "MCP_FIXTURE_MODE": mode},
            "cwd": str(tmp_path),
            "allowed_executables": [os.path.realpath(sys.executable)],
            "allowed_tools": list(tools),
            "bindings": {name: bindings[name] for name in tools},
            "effects": {name: "read" for name in tools},
            "budget_classes": {name: "data.read" for name in tools},
            "policy_metadata": {name: {"read_only": True} for name in tools},
            "source_description": "Read-only reference documents.",
            "max_result_bytes": max_result_bytes,
            "max_frame_bytes": max_frame_bytes,
            "max_descriptor_bytes": max_descriptor_bytes,
            "max_stderr_bytes": max_stderr_bytes,
            "max_in_flight": max_in_flight,
            "close_timeout_ms": close_timeout_ms,
            "startup_timeout_ms": startup_timeout_ms,
        },
    )


def _broker(source: ConfiguredSource) -> tuple[ProviderRegistry, CapabilityBroker, object]:
    bound = open_mcp_stdio_source(source)
    registry = ProviderRegistry((bound,))
    broker = CapabilityBroker(registry.capability_registrations())
    return registry, broker, registry.broker_globals(broker)[source.source_id]


def _official_source(tmp_path: Path, node: str) -> ConfiguredSource:
    return ConfiguredSource(
        "reference_docs",
        "reference_filesystem",
        {
            "command": os.path.realpath(node),
            "args": [str(REFERENCE_SERVER), str(tmp_path)],
            "env": {},
            "cwd": str(tmp_path),
            "allowed_executables": [os.path.realpath(node)],
            "allowed_tools": ["read_text_file"],
            "bindings": {"read_text_file": "read_text_file"},
            "effects": {"read_text_file": "read"},
            "budget_classes": {"read_text_file": "data.read"},
            "policy_metadata": {"read_text_file": {"read_only": True}},
            "source_description": "Read-only reference documents.",
        },
    )


class _CountingAttemptAuthority:
    def __init__(self, *, deadline: float | None = None) -> None:
        self.deadline = deadline
        self.active: set[str] = set()
        self.settlements: list[tuple[str, str | None, bool]] = []
        self._lock = threading.Lock()

    def admit(self, call: CapabilityCall) -> CapabilityAdmission:
        with self._lock:
            self.active.add(call.call_id)
        return CapabilityAdmission(
            CapabilityReservation(tokens=0, subcalls=0, wall_ms=1_000),
            self.deadline,
        )

    def checkpoint(
        self, call: CapabilityCall, cumulative: CapabilityCheckpoint
    ) -> CapabilityCheckpoint:
        del call
        return cumulative

    def settle(
        self,
        call: CapabilityCall,
        result: object,
        error: object,
        checkpoint: CapabilityCheckpoint,
        *,
        attempted: bool,
    ) -> CapabilityMetadata:
        del result, checkpoint
        code = getattr(error, "code", None)
        with self._lock:
            self.active.remove(call.call_id)
            self.settlements.append((call.call_id, code, attempted))
        return CapabilityMetadata()


def test_config_is_explicit_and_does_not_accept_ambient_authority(tmp_path: Path) -> None:
    with pytest.raises(McpConfigurationError, match="absolute executable"):
        open_mcp_stdio_source(
            ConfiguredSource(
                "bad",
                "reference_filesystem",
                {
                    "command": "python",
                    "allowed_executables": [os.path.realpath(sys.executable)],
                    "allowed_tools": ["ReadFile"],
                    "bindings": {"ReadFile": "read_file"},
                    "effects": {"ReadFile": "read"},
                    "budget_classes": {"ReadFile": "data.read"},
                },
            )
        )
    relative_allowlist = _source(tmp_path).config_dict()
    relative_allowlist["allowed_executables"] = ["python"]
    with pytest.raises(McpConfigurationError, match="absolute paths"):
        open_mcp_stdio_source(ConfiguredSource("bad", "reference_filesystem", relative_allowlist))
    ambient_cwd = _source(tmp_path).config_dict()
    del ambient_cwd["cwd"]
    with pytest.raises(McpConfigurationError, match="cwd must be an absolute"):
        open_mcp_stdio_source(ConfiguredSource("bad", "reference_filesystem", ambient_cwd))
    effectful = _source(tmp_path).config_dict()
    effectful["effects"] = {"ReadFile": "effectful"}
    with pytest.raises(McpConfigurationError, match="read-only"):
        open_mcp_stdio_source(ConfiguredSource("bad", "reference_filesystem", effectful))
    value = _source(tmp_path, tools=("environment",))
    registry, _, remote = _broker(value)
    try:
        names = remote.environment()["names"]
        assert {"MCP_FIXTURE_MODE", "MCP_TEST_ROOT"}.issubset(names)
        assert "HOME" not in names
        assert "PATH" not in names
    finally:
        registry.close()


def test_paginated_snapshot_preserves_raw_identity_and_ignores_annotations(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    bound = open_mcp_stdio_source(source)
    try:
        manifest = bound.registration.manifest
        operation = manifest.operations[0]
        assert operation.operation_id == "ReadFile"
        assert operation.binding_name == "read_file"
        assert operation.parameters.dialect == "https://json-schema.org/draft/2020-12/schema"
        assert operation.parameters.provenance.endswith(
            f"ReadFile/inputSchema@{MCP_PROTOCOL_VERSION}"
        )
        assert bound.registration.effects["ReadFile"].value == "read"
        assert manifest.to_dict() == json.loads(json.dumps(manifest.to_dict()))
    finally:
        bound.close()


def test_empty_pagination_cursor_is_preserved_as_an_opaque_string(tmp_path: Path) -> None:
    bound = open_mcp_stdio_source(_source(tmp_path, mode="empty-cursor"))
    try:
        assert bound.registration.manifest.operations[0].operation_id == "ReadFile"
    finally:
        bound.close()


def test_manifest_mapping_rejects_invention_and_tasks_but_preserves_dialects(
    tmp_path: Path,
) -> None:
    config = _source(tmp_path).config_dict()
    base = {
        "name": "ReadFile",
        "description": "Read a file.",
        "inputSchema": {"type": "object"},
    }
    policy = McpManifestPolicy(
        allowed_tools=("ReadFile",),
        bindings=config["bindings"],
        budget_classes=config["budget_classes"],
        max_descriptor_bytes=1_048_576,
    )
    first = mcp_tools_to_manifest("reference_filesystem", [base], policy)
    second = mcp_tools_to_manifest("reference_filesystem", [dict(base)], policy)
    assert first.digest == second.digest
    assert first.operations[0].parameters.dialect.endswith("2020-12/schema")

    with pytest.raises(McpDescriptorError, match="will not invent"):
        mcp_tools_to_manifest("reference_filesystem", [{**base, "description": ""}], policy)
    with pytest.raises(McpDescriptorError, match="will not invent"):
        mcp_tools_to_manifest(
            "reference_filesystem",
            [{**base, "description": "", "title": "ReadFile"}],
            policy,
        )
    with pytest.raises(McpDescriptorError, match="task execution"):
        mcp_tools_to_manifest(
            "reference_filesystem",
            [{**base, "execution": {"taskSupport": "required"}}],
            policy,
        )
    draft_seven = mcp_tools_to_manifest(
        "reference_filesystem",
        [
            {
                **base,
                "inputSchema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                },
            }
        ],
        policy,
    )
    assert draft_seven.operations[0].parameters.dialect == "http://json-schema.org/draft-07/schema#"


def test_mcp_and_sqlite_share_generic_registry_bindings_and_prompt(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("brokered MCP fact\n", encoding="utf-8")
    database = tmp_path / "facts.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE facts(value TEXT)")
    connection.execute("INSERT INTO facts VALUES ('sqlite fact')")
    connection.commit()
    connection.close()

    mcp = open_mcp_stdio_source(_source(tmp_path))
    sql = sqlite_provider().bind(ConfiguredSource("db", "sqlite", {"sqlite_path": str(database)}))
    registry = ProviderRegistry((sql, mcp))
    broker = CapabilityBroker(registry.capability_registrations())
    generated = registry.broker_globals(broker)
    try:
        assert generated["remote_docs"].read_file(path="guide.md") == {
            "text": "brokered MCP fact\n"
        }
        assert generated["db"].query("SELECT value FROM facts") == [{"value": "sqlite fact"}]
        assert set(vars(generated["remote_docs"])) == {"read_file"}
        assert set(vars(generated["db"])) == {"query", "get_schema"}
        prompt = registry.prompt_fragment()
        assert "remote_docs.read_file" in prompt
        assert "db.query" in prompt
        assert "MCP" not in prompt
        assert str(FIXTURE) not in prompt
        assert str(tmp_path) not in prompt
    finally:
        registry.close()


def test_model_projection_cannot_observe_transport_choice(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("same logical result", encoding="utf-8")
    source = _source(tmp_path)
    remote_bound = open_mcp_stdio_source(source)
    manifest = remote_bound.registration.manifest

    def bind_in_process(bound_source: ConfiguredSource, context: object = None) -> ProviderRuntime:
        del bound_source, context

        def read_file(execution, *, path: str) -> CapabilityOutcome:
            execution.check()
            assert path == "guide.md"
            return CapabilityOutcome(result={"text": "same logical result"})

        return ProviderRuntime(
            {"ReadFile": read_file},
            source_description="Read-only reference documents.",
        )

    local_bound = ProviderRegistration(
        manifest=manifest,
        effects={"ReadFile": SideEffect.READ},
        binder=bind_in_process,
        policy_metadata={"ReadFile": {"read_only": True}},
    ).bind(ConfiguredSource("remote_docs", "reference_filesystem"))
    remote_registry = ProviderRegistry((remote_bound,))
    local_registry = ProviderRegistry((local_bound,))
    remote_broker = CapabilityBroker(remote_registry.capability_registrations())
    local_broker = CapabilityBroker(local_registry.capability_registrations())
    try:
        assert remote_registry.prompt_fragment() == local_registry.prompt_fragment()
        assert remote_registry.accessor_manifest() == local_registry.accessor_manifest()
        assert remote_broker.describe().to_dict() == local_broker.describe().to_dict()
        remote = remote_registry.broker_globals(remote_broker)["remote_docs"]
        local = local_registry.broker_globals(local_broker)["remote_docs"]
        assert remote.read_file(path="guide.md") == local.read_file(path="guide.md")
        assert remote.read_file.__name__ == local.read_file.__name__ == "read_file"
    finally:
        remote_registry.close()
        local_registry.close()


def test_structured_content_is_preferred_and_content_fallback_is_lossless(
    tmp_path: Path,
) -> None:
    (tmp_path / "note.txt").write_text("structured", encoding="utf-8")
    source = _source(tmp_path, tools=("ReadFile", "content.blocks", "fail"))
    registry, _, remote = _broker(source)
    try:
        assert remote.read_file(path="note.txt") == {"text": "structured"}
        assert remote.content_blocks() == {
            "content": [
                {"type": "text", "text": "alpha"},
                {
                    "type": "resource_link",
                    "uri": "file:///fixture/guide.md",
                    "name": "guide",
                    "mimeType": "text/markdown",
                },
                {"type": "image", "data": "AA==", "mimeType": "image/png"},
            ],
            "losses": [],
        }
        with pytest.raises(CapabilityCallError) as error:
            remote.fail()
        assert error.value.error.code == "mcp.tool_error"
        assert error.value.error.message == "fixture refusal"
    finally:
        registry.close()


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ({}, "requires content"),
        ({"content": [], "isError": "false"}, "must be a boolean"),
        ({"content": [], "structuredContent": None}, "must be an object"),
    ],
)
def test_result_normalization_rejects_invented_or_invalid_protocol_values(
    result: dict[str, object], message: str
) -> None:
    outcome = normalize_mcp_tool_result(
        result=result,
        expects_structured=False,
        max_result_bytes=1024,
        latency_ms=0,
        response_bytes=0,
    )
    assert outcome.error is not None
    assert outcome.error.code == "mcp.invalid_result"
    assert message in outcome.error.message


def test_positional_arguments_fail_before_transport_dispatch(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("unused", encoding="utf-8")
    registry, _, remote = _broker(_source(tmp_path))
    try:
        with pytest.raises(CapabilityCallError) as error:
            remote.read_file("note.txt")
        assert error.value.error.code == "mcp.invalid_arguments"
        assert registry.stats()["remote_docs"]["calls"] == 0
    finally:
        registry.close()


def test_server_argument_names_cannot_collide_with_handler_implementation(
    tmp_path: Path,
) -> None:
    registry, _, remote = _broker(_source(tmp_path, tools=("reserved.params",)))
    try:
        assert remote.reserved_params(execution="remote", _operation="value") == {
            "execution": "remote",
            "_operation": "value",
        }
    finally:
        registry.close()


def test_outgoing_frame_bound_rejects_before_dispatch_without_closing_session(
    tmp_path: Path,
) -> None:
    (tmp_path / "note.txt").write_text("still usable", encoding="utf-8")
    registry, _, remote = _broker(_source(tmp_path, max_frame_bytes=2048))
    try:
        with pytest.raises(CapabilityCallError) as error:
            remote.read_file(path="x" * 4096)
        assert error.value.error.code == "mcp.protocol_error"
        assert registry.stats()["remote_docs"]["calls"] == 0
        assert remote.read_file(path="note.txt") == {"text": "still usable"}
    finally:
        registry.close()


def test_concurrent_calls_route_by_json_rpc_identity(tmp_path: Path) -> None:
    for index in range(20):
        (tmp_path / f"{index}.txt").write_text(f"value-{index}", encoding="utf-8")
    registry, _, remote = _broker(_source(tmp_path))
    values: dict[int, str] = {}
    failures: list[BaseException] = []

    def read(index: int) -> None:
        try:
            values[index] = remote.read_file(path=f"{index}.txt")["text"]
        except BaseException as exc:
            failures.append(exc)

    workers = [threading.Thread(target=read, args=(index,)) for index in range(20)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5)
        assert not failures
        assert values == {index: f"value-{index}" for index in range(20)}
        assert registry.stats()["remote_docs"]["calls"] == 20
    finally:
        registry.close()


def test_result_bounds_and_durable_trace_exclude_protocol_content(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("private payload" * 100, encoding="utf-8")
    source = _source(tmp_path, max_result_bytes=256)
    bound = open_mcp_stdio_source(source)
    registry = ProviderRegistry((bound,))
    durable: list[dict[str, object]] = []
    broker = CapabilityBroker(
        registry.capability_registrations(),
        observer=lambda result: durable.append(result.to_trace_dict()),
    )
    remote = registry.broker_globals(broker)["remote_docs"]
    try:
        with pytest.raises(CapabilityCallError) as error:
            remote.read_file(path="large.txt")
        assert error.value.error.code == "mcp.result_too_large"
        encoded = json.dumps(durable)
        assert "private payload" not in encoded
        assert "large.txt" not in encoded
        assert str(tmp_path) not in encoded
        assert "MCP_TEST_ROOT" not in encoded
    finally:
        registry.close()


def test_mid_call_cancellation_terminates_session_and_settles_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = LifecycleGate()
    real_read = os.read
    marker = b"slow call started"
    observed_stderr = bytearray()

    def observed_read(descriptor: int, size: int) -> bytes:
        chunk = real_read(descriptor, size)
        if threading.current_thread().name == "droste-mcp-stderr":
            observed_stderr.extend(chunk)
            if marker in observed_stderr:
                gate.arrive()
            elif len(observed_stderr) > len(marker):
                del observed_stderr[: -len(marker)]
        return chunk

    monkeypatch.setattr("droste.sources._mcp_stdio_transport.os.read", observed_read)
    source = _source(tmp_path, tools=("slow-read",))
    bound = open_mcp_stdio_source(source)
    registry = ProviderRegistry((bound,))
    authority = RecordingAttemptAuthority(reservation=CapabilityReservation(wall_ms=1_000))
    broker = CapabilityBroker(
        registry.capability_registrations(), run_id="run", attempt_authority=authority
    )
    capability_id = CapabilityId(
        CapabilityKind.DATA,
        "slow-read",
        source_id="remote_docs",
        provider_type="reference_filesystem",
    )
    call = CapabilityCall(capability_id, "slow-call", "run")

    def cancel() -> None:
        assert broker.cancel("slow-call") is True

    try:
        outcome = run_while_blocked(
            lambda: broker.dispatch(call), gate=gate, while_blocked=cancel
        ).require_value()
        assert outcome.status.value == "cancelled"
        assert outcome.error.code == "cancelled"
        assert broker.cancel("slow-call") is False
        settlement = authority.require_single_settlement("slow-call")
        assert settlement.error_code == "cancelled"
        assert settlement.attempted is True
        assert authority.active_calls == frozenset()
    finally:
        registry.close()


def test_max_in_flight_bound_rejects_excess_call_without_replay_hint(tmp_path: Path) -> None:
    bound = open_mcp_stdio_source(
        _source(
            tmp_path,
            tools=("slow-read",),
            close_timeout_ms=50,
            max_in_flight=1,
        )
    )
    registry = ProviderRegistry((bound,))
    broker = CapabilityBroker(registry.capability_registrations(), run_id="run")
    capability_id = CapabilityId(
        CapabilityKind.DATA,
        "slow-read",
        source_id="remote_docs",
        provider_type="reference_filesystem",
    )
    first_outcomes: list[CapabilityResult] = []
    worker = threading.Thread(
        target=lambda: first_outcomes.append(
            broker.dispatch(CapabilityCall(capability_id, "in-flight", "run"))
        )
    )
    worker.start()
    deadline = time.monotonic() + 2
    while bound.runtime.stats()["stderr_bytes"] < 20 and time.monotonic() < deadline:
        time.sleep(0.01)

    excess = broker.dispatch(CapabilityCall(capability_id, "excess", "run"))

    assert excess.error.code == "mcp.transport_error"
    assert excess.error.retryable is False
    assert broker.cancel("in-flight") is True
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert first_outcomes[0].error.code == "cancelled"
    registry.close()


def test_mid_call_deadline_terminates_session_and_settles_once(tmp_path: Path) -> None:
    bound = open_mcp_stdio_source(_source(tmp_path, tools=("slow-read",), close_timeout_ms=50))
    registry = ProviderRegistry((bound,))
    authority = _CountingAttemptAuthority(deadline=time.monotonic() + 0.05)
    broker = CapabilityBroker(
        registry.capability_registrations(), run_id="run", attempt_authority=authority
    )
    call = CapabilityCall(
        CapabilityId(
            CapabilityKind.DATA,
            "slow-read",
            source_id="remote_docs",
            provider_type="reference_filesystem",
        ),
        "deadline-call",
        "run",
    )

    outcome = broker.dispatch(call)

    assert outcome.status.value == "cancelled"
    assert outcome.error.code == "deadline_exceeded"
    assert authority.settlements == [("deadline-call", "deadline_exceeded", True)]
    assert authority.active == set()
    registry.close()


def test_cancellation_interrupts_pipe_filling_write_and_unblocks_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = LifecycleGate()
    first_dispatching = threading.Event()
    request_write_started = threading.Event()
    sibling_waiting = threading.Event()
    real_select = select.select
    real_write = os.write
    real_acquire_write_lock = McpStdioSession._acquire_write_lock

    def observed_write(descriptor, data) -> int:
        written = real_write(descriptor, data)
        if first_dispatching.is_set() and not sibling_waiting.is_set() and written:
            request_write_started.set()
        return written

    def observed_select(readable, writable, exceptional, timeout=None):
        result = real_select(readable, writable, exceptional, timeout)
        if (
            writable
            and not result[1]
            and first_dispatching.is_set()
            and request_write_started.is_set()
            and not sibling_waiting.is_set()
        ):
            gate.arrive()
        return result

    def observed_acquire_write_lock(
        session: McpStdioSession, execution, expires: float | None
    ) -> None:
        if execution is not None and execution.call_id == "blocked-sibling":
            if session._write_lock.acquire(blocking=False):
                session._write_lock.release()
                raise AssertionError("MCP sibling reached an unlocked writer")
            sibling_waiting.set()
        real_acquire_write_lock(session, execution, expires)

    source = _source(tmp_path, mode="stop-reading", close_timeout_ms=50)
    bound = open_mcp_stdio_source(source)
    registry = ProviderRegistry((bound,))
    monkeypatch.setattr("droste.sources._mcp_stdio_transport.os.write", observed_write)
    monkeypatch.setattr("droste.sources._mcp_stdio_transport.select.select", observed_select)
    monkeypatch.setattr(McpStdioSession, "_acquire_write_lock", observed_acquire_write_lock)
    authority = _CountingAttemptAuthority()
    broker = CapabilityBroker(
        registry.capability_registrations(), run_id="run", attempt_authority=authority
    )
    capability_id = CapabilityId(
        CapabilityKind.DATA,
        "ReadFile",
        source_id="remote_docs",
        provider_type="reference_filesystem",
    )
    calls = (
        CapabilityCall(
            capability_id,
            "blocked-write",
            "run",
            kwargs={"path": "x" * 524_288},
        ),
        CapabilityCall(capability_id, "blocked-sibling", "run", kwargs={"path": "small"}),
    )
    outcomes: dict[str, object] = {}
    second = threading.Thread(
        target=lambda: outcomes.setdefault("second", broker.dispatch(calls[1])),
        daemon=True,
    )
    started = time.monotonic()

    def cancel_after_sibling_waits() -> None:
        reached_sibling = False
        try:
            second.start()
            reached_sibling = sibling_waiting.wait(timeout=2)
        finally:
            cancelled = broker.cancel("blocked-write")
        assert reached_sibling
        assert cancelled is True

    def dispatch_first():
        first_dispatching.set()
        return broker.dispatch(calls[0])

    try:
        first_outcome = run_while_blocked(
            dispatch_first,
            gate=gate,
            while_blocked=cancel_after_sibling_waits,
        ).require_value()
    finally:
        if second.ident is not None:
            second.join(timeout=2)
        registry.close()
        if second.is_alive():
            second.join(timeout=2)
        assert not second.is_alive(), "blocked MCP sibling survived runtime close"
    assert not second.is_alive()
    assert time.monotonic() - started < 2
    second_outcome = outcomes["second"]
    assert first_outcome.status.value == "cancelled"
    assert second_outcome.error.code == "mcp.transport_error"
    require_unknown_completion(second_outcome.error, attempts=1)
    assert sorted(call_id for call_id, _, _ in authority.settlements) == [
        "blocked-sibling",
        "blocked-write",
    ]
    assert authority.active == set()


def test_close_escalates_to_process_group_kill_and_remains_idempotent(tmp_path: Path) -> None:
    bound = open_mcp_stdio_source(_source(tmp_path, mode="ignore-close", close_timeout_ms=50))
    registry = ProviderRegistry((bound,))
    started = time.monotonic()
    registry.close()
    registry.close()
    assert time.monotonic() - started < 1


def test_close_reaps_descendants_after_server_parent_exits(tmp_path: Path) -> None:
    bound = open_mcp_stdio_source(_source(tmp_path, mode="orphan-child", close_timeout_ms=50))
    child_pid = int((tmp_path / "child.pid").read_text(encoding="ascii"))
    bound.close()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("MCP process-group descendant survived runtime close")


def test_paginated_discovery_has_one_total_startup_deadline(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        open_mcp_stdio_source(
            _source(
                tmp_path,
                mode="hang-list",
                startup_timeout_ms=100,
                close_timeout_ms=50,
            )
        )
    assert time.monotonic() - started < 1


def test_initialization_and_discovery_share_one_startup_deadline(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        open_mcp_stdio_source(
            _source(
                tmp_path,
                mode="slow-startup",
                startup_timeout_ms=120,
                close_timeout_ms=50,
            )
        )
    assert time.monotonic() - started < 0.5


def test_initialize_timeout_closes_without_forbidden_cancellation_notification(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="timed out"):
        open_mcp_stdio_source(
            _source(
                tmp_path,
                mode="init-timeout",
                startup_timeout_ms=100,
                close_timeout_ms=50,
            )
        )
    assert not (tmp_path / "initialize-cancelled").exists()


@pytest.mark.parametrize(
    ("mode", "configuration", "message"),
    [
        ("oversized-frame", {"max_frame_bytes": 1024}, "frame bound"),
        ("stderr-flood", {"max_stderr_bytes": 1024}, "stderr exceeded"),
    ],
)
def test_transport_enforces_stdout_frame_and_total_stderr_bounds(
    tmp_path: Path,
    mode: str,
    configuration: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        open_mcp_stdio_source(_source(tmp_path, mode=mode, **configuration))


def test_aggregate_descriptor_bound_is_independent_of_frame_bound(tmp_path: Path) -> None:
    with pytest.raises(McpDescriptorError, match="snapshot exceeds"):
        open_mcp_stdio_source(_source(tmp_path, max_frame_bytes=2048, max_descriptor_bytes=1024))


def test_server_request_receives_best_effort_method_not_found(tmp_path: Path) -> None:
    bound = open_mcp_stdio_source(_source(tmp_path, mode="server-request"))
    try:
        response_path = tmp_path / "server-response.json"
        deadline = time.monotonic() + 1
        while not response_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        response = json.loads(response_path.read_text(encoding="utf-8"))
        assert response == {
            "jsonrpc": "2.0",
            "id": "server-1",
            "error": {"code": -32601, "message": "Method not found"},
        }
    finally:
        bound.close()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("old-version", "negotiate"),
        ("cursor-loop", "repeated"),
        ("invalid-json", "invalid JSON"),
        ("partial-eof", "mid-frame"),
        ("unknown-id", "active request"),
        ("result-and-error", "exactly one"),
    ],
)
def test_bootstrap_failure_closes_partial_runtime(tmp_path: Path, mode: str, message: str) -> None:
    with pytest.raises((McpDescriptorError, RuntimeError), match=message):
        open_mcp_stdio_source(_source(tmp_path, mode=mode))


@pytest.mark.skipif(
    not REFERENCE_SERVER.exists(), reason="pinned reference MCP server not installed"
)
def test_pinned_official_filesystem_server_answers_through_generic_broker(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable")
    (tmp_path / "guide.md").write_text("official MCP reference fact\n", encoding="utf-8")
    source = _official_source(tmp_path, node)
    database = tmp_path / "reference.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE facts(value TEXT)")
    connection.execute("INSERT INTO facts VALUES ('official sqlite fact')")
    connection.commit()
    connection.close()
    mcp = open_mcp_stdio_source(source)
    sql = sqlite_provider().bind(ConfiguredSource("db", "sqlite", {"sqlite_path": str(database)}))
    registry = ProviderRegistry((sql, mcp))
    broker = CapabilityBroker(registry.capability_registrations())
    generated = registry.broker_globals(broker)
    remote = generated["reference_docs"]
    try:
        assert remote.read_text_file(path="guide.md") == {
            "content": "official MCP reference fact\n"
        }
        assert generated["db"].query("SELECT value FROM facts") == [
            {"value": "official sqlite fact"}
        ]
        stats = registry.stats()["reference_docs"]
        assert stats["calls"] == 1
        assert stats["latency_ms"] > 0
        assert "write_file" not in vars(remote)
        assert "MCP" not in registry.prompt_fragment()
        assert "reference_docs.read_text_file" in registry.prompt_fragment()
        assert "db.query" in registry.prompt_fragment()
    finally:
        registry.close()


@pytest.mark.skipif(
    not REFERENCE_SERVER.exists(), reason="pinned reference MCP server not installed"
)
def test_rlm_answers_question_through_official_mcp_and_sqlite(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable")
    (tmp_path / "guide.md").write_text("filesystem fact", encoding="utf-8")
    database = tmp_path / "question.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE facts(value TEXT)")
    connection.execute("INSERT INTO facts VALUES ('sqlite fact')")
    connection.commit()
    connection.close()

    registry = ProviderRegistry(
        (
            open_mcp_stdio_source(_official_source(tmp_path, node)),
            sqlite_provider().bind(
                ConfiguredSource("db", "sqlite", {"sqlite_path": str(database)})
            ),
        )
    )
    observed: list[CapabilityResult] = []
    subcalls = MockSubcallClient()
    environment_config = EnvironmentConfig(kind="native")
    execution_context = create_environment_context(environment_config)
    environment = create_environment(
        environment_config,
        context={},
        registry=registry,
        subcalls=subcalls,
        execution_context=execution_context,
        capability_observer=observed.append,
    )
    root = MockLLMClient(
        [
            MockResponse(
                text=(
                    "```python\n"
                    "document = reference_docs.read_text_file(path='guide.md')['content']\n"
                    "row = db.query('SELECT value FROM facts')[0]['value']\n"
                    "answer['content'] = document + ' + ' + row\n"
                    "answer['ready'] = True\n"
                    "```"
                ),
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        ]
    )

    result = run_rlm(
        "Combine the fact in guide.md with the database fact.",
        environment=environment,
        root_llm=root,
        subcalls=subcalls,
        config=RLMConfig(),
        context=execution_context,
    )

    assert result.ready is True
    assert result.answer == "filesystem fact + sqlite fact"
    assert [item.call.capability_id.operation for item in observed] == [
        "read_text_file",
        "query",
    ]
