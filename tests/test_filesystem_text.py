"""First-party filesystem/text provider conformance and security tests."""

from __future__ import annotations

import base64
import json
import os
import sqlite3
from dataclasses import FrozenInstanceError
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

import pytest

from droste import (
    CapabilityBroker,
    CapabilityCall,
    CapabilityCallError,
    ConfiguredSource,
    EnvironmentConfig,
    ProviderCatalog,
    SideEffect,
    create_environment,
    create_environment_context,
)
from droste.sources.bridge import BridgeProvider, ProviderService
from droste.sources.filesystem_text import (
    FILESYSTEM_TEXT_PROVIDER_MANIFEST,
    FilesystemTextConfig,
    filesystem_text_provider,
)
from droste.sources.sql_local import sqlite_provider
from droste.testing import MockSubcallClient


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "nested").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "README.md").write_text(
        "# Guide\n\nWelcome to Droste.\n\n## Details\n\nNested facts.\n", encoding="utf-8"
    )
    (root / "notes.txt").write_text("alpha\nbeta alpha\ngamma\n", encoding="utf-8")
    (root / "nested" / "code.py").write_text("print('alpha')\n", encoding="utf-8")
    (root / ".git" / "config").write_text("secret\n", encoding="utf-8")
    (root / "binary.bin").write_bytes(b"text\x00binary")
    return root


def _registry(root: Path, *, config: dict | None = None, default: bool = False):
    value = {"root": str(root), **(config or {})}
    return ProviderCatalog((filesystem_text_provider(),)).bind(
        (ConfiguredSource("docs", "filesystem_text", value),),
        default_source_id="docs" if default else None,
    )


def _globals(root: Path, *, config: dict | None = None):
    registry = _registry(root, config=config)
    broker = CapabilityBroker(registry.capability_registrations())
    return registry, broker, registry.broker_globals(broker)["docs"]


def _error(call) -> CapabilityCallError:
    with pytest.raises(CapabilityCallError) as exc_info:
        call()
    return exc_info.value


def _tamper_cursor(cursor: str, position: object) -> str:
    padding = "=" * (-len(cursor) % 4)
    payload = json.loads(base64.urlsafe_b64decode(cursor + padding))
    payload["position"] = position
    return (
        base64.urlsafe_b64encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        .decode()
        .rstrip("=")
    )


class _DuplexSession:
    """One-slot pull pump that requests cancellation after traversal begins."""

    def __init__(self, service: ProviderService, method: str, payload: str) -> None:
        self._frames: Queue[str | BaseException] = Queue(maxsize=1)
        self._acks: Queue[str | BaseException] = Queue(maxsize=1)
        self._closed = Event()
        self._receives = 0
        self._call_id = json.loads(payload)["execution"]["call_id"]

        def emit(frame: str) -> str:
            self._frames.put(frame, timeout=2)
            ack = self._acks.get(timeout=2)
            if isinstance(ack, BaseException):
                raise ack
            return ack

        def run() -> None:
            try:
                response = service.handle_duplex(method, payload, emit)
                envelope = json.loads(response)
                if not envelope["ok"]:
                    self._frames.put(RuntimeError(envelope["error"]["message"]), timeout=2)
            except BaseException as exc:
                if not self._closed.is_set():
                    self._frames.put(exc, timeout=2)

        self._worker = Thread(target=run, daemon=True)
        self._worker.start()

    def receive(self) -> str:
        self._receives += 1
        try:
            frame = self._frames.get(timeout=2)
        except Empty as exc:
            raise TimeoutError("provider produced no duplex frame") from exc
        if isinstance(frame, BaseException):
            raise frame
        return frame

    def send(self, ack: str) -> None:
        self._acks.put(ack, timeout=2)

    def cancellation_requested(self, call_id: str) -> bool:
        return call_id == self._call_id and self._receives >= 3

    def close(self) -> None:
        self._closed.set()
        try:
            self._acks.put_nowait(ConnectionError("session closed"))
        except Exception:
            pass
        self._worker.join(timeout=2)


def test_config_is_frozen_strict_and_source_agnostic(tmp_path: Path) -> None:
    config = FilesystemTextConfig(
        root=str(_tree(tmp_path)), include=("**/*.md",), enrichers=("markdown",)
    )
    assert config.include == ("**/*.md",)
    with pytest.raises(FrozenInstanceError):
        config.root = "/elsewhere"  # type: ignore[misc]
    with pytest.raises(ValueError, match="absolute"):
        FilesystemTextConfig(root="relative")
    with pytest.raises(ValueError, match="unknown"):
        FilesystemTextConfig.from_mapping({"root": config.root, "sql": True})
    with pytest.raises(ValueError, match="complete POSIX segment"):
        FilesystemTextConfig(root=config.root, include=("docs/**.md",))


def test_root_must_be_a_real_directory_not_a_symlink(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    alias = tmp_path / "docs-link"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink directory"):
        _registry(alias)


def test_manifest_drives_bindings_accessors_prompt_and_broker(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    registry = _registry(root, default=True)
    broker = CapabilityBroker(registry.capability_registrations())
    globals_ = registry.broker_globals(broker)
    names = tuple(
        operation.binding_name for operation in FILESYSTEM_TEXT_PROVIDER_MANIFEST.operations
    )

    assert tuple(vars(globals_["docs"])) == names
    assert registry.accessor_manifest().namespaced == frozenset(("docs", name) for name in names)
    assert registry.accessor_manifest().flat == frozenset(names)
    prompt = registry.prompt_fragment()
    for operation in FILESYSTEM_TEXT_PROVIDER_MANIFEST.operations:
        assert f"docs.{operation.binding_name}(" in prompt
        assert operation.description in prompt
    assert str(root) not in prompt
    assert "SQL" not in prompt
    assert all(
        item.descriptor.operation is operation
        for item, operation in zip(
            registry.capability_registrations(), FILESYSTEM_TEXT_PROVIDER_MANIFEST.operations
        )
    )
    assert all(item.side_effect is SideEffect.READ for item in broker.describe().descriptors)


def test_list_is_deterministic_filtered_bounded_and_cursor_revalidated(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _, _, docs = _globals(root, config={"include": ["**/*.md", "**/*.txt", "**/*.py"]})

    first = docs.list_files(limit=2)
    assert [item["path"] for item in first["items"]] == ["README.md", "nested/code.py"]
    assert first["next_cursor"]
    second = docs.list_files(limit=2, cursor=first["next_cursor"])
    assert [item["path"] for item in second["items"]] == ["notes.txt"]
    assert second["next_cursor"] is None
    assert all(item["evidence"]["source_id"] == "docs" for item in first["items"])

    (root / "added.txt").write_text("new", encoding="utf-8")
    error = _error(lambda: docs.list_files(cursor=first["next_cursor"], limit=1))
    assert error.error.code == "filesystem.changed"
    assert error.error.retryable is True


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../outside", "nested/../README.md", "./README.md", "a//b", "a/"],
)
def test_traversal_paths_fail_before_io(tmp_path: Path, path: str) -> None:
    _, _, docs = _globals(_tree(tmp_path))
    error = _error(lambda: docs.read(path))
    assert error.error.code == "filesystem.invalid_path"


def test_intermediate_and_final_symlinks_never_escape(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("OUTSIDE", encoding="utf-8")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    (root / "secret-link.txt").symlink_to(outside / "secret.txt")
    _, _, docs = _globals(root, config={"include": ["**/*"]})

    for path in ("escape/secret.txt", "secret-link.txt"):
        error = _error(lambda path=path: docs.read(path))
        assert error.error.code == "filesystem.excluded"
        assert "OUTSIDE" not in error.error.message
        assert str(outside) not in error.error.message
    assert "escape/secret.txt" not in [item["path"] for item in docs.list_files()["items"]]
    assert _error(lambda: docs.list_files(path="escape")).error.code == "filesystem.excluded"


def test_pinned_root_identity_survives_path_replacement(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _, _, docs = _globals(root)
    original = root.with_name("docs-original")
    root.rename(original)
    root.mkdir()
    (root / "notes.txt").write_text("replacement root", encoding="utf-8")

    assert docs.read("notes.txt")["text"] == "alpha\nbeta alpha\ngamma\n"


def test_glob_segments_and_exclusion_have_one_policy(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _, _, docs = _globals(
        root,
        config={"include": ["**/*.txt", "**/*.py"], "exclude": ["nested/**"]},
    )

    assert [item["path"] for item in docs.list_files(glob="*")["items"]] == ["notes.txt"]
    assert docs.list_files(glob="**/*.py")["items"] == []
    assert _error(lambda: docs.read("nested/code.py")).error.code == "filesystem.excluded"


def test_glob_matching_is_segment_bounded_and_deterministic(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    name = "a" * 40 + "b.txt"
    (root / name).write_text("value", encoding="utf-8")
    _, _, docs = _globals(root, config={"include": ["**/*.txt", "**/*.py"]})

    repeated = "*a*a*a*a*a*a*a*a*b.txt"
    assert [item["path"] for item in docs.list_files(glob=repeated)["items"]] == [name]
    assert [item["path"] for item in docs.list_files(glob="**/[a-z]*.py")["items"]] == [
        "nested/code.py"
    ]


def test_scan_limits_and_forged_cursors_fail_typed(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _, _, docs = _globals(root, config={"max_scan_entries": 2})
    assert _error(lambda: docs.list_files()).error.code == "filesystem.oversized"

    _, _, ordinary = _globals(root)
    assert _error(lambda: ordinary.list_files(cursor="not+a+cursor")).error.code == (
        "filesystem.invalid_cursor"
    )

    first = ordinary.list_files(limit=1)
    forged_list = _tamper_cursor(first["next_cursor"], 3)
    assert _error(lambda: ordinary.list_files(cursor=forged_list)).error.code == (
        "filesystem.invalid_cursor"
    )
    grep = ordinary.grep("alpha", paths=["notes.txt"], limit=1)
    forged_grep = _tamper_cursor(grep["next_cursor"], {"file_index": 0, "offset": 0, "line": 999})
    assert (
        _error(lambda: ordinary.grep("alpha", paths=["notes.txt"], cursor=forged_grep)).error.code
        == "filesystem.invalid_cursor"
    )


def test_entry_and_explicit_path_bounds_apply_before_io(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    for index in range(100):
        (root / f"item-{index:03}.txt").write_text("value", encoding="utf-8")
    _, _, docs = _globals(root, config={"max_scan_entries": 10})

    assert _error(lambda: docs.list_files()).error.code == "filesystem.oversized"
    paths = [f"missing-{index}.txt" for index in range(11)]
    assert _error(lambda: docs.grep("value", paths=paths)).error.code == "filesystem.oversized"


def test_list_packs_complete_serialized_pages_under_one_bound(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "page-a.txt").write_text("a", encoding="utf-8")
    (root / "page-b.txt").write_text("b", encoding="utf-8")
    _, _, docs = _globals(
        root,
        config={
            "include": ["**/*.txt"],
            "max_read_bytes": 512,
            "max_result_bytes": 800,
            "max_line_bytes": 512,
        },
    )

    first = docs.list_files()
    assert len(first["items"]) == 1
    assert first["next_cursor"]
    assert len(json.dumps(first, sort_keys=True, separators=(",", ":")).encode()) <= 800
    second = docs.list_files(cursor=first["next_cursor"])
    assert second["items"]


def test_grep_packs_complete_serialized_pages_under_one_bound(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "page.txt").write_text("target one\ntarget two\ntarget three\n", encoding="utf-8")
    _, _, docs = _globals(
        root,
        config={
            "include": ["page.txt"],
            "max_read_bytes": 512,
            "max_scan_bytes": 512,
            "max_result_bytes": 900,
            "max_line_bytes": 512,
        },
    )

    first = docs.grep("target")
    assert len(first["items"]) == 1
    assert first["next_cursor"]
    assert len(json.dumps(first, sort_keys=True, separators=(",", ":")).encode()) <= 900
    second = docs.grep("target", cursor=first["next_cursor"])
    assert second["items"][0]["line"] == 2


def test_request_glob_complexity_errors_are_typed(tmp_path: Path) -> None:
    _, _, docs = _globals(_tree(tmp_path))
    error = _error(lambda: docs.list_files(glob="**/nested/**/file.txt"))
    assert error.error.code == "filesystem.invalid_call"
    assert _error(lambda: docs.grep("\ud800")).error.code == "filesystem.invalid_call"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO required")
def test_special_file_rejection_does_not_block(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    os.mkfifo(root / "pipe")
    _, _, docs = _globals(root, config={"include": ["**/*"]})
    error = _error(lambda: docs.read("pipe"))
    assert error.error.code == "filesystem.unsupported_type"


def test_read_strict_text_ranges_revisions_and_bounds(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _, _, docs = _globals(root, config={"max_read_bytes": 32})
    stat = docs.stat("notes.txt")
    line = docs.read("notes.txt", range={"line_start": 2, "line_end": 2}, revision=stat["revision"])
    assert line["text"] == "beta alpha\n"
    assert line["evidence"]["ranges"][0]["line_start"] == 2
    assert docs.read("notes.txt", range={"byte_start": 0, "byte_end": 5})["text"] == "alpha"

    (root / "notes.txt").write_text("changed", encoding="utf-8")
    changed = _error(lambda: docs.read("notes.txt", revision=stat["revision"]))
    assert changed.error.code == "filesystem.changed"
    oversized = _error(lambda: docs.read("README.md"))
    assert oversized.error.code == "filesystem.oversized"
    binary = _error(lambda: docs.read("binary.bin"))
    assert binary.error.code == "filesystem.binary"


def test_read_enforces_serialized_bound_and_detects_late_nul(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "controls.txt").write_bytes(b"\x01" * 90)
    _, _, small = _globals(
        root,
        config={
            "max_read_bytes": 100,
            "max_scan_bytes": 100,
            "max_result_bytes": 100,
            "max_line_bytes": 100,
        },
    )
    assert _error(lambda: small.read("controls.txt")).error.code == "filesystem.oversized"

    (root / "late-nul.txt").write_bytes(b"a" * 9000 + b"\x00")
    _, _, large = _globals(
        root,
        config={
            "max_read_bytes": 10_000,
            "max_result_bytes": 20_000,
            "max_line_bytes": 10_000,
        },
    )
    assert _error(lambda: large.read("late-nul.txt")).error.code == "filesystem.binary"


def test_full_read_line_evidence_matches_addressable_lines(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "trailing.txt").write_text("a\nb\n", encoding="utf-8")
    (root / "empty.txt").write_text("", encoding="utf-8")
    _, _, docs = _globals(root)

    trailing = docs.read("trailing.txt")["evidence"]["ranges"][0]
    assert (trailing["line_start"], trailing["line_end"]) == (1, 2)
    empty = docs.read("empty.txt")["evidence"]["ranges"][0]
    assert empty["byte_start"] == empty["byte_end"] == 0
    assert empty["line_start"] is empty["line_end"] is None


def test_byte_range_must_align_to_utf8_codepoints(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "unicode.txt").write_text("a☃b", encoding="utf-8")
    (root / "invalid.txt").write_bytes(b"plain\xfftext")
    _, _, docs = _globals(root)
    error = _error(lambda: docs.read("unicode.txt", range={"byte_start": 2, "byte_end": 3}))
    assert error.error.code == "filesystem.invalid_range"
    binary = _error(lambda: docs.read("invalid.txt", range={"byte_start": 0, "byte_end": 10}))
    assert binary.error.code == "filesystem.binary"


def test_markdown_sections_are_optional_and_removable(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _, _, plain = _globals(root)
    unavailable = _error(lambda: plain.read("README.md", section="Details"))
    assert unavailable.error.code == "filesystem.enricher_unavailable"

    _, _, enriched = _globals(root, config={"enrichers": ["markdown"]})
    section = enriched.read("README.md", section="Details")
    assert section["text"] == "## Details\n\nNested facts.\n"
    assert section["evidence"]["ranges"][0]["section"] == "Details"


def test_grep_and_search_are_literal_bounded_and_paginated(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _, _, docs = _globals(root, config={"include": ["**/*.txt", "**/*.py"]})

    first = docs.grep("alpha", limit=1)
    assert first["items"][0]["path"] == "nested/code.py"
    assert first["next_cursor"]
    second = docs.grep("alpha", limit=2, cursor=first["next_cursor"])
    assert [item["line"] for item in second["items"]] == [1, 2]
    search = docs.search("BETA ALPHA", filters={"paths": ["notes.txt"]})
    assert [(item["path"], item["line"]) for item in search["items"]] == [("notes.txt", 2)]
    search_range = search["items"][0]["evidence"]["ranges"][0]
    assert search_range["byte_end"] - search_range["byte_start"] == len("beta alpha")


def test_large_log_uses_bounded_stable_scan_pages_and_read_ranges(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    lines = [f"{index:04} target event\n" for index in range(500)]
    (root / "events.log").write_text("".join(lines), encoding="utf-8")
    _, _, docs = _globals(
        root,
        config={
            "include": ["**/*.log"],
            "max_read_bytes": 4096,
            "max_scan_bytes": 256,
            "max_result_bytes": 4096,
            "max_line_bytes": 128,
        },
    )

    first = docs.grep("target", limit=3)
    second = docs.grep("target", limit=3, cursor=first["next_cursor"])
    assert [item["line"] for item in first["items"] + second["items"]] == list(range(1, 7))
    assert second["next_cursor"]
    first_line_bytes = len(lines[0].encode())
    read = docs.read("events.log", range={"byte_start": 0, "byte_end": first_line_bytes})
    assert read["text"] == lines[0]
    assert read["evidence"]["ranges"][0]["byte_end"] == first_line_bytes


def test_scan_line_bound_includes_line_ending(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "long.txt").write_bytes(b"12345678\n")
    _, _, docs = _globals(
        root,
        config={
            "include": ["**/*.txt"],
            "max_read_bytes": 64,
            "max_scan_bytes": 64,
            "max_result_bytes": 512,
            "max_line_bytes": 8,
        },
    )

    assert _error(lambda: docs.grep("1", paths=["long.txt"])).error.code == ("filesystem.oversized")


def test_excluded_missing_binary_oversized_and_changed_are_typed(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _, _, docs = _globals(root, config={"max_read_bytes": 4})
    assert _error(lambda: docs.read("missing.txt")).error.code == "filesystem.not_found"
    assert _error(lambda: docs.read(".git/config")).error.code == "filesystem.excluded"
    assert _error(lambda: docs.read("binary.bin")).error.code == "filesystem.binary"
    assert _error(lambda: docs.read("notes.txt")).error.code == "filesystem.oversized"


def test_native_and_unary_bridge_envelopes_match_exactly(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "oversized.txt").write_text("x" * 1_048_577, encoding="utf-8")
    native_registry = _registry(root)
    bound = native_registry.sources[0]
    service = ProviderService(bound)
    bridge = BridgeProvider(service.handle)
    effects = {operation.operation_id: SideEffect.READ for operation in bridge.manifest.operations}
    remote_registry = ProviderCatalog((bridge.registration(effects=effects),)).bind(
        (ConfiguredSource("docs", "filesystem_text"),)
    )
    native = CapabilityBroker(native_registry.capability_registrations(), run_id="run-1")
    remote = CapabilityBroker(remote_registry.capability_registrations(), run_id="run-1")

    cases = (
        (("notes.txt",), {}),
        (("missing.txt",), {}),
        (("binary.bin",), {}),
        ((".git/config",), {}),
        (("oversized.txt",), {}),
        (("notes.txt",), {"revision": "sha256:" + "0" * 64}),
    )
    for args, kwargs in cases:
        capability_id = next(
            item.descriptor.capability_id
            for item in native_registry.capability_registrations()
            if item.descriptor.capability_id.operation == "read"
        )
        call = CapabilityCall(capability_id, "call-1", "run-1", args=args, kwargs=kwargs)
        native_result = native.dispatch(call)
        assert native_result.to_dict() == remote.dispatch(call).to_dict()
        if native_result.ok:
            assert native_result.evidence[0].source_id == "docs"
            assert {item.name for item in native_result.usage} == {
                "bytes_scanned",
                "files_scanned",
            }


def test_duplex_bridge_observes_cancellation_during_filesystem_scan(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    for index in range(50):
        (root / f"file-{index:03}.txt").write_text("search me\n", encoding="utf-8")
    native_registry = _registry(root, config={"include": ["**/*.txt"]})
    service = ProviderService(native_registry.sources[0])
    bridge = BridgeProvider(
        service.handle,
        duplex_call=lambda method, payload: _DuplexSession(service, method, payload),
    )
    effects = {operation.operation_id: SideEffect.READ for operation in bridge.manifest.operations}
    registry = ProviderCatalog((bridge.registration(effects=effects),)).bind(
        (ConfiguredSource("docs", "filesystem_text"),)
    )
    capability_id = next(
        item.descriptor.capability_id
        for item in registry.capability_registrations()
        if item.descriptor.capability_id.operation == "grep"
    )

    result = CapabilityBroker(registry.capability_registrations()).call(capability_id, "search")

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "cancelled"
    assert result.result is None


def test_real_sqlite_and_filesystem_sources_share_one_registry_and_environment(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    database = tmp_path / "records.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE records(value TEXT)")
    connection.execute("INSERT INTO records VALUES ('database fact')")
    connection.commit()
    connection.close()
    registry = ProviderCatalog((sqlite_provider(), filesystem_text_provider())).bind(
        (
            ConfiguredSource("db", "sqlite", {"sqlite_path": str(database)}),
            ConfiguredSource("docs", "filesystem_text", {"root": str(root)}),
        )
    )
    config = EnvironmentConfig(kind="native")
    environment = create_environment(
        config,
        context={},
        registry=registry,
        subcalls=MockSubcallClient(),
        execution_context=create_environment_context(config),
    )
    globals_ = environment.globals()

    assert globals_["db"].query("SELECT value FROM records") == [{"value": "database fact"}]
    assert "Welcome" in globals_["docs"].read("README.md")["text"]
    assert set(vars(globals_["db"])) == {"query", "get_schema"}
    assert set(vars(globals_["docs"])) == {"list_files", "read", "grep", "search", "stat"}
    assert "docs.read" in registry.prompt_fragment()
    assert "db.query" in registry.prompt_fragment()


def test_platform_without_secure_primitives_fails_closed(tmp_path: Path, monkeypatch) -> None:
    root = _tree(tmp_path)
    monkeypatch.setattr(os, "supports_dir_fd", set())
    with pytest.raises(RuntimeError, match="dir_fd"):
        _registry(root)


def test_repeated_success_and_error_calls_do_not_leak_descriptors(
    tmp_path: Path, monkeypatch
) -> None:
    root = _tree(tmp_path)
    _, _, docs = _globals(root)
    real_open = os.open
    real_close = os.close
    counts = {"open": 0, "close": 0}

    def tracked_open(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        counts["open"] += 1
        return descriptor

    def tracked_close(descriptor):
        real_close(descriptor)
        counts["close"] += 1

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "close", tracked_close)
    for _ in range(25):
        assert docs.read("notes.txt")["text"].startswith("alpha")
        assert _error(lambda: docs.read("missing.txt")).error.code == "filesystem.not_found"

    # A previously unreachable provider may finalize during this loop and add
    # an unrelated close, but calls under test may not accumulate new opens.
    assert counts["open"] - counts["close"] <= 1
