"""Secure runtime for root-scoped, read-only text provider operations.

The provider deliberately has two parts: immutable configuration and pure
matching/cursor values, plus a small POSIX ``openat`` shell.  Generated code
receives only descriptor-generated broker bindings; the configured root and
live directory descriptor remain on the trusted side.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import hmac
import json
import os
import re
import secrets
import stat as stat_module
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from ..capabilities import (
    CapabilityError,
    CapabilityMetadata,
    CapabilityMetric,
    CapabilityOutcome,
    EvidenceLocation,
    EvidenceRange,
)

_CURSOR_VERSION = 1
_MAX_CURSOR_BYTES = 4096
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_CURSOR_TEXT = re.compile(r"[A-Za-z0-9_-]+\Z", re.ASCII)


@dataclass(frozen=True, slots=True)
class FilesystemTextConfig:
    """Resolved host configuration for one local text source."""

    root: str
    include: tuple[str, ...] = ("**/*",)
    exclude: tuple[str, ...] = (".git/**",)
    enrichers: tuple[str, ...] = ()
    max_read_bytes: int = 1_048_576
    max_scan_bytes: int = 16_777_216
    max_result_bytes: int = 1_048_576
    max_scan_entries: int = 10_000
    max_results: int = 200
    max_line_bytes: int = 262_144
    max_depth: int = 64

    def __post_init__(self) -> None:
        if not isinstance(self.root, str) or not self.root or "\x00" in self.root:
            raise ValueError("filesystem_text root must be a non-empty path")
        if not os.path.isabs(self.root):
            raise ValueError("filesystem_text root must be absolute")
        for name in ("include", "exclude"):
            patterns = tuple(getattr(self, name))
            if (name == "include" and not patterns) or not all(
                isinstance(item, str) and item for item in patterns
            ):
                raise ValueError(f"filesystem_text {name} requires valid non-empty POSIX patterns")
            for pattern in patterns:
                _compile_glob(pattern)
            object.__setattr__(self, name, patterns)
        enrichers = tuple(self.enrichers)
        if any(item != "markdown" for item in enrichers) or len(enrichers) != len(set(enrichers)):
            raise ValueError("filesystem_text enrichers currently support only 'markdown'")
        object.__setattr__(self, "enrichers", enrichers)
        for name in (
            "max_read_bytes",
            "max_scan_bytes",
            "max_result_bytes",
            "max_scan_entries",
            "max_results",
            "max_line_bytes",
            "max_depth",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"filesystem_text {name} must be a positive integer")
        if self.max_line_bytes > self.max_scan_bytes:
            raise ValueError("filesystem_text max_line_bytes cannot exceed max_scan_bytes")
        if self.max_read_bytes > self.max_result_bytes:
            raise ValueError("filesystem_text max_read_bytes cannot exceed max_result_bytes")
        if self.max_line_bytes > self.max_result_bytes:
            raise ValueError("filesystem_text max_line_bytes cannot exceed max_result_bytes")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FilesystemTextConfig:
        allowed = {
            "root",
            "include",
            "exclude",
            "enrichers",
            "max_read_bytes",
            "max_scan_bytes",
            "max_result_bytes",
            "max_scan_entries",
            "max_results",
            "max_line_bytes",
            "max_depth",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError("unknown filesystem_text configuration: " + ", ".join(sorted(unknown)))
        kwargs = dict(value)
        for name in ("include", "exclude", "enrichers"):
            if name in kwargs:
                raw = kwargs[name]
                if not isinstance(raw, list):
                    raise TypeError(f"filesystem_text {name} must be an array")
                kwargs[name] = tuple(raw)
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class _FileFact:
    path: str
    size_bytes: int
    mtime_ns: int
    revision: str

    def snapshot_value(self) -> tuple[str, int, int, str]:
        return (self.path, self.size_bytes, self.mtime_ns, self.revision)


_GlobToken = tuple[str, Any]
_GlobSegment = tuple[_GlobToken, ...]


@dataclass(frozen=True, slots=True)
class _GlobPattern:
    segments: tuple[_GlobSegment | None, ...]

    def matches(self, path: str) -> bool:
        parts = path.split("/")
        if None not in self.segments:
            return len(parts) == len(self.segments) and all(
                _segment_matches(pattern, part)
                for pattern, part in zip(self.segments, parts, strict=True)
            )
        recursive = self.segments.index(None)
        prefix = self.segments[:recursive]
        suffix = self.segments[recursive + 1 :]
        if len(parts) < len(prefix) + len(suffix):
            return False
        return all(
            _segment_matches(pattern, part) for pattern, part in zip(prefix, parts, strict=False)
        ) and all(
            _segment_matches(pattern, part)
            for pattern, part in zip(reversed(suffix), reversed(parts), strict=False)
        )


@dataclass(frozen=True, slots=True)
class _OperationResult:
    value: Any
    evidence: tuple[EvidenceLocation, ...] = ()
    bytes_scanned: int = 0
    files_scanned: int = 0

    def require_serialized_bound(self, max_bytes: int) -> _OperationResult:
        if _serialized_size(self.value) > max_bytes:
            _failure(
                "filesystem.oversized",
                "FilesystemOversized",
                f"serialized result exceeds the configured {max_bytes}-byte bound",
            )
        return self

    def outcome(self) -> CapabilityOutcome:
        usage = (
            CapabilityMetric("bytes_scanned", self.bytes_scanned, "byte"),
            CapabilityMetric("files_scanned", self.files_scanned, "file"),
        )
        return CapabilityOutcome(
            result=self.value,
            metadata=CapabilityMetadata(usage=usage, evidence=self.evidence),
        )


class _FilesystemFailure(Exception):
    def __init__(self, code: str, type_name: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.error = CapabilityError(code, type_name, message, retryable)


def _failure(code: str, type_name: str, message: str, *, retryable: bool = False) -> None:
    raise _FilesystemFailure(code, type_name, message, retryable=retryable)


def _validate_relative_path(value: Any, *, allow_root: bool = False) -> str:
    if not isinstance(value, str):
        _failure("filesystem.invalid_path", "FilesystemInvalidPath", "path must be a string")
    if (
        "\x00" in value
        or value.startswith("/")
        or len(value.encode("utf-8", errors="surrogatepass")) > 4096
        or any(0xD800 <= ord(char) <= 0xDFFF for char in value)
    ):
        _failure(
            "filesystem.invalid_path",
            "FilesystemInvalidPath",
            "path must be a relative POSIX path",
        )
    if value == "" and allow_root:
        return ""
    parts = value.split("/")
    if not value or any(
        part in {"", ".", ".."} or len(part.encode("utf-8")) > 255 for part in parts
    ):
        _failure(
            "filesystem.invalid_path",
            "FilesystemInvalidPath",
            "path must contain only non-empty relative POSIX components",
        )
    return "/".join(parts)


def _compile_glob(pattern: str) -> _GlobPattern:
    if not isinstance(pattern, str) or not pattern or pattern.startswith("/") or "\x00" in pattern:
        raise ValueError("filesystem_text patterns must be relative POSIX globs")
    parts = pattern.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("filesystem_text patterns cannot contain empty, '.' or '..' components")
    if (
        len(pattern.encode("utf-8")) > 4096
        or len(parts) > 64
        or any(len(part.encode("utf-8")) > 255 for part in parts)
    ):
        raise ValueError("filesystem_text patterns must fit bounded POSIX path limits")
    if any("**" in part and part != "**" for part in parts):
        raise ValueError("filesystem_text '**' must occupy one complete POSIX segment")
    if sum(part == "**" for part in parts) > 1:
        raise ValueError("filesystem_text patterns may contain at most one '**' segment")
    return _GlobPattern(tuple(None if part == "**" else _compile_segment(part) for part in parts))


def _compile_segment(pattern: str) -> _GlobSegment:
    tokens: list[_GlobToken] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            if not tokens or tokens[-1][0] != "star":
                tokens.append(("star", None))
        elif char == "?":
            tokens.append(("any", None))
        elif char == "[":
            end = pattern.find("]", index + 1)
            if end == -1:
                tokens.append(("literal", char))
            else:
                content = pattern[index + 1 : end]
                if not content:
                    tokens.extend((("literal", "["), ("literal", "]")))
                else:
                    tokens.append(("class", _compile_character_class(content)))
                index = end
        else:
            tokens.append(("literal", char))
        index += 1
    return tuple(tokens)


def _compile_character_class(
    content: str,
) -> tuple[bool, tuple[str, ...], tuple[tuple[str, str], ...]]:
    negated = content[0] in {"!", "^"}
    members = content[1:] if negated else content
    if not members:
        raise ValueError("filesystem_text glob character class must not be empty")
    literals: list[str] = []
    ranges: list[tuple[str, str]] = []
    index = 0
    while index < len(members):
        if index + 2 < len(members) and members[index + 1] == "-":
            start, end = members[index], members[index + 2]
            if ord(start) > ord(end):
                raise ValueError("filesystem_text glob character class range is reversed")
            ranges.append((start, end))
            index += 3
        else:
            literals.append(members[index])
            index += 1
    return negated, tuple(literals), tuple(ranges)


def _segment_matches(pattern: _GlobSegment, value: str) -> bool:
    previous = [False] * (len(value) + 1)
    previous[0] = True
    for kind, expected in pattern:
        current = [False] * (len(value) + 1)
        if kind == "star":
            current[0] = previous[0]
            for index in range(1, len(value) + 1):
                current[index] = previous[index] or current[index - 1]
        else:
            for index, char in enumerate(value, start=1):
                if previous[index - 1] and _glob_token_matches(kind, expected, char):
                    current[index] = True
        previous = current
    return previous[-1]


def _glob_token_matches(kind: str, expected: Any, char: str) -> bool:
    if kind == "any":
        return True
    if kind == "literal":
        return char == expected
    negated, literals, ranges = expected
    included = char in literals or any(start <= char <= end for start, end in ranges)
    return not included if negated else included


def _matches(path: str, patterns: tuple[_GlobPattern, ...]) -> bool:
    return any(pattern.matches(path) for pattern in patterns)


def _serialized_size(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _page_value(items: list[dict[str, Any]], next_cursor: str | None) -> dict[str, Any]:
    return {"items": items, "next_cursor": next_cursor}


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _encode_cursor(
    key: bytes, operation: str, fingerprint: str, snapshot: str, position: Any
) -> str:
    payload = {
        "version": _CURSOR_VERSION,
        "operation": operation,
        "fingerprint": fingerprint,
        "snapshot": snapshot,
        "position": position,
    }
    signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    encoded = json.dumps(
        {**payload, "signature": hmac.new(key, signed, hashlib.sha256).hexdigest()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(encoded).decode().rstrip("=")


def _decode_cursor(key: bytes, cursor: Any, operation: str, fingerprint: str) -> tuple[str, Any]:
    if (
        not isinstance(cursor, str)
        or not cursor
        or len(cursor) > _MAX_CURSOR_BYTES
        or _CURSOR_TEXT.fullmatch(cursor) is None
    ):
        _failure("filesystem.invalid_cursor", "FilesystemInvalidCursor", "cursor is not valid")
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(cursor + padding)
        payload = json.loads(raw)
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _FilesystemFailure(
            "filesystem.invalid_cursor", "FilesystemInvalidCursor", "cursor is not valid"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "version",
        "operation",
        "fingerprint",
        "snapshot",
        "position",
        "signature",
    }:
        _failure("filesystem.invalid_cursor", "FilesystemInvalidCursor", "cursor is not valid")
    signature = payload.pop("signature")
    signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if not isinstance(signature, str) or not hmac.compare_digest(
        signature, hmac.new(key, signed, hashlib.sha256).hexdigest()
    ):
        _failure("filesystem.invalid_cursor", "FilesystemInvalidCursor", "cursor is not valid")
    if (
        payload["version"] != _CURSOR_VERSION
        or payload["operation"] != operation
        or payload["fingerprint"] != fingerprint
        or not isinstance(payload["snapshot"], str)
    ):
        _failure(
            "filesystem.invalid_cursor",
            "FilesystemInvalidCursor",
            "cursor does not match this request",
        )
    return payload["snapshot"], payload["position"]


def _file_revision(info: os.stat_result) -> str:
    return _canonical_digest(
        {
            "device": info.st_dev,
            "inode": info.st_ino,
            "mode": info.st_mode,
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
        }
    )


def _fact(path: str, info: os.stat_result) -> _FileFact:
    return _FileFact(path, info.st_size, info.st_mtime_ns, _file_revision(info))


def _evidence(
    source_id: str,
    fact: _FileFact,
    *,
    byte_start: int | None = None,
    byte_end: int | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
    section: str | None = None,
) -> EvidenceLocation:
    ranges: tuple[EvidenceRange, ...] = ()
    if any(item is not None for item in (byte_start, line_start, section)):
        ranges = (
            EvidenceRange(
                byte_start=byte_start,
                byte_end=byte_end,
                line_start=line_start,
                line_end=line_end,
                section=section,
            ),
        )
    return EvidenceLocation(source_id, fact.path, fact.revision, ranges)


def _error_message(path: str, detail: str) -> str:
    return f"{path!r}: {detail}"


class _SecureRoot:
    """Small POSIX I/O shell pinned to one trusted directory identity."""

    def __init__(self, config: FilesystemTextConfig) -> None:
        self.config = config
        self._ensure_secure_platform()
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            self._root_fd = os.open(config.root, flags)
        except OSError as exc:
            raise ValueError(
                "filesystem_text root must be an existing non-symlink directory"
            ) from exc
        if not stat_module.S_ISDIR(os.fstat(self._root_fd).st_mode):
            os.close(self._root_fd)
            raise ValueError("filesystem_text root must be a directory")

    @staticmethod
    def _ensure_secure_platform() -> None:
        constants = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")
        if os.name != "posix" or any(not hasattr(os, name) for name in constants):
            raise RuntimeError("filesystem_text requires POSIX O_DIRECTORY/O_NOFOLLOW/O_CLOEXEC")
        if (
            os.open not in os.supports_dir_fd
            or os.stat not in os.supports_dir_fd
            or os.scandir not in os.supports_fd
            or not hasattr(os, "pread")
        ):
            raise RuntimeError("filesystem_text requires open/stat dir_fd support")
        if os.stat not in os.supports_follow_symlinks:
            raise RuntimeError("filesystem_text requires lstat-style stat support")

    def close(self) -> None:
        root_fd = self._root_fd
        if root_fd is not None:
            self._root_fd = None
            os.close(root_fd)

    def _open_root_fd(self) -> int:
        root_fd = self._root_fd
        if root_fd is None:
            raise RuntimeError("filesystem_text runtime is closed")
        return root_fd

    @contextmanager
    def directory(self, path: str) -> Iterator[int]:
        fd = os.open(
            ".",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=self._open_root_fd(),
        )
        try:
            for part in path.split("/") if path else ():
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=fd,
                )
                os.close(fd)
                fd = next_fd
            yield fd
        finally:
            os.close(fd)

    @contextmanager
    def open_file(self, path: str) -> Iterator[tuple[int, _FileFact]]:
        normalized = _validate_relative_path(path)
        parent, leaf = normalized.rsplit("/", 1) if "/" in normalized else ("", normalized)
        try:
            with self.directory(parent) as parent_fd:
                fd = os.open(
                    leaf,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                    dir_fd=parent_fd,
                )
        except FileNotFoundError as exc:
            raise _FilesystemFailure(
                "filesystem.not_found",
                "FilesystemNotFound",
                _error_message(normalized, "file was not found"),
            ) from exc
        except OSError as exc:
            if exc.errno in {errno.ENOTDIR, errno.ELOOP}:
                raise _FilesystemFailure(
                    "filesystem.excluded",
                    "FilesystemExcluded",
                    _error_message(normalized, "symlinks are not allowed"),
                ) from exc
            raise _FilesystemFailure(
                "filesystem.io_error",
                "FilesystemIOError",
                _error_message(normalized, "file could not be opened safely"),
                retryable=True,
            ) from exc
        try:
            info = os.fstat(fd)
            if not stat_module.S_ISREG(info.st_mode):
                _failure(
                    "filesystem.unsupported_type",
                    "FilesystemUnsupportedType",
                    _error_message(normalized, "only regular files are supported"),
                )
            yield fd, _fact(normalized, info)
        finally:
            os.close(fd)

    def lstat(self, path: str, *, allow_root: bool = False) -> os.stat_result:
        normalized = _validate_relative_path(path, allow_root=allow_root)
        if not normalized:
            return os.fstat(self._open_root_fd())
        parent, leaf = normalized.rsplit("/", 1) if "/" in normalized else ("", normalized)
        try:
            with self.directory(parent) as parent_fd:
                return os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise _FilesystemFailure(
                "filesystem.not_found",
                "FilesystemNotFound",
                _error_message(normalized, "path was not found"),
            ) from exc
        except OSError as exc:
            if exc.errno in {errno.ENOTDIR, errno.ELOOP}:
                raise _FilesystemFailure(
                    "filesystem.excluded",
                    "FilesystemExcluded",
                    _error_message(normalized, "symlinks are not allowed"),
                ) from exc
            raise _FilesystemFailure(
                "filesystem.io_error",
                "FilesystemIOError",
                _error_message(normalized, "path could not be inspected safely"),
                retryable=True,
            ) from exc

    @staticmethod
    def ensure_unchanged(fd: int, expected: _FileFact) -> None:
        if _file_revision(os.fstat(fd)) != expected.revision:
            _failure(
                "filesystem.changed",
                "FilesystemChanged",
                _error_message(expected.path, "file changed while it was being read"),
                retryable=True,
            )


class _RootedTextRuntime:
    """Provider operations over one policy and one secure root shell."""

    def __init__(self, source_id: str, config: FilesystemTextConfig) -> None:
        self.source_id = source_id
        self.config = config
        self._include = tuple(_compile_glob(item) for item in config.include)
        self._exclude = tuple(_compile_glob(item) for item in config.exclude)
        self._root = _SecureRoot(config)
        self._cursor_key = secrets.token_bytes(32)

    def close(self) -> None:
        self._root.close()

    def _is_excluded(self, path: str) -> bool:
        if _matches(path, self._exclude):
            return True
        return any(
            pattern.endswith("/**") and path == pattern[:-3] for pattern in self.config.exclude
        )

    def _require_allowed(self, path: str) -> None:
        if self._is_excluded(path) or not _matches(path, self._include):
            _failure(
                "filesystem.excluded",
                "FilesystemExcluded",
                _error_message(path, "path is outside the configured include/exclude policy"),
            )

    def _walk_files(
        self,
        execution: Any,
        *,
        base: str = "",
        extra_glob: str | None = None,
    ) -> tuple[_FileFact, ...]:
        normalized_base = _validate_relative_path(base, allow_root=True)
        if normalized_base and self._is_excluded(normalized_base):
            _failure(
                "filesystem.excluded",
                "FilesystemExcluded",
                _error_message(normalized_base, "directory is excluded"),
            )
        try:
            base_info = self._root.lstat(normalized_base, allow_root=True)
        except _FilesystemFailure:
            raise
        if stat_module.S_ISLNK(base_info.st_mode):
            _failure(
                "filesystem.excluded",
                "FilesystemExcluded",
                _error_message(normalized_base, "symlinks are not allowed"),
            )
        if not stat_module.S_ISDIR(base_info.st_mode):
            _failure(
                "filesystem.invalid_path",
                "FilesystemInvalidPath",
                _error_message(normalized_base, "list/search base must be a directory"),
            )
        try:
            extra = (_compile_glob(extra_glob),) if extra_glob is not None else ()
        except (ValueError, UnicodeEncodeError) as exc:
            raise _FilesystemFailure(
                "filesystem.invalid_call",
                "FilesystemInvalidCall",
                "glob must be a valid bounded relative POSIX pattern",
            ) from exc
        pending: list[tuple[str, int]] = [(normalized_base, 0)]
        files: list[_FileFact] = []
        entries_seen = 0
        while pending:
            directory, depth = pending.pop()
            if depth > self.config.max_depth:
                _failure(
                    "filesystem.oversized",
                    "FilesystemOversized",
                    "filesystem tree exceeds the configured depth bound",
                )
            try:
                with self._root.directory(directory) as directory_fd:
                    directory_before = os.fstat(directory_fd)
                    names: list[str] = []
                    with os.scandir(directory_fd) as entries:
                        for entry in entries:
                            execution.check()
                            name = entry.name
                            if not isinstance(name, str) or any(
                                0xD800 <= ord(char) <= 0xDFFF for char in name
                            ):
                                _failure(
                                    "filesystem.invalid_path",
                                    "FilesystemInvalidPath",
                                    "filesystem contains a filename that is not valid UTF-8",
                                )
                            entries_seen += 1
                            if entries_seen > self.config.max_scan_entries:
                                _failure(
                                    "filesystem.oversized",
                                    "FilesystemOversized",
                                    "filesystem scan exceeds max_scan_entries",
                                )
                            names.append(name)
                    names.sort(key=lambda item: item.encode("utf-8"))
                    for name in names:
                        execution.check()
                        if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name:
                            _failure(
                                "filesystem.invalid_path",
                                "FilesystemInvalidPath",
                                "filesystem returned an invalid POSIX component",
                            )
                        path = f"{directory}/{name}" if directory else name
                        if self._is_excluded(path):
                            continue
                        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                        if stat_module.S_ISLNK(info.st_mode):
                            continue
                        if stat_module.S_ISDIR(info.st_mode):
                            pending.append((path, depth + 1))
                        elif stat_module.S_ISREG(info.st_mode) and _matches(path, self._include):
                            if not extra or _matches(path, extra):
                                files.append(_fact(path, info))
                    if _file_revision(os.fstat(directory_fd)) != _file_revision(directory_before):
                        _failure(
                            "filesystem.changed",
                            "FilesystemChanged",
                            _error_message(directory or ".", "directory changed during traversal"),
                            retryable=True,
                        )
            except FileNotFoundError as exc:
                raise _FilesystemFailure(
                    "filesystem.changed",
                    "FilesystemChanged",
                    _error_message(directory or ".", "directory changed during traversal"),
                    retryable=True,
                ) from exc
            except OSError as exc:
                raise _FilesystemFailure(
                    "filesystem.io_error",
                    "FilesystemIOError",
                    _error_message(directory or ".", "directory could not be scanned safely"),
                    retryable=True,
                ) from exc
        return tuple(sorted(files, key=lambda item: item.path))

    def _explicit_files(self, execution: Any, paths: Any) -> tuple[_FileFact, ...]:
        if not isinstance(paths, (list, tuple)) or not paths:
            _failure(
                "filesystem.invalid_call",
                "FilesystemInvalidCall",
                "paths must be a non-empty array",
            )
        if len(paths) > self.config.max_scan_entries:
            _failure(
                "filesystem.oversized",
                "FilesystemOversized",
                "paths exceeds max_scan_entries",
            )
        facts: list[_FileFact] = []
        for raw in paths:
            execution.check()
            path = _validate_relative_path(raw)
            self._require_allowed(path)
            info = self._root.lstat(path)
            if stat_module.S_ISLNK(info.st_mode):
                _failure(
                    "filesystem.excluded",
                    "FilesystemExcluded",
                    _error_message(path, "symlinks are not allowed"),
                )
            if not stat_module.S_ISREG(info.st_mode):
                _failure(
                    "filesystem.unsupported_type",
                    "FilesystemUnsupportedType",
                    _error_message(path, "only regular files are supported"),
                )
            facts.append(_fact(path, info))
        unique = {item.path: item for item in facts}
        return tuple(unique[path] for path in sorted(unique))

    def _candidates(
        self,
        execution: Any,
        *,
        paths: Any = None,
        glob: Any = None,
    ) -> tuple[_FileFact, ...]:
        if paths is not None and glob is not None:
            _failure(
                "filesystem.invalid_call",
                "FilesystemInvalidCall",
                "choose paths or glob, not both",
            )
        if paths is not None:
            return self._explicit_files(execution, paths)
        if glob is not None and not isinstance(glob, str):
            _failure("filesystem.invalid_call", "FilesystemInvalidCall", "glob must be a string")
        return self._walk_files(execution, extra_glob=glob)

    def list(
        self,
        execution: Any,
        *,
        path: str = "",
        glob: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> _OperationResult:
        execution.check()
        page_size = self._page_size(limit)
        files = self._walk_files(execution, base=path, extra_glob=glob)
        fingerprint = _canonical_digest({"source_id": self.source_id, "path": path, "glob": glob})
        snapshot = _canonical_digest([item.snapshot_value() for item in files])
        offset = 0
        if cursor is not None:
            expected_snapshot, position = _decode_cursor(
                self._cursor_key, cursor, "list", fingerprint
            )
            if expected_snapshot != snapshot:
                _failure(
                    "filesystem.changed",
                    "FilesystemChanged",
                    "filesystem listing changed since the cursor was issued",
                    retryable=True,
                )
            if (
                isinstance(position, bool)
                or not isinstance(position, int)
                or not 0 <= position <= len(files)
            ):
                _failure(
                    "filesystem.invalid_cursor",
                    "FilesystemInvalidCursor",
                    "cursor position is not valid",
                )
            offset = position
        items: list[dict[str, Any]] = []
        evidence_items: list[EvidenceLocation] = []
        for fact in files[offset : offset + page_size]:
            evidence_item = _evidence(self.source_id, fact)
            item = self._fact_dict(fact, evidence_item)
            candidate_items = [*items, item]
            candidate_offset = offset + len(candidate_items)
            candidate_cursor = (
                _encode_cursor(self._cursor_key, "list", fingerprint, snapshot, candidate_offset)
                if candidate_offset < len(files)
                else None
            )
            if (
                _serialized_size(_page_value(candidate_items, candidate_cursor))
                > self.config.max_result_bytes
            ):
                if not items:
                    self._oversized(fact.path, self.config.max_result_bytes, subject="result page")
                break
            items.append(item)
            evidence_items.append(evidence_item)
        evidence = tuple(evidence_items)
        next_offset = offset + len(items)
        next_cursor = (
            _encode_cursor(self._cursor_key, "list", fingerprint, snapshot, next_offset)
            if next_offset < len(files)
            else None
        )
        execution.check()
        return _OperationResult(
            _page_value(items, next_cursor),
            evidence,
            files_scanned=len(files),
        )

    def stat(self, execution: Any, path: str) -> _OperationResult:
        execution.check()
        normalized = _validate_relative_path(path)
        if self._is_excluded(normalized):
            self._require_allowed(normalized)
        info = self._root.lstat(normalized)
        if stat_module.S_ISLNK(info.st_mode):
            _failure(
                "filesystem.excluded",
                "FilesystemExcluded",
                _error_message(normalized, "symlinks are not allowed"),
            )
        if stat_module.S_ISDIR(info.st_mode):
            fact = _fact(normalized, info)
            evidence = _evidence(self.source_id, fact)
            value = {
                "path": normalized,
                "kind": "directory",
                "size_bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "revision": fact.revision,
                "evidence": evidence.to_dict(),
            }
            execution.check()
            return _OperationResult(value, (evidence,))
        self._require_allowed(normalized)
        if not stat_module.S_ISREG(info.st_mode):
            _failure(
                "filesystem.unsupported_type",
                "FilesystemUnsupportedType",
                _error_message(normalized, "only regular files and directories are supported"),
            )
        fact = _fact(normalized, info)
        evidence = _evidence(self.source_id, fact)
        execution.check()
        return _OperationResult(self._fact_dict(fact, evidence), (evidence,), files_scanned=1)

    def read(
        self,
        execution: Any,
        path: str,
        *,
        range: Mapping[str, Any] | None = None,
        section: str | None = None,
        max_bytes: int | None = None,
        revision: str | None = None,
    ) -> _OperationResult:
        execution.check()
        bound = self.config.max_read_bytes if max_bytes is None else max_bytes
        if isinstance(bound, bool) or not isinstance(bound, int) or bound < 1:
            _failure(
                "filesystem.invalid_call",
                "FilesystemInvalidCall",
                "max_bytes must be a positive integer",
            )
        if bound > self.config.max_read_bytes:
            _failure(
                "filesystem.oversized",
                "FilesystemOversized",
                f"max_bytes exceeds the configured limit of {self.config.max_read_bytes}",
            )
        if range is not None and section is not None:
            _failure(
                "filesystem.invalid_call",
                "FilesystemInvalidCall",
                "choose range or section, not both",
            )
        self._require_allowed(_validate_relative_path(path))
        with self._root.open_file(path) as (fd, fact):
            if revision is not None and revision != fact.revision:
                _failure(
                    "filesystem.changed",
                    "FilesystemChanged",
                    _error_message(fact.path, "revision no longer matches"),
                    retryable=True,
                )
            probe = os.pread(fd, min(fact.size_bytes, 8192), 0)
            if b"\x00" in probe:
                self._binary(fact.path)
            if section is not None:
                result = self._read_section(execution, fd, fact, section, bound)
            elif range is None:
                if fact.size_bytes > bound:
                    self._oversized(fact.path, bound)
                data = self._read_exact(execution, fd, 0, fact.size_bytes)
                text = self._decode(data, fact.path)
                line_count = len(text.splitlines())
                evidence = _evidence(
                    self.source_id,
                    fact,
                    byte_start=0,
                    byte_end=fact.size_bytes,
                    line_start=1 if line_count else None,
                    line_end=line_count or None,
                )
                result = _OperationResult(
                    self._read_value(fact, text, evidence),
                    (evidence,),
                    bytes_scanned=len(data),
                    files_scanned=1,
                )
            else:
                result = self._read_range(execution, fd, fact, range, bound)
            self._root.ensure_unchanged(fd, fact)
        execution.check()
        return result

    def grep(
        self,
        execution: Any,
        pattern: str,
        *,
        paths: list[str] | None = None,
        glob: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> _OperationResult:
        if not isinstance(pattern, str) or not pattern or len(pattern) > 256:
            _failure(
                "filesystem.invalid_call",
                "FilesystemInvalidCall",
                "grep pattern must be 1..256 characters",
            )
        try:
            needle = pattern.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise _FilesystemFailure(
                "filesystem.invalid_call",
                "FilesystemInvalidCall",
                "grep pattern must be valid Unicode text",
            ) from exc
        return self._scan_lines(
            execution,
            operation="grep",
            request={"pattern": pattern, "paths": paths, "glob": glob},
            paths=paths,
            glob=glob,
            cursor=cursor,
            limit=limit,
            match=lambda raw, text: raw.find(needle),
            match_length=lambda _raw, _text: len(needle),
        )

    def search(
        self,
        execution: Any,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> _OperationResult:
        if not isinstance(query, str) or not query.strip() or len(query) > 256:
            _failure(
                "filesystem.invalid_call",
                "FilesystemInvalidCall",
                "search query must be 1..256 non-whitespace characters",
            )
        if filters is None:
            filters_dict: dict[str, Any] = {}
        elif isinstance(filters, Mapping) and set(filters).issubset({"paths", "glob"}):
            filters_dict = dict(filters)
        else:
            _failure(
                "filesystem.invalid_call",
                "FilesystemInvalidCall",
                "search filters may contain only paths or glob",
            )
        terms = tuple(query.casefold().split())

        def find(_raw: bytes, text: str) -> int:
            folded = text.casefold()
            return 0 if all(term in folded for term in terms) else -1

        return self._scan_lines(
            execution,
            operation="search",
            request={"query": query, "filters": filters_dict},
            paths=filters_dict.get("paths"),
            glob=filters_dict.get("glob"),
            cursor=cursor,
            limit=limit,
            match=find,
            match_length=lambda raw, _text: len(raw.rstrip(b"\r\n")),
        )

    def _scan_lines(
        self,
        execution: Any,
        *,
        operation: str,
        request: Mapping[str, Any],
        paths: Any,
        glob: Any,
        cursor: str | None,
        limit: int | None,
        match: Callable[[bytes, str], int],
        match_length: Callable[[bytes, str], int],
    ) -> _OperationResult:
        execution.check()
        page_size = self._page_size(limit)
        files = self._candidates(execution, paths=paths, glob=glob)
        fingerprint = _canonical_digest({"source_id": self.source_id, **request})
        snapshot = _canonical_digest([item.snapshot_value() for item in files])
        file_index, offset, line_number = 0, 0, 1
        if cursor is not None:
            expected_snapshot, position = _decode_cursor(
                self._cursor_key, cursor, operation, fingerprint
            )
            if expected_snapshot != snapshot:
                _failure(
                    "filesystem.changed",
                    "FilesystemChanged",
                    "filesystem search set changed since the cursor was issued",
                    retryable=True,
                )
            if not (
                isinstance(position, dict)
                and set(position) == {"file_index", "offset", "line"}
                and all(
                    isinstance(position[name], int) and not isinstance(position[name], bool)
                    for name in position
                )
                and 0 <= position["file_index"] <= len(files)
                and position["offset"] >= 0
                and position["line"] >= 1
            ):
                _failure(
                    "filesystem.invalid_cursor",
                    "FilesystemInvalidCursor",
                    "cursor position is not valid",
                )
            file_index, offset, line_number = (
                position["file_index"],
                position["offset"],
                position["line"],
            )
        items: list[dict[str, Any]] = []
        evidence_items: list[EvidenceLocation] = []
        bytes_scanned = 0
        files_scanned = 0
        next_position: dict[str, int] | None = None
        while file_index < len(files):
            fact = files[file_index]
            if offset > fact.size_bytes:
                _failure(
                    "filesystem.changed",
                    "FilesystemChanged",
                    _error_message(fact.path, "cursor offset no longer exists"),
                    retryable=True,
                )
            self._require_allowed(fact.path)
            with self._root.open_file(fact.path) as (fd, current):
                if current.revision != fact.revision:
                    _failure(
                        "filesystem.changed",
                        "FilesystemChanged",
                        _error_message(fact.path, "file changed since scan started"),
                        retryable=True,
                    )
                files_scanned += 1
                os.lseek(fd, offset, os.SEEK_SET)
                with os.fdopen(os.dup(fd), "rb", closefd=True) as stream:
                    while True:
                        execution.check()
                        line_offset = stream.tell()
                        raw = stream.readline(self.config.max_line_bytes + 1)
                        if not raw:
                            break
                        if len(raw) > self.config.max_line_bytes:
                            self._oversized(fact.path, self.config.max_line_bytes, subject="line")
                        if bytes_scanned and bytes_scanned + len(raw) > self.config.max_scan_bytes:
                            next_position = {
                                "file_index": file_index,
                                "offset": line_offset,
                                "line": line_number,
                            }
                            break
                        if b"\x00" in raw:
                            self._binary(fact.path)
                        text = self._decode(raw, fact.path)
                        bytes_scanned += len(raw)
                        found = match(raw, text)
                        if found >= 0:
                            length = match_length(raw, text)
                            evidence = _evidence(
                                self.source_id,
                                fact,
                                byte_start=line_offset + found,
                                byte_end=line_offset + found + length,
                                line_start=line_number,
                                line_end=line_number,
                            )
                            item = {
                                "path": fact.path,
                                "revision": fact.revision,
                                "line": line_number,
                                "text": text.rstrip("\r\n"),
                                "evidence": evidence.to_dict(),
                            }
                            resume_after_item = {
                                "file_index": file_index,
                                "offset": stream.tell(),
                                "line": line_number + 1,
                            }
                            candidate_items = [*items, item]
                            candidate_cursor = (
                                None
                                if file_index == len(files) - 1
                                and resume_after_item["offset"] >= fact.size_bytes
                                else _encode_cursor(
                                    self._cursor_key,
                                    operation,
                                    fingerprint,
                                    snapshot,
                                    resume_after_item,
                                )
                            )
                            if (
                                _serialized_size(_page_value(candidate_items, candidate_cursor))
                                > self.config.max_result_bytes
                            ):
                                if not items:
                                    self._oversized(
                                        fact.path,
                                        self.config.max_result_bytes,
                                        subject="result page",
                                    )
                                next_position = {
                                    "file_index": file_index,
                                    "offset": line_offset,
                                    "line": line_number,
                                }
                                break
                            evidence_items.append(evidence)
                            items.append(item)
                        offset = stream.tell()
                        line_number += 1
                        if len(items) >= page_size:
                            next_position = {
                                "file_index": file_index,
                                "offset": offset,
                                "line": line_number,
                            }
                            break
                self._root.ensure_unchanged(fd, fact)
            if next_position is not None:
                break
            file_index += 1
            offset = 0
            line_number = 1
        if next_position is not None:
            if (
                next_position["file_index"] == len(files) - 1
                and next_position["offset"] >= files[-1].size_bytes
            ):
                next_position = None
        next_cursor = (
            _encode_cursor(self._cursor_key, operation, fingerprint, snapshot, next_position)
            if next_position is not None
            else None
        )
        execution.check()
        return _OperationResult(
            _page_value(items, next_cursor),
            tuple(evidence_items),
            bytes_scanned=bytes_scanned,
            files_scanned=files_scanned,
        )

    def _read_range(
        self,
        execution: Any,
        fd: int,
        fact: _FileFact,
        range_value: Mapping[str, Any],
        bound: int,
    ) -> _OperationResult:
        if not isinstance(range_value, Mapping):
            _failure("filesystem.invalid_call", "FilesystemInvalidCall", "range must be an object")
        keys = set(range_value)
        if keys == {"byte_start", "byte_end"}:
            start, end = range_value["byte_start"], range_value["byte_end"]
            if not self._ordered_range(start, end) or end > fact.size_bytes:
                self._invalid_range(fact.path)
            if end - start > bound:
                self._oversized(fact.path, bound)
            data = self._read_exact(execution, fd, start, end - start)
            try:
                text = self._decode(data, fact.path)
            except _FilesystemFailure as exc:
                if exc.error.code == "filesystem.binary" and not self._range_boundary_aligned(
                    fd, fact.size_bytes, start, end
                ):
                    self._invalid_range(fact.path)
                raise
            evidence = _evidence(self.source_id, fact, byte_start=start, byte_end=end)
        elif keys == {"line_start", "line_end"}:
            start, end = range_value["line_start"], range_value["line_end"]
            if not self._ordered_range(start, end) or start < 1:
                self._invalid_range(fact.path)
            chunks: list[bytes] = []
            first_offset: int | None = None
            final_offset = 0
            scanned = 0
            os.lseek(fd, 0, os.SEEK_SET)
            with os.fdopen(os.dup(fd), "rb", closefd=True) as stream:
                line_no = 1
                while line_no <= end:
                    execution.check()
                    offset = stream.tell()
                    raw = stream.readline(self.config.max_line_bytes + 1)
                    if not raw:
                        break
                    if len(raw) > self.config.max_line_bytes:
                        self._oversized(fact.path, self.config.max_line_bytes, subject="line")
                    scanned += len(raw)
                    if scanned > self.config.max_scan_bytes:
                        self._oversized(fact.path, self.config.max_scan_bytes, subject="scan")
                    if line_no >= start:
                        if first_offset is None:
                            first_offset = offset
                        chunks.append(raw)
                        final_offset = stream.tell()
                        if sum(map(len, chunks)) > bound:
                            self._oversized(fact.path, bound)
                    line_no += 1
            if first_offset is None:
                self._invalid_range(fact.path)
            data = b"".join(chunks)
            text = self._decode(data, fact.path)
            evidence = _evidence(
                self.source_id,
                fact,
                byte_start=first_offset,
                byte_end=final_offset,
                line_start=start,
                line_end=min(end, start + len(chunks) - 1),
            )
        else:
            self._invalid_range(fact.path)
        return _OperationResult(
            self._read_value(fact, text, evidence),
            (evidence,),
            bytes_scanned=len(data),
            files_scanned=1,
        )

    def _read_section(
        self, execution: Any, fd: int, fact: _FileFact, section: Any, bound: int
    ) -> _OperationResult:
        if "markdown" not in self.config.enrichers:
            _failure(
                "filesystem.enricher_unavailable",
                "FilesystemEnricherUnavailable",
                "section reads require the optional markdown enricher",
            )
        if not fact.path.lower().endswith((".md", ".markdown")):
            _failure(
                "filesystem.enricher_unavailable",
                "FilesystemEnricherUnavailable",
                _error_message(fact.path, "is not a Markdown file"),
            )
        if not isinstance(section, str) or not section.strip():
            _failure(
                "filesystem.invalid_call",
                "FilesystemInvalidCall",
                "section must be a non-empty heading",
            )
        if fact.size_bytes > self.config.max_scan_bytes:
            self._oversized(fact.path, self.config.max_scan_bytes, subject="section scan")
        data = self._read_exact(execution, fd, 0, fact.size_bytes)
        text = self._decode(data, fact.path)
        lines = text.splitlines(keepends=True)
        byte_offsets: list[int] = []
        cursor = 0
        for line in lines:
            byte_offsets.append(cursor)
            cursor += len(line.encode("utf-8"))
        wanted = section.strip()
        start_index: int | None = None
        level = 0
        for index, line in enumerate(lines):
            execution.check()
            match = _HEADING.match(line.rstrip("\r\n"))
            if match and match.group(2).strip() == wanted:
                start_index = index
                level = len(match.group(1))
                break
        if start_index is None:
            _failure(
                "filesystem.not_found",
                "FilesystemSectionNotFound",
                _error_message(fact.path, f"Markdown section {wanted!r} was not found"),
            )
        end_index = len(lines)
        for index in range(start_index + 1, len(lines)):
            match = _HEADING.match(lines[index].rstrip("\r\n"))
            if match and len(match.group(1)) <= level:
                end_index = index
                break
        selected = "".join(lines[start_index:end_index])
        selected_bytes = selected.encode("utf-8")
        if len(selected_bytes) > bound:
            self._oversized(fact.path, bound, subject="section")
        byte_start = byte_offsets[start_index]
        byte_end = byte_start + len(selected_bytes)
        evidence = _evidence(
            self.source_id,
            fact,
            byte_start=byte_start,
            byte_end=byte_end,
            line_start=start_index + 1,
            line_end=max(start_index + 1, end_index),
            section=wanted,
        )
        return _OperationResult(
            self._read_value(fact, selected, evidence),
            (evidence,),
            bytes_scanned=len(data),
            files_scanned=1,
        )

    @staticmethod
    def _ordered_range(start: Any, end: Any) -> bool:
        return (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start <= end
        )

    @staticmethod
    def _range_boundary_aligned(fd: int, size: int, start: int, end: int) -> bool:
        def continuation(offset: int) -> bool:
            value = os.pread(fd, 1, offset)
            return bool(value) and value[0] & 0xC0 == 0x80

        return (start == 0 or not continuation(start)) and (end == size or not continuation(end))

    @staticmethod
    def _read_exact(execution: Any, fd: int, offset: int, length: int) -> bytes:
        chunks: list[bytes] = []
        read = 0
        while read < length:
            execution.check()
            chunk = os.pread(fd, min(65_536, length - read), offset + read)
            if not chunk:
                break
            chunks.append(chunk)
            read += len(chunk)
        if read != length:
            _failure(
                "filesystem.changed",
                "FilesystemChanged",
                "file changed while it was being read",
                retryable=True,
            )
        return b"".join(chunks)

    @staticmethod
    def _decode(data: bytes, path: str) -> str:
        if b"\x00" in data:
            _RootedTextRuntime._binary(path)
        try:
            return data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise _FilesystemFailure(
                "filesystem.binary",
                "FilesystemBinary",
                _error_message(path, "content is not strict UTF-8 text"),
            ) from exc

    @staticmethod
    def _binary(path: str) -> None:
        _failure(
            "filesystem.binary",
            "FilesystemBinary",
            _error_message(path, "binary content is not supported"),
        )

    @staticmethod
    def _oversized(path: str, bound: int, *, subject: str = "file") -> None:
        _failure(
            "filesystem.oversized",
            "FilesystemOversized",
            _error_message(path, f"{subject} exceeds the configured {bound}-byte bound"),
        )

    @staticmethod
    def _invalid_range(path: str) -> None:
        _failure(
            "filesystem.invalid_range",
            "FilesystemInvalidRange",
            _error_message(path, "range is invalid"),
        )

    def _page_size(self, limit: Any) -> int:
        if limit is None:
            return min(100, self.config.max_results)
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > self.config.max_results
        ):
            _failure(
                "filesystem.invalid_call",
                "FilesystemInvalidCall",
                f"limit must be between 1 and {self.config.max_results}",
            )
        return limit

    @staticmethod
    def _fact_dict(fact: _FileFact, evidence: EvidenceLocation) -> dict[str, Any]:
        return {
            "path": fact.path,
            "kind": "file",
            "size_bytes": fact.size_bytes,
            "mtime_ns": fact.mtime_ns,
            "revision": fact.revision,
            "evidence": evidence.to_dict(),
        }

    @staticmethod
    def _read_value(fact: _FileFact, text: str, evidence: EvidenceLocation) -> dict[str, Any]:
        return {
            "path": fact.path,
            "revision": fact.revision,
            "text": text,
            "evidence": evidence.to_dict(),
        }


__all__ = [
    "FilesystemTextConfig",
    "_FilesystemFailure",
    "_OperationResult",
    "_RootedTextRuntime",
]
