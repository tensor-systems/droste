"""Public manifest and binding shell for the local filesystem/text provider."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..capabilities import (
    JSON_SCHEMA_2020_12,
    CapabilityOutcome,
    PaginationMode,
    ProviderOperation,
    ResultDelivery,
    SchemaSpec,
    SideEffect,
)
from ..providers import (
    ConfiguredSource,
    ProviderManifest,
    ProviderRegistration,
    ProviderRuntime,
)
from ._filesystem_text_runtime import (
    FilesystemTextConfig,
    _FilesystemFailure,
    _OperationResult,
    _RootedTextRuntime,
)


def _object_schema(
    properties: Mapping[str, Any], *, required: tuple[str, ...] = ()
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "type": "object",
        "properties": dict(properties),
        "additionalProperties": False,
    }
    if required:
        value["required"] = list(required)
    return value


_CURSOR = {"type": ["string", "null"]}
_LIMIT = {"type": ["integer", "null"], "minimum": 1}
_PATH = {"type": "string"}
_EVIDENCE_SCHEMA = _object_schema(
    {
        "source_id": {"type": "string"},
        "path": {"type": "string"},
        "revision": {"type": ["string", "null"]},
        "ranges": {"type": "array", "items": {"type": "object"}},
    },
    required=("source_id", "path", "revision", "ranges"),
)
_FILE_ITEM = _object_schema(
    {
        "path": _PATH,
        "kind": {"enum": ["file"]},
        "size_bytes": {"type": "integer", "minimum": 0},
        "mtime_ns": {"type": "integer"},
        "revision": {"type": "string"},
        "evidence": _EVIDENCE_SCHEMA,
    },
    required=("path", "kind", "size_bytes", "mtime_ns", "revision", "evidence"),
)
_LIST_RESULT = _object_schema(
    {"items": {"type": "array", "items": _FILE_ITEM}, "next_cursor": _CURSOR},
    required=("items", "next_cursor"),
)
_MATCH_ITEM = _object_schema(
    {
        "path": _PATH,
        "revision": {"type": "string"},
        "line": {"type": "integer", "minimum": 1},
        "text": {"type": "string"},
        "evidence": _EVIDENCE_SCHEMA,
    },
    required=("path", "revision", "line", "text", "evidence"),
)
_MATCH_RESULT = _object_schema(
    {"items": {"type": "array", "items": _MATCH_ITEM}, "next_cursor": _CURSOR},
    required=("items", "next_cursor"),
)
_READ_RANGE_SCHEMA = {
    "oneOf": [
        _object_schema(
            {
                "byte_start": {"type": "integer", "minimum": 0},
                "byte_end": {"type": "integer", "minimum": 0},
            },
            required=("byte_start", "byte_end"),
        ),
        _object_schema(
            {
                "line_start": {"type": "integer", "minimum": 1},
                "line_end": {"type": "integer", "minimum": 1},
            },
            required=("line_start", "line_end"),
        ),
    ]
}
_SEARCH_FILTERS = _object_schema(
    {
        "paths": {"type": ["array", "null"], "items": _PATH, "minItems": 1},
        "glob": {"type": ["string", "null"]},
    }
)


def _schema(value: Mapping[str, Any], provenance: str) -> SchemaSpec:
    return SchemaSpec(value, JSON_SCHEMA_2020_12, provenance)


FILESYSTEM_TEXT_PROVIDER_MANIFEST = ProviderManifest(
    provider_type="filesystem_text",
    revision="1",
    operations=(
        ProviderOperation(
            "list",
            "list_files",
            "List allowed regular text files recursively in deterministic POSIX-path order.",
            _schema(
                _object_schema(
                    {
                        "path": {"type": "string", "default": ""},
                        "glob": {"type": ["string", "null"]},
                        "cursor": _CURSOR,
                        "limit": _LIMIT,
                    }
                ),
                "droste:provider/filesystem_text/list/parameters@1",
            ),
            _schema(_LIST_RESULT, "droste:provider/filesystem_text/list/result@1"),
            PaginationMode.CURSOR,
            ResultDelivery.INLINE,
            "data.list",
        ),
        ProviderOperation(
            "read",
            "read",
            "Read a bounded strict-UTF-8 file slice or optional Markdown section.",
            _schema(
                _object_schema(
                    {
                        "path": _PATH,
                        "range": {"anyOf": [_READ_RANGE_SCHEMA, {"type": "null"}]},
                        "section": {"type": ["string", "null"]},
                        "max_bytes": {"type": ["integer", "null"], "minimum": 1},
                        "revision": {"type": ["string", "null"]},
                    },
                    required=("path",),
                ),
                "droste:provider/filesystem_text/read/parameters@1",
            ),
            _schema(
                _object_schema(
                    {
                        "path": _PATH,
                        "revision": {"type": "string"},
                        "text": {"type": "string"},
                        "evidence": _EVIDENCE_SCHEMA,
                    },
                    required=("path", "revision", "text", "evidence"),
                ),
                "droste:provider/filesystem_text/read/result@1",
            ),
            PaginationMode.NONE,
            ResultDelivery.INLINE,
            "data.read",
        ),
        ProviderOperation(
            "grep",
            "grep",
            "Find case-sensitive literal text in bounded strict-UTF-8 lines.",
            _schema(
                _object_schema(
                    {
                        "pattern": {"type": "string", "minLength": 1, "maxLength": 256},
                        "paths": {"type": ["array", "null"], "items": _PATH},
                        "glob": {"type": ["string", "null"]},
                        "cursor": _CURSOR,
                        "limit": _LIMIT,
                    },
                    required=("pattern",),
                ),
                "droste:provider/filesystem_text/grep/parameters@1",
            ),
            _schema(_MATCH_RESULT, "droste:provider/filesystem_text/grep/result@1"),
            PaginationMode.CURSOR,
            ResultDelivery.INLINE,
            "data.scan",
        ),
        ProviderOperation(
            "search",
            "search",
            "Find lines containing every case-insensitive query term without an index.",
            _schema(
                _object_schema(
                    {
                        "query": {"type": "string", "minLength": 1, "maxLength": 256},
                        "filters": {"anyOf": [_SEARCH_FILTERS, {"type": "null"}]},
                        "cursor": _CURSOR,
                        "limit": _LIMIT,
                    },
                    required=("query",),
                ),
                "droste:provider/filesystem_text/search/parameters@1",
            ),
            _schema(_MATCH_RESULT, "droste:provider/filesystem_text/search/result@1"),
            PaginationMode.CURSOR,
            ResultDelivery.INLINE,
            "data.search",
        ),
        ProviderOperation(
            "stat",
            "stat",
            "Return bounded metadata and a revision for one root-relative path.",
            _schema(
                _object_schema({"path": _PATH}, required=("path",)),
                "droste:provider/filesystem_text/stat/parameters@1",
            ),
            _schema(
                _object_schema(
                    {
                        "path": _PATH,
                        "kind": {"enum": ["file", "directory"]},
                        "size_bytes": {"type": "integer", "minimum": 0},
                        "mtime_ns": {"type": "integer"},
                        "revision": {"type": "string"},
                        "evidence": {"type": ["object", "null"]},
                    },
                    required=("path", "kind", "size_bytes", "mtime_ns", "revision", "evidence"),
                ),
                "droste:provider/filesystem_text/stat/result@1",
            ),
            PaginationMode.NONE,
            ResultDelivery.INLINE,
            "data.metadata",
        ),
    ),
)


def _bind_filesystem_text(source: ConfiguredSource, context: Any = None) -> ProviderRuntime:
    del context
    runtime = _RootedTextRuntime(
        source.source_id, FilesystemTextConfig.from_mapping(source.config_dict())
    )

    def outcome(function: Callable[..., _OperationResult]) -> Callable[..., CapabilityOutcome]:
        def invoke(execution: Any, *args: Any, **kwargs: Any) -> CapabilityOutcome:
            try:
                return (
                    function(execution, *args, **kwargs)
                    .require_serialized_bound(runtime.config.max_result_bytes)
                    .outcome()
                )
            except _FilesystemFailure as exc:
                return CapabilityOutcome(error=exc.error)

        return invoke

    return ProviderRuntime(
        handlers={
            "list": outcome(runtime.list),
            "read": outcome(runtime.read),
            "grep": outcome(runtime.grep),
            "search": outcome(runtime.search),
            "stat": outcome(runtime.stat),
        },
        source_description=(
            "Root-scoped read-only text files. Paths are relative POSIX paths; reads and "
            "result pages are bounded."
        ),
        close_callback=runtime.close,
    )


def filesystem_text_provider() -> ProviderRegistration:
    """Return the reusable local filesystem/text provider registration."""

    effects = {
        operation.operation_id: SideEffect.READ
        for operation in FILESYSTEM_TEXT_PROVIDER_MANIFEST.operations
    }
    return ProviderRegistration(
        manifest=FILESYSTEM_TEXT_PROVIDER_MANIFEST,
        effects=effects,
        binder=_bind_filesystem_text,
        policy_metadata={
            operation.operation_id: {
                "read_only": True,
                "root_scoped": True,
                "local_only": True,
            }
            for operation in FILESYSTEM_TEXT_PROVIDER_MANIFEST.operations
        },
    )


__all__ = [
    "FILESYSTEM_TEXT_PROVIDER_MANIFEST",
    "FilesystemTextConfig",
    "filesystem_text_provider",
]
