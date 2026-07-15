"""MCP tools as descriptor-driven Droste providers over local stdio.

MCP is only the transport here.  ``tools/list`` is mapped once into the same
immutable provider values used by in-process sources; generated bindings call
the ordinary capability broker and never receive MCP vocabulary or process
configuration.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Any, Protocol

from ..capabilities import (
    JSON_SCHEMA_2020_12,
    CapabilityError,
    CapabilityExecutionContext,
    CapabilityMetadata,
    CapabilityMetric,
    CapabilityOutcome,
    PaginationMode,
    ProviderOperation,
    ResultDelivery,
    SchemaSpec,
    SideEffect,
)
from ..providers import (
    BoundSource,
    ConfiguredSource,
    ProviderManifest,
    ProviderRegistration,
    ProviderRuntime,
)
from ._mcp_stdio_transport import (
    MCP_PROTOCOL_VERSION,
    McpProtocolError,
    McpRemoteError,
    McpStdioSession,
    McpTransportError,
)

_TOOL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z", re.ASCII)
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z", re.ASCII)
_CONFIG_KEYS = {
    "command",
    "args",
    "env",
    "cwd",
    "allowed_executables",
    "allowed_tools",
    "bindings",
    "effects",
    "budget_classes",
    "policy_metadata",
    "source_description",
    "startup_timeout_ms",
    "close_timeout_ms",
    "max_frame_bytes",
    "max_descriptor_bytes",
    "max_result_bytes",
    "max_stderr_bytes",
    "max_tool_pages",
    "max_tools",
    "max_in_flight",
}


class McpConfigurationError(ValueError):
    """Trusted host MCP configuration is invalid or incomplete."""


class McpDescriptorError(ValueError):
    """A server descriptor cannot be represented without invention."""


class McpToolTransport(Protocol):
    """Trusted connector-neutral edge consumed by MCP provider binding.

    Cross-language hosts can implement this edge with the existing provider
    bridge: only the trusted provider shell sees raw MCP results, while the
    generated-code broker receives normalized ``CapabilityOutcome`` values.
    """

    def list_tools(self) -> tuple[dict[str, Any], ...]: ...

    def call_tool(
        self,
        execution: CapabilityExecutionContext,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], float, int]: ...

    def stats(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


class _OneShotRuntimeBinder:
    """Transfer exactly one acquired runtime into ProviderRegistration.bind."""

    def __init__(self, source: ConfiguredSource, runtime: ProviderRuntime) -> None:
        self._source = source
        self._runtime: ProviderRuntime | None = runtime
        self._lock = Lock()

    def __call__(self, source: ConfiguredSource, context: Any = None) -> ProviderRuntime:
        del context
        if source != self._source:
            raise McpConfigurationError("acquired MCP runtime belongs to a different source")
        with self._lock:
            runtime = self._runtime
            if runtime is None:
                raise RuntimeError("acquired MCP runtime was already bound")
            self._runtime = None
            return runtime


@dataclass(frozen=True, slots=True)
class McpManifestPolicy:
    """Host choices needed by the pure tools/list projection—no live launch state."""

    allowed_tools: tuple[str, ...]
    bindings: Mapping[str, str]
    budget_classes: Mapping[str, str]
    max_descriptor_bytes: int

    def __post_init__(self) -> None:
        if not all(isinstance(item, str) for item in self.allowed_tools):
            raise McpConfigurationError("MCP manifest allowed_tools must contain strings")
        allowed = tuple(sorted(self.allowed_tools))
        if not allowed or len(allowed) != len(set(allowed)):
            raise McpConfigurationError("MCP manifest allowed_tools must be unique and non-empty")
        expected = set(allowed)
        bindings = dict(self.bindings)
        budgets = dict(self.budget_classes)
        if not all(
            isinstance(key, str) and isinstance(item, str)
            for mapping in (bindings, budgets)
            for key, item in mapping.items()
        ):
            raise McpConfigurationError("MCP manifest mappings must map strings to strings")
        if set(bindings) != expected or set(budgets) != expected:
            raise McpConfigurationError(
                "MCP manifest bindings and budget classes must classify every tool exactly"
            )
        if (
            isinstance(self.max_descriptor_bytes, bool)
            or not isinstance(self.max_descriptor_bytes, int)
            or self.max_descriptor_bytes < 1024
        ):
            raise McpConfigurationError("MCP manifest descriptor bound must be at least 1024")
        object.__setattr__(self, "allowed_tools", allowed)
        object.__setattr__(self, "bindings", MappingProxyType(bindings))
        object.__setattr__(self, "budget_classes", MappingProxyType(budgets))


@dataclass(frozen=True, slots=True)
class McpBindingPolicy:
    """Transport-independent host policy for one acquired MCP source."""

    manifest: McpManifestPolicy
    effects: Mapping[str, SideEffect]
    policy_metadata: Mapping[str, Mapping[str, Any]]
    source_description: str
    max_result_bytes: int = 262_144

    def __post_init__(self) -> None:
        expected = set(self.manifest.allowed_tools)
        effects = dict(self.effects)
        if set(effects) != expected or not all(
            isinstance(item, SideEffect) and item is not SideEffect.UNSPECIFIED
            for item in effects.values()
        ):
            raise McpConfigurationError(
                "MCP binding effects must explicitly classify every allowed tool"
            )
        policies = {key: dict(value) for key, value in self.policy_metadata.items()}
        if not set(policies).issubset(expected) or not all(
            isinstance(value, Mapping) for value in self.policy_metadata.values()
        ):
            raise McpConfigurationError("MCP binding policy metadata names an unknown tool")
        if not isinstance(self.source_description, str):
            raise McpConfigurationError("MCP source description must be a string")
        if (
            isinstance(self.max_result_bytes, bool)
            or not isinstance(self.max_result_bytes, int)
            or not 256 <= self.max_result_bytes <= 8_388_608
        ):
            raise McpConfigurationError("MCP max_result_bytes must be in 256..8388608")
        object.__setattr__(self, "effects", MappingProxyType(effects))
        object.__setattr__(self, "policy_metadata", MappingProxyType(policies))


def _bounded_int(value: Any, name: str, *, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise McpConfigurationError(f"MCP {name} must be an integer in {minimum}..{maximum}")
    return value


def _string_tuple(value: Any, name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise McpConfigurationError(f"MCP {name} must be an array of strings")
    copied = tuple(value)
    if not allow_empty and not copied:
        raise McpConfigurationError(f"MCP {name} must not be empty")
    if not all(isinstance(item, str) and "\x00" not in item for item in copied):
        raise McpConfigurationError(f"MCP {name} must contain strings without NUL bytes")
    return copied


def _string_map(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise McpConfigurationError(f"MCP {name} must be an object")
    copied = dict(value)
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in copied.items()):
        raise McpConfigurationError(f"MCP {name} must map strings to strings")
    return copied


@dataclass(frozen=True, slots=True)
class _McpStdioConfig:
    """Explicit host authority for one local MCP process."""

    command: str
    args: tuple[str, ...]
    env: Mapping[str, str]
    cwd: str | None
    allowed_tools: tuple[str, ...]
    bindings: Mapping[str, str]
    effects: Mapping[str, SideEffect]
    budget_classes: Mapping[str, str]
    policy_metadata: Mapping[str, Mapping[str, Any]]
    source_description: str
    startup_timeout_ms: int
    close_timeout_ms: int
    max_frame_bytes: int
    max_descriptor_bytes: int
    max_result_bytes: int
    max_stderr_bytes: int
    max_tool_pages: int
    max_tools: int
    max_in_flight: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> _McpStdioConfig:
        raw = dict(value)
        unknown = set(raw) - _CONFIG_KEYS
        if unknown:
            raise McpConfigurationError(
                f"unknown MCP stdio configuration fields: {sorted(unknown)!r}"
            )
        command = raw.get("command")
        if not isinstance(command, str) or not os.path.isabs(command) or "\x00" in command:
            raise McpConfigurationError("MCP command must be an absolute executable path")
        command = os.path.realpath(command)
        raw_executables = _string_tuple(
            raw.get("allowed_executables"), "allowed_executables", allow_empty=False
        )
        if not all(os.path.isabs(item) for item in raw_executables):
            raise McpConfigurationError("MCP allowed_executables must contain absolute paths")
        executables = tuple(os.path.realpath(item) for item in raw_executables)
        if command not in executables:
            raise McpConfigurationError("MCP command is not in allowed_executables")
        if not os.path.isfile(command) or not os.access(command, os.X_OK):
            raise McpConfigurationError("MCP command must name an executable regular file")

        args = _string_tuple(raw.get("args", ()), "args")
        if len(args) > 128 or sum(len(item.encode("utf-8")) for item in args) > 32_768:
            raise McpConfigurationError("MCP arguments exceed the configured static bound")
        env = _string_map(raw.get("env", {}), "env")
        if not all(_ENV_NAME.fullmatch(key) and "\x00" not in item for key, item in env.items()):
            raise McpConfigurationError("MCP env contains an invalid name or NUL byte")
        if len(env) > 64 or sum(len(key) + len(item) for key, item in env.items()) > 65_536:
            raise McpConfigurationError("MCP env exceeds the configured static bound")
        cwd = raw.get("cwd")
        if (
            not isinstance(cwd, str)
            or not os.path.isabs(cwd)
            or not os.path.isdir(cwd)
            or os.path.islink(cwd)
        ):
            raise McpConfigurationError("MCP cwd must be an absolute non-symlink directory")

        allowed_tools = tuple(
            sorted(_string_tuple(raw.get("allowed_tools"), "allowed_tools", allow_empty=False))
        )
        if len(allowed_tools) != len(set(allowed_tools)):
            raise McpConfigurationError("MCP allowed_tools must not contain duplicates")
        bindings = _string_map(raw.get("bindings"), "bindings")
        raw_effects = _string_map(raw.get("effects"), "effects")
        budgets = _string_map(raw.get("budget_classes"), "budget_classes")
        expected = set(allowed_tools)
        for name, mapping in (
            ("bindings", bindings),
            ("effects", raw_effects),
            ("budget_classes", budgets),
        ):
            if set(mapping) != expected:
                raise McpConfigurationError(f"MCP {name} must classify every allowed tool exactly")
        effects: dict[str, SideEffect] = {}
        for tool_name, effect in raw_effects.items():
            try:
                parsed = SideEffect(effect)
            except ValueError as exc:
                raise McpConfigurationError(
                    f"MCP effect for {tool_name!r} must be read or effectful"
                ) from exc
            if parsed is SideEffect.UNSPECIFIED:
                raise McpConfigurationError(
                    f"MCP effect for {tool_name!r} must be read or effectful"
                )
            if parsed is not SideEffect.READ:
                raise McpConfigurationError(
                    "the local MCP stdio spike exposes read-only data tools only"
                )
            effects[tool_name] = parsed

        raw_policies = raw.get("policy_metadata", {})
        if not isinstance(raw_policies, Mapping) or not set(raw_policies).issubset(expected):
            raise McpConfigurationError("MCP policy_metadata must name only allowed tools")
        policies: dict[str, Mapping[str, Any]] = {}
        for tool_name, metadata in raw_policies.items():
            if not isinstance(metadata, Mapping):
                raise McpConfigurationError("MCP policy_metadata values must be objects")
            # ProviderRegistration takes the immutable snapshot at the public boundary.
            policies[tool_name] = dict(metadata)

        description = raw.get("source_description", "")
        if not isinstance(description, str):
            raise McpConfigurationError("MCP source_description must be a string")
        return cls(
            command=command,
            args=args,
            env=MappingProxyType(env),
            cwd=cwd,
            allowed_tools=allowed_tools,
            bindings=MappingProxyType(bindings),
            effects=MappingProxyType(effects),
            budget_classes=MappingProxyType(budgets),
            policy_metadata=MappingProxyType(policies),
            source_description=description,
            startup_timeout_ms=_bounded_int(
                raw.get("startup_timeout_ms"),
                "startup_timeout_ms",
                default=10_000,
                minimum=100,
                maximum=120_000,
            ),
            close_timeout_ms=_bounded_int(
                raw.get("close_timeout_ms"),
                "close_timeout_ms",
                default=2_000,
                minimum=10,
                maximum=30_000,
            ),
            max_frame_bytes=_bounded_int(
                raw.get("max_frame_bytes"),
                "max_frame_bytes",
                default=1_048_576,
                minimum=1024,
                maximum=16_777_216,
            ),
            max_descriptor_bytes=_bounded_int(
                raw.get("max_descriptor_bytes"),
                "max_descriptor_bytes",
                default=1_048_576,
                minimum=1024,
                maximum=16_777_216,
            ),
            max_result_bytes=_bounded_int(
                raw.get("max_result_bytes"),
                "max_result_bytes",
                default=262_144,
                minimum=256,
                maximum=8_388_608,
            ),
            max_stderr_bytes=_bounded_int(
                raw.get("max_stderr_bytes"),
                "max_stderr_bytes",
                default=65_536,
                minimum=0,
                maximum=1_048_576,
            ),
            max_tool_pages=_bounded_int(
                raw.get("max_tool_pages"),
                "max_tool_pages",
                default=32,
                minimum=1,
                maximum=256,
            ),
            max_tools=_bounded_int(
                raw.get("max_tools"),
                "max_tools",
                default=256,
                minimum=1,
                maximum=4096,
            ),
            max_in_flight=_bounded_int(
                raw.get("max_in_flight"),
                "max_in_flight",
                default=64,
                minimum=1,
                maximum=4096,
            ),
        )

    def session(self) -> McpStdioSession:
        return McpStdioSession(
            command=self.command,
            args=self.args,
            env=dict(self.env),
            cwd=self.cwd,
            startup_timeout_ms=self.startup_timeout_ms,
            close_timeout_ms=self.close_timeout_ms,
            max_frame_bytes=self.max_frame_bytes,
            max_stderr_bytes=self.max_stderr_bytes,
            max_tool_pages=self.max_tool_pages,
            max_tools=self.max_tools,
            max_in_flight=self.max_in_flight,
        )

    def manifest_policy(self) -> McpManifestPolicy:
        return McpManifestPolicy(
            allowed_tools=self.allowed_tools,
            bindings=self.bindings,
            budget_classes=self.budget_classes,
            max_descriptor_bytes=self.max_descriptor_bytes,
        )


def _schema_spec(raw: Any, *, tool_name: str, field: str) -> SchemaSpec:
    if not isinstance(raw, Mapping):
        raise McpDescriptorError(f"MCP tool {tool_name!r} {field} must be a JSON Schema object")
    schema = dict(raw)
    raw_dialect = schema.get("$schema", JSON_SCHEMA_2020_12)
    if not isinstance(raw_dialect, str) or not raw_dialect:
        raise McpDescriptorError(f"MCP tool {tool_name!r} {field} has an invalid $schema")
    return SchemaSpec(
        schema,
        raw_dialect,
        f"mcp:tools/list/{tool_name}/{field}@{MCP_PROTOCOL_VERSION}",
    )


def mcp_tools_to_manifest(
    provider_type: str,
    tools: Sequence[Mapping[str, Any]],
    policy: McpManifestPolicy,
) -> ProviderManifest:
    """Purely map one complete tools/list snapshot into Droste descriptors."""

    try:
        snapshot_bytes = _serialized_size(list(tools))
    except (TypeError, ValueError) as exc:
        raise McpDescriptorError(f"MCP tools/list snapshot is not finite JSON: {exc}") from exc
    if snapshot_bytes > policy.max_descriptor_bytes:
        raise McpDescriptorError("MCP tools/list snapshot exceeds the configured descriptor bound")
    by_name: dict[str, Mapping[str, Any]] = {}
    for raw in tools:
        if not isinstance(raw, Mapping):
            raise McpDescriptorError("MCP tools/list entries must be objects")
        name = raw.get("name")
        if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
            raise McpDescriptorError(
                "MCP tool names must use the supported 1..128 ASCII letter/digit/._- subset"
            )
        if name in by_name:
            raise McpDescriptorError(f"MCP tools/list contains duplicate tool {name!r}")
        by_name[name] = raw
    missing = set(policy.allowed_tools) - set(by_name)
    if missing:
        raise McpDescriptorError(f"MCP allowlist names missing tools: {sorted(missing)!r}")

    operations: list[ProviderOperation] = []
    for name in policy.allowed_tools:
        raw = by_name[name]
        execution = raw.get("execution")
        if isinstance(execution, Mapping) and execution.get("taskSupport") == "required":
            raise McpDescriptorError(
                f"MCP tool {name!r} requires task execution, which belongs to follow-up #91"
            )
        description = raw.get("description")
        if not isinstance(description, str) or not description.strip():
            raise McpDescriptorError(
                f"MCP tool {name!r} needs a declared description or title; Droste will not invent one"
            )
        if len(description.encode("utf-8")) > 8192:
            raise McpDescriptorError(f"MCP tool {name!r} description exceeds 8192 bytes")
        output = raw.get("outputSchema")
        result = (
            _schema_spec(output, tool_name=name, field="outputSchema")
            if output is not None
            else None
        )
        operations.append(
            ProviderOperation(
                operation_id=name,
                binding_name=policy.bindings[name],
                description=description,
                parameters=_schema_spec(
                    raw.get("inputSchema"), tool_name=name, field="inputSchema"
                ),
                result=result,
                pagination=PaginationMode.NONE,
                delivery=ResultDelivery.INLINE if result is not None else ResultDelivery.UNTYPED,
                budget_class=policy.budget_classes[name],
            )
        )
    return ProviderManifest(
        provider_type=provider_type,
        revision=f"mcp-{MCP_PROTOCOL_VERSION}",
        operations=tuple(operations),
    )


def _serialized_size(value: Any) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )


def _fallback_content(content: Sequence[Any]) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, Mapping) or not isinstance(block.get("type"), str):
            raise McpDescriptorError("MCP content blocks must be typed objects")
        blocks.append(dict(block))
    return {"content": blocks, "losses": []}


def normalize_mcp_tool_result(
    *,
    result: Mapping[str, Any],
    expects_structured: bool,
    max_result_bytes: int,
    latency_ms: float,
    response_bytes: int,
) -> CapabilityOutcome:
    """Normalize one MCP result without leaking raw protocol envelopes."""

    if "content" not in result:
        return CapabilityOutcome(
            error=CapabilityError(
                "mcp.invalid_result", "McpInvalidResult", "MCP result requires content"
            )
        )
    raw_content = result.get("content")
    if not isinstance(raw_content, Sequence) or isinstance(raw_content, (str, bytes)):
        return CapabilityOutcome(
            error=CapabilityError(
                "mcp.invalid_result", "McpInvalidResult", "MCP content must be an array"
            )
        )
    content = tuple(raw_content)
    is_error = result.get("isError", False)
    if not isinstance(is_error, bool):
        return CapabilityOutcome(
            error=CapabilityError(
                "mcp.invalid_result", "McpInvalidResult", "MCP isError must be a boolean"
            )
        )
    if is_error:
        messages = [
            block.get("text")
            for block in content
            if isinstance(block, Mapping)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        message = "\n".join(messages) or "MCP tool reported an error"
        encoded = message.encode("utf-8")[:2048]
        return CapabilityOutcome(
            error=CapabilityError(
                "mcp.tool_error",
                "McpToolError",
                encoded.decode("utf-8", errors="ignore"),
            ),
        )

    has_structured = "structuredContent" in result
    structured = result.get("structuredContent")
    if has_structured:
        if not isinstance(structured, Mapping):
            return CapabilityOutcome(
                error=CapabilityError(
                    "mcp.invalid_result",
                    "McpInvalidResult",
                    "MCP structuredContent must be an object",
                )
            )
        value: Any = dict(structured)
        ignored_content_blocks = len(content)
    elif expects_structured:
        return CapabilityOutcome(
            error=CapabilityError(
                "mcp.missing_structured_content",
                "McpMissingStructuredContent",
                "MCP tool declared outputSchema but returned no structuredContent",
            )
        )
    else:
        try:
            value = _fallback_content(content)
        except McpDescriptorError as exc:
            return CapabilityOutcome(
                error=CapabilityError("mcp.invalid_result", "McpInvalidResult", str(exc))
            )
        ignored_content_blocks = 0
    try:
        size = _serialized_size(value)
    except (TypeError, ValueError) as exc:
        return CapabilityOutcome(
            error=CapabilityError(
                "mcp.invalid_result", "McpInvalidResult", f"MCP result is not finite JSON: {exc}"
            )
        )
    if size > max_result_bytes:
        return CapabilityOutcome(
            error=CapabilityError(
                "mcp.result_too_large",
                "McpResultTooLarge",
                "normalized MCP result exceeds the configured bound",
            )
        )
    return CapabilityOutcome(
        result=value,
        metadata=CapabilityMetadata(
            usage=(
                CapabilityMetric("provider_latency", round(latency_ms, 3), "ms"),
                CapabilityMetric("response_bytes", response_bytes, "byte"),
                CapabilityMetric("result_bytes", size, "byte"),
                CapabilityMetric("compatibility_content_ignored", ignored_content_blocks, "block"),
            ),
        ),
    )


def _runtime(
    config: _McpStdioConfig | McpBindingPolicy,
    manifest: ProviderManifest,
    session: McpToolTransport,
) -> ProviderRuntime:
    def operation_handler(operation: ProviderOperation) -> Callable[..., CapabilityOutcome]:
        def invoke(
            execution: CapabilityExecutionContext,
            /,
            *args: Any,
            **kwargs: Any,
        ) -> CapabilityOutcome:
            if args:
                return CapabilityOutcome(
                    error=CapabilityError(
                        "mcp.invalid_arguments",
                        "McpInvalidArguments",
                        "MCP-backed generated bindings require keyword arguments",
                    )
                )
            try:
                raw, elapsed_ms, response_bytes = session.call_tool(
                    execution, operation.operation_id, kwargs
                )
            except McpRemoteError as exc:
                remote_codes = {
                    -32601: "mcp.method_not_found",
                    -32602: "mcp.invalid_params",
                    -32603: "mcp.remote_internal_error",
                }
                return CapabilityOutcome(
                    error=CapabilityError(
                        remote_codes.get(exc.code, "mcp.remote_error"),
                        type(exc).__name__,
                        str(exc),
                    )
                )
            except McpProtocolError as exc:
                return CapabilityOutcome(
                    error=CapabilityError("mcp.protocol_error", type(exc).__name__, str(exc))
                )
            except McpTransportError as exc:
                return CapabilityOutcome(
                    error=CapabilityError("mcp.transport_error", type(exc).__name__, str(exc))
                )
            return normalize_mcp_tool_result(
                result=raw,
                expects_structured=operation.result is not None,
                max_result_bytes=config.max_result_bytes,
                latency_ms=elapsed_ms,
                response_bytes=response_bytes,
            )

        return invoke

    handlers = {
        operation.operation_id: operation_handler(operation) for operation in manifest.operations
    }
    return ProviderRuntime(
        handlers=handlers,
        source_description=config.source_description,
        stats=session.stats,
        close_callback=session.close,
    )


def bind_mcp_transport_source(
    source: ConfiguredSource,
    transport: McpToolTransport,
    policy: McpBindingPolicy,
) -> BoundSource:
    """Freeze a trusted MCP transport behind the ordinary capability ABI.

    This is the cross-language host seam.  Transport/auth/session state stays
    outside generated code; the returned source is indistinguishable from an
    in-process or stdio provider at the broker boundary.
    """

    if not isinstance(source, ConfiguredSource):
        raise TypeError("bind_mcp_transport_source requires a ConfiguredSource")
    if not isinstance(policy, McpBindingPolicy):
        raise TypeError("bind_mcp_transport_source requires McpBindingPolicy")
    try:
        manifest = mcp_tools_to_manifest(
            source.provider_type, transport.list_tools(), policy.manifest
        )
        runtime = _runtime(policy, manifest, transport)
        registration = ProviderRegistration(
            manifest=manifest,
            effects=policy.effects,
            binder=_OneShotRuntimeBinder(source, runtime),
            policy_metadata=policy.policy_metadata,
        )
        return registration.bind(source)
    except BaseException:
        transport.close()
        raise


def open_mcp_stdio_source(source: ConfiguredSource) -> BoundSource:
    """Acquire, discover, and bind one lifecycle-owned local MCP source.

    MCP discovery necessarily performs I/O before a manifest exists.  This one
    transaction therefore returns a bound source rather than pretending a
    reusable static registration can exist first.
    """

    if not isinstance(source, ConfiguredSource):
        raise TypeError("open_mcp_stdio_source requires a ConfiguredSource")
    config = _McpStdioConfig.from_mapping(source.config_dict())
    session = config.session()
    return bind_mcp_transport_source(
        source,
        session,
        McpBindingPolicy(
            manifest=config.manifest_policy(),
            effects=config.effects,
            policy_metadata=config.policy_metadata,
            source_description=config.source_description,
            max_result_bytes=config.max_result_bytes,
        ),
    )


__all__ = [
    "MCP_PROTOCOL_VERSION",
    "McpConfigurationError",
    "McpDescriptorError",
    "McpBindingPolicy",
    "McpManifestPolicy",
    "McpToolTransport",
    "bind_mcp_transport_source",
    "mcp_tools_to_manifest",
    "normalize_mcp_tool_result",
    "open_mcp_stdio_source",
]
