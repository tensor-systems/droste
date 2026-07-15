"""MCP tools over production Streamable HTTP.

Configuration is immutable authority supplied by a trusted host.  Endpoint and
secret material never enters the provider manifest or generated bindings.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from ..capabilities import SideEffect
from ..providers import BoundSource, ConfiguredSource
from ._mcp_http_transport import (
    McpHttpHost,
    McpHttpSession,
    McpHttpTransportConfig,
    McpOAuthConfig,
    McpSecretRequest,
    canonical_https_url,
)
from .mcp_stdio import (
    McpBindingPolicy,
    McpConfigurationError,
    McpManifestPolicy,
    _bounded_int,
    _string_map,
    _string_tuple,
    bind_mcp_transport_source,
)

_CONFIG_KEYS = {
    "endpoint",
    "allowed_endpoints",
    "tenant_id",
    "auth",
    "allowed_tools",
    "bindings",
    "effects",
    "budget_classes",
    "policy_metadata",
    "source_description",
    "startup_timeout_ms",
    "request_timeout_ms",
    "close_timeout_ms",
    "max_frame_bytes",
    "max_descriptor_bytes",
    "max_result_bytes",
    "max_tool_pages",
    "max_tools",
    "max_in_flight",
    "reconnect_attempts",
    "backoff_ms",
    "max_debug_payload_bytes",
}


class _McpHttpConfig:
    __slots__ = ("binding_policy", "transport_config")

    def __init__(
        self, binding_policy: McpBindingPolicy, transport_config: McpHttpTransportConfig
    ) -> None:
        self.binding_policy = binding_policy
        self.transport_config = transport_config

    @classmethod
    def from_source(cls, source: ConfiguredSource) -> _McpHttpConfig:
        raw = source.config_dict()
        unknown = set(raw) - _CONFIG_KEYS
        if unknown:
            raise McpConfigurationError(
                f"unknown MCP HTTP configuration fields: {sorted(unknown)!r}"
            )
        endpoint = canonical_https_url(raw.get("endpoint"), field="endpoint").url
        allowed_endpoints = _string_tuple(
            raw.get("allowed_endpoints"), "allowed_endpoints", allow_empty=False
        )
        canonical_allowed = tuple(
            canonical_https_url(item, field="allowed_endpoints entry").url
            for item in allowed_endpoints
        )
        if len(canonical_allowed) != len(set(canonical_allowed)):
            raise McpConfigurationError("MCP allowed_endpoints must not contain duplicates")
        if endpoint not in canonical_allowed:
            raise McpConfigurationError("MCP endpoint must be exactly allowlisted")
        tenant_id = raw.get("tenant_id")
        if not isinstance(tenant_id, str) or not tenant_id or len(tenant_id.encode("utf-8")) > 256:
            raise McpConfigurationError("MCP tenant_id must be a non-empty bounded string")

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
        for tool, raw_effect in raw_effects.items():
            try:
                effect = SideEffect(raw_effect)
            except ValueError as exc:
                raise McpConfigurationError(
                    f"MCP effect for {tool!r} must be read or effectful"
                ) from exc
            if effect is SideEffect.UNSPECIFIED:
                raise McpConfigurationError(f"MCP effect for {tool!r} must be read or effectful")
            effects[tool] = effect

        raw_policies = raw.get("policy_metadata", {})
        if not isinstance(raw_policies, Mapping) or not set(raw_policies).issubset(expected):
            raise McpConfigurationError("MCP policy_metadata must name only allowed tools")
        policies: dict[str, Mapping[str, Any]] = {}
        for tool, value in raw_policies.items():
            if not isinstance(value, Mapping):
                raise McpConfigurationError("MCP policy_metadata values must be objects")
            policies[tool] = MappingProxyType(dict(value))
        description = raw.get("source_description", "")
        if not isinstance(description, str):
            raise McpConfigurationError("MCP source_description must be a string")

        auth = raw.get("auth", {"type": "none"})
        if not isinstance(auth, Mapping):
            raise McpConfigurationError("MCP auth must be an object")
        auth = dict(auth)
        auth_type = auth.get("type", "none")
        token_ref: str | None = None
        oauth: McpOAuthConfig | None = None
        if auth_type == "none":
            if set(auth) != {"type"}:
                raise McpConfigurationError("MCP unauthenticated config accepts only auth.type")
        elif auth_type == "bearer":
            if set(auth) != {"type", "token_ref"}:
                raise McpConfigurationError("MCP bearer auth requires only token_ref")
            token_ref = cls._secret_ref(auth.get("token_ref"), "token_ref")
        elif auth_type == "oauth_client_credentials":
            expected_auth = {
                "type",
                "client_id_ref",
                "client_secret_ref",
                "resource_metadata_url",
                "authorization_server",
                "token_endpoints",
                "scopes",
            }
            unknown_auth = set(auth) - expected_auth
            required_auth = expected_auth - {"scopes"}
            if unknown_auth or not required_auth.issubset(auth):
                raise McpConfigurationError(
                    "MCP OAuth config requires secret refs and exact discovery/token allowlists"
                )
            scopes = _string_tuple(auth.get("scopes", ()), "OAuth scopes")
            if any(not scope or any(item.isspace() for item in scope) for scope in scopes):
                raise McpConfigurationError("MCP OAuth scopes must be non-empty single tokens")
            token_endpoints = _string_tuple(
                auth.get("token_endpoints"), "OAuth token_endpoints", allow_empty=False
            )
            oauth = McpOAuthConfig(
                client_id_ref=cls._secret_ref(auth.get("client_id_ref"), "client_id_ref"),
                client_secret_ref=cls._secret_ref(
                    auth.get("client_secret_ref"), "client_secret_ref"
                ),
                resource_metadata_url=canonical_https_url(
                    auth.get("resource_metadata_url"), field="OAuth resource metadata URL"
                ).url,
                authorization_server=canonical_https_url(
                    auth.get("authorization_server"), field="OAuth authorization server"
                ).url,
                token_endpoints=tuple(
                    canonical_https_url(item, field="OAuth token endpoint allowlist entry").url
                    for item in token_endpoints
                ),
                scopes=scopes,
            )
            auth_type = "oauth"
        else:
            raise McpConfigurationError(
                "MCP auth.type must be none, bearer, or oauth_client_credentials"
            )

        max_descriptor_bytes = _bounded_int(
            raw.get("max_descriptor_bytes"),
            "max_descriptor_bytes",
            default=1_048_576,
            minimum=1024,
            maximum=16_777_216,
        )
        max_result_bytes = _bounded_int(
            raw.get("max_result_bytes"),
            "max_result_bytes",
            default=262_144,
            minimum=256,
            maximum=8_388_608,
        )
        binding = McpBindingPolicy(
            manifest=McpManifestPolicy(
                allowed_tools=allowed_tools,
                bindings=bindings,
                budget_classes=budgets,
                max_descriptor_bytes=max_descriptor_bytes,
            ),
            effects=effects,
            policy_metadata=policies,
            source_description=description,
            max_result_bytes=max_result_bytes,
        )
        transport = McpHttpTransportConfig(
            endpoint=endpoint,
            allowed_endpoints=canonical_allowed,
            tenant_id=tenant_id,
            source_id=source.source_id,
            auth_type=auth_type,
            token_ref=token_ref,
            oauth=oauth,
            startup_timeout_ms=_bounded_int(
                raw.get("startup_timeout_ms"),
                "startup_timeout_ms",
                default=15_000,
                minimum=100,
                maximum=120_000,
            ),
            request_timeout_ms=_bounded_int(
                raw.get("request_timeout_ms"),
                "request_timeout_ms",
                default=30_000,
                minimum=100,
                maximum=300_000,
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
            reconnect_attempts=_bounded_int(
                raw.get("reconnect_attempts"),
                "reconnect_attempts",
                default=2,
                minimum=0,
                maximum=5,
            ),
            backoff_ms=_bounded_int(
                raw.get("backoff_ms"),
                "backoff_ms",
                default=100,
                minimum=10,
                maximum=5_000,
            ),
            max_debug_payload_bytes=_bounded_int(
                raw.get("max_debug_payload_bytes"),
                "max_debug_payload_bytes",
                default=0,
                minimum=0,
                maximum=4096,
            ),
        )
        return cls(binding, transport)

    @staticmethod
    def _secret_ref(value: Any, name: str) -> str:
        if (
            not isinstance(value, str)
            or not value
            or len(value.encode("utf-8")) > 512
            or "\x00" in value
        ):
            raise McpConfigurationError(f"MCP {name} must be a non-empty bounded reference")
        return value


def open_mcp_http_source(source: ConfiguredSource, host: McpHttpHost) -> BoundSource:
    """Acquire one lifecycle-owned remote MCP source over Streamable HTTP."""

    if not isinstance(source, ConfiguredSource):
        raise TypeError("open_mcp_http_source requires a ConfiguredSource")
    if not isinstance(host, McpHttpHost):
        raise TypeError("open_mcp_http_source requires McpHttpHost")
    config = _McpHttpConfig.from_source(source)
    session = McpHttpSession(config.transport_config, host)
    return bind_mcp_transport_source(source, session, config.binding_policy)


__all__ = ["McpHttpHost", "McpSecretRequest", "open_mcp_http_source"]
