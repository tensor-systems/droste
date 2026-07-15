"""Immutable, content-addressed identity for the model-facing RLM scaffold.

The values and compatibility functions in this module are pure.  Hosts resolve
filesystem, package, model-registry, and source-control facts before building a
manifest and persist the resulting value outside the engine.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from ..prompts.pack import (
    PROMPT_PACK_SCHEMA_VERSION,
    PromptPack,
    prompt_pack_content_sha256,
)
from ..protocols.subcall_capacity import SubcallInputCapacity
from .budget import Budget
from .config import DEFAULT_SUBCALL_CONCURRENCY, SandboxLimits, validate_subcall_concurrency
from .trace import TRACE_ABI_VERSION

SCAFFOLD_MANIFEST_VERSION = 2
_SUPPORTED_SCAFFOLD_MANIFEST_VERSIONS = frozenset({1, SCAFFOLD_MANIFEST_VERSION})
KERNEL_ABI_VERSION = 1
CAPABILITY_ABI_VERSION = 1
TERMINAL_CONTRACT_ID = "answer-ready-v1"
SUBCALL_IDENTITY_CONTRACT_ID = "capability-call-parent-v1"

_JSON_SCALARS = (str, int, float, bool, type(None))


def _freeze_json(value: Any, *, path: str = "value") -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} object keys must be strings")
            normalized[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(dict(sorted(normalized.items())))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, path=f"{path}[]") for item in value)
    if isinstance(value, _JSON_SCALARS):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise ValueError(f"{path} numbers must be finite")
        return value
    raise TypeError(f"{path} contains unsupported type {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the manifest's specified UTF-8 canonical JSON representation."""

    return json.dumps(
        _thaw_json(_freeze_json(value)),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def content_digest(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MANIFEST_TOP_LEVEL_FIELDS = frozenset(
    {
        "engine",
        "abis",
        "prompt_pack",
        "capabilities",
        "contracts",
        "inference",
        "budget",
        "sandbox",
        "parent_child",
    }
)


def _raise_shape(name: str, missing: set[str], unknown: set[str]) -> None:
    detail: list[str] = []
    if missing:
        detail.append("missing " + ", ".join(sorted(missing)))
    if unknown:
        detail.append("unknown " + ", ".join(sorted(unknown)))
    raise ValueError(f"{name} has " + "; ".join(detail))


def _exact_object(
    value: Any,
    *,
    name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    allowed = required | (optional or set())
    missing = required - value.keys()
    unknown = value.keys() - allowed
    if missing or unknown:
        _raise_shape(name, set(missing), set(unknown))
    return value


def _integer(value: Any, name: str, *, nullable: bool = False, positive: bool = False) -> None:
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer" + (" or null" if nullable else ""))
    if positive and value < 1:
        raise ValueError(f"{name} must be positive")


def _text(value: Any, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string" + (" or null" if nullable else ""))


def _digest(value: Any, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"{name} must be a sha256 digest" + (" or null" if nullable else ""))


def _validate_model(value: Any, name: str) -> None:
    model = _exact_object(value, name=name, required={"id", "revision"})
    _text(model["id"], f"{name}.id", nullable=True)
    _text(model["revision"], f"{name}.revision", nullable=True)
    if model["id"] is None and model["revision"] is not None:
        raise ValueError(f"{name}.revision requires a model id")


def _validate_manifest_body(body: Mapping[str, Any], *, schema_version: int) -> None:
    missing = _MANIFEST_TOP_LEVEL_FIELDS - body.keys()
    unknown = body.keys() - _MANIFEST_TOP_LEVEL_FIELDS
    if missing or unknown:
        _raise_shape("scaffold manifest body", set(missing), set(unknown))

    engine = _exact_object(body["engine"], name="engine", required={"version", "source_revision"})
    _text(engine["version"], "engine.version")
    _text(engine["source_revision"], "engine.source_revision", nullable=True)

    abis = _exact_object(
        body["abis"],
        name="abis",
        required={"kernel", "capability", "trace", "prompt_pack", "provider", "runner"},
    )
    for name in ("kernel", "capability", "trace", "prompt_pack", "provider"):
        _integer(abis[name], f"abis.{name}", positive=True)
    _integer(abis["runner"], "abis.runner", nullable=True, positive=True)

    prompt = _exact_object(
        body["prompt_pack"],
        name="prompt_pack",
        required={"id", "revision", "profile", "content_hash"},
    )
    for name in ("id", "revision", "profile"):
        _text(prompt[name], f"prompt_pack.{name}")
    _digest(prompt["content_hash"], "prompt_pack.content_hash")

    capabilities = _exact_object(
        body["capabilities"],
        name="capabilities",
        required={"manifest_hash", "model_visible_globals"},
    )
    _digest(capabilities["manifest_hash"], "capabilities.manifest_hash")
    globals_value = capabilities["model_visible_globals"]
    if not isinstance(globals_value, tuple) or any(
        not isinstance(value, str) or not value for value in globals_value
    ):
        raise TypeError("capabilities.model_visible_globals must be a string array")
    if tuple(sorted(set(globals_value))) != globals_value:
        raise ValueError("capabilities.model_visible_globals must be sorted and unique")

    contracts = _exact_object(
        body["contracts"],
        name="contracts",
        required={
            "terminal",
            "subcall_identity",
            "templates",
            "overrides",
        },
    )
    for name in ("terminal", "subcall_identity"):
        _text(contracts[name], f"contracts.{name}")
    templates = _exact_object(
        contracts["templates"],
        name="contracts.templates",
        required={
            "refinement",
            "missing_code_repair",
            "error_repair",
            "extract_system",
            "extract_user",
        },
    )
    for name, value in templates.items():
        _digest(value, f"contracts.templates.{name}")
    overrides = _exact_object(
        contracts["overrides"],
        name="contracts.overrides",
        required={
            "system_prompt",
            "system_prompt_additions",
            "user_prompt",
            "refinement_prompt",
        },
    )
    for name, value in overrides.items():
        _digest(value, f"contracts.overrides.{name}", nullable=True)

    inference_fields = {
        "root",
        "subcall",
        "root_sampling",
        "subcall_sampling",
        "output_limits",
        "concurrency",
        "seed",
    }
    if schema_version >= 2:
        inference_fields.add("input_capacity")
    inference = _exact_object(
        body["inference"],
        name="inference",
        required=inference_fields,
    )
    _validate_model(inference["root"], "inference.root")
    _validate_model(inference["subcall"], "inference.subcall")
    for name in ("root_sampling", "subcall_sampling"):
        if not isinstance(inference[name], Mapping):
            raise TypeError(f"inference.{name} must be an object")
    limits = _exact_object(
        inference["output_limits"],
        name="inference.output_limits",
        required={"root_tokens", "subcall_tokens"},
    )
    _integer(limits["root_tokens"], "inference.output_limits.root_tokens", positive=True)
    _integer(limits["subcall_tokens"], "inference.output_limits.subcall_tokens", positive=True)
    if schema_version >= 2:
        capacity = _exact_object(
            inference["input_capacity"],
            name="inference.input_capacity",
            required={"subcall"},
        )
        SubcallInputCapacity.from_dict(capacity["subcall"])
    _integer(inference["concurrency"], "inference.concurrency", positive=True)
    _integer(inference["seed"], "inference.seed", nullable=True)

    if not isinstance(body["budget"], Mapping):
        raise TypeError("budget must be an object")
    Budget.from_dict(body["budget"])
    sandbox = _exact_object(
        body["sandbox"],
        name="sandbox",
        required={"output_chars", "execution_timeout_ms", "capture_output_chars"},
    )
    SandboxLimits(
        output_chars=sandbox["output_chars"],
        execution_timeout_ms=sandbox["execution_timeout_ms"],
        capture_output_chars=sandbox["capture_output_chars"],
    )
    parent = _exact_object(
        body["parent_child"],
        name="parent_child",
        required={"trace_depth", "identity"},
    )
    _text(parent["trace_depth"], "parent_child.trace_depth")
    _text(parent["identity"], "parent_child.identity")


@dataclass(frozen=True, slots=True)
class EngineIdentity:
    """Host-resolved engine identity; source revision is optional but explicit."""

    version: str
    source_revision: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("engine version must not be empty")
        if self.source_revision is not None and (
            not isinstance(self.source_revision, str) or not self.source_revision
        ):
            raise ValueError("engine source_revision must be a non-empty string")

    def as_dict(self) -> dict[str, Any]:
        return {"version": self.version, "source_revision": self.source_revision}


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    model_id: str | None
    revision: str | None = None

    def __post_init__(self) -> None:
        if self.model_id is not None and (not isinstance(self.model_id, str) or not self.model_id):
            raise ValueError("model_id must be a non-empty string or null")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision):
            raise ValueError("model revision must be a non-empty string")

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.model_id, "revision": self.revision}


@dataclass(frozen=True, slots=True)
class RolloutConfiguration:
    """Resolved inference facts supplied by the host at the run boundary."""

    root_revision: str | None = None
    subcall_model: str | None = None
    subcall_revision: str | None = None
    root_sampling: Mapping[str, Any] = field(default_factory=dict)
    subcall_sampling: Mapping[str, Any] = field(default_factory=dict)
    concurrency: int = DEFAULT_SUBCALL_CONCURRENCY
    seed: int | None = None
    runner_protocol: int | None = None
    source_revision: str | None = None
    subcall_input_capacity: SubcallInputCapacity = field(
        default_factory=SubcallInputCapacity.unknown
    )

    def __post_init__(self) -> None:
        for name in ("root_revision", "subcall_model", "subcall_revision", "source_revision"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"rollout {name} must be a non-empty string")
        validate_subcall_concurrency(self.concurrency)
        if not isinstance(self.subcall_input_capacity, SubcallInputCapacity):
            raise TypeError("rollout subcall_input_capacity must be SubcallInputCapacity")
        for name in ("seed", "runner_protocol"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise TypeError(f"rollout {name} must be an integer")
        object.__setattr__(
            self, "root_sampling", _freeze_json(self.root_sampling, path="root_sampling")
        )
        object.__setattr__(
            self,
            "subcall_sampling",
            _freeze_json(self.subcall_sampling, path="subcall_sampling"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_revision": self.root_revision,
            "subcall_model": self.subcall_model,
            "subcall_revision": self.subcall_revision,
            "root_sampling": _thaw_json(self.root_sampling),
            "subcall_sampling": _thaw_json(self.subcall_sampling),
            "subcall_input_capacity": self.subcall_input_capacity.as_dict(),
            "concurrency": self.concurrency,
            "seed": self.seed,
            "runner_protocol": self.runner_protocol,
            "source_revision": self.source_revision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RolloutConfiguration":
        expected = {
            "root_revision",
            "subcall_model",
            "subcall_revision",
            "root_sampling",
            "subcall_sampling",
            "subcall_input_capacity",
            "concurrency",
            "seed",
            "runner_protocol",
            "source_revision",
        }
        unknown = value.keys() - expected
        if unknown:
            raise ValueError(
                "rollout configuration has unknown fields: " + ", ".join(sorted(unknown))
            )
        fields = {key: value[key] for key in expected if key in value}
        if "subcall_input_capacity" in fields:
            fields["subcall_input_capacity"] = SubcallInputCapacity.from_dict(
                fields["subcall_input_capacity"]
            )
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class ScaffoldManifest:
    """One resolved scaffold value; ``manifest_id`` hashes only ``as_dict``."""

    body: Mapping[str, Any]
    schema_version: int = SCAFFOLD_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in _SUPPORTED_SCAFFOLD_MANIFEST_VERSIONS:
            raise ValueError(f"unsupported scaffold manifest version: {self.schema_version}")
        frozen = _freeze_json(self.body, path="scaffold_manifest")
        if not isinstance(frozen, Mapping):
            raise TypeError("scaffold manifest body must be an object")
        _validate_manifest_body(frozen, schema_version=self.schema_version)
        object.__setattr__(self, "body", frozen)

    @property
    def manifest_id(self) -> str:
        return content_digest(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, **_thaw_json(self.body)}

    def as_wire_dict(self) -> dict[str, Any]:
        """Return the complete transport value, including its derived identity."""

        return {**self.as_dict(), "id": self.manifest_id}

    def identity_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "id": self.manifest_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScaffoldManifest":
        if not isinstance(value, Mapping):
            raise TypeError("scaffold manifest must be an object")
        allowed = {"schema_version", "id", *_MANIFEST_TOP_LEVEL_FIELDS}
        missing = {"schema_version", *_MANIFEST_TOP_LEVEL_FIELDS} - value.keys()
        unknown = value.keys() - allowed
        if missing or unknown:
            _raise_shape("scaffold manifest", missing, unknown)
        schema_version = value["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise TypeError("scaffold manifest schema_version must be an integer")
        body = {key: value[key] for key in _MANIFEST_TOP_LEVEL_FIELDS}
        manifest = cls(body=body, schema_version=schema_version)
        claimed_id = value.get("id")
        if claimed_id is not None and claimed_id != manifest.manifest_id:
            raise ValueError("scaffold manifest id does not match canonical content")
        return manifest


@dataclass(frozen=True, slots=True)
class ScaffoldRequirements:
    """Checkpoint-declared exact or partial requirements, independent of a registry."""

    manifest_id: str | None = None
    required: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.manifest_id is not None and (
            not isinstance(self.manifest_id, str) or not _DIGEST.fullmatch(self.manifest_id)
        ):
            raise ValueError("checkpoint manifest_id must be a sha256 digest")
        frozen = _freeze_json(self.required, path="checkpoint requirements")
        if not isinstance(frozen, Mapping):
            raise TypeError("checkpoint requirements must be an object")
        object.__setattr__(self, "required", frozen)


@dataclass(frozen=True, slots=True)
class ScaffoldMismatch:
    path: str
    expected: Any
    actual: Any

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "expected": self.expected, "actual": self.actual}


class ScaffoldCompatibilityError(ValueError):
    """Typed pre-inference refusal for an incompatible checkpoint scaffold."""

    def __init__(self, mismatches: tuple[ScaffoldMismatch, ...]) -> None:
        self.mismatches = mismatches
        details = "; ".join(
            f"{item.path}: expected {item.expected!r}, got {item.actual!r}" for item in mismatches
        )
        super().__init__("checkpoint scaffold is incompatible: " + details)


def scaffold_mismatches(
    manifest: ScaffoldManifest, requirements: ScaffoldRequirements
) -> tuple[ScaffoldMismatch, ...]:
    """Pure recursive comparison; requirement objects are partial matches."""

    found: list[ScaffoldMismatch] = []
    if requirements.manifest_id is not None and requirements.manifest_id != manifest.manifest_id:
        found.append(
            ScaffoldMismatch("manifest_id", requirements.manifest_id, manifest.manifest_id)
        )

    def compare(expected: Any, actual: Any, path: str) -> None:
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping):
                found.append(ScaffoldMismatch(path, _thaw_json(expected), _thaw_json(actual)))
                return
            for key, item in expected.items():
                child = f"{path}.{key}" if path else key
                if key not in actual:
                    found.append(ScaffoldMismatch(child, _thaw_json(item), "<missing>"))
                else:
                    compare(item, actual[key], child)
            return
        if expected != actual:
            found.append(ScaffoldMismatch(path, _thaw_json(expected), _thaw_json(actual)))

    compare(requirements.required, _freeze_json(manifest.as_dict()), "")
    return tuple(found)


def require_scaffold_compatibility(
    manifest: ScaffoldManifest, requirements: ScaffoldRequirements | None
) -> None:
    if requirements is None:
        return
    mismatches = scaffold_mismatches(manifest, requirements)
    if mismatches:
        raise ScaffoldCompatibilityError(mismatches)


def _template_contracts(pack: PromptPack) -> dict[str, str]:
    return {
        name: content_digest({"text": getattr(pack.templates, name)})
        for name in (
            "refinement",
            "missing_code_repair",
            "error_repair",
            "extract_system",
            "extract_user",
        )
    }


def _normalized_capabilities(manifest: Any) -> dict[str, Any]:
    value = manifest.to_dict()
    value["capabilities"] = sorted(
        value["capabilities"],
        key=lambda item: canonical_json_bytes(item["capability_id"]),
    )
    return value


def build_scaffold_manifest(
    *,
    engine: EngineIdentity,
    prompt_pack: PromptPack,
    capability_manifest: Any,
    provider_protocol: int,
    model_visible_globals: tuple[str, ...],
    root_model: str | None,
    rollout: RolloutConfiguration,
    budget: Budget,
    sandbox: SandboxLimits,
    system_prompt_override: str | None = None,
    system_prompt_additions: str | None = None,
    user_prompt_override: str | None = None,
    refinement_prompt_override: str | None = None,
) -> ScaffoldManifest:
    """Purely compose existing identities into one content-addressed value."""

    capabilities = _normalized_capabilities(capability_manifest)
    prompt_digest = "sha256:" + prompt_pack_content_sha256(prompt_pack)
    subcall_model = rollout.subcall_model if rollout.subcall_model is not None else root_model
    overrides = {
        "system_prompt": (
            content_digest({"text": system_prompt_override})
            if system_prompt_override is not None
            else None
        ),
        "system_prompt_additions": (
            content_digest({"text": system_prompt_additions}) if system_prompt_additions else None
        ),
        "user_prompt": (
            content_digest({"text": user_prompt_override})
            if user_prompt_override is not None
            else None
        ),
        "refinement_prompt": (
            content_digest({"text": refinement_prompt_override})
            if refinement_prompt_override is not None
            else None
        ),
    }
    return ScaffoldManifest(
        {
            "engine": engine.as_dict(),
            "abis": {
                "kernel": KERNEL_ABI_VERSION,
                "capability": CAPABILITY_ABI_VERSION,
                "trace": TRACE_ABI_VERSION,
                "prompt_pack": PROMPT_PACK_SCHEMA_VERSION,
                "provider": provider_protocol,
                "runner": rollout.runner_protocol,
            },
            "prompt_pack": {
                "id": prompt_pack.pack_id,
                "revision": prompt_pack.revision,
                "profile": prompt_pack.profile,
                "content_hash": prompt_digest,
            },
            "capabilities": {
                "manifest_hash": content_digest(capabilities),
                "model_visible_globals": sorted(set(model_visible_globals)),
            },
            "contracts": {
                "terminal": TERMINAL_CONTRACT_ID,
                "subcall_identity": SUBCALL_IDENTITY_CONTRACT_ID,
                "templates": _template_contracts(prompt_pack),
                "overrides": overrides,
            },
            "inference": {
                "root": ModelIdentity(root_model, rollout.root_revision).as_dict(),
                "subcall": ModelIdentity(subcall_model, rollout.subcall_revision).as_dict(),
                "root_sampling": _thaw_json(rollout.root_sampling),
                "subcall_sampling": _thaw_json(rollout.subcall_sampling),
                "output_limits": {
                    "root_tokens": budget.root_output_tokens,
                    "subcall_tokens": budget.subcall_output_tokens,
                },
                "input_capacity": {
                    "subcall": rollout.subcall_input_capacity.as_dict(),
                },
                "concurrency": rollout.concurrency,
                "seed": rollout.seed,
            },
            "budget": budget.as_dict(),
            "sandbox": {
                "output_chars": sandbox.output_chars,
                "execution_timeout_ms": sandbox.execution_timeout_ms,
                "capture_output_chars": sandbox.resolved_capture_output_chars,
            },
            "parent_child": {
                "trace_depth": "root-zero-child-increment",
                "identity": SUBCALL_IDENTITY_CONTRACT_ID,
            },
        }
    )


@dataclass(frozen=True, slots=True)
class OutcomeJoinKey:
    """Content-free key for trainer-owned task/verifier/reward metadata."""

    run_id: str
    scaffold_manifest_id: str

    def as_dict(self) -> dict[str, str]:
        return {"run_id": self.run_id, "scaffold_manifest_id": self.scaffold_manifest_id}


__all__ = [
    "CAPABILITY_ABI_VERSION",
    "KERNEL_ABI_VERSION",
    "SCAFFOLD_MANIFEST_VERSION",
    "EngineIdentity",
    "ModelIdentity",
    "OutcomeJoinKey",
    "RolloutConfiguration",
    "ScaffoldCompatibilityError",
    "ScaffoldManifest",
    "ScaffoldMismatch",
    "ScaffoldRequirements",
    "build_scaffold_manifest",
    "canonical_json_bytes",
    "content_digest",
    "require_scaffold_compatibility",
    "scaffold_mismatches",
]
