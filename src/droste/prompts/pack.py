"""Versioned prompt-pack values, validation, loading, and pure resolution.

Prompt content is data. Loaders are the I/O boundary; parsing, validation,
selection, and rendering operate only on immutable values.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from string import Formatter
from typing import Any, Literal, Mapping

PROMPT_PACK_SCHEMA_VERSION = 1
PROMPT_SLOT_NAMES = frozenset({"capabilities", "budget", "question", "history", "output_contract"})
DEFAULT_PROMPT_PROFILE = "full"

CODE_OUTPUT_CONTRACT = (
    "**TEXT OUTPUT ONLY. No function/tool calling.**\n\n"
    "Each turn, return exactly one fenced `python` code block."
)
EXTRACT_OUTPUT_CONTRACT = "Reply with answer text only: no code block and no preamble."

_TEMPLATE_FIELDS = (
    "system",
    "user",
    "refinement",
    "missing_code_repair",
    "error_repair",
    "extract_system",
    "extract_user",
)
_REQUIRED_TEMPLATE_SLOTS = {
    "system": frozenset({"capabilities", "budget", "output_contract"}),
    "user": frozenset({"question", "history"}),
    "refinement": frozenset({"history", "output_contract"}),
    "missing_code_repair": frozenset({"output_contract"}),
    "error_repair": frozenset({"history", "output_contract"}),
    "extract_system": frozenset({"output_contract"}),
    "extract_user": frozenset({"question", "history"}),
}
_BUILTIN_PACK_FILES = (
    "generic-full-v1.toml",
    "generic-minimal-v1.toml",
    "generic-none-v1.toml",
)
_BUILTIN_PACK_BY_PROFILE = {
    "full": "generic-full-v1.toml",
    "minimal": "generic-minimal-v1.toml",
    "none": "generic-none-v1.toml",
}

ResolutionTier = Literal[
    "caller",
    "consumer_model_family",
    "consumer_generic",
    "engine_model_family",
    "generic",
]


class PromptPackError(ValueError):
    """A prompt-pack artifact or catalog violates the stable contract."""


@dataclass(frozen=True, slots=True)
class PromptPackProvenance:
    source: str
    notes: str
    benchmark: str | None = None
    score: str | None = None


@dataclass(frozen=True, slots=True)
class PromptPolicyDefaults:
    enforce_contract: bool = True


@dataclass(frozen=True, slots=True)
class PromptTemplates:
    system: str
    user: str
    refinement: str
    missing_code_repair: str
    error_repair: str
    extract_system: str
    extract_user: str


@dataclass(frozen=True, slots=True)
class PromptPack:
    schema_version: int
    pack_id: str
    revision: str
    profile: str
    templates: PromptTemplates
    policy_defaults: PromptPolicyDefaults
    provenance: PromptPackProvenance
    tips: tuple[str, ...] = ()
    unable_sentinel: str = "unable to determine from the work so far"

    def __post_init__(self) -> None:
        # Copy caller-owned sequences so the frozen pack never aliases mutable
        # strategy state. Full validation happens at parse/binding/resolution.
        object.__setattr__(self, "tips", tuple(self.tips))


@dataclass(frozen=True, slots=True)
class PromptPackBinding:
    model_family: str
    profile: str
    pack: PromptPack

    def __post_init__(self) -> None:
        validate_prompt_pack(self.pack)
        family = _selector(self.model_family, "model_family")
        profile = _selector(self.profile, "profile")
        if profile != self.pack.profile:
            raise PromptPackError(
                f"binding profile {profile!r} does not match pack profile {self.pack.profile!r}"
            )
        object.__setattr__(self, "model_family", family)
        object.__setattr__(self, "profile", profile)


@dataclass(frozen=True, slots=True)
class PromptPackCatalog:
    bindings: tuple[PromptPackBinding, ...] = ()

    def __post_init__(self) -> None:
        bindings = tuple(self.bindings)
        object.__setattr__(self, "bindings", bindings)
        selectors = [(binding.model_family, binding.profile) for binding in bindings]
        if len(selectors) != len(set(selectors)):
            raise PromptPackError("prompt-pack catalog contains duplicate selectors")


@dataclass(frozen=True, slots=True)
class ResolvedPromptPack:
    pack: PromptPack
    tier: ResolutionTier
    requested_model: str
    requested_profile: str
    matched_model_family: str

    def record(self) -> "PromptPackRecord":
        return PromptPackRecord(
            pack_id=self.pack.pack_id,
            revision=self.pack.revision,
            profile=self.pack.profile,
            resolution_tier=self.tier,
            model_family=self.matched_model_family,
            provenance_source=self.pack.provenance.source,
            provenance_benchmark=self.pack.provenance.benchmark,
            provenance_score=self.pack.provenance.score,
        )


@dataclass(frozen=True, slots=True)
class PromptPackRecord:
    pack_id: str
    revision: str
    profile: str
    resolution_tier: ResolutionTier
    model_family: str
    provenance_source: str
    provenance_benchmark: str | None = None
    provenance_score: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "id": self.pack_id,
            "revision": self.revision,
            "profile": self.profile,
            "resolution_tier": self.resolution_tier,
            "model_family": self.model_family,
            "provenance_source": self.provenance_source,
            "provenance_benchmark": self.provenance_benchmark,
            "provenance_score": self.provenance_score,
        }


@dataclass(frozen=True, slots=True)
class PromptSlots:
    capabilities: str = ""
    budget: str = ""
    question: str = ""
    history: str = ""
    output_contract: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "capabilities": self.capabilities,
            "budget": self.budget,
            "question": self.question,
            "history": self.history,
            "output_contract": self.output_contract,
        }


def _selector(value: str, field: str) -> str:
    normalized = str(value).strip().casefold()
    if not normalized:
        raise PromptPackError(f"{field} must be a non-empty string")
    return normalized


def _strict_keys(
    value: Mapping[str, Any], *, required: set[str], allowed: set[str], path: str
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise PromptPackError(f"{path} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise PromptPackError(f"{path} has unknown fields: {', '.join(unknown)}")


def _table(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromptPackError(f"{path} must be a table")
    return value


def _text(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        suffix = "a string" if allow_empty else "a non-empty string"
        raise PromptPackError(f"{path} must be {suffix}")
    return value


def _validate_template(name: str, template: str) -> None:
    slots: set[str] = set()
    try:
        parsed = Formatter().parse(template)
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if field_name not in PROMPT_SLOT_NAMES:
                raise PromptPackError(
                    f"templates.{name} uses unknown slot {{{field_name}}}; "
                    f"allowed slots: {', '.join(sorted(PROMPT_SLOT_NAMES))}"
                )
            if format_spec or conversion:
                raise PromptPackError(
                    f"templates.{name} slot {{{field_name}}} may not use conversions or formats"
                )
            slots.add(field_name)
    except ValueError as exc:
        raise PromptPackError(f"templates.{name} has malformed braces: {exc}") from exc
    missing = sorted(_REQUIRED_TEMPLATE_SLOTS[name] - slots)
    if missing:
        rendered = ", ".join(f"{{{slot}}}" for slot in missing)
        raise PromptPackError(f"templates.{name} is missing required slots: {rendered}")


def parse_prompt_pack(data: Mapping[str, Any], *, source: str = "<memory>") -> PromptPack:
    """Purely validate and convert decoded TOML data into an immutable pack."""
    if not isinstance(data, Mapping):
        raise PromptPackError(f"{source}: prompt pack must be a table")
    required = {
        "schema_version",
        "id",
        "revision",
        "profile",
        "unable_sentinel",
        "tips",
        "provenance",
        "policy_defaults",
        "templates",
    }
    _strict_keys(data, required=required, allowed=required, path=source)
    schema_version = data["schema_version"]
    if type(schema_version) is not int or schema_version != PROMPT_PACK_SCHEMA_VERSION:
        raise PromptPackError(
            f"{source}: schema_version must be {PROMPT_PACK_SCHEMA_VERSION}, got {schema_version!r}"
        )

    profile = _selector(_text(data["profile"], f"{source}.profile"), "profile")
    raw_tips = data["tips"]
    if not isinstance(raw_tips, list) or any(not isinstance(tip, str) for tip in raw_tips):
        raise PromptPackError(f"{source}.tips must be a list of strings")

    provenance_data = _table(data["provenance"], f"{source}.provenance")
    _strict_keys(
        provenance_data,
        required={"source", "notes"},
        allowed={"source", "notes", "benchmark", "score"},
        path=f"{source}.provenance",
    )
    benchmark = provenance_data.get("benchmark")
    score = provenance_data.get("score")
    if benchmark is not None:
        benchmark = _text(benchmark, f"{source}.provenance.benchmark")
    if score is not None:
        score = _text(score, f"{source}.provenance.score")
    if (benchmark is None) != (score is None):
        raise PromptPackError(f"{source}.provenance benchmark and score must be provided together")
    provenance = PromptPackProvenance(
        source=_text(provenance_data["source"], f"{source}.provenance.source"),
        notes=_text(provenance_data["notes"], f"{source}.provenance.notes"),
        benchmark=benchmark,
        score=score,
    )

    policy_data = _table(data["policy_defaults"], f"{source}.policy_defaults")
    _strict_keys(
        policy_data,
        required={"enforce_contract"},
        allowed={"enforce_contract"},
        path=f"{source}.policy_defaults",
    )
    if type(policy_data["enforce_contract"]) is not bool:
        raise PromptPackError(f"{source}.policy_defaults.enforce_contract must be a boolean")

    templates_data = _table(data["templates"], f"{source}.templates")
    template_fields = set(_TEMPLATE_FIELDS)
    _strict_keys(
        templates_data,
        required=template_fields,
        allowed=template_fields,
        path=f"{source}.templates",
    )
    template_values: dict[str, str] = {}
    for name in _TEMPLATE_FIELDS:
        template = _text(templates_data[name], f"{source}.templates.{name}")
        _validate_template(name, template)
        template_values[name] = template

    return PromptPack(
        schema_version=schema_version,
        pack_id=_text(data["id"], f"{source}.id"),
        revision=_text(data["revision"], f"{source}.revision"),
        profile=profile,
        templates=PromptTemplates(**template_values),
        policy_defaults=PromptPolicyDefaults(enforce_contract=policy_data["enforce_contract"]),
        provenance=provenance,
        tips=tuple(raw_tips),
        unable_sentinel=_text(data["unable_sentinel"], f"{source}.unable_sentinel"),
    )


def validate_prompt_pack(pack: PromptPack) -> None:
    """Pure validation for directly constructed as well as loaded packs."""
    if pack.schema_version != PROMPT_PACK_SCHEMA_VERSION:
        raise PromptPackError(
            f"schema_version must be {PROMPT_PACK_SCHEMA_VERSION}, got {pack.schema_version!r}"
        )
    _text(pack.pack_id, "pack.id")
    _text(pack.revision, "pack.revision")
    if _selector(pack.profile, "profile") != pack.profile:
        raise PromptPackError("pack.profile must already be normalized")
    if type(pack.policy_defaults.enforce_contract) is not bool:
        raise PromptPackError("pack.policy_defaults.enforce_contract must be a boolean")
    _text(pack.provenance.source, "pack.provenance.source")
    _text(pack.provenance.notes, "pack.provenance.notes")
    if (pack.provenance.benchmark is None) != (pack.provenance.score is None):
        raise PromptPackError("pack provenance benchmark and score must be provided together")
    if pack.provenance.benchmark is not None:
        _text(pack.provenance.benchmark, "pack.provenance.benchmark")
        _text(pack.provenance.score, "pack.provenance.score")
    if any(not isinstance(tip, str) or not tip.strip() for tip in pack.tips):
        raise PromptPackError("pack.tips must contain only non-empty strings")
    _text(pack.unable_sentinel, "pack.unable_sentinel")
    for name in _TEMPLATE_FIELDS:
        template = _text(getattr(pack.templates, name), f"pack.templates.{name}")
        _validate_template(name, template)


def load_prompt_pack(path: str | Path) -> PromptPack:
    """I/O boundary for a caller-owned TOML prompt pack."""
    artifact = Path(path)
    try:
        with artifact.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PromptPackError(f"cannot load prompt pack {artifact}: {exc}") from exc
    return parse_prompt_pack(data, source=str(artifact))


@lru_cache(maxsize=None)
def load_prompt_pack_resource(filename: str) -> PromptPack:
    """I/O boundary for a prompt pack bundled in the installed wheel."""
    if Path(filename).name != filename or not filename.endswith(".toml"):
        raise PromptPackError("built-in prompt-pack filename must be a plain .toml name")
    resource = files("droste.prompts").joinpath("packs", filename)
    try:
        data = tomllib.loads(resource.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PromptPackError(f"cannot load built-in prompt pack {filename}: {exc}") from exc
    return parse_prompt_pack(data, source=f"droste.prompts/packs/{filename}")


@lru_cache(maxsize=1)
def load_builtin_prompt_catalog() -> PromptPackCatalog:
    """Load and process-cache the engine-owned immutable resource catalog."""
    packs = tuple(load_prompt_pack_resource(filename) for filename in _BUILTIN_PACK_FILES)
    return PromptPackCatalog(
        tuple(
            PromptPackBinding(model_family="generic", profile=pack.profile, pack=pack)
            for pack in packs
        )
    )


def load_builtin_prompt_pack(profile: str = DEFAULT_PROMPT_PROFILE) -> PromptPack:
    normalized = _selector(profile, "profile")
    filename = _BUILTIN_PACK_BY_PROFILE.get(normalized)
    if filename is None:
        raise PromptPackError(f"built-in prompt profile not found: {profile!r}")
    return load_prompt_pack_resource(filename)


def infer_model_family(model: str) -> str:
    """Pure, conservative family classification used only for pack selection."""
    normalized = str(model).strip().casefold().replace(":", "/")
    leaf = normalized.rsplit("/", 1)[-1]
    provider = normalized.split("/", 1)[0] if "/" in normalized else ""
    if leaf.startswith("claude") or provider == "anthropic":
        return "anthropic"
    if leaf.startswith(("gpt", "o1", "o3", "o4")) or provider == "openai":
        return "openai"
    if leaf.startswith("gemini") or provider == "google":
        return "google"
    if not leaf:
        return "generic"
    return leaf.split("-", 1)[0]


def select_prompt_pack(
    catalog: PromptPackCatalog, *, model_family: str, profile: str
) -> PromptPack | None:
    selector = (_selector(model_family, "model_family"), _selector(profile, "profile"))
    for binding in catalog.bindings:
        if (binding.model_family, binding.profile) == selector:
            return binding.pack
    return None


def resolve_prompt_pack(
    *,
    model: str,
    profile: str = DEFAULT_PROMPT_PROFILE,
    caller_pack: PromptPack | None = None,
    consumer_catalog: PromptPackCatalog | None = None,
    engine_catalog: PromptPackCatalog | None = None,
) -> ResolvedPromptPack:
    """Purely choose one complete pack; no partial merge or mutable registry."""
    requested_profile = _selector(profile or DEFAULT_PROMPT_PROFILE, "profile")
    family = infer_model_family(model)
    if caller_pack is not None:
        validate_prompt_pack(caller_pack)
        return ResolvedPromptPack(caller_pack, "caller", model, requested_profile, family)

    catalogs: tuple[tuple[PromptPackCatalog | None, ResolutionTier, ResolutionTier], ...] = (
        (consumer_catalog, "consumer_model_family", "consumer_generic"),
        (engine_catalog, "engine_model_family", "generic"),
    )
    for catalog, family_tier, generic_tier in catalogs:
        if catalog is None:
            continue
        selected = select_prompt_pack(catalog, model_family=family, profile=requested_profile)
        if selected is not None:
            return ResolvedPromptPack(selected, family_tier, model, requested_profile, family)
        selected = select_prompt_pack(catalog, model_family="generic", profile=requested_profile)
        if selected is not None:
            return ResolvedPromptPack(selected, generic_tier, model, requested_profile, "generic")

    # Preserve the legacy behavior for an unknown profile: the generic full
    # profile is the final compatibility fallback, never a partial merge.
    if engine_catalog is not None and requested_profile != DEFAULT_PROMPT_PROFILE:
        selected = select_prompt_pack(
            engine_catalog,
            model_family="generic",
            profile=DEFAULT_PROMPT_PROFILE,
        )
        if selected is not None:
            return ResolvedPromptPack(selected, "generic", model, requested_profile, "generic")
    raise PromptPackError(
        f"no prompt pack for model family {family!r} and profile {requested_profile!r}"
    )


def render_prompt_template(template: str, slots: PromptSlots) -> str:
    """Purely render a validated template from the stable slot value."""
    return template.format_map(slots.as_dict()).strip()


def render_system_prompt(pack: PromptPack, slots: PromptSlots) -> str:
    rendered = render_prompt_template(pack.templates.system, slots)
    if not pack.tips:
        return rendered
    return rendered + "\n\n## Tips\n" + "\n\n".join(pack.tips)
