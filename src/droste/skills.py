"""Versioned, additive RLM strategy artifacts and their brokered provider.

Skills are immutable Markdown data.  Parsing, selection, and hashing are pure;
filesystem/package-resource reads and live provider handlers stay at the edge.
Prompt packs remain the exclusive deterministic harness configuration.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

from .capabilities import (
    JSON_SCHEMA_2020_12,
    CapabilityExecutionContext,
    PaginationMode,
    ProviderOperation,
    ResultDelivery,
    SchemaSpec,
    SideEffect,
)
from .providers import (
    ConfiguredSource,
    ProviderManifest,
    ProviderRegistration,
    ProviderRuntime,
)

RLM_SKILL_SCHEMA_VERSION = 1
_BUILTIN_SKILLS = ("chunking-v1.md", "decomposition-example-v1.md")


class RLMSkillError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RLMSkillProvenance:
    source: str
    benchmark: str | None = None
    delta: str | None = None


@dataclass(frozen=True, slots=True)
class RLMSkill:
    schema_version: int
    skill_id: str
    revision: str
    summary: str
    body: str
    model_families: tuple[str, ...]
    provenance: RLMSkillProvenance

    def __post_init__(self) -> None:
        if self.schema_version != RLM_SKILL_SCHEMA_VERSION:
            raise RLMSkillError(f"unsupported RLM skill schema: {self.schema_version}")
        for name in ("skill_id", "revision", "summary", "body"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise RLMSkillError(f"skill {name} must not be empty")
        families = tuple(str(value).strip().casefold() for value in self.model_families)
        if not families or any(not value for value in families):
            raise RLMSkillError("skill model_families must not be empty")
        if len(families) != len(set(families)):
            raise RLMSkillError("skill model_families contains duplicates")
        object.__setattr__(self, "model_families", families)
        if not self.provenance.source:
            raise RLMSkillError("skill provenance source must not be empty")
        if (self.provenance.benchmark is None) != (self.provenance.delta is None):
            raise RLMSkillError("skill provenance benchmark and delta must be provided together")

    @property
    def content_hash(self) -> str:
        payload = json.dumps(
            self.as_dict(include_body=True, include_hash=False),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def as_dict(self, *, include_body: bool, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "id": self.skill_id,
            "revision": self.revision,
            "summary": self.summary,
            "model_families": list(self.model_families),
            "provenance": {
                "source": self.provenance.source,
                "benchmark": self.provenance.benchmark,
                "delta": self.provenance.delta,
            },
        }
        if include_body:
            value["body"] = self.body
        if include_hash:
            value["content_hash"] = self.content_hash
        return value


@dataclass(frozen=True, slots=True)
class RLMSkillCatalog:
    skills: tuple[RLMSkill, ...] = ()

    def __post_init__(self) -> None:
        skills = tuple(self.skills)
        keys = [(skill.skill_id, skill.revision) for skill in skills]
        if len(keys) != len(set(keys)):
            raise RLMSkillError("skill catalog contains duplicate id/revision pairs")
        object.__setattr__(self, "skills", skills)

    def list(self, *, model_family: str | None = None) -> tuple[RLMSkill, ...]:
        family = str(model_family or "").strip().casefold()
        return tuple(
            skill
            for skill in self.skills
            if not family or "generic" in skill.model_families or family in skill.model_families
        )

    def load(self, skill_id: str, revision: str | None = None) -> RLMSkill:
        matches = tuple(
            skill
            for skill in self.skills
            if skill.skill_id == skill_id and (revision is None or skill.revision == revision)
        )
        if not matches:
            raise KeyError(f"unknown RLM skill {skill_id!r}")
        if revision is None and len(matches) != 1:
            raise RLMSkillError(f"skill {skill_id!r} requires an explicit revision")
        return matches[0]


def parse_rlm_skill(text: str, *, source: str = "<memory>") -> RLMSkill:
    """Parse ``+++`` TOML frontmatter plus a Markdown body without I/O."""

    if not isinstance(text, str) or not text.startswith("+++\n"):
        raise RLMSkillError(f"{source}: skill must start with +++ TOML frontmatter")
    head, separator, body = text[4:].partition("\n+++\n")
    if not separator:
        raise RLMSkillError(f"{source}: skill frontmatter is not terminated")
    try:
        data = tomllib.loads(head)
    except tomllib.TOMLDecodeError as exc:
        raise RLMSkillError(f"{source}: invalid TOML frontmatter: {exc}") from exc
    expected = {
        "schema_version",
        "id",
        "revision",
        "summary",
        "model_families",
        "provenance",
    }
    missing = expected - data.keys()
    unknown = data.keys() - expected
    if missing or unknown:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            detail.append("unknown " + ", ".join(sorted(unknown)))
        raise RLMSkillError(f"{source}: frontmatter has " + "; ".join(detail))
    provenance = data["provenance"]
    if not isinstance(provenance, Mapping):
        raise RLMSkillError(f"{source}: provenance must be a table")
    provenance_unknown = provenance.keys() - {"source", "benchmark", "delta"}
    if "source" not in provenance or provenance_unknown:
        raise RLMSkillError(f"{source}: invalid provenance fields")
    families = data["model_families"]
    if not isinstance(families, list) or any(not isinstance(value, str) for value in families):
        raise RLMSkillError(f"{source}: model_families must be a string list")
    return RLMSkill(
        schema_version=data["schema_version"],
        skill_id=data["id"],
        revision=data["revision"],
        summary=data["summary"],
        body=body.strip(),
        model_families=tuple(families),
        provenance=RLMSkillProvenance(
            source=provenance["source"],
            benchmark=provenance.get("benchmark"),
            delta=provenance.get("delta"),
        ),
    )


def load_rlm_skill(path: str | Path) -> RLMSkill:
    artifact = Path(path)
    try:
        text = artifact.read_text(encoding="utf-8")
    except OSError as exc:
        raise RLMSkillError(f"cannot load skill {artifact}: {exc}") from exc
    return parse_rlm_skill(text, source=str(artifact))


@lru_cache(maxsize=1)
def load_builtin_skill_catalog() -> RLMSkillCatalog:
    root = files("droste").joinpath("skill_artifacts")
    skills = tuple(
        parse_rlm_skill(
            root.joinpath(filename).read_text(encoding="utf-8"),
            source=f"droste/skill_artifacts/{filename}",
        )
        for filename in _BUILTIN_SKILLS
    )
    return RLMSkillCatalog(skills)


def _schema(value: Mapping[str, Any], suffix: str) -> SchemaSpec:
    return SchemaSpec(value, JSON_SCHEMA_2020_12, f"droste:rlm-skills/{suffix}@1")


RLM_SKILLS_PROVIDER_MANIFEST = ProviderManifest(
    provider_type="rlm_skills",
    revision="1",
    operations=(
        ProviderOperation(
            operation_id="skills.list",
            binding_name="available",
            description="List versioned RLM strategy skills and their content hashes.",
            parameters=_schema(
                {
                    "type": "object",
                    "properties": {"model_family": {"type": ["string", "null"]}},
                    "additionalProperties": False,
                },
                "list/parameters",
            ),
            result=_schema({"type": "array", "items": {"type": "object"}}, "list/result"),
            pagination=PaginationMode.NONE,
            delivery=ResultDelivery.INLINE,
            budget_class="strategy.read",
        ),
        ProviderOperation(
            operation_id="skills.load",
            binding_name="load",
            description="Load one exact RLM strategy skill as Markdown.",
            parameters=_schema(
                {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                        "revision": {"type": ["string", "null"]},
                    },
                    "required": ["skill_id"],
                    "additionalProperties": False,
                },
                "load/parameters",
            ),
            result=_schema({"type": "object"}, "load/result"),
            pagination=PaginationMode.NONE,
            delivery=ResultDelivery.INLINE,
            budget_class="strategy.read",
        ),
    ),
)


def rlm_skills_provider(catalog: RLMSkillCatalog) -> ProviderRegistration:
    """Adapt immutable skill values to the generic capability broker."""

    if not isinstance(catalog, RLMSkillCatalog):
        raise TypeError("rlm_skills_provider requires an RLMSkillCatalog")

    def bind(source: ConfiguredSource, context: Any = None) -> ProviderRuntime:
        del source, context

        def list_skills(
            execution: CapabilityExecutionContext,
            model_family: str | None = None,
        ) -> list[dict[str, Any]]:
            execution.check()
            return [
                skill.as_dict(include_body=False)
                for skill in catalog.list(model_family=model_family)
            ]

        def load_skill(
            execution: CapabilityExecutionContext,
            skill_id: str,
            revision: str | None = None,
        ) -> dict[str, Any]:
            execution.check()
            return catalog.load(skill_id, revision).as_dict(include_body=True)

        return ProviderRuntime(
            handlers={"skills.list": list_skills, "skills.load": load_skill},
            source_description=(
                "Versioned, additive RLM strategy. Inspect metadata with `skills.available()` and "
                "load only the exact skill needed with `skills.load(skill_id, revision)`."
            ),
        )

    return ProviderRegistration(
        manifest=RLM_SKILLS_PROVIDER_MANIFEST,
        effects={"skills.list": SideEffect.READ, "skills.load": SideEffect.READ},
        binder=bind,
    )


__all__ = [
    "RLM_SKILL_SCHEMA_VERSION",
    "RLM_SKILLS_PROVIDER_MANIFEST",
    "RLMSkill",
    "RLMSkillCatalog",
    "RLMSkillError",
    "RLMSkillProvenance",
    "load_builtin_skill_catalog",
    "load_rlm_skill",
    "parse_rlm_skill",
    "rlm_skills_provider",
]
