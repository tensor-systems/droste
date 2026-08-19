# Prompt packs (schema v2)

A prompt pack is the complete deterministic harness strategy for one run.
Droste resolves exactly one pack before inference and never merges packs or
changes strategy mid-run.

The stable template slots are:

| Slot | Value |
| --- | --- |
| `capabilities` | Generated model-visible bindings |
| `budget` | Resolved compute and output limits |
| `question` | Caller question |
| `history` | Bounded loop transcript |
| `output_contract` | Required model response form |

Missing or unknown slots fail validation. A pack contains all initial,
refinement, repair, extraction, and transcript-elision templates plus policy
defaults and provenance. Parsing, validation, selection, hashing, and rendering
are pure; only loader functions read files or package resources.

Resolution checks caller bindings, consumer model-family and generic bindings,
engine model-family bindings, then the generic fallback for the requested
profile. Built-in profiles are `full`, `minimal`, and `none`. Selection records
the tier and canonical content SHA-256 in results and the scaffold manifest.

Use `parse_prompt_pack()`, `PromptPackCatalog`, and `resolve_prompt_pack()` for
custom packs. A declared `content_sha256` must equal the hash of the parsed
content; a revision label cannot conceal a content change.

## RLM skills (schema v1)

RLM skills are optional, immutable Markdown-plus-TOML strategy artifacts. They
are separate from prompt packs: loading a skill never changes the run's
deterministic prompt-pack contract.

A skill declares `schema_version`, `id`, `revision`, `summary`,
`model_families`, provenance, and a non-empty Markdown body. Built-in skills are
loaded through the explicitly registered read-only `rlm_skills` provider. Its
`list` operation returns metadata; `load` returns one exact body and content
hash. No skill provider or prompt text is installed automatically.

Skill bodies are configurable content. Durable capability events record only
the provider identity and outcome, never the body.
