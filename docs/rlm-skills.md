# RLM skills

RLM skills are versioned, additive strategy documents the root model may inspect
and load during a run. They are not executable, are not subcalls, and do not
replace prompt packs. A prompt pack is the single resolved harness contract;
skills are optional data reached through the same broker as other read-only
providers.

Each artifact is Markdown with strict `+++` TOML frontmatter:

```toml
+++
schema_version = 1
id = "acme.chunking"
revision = "1.0.0"
summary = "Choose chunks that fit semantic and output limits."
model_families = ["generic"]

[provenance]
source = "acme"
+++
Keep bulk data in Python and print only bounded summaries.
```

Required fields, provenance fields, and model-family selectors fail closed.
Every parsed `RLMSkill` has a canonical `content_hash`. A `generic` skill is
eligible for every named family; a family-specific skill is never selected for
another family. Loading an ID with multiple revisions requires an exact
revision.

Droste ships `droste.chunking@1.0.0` and a removable decomposition example.
They make no benchmark claim. Hosts may load filesystem artifacts with
`load_rlm_skill` or compose immutable values in an `RLMSkillCatalog`.

## Broker wiring

```python
from droste import (
    ConfiguredSource, ProviderCatalog, load_builtin_skill_catalog,
    rlm_skills_provider,
)

registration = rlm_skills_provider(load_builtin_skill_catalog())
registry = ProviderCatalog((registration,)).bind(
    (ConfiguredSource("skills", "rlm_skills"),)
)
```

When that registry is used to create the run environment, generated code sees:

```python
items = skills.available(model_family="qwen")
skill = skills.load("droste.chunking", "1.0.0")
```

The stable provider operations are `skills.list` and `skills.load`; the Python
binding uses `available` because `list` is a Python builtin. Both operations are
read-only and their normal capability events record identity/status without the
skill body. Hosts decide whether the provider is present. Droste never
auto-registers it or silently adds skill text to a prompt.

