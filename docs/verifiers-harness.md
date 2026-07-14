# Verifiers v1 harness

The optional `droste_verifiers` package exposes exactly one Verifiers v1
`Harness`: `DrosteHarness`. Install it with the matching Droste release:

```bash
uv add 'droste[verifiers]==<version>'
```

The extra pins `verifiers==0.2.0`; its upstream dependency markers currently
make the harness available on Python 3.11–3.13. Core Droste supports Python
3.11–3.14. On 3.14 the optional harness is not installed and its tests skip
cleanly until upstream support lands. The harness itself creates a PEP 723
program that pins the same exact Droste version inside the selected Verifiers
runtime. Runtime preparation remains Verifiers' content-addressed/cached
responsibility.

A minimal Verifiers config is:

```toml
num_tasks = 1

[taskset]
id = "gsm8k-v1"

[harness]
id = "droste_verifiers"
droste_version = "<same installed version>"
depth = 0
data_paths = ["."]
prompt_profile = "minimal"
token_budget = 20000
max_subcalls = 0
wall_ms = 60000
root_output_tokens = 2048
subcall_output_tokens = 1024
```

`depth = 0` disables semantic subcalls and is useful for a zero-subcall control.
`depth = 1` enables Droste's current flat `llm_query`/batch calls up to
`max_subcalls`; recursive child RLM loops are not implied. `data_paths` are
runtime-local symbolic handles passed to the Droste CLI. Their contents are not
copied into the root prompt.

Both root and subcall clients use the Verifiers interception endpoint and bearer
secret. The model ID and the canonical sampling object from `ModelContext` are
recorded in Droste's scaffold manifest. Optional `root_revision`,
`subcall_revision`, and `seed` should be resolved by the experiment controller.
Task system prompts are deterministically folded into the single task question,
matching the Verifiers base harness behavior. Verifiers MCP URLs are refused
until Droste can project them through its brokered capability ABI.

The harness records `droste_iterations`, token/subcall counters, successful
subcalls, returned stdout characters, terminal readiness/extraction, and
configured depth as metrics. Task rewards remain taskset-owned.

## Trace-branch limitation

Droste emits content-free capability events with a unique call ID and root,
parent, and optional child run IDs. Verifiers v1's public Harness/interception
API records the intercepted root and subcall requests, but currently exposes no
supported field or callback through which a harness can attach Droste's call ID
as the canonical graph parent. Requests are therefore distinguishable as
intercepted message branches, while an exact Droste-call-to-Verifiers-node join
is not yet portable.

The harness intentionally does not depend on an internal header or private
trace mutation method. Exact canonical branch correlation needs a supported
Verifiers parent/correlation seam; Droste's Trace ABI already preserves the
identity needed to use it when one exists.
