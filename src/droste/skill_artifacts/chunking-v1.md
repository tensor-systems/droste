+++
schema_version = 1
id = "droste.chunking"
revision = "1.0.0"
summary = "Plan semantic map/reduce against both input and output limits."
model_families = ["generic"]

[provenance]
source = "droste"
+++
Inspect the symbolic context first. Keep bulk data in Python variables, use
deterministic code to narrow it, and send semantic work through bounded
`llm_query` calls. Batch independent chunks when their expected responses fit
the per-call output limit. Reduce partial results in code or a targeted final
subcall; do not print the bulk intermediate state into root history.
