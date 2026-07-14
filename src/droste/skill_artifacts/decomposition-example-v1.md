+++
schema_version = 1
id = "droste.decomposition.example"
revision = "1.0.0"
summary = "An unrelated first-decomposition example kept as removable data."
model_families = ["generic"]

[provenance]
source = "RLM paper v3-inspired removable example; no benchmark claim"
+++
Example: for a corpus-wide classification, first inspect the record shape and
size. Partition records into batches whose expected labels fit the output
ceiling, ask the same explicit classification question for every batch, retain
the structured results in a Python variable, then aggregate and verify before
submitting the answer. The example is strategy, not a required execution path.
