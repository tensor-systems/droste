# Verifiers v1 × droste

*Analysis of where droste fits into Prime Intellect's
[verifiers v1](https://www.primeintellect.ai/blog/verifiers-v1) (2026-07).
Tracking issues: [#63](https://github.com/tensor-systems/droste/issues/63)
(harness adapter), [#64](https://github.com/tensor-systems/droste/issues/64)
(trainability).*

## What verifiers v1 is

Prime Intellect's rebuild of their RL/eval framework around three decoupled
layers:

- **Tasksets** — what the work is. A `TaskData` class of parameters plus a
  `Task` with `@vf.reward`-decorated scoring functions that examine the full
  trace. Harbor datasets (Terminal Bench 2, etc.) plug in with inherited
  rewards; NeMo Gym and OpenEnv have alpha support.
- **Harnesses** — the agent that does it. Tasksets are harness-agnostic by
  design: "any compatible harness, from a simple model call to a coding agent
  with subagents and compaction."
- **Runtimes** — where code executes. A minimal `run`/`read`/`write` class;
  subprocess and Docker locally, Prime/Modal sandboxes remotely.

The glue is an **interception server**: an HTTP proxy between the harness and
the inference API that records every request/response, can rewrite sampling
params and tool responses, and normalizes three dialects (OpenAI Chat
Completions, OpenAI Responses, Anthropic Messages) into canonical `vf.types`.
Pools of these servers scale elastically (~32 rollouts per server).

Traces are **DAGs, not transcripts**: each message is stored once; compaction
and subagent calls create branches; every root-to-leaf path is an independent
training sample. That gives linear rather than quadratic scaling on long
horizons, and — their words — makes "training across compactions and subagents
feasible, and unlocks long-horizon training past a model's context window."
Traces feed prime-rl directly; a `TrainClient` (built on their `renderers`
library) preserves token-in-token-out fidelity with exact token IDs and
logprobs, while `EvalClient` is a blind HTTP proxy for eval fidelity. Runs are
configured in TOML (model, sampling, taskset ID, harness version) and launched
via the `prime` CLI.

## Fit 1 — droste as a harness → #63

RLM is exactly the kind of non-obvious harness the taskset/harness split was
built for: instead of an agent loop, the root model writes code over the data
and spawns sub-calls from inside the REPL. Packaging droste as a verifiers
harness buys **independent, apples-to-apples benchmark numbers** for RLM vs.
plain agent loops on shared tasksets, plus entry into an ecosystem where new
tasksets keep arriving for free.

It should also be cheap. All LLM calls already flow through one client seam,
so "run droste against the interception server" is a base-URL and
client-selection change, not a new integration surface. The interception
server speaks Anthropic Messages and OpenAI dialects — both of which our
client layer already produces.

Things to watch (record findings on #63):

- **Sampling-param interception vs. our policy layer.** Verifiers rewrites
  sampling params in-flight; droste's policy layer also has opinions. Decide
  who wins and make it explicit.
- **Budget interaction.** Droste's budget object and verifiers' per-rollout
  limits are parallel mechanisms; the harness adapter should map one onto the
  other rather than running both blind.
- The blog describes the taskset ABI in detail but not the harness ABI —
  step 1 is pinning that down from `PrimeIntellect-ai/verifiers` source.

## Fit 2 — trainability: sub-calls as trace branches → #64

Verifiers' branching traces were designed for subagent calls, and droste's
REPL-spawned sub-calls are subagent branches. If droste rollouts land in
canonical trace format with the branch structure matching the sub-call tree,
prime-rl can train on them — "train a model to be a better RLM root model"
stops being hypothetical. This is the same thesis as
[principles.md](principles.md): mechanism (REPL, budget, recursion) scales
with compute; verifiers supplies the missing training loop.

Two prerequisites keep this honest:

- **Today's sub-calls are flat.** `llm_query` and its batch variants are
  plain LLM calls; a true recursive `rlm_query` (a child RLM loop) is
  designed but not implemented (#2). Branch mapping starts with what exists —
  `llm_query` sub-calls and compaction — and recursion deepens the tree when
  #2 lands, it doesn't gate the first version.
- **Branch identity.** Whether sub-calls made through the interception server
  are recorded as branches of the parent trace (vs. unrelated flat traces)
  likely hinges on how the adapter threads parent/session identity through
  sub-call requests. The NDJSON event vocabulary (#35) gives us droste-side
  correlation hooks either way.

Blocked on #63.

## Fit 3 — running behind a metered gateway

The interception server proxies to whatever compatible endpoint you
configure, so gateways compose with it:
`harness → interception server → gateway → provider`. Droste already runs
against [ModelRelay](https://modelrelay.ai), which adds what verifiers leaves
out — multi-provider routing through one key, per-run spend limits (RL
rollouts are enormous inference consumers, and verifiers has no metering
story), and quality presets that ride the model string, making a
sweep-across-tiers a one-line TOML change with zero harness changes.

One caveat for the training path: `TrainClient` needs exact token IDs and
logprobs, so any gateway in the chain must pass token-level response fields
and unrecognized sampling params through untouched. Two proxies in one
request path is also a real (if modest) latency tax — fine for evals, worth
measuring before long training runs.

## Non-fit (for now) — pyodide sandbox as a `Runtime`

The `Runtime` interface is small enough (`run`/`read`/`write`) that our
Deno+Pyodide substrate could implement it, and it would be the only option
that runs untrusted code on a stock Mac with no Docker daemon. But our sandbox
is scoped to per-query code execution, not general environment containers, and
nothing above depends on it. Possible later; not a reason to integrate.

## Sequencing

1. **#63** — thin harness adapter, one Harbor taskset, end-to-end eval.
   Everything else is contingent on what this run teaches us.
2. **#64** — branch-structure verification on a toy taskset, after #63.
3. Training runs and a `Runtime` impl — only if the above earn it.
