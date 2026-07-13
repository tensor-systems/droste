# Verifiers v1 × droste / ModelRelay

*Analysis of where our stack fits into Prime Intellect's
[verifiers v1](https://www.primeintellect.ai/blog/verifiers-v1) (2026-07).
Tracking issues: [#63](https://github.com/tensor-systems/droste/issues/63)
(harness adapter), [#64](https://github.com/tensor-systems/droste/issues/64)
(trainability), [modelrelay#1684](https://github.com/tensor-systems/modelrelay/issues/1684)
(proxy fidelity).*

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

## Fit 1 — droste as a harness (the big one) → #63

RLM is exactly the kind of non-obvious harness the taskset/harness split was
built for: instead of an agent loop, the root model writes code over the data
and recurses via `rlm_query`. Packaging droste as a verifiers harness buys:

- **Independent, apples-to-apples benchmark numbers** for RLM vs. plain agent
  loops on shared tasksets. This is the credibility artifact the public launch
  currently lacks — our numbers, on our tasks, convince nobody.
- Entry into an ecosystem where new tasksets keep arriving for free.

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

## Fit 2 — trainability: recursion as trace branches → #64

Verifiers' branching traces were designed for subagent calls, and droste's
recursive `rlm_query` calls *are* subagent branches. If droste rollouts land
in canonical trace format with the branch structure matching the recursion
tree, prime-rl can train on them — "train a model to be a better RLM root
model" stops being hypothetical. This is the same thesis as our
architecture-principles north star: mechanism (REPL, budget, recursion) scales
with compute; verifiers supplies the missing training loop.

The open question is whether sub-calls made through the interception server
get recorded as branches of the parent trace or as unrelated flat traces —
that likely hinges on how the adapter threads parent/session identity through
sub-call requests. The NDJSON event vocabulary (#35) gives us droste-side
correlation hooks either way. Blocked on #63.

## Fit 3 — ModelRelay as the inference endpoint → modelrelay#1684

The interception server proxies to whatever compatible API you configure, and
ModelRelay is one. The chain is
`harness → interception server → ModelRelay → labs`, and it adds three things
verifiers doesn't have:

- **Per-rollout budgets and metering.** RL rollouts are enormous inference
  consumers; verifiers has no billing story. Spend limits cover it today, and
  the x402 reservation→settle "upto" scheme (modelrelay#1655) is literally
  "reserve a budget for this rollout, settle actuals."
- **Presets as a sweep axis.** `preset:<code>/<role>` rides the model string,
  and verifiers configures the model per-run in TOML — sweeping an eval across
  quality tiers is a one-line config change, zero harness changes.
- **Multi-provider routing through one key**, where verifiers otherwise makes
  the harness author juggle provider keys.

The risk is fidelity: `TrainClient` needs exact token IDs and logprobs, so the
relay must pass token-level fields through byte-identical (both dialects,
streaming and unary), and must forward sampling params it doesn't recognize
rather than stripping to a known schema. This is also the same wire as the
in-boundary root model work (modelrelay#1680) — check they don't collide.
Both proxies in one request path is a real (if modest) latency tax during
training; acceptable for evals, measure before committing for RL.

Speculative but worth naming: Prime's ecosystem direction (environment hub,
third-party tasksets) creates a marketplace where environment authors need
metered, billable inference for judges and reward models — Build→Ship→Monetize
with "app developer" replaced by "environment author." Same shape as the x402
four-role analysis. No issue filed; revisit if #63 lands well.

## Non-fit (for now) — pyodide sandbox as a `Runtime`

The `Runtime` interface is small enough (`run`/`read`/`write`) that our
Deno+Pyodide substrate could implement it, and it would be the only option
that runs untrusted code on a stock Mac with no Docker daemon. But our sandbox
is scoped to per-query code execution, not general environment containers, and
nothing above depends on it. Possible later; not a reason to integrate.

## Sequencing

1. **#63** — thin harness adapter, one Harbor taskset, end-to-end eval.
   Everything else is contingent on what this run teaches us.
2. **modelrelay#1684** — passthrough test + fixes; independent of #63, small.
3. **#64** — branch-structure verification on a toy taskset, after #63.
4. Training runs, marketplace positioning, `Runtime` impl — only if the above
   earn it.
