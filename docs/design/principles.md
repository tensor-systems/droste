# Architecture principles

*The north star for droste's evolution. `architecture.md` describes what is; this
describes what everything should converge toward, and the test every new piece of
structure must pass. Grounded in the 2026-07 architecture review and its
topology spike.*

## The test: structure must be removable

Sutton's Bitter Lesson, applied as an engineering discipline rather than a slogan:
the only things that scale with compute are **search** and **learning**. Structure
that encodes *how humans think the problem should be solved* buys a short-term win
and becomes a ceiling. The discipline is not "no structure" — it is:

> Every piece of human strategy in the system must be **deletable without a
> rewrite** the day a model no longer needs it.

RLM itself passes this test: it replaces retrieval theory (embeddings, chunking
heuristics, rerankers, orchestration graphs) with a REPL — **code is the action
space**, and the model searches the problem by writing and running programs. The
structure we do ship divides into two kinds:

- **Mechanism** — structure that scales *with* compute: the REPL, the budget, the
  capability protocol. More compute → more code, deeper recursion, wider search.
  This is the invariant core, and it is deliberately tiny.
- **Strategy** — structure that substitutes human judgment *for* compute: prompts,
  tips, policy defaults, model pairings. All of it lives in a swappable data layer,
  ships as removable defaults, and is expected to shrink toward empty as models
  improve.

The failure modes this guards against are all live in the current code: a validator
that is convention rather than boundary (since hardened by the A′ sandbox split),
recursion capped at depth 1 by an
architectural constant (#2), and orchestration strategy welded into the engine as
Python string constants (`prompts/tips.py`).

## The stack

```
┌────────────────────────────────────────────────────────────────┐
│  PROMPT PACKS + RLM SKILLS          strategy-as-data, swappable │
├────────────────────────────────────────────────────────────────┤
│  PROVIDERS            open edge: in-proc · MCP · HTTP transports │
├────────────────────────────────────────────────────────────────┤
│  BROKER            budget ledger · trust boundary · MCP client  │
├────────────────────────────────────────────────────────────────┤
│  BRIDGE                       call / emit · JSON-RPC 2.0        │
├────────────────────────────────────────────────────────────────┤
│  REPL                    Python · persistent namespace · WASM   │
└────────────────────────────────────────────────────────────────┘
   invariant core (bottom three) · open ecosystem · deletable top
```

## 1. One capability protocol, location- and language-transparent

`query()`, `search()`, `llm_query()`, a future `rlm_query()`, host tools, and
partner APIs are all the same thing: **untrusted code emitting a capability
request, and something trusted answering it.** One wire shape (JSON-RPC 2.0 with a
`register` step; descriptors shaped like tool schemas, because models are trained
on that shape). The sandbox cannot tell — and must not care — whether the provider
answering is:

- a second interpreter in the same process (an embedded data-source service),
- host code in the embedding app (Swift, Go, anything),
- a remote HTTP API (`wrapper_v1` demotes to exactly this — a remote provider),
- **another RLM** (recursion is a provider that answers by running a child loop).

Two primitives cross the boundary, nothing else:

```
bridge.call(method, params) -> result     # sync from Python; request/response
bridge.emit(event)                        # fire-and-forget; the event stream (#1)
```

**Why this is not tool calling reinvented:** provider tool calling is
*decoder-mediated orchestration* — every call routes through the model's decoder,
every result lands in context, and loops/joins/batching live in the token stream,
the most expensive and least reliable computer available. Droste is
*code-mediated orchestration*: the model writes a program; the program does the
cheap deterministic control flow — loops, joins, batching, local reduction — and
the model re-enters only when judgment is needed or the reduced result matters.
Many calls happen inside one execution step, and only the reduced output (plus
traces, errors, and refinement feedback) returns to the model. The win is where
orchestration lives, not a free lunch on context. Same descriptor vocabulary,
inverted mechanics — and we converge on tool calling's *schema* precisely so we
can ride its ecosystem (next section) without adopting its economics.

## 2. The capability ABI is ours; MCP is the default external transport

The enduring droste idea is **model-written programs over brokered capabilities
with bounded semantic subcalls and reduced feedback** — not any particular
protocol. The stable abstraction is internal:

```
capability manifest → Python binding → brokered operation call → trace/budget ledger
```

Transports *implement* that abstraction behind the broker, and adding one must
never force a core rewrite:

- **in-process** (direct call, no serialization) — first-class, the hot local
  path: a host's embedded data source, the SQL/fs sources.
- **MCP** — the *default external* transport: the broker acts as an MCP client,
  so a data source / host tool / sub-RLM can be an MCP server in any language.
- **HTTP** (`wrapper_v1` demotes to this) — remains possible; keeps its
  budget/SSRF guards, relocated into the broker.

What the MCP default buys: **every existing MCP server is a droste data source
for free** — "integrate droste" becomes "point it at your MCP server" — and the
registry stops being a Python-only abstraction: manifests auto-generate both
the sandbox bindings *and* the prompt's `{capabilities}` fragment, so the prompt
can never lie about the API surface. But MCP is how capabilities are *populated*,
not what droste *is*.

The security posture is ours regardless of transport: **the REPL never speaks
MCP.** The sandbox calls generated Python functions; those functions go through
the broker, which owns auth, policy, budgets, validation, side-effect gating, and
trace emission — MCP assumes a trusted caller, and ours is not.

A network boundary and a trust boundary are the same shape. Making capabilities
location-transparent *is* the sandbox split: the untrusted REPL holds no DB
handle, no credential, no raw tool — only the ability to make requests.

## 3. Define little, but define it sharply: the five ABIs

"Define as little as possible" is half the discipline; the other half is that what
we *do* define, we define precisely — these are the boundaries products will
otherwise reinvent, badly and divergently. Five, each small, versioned, and
consumer-validated before frozen:

1. **Kernel ABI** — the REPL contract: the `answer` dict, stdout-as-feedback,
   persistent state across iterations, reserved globals.
2. **Capability ABI** — the manifest (tool-shaped descriptors), Python binding
   generation, parameter/result schemas, and **side-effect metadata**
   (read-only vs effectful — the generalization of today's hard-coded
   SqlValidator gate, and the hook confirmation policies hang from).
3. **Broker ABI** — `call`, `emit`, the budget ledger, policy/confirmation hooks.
4. **PromptPack ABI** — the stable slots, `(model, profile)` resolution,
   provenance fields.
5. **Trace ABI** — the run record: code executed, outputs, capability calls,
   usage, policy events, final answer. This unifies what already exists piecemeal
   (`trajectory`/`IterationRecord`, `retrieved_guids`, the #1 event stream) into
   one named boundary — it is what benchmarking, citations, replay, and billing
   all consume, and none of them should parse ad-hoc internals.

## 4. Compute is a budget, not an architectural constant

`max_depth = 1` is a hard cap on how much compute the method may spend on a hard
problem — exactly the ceiling the Bitter Lesson forbids. The paper's OOLONG-Pairs
ablation (58 → 76 at depth 3) is the lesson restated in our own benchmark: more
search buys more accuracy, and a depth-1 architecture forbids the spend even when
it would pay.

The replacement is **one budget object**, set by the caller, metered by the broker:

```json
{ "tokens": 500000, "subcalls": 100, "depth": 3, "wall_ms": 300000 }
```

Depth, breadth, and iteration count stop being engine constants and become
emergent consequences of how much compute the caller authorized. Child runs
(`rlm_query`) draw from the **parent's** ledger — recursion is metered, not gated.
This reconciles Sutton with the P&L: the architecture has no ceiling on
intelligence; the *invoice* has a caller-set ceiling with conservative defaults.
Scaling a hard problem means handing it a bigger budget, nothing else.

## 5. Strategy is data: prompt packs + RLM skills

All human strategy lives above the engine in two artifact kinds, both versioned
data, neither code:

**Prompt packs** — deterministic harness configuration, resolved at run start by
`(model, profile)` with a fallback chain (caller pack → product pack → droste
default for the model family → generic). A pack carries the base prompt, the
refinement/repair/extract templates, and policy defaults. The **slot contract is
the only stable API**: the engine guarantees each template a fixed variable set —
`{capabilities}` (generated from the provider registry, never authored),
`{budget}`, `{question}`, `{history}`, `{output_contract}`. Content may change
wildly between packs; slots may not. Packs are exclusive — one governs a run —
because they carry the machinery that must be deterministic (the repair path fires
when the model is already misbehaving; that is the worst moment for model-chosen
strategy).

**RLM skills** — additive, composable strategy in the skills format the agent
ecosystems (Claude Code and similar agent tools) already converged on: markdown +
frontmatter. The difference in genre: agent skills teach *tool workflows*; RLM
skills teach *code-writing under a metered budget* — chunking budgets,
fat-prompts-small-batches, "string matching finds WHERE, llm_query understands
WHAT." Today's `tips.py` profiles become the first skills. Two properties agent
ecosystems lack:

- **Benchmarked**: every skill can carry a provenance line — which bench, what
  delta. "This skill helps" is a measured claim.
- **Loadable mid-run through the bridge**: a `skills.*` provider lets the model
  fetch strategy when it discovers it needs it, costed against the same budget.
  Progressive disclosure fits RLM economics better than agent sessions: root
  tokens are paid every iteration, so strategy you didn't need is strategy you
  shouldn't have loaded.

The exit ramp is built in: the pack for a strong-enough model is three lines
("REPL. Capabilities: `{capabilities}`. Budget: `{budget}`.") and the default
skill set shrinks toward empty. Deleting strategy is swapping a data file.

An embedder's own extension formats (connectors, recipes) map onto this substrate:
connectors are providers, recipes are RLM skills. A product-specific extension
format is a branded profile of the packs/skills mechanism, not a parallel system.

## 6. Trust topology

Settled by spike (2026-07-06): the untrusted REPL and the trusted side are
separate interpreter contexts bridged by `call`/`emit` (Option A, phased **A′ →
A″**; a second Pyodide context costs ~+42 MB, not 2×). One rule governs every
future topology change, including nested recursion and the Worker evolution:

> **Never re-enter a suspended interpreter. Every concurrently-active role gets
> its own context.**

## Non-goals and cautions

- **Don't over-unify.** The pull to make everything a skill (base prompt, repair
  templates) trades determinism away where the harness needs it most. Two layers —
  one deterministic, one discoverable — is the right amount of structure.
- **Don't strip strategy prematurely.** Today's models measurably need the tips;
  the discipline is removability, not asceticism. Ship defaults; measure; shrink.
- **The engine stays intent-blind.** It never infers what the caller wants from
  question text (existing policy-hints rule). Callers pass policy; packs carry
  defaults; the core enforces mechanics only.
- **Python stays the REPL language** — not from partisanship but because it is the
  language models know most deeply, which is itself a Bitter Lesson call: use the
  substrate the model already learned rather than inventing a DSL it must be
  taught.

## Where the work lands

| principle | issue |
|---|---|
| bridge (`call`/`emit`, JSON-RPC 2.0, register) + A′ split | shipped (Pyodide substrate); broker generalization: #9 |
| providers are MCP; registry unification; wrapper_v1 demotion | shipped; MCP spike: #5 |
| one budget object; recursion as metered provider | #4 (budget) + #2 (recursion) |
| event stream over `emit` | #1 |
| prompt packs + RLM skills | #3 |
