# Unified Data Sources — design spec

**Status:** draft / for review
**Issue:** tensor-systems/rlm-core#9
**Consumers blocked on this:** modelrelay/modelrelay#1553 (SQL), #1561 (filesystem), epic #1556

This spec does two things:
1. Draws the boundaries between **Cozy**, **rlm-core**, and **ModelRelay** (shared context).
2. Specifies how to **unify the two parallel "data source" mechanisms** into one abstraction so SQL and filesystem sources become straightforward.

---

## 1. Boundaries: Cozy / rlm-core / ModelRelay

### Who owns what

| Layer | Repo | Language | Owns | Explicitly does NOT own |
|---|---|---|---|---|
| **rlm-core** (engine) | `tensor-systems/rlm-core` | Python (+ Pyodide/Deno substrate) | The RLM loop (`run_rlm`), the code-exec sandbox, the protocols (`RLMEnvironment`, `DataSource`, `LLMClient`, `SubcallClient`), the `DataSourceRegistry`, the `rlm_runner` entrypoint | Provider API keys, billing, auth, DB drivers, persistence. **LLM calls go *out* via host-provided HTTP endpoints** (`root_endpoint`/`subcall_endpoint` + token) — the engine never holds keys. |
| **ModelRelay** (commercial host) | `modelrelay/modelrelay` | Go | Embeds rlm-core (vendored in `platform/rlmrunner/assets`, synced via `scripts/sync-rlm-core.sh`); drives the runner; **provides the LLM endpoints the runner calls back into** (multi-lab routing + managed keys); billing/metering/PAYGO; auth/customer tokens; Stripe Connect; hosted sandbox fleet; `wrapper_v1` *server*; `/rlm/execute` + `/rlm/context`; observability; VPC/compliance | The RLM algorithm itself (delegated to rlm-core) |
| **Cozy** (consumer app) | `tensor-systems/cozybot` | Swift/macOS + Python | Runs rlm-core **in-process, on-device**; implements its **own concrete `DataSource`** over a local SQLite `chat.db` (iMessages); is a **ModelRelay customer** for inference + billing | The engine (uses rlm-core), inference (uses ModelRelay) |

### The one-sentence version
**rlm-core is the loop + the contracts (no keys, no DB, no billing). ModelRelay is the paid host that supplies inference + billing + remote data + hosted execution. Cozy is a consumer that runs the engine locally with a local data source and rents inference from ModelRelay.**

### Diagram

```mermaid
flowchart TB
    subgraph engine["rlm-core (open engine)"]
      loop["RLM loop · sandbox · protocols<br/>DataSourceRegistry · rlm_runner"]
    end

    subgraph mr["ModelRelay (commercial host)"]
      api["/rlm/execute · /rlm/context"]
      infer["multi-lab inference<br/>(managed keys, routing)"]
      bill["billing · auth · Stripe Connect"]
      wrap["wrapper_v1 server (remote data)"]
      mrEngine["embedded rlm-core"]
      api --> mrEngine
      mrEngine -. "LLM calls (root/subcall endpoints)" .-> infer
      api --> bill
    end

    subgraph cozy["Cozy (consumer app, on-device)"]
      cozyEngine["embedded rlm-core"]
      cozyDS["local DataSource<br/>(SQLite chat.db)"]
      cozyEngine --> cozyDS
    end

    mrEngine -. embeds .-> engine
    cozyEngine -. embeds .-> engine
    cozyEngine -. "inference + billing (customer)" .-> mr
```

### "Runs where the data is" — three deployment modes (same engine)

| Mode | Engine runs | Data source | Inference | Example |
|---|---|---|---|---|
| **On-device** | user's machine | in-process, local (SQLite) | ModelRelay | **Cozy** today |
| **Hosted** | ModelRelay cloud | `wrapper_v1` (remote HTTP) | ModelRelay | `/rlm/execute` today |
| **In-boundary / VPC** | customer's VPC | in-process (SQL/fs next to the data) | ModelRelay | the SQL/fs target (#1553/#1561) |

The point of this spec: the **in-boundary mode** needs in-process SQL/fs data sources, and today the hosted runner can only do `wrapper_v1`. That's the gap.

---

## 2. Problem: two things both called "data source"

Today there are **two parallel, non-interoperating mechanisms**:

1. **`wrapper_v1`** — `rlm_runner/runner.py::DataSourceWrapper`. A thin **remote HTTP** client (`/search`, `/get`, `/content`) with budgets + SSRF guards. The runner special-cases the request's `data_source` dict and injects flat globals `data_source_search` / `data_source_get` / `data_source_content`. **This is the only path the runner wires.**
2. **`DataSource` protocol + `DataSourceRegistry`** — `rlm_core/protocols/data_source.py` + `registry.py`. A rich **in-process** abstraction: `capabilities` (`sql`/`search`/`get`) + `query(sql)` / `search` / `find` / `get` / `get_schema` / `get_stats` / `sample` / chat helpers. Exposed as **namespaced** globals (`env[name] = {query, search, ...}`, optionally flattened for a default source). **Exported for consumers (Cozy builds one); the runner never touches it; nothing in rlm-core constructs it.**

Consequences:
- The richer abstraction (with `query()` for SQL, `get_schema()` for introspection) is **unreachable through the hosted runner**.
- "Data source" is ambiguous — a remote HTTP contract *and* an in-process protocol.
- SQL/fs sources have nowhere to plug into the hosted path.

---

## 3. Design: one abstraction, multiple transports

**Principle:** `DataSource` (the protocol) is the **single** "data source" concept. Everything is a `DataSource`; transports differ.

- **In-process transport** — a Python `DataSource` object (local/VPC/embedded). Used by Cozy, and by the future SQL/fs sources.
- **Remote transport** — HTTP. **`wrapper_v1` becomes a `DataSource` implementation** (`WrapperV1DataSource`) that proxies `search`/`get`/`content` to a partner API. It is *registered like any other source*, not a parallel path. Its budgets/SSRF guards move with it.

### 3.1 Make the runner use the registry

Replace the `wrapper_v1` special-casing in `run()` with: build a `DataSourceRegistry` from the request, then merge `registry.globals()` into the environment globals and use `registry.prompt_fragment()`.

```python
# runner.run(), replacing the DataSourceWrapper special-case
sources = build_data_sources(request)                 # list[DataSource]
registry = DataSourceRegistry(sources, default_source_name=request.get("default_source"))
environment = RunnerEnvironment(context=context, subcalls=subcalls, ...)
environment.merge_globals(registry.globals())          # namespaced + default-flattened
environment.set_data_prompt(registry.prompt_fragment())
```

`RunnerEnvironment` drops its `data_source`-dict handling; data-source globals/prompt now come from the registry. (`RLMEnvironment` protocol unchanged: `capabilities`/`globals`/`prompt_fragment`/`execute`/`close`.)

### 3.2 `WrapperV1DataSource` (the remote adapter)

Wrap the existing `DataSourceWrapper` HTTP logic in the `DataSource` protocol:

```python
class WrapperV1DataSource:                 # implements DataSource
    def name(self): return self._name      # default "wrapper"
    def capabilities(self): return {"search": True, "get": True}
    def get_schema(self): return "Remote wrapper_v1 source (search/get/content)."
    def search(self, query, filters=None, page=None): ...   # → HTTP /search
    def get(self, id): ...                                   # → HTTP /get
    def content(self, id, format="text", max_bytes=None): ...# → HTTP /content (extra method, exposed via hasattr)
```

The registry already exposes `search`/`get` by capability and any extra method (`content`) via the `hasattr` path.

### 3.3 Request shape (new)

```jsonc
{
  "data_sources": [
    { "type": "wrapper_v1", "name": "partner", "base_url": "...", "token": "...",
      "allowed_hosts": [...], "limits": { "max_requests": 20, "max_response_bytes": 1048576 } },
    { "type": "sql", "name": "db", "dsn": "postgres://readonly@db/app", "policy_ref": "..." },
    { "type": "fs",  "name": "vault", "root": "/data/vault", "glob": "**/*.md" }
  ],
  "default_source": "db"        // optional: flattens that source's methods to top-level globals
}
```

Singular `data_source` may remain as sugar for a one-element list during migration, or be dropped (see §6 — clean break is acceptable).

### 3.4 Who constructs SQL/fs sources? (the key decision)

rlm-core defines the protocol but ships **no concrete SQL/fs** source. Two options:

- **Option A — rlm-core ships reference `sql` + `fs` DataSources.** `build_data_sources()` maps `type → class`. *Pro:* batteries-included; any consumer gets SQL/fs. *Con:* the engine takes on DB drivers / filesystem concerns — heavier, against the "minimal embeddable engine" positioning.
- **Option B — keep the engine minimal; consumers inject sources via the `adapter_module` seam.** The runner already supports `request.adapter_module` → delegates to a consumer-provided `run(request)`. ModelRelay (and Cozy) implement their own `sql`/`fs` DataSources and build the registry in their adapter. *Pro:* engine stays protocol-only (on-brand "Flask, not Django"); DB drivers/policy live with the consumer that owns security. *Con:* each consumer implements the sources.

**Recommendation: Option B as the contract, with a thin Option-A convenience later.** Do §3.1–3.3 (registry-through-runner + `WrapperV1DataSource` + request shape) in rlm-core now; let ModelRelay implement `sql`/`fs` DataSources behind its adapter (where `sqlvalidate`/path-policy already live). If a second consumer wants them, promote reference impls into an **optional** `rlm_core.sources` module — keeping the base engine minimal. This keeps DB drivers and security policy in the consumer that owns the boundary, which is also the right place for them (see §4).

---

## 4. Security / policy boundaries

Each transport owns its guardrails — and they live with whoever owns the boundary, **not** the core loop:

- **wrapper_v1**: keeps existing per-request **budgets** (`max_requests`, `max_response_bytes`, `timeout_ms`) — these live in `DataSourceWrapper` and move into `WrapperV1DataSource`. **Correction:** SSRF / `allowed_hosts` enforcement does **not** live in rlm-core — `DataSourceWrapper._call()` only checks `base_url`/`token` then calls `urlopen`. The host allowlist + network policy is enforced by **ModelRelay's Go layer** (`cmd/modelrelay-api/server/rlm_datasource.go`) before the request reaches the runner. `allowed_hosts` in the config is descriptive (surfaced in the prompt); the engine does not gate on it. Either keep SSRF in the host (current) or add it to `WrapperV1DataSource` — but the spec must not claim the engine enforces it.
- **sql**: read-only, SELECT-only, **table/column allowlists** (note: `sqlprofiles.Policy` has **no row-level predicate** today — row/tenant scoping must be added before claiming it) — enforced by ModelRelay's `sqlvalidate`/`sqlprofiles` **inside its SQL `DataSource.query()`** before execution. rlm-core never sees credentials.
- **fs**: read-only, root/path allowlist, glob scoping — enforced in ModelRelay's fs `DataSource`.

This is why Option B is the safer default: **the engine stays credential- and policy-free**; the consumer that holds the data/connection also holds the policy.

---

## 5. Capability → sandbox surface

`DataSourceRegistry.globals()` (implemented in #11):
- `env["<name>"]` → an **attribute-accessible namespace object** (`SimpleNamespace`) with the methods the source's `capabilities()` enable: `search`, `query` (if `sql`), `get`, `get_recent`, `get_schema` (if `schema`), `get_stats`, plus `content`/`find`/`sample`/chat helpers via `hasattr`. **Fixed in #11:** the namespace was previously a plain `dict`, so `db.query(...)` raised `AttributeError` (only `db["query"]` worked) — it is now an attribute namespace so the documented `db.query(...)` API actually runs.
- If `default_source` matches, those methods are also flattened to top-level globals. Reserved names (`answer`/`context`/`llm_query`/…) and duplicate/unknown `default_source` are now rejected.

Prompt: `registry.prompt_fragment()` emits a `## Data Sources` section listing each source's `get_schema()` — replacing the hard-coded "Data source: wrapper_v1 …" line. Model guidance becomes accurate per source (e.g., `db.query("SELECT …")`, `vault.search("…")`).

---

## 6. Migration

rlm-core's stated principle is **no backward-compat shims**; ModelRelay has **zero users**. So: clean break.

1. **rlm-core**: land §3.1–3.3 + §3.2. Drop the flat `data_source_*` special-case (or keep only as a `default_source` convenience for a single `wrapper_v1`). Bump minor version.
2. **ModelRelay**: re-sync embedded rlm-core (`sync-rlm-core.sh`); update `platform/rlmrunner` `RunnerRequest` to emit `data_sources`; implement `sql`/`fs` DataSources behind its adapter; update `/rlm/execute` request docs + the data-source docs (`integrations/wrapper-v1.md` reframed as "the remote data-source transport").
3. **Cozy**: already builds a registry in-process — confirm it matches the (unchanged) `DataSourceRegistry` API. No transport change.

Note: main's embedded copy is currently one commit behind `0.2.2` (the `find()` helper, #7) — the re-sync in step 2 picks that up too. **This drift is the symptom §7 fixes.**

---

## 7. Engine integration contract (pinning + the adapter seam)

This migration assumes two things that aren't currently guaranteed: (a) the rlm-core embedded in ModelRelay actually *is* the engine this spec describes, and (b) the `adapter_module` seam can safely carry production data sources. Both need hardening **before** §3 lands, because §3 is what makes them load-bearing.

### 7.1 Pin the engine; don't rsync-vendor it

Today ModelRelay embeds rlm-core by `rsync -a --delete` from a loose sibling checkout (`scripts/sync-rlm-core.sh`), run **manually**. Failure modes this already produces:

- **Silent drift.** The embedded copy lags the real engine until a human remembers to re-run the script — exactly why main is "one commit behind `0.2.2`" (§6 note), and the divergence hit on the 1536 branch. Nothing flags it.
- **No provenance.** Nothing records *which* rlm-core commit is embedded. You can't answer "is the vendored tree the spec'd engine?" without a manual diff.
- **No enforcement.** `--delete` mirrors a mutable working tree; a hand-edit to the vendored copy (or an un-synced engine fix) is invisible to CI.

**Fix — pin + assert parity:**

1. **Record the pin.** Write the embedded engine's version/SHA next to the assets (e.g. `platform/rlmrunner/assets/RLM_CORE_VERSION` = a git tag or commit SHA), set by `sync-rlm-core.sh` when it syncs.
2. **CI parity gate.** A CI step re-runs the sync against the *pinned* ref into a temp dir and `git diff --exit-code` (or checksum) vs the committed assets. Fails the build if someone forgot to re-sync, hand-edited the vendored tree, or the pin and the tree disagree. This makes "embedded == spec'd engine" a *checked invariant*, not a hope.
3. **Sync from an immutable ref, not a working tree.** Have the script fetch a tagged rlm-core release (git tag, or a published artifact — see decision 5) rather than `../../tensor-systems/rlm-core` as it happens to sit on disk.

This is a process/CI change, not engine code — but it lives in this spec because §6 step 2 ("re-sync the embedded rlm-core") is only safe if re-syncing is verifiable and pinned.

### 7.2 Harden the `adapter_module` seam

Option B (§3.4, recommended) elevates `request.adapter_module → consumer.run(request)` from a convenience to **the official extension point** for SQL/fs sources. As-is it's a stringly-typed module path with an implicit `run(request)` contract and no validation — fine as an escape hatch, not fine as a load-bearing, security-relevant seam in a hosted runner. Before it carries production data sources it needs:

- **A declared interface.** A typed `Adapter` protocol with an explicit `build_data_sources(request) -> list[DataSource]` hook (composing with §3.1's registry construction), not a bare `run(request)` that re-implements the whole loop. The engine builds the registry; the adapter only supplies sources.
- **An import allowlist.** The hosted runner must not import an arbitrary, request-controlled module path — that's remote-code-execution-by-config, the code-side analogue of the wrapper's SSRF guard. Resolve `adapter_module` against a fixed allowlist (or a single configured adapter per deployment).
- **A compat check.** Version/handshake between adapter and engine so a stale adapter against a newer engine fails loudly, not subtly — the same drift problem as §7.1, one layer up.

**Sequencing clarification (avoid the contradiction):** #11 ships §3.1–3.3 (registry-through-runner + `WrapperV1DataSource` + request shape) **without** using the adapter for any production source — `sql`/`fs` types *raise* and route nowhere yet. So the seam is **not load-bearing in #11**, and #11 does not require the hardening. The hardening (typed `Adapter` protocol + import allowlist + compat check) is a **prerequisite for #1553/#1561** — i.e. before any SQL/fs source is actually constructed and run via `adapter_module` — not for the #11 plumbing. Land it with the first consumer adapter.

---

## 8. Sequencing

```
rlm-core#9 (this spec)
  ├─ registry-through-runner + WrapperV1DataSource + request shape   [rlm-core]
  └─ then, in modelrelay:
       ├─ #1553 SQL DataSource (sqlvalidate policy) + adapter wiring
       └─ #1561 fs/Markdown DataSource
```

#1553 and #1561 become **consumer-side `DataSource` implementations** once #9 lands — the hard, shared part is the registry-through-runner plumbing here (shipped in #11). §7 applies to the **modelrelay steps**: §7.1 pinning gates the re-sync (re-sync must be verifiable), and §7.2 adapter hardening must land **with the first SQL/fs source** that goes through `adapter_module` — not before #11's wrapper-only plumbing.

---

## 9. Open decisions (confirm before building)

1. **Option A vs B** for SQL/fs construction (recommended: **B** — engine stays minimal; consumers inject via adapter). 
2. **Keep singular `data_source`** as sugar, or require `data_sources` (clean break)?
3. **`content` verb**: keep as a wrapper-only extra method (via `hasattr`), or promote to a first-class capability in `DataSourceCapabilities`?
4. **Default-flatten behaviour**: keep top-level flattening for a `default_source`, or always namespace (clearer prompts, slightly more verbose model code)?
5. **Engine distribution** (§7.1): keep manual rsync-vendor (status quo), or pin a SHA + CI parity gate, or fetch a versioned artifact (git tag / PyPI)? Recommended: **pin + CI parity gate now** (cheapest, kills the silent drift), versioned artifact later.
6. **`adapter_module` hardening** (§7.2): keep the bare `run(request)` escape hatch, or require a typed `Adapter` protocol + import allowlist + compat check before Option B sources ship? Recommended: **typed protocol + allowlist**, since Option B makes this seam load-bearing and security-relevant.
