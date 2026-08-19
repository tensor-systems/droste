# Deno and Pyodide substrate

The bundled relay runs Droste in CPython on WASM. Deno owns network, process,
credential, and provider-file authority; generated Python receives only
brokered model and provider calls.

Relay sources ship in the wheel under `droste/substrates/_relay`. Stage the
files from the installed package so the engine and relay versions stay pinned:

```bash
relay_dir="$(droste relay-path)"
cp "$relay_dir"/*.ts <build-staging-directory>/
```

The working host adapter is
[`examples/pyodide-host/pyodide_host_adapter.py`](../examples/pyodide-host/pyodide_host_adapter.py).

## Host adapter contract

The trusted launcher selects an importable adapter module. Request JSON cannot
select code. The module exposes:

```python
def build_db_service(db_path, contacts_db_path=None) -> tuple[ProviderService, dict]:
    ...

def run_for_host_pyodide(
    request, host_fetch, bridge_call, duplex_bridge_call=None, meta=None
) -> dict:
    ...
```

`build_db_service` runs in the trusted provider interpreter and returns one
owned service plus opaque JSON metadata. `run_for_host_pyodide` runs in the
generated-code interpreter. It must use `EnvironmentConfig(kind="pyodide")`,
declare `host_managed_timeout=True` and `host_managed_isolation=True`, and set
the Python execution timeout to zero. Those declarations verify host wiring;
the launcher must still enforce the WASM boundary and hard deadline.

A database-backed adapter passes a verified provider through `BridgeProvider`.
The bundled relay supplies the bridge-v2 duplex pump for mid-call checkpoints
and cancellation acknowledgement. The generated-code interpreter never opens a
database or host path directly.

`host_fetch` is the only model-network seam. The bundled broker restricts it to
ModelRelay Responses endpoints and injects credentials host-side. Supporting a
different backend requires a host-specific broker/client implementation.

## Process contract

Launchers provide a dedicated writable event descriptor, fd3 by convention:

```bash
DENO_EXTRA_STDIO_FDS=3 DROSTE_RELAY_EVENT_FD=3 \
  deno run --allow-net=api.modelrelay.ai --allow-read --allow-env \
  relay.ts <sources> <adapter_module> 3>events.ndjson
```

The three output lanes are independent:

| Lane | Content |
| --- | --- |
| fd1 | Exactly one adapter-owned JSON response line |
| configured descriptor | Trace ABI v10 NDJSON only |
| fd2 | Diagnostics only |

External launchers must include the event descriptor in
`DENO_EXTRA_STDIO_FDS`; inheriting fd3 at the OS level alone is insufficient.
Descriptors 0–2, missing markers, malformed values, and unwritable channels
fail before Pyodide work with `RelayEventChannelError`. Preflight and
pre-admission refusal intentionally emit no event frames.

Drain fd2 and the event descriptor concurrently. A hard stop may leave a valid
nonterminal prefix. Treat an incomplete final frame as a transport error and do
not fabricate `done`.

## Security boundary

The provider interpreter owns corpus files and live provider runtimes. The Deno
broker owns model credentials. The generated-code interpreter has neither. Its
only external effects are the model and provider calls that those trusted
components admit.

`RawExecutor` intentionally runs ordinary Python because WASM is the isolation
boundary. `SandboxLimits` still bounds captured output, while the host owns the
wall-clock kill. Under Pyodide, SQLite's thread-based per-query timer is not
available; host termination is the final timeout.

## Tests

Run the substrate suite:

```bash
deno test --allow-read --allow-env --allow-ffi pyodide/
```

On a cold cache, allow network access to `cdn.jsdelivr.net` and writes to the
Deno cache so Pyodide can fetch its SQLite wheel. Run the complete relay and
adapter path with:

```bash
deno test --allow-run --allow-read --allow-write \
  --allow-net=127.0.0.1 --allow-env examples/pyodide-host/e2e_test.ts
```

The end-to-end test spawns the real relay, imports the real adapter, executes in
real Pyodide, queries a temporary SQLite database through the broker, and uses a
loopback model server. Python-side substrate tests run with `uv run pytest`.

## Limitations

- Unary-only custom provider bridges cannot receive a new soft-cancellation
  request during a synchronous remote call; the host still needs a hard
  timeout.
- SQLite per-query timers are unavailable under Pyodide.
- The bundled model broker is ModelRelay-specific.
