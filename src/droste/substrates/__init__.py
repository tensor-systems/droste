"""Adapters that let the droste core loop run under alternative runtimes.

Each submodule targets one runtime (e.g. Pyodide/WASM) and stays
host-agnostic: it knows nothing about any particular embedder's data or
product wiring. A host embeds droste by writing its own adapter module — a
small Python file exposing the contract `pyodide/relay.ts` expects — that
calls into the pieces exported here. See `examples/pyodide-host/` for
droste's own minimal, dependency-free reference adapter and
`pyodide/README.md` for the full contract.
"""
