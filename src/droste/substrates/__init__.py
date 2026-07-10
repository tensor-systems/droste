"""Adapters that let the droste core loop run under alternative runtimes.

Each submodule targets one runtime (e.g. Pyodide/WASM) and stays
host-agnostic: it knows nothing about any particular embedder's data or
product wiring. A host embeds droste by writing its own adapter module — a
small Python file exposing the contract the Deno relay expects — that
calls into the pieces exported here. See `examples/pyodide-host/` for
droste's own minimal, dependency-free reference adapter and
`pyodide/README.md` for the full contract.

The Deno half of the Pyodide substrate (relay.ts and friends) ships INSIDE
this package (#33): `relay_dir()` (or `droste relay-path` on the command
line) locates it, so an embedder's build stages the relay from the installed
wheel — one pinned, hash-verified artifact — instead of downloading a
separate release tarball with its own sha pin.
"""

from importlib import resources
from pathlib import Path

# The Deno relay sources bundled as package data.
_RELAY_FILES = ("relay.ts", "stream.ts", "broker.ts", "events.ts", "offline-probe.ts", "deps.ts")


def relay_dir() -> Path:
    """Directory holding the bundled Deno relay sources (version-locked to
    this package by construction — they ship in the same wheel).

    Raises FileNotFoundError if the relay assets are missing or incomplete
    (e.g. a repackaged distribution that stripped non-Python files), rather
    than letting an embedder's build stage a partial relay.
    """
    path = Path(str(resources.files(__package__).joinpath("_relay")))
    missing = [name for name in _RELAY_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"bundled relay assets missing from {path}: {', '.join(missing)}")
    return path
