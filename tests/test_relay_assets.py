"""The Deno relay ships inside the wheel (#33): relay_dir() locates the
bundled sources, `droste relay-path` exposes it to build scripts, and the
Pyodide pin lives in exactly one file."""

from __future__ import annotations

import re

from droste.substrates import _RELAY_FILES, relay_dir


def test_relay_dir_contains_every_bundled_source() -> None:
    path = relay_dir()
    assert path.is_dir()
    for name in _RELAY_FILES:
        assert (path / name).is_file(), f"missing bundled relay file {name}"


def test_relay_imports_resolve_within_the_bundle() -> None:
    # The relay must be stageable from the installed package alone: every
    # relative import inside the bundled .ts files must resolve to another
    # bundled file (an import reaching outside the bundle would only work in
    # a git checkout).
    path = relay_dir()
    for name in _RELAY_FILES:
        for match in re.finditer(r'from "\./([^"]+)"', (path / name).read_text()):
            target = match.group(1)
            assert (path / target).is_file(), f"{name} imports non-bundled ./{target}"


def test_pyodide_pin_has_one_site() -> None:
    # deps.ts is the single bump site; every other bundled file imports from it.
    path = relay_dir()
    pinned = [name for name in _RELAY_FILES if "npm:pyodide@" in (path / name).read_text()]
    assert pinned == ["deps.ts"]


def test_cli_relay_path_prints_the_directory(capsys) -> None:
    from droste_cli.main import main

    assert main(["relay-path"]) == 0
    out = capsys.readouterr().out.strip()
    assert out == str(relay_dir())


def test_cli_relay_path_rejects_arguments(capsys) -> None:
    from droste_cli.main import main

    assert main(["relay-path", "extra"]) == 2
    assert "takes no arguments" in capsys.readouterr().err


def test_relay_emits_startup_handshake() -> None:
    # The relay's startup handshake must report both protocol constants; pin
    # the source-level contract (the e2e deno test asserts the runtime event).
    relay = (relay_dir() / "relay.ts").read_text()
    assert '"type": "startup"' in relay
    assert "RUNNER_PROTOCOL_VERSION" in relay
    assert "PROVIDER_PROTOCOL_VERSION" in relay
    assert '"provider_protocol"' in relay


def test_relay_has_no_direct_database_mount_mode() -> None:
    relay = (relay_dir() / "relay.ts").read_text()
    assert "RLM_DB_SERVICE" not in relay
    mount_calls = re.findall(
        r"\b([A-Za-z_$][\w$]*)\s*\.\s*mountNodeFS\s*\(",
        relay,
    )
    mounts = re.findall(
        r"\b([A-Za-z_$][\w$]*)\s*\.\s*mountNodeFS\s*\(\s*['\"]([^'\"]+)['\"]",
        relay,
    )
    assert len(mounts) == len(mount_calls), "relay mount points must be reviewable literals"
    assert mounts == [
        ("interp", "/app"),
        ("dbsvc", "/data"),
        ("dbsvc", "/contacts"),
    ], "only source files and the trusted provider interpreter may receive mounts"
    delete_path = relay.find("delete request.db_path")
    split_credentials = relay.find("splitCredentials(request)")
    assert delete_path >= 0, "relay must remove db_path before sandbox construction"
    assert split_credentials >= 0, "relay must broker credentials after path removal"
    assert delete_path < split_credentials
