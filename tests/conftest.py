"""Shared test isolation.

Every test runs with an isolated ``XDG_CONFIG_HOME`` and without ambient
provider credentials: the login flow makes stored credentials
and env keys part of the CLI's resolution order, so a developer's real
``~/.config/droste/credentials.json`` or exported ``OPENAI_API_KEY`` must
never leak into test behavior.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_credentials(tmp_path_factory, monkeypatch):
    config_home = tmp_path_factory.mktemp("xdg-config")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY", "DROSTE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    yield
