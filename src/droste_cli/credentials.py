"""Stored ModelRelay credentials for the logged-in path.

One file, one credential: ``$XDG_CONFIG_HOME/droste/credentials.json``
(default ``~/.config/droste/credentials.json``), written atomically at 0600.
The write pattern (O_EXCL temp file + fsync + rename) closes the window where
a plain write-then-chmod would expose the key at the process umask.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any

CREDENTIALS_VERSION = 1


class CredentialsError(Exception):
    """Credentials file exists but cannot be used; message is user-facing."""


@dataclass(frozen=True)
class Credentials:
    api_key: str
    base_url: str = ""
    provider: str = "modelrelay"  # "modelrelay" | "byok"
    api_key_id: str = ""
    email: str = ""
    default_model: str = ""
    created_at: str = ""


def credentials_path() -> str:
    config_home = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    return os.path.join(config_home, "droste", "credentials.json")


def load_credentials() -> Credentials | None:
    """Return stored credentials, or None when the user never logged in.

    A present-but-broken file raises: silently ignoring it would flip the run
    onto the "no credentials" error path and hide the real problem.
    """
    path = credentials_path()
    try:
        with open(path, encoding="utf-8") as fh:
            data: Any = json.load(fh)
    except FileNotFoundError:
        return None
    except Exception as exc:
        raise CredentialsError(f"cannot read {path}: {exc} (re-run `droste login`)") from exc
    if not isinstance(data, dict):
        raise CredentialsError(f"{path} is not a JSON object (re-run `droste login`)")
    provider = str(data.get("provider") or "modelrelay")
    if provider not in ("modelrelay", "byok"):
        raise CredentialsError(f"{path} has unknown provider {provider!r} (re-run `droste login`)")
    base_url = str(data.get("base_url") or "").rstrip("/")
    api_key = str(data.get("api_key") or "")
    if not api_key or (provider == "modelrelay" and not base_url):
        raise CredentialsError(f"{path} is incomplete (re-run `droste login`)")
    return Credentials(
        api_key=api_key,
        base_url=base_url,
        provider=provider,
        api_key_id=str(data.get("api_key_id") or ""),
        email=str(data.get("email") or ""),
        default_model=str(data.get("default_model") or ""),
        created_at=str(data.get("created_at") or ""),
    )


def save_credentials(creds: Credentials) -> str:
    """Atomically write the credentials file at 0600; returns the path."""
    path = credentials_path()
    directory = os.path.dirname(path)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    payload = {
        "version": CREDENTIALS_VERSION,
        "provider": creds.provider,
        "base_url": creds.base_url.rstrip("/"),
        "api_key": creds.api_key,
        "api_key_id": creds.api_key_id,
        "email": creds.email,
        "default_model": creds.default_model,
        "created_at": creds.created_at,
    }
    body = json.dumps(payload, indent=2) + "\n"
    # mkstemp: unique 0600 temp file in the same directory, so concurrent
    # saves can't unlink or replace each other's in-flight temp file — last
    # completed rename wins atomically.
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".credentials-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise
    return path


def delete_credentials() -> bool:
    """Remove the credentials file; returns True when one existed."""
    try:
        os.remove(credentials_path())
        return True
    except FileNotFoundError:
        return False
