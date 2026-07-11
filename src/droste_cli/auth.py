"""`droste login` / `logout` / `whoami`.

In an interactive terminal, `droste login` is a chooser: ModelRelay with
free credits (the default), or your own API key. Both choices are STORED in
the credentials file — picking how droste runs is a deliberate one-time
setup step, not a side effect of whatever happens to be exported in the
shell. A detected env key is offered as a one-keystroke import.

The ModelRelay choice is a browser loopback OAuth flow (RFC 8252): start a
local server on
an ephemeral loopback port, ask the platform for the GitHub authorization
URL with that port as `return_to`, and capture the tokens the OAuth
callback form-POSTs back.

Free credits are gated by a $0 card check: after sign-in the CLI opens a
hosted checkout page (card saved for verification, never charged) and polls
the confirm endpoint until the card is verified. Each card can claim the
signup grant once, ever; prepaid cards are declined. The card UX itself is
entirely web — the CLI only opens the URL and reads the outcome.

What gets stored is a single long-lived API key (plus base_url/email/default
model) in ``~/.config/droste/credentials.json`` — no OAuth tokens are kept,
so there is nothing to refresh and `logout` is just removing the file. Keys
can be revoked any time from the ModelRelay dashboard.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from droste.clients.useragent import USER_AGENT

from .credentials import (
    Credentials,
    credentials_path,
    delete_credentials,
    load_credentials,
    save_credentials,
)

DEFAULT_BASE_URL = "https://api.modelrelay.ai/api/v1"
# The default root model for logged-in runs.
DEFAULT_LOGGED_IN_MODEL = "gemini-3.5-flash"

OAUTH_TIMEOUT_SECONDS = 180.0
CARD_TIMEOUT_SECONDS = 300.0
CARD_POLL_INTERVAL_SECONDS = 2.0
HTTP_TIMEOUT_SECONDS = 15.0

_SUCCESS_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Signed in</title></head>
<body style="font-family:-apple-system,system-ui,sans-serif;text-align:center;padding:3rem">
<h2>Signed in to ModelRelay</h2>
<p>You can close this tab and return to the terminal.</p>
</body></html>"""

_FAIL_HTML = """<!doctype html><html><body style="font-family:sans-serif;text-align:center;padding:3rem">
<h2>Sign-in failed</h2><p>{msg}</p></body></html>"""


class AuthError(Exception):
    """User-facing auth failure: message to stderr, exit code 2."""


def _say(message: str) -> None:
    print(f"droste: {message}", file=sys.stderr, flush=True)


def _is_remote_session() -> bool:
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))


def _open_browser(url: str, opener: Callable[[str], bool] | None) -> None:
    """Open the browser unless we're clearly remote; always print the URL so
    the user can hand-off manually (SSH, no default browser, ...)."""
    print(f"  if your browser doesn't open, visit:\n  {url}", file=sys.stderr, flush=True)
    if _is_remote_session():
        return
    try:
        (opener or webbrowser.open)(url)
    except Exception:
        pass


def _api_request(
    base_url: str,
    method: str,
    path: str,
    *,
    token: str = "",
    api_key: str = "",
    payload: dict[str, Any] | None = None,
    timeout: float = HTTP_TIMEOUT_SECONDS,
) -> tuple[int, Any]:
    """One platform API call; returns (status, parsed-JSON-or-None).

    ``token`` is an OAuth bearer token (Authorization header); ``api_key`` is
    an mr_sk_* secret key (X-ModelRelay-Api-Key header — the native API
    rejects API keys sent as Bearer). HTTP error statuses are returned, not
    raised — callers branch on expected statuses (404 capability-off, 409
    pending, 422 prepaid). Network-level failures raise AuthError.
    """
    url = base_url.rstrip("/") + path
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = "Bearer " + token
    if api_key:
        headers["X-ModelRelay-Api-Key"] = api_key
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = int(resp.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read() if exc.fp else b""
        status = int(getattr(exc, "code", 0))
    except Exception as exc:
        raise AuthError(f"cannot reach {url}: {exc}") from exc
    data: Any = None
    if raw:
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            data = raw.decode("utf-8", errors="replace")
    return status, data


class _HandoffResult:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.fields: dict[str, str] = {}
        self.error: str = ""

    def resolve(self, fields: dict[str, str]) -> None:
        self.fields = fields
        self.event.set()

    def fail(self, error: str) -> None:
        self.error = error
        self.event.set()


def _start_loopback_server(nonce: str) -> tuple[ThreadingHTTPServer, int, _HandoffResult]:
    result = _HandoffResult()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            fields = {k: v[0] for k, v in urllib.parse.parse_qs(raw).items()}
            # Unsolicited traffic (wrong path, or a blind localhost probe
            # that doesn't even carry a handoff_nonce field) is rejected but
            # must NOT abort the wait — the real callback may still arrive.
            # A well-formed callback with the WRONG nonce is terminal: that
            # is the CSRF signal this nonce exists to catch.
            if (
                urllib.parse.urlparse(self.path).path != "/callback"
                or "handoff_nonce" not in fields
            ):
                self._respond(404, _FAIL_HTML.format(msg="unexpected request"))
                return
            if fields.get("handoff_nonce") != nonce:
                self._respond(400, _FAIL_HTML.format(msg="state mismatch"))
                result.fail("callback nonce mismatch (possible CSRF) - aborted")
                return
            if not fields.get("access_token"):
                self._respond(400, _FAIL_HTML.format(msg="no token was returned"))
                result.fail("callback contained no access token")
                return
            self._respond(200, _SUCCESS_HTML)
            result.resolve(fields)

        def _respond(self, status: int, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], result


def _oauth_handoff(
    base_url: str,
    opener: Callable[[str], bool] | None,
    timeout_seconds: float,
) -> dict[str, str]:
    """Run the loopback OAuth flow; returns the callback form fields
    (access_token, refresh_token, and issued_key_* on fresh signups)."""
    nonce = secrets.token_hex(16)
    server, port, result = _start_loopback_server(nonce)
    try:
        start_path = (
            "/auth/oauth/start?provider=github"
            f"&return_to={urllib.parse.quote(f'http://127.0.0.1:{port}/callback', safe='')}"
            f"&handoff_nonce={urllib.parse.quote(nonce, safe='')}"
        )
        status, data = _api_request(base_url, "POST", start_path)
        if status != 200 or not isinstance(data, dict) or not data.get("redirect_url"):
            raise AuthError(f"oauth start failed (HTTP {status}): {str(data)[:200]}")
        redirect_url = str(data["redirect_url"])

        _say("signing in to ModelRelay with GitHub...")
        _open_browser(redirect_url, opener)
        _say("waiting for browser sign-in...")

        if not result.event.wait(timeout_seconds):
            raise AuthError("timed out waiting for browser sign-in")
        if result.error:
            raise AuthError(result.error)
        return result.fields
    finally:
        server.shutdown()


def _ensure_card_verification(
    base_url: str,
    token: str,
    opener: Callable[[str], bool] | None,
    *,
    timeout_seconds: float,
    poll_interval: float,
) -> str:
    """Run the $0 card check that gates free credits.

    Returns one of: "granted", "verified" (card ok, grant already claimed
    or previously applied), "prepaid", "pending" (user didn't finish),
    "unavailable" (server has no card verification — self-host/dev).
    Never raises for card-step outcomes: login proceeds and the caller
    prints the honest state.
    """
    status, data = _api_request(base_url, "GET", "/account/card-verification", token=token)
    if status in (404, 501, 503):
        return "unavailable"
    if status != 200:
        raise AuthError(f"card verification status failed (HTTP {status}): {str(data)[:200]}")
    if isinstance(data, dict) and data.get("verified"):
        return "granted" if data.get("credit_granted") else "verified"

    _say("free credits require a quick card check - never charged unless you upgrade.")
    status, data = _api_request(base_url, "POST", "/account/card-verification", token=token)
    if status == 409:
        return "verified"
    if status != 200 or not isinstance(data, dict) or not data.get("checkout_url"):
        raise AuthError(f"could not start card verification (HTTP {status}): {str(data)[:200]}")
    checkout_url = str(data["checkout_url"])
    checkout_session_id = str(data.get("checkout_session_id") or "")

    _open_browser(checkout_url, opener)
    _say("waiting for card verification...")

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status, outcome = _api_request(
            base_url,
            "POST",
            "/account/card-verification/confirm",
            token=token,
            payload={"checkout_session_id": checkout_session_id},
        )
        if status == 200 and isinstance(outcome, dict):
            if outcome.get("credit_granted"):
                return "granted"
            return "verified"
        if status == 422:
            return "prepaid"
        if status not in (409,):  # 409 = checkout still open; keep waiting
            raise AuthError(f"card verification failed (HTTP {status}): {str(outcome)[:200]}")
        time.sleep(poll_interval)
    return "pending"


def _mint_api_key(base_url: str, token: str, handoff: dict[str, str]) -> tuple[str, str]:
    """Returns (secret_key, key_id): the signup-issued key when the OAuth
    handoff carried one (fresh signups), else a freshly minted droste-cli key."""
    issued = handoff.get("issued_key_secret") or ""
    if issued:
        return issued, handoff.get("issued_key_id") or ""
    status, data = _api_request(
        base_url, "POST", "/api-keys", token=token, payload={"label": "droste-cli"}
    )
    if status not in (200, 201) or not isinstance(data, dict):
        raise AuthError(f"could not create an API key (HTTP {status}): {str(data)[:200]}")
    key = data.get("api_key") if isinstance(data.get("api_key"), dict) else {}
    secret = str(key.get("secret_key") or "")
    if not secret:
        raise AuthError("API key response contained no secret_key")
    return secret, str(key.get("id") or "")


def _fetch_balance_cents(base_url: str, api_key: str) -> int | None:
    status, data = _api_request(base_url, "GET", "/account/balance", api_key=api_key)
    if status == 200 and isinstance(data, dict) and "balance_cents" in data:
        try:
            return int(data["balance_cents"])
        except (TypeError, ValueError):
            return None
    return None


def run_login(
    base_url: str | None = None,
    *,
    opener: Callable[[str], bool] | None = None,
    oauth_timeout: float = OAUTH_TIMEOUT_SECONDS,
    card_timeout: float = CARD_TIMEOUT_SECONDS,
    card_poll_interval: float = CARD_POLL_INTERVAL_SECONDS,
) -> int:
    """`droste login`: interactive chooser on a TTY, straight to the
    ModelRelay sign-in otherwise (non-TTY has nobody to ask)."""
    if sys.stdin.isatty() and sys.stderr.isatty():
        choice = _prompt_choice()
        if choice == "2":
            _setup_byok()
            _say('try: droste "why did it crash?" ./logs')
            return 0
    return login_modelrelay(
        base_url,
        opener=opener,
        oauth_timeout=oauth_timeout,
        card_timeout=card_timeout,
        card_poll_interval=card_poll_interval,
    )


def _ask(prompt: str) -> str:
    """Prompt on stderr, read a line; Ctrl-D is a clean cancel."""
    sys.stderr.write(prompt)
    sys.stderr.flush()
    try:
        return input().strip()
    except EOFError as exc:
        raise AuthError("setup cancelled") from exc


def _prompt_choice() -> str:
    print(
        "droste: choose how to run:\n"
        "  1) ModelRelay - free credits, GitHub sign-in, no key (recommended)\n"
        "  2) your own API key (any OpenAI-compatible endpoint, or Anthropic)",
        file=sys.stderr,
        flush=True,
    )
    answer = _ask("choice [1]: ")
    return "2" if answer == "2" else "1"


def _mask_key(key: str) -> str:
    return key[:8] + "..." if len(key) > 11 else "..."


def _setup_byok() -> None:
    """Store a bring-your-own-key choice in the credentials file."""
    import getpass

    env_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""
    api_key = ""
    if env_key:
        if _ask(f"use {_mask_key(env_key)} from your environment? [Y/n]: ").lower() not in (
            "n",
            "no",
        ):
            api_key = env_key
    if not api_key:
        try:
            api_key = getpass.getpass("paste your API key: ").strip()
        except EOFError as exc:
            raise AuthError("setup cancelled") from exc
    if not api_key:
        raise AuthError("no API key entered")

    base = ""
    if not api_key.startswith("sk-ant-"):
        base = _ask("API base URL [https://api.openai.com/v1]: ").rstrip("/")

    model = _ask("default model (e.g. gpt-5.2-mini): ")
    if not model:
        raise AuthError("a default model is required for your own key")

    path = save_credentials(
        Credentials(
            api_key=api_key,
            base_url=base,
            provider="byok",
            default_model=model,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    )
    _say(f"saved (credentials: {path})")


def run_interactive_setup() -> Credentials:
    """First run with no credentials on a TTY: the same chooser, then the
    run continues with whatever was set up."""
    choice = _prompt_choice()
    if choice == "2":
        _setup_byok()
    else:
        code = login_modelrelay(None)
        if code != 0:
            raise AuthError("login failed")
    creds = load_credentials()
    if creds is None:
        raise AuthError("setup did not produce credentials")
    return creds


def login_modelrelay(
    base_url: str | None = None,
    *,
    opener: Callable[[str], bool] | None = None,
    oauth_timeout: float = OAUTH_TIMEOUT_SECONDS,
    card_timeout: float = CARD_TIMEOUT_SECONDS,
    card_poll_interval: float = CARD_POLL_INTERVAL_SECONDS,
) -> int:
    base = (base_url or DEFAULT_BASE_URL).rstrip("/")

    handoff = _oauth_handoff(base, opener, oauth_timeout)
    token = handoff["access_token"]

    status, me = _api_request(base, "GET", "/auth/me", token=token)
    email = ""
    if status == 200 and isinstance(me, dict):
        user = me.get("user") if isinstance(me.get("user"), dict) else {}
        email = str(user.get("email") or "")

    card_state = _ensure_card_verification(
        base, token, opener, timeout_seconds=card_timeout, poll_interval=card_poll_interval
    )

    secret_key, key_id = _mint_api_key(base, token, handoff)

    path = save_credentials(
        Credentials(
            api_key=secret_key,
            base_url=base,
            provider="modelrelay",
            api_key_id=key_id,
            email=email,
            default_model=DEFAULT_LOGGED_IN_MODEL,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    )

    _say(f"logged in as {email or 'your ModelRelay account'} (credentials: {path})")
    if card_state == "granted":
        _say("card verified - free credits are on your account.")
    elif card_state == "verified":
        _say(
            "card verified. this card already claimed free credits elsewhere, "
            "so no new credits were granted."
        )
    elif card_state == "prepaid":
        _say(
            "prepaid cards can't be used for verification, so no free credits were "
            "granted. use a different card from the dashboard, or bring your own "
            "key (OPENAI_API_KEY)."
        )
    elif card_state == "pending":
        _say(
            "card check not completed - finish it in the browser tab, then run "
            "`droste login` again to claim your credits."
        )
    balance = _fetch_balance_cents(base, secret_key)
    if balance is not None:
        _say(f"balance: ${balance / 100:.2f}")
    _say('try: droste "why did it crash?" ./logs')
    return 0


def run_logout() -> int:
    existed = delete_credentials()
    if existed:
        _say("stored credentials removed.")
    else:
        _say("nothing stored.")
    return 0


def run_whoami() -> int:
    creds = load_credentials()
    if creds is None:
        _say("not set up yet - run `droste login`.")
        return 2
    if creds.provider == "byok":
        print("provider: your own key")
        print(f"key:      {_mask_key(creds.api_key)}")
        print(f"base_url: {creds.base_url or '(default)'}")
        print(f"model:    {creds.default_model or '(none)'}")
        print(f"file:     {credentials_path()}")
        return 0
    print(f"email:    {creds.email or '(unknown)'}")
    print(f"base_url: {creds.base_url}")
    print(f"model:    {creds.default_model or '(none)'}")
    print(f"file:     {credentials_path()}")
    balance = _fetch_balance_cents(creds.base_url, creds.api_key)
    if balance is not None:
        print(f"balance:  ${balance / 100:.2f}")
    else:
        status, _ = _api_request(creds.base_url, "GET", "/account/balance", api_key=creds.api_key)
        if status == 401:
            _say("stored credentials were rejected (revoked key?) - run `droste login`.")
            return 2
    return 0
