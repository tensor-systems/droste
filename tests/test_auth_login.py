"""`droste login`/`logout`/`whoami` against a fake ModelRelay server (droste#55).

Covers the login contract: loopback OAuth handoff (nonce-checked), the $0
card-check gate (granted / reused-fingerprint / prepaid / unavailable), key
minting vs the signup-issued key, 0600 credentials on disk, and the
logout/whoami surfaces.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from droste_cli import auth
from droste_cli.credentials import credentials_path, load_credentials


class FakeModelRelay:
    """Just enough of the platform for the login flow.

    The OAuth provider hop is simulated by the test's browser opener: when the
    CLI "opens" the authorize URL, the opener form-POSTs the handoff (tokens +
    nonce) straight to the CLI's loopback callback — exactly what the real
    /auth/oauth/callback page's auto-submitting form does.
    """

    def __init__(self) -> None:
        self.start_requests: list[dict[str, str]] = []
        self.confirm_attempts = 0
        self.card_complete = False
        self.card_supported = True
        self.card_already_verified = False
        self.prepaid = False
        self.grant_already_used = False
        self.minted_keys: list[dict] = []
        self.handoff_extra: dict[str, str] = {}
        self.balance_cents: int | None = 321
        self.balance_auth_headers: list[dict[str, str]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _json(self, status: int, payload) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length).decode("utf-8") if length else ""
                payload = json.loads(raw) if raw else {}
                if parsed.path == "/auth/oauth/start":
                    fake.start_requests.append(query)
                    self._json(200, {"redirect_url": fake.base_url + "/fake-authorize"})
                    return
                if parsed.path == "/account/card-verification":
                    if not fake.card_supported:
                        self._json(404, {"error": "not found"})
                        return
                    if fake.card_already_verified:
                        self._json(409, {"error": "card already verified"})
                        return
                    self._json(
                        200,
                        {
                            "verification_id": "11111111-1111-1111-1111-111111111111",
                            "checkout_session_id": "cs_fake_123",
                            "checkout_url": fake.base_url + "/fake-checkout",
                            "expires_at": "2026-01-01T00:00:00Z",
                        },
                    )
                    return
                if parsed.path == "/account/card-verification/confirm":
                    fake.confirm_attempts += 1
                    assert payload.get("checkout_session_id") == "cs_fake_123"
                    if fake.prepaid:
                        self._json(422, {"error": "prepaid cards are not accepted"})
                        return
                    if not fake.card_complete:
                        self._json(409, {"error": "checkout session not completed"})
                        return
                    self._json(
                        200,
                        {
                            "verified": True,
                            "credit_granted": not fake.grant_already_used,
                            "grant_amount_cents": 100,
                            "grant_already_used_by_card": fake.grant_already_used,
                        },
                    )
                    return
                if parsed.path == "/api-keys":
                    fake.minted_keys.append(payload)
                    self._json(
                        201,
                        {
                            "api_key": {
                                "id": "22222222-2222-2222-2222-222222222222",
                                "secret_key": "mr_sk_minted_secret",
                                "label": payload.get("label", ""),
                            }
                        },
                    )
                    return
                self._json(404, {"error": "not found"})

            def do_GET(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/auth/me":
                    self._json(200, {"user": {"email": "dev@example.com"}})
                    return
                if parsed.path == "/account/card-verification":
                    if not fake.card_supported:
                        self._json(404, {"error": "not found"})
                        return
                    if fake.card_already_verified:
                        self._json(
                            200,
                            {"verified": True, "credit_granted": True, "grant_amount_cents": 100},
                        )
                        return
                    self._json(200, {"verified": False, "credit_granted": False})
                    return
                if parsed.path == "/account/balance":
                    self.headers_dict = dict(self.headers)
                    fake.balance_auth_headers.append(dict(self.headers))
                    if fake.balance_cents is None:
                        self._json(401, {"error": "unauthorized"})
                        return
                    self._json(200, {"balance_cents": fake.balance_cents, "currency": "usd"})
                    return
                self._json(404, {"error": "not found"})

            def log_message(self, *args) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def shutdown(self) -> None:
        self._server.shutdown()

    def make_opener(self, *, tokens: dict[str, str] | None = None):
        """A fake browser: the authorize URL triggers the OAuth handoff POST
        to the CLI's loopback; the checkout URL completes the card session."""
        fake = self

        def opener(url: str) -> bool:
            if url.endswith("/fake-authorize"):
                start = fake.start_requests[-1]
                fields = {
                    "access_token": "test-access-token",
                    "refresh_token": "test-refresh-token",
                    "handoff_nonce": start["handoff_nonce"],
                }
                fields.update(tokens or {})
                body = urllib.parse.urlencode(fields).encode("utf-8")
                req = urllib.request.Request(start["return_to"], data=body, method="POST")
                urllib.request.urlopen(req, timeout=5)
                return True
            if url.endswith("/fake-checkout"):
                fake.card_complete = True
                return True
            return True

        return opener


@pytest.fixture()
def fake_platform():
    fake = FakeModelRelay()
    yield fake
    fake.shutdown()


def _login(fake: FakeModelRelay, **kwargs) -> int:
    return auth.run_login(
        fake.base_url,
        opener=fake.make_opener(**kwargs),
        oauth_timeout=10,
        card_timeout=10,
        card_poll_interval=0.05,
    )


def test_login_end_to_end_grants_credits(fake_platform, capsys):
    code = _login(fake_platform)
    assert code == 0

    creds = load_credentials()
    assert creds is not None
    assert creds.api_key == "mr_sk_minted_secret"
    assert creds.base_url == fake_platform.base_url
    assert creds.email == "dev@example.com"
    assert creds.default_model == auth.DEFAULT_LOGGED_IN_MODEL
    assert creds.api_key_id == "22222222-2222-2222-2222-222222222222"

    mode = stat.S_IMODE(os.stat(credentials_path()).st_mode)
    assert mode == 0o600

    err = capsys.readouterr().err
    assert "logged in as dev@example.com" in err
    assert "free credits" in err
    assert "never charged" in err
    assert "balance: $3.21" in err
    # The card check polled at least once before the fake checkout completed.
    assert fake_platform.confirm_attempts >= 1
    # The minted key was requested with the CLI label.
    assert fake_platform.minted_keys == [{"label": "droste-cli"}]
    # Balance was fetched with the API key header, not a bearer token
    # (urllib normalizes header casing, so compare case-insensitively).
    normalized = [
        {k.lower(): v for k, v in headers.items()} for headers in fake_platform.balance_auth_headers
    ]
    assert any(h.get("x-modelrelay-api-key") == "mr_sk_minted_secret" for h in normalized)
    assert all("authorization" not in h for h in normalized)


def test_login_prefers_signup_issued_key(fake_platform):
    code = _login(
        fake_platform,
        tokens={
            "issued_key_secret": "mr_sk_issued_at_signup",
            "issued_key_id": "33333333-3333-3333-3333-333333333333",
        },
    )
    assert code == 0
    creds = load_credentials()
    assert creds is not None and creds.api_key == "mr_sk_issued_at_signup"
    assert fake_platform.minted_keys == []  # no extra key minted


def test_login_prepaid_card_completes_without_credits(fake_platform, capsys):
    fake_platform.prepaid = True
    code = _login(fake_platform)
    assert code == 0
    assert load_credentials() is not None  # login still lands
    err = capsys.readouterr().err
    assert "prepaid" in err
    assert "no free credits" in err


def test_login_reused_fingerprint_is_honest(fake_platform, capsys):
    fake_platform.grant_already_used = True
    code = _login(fake_platform)
    assert code == 0
    err = capsys.readouterr().err
    assert "already claimed free credits" in err


def test_login_card_verification_unavailable_skips_quietly(fake_platform, capsys):
    fake_platform.card_supported = False
    code = _login(fake_platform)
    assert code == 0
    assert load_credentials() is not None
    err = capsys.readouterr().err
    assert "card" not in err.lower() or "card check" not in err


def test_login_rejects_wrong_nonce(fake_platform):
    def bad_opener(url: str) -> bool:
        if url.endswith("/fake-authorize"):
            start = fake_platform.start_requests[-1]
            body = urllib.parse.urlencode(
                {"access_token": "stolen", "handoff_nonce": "attacker-nonce"}
            ).encode("utf-8")
            req = urllib.request.Request(start["return_to"], data=body, method="POST")
            try:
                urllib.request.urlopen(req, timeout=5)
            except urllib.error.HTTPError:
                pass  # loopback answers 400; the CLI aborts
        return True

    with pytest.raises(auth.AuthError, match="nonce mismatch"):
        auth.run_login(
            fake_platform.base_url,
            opener=bad_opener,
            oauth_timeout=10,
            card_timeout=5,
            card_poll_interval=0.05,
        )
    assert load_credentials() is None


def test_unsolicited_loopback_posts_do_not_abort_login(fake_platform):
    # A blind localhost probe (no handoff_nonce field, or wrong path) must
    # not DoS the 3-minute wait; the real callback still lands afterwards.
    real_opener = fake_platform.make_opener()

    def noisy_opener(url: str) -> bool:
        if url.endswith("/fake-authorize"):
            start = fake_platform.start_requests[-1]
            for probe_body, probe_path in (
                (b"garbage=1", ""),
                (b"", ""),
                (b"x=y", "/not-callback"),
            ):
                target = start["return_to"]
                if probe_path:
                    target = target.rsplit("/callback", 1)[0] + probe_path
                req = urllib.request.Request(target, data=probe_body, method="POST")
                try:
                    urllib.request.urlopen(req, timeout=5)
                except urllib.error.HTTPError:
                    pass  # rejected, but the wait must survive
        return real_opener(url)

    assert (
        auth.run_login(
            fake_platform.base_url,
            opener=noisy_opener,
            oauth_timeout=10,
            card_timeout=10,
            card_poll_interval=0.05,
        )
        == 0
    )
    assert load_credentials() is not None


def test_logout_removes_credentials(fake_platform, capsys):
    _login(fake_platform)
    assert load_credentials() is not None
    assert auth.run_logout() == 0
    assert load_credentials() is None
    assert "removed" in capsys.readouterr().err
    # Idempotent.
    assert auth.run_logout() == 0


def test_whoami_shows_identity_and_balance(fake_platform, capsys):
    _login(fake_platform)
    capsys.readouterr()
    assert auth.run_whoami() == 0
    out = capsys.readouterr().out
    assert "dev@example.com" in out
    assert "$3.21" in out
    assert fake_platform.base_url in out


def test_whoami_without_login(capsys):
    assert auth.run_whoami() == 2
    assert "droste login" in capsys.readouterr().err


def test_whoami_rejected_key_suggests_relogin(fake_platform, capsys):
    _login(fake_platform)
    fake_platform.balance_cents = None  # balance endpoint now 401s
    capsys.readouterr()
    assert auth.run_whoami() == 2
    assert "run `droste login`" in capsys.readouterr().err


def test_concurrent_credential_saves_never_corrupt(tmp_path):
    # Unique temp files: parallel saves race safely; the survivor is one
    # complete, parseable credential set (codex review finding).
    from droste_cli.credentials import Credentials, save_credentials

    errors: list[Exception] = []

    def save(idx: int) -> None:
        try:
            save_credentials(
                Credentials(base_url=f"https://api-{idx}.test", api_key=f"mr_sk_{idx}")
            )
        except Exception as exc:  # pragma: no cover - failure detail
            errors.append(exc)

    threads = [threading.Thread(target=save, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    creds = load_credentials()
    assert creds is not None
    assert creds.api_key.startswith("mr_sk_")
    assert creds.base_url.startswith("https://api-")
    # No stray temp files left behind.
    import glob

    directory = os.path.dirname(credentials_path())
    assert glob.glob(os.path.join(directory, "*.tmp")) == []


# --- interactive chooser (droste login on a TTY) ---


def _feed_inputs(monkeypatch, answers):
    answers = list(answers)
    monkeypatch.setattr("builtins.input", lambda *a: answers.pop(0))


def _fake_tty(monkeypatch):
    # The chooser requires BOTH stdin and stderr to be TTYs.
    monkeypatch.setattr("sys.stdin", type("T", (), {"isatty": staticmethod(lambda: True)})())
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)


def test_login_chooser_byok_imports_env_key(fake_platform, monkeypatch, capsys):
    _fake_tty(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key-123456")
    # choice 2, accept env key, default base URL, model
    _feed_inputs(monkeypatch, ["2", "", "", "gpt-5.2-mini"])
    assert auth.run_login(fake_platform.base_url) == 0
    creds = load_credentials()
    assert creds is not None
    assert creds.provider == "byok"
    assert creds.api_key == "sk-env-key-123456"
    assert creds.base_url == ""
    assert creds.default_model == "gpt-5.2-mini"
    err = capsys.readouterr().err
    assert "choose how to run" in err
    assert "sk-env-key-123456" not in err  # never echo the full key


def test_login_chooser_default_choice_is_modelrelay(fake_platform, monkeypatch):
    _fake_tty(monkeypatch)
    _feed_inputs(monkeypatch, [""])  # Enter = choice 1
    # Patch the OAuth leg so the chooser routing is what's under test.
    called = {}

    def fake_login(base_url, **kw):
        called["base"] = base_url
        return 0

    monkeypatch.setattr(auth, "login_modelrelay", fake_login)
    assert auth.run_login("https://example.test/api/v1") == 0
    assert called["base"] == "https://example.test/api/v1"


def test_byok_setup_requires_model(monkeypatch):
    _fake_tty(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key-123456")
    _feed_inputs(monkeypatch, ["2", "", "", ""])  # empty model
    with pytest.raises(auth.AuthError, match="default model"):
        auth.run_login()


def test_eof_during_chooser_is_clean_cancel(monkeypatch):
    _fake_tty(monkeypatch)

    def eof(*a):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    with pytest.raises(auth.AuthError, match="cancelled"):
        auth.run_login()
    assert load_credentials() is None
