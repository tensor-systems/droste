"""droste CLI: arg parsing, the unix contract, end-to-end over the stub server."""

from __future__ import annotations

import json
import sqlite3

import pytest
from test_openai_compat_client import StubOpenAIServer

from droste_cli.main import build_parser, main


@pytest.fixture()
def stub_server():
    server = StubOpenAIServer()
    yield server
    server.shutdown()


def make_sqlite(path):
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE t (id INTEGER, name TEXT); INSERT INTO t VALUES (1, 'ada');")
    conn.commit()
    conn.close()


# --- arg parsing (the naked binary is the verb; `ask` is an alias) ---


def test_parse_flags_and_positionals():
    args = build_parser().parse_args(
        [
            "a.txt",
            "b.txt",
            "what changed?",
            "--model",
            "m",
            "--base-url",
            "http://x/v1",
            "--api-key",
            "k",
            "--subcall-model",
            "sm",
            "--subcall-max-output-tokens",
            "512",
            "--reasoning-effort",
            "low",
            "--max-iterations",
            "3",
            "--max-subcalls",
            "7",
            "--json",
            "--verbose",
            "--quiet",
        ]
    )
    assert args.inputs == ["a.txt", "b.txt", "what changed?"]
    assert (args.model, args.base_url, args.api_key) == ("m", "http://x/v1", "k")
    assert (args.subcall_model, args.subcall_max_output_tokens) == ("sm", 512)
    assert args.reasoning_effort == "low"
    assert (args.max_iterations, args.max_subcalls) == (3, 7)
    assert args.json and args.verbose and args.quiet


def test_parse_defaults():
    args = build_parser().parse_args(["q"])
    assert args.subcall_max_output_tokens == 2048
    assert args.max_iterations == 20
    assert args.max_subcalls == 50
    assert args.max_bytes == 50_000_000
    assert args.max_file_bytes == 2_000_000
    assert not args.json and not args.verbose and not args.quiet
    assert args.db is None


def test_model_env_default(monkeypatch):
    monkeypatch.setenv("DROSTE_MODEL", "env-model")
    args = build_parser().parse_args(["q"])
    assert args.model == "env-model"


def test_missing_model_is_usage_error(tmp_path, monkeypatch, capsys):
    # BYOK (env key present) still requires an explicit model.
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert main([str(f), "q"]) == 2
    assert "--model is required" in capsys.readouterr().err


def test_no_credentials_error_points_at_login(tmp_path, capsys):
    # Nothing configured, non-interactive (pytest stdin is not a TTY):
    # a terse error pointing at the setup command.
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert main([str(f), "q"]) == 2
    err = capsys.readouterr().err
    assert "no credentials" in err
    assert "droste login" in err


def test_empty_cwd_is_usage_error(tmp_path, monkeypatch, capsys):
    # No data args and nothing readable here → a clean usage error, not a
    # model call over nothing.
    monkeypatch.chdir(tmp_path)
    assert main(["q", "--model", "m"]) == 2
    assert "nothing to ask over" in capsys.readouterr().err


def test_pathlike_typo_is_usage_error(capsys):
    assert main(["reprot.txt", "q", "--model", "m"]) == 2
    assert "typo" in capsys.readouterr().err


def test_db_missing_file_errors(tmp_path, capsys):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert main([str(f), "--db", "/nonexistent/app.db", "q", "--model", "m"]) == 2
    assert "database not found" in capsys.readouterr().err


# --- --db wiring to the local SQL source ---


def test_db_exposes_sqlite_as_source(tmp_path):
    from droste_cli.main import _load_sql_source

    db = tmp_path / "app.db"
    make_sqlite(db)

    source = _load_sql_source(str(db))
    assert source.name() == "db"
    assert "t(" in source.get_schema()
    rows = source.query("SELECT name FROM t")
    assert rows == [{"name": "ada"}]


def test_db_directory_is_usage_error(tmp_path, capsys):
    code = main(["--db", str(tmp_path), "q", "--model", "m"])
    assert code == 2
    assert "not a file" in capsys.readouterr().err


def test_db_non_sqlite_file_is_usage_error(tmp_path, capsys):
    bogus = tmp_path / "notes.txt"
    bogus.write_text("this is not a database")
    code = main(["--db", str(bogus), "q", "--model", "m"])
    assert code == 2
    assert "SQLite database" in capsys.readouterr().err


# --- end-to-end over the stub OpenAI-compatible server ---


def _e2e_argv(server, *positionals, extra=()):
    return [
        *[str(p) for p in positionals],
        "--model",
        "root-model",
        "--base-url",
        server.base_url,
        "--api-key",
        "k",
        "--max-iterations",
        "2",
        *extra,
    ]


ANSWER_FROM_FILE = (
    "```python\nanswer['content'] = context['files'][0]['text']\nanswer['ready'] = True\n```"
)


def test_ask_file_end_to_end(stub_server, tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("the launch is on friday")
    stub_server.root_responses = [ANSWER_FROM_FILE]
    exit_code = main(_e2e_argv(stub_server, doc, "what does the file say?"))
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "the launch is on friday"
    # The report line names what was read, before spend.
    assert "loaded 1 file" in captured.err
    # The system prompt must carry the context size + preview description.
    system_prompt = stub_server.requests[0]["messages"][0]["content"]
    assert "doc.txt" in system_prompt
    assert "1 file(s)" in system_prompt


def test_ask_alias_still_works(stub_server, tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("alias content")
    stub_server.root_responses = [ANSWER_FROM_FILE]
    exit_code = main(["ask", *_e2e_argv(stub_server, doc, "say it")])
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "alias content"


def test_ask_directory_end_to_end(stub_server, tmp_path, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("alpha")
    (docs / "b.md").write_text("bravo")
    (docs / "blob.bin").write_bytes(b"\x00\x01")
    stub_server.root_responses = [
        "```python\n"
        "answer['content'] = ' '.join(f['text'] for f in context['files'])\n"
        "answer['ready'] = True\n"
        "```",
    ]
    exit_code = main(_e2e_argv(stub_server, docs, "what do the docs say?"))
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "alpha bravo"
    assert "loaded 2 files" in captured.err
    assert "1 binary" in captured.err


def test_ask_cwd_default_end_to_end(stub_server, tmp_path, monkeypatch, capsys):
    (tmp_path / "notes.md").write_text("cwd content")
    monkeypatch.chdir(tmp_path)
    stub_server.root_responses = [ANSWER_FROM_FILE]
    exit_code = main(_e2e_argv(stub_server, "what is here?"))
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "cwd content"
    assert "from ." in captured.err


def test_ask_positional_sqlite_end_to_end(stub_server, tmp_path, capsys):
    db = tmp_path / "app.db"
    make_sqlite(db)
    stub_server.root_responses = [
        "```python\n"
        'rows = db.query("SELECT name FROM t")\n'
        "answer['content'] = rows[0]['name']\n"
        "answer['ready'] = True\n"
        "```",
    ]
    exit_code = main(_e2e_argv(stub_server, db, "who is in the table?"))
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "ada"
    assert f"db: {db}" in captured.err


def test_quiet_suppresses_report(stub_server, tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("quiet content")
    stub_server.root_responses = [ANSWER_FROM_FILE]
    exit_code = main(_e2e_argv(stub_server, doc, "say it", extra=["--quiet"]))
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "loaded" not in captured.err


def test_ask_json_shape(stub_server, tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("payload text")
    stub_server.root_responses = [
        "```python\nanswer['content'] = 'json answer'\nanswer['ready'] = True\n```",
    ]
    exit_code = main(_e2e_argv(stub_server, doc, "q", extra=["--json"]))
    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["answer"] == "json answer"
    assert payload["ready"] is True
    assert payload["extracted"] is False
    assert payload["iterations"] == 1
    assert payload["subcalls"] == 0
    assert payload["tokens_used"] > 0
    assert payload["model"] == "root-model"
    assert payload["files"] == 1
    assert payload["db"] is None
    assert payload["error"] is None


def test_ask_extracted_answer_exits_zero_with_note(stub_server, tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("partial work")
    stub_server.root_responses = [
        # One iteration that never sets ready...
        "```python\nprint(context['files'][0]['text'])\n```",
        # ...then the extract-fallback pass answers from the trajectory.
        "extracted final answer",
    ]
    exit_code = main(
        [
            str(doc),
            "q",
            "--model",
            "root-model",
            "--base-url",
            stub_server.base_url,
            "--api-key",
            "k",
            "--max-iterations",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "extracted final answer"
    assert "extracted from partial work" in captured.err


def test_ask_extract_failure_surfaces_note_and_json_field(monkeypatch, tmp_path, capsys):
    """When the post-exhaustion extract call itself fails, the CLI must say so
    (not silently present raw loop output as if nothing went wrong) — the fix
    for a real bug where this failure was swallowed with zero trace anywhere."""
    import sys

    from droste.exceptions import RLMError
    from droste.loop.rlm import RLMResult

    # droste_cli/__init__.py does `from .main import main`, which shadows the
    # `droste_cli.main` package attribute with the function — grab the real
    # submodule from sys.modules to patch the name `main()` actually resolves.
    main_module = sys.modules["droste_cli.main"]

    def fake_run_rlm(*args, **kwargs):
        return RLMResult(
            answer="raw debug print() output",
            ready=False,
            iterations=2,
            tokens_used=10,
            sub_calls_made=0,
            trajectory=[],
            extracted=False,
            extract_error=RLMError(type="RuntimeError", message="provider timeout"),
        )

    monkeypatch.setattr(main_module, "run_rlm", fake_run_rlm)
    doc = tmp_path / "doc.txt"
    doc.write_text("x")
    exit_code = main(
        [
            str(doc),
            "q",
            "--model",
            "m",
            "--base-url",
            "http://127.0.0.1:1",
            "--api-key",
            "k",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    payload = json.loads(captured.out)
    assert payload["extracted"] is False
    assert payload["extract_error"] == {"type": "RuntimeError", "message": "provider timeout"}
    assert "extraction failed" in captured.err
    assert "RuntimeError: provider timeout" in captured.err
    assert "raw loop output, not a synthesized answer" in captured.err


def test_ask_root_failure_is_nonzero(stub_server, tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("x")
    stub_server.fail_status = 503
    stub_server.fail_body = b"provider down"
    exit_code = main(_e2e_argv(stub_server, doc, "q"))
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "provider down" in captured.err


# --- upfront validation of usage errors (exit 2, not a mid-run crash) ---


def test_negative_subcall_max_output_tokens_is_usage_error(tmp_path, capsys):
    f = tmp_path / "a.txt"
    f.write_text("hi")
    code = main([str(f), "q", "--model", "m", "--subcall-max-output-tokens", "-1"])
    assert code == 2
    assert "must be >= 0" in capsys.readouterr().err


def test_max_iterations_below_one_is_usage_error(tmp_path, capsys):
    f = tmp_path / "a.txt"
    f.write_text("hi")
    code = main([str(f), "q", "--model", "m", "--max-iterations", "0"])
    assert code == 2
    assert "must be >= 1" in capsys.readouterr().err


def test_bad_byte_budgets_are_usage_errors(tmp_path, capsys):
    f = tmp_path / "a.txt"
    f.write_text("hi")
    assert main([str(f), "q", "--model", "m", "--max-bytes", "0"]) == 2
    assert main([str(f), "q", "--model", "m", "--max-file-bytes", "0"]) == 2


def test_verbose_streams_clean_progress_without_loop_dump(stub_server, tmp_path, capsys):
    # --verbose = one-line progress on stderr; the full loop trace (code
    # blocks, "LLM Response:") stays behind --trace.
    doc = tmp_path / "doc.txt"
    doc.write_text("content")
    stub_server.root_responses = [ANSWER_FROM_FILE]
    exit_code = main(_e2e_argv(stub_server, doc, "q", extra=["--verbose"]))
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "droste: Iteration" in captured.err
    assert "LLM Response" not in captured.err
    assert "====" not in captured.err
    assert captured.out.strip() == "content"  # stdout is the answer, only


def test_trace_dumps_full_loop(stub_server, tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("content")
    stub_server.root_responses = [ANSWER_FROM_FILE]
    exit_code = main(_e2e_argv(stub_server, doc, "q", extra=["--trace"]))
    captured = capsys.readouterr()
    assert exit_code == 0
    # Trace renders progress through render_verbose (#35): banner form,
    # not the --verbose echo's "droste: " marker.
    assert "Iteration 1/" in captured.err and "=" * 60 in captured.err
    assert "LLM Response" in captured.err  # dump goes to stderr too
    assert captured.out.strip() == "content"  # stdout stays answer-only


def test_verbose_streams_generated_code_live(stub_server, tmp_path, capsys):
    # --verbose shows the root model's code AS IT GENERATES (SSE deltas on
    # stderr), with progress lines starting on fresh lines.
    doc = tmp_path / "doc.txt"
    doc.write_text("content")
    stub_server.root_responses = [ANSWER_FROM_FILE]
    exit_code = main(_e2e_argv(stub_server, doc, "q", extra=["--verbose"]))
    captured = capsys.readouterr()
    assert exit_code == 0
    # The generated code itself appears on stderr (streamed)...
    assert "answer['ready'] = True" in captured.err
    # ...requested as a streaming call...
    assert any(r.get("stream") for r in stub_server.requests)
    # ...and stdout stays answer-only.
    assert captured.out.strip() == "content"


def test_default_run_does_not_stream(stub_server, tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("content")
    stub_server.root_responses = [ANSWER_FROM_FILE]
    exit_code = main(_e2e_argv(stub_server, doc, "q"))
    assert exit_code == 0
    assert not any(r.get("stream") for r in stub_server.requests)


def test_keyless_default_endpoint_fails_upfront(tmp_path, monkeypatch, capsys):
    # An explicit api.openai.com endpoint without a key can only die
    # mid-flight with a raw 401 — fail upfront with one clean line.
    f = tmp_path / "a.txt"
    f.write_text("x")
    code = main(
        [
            str(f),
            "q",
            "--model",
            "gpt-5.2-mini",
            "--base-url",
            "https://api.openai.com/v1",
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "no API key" in err
    assert "401" not in err  # one clean line, not a raw HTTP dump


def test_keyless_custom_base_url_is_allowed(stub_server, tmp_path, monkeypatch, capsys):
    # Ollama-style endpoints are legitimately keyless.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    doc = tmp_path / "doc.txt"
    doc.write_text("local content")
    stub_server.root_responses = [ANSWER_FROM_FILE]
    code = main(
        [
            str(doc),
            "q",
            "--model",
            "root-model",
            "--base-url",
            stub_server.base_url,
            "--max-iterations",
            "2",
        ]
    )
    assert code == 0
    assert capsys.readouterr().out.strip() == "local content"


def test_file_args_ignore_inherited_stdin(stub_server, tmp_path, capsys, monkeypatch):
    # `droste "q" file` with an unrelated inherited pipe: the pipe's content
    # must not leak into the context (unix rule: file args win, no `-` given).
    import io

    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(b"unrelated pipe noise")))
    doc = tmp_path / "doc.txt"
    doc.write_text("file content")
    stub_server.root_responses = [
        "```python\n"
        "answer['content'] = ' | '.join(f['path'] for f in context['files'])\n"
        "answer['ready'] = True\n"
        "```",
    ]
    code = main(_e2e_argv(stub_server, doc, "q"))
    out = capsys.readouterr().out
    assert code == 0
    assert "<stdin>" not in out


def test_key_check_matches_hostname_not_substring(tmp_path, monkeypatch, capsys):
    # codex review: a proxy URL merely containing the string must not
    # trip the guard; host comparison is exact and case-insensitive.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    f = tmp_path / "a.txt"
    f.write_text("x")
    # Path contains the string but the host is a keyless proxy: allowed
    # through the guard (fails later at connect, which is correct).
    code = main(
        [
            str(f),
            "q",
            "--model",
            "m",
            "--base-url",
            "http://127.0.0.1:9/api.openai.com",
            "--max-iterations",
            "1",
        ]
    )
    assert code == 1  # connection failure, NOT the exit-2 key usage error
    assert "no API key" not in capsys.readouterr().err
    # Uppercase host is still the real endpoint: guard fires.
    code = main([str(f), "q", "--model", "m", "--base-url", "https://API.OPENAI.COM/v1"])
    assert code == 2
    assert "no API key" in capsys.readouterr().err


# --- stored ModelRelay credentials: resolution order + e2e ---


def _write_credentials(base_url: str, model: str = "cred-model") -> None:
    from droste_cli.credentials import Credentials, save_credentials

    save_credentials(
        Credentials(
            base_url=base_url,
            api_key="mr_sk_stored",
            email="dev@example.com",
            default_model=model,
        )
    )


@pytest.fixture()
def stub_native_server():
    from test_modelrelay_client import StubResponsesServer

    server = StubResponsesServer()
    yield server
    server.shutdown()


def test_stored_credentials_run_end_to_end(stub_native_server, tmp_path, capsys):
    # No flags, no env keys: the stored login runs over native /responses
    # and the model defaults from the credentials.
    _write_credentials(stub_native_server.base_url)
    doc = tmp_path / "doc.txt"
    doc.write_text("logged-in content")
    stub_native_server.root_responses = [ANSWER_FROM_FILE]

    exit_code = main([str(doc), "what does it say?", "--max-iterations", "2"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "logged-in content"

    request = stub_native_server.requests[0]
    assert request["model"] == "cred-model"
    headers = stub_native_server.headers[0]
    assert headers.get("x-modelrelay-api-key") == "mr_sk_stored"


def test_stored_login_wins_over_ambient_env_keys(
    stub_server, stub_native_server, tmp_path, monkeypatch, capsys
):
    # Running `droste login` IS the explicit choice, so ModelRelay stays
    # the default even with OPENAI_* exported in the shell (most developers
    # have ambient keys; the free credits must not be silently bypassed).
    # Flags or `droste logout` switch back to BYOK.
    _write_credentials(stub_native_server.base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
    monkeypatch.setenv("OPENAI_BASE_URL", stub_server.base_url)
    doc = tmp_path / "doc.txt"
    doc.write_text("logged-in content")
    stub_native_server.root_responses = [ANSWER_FROM_FILE]

    exit_code = main([str(doc), "q", "--max-iterations", "2"])
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "logged-in content"
    assert stub_native_server.requests, "the stored login must be used"
    assert not stub_server.requests, "ambient env keys must not shadow the login"


def test_flags_override_stored_login_for_one_run(stub_server, stub_native_server, tmp_path, capsys):
    # Per-run flags are the BYOK escape hatch while logged in.
    _write_credentials(stub_native_server.base_url)
    doc = tmp_path / "doc.txt"
    doc.write_text("byok content")
    stub_server.root_responses = [ANSWER_FROM_FILE]

    exit_code = main(
        [
            str(doc),
            "q",
            "--model",
            "root-model",
            "--base-url",
            stub_server.base_url,
            "--api-key",
            "k",
            "--max-iterations",
            "2",
        ]
    )
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "byok content"
    assert stub_server.requests, "flags must route to BYOK"
    assert not stub_native_server.requests, "stored credentials must not be used"


def test_env_keys_used_when_not_logged_in(stub_server, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "byok-key")
    monkeypatch.setenv("OPENAI_BASE_URL", stub_server.base_url)
    doc = tmp_path / "doc.txt"
    doc.write_text("byok content")
    stub_server.root_responses = [ANSWER_FROM_FILE]
    exit_code = main([str(doc), "q", "--model", "root-model", "--max-iterations", "2"])
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "byok content"


def test_malformed_env_key_without_login_fails_loudly(tmp_path, monkeypatch, capsys):
    # A set-but-malformed key still selects the BYOK path (when not logged
    # in): fail loudly there, never mask it with the no-credentials error.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-anthropic-key")
    doc = tmp_path / "doc.txt"
    doc.write_text("x")
    code = main([str(doc), "q", "--model", "gpt-5.2-mini"])
    assert code == 2
    err = capsys.readouterr().err
    assert "no API key" in err
    assert "no credentials" not in err


def test_explicit_model_beats_credentials_default(stub_native_server, tmp_path, capsys):
    _write_credentials(stub_native_server.base_url)
    doc = tmp_path / "doc.txt"
    doc.write_text("x")
    stub_native_server.root_responses = [ANSWER_FROM_FILE]
    exit_code = main([str(doc), "q", "--model", "override-model", "--max-iterations", "2"])
    assert exit_code == 0
    assert stub_native_server.requests[0]["model"] == "override-model"


def test_corrupt_credentials_file_is_actionable(tmp_path, capsys):
    import os

    from droste_cli.credentials import credentials_path

    path = credentials_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("not json")
    doc = tmp_path / "doc.txt"
    doc.write_text("x")
    assert main([str(doc), "q"]) == 2
    assert "droste login" in capsys.readouterr().err


def test_stored_byok_credentials_run_end_to_end(stub_server, tmp_path, capsys):
    # A key chosen during setup is stored and runs the BYOK path — no env
    # vars involved (choosing keys is a setup step, not an export).
    from droste_cli.credentials import Credentials, save_credentials

    save_credentials(
        Credentials(
            api_key="sk-stored-byok",
            base_url=stub_server.base_url,
            provider="byok",
            default_model="root-model",
        )
    )
    doc = tmp_path / "doc.txt"
    doc.write_text("byok stored content")
    stub_server.root_responses = [ANSWER_FROM_FILE]
    exit_code = main([str(doc), "q", "--max-iterations", "2"])
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "byok stored content"
    assert stub_server.requests[0]["model"] == "root-model"


def test_first_run_interactive_setup_then_continues(stub_server, tmp_path, monkeypatch, capsys):
    # No credentials + interactive terminal: the chooser runs in-line and
    # the original ask continues with the stored choice.
    monkeypatch.setattr("sys.stdin", type("T", (), {"isatty": staticmethod(lambda: True)})())
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-import-1")
    answers = ["2", "", stub_server.base_url, "root-model"]
    monkeypatch.setattr("builtins.input", lambda *a: answers.pop(0))

    doc = tmp_path / "doc.txt"
    doc.write_text("setup then run")
    stub_server.root_responses = [ANSWER_FROM_FILE]
    exit_code = main([str(doc), "q", "--max-iterations", "2"])
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "setup then run"

    from droste_cli.credentials import load_credentials

    creds = load_credentials()
    assert creds is not None and creds.provider == "byok"


def test_stored_byok_ignores_ambient_base_url(stub_server, tmp_path, monkeypatch, capsys):
    # Stored credentials pin the endpoint: ambient OPENAI_BASE_URL must not
    # redirect a stored key to a different server.
    from droste_cli.credentials import Credentials, save_credentials

    save_credentials(
        Credentials(
            api_key="sk-stored-byok",
            base_url=stub_server.base_url,
            provider="byok",
            default_model="root-model",
        )
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/nowhere")
    doc = tmp_path / "doc.txt"
    doc.write_text("pinned endpoint")
    stub_server.root_responses = [ANSWER_FROM_FILE]
    exit_code = main([str(doc), "q", "--max-iterations", "2"])
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "pinned endpoint"
    assert stub_server.requests, "the stored endpoint must be used"


def test_stored_anthropic_key_routes_to_anthropic_despite_env(tmp_path, monkeypatch):
    # A stored sk-ant key selects the Anthropic client even when
    # OPENAI_BASE_URL is exported (never send an Anthropic key elsewhere).
    from droste_cli.credentials import Credentials, save_credentials
    from droste_cli.main import build_parser, resolve_run_target

    save_credentials(
        Credentials(api_key="sk-ant-stored", provider="byok", default_model="claude-x")
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/nowhere")
    args = build_parser().parse_args(["q"])
    provider, creds = resolve_run_target(args)
    assert provider == "anthropic"
    assert creds is None
    assert args.api_key == "sk-ant-stored"
    assert args.model == "claude-x"


def test_unknown_provider_in_credentials_is_rejected(tmp_path, capsys):
    import json
    import os as _os

    from droste_cli.credentials import credentials_path

    path = credentials_path()
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"provider": "byo", "api_key": "k", "base_url": "https://x"}, fh)
    doc = tmp_path / "doc.txt"
    doc.write_text("x")
    assert main([str(doc), "q"]) == 2
    assert "unknown provider" in capsys.readouterr().err
