"""droste CLI (#28, #44): arg parsing, the unix contract, end-to-end over the stub server."""

from __future__ import annotations

import json
import sqlite3

import pytest

from droste_cli.main import main, build_parser

from test_openai_compat_client import StubOpenAIServer


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
            "a.txt", "b.txt", "what changed?",
            "--model", "m", "--base-url", "http://x/v1", "--api-key", "k",
            "--subcall-model", "sm", "--subcall-max-output-tokens", "512",
            "--reasoning-effort", "low", "--max-iterations", "3",
            "--max-subcalls", "7", "--json", "--verbose", "--quiet",
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
    monkeypatch.delenv("DROSTE_MODEL", raising=False)
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert main([str(f), "q"]) == 2
    assert "--model is required" in capsys.readouterr().err


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


# --- --db wiring to the local SQL source (droste#29) ---


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
        "--model", "root-model",
        "--base-url", server.base_url,
        "--api-key", "k",
        "--max-iterations", "2",
        *extra,
    ]


ANSWER_FROM_FILE = (
    "```python\n"
    "answer['content'] = context['files'][0]['text']\n"
    "answer['ready'] = True\n"
    "```"
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
        "rows = db.query(\"SELECT name FROM t\")\n"
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
        "```python\n"
        "answer['content'] = 'json answer'\n"
        "answer['ready'] = True\n"
        "```",
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
            str(doc), "q",
            "--model", "root-model",
            "--base-url", stub_server.base_url,
            "--api-key", "k",
            "--max-iterations", "1",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "extracted final answer"
    assert "extracted from partial work" in captured.err


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
