"""droste CLI (#28): arg parsing, context assembly, end-to-end over the stub server."""

from __future__ import annotations

import importlib
import json

import pytest

from droste_cli.main import CLIError, build_context, build_parser, main

from test_openai_compat_client import StubOpenAIServer


@pytest.fixture()
def stub_server():
    server = StubOpenAIServer()
    yield server
    server.shutdown()


# --- arg parsing ---


def test_parse_ask_files_and_question():
    args = build_parser().parse_args(["ask", "a.txt", "b.txt", "what changed?"])
    assert args.command == "ask"
    assert args.files == ["a.txt", "b.txt"]
    assert args.question == "what changed?"
    assert args.subcall_max_output_tokens == 2048
    assert args.max_iterations == 20
    assert args.max_subcalls == 50
    assert not args.json and not args.verbose and args.db is None


def test_parse_ask_db_only():
    args = build_parser().parse_args(["ask", "--db", "app.db", "who churned?"])
    assert args.files == []
    assert args.question == "who churned?"
    assert args.db == "app.db"


def test_parse_engine_knob_flags():
    args = build_parser().parse_args(
        [
            "ask", "a.txt", "q",
            "--model", "m", "--base-url", "http://x/v1", "--api-key", "k",
            "--subcall-model", "sm", "--subcall-max-output-tokens", "512",
            "--reasoning-effort", "low", "--max-iterations", "3",
            "--max-subcalls", "7", "--json", "--verbose",
        ]
    )
    assert (args.model, args.base_url, args.api_key) == ("m", "http://x/v1", "k")
    assert (args.subcall_model, args.subcall_max_output_tokens) == ("sm", 512)
    assert args.reasoning_effort == "low"
    assert (args.max_iterations, args.max_subcalls) == (3, 7)
    assert args.json and args.verbose


def test_model_env_default(monkeypatch):
    monkeypatch.setenv("DROSTE_MODEL", "env-model")
    args = build_parser().parse_args(["ask", "a.txt", "q"])
    assert args.model == "env-model"


def test_missing_model_is_usage_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DROSTE_MODEL", raising=False)
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert main(["ask", str(f), "q"]) == 2
    assert "--model is required" in capsys.readouterr().err


def test_no_files_and_no_db_is_usage_error(capsys):
    assert main(["ask", "q", "--model", "m"]) == 2
    assert "nothing to ask over" in capsys.readouterr().err


# --- context assembly ---


def test_build_context_shape(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("alpha content")
    b = tmp_path / "notes.md"
    b.write_text("beta")
    context = build_context([str(a), str(b)])
    assert set(context.keys()) == {"files"}
    first = context["files"][0]
    assert first["path"] == str(a)
    assert first["name"] == "a.txt"
    assert first["text"] == "alpha content"
    assert first["size"] == len("alpha content")
    assert context["files"][1]["name"] == "notes.md"


def test_build_context_empty_is_none():
    assert build_context([]) is None


def test_build_context_missing_file_raises_cli_error(tmp_path):
    with pytest.raises(CLIError, match="cannot read"):
        build_context([str(tmp_path / "missing.txt")])


# --- --db wiring to the local SQL source (rlm-core#29) ---


def test_db_exposes_sqlite_as_source(tmp_path):
    import sqlite3

    from droste_cli.main import _load_sql_source

    db = tmp_path / "app.db"
    conn = sqlite3.connect(db)
    conn.executescript("CREATE TABLE t (id INTEGER, name TEXT); INSERT INTO t VALUES (1, 'ada');")
    conn.commit()
    conn.close()

    source = _load_sql_source(str(db))
    assert source.name() == "db"
    assert "t(" in source.get_schema()
    rows = source.query("SELECT name FROM t")
    assert rows == [{"name": "ada"}]


def test_db_missing_file_errors(capsys):
    assert main(["ask", "--db", "/nonexistent/app.db", "q", "--model", "m"]) == 2
    assert "database not found" in capsys.readouterr().err


# --- end-to-end over the stub OpenAI-compatible server ---


def _e2e_argv(server, path, extra=()):
    return [
        "ask", str(path), "what does the file say?",
        "--model", "root-model",
        "--base-url", server.base_url,
        "--api-key", "k",
        "--max-iterations", "2",
        *extra,
    ]


def test_ask_file_end_to_end(stub_server, tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("the launch is on friday")
    stub_server.root_responses = [
        "```python\n"
        "answer['content'] = context['files'][0]['text']\n"
        "answer['ready'] = True\n"
        "```",
    ]
    exit_code = main(_e2e_argv(stub_server, doc))
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "the launch is on friday"
    # The system prompt must carry the context size + preview description.
    system_prompt = stub_server.requests[0]["messages"][0]["content"]
    assert "doc.txt" in system_prompt
    assert "1 file(s)" in system_prompt


def test_ask_json_shape(stub_server, tmp_path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("payload text")
    stub_server.root_responses = [
        "```python\n"
        "answer['content'] = 'json answer'\n"
        "answer['ready'] = True\n"
        "```",
    ]
    exit_code = main(_e2e_argv(stub_server, doc, extra=["--json"]))
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
            "ask", str(doc), "q",
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
    exit_code = main(_e2e_argv(stub_server, doc))
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "provider down" in captured.err


# --- upfront validation of usage errors (exit 2, not a mid-run crash) ---


def test_negative_subcall_max_output_tokens_is_usage_error(tmp_path, capsys):
    f = tmp_path / "a.txt"
    f.write_text("hi")
    code = main(["ask", str(f), "q", "--model", "m", "--subcall-max-output-tokens", "-1"])
    assert code == 2
    assert "must be >= 0" in capsys.readouterr().err


def test_max_iterations_below_one_is_usage_error(tmp_path, capsys):
    f = tmp_path / "a.txt"
    f.write_text("hi")
    code = main(["ask", str(f), "q", "--model", "m", "--max-iterations", "0"])
    assert code == 2
    assert "must be >= 1" in capsys.readouterr().err


def test_db_directory_is_usage_error(tmp_path, capsys):
    code = main(["ask", "--db", str(tmp_path), "q", "--model", "m"])
    assert code == 2
    assert "not a file" in capsys.readouterr().err


def test_db_non_sqlite_file_is_usage_error(tmp_path, capsys):
    bogus = tmp_path / "notes.txt"
    bogus.write_text("this is not a database")
    code = main(["ask", "--db", str(bogus), "q", "--model", "m"])
    assert code == 2
    assert "SQLite database" in capsys.readouterr().err
