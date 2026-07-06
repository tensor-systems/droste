"""Input classification and ingestion (#44): the fact-based magic layer.

The contract under test: args that exist are data, the one that doesn't is
the question, SQLite is detected by magic bytes, ambiguity errors loudly,
and skips are counted — never silent.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

from droste_cli.inputs import (
    Classified,
    InputError,
    classify,
    is_sqlite_file,
    load_inputs,
    walk_directory,
    _WalkStats,
)


def make_sqlite(path):
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (1);")
    conn.commit()
    conn.close()


# --- classification: filesystem facts, not guesses ---


def test_classify_files_dirs_dbs_and_question(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("hello")
    d = tmp_path / "docs"
    d.mkdir()
    db = tmp_path / "app.db"
    make_sqlite(db)

    c = classify([str(f), str(d), str(db), "what changed?"])
    assert c.files == [str(f)]
    assert c.dirs == [str(d)]
    assert c.dbs == [str(db)]
    assert c.question == "what changed?"


def test_classify_order_does_not_matter(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    first = classify(["why?", str(f)])
    second = classify([str(f), "why?"])
    assert first.question == second.question == "why?"
    assert first.files == second.files == [str(f)]


def test_classify_sqlite_by_magic_bytes_not_extension(tmp_path):
    # A .txt that IS a SQLite database is classified as a database…
    disguised = tmp_path / "data.txt"
    make_sqlite(disguised)
    assert is_sqlite_file(str(disguised))
    assert classify([str(disguised), "q"]).dbs == [str(disguised)]

    # …and a .db that is plain text is classified as a file.
    fake = tmp_path / "notes.db"
    fake.write_text("just text")
    c = classify([str(fake), "q"])
    assert c.files == [str(fake)]
    assert c.dbs == []


def test_classify_two_questions_is_an_error(tmp_path):
    with pytest.raises(InputError, match="multiple question candidates"):
        classify(["what changed", "since yesterday"])


def test_classify_pathlike_typo_is_an_error_not_a_question():
    with pytest.raises(InputError, match="looks\\s+like a path"):
        classify(["reprot.txt", "q"])
    with pytest.raises(InputError, match="looks\\s+like a path"):
        classify(["missing/dir", "q"])


def test_classify_dash_means_stdin():
    c = classify(["-", "q"])
    assert c.reads_stdin is True
    assert c.stdin_explicit is True
    assert c.question == "q"


def test_classify_question_with_slash_and_spaces_is_a_question():
    # codex review (#44): "client/server" inside a spaced question must not
    # trip the path-typo detector.
    c = classify(["what is the client/server split?"])
    assert c.question == "what is the client/server split?"


def test_classify_tilde_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    f = tmp_path / "home.txt"
    f.write_text("x")
    c = classify(["~/home.txt", "q"])
    assert c.files == [str(f)]


# --- directory walking: deterministic, capped, counted ---


def test_walk_skips_junk_hidden_binary_and_capped(tmp_path):
    (tmp_path / "keep.md").write_text("text")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.txt").write_text("more")
    (tmp_path / ".hidden").write_text("secret")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("ignored")
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")
    (tmp_path / "big.txt").write_text("x" * 100)

    stats = _WalkStats()
    files = walk_directory(str(tmp_path), max_file_bytes=50, stats=stats)
    names = sorted(os.path.basename(p) for p in files)
    assert names == ["inner.txt", "keep.md"]
    assert stats.binary == 1
    assert stats.over_cap == 1
    assert stats.ignored == 2  # .hidden file + node_modules dir


def test_walk_is_deterministically_sorted(tmp_path):
    for name in ["c.txt", "a.txt", "b.txt"]:
        (tmp_path / name).write_text("x")
    stats = _WalkStats()
    files = walk_directory(str(tmp_path), max_file_bytes=1000, stats=stats)
    assert [os.path.basename(p) for p in files] == ["a.txt", "b.txt", "c.txt"]


# --- loading: budgets, defaults, the report line ---


def test_load_requires_a_question(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    with pytest.raises(InputError, match="no question"):
        load_inputs(Classified(files=[str(f)]))


def test_load_explicit_file_bypasses_per_file_cap(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("x" * 500)
    loaded = load_inputs(
        Classified(question="q", files=[str(big)]), max_file_bytes=50
    )
    assert loaded.context is not None
    assert loaded.context["files"][0]["size"] == 500


def test_load_total_budget_is_a_loud_error(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x" * 100)
    with pytest.raises(InputError, match="total budget"):
        load_inputs(
            Classified(question="q", files=[str(f)]), max_total_bytes=10
        )


def test_load_multiple_dbs_is_an_error(tmp_path):
    one = tmp_path / "one.db"
    two = tmp_path / "two.db"
    make_sqlite(one)
    make_sqlite(two)
    with pytest.raises(InputError, match="multiple databases"):
        load_inputs(Classified(question="q", dbs=[str(one), str(two)]))
    with pytest.raises(InputError, match="multiple databases"):
        load_inputs(Classified(question="q", dbs=[str(one)]), db_flag=str(two))


def test_load_no_data_walks_cwd(tmp_path, monkeypatch):
    (tmp_path / "readme.md").write_text("hello world")
    monkeypatch.chdir(tmp_path)
    loaded = load_inputs(Classified(question="q"))
    assert loaded.context is not None
    assert [f["name"] for f in loaded.context["files"]] == ["readme.md"]
    assert "from ." in loaded.report


def test_load_no_data_empty_cwd_is_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(InputError, match="nothing to ask over"):
        load_inputs(Classified(question="q"))


def test_load_explicit_empty_dir_errors_instead_of_cwd_fallback(tmp_path, monkeypatch):
    # codex review (#44): an explicitly-passed dir with nothing readable must
    # error, never quietly swap in the current directory.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "unrelated.md").write_text("should never load")
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(cwd)
    with pytest.raises(InputError, match="no readable text files in"):
        load_inputs(Classified(question="q", dirs=[str(empty)]))


def test_load_explicit_dash_with_empty_stdin_errors(tmp_path, monkeypatch):
    # Explicit `-` with nothing piped is an error…
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "unrelated.md").write_text("should never load")
    monkeypatch.chdir(cwd)
    with pytest.raises(InputError, match="requested but the pipe was empty"):
        load_inputs(
            Classified(question="q", reads_stdin=True, stdin_explicit=True),
            stdin_text=None,
        )
    # …while an empty *implicit* pipe (cron's /dev/null) still gets the cwd
    # default.
    loaded = load_inputs(
        Classified(question="q", reads_stdin=True), stdin_text=None
    )
    assert loaded.context is not None
    assert [f["name"] for f in loaded.context["files"]] == ["unrelated.md"]


def test_load_budget_counts_utf8_bytes_not_characters(tmp_path):
    # codex review (#44): 100 é characters = 200 UTF-8 bytes; a 150-byte
    # budget must reject them even though len(text) == 100.
    f = tmp_path / "accents.txt"
    f.write_text("é" * 100, encoding="utf-8")
    with pytest.raises(InputError, match="total budget"):
        load_inputs(
            Classified(question="q", files=[str(f)]), max_total_bytes=150
        )


def test_load_stdin_becomes_a_context_file():
    loaded = load_inputs(
        Classified(question="q", reads_stdin=True), stdin_text="piped log line"
    )
    assert loaded.context is not None
    assert loaded.context["files"][0]["path"] == "<stdin>"
    assert loaded.context["files"][0]["text"] == "piped log line"


def test_report_line_counts_everything(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("text")
    (docs / "blob.bin").write_bytes(b"\x00")
    db = tmp_path / "app.db"
    make_sqlite(db)
    loaded = load_inputs(
        Classified(question="q", dirs=[str(docs)], dbs=[str(db)])
    )
    assert "loaded 1 file" in loaded.report
    assert f"db: {db}" in loaded.report
    assert "1 binary" in loaded.report


def test_stdin_read_is_bounded_by_budget(monkeypatch):
    # codex review (#44): a pipe larger than the budget errors immediately
    # instead of buffering the whole stream.
    import io

    fake = io.TextIOWrapper(io.BytesIO(b"x" * 100))
    monkeypatch.setattr("sys.stdin", fake)
    from droste_cli.inputs import read_piped_stdin

    with pytest.raises(InputError, match="stdin exceeds the total budget"):
        read_piped_stdin(limit=50, explicit=True)

    fake_small = io.TextIOWrapper(io.BytesIO(b"y" * 10))
    monkeypatch.setattr("sys.stdin", fake_small)
    assert read_piped_stdin(limit=50, explicit=True) == "y" * 10


def test_walk_counts_unreadable_subtrees(tmp_path):
    # codex review (#44): os.walk scan errors (permissions) are counted, not
    # silently swallowed.
    (tmp_path / "ok.md").write_text("fine")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "secret.md").write_text("hidden")
    locked.chmod(0o000)
    try:
        stats = _WalkStats()
        files = walk_directory(str(tmp_path), max_file_bytes=1000, stats=stats)
        assert [os.path.basename(p) for p in files] == ["ok.md"]
        assert stats.unreadable == 1
    finally:
        locked.chmod(0o755)


def test_stdin_whitespace_only_is_still_data(monkeypatch):
    # codex review (#44): a pipe of blank lines is data (a question about
    # formatting is valid); only a zero-length read means "no stdin".
    import io

    from droste_cli.inputs import read_piped_stdin

    fake = io.TextIOWrapper(io.BytesIO(b"\n\n  \n"))
    monkeypatch.setattr("sys.stdin", fake)
    assert read_piped_stdin(limit=50, explicit=True) == "\n\n  \n"

    empty = io.TextIOWrapper(io.BytesIO(b""))
    monkeypatch.setattr("sys.stdin", empty)
    assert read_piped_stdin(limit=50, explicit=True) is None


def test_load_over_budget_explicit_file_errors_without_reading(tmp_path, monkeypatch):
    # codex review (#44): the budget error fires on the size pre-check —
    # before the file is materialized.
    big = tmp_path / "big.txt"
    big.write_text("x" * 1000)

    def boom(path):
        raise AssertionError("over-budget file must not be read")

    monkeypatch.setattr("droste_cli.inputs._read_text", boom)
    with pytest.raises(InputError, match="total budget"):
        load_inputs(
            Classified(question="q", files=[str(big)]), max_total_bytes=100
        )


def test_walk_skips_fifos_and_special_files(tmp_path):
    # codex review (#44): a FIFO with no writer must be skipped, not opened
    # (opening it would block forever).
    (tmp_path / "ok.md").write_text("fine")
    os.mkfifo(tmp_path / "pipe.fifo")
    stats = _WalkStats()
    files = walk_directory(str(tmp_path), max_file_bytes=1000, stats=stats)
    assert [os.path.basename(p) for p in files] == ["ok.md"]
    assert stats.ignored == 1


def test_walk_counts_symlinked_dirs_as_ignored(tmp_path):
    # codex review (#44): symlinked subdirs are not followed (cycle safety) —
    # they must be counted, never silently omitted.
    real = tmp_path / "real"
    real.mkdir()
    (real / "inside.md").write_text("content")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "top.md").write_text("top")
    os.symlink(real, docs / "api")
    stats = _WalkStats()
    files = walk_directory(str(docs), max_file_bytes=1000, stats=stats)
    assert [os.path.basename(p) for p in files] == ["top.md"]
    assert stats.ignored == 1


def test_load_explicit_dash_empty_pipe_errors_even_with_other_data(tmp_path):
    # codex review (#44): `cat empty | droste - config.yml "q"` must not
    # silently proceed without the requested stdin.
    f = tmp_path / "config.yml"
    f.write_text("key: value")
    with pytest.raises(InputError, match="requested but the pipe was empty"):
        load_inputs(
            Classified(question="q", files=[str(f)], reads_stdin=True, stdin_explicit=True),
            stdin_text=None,
        )


def test_load_explicit_empty_dir_errors_even_with_other_inputs(tmp_path):
    # codex review (#44): `droste empty-dir notes.txt "q"` must error on the
    # empty dir, not silently proceed over notes.txt alone.
    f = tmp_path / "notes.txt"
    f.write_text("content")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(InputError, match="no readable text files in"):
        load_inputs(
            Classified(question="q", files=[str(f)], dirs=[str(empty)])
        )


def test_slow_pipe_producer_is_waited_for(monkeypatch):
    # grep-style: a bare invocation under a pipeline waits for its producer —
    # slow first bytes are data, not absence (codex review, #53).
    import io
    import os as _os
    import threading

    r_fd, w_fd = _os.pipe()

    def slow_writer():
        import time

        time.sleep(0.4)
        _os.write(w_fd, b"late data")
        _os.close(w_fd)

    threading.Thread(target=slow_writer, daemon=True).start()
    reader = io.TextIOWrapper(_os.fdopen(r_fd, "rb"))
    monkeypatch.setattr("sys.stdin", reader)
    from droste_cli.inputs import read_piped_stdin

    assert read_piped_stdin(limit=100) == "late data"
