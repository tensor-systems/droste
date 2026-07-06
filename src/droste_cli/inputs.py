"""Argument classification and directory ingestion for the droste CLI (#44).

The CLI's contract in one breath: *args that exist are data, the one that
doesn't is the question, no args means here, pipes are data too, and it
always tells you what it read.*

All magic here is built on **facts, not guesses**: a path either exists or it
doesn't; a file either starts with the SQLite magic bytes or it doesn't; stdin
either is a pipe or it isn't. Nothing infers intent from phrasing, and every
ambiguity resolves to a loud error instead of a silent choice:

- a non-existent arg that *looks like* a path (has a separator or an
  extension, no spaces) is a typo, not a question;
- two non-path args means two question candidates — error, never joined;
- blowing the total-size budget is an error, never a silent truncation.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

SQLITE_MAGIC = b"SQLite format 3\x00"

#: Directory names that are always pruned during a walk — the junk everyone
#: means to skip. Deliberately a fixed list (not .gitignore semantics, which
#: drag in negations/nesting): predictable beats clever at v1.
SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        "dist",
        "build",
        "target",
        ".next",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)

DEFAULT_MAX_FILE_BYTES = 2_000_000
DEFAULT_MAX_TOTAL_BYTES = 50_000_000

#: A non-existent positional with no spaces, an extension or separator, is a
#: mistyped path, not a question ("reprot.txt" should error, not be asked).
_PATHLIKE = re.compile(r"^[^\s]+\.[A-Za-z0-9]{1,8}$")


class InputError(Exception):
    """User-facing input problem: message to stderr, exit code 2."""


@dataclass
class Classified:
    """What the positionals turned out to be, by filesystem fact."""

    question: str | None = None
    files: list[str] = field(default_factory=list)
    dirs: list[str] = field(default_factory=list)
    dbs: list[str] = field(default_factory=list)
    reads_stdin: bool = False
    #: True only for an explicit `-` positional — an empty explicit stdin is
    #: an error, while an empty *implicit* pipe (cron's /dev/null) just means
    #: "no stdin data" and must not block the cwd default.
    stdin_explicit: bool = False


@dataclass
class LoadedInputs:
    """Materialized context plus the ingestion report."""

    context: dict[str, Any] | None
    db_path: str | None
    question: str
    #: One human line describing everything that was read — printed to stderr
    #: on every run. The report is what makes the magic trustworthy: the tool
    #: never reads anything it doesn't say.
    report: str


def _has_whitespace(text: str) -> bool:
    return any(ch.isspace() for ch in text)


def is_sqlite_file(path: str) -> bool:
    """Fact check: does the file start with the 16 SQLite magic bytes?"""
    try:
        with open(path, "rb") as handle:
            return handle.read(len(SQLITE_MAGIC)) == SQLITE_MAGIC
    except OSError:
        return False


def classify(positionals: list[str], *, stdin_is_tty: bool = True) -> Classified:
    """Sort positionals into question/files/dirs/dbs by filesystem facts."""
    result = Classified(reads_stdin=not stdin_is_tty)
    question_candidates: list[str] = []

    for arg in positionals:
        if arg == "-":
            result.reads_stdin = True
            result.stdin_explicit = True
            continue
        path = os.path.expanduser(arg)
        if os.path.isdir(path):
            result.dirs.append(path)
        elif os.path.isfile(path):
            if is_sqlite_file(path):
                result.dbs.append(path)
            else:
                result.files.append(path)
        elif os.path.exists(path):
            raise InputError(f"{arg}: not a regular file or directory")
        elif not _has_whitespace(arg) and (
            os.sep in arg or (os.altsep and os.altsep in arg) or "/" in arg or _PATHLIKE.match(arg)
        ):
            # No-whitespace only: "what is the client/server split?" is a
            # question; "missing/dir" is a typo.
            raise InputError(
                f"{arg}: no such file or directory (a non-existent arg that looks "
                "like a path is treated as a typo, not as the question)"
            )
        else:
            question_candidates.append(arg)

    if len(question_candidates) > 1:
        listed = " | ".join(question_candidates)
        raise InputError(
            f"multiple question candidates ({listed}); quote the question as a single argument"
        )
    if question_candidates:
        result.question = question_candidates[0]
    return result


def _is_binary(chunk: bytes) -> bool:
    return b"\x00" in chunk


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


@dataclass
class _WalkStats:
    binary: int = 0
    over_cap: int = 0
    ignored: int = 0
    unreadable: int = 0


def walk_directory(
    root: str,
    *,
    max_file_bytes: int,
    stats: _WalkStats,
) -> list[str]:
    """Collect readable text files under ``root``, deterministically ordered.

    Prunes SKIP_DIR_NAMES and hidden entries; skips binaries (NUL byte in the
    first 8 KB) and files over the per-file cap. Skips are *counted*, and the
    caller reports them — never silent.
    """
    collected: list[str] = []

    def _scan_error(_exc: OSError) -> None:
        # os.walk swallows unreadable subtrees by default; counting them keeps
        # the no-silent-skip contract.
        stats.unreadable += 1

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=_scan_error):
        kept_dirs = []
        for name in sorted(dirnames):
            if (
                name in SKIP_DIR_NAMES
                or name.startswith(".")
                # os.walk won't recurse into symlinked dirs (followlinks=False,
                # deliberately: cycle safety) — count them so the omission is
                # never silent.
                or os.path.islink(os.path.join(dirpath, name))
            ):
                stats.ignored += 1
            else:
                kept_dirs.append(name)
        dirnames[:] = kept_dirs

        for name in sorted(filenames):
            if name.startswith("."):
                stats.ignored += 1
                continue
            path = os.path.join(dirpath, name)
            if os.path.islink(path) or not os.path.isfile(path):
                # Symlinks, FIFOs, sockets, devices: opening a pipe with no
                # writer would block forever. Regular files only.
                stats.ignored += 1
                continue
            try:
                size = os.path.getsize(path)
                if size > max_file_bytes:
                    stats.over_cap += 1
                    continue
                with open(path, "rb") as handle:
                    head = handle.read(8192)
            except OSError:
                stats.ignored += 1
                continue
            if _is_binary(head):
                stats.binary += 1
                continue
            collected.append(path)
    return collected


def _human_bytes(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f} MB"
    if count >= 1_000:
        return f"{count / 1_000:.1f} kB"
    return f"{count} B"


def load_inputs(
    classified: Classified,
    *,
    db_flag: str | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    stdin_text: str | None = None,
) -> LoadedInputs:
    """Materialize the classified inputs into the sandbox context payload.

    Explicit files are intentional — they bypass the binary check and the
    per-file cap (only the total budget applies). Directory walks get the
    full skip logic. Exactly one database is allowed (from magic-byte
    detection or --db); more is an error, not a pick.
    """
    if classified.question is None:
        raise InputError("no question provided")

    dbs = list(classified.dbs)
    if db_flag:
        dbs.append(os.path.expanduser(db_flag))
    if len(dbs) > 1:
        raise InputError("multiple databases (" + ", ".join(dbs) + "); pass exactly one")
    db_path = dbs[0] if dbs else None

    stats = _WalkStats()
    paths: list[str] = list(classified.files)
    for root in classified.dirs:
        # Per-dir contribution check: an explicitly named directory that
        # yields nothing readable is an error even when other inputs exist —
        # a mistyped or empty target must never be silently omitted.
        dir_stats = _WalkStats()
        found = walk_directory(root, max_file_bytes=max_file_bytes, stats=dir_stats)
        stats.binary += dir_stats.binary
        stats.over_cap += dir_stats.over_cap
        stats.ignored += dir_stats.ignored
        stats.unreadable += dir_stats.unreadable
        if not found:
            raise InputError(
                f"no readable text files in {root} (skipped: "
                f"{dir_stats.binary} binary, {dir_stats.over_cap} over cap, "
                f"{dir_stats.ignored} ignored, {dir_stats.unreadable} unreadable)"
            )
        paths.extend(found)

    # The cwd default applies ONLY when no data source was requested at all —
    # explicit-but-empty inputs (a dir with nothing readable, `-` on empty
    # stdin) must error, never quietly swap in the current directory.
    has_stdin_data = stdin_text is not None and stdin_text != ""
    if classified.stdin_explicit and not has_stdin_data:
        raise InputError("stdin (-) was requested but the pipe was empty")
    data_requested = bool(
        classified.files
        or classified.dirs
        or db_path
        or classified.stdin_explicit
        or has_stdin_data
    )
    walked_cwd = False
    if not data_requested:
        paths = walk_directory(".", max_file_bytes=max_file_bytes, stats=stats)
        walked_cwd = True
        if not paths:
            raise InputError(
                "nothing to ask over: no readable text files here — pass "
                "files, a directory, or --db PATH"
            )

    files: list[dict[str, Any]] = []
    total = 0  # budget + report are byte-based (UTF-8), not character-based
    for path in paths:
        try:
            # Size pre-check so an over-budget file errors before it is ever
            # materialized (explicit files bypass the per-file cap).
            if total + os.path.getsize(path) > max_total_bytes:
                raise InputError(
                    f"inputs exceed the total budget ({_human_bytes(max_total_bytes)}); "
                    "narrow the target or raise --max-bytes"
                )
            text = _read_text(path)
        except OSError as exc:
            raise InputError(f"cannot read {path}: {exc}") from exc
        total += len(text.encode("utf-8"))
        if total > max_total_bytes:
            raise InputError(
                f"inputs exceed the total budget ({_human_bytes(max_total_bytes)}); "
                "narrow the target or raise --max-bytes"
            )
        display = os.path.relpath(path) if not os.path.isabs(path) else path
        files.append(
            {"path": display, "name": os.path.basename(path), "text": text, "size": len(text)}
        )

    if has_stdin_data:
        assert stdin_text is not None
        total += len(stdin_text.encode("utf-8"))
        if total > max_total_bytes:
            raise InputError(
                f"inputs exceed the total budget ({_human_bytes(max_total_bytes)}); "
                "narrow the target or raise --max-bytes"
            )
        files.append(
            {"path": "<stdin>", "name": "<stdin>", "text": stdin_text, "size": len(stdin_text)}
        )

    # The report line — always printed, exactly one line.
    parts: list[str] = []
    if files:
        origin = " from ." if walked_cwd else ""
        parts.append(
            f"loaded {len(files)} file{'s' if len(files) != 1 else ''} ({_human_bytes(total)}){origin}"
        )
    if db_path:
        parts.append(f"db: {db_path}")
    skips: list[str] = []
    if stats.binary:
        skips.append(f"{stats.binary} binary")
    if stats.over_cap:
        skips.append(f"{stats.over_cap} over {_human_bytes(max_file_bytes)} cap")
    if stats.ignored:
        skips.append(f"{stats.ignored} ignored")
    if stats.unreadable:
        skips.append(f"{stats.unreadable} unreadable")
    if skips:
        parts.append("skipped: " + ", ".join(skips))
    report = " · ".join(parts) if parts else "no local inputs"

    context = {"files": files} if files else None
    return LoadedInputs(
        context=context, db_path=db_path, question=classified.question, report=report
    )


def read_piped_stdin(limit: int = DEFAULT_MAX_TOTAL_BYTES, *, explicit: bool = False) -> str | None:
    """Read stdin when it's a pipe/redirect; None on a terminal.

    Reads at most ``limit`` bytes plus one — a stream over the budget errors
    immediately instead of buffering (or hanging on) an unbounded pipe. An
    empty read (e.g. cron's /dev/null) counts as no input.

    The caller only invokes this when stdin is the intended input: an
    explicit ``-``, or a pipe with no other data args (grep-style — a bare
    invocation under a pipeline waits for its producer, slow ones included).
    When other data args are present the caller never calls this at all,
    which is what keeps ``droste "q" file`` from hanging on an inherited
    silent pipe in scripts/CI.
    """
    if sys.stdin is None or sys.stdin.isatty():
        return None
    try:
        raw = sys.stdin.buffer.read(limit + 1)
    except (OSError, AttributeError):
        # e.g. pytest's captured stdin: not a tty, but not readable either.
        return None
    if len(raw) > limit:
        raise InputError(
            f"stdin exceeds the total budget ({_human_bytes(limit)}); "
            "narrow the input or raise --max-bytes"
        )
    if not raw:
        return None
    return raw.decode("utf-8", errors="replace")
