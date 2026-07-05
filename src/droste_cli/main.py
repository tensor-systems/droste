"""droste — ask questions over files, folders, and SQLite from the terminal.

The contract (#44): *args that exist are data, the one that doesn't is the
question, no args means here, pipes are data too, and it always tells you
what it read.*

    droste "what changed between these?" report.txt logs.txt
    droste "which customers churned last month?" app.db
    droste "how does auth work here?" ./docs
    cd ~/notes && droste "what did I decide about pricing?"
    tail -5000 app.log | droste "why did it crash?"

`ask` survives as an alias (``droste ask …``) for scripts and muscle memory.
Files are materialized as the sandbox's `context` variable
(``{"files": [{path, name, text, size}, ...]}``); the model sees names and
sizes — not the raw bytes — and pulls data in via code. SQLite files are
recognized by their magic bytes and go through the engine's local-mode SQL
data source (read-only policy as a guardrail, not a boundary; OS permissions
are the boundary).

Pointing --base-url at ModelRelay lights up the platform features (validated
SQL policies, server-enforced subcall cost controls, audit); it is documented,
not required. `droste` is the engine CLI; `mrl` remains the platform CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from droste import (
    DEFAULT_MAX_CALLS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_OUTPUT_CHARS,
    OpenAICompatClient,
    OpenAICompatSubcallClient,
    RLMConfig,
    create_execution_context,
    run_rlm,
)
from droste.clients.openai_compat import DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS
from droste.registry import DataSourceRegistry

from .inputs import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_TOTAL_BYTES,
    InputError,
    classify,
    load_inputs,
    read_piped_stdin,
)


class CLIError(Exception):
    """User-facing CLI error: message to stderr, exit code 2."""


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("droste")
    except Exception:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="droste",
        description=(
            "Ask questions over files, folders, and SQLite with a recursive "
            "language model. Args that exist are data; the one that doesn't "
            "is the question; no data args means the current directory."
        ),
    )
    parser.add_argument("--version", action="version", version=f"droste {_package_version()}")
    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="QUESTION|PATH",
        help="the question (quoted) plus any files, directories, or SQLite databases",
    )
    parser.add_argument("--db", metavar="PATH", help="SQLite database to expose as the `db` source")
    parser.add_argument(
        "--model",
        default=os.environ.get("DROSTE_MODEL", ""),
        help="root model id (env: DROSTE_MODEL)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint base URL (env: OPENAI_BASE_URL; default: api.openai.com/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (env: OPENAI_API_KEY; the flag overrides the env)",
    )
    parser.add_argument("--subcall-model", default="", help="model for llm_query subcalls (default: --model)")
    parser.add_argument(
        "--subcall-max-output-tokens",
        type=int,
        default=DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS,
        help=f"per-subcall output token bound (default: {DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS}; 0 disables)",
    )
    parser.add_argument("--reasoning-effort", default="", help="reasoning effort passed through to the endpoint")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"max refinement iterations (default: {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--max-subcalls",
        type=int,
        default=DEFAULT_MAX_CALLS,
        help=f"max total llm_query subcalls (default: {DEFAULT_MAX_CALLS})",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_TOTAL_BYTES,
        help=f"total input budget in bytes (default: {DEFAULT_MAX_TOTAL_BYTES})",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help=(
            "per-file cap for directory walks in bytes "
            f"(default: {DEFAULT_MAX_FILE_BYTES}; explicit files are exempt)"
        ),
    )
    parser.add_argument("--json", action="store_true", help="print a JSON result object for scripting")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="stream one-line progress to stderr (watch it think)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="full loop trace on stderr: generated code, execution output, LLM responses (implies --verbose)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress the loaded-inputs report line")
    return parser


def _load_sql_source(db_path: str) -> Any:
    """Build the local-mode read-only SQLite data source (droste#29).

    Exposes the database as the `db` source: SELECT-only, policy-gated, opened
    mode=ro. The policy is a guardrail, not a security boundary — OS file
    permissions are the real control (see the source's threat-model docstring).
    """
    if not os.path.isfile(db_path):
        raise CLIError(f"database not found (or not a file): {db_path}")
    from droste.sources.sql_local import local_sql_source_factory

    source = local_sql_source_factory({"name": "db", "sqlite_path": db_path})
    # Fail fast with a clean usage error if the file is not a readable SQLite
    # database, rather than crashing mid-run on the first query.
    try:
        source.get_schema()
    except Exception as exc:
        raise CLIError(f"cannot open {db_path} as a SQLite database: {exc}") from exc
    return source


def _print_progress(status: str) -> None:
    print(f"droste: {status}", file=sys.stderr, flush=True)


def run_ask(args: argparse.Namespace) -> int:
    if not args.model:
        raise CLIError("--model is required (or set DROSTE_MODEL)")
    # Validate numeric budgets up front so bad values surface as clean usage
    # errors (exit 2) instead of crashing mid-run as exit 1.
    if args.subcall_max_output_tokens < 0:
        raise CLIError("--subcall-max-output-tokens must be >= 0 (0 disables the cap)")
    if args.max_iterations < 1:
        raise CLIError("--max-iterations must be >= 1")
    if args.max_subcalls < 0:
        raise CLIError("--max-subcalls must be >= 0")
    if args.max_bytes < 1:
        raise CLIError("--max-bytes must be >= 1")
    if args.max_file_bytes < 1:
        raise CLIError("--max-file-bytes must be >= 1")

    try:
        classified = classify(args.inputs, stdin_is_tty=sys.stdin is None or sys.stdin.isatty())
        stdin_text = read_piped_stdin(limit=args.max_bytes) if classified.reads_stdin else None
        loaded = load_inputs(
            classified,
            db_flag=args.db,
            max_file_bytes=args.max_file_bytes,
            max_total_bytes=args.max_bytes,
            stdin_text=stdin_text,
        )
    except InputError as exc:
        raise CLIError(str(exc)) from exc

    # The trust line: everything the run reads, in one glance, before spend.
    if not args.quiet:
        print(f"droste: {loaded.report}", file=sys.stderr, flush=True)

    registry = None
    if loaded.db_path:
        source = _load_sql_source(loaded.db_path)
        registry = DataSourceRegistry([source], default_source_name=source.name())

    exec_context = create_execution_context(
        max_calls=args.max_subcalls,
        max_iterations=args.max_iterations,
        max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
        # Progress lines only stream with --verbose/--trace; the default
        # emitter would print JSON progress events to stderr unconditionally.
        on_progress=_print_progress if (args.verbose or args.trace) else (lambda status: None),
    )

    root = OpenAICompatClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    subcalls = OpenAICompatSubcallClient(
        model=args.subcall_model or args.model,
        context=exec_context,
        base_url=args.base_url,
        api_key=args.api_key,
        max_output_tokens=args.subcall_max_output_tokens,
        reasoning_effort=args.reasoning_effort,
    )

    # RunnerEnvironment provides the in-process REPL plus the context
    # name + size prompt description (0.5.x prompt work) for free.
    from droste_runner.runner import RunnerEnvironment

    environment = RunnerEnvironment(
        context=loaded.context,
        registry=registry,
        subcalls=subcalls,
        max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
        exec_timeout_ms=0,
    )

    config = RLMConfig(
        max_iterations=args.max_iterations,
        max_calls=args.max_subcalls,
        root_model=args.model,
        # config.verbose is the FULL loop dump (code, outputs, responses) —
        # that's --trace. --verbose alone is the clean one-line stream.
        verbose=args.trace,
    )

    result = run_rlm(
        loaded.question,
        environment=environment,
        root_llm=root,
        subcalls=subcalls,
        config=config,
        context=exec_context,
    )

    if args.json:
        payload: dict[str, Any] = {
            "answer": result.answer,
            "ready": result.ready,
            "extracted": result.extracted,
            "iterations": result.iterations,
            "tokens_used": result.tokens_used,
            "subcalls": result.sub_calls_made,
            "model": args.model,
            "files": len(loaded.context["files"]) if loaded.context else 0,
            "db": loaded.db_path,
            "error": None,
        }
        if result.error:
            payload["error"] = {
                "type": result.error.type,
                "message": result.error.message,
            }
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(result.answer)

    if result.ready:
        return 0
    if result.extracted:
        # Like mrl: usable answer, honest provenance.
        print(
            "droste: note: max iterations reached; answer extracted from partial work (unconfirmed)",
            file=sys.stderr,
        )
        return 0
    if result.error:
        print(f"droste: error: {result.error.type}: {result.error.message}", file=sys.stderr)
    else:
        print("droste: no confirmed answer produced", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `ask` is a compatibility alias: the naked binary is the verb.
    if argv and argv[0] == "ask":
        argv = argv[1:]
    parser = build_parser()
    args = parser.parse_intermixed_args(argv)
    try:
        return run_ask(args)
    except CLIError as exc:
        print(f"droste: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("droste: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
