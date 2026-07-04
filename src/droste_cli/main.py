"""droste — ask questions over files and SQLite from the terminal (#28).

The five-minute out-of-box moment: point the engine at your own data with your
own key (BYOK, any OpenAI-compatible endpoint) and ask.

    droste ask report.txt logs.txt "what changed between these?" --model gpt-5.2-mini
    droste ask --db app.db "which customers churned last month?" --model gpt-5.2-mini

Files are materialized as the sandbox's `context` variable
(``{"files": [{path, name, text, size}, ...]}``); the model sees sizes and
sizes — not the raw bytes — and pulls data in via code. SQLite goes through
the engine's local-mode SQL data source (read-only policy as a guardrail, not
a boundary; OS permissions are the boundary).

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

from rlm_core import (
    DEFAULT_MAX_CALLS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_OUTPUT_CHARS,
    OpenAICompatClient,
    OpenAICompatSubcallClient,
    RLMConfig,
    create_execution_context,
    run_rlm,
)
from rlm_core.clients.openai_compat import DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS
from rlm_core.registry import DataSourceRegistry



class CLIError(Exception):
    """User-facing CLI error: message to stderr, exit code 2."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="droste",
        description="Ask questions over files and SQLite with a recursive language model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="ask a question over files (and/or --db)")
    ask.add_argument("files", nargs="*", help="files to load into the context")
    ask.add_argument("question", help="the question to answer")
    ask.add_argument("--db", metavar="PATH", help="SQLite database to expose as the `db` source")
    ask.add_argument(
        "--model",
        default=os.environ.get("DROSTE_MODEL", ""),
        help="root model id (env: DROSTE_MODEL)",
    )
    ask.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint base URL (env: OPENAI_BASE_URL; default: api.openai.com/v1)",
    )
    ask.add_argument(
        "--api-key",
        default=None,
        help="API key (env: OPENAI_API_KEY; the flag overrides the env)",
    )
    ask.add_argument("--subcall-model", default="", help="model for llm_query subcalls (default: --model)")
    ask.add_argument(
        "--subcall-max-output-tokens",
        type=int,
        default=DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS,
        help=f"per-subcall output token bound (default: {DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS}; 0 disables)",
    )
    ask.add_argument("--reasoning-effort", default="", help="reasoning effort passed through to the endpoint")
    ask.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"max refinement iterations (default: {DEFAULT_MAX_ITERATIONS})",
    )
    ask.add_argument(
        "--max-subcalls",
        type=int,
        default=DEFAULT_MAX_CALLS,
        help=f"max total llm_query subcalls (default: {DEFAULT_MAX_CALLS})",
    )
    ask.add_argument("--json", action="store_true", help="print a JSON result object for scripting")
    ask.add_argument("--verbose", action="store_true", help="stream progress and loop traces to stderr")
    return parser


def build_context(paths: list[str]) -> dict[str, Any] | None:
    """Materialize files as the sandbox `context` variable.

    Same shape as the bench harness / ModelRelay attachments
    ({"files": [{path, name, text, size}]}), so the runner environment's
    name + size prompt description (0.5.x prompt work) applies as-is.
    """
    if not paths:
        return None
    files: list[dict[str, Any]] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError as exc:
            raise CLIError(f"cannot read {path}: {exc}") from exc
        files.append(
            {
                "path": path,
                "name": os.path.basename(path),
                "text": text,
                "size": len(text),
            }
        )
    return {"files": files}


def _load_sql_source(db_path: str) -> Any:
    """Build the local-mode read-only SQLite data source (rlm-core#29).

    Exposes the database as the `db` source: SELECT-only, policy-gated, opened
    mode=ro. The policy is a guardrail, not a security boundary — OS file
    permissions are the real control (see the source's threat-model docstring).
    """
    if not os.path.isfile(db_path):
        raise CLIError(f"database not found (or not a file): {db_path}")
    from rlm_core.sources.sql_local import local_sql_source_factory

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
    if not args.files and not args.db:
        raise CLIError("nothing to ask over: pass at least one file, or --db PATH")
    # Validate numeric budgets up front so bad values surface as clean usage
    # errors (exit 2) instead of crashing mid-run as exit 1.
    if args.subcall_max_output_tokens < 0:
        raise CLIError("--subcall-max-output-tokens must be >= 0 (0 disables the cap)")
    if args.max_iterations < 1:
        raise CLIError("--max-iterations must be >= 1")
    if args.max_subcalls < 0:
        raise CLIError("--max-subcalls must be >= 0")

    context_payload = build_context(args.files)
    registry = None
    if args.db:
        source = _load_sql_source(args.db)
        registry = DataSourceRegistry([source], default_source_name=source.name())

    exec_context = create_execution_context(
        max_calls=args.max_subcalls,
        max_iterations=args.max_iterations,
        max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
        # Progress lines only stream with --verbose; the default emitter would
        # print JSON progress events to stderr unconditionally.
        on_progress=_print_progress if args.verbose else (lambda status: None),
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
    from rlm_runner.runner import RunnerEnvironment

    environment = RunnerEnvironment(
        context=context_payload,
        registry=registry,
        subcalls=subcalls,
        max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
        exec_timeout_ms=0,
    )

    config = RLMConfig(
        max_iterations=args.max_iterations,
        max_calls=args.max_subcalls,
        root_model=args.model,
        verbose=args.verbose,
    )

    result = run_rlm(
        args.question,
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
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "ask":
            return run_ask(args)
        parser.error(f"unknown command: {args.command}")
        return 2
    except CLIError as exc:
        print(f"droste: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("droste: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
