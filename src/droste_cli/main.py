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
The model's generated Python runs **in-process with your privileges** — the
CLI is a power tool with the trust level of running a script the model wrote
(the SQL source is read-only-gated; the Python is not sandboxed). Files are
materialized as the REPL's `context` variable
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

from .credentials import Credentials, CredentialsError, load_credentials
from .inputs import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_TOTAL_BYTES,
    InputError,
    classify,
    load_inputs,
    read_piped_stdin,
)

NO_CREDENTIALS_MESSAGE = (
    "no credentials. run `droste login` to choose how to run (free credits, "
    "or your own key); scripts can pass --api-key or set OPENAI_API_KEY / "
    "ANTHROPIC_API_KEY"
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
        epilog=(
            "setup: droste login chooses how to run (ModelRelay free credits, "
            "or your own key) and stores it; droste whoami / droste logout. "
            "--api-key / --base-url override for a single run; scripts can "
            "set OPENAI_API_KEY / ANTHROPIC_API_KEY."
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
    parser.add_argument(
        "--subcall-model", default="", help="model for llm_query subcalls (default: --model)"
    )
    parser.add_argument(
        "--subcall-max-output-tokens",
        type=int,
        default=DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS,
        help=f"per-subcall output token bound (default: {DEFAULT_SUBCALL_MAX_OUTPUT_TOKENS}; 0 disables)",
    )
    parser.add_argument(
        "--reasoning-effort", default="", help="reasoning effort passed through to the endpoint"
    )
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
    parser.add_argument(
        "--json", action="store_true", help="print a JSON result object for scripting"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="stream progress and the model's code as it is written to stderr (watch it think)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="full loop trace on stderr: generated code, execution output, LLM responses (implies --verbose)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="suppress the loaded-inputs report line"
    )
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


class _StreamEcho:
    """stderr echo for streamed root-model output (--verbose).

    Fragments arrive mid-line; progress lines must start on a fresh line, so
    the echo tracks whether the stream left the cursor mid-line and the
    progress printer consults it.
    """

    def __init__(self) -> None:
        self.mid_line = False

    def __call__(self, text: str) -> None:
        sys.stderr.write(text)
        sys.stderr.flush()
        self.mid_line = not text.endswith("\n")

    def progress(self, status: str) -> None:
        if self.mid_line:
            sys.stderr.write("\n")
            self.mid_line = False
        print(f"droste: {status}", file=sys.stderr, flush=True)


def select_provider(args: argparse.Namespace) -> str:
    """Fact-based provider detection — no guessing, documented order:

    1. An explicit endpoint (``--base-url`` or ``OPENAI_BASE_URL``) always
       means the OpenAI-compatible client: the user chose where to go.
    2. ``--api-key sk-ant-…`` is an Anthropic key (their published prefix).
    3. Any other explicit ``--api-key`` is OpenAI-compatible.
    4. A ``claude-*`` model with ``ANTHROPIC_API_KEY`` set goes to Anthropic
       even when ``OPENAI_API_KEY`` is also set (the model names the vendor).
    5. Otherwise ``OPENAI_API_KEY`` wins, then ``ANTHROPIC_API_KEY``.
    """
    from droste.clients.anthropic import ANTHROPIC_KEY_PREFIX

    if args.base_url or os.environ.get("OPENAI_BASE_URL"):
        return "openai"
    if args.api_key is not None:
        return "anthropic" if args.api_key.startswith(ANTHROPIC_KEY_PREFIX) else "openai"
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    has_anthropic = anthropic_key.startswith(ANTHROPIC_KEY_PREFIX)
    if str(args.model).startswith("claude") and has_anthropic:
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if has_anthropic:
        return "anthropic"
    return "openai"


def resolve_run_target(args: argparse.Namespace) -> tuple[str, Credentials | None]:
    """Credential resolution (droste#55):

    1. Per-run flags (``--base-url`` / ``--api-key``) always win: an explicit
       flag is a statement about THIS run.
    2. Stored credentials — set up once via `droste login` (ModelRelay with
       free credits, or your own key). Choosing how droste runs is a
       deliberate setup step, never a side effect of exported env vars.
    3. No credentials on an interactive terminal: run the same chooser
       in-line, then continue the run.
    4. Non-interactive (scripts/CI): env keys (``OPENAI_BASE_URL`` /
       ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``) as a fallback, else a
       terse error pointing at `droste login`.
    """
    if args.base_url is not None or args.api_key is not None:
        return select_provider(args), None
    try:
        creds = load_credentials()
    except CredentialsError as exc:
        raise CLIError(str(exc)) from exc
    if creds is None and sys.stdin.isatty() and sys.stderr.isatty():
        from . import auth

        try:
            creds = auth.run_interactive_setup()
        except auth.AuthError as exc:
            raise CLIError(str(exc)) from exc
    if creds is not None:
        if creds.provider == "byok":
            # Stored credentials are authoritative: pin the endpoint too, so
            # ambient OPENAI_BASE_URL can never redirect a stored key.
            from droste.clients.anthropic import ANTHROPIC_KEY_PREFIX
            from droste.clients.openai_compat import DEFAULT_BASE_URL

            args.api_key = creds.api_key
            if not args.model and creds.default_model:
                args.model = creds.default_model
            if creds.api_key.startswith(ANTHROPIC_KEY_PREFIX):
                return "anthropic", None
            args.base_url = creds.base_url or DEFAULT_BASE_URL
            return "openai", None
        return "modelrelay", creds
    # ANY set env key counts — including a malformed one. A bad
    # ANTHROPIC_API_KEY must fail loudly on the BYOK path, not be masked
    # by the no-credentials error.
    env_byok = (
        bool(os.environ.get("OPENAI_BASE_URL"))
        or bool(os.environ.get("OPENAI_API_KEY"))
        or bool(os.environ.get("ANTHROPIC_API_KEY"))
    )
    if env_byok:
        return select_provider(args), None
    raise CLIError(NO_CREDENTIALS_MESSAGE)


def _print_progress(status: str) -> None:
    print(f"droste: {status}", file=sys.stderr, flush=True)


def run_ask(args: argparse.Namespace) -> int:
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
        # Implicit stdin is consumed only when nothing else provides data —
        # `droste "q" file.txt` in a script must not hang on (or slurp) an
        # unrelated inherited pipe; unix tools ignore stdin when file args
        # are given. Explicit `-` always reads.
        wants_stdin = classified.stdin_explicit or (
            classified.reads_stdin
            and not (classified.files or classified.dirs or classified.dbs or args.db)
        )
        stdin_text = (
            read_piped_stdin(limit=args.max_bytes, explicit=classified.stdin_explicit)
            if wants_stdin
            else None
        )
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

    provider, creds = resolve_run_target(args)

    # Logged-in runs default the model from the stored credentials; BYOK
    # still requires an explicit model (we won't guess someone's bill).
    if provider == "modelrelay" and not args.model and creds is not None:
        args.model = creds.default_model
    if not args.model:
        raise CLIError("--model is required (or set DROSTE_MODEL)")

    # The default endpoint (api.openai.com) always requires a key, so a
    # keyless run there can only die mid-flight with a raw 401 — catch it
    # before any client is built. A custom --base-url (Ollama, vLLM, ...)
    # may legitimately be keyless. Input errors above stay most-specific.
    if provider == "openai":
        from urllib.parse import urlparse

        from droste.clients.openai_compat import resolve_api_key, resolve_base_url

        resolved_host = (urlparse(resolve_base_url(args.base_url)).hostname or "").lower()
        if not resolve_api_key(args.api_key) and resolved_host == "api.openai.com":
            raise CLIError(
                "no API key: set OPENAI_API_KEY or ANTHROPIC_API_KEY, or pass "
                "--api-key (custom --base-url endpoints may run keyless)"
            )
    elif args.reasoning_effort:
        raise CLIError(
            "--reasoning-effort is not an Anthropic API parameter; Claude "
            "thinking is controlled via the API's thinking params"
        )

    # --verbose/--trace stream the root model's output (the generated code)
    # to stderr as it is written, between one-line progress markers — the
    # "watch it think" view. --trace additionally dumps the full loop.
    echo = _StreamEcho() if (args.verbose or args.trace) else None

    exec_context = create_execution_context(
        max_calls=args.max_subcalls,
        max_iterations=args.max_iterations,
        max_output_chars=DEFAULT_MAX_OUTPUT_CHARS,
        # Progress lines only stream with --verbose/--trace; the default
        # emitter would print JSON progress events to stderr unconditionally.
        on_progress=echo.progress if echo else (lambda status: None),
        # The CLI renders "watch it think" via the --verbose/--trace echo above,
        # not the structured NDJSON events (those feed programmatic hosts through
        # the relay's stderr forwarder). Swallow them here so a plain `droste ask`
        # never dumps raw code/iteration/output JSON to stderr.
        on_event=lambda event: None,
    )

    if provider == "modelrelay":
        from droste.clients.modelrelay import ModelRelayClient, ModelRelaySubcallClient

        assert creds is not None
        root = ModelRelayClient(
            model=args.model,
            base_url=creds.base_url,
            api_key=creds.api_key,
            reasoning_effort=args.reasoning_effort,
            on_delta=echo,
        )
        # Subcalls default to reasoning_effort="none" + a bounded output
        # budget (ModelRelay's own server-side subcall defaults);
        # --reasoning-effort overrides both root and subcalls.
        subcalls = ModelRelaySubcallClient(
            model=args.subcall_model or args.model,
            context=exec_context,
            base_url=creds.base_url,
            api_key=creds.api_key,
            max_output_tokens=args.subcall_max_output_tokens,
            reasoning_effort=args.reasoning_effort or "none",
        )
    elif provider == "anthropic":
        from droste import AnthropicClient, AnthropicSubcallClient

        root = AnthropicClient(
            model=args.model,
            api_key=args.api_key,
            on_delta=echo,
        )
        # The Messages API requires a positive max_tokens; 0 (the compat
        # client's "unbounded" opt-out) falls back to the default bound.
        subcalls = AnthropicSubcallClient(
            model=args.subcall_model or args.model,
            context=exec_context,
            api_key=args.api_key,
            max_output_tokens=args.subcall_max_output_tokens or 2048,
        )
    else:
        root = OpenAICompatClient(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            on_delta=echo,
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
    # name + size prompt description for free.
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
        # Usable answer, honest provenance.
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


def _run_auth_command(argv: list[str]) -> int:
    from . import auth

    command, rest = argv[0], argv[1:]
    if command == "login":
        login_parser = argparse.ArgumentParser(prog="droste login")
        login_parser.add_argument(
            "--base-url",
            default=None,
            help=f"ModelRelay API base URL (default: {auth.DEFAULT_BASE_URL})",
        )
        login_args = login_parser.parse_args(rest)
        return auth.run_login(login_args.base_url)
    if rest:
        raise CLIError(f"droste {command} takes no arguments")
    if command == "logout":
        return auth.run_logout()
    return auth.run_whoami()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("login", "logout", "whoami"):
        from . import auth

        try:
            return _run_auth_command(argv)
        except (CLIError, auth.AuthError) as exc:
            print(f"droste: error: {exc}", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("droste: interrupted", file=sys.stderr)
            return 130
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
