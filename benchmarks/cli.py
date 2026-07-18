from __future__ import annotations

import argparse
import json
from pathlib import Path

from .live import run_modelrelay_suite
from .longbench_codeqa import materialize_longbench_codeqa
from .models import load_manifest
from .oolong import materialize_oolong
from .report import aggregate, load_artifacts, render_markdown, summary_dict
from .runner import run_fixture_suite
from .sniah import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEED,
    DEFAULT_TASK_COUNT,
    materialize_sniah,
)

_ROOT = Path(__file__).resolve().parent
_SMOKE_MANIFEST = _ROOT / "manifests" / "smoke-v1.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a versioned suite manifest")
    validate.add_argument("manifest", type=Path)

    smoke = commands.add_parser("smoke", help="run the deterministic zero-cost smoke suite")
    smoke.add_argument("--output", type=Path, required=True)

    materialize = commands.add_parser(
        "materialize-oolong", help="materialize the pinned public OOLONG 131K task slice"
    )
    materialize.add_argument("--output", type=Path, required=True)

    materialize_sniah_parser = commands.add_parser(
        "materialize-sniah", help="materialize deterministic noise S-NIAH tasks"
    )
    materialize_sniah_parser.add_argument("--output", type=Path, required=True)
    materialize_sniah_parser.add_argument(
        "--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH
    )
    materialize_sniah_parser.add_argument("--task-count", type=int, default=DEFAULT_TASK_COUNT)
    materialize_sniah_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    materialize_codeqa = commands.add_parser(
        "materialize-longbench-codeqa",
        help="materialize the pinned LongBench-v2 CodeQA 20-task cost-bounded subsample",
    )
    materialize_codeqa.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run", help="run selected live ModelRelay benchmark arms")
    run.add_argument("manifest", type=Path)
    run.add_argument("--benchmark", required=True)
    run.add_argument("--arm", action="append", required=True, dest="arms")
    run.add_argument("--task-id", action="append", dest="task_ids")
    run.add_argument("--limit", type=int, default=0)
    run.add_argument(
        "--max-cost-microusd",
        type=int,
        help="stop cleanly before another arm when cumulative artifact cost reaches this cap",
    )
    run.add_argument("--output", type=Path, required=True)

    report = commands.add_parser("report", help="render a report from run artifacts")
    report.add_argument("manifest", type=Path)
    report.add_argument("artifacts", type=Path)
    report.add_argument("--task-id", action="append", dest="task_ids")
    report.add_argument("--json", dest="json_output", type=Path)
    report.add_argument("--markdown", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        manifest = load_manifest(args.manifest)
        print(f"valid: {manifest.suite_id} {manifest.suite_version} ({manifest.sha256})")
        return 0
    if args.command == "smoke":
        manifest = load_manifest(_SMOKE_MANIFEST)
        artifacts = run_fixture_suite(manifest, args.output)
        print(render_markdown(manifest, aggregate(artifacts)), end="")
        return 0
    if args.command == "materialize-oolong":
        result = materialize_oolong(args.output)
        print(
            f"materialized {result.task_count} tasks and {result.context_count} contexts; "
            f"tasks SHA-256: {result.tasks_sha256}"
        )
        return 0
    if args.command == "materialize-sniah":
        result = materialize_sniah(
            args.output,
            context_length=args.context_length,
            task_count=args.task_count,
            seed=args.seed,
        )
        print(
            f"materialized {result.task_count} tasks and {result.context_count} contexts; "
            f"tasks SHA-256: {result.tasks_sha256}"
        )
        return 0
    if args.command == "materialize-longbench-codeqa":
        result = materialize_longbench_codeqa(args.output)
        print(
            f"materialized {result.task_count} tasks and {result.context_count} contexts; "
            f"tasks SHA-256: {result.tasks_sha256}"
        )
        return 0
    if args.command == "run":
        manifest = load_manifest(args.manifest)
        artifacts = run_modelrelay_suite(
            manifest,
            args.output,
            benchmark_id=args.benchmark,
            arm_ids=set(args.arms),
            task_ids=args.task_ids,
            limit=args.limit,
            max_cost_microusd=args.max_cost_microusd,
            progress=lambda message: print(message, flush=True),
        )
        print(f"wrote {len(artifacts)} immutable run artifacts")
        return 0
    manifest = load_manifest(args.manifest)
    rows = aggregate(load_artifacts(args.artifacts, manifest, task_ids=args.task_ids))
    markdown = render_markdown(manifest, rows)
    if args.markdown:
        args.markdown.write_text(markdown)
    else:
        print(markdown, end="")
    if args.json_output:
        args.json_output.write_text(
            json.dumps(
                summary_dict(manifest, rows),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
    return 0
