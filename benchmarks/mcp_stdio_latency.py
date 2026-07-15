"""Measure the #5 stdio spike against the equivalent in-process read path."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from pathlib import Path
from time import perf_counter

from droste import CapabilityBroker, ConfiguredSource, ProviderCatalog, ProviderRegistry
from droste.sources.filesystem_text import filesystem_text_provider
from droste.sources.mcp_stdio import open_mcp_stdio_source


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be positive")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "guide.md").write_text("latency comparison fact\n", encoding="utf-8")
        mcp_source = ConfiguredSource(
            "remote_docs",
            "reference_filesystem",
            {
                "command": os.path.realpath(args.node),
                "args": [str(Path(args.server).resolve()), str(root)],
                "env": {},
                "cwd": str(root),
                "allowed_executables": [os.path.realpath(args.node)],
                "allowed_tools": ["read_text_file"],
                "bindings": {"read_text_file": "read_text_file"},
                "effects": {"read_text_file": "read"},
                "budget_classes": {"read_text_file": "data.read"},
                "source_description": "Read-only reference documents.",
            },
        )
        started = perf_counter()
        mcp = open_mcp_stdio_source(mcp_source)
        cold_ms = (perf_counter() - started) * 1000
        local = (
            ProviderCatalog((filesystem_text_provider(),))
            .bind((ConfiguredSource("local_docs", "filesystem_text", {"root": str(root)}),))
            .sources[0]
        )
        registry = ProviderRegistry((mcp, local))
        broker = CapabilityBroker(registry.capability_registrations())
        generated = registry.broker_globals(broker)
        remote_ms: list[float] = []
        local_ms: list[float] = []
        try:
            for _ in range(args.iterations):
                started = perf_counter()
                remote = generated["remote_docs"].read_text_file(path="guide.md")
                remote_ms.append((perf_counter() - started) * 1000)
                started = perf_counter()
                local_result = generated["local_docs"].read(path="guide.md")
                local_ms.append((perf_counter() - started) * 1000)
                assert remote["content"] == local_result["text"]
        finally:
            registry.close()
        print(
            json.dumps(
                {
                    "iterations": args.iterations,
                    "cold_bind_ms": round(cold_ms, 3),
                    "mcp_warm_median_ms": round(statistics.median(remote_ms), 3),
                    "mcp_warm_p95_ms": round(_percentile(remote_ms, 0.95), 3),
                    "in_process_median_ms": round(statistics.median(local_ms), 3),
                    "in_process_p95_ms": round(_percentile(local_ms, 0.95), 3),
                    "median_overhead_ms": round(
                        statistics.median(remote_ms) - statistics.median(local_ms), 3
                    ),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
