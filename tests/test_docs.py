from __future__ import annotations

import re
import runpy
import unicodedata
from pathlib import Path
from urllib.parse import unquote

from droste.execution.manifest import SCAFFOLD_MANIFEST_VERSION
from droste.execution.trace import TRACE_ABI_VERSION
from droste.prompts.pack import PROMPT_PACK_SCHEMA_VERSION
from droste.providers import PROVIDER_PROTOCOL_VERSION
from droste_runner.protocol import RUNNER_PROTOCOL_VERSION

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
LINK = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")


def _maintained_markdown() -> tuple[Path, ...]:
    roots = (
        REPO / "README.md",
        REPO / "CONTRIBUTING.md",
        REPO / "UPGRADING.md",
        REPO / "benchmarks" / "README.md",
        REPO / "integrations" / "README.md",
        REPO / "pyodide" / "README.md",
    )
    docs = tuple(path for path in DOCS.rglob("*.md") if "assets" not in path.parts)
    return tuple(sorted((*roots, *docs)))


def _destination(raw: str) -> str:
    value = raw.strip()
    if value.startswith("<") and ">" in value:
        return value[1 : value.index(">")]
    return value.split(maxsplit=1)[0]


def _slug(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"[`*_~]", "", value)
    value = unicodedata.normalize("NFKD", value).casefold()
    value = "".join(character for character in value if character.isalnum() or character in " -_")
    return re.sub(r"[ _]+", "-", value).strip("-")


def _anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    fenced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        match = None if fenced else HEADING.match(line)
        if match is None:
            continue
        base = _slug(match.group(1))
        count = counts.get(base, 0)
        counts[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


def _local_links(path: Path) -> tuple[tuple[Path, str | None], ...]:
    links: list[tuple[Path, str | None]] = []
    for raw in LINK.findall(path.read_text(encoding="utf-8")):
        destination = unquote(_destination(raw))
        if destination.startswith(("http://", "https://", "mailto:")):
            continue
        target_name, separator, anchor = destination.partition("#")
        target = path if not target_name else (path.parent / target_name).resolve()
        links.append((target, anchor if separator else None))
    return tuple(links)


def test_documented_contract_versions_match_runtime_constants() -> None:
    headings = {
        DOCS / "providers.md": f"# Providers (protocol v{PROVIDER_PROTOCOL_VERSION})",
        DOCS / "reference" / "runner.md": f"# Runner protocol v{RUNNER_PROTOCOL_VERSION}",
        DOCS / "reference" / "trace.md": f"# Trace ABI v{TRACE_ABI_VERSION}",
        DOCS
        / "reference"
        / "prompt-packs.md": f"# Prompt packs (schema v{PROMPT_PACK_SCHEMA_VERSION})",
        DOCS / "reference" / "scaffold.md": f"# Scaffold manifest v{SCAFFOLD_MANIFEST_VERSION}",
    }
    for path, expected in headings.items():
        assert path.read_text(encoding="utf-8").splitlines()[0] == expected


def test_maintained_markdown_links_and_anchors_resolve() -> None:
    failures: list[str] = []
    for source in _maintained_markdown():
        for target, anchor in _local_links(source):
            if not target.exists():
                failures.append(f"{source.relative_to(REPO)} -> missing {target}")
                continue
            if anchor and target.suffix.casefold() == ".md" and anchor not in _anchors(target):
                failures.append(
                    f"{source.relative_to(REPO)} -> missing #{anchor} in {target.relative_to(REPO)}"
                )
    assert not failures, "\n".join(failures)


def test_every_public_doc_is_reachable_from_the_index() -> None:
    public = set(DOCS.glob("*.md")) | set((DOCS / "reference").glob("*.md"))
    reachable: set[Path] = set()
    pending = [DOCS / "README.md"]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        for target, _ in _local_links(current):
            if target in public and target not in reachable:
                pending.append(target)
    assert public <= reachable, sorted(str(path.relative_to(REPO)) for path in public - reachable)


def test_embedding_example_imports_without_dispatch() -> None:
    namespace = runpy.run_path(str(REPO / "examples" / "embedding.py"), run_name="docs_test")
    assert callable(namespace["ask"])
