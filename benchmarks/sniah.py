"""Deterministic, zero-I/O RULER S-NIAH task generation.

Methodology provenance: Hsieh et al. (2024), "RULER: What's the Real Context
Size of Your Long-Context Language Models?" (arXiv:2404.06654), and NVIDIA's
Apache-2.0-licensed RULER implementation at commit
38da79d79519ef87aa46ae804f838e1eab7f86d7, specifically
``scripts/data/synthetic/niah.py`` and ``constants.py``.

This is an independent implementation of the published algorithm. It fetches
and redistributes no RULER corpus or generated examples. The fixed noise
sentence is part of the published methodology. To keep materialization fully
stdlib and offline, Droste uses a small in-repo adjective/noun bank instead of
RULER's ``wonderwords`` files and pins a deterministic word/punctuation/newline
counter instead of accepting an external model tokenizer. Those choices are
recorded in every task and the manifest split.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Literal

from .runner import BenchmarkRunError

RULER_COMMIT = "38da79d79519ef87aa46ae804f838e1eab7f86d7"
GENERATOR_VERSION = "droste-native-generator-v1"
TOKENIZER_VERSION = "wordpunct-newline-v1"
NOISE_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
)
NEEDLE_TEMPLATE = "One of the special magic {needle_type} for {key} is: {value}."
DEFAULT_CONTEXT_LENGTH = 32_768
DEFAULT_TASK_COUNT = 50
DEFAULT_SEED = 42
DEFAULT_NEEDLE_TYPE: Literal["words"] = "words"
RESERVED_OUTPUT_TOKENS = 128
MODEL_TEMPLATE_TOKENS = 256

_TOKEN_RE = re.compile(r"\w+|[^\w\s]|\n", re.UNICODE)

# These original, deliberately compact banks replace RULER's runtime
# ``wonderwords`` dependency. Their sorted Cartesian product has the same
# adjective-noun shape as RULER's word needles.
_ADJECTIVES = tuple(
    sorted(
        {
            "amber",
            "ancient",
            "brisk",
            "calm",
            "candid",
            "clever",
            "cool",
            "crisp",
            "eager",
            "faint",
            "gentle",
            "grand",
            "green",
            "hidden",
            "kind",
            "lively",
            "lucid",
            "mellow",
            "merry",
            "nimble",
            "plain",
            "proud",
            "quiet",
            "rapid",
            "rare",
            "round",
            "silver",
            "steady",
            "subtle",
            "swift",
            "tidy",
            "vivid",
        }
    )
)
_NOUNS = tuple(
    sorted(
        {
            "anchor",
            "badger",
            "beacon",
            "birch",
            "brook",
            "canyon",
            "cedar",
            "comet",
            "coral",
            "crane",
            "dawn",
            "delta",
            "ember",
            "falcon",
            "fern",
            "fjord",
            "grove",
            "harbor",
            "heron",
            "island",
            "juniper",
            "lantern",
            "maple",
            "meadow",
            "otter",
            "pebble",
            "quartz",
            "raven",
            "ridge",
            "sparrow",
            "willow",
            "zephyr",
        }
    )
)
_WORDS = tuple(f"{adjective}-{noun}" for adjective in _ADJECTIVES for noun in _NOUNS)


@dataclass(frozen=True)
class MaterializedSniah:
    tasks_path: Path
    tasks_sha256: str
    task_count: int
    context_count: int


def count_tokens(text: str) -> int:
    """Count pinned word, punctuation, and newline units."""

    return sum(1 for _ in _TOKEN_RE.finditer(text))


def _encode_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise BenchmarkRunError(f"refusing to overwrite materialized file: {path}") from exc
    finally:
        Path(temporary).unlink(missing_ok=True)


def _word(rng: Random) -> str:
    return _WORDS[rng.randrange(len(_WORDS))]


def _prompt_parts(key: str) -> tuple[str, str, str]:
    preamble = (
        "A special magic word is hidden within the following text. "
        "Make sure to memorize it. I will quiz you about the word afterwards."
    )
    query = f"What is the special magic word for {key} mentioned in the provided text?"
    answer_prefix = f"The special magic word for {key} mentioned in the provided text is"
    return preamble, query, answer_prefix


def _context_for(repetitions: int, insertion_index: int, needle: str) -> str:
    sentences = [NOISE_SENTENCE] * repetitions
    sentences.insert(insertion_index, needle)
    return "\n".join(sentences)


def _full_prompt(
    repetitions: int,
    insertion_index: int,
    needle: str,
    preamble: str,
    query: str,
    answer_prefix: str,
) -> tuple[str, str]:
    haystack = _context_for(repetitions, insertion_index, needle)
    context = f"{preamble}\n{haystack}"
    return context, f"{context}\n{query}\n{answer_prefix}"


def _max_repetitions(
    context_length: int,
    needle: str,
    preamble: str,
    query: str,
    answer_prefix: str,
) -> int:
    available = context_length - RESERVED_OUTPUT_TOKENS - MODEL_TEMPLATE_TOKENS
    if available <= 0:
        raise ValueError(
            "context_length is too small for reserved output and model template tokens"
        )

    def fits(repetitions: int) -> bool:
        _, prompt = _full_prompt(repetitions, 0, needle, preamble, query, answer_prefix)
        return count_tokens(prompt) <= available

    if not fits(1):
        raise ValueError("context_length is too small for one noise sentence")
    lower, upper = 1, 2
    while fits(upper):
        lower, upper = upper, upper * 2
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if fits(middle):
            lower = middle
        else:
            upper = middle
    return lower


def generate_task(
    *,
    context_length: int,
    seed: int,
    task_id: str,
    needle_type: Literal["words"] = DEFAULT_NEEDLE_TYPE,
    depth: float | None = None,
) -> dict[str, Any]:
    """Generate one RULER-style noise S-NIAH task from explicit parameters."""

    if needle_type != "words":
        raise ValueError("generator v1 supports only the pinned words configuration")
    if not isinstance(context_length, int) or isinstance(context_length, bool):
        raise TypeError("context_length must be an integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if depth is not None and not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be between 0.0 and 1.0")

    rng = Random(seed)
    key = _word(rng)
    value = _word(rng)
    needle = NEEDLE_TEMPLATE.format(needle_type=needle_type, key=key, value=value)
    preamble, query, answer_prefix = _prompt_parts(key)
    repetitions = _max_repetitions(context_length, needle, preamble, query, answer_prefix)
    insertion_index = (
        rng.randrange(repetitions)
        if depth is None
        else min(int(repetitions * depth), repetitions - 1)
    )
    context, prompt = _full_prompt(
        repetitions, insertion_index, needle, preamble, query, answer_prefix
    )
    input_tokens = count_tokens(prompt)
    total_tokens = input_tokens + RESERVED_OUTPUT_TOKENS + MODEL_TEMPLATE_TOKENS
    if total_tokens > context_length:
        raise AssertionError("generated task exceeded its token budget")

    context_bytes = context.encode("utf-8")
    context_sha256 = hashlib.sha256(context_bytes).hexdigest()
    return {
        "answer_prefix": answer_prefix,
        "context": context,
        "context_length": context_length,
        "context_path": f"contexts/{context_sha256}.txt",
        "context_sha256": context_sha256,
        "generator_version": GENERATOR_VERSION,
        "haystack_repetitions": repetitions,
        "haystack_type": "noise",
        "id": task_id,
        "input_tokens": input_tokens,
        "model_template_tokens": MODEL_TEMPLATE_TOKENS,
        "needle": needle,
        "needle_depth": insertion_index / repetitions,
        "needle_index": insertion_index,
        "needle_key": key,
        "needle_type": needle_type,
        "needle_value": value,
        "position": insertion_index / repetitions,
        "query": query,
        # RULER supplies this answer prefix to elicit the value. Keeping it in
        # the live question makes the exact-match contract unambiguous.
        "question": f"{query}\n{answer_prefix}",
        "reference": value,
        "reserved_output_tokens": RESERVED_OUTPUT_TOKENS,
        "ruler_commit": RULER_COMMIT,
        "seed": seed,
        "tokenizer": TOKENIZER_VERSION,
        "total_tokens": total_tokens,
    }


def generate_tasks(
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    task_count: int = DEFAULT_TASK_COUNT,
    seed: int = DEFAULT_SEED,
    needle_type: Literal["words"] = DEFAULT_NEEDLE_TYPE,
) -> list[dict[str, Any]]:
    """Generate task seeds and records deterministically from one suite seed."""

    if not isinstance(task_count, int) or isinstance(task_count, bool) or task_count < 1:
        raise ValueError("task_count must be a positive integer")
    suite_rng = Random(seed)
    return [
        generate_task(
            context_length=context_length,
            seed=suite_rng.getrandbits(64),
            task_id=f"sniah-{index:03d}",
            needle_type=needle_type,
        )
        for index in range(task_count)
    ]


def materialize_sniah(
    output_dir: Path,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    task_count: int = DEFAULT_TASK_COUNT,
    seed: int = DEFAULT_SEED,
    needle_type: Literal["words"] = DEFAULT_NEEDLE_TYPE,
) -> MaterializedSniah:
    """Materialize a byte-reproducible synthetic S-NIAH task set."""

    tasks = generate_tasks(
        context_length=context_length,
        task_count=task_count,
        seed=seed,
        needle_type=needle_type,
    )
    contexts = {str(task["context_sha256"]): str(task["context"]).encode("utf-8") for task in tasks}
    tasks_bytes = _encode_json(tasks)
    tasks_path = output_dir / "tasks.json"
    targets = [tasks_path, *(output_dir / "contexts" / f"{key}.txt" for key in contexts)]
    existing = next((path for path in targets if path.exists()), None)
    if existing is not None:
        raise BenchmarkRunError(f"refusing to overwrite materialized file: {existing}")
    for context_hash, content in sorted(contexts.items()):
        _write_exclusive(output_dir / "contexts" / f"{context_hash}.txt", content)
    _write_exclusive(tasks_path, tasks_bytes)
    return MaterializedSniah(
        tasks_path=tasks_path,
        tasks_sha256=hashlib.sha256(tasks_bytes).hexdigest(),
        task_count=len(tasks),
        context_count=len(contexts),
    )
