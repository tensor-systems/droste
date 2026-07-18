"""One-off BrowseComp-Plus semantic judge pass over published predictions."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from droste import ModelRelayClient
from droste_cli.credentials import load_credentials

from .live import _fetch_pricing

JUDGE_PROMPT = (
    """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]."""
    + " \n\n"
    + """[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
"""
)

_CORRECT_RE = re.compile(r"correct:\s*(yes|no)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"confidence:\s*(\d{1,3})\s*%?", re.IGNORECASE)


def parse_verdict(response: str) -> tuple[bool, int]:
    """Parse the official fields, defaulting an unparseable verdict to no."""

    plain = response.replace("**", "").replace("__", "")
    correct_matches = list(_CORRECT_RE.finditer(plain))
    correct_match = correct_matches[-1] if correct_matches else None
    confidence_match = _CONFIDENCE_RE.search(plain)
    correct = bool(correct_match and correct_match.group(1).lower() == "yes")
    confidence = int(confidence_match.group(1)) if confidence_match else 100
    return correct, min(100, confidence)


@dataclass(frozen=True)
class Judged:
    task_id: str
    response: str
    input_tokens: int
    output_tokens: int


def _judge(
    *,
    task_id: str,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    max_output_tokens: int,
) -> Judged:
    client = ModelRelayClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_output_tokens=max_output_tokens,
        reasoning_effort="none",
        temperature=0,
        timeout=120,
    )
    response, usage = client.responses_create(
        [{"role": "user", "content": prompt}],
        model=model,
        return_usage=True,
    )
    return Judged(task_id, response, usage.prompt_tokens, usage.completion_tokens)


def _cost_microusd(
    input_tokens: int,
    output_tokens: int,
    *,
    input_cents: int,
    output_cents: int,
    fee_percent: int,
) -> int:
    numerator = input_tokens * input_cents + output_tokens * output_cents
    base = Decimal(numerator) / Decimal(100)
    total = base * Decimal(100 + fee_percent) / Decimal(100)
    return int(total.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = sorted(args.artifacts.glob("*--droste-terra-luna-browsecomp-plus-1k--*.json"))
    if len(artifacts) != 150:
        raise RuntimeError(f"expected 150 Droste artifacts, found {len(artifacts)}")

    tasks_data = json.loads(args.tasks.read_text())
    tasks = {str(task["id"]): task for task in tasks_data}
    records = {
        str(item["task_id"]): item for item in map(lambda p: json.loads(p.read_text()), artifacts)
    }
    if set(records) != set(tasks):
        raise RuntimeError("artifact task IDs do not match the 150 materialized tasks")
    for task_id, record in records.items():
        if record.get("reference") != tasks[task_id].get("reference"):
            raise RuntimeError(f"reference mismatch for task {task_id}")

    credentials = load_credentials()
    if credentials is None or credentials.provider != "modelrelay":
        raise RuntimeError("ModelRelay credentials required; run `droste login`")
    pricing = _fetch_pricing({args.model})
    price = pricing.prices[args.model]
    cost_args = {
        "input_cents": price.input_cost_per_million_cents,
        "output_cents": price.output_cost_per_million_cents,
        "fee_percent": pricing.platform_fee_percent,
    }

    output: dict[str, Any]
    if args.output.exists():
        output = json.loads(args.output.read_text())
        if output.get("judge_model") != args.model:
            raise RuntimeError("existing output uses a different judge model")
    else:
        output = {
            "judge_model": args.model,
            "max_cost_microusd": args.max_cost_microusd,
            "judge_cost_microusd": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tasks": 150,
            "results": {},
        }

    results = output["results"]
    prompts: dict[str, str] = {}
    for task_id, record in records.items():
        if task_id in results:
            continue
        prediction = record.get("prediction")
        if prediction is None:
            results[task_id] = {
                "correct": False,
                "confidence": 100,
                "judge_response": "No prediction: HTTP 504 timeout; counted incorrect without an LLM call.",
            }
            continue
        task = tasks[task_id]
        prompts[task_id] = JUDGE_PROMPT.format(
            question=task["question"],
            response=prediction,
            correct_answer=task["reference"],
        )

    pending = sorted(prompts, key=lambda value: (len(value), value))
    for offset in range(0, len(pending), args.concurrency):
        batch_ids = pending[offset : offset + args.concurrency]
        # UTF-8 bytes are a conservative token ceiling for this English prompt.
        reserved_cost = sum(
            _cost_microusd(len(prompts[task_id].encode()), args.max_output_tokens, **cost_args)
            for task_id in batch_ids
        )
        if output["judge_cost_microusd"] + reserved_cost > args.max_cost_microusd:
            _write_output(args.output, output)
            raise RuntimeError("$5 judge cost ceiling would be exceeded before dispatch")

        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(
                    _judge,
                    task_id=task_id,
                    prompt=prompts[task_id],
                    model=args.model,
                    api_key=credentials.api_key,
                    base_url=credentials.base_url,
                    max_output_tokens=args.max_output_tokens,
                ): task_id
                for task_id in batch_ids
            }
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    judged = future.result()
                except Exception as exc:
                    failures.append(f"{task_id}: {exc}")
                    continue
                correct, confidence = parse_verdict(judged.response)
                results[task_id] = {
                    "correct": correct,
                    "confidence": confidence,
                    "judge_response": judged.response,
                }
                output["input_tokens"] += judged.input_tokens
                output["output_tokens"] += judged.output_tokens
                output["judge_cost_microusd"] = _cost_microusd(
                    output["input_tokens"], output["output_tokens"], **cost_args
                )
                print(f"{len(results)}/150 task={task_id} correct={correct}", flush=True)
        _write_output(args.output, output)
        if failures:
            raise RuntimeError("judge calls failed; rerun to resume:\n" + "\n".join(failures))

    correct_count = sum(bool(result["correct"]) for result in results.values())
    output.update(
        {
            "automatic_incorrect": sum(
                record.get("prediction") is None for record in records.values()
            ),
            "judged_tasks": sum(
                record.get("prediction") is not None for record in records.values()
            ),
            "correct_count": correct_count,
            "accuracy": correct_count / 150,
            "exact_match_correct": sum(record.get("score") == 1.0 for record in records.values()),
            "exact_match_accuracy": sum(record.get("score") == 1.0 for record in records.values())
            / 150,
        }
    )
    _write_output(args.output, output)
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=root / "benchmarks/results/browsecomp-plus-1k-2026-07-18/artifacts",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=root / "benchmarks/.data/browsecomp-plus-1k-seed-166001-v1/tasks.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "benchmarks/results/browsecomp-plus-1k-2026-07-18/judge-results.json",
    )
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-cost-microusd", type=int, default=5_000_000)
    args = parser.parse_args()
    output = run(args)
    print(json.dumps({key: value for key, value in output.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
