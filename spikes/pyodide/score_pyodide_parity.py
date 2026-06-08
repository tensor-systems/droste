"""Score Pyodide RLM answers with the NATIVE semantic_extraction scoring, and
compare to the native baseline. Run from cozybot/tools/rcl-rlm:

    uv run python <this> /tmp/pyodide_sem_results.json
"""
import json
import sys

sys.path.insert(0, "benchmarks")
from semantic_extraction import evaluate_semantic_result, get_semantic_queries  # noqa: E402

# Native baseline captured this session (spikes/pyodide/baseline-native-semantic.txt).
BASELINE = {
    "sem-stressed": (False, 0.22), "sem-excited": (True, 0.51), "sem-plans": (False, 0.17),
    "sem-gratitude": (True, 0.30), "sem-frustration": (True, 0.30), "sem-asking-help": (True, 0.30),
    "sem-apologizing": (True, 0.30), "sem-sarcasm": (True, 0.38),
}

results = {r["id"]: r for r in json.load(open(sys.argv[1]))}
queries = {q.id: q for q in get_semantic_queries()}

print(f"{'query':18} {'NATIVE':>12}   {'PYODIDE':>20}")
print("-" * 56)
passed = 0
for qid, q in queries.items():
    r = results.get(qid, {"answer": "", "subcalls": 0, "tokens": 0})
    ev = evaluate_semantic_result(r["answer"], q, r["subcalls"], r["tokens"])
    passed += int(ev.passed)
    bp, bs = BASELINE.get(qid, (None, 0.0))
    print(f"{qid:18} {('PASS' if bp else 'FAIL')} {bs:.2f}      "
          f"{('PASS' if ev.passed else 'FAIL')} {ev.score:.2f}  sub={ev.subcalls} tok={r.get('tokens',0)}")

n = len(queries)
print("-" * 56)
print(f"PYODIDE: {passed}/{n} ({passed / n * 100:.0f}%)   |   NATIVE BASELINE: 6/8 (75%)")
