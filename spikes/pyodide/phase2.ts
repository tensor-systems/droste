// Phase 2 — step 1: prove RunnerEnvironment.execute() RUNS under Pyodide (not
// just imports) after the signal-timeout port.   Run:  ./run.sh phase2.ts
import { loadPyodide } from "npm:pyodide";

const zipPath = Deno.args[0];
const pyodide = await loadPyodide();
const zip = await Deno.readFile(zipPath);
pyodide.unpackArchive(zip, "zip", { extractDir: "/app" });
await pyodide.runPythonAsync(`import sys; sys.path.insert(0, "/app")`);

const result = await pyodide.runPythonAsync(`
import json
from rlm_runner.runner import RunnerEnvironment

# Stub the host-bridged subcall client (real one proxies llm_query to ModelRelay).
class StubSubcalls:
    def llm_query(self, prompt, *a, **k):
        return f"[stub answer to: {prompt}]"
    def llm_batch(self, prompts, *a, **k):
        return [f"[stub:{p}]" for p in prompts]

env = RunnerEnvironment(
    context={}, data_source=None, subcalls=StubSubcalls(),
    max_output_chars=10000, exec_timeout_ms=5000,   # timeout set; no SIGALRM in WASM
)

# Model-style generated code: print, call a sub-LLM, populate answer.
code = """
print('running in pyodide REPL')
answer['content'] = llm_query('summarize the Winona thread')
answer['ready'] = True
"""
res = env.execute(code)
json.dumps({
    "stdout": res.stdout.strip(),
    "timed_out": res.timed_out,
    "exit_code": res.exit_code,
    "answer": env.globals()["answer"],
})
`);

console.log("[RunnerEnvironment.execute() in Pyodide]:");
console.log(JSON.stringify(JSON.parse(result), null, 2));
