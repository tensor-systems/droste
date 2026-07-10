// llm_batch threading verification harness: pin what the runner's threads/signals-dependent paths
// do under Pyodide (no OS threads, no signals), and prove the host-side fix.
//
// Findings (2026-07-07, Deno 2.9.1 / Pyodide 0.29.4):
//   - HTTPSubcallClient.llm_batch (ThreadPoolExecutor)  -> RuntimeError, CRASHES
//   - HTTPSubcallClient.llm_batch_with_errors (raw Thread) -> RuntimeError, CRASHES
//   - SqlDataSource query timeout (threading.Timer)     -> RuntimeError, CRASHES
//   - signal.setitimer / SIGALRM                        -> not present
//   - sequential subcalls                               -> work (no parallelism)
//   - host-side Promise.all fan-out                     -> works, PARALLELISM KEPT
// Conclusion: batch fan-out AND query timeouts must move to the Deno host. See
// the findings note. The host-side fan-out is built in the A-prime broker.
//
// Run: deno run --allow-read --allow-env --allow-ffi verify_16_threading.ts
import { loadPyodide } from "npm:pyodide@0.29.4";
const py = await loadPyodide({ stdout: () => {}, stderr: () => {} });
const log = (s: string) => console.log(s);
const now = () => performance.now();
const errLine = (e: unknown) =>
  (e instanceof Error ? e.message.split("\n").filter((l) => /Error|thread/.test(l)).slice(-1)[0] : String(e))?.trim();

const SUB = `
import time
def _subcall(i):
    time.sleep(0.1)   # 5 x 0.1s = 0.5s serial, ~0.1s if parallel
    return f"r{i}"
`;

log("== 1. llm_batch: ThreadPoolExecutor + as_completed ==");
try {
  const o = await py.runPythonAsync(`${SUB}
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
res=[""]*5
with ThreadPoolExecutor(max_workers=5) as ex:
    futs={ex.submit(_subcall,i):i for i in range(5)}
    for f in as_completed(futs): res[futs[f]]=f.result()
f"OK {res}"`);
  log("  " + o);
} catch (e) { log("  CRASHES: " + errLine(e)); }

log("\n== 2. llm_batch_with_errors: raw threading.Thread ==");
try {
  const o = await py.runPythonAsync(`${SUB}
import threading
res=[""]*5
def one(i): res[i]=_subcall(i)
ts=[threading.Thread(target=one,args=(i,),daemon=True) for i in range(5)]
[t.start() for t in ts]; [t.join() for t in ts]
f"OK {res}"`);
  log("  " + o);
} catch (e) { log("  CRASHES: " + errLine(e)); }

log("\n== 3. SqlDataSource query timeout: threading.Timer ==");
try {
  const o = await py.runPythonAsync(`
import threading, time
fired=[]
threading.Timer(0.05, lambda: fired.append(1)).start()
time.sleep(0.15)
f"timer fired={bool(fired)}"`);
  log("  " + o);
} catch (e) { log("  CRASHES: " + errLine(e)); }

log("\n== 4. signal.setitimer / SIGALRM (alt timeout path) ==");
const hasSig = await py.runPythonAsync(`import signal; hasattr(signal,"setitimer")`);
log(`  signal.setitimer present: ${hasSig}`);

log("\n== 5. sequential subcalls (correctness baseline) ==");
{
  const t = now();
  const o = await py.runPythonAsync(`${SUB}\n[_subcall(i) for i in range(5)]`);
  log(`  OK ${o}  wall=${Math.round(now() - t)}ms`);
}

log("\n== 6. THE FIX: host-side fan-out via Promise.all ==");
py.globals.set("_host_batch", async (j: string): Promise<string> => {
  const reqs: number[] = JSON.parse(j);
  return JSON.stringify(await Promise.all(reqs.map((i) => new Promise<string>((r) => setTimeout(() => r(`r${i}`), 100)))));
});
{
  const t = now();
  const o = await py.runPythonAsync(`
import json
from pyodide.ffi import run_sync
def llm_batch_hosted(prompts):
    return json.loads(run_sync(_host_batch(json.dumps(prompts))))
llm_batch_hosted(list(range(5)))`);
  const dt = Math.round(now() - t);
  log(`  ${o}  wall=${dt}ms  => ${dt < 300 ? "PARALLELISM PRESERVED" : "serialized"}`);
}
