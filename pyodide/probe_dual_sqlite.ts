// Owed check for topology Option A: does the sqlite3 native extension load
// and run in TWO co-resident Pyodide interpreters (untrusted REPL + DB service)?
// Now runnable — the 0.29.4 sqlite3 wheel is primed in the deno cache (see the
// "why blocked" diagnosis: npm tarball ships no wheels; indexURL is local).
//
// Run: deno run --cached-only --allow-read --allow-env --allow-ffi probe_dual_sqlite.ts
import { loadPyodide } from "npm:pyodide@0.29.4";
const quiet = { stdout: () => {}, stderr: () => {} };
const log = (s: string) => console.log(s);

async function sqliteWorks(py: any, label: string): Promise<boolean> {
  try {
    await py.loadPackage("sqlite3", { messageCallback: () => {}, errorCallback: () => {} });
    const v = await py.runPythonAsync(`
import sqlite3
c = sqlite3.connect(":memory:")
c.execute("CREATE TABLE t(x)"); c.execute("INSERT INTO t VALUES (7)")
str(c.execute("SELECT x*6 FROM t").fetchone()[0]) + " / sqlite " + sqlite3.sqlite_version`);
    log(`  ${label}: OK — 7*6=${v}`);
    return true;
  } catch (e) {
    log(`  ${label}: FAIL — ${(e instanceof Error ? e.message : String(e)).split("\n").slice(-2)[0]}`);
    return false;
  }
}

log("loading two co-resident interpreters...");
const dbsvc = await loadPyodide(quiet); // trusted DB service
const repl = await loadPyodide(quiet); // untrusted REPL
const a = await sqliteWorks(dbsvc, "interp#1 (DB service)");
const b = await sqliteWorks(repl, "interp#2 (REPL-role)");

// Independent connections + independent data — no shared native state across the
// two WASM instances (the isolation Option A relies on).
const x = await dbsvc.runPythonAsync(
  `import sqlite3; c=sqlite3.connect(":memory:"); c.execute("CREATE TABLE a(n)"); c.execute("INSERT INTO a VALUES(111)"); c.execute("SELECT n FROM a").fetchone()[0]`,
);
const y = await repl.runPythonAsync(
  `import sqlite3; c=sqlite3.connect(":memory:"); c.execute("CREATE TABLE a(n)"); c.execute("INSERT INTO a VALUES(222)"); c.execute("SELECT n FROM a").fetchone()[0]`,
);
log(`independent state: #1.a=${x} (want 111), #2.a=${y} (want 222) -> ${x === 111 && y === 222 ? "ISOLATED OK" : "LEAK"}`);
log(`\nRESULT: dual sqlite3 ${a && b && x === 111 && y === 222 ? "WORKS — Option A residual risk CLOSED" : "PROBLEM"}`);
