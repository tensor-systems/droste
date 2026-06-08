import { loadPyodide } from "npm:pyodide";
const py = await loadPyodide();
await py.loadPackage("sqlite3");
const out = await py.runPythonAsync(`
import sqlite3
c = sqlite3.connect(":memory:"); c.execute("CREATE TABLE t(x)"); c.execute("INSERT INTO t VALUES (42)")
"OFFLINE_OK: pyodide " + __import__("pyodide").__version__ + ", sqlite " + sqlite3.sqlite_version + ", query=" + str(c.execute("SELECT x FROM t").fetchone()[0])
`);
console.log(out);
