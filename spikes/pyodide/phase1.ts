// Phase 1: data-layer fidelity + volume.
// Runs the VERBATIM MessageDatabase against the 156MB benchmark corpus in two
// runtimes — native CPython and Pyodide (DB FS-mounted, not copied) — and
// compares set-digests. Identical digests ⇒ the host-brokered data layer
// preserves exactly what the LLM sees.   Run:  ./run.sh phase1.ts
import { loadPyodide } from "npm:pyodide";

const [zipPath, dbDir, stageDir] = Deno.args;
const dbFile = `${dbDir}/shadow_corpus_test.db`;
const contactsFile = `${dbDir}/contacts.db`;

// ---------- native baseline (system python, same staged data layer) ----------
console.log("[native] running MessageDatabase under system CPython...");
const native = new Deno.Command("python3", {
  args: [
    "-c",
    `import json, probe; print("__PROBE__" + json.dumps(probe.run(${JSON.stringify(dbFile)}, ${JSON.stringify(contactsFile)})))`,
  ],
  env: { PYTHONPATH: stageDir },
  stdout: "piped",
  stderr: "piped",
}).outputSync();
const nativeOut = new TextDecoder().decode(native.stdout);
const nativeErr = new TextDecoder().decode(native.stderr);
const nativeLine = nativeOut.split("\n").find((l) => l.startsWith("__PROBE__"));
if (!nativeLine) {
  console.error("native probe failed:\n", nativeErr.slice(-1500));
  Deno.exit(1);
}
const nativeRes = JSON.parse(nativeLine.replace("__PROBE__", ""));

// ---------- Pyodide ----------
console.log("[pyodide] loading runtime...");
const pyodide = await loadPyodide();
// sqlite3 is unvendored from the base Pyodide distribution — load it explicitly.
// (For v1: include the sqlite3 package in the bundled Pyodide.)
await pyodide.loadPackage("sqlite3");
const zip = await Deno.readFile(zipPath);
pyodide.unpackArchive(zip, "zip", { extractDir: "/app" });
await pyodide.runPythonAsync(`import sys; sys.path.insert(0, "/app")`);

// Mount the host DB dir into Pyodide's FS (NO copy into WASM heap → volume test).
let mountMode = "none";
try {
  // @ts-ignore - mountNodeFS exists on the Pyodide API in Node-like hosts
  pyodide.mountNodeFS("/data", dbDir);
  mountMode = "mountNodeFS";
} catch (_e) {
  try {
    pyodide.FS.mkdir("/data");
    pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: dbDir }, "/data");
    mountMode = "FS.mount(NODEFS)";
  } catch (e2) {
    console.error("[pyodide] FS mount of host dir failed:", e2);
    Deno.exit(2);
  }
}
console.log(`[pyodide] mounted ${dbDir} → /data via ${mountMode}`);

pyodide.globals.set("__db", "/data/shadow_corpus_test.db");
pyodide.globals.set("__contacts", "/data/contacts.db");
const pyRes = JSON.parse(
  await pyodide.runPythonAsync(`import json, probe; json.dumps(probe.run(__db, __contacts))`),
);

// ---------- compare ----------
const same = (k: string) => nativeRes[k] === pyRes[k];
const verdict =
  same("message_count") && same("get_messages_digest") && same("view_digest");

console.log("\n──────────────── FIDELITY ────────────────");
console.log("mount mode (Pyodide)     :", mountMode);
console.log("sqlite  native / pyodide :", nativeRes.sqlite_version, "/", pyRes.sqlite_version);
console.log("messages native / pyodide:", nativeRes.message_count, "/", pyRes.message_count);
console.log("get_messages columns     :", JSON.stringify(pyRes.get_messages_columns));
console.log("get_messages rows        :", nativeRes.get_messages_rows, "/", pyRes.get_messages_rows);
console.log("get_messages digest match:", same("get_messages_digest"), nativeRes.get_messages_digest.slice(0, 12), "/", pyRes.get_messages_digest.slice(0, 12));
console.log("view(query) rows         :", nativeRes.view_rows, "/", pyRes.view_rows);
console.log("view  digest match       :", same("view_digest"), nativeRes.view_digest.slice(0, 12), "/", pyRes.view_digest.slice(0, 12));
console.log("──────────────────────────────────────────");
console.log(verdict ? "✅ FIDELITY PARITY: native == Pyodide" : "❌ DIVERGENCE — investigate above");
