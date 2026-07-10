// deps.ts — the ONE Pyodide pin site (#33). Every .ts file in this repo
// imports loadPyodide from here; bump the version on this line and nowhere
// else (the CI deno cache key names the same version — bump it alongside).
//
// Why pinned at all: an unpinned `npm:pyodide` lets `deno cache` drift to a
// new major (e.g. 314.0.0) whose sqlite3 package/API differs, breaking the
// offline bundle ("No known package with name 'sqlite3'"). This is the
// version the relay is built and tested against (run_sync + mountNodeFS +
// loadPackage("sqlite3")).
export { loadPyodide } from "npm:pyodide@0.29.4";
