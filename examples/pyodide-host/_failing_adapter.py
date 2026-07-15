"""Test-only adapter proving relay exception envelope attribution."""


def run_for_host_pyodide(request, host_fetch, bridge_call=None, meta=None):
    raise RuntimeError("adapter boom")
