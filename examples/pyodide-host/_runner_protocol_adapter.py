"""Test-only adapter exposing the canonical runner admission boundary."""

from droste_runner.run import run_worker_request


def run_for_host_pyodide(
    request,
    host_fetch,
    bridge_call=None,
    duplex_bridge_call=None,
    meta=None,
):
    return run_worker_request(request).response
