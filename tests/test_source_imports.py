from __future__ import annotations

import subprocess
import sys


def test_sources_package_lazy_loads_native_mcp_http_exports() -> None:
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import droste.sources as sources; "
            "assert 'droste.sources.mcp_http' not in sys.modules; "
            "assert sources.McpHttpHost.__name__ == 'McpHttpHost'; "
            "assert 'droste.sources.mcp_http' in sys.modules",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert check.returncode == 0, check.stderr
