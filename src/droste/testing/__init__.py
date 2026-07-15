from importlib.resources import files

from .environment import MockEnvironment
from .llm_client import MockLLMClient, MockResponse
from .provider import FAKE_RECORDS_MANIFEST, fake_records_provider
from .subcall_client import MockSubcallClient


def trace_v2_lifecycle_ndjson() -> bytes:
    """Return the shared Trace ABI v2 lifecycle conformance corpus."""

    return files(__package__).joinpath("fixtures/trace-v2-lifecycle.ndjson").read_bytes()


__all__ = [
    "MockEnvironment",
    "FAKE_RECORDS_MANIFEST",
    "MockLLMClient",
    "MockResponse",
    "MockSubcallClient",
    "fake_records_provider",
    "trace_v2_lifecycle_ndjson",
]
