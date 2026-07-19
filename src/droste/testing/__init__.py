from importlib.resources import files

from .environment import MockEnvironment
from .lifecycle import (
    DEFAULT_LIFECYCLE_TIMEOUT,
    LifecycleGate,
    RecordingAttemptAuthority,
    Settlement,
    ThreadOutcome,
    require_ordered_terminal_events,
    require_unknown_completion,
    run_while_blocked,
)
from .llm_client import MockLLMClient, MockResponse
from .provider import FAKE_RECORDS_MANIFEST, fake_records_provider
from .subcall_client import MockSubcallClient


def trace_v4_lifecycle_ndjson() -> bytes:
    """Return the shared Trace ABI v4 lifecycle conformance corpus."""

    return files(__package__).joinpath("fixtures/trace-v4-lifecycle.ndjson").read_bytes()


def trace_v4_execution_ndjson() -> bytes:
    """Return the shared Trace ABI v4 response/code/output/error conformance corpus."""

    return files(__package__).joinpath("fixtures/trace-v4-execution.ndjson").read_bytes()


def runner_v8_refusal_ndjson() -> bytes:
    """Return the pre-admission runner-v8 refusal fixture."""

    return files(__package__).joinpath("fixtures/runner-v8-refusal.ndjson").read_bytes()


__all__ = [
    "MockEnvironment",
    "FAKE_RECORDS_MANIFEST",
    "MockLLMClient",
    "MockResponse",
    "MockSubcallClient",
    "DEFAULT_LIFECYCLE_TIMEOUT",
    "LifecycleGate",
    "RecordingAttemptAuthority",
    "Settlement",
    "ThreadOutcome",
    "fake_records_provider",
    "runner_v8_refusal_ndjson",
    "require_ordered_terminal_events",
    "require_unknown_completion",
    "run_while_blocked",
    "trace_v4_execution_ndjson",
    "trace_v4_lifecycle_ndjson",
]
