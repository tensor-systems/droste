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


def conformance_fixture_names() -> tuple[str, ...]:
    """Every NDJSON fixture in the shipped conformance corpus, enumerated.

    The corpus is named in packaging checks, the release tarball, and
    downstream suites in other repositories. Listing the filenames in each of
    those means an ABI rename has to be found in all of them, and it will be
    missed in whichever one has no pull request to exercise it -- that is
    exactly how the v8 -> v9 rename reached a release and broke it, in the one
    workflow that only ever runs on a tag.

    Ask here instead, and a rename is invisible to every consumer.
    """

    return tuple(
        sorted(
            path.name
            for path in files(__package__).joinpath("fixtures").iterdir()
            if path.name.endswith(".ndjson")
        )
    )


def trace_v9_lifecycle_ndjson() -> bytes:
    """Return the shared Trace ABI v7 lifecycle conformance corpus."""

    return files(__package__).joinpath("fixtures/trace-v9-lifecycle.ndjson").read_bytes()


def trace_v9_execution_ndjson() -> bytes:
    """Return the shared Trace ABI v7 response/code/output/error conformance corpus."""

    return files(__package__).joinpath("fixtures/trace-v9-execution.ndjson").read_bytes()


def runner_v10_refusal_ndjson() -> bytes:
    """Return the pre-admission runner-v10 refusal fixture."""

    return files(__package__).joinpath("fixtures/runner-v10-refusal.ndjson").read_bytes()


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
    "runner_v10_refusal_ndjson",
    "require_ordered_terminal_events",
    "require_unknown_completion",
    "run_while_blocked",
    "conformance_fixture_names",
    "trace_v9_execution_ndjson",
    "trace_v9_lifecycle_ndjson",
]
