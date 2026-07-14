"""RawExecutor must not pre-truncate oversized stdout (#44): the output
budget has one chokepoint — the loop's _enforce_output_budget — which raises
SandboxError so the model learns its output was over budget and can narrow
the query. Pre-truncation made the budget check pass and handed the model
silently incomplete data, unlike the native executor path."""

from __future__ import annotations

from droste import RLMConfig, SandboxLimits, run_rlm
from droste.capabilities import broker_subcalls
from droste.execution import BudgetLedger
from droste.protocols.llm_client import TokenUsage
from droste.protocols.subcall_client import SubcallClient
from droste.substrates.pyodide import RawExecutor
from droste.testing import MockLLMClient, MockResponse, MockSubcallClient


def test_raw_executor_returns_full_output() -> None:
    executor = RawExecutor(db=None, max_output_chars=10)
    out = executor.execute("print('x' * 100)")
    assert len(out) == 101  # 100 chars + newline — nothing sliced away


class _RawExecutorEnvironment:
    """Minimal RLMEnvironment over RawExecutor, shaped like the adapters
    that wire it (execute returns the raw string; the loop owns the budget)."""

    def __init__(self) -> None:
        self._executor = RawExecutor(db=None)
        self._globals: dict = {"answer": {"content": "", "ready": False}}

    def capabilities(self):
        return {"tools_in_root": False, "max_output_chars": 50}

    def globals(self):
        return self._globals

    def sandbox_subcalls(self, subcalls: SubcallClient, ledger: BudgetLedger) -> SubcallClient:
        return broker_subcalls(subcalls, ledger)

    def prompt_fragment(self) -> str:
        return ""

    def execute(self, code: str):
        return self._executor.execute(code, extra_globals=self._globals)

    def close(self) -> None:
        self._executor.close()


def test_oversized_output_raises_the_budget_error_end_to_end() -> None:
    # One reply prints over budget; the loop must surface SandboxError (the
    # model's cue to aggregate instead of dumping rows), and the repair reply
    # then completes within budget.
    oversized = MockResponse(
        text="""```python\nprint('x' * 200)\nanswer['ready'] = True\n```""",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    repaired = MockResponse(
        text="""```python\nanswer['content'] = 'ok'\nanswer['ready'] = True\n```""",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    events: list[dict] = []
    result = run_rlm(
        question="q",
        environment=_RawExecutorEnvironment(),
        root_llm=MockLLMClient(responses=[oversized, repaired]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(sandbox=SandboxLimits(output_chars=50)),
        on_event=events.append,
    )
    assert result.ready and result.answer == "ok"
    errors = [e for e in events if e["type"] == "execution_error"]
    assert errors and errors[0]["error_type"] == "SandboxError"
    assert "Sandbox output exceeded 50 characters (attempted 201)" in errors[0]["message"]


def test_oversized_output_retains_returned_stdout_count_when_attempt_is_retained() -> None:
    oversized = MockResponse(
        text="""```python\nprint('x' * 200)\nanswer['ready'] = True\n```""",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    result = run_rlm(
        question="q",
        environment=_RawExecutorEnvironment(),
        root_llm=MockLLMClient(responses=[oversized]),
        subcalls=MockSubcallClient(),
        config=RLMConfig(sandbox=SandboxLimits(output_chars=50)),
    )

    assert result.error is not None
    assert [entry.stdout_chars for entry in result.trajectory] == [201]
    assert result.stdout_chars == 201
