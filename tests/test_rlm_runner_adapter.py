import sys
import types

from rlm_runner import runner


def test_rlm_runner_adapter_delegates() -> None:
    module_name = "tests_dummy_rlm_adapter"
    module = types.ModuleType(module_name)

    def run(req: dict) -> dict:
        return {"answer": req.get("answer", "ok")}

    module.run = run  # type: ignore[attr-defined]
    sys.modules[module_name] = module

    try:
        result = runner.run({"adapter_module": module_name, "answer": "hello"})
    finally:
        sys.modules.pop(module_name, None)

    assert result["answer"] == "hello"
