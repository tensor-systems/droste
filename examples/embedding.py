"""Minimal in-process Droste embedding using an OpenAI-compatible endpoint."""

from __future__ import annotations

import os

from droste import (
    Budget,
    EnvironmentConfig,
    OpenAICompatClient,
    OpenAICompatSubcallClient,
    RLMConfig,
    create_environment,
    create_environment_context,
    run_rlm,
)


def ask(question: str, data: str, *, model: str) -> str:
    config = EnvironmentConfig(kind="native", budget=Budget(subcalls=50, depth=1))
    context = create_environment_context(config)
    root = OpenAICompatClient(model=model)
    subcalls = OpenAICompatSubcallClient(model=model, context=context)
    environment = create_environment(
        config,
        context=data,
        registry=None,
        subcalls=subcalls,
        execution_context=context,
    )
    result = run_rlm(
        question,
        environment=environment,
        root_llm=root,
        subcalls=subcalls,
        config=RLMConfig(root_model=model),
        context=context,
    )
    return result.answer


if __name__ == "__main__":
    print(ask("What happened?", "your source data", model=os.environ["DROSTE_MODEL"]))
