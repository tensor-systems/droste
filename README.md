# rlm-core

Core library for the Reflexive Loop Model (RLM) - an iterative code-generation and execution framework for LLM-powered agents.

## Overview

RLM enables LLMs to solve complex problems by generating and executing code in a controlled loop:

1. LLM receives a question and generates Python code
2. Code executes in a sandboxed environment
3. Output feeds back to the LLM for refinement
4. Loop continues until `answer["ready"] = True`

## Installation

```bash
# From private index
uv pip install --index-url "https://${PYPI_TOKEN}:x@rlm-pypi.hyperpredict.workers.dev/simple" rlm-core

# For offline builds
uv pip download --dest wheelhouse --index-url "https://${PYPI_TOKEN}:x@rlm-pypi.hyperpredict.workers.dev/simple" rlm-core
uv pip install --no-index --find-links wheelhouse rlm-core
```

## Quick Start

```python
from rlm_core import run_rlm, RLMConfig

result = run_rlm(
    question="What is 2 + 2?",
    environment=my_environment,  # RLMEnvironment implementation
    root_llm=my_llm_client,      # LLMClient implementation
    subcalls=my_subcall_client,  # SubcallClient implementation
    config=RLMConfig(max_iterations=5),
)

print(result.answer)
print(f"Completed in {result.iterations} iterations")
```

## Core Concepts

### Protocols

Implement these protocols to integrate with your infrastructure:

- **`RLMEnvironment`** - Sandboxed code execution environment
- **`LLMClient`** - Chat completion interface for the root LLM
- **`SubcallClient`** - Interface for `llm_query()` and `llm_batch()` calls from generated code
- **`DataSource`** - Optional data source integration

### Configuration

```python
RLMConfig(
    max_iterations=10,      # Max refinement loops
    max_depth=3,            # Max nested subcall depth
    max_calls=50,           # Max total subcalls
    max_output_chars=50000, # Output budget per iteration
    verbose=False,          # Debug logging
)
```

### Result

```python
RLMResult(
    answer="...",           # Final answer
    ready=True,             # Whether answer["ready"] was set
    iterations=3,           # Iterations used
    tokens_used=1500,       # Total tokens consumed
    sub_calls_made=12,      # Total subcalls made
    trajectory=[...],       # Full execution history
    error=None,             # Any error that occurred
)
```

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Build
uv build
```

## License

Proprietary - Tensor Systems
