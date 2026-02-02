# Agent Guidelines

## Package Management

This project uses **uv** as the default package manager.

- Install dependencies: `uv sync`
- Add a dependency: `uv add <package>`
- Add a dev dependency: `uv add --group dev <package>`
- Run tests: `uv run pytest`

## Installing from Private Index

To install `rlm-core` from the private PyPI index:

```bash
uv pip install --index-url https://${PYPI_TOKEN}:x@rlm-pypi.hyperpredict.workers.dev/simple rlm-core
```

Or add to your `pyproject.toml`:

```toml
[[tool.uv.index]]
url = "https://rlm-pypi.hyperpredict.workers.dev/simple"
```

Then set `UV_INDEX_PYPI_TOKEN` or use a `.netrc` file for authentication.

## Publishing

Publishing is automated via GitHub Actions on `v*` tags. To publish manually:

```bash
uv build
uv run python scripts/publish.py
```

Requires `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` environment variables.
