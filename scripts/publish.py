#!/usr/bin/env python3
"""
Publish droste package to R2-backed PyPI index.

Usage:
    uv build
    uv run python scripts/publish.py

Requires CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN environment variables.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path


def get_sha256(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def upload_to_r2(filepath: Path, bucket: str = "rlm-pypi") -> None:
    """Upload a file to R2 bucket using wrangler."""
    filename = filepath.name
    sha256 = get_sha256(filepath)

    print(f"Uploading {filename} (sha256: {sha256[:16]}...)")

    # Upload file with SHA256 in custom metadata
    cmd = [
        "npx", "wrangler", "r2", "object", "put",
        f"{bucket}/packages/{filename}",
        "--file", str(filepath),
        "--content-type", "application/octet-stream",
        "--remote",  # Upload to remote R2, not local storage
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error uploading {filename}:")
        print(result.stderr)
        sys.exit(1)

    print(f"  ✓ Uploaded to packages/{filename}")


def find_dist_files() -> list[Path]:
    """Find wheel and sdist files in dist/."""
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("Error: dist/ directory not found. Run 'uv build' first.")
        sys.exit(1)

    files = list(dist_dir.glob("*.whl")) + list(dist_dir.glob("*.tar.gz"))
    if not files:
        print("Error: No wheel or sdist files found in dist/")
        sys.exit(1)

    return files


def main() -> None:
    # Check we're in the right directory
    if not Path("pyproject.toml").exists():
        print("Error: Must be run from project root (where pyproject.toml is)")
        sys.exit(1)

    # Find distribution files
    files = find_dist_files()
    print(f"Found {len(files)} distribution file(s):")
    for f in files:
        print(f"  - {f.name}")
    print()

    # Upload each file
    for filepath in files:
        upload_to_r2(filepath)

    print()
    print("✓ All files uploaded successfully!")
    print()
    print("To install:")
    print("  uv pip install --index-url https://rlm-pypi.<your-subdomain>.workers.dev/simple droste")


if __name__ == "__main__":
    main()
