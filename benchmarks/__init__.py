"""Repository-local reproducible benchmark tooling.

This package is deliberately excluded from the published wheel. It owns the
versioned evaluation contract without coupling benchmark dependencies to the
Droste runtime.
"""

from .models import RunArtifact, SuiteManifest, load_manifest

__all__ = ["RunArtifact", "SuiteManifest", "load_manifest"]
