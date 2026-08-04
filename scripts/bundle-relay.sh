#!/usr/bin/env bash
# Assemble the relay convenience tarball for non-Python consumers (#33).
#
# The EMBEDDER path is the wheel itself — the relay ships as package data under
# droste/substrates/_relay (see `droste relay-path`), so wheel and relay are
# version-locked by construction. This tarball exists for consumers that cannot
# install a Python package.
#
# Lives in a script rather than inline in release.yml so CI can run the exact
# same code on every pull request. Inline, it ran only on a tag, which is how a
# fixture rename shipped a broken release: nothing exercised it beforehand.
#
# Usage: bundle-relay.sh <version-label> <commit-sha> [output-dir]
set -euo pipefail

VERSION_LABEL="${1:?usage: bundle-relay.sh <version-label> <commit-sha> [output-dir]}"
COMMIT_SHA="${2:?usage: bundle-relay.sh <version-label> <commit-sha> [output-dir]}"
OUT_DIR="${3:-relay-dist}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

staging="droste-relay-$VERSION_LABEL"
rm -rf "$staging"
mkdir -p "$staging/conformance" "$OUT_DIR"

cp src/droste/substrates/_relay/*.ts pyodide/README.md "$staging/"

# Enumerated from the package, never listed here: a corpus rename must not
# require finding this file. Fails loudly if the corpus is empty rather than
# shipping a tarball with a silently missing conformance directory.
fixtures="$(cd src/droste/testing/fixtures && ls *.ndjson | sort)"  # single source: the corpus directory itself
if [ -z "$fixtures" ]; then
  echo "::error::conformance corpus is empty — refusing to bundle a relay tarball without it" >&2
  exit 1
fi
while IFS= read -r fixture; do
  cp "src/droste/testing/fixtures/$fixture" "$staging/conformance/"
done <<< "$fixtures"

printf '%s %s\n' "$VERSION_LABEL" "$COMMIT_SHA" > "$staging/DROSTE_VERSION"
tar czf "$OUT_DIR/$staging.tar.gz" "$staging"
rm -rf "$staging"
echo "bundled $OUT_DIR/$staging.tar.gz"
