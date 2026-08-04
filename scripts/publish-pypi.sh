#!/usr/bin/env bash
# Publish dist/* to PyPI, converging rather than executing once.
#
# A release is retried by re-running the workflow or re-pushing the tag, so
# every step has to be safe to run twice. Publishing is the one step that
# cannot be undone, which makes it the one that most needs to be re-runnable.
#
# `uv publish --check-url` was supposed to provide that and does not. It
# predicts from the Simple index, which is CDN-cached, and it compares against
# freshly built artifacts — but sdists are not byte-reproducible, so a rebuild
# is a *different* file with the same name. Either way the upload is attempted
# and PyPI rejects it, so a release whose later step failed could never be
# retried at the same version.
#
# The authoritative question is not "does the index list files that match the
# ones I just built" but "does this version exist on PyPI". Ask that, of the
# API rather than the cache, and only after the upload has actually been tried.
#
# Usage: publish-pypi.sh <version> [--dry-run]
set -euo pipefail

VERSION="${1:?usage: publish-pypi.sh <version> [--dry-run]}"
DRY_RUN="${2:-}"

# `uv publish` uploads dist/* wholesale, so anything stale left in that
# directory is published alongside the release. A workspace that is not
# scrubbed between builds — a self-hosted runner, a developer's checkout —
# turns a leftover artifact into an unintended upload of some older version.
# Refuse rather than trust the directory to hold only what this release built.
unexpected=""
for artifact in dist/*; do
  [ -e "$artifact" ] || continue
  case "$(basename "$artifact")" in
    "droste-$VERSION.tar.gz" | "droste-$VERSION-"*.whl) ;;
    *) unexpected="$unexpected $(basename "$artifact")" ;;
  esac
done
if [ -n "$unexpected" ]; then
  echo "::error::dist/ holds artifacts that are not droste $VERSION:$unexpected" >&2
  echo "::error::clear dist/ and rebuild — publishing would upload them too" >&2
  exit 1
fi

if [ "$DRY_RUN" = "--dry-run" ]; then
  echo "dry run: validating dist/ without uploading"
  uv publish --dry-run
  exit 0
fi

version_is_published() {
  # Query the JSON API for the exact version. 404 means absent; any other
  # failure is inconclusive and must not be read as "already published".
  local status
  status="$(curl -s -o /dev/null -w '%{http_code}' \
    "https://pypi.org/pypi/droste/$VERSION/json")"
  case "$status" in
    200) return 0 ;;
    404) return 1 ;;
    *)
      echo "::error::PyPI returned $status for droste $VERSION — cannot tell whether it is published" >&2
      exit 1
      ;;
  esac
}

if version_is_published; then
  echo "droste $VERSION is already on PyPI — nothing to publish"
  exit 0
fi

if uv publish --trusted-publishing always; then
  echo "published droste $VERSION"
  exit 0
fi

# The upload failed. That is only acceptable if the version is now present,
# which happens when a previous attempt uploaded it (or this attempt partially
# succeeded before erroring on a duplicate filename). Re-ask the authority.
if version_is_published; then
  echo "publish reported an error but droste $VERSION is on PyPI — treating as already published"
  exit 0
fi

echo "::error::publish failed and droste $VERSION is not on PyPI" >&2
exit 1
