#!/usr/bin/env bash
# Stage rlm-core + the (verbatim) rcl_rlm data layer into a zip Pyodide can load,
# then run a spike script under Deno.
#
#   ./run.sh spike.ts      # WASM viability probe (imports + bridge + blockers)
#   ./run.sh phase1.ts     # data-layer fidelity: native vs Pyodide on the corpus
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RLMCORE_SRC="$HERE/../../src"                              # rlm-core/src
RCL="$HERE/../../../cozybot/tools/rcl-rlm/src/rcl_rlm"     # sibling cozybot
STAGE="$HERE/_stage"
ZIP="$HERE/_stage.zip"
SCRIPT="${1:-spike.ts}"
DBDIR="${2:-$HOME/Library/Application Support/RecallRLM}"

rm -rf "$STAGE" "$ZIP"
mkdir -p "$STAGE"
cp -R "$RLMCORE_SRC/rlm_core" "$RLMCORE_SRC/rlm_runner" "$STAGE/"
# Full rcl_rlm package with its real (refactored) __init__. Importing the data
# layer (rcl_rlm.message_database) must NOT drag in the network stack — the lazy
# __init__ guarantees this: modelrelay/httpx + the full RLM stack load only on
# demand, never during a data-layer import. (Previously this staged a 3-file stub
# with an empty __init__ to dodge the eager httpx import — no longer needed.)
cp -R "$RCL" "$STAGE/rcl_rlm"
cp "$HERE/probe.py" "$STAGE/"
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
( cd "$STAGE" && zip -rq "$ZIP" . -x '*__pycache__*' )
echo "staged: $ZIP   (db dir: $DBDIR)"

# Native baseline path (system python loads the same staged data layer).
export STAGE_DIR="$STAGE"
deno run --allow-read --allow-write --allow-net --allow-env --allow-run \
  "$HERE/$SCRIPT" "$ZIP" "$DBDIR" "$STAGE"
