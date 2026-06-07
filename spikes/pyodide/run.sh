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
mkdir -p "$STAGE/rcl_rlm"
cp -R "$RLMCORE_SRC/rlm_core" "$RLMCORE_SRC/rlm_runner" "$STAGE/"
# Minimal rcl_rlm: just the data layer, with a stub __init__ so importing it does
# NOT drag in the network stack (rcl_rlm/__init__.py eagerly imports httpx via
# .modelrelay, which cannot load under Pyodide — see README).
: > "$STAGE/rcl_rlm/__init__.py"
cp "$RCL/message_database.py" "$RCL/sql_validator.py" "$RCL/exceptions.py" "$STAGE/rcl_rlm/"
cp "$HERE/probe.py" "$STAGE/"
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
( cd "$STAGE" && zip -rq "$ZIP" . -x '*__pycache__*' )
echo "staged: $ZIP   (db dir: $DBDIR)"

# Native baseline path (system python loads the same staged data layer).
export STAGE_DIR="$STAGE"
deno run --allow-read --allow-write --allow-net --allow-env --allow-run \
  "$HERE/$SCRIPT" "$ZIP" "$DBDIR" "$STAGE"
