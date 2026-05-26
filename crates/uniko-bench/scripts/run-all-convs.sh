#!/bin/bash
# Run a uniko-bench profile over every conversation in
# data/locomo10.json, one conversation per `run.sh` invocation.
#
# Why per-conversation invocations: uni-xervo's vertexai provider
# reads `VERTEXAI_API_TOKEN` once at construction and caches it
# (provider/vertexai.rs:108-130).  gcloud OAuth tokens expire after
# ~60 min, and a single 199-question gemini-3.1-pro-preview judge
# conversation runs ~32 min.  Wrapping each conv in its own bench
# invocation makes `run.sh` refresh the token at the start of each
# leg, so the running provider never holds a token long enough to
# expire mid-run.
#
# Usage:
#   ./crates/uniko-bench/scripts/run-all-convs.sh <profile> [data] [output_prefix]
#
# Defaults:
#   profile       = crates/uniko-bench/bench-configs/locomo-bge-gemini31.json
#   data          = data/locomo10.json
#   output_prefix = data/locomo_gemini31
#
# Per-conv outputs land at `<output_prefix>_<conv-id>.json` and the
# matching `<...>_events.jsonl`.  Each leg is logged to
# `<output_prefix>_<conv-id>.log` so a tail can follow progress.
#
# Set REUSE_KB=1 to pass `--reuse` (skip ingest when the per-conv KB
# dir already exists).  Off by default — every leg ingests fresh so
# embedding-dim mismatches between profiles can't bite.

set -euo pipefail

PROFILE="${1:-crates/uniko-bench/bench-configs/locomo-bge-gemini31.json}"
DATA="${2:-data/locomo10.json}"
OUTPUT_PREFIX="${3:-data/locomo_gemini31}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH="${SCRIPT_DIR}/../run.sh"

# Source list — extracted from data/locomo10.json sample_id field.
# Update this list if the dataset gains / loses conversations.
CONVS=(
    conv-26
    conv-30
    conv-41
    conv-42
    conv-43
    conv-44
    conv-47
    conv-48
    conv-49
    conv-50
)

REUSE_ARG=""
if [ -n "${REUSE_KB:-}" ]; then
    REUSE_ARG="--reuse"
fi

echo "Profile:  ${PROFILE}"
echo "Data:     ${DATA}"
echo "Output:   ${OUTPUT_PREFIX}_<conv>.json"
echo "Convs:    ${CONVS[*]}"
echo "Reuse:    ${REUSE_KB:-no}"
echo

START_TS=$(date -Iseconds)
echo "[start] ${START_TS}"

for conv in "${CONVS[@]}"; do
    conv_start=$(date +%s)
    out_json="${OUTPUT_PREFIX}_${conv}.json"
    leg_log="${OUTPUT_PREFIX}_${conv}.log"
    echo
    echo "=== ${conv}  start=$(date -Iseconds) ==="
    # Each invocation re-runs gcloud auth print-access-token via
    # run.sh, so the token cached by the vertexai provider is at most
    # one conversation old.
    "${RUN_SH}" \
        --bench-config "${PROFILE}" \
        --data "${DATA}" \
        --conversations "${conv}" \
        ${REUSE_ARG} \
        --output "${out_json}" \
        > "${leg_log}" 2>&1
    conv_end=$(date +%s)
    elapsed=$((conv_end - conv_start))
    echo "[done]  ${conv}  elapsed=${elapsed}s  out=${out_json}"
done

END_TS=$(date -Iseconds)
echo
echo "[finish] ${END_TS}"
echo "All ${#CONVS[@]} conversations complete."
echo
echo "Per-conv summaries: ${OUTPUT_PREFIX}_<conv>.json"
echo "Per-conv events:    ${OUTPUT_PREFIX}_<conv>_events.jsonl"
echo "Per-conv logs:      ${OUTPUT_PREFIX}_<conv>.log"
