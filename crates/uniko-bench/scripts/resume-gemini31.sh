#!/bin/bash
# Resume the gemini-3.1 10-conv bench.  Wraps the same per-conv
# token-refresh strategy as run-all-convs.sh but (a) skips any conv
# whose output JSON already exists, (b) drops `set -e` so a single
# auth failure or model error doesn't abort the whole queue.
#
# The 2026-05-25 first attempt died inside conv-30 with an
# Unauthorized error — the gcloud OAuth token expired ~60min into a
# long conv that has 369 ingest turns + 105 questions under the
# slow `gemini-3.1-pro-preview` judge.  Per-conv refresh isn't enough
# for the long conversations; if you see Unauthorized again, switch
# to a per-category split instead.
#
# Defaults match the original run.  Override via env if needed.

cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 2

PROFILE="${PROFILE:-crates/uniko-bench/bench-configs/locomo-bge-gemini31.json}"
DATA="${DATA:-data/locomo10.json}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-data/locomo_gemini31}"
RUN_SH="${RUN_SH:-crates/uniko-bench/run.sh}"

CONVS=(
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

START_TS=$(date -Iseconds)
echo "[start] ${START_TS}"
echo "profile: ${PROFILE}"
echo "data:    ${DATA}"
echo "prefix:  ${OUTPUT_PREFIX}"
echo "convs:   ${CONVS[*]}"
echo

for conv in "${CONVS[@]}"; do
    out="${OUTPUT_PREFIX}_${conv}.json"
    log="${OUTPUT_PREFIX}_${conv}.log"
    if [ -s "${out}" ]; then
        echo "[skip ] ${conv}  out=${out} already populated"
        continue
    fi
    conv_start=$(date +%s)
    echo "[run  ] ${conv}  start=$(date -Iseconds)"
    "${RUN_SH}" \
        --bench-config "${PROFILE}" \
        --data "${DATA}" \
        --conversations "${conv}" \
        --reuse \
        --output "${out}" \
        > "${log}" 2>&1
    rc=$?
    conv_end=$(date +%s)
    elapsed=$((conv_end - conv_start))
    if [ "${rc}" -eq 0 ] && [ -s "${out}" ]; then
        echo "[done ] ${conv}  elapsed=${elapsed}s  rc=${rc}"
    else
        echo "[FAIL ] ${conv}  elapsed=${elapsed}s  rc=${rc}  log=${log}"
    fi
done

END_TS=$(date -Iseconds)
echo
echo "[finish] ${END_TS}"
