#!/bin/bash
# Retry the 4 OAuth-failed gemini-3.1 convs, one category at a time.
# Each run touches only the questions in a single category, so the
# slowest single run (conv-42 cat 4, ~111 questions × ~10s) is ~20min
# — well inside the ~60min Vertex OAuth lifetime per process.
#
# Output: one JSON per (conv, category) at
#   data/locomo_gemini31_${conv}_cat${cat}.json
# Aggregate later with a separate merge script.
#
# Skip-if-populated: existing non-empty output files are skipped so
# this script is safe to re-invoke after failures.
#
# Per-conv KBs already exist under data/kb/${conv}, so --reuse skips
# ingestion (~70s) on every run.

cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 2

PROFILE="${PROFILE:-crates/uniko-bench/bench-configs/locomo-bge-gemini31.json}"
DATA="${DATA:-data/locomo10.json}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-data/locomo_gemini31}"
RUN_SH="${RUN_SH:-crates/uniko-bench/run.sh}"

CONVS=(conv-42 conv-44 conv-48 conv-50)
CATS=(1 2 3 4 5)

START_TS=$(date -Iseconds)
echo "[start] ${START_TS}"
echo "profile: ${PROFILE}"
echo "convs:   ${CONVS[*]}"
echo "cats:    ${CATS[*]}"
echo

for conv in "${CONVS[@]}"; do
    for cat in "${CATS[@]}"; do
        out="${OUTPUT_PREFIX}_${conv}_cat${cat}.json"
        log="${OUTPUT_PREFIX}_${conv}_cat${cat}.log"
        if [ -s "${out}" ]; then
            echo "[skip ] ${conv} cat=${cat}  out=${out} already populated"
            continue
        fi
        t0=$(date +%s)
        echo "[run  ] ${conv} cat=${cat}  start=$(date -Iseconds)"
        "${RUN_SH}" \
            --bench-config "${PROFILE}" \
            --data "${DATA}" \
            --conversations "${conv}" \
            --categories "${cat}" \
            --reuse \
            --output "${out}" \
            > "${log}" 2>&1
        rc=$?
        t1=$(date +%s)
        elapsed=$((t1 - t0))
        if [ "${rc}" -eq 0 ] && [ -s "${out}" ]; then
            echo "[done ] ${conv} cat=${cat}  elapsed=${elapsed}s"
        else
            echo "[FAIL ] ${conv} cat=${cat}  elapsed=${elapsed}s  rc=${rc}  log=${log}"
        fi
    done
done

echo
echo "[finish] $(date -Iseconds)"
