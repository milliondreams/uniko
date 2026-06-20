#!/bin/bash
# Run a uniko-bench binary with the GPU runtime env already set.
#
# This script builds uniko-bench with `--features gpu-cuda` (the crate is
# CPU-by-default so the plain workspace build stays CUDA-free for CI). The
# resulting binary needs three runtime pieces ORT/CUDA can't find on
# its own:
#
#   1. The CUDA execution provider (libonnxruntime_providers_cuda.so)
#      lives under ~/.cache/ort.pyke.io and must be on LD_LIBRARY_PATH
#      or ORT silently falls back to CPU.
#   2. cuDNN 9.x built for CUDA 13 (cublasLt 13 ABI). The system cuDNN
#      under /usr/local/cuda*/lib/ is built for CUDA 12 — incompatible.
#      The `nvidia-cudnn-cu13` pip package ships a working cuDNN at
#      ~/.local/lib/python3.13/site-packages/nvidia/cudnn/lib.
#   3. A linker shim that drops `-L /usr/lib` so the 64-bit libcuda.so
#      under /usr/lib64 is preferred over the 32-bit multilib stub.
#      Used only by builds, not at runtime.  See scripts/cc_no_usrlib.sh.
#
# Two modes:
#
#   ./crates/uniko-bench/run.sh [args...]
#       Run mode — exec the prebuilt binary at
#       target/release/<BIN> with the runtime env set. No cargo, no
#       RUSTFLAGS, no rebuild. Fast startup. If the binary is
#       missing, falls through to a one-shot build first.
#
#   ./crates/uniko-bench/run.sh --build
#       Build mode — invoke `cargo build --release` with the linker
#       shim so the build picks up source changes. After it
#       completes, re-run without `--build` to launch the bench.
#
# Splitting the modes keeps cargo's incremental cache in one stable
# slot (RUSTFLAGS=<shim>) instead of thrashing it against bare
# `cargo build` invocations from other shells / IDEs.
#
# Env knobs:
#
#   BIN=longmemeval-bench   Pick which uniko-bench binary to launch.
#                           Default: uniko-bench.
#   NO_GPU=1                Build/run without `gpu-cuda` (and without
#                           the linker shim or LD_LIBRARY_PATH).
#                           Slow but works on machines lacking CUDA.

set -euo pipefail

BIN="${BIN:-uniko-bench}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BIN_PATH="${REPO_ROOT}/target/release/${BIN}"

# CPU-only escape hatch — defers to bare cargo on a separate target
# slot.  Useful for laptops / CI without the CUDA toolkit.
if [ -n "${NO_GPU:-}" ]; then
    if [ "${1:-}" = "--build" ]; then
        shift
        exec cargo build --release --manifest-path "${REPO_ROOT}/Cargo.toml" \
            -p uniko-bench --bin "${BIN}" --no-default-features
    fi
    exec cargo run --release --manifest-path "${REPO_ROOT}/Cargo.toml" \
        -p uniko-bench --bin "${BIN}" --no-default-features -- "$@"
fi

# Resolve runtime env (ORT cache + cuDNN 13 + CUDA toolkit libs).
ORT_LIB=$(find "${HOME}/.cache/ort.pyke.io" -name 'libonnxruntime_providers_cuda.so' \
            -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}')
if [ -z "${ORT_LIB}" ]; then
    echo "warn: no libonnxruntime_providers_cuda.so under ~/.cache/ort.pyke.io;" >&2
    echo "      CUDA EP may not register. Re-run after a successful build to populate it." >&2
fi

CUDNN13_LIB="${HOME}/.local/lib/python3.13/site-packages/nvidia/cudnn/lib"
if [ ! -d "${CUDNN13_LIB}" ]; then
    echo "warn: cuDNN13 not at ${CUDNN13_LIB};" >&2
    echo "      install with: pip install --user nvidia-cudnn-cu13" >&2
fi

export LD_LIBRARY_PATH="${ORT_LIB}:${CUDNN13_LIB}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# Auto-populate VERTEXAI_API_TOKEN from `gcloud auth print-access-token`
# unless the caller already set it.  uni-xervo's `remote/vertexai`
# provider reads this env var (lib.rs:158 wires the alias to it).
# Tokens expire ~1h; a long bench may need re-auth mid-run, but the
# common LoCoMo conv-26 run finishes well under that.
if [ -z "${VERTEXAI_API_TOKEN:-}" ] && command -v gcloud >/dev/null 2>&1; then
    if tok=$(gcloud auth print-access-token 2>/dev/null) && [ -n "$tok" ]; then
        export VERTEXAI_API_TOKEN="$tok"
    fi
fi

# Build mode: invoke cargo with the linker shim and exit.
if [ "${1:-}" = "--build" ]; then
    shift
    export RUSTFLAGS="-C linker=${SCRIPT_DIR}/scripts/cc_no_usrlib.sh"
    exec cargo build --release --manifest-path "${REPO_ROOT}/Cargo.toml" \
        -p uniko-bench --bin "${BIN}" --features gpu-cuda "$@"
fi

# Run mode: exec the prebuilt binary directly so cargo's cache stays
# pristine.  Source changes are *not* picked up automatically — pass
# `--build` first when the binary needs refreshing.
if [ ! -x "${BIN_PATH}" ]; then
    echo "info: ${BIN_PATH} missing; building once..." >&2
    export RUSTFLAGS="-C linker=${SCRIPT_DIR}/scripts/cc_no_usrlib.sh"
    cargo build --release --manifest-path "${REPO_ROOT}/Cargo.toml" \
        -p uniko-bench --bin "${BIN}" --features gpu-cuda
fi

exec "${BIN_PATH}" "$@"
