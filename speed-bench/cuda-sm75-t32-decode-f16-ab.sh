#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE_GPU="${PROFILE_GPU:-0}"
CUDA_ARCH="${CUDA_ARCH:-sm_75}"
TIMING_ROUNDS="${TIMING_ROUNDS:-9}"
TIMING_REPEATS="${TIMING_REPEATS:-100}"
WARMUPS="${WARMUPS:-5}"
RUN_SANITIZER="${RUN_SANITIZER:-1}"
SKIP_BUILD="${SKIP_BUILD:-0}"
CREATE_ARCHIVE="${CREATE_ARCHIVE:-1}"
OUT="${T32_DECODE_F16_AB_DIR:-$ROOT/sm75-t32-decode-f16-ab-$(date -u +%Y%m%dT%H%M%SZ)}"

case "$PROFILE_GPU" in
    ''|*[!0-9]*) echo "error: PROFILE_GPU must be one CUDA ordinal" >&2; exit 2 ;;
esac
[[ "$CUDA_ARCH" == "sm_75" ]] || {
    echo "error: this diagnostic requires CUDA_ARCH=sm_75" >&2
    exit 2
}
[[ ! -e "$OUT" ]] || {
    echo "error: output path already exists: $OUT" >&2
    exit 2
}
mkdir -p "$OUT"

finish() {
    local status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$OUT/exit-status.txt"
    if [[ "$CREATE_ARCHIVE" == 1 ]]; then
        tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
        echo "Archive to return: $OUT.tar.gz"
    fi
    exit "$status"
}
trap finish EXIT

{
    echo "git_commit=$(git rev-parse HEAD)"
    echo "profile_gpu=$PROFILE_GPU"
    echo "cuda_arch=$CUDA_ARCH"
    echo "timing_rounds=$TIMING_ROUNDS"
    echo "timing_repeats=$TIMING_REPEATS"
    echo "warmups=$WARMUPS"
    echo "run_sanitizer=$RUN_SANITIZER"
    echo "scope=single-device-production-rank-shape"
    echo "production_dispatch_changed=0"
} > "$OUT/config.txt"
git status --short > "$OUT/git-status.txt"
uname -a > "$OUT/uname.txt"
nvidia-smi \
    --query-gpu=index,name,pci.bus_id,uuid,power.limit \
    --format=csv > "$OUT/gpu-inventory.csv"

if [[ "$SKIP_BUILD" != 1 ]]; then
    make -j"$(nproc)" CUDA_ARCH="$CUDA_ARCH" \
        tests/cuda_sm75_t32_decode_f16_ab \
        2>&1 | tee "$OUT/build.log"
elif [[ ! -x tests/cuda_sm75_t32_decode_f16_ab ]]; then
    echo "error: SKIP_BUILD=1 but the diagnostic executable is missing" >&2
    exit 1
fi

echo "T32 single-token decode F16-output three-arm diagnostic..."
CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
DS4_T32_DECODE_ROUNDS="$TIMING_ROUNDS" \
DS4_T32_DECODE_REPEATS="$TIMING_REPEATS" \
DS4_T32_DECODE_WARMUPS="$WARMUPS" \
./tests/cuda_sm75_t32_decode_f16_ab \
    2>&1 | tee "$OUT/result.log"

if [[ "$RUN_SANITIZER" == 1 ]]; then
    command -v compute-sanitizer >/dev/null || {
        echo "error: RUN_SANITIZER=1 but compute-sanitizer is unavailable" >&2
        exit 1
    }
    echo "Compute Sanitizer: bounded T32 single-token three-arm diagnostic..."
    CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
    DS4_T32_DECODE_ROUNDS=1 \
    DS4_T32_DECODE_REPEATS=1 \
    DS4_T32_DECODE_WARMUPS=1 \
    compute-sanitizer --tool memcheck --error-exitcode=99 \
        ./tests/cuda_sm75_t32_decode_f16_ab \
        > "$OUT/compute-sanitizer.log" 2>&1
    grep -q 'ERROR SUMMARY: 0 errors' "$OUT/compute-sanitizer.log" || {
        echo "error: Compute Sanitizer did not report a clean run" >&2
        exit 1
    }
fi

echo "SM75 T32 single-token decode F16-output A/B complete: $OUT"
