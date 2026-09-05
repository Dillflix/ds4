#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

CUDA_ARCH=${CUDA_ARCH:-sm_75}
PROFILE_GPU=${PROFILE_GPU:-0}
PEER_GPU=${PEER_GPU:-1}
RUN_SANITIZER=${RUN_SANITIZER:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${T32_HEADSHARD_EXACTNESS_DIR:-$repo_dir/sm75-t32-headshard-exactness-$stamp}
target=tests/cuda_sm75_t32_prefill_headshard_exact
xdev_target=tests/cuda_sm75_t32_headshard_xdev_exact

[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be sm_75"
[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be an integer"
[[ $PEER_GPU =~ ^[0-9]+$ ]] || die "PEER_GPU must be an integer"
[[ $PROFILE_GPU != "$PEER_GPU" ]] || die "PROFILE_GPU and PEER_GPU must differ"
for flag in RUN_SANITIZER SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
for tool in cat date env git grep make mkdir nproc nvidia-smi tail tar; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
if (( RUN_SANITIZER )); then
    command -v compute-sanitizer >/dev/null 2>&1 ||
        die "compute-sanitizer not found"
fi
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/provenance"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=build
finish() {
    status=$?
    trap - EXIT INT TERM HUP
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    if (( CREATE_ARCHIVE )); then
        archive="$OUTPUT_DIR.tar.gz"
        tar -C "$(dirname "$OUTPUT_DIR")" -czf "$archive" \
            "$(basename "$OUTPUT_DIR")" || status=1
        printf 'Archive to return: %s\n' "$archive"
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM HUP

if (( SKIP_BUILD == 0 )); then
    make -B -j"$(nproc)" "$target" "$xdev_target" CUDA_ARCH="$CUDA_ARCH" \
        >"$OUTPUT_DIR/build.log" 2>&1 || {
            tail -n 200 "$OUTPUT_DIR/build.log" >&2
            die "build failed"
        }
else
    make -q "$target" "$xdev_target" CUDA_ARCH="$CUDA_ARCH" ||
        die "SKIP_BUILD=1 found a stale diagnostic"
fi

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,power.limit \
        --format=csv
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

phase=exactness
env -u DS4_T32_HEADSHARD_SANITIZER_SMOKE \
    CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
    "./$target" >"$OUTPUT_DIR/exactness.log" 2>&1 || {
    tail -n 240 "$OUTPUT_DIR/exactness.log" >&2
    die "T32 full-versus-head-shard exactness diagnostic failed"
}
grep -Fq 'harness_status=ok' "$OUTPUT_DIR/exactness.log" ||
    die "exactness diagnostic omitted success marker"
cat "$OUTPUT_DIR/exactness.log"

phase=physical-pair-exactness
env CUDA_VISIBLE_DEVICES="$PROFILE_GPU,$PEER_GPU" \
    "./$xdev_target" >"$OUTPUT_DIR/physical-pair-exactness.log" 2>&1 || {
    tail -n 240 "$OUTPUT_DIR/physical-pair-exactness.log" >&2
    die "T32 physical-pair boundary exactness diagnostic failed"
}
grep -Fq 'harness_status=ok' "$OUTPUT_DIR/physical-pair-exactness.log" ||
    die "physical-pair diagnostic omitted success marker"
cat "$OUTPUT_DIR/physical-pair-exactness.log"

if (( RUN_SANITIZER )); then
    phase=sanitizer
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        DS4_T32_HEADSHARD_SANITIZER_SMOKE=1 \
        compute-sanitizer --tool memcheck --error-exitcode=99 \
        "./$target" >"$OUTPUT_DIR/sanitizer.log" 2>&1 || {
            tail -n 240 "$OUTPUT_DIR/sanitizer.log" >&2
            die "Compute Sanitizer failed"
        }
    grep -Fq 'ERROR SUMMARY: 0 errors' "$OUTPUT_DIR/sanitizer.log" ||
        die "Compute Sanitizer omitted a clean summary"
fi

phase=summary
grep -E '^(algorithm=|first_shipping_exact_algorithm=|diagnostic_conclusion=|boundary=|downstream_boundary_conclusion=|harness_status=)' \
    "$OUTPUT_DIR/exactness.log" >"$OUTPUT_DIR/summary.txt"
grep -E '^(boundary=|output_a_rank=|physical_pair_protocol=|production_ordered_protocol=|harness_status=)' \
    "$OUTPUT_DIR/physical-pair-exactness.log" >>"$OUTPUT_DIR/summary.txt"
printf 'SM75 T32 prefill head-shard exactness diagnostic complete: %s\n' \
    "$OUTPUT_DIR"
