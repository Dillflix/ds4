#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

CUDA_ARCH=${CUDA_ARCH:-sm_75}
PROFILE_GPU=${PROFILE_GPU:-0}
RUN_SANITIZER=${RUN_SANITIZER:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
DIAGNOSTIC_SCOPE=${DIAGNOSTIC_SCOPE:-full}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-600}
B_TIMING_ROUNDS=${B_TIMING_ROUNDS:-7}
B_TIMING_REPEATS=${B_TIMING_REPEATS:-10}
B_TIMING_WARMUPS=${B_TIMING_WARMUPS:-3}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${TOKEN_ROW_ARITHMETIC_DIR:-$repo_dir/sm75-token-row-arithmetic-$stamp}
target=tests/cuda_sm75_token_row_arithmetic

[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be sm_75"
[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be an integer"
[[ $DIAGNOSTIC_SCOPE == q-b || $DIAGNOSTIC_SCOPE == q-b-native ||
   $DIAGNOSTIC_SCOPE == output-a-native ||
   $DIAGNOSTIC_SCOPE == output-b-canonical ||
   $DIAGNOSTIC_SCOPE == output-b-native ||
   $DIAGNOSTIC_SCOPE == full ]] ||
    die "DIAGNOSTIC_SCOPE must be q-b, q-b-native, output-a-native, output-b-canonical, output-b-native, or full"
[[ $CASE_TIMEOUT_SECONDS =~ ^[1-9][0-9]*$ ]] ||
    die "CASE_TIMEOUT_SECONDS must be a positive integer"
for flag in RUN_SANITIZER SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
for value_name in B_TIMING_ROUNDS B_TIMING_REPEATS B_TIMING_WARMUPS; do
    value=${!value_name}
    [[ $value =~ ^[1-9][0-9]*$ ]] || die "$value_name must be a positive integer"
done
for tool in cat date env git grep journalctl make mkdir nproc nvidia-smi sudo tail tar timeout; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
if (( RUN_SANITIZER )); then
    command -v compute-sanitizer >/dev/null 2>&1 ||
        die "compute-sanitizer not found"
fi
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/provenance" "$OUTPUT_DIR/health"
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
    make -B -j"$(nproc)" "$target" CUDA_ARCH="$CUDA_ARCH" \
        >"$OUTPUT_DIR/build.log" 2>&1 || {
            tail -n 200 "$OUTPUT_DIR/build.log" >&2
            die "build failed"
        }
else
    make -q "$target" CUDA_ARCH="$CUDA_ARCH" ||
        die "SKIP_BUILD=1 found a stale diagnostic"
fi

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,power.limit \
        --format=csv
    printf 'profile_gpu=%s\ndiagnostic_scope=%s\ncase_timeout_seconds=%s\n' \
        "$PROFILE_GPU" "$DIAGNOSTIC_SCOPE" "$CASE_TIMEOUT_SECONDS"
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

# This is an arithmetic-path diagnostic, so inherited engine experiments must
# not silently select a different GEMM, attention, cache, or quality path.
# Keep only the harness-owned selector, which the diagnostic itself consumes.
clean_env=(env -u CUDA_VISIBLE_DEVICES -u CUDA_LAUNCH_BLOCKING)
while IFS='=' read -r name _; do
    if [[ $name == DS4_* ]]; then
        clean_env+=(-u "$name")
    fi
done < <(env)
clean_env+=(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$PROFILE_GPU"
    B_TIMING_ROUNDS="$B_TIMING_ROUNDS"
    B_TIMING_REPEATS="$B_TIMING_REPEATS"
    B_TIMING_WARMUPS="$B_TIMING_WARMUPS")
if [[ $DIAGNOSTIC_SCOPE == q-b || $DIAGNOSTIC_SCOPE == q-b-native ]]; then
    clean_env+=(DS4_TOKEN_ROW_ARITHMETIC_STOP_AFTER_Q_B=1)
fi
if [[ $DIAGNOSTIC_SCOPE == q-b-native ]]; then
    clean_env+=(DS4_TOKEN_ROW_ARITHMETIC_NATIVE_Q_B=1)
fi
if [[ $DIAGNOSTIC_SCOPE == output-b-canonical ]]; then
    clean_env+=(DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_CANONICAL=1)
fi
if [[ $DIAGNOSTIC_SCOPE == output-b-native ]]; then
    clean_env+=(DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_NATIVE=1)
fi
if [[ $DIAGNOSTIC_SCOPE == output-a-native ]]; then
    clean_env+=(DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_A_NATIVE=1)
fi

capture_gpu_health() {
    local output=$1
    timeout 20s nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,pci.bus_id,uuid,memory.used,memory.free,power.limit \
        --format=csv,noheader,nounits >"$output" 2>&1
}

capture_kernel_since() {
    local since=$1
    local output=$2
    if sudo -n true >/dev/null 2>&1; then
        sudo -n journalctl -k --since "$since" --no-pager >"$output" 2>&1 || true
    else
        journalctl -k --since "$since" --no-pager >"$output" 2>&1 || true
    fi
}

capture_gpu_health "$OUTPUT_DIR/health/pre-gpu.csv" ||
    die "could not capture pre-run GPU health"
arm_start=$(date --iso-8601=seconds)

phase=diagnostic
set +e
timeout --signal=TERM --kill-after=10 "$CASE_TIMEOUT_SECONDS" \
    "${clean_env[@]}" "./$target" >"$OUTPUT_DIR/diagnostic.log" 2>&1
diagnostic_status=$?
set -e
capture_kernel_since "$arm_start" "$OUTPUT_DIR/health/kernel.log"
capture_gpu_health "$OUTPUT_DIR/health/post-gpu.csv" || {
    tail -n 240 "$OUTPUT_DIR/diagnostic.log" >&2
    die "post-run GPU health unavailable after diagnostic status $diagnostic_status"
}
! grep -Eiq 'NVRM: Xid|GPU has fallen off|GPU Unavailable|Critical Xid' \
    "$OUTPUT_DIR/diagnostic.log" "$OUTPUT_DIR/health/kernel.log" || {
        tail -n 240 "$OUTPUT_DIR/diagnostic.log" >&2
        die "GPU fault recorded during token-row arithmetic diagnostic"
    }
if (( diagnostic_status != 0 )); then
    tail -n 240 "$OUTPUT_DIR/diagnostic.log" >&2
    die "token-row arithmetic diagnostic failed with status $diagnostic_status"
fi
grep -Fq 'harness_status=ok' "$OUTPUT_DIR/diagnostic.log" ||
    die "diagnostic omitted success marker"
if [[ $DIAGNOSTIC_SCOPE == q-b-native ]]; then
    grep -Fq 'ds4: token-row native-stream dispatch stage=attn_q_b' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "native q_b scope missed the production native-stream dispatch"
    for checkpoint in q-b-native-dequant q-b-activation-f32-to-f16 \
        q-b-cublas-gemm q-b-head-rms-rope; do
        grep -Fq "stage=$checkpoint device=0 status=no error" \
            "$OUTPUT_DIR/diagnostic.log" ||
            die "native q_b scope missed clean $checkpoint checkpoint"
    done
fi
if [[ $DIAGNOSTIC_SCOPE == output-b-canonical ||
      $DIAGNOSTIC_SCOPE == output-b-native ]]; then
    output_b_kind=${DIAGNOSTIC_SCOPE#output-b-}
    grep -Fq "diagnostic_scope=output-b-$output_b_kind-single-launch" \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "$output_b_kind output-B probe omitted its scope marker"
    grep -Fq 'projection_launches=1' "$OUTPUT_DIR/diagnostic.log" ||
        die "$output_b_kind output-B probe did not remain single-launch"
    output_b_checkpoints=(activation-f32-to-f16 cublas-gemm)
    if [[ $DIAGNOSTIC_SCOPE == output-b-native ]]; then
        output_b_checkpoints=(native-dequant "${output_b_checkpoints[@]}")
        grep -Fq 'ds4: token-row native-stream dispatch stage=attn_output_b' \
            "$OUTPUT_DIR/diagnostic.log" ||
            die "native output-B probe missed native-stream dispatch"
    fi
    for checkpoint in "${output_b_checkpoints[@]}"; do
        grep -Fq "ds4: local output-B checkpoint stage=$checkpoint device=0 status=no error" \
            "$OUTPUT_DIR/diagnostic.log" ||
            die "$output_b_kind output-B probe missed clean $checkpoint checkpoint"
    done
    grep -Fq 'canary_prefix_mismatches=0' "$OUTPUT_DIR/diagnostic.log" ||
        die "$output_b_kind output-B probe damaged its prefix canary"
    grep -Fq 'canary_suffix_mismatches=0' "$OUTPUT_DIR/diagnostic.log" ||
        die "$output_b_kind output-B probe damaged its suffix canary"
fi
if [[ $DIAGNOSTIC_SCOPE == output-a-native ]]; then
    grep -Fq 'diagnostic_scope=output-a-native-single-launch' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "native output-A probe omitted its scope marker"
    grep -Fq 'projection_launches=1' "$OUTPUT_DIR/diagnostic.log" ||
        die "native output-A probe did not remain single-launch"
    grep -Fq 'output_b=not-entered' "$OUTPUT_DIR/diagnostic.log" ||
        die "native output-A probe did not prove its B exclusion"
    grep -Fq 'ds4: token-row native-stream dispatch stage=attn_output_a' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "native output-A probe missed native-stream dispatch"
    for checkpoint in native-dequant heads-f32-to-f16 cublas-gemm low-unpack; do
        grep -Fq "ds4: local output-A checkpoint stage=$checkpoint device=0 status=no error" \
            "$OUTPUT_DIR/diagnostic.log" ||
            die "native output-A probe missed clean $checkpoint checkpoint"
    done
    grep -Fq 'canary_prefix_mismatches=0' "$OUTPUT_DIR/diagnostic.log" ||
        die "native output-A probe damaged its prefix canary"
    grep -Fq 'canary_suffix_mismatches=0' "$OUTPUT_DIR/diagnostic.log" ||
        die "native output-A probe damaged its suffix canary"
fi
cat "$OUTPUT_DIR/diagnostic.log"

if (( RUN_SANITIZER )); then
    phase=sanitizer
    sanitizer_start=$(date --iso-8601=seconds)
    "${clean_env[@]}" \
        DS4_TOKEN_ROW_ARITHMETIC_SANITIZER_SMOKE=1 \
        timeout --signal=TERM --kill-after=10 "$CASE_TIMEOUT_SECONDS" \
        compute-sanitizer --tool memcheck --error-exitcode=99 \
        "./$target" >"$OUTPUT_DIR/sanitizer.log" 2>&1 || {
            tail -n 240 "$OUTPUT_DIR/sanitizer.log" >&2
            die "Compute Sanitizer failed"
        }
    capture_kernel_since "$sanitizer_start" \
        "$OUTPUT_DIR/health/sanitizer-kernel.log"
    capture_gpu_health "$OUTPUT_DIR/health/post-sanitizer-gpu.csv" ||
        die "post-sanitizer GPU health unavailable"
    ! grep -Eiq 'NVRM: Xid|GPU has fallen off|GPU Unavailable|Critical Xid' \
        "$OUTPUT_DIR/sanitizer.log" \
        "$OUTPUT_DIR/health/sanitizer-kernel.log" ||
        die "GPU fault recorded during Compute Sanitizer"
    grep -Fq 'ERROR SUMMARY: 0 errors' "$OUTPUT_DIR/sanitizer.log" ||
        die "Compute Sanitizer omitted a clean summary"
fi

phase=summary
grep -E '^(ds4: local native-stream checkpoint|ds4: local output-A checkpoint|ds4: local output-B checkpoint|boundary=|b_algorithm=|first_shipping_exact_b_algorithm=|b_algorithm_conclusion=|b_timing|fastest_shipping_exact_b_|diagnostic_scope=|n_tokens=|groups=|group_dim=|rank=|low_dim=|input_dim=|output_dim=|algorithm=|projection_launches=|peer_access=|native_stream=|output_b=|canary_|low_finite=|low_nonzero=|low_fnv1a64=|output_finite=|output_nonzero=|output_fnv1a64=|diagnostic_conclusion=|harness_status=)' \
    "$OUTPUT_DIR/diagnostic.log" >"$OUTPUT_DIR/summary.txt"
printf 'SM75 token-row arithmetic diagnostic complete: %s\n' "$OUTPUT_DIR"
