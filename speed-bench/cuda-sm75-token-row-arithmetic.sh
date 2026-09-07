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
OUTPUT_AB_REPEAT_CALLS=${OUTPUT_AB_REPEAT_CALLS:-256}
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
   $DIAGNOSTIC_SCOPE == output-ab-native ||
   $DIAGNOSTIC_SCOPE == output-ab-native-repeat ||
   $DIAGNOSTIC_SCOPE == projection-chain-native ||
   $DIAGNOSTIC_SCOPE == projection-chain-native-repeat ||
   $DIAGNOSTIC_SCOPE == output-b-canonical ||
   $DIAGNOSTIC_SCOPE == output-b-canonical-replay ||
   $DIAGNOSTIC_SCOPE == output-b-native ||
   $DIAGNOSTIC_SCOPE == full ]] ||
    die "DIAGNOSTIC_SCOPE must be q-b, q-b-native, output-a-native, output-ab-native, output-ab-native-repeat, projection-chain-native, projection-chain-native-repeat, output-b-canonical, output-b-canonical-replay, output-b-native, or full"
[[ $CASE_TIMEOUT_SECONDS =~ ^[1-9][0-9]*$ ]] ||
    die "CASE_TIMEOUT_SECONDS must be a positive integer"
[[ $OUTPUT_AB_REPEAT_CALLS =~ ^[1-9][0-9]*$ ]] ||
    die "OUTPUT_AB_REPEAT_CALLS must be a positive integer"
(( OUTPUT_AB_REPEAT_CALLS <= 4096 )) ||
    die "OUTPUT_AB_REPEAT_CALLS must not exceed 4096"
if [[ $DIAGNOSTIC_SCOPE == output-ab-native-repeat ||
      $DIAGNOSTIC_SCOPE == projection-chain-native-repeat ]]; then
    (( OUTPUT_AB_REPEAT_CALLS >= 2 )) ||
        die "OUTPUT_AB_REPEAT_CALLS must be at least 2 in repeat scope"
fi
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
    printf 'profile_gpu=%s\ndiagnostic_scope=%s\ncase_timeout_seconds=%s\noutput_ab_repeat_calls=%s\n' \
        "$PROFILE_GPU" "$DIAGNOSTIC_SCOPE" "$CASE_TIMEOUT_SECONDS" \
        "$OUTPUT_AB_REPEAT_CALLS"
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
if [[ $DIAGNOSTIC_SCOPE == output-b-canonical-replay ]]; then
    clean_env+=(DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_CANONICAL_REPLAY=1)
fi
if [[ $DIAGNOSTIC_SCOPE == output-b-native ]]; then
    clean_env+=(DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_NATIVE=1)
fi
if [[ $DIAGNOSTIC_SCOPE == output-a-native ]]; then
    clean_env+=(DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_A_NATIVE=1)
fi
if [[ $DIAGNOSTIC_SCOPE == output-ab-native ]]; then
    clean_env+=(DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_AB_NATIVE=1)
fi
if [[ $DIAGNOSTIC_SCOPE == output-ab-native-repeat ]]; then
    clean_env+=(DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_AB_NATIVE_REPEAT=1
        DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_AB_REPEAT_CALLS="$OUTPUT_AB_REPEAT_CALLS")
fi
if [[ $DIAGNOSTIC_SCOPE == projection-chain-native ]]; then
    clean_env+=(DS4_TOKEN_ROW_ARITHMETIC_PROJECTION_CHAIN_NATIVE=1)
fi
if [[ $DIAGNOSTIC_SCOPE == projection-chain-native-repeat ]]; then
    clean_env+=(DS4_TOKEN_ROW_ARITHMETIC_PROJECTION_CHAIN_NATIVE_REPEAT=1
        DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_AB_REPEAT_CALLS="$OUTPUT_AB_REPEAT_CALLS")
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
if [[ $DIAGNOSTIC_SCOPE == output-b-canonical-replay ]]; then
    grep -Fq 'diagnostic_scope=output-b-canonical-replay' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "canonical output-B replay omitted its scope marker"
    grep -Fq 'resident_f16_cache_bytes=201326592' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "canonical output-B replay did not preserve the three-weight FP16 cache"
    grep -Fq 'peer_access=none' "$OUTPUT_DIR/diagnostic.log" ||
        die "canonical output-B replay unexpectedly enabled peer access"
    grep -Fq 'exhaustive_algorithm_sweep=off' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "canonical output-B replay unexpectedly entered the unsafe sweep"
    grep -Fq 'replay_step=default-suffix-row1-256,event=complete,status=ok' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "canonical output-B replay did not complete its final synchronized step"
    ! grep -Eq 'untouched_payload_mismatches=[1-9][0-9]*' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "canonical output-B replay overwrote an unselected row half"
    ! grep -Eq 'selected_poison_words=[1-9][0-9]*' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "canonical output-B replay left selected output unwritten"
    ! grep -Eq 'expected_bit_mismatches=[1-9][0-9]*' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "canonical output-B replay changed arithmetic across row extents"
    grep -Fq 'fidelity=bounded-synchronized-transition-probe' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "canonical output-B replay omitted its bounded-fidelity marker"
    grep -Fq 'original_device_working_set_reproduced=0' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "canonical output-B replay omitted its working-set limitation"
    grep -Fq 'diagnostic_conclusion=canonical-output-b-replay-clean' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "canonical output-B replay omitted its clean conclusion"
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
if [[ $DIAGNOSTIC_SCOPE == output-ab-native ||
      $DIAGNOSTIC_SCOPE == output-ab-native-repeat ||
      $DIAGNOSTIC_SCOPE == projection-chain-native ||
      $DIAGNOSTIC_SCOPE == projection-chain-native-repeat ]]; then
    expected_scope=output-ab-native-single-call
    expected_calls=1
    expected_reference_calls=0
    expected_stress_calls=1
    expected_conclusion=native-stream-output-ab-single-call-clean
    if [[ $DIAGNOSTIC_SCOPE == output-ab-native-repeat ]]; then
        expected_scope=output-ab-native-repeat
        expected_calls=$((OUTPUT_AB_REPEAT_CALLS + 1))
        expected_reference_calls=1
        expected_stress_calls=$OUTPUT_AB_REPEAT_CALLS
        expected_conclusion=native-stream-output-ab-repeat-clean
    elif [[ $DIAGNOSTIC_SCOPE == projection-chain-native ]]; then
        expected_scope=projection-chain-native
        expected_conclusion=native-stream-projection-chain-clean
    elif [[ $DIAGNOSTIC_SCOPE == projection-chain-native-repeat ]]; then
        expected_scope=projection-chain-native-repeat
        expected_calls=$((OUTPUT_AB_REPEAT_CALLS + 1))
        expected_reference_calls=1
        expected_stress_calls=$OUTPUT_AB_REPEAT_CALLS
        expected_conclusion=native-stream-projection-chain-repeat-clean
    fi
    grep -Fq "diagnostic_scope=$expected_scope" \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "native A-to-B probe omitted its scope marker"
    grep -Fq 'ds4: rebased borrowed native-Q8 cache views device=0 count=1' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "native A-to-B probe did not exercise the borrowed-view rebase"
    if [[ $DIAGNOSTIC_SCOPE == projection-chain-native ||
          $DIAGNOSTIC_SCOPE == projection-chain-native-repeat ]]; then
        grep -Fq 'ds4: rebased borrowed native-Q8 cache views device=0 count=2' \
            "$OUTPUT_DIR/diagnostic.log" ||
            die "native projection chain missed its second borrowed-view rebase"
        grep -Fq "q_b_launches=$expected_calls" "$OUTPUT_DIR/diagnostic.log" ||
            die "native projection chain reported the wrong Q_B call count"
        grep -Fq 'q_b_to_a_sync=0' "$OUTPUT_DIR/diagnostic.log" ||
            die "native projection chain unexpectedly fenced Q_B-to-A"
        grep -Fq 'ds4: token-row native-stream dispatch stage=attn_q_b' \
            "$OUTPUT_DIR/diagnostic.log" ||
            die "native projection chain missed Q_B native-stream dispatch"
    fi
    grep -Fq "reference_calls=$expected_reference_calls" \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "native A-to-B probe reported the wrong reference-call count"
    grep -Fq "stress_calls=$expected_stress_calls" \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "native A-to-B probe reported the wrong stress-call count"
    grep -Fq "attention_output_calls=$expected_calls" \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "native A-to-B probe reported the wrong total call count"
    grep -Fq 'production_order=unfenced-a-to-b' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "native A-to-B probe did not preserve production ordering"
    grep -Fq 'handoff_sync_before_b=0' "$OUTPUT_DIR/diagnostic.log" ||
        die "native A-to-B probe unexpectedly fenced the handoff"
    for stage in attn_output_a attn_output_b; do
        grep -Fq "ds4: token-row native-stream dispatch stage=$stage" \
            "$OUTPUT_DIR/diagnostic.log" ||
            die "native A-to-B probe missed $stage native-stream dispatch"
    done
    if [[ $DIAGNOSTIC_SCOPE == output-ab-native ]]; then
        for checkpoint in native-dequant activation-f32-to-f16 cublas-gemm; do
            grep -Fq "ds4: local output-B checkpoint stage=$checkpoint device=0 status=no error" \
                "$OUTPUT_DIR/diagnostic.log" ||
                die "native A-to-B probe missed clean B $checkpoint checkpoint"
        done
    fi
    for canary in low_canary_prefix_mismatches \
        low_canary_suffix_mismatches out_canary_prefix_mismatches \
        out_canary_suffix_mismatches; do
        grep -Fq "$canary=0" "$OUTPUT_DIR/diagnostic.log" ||
            die "native A-to-B probe damaged $canary"
    done
    grep -Fq 'low_repeat_bit_mismatches=0' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "native A-to-B probe changed its repeated low output"
    grep -Fq 'output_repeat_bit_mismatches=0' \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "native A-to-B probe changed its repeated final output"
    grep -Fq "diagnostic_conclusion=$expected_conclusion" \
        "$OUTPUT_DIR/diagnostic.log" ||
        die "native A-to-B probe omitted its clean conclusion"
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
grep -E '^(ds4: rebased borrowed native-Q8 cache views|ds4: local native-stream checkpoint|ds4: local output-A checkpoint|ds4: local output-B checkpoint|boundary=|b_algorithm=|first_shipping_exact_b_algorithm=|b_algorithm_conclusion=|b_timing|fastest_shipping_exact_b_|diagnostic_scope=|n_tokens=|groups=|group_dim=|rank=|low_dim=|input_dim=|output_dim=|algorithm=|projection_launches=|q_b_launches=|q_b_to_a_sync=|reference_calls=|stress_calls=|attention_output_calls=|output_a_launches=|output_b_launches=|peer_access=|native_stream=|output_b=|production_order=|handoff_sync_before_b=|canary_|low_canary_|out_canary_|low_finite=|low_nonzero=|low_fnv1a64=|low_repeat_bit_mismatches=|output_finite=|output_nonzero=|output_fnv1a64=|output_repeat_bit_mismatches=|diagnostic_conclusion=|harness_status=)' \
    "$OUTPUT_DIR/diagnostic.log" >"$OUTPUT_DIR/summary.txt"
printf 'SM75 token-row arithmetic diagnostic complete: %s\n' "$OUTPUT_DIR"
