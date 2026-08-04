#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Build, validate, sanitize, and position-balance the production SM75 Q4
wide-CTA specializations without opening a GGUF.

Optional environment:
  PROFILE_GPU=0                  physical SM75 GPU exposed as logical device 0
  CUDA_ARCH=sm_75                must remain exactly sm_75
  TIMING_POSITION_CYCLES=2       complete Latin position-balance cycles
  TIMING_REPEATS=10              production calls inside each timed process
  SANITIZER_TIMEOUT_SECONDS=900  timeout for each sanitizer tool
  WIDE_CTA_DIR=/absolute/path    new evidence directory

The run always rebuilds with production fast-math flags and verbose PTXAS,
requires memcheck, racecheck, and synccheck, and archives partial evidence on
success, failure, or interruption. It never changes clocks, git state, or
production defaults.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

script_rel=speed-bench/cuda-sm75-wide-cta.sh
parser_rel=speed-bench/summarize-sm75-wide-cta.py
resource_rel=tests/cuda_sm75_wide_cta_resources
correctness_rel=tests/cuda_sm75_wide_cta_correctness
profile_rel=tests/cuda_sm75_profile_harness

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
TIMING_POSITION_CYCLES=${TIMING_POSITION_CYCLES:-2}
TIMING_REPEATS=${TIMING_REPEATS:-10}
SANITIZER_TIMEOUT_SECONDS=${SANITIZER_TIMEOUT_SECONDS:-900}
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${WIDE_CTA_DIR:-$repo_dir/speed-bench/local-runs/sm75-wide-cta-$run_stamp}
while [[ $OUTPUT_DIR != / && $OUTPUT_DIR == */ ]]; do OUTPUT_DIR=${OUTPUT_DIR%/}; done

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a GPU index"
[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be exactly sm_75"
[[ $TIMING_POSITION_CYCLES =~ ^[1-9][0-9]*$ ]] ||
    die "TIMING_POSITION_CYCLES must be positive"
[[ $TIMING_REPEATS =~ ^[1-9][0-9]*$ ]] ||
    die "TIMING_REPEATS must be positive"
(( 10#$TIMING_REPEATS <= 100 )) || die "TIMING_REPEATS must not exceed 100"
[[ $SANITIZER_TIMEOUT_SECONDS =~ ^[1-9][0-9]*$ ]] ||
    die "SANITIZER_TIMEOUT_SECONDS must be positive"
[[ $OUTPUT_DIR == /* ]] || die "WIDE_CTA_DIR must be an absolute path"
[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
[[ ! -e $OUTPUT_DIR.tar.gz ]] || die "archive path already exists: $OUTPUT_DIR.tar.gz"

mkdir -p "$OUTPUT_DIR"/{provenance,resources,correctness,sanitizer,timing/logs}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
ARCHIVE_PATH=$OUTPUT_DIR.tar.gz
current_phase=initialization
caught_signal=none

write_status() {
    local state=$1 status=$2 archive_status=$3
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\nsignal=%s\narchive_status=%s\ndate_utc=%s\n' \
        "$state" "$status" "$current_phase" "$caught_signal" \
        "$archive_status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
}

finalize() {
    local status=$?
    local final_status=$status
    local state=failed archive_status=failed
    local partial=$ARCHIVE_PATH.partial.$$
    trap - EXIT INT TERM HUP
    if (( status == 0 )) && [[ $current_phase == complete ]]; then state=finished; fi
    # A successful archive must be self-describing.  Write the state that the
    # archive will have before creating it; if creation fails, correct the
    # retained directory below and return failure.
    write_status "$state" "$status" finished
    if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
            "$(basename "$OUTPUT_DIR")" && mv -- "$partial" "$ARCHIVE_PATH"; then
        archive_status=finished
        printf 'archive=%s\n' "$ARCHIVE_PATH"
    else
        final_status=1
        printf 'error: could not create archive: %s\n' "$ARCHIVE_PATH" >&2
    fi
    write_status "$state" "$final_status" "$archive_status"
    exit "$final_status"
}

on_signal() {
    caught_signal=$1
    current_phase="interrupted-$1"
    exit "$2"
}

trap finalize EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

current_phase=preflight
for tool in make nproc python3 cuobjdump c++filt nvidia-smi sha256sum \
        compute-sanitizer timeout tar mv git awk cmp date grep tail tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
for source in "$script_rel" "$parser_rel" ds4_cuda.cu Makefile \
        .github/workflows/cuda-quantizer-build.yml .gitignore \
        speed-bench/README.md speed-bench/cuda-sm75-production-scalar.sh \
        tests/cuda_long_context_smoke.c \
        tests/cuda_sm75_wide_cta_resources.c \
        tests/cuda_sm75_wide_cta_correctness.c \
        tests/cuda_sm75_profile_harness.c; do
    [[ -f $source ]] || die "required source not found: $source"
done

compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] ||
    die "GPU $PROFILE_GPU is not SM75 (${compute_cap:-unknown})"
free_mib=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $free_mib =~ ^[0-9]+$ ]] || die "could not read free VRAM"
(( free_mib >= 4096 )) || die "at least 4096 MiB free VRAM is required"

evidence_nvccflags="-O3 -g -lineinfo --use_fast_math -arch=sm_75 -Xcompiler -march=native -Xcompiler -pthread -Xptxas -v"
{
    printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'git_commit=%s\ngit_branch=%s\n' \
        "$(git rev-parse HEAD 2>/dev/null || printf unknown)" \
        "$(git branch --show-current 2>/dev/null || printf unknown)"
    printf 'cuda_arch=%s\nprofile_gpu_physical=%s\ncompute_capability=%s\n' \
        "$CUDA_ARCH" "$PROFILE_GPU" "$compute_cap"
    printf 'free_mib_at_preflight=%s\ntiming_position_cycles=%s\ntiming_repeats=%s\n' \
        "$free_mib" "$TIMING_POSITION_CYCLES" "$TIMING_REPEATS"
    printf 'sanitizer_timeout_seconds=%s\n' "$SANITIZER_TIMEOUT_SECONDS"
    printf 'gate_widths=256,384,512\ndown_widths=256,384,512,640\n'
    printf 'timing_order=latin-position-balanced\n'
    printf 'timing_isolation=selected-family-width-only;other-family-width-256\n'
    printf 'nvccflags=%s\n' "$evidence_nvccflags"
    printf '\n[git status]\n'; git status --short 2>/dev/null || true
    printf '\n[gpu]\n'; nvidia-smi -i "$PROFILE_GPU" 2>&1
    printf '\n[nvcc]\n'; "${NVCC:-/usr/local/cuda/bin/nvcc}" --version 2>&1
    printf '\n[cuobjdump]\n'; cuobjdump --version 2>&1
    printf '\n[compute-sanitizer]\n'; compute-sanitizer --version 2>&1
} >"$OUTPUT_DIR/manifest.txt"

sha256sum "$script_rel" "$parser_rel" ds4_cuda.cu Makefile \
    .github/workflows/cuda-quantizer-build.yml .gitignore \
    speed-bench/README.md speed-bench/cuda-sm75-production-scalar.sh \
    tests/cuda_long_context_smoke.c \
    tests/cuda_sm75_wide_cta_resources.c \
    tests/cuda_sm75_wide_cta_correctness.c \
    tests/cuda_sm75_profile_harness.c \
    >"$OUTPUT_DIR/provenance/source-sha256.txt"
git diff --binary HEAD -- "$script_rel" "$parser_rel" ds4_cuda.cu Makefile \
    .github/workflows/cuda-quantizer-build.yml .gitignore \
    speed-bench/README.md speed-bench/cuda-sm75-production-scalar.sh \
    tests/cuda_long_context_smoke.c \
    tests/cuda_sm75_wide_cta_resources.c \
    tests/cuda_sm75_wide_cta_correctness.c \
    tests/cuda_sm75_profile_harness.c \
    >"$OUTPUT_DIR/provenance/tracked-working-tree.patch" || true

current_phase=build
set +e
make -B -j"$(nproc)" "$resource_rel" "$correctness_rel" "$profile_rel" \
    CUDA_ARCH="$CUDA_ARCH" NVCCFLAGS="$evidence_nvccflags" \
    >"$OUTPUT_DIR/build.log" 2>&1
build_rc=$?
set -e
if (( build_rc != 0 )); then
    tail -n 160 "$OUTPUT_DIR/build.log" >&2 || true
    die "forced SM75 evidence build failed"
fi
for binary in "$resource_rel" "$correctness_rel" "$profile_rel"; do
    [[ -x $binary ]] || die "built binary is missing or not executable: $binary"
done
sha256sum "$resource_rel" "$correctness_rel" "$profile_rel" \
    >"$OUTPUT_DIR/provenance/binary-sha256.txt"

current_phase=resource-runtime
CUDA_VISIBLE_DEVICES="$PROFILE_GPU" "./$resource_rel" \
    >"$OUTPUT_DIR/resources/runtime-resources.csv" \
    2>"$OUTPUT_DIR/resources/runtime-resources.log" || {
        tail -n 120 "$OUTPUT_DIR/resources/runtime-resources.log" >&2 || true
        die "runtime resource matrix failed"
    }

current_phase=resource-static
cuobjdump --list-elf "$resource_rel" \
    >"$OUTPUT_DIR/resources/elf-list.txt" 2>&1
grep -Eq '\.sm_75\.cubin([[:space:]]|$)' \
    "$OUTPUT_DIR/resources/elf-list.txt" ||
    die "resource binary does not contain an sm_75 cubin"
cuobjdump --dump-resource-usage "$resource_rel" \
    >"$OUTPUT_DIR/resources/cuobjdump-resource-usage.txt" 2>&1
cuobjdump --dump-sass "$resource_rel" \
    >"$OUTPUT_DIR/resources/sass.txt" 2>&1

PYTHONDONTWRITEBYTECODE=1 python3 "$parser_rel" \
    --runtime-resources "$OUTPUT_DIR/resources/runtime-resources.csv" \
    --ptxas-log "$OUTPUT_DIR/build.log" \
    --sass "$OUTPUT_DIR/resources/sass.txt" \
    --resource-output "$OUTPUT_DIR/resources/resource-merged.csv" \
    >"$OUTPUT_DIR/resources/parser.log" 2>&1 || {
        tail -n 160 "$OUTPUT_DIR/resources/parser.log" >&2 || true
        die "resource/PTXAS/SASS merge failed"
    }
grep -Fxq 'resource_status=ok' "$OUTPUT_DIR/resources/parser.log" ||
    die "resource parser omitted its success marker"

verify_correctness_evidence() {
    local log=$1 family path row_span width out_dim write_aux
    local -a widths
    local marker_count
    marker_count=$(grep -Ec '^wide-cta-nonvacuous: ' "$log" || true)
    [[ $marker_count == 6 ]] || {
        printf 'error: %s contains %s non-vacuity records; expected exactly 6\n' \
            "$log" "$marker_count" >&2
        return 1
    }
    for out_dim in 504 520 4096; do
        for write_aux in 0 1; do
            marker_count=$(grep -Ec "^wide-cta-nonvacuous: dim=${out_dim} write_aux=${write_aux} resident_pairs=63 populations=1\\|3\\|4\\|7\\|8\\|9\\|15\\|16$" "$log" || true)
            [[ $marker_count == 1 ]] || {
                printf 'error: %s has %s non-vacuity records for dim=%s write_aux=%s; expected exactly one\n' \
                    "$log" "$marker_count" "$out_dim" "$write_aux" >&2
                return 1
            }
        done
    done
    marker_count=$(grep -Ec '^ds4: sm75-scalar-dispatch ' "$log" || true)
    [[ $marker_count == 21 ]] || {
        printf 'error: %s contains %s scalar-dispatch markers; expected exactly 21\n' \
            "$log" "$marker_count" >&2
        return 1
    }
    for family in gate down; do
        if [[ $family == gate ]]; then
            path=q4-gate-tile8
            widths=(256 384 512)
        else
            path=q4-down-tile16
            widths=(256 384 512 640)
        fi
        for row_span in 512 1024 2048; do
            for width in "${widths[@]}"; do
                marker_count=$(grep -Ec "^ds4: sm75-scalar-dispatch path=${path} scalar=1 .*row_span=${row_span} cta_threads=${width} stage_rows=0$" "$log" || true)
                [[ $marker_count == 1 ]] || {
                    printf 'error: %s has %s markers for path=%s row_span=%s cta_threads=%s; expected exactly one\n' \
                        "$log" "$marker_count" "$path" "$row_span" "$width" >&2
                    return 1
                }
            done
        done
    done
}

current_phase=correctness
CUDA_VISIBLE_DEVICES="$PROFILE_GPU" "./$correctness_rel" \
    >"$OUTPUT_DIR/correctness/correctness.log" 2>&1 || {
        tail -n 160 "$OUTPUT_DIR/correctness/correctness.log" >&2 || true
        die "SM75 wide-CTA correctness failed"
    }
grep -Fxq 'sm75 wide-cta correctness: OK' \
    "$OUTPUT_DIR/correctness/correctness.log" ||
    die "correctness harness omitted its success marker"
verify_correctness_evidence "$OUTPUT_DIR/correctness/correctness.log" ||
    die "correctness harness did not prove every literal specialization launch"
for width in 256 384 512; do
    grep -Eq "gate_cta=${width}([^0-9]|$)" \
        "$OUTPUT_DIR/correctness/correctness.log" ||
        die "correctness evidence omitted gate width $width"
done
for width in 256 384 512 640; do
    grep -Eq "down_cta=${width}([^0-9]|$)" \
        "$OUTPUT_DIR/correctness/correctness.log" ||
        die "correctness evidence omitted down width $width"
done

current_phase=sanitizer
printf 'tool,covered_gate_widths,covered_down_widths,exit_status,success_marker,dispatch_matrix,clean_summary,status,log\n' \
    >"$OUTPUT_DIR/sanitizer/status.csv"
sanitizer_failed=0
run_sanitizer() {
    local tool=$1 error_exitcode=$2
    shift 2
    local log="$OUTPUT_DIR/sanitizer/$tool.log"
    local rc marker=0 dispatch=0 summary=0 status=fail
    set +e
    timeout --signal=INT --kill-after=30s "${SANITIZER_TIMEOUT_SECONDS}s" \
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        compute-sanitizer --tool "$tool" --error-exitcode "$error_exitcode" \
            "$@" "./$correctness_rel" >"$log" 2>&1
    rc=$?
    set -e
    grep -Fxq 'sm75 wide-cta correctness: OK' "$log" && marker=1
    verify_correctness_evidence "$log" && dispatch=1
    if grep -Eiq 'ERROR SUMMARY:[[:space:]]*0 errors|RACECHECK SUMMARY:[[:space:]]*0 hazards' "$log"; then
        summary=1
    fi
    if (( rc == 0 && marker == 1 && dispatch == 1 && summary == 1 )) &&
            ! grep -Eiq 'Internal Sanitizer Error|Barrier error detected|Race reported|Potential (RAW|WAR|WAW) hazard' "$log"; then
        status=pass
    fi
    printf '%s,"256|384|512","256|384|512|640",%s,%s,%s,%s,%s,%s\n' \
        "$tool" "$rc" "$marker" "$dispatch" "$summary" "$status" "$log" \
        >>"$OUTPUT_DIR/sanitizer/status.csv"
    if [[ $status != pass ]]; then
        tail -n 160 "$log" >&2 || true
        return 1
    fi
}

run_sanitizer memcheck 86 || sanitizer_failed=1
run_sanitizer racecheck 87 \
    --racecheck-detect-level warn \
    --racecheck-report analysis \
    --racecheck-deadlock-timeout 30000 || sanitizer_failed=1
run_sanitizer synccheck 88 || sanitizer_failed=1
(( sanitizer_failed == 0 )) || die "one or more sanitizer gates failed"

current_phase=timing
timing_samples="$OUTPUT_DIR/timing/samples.csv"
printf 'family,scenario,cycle,round,sample_slot,width,timed_repeats,timed_total_ms,timed_per_call_ms,status\n' \
    >"$timing_samples"
printf 'family,scenario,point,timestamp,gpu_clock_mhz,memory_clock_mhz,temperature_c,power_w\n' \
    >"$OUTPUT_DIR/timing/telemetry.csv"

capture_telemetry() {
    local family=$1 scenario=$2 point=$3
    local row
    row=$(nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=timestamp,clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw \
        --format=csv,noheader,nounits 2>/dev/null) || return 1
    printf '%s,%s,%s,%s\n' "$family" "$scenario" "$point" "$row" \
        >>"$OUTPUT_DIR/timing/telemetry.csv"
}

log_value() {
    local key=$1 log=$2
    awk -F= -v wanted="$key" '$1 == wanted { print $2; exit }' "$log"
}

run_timing_sample() {
    local family=$1 scenario=$2 cycle=$3 round=$4 slot=$5 width=$6
    local target gate_width down_width dispatch_path
    case "$family" in
        gate)
            target=q4-gate; gate_width=$width; down_width=256
            dispatch_path=q4-gate-tile8
            ;;
        down)
            target=q4-down; gate_width=256; down_width=$width
            dispatch_path=q4-down-tile16
            ;;
        *) die "unknown timing family: $family" ;;
    esac
    local log="$OUTPUT_DIR/timing/logs/$family-$scenario-c$cycle-r$round-s$slot-w$width.log"
    local -a command=(env
        -u DS4_CUDA_MOE_Q4_GATE_SCALAR_CTA_SM75
        -u DS4_CUDA_MOE_Q4_DOWN_SCALAR_CTA_SM75
        -u DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75
        -u DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75
        -u DS4_PROFILE_Q4_GATE_CTA_THREADS
        -u DS4_PROFILE_Q4_DOWN_CTA_THREADS
        CUDA_VISIBLE_DEVICES="$PROFILE_GPU"
        DS4_PROFILE_SCALAR_TARGET="$target"
        DS4_PROFILE_SCALAR=1
        DS4_PROFILE_REPEATS="$TIMING_REPEATS"
        DS4_PROFILE_Q4_GATE_CTA_THREADS="$gate_width"
        DS4_PROFILE_Q4_DOWN_CTA_THREADS="$down_width"
        DS4_CUDA_MOE_SCALAR_AUDIT=1)
    "${command[@]}" "./$profile_rel" "q4-$scenario" >"$log" 2>&1 || {
        tail -n 140 "$log" >&2 || true
        die "timing process failed for $family/$scenario/width$width"
    }
    grep -Fxq 'harness_status=ok' "$log" ||
        die "timing harness omitted success for $family/$scenario/width$width"
    grep -Fxq "scalar_target=$target" "$log" ||
        die "timing harness selected the wrong scalar target"
    grep -Fxq 'scalar_enabled=1' "$log" ||
        die "timing harness did not enable the scalar candidate"
    grep -Eq "sm75-scalar-dispatch path=${dispatch_path} scalar=1 .*row_span=512 cta_threads=${width}([^0-9]|$)" "$log" ||
        die "timing log did not prove $family CTA width $width"
    local repeats total per_call
    repeats=$(log_value timed_repeats "$log")
    total=$(log_value timed_total_ms "$log")
    per_call=$(log_value timed_per_call_ms "$log")
    [[ $repeats == "$TIMING_REPEATS" ]] ||
        die "timing harness recorded unexpected repeat count"
    [[ $total =~ ^[0-9]+([.][0-9]+)?$ && $per_call =~ ^[0-9]+([.][0-9]+)?$ ]] ||
        die "timing harness recorded invalid durations"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,ok\n' \
        "$family" "$scenario" "$cycle" "$round" "$slot" "$width" \
        "$repeats" "$total" "$per_call" >>"$timing_samples"
}

for family in gate down; do
    if [[ $family == gate ]]; then
        widths=(256 384 512)
    else
        widths=(256 384 512 640)
    fi
    width_count=${#widths[@]}
    for scenario in early late; do
        capture_telemetry "$family" "$scenario" before ||
            die "could not capture pre-timing telemetry"
        round=0
        for ((cycle=1; cycle<=TIMING_POSITION_CYCLES; cycle++)); do
            for ((rotation=0; rotation<width_count; rotation++)); do
                round=$((round + 1))
                for ((slot_index=0; slot_index<width_count; slot_index++)); do
                    index=$(((slot_index + rotation + cycle - 1) % width_count))
                    width=${widths[$index]}
                    run_timing_sample "$family" "$scenario" "$cycle" \
                        "$round" "$((slot_index + 1))" "$width"
                done
            done
        done
        capture_telemetry "$family" "$scenario" after ||
            die "could not capture post-timing telemetry"
    done
done

current_phase=timing-summary
PYTHONDONTWRITEBYTECODE=1 python3 "$parser_rel" \
    --runtime-resources "$OUTPUT_DIR/resources/runtime-resources.csv" \
    --ptxas-log "$OUTPUT_DIR/build.log" \
    --sass "$OUTPUT_DIR/resources/sass.txt" \
    --resource-output "$OUTPUT_DIR/resources/resource-merged-final.csv" \
    --timing-samples "$timing_samples" \
    --timing-output "$OUTPUT_DIR/timing/summary.csv" \
    --position-output "$OUTPUT_DIR/timing/position-balance.csv" \
    >"$OUTPUT_DIR/timing/parser.log" 2>&1 || {
        tail -n 160 "$OUTPUT_DIR/timing/parser.log" >&2 || true
        die "timing summary validation failed"
    }
grep -Fxq 'timing_status=ok' "$OUTPUT_DIR/timing/parser.log" ||
    die "timing parser omitted its success marker"
cmp -s "$OUTPUT_DIR/resources/resource-merged.csv" \
    "$OUTPUT_DIR/resources/resource-merged-final.csv" ||
    die "resource evidence changed between validation passes"

current_phase=final-validation
for required in \
        "$OUTPUT_DIR/resources/resource-merged.csv" \
        "$OUTPUT_DIR/correctness/correctness.log" \
        "$OUTPUT_DIR/sanitizer/status.csv" \
        "$OUTPUT_DIR/timing/samples.csv" \
        "$OUTPUT_DIR/timing/summary.csv" \
        "$OUTPUT_DIR/timing/position-balance.csv"; do
    [[ -s $required ]] || die "required evidence is missing or empty: $required"
done
printf 'wide_cta_evidence_status=ok\nproduction_default_changed=false\n' \
    >"$OUTPUT_DIR/acceptance.txt"
current_phase=complete
printf 'SM75 wide-CTA evidence complete: %s\n' "$OUTPUT_DIR"
