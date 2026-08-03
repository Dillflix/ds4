#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Build, validate, benchmark, disassemble, and optionally profile the bounded
SM75 packed-INT4 scaled-integer-dot experiment. No GGUF is opened.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  EXACT_CASES=256
  BENCH_REPEATS=128
  BENCH_LAUNCHES=20
  BENCH_ROUNDS=9
  STREAM_WEIGHT_CASES=16384
  PROFILE_REPEATS=128
  SKIP_BUILD=0
  RUN_SANITIZER=1       run memcheck when compute-sanitizer is installed
  RUN_NCU=1
  NCU_USE_SUDO=0
  CREATE_ARCHIVE=1
  INT4_EXPERIMENT_DIR=/absolute/output/directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

script_rel=speed-bench/cuda-sm75-int4-mma-experiment.sh
source_rel=tests/cuda_sm75_int4_mma.cu
binary_rel=tests/cuda_sm75_int4_mma
makefile_rel=Makefile

variants=(
    i8-standard
    i4-mixed-standard
    i4-mixed-group32-w
    i4-mixed-native-w
    i4-mixed-native-aw
    i4-u4-corrected-native-aw
)
kernels=(
    sm75_q4_i8_standard_kernel
    sm75_q4_i4_mixed_standard_kernel
    sm75_q4_i4_mixed_group32_w_kernel
    sm75_q4_i4_mixed_native_w_kernel
    sm75_q4_i4_mixed_native_aw_kernel
    sm75_q4_i4_u4_corrected_native_aw_kernel
)

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
EXACT_CASES=${EXACT_CASES:-256}
BENCH_REPEATS=${BENCH_REPEATS:-128}
BENCH_LAUNCHES=${BENCH_LAUNCHES:-20}
BENCH_ROUNDS=${BENCH_ROUNDS:-9}
STREAM_WEIGHT_CASES=${STREAM_WEIGHT_CASES:-16384}
PROFILE_REPEATS=${PROFILE_REPEATS:-128}
SKIP_BUILD=${SKIP_BUILD:-0}
RUN_SANITIZER=${RUN_SANITIZER:-1}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${INT4_EXPERIMENT_DIR:-$repo_dir/sm75-int4-mma-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be exactly sm_75"
[[ $OUTPUT_DIR == /* ]] || die "INT4_EXPERIMENT_DIR must be an absolute path"
for value_name in EXACT_CASES BENCH_REPEATS BENCH_LAUNCHES BENCH_ROUNDS \
        STREAM_WEIGHT_CASES PROFILE_REPEATS; do
    value=${!value_name}
    [[ $value =~ ^[1-9][0-9]*$ ]] || die "$value_name must be a positive integer"
done
(( STREAM_WEIGHT_CASES >= 8192 )) ||
    die "STREAM_WEIGHT_CASES must be at least 8192 to exceed RTX 8000 L2"
for value_name in SKIP_BUILD RUN_SANITIZER RUN_NCU NCU_USE_SUDO CREATE_ARCHIVE; do
    value=${!value_name}
    [[ $value == 0 || $value == 1 ]] || die "$value_name must be 0 or 1"
done

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
command -v cuobjdump >/dev/null 2>&1 || die "cuobjdump not found"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum not found"
command -v stat >/dev/null 2>&1 || die "stat not found"
command -v tar >/dev/null 2>&1 || die "tar not found"
[[ -x $script_rel ]] ||
    die "$script_rel is not executable; commit it with mode 100755"
compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] ||
    die "physical GPU $PROFILE_GPU has compute capability ${compute_cap:-unknown}; SM75 is required"

git_status_before=$(git status --short --untracked-files=all)
[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
ARCHIVE_PATH=$OUTPUT_DIR.tar.gz
[[ $CREATE_ARCHIVE == 0 || ! -e $ARCHIVE_PATH ]] ||
    die "archive path already exists: $ARCHIVE_PATH"
mkdir -p "$OUTPUT_DIR/ncu"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
ARCHIVE_PATH=$OUTPUT_DIR.tar.gz
mkdir -p "$OUTPUT_DIR/provenance" "$OUTPUT_DIR/sass-kernels"

current_phase=initialization
caught_signal=

take_output_ownership() {
    if [[ $NCU_USE_SUDO == 1 ]] && command -v sudo >/dev/null 2>&1; then
        sudo -n chown -R -- "$(id -u):$(id -g)" "$OUTPUT_DIR" \
            >/dev/null 2>&1 || true
    fi
}

write_run_status() {
    local state=$1
    local status=$2
    local archive_status=$3
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\nsignal=%s\narchive_status=%s\ndate_utc=%s\n' \
        "$state" "$status" "$current_phase" "${caught_signal:-none}" \
        "$archive_status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
}

finalize() {
    local status=$?
    trap - EXIT INT TERM HUP
    if [[ -d $OUTPUT_DIR ]]; then
        take_output_ownership
        local state=failed
        if [[ -n $caught_signal ]]; then
            state=interrupted
        elif (( status == 0 )) && [[ $current_phase == complete ]]; then
            state=complete
        fi
        if [[ $CREATE_ARCHIVE == 1 ]]; then
            local partial_archive=$ARCHIVE_PATH.partial.$$
            write_run_status "$state" "$status" created
            if [[ ! -e $ARCHIVE_PATH ]] &&
                    tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial_archive" \
                        "$(basename "$OUTPUT_DIR")" &&
                    mv -- "$partial_archive" "$ARCHIVE_PATH"; then
                printf 'Archive to return: %s\n' "$ARCHIVE_PATH"
            else
                rm -f -- "$partial_archive"
                status=1
                state=failed
                write_run_status "$state" "$status" failed
                printf 'error: could not create archive: %s\n' "$ARCHIVE_PATH" >&2
            fi
        else
            write_run_status "$state" "$status" disabled
        fi
    fi
    exit "$status"
}
trap finalize EXIT
trap 'caught_signal=INT; exit 130' INT
trap 'caught_signal=TERM; exit 143' TERM
trap 'caught_signal=HUP; exit 129' HUP

cp -- "$script_rel" "$source_rel" "$makefile_rel" "$OUTPUT_DIR/provenance/"
sha256sum "$script_rel" "$source_rel" "$makefile_rel" \
    >"$OUTPUT_DIR/provenance/source-sha256.txt"
git diff --no-ext-diff --binary HEAD -- \
    "$script_rel" "$source_rel" "$makefile_rel" \
    >"$OUTPUT_DIR/provenance/tracked-working-tree.patch" || true

gpu_state_log=$OUTPUT_DIR/gpu-state.log
gpu_state_error_log=$OUTPUT_DIR/gpu-state-errors.log
capture_gpu_state() {
    local label=$1
    {
        printf '\n[%s] date_utc=%s\n' "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        nvidia-smi -i "$PROFILE_GPU" \
            --query-gpu=index,pstate,temperature.gpu,clocks.current.graphics,clocks.current.sm,clocks.current.memory,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used,memory.free \
            --format=csv
    } >>"$gpu_state_log" 2>>"$gpu_state_error_log" ||
        printf '[%s] GPU telemetry query failed\n' "$label" \
            >>"$gpu_state_error_log"
}

{
    printf 'date_utc=%s\nrepo=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
    printf 'git_commit=%s\ngit_branch=%s\n' "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'profile_gpu=%s\ncuda_arch=%s\ncompute_capability=%s\n' \
        "$PROFILE_GPU" "$CUDA_ARCH" "$compute_cap"
    printf 'exact_cases=%s\nbench_repeats=%s\nbench_launches=%s\nbench_rounds=%s\n' \
        "$EXACT_CASES" "$BENCH_REPEATS" "$BENCH_LAUNCHES" "$BENCH_ROUNDS"
    printf 'hot_weight_cases=16\nstream_weight_cases=%s\nprofile_repeats=%s\n' \
        "$STREAM_WEIGHT_CASES" "$PROFILE_REPEATS"
    printf 'stream_min_weight_payload_bytes=%s\n' \
        "$((STREAM_WEIGHT_CASES * 1088))"
    printf 'full_model_loaded=false\nproduction_dispatch_changed=false\n'
    printf 'script_mode=%s\nsource_mode=%s\nmakefile_mode=%s\n' \
        "$(stat -c %a "$script_rel")" "$(stat -c %a "$source_rel")" \
        "$(stat -c %a "$makefile_rel")"
    printf '\n[gpu]\n'
    nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,name,pci.bus_id,memory.total,memory.free,ecc.mode.current,driver_version \
        --format=csv
    printf '\n[git status before output creation]\n%s\n' "${git_status_before:-clean}"
    printf '\n[source sha256]\n'; cat "$OUTPUT_DIR/provenance/source-sha256.txt"
    printf '\n[toolchain]\n'
    command -v nvcc || true
    nvcc --version 2>/dev/null || true
    cuobjdump --version 2>/dev/null || true
    ncu --version 2>/dev/null || true
    compute-sanitizer --version 2>/dev/null || true
} >"$OUTPUT_DIR/manifest.txt"

capture_gpu_state initialization

current_phase=build
if [[ $SKIP_BUILD == 0 ]]; then
    make -B "$binary_rel" CUDA_ARCH="$CUDA_ARCH" \
        >"$OUTPUT_DIR/build.log" 2>&1 || {
            tail -n 120 "$OUTPUT_DIR/build.log" >&2 || true
            die "SM75 INT4 harness build failed"
        }
else
    set +e
    make -q "$binary_rel" CUDA_ARCH="$CUDA_ARCH" \
        >"$OUTPUT_DIR/build.log" 2>&1
    make_query_rc=$?
    set -e
    case $make_query_rc in
        0) ;;
        1) die "SKIP_BUILD=1 rejected a stale harness; rerun with SKIP_BUILD=0" ;;
        *) die "could not validate the skipped harness build; rerun with SKIP_BUILD=0" ;;
    esac
    [[ $binary_rel -nt $source_rel && $binary_rel -nt $makefile_rel ]] ||
        die "SKIP_BUILD=1 rejected a harness older than its source or Makefile"
    printf 'build skipped after make -q and mtime validation\n' \
        >>"$OUTPUT_DIR/build.log"
fi
[[ -x $binary_rel ]] ||
    die "$binary_rel is missing; rerun with SKIP_BUILD=0"
sha256sum "$binary_rel" >"$OUTPUT_DIR/provenance/binary-sha256.txt"
{
    printf '\n[binary]\n'
    cat "$OUTPUT_DIR/provenance/binary-sha256.txt"
    stat -c 'mode=%a size=%s mtime=%y file=%n' "$binary_rel"
} >>"$OUTPUT_DIR/manifest.txt"

current_phase=sass
cuobjdump --list-elf "$binary_rel" >"$OUTPUT_DIR/elf-list.txt" 2>&1
grep -Eqi 'sm_?75' "$OUTPUT_DIR/elf-list.txt" ||
    die "harness binary does not contain an sm_75 cubin"
cuobjdump --dump-resource-usage "$binary_rel" \
    >"$OUTPUT_DIR/resource-usage.txt" 2>&1 || true
cuobjdump --dump-sass "$binary_rel" >"$OUTPUT_DIR/sass.txt" 2>&1
: >"$OUTPUT_DIR/sass-int4.txt"
: >"$OUTPUT_DIR/sass-relevant.txt"
printf 'variant,kernel,total_imma,imma_8816,imma_8832,s4_u4,u4_u4,lop3,prmt,shf,bfe,iadd3\n' \
    >"$OUTPUT_DIR/sass-summary.csv"
for i in "${!variants[@]}"; do
    variant=${variants[$i]}
    kernel=${kernels[$i]}
    kernel_sass=$OUTPUT_DIR/sass-kernels/$variant.sass.txt
    awk -v wanted="$kernel" '
        /Function : / {
            current = $0
            sub(/^.*Function :[[:space:]]*/, "", current)
            sub(/[[:space:]]*$/, "", current)
            emit = current == wanted
        }
        emit { print }
    ' "$OUTPUT_DIR/sass.txt" >"$kernel_sass"
    [[ -s $kernel_sass ]] || die "SASS contains no function section for $kernel"

    total_imma=$(grep -Ec 'IMMA' "$kernel_sass" || true)
    imma_8816=$(grep -Ec 'IMMA[^[:space:]]*8816|IMMA\.8816' "$kernel_sass" || true)
    imma_8832=$(grep -Ec 'IMMA[^[:space:]]*8832|IMMA\.8832' "$kernel_sass" || true)
    s4_u4=$(grep -Eic 'IMMA[^[:space:]]*S4[^[:space:]]*U4' "$kernel_sass" || true)
    u4_u4=$(grep -Eic 'IMMA[^[:space:]]*U4[^[:space:]]*U4' "$kernel_sass" || true)
    lop3=$(grep -Ec 'LOP3' "$kernel_sass" || true)
    prmt=$(grep -Ec 'PRMT' "$kernel_sass" || true)
    shf=$(grep -Ec 'SHF' "$kernel_sass" || true)
    bfe=$(grep -Ec 'BFE' "$kernel_sass" || true)
    iadd3=$(grep -Ec 'IADD3' "$kernel_sass" || true)
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$variant" "$kernel" "$total_imma" "$imma_8816" "$imma_8832" \
        "$s4_u4" "$u4_u4" "$lop3" "$prmt" "$shf" "$bfe" "$iadd3" \
        >>"$OUTPUT_DIR/sass-summary.csv"
    {
        printf '\n===== %s (%s) =====\n' "$variant" "$kernel"
        grep -E 'Function :|IMMA|LOP3|PRMT|SHF|BFE|IADD3' "$kernel_sass" || true
    } >>"$OUTPUT_DIR/sass-relevant.txt"
    if [[ $variant == i8-standard ]]; then
        (( imma_8816 > 0 )) || die "$kernel contains no 8x8x16 IMMA instruction"
    else
        (( imma_8832 > 0 )) || die "$kernel contains no 8x8x32 IMMA instruction"
        {
            printf '\n===== %s (%s) =====\n' "$variant" "$kernel"
            grep -E 'IMMA[^[:space:]]*8832|IMMA\.8832' "$kernel_sass" || true
        } >>"$OUTPUT_DIR/sass-int4.txt"
    fi
done

current_phase=correctness
capture_gpu_state pre-correctness
env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
    ./tests/cuda_sm75_int4_mma --device 0 --cases "$EXACT_CASES" \
        --correctness-only >"$OUTPUT_DIR/correctness.log" 2>&1 || {
            cat "$OUTPUT_DIR/correctness.log" >&2 || true
            die "packed-INT4 exactness failed"
        }
grep -q '^exact_status=ok$' "$OUTPUT_DIR/correctness.log" ||
    die "correctness harness did not report exact_status=ok"
capture_gpu_state post-correctness

current_phase=sanitizer
capture_gpu_state pre-sanitizer
if [[ $RUN_SANITIZER == 1 ]] && command -v compute-sanitizer >/dev/null 2>&1; then
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        compute-sanitizer --tool memcheck --error-exitcode 3 \
            ./tests/cuda_sm75_int4_mma --device 0 --cases 16 \
                --correctness-only >"$OUTPUT_DIR/memcheck.log" 2>&1 || {
                    tail -n 120 "$OUTPUT_DIR/memcheck.log" >&2 || true
                    die "compute-sanitizer memcheck failed"
                }
else
    printf 'skipped: RUN_SANITIZER=%s compute-sanitizer=%s\n' \
        "$RUN_SANITIZER" "$(command -v compute-sanitizer || true)" \
        >"$OUTPUT_DIR/memcheck.log"
fi
capture_gpu_state post-sanitizer

current_phase=benchmark-hot
capture_gpu_state pre-benchmark-hot
env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
    ./tests/cuda_sm75_int4_mma --device 0 --benchmark-only \
        --bench-cases 16 --rounds "$BENCH_ROUNDS" \
        --repeats "$BENCH_REPEATS" --launches "$BENCH_LAUNCHES" \
        >"$OUTPUT_DIR/benchmark-hot.log" 2>&1 || {
            cat "$OUTPUT_DIR/benchmark-hot.log" >&2 || true
            die "packed-INT4 hot-weight benchmark failed"
        }
awk '/^variant,total_ms,/{emit=1} emit && /,/{print}' \
    "$OUTPUT_DIR/benchmark-hot.log" >"$OUTPUT_DIR/benchmark-hot.csv"
[[ $(wc -l <"$OUTPUT_DIR/benchmark-hot.csv") -eq 7 ]] ||
    die "hot-weight benchmark CSV does not contain all six variants"
capture_gpu_state post-benchmark-hot

current_phase=benchmark-streaming
capture_gpu_state pre-benchmark-streaming
env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
    ./tests/cuda_sm75_int4_mma --device 0 --benchmark-only \
        --bench-cases "$STREAM_WEIGHT_CASES" --rounds "$BENCH_ROUNDS" \
        --repeats "$BENCH_REPEATS" --launches "$BENCH_LAUNCHES" \
        >"$OUTPUT_DIR/benchmark-streaming.log" 2>&1 || {
            cat "$OUTPUT_DIR/benchmark-streaming.log" >&2 || true
            die "packed-INT4 streaming-weight benchmark failed"
        }
awk '/^variant,total_ms,/{emit=1} emit && /,/{print}' \
    "$OUTPUT_DIR/benchmark-streaming.log" \
    >"$OUTPUT_DIR/benchmark-streaming.csv"
[[ $(wc -l <"$OUTPUT_DIR/benchmark-streaming.csv") -eq 7 ]] ||
    die "streaming-weight benchmark CSV does not contain all six variants"
capture_gpu_state post-benchmark-streaming

if [[ $RUN_NCU == 1 ]]; then
    current_phase=nsight
    command -v ncu >/dev/null 2>&1 || die "RUN_NCU=1 but ncu was not found"
    ncu_bin=$(command -v ncu)
    ncu_command=("$ncu_bin")
    if [[ $NCU_USE_SUDO == 1 ]]; then
        command -v sudo >/dev/null 2>&1 || die "sudo not found"
        sudo -v
        ncu_command=(sudo -E "$ncu_bin")
    fi
    required_metric_names=(
        gpu__time_duration.sum \
        launch__registers_per_thread \
        launch__shared_mem_per_block \
        launch__occupancy_limit_registers \
        sm__warps_active.avg.pct_of_peak_sustained_active \
        smsp__warps_eligible.avg.per_cycle_active \
        sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_elapsed \
        sm__inst_executed_pipe_ipa.avg.pct_of_peak_sustained_elapsed \
        l1tex__t_sector_hit_rate.pct \
        lts__t_sector_hit_rate.pct \
        lts__t_sectors_lookup_miss.sum \
        dram__bytes.sum.per_second \
        smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio \
        smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio)
    optional_metric_candidates=(
        dram__bytes.sum
        dram__bytes_read.sum
        dram__bytes_write.sum
        l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
        l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
        l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum
        l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum
        l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
        l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
        l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum
        lts__t_sectors_op_read.sum
        lts__t_sectors_op_write.sum
        lts__t_sectors_op_atom.sum
        lts__t_bytes.sum
        sm__inst_executed_pipe_ipa.sum
        sm__inst_executed_pipe_tensor.sum
        smsp__inst_executed.sum
        smsp__inst_executed_pipe_ipa.sum
        smsp__inst_executed_pipe_tensor.sum
        smsp__sass_thread_inst_executed_op_integer_pred_on.sum
        smsp__sass_thread_inst_executed_op_memory_pred_on.sum
    )

    current_phase=nsight-metric-query
    available_metrics_raw=$OUTPUT_DIR/ncu/available-metrics.raw.txt
    available_metrics=$OUTPUT_DIR/ncu/available-metric-names.txt
    available_metrics_log=$OUTPUT_DIR/ncu/available-metrics-query.log
    query_args=(--config-file off --devices 0 --query-metrics)
    if "$ncu_bin" --help 2>/dev/null | grep -q -- '--query-metrics-mode'; then
        query_args+=(--query-metrics-mode all)
    fi
    metric_query_rc=0
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        "$ncu_bin" "${query_args[@]}" >"$available_metrics_raw" \
            2>"$available_metrics_log" || metric_query_rc=$?
    if (( metric_query_rc == 0 )); then
        grep -Eo '[[:alnum:]_]+__[[:alnum:]_.]+' "$available_metrics_raw" |
            sort -u >"$available_metrics" || true
    else
        : >"$available_metrics"
        printf 'optional metric discovery failed with exit %s; using required metrics only\n' \
            "$metric_query_rc" >>"$available_metrics_log"
    fi

    metric_names=("${required_metric_names[@]}")
    : >"$OUTPUT_DIR/ncu/optional-metrics-selected.txt"
    if [[ -s $available_metrics ]]; then
        for candidate in "${optional_metric_candidates[@]}"; do
            if grep -Fxq -- "$candidate" "$available_metrics"; then
                metric_names+=("$candidate")
                printf '%s\n' "$candidate" \
                    >>"$OUTPUT_DIR/ncu/optional-metrics-selected.txt"
            fi
        done
    fi
    printf '%s\n' "${required_metric_names[@]}" \
        >"$OUTPUT_DIR/ncu/required-metrics.txt"
    printf '%s\n' "${metric_names[@]}" \
        >"$OUTPUT_DIR/ncu/selected-metrics.txt"
    metrics=$(IFS=,; printf '%s' "${metric_names[*]}")

    current_phase=nsight
    capture_gpu_state pre-nsight
    profile_case_args=()
    if grep -Fq -- '--bench-cases' "$source_rel"; then
        profile_case_args=(--bench-cases "$STREAM_WEIGHT_CASES")
    fi
    printf 'profile_uses_stream_weight_cases=%s\n' \
        "$([[ ${#profile_case_args[@]} -gt 0 ]] && printf true || printf false)" \
        >"$OUTPUT_DIR/ncu/profile-case-selection.txt"
    for i in "${!variants[@]}"; do
        variant=${variants[$i]}
        kernel=${kernels[$i]}
        base="$OUTPUT_DIR/ncu/$variant"
        printf 'Nsight Compute: %s...\n' "$variant"
        rc=0
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            "${ncu_command[@]}" --config-file off \
                --target-processes application-only --devices 0 \
                --kernel-name-base function --kernel-name "$kernel" \
                --launch-count 1 --replay-mode kernel --cache-control none \
                --clock-control none --force-overwrite --export "$base" \
                --metrics "$metrics" --disable-extra-suffixes \
                ./tests/cuda_sm75_int4_mma --device 0 --profile "$variant" \
                    --repeats "$PROFILE_REPEATS" "${profile_case_args[@]}" \
                >"$base.log" 2>&1 || rc=$?
        take_output_ownership
        if (( rc != 0 )); then
            tail -n 120 "$base.log" >&2 || true
            die "Nsight Compute failed for $variant (exit $rc)"
        fi
        grep -Eq '==ERROR==|No kernels were profiled|Failed to (profile|create report)' \
            "$base.log" && die "Nsight Compute captured no valid kernel for $variant"
        [[ -s $base.ncu-rep ]] || die "missing Nsight report for $variant"
        "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
            >"$base.csv" 2>"$base-import.log" ||
            die "could not import Nsight report for $variant"
        grep -q "$kernel" "$base.csv" ||
            die "Nsight report does not contain expected kernel $kernel"
        capture_gpu_state "post-nsight-$variant"
    done
fi

current_phase=final-telemetry
capture_gpu_state final
current_phase=complete
printf 'SM75 packed-INT4 experiment complete: %s\n' "$OUTPUT_DIR"
