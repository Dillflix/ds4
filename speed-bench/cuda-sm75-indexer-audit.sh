#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Audit the production-shape DeepSeek-V4 indexer on one SM75 GPU without opening
a GGUF. The harness uses the final 512-token chunk at 32K (8192 compressed
rows), validates every score bit against the shipping WMMA128 path, and checks
the exact ordered top-512 set.

The pass measures:
  * existing score tiles 128, 64, 32, and 16;
  * monolithic-8192 versus existing chunked-tree top-k;
  * optional Nsight Compute captures for each score tile and both top-k stages.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  TIMING_REPEATS=10
  RUN_NCU=1
  NCU_USE_SUDO=0
  NCU_SET=focused          focused, targeted, or full
  SKIP_BUILD=0
  RESUME=0                reuse timing/score NCU results in INDEXER_AUDIT_DIR
  CREATE_ARCHIVE=1
  INDEXER_AUDIT_DIR=/absolute/output/directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
TIMING_REPEATS=${TIMING_REPEATS:-10}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
NCU_SET=${NCU_SET:-focused}
SKIP_BUILD=${SKIP_BUILD:-0}
RESUME=${RESUME:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${INDEXER_AUDIT_DIR:-$repo_dir/sm75-indexer-audit-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $TIMING_REPEATS =~ ^[1-9][0-9]*$ ]] || die "TIMING_REPEATS must be positive"
(( TIMING_REPEATS <= 100 )) || die "TIMING_REPEATS must not exceed 100"
[[ $RUN_NCU == 0 || $RUN_NCU == 1 ]] || die "RUN_NCU must be 0 or 1"
[[ $NCU_USE_SUDO == 0 || $NCU_USE_SUDO == 1 ]] || die "NCU_USE_SUDO must be 0 or 1"
[[ $SKIP_BUILD == 0 || $SKIP_BUILD == 1 ]] || die "SKIP_BUILD must be 0 or 1"
[[ $RESUME == 0 || $RESUME == 1 ]] || die "RESUME must be 0 or 1"
[[ $CREATE_ARCHIVE == 0 || $CREATE_ARCHIVE == 1 ]] || die "CREATE_ARCHIVE must be 0 or 1"
[[ $NCU_SET == focused || $NCU_SET == targeted || $NCU_SET == full ]] ||
    die "NCU_SET must be focused, targeted, or full"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
command -v python3 >/dev/null 2>&1 || die "python3 not found"
command -v tar >/dev/null 2>&1 || die "tar not found"

compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] ||
    die "physical GPU $PROFILE_GPU has compute capability ${compute_cap:-unknown}; SM75 is required"
free_mib=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $free_mib =~ ^[0-9]+$ ]] || die "could not read free VRAM for GPU $PROFILE_GPU"
(( free_mib >= 4096 )) ||
    die "GPU $PROFILE_GPU has only ${free_mib} MiB free; at least 4096 MiB is required"

if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" tests/cuda_sm75_profile_harness CUDA_ARCH="$CUDA_ARCH"
else
    make -q tests/cuda_sm75_profile_harness CUDA_ARCH="$CUDA_ARCH" ||
        die "SKIP_BUILD=1 found a stale profile harness"
fi
[[ -x tests/cuda_sm75_profile_harness ]] ||
    die "profile harness is missing; rerun with SKIP_BUILD=0"

if [[ $RESUME == 1 ]]; then
    [[ -n ${INDEXER_AUDIT_DIR:-} ]] ||
        die "RESUME=1 requires an explicit INDEXER_AUDIT_DIR"
    [[ -d $OUTPUT_DIR ]] || die "resume directory does not exist: $OUTPUT_DIR"
    for required in score-tile-timing.csv topk-timing.csv \
            timing/score-tile-128.log timing/score-tile-64.log \
            timing/score-tile-32.log timing/score-tile-16.log \
            ncu/score-wmma128.csv ncu/score-wmma64.csv \
            ncu/score-wmma32.csv ncu/score-wmma16.csv; do
        [[ -s $OUTPUT_DIR/$required ]] ||
            die "resume evidence is missing or empty: $OUTPUT_DIR/$required"
    done
else
    [[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/timing" "$OUTPUT_DIR/ncu"
fi
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

current_phase=initialization
finalize() {
    local status=$?
    trap - EXIT
    if [[ -d $OUTPUT_DIR ]]; then
        printf 'state=finished\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
            "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >"$OUTPUT_DIR/run-status.txt"
        if [[ $CREATE_ARCHIVE == 1 ]]; then
            local archive="$OUTPUT_DIR.tar.gz"
            if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$archive" \
                    "$(basename "$OUTPUT_DIR")"; then
                printf 'Archive to return: %s\n' "$archive"
            else
                status=1
            fi
        fi
    fi
    exit "$status"
}
trap finalize EXIT

if [[ $RESUME == 0 ]]; then
    {
        printf 'date_utc=%s\nrepo=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
        printf 'git_commit=%s\ngit_branch=%s\n' "$(git rev-parse HEAD)" "$(git branch --show-current)"
        printf 'profile_gpu_physical=%s\nprofile_gpu_logical=0\n' "$PROFILE_GPU"
        printf 'cuda_arch=%s\ncompute_capability=%s\nfree_mib_at_preflight=%s\n' \
            "$CUDA_ARCH" "$compute_cap" "$free_mib"
        printf 'timing_repeats=%s\nrun_ncu=%s\nncu_set=%s\n' \
            "$TIMING_REPEATS" "$RUN_NCU" "$NCU_SET"
        printf 'shape=512x64x128-by-8192\nposition=32256\ntop_k=512\n'
        printf 'full_model_loaded=false\nscore_reference=shipping-wmma128\n'
        printf '\n[gpu]\n'
        nvidia-smi -i "$PROFILE_GPU" \
            --query-gpu=index,name,pci.bus_id,memory.total,memory.free,ecc.mode.current,driver_version \
            --format=csv
        printf '\n[git status]\n'; git status --short
    } >"$OUTPUT_DIR/manifest.txt"
else
    printf 'date_utc=%s\nresume=true\ngit_commit=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        >"$OUTPUT_DIR/resume-manifest.txt"
fi

value_from_log() {
    local key=$1 file=$2
    awk -F= -v key="$key" '$1 == key { print $2; found=1; exit } END { if (!found) exit 1 }' "$file"
}

run_timing() {
    local tile=$1 topk=$2 label=$3
    local log="$OUTPUT_DIR/timing/$label.log"
    printf 'Timing and validating tile=%s topk=%s...\n' "$tile" "$topk"
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        DS4_PROFILE_INDEXER_TILE="$tile" \
        DS4_PROFILE_INDEXER_TOPK="$topk" \
        DS4_PROFILE_REPEATS="$TIMING_REPEATS" \
        ./tests/cuda_sm75_profile_harness indexer-32k >"$log" 2>&1 || {
            tail -n 120 "$log" >&2 || true
            die "indexer harness failed for tile=$tile topk=$topk"
        }
    grep -q '^score_validation=bit-exact$' "$log" ||
        die "score validation missing for tile=$tile topk=$topk"
    grep -q '^topk_validation=exact-order-and-set$' "$log" ||
        die "top-k validation missing for tile=$tile topk=$topk"
    grep -q '^harness_status=ok$' "$log" ||
        die "harness success marker missing for tile=$tile topk=$topk"
}

if [[ $RESUME == 0 ]]; then
    current_phase=timing-score-tiles
    printf 'tile,score_ms,speedup_vs_128\n' >"$OUTPUT_DIR/score-tile-timing.csv"
    for tile in 128 64 32 16; do
        run_timing "$tile" monolithic "score-tile-$tile"
    done
    base_score_ms=$(value_from_log score_timed_per_call_ms \
        "$OUTPUT_DIR/timing/score-tile-128.log")
    for tile in 128 64 32 16; do
        score_ms=$(value_from_log score_timed_per_call_ms \
            "$OUTPUT_DIR/timing/score-tile-$tile.log")
        speedup=$(awk -v base="$base_score_ms" -v value="$score_ms" \
            'BEGIN { printf "%.6f", base / value }')
        printf '%s,%s,%s\n' "$tile" "$score_ms" "$speedup" \
            >>"$OUTPUT_DIR/score-tile-timing.csv"
    done

    current_phase=timing-topk
    run_timing 128 chunked topk-chunked
    printf 'path,topk_ms,speedup_vs_monolithic\n' >"$OUTPUT_DIR/topk-timing.csv"
    base_topk_ms=$(value_from_log topk_timed_per_call_ms \
        "$OUTPUT_DIR/timing/score-tile-128.log")
    for path in monolithic chunked; do
        if [[ $path == monolithic ]]; then
            log="$OUTPUT_DIR/timing/score-tile-128.log"
        else
            log="$OUTPUT_DIR/timing/topk-chunked.log"
        fi
        topk_ms=$(value_from_log topk_timed_per_call_ms "$log")
        speedup=$(awk -v base="$base_topk_ms" -v value="$topk_ms" \
            'BEGIN { printf "%.6f", base / value }')
        printf '%s,%s,%s\n' "$path" "$topk_ms" "$speedup" \
            >>"$OUTPUT_DIR/topk-timing.csv"
    done
else
    current_phase=reuse-timing-and-score-ncu
    printf 'Reusing validated timing and score Nsight results in %s\n' "$OUTPUT_DIR"
fi

if [[ $RUN_NCU == 0 ]]; then
    current_phase=complete-without-ncu
    printf 'SM75 indexer timing audit complete: %s\n' "$OUTPUT_DIR"
    exit 0
fi

command -v ncu >/dev/null 2>&1 || die "Nsight Compute CLI (ncu) not found"
ncu_bin=$(command -v ncu)
[[ $ncu_bin == /* && -x $ncu_bin ]] || die "cannot resolve ncu executable: $ncu_bin"
ncu_command=("$ncu_bin")
if [[ $NCU_USE_SUDO == 1 ]]; then
    command -v sudo >/dev/null 2>&1 || die "sudo not found"
    sudo -v
    ncu_command=(sudo -E "$ncu_bin")
fi

desired_metric_names=(
    gpu__time_duration.sum
    launch__registers_per_thread
    launch__shared_mem_per_block
    launch__occupancy_limit_blocks
    launch__occupancy_limit_registers
    launch__occupancy_limit_shared_mem
    launch__occupancy_limit_warps
    sm__warps_active.avg.pct_of_peak_sustained_active
    smsp__warps_eligible.avg.per_cycle_active
    sm__inst_issued.avg.per_cycle_active
    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active
    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed
    sm__inst_executed_pipe_tensor.sum
    smsp__inst_executed_pipe_tensor.sum
    dram__bytes.sum.per_second
    dram__cycles_active.avg.pct_of_peak_sustained_elapsed
    l1tex__throughput.avg.pct_of_peak_sustained_active
    l1tex__t_sector_hit_rate.pct
    lts__throughput.avg.pct_of_peak_sustained_elapsed
    lts__t_sector_hit_rate.pct
    smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
    smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
    smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
)
required_metric_names=(
    gpu__time_duration.sum
    launch__registers_per_thread
    launch__shared_mem_per_block
    launch__occupancy_limit_blocks
    launch__occupancy_limit_registers
    launch__occupancy_limit_shared_mem
    launch__occupancy_limit_warps
    sm__warps_active.avg.pct_of_peak_sustained_active
    dram__bytes.sum.per_second
    l1tex__t_sector_hit_rate.pct
    lts__t_sector_hit_rate.pct
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
    smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
)

# Nsight versions differ in which derived SM75 metrics their query lists.
# Keep the metrics already proven collectable on this machine mandatory and
# select every additional tensor/barrier metric only when the installed tool
# advertises it. This avoids losing the entire audit to one renamed counter.
available_metrics_raw="$OUTPUT_DIR/ncu/available-metrics.raw.txt"
available_metrics="$OUTPUT_DIR/ncu/available-metric-names.txt"
available_metrics_log="$OUTPUT_DIR/ncu/available-metrics-query.log"
query_args=(--config-file off --devices 0 --query-metrics)
ncu_help=$("$ncu_bin" --help 2>/dev/null || true)
if grep -Fq -- '--query-metrics-mode' <<<"$ncu_help"; then
    query_args+=(--query-metrics-mode all)
fi
metric_query_rc=0
env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
    "${ncu_command[@]}" "${query_args[@]}" >"$available_metrics_raw" \
    2>"$available_metrics_log" || metric_query_rc=$?
if (( metric_query_rc == 0 )); then
    grep -Eo '[[:alnum:]_]+__[[:alnum:]_.]+' "$available_metrics_raw" |
        sort -u >"$available_metrics" || true
else
    : >"$available_metrics"
    printf 'optional metric discovery failed with exit %s; collecting required metrics only\n' \
        "$metric_query_rc" >>"$available_metrics_log"
fi

selected_metric_names=("${required_metric_names[@]}")
: >"$OUTPUT_DIR/ncu/metrics-unavailable.txt"
for metric in "${desired_metric_names[@]}"; do
    required=0
    for required_metric in "${required_metric_names[@]}"; do
        if [[ $metric == "$required_metric" ]]; then
            required=1
            break
        fi
    done
    (( required == 0 )) || continue
    if grep -Fxq -- "$metric" "$available_metrics"; then
        selected_metric_names+=("$metric")
    else
        printf '%s\n' "$metric" >>"$OUTPUT_DIR/ncu/metrics-unavailable.txt"
    fi
done
printf '%s\n' "${desired_metric_names[@]}" >"$OUTPUT_DIR/ncu/metrics-desired.txt"
printf '%s\n' "${required_metric_names[@]}" >"$OUTPUT_DIR/ncu/metrics-required.txt"
printf '%s\n' "${selected_metric_names[@]}" >"$OUTPUT_DIR/ncu/metrics-selected.txt"
focused_metrics=$(IFS=,; printf '%s' "${selected_metric_names[*]}")

profile_one() {
    local name=$1 tile=$2 topk=$3 kernel_regex=$4 validation_regex=$5
    local base="$OUTPUT_DIR/ncu/$name"
    local -a collection
    if [[ $NCU_SET == full ]]; then
        collection=(--set full)
    elif [[ $NCU_SET == targeted ]]; then
        collection=(--section SpeedOfLight --section LaunchStats --section Occupancy
                    --section SchedulerStats --section WarpStateStats
                    --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis)
    else
        collection=(--metrics "$focused_metrics" --disable-extra-suffixes)
    fi
    printf 'Nsight Compute: %s...\n' "$name"
    local rc=0
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        DS4_PROFILE_INDEXER_TILE="$tile" \
        DS4_PROFILE_INDEXER_TOPK="$topk" \
        DS4_PROFILE_REPEATS=0 \
        "${ncu_command[@]}" --config-file off --verbose \
            --target-processes application-only --devices 0 \
            --kernel-name-base function --kernel-name "$kernel_regex" \
            --launch-count 1 --replay-mode kernel --cache-control none \
            --clock-control none --force-overwrite --export "$base" \
            "${collection[@]}" \
            ./tests/cuda_sm75_profile_harness indexer-32k \
            >"$base.log" 2>&1 || rc=$?
    if (( rc != 0 )); then
        tail -n 120 "$base.log" >&2 || true
        grep -q ERR_NVGPUCTRPERM "$base.log" &&
            printf 'error: rerun with NCU_USE_SUDO=1 for restricted counters\n' >&2 || true
        die "Nsight Compute failed for $name (exit $rc)"
    fi
    if grep -Eq '==ERROR==|No kernels were profiled|Failed to (profile|create report)' \
            "$base.log"; then
        tail -n 120 "$base.log" >&2 || true
        die "Nsight Compute produced a zero-kernel or failed capture for $name"
    fi
    [[ -s $base.ncu-rep ]] || die "missing Nsight report: $base.ncu-rep"
    if [[ $NCU_USE_SUDO == 1 ]]; then
        sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
    fi
    "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
        >"$base.csv" 2>"$base-import.log" ||
        die "could not import Nsight report: $base.ncu-rep"
    python3 speed-bench/validate-ncu-capture.py \
        "$base.csv" "$validation_regex" 0 \
        --process cuda_sm75_profile_harness \
        >"$base-validation.txt" 2>&1 || {
            cat "$base-validation.txt" >&2 || true
            die "Nsight report validation failed for $name"
        }
    cat "$base-validation.txt"
}

if [[ $RESUME == 0 ]]; then
    current_phase=nsight-score-tiles
    profile_one score-wmma128 128 monolithic \
        'regex:indexer_scores_wmma128_kernel.*' 'indexer_scores_wmma128_kernel.*'
    profile_one score-wmma64 64 monolithic \
        'regex:indexer_scores_wmma64_kernel.*' 'indexer_scores_wmma64_kernel.*'
    profile_one score-wmma32 32 monolithic \
        'regex:indexer_scores_wmma32_kernel.*' 'indexer_scores_wmma32_kernel.*'
    profile_one score-wmma16 16 monolithic \
        'regex:indexer_scores_wmma_kernel.*' 'indexer_scores_wmma_kernel.*'
fi

current_phase=nsight-topk
profile_one topk-monolithic 128 monolithic \
    'regex:indexer_topk_.*' 'indexer_topk_.*'
profile_one topk-chunk 128 chunked \
    'regex:indexer_topk_chunk_pow2_kernel.*' \
    'indexer_topk_chunk_pow2_kernel.*'
profile_one topk-final-merge 128 chunked \
    'regex:indexer_topk_merge_pow2_kernel.*' \
    'indexer_topk_merge_pow2_kernel.*'

current_phase=complete
printf 'SM75 indexer audit complete: %s\n' "$OUTPUT_DIR"
