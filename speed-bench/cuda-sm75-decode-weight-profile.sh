#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Capture one-token production decode weight kernels on one SM75 GPU with
Nsight Compute. No GGUF is opened. Every scenario uses the shipping dispatch,
a synthetic zero-weight model, and exact-zero output validation.

Captured families:
  * Q4-32 and Q3A4 routed gate/up;
  * every materially distinct packed-Q8 decode access pattern observed in the
    32K production trace (single T32, pair, K-slice, grouped-A, shared-mid);
  * the 256/512/1024-wide ordered F16 compressor-pair kernels.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  PROFILE_SET=all         all, routed, q8, or f16
  SCENARIOS=...           explicit comma-separated scenario override
  NCU_SET=focused         focused, targeted, or full
  NCU_CACHE_CONTROL=all   all (cold streamed weights) or none (replay-warm)
  NCU_USE_SUDO=0
  SKIP_BUILD=0
  RESUME=0
  CREATE_ARCHIVE=1
  DECODE_WEIGHT_PROFILE_DIR=/absolute/output/directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
PROFILE_SET=${PROFILE_SET:-all}
NCU_SET=${NCU_SET:-focused}
NCU_CACHE_CONTROL=${NCU_CACHE_CONTROL:-all}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
RESUME=${RESUME:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${DECODE_WEIGHT_PROFILE_DIR:-$repo_dir/sm75-decode-weight-profile-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $PROFILE_SET == all || $PROFILE_SET == routed || $PROFILE_SET == q8 ||
   $PROFILE_SET == f16 ]] || die "PROFILE_SET must be all, routed, q8, or f16"
[[ $NCU_SET == focused || $NCU_SET == targeted || $NCU_SET == full ]] ||
    die "NCU_SET must be focused, targeted, or full"
[[ $NCU_CACHE_CONTROL == all || $NCU_CACHE_CONTROL == none ]] ||
    die "NCU_CACHE_CONTROL must be all or none"
[[ $NCU_USE_SUDO == 0 || $NCU_USE_SUDO == 1 ]] || die "NCU_USE_SUDO must be 0 or 1"
[[ $SKIP_BUILD == 0 || $SKIP_BUILD == 1 ]] || die "SKIP_BUILD must be 0 or 1"
[[ $RESUME == 0 || $RESUME == 1 ]] || die "RESUME must be 0 or 1"
[[ $CREATE_ARCHIVE == 0 || $CREATE_ARCHIVE == 1 ]] || die "CREATE_ARCHIVE must be 0 or 1"

for command in nvidia-smi python3 ncu tar; do
    command -v "$command" >/dev/null 2>&1 || die "$command not found"
done
ncu_bin=$(command -v ncu)
[[ $ncu_bin == /* && -x $ncu_bin ]] || die "cannot resolve ncu executable: $ncu_bin"

compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] ||
    die "physical GPU $PROFILE_GPU has compute capability ${compute_cap:-unknown}; SM75 is required"
free_mib=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $free_mib =~ ^[0-9]+$ ]] || die "could not read free VRAM for GPU $PROFILE_GPU"
(( free_mib >= 4096 )) || die "GPU $PROFILE_GPU has only ${free_mib} MiB free; 4096 MiB is required"

target=tests/cuda_sm75_decode_weight_profile
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "$target" CUDA_ARCH="$CUDA_ARCH"
else
    make -q "$target" CUDA_ARCH="$CUDA_ARCH" ||
        die "SKIP_BUILD=1 found a stale decode-weight harness"
fi
[[ -x $target ]] || die "$target is missing; rerun with SKIP_BUILD=0"

routed_scenarios=(q4-32-gate-up q3a4-gate-up)
q8_scenarios=(q8-single-t32 q8-pair-2048 q8-pair-1024 q8-kslice-t256 q8-grouped-a-half q8-shared-mid)
f16_scenarios=(f16-pair-256 f16-pair-512 f16-pair-1024)
case "$PROFILE_SET" in
    all) scenarios=("${routed_scenarios[@]}" "${q8_scenarios[@]}" "${f16_scenarios[@]}") ;;
    routed) scenarios=("${routed_scenarios[@]}") ;;
    q8) scenarios=("${q8_scenarios[@]}") ;;
    f16) scenarios=("${f16_scenarios[@]}") ;;
esac
if [[ -n ${SCENARIOS:-} ]]; then
    IFS=, read -r -a scenarios <<<"$SCENARIOS"
    (( ${#scenarios[@]} > 0 )) || die "SCENARIOS is empty"
fi
scenario_csv=$(IFS=,; printf '%s' "${scenarios[*]}")

if [[ $RESUME == 1 ]]; then
    [[ -n ${DECODE_WEIGHT_PROFILE_DIR:-} ]] ||
        die "RESUME=1 requires DECODE_WEIGHT_PROFILE_DIR"
    [[ -d $OUTPUT_DIR ]] || die "resume directory does not exist: $OUTPUT_DIR"
else
    [[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/smoke" "$OUTPUT_DIR/ncu"
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
        printf 'profile_set=%s\nscenarios=%s\n' "$PROFILE_SET" "$scenario_csv"
        printf 'ncu_set=%s\nncu_cache_control=%s\n' "$NCU_SET" "$NCU_CACHE_CONTROL"
        printf 'n_tokens=1\nfull_model_loaded=false\nq8_arithmetic=production-dp4a\n'
        printf 'q8_f16_cache=disabled\nq32_decode_graph=disabled\n'
        printf '\n[gpu]\n'
        nvidia-smi -i "$PROFILE_GPU" \
            --query-gpu=index,name,pci.bus_id,memory.total,memory.free,ecc.mode.current,driver_version \
            --format=csv
        printf '\n[git status]\n'; git status --short
    } >"$OUTPUT_DIR/manifest.txt"
fi

valid_scenario() {
    case "$1" in
        q4-32-gate-up|q3a4-gate-up|q8-single-t32|q8-pair-2048|q8-pair-1024|\
        q8-kslice-t256|q8-grouped-a-half|q8-shared-mid|f16-pair-256|\
        f16-pair-512|f16-pair-1024) return 0 ;;
        *) return 1 ;;
    esac
}

current_phase=smoke-validation
for scenario in "${scenarios[@]}"; do
    valid_scenario "$scenario" || die "unknown scenario in SCENARIOS: $scenario"
    log="$OUTPUT_DIR/smoke/$scenario.log"
    if [[ $RESUME == 1 && -s $log ]]; then
        printf 'Reusing smoke validation: %s\n' "$scenario"
    else
        printf 'Smoke validation: %s...\n' "$scenario"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$target "$scenario" \
            >"$log" 2>&1 || {
                tail -n 120 "$log" >&2 || true
                die "decode-weight harness failed for $scenario"
            }
    fi
    grep -q '^output_validation=exact-zero$' "$log" ||
        die "exact output validation missing for $scenario"
    grep -q '^harness_status=ok$' "$log" || die "harness success marker missing for $scenario"
done

ncu_command=("$ncu_bin")
if [[ $NCU_USE_SUDO == 1 ]]; then
    command -v sudo >/dev/null 2>&1 || die "sudo not found"
    sudo -v
    ncu_command=(sudo -E "$ncu_bin")
fi

required_metric_names=(
    gpu__time_duration.sum
    dram__bytes.sum.per_second
    dram__bytes.avg.pct_of_peak_sustained_elapsed
    lts__t_sector_hit_rate.pct
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
    smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
    sm__warps_active.avg.pct_of_peak_sustained_active
)
desired_metric_names=(
    "${required_metric_names[@]}"
    dram__bytes_read.sum
    dram__bytes_write.sum
    lts__throughput.avg.pct_of_peak_sustained_elapsed
    smsp__warps_eligible.avg.per_cycle_active
    launch__registers_per_thread
    launch__shared_mem_per_block
    launch__block_size
    launch__grid_size
)

available_raw="$OUTPUT_DIR/ncu/available-metrics.raw.txt"
available_names="$OUTPUT_DIR/ncu/available-metric-names.txt"
available_log="$OUTPUT_DIR/ncu/available-metrics-query.log"
query_args=(--config-file off --devices 0 --query-metrics)
ncu_help=$("$ncu_bin" --help 2>/dev/null || true)
grep -Fq -- '--query-metrics-mode' <<<"$ncu_help" && query_args+=(--query-metrics-mode all)
metric_query_rc=0
env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" "${ncu_command[@]}" "${query_args[@]}" \
    >"$available_raw" 2>"$available_log" || metric_query_rc=$?
if (( metric_query_rc == 0 )); then
    grep -Eo '[[:alnum:]_]+__[[:alnum:]_.]+' "$available_raw" | sort -u \
        >"$available_names" || true
else
    : >"$available_names"
    printf 'metric discovery failed with exit %s; attempting requested metrics directly\n' \
        "$metric_query_rc" >>"$available_log"
fi

selected_metric_names=("${required_metric_names[@]}")
: >"$OUTPUT_DIR/ncu/metrics-unavailable.txt"
for metric in "${desired_metric_names[@]}"; do
    required=0
    for item in "${required_metric_names[@]}"; do
        [[ $metric == "$item" ]] && { required=1; break; }
    done
    (( required == 0 )) || continue
    if grep -Fxq -- "$metric" "$available_names"; then
        selected_metric_names+=("$metric")
    else
        printf '%s\n' "$metric" >>"$OUTPUT_DIR/ncu/metrics-unavailable.txt"
    fi
done
printf '%s\n' "${required_metric_names[@]}" >"$OUTPUT_DIR/ncu/metrics-required.txt"
printf '%s\n' "${selected_metric_names[@]}" >"$OUTPUT_DIR/ncu/metrics-selected.txt"
focused_metrics=$(IFS=,; printf '%s' "${selected_metric_names[*]}")

kernel_regex() {
    case "$1" in
        # With --kernel-name-base function, Nsight strips the template value
        # from this symbol. Each scenario launches exactly one specialization,
        # so the harness scenario—not a missing suffix—provides separation.
        q4-32-gate-up|q3a4-gate-up) printf '%s' 'moe_gate_up_mid_decode_sm75_q32_owned_kernel.*' ;;
        q8-single-t32) printf '%s' 'matmul_q8_0_preq_warp8_kernel.*' ;;
        q8-pair-2048|q8-pair-1024) printf '%s' 'matmul_q8_0_pair_preq_warp8_kernel.*' ;;
        q8-kslice-t256) printf '%s' 'matmul_q8_0_kslice_preq_warp8_kernel.*' ;;
        q8-grouped-a-half) printf '%s' 'grouped_q8_0_a_preq_warp8_kernel.*' ;;
        q8-shared-mid) printf '%s' 'shared_mid_q8_0_preq_warp8_exact_kernel.*' ;;
        f16-pair-256|f16-pair-512|f16-pair-1024) printf '%s' 'matmul_f16_pair_ordered_chunks_kernel.*' ;;
    esac
}

block_size() {
    case "$1" in f16-pair-*) printf '32' ;; *) printf '256' ;; esac
}

profile_one() {
    local scenario=$1 regex block base rc=0
    regex=$(kernel_regex "$scenario")
    block=$(block_size "$scenario")
    base="$OUTPUT_DIR/ncu/$scenario"
    if [[ $RESUME == 1 && -s $base.csv && -s $base.ncu-rep ]]; then
        printf 'Reusing Nsight capture: %s\n' "$scenario"
        return
    fi
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
    printf 'Nsight Compute: %s...\n' "$scenario"
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        "${ncu_command[@]}" --config-file off --verbose \
            --target-processes application-only --devices 0 \
            --kernel-name-base function --kernel-name "regex:$regex" \
            --launch-count 1 --replay-mode kernel \
            --cache-control "$NCU_CACHE_CONTROL" --clock-control none \
            --force-overwrite --export "$base" "${collection[@]}" \
            ./$target "$scenario" >"$base.log" 2>&1 || rc=$?
    if (( rc != 0 )); then
        tail -n 120 "$base.log" >&2 || true
        grep -q ERR_NVGPUCTRPERM "$base.log" &&
            printf 'error: rerun with NCU_USE_SUDO=1 for restricted counters\n' >&2 || true
        die "Nsight Compute failed for $scenario (exit $rc)"
    fi
    if grep -Eq '==ERROR==|No kernels were profiled|Failed to (profile|create report)' "$base.log"; then
        tail -n 120 "$base.log" >&2 || true
        die "Nsight Compute produced a zero-kernel or failed capture for $scenario"
    fi
    [[ -s $base.ncu-rep ]] || die "missing Nsight report: $base.ncu-rep"
    if [[ $NCU_USE_SUDO == 1 ]]; then
        sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
    fi
    "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
        >"$base.csv" 2>"$base-import.log" ||
        die "could not import Nsight report for $scenario"
    python3 speed-bench/validate-ncu-capture.py \
        "$base.csv" "$regex" 0 --process cuda_sm75_decode_weight_profile \
        --block-size "$block" >"$base-validation.txt" 2>&1 || {
            cat "$base-validation.txt" >&2 || true
            die "Nsight report validation failed for $scenario"
        }
    cat "$base-validation.txt"
}

current_phase=nsight-compute
for scenario in "${scenarios[@]}"; do profile_one "$scenario"; done

current_phase=summarization
python3 speed-bench/summarize-sm75-decode-weight-profile.py \
    "$OUTPUT_DIR/ncu" "$OUTPUT_DIR" --scenarios "$scenario_csv" \
    | tee "$OUTPUT_DIR/summary.stdout.txt"

current_phase=complete
printf 'SM75 decode weight-kernel profile complete: %s\n' "$OUTPUT_DIR"
