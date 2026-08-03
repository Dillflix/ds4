#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Profile the routed-expert and native dense-Q8 SM75 kernels through a bounded,
single-GPU synthetic harness. No GGUF is opened.

The production-shape scenarios are:
  q4-early   layer-3 aggregate: 1879 pairs, 99 active experts, 183 tile16
  q4-late    layer-36 aggregate: 2186 pairs, 76 active experts, 189 tile16
  q2-early   same layer-3 routing, IQ2_XXS gate/up plus Q2_K down
  q2-late    same layer-36 routing, IQ2_XXS gate/up plus Q2_K down
  q8-q-b     T32:  512 x (1024 -> 32768), attention q_b
  q8-shared  T64:  512 x (2048 -> 4096), shared down
  q8-attn    T128: 512 x (4096 -> 1024), attention q_a
  q8-out-b   T256: 512 x (8192 -> 4096), attention output b

Optional environment:
  PROFILE_GPU=0             physical GPU to expose as logical device 0
  CUDA_ARCH=sm_75
  NCU_USE_SUDO=0            use sudo -E for restricted performance counters
  NCU_SET=focused           focused, targeted, or full
  PROFILE_SET=all           all, remaining, experts, or q8
                            remaining = Q2_K down plus Q8 T32/T256
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  PROFILE_DIR=/absolute/output/directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
NCU_SET=${NCU_SET:-focused}
PROFILE_SET=${PROFILE_SET:-all}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${PROFILE_DIR:-$repo_dir/sm75-kernel-profile-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $NCU_USE_SUDO == 0 || $NCU_USE_SUDO == 1 ]] || die "NCU_USE_SUDO must be 0 or 1"
[[ $SKIP_BUILD == 0 || $SKIP_BUILD == 1 ]] || die "SKIP_BUILD must be 0 or 1"
[[ $CREATE_ARCHIVE == 0 || $CREATE_ARCHIVE == 1 ]] || die "CREATE_ARCHIVE must be 0 or 1"
[[ $NCU_SET == focused || $NCU_SET == targeted || $NCU_SET == full ]] ||
    die "NCU_SET must be focused, targeted, or full"
[[ $PROFILE_SET == all || $PROFILE_SET == remaining ||
   $PROFILE_SET == experts || $PROFILE_SET == q8 ]] ||
    die "PROFILE_SET must be all, remaining, experts, or q8"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
command -v ncu >/dev/null 2>&1 || die "Nsight Compute CLI (ncu) not found"
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
fi
[[ -x tests/cuda_sm75_profile_harness ]] ||
    die "tests/cuda_sm75_profile_harness is missing; rerun with SKIP_BUILD=0"

[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/smoke" "$OUTPUT_DIR/ncu"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

current_phase=initialization
archive_ready=1
finalize() {
    local status=$?
    trap - EXIT
    if [[ $archive_ready == 1 && -d $OUTPUT_DIR ]]; then
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

{
    printf 'date_utc=%s\nrepo=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
    printf 'git_commit=%s\ngit_branch=%s\n' "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'profile_gpu_physical=%s\nprofile_gpu_logical=0\n' "$PROFILE_GPU"
    printf 'cuda_arch=%s\ncompute_capability=%s\nfree_mib_at_preflight=%s\n' \
        "$CUDA_ARCH" "$compute_cap" "$free_mib"
    printf 'ncu_set=%s\nncu_replay_mode=kernel\nprofile_set=%s\n' \
        "$NCU_SET" "$PROFILE_SET"
    printf 'harness_device_limit_bytes=3221225472\nfull_model_loaded=false\n'
    printf 'ds4_cli_lock_linked=false\nhelper_processes=false\n'
    printf '\n[gpu]\n'
    nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,name,pci.bus_id,memory.total,memory.free,ecc.mode.current,driver_version \
        --format=csv
    printf '\n[ncu]\n'; ncu --version 2>&1
    printf '\n[git status]\n'; git status --short
} >"$OUTPUT_DIR/manifest.txt"

case "$PROFILE_SET" in
    all) scenarios=(q4-early q4-late q2-early q2-late
                    q8-q-b q8-shared q8-attn q8-out-b) ;;
    remaining) scenarios=(q2-early q2-late q8-q-b q8-out-b) ;;
    experts) scenarios=(q4-early q4-late q2-early q2-late) ;;
    q8) scenarios=(q8-q-b q8-shared q8-attn q8-out-b) ;;
esac
current_phase=harness-smoke
for scenario in "${scenarios[@]}"; do
    printf 'Harness smoke: %s...\n' "$scenario"
    audit_env=()
    if [[ $scenario == q4-* || $scenario == q2-* ]]; then
        audit_env=("DS4_PROFILE_AUDIT_CSV=$OUTPUT_DIR/smoke/$scenario-tile-audit.csv")
    fi
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" "${audit_env[@]}" \
        ./tests/cuda_sm75_profile_harness "$scenario" \
        >"$OUTPUT_DIR/smoke/$scenario.log" 2>&1 || {
            tail -n 100 "$OUTPUT_DIR/smoke/$scenario.log" >&2 || true
            die "plain harness smoke failed for $scenario"
        }
    grep -q '^harness_status=ok$' "$OUTPUT_DIR/smoke/$scenario.log" ||
        die "plain harness did not validate $scenario"
done

ncu_bin=$(command -v ncu)
[[ $ncu_bin == /* && -x $ncu_bin ]] || die "cannot resolve ncu executable: $ncu_bin"
ncu_command=("$ncu_bin")
if [[ $NCU_USE_SUDO == 1 ]]; then
    command -v sudo >/dev/null 2>&1 || die "sudo not found"
    sudo -v
    ncu_command=(sudo -E "$ncu_bin")
fi

focused_metric_names=(
    gpu__time_duration.sum
    launch__registers_per_thread
    launch__shared_mem_per_block
    launch__occupancy_limit_blocks
    launch__occupancy_limit_registers
    launch__occupancy_limit_shared_mem
    launch__occupancy_limit_warps
    launch__occupancy_per_shared_mem_size
    sm__warps_active.avg.pct_of_peak_sustained_active
    smsp__warps_eligible.avg.per_cycle_active
    sm__inst_issued.avg.per_cycle_active
    sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active
    sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_elapsed
    sm__inst_executed_pipe_ipa.avg.pct_of_peak_sustained_elapsed
    sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed
    dram__bytes.sum.per_second
    dram__cycles_active.avg.pct_of_peak_sustained_elapsed
    l1tex__throughput.avg.pct_of_peak_sustained_active
    l1tex__m_xbar2l1tex_read_sectors.avg.pct_of_peak_sustained_elapsed
    l1tex__t_sector_hit_rate.pct
    lts__throughput.avg.pct_of_peak_sustained_elapsed
    lts__t_sectors.avg.pct_of_peak_sustained_elapsed
    lts__t_sector_hit_rate.pct
    lts__t_sectors_lookup_miss.sum
    lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed
    smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
    smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
    smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
)
focused_metrics=$(IFS=,; printf '%s' "${focused_metric_names[*]}")

profile_one() {
    local name=$1 scenario=$2 kernel_regex=$3 capture_set=${4:-$NCU_SET}
    local validation_regex=${5:-${kernel_regex#regex:}}
    local base="$OUTPUT_DIR/ncu/$name"
    local -a collection
    if [[ $capture_set == preflight ]]; then
        collection=(--metrics gpu__time_duration.sum --disable-extra-suffixes)
    elif [[ $capture_set == full ]]; then
        collection=(--set full)
    elif [[ $capture_set == targeted ]]; then
        collection=(--section SpeedOfLight --section LaunchStats --section Occupancy
                    --section SchedulerStats --section WarpStateStats
                    --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis)
    else
        collection=(--metrics "$focused_metrics" --disable-extra-suffixes)
    fi

    printf 'Nsight Compute kernel replay: %s...\n' "$name"
    local rc=0
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        "${ncu_command[@]}" --config-file off --verbose \
            --target-processes application-only --devices 0 \
            --kernel-name-base function --kernel-name "$kernel_regex" \
            --launch-count 1 --replay-mode kernel --cache-control none \
            --clock-control none --force-overwrite --export "$base" \
            "${collection[@]}" \
            ./tests/cuda_sm75_profile_harness "$scenario" \
            >"$base.log" 2>&1 || rc=$?
    if (( rc != 0 )); then
        tail -n 100 "$base.log" >&2 || true
        grep -q ERR_NVGPUCTRPERM "$base.log" &&
            printf 'error: rerun with NCU_USE_SUDO=1 for restricted counters\n' >&2 || true
        die "Nsight Compute failed for $name (exit $rc)"
    fi
    if grep -Eq '==ERROR==|No kernels were profiled|Failed to (profile|create report)' \
            "$base.log"; then
        tail -n 100 "$base.log" >&2 || true
        die "Nsight Compute produced a zero-kernel or failed capture for $name"
    fi
    [[ -s $base.ncu-rep ]] || die "missing Nsight report: $base.ncu-rep"
    if [[ $NCU_USE_SUDO == 1 ]]; then
        sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
    fi
    "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
        >"$base.csv" 2>"$base-import.log" ||
        die "could not import Nsight report: $base.ncu-rep"
    [[ -s $base.csv ]] || die "empty Nsight CSV: $base.csv"
    python3 speed-bench/validate-ncu-capture.py \
        "$base.csv" "$validation_regex" 0 \
        --process cuda_sm75_profile_harness \
        >"$base-validation.txt" 2>&1 || {
            cat "$base-validation.txt" >&2 || true
            die "Nsight report validation failed for $name"
        }
    cat "$base-validation.txt"
}

# Fail quickly on permissions, attachment, filtering, and replay using only the
# ~20 MiB q8-shared model/tensor state before any Q4 allocation is profiled.
current_phase=nsight-preflight
profile_one preflight-q8-shared q8-shared \
    'regex:matmul_q8_0_mma_sm75_exact_kernel.*' preflight \
    'matmul_q8_0_mma_sm75_exact_kernel.*64'

current_phase=nsight-focused
if [[ $PROFILE_SET == all || $PROFILE_SET == experts ]]; then
    profile_one early-layer3-q4-gate-up-tile8 q4-early \
        'regex:moe_gate_up_mid_q4K_tile8_mma_kernel.*'
    profile_one late-layer36-q4-gate-up-tile8 q4-late \
        'regex:moe_gate_up_mid_q4K_tile8_mma_kernel.*'
    profile_one early-layer3-q4-down-tile16 q4-early \
        'regex:moe_down_q4K_tile16_mma_sm75_kernel.*'
    profile_one late-layer36-q4-down-tile16 q4-late \
        'regex:moe_down_q4K_tile16_mma_sm75_kernel.*'
    profile_one early-layer3-iq2-gate-up-tile16 q2-early \
        'regex:moe_gate_up_mid_iq2_tile16_mma_sm75_kernel.*'
    profile_one late-layer36-iq2-gate-up-tile16 q2-late \
        'regex:moe_gate_up_mid_iq2_tile16_mma_sm75_kernel.*'
fi
if [[ $PROFILE_SET == all || $PROFILE_SET == experts ||
      $PROFILE_SET == remaining ]]; then
    profile_one early-layer3-q2-down-tile16 q2-early \
        'regex:moe_down_expert_tile16_rowspan_kernel.*512.*'
    profile_one late-layer36-q2-down-tile16 q2-late \
        'regex:moe_down_expert_tile16_rowspan_kernel.*512.*'
fi
if [[ $PROFILE_SET == all || $PROFILE_SET == q8 ||
      $PROFILE_SET == remaining ]]; then
    profile_one attention-q-b-layer9-dense-q8-t32 q8-q-b \
        'regex:matmul_q8_0_mma_sm75_exact_kernel.*' "$NCU_SET" \
        'matmul_q8_0_mma_sm75_exact_kernel.*32'
    if [[ $PROFILE_SET != remaining ]]; then
        profile_one shared-layer8-dense-q8-t64 q8-shared \
            'regex:matmul_q8_0_mma_sm75_exact_kernel.*' "$NCU_SET" \
            'matmul_q8_0_mma_sm75_exact_kernel.*64'
        profile_one attention-layer9-dense-q8-t128 q8-attn \
            'regex:matmul_q8_0_mma_sm75_exact_kernel.*' "$NCU_SET" \
            'matmul_q8_0_mma_sm75_exact_kernel.*128'
    fi
    profile_one attention-output-b-layer9-dense-q8-t256 q8-out-b \
        'regex:matmul_q8_0_mma_sm75_exact_kernel.*' "$NCU_SET" \
        'matmul_q8_0_mma_sm75_exact_kernel.*256'
fi

current_phase=complete
printf 'Bounded SM75 kernel profile complete: %s\n' "$OUTPUT_DIR"
