#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Capture the next full-Q4 SM75 evidence pass without changing production
dispatch:
  1. fixed production prefill benchmark and Nsight Systems branch trace;
  2. complete Q8->F16 cache decision coverage for the first frontier;
  3. Nsight Compute on early/late Q4 gate/up tile8;
  4. Nsight Compute on two proven-uncached dense-Q8 projections selected
     from the cache audit (one attention and one shared projection when
     available).

Required environment:
  MODEL=/absolute/path/to/full-Q4.gguf

Prompt suite:
  PROMPT=/absolute/path/prompt.txt
  PROMPT_MANIFEST=/absolute/path/prompts.tsv
    Tab-separated: label<TAB>path. Blank lines and # comments are ignored.
  Default: speed-bench/promessi_sposi.txt

Optional environment:
  GPU_DEVICES=0,2,1,3
  GPU_VRAM=auto
  CUDA_ARCH=sm_75
  CTX_START=2048
  CTX_MAX=65536
  STEP_MUL=2
  PREFILL_CHUNK=2048
  PROFILE_TOKENS=2048
  EARLY_LAYER=3
  LATE_LAYER=36
  NCU_SET=focused           focused, targeted, or full
  NCU_USE_SUDO=0            run ncu through sudo -E
  RUN_NSYS=1
  RUN_NCU=1
  SKIP_BUILD=0
  SKIP_BASELINE=0
  SKIP_COVERAGE=0           reuse coverage files in Q4_EVIDENCE_DIR
  Q4_EVIDENCE_DIR=/absolute/path/output-directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
: "${MODEL:?set MODEL to the absolute full-Q4 GGUF path}"
[[ -f $MODEL ]] || die "model not found: $MODEL"
command -v python3 >/dev/null 2>&1 || die "python3 not found"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"

GPU_DEVICES=${GPU_DEVICES:-0,2,1,3}
GPU_VRAM=${GPU_VRAM:-auto}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
CTX_START=${CTX_START:-2048}
CTX_MAX=${CTX_MAX:-65536}
STEP_MUL=${STEP_MUL:-2}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
PROFILE_TOKENS=${PROFILE_TOKENS:-2048}
EARLY_LAYER=${EARLY_LAYER:-3}
LATE_LAYER=${LATE_LAYER:-36}
NCU_SET=${NCU_SET:-focused}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
RUN_NSYS=${RUN_NSYS:-1}
RUN_NCU=${RUN_NCU:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
SKIP_BASELINE=${SKIP_BASELINE:-0}
SKIP_COVERAGE=${SKIP_COVERAGE:-0}
EVIDENCE_DIR=${Q4_EVIDENCE_DIR:-$repo_dir/q4-prefill-evidence-$(date -u +%Y%m%dT%H%M%SZ)}

for item in "CTX_START:$CTX_START" "CTX_MAX:$CTX_MAX" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" "PROFILE_TOKENS:$PROFILE_TOKENS" \
            "EARLY_LAYER:$EARLY_LAYER" "LATE_LAYER:$LATE_LAYER"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( CTX_START > 0 && CTX_MAX >= CTX_START && PROFILE_TOKENS > 0 )) ||
    die "invalid context/profile range"
(( EARLY_LAYER < 22 && LATE_LAYER >= 22 && LATE_LAYER < 43 )) ||
    die "EARLY_LAYER/LATE_LAYER must straddle the forced 22/21 split"
[[ $NCU_SET == focused || $NCU_SET == targeted || $NCU_SET == full ]] ||
    die "NCU_SET must be focused, targeted, or full"
for flag in NCU_USE_SUDO RUN_NSYS RUN_NCU SKIP_BUILD SKIP_BASELINE SKIP_COVERAGE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
IFS=',' read -r -a gpu_devices <<<"$GPU_DEVICES"
(( ${#gpu_devices[@]} == 4 )) || die "this evidence pass requires four GPU devices"

declare -a prompt_labels=() prompt_paths=()
if [[ -n ${PROMPT_MANIFEST:-} ]]; then
    [[ -f $PROMPT_MANIFEST ]] || die "prompt manifest not found: $PROMPT_MANIFEST"
    while IFS=$'\t' read -r label path extra; do
        [[ -n $label && ${label:0:1} != "#" ]] || continue
        [[ -n $path && -z ${extra:-} ]] || die "invalid prompt manifest row for $label"
        [[ $path == /* ]] || path="$repo_dir/$path"
        [[ -f $path ]] || die "prompt not found: $path"
        [[ $label =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe prompt label: $label"
        prompt_labels+=("$label"); prompt_paths+=("$path")
    done <"$PROMPT_MANIFEST"
else
    prompt=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
    [[ -f $prompt ]] || die "prompt not found: $prompt"
    prompt_labels+=(promessi); prompt_paths+=("$prompt")
fi
(( ${#prompt_paths[@]} > 0 )) || die "prompt suite is empty"

current_phase=initialization
archive_ready=0
finalize() {
    local status=$?
    trap - EXIT
    if [[ $archive_ready == 1 && -d ${EVIDENCE_DIR:-} ]]; then
        local status_tmp="$EVIDENCE_DIR/run-status.txt.tmp"
        printf 'state=finished\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
            "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >"$status_tmp"
        mv -f "$status_tmp" "$EVIDENCE_DIR/run-status.txt"
        local archive="$EVIDENCE_DIR.tar.gz"
        if tar -C "$(dirname "$EVIDENCE_DIR")" -czf "$archive" \
                "$(basename "$EVIDENCE_DIR")"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
        fi
    fi
    exit "$status"
}
trap finalize EXIT

mkdir -p "$EVIDENCE_DIR" "$EVIDENCE_DIR/runtime" "$EVIDENCE_DIR/coverage" \
         "$EVIDENCE_DIR/nsys" "$EVIDENCE_DIR/ncu"
EVIDENCE_DIR=$(cd "$EVIDENCE_DIR" && pwd)
archive_ready=1

initial_manifest="$EVIDENCE_DIR/manifest.txt"
resume_evidence=0
if [[ $SKIP_BASELINE == 1 || $SKIP_COVERAGE == 1 ]]; then
    [[ -f $initial_manifest ]] ||
        die "resume requested but initial manifest is missing: $initial_manifest"
    resume_evidence=1
elif [[ -e $initial_manifest ]]; then
    die "output directory already has a manifest; choose a new Q4_EVIDENCE_DIR or use resume flags"
fi

# A hard reset must not leave an earlier status looking like the result of the
# current resume.  Publish the running marker atomically before doing work.
status_tmp="$EVIDENCE_DIR/run-status.txt.tmp"
printf 'state=running\nlast_phase=%s\ndate_utc=%s\n' \
    "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$status_tmp"
mv -f "$status_tmp" "$EVIDENCE_DIR/run-status.txt"

if [[ $SKIP_BUILD == 0 ]]; then
    current_phase=build
    make -B -j"$(nproc)" ds4-bench tests/test_engine_mgpu_placement CUDA_ARCH="$CUDA_ARCH"
    ./tests/test_engine_mgpu_placement
fi
[[ -x ./ds4-bench ]] || die "ds4-bench is missing"
if [[ $RUN_NCU == 1 ]] &&
   ! LC_ALL=C grep -aFq 'DS4_CUDA_NCU_TARGET_DEVICE' ./ds4-bench; then
    die "ds4-bench lacks the targeted CUDA profiler support; rerun with SKIP_BUILD=0"
fi

# Presence-based debug flags must be absent. The cache and 22/21 placement are
# the production configuration being measured.
unset DS4_CUDA_MOE_PROFILE DS4_CUDA_ATTN_OUTPUT_PROFILE
unset DS4_METAL_LAYER_STAGE_PROFILE DS4_METAL_GRAPH_PREFILL_PROFILE
unset DS4_CUDA_PREFILL_PIPELINE_SEQUENTIAL DS4_CUDA_PREFILL_PIPELINE_SYNC_BOUNDARY
unset DS4_CUDA_PREFILL_TILE_AUDIT_CSV DS4_CUDA_Q8_CACHE_AUDIT_CSV
unset DS4_CUDA_NCU_TARGET_MODULE DS4_CUDA_NCU_TARGET_LAYER DS4_CUDA_NCU_TARGET_POS
unset DS4_CUDA_NCU_TARGET_DEVICE
unset DS4_NSYS_CAPTURE_PREFILL
unset DS4_CUDA_NO_Q8_F16_CACHE DS4_CUDA_Q8_F32_ALL DS4_CUDA_Q8_F32_LARGE
unset DS4_CUDA_ATTN_Q_B_F32_CACHE DS4_CUDA_Q8_F32_PRELOAD
unset DS4_CUDA_MOE_NO_Q4_MMA DS4_CUDA_MOE_NO_Q4_MMA_TILE16
export DS4_CUDA_EP_STAGE_SPLIT=22
export DS4_CUDA_TP_PREFILL_ATTN_HEADS=0
export DS4_CUDA_PREFILL_PIPELINE=1
export DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
export DS4_CUDA_PREFILL_PIPELINE_MB=512

bench_common=(
    -m "$MODEL" --backend cuda --gpu-devices "$GPU_DEVICES"
    --gpu-vram "$GPU_VRAM" --cuda-tensor-parallel --gen-tokens 0
)

# Preserve the initial manifest on a resume and reject configuration mixing.
# The commit may legitimately change to contain a profiler-script fix, so it is
# recorded in a separate resume manifest rather than treated as run identity.
model_bytes=$(stat -c %s "$MODEL")
if [[ $resume_evidence == 1 ]]; then
    manifest_value() {
        local key=$1
        awk -F= -v wanted="$key" '
            $1 == wanted { print substr($0, index($0, "=") + 1); exit }
        ' "$initial_manifest"
    }
    require_manifest_value() {
        local key=$1 expected=$2 actual
        actual=$(manifest_value "$key")
        [[ -n $actual ]] || die "initial manifest has no $key value"
        [[ $actual == "$expected" ]] ||
            die "resume mismatch for $key: initial=$actual current=$expected"
    }
    require_manifest_value model "$MODEL"
    require_manifest_value model_bytes "$model_bytes"
    require_manifest_value gpu_devices "$GPU_DEVICES"
    require_manifest_value gpu_vram "$GPU_VRAM"
    require_manifest_value cuda_arch "$CUDA_ARCH"
    require_manifest_value split 22/21
    require_manifest_value ctx_start "$CTX_START"
    require_manifest_value ctx_max "$CTX_MAX"
    require_manifest_value step_mul "$STEP_MUL"
    require_manifest_value prefill_chunk "$PREFILL_CHUNK"
    require_manifest_value profile_tokens "$PROFILE_TOKENS"
    mapfile -t manifest_prompt_rows < <(awk '
        /^\[prompts\]$/ { in_prompts=1; next }
        /^\[/ && in_prompts { exit }
        in_prompts && NF { print }
    ' "$initial_manifest")
    (( ${#manifest_prompt_rows[@]} == ${#prompt_paths[@]} )) ||
        die "resume prompt count differs from initial manifest"
    for i in "${!prompt_paths[@]}"; do
        prompt_row=$(printf '%s\t%s' "${prompt_labels[$i]}" "${prompt_paths[$i]}")
        [[ ${manifest_prompt_rows[$i]} == "$prompt_row" ]] ||
            die "resume prompt order/path differs at index $i: ${prompt_labels[$i]}"
    done
fi

if [[ $SKIP_BASELINE == 1 ]]; then
    for label in "${prompt_labels[@]}"; do
        [[ -s $EVIDENCE_DIR/runtime/$label.csv ]] ||
            die "SKIP_BASELINE=1 but runtime/$label.csv is missing or empty"
    done
fi
manifest_out="$initial_manifest"
if [[ $resume_evidence == 1 ]]; then
    manifest_out="$EVIDENCE_DIR/resume-manifest-$(date -u +%Y%m%dT%H%M%SZ).txt"
fi
{
    [[ $resume_evidence == 0 ]] || printf 'resume_of=%s\n' "$initial_manifest"
    printf 'date_utc=%s\nrepo=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
    printf 'git_commit=%s\ngit_branch=%s\n' "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\n' "$MODEL" "$model_bytes"
    printf 'gpu_devices=%s\ngpu_vram=%s\ncuda_arch=%s\nsplit=22/21\n' \
        "$GPU_DEVICES" "$GPU_VRAM" "$CUDA_ARCH"
    printf 'ctx_start=%s\nctx_max=%s\nstep_mul=%s\nprefill_chunk=%s\nprofile_tokens=%s\n' \
        "$CTX_START" "$CTX_MAX" "$STEP_MUL" "$PREFILL_CHUNK" "$PROFILE_TOKENS"
    printf 'ncu_set=%s\nncu_replay_mode=application\nncu_app_replay_buffer=file\n' "$NCU_SET"
    printf 'ncu_app_replay_match=grid\nncu_app_replay_mode=balanced\n'
    printf 'ncu_target_processes=all\nncu_target_process_filter=none\n'
    printf 'ncu_preflight=early_q4_ungated_then_gated_duration_only\n'
    printf 'ncu_config_file=off\nncu_scope_device_verified=true\n'
    printf '\n[prompts]\n'
    for i in "${!prompt_paths[@]}"; do
        printf '%s\t%s\n' "${prompt_labels[$i]}" "${prompt_paths[$i]}"
    done
    printf '\n[git status]\n'; git status --short
    printf '\n[gpu inventory]\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,ecc.mode.current,driver_version --format=csv
    printf '\n[gpu topology]\n'; nvidia-smi topo -m
    printf '\n[nvcc]\n'; nvcc --version 2>&1 || true
    printf '\n[ncu]\n'; ncu --version 2>&1 || true
    printf '\n[nsys]\n'; nsys --version 2>&1 || true
} >"$manifest_out"

if [[ $SKIP_BASELINE == 0 ]]; then
    current_phase=production-baseline
    for i in "${!prompt_paths[@]}"; do
        label=${prompt_labels[$i]}; prompt=${prompt_paths[$i]}
        printf 'Production full-Q4 benchmark: %s...\n' "$label"
        ./ds4-bench "${bench_common[@]}" --prompt-file "$prompt" \
            --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
            --ctx-alloc "$((CTX_MAX + 1))" --step-mul "$STEP_MUL" \
            --prefill-chunk "$PREFILL_CHUNK" \
            --csv "$EVIDENCE_DIR/runtime/$label.csv" \
            >"$EVIDENCE_DIR/runtime/$label.log" 2>&1
    done
fi

audit_prompt=${prompt_paths[0]}
if [[ $SKIP_COVERAGE == 0 ]]; then
    current_phase=q8-cache-coverage
    printf 'Capturing full-Q4 Q8 cache decisions...\n'
    DS4_CUDA_Q8_CACHE_AUDIT_CSV="$EVIDENCE_DIR/coverage/q8-cache.csv" \
        ./ds4-bench "${bench_common[@]}" --prompt-file "$audit_prompt" \
            --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS" \
            --ctx-alloc "$((PROFILE_TOKENS + 1))" --step-incr "$PROFILE_TOKENS" \
            --prefill-chunk "$PREFILL_CHUNK" \
            --csv "$EVIDENCE_DIR/coverage/benchmark.csv" \
            >"$EVIDENCE_DIR/coverage/benchmark.log" 2>&1
    python3 speed-bench/summarize-q8-cache-audit.py \
        "$EVIDENCE_DIR/coverage/q8-cache.csv" \
        "$EVIDENCE_DIR/coverage/q8-cache-summary.csv" \
        "$EVIDENCE_DIR/coverage/native-q8-targets.tsv" \
        | tee "$EVIDENCE_DIR/coverage/q8-cache-summary.txt"
else
    for reused in q8-cache.csv q8-cache-summary.csv q8-cache-summary.txt \
                  native-q8-targets.tsv; do
        [[ -s $EVIDENCE_DIR/coverage/$reused ]] ||
            die "SKIP_COVERAGE=1 but coverage/$reused is missing or empty"
    done
    native_target_count=$(awk 'NR > 1 && NF { n++ } END { print n + 0 }' \
        "$EVIDENCE_DIR/coverage/native-q8-targets.tsv")
    (( native_target_count >= 2 )) ||
        die "reused coverage has fewer than two native-Q8 profiler targets"
fi

if [[ $RUN_NSYS == 1 ]]; then
    current_phase=nsight-systems
    command -v nsys >/dev/null 2>&1 || die "Nsight Systems CLI (nsys) not found"
    printf 'Capturing production runtime branches with Nsight Systems...\n'
    export DS4_NSYS_CAPTURE_PREFILL=1
    nsys profile --force-overwrite=true --sample=none \
        --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi \
        --capture-range-end=stop --output="$EVIDENCE_DIR/nsys/full-q4-prefill" \
        ./ds4-bench "${bench_common[@]}" --prompt-file "$audit_prompt" \
            --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS" \
            --ctx-alloc "$((PROFILE_TOKENS + 1))" --step-incr "$PROFILE_TOKENS" \
            --prefill-chunk "$PREFILL_CHUNK" \
            --csv "$EVIDENCE_DIR/nsys/benchmark.csv" \
            >"$EVIDENCE_DIR/nsys/capture.log" 2>&1
    unset DS4_NSYS_CAPTURE_PREFILL
    for report in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum cuda_gpu_trace; do
        nsys stats --report "$report" --format csv \
            "$EVIDENCE_DIR/nsys/full-q4-prefill.nsys-rep" \
            >"$EVIDENCE_DIR/nsys/$report.csv" 2>"$EVIDENCE_DIR/nsys/$report.log" || true
    done
    kernel_summary="$EVIDENCE_DIR/nsys/cuda_gpu_kern_sum.csv"
    grep -E 'moe_gate_up_mid_q4K_tile8_mma_kernel|moe_down_q4K_tile16_mma_sm75_kernel|matmul_q8_0_mma_sm75_exact_kernel' \
        "$kernel_summary" >"$EVIDENCE_DIR/nsys/required-runtime-branches.txt" || true
    grep -q 'moe_gate_up_mid_q4K_tile8_mma_kernel' "$kernel_summary" ||
        die "runtime trace did not execute the required Q4 gate/up tile8 kernel"
    grep -q 'moe_down_q4K_tile16_mma_sm75_kernel' "$kernel_summary" ||
        die "runtime trace did not execute the required SM75 Q4 down tile16 kernel"
    if grep -q 'moe_gate_up_mid_iq2' "$kernel_summary"; then
        die "runtime trace contains IQ2 gate/up; MODEL is not the requested full-Q4 path"
    fi
fi

ncu_capture() {
    local name=$1 device=$2 module=$3 layer=$4 pos=$5 kernel=$6
    local capture_set=${7:-$NCU_SET}
    local scope_mode=${8:-targeted}
    local base="$EVIDENCE_DIR/ncu/$name"
    local -a sections profiler_args target_env
    if [[ $capture_set == preflight ]]; then
        sections=(--metrics gpu__time_duration.sum --disable-extra-suffixes)
    elif [[ $capture_set == full ]]; then
        sections=(--set full)
    elif [[ $capture_set == targeted ]]; then
        sections=(--section SpeedOfLight --section LaunchStats --section Occupancy
                  --section SchedulerStats --section WarpStateStats
                  --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis)
    else
        # These are the exact questions this pass must answer.  Keeping the
        # list narrow avoids the 14 replay passes required by the previous
        # broad section set.  Every metric below was present in the prior
        # TU102 report generated by the same installed Nsight Compute build.
        local -a focused_metric_names=(
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
            dram__bytes.sum.per_second
            dram__cycles_active.avg.pct_of_peak_sustained_elapsed
            l1tex__throughput.avg.pct_of_peak_sustained_active
            l1tex__m_xbar2l1tex_read_sectors.avg.pct_of_peak_sustained_elapsed
            l1tex__t_sector_hit_rate.pct
            lts__throughput.avg.pct_of_peak_sustained_elapsed
            lts__t_sectors.avg.pct_of_peak_sustained_elapsed
            lts__t_sector_hit_rate.pct
            lts__t_sectors_lookup_miss.sum
            smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
            smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
            smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
        )
        local focused_metrics
        local IFS=,
        focused_metrics="${focused_metric_names[*]}"
        sections=(--metrics "$focused_metrics" --disable-extra-suffixes)
    fi
    if [[ $scope_mode == ungated ]]; then
        # Nsight treats --profile-from-start and
        # --disable-profiler-start-stop as mutually exclusive. The latter
        # ignores application Start/Stop calls and profiles from launch.
        profiler_args=(--disable-profiler-start-stop)
        target_env=(env
            -u DS4_CUDA_NCU_TARGET_MODULE
            -u DS4_CUDA_NCU_TARGET_LAYER
            -u DS4_CUDA_NCU_TARGET_POS
            -u DS4_CUDA_NCU_TARGET_DEVICE
            "DS4_CUDA_Q8_CACHE_AUDIT_CSV=$base-cache.csv"
            "DS4_LOCK_FILE=$ncu_lock_file")
    elif [[ $scope_mode == targeted ]]; then
        profiler_args=(--profile-from-start off)
        target_env=(env
            "DS4_CUDA_NCU_TARGET_MODULE=$module"
            "DS4_CUDA_NCU_TARGET_LAYER=$layer"
            "DS4_CUDA_NCU_TARGET_POS=$pos"
            "DS4_CUDA_NCU_TARGET_DEVICE=$device"
            "DS4_CUDA_Q8_CACHE_AUDIT_CSV=$base-cache.csv"
            "DS4_LOCK_FILE=$ncu_lock_file")
    else
        die "invalid internal NCU scope mode: $scope_mode"
    fi
    printf 'Nsight Compute: %s (scope=%s module=%s layer=%s pos=%s device=%s; application replay)...\n' \
        "$name" "$scope_mode" "$module" "$layer" "$pos" "$device"
    printf '  Nsight will relaunch ds4-bench for each metric pass; details: %s.log\n' "$base"
    local rc=0
    "${target_env[@]}" "${ncu_command[@]}" --config-file off --verbose \
            --target-processes all \
            --devices "$device" \
            --filter-mode per-gpu "${profiler_args[@]}" \
            --kernel-name-base function --kernel-name "$kernel" \
            --launch-count 1 --replay-mode application \
            --app-replay-buffer file --app-replay-match grid \
            --app-replay-mode balanced --cache-control none \
            --clock-control none --forward-signals true \
            --force-overwrite --export "$base" "${sections[@]}" \
            ./ds4-bench "${bench_common[@]}" --prompt-file "$audit_prompt" \
                --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS" \
                --ctx-alloc "$((PROFILE_TOKENS + 1))" --step-incr "$PROFILE_TOKENS" \
                --prefill-chunk "$PREFILL_CHUNK" --csv "$base-benchmark.csv" \
                >"$base.log" 2>&1 || rc=$?
    if (( rc != 0 )); then
        printf 'error: Nsight Compute capture %s failed (exit %s); see %s.log\n' \
            "$name" "$rc" "$base" >&2
        if [[ $scope_mode == ungated ]]; then
            printf '%s\n' \
                'error: boundary=application-replay-root-attachment (DS4 profiler Start/Stop was disabled)' >&2
        else
            printf '%s\n' \
                'error: boundary=targeted-profiler-range (the ungated root-attachment preflight already passed)' >&2
        fi
        grep -q ERR_NVGPUCTRPERM "$base.log" &&
            printf 'error: rerun with NCU_USE_SUDO=1 for restricted counters\n' >&2 || true
        tail -n 80 "$base.log" >&2 || true
        return "$rc"
    fi
    if grep -Eq '==ERROR==|No kernels were profiled|Failed to (profile|create report)' \
            "$base.log"; then
        tail -n 80 "$base.log" >&2 || true
        if [[ $scope_mode == ungated ]]; then
            printf '%s\n' \
                'error: boundary=application-replay-root-attachment (DS4 profiler Start/Stop was disabled)' >&2
        else
            printf '%s\n' \
                'error: boundary=targeted-profiler-range (the ungated root-attachment preflight already passed)' >&2
        fi
        die "Nsight Compute reported a zero-kernel or failed capture for $name"
    fi
    if [[ $scope_mode == targeted ]]; then
        local start_marker="ds4: starting targeted CUDA profile module=$module layer=$layer pos=$pos"
        local stop_marker="ds4: stopped targeted CUDA profile module=$module layer=$layer pos=$pos"
        local device_marker="ds4: CUDA profiler scope active on physical device $device (explicit=true)"
        grep -Fq "$start_marker" "$base.log" || {
            tail -n 80 "$base.log" >&2 || true
            die "targeted profiler start marker is missing for $name"
        }
        grep -Fq "$device_marker" "$base.log" || {
            tail -n 80 "$base.log" >&2 || true
            die "targeted profiler device marker is missing for $name"
        }
        grep -Fq "$stop_marker" "$base.log" || {
            tail -n 80 "$base.log" >&2 || true
            die "targeted profiler stop marker is missing for $name"
        }
    fi
    [[ -s $base.ncu-rep ]] || die "Nsight Compute did not produce a nonempty $base.ncu-rep"
    "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
        >"$base.csv" 2>"$base-import.log" ||
        die "failed to import Nsight Compute report: $base.ncu-rep"
    [[ -s $base.csv ]] || die "Nsight Compute import produced an empty CSV: $base.csv"
    python3 speed-bench/validate-ncu-capture.py \
        "$base.csv" "${kernel#regex:}" "$device" \
        >"$base-validation.txt" 2>&1 || {
            cat "$base-validation.txt" >&2 || true
            die "Nsight Compute report validation failed for $name"
        }
    cat "$base-validation.txt"
}

if [[ $RUN_NCU == 1 ]]; then
    current_phase=nsight-compute
    command -v ncu >/dev/null 2>&1 || die "Nsight Compute CLI (ncu) not found"
    ncu_bin=$(command -v ncu)
    [[ $ncu_bin == /* && -x $ncu_bin ]] || die "cannot resolve ncu executable: $ncu_bin"
    ncu_command=("$ncu_bin")
    ncu_lock_file="$EVIDENCE_DIR/ncu/ds4-profile.lock"
    : >"$ncu_lock_file"
    if [[ $NCU_USE_SUDO == 1 ]]; then
        command -v sudo >/dev/null 2>&1 || die "sudo not found"
        sudo -v
        ncu_command=(sudo -E "$ncu_bin")
    fi

    printf 'Nsight Compute ungated attachment preflight (one duration metric)...\n'
    ncu_capture "preflight-ungated-first-q4-gate-up-tile8" \
        "${gpu_devices[0]}" routed_moe "$EARLY_LAYER" 0 \
        'regex:moe_gate_up_mid_q4K_tile8_mma_kernel.*' preflight ungated

    printf 'Nsight Compute targeted-range preflight (one duration metric)...\n'
    ncu_capture "preflight-targeted-early-layer${EARLY_LAYER}-q4-gate-up-tile8" \
        "${gpu_devices[0]}" routed_moe "$EARLY_LAYER" 0 \
        'regex:moe_gate_up_mid_q4K_tile8_mma_kernel.*' preflight targeted

    ncu_capture "early-layer${EARLY_LAYER}-q4-gate-up-tile8" \
        "${gpu_devices[0]}" routed_moe "$EARLY_LAYER" 0 \
        'regex:moe_gate_up_mid_q4K_tile8_mma_kernel.*'
    ncu_capture "late-layer${LATE_LAYER}-q4-gate-up-tile8" \
        "${gpu_devices[1]}" routed_moe "$LATE_LAYER" 0 \
        'regex:moe_gate_up_mid_q4K_tile8_mma_kernel.*'

    tail -n +2 "$EVIDENCE_DIR/coverage/native-q8-targets.tsv" |
    while IFS=$'\t' read -r kind module layer pos device label in_dim out_dim reason; do
        [[ -n $module ]] || continue
        name="${kind}-${module}-layer${layer}-pos${pos}-dense-q8"
        ncu_capture "$name" "$device" "$module" "$layer" "$pos" \
            'regex:matmul_q8_0_mma_sm75_exact_kernel.*'
        awk -F, -v m="$module" -v l="$layer" -v p="$pos" -v d="$device" '
            NR > 1 && $2 == m && $4 == l && $5 == p && $6 == d && $12 == "native_q8" { found=1 }
            END { exit found ? 0 : 1 }
        ' "$EVIDENCE_DIR/ncu/$name-cache.csv" ||
            die "NCU target $module layer $layer was not native Q8 in its capture run"
    done
fi

current_phase=complete
printf 'Full-Q4 evidence pass complete: %s\n' "$EVIDENCE_DIR"
