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
  NCU_SET=targeted          targeted or full
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
NCU_SET=${NCU_SET:-targeted}
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
[[ $NCU_SET == targeted || $NCU_SET == full ]] || die "NCU_SET must be targeted or full"
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
        {
            printf 'exit_status=%s\nlast_phase=%s\n' "$status" "$current_phase"
            printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        } >"$EVIDENCE_DIR/run-status.txt"
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

if [[ $SKIP_BUILD == 0 ]]; then
    current_phase=build
    make -B -j"$(nproc)" ds4-bench tests/test_engine_mgpu_placement CUDA_ARCH="$CUDA_ARCH"
    ./tests/test_engine_mgpu_placement
fi
[[ -x ./ds4-bench ]] || die "ds4-bench is missing"

# Presence-based debug flags must be absent. The cache and 22/21 placement are
# the production configuration being measured.
unset DS4_CUDA_MOE_PROFILE DS4_CUDA_ATTN_OUTPUT_PROFILE
unset DS4_METAL_LAYER_STAGE_PROFILE DS4_METAL_GRAPH_PREFILL_PROFILE
unset DS4_CUDA_PREFILL_PIPELINE_SEQUENTIAL DS4_CUDA_PREFILL_PIPELINE_SYNC_BOUNDARY
unset DS4_CUDA_PREFILL_TILE_AUDIT_CSV DS4_CUDA_Q8_CACHE_AUDIT_CSV
unset DS4_CUDA_NCU_TARGET_MODULE DS4_CUDA_NCU_TARGET_LAYER DS4_CUDA_NCU_TARGET_POS
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

{
    printf 'date_utc=%s\nrepo=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
    printf 'git_commit=%s\ngit_branch=%s\n' "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\n' "$MODEL" "$(stat -c %s "$MODEL")"
    printf 'gpu_devices=%s\ngpu_vram=%s\ncuda_arch=%s\nsplit=22/21\n' \
        "$GPU_DEVICES" "$GPU_VRAM" "$CUDA_ARCH"
    printf 'ctx_start=%s\nctx_max=%s\nstep_mul=%s\nprefill_chunk=%s\nprofile_tokens=%s\n' \
        "$CTX_START" "$CTX_MAX" "$STEP_MUL" "$PREFILL_CHUNK" "$PROFILE_TOKENS"
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
} >"$EVIDENCE_DIR/manifest.txt"

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
    [[ -f $EVIDENCE_DIR/coverage/native-q8-targets.tsv ]] ||
        die "SKIP_COVERAGE=1 but native-q8-targets.tsv is missing"
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
    local base="$EVIDENCE_DIR/ncu/$name"
    local -a sections
    if [[ $NCU_SET == full ]]; then sections=(--set full); else
        sections=(--section SpeedOfLight --section LaunchStats --section Occupancy
                  --section SchedulerStats --section WarpStateStats
                  --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis)
    fi
    printf 'Nsight Compute: %s (module=%s layer=%s pos=%s device=%s)...\n' \
        "$name" "$module" "$layer" "$pos" "$device"
    local rc=0
    DS4_CUDA_NCU_TARGET_MODULE="$module" DS4_CUDA_NCU_TARGET_LAYER="$layer" \
    DS4_CUDA_NCU_TARGET_POS="$pos" \
    DS4_CUDA_Q8_CACHE_AUDIT_CSV="$base-cache.csv" DS4_LOCK_FILE="$ncu_lock_file" \
        "${ncu_command[@]}" --target-processes all --devices "$device" \
            --filter-mode per-gpu --profile-from-start off \
            --kernel-name-base function --kernel-name "$kernel" \
            --launch-count 1 --replay-mode kernel --clock-control none \
            --force-overwrite --export "$base" "${sections[@]}" \
            ./ds4-bench "${bench_common[@]}" --prompt-file "$audit_prompt" \
                --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS" \
                --ctx-alloc "$((PROFILE_TOKENS + 1))" --step-incr "$PROFILE_TOKENS" \
                --prefill-chunk "$PREFILL_CHUNK" --csv "$base-benchmark.csv" \
                >"$base.log" 2>&1 || rc=$?
    if (( rc != 0 )); then
        printf 'error: Nsight Compute capture %s failed (exit %s); see %s.log\n' \
            "$name" "$rc" "$base" >&2
        grep -q ERR_NVGPUCTRPERM "$base.log" &&
            printf 'error: rerun with NCU_USE_SUDO=1 for restricted counters\n' >&2 || true
        return "$rc"
    fi
    [[ -f $base.ncu-rep ]] || die "Nsight Compute did not produce $base.ncu-rep"
    ncu --import "$base.ncu-rep" --csv --page raw >"$base.csv" 2>"$base-import.log" || true
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
