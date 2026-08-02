#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Capture the evidence needed to tune four-GPU SM75 prefill without changing the
production defaults:
  1. device-only routed-MoE tile records, copied once per GPU after prefill;
  2. repeated 21/22 versus 25/18 pipeline-placement benchmarks;
  3. targeted Nsight Compute reports for early/late IQ2 gate-up and Q4 down.

Required environment:
  MODEL=/absolute/path/DeepSeek-V4-Flash-0731-IQ2-IQ2-Q4.gguf

Prompt suite (choose one):
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
  REPEATS=3
  PREFILL_MB=512
  PREFILL_CHUNK=2048
  TILE_TOKENS=2048
  TILE_AUDIT_CAPACITY=4096
  EARLY_LAYER=3
  LATE_LAYER=36
  RUN_NCU=1                 Set to 0 to omit Nsight Compute
  NCU_SET=targeted          Set to full for every Nsight Compute section
  SKIP_BUILD=1
  AUDIT_DIR=/absolute/path/output-directory
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

[[ ${1:-} != -h && ${1:-} != --help ]] || {
    usage
    exit 0
}

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

: "${MODEL:?set MODEL to the absolute GGUF path}"
[[ -f $MODEL ]] || die "model not found: $MODEL"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
command -v python3 >/dev/null 2>&1 || die "python3 not found"

GPU_DEVICES=${GPU_DEVICES:-0,2,1,3}
GPU_VRAM=${GPU_VRAM:-auto}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
CTX_START=${CTX_START:-2048}
CTX_MAX=${CTX_MAX:-65536}
STEP_MUL=${STEP_MUL:-2}
REPEATS=${REPEATS:-3}
PREFILL_MB=${PREFILL_MB:-512}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
TILE_TOKENS=${TILE_TOKENS:-2048}
TILE_AUDIT_CAPACITY=${TILE_AUDIT_CAPACITY:-4096}
EARLY_LAYER=${EARLY_LAYER:-3}
LATE_LAYER=${LATE_LAYER:-36}
RUN_NCU=${RUN_NCU:-1}
NCU_SET=${NCU_SET:-targeted}
AUDIT_DIR=${AUDIT_DIR:-$repo_dir/prefill-deep-audit-$(date -u +%Y%m%dT%H%M%SZ)}

for item in \
    "CTX_START:$CTX_START" "CTX_MAX:$CTX_MAX" "REPEATS:$REPEATS" \
    "PREFILL_MB:$PREFILL_MB" "PREFILL_CHUNK:$PREFILL_CHUNK" \
    "TILE_TOKENS:$TILE_TOKENS" \
    "TILE_AUDIT_CAPACITY:$TILE_AUDIT_CAPACITY" \
    "EARLY_LAYER:$EARLY_LAYER" "LATE_LAYER:$LATE_LAYER"; do
    name=${item%%:*}
    value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( CTX_START > 0 && CTX_MAX >= CTX_START )) || die "invalid context range"
(( REPEATS > 0 && PREFILL_MB > 0 && PREFILL_CHUNK > PREFILL_MB &&
   TILE_TOKENS > PREFILL_MB && TILE_TOKENS <= PREFILL_CHUNK )) ||
    die "REPEATS/PREFILL_MB/PREFILL_CHUNK/TILE_TOKENS are inconsistent"
(( EARLY_LAYER >= 0 && EARLY_LAYER < 21 )) ||
    die "EARLY_LAYER must be in the baseline first stage (0..20)"
(( LATE_LAYER >= 21 && LATE_LAYER < 43 )) ||
    die "LATE_LAYER must be in the baseline second stage (21..42)"
[[ $RUN_NCU == 0 || $RUN_NCU == 1 ]] || die "RUN_NCU must be 0 or 1"
[[ $NCU_SET == targeted || $NCU_SET == full ]] ||
    die "NCU_SET must be targeted or full"

IFS=',' read -r -a gpu_devices <<<"$GPU_DEVICES"
(( ${#gpu_devices[@]} == 4 )) ||
    die "this audit requires four GPU_DEVICES in logical tier order"

declare -a prompt_labels=()
declare -a prompt_paths=()
if [[ -n ${PROMPT_MANIFEST:-} ]]; then
    [[ -f $PROMPT_MANIFEST ]] || die "prompt manifest not found: $PROMPT_MANIFEST"
    while IFS=$'\t' read -r label path extra; do
        [[ -n $label && ${label:0:1} != "#" ]] || continue
        [[ -n $path && -z ${extra:-} ]] ||
            die "invalid prompt manifest row for $label"
        [[ $path == /* ]] || path="$repo_dir/$path"
        [[ -f $path ]] || die "prompt not found: $path"
        [[ $label =~ ^[A-Za-z0-9._-]+$ ]] ||
            die "prompt label must be filename-safe: $label"
        prompt_labels+=("$label")
        prompt_paths+=("$path")
    done <"$PROMPT_MANIFEST"
else
    prompt=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
    [[ -f $prompt ]] || die "prompt not found: $prompt"
    prompt_labels+=("promessi")
    prompt_paths+=("$prompt")
fi
(( ${#prompt_paths[@]} > 0 )) || die "prompt suite is empty"

mkdir -p "$AUDIT_DIR" "$AUDIT_DIR/placement" "$AUDIT_DIR/tile" "$AUDIT_DIR/ncu"
AUDIT_DIR=$(cd "$AUDIT_DIR" && pwd)

if [[ ${SKIP_BUILD:-0} != 1 ]]; then
    make -B -j"$(nproc)" ds4-bench tests/test_engine_mgpu_placement \
        CUDA_ARCH="$CUDA_ARCH"
    ./tests/test_engine_mgpu_placement
fi
[[ -x ./ds4-bench ]] || die "ds4-bench is missing; run without SKIP_BUILD=1"

# Lock every run to the same proved production path. Presence-based diagnostic
# flags must be unset, not assigned zero.
unset DS4_CUDA_MOE_PROFILE
unset DS4_CUDA_ATTN_OUTPUT_PROFILE
unset DS4_METAL_LAYER_STAGE_PROFILE
unset DS4_METAL_GRAPH_PREFILL_PROFILE
unset DS4_CUDA_PREFILL_PIPELINE_SEQUENTIAL
unset DS4_CUDA_PREFILL_PIPELINE_SYNC_BOUNDARY
unset DS4_CUDA_PREFILL_TILE_AUDIT_CSV
unset DS4_CUDA_PREFILL_TILE_AUDIT_CAPACITY
unset DS4_NSYS_CAPTURE_PREFILL
unset DS4_CUDA_MOE_NO_Q4_MMA
unset DS4_CUDA_MOE_NO_Q4_MMA_TILE16
unset DS4_CUDA_MOE_NO_IQ2_MMA_TILE16_SM75
unset DS4_CUDA_MOE_NO_Q4_MMA_TILE16_SM75
export DS4_CUDA_TP_PREFILL_ATTN_HEADS=0
export DS4_CUDA_PREFILL_PIPELINE=1
export DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
export DS4_CUDA_PREFILL_PIPELINE_MB="$PREFILL_MB"

bench_common=(
    -m "$MODEL"
    --backend cuda
    --gpu-devices "$GPU_DEVICES"
    --gpu-vram "$GPU_VRAM"
    --cuda-tensor-parallel
    --gen-tokens 0
)

{
    printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'repo=%s\n' "$repo_dir"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'git_branch=%s\n' "$(git branch --show-current)"
    printf 'model=%s\n' "$MODEL"
    printf 'model_bytes=%s\n' "$(stat -c %s "$MODEL")"
    printf 'gpu_devices=%s\n' "$GPU_DEVICES"
    printf 'gpu_vram=%s\n' "$GPU_VRAM"
    printf 'cuda_arch=%s\n' "$CUDA_ARCH"
    printf 'ctx_start=%s\nctx_max=%s\nstep_mul=%s\n' \
        "$CTX_START" "$CTX_MAX" "$STEP_MUL"
    printf 'repeats=%s\nprefill_mb=%s\nprefill_chunk=%s\n' \
        "$REPEATS" "$PREFILL_MB" "$PREFILL_CHUNK"
    printf 'tile_tokens=%s\ntile_audit_capacity=%s\n' \
        "$TILE_TOKENS" "$TILE_AUDIT_CAPACITY"
    printf 'early_layer=%s\nlate_layer=%s\nrun_ncu=%s\nncu_set=%s\n' \
        "$EARLY_LAYER" "$LATE_LAYER" "$RUN_NCU" "$NCU_SET"
    printf '\n[prompts]\n'
    for i in "${!prompt_paths[@]}"; do
        printf '%s\t%s\n' "${prompt_labels[$i]}" "${prompt_paths[$i]}"
    done
    printf '\n[git status]\n'
    git status --short
    printf '\n[nvidia-smi inventory]\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,ecc.mode.current,driver_version --format=csv
    printf '\n[nvidia-smi topology]\n'
    nvidia-smi topo -m
    printf '\n[nvcc]\n'
    nvcc --version 2>&1 || true
    printf '\n[ncu]\n'
    ncu --version 2>&1 || true
    printf '\n[ncu sections]\n'
    ncu --list-sections 2>&1 || true
} >"$AUDIT_DIR/manifest.txt"

runs_tsv="$AUDIT_DIR/placement-runs.tsv"
printf 'split\tprompt\trepeat\tcsv\n' >"$runs_tsv"
for repeat in $(seq 1 "$REPEATS"); do
    if (( repeat % 2 == 1 )); then
        split_order=(21 25)
    else
        split_order=(25 21)
    fi
    for i in "${!prompt_paths[@]}"; do
        label=${prompt_labels[$i]}
        prompt=${prompt_paths[$i]}
        for split in "${split_order[@]}"; do
            stem="split${split}-${label}-r${repeat}"
            csv="$AUDIT_DIR/placement/$stem.csv"
            log="$AUDIT_DIR/placement/$stem.log"
            printf 'Benchmarking split %s/%s, prompt %s, repeat %s/%s...\n' \
                "$split" "$((43 - split))" "$label" "$repeat" "$REPEATS"
            DS4_CUDA_EP_STAGE_SPLIT="$split" \
                ./ds4-bench "${bench_common[@]}" \
                    --prompt-file "$prompt" \
                    --ctx-start "$CTX_START" \
                    --ctx-max "$CTX_MAX" \
                    --ctx-alloc "$((CTX_MAX + 1))" \
                    --step-mul "$STEP_MUL" \
                    --prefill-chunk "$PREFILL_CHUNK" \
                    --csv "$csv" \
                    >"$log" 2>&1
            printf '%s\t%s\t%s\t%s\n' "$split" "$label" "$repeat" "$csv" \
                >>"$runs_tsv"
        done
    done
done

python3 "$repo_dir/speed-bench/summarize-prefill-placement.py" \
    "$runs_tsv" "$AUDIT_DIR/placement-summary.csv" \
    | tee "$AUDIT_DIR/placement-summary.txt"

audit_prompt=${prompt_paths[0]}
printf 'Capturing deferred tile counters for the baseline 21/22 split...\n'
DS4_CUDA_EP_STAGE_SPLIT=21 \
DS4_CUDA_PREFILL_TILE_AUDIT_CSV="$AUDIT_DIR/tile/tile-audit.csv" \
DS4_CUDA_PREFILL_TILE_AUDIT_CAPACITY="$TILE_AUDIT_CAPACITY" \
    ./ds4-bench "${bench_common[@]}" \
        --prompt-file "$audit_prompt" \
        --ctx-start "$TILE_TOKENS" \
        --ctx-max "$TILE_TOKENS" \
        --ctx-alloc "$((TILE_TOKENS + 1))" \
        --step-incr "$TILE_TOKENS" \
        --prefill-chunk "$PREFILL_CHUNK" \
        --csv "$AUDIT_DIR/tile/benchmark.csv" \
        >"$AUDIT_DIR/tile/benchmark.log" 2>&1
python3 "$repo_dir/speed-bench/summarize-tile-audit.py" \
    "$AUDIT_DIR/tile/tile-audit.csv" "$AUDIT_DIR/tile/tile-summary.csv" \
    >"$AUDIT_DIR/tile/tile-summary.txt"

ncu_capture() {
    local label=$1
    local physical_device=$2
    local kernel_regex=$3
    local launch_skip=$4
    local report_base="$AUDIT_DIR/ncu/$label"
    local -a section_args
    if [[ $NCU_SET == full ]]; then
        section_args=(--set full)
    else
        section_args=(
            --section SpeedOfLight
            --section LaunchStats
            --section Occupancy
            --section SchedulerStats
            --section WarpStateStats
            --section MemoryWorkloadAnalysis
            --section ComputeWorkloadAnalysis
        )
    fi
    printf 'Nsight Compute: %s on physical GPU %s (matching launch %s)...\n' \
        "$label" "$physical_device" "$launch_skip"
    DS4_CUDA_EP_STAGE_SPLIT=21 \
        ncu \
            --target-processes all \
            --devices "$physical_device" \
            --filter-mode per-gpu \
            --kernel-name-base function \
            --kernel-name "$kernel_regex" \
            --launch-skip "$launch_skip" \
            --launch-count 1 \
            --replay-mode kernel \
            --clock-control none \
            --force-overwrite \
            --export "$report_base" \
            "${section_args[@]}" \
            ./ds4-bench "${bench_common[@]}" \
                --prompt-file "$audit_prompt" \
                --ctx-start "$TILE_TOKENS" \
                --ctx-max "$TILE_TOKENS" \
                --ctx-alloc "$((TILE_TOKENS + 1))" \
                --step-incr "$TILE_TOKENS" \
                --prefill-chunk "$PREFILL_CHUNK" \
                --csv "$report_base-benchmark.csv" \
                >"$report_base.log" 2>&1
    local report="$report_base.ncu-rep"
    [[ -f $report ]] || die "Nsight Compute did not produce $report"
    ncu --import "$report" --csv --page raw \
        >"$report_base.csv" 2>"$report_base-import.log" || true
}

if [[ $RUN_NCU == 1 ]]; then
    command -v ncu >/dev/null 2>&1 || die "Nsight Compute CLI (ncu) not found"
    ncu_capture \
        "early-layer${EARLY_LAYER}-iq2-gate-up" \
        "${gpu_devices[0]}" \
        'regex:moe_gate_up_mid_iq2_tile16_mma_sm75_kernel.*' \
        "$EARLY_LAYER"
    ncu_capture \
        "early-layer${EARLY_LAYER}-q4-down" \
        "${gpu_devices[0]}" \
        'regex:moe_down_q4K_tile16_mma_sm75_kernel.*' \
        "$EARLY_LAYER"
    ncu_capture \
        "late-layer${LATE_LAYER}-iq2-gate-up" \
        "${gpu_devices[1]}" \
        'regex:moe_gate_up_mid_iq2_tile16_mma_sm75_kernel.*' \
        "$((LATE_LAYER - 21))"
    ncu_capture \
        "late-layer${LATE_LAYER}-q4-down" \
        "${gpu_devices[1]}" \
        'regex:moe_down_q4K_tile16_mma_sm75_kernel.*' \
        "$((LATE_LAYER - 21))"
fi

archive="$AUDIT_DIR.tar.gz"
tar -C "$(dirname "$AUDIT_DIR")" -czf "$archive" "$(basename "$AUDIT_DIR")"
printf 'Deep prefill audit complete: %s\n' "$AUDIT_DIR"
printf 'Archive to return: %s\n' "$archive"
