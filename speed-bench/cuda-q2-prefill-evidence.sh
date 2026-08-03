#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Capture the stock-Q2 SM75 production evidence needed to rank architecture
optimization targets:
  1. the fixed, uninstrumented prompt-suite benchmark;
  2. non-perturbing routed-MoE tile coverage and complete Q8-cache coverage;
  3. one bounded Nsight Systems prefill trace with kernel/API summaries.

Nsight Compute is intentionally delegated to cuda-sm75-kernel-profile.sh so
the full model is never replayed under NCU.

Required environment:
  MODEL_Q2=/absolute/path/to/stock-Q2.gguf

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
  RUN_NSYS=1
  SKIP_BUILD=0
  SKIP_BASELINE=0
  SKIP_COVERAGE=0
  CREATE_ARCHIVE=1
  Q2_EVIDENCE_DIR=/absolute/path/output-directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
: "${MODEL_Q2:?set MODEL_Q2 to the absolute stock-Q2 GGUF path}"
[[ $MODEL_Q2 == /* ]] || die "MODEL_Q2 must be an absolute path"
[[ -f $MODEL_Q2 ]] || die "stock-Q2 model not found: $MODEL_Q2"

GPU_DEVICES=${GPU_DEVICES:-0,2,1,3}
GPU_VRAM=${GPU_VRAM:-auto}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
CTX_START=${CTX_START:-2048}
CTX_MAX=${CTX_MAX:-65536}
STEP_MUL=${STEP_MUL:-2}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
PROFILE_TOKENS=${PROFILE_TOKENS:-2048}
RUN_NSYS=${RUN_NSYS:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
SKIP_BASELINE=${SKIP_BASELINE:-0}
SKIP_COVERAGE=${SKIP_COVERAGE:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
EVIDENCE_DIR=${Q2_EVIDENCE_DIR:-$repo_dir/q2-prefill-evidence-$(date -u +%Y%m%dT%H%M%SZ)}

for tool in python3 nvidia-smi tar; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
for item in "CTX_START:$CTX_START" "CTX_MAX:$CTX_MAX" \
            "STEP_MUL:$STEP_MUL" "PREFILL_CHUNK:$PREFILL_CHUNK" \
            "PROFILE_TOKENS:$PROFILE_TOKENS"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( CTX_START > 0 && CTX_MAX >= CTX_START && STEP_MUL >= 1 &&
   PREFILL_CHUNK > 0 && PROFILE_TOKENS > 0 )) || die "invalid benchmark range"
for flag in RUN_NSYS SKIP_BUILD SKIP_BASELINE SKIP_COVERAGE CREATE_ARCHIVE; do
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
        printf 'state=finished\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
            "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >"$EVIDENCE_DIR/run-status.txt"
        if [[ $CREATE_ARCHIVE == 1 ]]; then
            local archive="$EVIDENCE_DIR.tar.gz"
            if tar -C "$(dirname "$EVIDENCE_DIR")" -czf "$archive" \
                    "$(basename "$EVIDENCE_DIR")"; then
                printf 'Archive to return: %s\n' "$archive"
            else
                status=1
            fi
        fi
    fi
    exit "$status"
}
trap finalize EXIT

resume=0
if [[ -e $EVIDENCE_DIR ]]; then
    if [[ $SKIP_BUILD == 1 || $SKIP_BASELINE == 1 || $SKIP_COVERAGE == 1 ]]; then
        resume=1
    else
        die "output path already exists: $EVIDENCE_DIR"
    fi
fi
mkdir -p "$EVIDENCE_DIR/runtime" "$EVIDENCE_DIR/coverage" "$EVIDENCE_DIR/nsys"
EVIDENCE_DIR=$(cd "$EVIDENCE_DIR" && pwd)
archive_ready=1
if [[ $resume == 1 && ! -s $EVIDENCE_DIR/manifest.txt ]]; then
    die "resume requested but manifest is missing: $EVIDENCE_DIR/manifest.txt"
fi
printf 'state=running\nlast_phase=%s\ndate_utc=%s\n' \
    "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >"$EVIDENCE_DIR/run-status.txt"

if [[ $SKIP_BUILD == 0 ]]; then
    current_phase=build
    make -B -j"$(nproc)" ds4-bench tests/test_engine_mgpu_placement \
        tests/cuda_sm75_profile_harness CUDA_ARCH="$CUDA_ARCH"
    ./tests/test_engine_mgpu_placement
fi
[[ -x ./ds4-bench ]] || die "ds4-bench is missing; rerun with SKIP_BUILD=0"

# Normalize every presence-based experimental switch. This is the production
# 22/21 baseline, not a dispatch experiment.
unset DS4_CUDA_MOE_PROFILE DS4_CUDA_ATTN_OUTPUT_PROFILE
unset DS4_METAL_LAYER_STAGE_PROFILE DS4_METAL_GRAPH_PREFILL_PROFILE
unset DS4_CUDA_PREFILL_PIPELINE_SEQUENTIAL DS4_CUDA_PREFILL_PIPELINE_SYNC_BOUNDARY
unset DS4_CUDA_PREFILL_TILE_AUDIT_CSV DS4_CUDA_Q8_CACHE_AUDIT_CSV
unset DS4_CUDA_NO_Q8_F16_CACHE DS4_CUDA_Q8_F32_ALL DS4_CUDA_Q8_F32_LARGE
unset DS4_CUDA_MOE_NO_IQ2_MMA_SM75 DS4_CUDA_MOE_NO_IQ2_MMA_TILE16_SM75
unset DS4_CUDA_MOE_NO_ATOMIC_DOWN DS4_CUDA_MOE_NO_DOWN_TILE16
unset DS4_CUDA_MOE_NO_DOWN_ROW2048 DS4_CUDA_MOE_DOWN_ROW1024
unset DS4_CUDA_MOE_DOWN_ROW2048 DS4_NSYS_CAPTURE_PREFILL
export DS4_CUDA_EP_STAGE_SPLIT=22
export DS4_CUDA_TP_PREFILL_ATTN_HEADS=0
export DS4_CUDA_PREFILL_PIPELINE=1
export DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
export DS4_CUDA_PREFILL_PIPELINE_MB=512

bench_common=(
    -m "$MODEL_Q2" --backend cuda --gpu-devices "$GPU_DEVICES"
    --gpu-vram "$GPU_VRAM" --cuda-tensor-parallel --gen-tokens 0
)

model_bytes=$(stat -c %s "$MODEL_Q2")
if [[ $resume == 0 ]]; then
    {
        printf 'date_utc=%s\nrepo=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
        printf 'git_commit=%s\ngit_branch=%s\n' "$(git rev-parse HEAD)" "$(git branch --show-current)"
        printf 'model_q2=%s\nmodel_bytes=%s\n' "$MODEL_Q2" "$model_bytes"
        printf 'expected_gate_up=IQ2_XXS\nexpected_down=Q2_K\n'
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
        nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,ecc.mode.current,driver_version --format=csv
        printf '\n[gpu topology]\n'; nvidia-smi topo -m
        printf '\n[nvcc]\n'; nvcc --version 2>&1 || true
        printf '\n[nsys]\n'; nsys --version 2>&1 || true
    } >"$EVIDENCE_DIR/manifest.txt"
else
    grep -Fqx "model_q2=$MODEL_Q2" "$EVIDENCE_DIR/manifest.txt" ||
        die "resume MODEL_Q2 differs from the initial manifest"
    grep -Fqx "model_bytes=$model_bytes" "$EVIDENCE_DIR/manifest.txt" ||
        die "resume model size differs from the initial manifest"
fi

if [[ $SKIP_BASELINE == 0 ]]; then
    current_phase=production-baseline
    for i in "${!prompt_paths[@]}"; do
        label=${prompt_labels[$i]}; prompt=${prompt_paths[$i]}
        printf 'Production stock-Q2 benchmark: %s...\n' "$label"
        ./ds4-bench "${bench_common[@]}" --prompt-file "$prompt" \
            --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
            --ctx-alloc "$((CTX_MAX + 1))" --step-mul "$STEP_MUL" \
            --prefill-chunk "$PREFILL_CHUNK" \
            --csv "$EVIDENCE_DIR/runtime/$label.csv" \
            >"$EVIDENCE_DIR/runtime/$label.log" 2>&1
    done
else
    for label in "${prompt_labels[@]}"; do
        [[ -s $EVIDENCE_DIR/runtime/$label.csv ]] ||
            die "SKIP_BASELINE=1 but runtime/$label.csv is missing"
    done
fi

audit_prompt=${prompt_paths[0]}
if [[ $SKIP_COVERAGE == 0 ]]; then
    current_phase=production-coverage
    printf 'Capturing stock-Q2 tile and Q8-cache coverage...\n'
    DS4_CUDA_PREFILL_TILE_AUDIT_CSV="$EVIDENCE_DIR/coverage/tile-audit.csv" \
    DS4_CUDA_Q8_CACHE_AUDIT_CSV="$EVIDENCE_DIR/coverage/q8-cache.csv" \
        ./ds4-bench "${bench_common[@]}" --prompt-file "$audit_prompt" \
            --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS" \
            --ctx-alloc "$((PROFILE_TOKENS + 1))" --step-incr "$PROFILE_TOKENS" \
            --prefill-chunk "$PREFILL_CHUNK" \
            --csv "$EVIDENCE_DIR/coverage/benchmark.csv" \
            >"$EVIDENCE_DIR/coverage/benchmark.log" 2>&1
    python3 speed-bench/summarize-tile-audit.py \
        "$EVIDENCE_DIR/coverage/tile-audit.csv" \
        "$EVIDENCE_DIR/coverage/tile-summary.csv" \
        >"$EVIDENCE_DIR/coverage/tile-summary.txt"
    python3 speed-bench/summarize-q8-cache-audit.py \
        "$EVIDENCE_DIR/coverage/q8-cache.csv" \
        "$EVIDENCE_DIR/coverage/q8-cache-summary.csv" \
        "$EVIDENCE_DIR/coverage/native-q8-targets.tsv" \
        --min-targets 0 \
        >"$EVIDENCE_DIR/coverage/q8-cache-summary.txt"
else
    for reused in tile-audit.csv tile-summary.csv q8-cache.csv \
                  q8-cache-summary.csv native-q8-targets.tsv; do
        [[ -s $EVIDENCE_DIR/coverage/$reused ]] ||
            die "SKIP_COVERAGE=1 but coverage/$reused is missing"
    done
fi

if [[ $RUN_NSYS == 1 ]]; then
    current_phase=nsight-systems
    command -v nsys >/dev/null 2>&1 || die "Nsight Systems CLI (nsys) not found"
    printf 'Capturing stock-Q2 production prefill with Nsight Systems...\n'
    export DS4_NSYS_CAPTURE_PREFILL=1
    nsys profile --force-overwrite=true --sample=none \
        --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi \
        --capture-range-end=stop --output="$EVIDENCE_DIR/nsys/stock-q2-prefill" \
        ./ds4-bench "${bench_common[@]}" --prompt-file "$audit_prompt" \
            --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS" \
            --ctx-alloc "$((PROFILE_TOKENS + 1))" --step-incr "$PROFILE_TOKENS" \
            --prefill-chunk "$PREFILL_CHUNK" \
            --csv "$EVIDENCE_DIR/nsys/benchmark.csv" \
            >"$EVIDENCE_DIR/nsys/capture.log" 2>&1
    unset DS4_NSYS_CAPTURE_PREFILL
    report_path="$EVIDENCE_DIR/nsys/stock-q2-prefill.nsys-rep"
    [[ -s $report_path ]] || die "Nsight Systems did not create $report_path"
    for report in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum \
                  cuda_gpu_trace nvtx_sum; do
        nsys stats --report "$report" --format csv "$report_path" \
            >"$EVIDENCE_DIR/nsys/$report.csv" \
            2>"$EVIDENCE_DIR/nsys/$report.log" || true
    done
    kernel_summary="$EVIDENCE_DIR/nsys/cuda_gpu_kern_sum.csv"
    [[ -s $kernel_summary ]] || die "Nsight Systems kernel summary is empty"
    grep -E 'moe_gate_up_mid_iq2_tile16_mma_sm75_kernel|moe_down_expert_tile16_rowspan_kernel|matmul_q8_0_mma_sm75_exact_kernel|ampere_|turing_' \
        "$kernel_summary" >"$EVIDENCE_DIR/nsys/relevant-runtime-branches.txt" || true
    grep -q 'moe_gate_up_mid_iq2_tile16_mma_sm75_kernel' "$kernel_summary" ||
        die "stock-Q2 trace did not execute the SM75 IQ2 gate/up tile16 kernel"
    grep -q 'moe_down_expert_tile16_rowspan_kernel' "$kernel_summary" ||
        die "stock-Q2 trace did not execute the Q2_K tile16 row-span down kernel"
    if grep -q 'moe_gate_up_mid_q4K' "$kernel_summary"; then
        die "trace contains Q4 gate/up; MODEL_Q2 is not the stock-Q2 recipe"
    fi
fi

current_phase=complete
printf 'Stock-Q2 SM75 evidence complete: %s\n' "$EVIDENCE_DIR"
