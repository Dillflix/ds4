#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Calibrate, generate, and verify the maximum-Q4 routed recipe whose complete
dense-Q8 FP16 cache plan is resident at a 262144-token context allocation.

Usage:
  bash produce-verify-q4-max-cache-256k.sh \
    HF_DIR FULL_Q4.gguf [OUTPUT.gguf] [RESULTS.csv]

The full-Q4 input is used only for a short capacity calibration and as GGUF
metadata. Tensor payloads are regenerated from HF_DIR. The generated file tags
and packs each remaining routed Q4 tensor for the SM75 native kernels; selected
Q2/IQ2 tensors retain their standard layouts.

Defaults:
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  DS4_CUDA_EP_STAGE_SPLIT=22
  TARGET_CONTEXT=262144
  TARGET_GEN_TOKENS=128
  CACHE_EXTRA_HEADROOM_MIB_PER_DEVICE=512
  Q4_TIE_BREAK=preserve-down
  RUN_FULL_BENCHMARK=0

Set RUN_FULL_BENCHMARK=1 after verification to sweep 2K..256K. The default run
quantizes once and performs only a 2K-token inference with the final 256K
allocation; this proves placement and complete cache residency without paying
for the full sweep.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
[[ $# -ge 2 && $# -le 4 ]] || { usage >&2; exit 2; }
[[ $(uname -s) == Linux ]] || die "this CUDA production script must run on Linux"
require_command make
require_command nproc
require_command python3
require_command realpath

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ -d $1 ]] || die "HF directory not found: $1"
[[ -f $2 ]] || die "full-Q4 GGUF not found: $2"
hf_dir=$(realpath "$1")
full_q4=$(realpath "$2")
out=${3:-${OUT:-/mnt/nfs-images/models/gguf/ds4/DeepSeek-V4-Flash-0731-Q4Max-FullF16Cache-256K-SM75.gguf}}
out=$(realpath -m "$out")
results=${4:-${RESULTS_CSV:-${out%.gguf}.bench.csv}}
results=$(realpath -m "$results")
cd "$script_dir"

gpu_devices=${GPU_DEVICES:-0,3,1,2}
gpu_vram=${GPU_VRAM:-auto}
cuda_arch=${CUDA_ARCH:-sm_75}
make_jobs=${MAKE_JOBS:-$(nproc)}
target_context=${TARGET_CONTEXT:-262144}
target_gen_tokens=${TARGET_GEN_TOKENS:-128}
extra_headroom=${CACHE_EXTRA_HEADROOM_MIB_PER_DEVICE:-512}
q2_layer_order=${Q2_DOWN_LAYER_ORDER:-3-42}
iq2_layer_order=${IQ2_GATE_UP_LAYER_ORDER:-$q2_layer_order}
tie_break=${Q4_TIE_BREAK:-preserve-down}
run_full_benchmark=${RUN_FULL_BENCHMARK:-0}
prompt=${PROMPT_FILE:-$script_dir/speed-bench/promessi_sposi.txt}
prompt=$(realpath "$prompt")

[[ $target_context =~ ^[1-9][0-9]*$ ]] || die "TARGET_CONTEXT must be positive"
[[ $target_gen_tokens =~ ^[0-9]+$ ]] || die "TARGET_GEN_TOKENS must be nonnegative"
[[ $extra_headroom =~ ^[0-9]+$ ]] || die "CACHE_EXTRA_HEADROOM_MIB_PER_DEVICE must be nonnegative"
[[ $run_full_benchmark == 0 || $run_full_benchmark == 1 ]] || die "RUN_FULL_BENCHMARK must be 0 or 1"
[[ $tie_break == preserve-down || $tie_break == prefer-q2 || $tie_break == least-overshoot ]] || \
    die "Q4_TIE_BREAK must be preserve-down, prefer-q2, or least-overshoot"
[[ -f $prompt ]] || die "prompt file not found: $prompt"

export DS4_CUDA_EP_STAGE_SPLIT=${DS4_CUDA_EP_STAGE_SPLIT:-22}
export DS4_CUDA_Q8_F16_ALL=1
prefix=${out%.gguf}
baseline_plan=${BASELINE_PLAN_AUDIT:-$prefix.full-q4-plan.csv}
baseline_log=${BASELINE_LOG:-$prefix.full-q4-plan.log}
baseline_csv=${BASELINE_CSV:-$prefix.full-q4-calibration.csv}
selection_manifest=${SELECTION_MANIFEST:-$prefix.selection.txt}
verification_plan=${VERIFICATION_PLAN_AUDIT:-$prefix.cache-verification.csv}
verification_csv=${VERIFICATION_CSV:-$prefix.verification-bench.csv}
mkdir -p -- "$(dirname -- "$out")" "$(dirname -- "$results")"

ctx_alloc=$((target_context + target_gen_tokens + 1))
printf 'Building the SM75 engine for 256K capacity calibration...\n'
make -j "$make_jobs" ds4-bench CUDA_ARCH="$cuda_arch"

printf 'Calibrating full Q4 at ctx_alloc=%s with all Q8 FP16 candidates enabled...\n' "$ctx_alloc"
DS4_CUDA_Q8_PLAN_AUDIT_CSV="$baseline_plan" \
./ds4-bench \
    --cuda --cuda-tensor-parallel \
    --gpu-devices "$gpu_devices" --gpu-vram "$gpu_vram" \
    --warm-weights -m "$full_q4" --prompt-file "$prompt" \
    --ctx-start 2048 --ctx-max 2048 --ctx-alloc "$ctx_alloc" \
    --step-incr 2048 --gen-tokens 0 --csv "$baseline_csv" \
    >"$baseline_log" 2>&1
[[ -s $baseline_plan ]] || die "full-Q4 calibration did not produce $baseline_plan"

selection=$(python3 speed-bench/select-q4-max-for-cache.py \
    --plan-audit "$baseline_plan" \
    --layout-log "$baseline_log" \
    --gpu-devices "$gpu_devices" \
    --extra-headroom-mib-per-device "$extra_headroom" \
    --q2-layer-order "$q2_layer_order" \
    --iq2-layer-order "$iq2_layer_order" \
    --tie-break "$tie_break")
IFS=$'\t' read -r q2_layers iq2_layers tensor_overrides <<< "$selection"
[[ -n $q2_layers && -n $iq2_layers && -n $tensor_overrides ]] || \
    die "selector returned an empty quantization plan"

{
    printf 'full_q4=%s\n' "$full_q4"
    printf 'target_context=%s\n' "$target_context"
    printf 'target_gen_tokens=%s\n' "$target_gen_tokens"
    printf 'target_ctx_alloc=%s\n' "$ctx_alloc"
    printf 'gpu_devices=%s\n' "$gpu_devices"
    printf 'stage_split=%s\n' "$DS4_CUDA_EP_STAGE_SPLIT"
    printf 'cache_policy=all-q8-f16\n'
    printf 'cache_extra_headroom_mib_per_device=%s\n' "$extra_headroom"
    printf 'objective=maximize-routed-q4-tensor-count\n'
    printf 'tie_break=%s\n' "$tie_break"
    printf 'q2_down_layer_order=%s\n' "$q2_layer_order"
    printf 'iq2_gate_up_layer_order=%s\n' "$iq2_layer_order"
    printf 'q2_down_layers=%s\n' "$q2_layers"
    printf 'iq2_gate_up_layers=%s\n' "$iq2_layers"
    printf 'tensor_type_overrides=%s\n' "$tensor_overrides"
} | tee "$selection_manifest"

export GPU_DEVICES="$gpu_devices"
export GPU_VRAM="$gpu_vram"
export CUDA_ARCH="$cuda_arch"
export MAKE_JOBS="$make_jobs"
export DS4_TEMPLATE_GGUF="$full_q4"
export ROUTED_W1=q4_k
export ROUTED_W2=q4_k
export ROUTED_W3=q4_k
export TENSOR_TYPE_OVERRIDES="$tensor_overrides"
export SM75_NATIVE_Q4=1
export QUANT_RECIPE="q4-max-256k:q2_down=$q2_layers;iq2_gate_up=$iq2_layers"
export PLOT_TITLE="RTX 8000 2x2 TP - max Q4 with complete 256K F16 cache"
export CTX_START=2048
export CTX_MAX=2048
export CTX_ALLOC="$ctx_alloc"
export STEP_INCR=2048
export STEP_MUL=1
export GEN_TOKENS=0
export DS4_CUDA_Q8_PLAN_AUDIT_CSV="$verification_plan"

bash "$script_dir/produce-benchmark-iq2-iq2-q4.sh" \
    "$hf_dir" "$out" "$verification_csv"

python3 speed-bench/select-q4-max-for-cache.py \
    --plan-audit "$verification_plan" --verify-only

if [[ $run_full_benchmark == 1 ]]; then
    printf 'Running the optional 2K..256K frontier sweep...\n'
    unset DS4_CUDA_Q8_PLAN_AUDIT_CSV
    ./ds4-bench \
        --cuda --cuda-tensor-parallel \
        --gpu-devices "$gpu_devices" --gpu-vram "$gpu_vram" \
        --warm-weights -m "$out" --prompt-file "$prompt" \
        --ctx-start 2048 --ctx-max "$target_context" \
        --ctx-alloc "$ctx_alloc" --step-mul 2 --step-incr 2048 \
        --gen-tokens "$target_gen_tokens" --csv "$results"
fi

printf '\nMaximum-Q4 256K model generated and cache-verified.\n'
printf 'Model:             %s\n' "$out"
printf 'Q2 down layers:    %s\n' "$q2_layers"
printf 'IQ2 gate/up layers:%s\n' "$iq2_layers"
printf 'Selection:         %s\n' "$selection_manifest"
printf 'Plan audit:        %s\n' "$verification_plan"
