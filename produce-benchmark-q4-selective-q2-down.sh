#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Generate and benchmark a DeepSeek-V4-Flash Q4-first model with the minimum
number of Q2_K routed-down tensors needed to make the dense Q8 FP16 cache fully
resident. If down-only conversion is impossible for a stage, add the minimum
number of matched IQ2_XXS routed gate/up layer pairs needed to close the gap.

Usage:
  bash produce-benchmark-q4-selective-q2-down.sh \
    HF_DIR FULL_Q4.gguf [OUTPUT.gguf] [RESULTS.csv]

The script performs a short full-Q4 cache audit using the final context
allocation, selects Q2_K down tensors independently for each pipeline stage,
quantizes from HF source, benchmarks only the new model, and fails if its audit
still contains a budget-limited native-Q8 fallback.

Defaults:
  GPU_DEVICES=0,2,1,3
  GPU_VRAM=auto
  CUDA_ARCH=sm_75
  DS4_CUDA_EP_STAGE_SPLIT=22
  CTX_MAX=65536
  GEN_TOKENS=128
  CACHE_ALL_Q8=1
  CACHE_EXTRA_HEADROOM_MIB=512
  Q2_DOWN_LAYER_ORDER=3-42
  IQ2_GATE_UP_LAYER_ORDER=Q2_DOWN_LAYER_ORDER

Q2_DOWN_LAYER_ORDER is a preference order, not merely a set. Replace it with a
quality-ranked layer order when the fixed quality suite supplies one.
IQ2_GATE_UP_LAYER_ORDER independently controls the IQ2 pair preference.
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
[[ $(uname -s) == Linux ]] || die "this production/CUDA script must run on Linux"
require_command make
require_command nproc
require_command python3
require_command realpath

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ -d $1 ]] || die "HF directory not found: $1"
[[ -f $2 ]] || die "full-Q4 GGUF not found: $2"
hf_dir=$(realpath "$1")
full_q4=$(realpath "$2")
if [[ $# -ge 3 ]]; then
    out=$(realpath -m "$3")
elif [[ -n ${OUT:-} ]]; then
    out=$(realpath -m "$OUT")
else
    out=$script_dir/gguf/DeepSeek-V4-Flash-0731-Q4GateUp-SelectiveQ2Down.gguf
fi
csv=${4:-${RESULTS_CSV:-${out%.gguf}.bench.csv}}
csv=$(realpath -m "$csv")
cd "$script_dir"

gpu_devices=${GPU_DEVICES:-0,2,1,3}
gpu_vram=${GPU_VRAM:-auto}
cuda_arch=${CUDA_ARCH:-sm_75}
make_jobs=${MAKE_JOBS:-$(nproc)}
ctx_max=${CTX_MAX:-65536}
gen_tokens=${GEN_TOKENS:-128}
extra_headroom=${CACHE_EXTRA_HEADROOM_MIB:-512}
cache_all_q8=${CACHE_ALL_Q8:-1}
layer_order=${Q2_DOWN_LAYER_ORDER:-3-42}
iq2_layer_order=${IQ2_GATE_UP_LAYER_ORDER:-$layer_order}
if [[ -n ${PROMPT_FILE:-} ]]; then
    prompt=$(realpath "$PROMPT_FILE")
else
    prompt=$script_dir/speed-bench/promessi_sposi.txt
fi
[[ -f $prompt ]] || die "benchmark prompt not found: $prompt"
[[ $ctx_max =~ ^[1-9][0-9]*$ ]] || die "CTX_MAX must be a positive integer"
[[ $gen_tokens =~ ^[0-9]+$ ]] || die "GEN_TOKENS must be a nonnegative integer"
[[ $extra_headroom =~ ^[0-9]+$ ]] || die "CACHE_EXTRA_HEADROOM_MIB must be nonnegative"
[[ $cache_all_q8 == 0 || $cache_all_q8 == 1 ]] || die "CACHE_ALL_Q8 must be 0 or 1"

if [[ $cache_all_q8 == 1 ]]; then
    # Make every dense Q8 weight consulted by an FP16-capable runtime path
    # eligible. The final audit below then verifies the literal complete set,
    # rather than only DS4's default shape/label allow-list.
    export DS4_CUDA_Q8_F16_ALL=1
else
    unset DS4_CUDA_Q8_F16_ALL
fi

export DS4_CUDA_EP_STAGE_SPLIT=${DS4_CUDA_EP_STAGE_SPLIT:-22}
prefix=${out%.gguf}
baseline_audit=${BASELINE_CACHE_AUDIT:-$prefix.full-q4-cache.csv}
baseline_log=${BASELINE_CACHE_LOG:-$prefix.full-q4-cache.log}
baseline_bench=${BASELINE_CACHE_BENCH:-$prefix.full-q4-cache-bench.csv}
selection_manifest=${SELECTION_MANIFEST:-$prefix.selection.txt}
verification_audit=${VERIFICATION_CACHE_AUDIT:-$prefix.cache-verification.csv}
mkdir -p -- "$(dirname -- "$out")" "$(dirname -- "$csv")"

printf 'Building the SM75 benchmark for the full-Q4 cache calibration...\n'
make -j "$make_jobs" ds4-bench CUDA_ARCH="$cuda_arch"

ctx_alloc=$((ctx_max + gen_tokens + 1))
printf 'Auditing full Q4 with target context allocation %s...\n' "$ctx_alloc"
DS4_CUDA_Q8_CACHE_AUDIT_CSV="$baseline_audit" \
./ds4-bench \
    -m "$full_q4" --cuda --cuda-tensor-parallel \
    --gpu-devices "$gpu_devices" --gpu-vram "$gpu_vram" \
    --prompt-file "$prompt" --ctx-start 2048 --ctx-max 2048 \
    --ctx-alloc "$ctx_alloc" --step-incr 2048 --gen-tokens 0 \
    --csv "$baseline_bench" >"$baseline_log" 2>&1
[[ -s $baseline_audit ]] || die "full-Q4 audit did not produce $baseline_audit"

selection=$(python3 speed-bench/select-q2-down-for-cache.py \
    --audit "$baseline_audit" \
    --layout-log "$baseline_log" \
    --gpu-devices "$gpu_devices" \
    --extra-headroom-mib "$extra_headroom" \
    --layer-order "$layer_order" \
    --iq2-layer-order "$iq2_layer_order")
IFS=$'\t' read -r q2_layers iq2_gate_up_layers tensor_overrides <<< "$selection"
[[ -n $q2_layers && -n $iq2_gate_up_layers && -n $tensor_overrides ]] || \
    die "empty selective quant plan"

{
    printf 'full_q4=%s\n' "$(realpath "$full_q4")"
    printf 'baseline_audit=%s\n' "$(realpath "$baseline_audit")"
    printf 'target_ctx_alloc=%s\n' "$ctx_alloc"
    printf 'gpu_devices=%s\n' "$gpu_devices"
    printf 'stage_split=%s\n' "$DS4_CUDA_EP_STAGE_SPLIT"
    printf 'cache_all_q8=%s\n' "$cache_all_q8"
    printf 'cache_extra_headroom_mib=%s\n' "$extra_headroom"
    printf 'q2_down_layer_order=%s\n' "$layer_order"
    printf 'iq2_gate_up_layer_order=%s\n' "$iq2_layer_order"
    printf 'q2_down_layers=%s\n' "$q2_layers"
    printf 'iq2_gate_up_layers=%s\n' "$iq2_gate_up_layers"
    printf 'tensor_type_overrides=%s\n' "$tensor_overrides"
} | tee "$selection_manifest"

export ROUTED_W1=q4_k
export ROUTED_W2=q4_k
export ROUTED_W3=q4_k
export TENSOR_TYPE_OVERRIDES="$tensor_overrides"
export QUANT_RECIPE="base:gate=q4_k,up=q4_k,down=q4_k;q2_down_layers:$q2_layers;iq2_gate_up_layers:$iq2_gate_up_layers"
export PLOT_TITLE="RTX 8000 2x2 TP - Q4-first selective Q2 down + IQ2 gate/up"
export DS4_CUDA_Q8_CACHE_AUDIT_CSV="$verification_audit"

bash "$script_dir/produce-benchmark-iq2-iq2-q4.sh" "$hf_dir" "$out" "$csv"

python3 speed-bench/select-q2-down-for-cache.py \
    --audit "$verification_audit" --verify-only

printf '\nSelective model verified.\nModel:          %s\nQ2 down layers: %s\nIQ2 G/U layers: %s\nSelection:      %s\nAudit:          %s\n' \
    "$out" "$q2_layers" "$iq2_gate_up_layers" "$selection_manifest" "$verification_audit"
