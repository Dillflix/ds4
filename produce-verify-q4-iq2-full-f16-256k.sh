#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Calibrate, produce, and verify a Q4/IQ2 routed-expert model that supports a
262144-token context allocation with complete dense-Q8 FP16 coverage.

Usage:
  bash produce-verify-q4-iq2-full-f16-256k.sh \
    HF_DIR FULL_Q4.gguf [OUTPUT.gguf] [RESULTS.csv]

The recipe is measured, not hardcoded. A known all-IQ2-gate/up + Q4-down
calibration model is loaded with the final 256K allocation, complete dense
FP16 coverage, and worst-case all-local T256 policy. Post-warm-up free VRAM is measured
on every device. The selector promotes as many matched gate/up layer pairs to
Q4_K as both devices in each NVLink stage can hold while preserving the CUDA
cache reserve plus the requested safety margin. Routed down remains Q4_K in
every layer. The output keeps tagged native SM75 packing on every routed
tensor that remains Q4_K.

Defaults:
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  DS4_CUDA_EP_STAGE_SPLIT=22
  TARGET_CONTEXT=262144
  TARGET_GEN_TOKENS=128
  CACHE_EXTRA_HEADROOM_MIB_PER_DEVICE=512
  EXPECTED_DENSE_CANDIDATES=344
  IQ2_GATE_UP_LAYER_ORDER=3-42
  CALIBRATION_MODEL=./gguf/DeepSeek-V4-Flash-0731-IQ2-IQ2-Q4.gguf

IQ2_GATE_UP_LAYER_ORDER is a preference order, not a sensitivity ranking. The
default is deterministic and makes no quality-optimality claim. Supply a
quality-ranked order when one is available, then run the fixed quality suite
against the produced model.

Resume controls:
  REUSE_BASELINE=1  reuse the all-IQ2 plan, layout, and memory calibration
  REUSE_MODEL=1     skip quantization and only rerun final verification
  OVERWRITE=1       replace an existing output atomically
  RUN_FULL_SWEEP=1  after verification, benchmark 2K..256K (prompt permitting)
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
calibration_model=${CALIBRATION_MODEL:-$script_dir/gguf/DeepSeek-V4-Flash-0731-IQ2-IQ2-Q4.gguf}
[[ -f $calibration_model ]] || die "all-IQ2 gate/up calibration GGUF not found: $calibration_model"
calibration_model=$(realpath "$calibration_model")
out=${3:-${OUT:-/mnt/nfs-images/models/gguf/ds4/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf}}
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
expected_candidates=${EXPECTED_DENSE_CANDIDATES:-344}
iq2_layer_order=${IQ2_GATE_UP_LAYER_ORDER:-3-42}
reuse_baseline=${REUSE_BASELINE:-0}
run_full_sweep=${RUN_FULL_SWEEP:-0}
prompt=${PROMPT_FILE:-$script_dir/speed-bench/promessi_sposi.txt}
prompt=$(realpath "$prompt")

[[ $target_context =~ ^[1-9][0-9]*$ ]] || die "TARGET_CONTEXT must be positive"
[[ $target_gen_tokens =~ ^[0-9]+$ ]] || die "TARGET_GEN_TOKENS must be nonnegative"
[[ $extra_headroom =~ ^[0-9]+$ ]] || die "CACHE_EXTRA_HEADROOM_MIB_PER_DEVICE must be nonnegative"
[[ $expected_candidates =~ ^[1-9][0-9]*$ ]] || die "EXPECTED_DENSE_CANDIDATES must be positive"
[[ $reuse_baseline == 0 || $reuse_baseline == 1 ]] || die "REUSE_BASELINE must be 0 or 1"
[[ $run_full_sweep == 0 || $run_full_sweep == 1 ]] || die "RUN_FULL_SWEEP must be 0 or 1"
[[ -f $prompt ]] || die "prompt file not found: $prompt"

export DS4_CUDA_EP_STAGE_SPLIT=${DS4_CUDA_EP_STAGE_SPLIT:-22}
export DS4_CUDA_TP_PREFILL_ATTN_HEADS=0
export DS4_CUDA_Q8_F16_ALL=1
export DS4_CUDA_Q8_F16_FREEZE_HOME_PLAN=1
export DS4_CUDA_Q8_F16_PARTNER_CLASSES=none
export DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=2048
export DS4_CUDA_Q8_T256_PLACEMENT=all-local
export DS4_CUDA_NO_Q8_F16_PARTNER_OFFLOAD=1
unset DS4_CUDA_Q8_F16_PARTNER_LAYERS
unset DS4_CUDA_Q8_PARTNER_ARITHMETIC
prefix=${out%.gguf}
baseline_plan=${BASELINE_PLAN_AUDIT:-$prefix.all-iq2-256k-plan.csv}
baseline_log=${BASELINE_LOG:-$prefix.all-iq2-256k-plan.log}
baseline_csv=${BASELINE_CSV:-$prefix.all-iq2-256k-calibration.csv}
baseline_memory=${BASELINE_MEMORY_STATE:-$prefix.all-iq2-256k-memory.csv}
baseline_bindings=${BASELINE_BINDING_STATE:-$prefix.all-iq2-256k-bindings.csv}
baseline_allocations=${BASELINE_ALLOCATION_STATE:-$prefix.all-iq2-256k-allocations.csv}
selection_manifest=${SELECTION_MANIFEST:-$prefix.selection.txt}
verification_plan=${VERIFICATION_PLAN_AUDIT:-$prefix.full-f16-verification.csv}
verification_bindings=${VERIFICATION_BINDINGS:-$prefix.full-f16-bindings.csv}
verification_allocations=${VERIFICATION_ALLOCATIONS:-$prefix.full-f16-allocations.csv}
verification_csv=${VERIFICATION_CSV:-$prefix.verification-bench.csv}
mkdir -p -- "$(dirname -- "$out")" "$(dirname -- "$results")"

ctx_alloc=$((target_context + target_gen_tokens + 1))
if [[ $reuse_baseline == 1 ]]; then
    [[ -s $baseline_plan ]] || die "REUSE_BASELINE=1 but plan is missing: $baseline_plan"
    [[ -s $baseline_log ]] || die "REUSE_BASELINE=1 but layout log is missing: $baseline_log"
    [[ -s $baseline_memory ]] || die "REUSE_BASELINE=1 but memory state is missing: $baseline_memory"
    [[ -s $baseline_bindings ]] || die "REUSE_BASELINE=1 but binding state is missing: $baseline_bindings"
    [[ -s $baseline_allocations ]] || die "REUSE_BASELINE=1 but allocation state is missing: $baseline_allocations"
    printf 'Reusing the all-IQ2 256K capacity calibration.\n'
else
    printf 'Building the SM75 engine for the all-IQ2 256K capacity calibration...\n'
    make -j "$make_jobs" ds4-bench CUDA_ARCH="$cuda_arch"
    printf 'Auditing all-IQ2 gate/up at ctx_alloc=%s with complete dense FP16...\n' "$ctx_alloc"
    DS4_BENCH_UNTIMED_WARMUP_TOKENS=2048 \
    DS4_CUDA_Q8_PLAN_AUDIT_CSV="$baseline_plan" \
    DS4_CUDA_Q8_BINDING_STATE_CSV="$baseline_bindings" \
    DS4_CUDA_Q8_ALLOCATION_STATE_CSV="$baseline_allocations" \
    DS4_CUDA_MEMORY_STATE_CSV="$baseline_memory" \
    ./ds4-bench \
        --cuda --cuda-tensor-parallel \
        --gpu-devices "$gpu_devices" --gpu-vram "$gpu_vram" \
        -m "$calibration_model" --prompt-file "$prompt" \
        --ctx-start 2048 --ctx-max 2048 --ctx-alloc "$ctx_alloc" \
        --step-incr 2048 --gen-tokens 0 --csv "$baseline_csv" \
        >"$baseline_log" 2>&1 || {
            tail -n 200 "$baseline_log" >&2
            die "all-IQ2 256K capacity calibration failed"
        }
    [[ -s $baseline_plan ]] || die "calibration did not produce $baseline_plan"
    [[ -s $baseline_memory ]] || die "calibration did not produce $baseline_memory"
    [[ -s $baseline_bindings ]] || die "calibration did not produce $baseline_bindings"
    [[ -s $baseline_allocations ]] || die "calibration did not produce $baseline_allocations"
    if grep -Fq 'q8 fp16 plan rejected' "$baseline_log"; then
        die "calibration cache plan was rejected; capacity selection is invalid"
    fi
fi

selection=$(python3 speed-bench/select-q4-iq2-full-f16.py \
    --plan-audit "$baseline_plan" \
    --layout-log "$baseline_log" \
    --device-memory "$baseline_memory" \
    --gpu-devices "$gpu_devices" \
    --extra-headroom-mib-per-device "$extra_headroom" \
    --iq2-layer-order "$iq2_layer_order" \
    --expected-candidates "$expected_candidates")
IFS=$'\t' read -r iq2_layers tensor_overrides <<< "$selection"
[[ -n $iq2_layers && -n $tensor_overrides ]] || die "selector returned an empty Q4/IQ2 plan"

{
    printf 'full_q4=%s\n' "$full_q4"
    printf 'all_iq2_calibration_model=%s\n' "$calibration_model"
    printf 'target_context=%s\n' "$target_context"
    printf 'target_gen_tokens=%s\n' "$target_gen_tokens"
    printf 'target_ctx_alloc=%s\n' "$ctx_alloc"
    printf 'gpu_devices=%s\n' "$gpu_devices"
    printf 'stage_split=%s\n' "$DS4_CUDA_EP_STAGE_SPLIT"
    printf 'dense_cache_policy=all-q8-f16\n'
    printf 'capacity_calibration_t256_placement=all-local\n'
    printf 'placement_target=neutral-worst-case-home-footprint\n'
    printf 'cache_extra_headroom_mib_per_device=%s\n' "$extra_headroom"
    printf 'expected_dense_candidates=%s\n' "$expected_candidates"
    printf 'routed_down_type=q4_k\n'
    printf 'iq2_gate_up_layer_order=%s\n' "$iq2_layer_order"
    printf 'iq2_layer_order_quality_ranked=false-unless-user-supplied\n'
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
export QUANT_RECIPE="q4-iq2-256k:iq2_gate_up_layers=$iq2_layers;down=q4_k"
export PLOT_TITLE="RTX 8000 2x2 TP - Q4/IQ2 complete dense FP16 at 256K"
export CTX_START=2048
export CTX_MAX=2048
export CTX_ALLOC="$ctx_alloc"
export STEP_INCR=2048
export STEP_MUL=1
export GEN_TOKENS=0
export PROMPT_FILE="$prompt"
export DS4_CUDA_Q8_PLAN_AUDIT_CSV="$verification_plan"
export DS4_CUDA_Q8_BINDING_STATE_CSV="$verification_bindings"
export DS4_CUDA_Q8_ALLOCATION_STATE_CSV="$verification_allocations"
export DS4_CUDA_MEMORY_STATE_CSV="$prefix.full-f16-memory.csv"
export DS4_BENCH_UNTIMED_WARMUP_TOKENS=2048

bash "$script_dir/produce-benchmark-iq2-iq2-q4.sh" \
    "$hf_dir" "$out" "$verification_csv"

python3 speed-bench/select-q4-iq2-full-f16.py \
    --plan-audit "$verification_plan" \
    --expected-candidates "$expected_candidates" \
    --verify-only
grep -Fq "q8 fp16 benefit plan materialized $expected_candidates/$expected_candidates candidates" \
    "${BENCH_LOG:-${out%.gguf}.bench.log}" || \
    die "runtime log does not prove complete dense FP16 materialization"
[[ -s $verification_bindings ]] || die "runtime omitted dense FP16 binding evidence"
[[ -s $verification_allocations ]] || die "runtime omitted dense FP16 allocation evidence"

if [[ $run_full_sweep == 1 ]]; then
    printf 'Running optional 2K..256K frontier sweep...\n'
    unset DS4_CUDA_Q8_PLAN_AUDIT_CSV
    unset DS4_CUDA_Q8_BINDING_STATE_CSV
    unset DS4_CUDA_Q8_ALLOCATION_STATE_CSV
    unset DS4_CUDA_MEMORY_STATE_CSV
    unset DS4_BENCH_UNTIMED_WARMUP_TOKENS
    ./ds4-bench \
        --cuda --cuda-tensor-parallel \
        --gpu-devices "$gpu_devices" --gpu-vram "$gpu_vram" \
        -m "$out" --prompt-file "$prompt" \
        --ctx-start 2048 --ctx-max "$target_context" \
        --ctx-alloc "$ctx_alloc" --step-mul 2 --step-incr 2048 \
        --gen-tokens "$target_gen_tokens" --csv "$results"
fi

printf '\nQ4/IQ2 256K model generated and cache-verified.\n'
printf 'Model:              %s\n' "$out"
printf 'IQ2 gate/up layers: %s\n' "$iq2_layers"
printf 'Q4 down layers:     3-42 (all routed layers)\n'
printf 'Dense FP16:         %s/%s candidates resident\n' \
    "$expected_candidates" "$expected_candidates"
printf 'Selection:          %s\n' "$selection_manifest"
printf 'Plan audit:         %s\n' "$verification_plan"
