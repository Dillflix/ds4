#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Calibrate, produce, and verify a Q4/IQ2 routed-expert model that supports a
262144-token context allocation with complete dense-Q8 FP16 coverage.

Usage:
  bash produce-verify-q4-iq2-full-f16-256k.sh \
    HF_DIR [OUTPUT.gguf] [RESULTS.csv]

The recipe is measured, not hardcoded. A known all-IQ2-gate/up + Q4-down
calibration model is loaded with the final 256K allocation, complete dense
FP16 coverage, and worst-case all-local T256 policy. Post-warm-up free VRAM is measured
on every device. The selector promotes as many matched gate/up layer pairs to
Q4_K as both devices in each NVLink stage can hold while preserving the CUDA
cache reserve plus the requested safety margin. Routed down remains Q4_K in
every layer. The output keeps tagged native SM75 packing on every routed
tensor that remains Q4_K. Final verification uses the worst-case all-local
T256 capacity envelope and requires every one of the 344 dense candidates to
remain resident. Runtime T256 placement is deliberately left to the separate
native/all-local/balanced/all-partner production A/B; model generation does
not assume that all-partner remains optimal after the routed quant changes.
The final GGUF path is published atomically only after that verification
passes; a failed benchmark or incomplete cache plan cannot leave an
unverified model at OUTPUT.gguf.

No full-Q4 GGUF is required. The quantizer regenerates tensors from HF_DIR,
using the automatically cached 8 MiB metadata template and routed-MoE imatrix.
If CALIBRATION_MODEL does not exist, the runner first creates a disposable
all-IQ2-gate/up + Q4-down calibration model from the same HF checkpoint. It is
removed after capacity selection, before final quantization, unless
KEEP_GENERATED_CALIBRATION=1. The persisted calibration evidence supports a
REUSE_BASELINE=1 retry without regenerating the disposable model.

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
  GENERATED_CALIBRATION_MODEL=<OUTPUT directory>/.DeepSeek-V4-Flash-0731-IQ2-IQ2-Q4.calibration.gguf
  KEEP_GENERATED_CALIBRATION=0

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
[[ $# -ge 1 && $# -le 3 ]] || { usage >&2; exit 2; }
[[ $(uname -s) == Linux ]] || die "this CUDA production script must run on Linux"
require_command make
require_command nproc
require_command python3
require_command realpath

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ -d $1 ]] || die "HF directory not found: $1"
hf_dir=$(realpath "$1")
[[ -f $hf_dir/model.safetensors.index.json ]] || \
    die "HF directory has no model.safetensors.index.json: $hf_dir"
out=${2:-${OUT:-/mnt/nfs-images/models/gguf/ds4/DeepSeek-V4-Flash-0731-Q4-IQ2-FullF16-256K-SM75.gguf}}
out=$(realpath -m "$out")
results=${3:-${RESULTS_CSV:-${out%.gguf}.bench.csv}}
results=$(realpath -m "$results")
reuse_model=${REUSE_MODEL:-0}
overwrite=${OVERWRITE:-0}
[[ $reuse_model == 0 || $reuse_model == 1 ]] || die "REUSE_MODEL must be 0 or 1"
[[ $overwrite == 0 || $overwrite == 1 ]] || die "OVERWRITE must be 0 or 1"
if [[ $reuse_model == 1 ]]; then
    [[ -f $out ]] || die "REUSE_MODEL=1 but OUTPUT.gguf does not exist: $out"
elif [[ -e $out && $overwrite != 1 ]]; then
    die "OUTPUT.gguf already exists; set OVERWRITE=1 or REUSE_MODEL=1: $out"
fi
verification_candidate=
cleanup_verification_candidate() {
    if [[ -n $verification_candidate && -f $verification_candidate ]]; then
        rm -f -- "$verification_candidate"
    fi
}
trap cleanup_verification_candidate EXIT
default_calibration=$script_dir/gguf/DeepSeek-V4-Flash-0731-IQ2-IQ2-Q4.gguf
generated_calibration=${GENERATED_CALIBRATION_MODEL:-$(dirname -- "$out")/.DeepSeek-V4-Flash-0731-IQ2-IQ2-Q4.calibration.gguf}
generated_calibration_by_runner=0
if [[ -n ${CALIBRATION_MODEL:-} ]]; then
    calibration_model=$CALIBRATION_MODEL
    [[ -f $calibration_model || ${REUSE_BASELINE:-0} == 1 ]] || \
        die "CALIBRATION_MODEL does not exist: $calibration_model"
elif [[ -f $default_calibration ]]; then
    calibration_model=$default_calibration
else
    calibration_model=$generated_calibration
    generated_calibration_by_runner=1
fi
calibration_model=$(realpath -m "$calibration_model")
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
keep_generated_calibration=${KEEP_GENERATED_CALIBRATION:-0}
prompt=${PROMPT_FILE:-$script_dir/speed-bench/promessi_sposi.txt}
prompt=$(realpath "$prompt")

[[ $target_context =~ ^[1-9][0-9]*$ ]] || die "TARGET_CONTEXT must be positive"
[[ $target_gen_tokens =~ ^[0-9]+$ ]] || die "TARGET_GEN_TOKENS must be nonnegative"
[[ $extra_headroom =~ ^[0-9]+$ ]] || die "CACHE_EXTRA_HEADROOM_MIB_PER_DEVICE must be nonnegative"
[[ $expected_candidates =~ ^[1-9][0-9]*$ ]] || die "EXPECTED_DENSE_CANDIDATES must be positive"
[[ $reuse_baseline == 0 || $reuse_baseline == 1 ]] || die "REUSE_BASELINE must be 0 or 1"
[[ $run_full_sweep == 0 || $run_full_sweep == 1 ]] || die "RUN_FULL_SWEEP must be 0 or 1"
[[ $keep_generated_calibration == 0 || $keep_generated_calibration == 1 ]] || \
    die "KEEP_GENERATED_CALIBRATION must be 0 or 1"
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
baseline_plan=${BASELINE_PLAN_AUDIT:-$prefix.all-iq2-all-local-256k-plan.csv}
baseline_log=${BASELINE_LOG:-$prefix.all-iq2-all-local-256k-plan.log}
baseline_csv=${BASELINE_CSV:-$prefix.all-iq2-all-local-256k-calibration.csv}
baseline_memory=${BASELINE_MEMORY_STATE:-$prefix.all-iq2-all-local-256k-memory.csv}
baseline_bindings=${BASELINE_BINDING_STATE:-$prefix.all-iq2-all-local-256k-bindings.csv}
baseline_allocations=${BASELINE_ALLOCATION_STATE:-$prefix.all-iq2-all-local-256k-allocations.csv}
selection_manifest=${SELECTION_MANIFEST:-$prefix.selection.txt}
verification_plan=${VERIFICATION_PLAN_AUDIT:-$prefix.full-f16-verification.csv}
verification_bindings=${VERIFICATION_BINDINGS:-$prefix.full-f16-bindings.csv}
verification_allocations=${VERIFICATION_ALLOCATIONS:-$prefix.full-f16-allocations.csv}
verification_csv=${VERIFICATION_CSV:-$prefix.verification-bench.csv}
mkdir -p -- "$(dirname -- "$out")" "$(dirname -- "$results")"

ctx_alloc=$((target_context + target_gen_tokens + 1))
if [[ $reuse_baseline != 1 && ! -f $calibration_model ]]; then
    mkdir -p -- "$(dirname -- "$calibration_model")"
    printf 'Generating disposable all-IQ2 gate/up + native-Q4 down calibration model...\n'
    unset DS4_TEMPLATE_GGUF Q4_TEMPLATE
    QUANTIZE_ONLY=1 \
    REUSE_MODEL=0 \
    OVERWRITE=0 \
    GPU_DEVICES="$gpu_devices" \
    GPU_VRAM="$gpu_vram" \
    CUDA_ARCH="$cuda_arch" \
    MAKE_JOBS="$make_jobs" \
    ROUTED_W1=iq2_xxs \
    ROUTED_W2=q4_k \
    ROUTED_W3=iq2_xxs \
    SM75_NATIVE_Q4=1 \
    QUANT_RECIPE='calibration:gate=iq2_xxs,up=iq2_xxs,down=q4_k' \
    PLAN_LOG="$prefix.generated-calibration-quant-plan.txt" \
    QUANT_LOG="$prefix.generated-calibration-quantize.log" \
    bash "$script_dir/produce-benchmark-iq2-iq2-q4.sh" \
        "$hf_dir" "$calibration_model"
    [[ -s $calibration_model ]] || \
        die "calibration generation did not produce $calibration_model"
fi
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
grep -Fq 't256-placement=all-local' "$baseline_log" ||
    die "capacity calibration is not the required all-local T256 policy"

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
    printf 'hf_source=%s\n' "$hf_dir"
    printf 'all_iq2_calibration_model=%s\n' "$calibration_model"
    printf 'calibration_generated_by_runner=%s\n' "$generated_calibration_by_runner"
    printf 'target_context=%s\n' "$target_context"
    printf 'target_gen_tokens=%s\n' "$target_gen_tokens"
    printf 'target_ctx_alloc=%s\n' "$ctx_alloc"
    printf 'gpu_devices=%s\n' "$gpu_devices"
    printf 'stage_split=%s\n' "$DS4_CUDA_EP_STAGE_SPLIT"
    printf 'dense_cache_policy=all-q8-f16\n'
    printf 'capacity_calibration_t256_placement=all-local\n'
    printf 'verification_t256_placement=all-local\n'
    printf 'placement_target=neutral-worst-case-home-footprint\n'
    printf 'runtime_placement_selection=deferred-to-production-ab\n'
    printf 'cache_extra_headroom_mib_per_device=%s\n' "$extra_headroom"
    printf 'expected_dense_candidates=%s\n' "$expected_candidates"
    printf 'routed_down_type=q4_k\n'
    printf 'iq2_gate_up_layer_order=%s\n' "$iq2_layer_order"
    printf 'iq2_layer_order_quality_ranked=false-unless-user-supplied\n'
    printf 'iq2_gate_up_layers=%s\n' "$iq2_layers"
    printf 'tensor_type_overrides=%s\n' "$tensor_overrides"
} | tee "$selection_manifest"

if [[ $generated_calibration_by_runner == 1 &&
      $keep_generated_calibration != 1 &&
      -f $calibration_model ]]; then
    expected_generated=$(realpath -m "$generated_calibration")
    [[ $calibration_model == "$expected_generated" ]] || \
        die "refusing to remove unexpected calibration path: $calibration_model"
    rm -f -- "$calibration_model"
    printf 'Removed generated calibration model after capacity selection: %s\n' \
        "$calibration_model"
fi

export GPU_DEVICES="$gpu_devices"
export GPU_VRAM="$gpu_vram"
export CUDA_ARCH="$cuda_arch"
export MAKE_JOBS="$make_jobs"
unset DS4_TEMPLATE_GGUF Q4_TEMPLATE
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
export DS4_CUDA_Q8_F16_PARTNER_CLASSES=none
export DS4_CUDA_Q8_T256_PLACEMENT=all-local
export DS4_CUDA_NO_Q8_F16_PARTNER_OFFLOAD=1
unset DS4_CUDA_Q8_F16_PARTNER_LAYERS
unset DS4_CUDA_Q8_PARTNER_ARITHMETIC

verification_model=$out
if [[ $reuse_model != 1 ]]; then
    verification_candidate=$(dirname -- "$out")/.$(basename -- "${out%.gguf}").unverified.$$.gguf
    [[ ! -e $verification_candidate ]] || \
        die "temporary verification model already exists: $verification_candidate"
    verification_model=$verification_candidate
fi
export PLAN_LOG=${PLAN_LOG:-$prefix.quant-plan.txt}
export QUANT_LOG=${QUANT_LOG:-$prefix.quantize.log}
export BENCH_LOG=${BENCH_LOG:-$prefix.bench.log}
export METADATA_LOG=${METADATA_LOG:-$prefix.bench-metadata.txt}
export RESULTS_SVG=${RESULTS_SVG:-$prefix.bench.svg}

bash "$script_dir/produce-benchmark-iq2-iq2-q4.sh" \
    "$hf_dir" "$verification_model" "$verification_csv"

python3 speed-bench/select-q4-iq2-full-f16.py \
    --plan-audit "$verification_plan" \
    --expected-candidates "$expected_candidates" \
    --verify-only
grep -Fq "q8 fp16 benefit plan materialized $expected_candidates/$expected_candidates candidates" \
    "${BENCH_LOG:-${out%.gguf}.bench.log}" || \
    die "runtime log does not prove complete dense FP16 materialization"
grep -Fq 't256-placement=all-local' \
    "${BENCH_LOG:-${out%.gguf}.bench.log}" || \
    die "runtime log does not prove the all-local T256 capacity envelope"
grep -Fq 'T256-output_b=43/43' \
    "${BENCH_LOG:-${out%.gguf}.bench.log}" || \
    die "runtime log does not prove complete T256 FP16 materialization"
grep -Fq 'partner=0 ' \
    "${BENCH_LOG:-${out%.gguf}.bench.log}" || \
    die "runtime log does not prove zero partner-resident dense bindings"
[[ -s $verification_bindings ]] || die "runtime omitted dense FP16 binding evidence"
[[ -s $verification_allocations ]] || die "runtime omitted dense FP16 allocation evidence"

if [[ -n $verification_candidate ]]; then
    if [[ $overwrite == 1 ]]; then
        mv -f -- "$verification_candidate" "$out"
    else
        mv -- "$verification_candidate" "$out"
    fi
    verification_candidate=
fi

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
