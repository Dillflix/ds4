#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Produce and benchmark a DeepSeek V4 Flash GGUF with:
  routed gate = IQ2_XXS, routed up = IQ2_XXS, routed down = Q4_K

Usage:
  bash produce-benchmark-iq2-iq2-q4.sh \
    HF_DIR TEMPLATE.gguf IMATRIX.dat OUTPUT.gguf [RESULTS.csv]

Defaults target two NVLink pairs of 48 GB Turing GPUs:
  CUDA_ARCH=sm_75
  GPU_DEVICES=0,2,1,3       # DwarfStar pairs 0<->1 and 2<->3
  GPU_VRAM=auto
  QUANT_THREADS=8
  CTX_START=2048
  CTX_MAX=65536
  STEP_INCR=2048
  GEN_TOKENS=128

Override any default as an environment variable, for example:
  GPU_DEVICES=0,4,1,5 CTX_MAX=32768 bash produce-benchmark-iq2-iq2-q4.sh ...

The script refuses to replace OUTPUT.gguf. Set OVERWRITE=1 to replace it
atomically, or REUSE_MODEL=1 to skip quantization and benchmark an existing
OUTPUT.gguf again.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_positive_integer() {
    local name=$1
    local value=$2
    [[ $value =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
}

[[ ${1:-} != -h && ${1:-} != --help ]] || {
    usage
    exit 0
}
[[ $# -eq 4 || $# -eq 5 ]] || {
    usage >&2
    exit 2
}
[[ $(uname -s) == Linux ]] || die "this production/CUDA script must run on Linux"

require_command make
require_command nvidia-smi
require_command nproc
require_command realpath
require_command tee
require_command awk
require_command df

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ -d $1 ]] || die "HF directory not found: $1"
[[ -f $2 ]] || die "template GGUF not found: $2"
[[ -f $3 ]] || die "imatrix not found: $3"
hf_dir=$(realpath "$1")
template=$(realpath "$2")
imatrix=$(realpath "$3")
out_arg=$4

[[ -f $hf_dir/model.safetensors.index.json ]] || \
    die "HF directory has no model.safetensors.index.json: $hf_dir"

mkdir -p -- "$(dirname -- "$out_arg")"
out=$(realpath -m "$out_arg")
[[ $out != "$template" ]] || die "OUTPUT.gguf must not be the template GGUF"

prefix=${out%.gguf}
[[ $prefix != "$out" ]] || prefix=$out
csv_arg=${5:-${RESULTS_CSV:-$prefix.bench.csv}}
mkdir -p -- "$(dirname -- "$csv_arg")"
csv=$(realpath -m "$csv_arg")
plan_log=$(realpath -m "${PLAN_LOG:-$prefix.quant-plan.txt}")
quant_log=$(realpath -m "${QUANT_LOG:-$prefix.quantize.log}")
bench_log=$(realpath -m "${BENCH_LOG:-$prefix.bench.log}")
metadata=$(realpath -m "${METADATA_LOG:-$prefix.bench-metadata.txt}")
svg=$(realpath -m "${RESULTS_SVG:-$prefix.bench.svg}")
for artifact in "$plan_log" "$quant_log" "$bench_log" "$metadata" "$svg"; do
    mkdir -p -- "$(dirname -- "$artifact")"
done

cuda_arch=${CUDA_ARCH:-sm_75}
gpu_devices=${GPU_DEVICES:-0,2,1,3}
gpu_vram=${GPU_VRAM:-auto}
quant_threads=${QUANT_THREADS:-8}
make_jobs=${MAKE_JOBS:-$(nproc)}
ctx_start=${CTX_START:-2048}
ctx_max=${CTX_MAX:-65536}
step_incr=${STEP_INCR:-2048}
gen_tokens=${GEN_TOKENS:-128}
prompt_arg=${PROMPT_FILE:-$script_dir/speed-bench/promessi_sposi.txt}
[[ -f $prompt_arg ]] || die "benchmark prompt not found: $prompt_arg"
prompt=$(realpath "$prompt_arg")

require_positive_integer QUANT_THREADS "$quant_threads"
require_positive_integer MAKE_JOBS "$make_jobs"
require_positive_integer CTX_START "$ctx_start"
require_positive_integer CTX_MAX "$ctx_max"
require_positive_integer STEP_INCR "$step_incr"
[[ $gen_tokens =~ ^[0-9]+$ ]] || die "GEN_TOKENS must be a nonnegative integer"
(( ctx_start <= ctx_max )) || die "CTX_START must be no greater than CTX_MAX"

IFS=',' read -r -a gpu_list <<< "$gpu_devices"
[[ ${#gpu_list[@]} -eq 4 ]] || \
    die "GPU_DEVICES must list four devices as home0,home1,partner0,partner1"
for gpu in "${gpu_list[@]}"; do
    [[ $gpu =~ ^[0-9]+$ ]] || die "bad GPU device index: $gpu"
done
for ((i = 0; i < 4; i++)); do
    for ((j = i + 1; j < 4; j++)); do
        [[ ${gpu_list[i]} != "${gpu_list[j]}" ]] || \
            die "GPU_DEVICES contains duplicate device ${gpu_list[i]}"
    done
done

reuse_model=${REUSE_MODEL:-0}
overwrite=${OVERWRITE:-0}
[[ $reuse_model == 0 || $reuse_model == 1 ]] || die "REUSE_MODEL must be 0 or 1"
[[ $overwrite == 0 || $overwrite == 1 ]] || die "OVERWRITE must be 0 or 1"
if [[ $reuse_model == 1 ]]; then
    [[ -f $out ]] || die "REUSE_MODEL=1 but OUTPUT.gguf does not exist: $out"
elif [[ -e $out && $overwrite != 1 ]]; then
    die "OUTPUT.gguf already exists; set OVERWRITE=1 or REUSE_MODEL=1: $out"
fi

cd "$script_dir"

printf 'Building the quantizer and %s CUDA benchmark...\n' "$cuda_arch"
make -j "$make_jobs" -C gguf-tools deepseek4-quantize
# Force the CUDA object to rebuild: changing CUDA_ARCH alone is not a Makefile
# dependency and could otherwise leave an object compiled for another GPU.
make -B -j "$make_jobs" ds4-bench tests/test_engine_mgpu_placement \
    CUDA_ARCH="$cuda_arch"
./tests/test_engine_mgpu_placement

quantizer=$script_dir/gguf-tools/deepseek4-quantize
quant_args=(
    --hf "$hf_dir"
    --template "$template"
    --imatrix "$imatrix"
    --routed-w1 iq2_xxs
    --routed-w3 iq2_xxs
    --routed-w2 q4_k
    --threads "$quant_threads"
)

partial=
cleanup() {
    if [[ -n $partial && -f $partial ]]; then
        rm -f -- "$partial"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ $reuse_model != 1 ]]; then
    printf 'Validating the IQ2/IQ2/Q4 quantization plan...\n'
    "$quantizer" "${quant_args[@]}" --dry-run 2>&1 | tee "$plan_log"

    planned_bytes=$(awk -F: '/^approx_file_bytes:/ {
        gsub(/[[:space:]]/, "", $2); value = $2
    } END {print value}' "$plan_log")
    available_bytes=$(df -PB1 "$(dirname -- "$out")" | awk 'NR == 2 {print $4}')
    if [[ $planned_bytes =~ ^[0-9]+$ && $available_bytes =~ ^[0-9]+$ ]]; then
        required_bytes=$((planned_bytes + 1073741824))
        (( available_bytes >= required_bytes )) || \
            die "output needs $((required_bytes / 1073741824)) GiB free; only $((available_bytes / 1073741824)) GiB is available"
    fi

    partial=$(dirname -- "$out")/.$(basename -- "$out").partial.$$
    [[ ! -e $partial ]] || die "temporary output already exists: $partial"
    printf 'Producing %s (the final rename is atomic)...\n' "$out"
    "$quantizer" "${quant_args[@]}" --out "$partial" 2>&1 | tee "$quant_log"
    [[ -s $partial ]] || die "quantizer produced an empty output file"
    if [[ $overwrite == 1 ]]; then
        mv -f -- "$partial" "$out"
    else
        mv -- "$partial" "$out"
    fi
    partial=
fi

pair0="${gpu_list[0]}<->${gpu_list[2]}"
pair1="${gpu_list[1]}<->${gpu_list[3]}"
{
    printf 'model=%s\n' "$out"
    printf 'model_bytes=%s\n' "$(stat -c %s "$out")"
    printf 'quant=gate:iq2_xxs,up:iq2_xxs,down:q4_k\n'
    printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
    if [[ -n $(git status --short 2>/dev/null) ]]; then
        printf 'git_dirty=true\n'
    else
        printf 'git_dirty=false\n'
    fi
    printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'cuda_arch=%s\n' "$cuda_arch"
    printf 'gpu_devices=%s\n' "$gpu_devices"
    printf 'tp_pairs=%s,%s\n' "$pair0" "$pair1"
    printf 'gpu_vram=%s\n' "$gpu_vram"
    printf 'ctx_start=%s\nctx_max=%s\nstep_incr=%s\ngen_tokens=%s\n' \
        "$ctx_start" "$ctx_max" "$step_incr" "$gen_tokens"
    printf '\nGPU inventory:\n'
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader
    printf '\nGPU topology:\n'
    nvidia-smi topo -m
    printf '\nGit status:\n'
    git status --short 2>/dev/null || true
} | tee "$metadata"

if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    printf 'warning: CUDA_VISIBLE_DEVICES=%s; --gpu-devices indexes that visible set\n' \
        "$CUDA_VISIBLE_DEVICES" >&2
fi

printf 'Benchmarking with TP pairs %s and %s...\n' "$pair0" "$pair1"
./ds4-bench \
    --cuda \
    --cuda-tensor-parallel \
    --gpu-vram "$gpu_vram" \
    --gpu-devices "$gpu_devices" \
    --warm-weights \
    -m "$out" \
    --prompt-file "$prompt" \
    --ctx-start "$ctx_start" \
    --ctx-max "$ctx_max" \
    --step-incr "$step_incr" \
    --gen-tokens "$gen_tokens" \
    --csv "$csv" 2>&1 | tee "$bench_log"

if command -v python3 >/dev/null 2>&1; then
    python3 speed-bench/plot_speed.py "$csv" \
        --output "$svg" \
        --title "RTX 8000 2x2 TP - IQ2 gate/up, Q4 down"
fi

printf '\nDone.\nModel:    %s\nCSV:      %s\nMetadata: %s\nLog:      %s\n' \
    "$out" "$csv" "$metadata" "$bench_log"
if [[ -f $svg ]]; then
    printf 'Chart:    %s\n' "$svg"
fi
