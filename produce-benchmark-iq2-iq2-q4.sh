#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Produce and benchmark a DeepSeek V4 Flash GGUF with:
  routed gate = IQ2_XXS, routed up = IQ2_XXS, routed down = Q4_K

Usage:
  bash produce-benchmark-iq2-iq2-q4.sh \
    HF_DIR [OUTPUT.gguf] [RESULTS.csv]

The published Q4 GGUF metadata and routed-MoE imatrix are downloaded and
cached automatically. Only the small metadata prefix of the 165 GB Q4 GGUF is
downloaded; tensor payloads are regenerated from HF_DIR.

Advanced/legacy usage:
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

Optional asset overrides:
  DS4_TEMPLATE_GGUF=/path/to/template.gguf
  DS4_IMATRIX=/path/to/routed-moe-imatrix.dat
  DS4_ASSET_CACHE=/path/to/cache

Advanced recipe overrides (used by the selective-Q2 wrapper):
  ROUTED_W1=iq2_xxs
  ROUTED_W2=q4_k
  ROUTED_W3=iq2_xxs
  TENSOR_TYPE_OVERRIDES='blk.3.ffn_down_exps.weight=q2_k,...'
  QUANT_RECIPE='gate:iq2_xxs,up:iq2_xxs,down:q4_k'
  PLOT_TITLE='RTX 8000 2x2 TP - IQ2 gate/up, Q4 down'

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
[[ $(uname -s) == Linux ]] || die "this production/CUDA script must run on Linux"

require_command make
require_command nvidia-smi
require_command nproc
require_command realpath
require_command tee
require_command awk
require_command df

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

case $# in
    1|2|3)
        hf_arg=$1
        out_arg=${2:-${OUT:-$PWD/gguf/DeepSeek-V4-Flash-0731-IQ2-IQ2-Q4.gguf}}
        csv_arg=${3:-}
        template_arg=${DS4_TEMPLATE_GGUF:-${Q4_TEMPLATE:-}}
        imatrix_arg=${DS4_IMATRIX:-${IMATRIX:-}}
        ;;
    4|5)
        # Keep the original explicit-path interface working.
        hf_arg=$1
        template_arg=$2
        imatrix_arg=$3
        out_arg=$4
        csv_arg=${5:-}
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

[[ -d $hf_arg ]] || die "HF directory not found: $hf_arg"
hf_dir=$(realpath "$hf_arg")

[[ -f $hf_dir/model.safetensors.index.json ]] || \
    die "HF directory has no model.safetensors.index.json: $hf_dir"

mkdir -p -- "$(dirname -- "$out_arg")"
out=$(realpath -m "$out_arg")

prefix=${out%.gguf}
[[ $prefix != "$out" ]] || prefix=$out
csv_arg=${csv_arg:-${RESULTS_CSV:-$prefix.bench.csv}}
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
routed_w1=${ROUTED_W1:-iq2_xxs}
routed_w2=${ROUTED_W2:-q4_k}
routed_w3=${ROUTED_W3:-iq2_xxs}
quant_recipe=${QUANT_RECIPE:-gate:$routed_w1,up:$routed_w3,down:$routed_w2}
plot_title=${PLOT_TITLE:-RTX 8000 2x2 TP - IQ2 gate/up, Q4 down}
tensor_type_overrides=${TENSOR_TYPE_OVERRIDES:-}
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

printf 'Building the quantizer...\n'
make -j "$make_jobs" -C gguf-tools deepseek4-quantize
quantizer=$script_dir/gguf-tools/deepseek4-quantize

template_is_valid() {
    local candidate=$1
    "$quantizer" \
        --hf "$hf_dir" \
        --template "$candidate" \
        --routed-w1 "$routed_w1" \
        --routed-w3 "$routed_w3" \
        --routed-w2 "$routed_w2" \
        --dry-run >/dev/null 2>&1
}

cache_root=${XDG_CACHE_HOME:-${HOME:?HOME is required}/.cache}
asset_revision=1cd7b564460821938add0475a60b942c409295e0
asset_cache=${DS4_ASSET_CACHE:-$cache_root/ds4/iq2-iq2-q4/$asset_revision}
mkdir -p -- "$asset_cache"

template_repo=antirez/deepseek-v4-gguf
template_name=DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf
template_cache=$asset_cache/$template_name.metadata-only.gguf
template_url=https://huggingface.co/$template_repo/resolve/$asset_revision/$template_name

if [[ -n $template_arg ]]; then
    [[ -f $template_arg ]] || die "template GGUF not found: $template_arg"
    template=$(realpath "$template_arg")
    template_is_valid "$template" || die "template GGUF metadata is invalid: $template"
else
    if [[ ! -s $template_cache ]] || ! template_is_valid "$template_cache"; then
        require_command curl
        printf 'Fetching the published Q4 template metadata (not its tensor payloads)...\n'
        template_partial=$template_cache.partial.$$
        template_ok=0
        for template_mib in 8 16 32 64 128 256; do
            template_bytes=$((template_mib * 1048576))
            rm -f -- "$template_partial"
            printf '  trying a %s MiB metadata prefix...\n' "$template_mib"
            if curl --fail --location --retry 3 \
                --range "0-$((template_bytes - 1))" \
                --max-filesize "$template_bytes" \
                --output "$template_partial" \
                "$template_url"; then
                downloaded_bytes=$(stat -c %s "$template_partial")
                (( downloaded_bytes <= template_bytes )) || \
                    die "template server ignored the bounded range request"
                if template_is_valid "$template_partial"; then
                    mv -f -- "$template_partial" "$template_cache"
                    template_ok=1
                    break
                fi
            fi
        done
        rm -f -- "$template_partial"
        (( template_ok == 1 )) || \
            die "could not obtain a complete template metadata prefix; set DS4_TEMPLATE_GGUF to a local Q4 GGUF"
    fi
    template=$(realpath "$template_cache")
fi

imatrix_repo=antirez/deepseek-v4-gguf
imatrix_rel=imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat
imatrix_cache=$asset_cache/$imatrix_rel
if [[ -n $imatrix_arg ]]; then
    [[ -f $imatrix_arg ]] || die "imatrix not found: $imatrix_arg"
    imatrix=$(realpath "$imatrix_arg")
else
    if [[ ! -s $imatrix_cache ]]; then
        require_command hf
        printf 'Fetching the published routed-MoE calibration matrix...\n'
        hf download "$imatrix_repo" "$imatrix_rel" \
            --revision "$asset_revision" \
            --local-dir "$asset_cache"
    fi
    [[ -s $imatrix_cache ]] || die "imatrix download did not produce: $imatrix_cache"
    imatrix=$(realpath "$imatrix_cache")
fi

[[ $out != "$template" ]] || die "OUTPUT.gguf must not be the template GGUF"

printf 'Building the %s CUDA benchmark...\n' "$cuda_arch"
make -j "$make_jobs" -C gguf-tools deepseek4-quantize-cuda test-quants-cuda \
    CUDA_ARCH="$cuda_arch"
quantizer=$script_dir/gguf-tools/deepseek4-quantize-cuda
gguf-tools/test-quants-cuda "$gpu_devices"
# Force the CUDA object to rebuild: changing CUDA_ARCH alone is not a Makefile
# dependency and could otherwise leave an object compiled for another GPU.
make -B -j "$make_jobs" ds4-bench tests/test_engine_mgpu_placement \
    CUDA_ARCH="$cuda_arch"
./tests/test_engine_mgpu_placement

quant_args=(
    --hf "$hf_dir"
    --template "$template"
    --imatrix "$imatrix"
    --routed-w1 "$routed_w1"
    --routed-w3 "$routed_w3"
    --routed-w2 "$routed_w2"
    --threads "$quant_threads"
    --quant-backend cuda
    --quant-gpu-devices "$gpu_devices"
)
if [[ -n $tensor_type_overrides ]]; then
    IFS=',' read -r -a override_list <<< "$tensor_type_overrides"
    for override in "${override_list[@]}"; do
        [[ $override == *=* ]] || die "bad TENSOR_TYPE_OVERRIDES entry: $override"
        quant_args+=(--tensor-type "$override")
    done
fi

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
    printf 'Validating the %s quantization plan...\n' "$quant_recipe"
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
    printf 'quant=%s\n' "$quant_recipe"
    printf 'tensor_type_overrides=%s\n' "$tensor_type_overrides"
    printf 'quant_backend=cuda\n'
    printf 'quant_gpu_devices=%s\n' "$gpu_devices"
    printf 'template=%s\n' "$template"
    printf 'template_bytes=%s\n' "$(stat -c %s "$template")"
    printf 'imatrix=%s\n' "$imatrix"
    printf 'imatrix_bytes=%s\n' "$(stat -c %s "$imatrix")"
    printf 'bootstrap_repo=%s\n' "$template_repo"
    printf 'bootstrap_revision=%s\n' "$asset_revision"
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
        --title "$plot_title"
fi

printf '\nDone.\nModel:    %s\nCSV:      %s\nMetadata: %s\nLog:      %s\n' \
    "$out" "$csv" "$metadata" "$bench_log"
if [[ -f $svg ]]; then
    printf 'Chart:    %s\n' "$svg"
fi
