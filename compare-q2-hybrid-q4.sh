#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Run the fixed DeepSeek V4 Flash Q2 vs IQ2/IQ2/Q4 vs Q4 suite.

Usage:
  ./compare-q2-hybrid-q4.sh Q2.gguf HYBRID.gguf Q4.gguf [RESULTS_DIR]

The suite runs models one at a time on the same four-GPU layout and records:
  * 100 official-continuation quality cases (exact-math target-token NLL)
  * prompt and steady-generation throughput at 2K, 4K, ... 64K context
  * first-token latency, peak per-card VRAM, utilization, power, temperature
  * three performance repetitions in a balanced model order

Defaults for 4x RTX 8000 in two NVLink pairs:
  GPU_DEVICES=0,2,1,3
  GPU_VRAM=auto
  CUDA_ARCH=sm_75
  QUALITY_CTX=4096
  PERF_CTX_START=2048
  PERF_CTX_MAX=65536
  PERF_STEP_MUL=2
  PERF_GEN_TOKENS=128
  PERF_REPEATS=3

Set RESUME=1 to reuse complete result files after an interruption.
The suite refuses to start while another CUDA compute process is resident;
set ALLOW_BUSY_GPUS=1 only when that process is intentional.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

[[ ${1:-} != -h && ${1:-} != --help ]] || {
    usage
    exit 0
}
[[ $# -ge 3 && $# -le 4 ]] || {
    usage >&2
    exit 2
}
[[ $(uname -s) == Linux ]] || die "the CUDA comparison suite must run on Linux"

for command in make nvidia-smi python3 realpath stat tee awk nproc; do
    require_command "$command"
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
q2=$(realpath "$1")
hybrid=$(realpath "$2")
q4=$(realpath "$3")
for model in "$q2" "$hybrid" "$q4"; do
    [[ -f $model ]] || die "model not found: $model"
done
[[ $q2 != "$hybrid" && $q2 != "$q4" && $hybrid != "$q4" ]] || \
    die "Q2, hybrid, and Q4 must be three distinct files"

results_arg=${4:-$PWD/comparison-$(date -u +%Y%m%dT%H%M%SZ)}
results=$(realpath -m "$results_arg")
resume=${RESUME:-0}
[[ $resume == 0 || $resume == 1 ]] || die "RESUME must be 0 or 1"
if [[ -e $results && $resume != 1 ]]; then
    die "results directory exists; choose another or set RESUME=1: $results"
fi
mkdir -p "$results"/{quality,performance,comparisons,logs,gpu}

cuda_arch=${CUDA_ARCH:-sm_75}
gpu_devices=${GPU_DEVICES:-0,2,1,3}
gpu_vram=${GPU_VRAM:-auto}
quality_ctx=${QUALITY_CTX:-4096}
perf_ctx_start=${PERF_CTX_START:-2048}
perf_ctx_max=${PERF_CTX_MAX:-65536}
perf_step_mul=${PERF_STEP_MUL:-2}
perf_gen_tokens=${PERF_GEN_TOKENS:-128}
perf_repeats=${PERF_REPEATS:-3}
gpu_sample_seconds=${GPU_SAMPLE_SECONDS:-1}
make_jobs=${MAKE_JOBS:-$(nproc)}
allow_busy=${ALLOW_BUSY_GPUS:-0}
manifest_arg=${QUALITY_MANIFEST:-$script_dir/gguf-tools/quality-testing/data/flash/manifest.tsv}
prompt_arg=${PERF_PROMPT_FILE:-$script_dir/speed-bench/promessi_sposi.txt}

for pair in \
    "QUALITY_CTX:$quality_ctx" \
    "PERF_CTX_START:$perf_ctx_start" \
    "PERF_CTX_MAX:$perf_ctx_max" \
    "PERF_GEN_TOKENS:$perf_gen_tokens" \
    "PERF_REPEATS:$perf_repeats" \
    "MAKE_JOBS:$make_jobs"; do
    name=${pair%%:*}
    value=${pair#*:}
    [[ $value =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
done
[[ $perf_step_mul =~ ^[1-9][0-9]*([.][0-9]+)?$ ]] || \
    die "PERF_STEP_MUL must be at least 1"
(( perf_ctx_start <= perf_ctx_max )) || die "PERF_CTX_START exceeds PERF_CTX_MAX"
[[ $allow_busy == 0 || $allow_busy == 1 ]] || die "ALLOW_BUSY_GPUS must be 0 or 1"

IFS=',' read -r -a gpu_list <<< "$gpu_devices"
[[ ${#gpu_list[@]} -eq 4 ]] || \
    die "GPU_DEVICES must list four devices as home0,home1,partner0,partner1"
for gpu in "${gpu_list[@]}"; do
    [[ $gpu =~ ^[0-9]+$ ]] || die "bad GPU device index: $gpu"
done

manifest=$(realpath "$manifest_arg")
prompt=$(realpath "$prompt_arg")
[[ -f $manifest ]] || die "quality manifest not found: $manifest"
[[ -f $prompt ]] || die "performance prompt not found: $prompt"
quality_cases=$(awk 'NF && $1 !~ /^#/ {n++} END {print n+0}' "$manifest")
(( quality_cases > 0 )) || die "quality manifest has no cases: $manifest"

cat > "$results/config.env" <<EOF
SUITE_VERSION=1
CUDA_ARCH=$cuda_arch
GPU_DEVICES=$gpu_devices
GPU_VRAM=$gpu_vram
QUALITY_MANIFEST=$manifest
QUALITY_CASES=$quality_cases
QUALITY_CTX=$quality_ctx
PERF_PROMPT_FILE=$prompt
PERF_CTX_START=$perf_ctx_start
PERF_CTX_MAX=$perf_ctx_max
PERF_STEP_MUL=$perf_step_mul
PERF_GEN_TOKENS=$perf_gen_tokens
PERF_REPEATS=$perf_repeats
GPU_SAMPLE_SECONDS=$gpu_sample_seconds
GIT_COMMIT=$(git -C "$script_dir" rev-parse HEAD 2>/dev/null || printf unknown)
STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

printf 'label\tpath\tbytes\tmodified_utc\n' > "$results/models.tsv"
for spec in "q2:$q2" "hybrid:$hybrid" "q4:$q4"; do
    label=${spec%%:*}
    model=${spec#*:}
    printf '%s\t%s\t%s\t%s\n' \
        "$label" "$model" "$(stat -c %s "$model")" \
        "$(date -u -d "@$(stat -c %Y "$model")" +%Y-%m-%dT%H:%M:%SZ)" \
        >> "$results/models.tsv"
done
nvidia-smi -L > "$results/gpus.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,pstate,power.limit \
    --format=csv > "$results/gpu-inventory.csv"

printf 'Building the CUDA benchmark and four-GPU quality scorer...\n'
make -B -j "$make_jobs" ds4-bench CUDA_ARCH="$cuda_arch"
make -B -j "$make_jobs" -C gguf-tools quality-score CUDA_ARCH="$cuda_arch"

monitor_pid=
cleanup() {
    if [[ -n ${monitor_pid:-} ]]; then
        kill "$monitor_pid" >/dev/null 2>&1 || true
        wait "$monitor_pid" >/dev/null 2>&1 || true
        monitor_pid=
    fi
}
trap cleanup EXIT

check_idle_gpus() {
    local busy
    busy=$(nvidia-smi \
        --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null || true)
    if [[ -n $busy && $allow_busy != 1 ]]; then
        printf '%s\n' "$busy" >&2
        die "CUDA compute processes are already resident; stop them or set ALLOW_BUSY_GPUS=1"
    fi
}

gpu_monitor() {
    local output=$1
    printf 'timestamp,index,utilization_gpu_percent,utilization_memory_percent,memory_used_mib,power_draw_w,temperature_c\n' > "$output"
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu \
            --format=csv,noheader,nounits >> "$output" 2>/dev/null || true
        sleep "$gpu_sample_seconds"
    done
}

run_monitored() {
    local tag=$1
    local log=$2
    shift 2
    check_idle_gpus
    gpu_monitor "$results/gpu/$tag.csv" &
    monitor_pid=$!
    set +e
    "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    cleanup
    return "$rc"
}

quality_complete() {
    local output=$1
    [[ -s $output ]] || return 1
    local rows
    rows=$(awk 'NR > 1 {n++} END {print n+0}' "$output")
    [[ $rows -eq $quality_cases ]]
}

performance_complete() {
    local output=$1
    [[ -s $output ]] || return 1
    local final_ctx
    final_ctx=$(awk -F, 'NR > 1 {v=$1} END {print v+0}' "$output")
    [[ $final_ctx -eq $perf_ctx_max ]]
}

score_model() {
    local label=$1
    local model=$2
    local output=$results/quality/$label.tsv
    if [[ $resume == 1 ]] && quality_complete "$output"; then
        printf 'Reusing complete quality result: %s\n' "$output"
        return
    fi
    rm -f "$output"
    printf '\nQuality: %s (%s official cases)\n' "$label" "$quality_cases"
    run_monitored "quality-$label" "$results/logs/quality-$label.log" \
        "$script_dir/gguf-tools/quality-testing/score_official" \
        "$model" "$manifest" "$output" "$quality_ctx" \
        --gpu-vram "$gpu_vram" \
        --gpu-devices "$gpu_devices" \
        --cuda-tensor-parallel \
        --warm-weights
}

run_performance() {
    local label=$1
    local model=$2
    local repeat=$3
    local output=$results/performance/$label.run$repeat.csv
    if [[ $resume == 1 ]] && performance_complete "$output"; then
        printf 'Reusing complete performance result: %s\n' "$output"
        return
    fi
    rm -f "$output"
    printf '\nPerformance: %s, repetition %s/%s\n' "$label" "$repeat" "$perf_repeats"
    run_monitored "performance-$label-run$repeat" \
        "$results/logs/performance-$label-run$repeat.log" \
        "$script_dir/ds4-bench" \
        --model "$model" \
        --cuda \
        --cuda-tensor-parallel \
        --gpu-vram "$gpu_vram" \
        --gpu-devices "$gpu_devices" \
        --warm-weights \
        --prompt-file "$prompt" \
        --ctx-start "$perf_ctx_start" \
        --ctx-max "$perf_ctx_max" \
        --step-mul "$perf_step_mul" \
        --gen-tokens "$perf_gen_tokens" \
        --csv "$output"
}

score_model q2 "$q2"
score_model hybrid "$hybrid"
score_model q4 "$q4"

# A balanced three-order rotation avoids always testing Q4 hottest and Q2 coldest.
labels=(q2 hybrid q4)
models=("$q2" "$hybrid" "$q4")
for ((repeat = 1; repeat <= perf_repeats; repeat++)); do
    offset=$(((repeat - 1) % 3))
    for ((slot = 0; slot < 3; slot++)); do
        index=$(((offset + slot) % 3))
        run_performance "${labels[index]}" "${models[index]}" "$repeat"
    done
done

python3 "$script_dir/gguf-tools/quality-testing/compare_scores.py" \
    "$results/quality/q2.tsv" "$results/quality/hybrid.tsv" \
    > "$results/comparisons/q2-vs-hybrid.txt"
python3 "$script_dir/gguf-tools/quality-testing/compare_scores.py" \
    "$results/quality/hybrid.tsv" "$results/quality/q4.tsv" \
    > "$results/comparisons/hybrid-vs-q4.txt"
python3 "$script_dir/gguf-tools/quality-testing/compare_scores.py" \
    "$results/quality/q2.tsv" "$results/quality/q4.tsv" \
    > "$results/comparisons/q2-vs-q4.txt"

python3 "$script_dir/gguf-tools/quality-testing/summarize_three_model_suite.py" "$results"
printf '\nSuite complete: %s\n' "$results"
printf 'Open %s\n' "$results/summary.md"
