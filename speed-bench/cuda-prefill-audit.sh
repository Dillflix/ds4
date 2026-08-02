#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Capture one uninstrumented 2K prefill benchmark and one Nsight Systems trace of
the same production four-GPU CUDA path.

Required environment:
  MODEL=/absolute/path/model.gguf
  PROMPT=/absolute/path/prompt.txt

Optional environment:
  GPU_DEVICES=0,2,1,3       Logical tier order; default matches two NVLink pairs
  GPU_VRAM=auto
  CUDA_ARCH=sm_75
  PREFILL_TOKENS=2048
  PREFILL_MB=512
  AUDIT_DIR=/absolute/path/output-directory
  SKIP_BUILD=1              Reuse the current ds4-bench binary
  SKIP_BASELINE=1           Capture only the Nsight run
  NSYS_TRACE=cuda,nvtx,osrt,cublas
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

[[ ${1:-} != "-h" && ${1:-} != "--help" ]] || {
    usage
    exit 0
}

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

: "${MODEL:?set MODEL to the absolute GGUF path}"
: "${PROMPT:?set PROMPT to the absolute benchmark prompt path}"
[[ -f $MODEL ]] || die "model not found: $MODEL"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
command -v nsys >/dev/null 2>&1 || die "Nsight Systems CLI (nsys) not found"

GPU_DEVICES=${GPU_DEVICES:-0,2,1,3}
GPU_VRAM=${GPU_VRAM:-auto}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
PREFILL_TOKENS=${PREFILL_TOKENS:-2048}
PREFILL_MB=${PREFILL_MB:-512}
NSYS_TRACE=${NSYS_TRACE:-cuda,nvtx,osrt,cublas}
AUDIT_DIR=${AUDIT_DIR:-$repo_dir/prefill-audit-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PREFILL_TOKENS =~ ^[0-9]+$ && $PREFILL_TOKENS -gt 0 ]] ||
    die "PREFILL_TOKENS must be a positive integer"
[[ $PREFILL_MB =~ ^[0-9]+$ && $PREFILL_MB -gt 0 && $PREFILL_MB -lt $PREFILL_TOKENS ]] ||
    die "PREFILL_MB must be positive and smaller than PREFILL_TOKENS"

mkdir -p "$AUDIT_DIR"
AUDIT_DIR=$(cd "$AUDIT_DIR" && pwd)

if [[ ${SKIP_BUILD:-0} != 1 ]]; then
    make -B -j"$(nproc)" ds4-bench CUDA_ARCH="$CUDA_ARCH"
fi
[[ -x ./ds4-bench ]] || die "ds4-bench is missing; run without SKIP_BUILD=1"

# These are the proved baseline semantics.  Flags whose code checks only for
# presence must be removed rather than assigned zero.
unset DS4_CUDA_MOE_PROFILE
unset DS4_CUDA_ATTN_OUTPUT_PROFILE
unset DS4_METAL_LAYER_STAGE_PROFILE
unset DS4_METAL_GRAPH_PREFILL_PROFILE
unset DS4_CUDA_PREFILL_PIPELINE_SEQUENTIAL
unset DS4_CUDA_PREFILL_PIPELINE_SYNC_BOUNDARY
export DS4_CUDA_TP_PREFILL_ATTN_HEADS=0
export DS4_CUDA_PREFILL_PIPELINE=1
export DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
export DS4_CUDA_PREFILL_PIPELINE_MB="$PREFILL_MB"

bench_args=(
    -m "$MODEL"
    --prompt-file "$PROMPT"
    --backend cuda
    --gpu-devices "$GPU_DEVICES"
    --gpu-vram "$GPU_VRAM"
    --cuda-tensor-parallel
    --ctx-start "$PREFILL_TOKENS"
    --ctx-max "$PREFILL_TOKENS"
    --ctx-alloc "$((PREFILL_TOKENS + 1))"
    --step-incr "$PREFILL_TOKENS"
    --gen-tokens 0
    --prefill-chunk "$PREFILL_TOKENS"
)

{
    printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'repo=%s\n' "$repo_dir"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'git_branch=%s\n' "$(git branch --show-current)"
    printf 'model=%s\n' "$MODEL"
    printf 'model_bytes=%s\n' "$(stat -c %s "$MODEL")"
    printf 'prompt=%s\n' "$PROMPT"
    printf 'gpu_devices=%s\n' "$GPU_DEVICES"
    printf 'gpu_vram=%s\n' "$GPU_VRAM"
    printf 'cuda_arch=%s\n' "$CUDA_ARCH"
    printf 'prefill_tokens=%s\n' "$PREFILL_TOKENS"
    printf 'prefill_microbatch=%s\n' "$PREFILL_MB"
    printf 'nsys_trace=%s\n' "$NSYS_TRACE"
    printf '\n[git status]\n'
    git status --short
    printf '\n[nvidia-smi inventory]\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,ecc.mode.current,driver_version --format=csv
    printf '\n[nvidia-smi topology]\n'
    nvidia-smi topo -m
    printf '\n[nvcc]\n'
    nvcc --version 2>&1 || true
    printf '\n[nsys]\n'
    nsys --version 2>&1 || true
    printf '\n[numa]\n'
    numactl --hardware 2>&1 || true
} >"$AUDIT_DIR/manifest.txt"

if [[ ${SKIP_BASELINE:-0} != 1 ]]; then
    printf 'Running the uninstrumented baseline...\n'
    ./ds4-bench "${bench_args[@]}" \
        --csv "$AUDIT_DIR/baseline.csv" \
        > >(tee "$AUDIT_DIR/baseline.stdout.log") \
        2> >(tee "$AUDIT_DIR/baseline.stderr.log" >&2)
fi

sampler_pid=
cleanup() {
    if [[ -n $sampler_pid ]] && kill -0 "$sampler_pid" 2>/dev/null; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

printf 'Capturing the production prefill with Nsight Systems...\n'
nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu \
    --format=csv -lms 250 >"$AUDIT_DIR/gpu-samples.csv" &
sampler_pid=$!

export DS4_NSYS_CAPTURE_PREFILL=1
export DS4_CUDA_PREFILL_AUDIT=1
nsys profile \
    --trace="$NSYS_TRACE" \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --output="$AUDIT_DIR/prefill-2k" \
    ./ds4-bench "${bench_args[@]}" \
        --csv "$AUDIT_DIR/trace.csv" \
        > >(tee "$AUDIT_DIR/trace.stdout.log") \
        2> >(tee "$AUDIT_DIR/trace.stderr.log" >&2)

cleanup
sampler_pid=
unset DS4_NSYS_CAPTURE_PREFILL
unset DS4_CUDA_PREFILL_AUDIT

report="$AUDIT_DIR/prefill-2k.nsys-rep"
[[ -f $report ]] || die "Nsight did not produce $report"

printf 'Exporting trace summaries...\n'
help_reports=$(nsys stats --help-reports 2>&1 || true)
printf '%s\n' "$help_reports" >"$AUDIT_DIR/nsys-help-reports.txt"
for report_name in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum cuda_gpu_trace nvtx_sum; do
    if grep -q "${report_name}" <<<"$help_reports"; then
        nsys stats --report "$report_name" --format csv "$report" \
            >"$AUDIT_DIR/${report_name}.csv" 2>"$AUDIT_DIR/${report_name}.stderr.log" || true
    fi
done
nsys export --type sqlite --force-overwrite=true \
    --output "$AUDIT_DIR/prefill-2k.sqlite" "$report" \
    >"$AUDIT_DIR/nsys-export.stdout.log" \
    2>"$AUDIT_DIR/nsys-export.stderr.log" || true

printf 'Audit complete: %s\n' "$AUDIT_DIR"
printf 'Return the directory as a tar archive; it contains the baseline CSV, exact plan, GPU samples, trace, and summaries.\n'
