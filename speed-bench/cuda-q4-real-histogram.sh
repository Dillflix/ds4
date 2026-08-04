#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Capture the exact routed-expert histogram used by production Q4 gate/up and
down, then derive tile8, tile16, and adaptive 16/8/4 plans without host copies
inside the measured prefill.

Required environment:
  MODEL=/absolute/path/to/full-Q4.gguf

Optional environment:
  PROMPT=/absolute/path/prompt.txt  # default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,2,1,3
  GPU_VRAM=auto
  STAGE_SPLIT=22
  CTX_TOKENS=2048
  PREFILL_CHUNK=2048
  AUDIT_CAPACITY=4096             # records per physical GPU
  CUDA_ARCH=sm_75
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  Q4_HISTOGRAM_DIR=...            # new output directory

The same router counts feed gate/up and down, so one deferred device capture is
sufficient for both projections. The full-Q4 recipe is verified before results
are accepted. Model hashing is deliberately omitted.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
: "${MODEL:?set MODEL to the absolute full-Q4 GGUF path}"
[[ $MODEL == /* ]] || die "MODEL must be an absolute path"
[[ -f $MODEL ]] || die "model not found: $MODEL"

PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,2,1,3}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
CTX_TOKENS=${CTX_TOKENS:-2048}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
AUDIT_CAPACITY=${AUDIT_CAPACITY:-4096}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${Q4_HISTOGRAM_DIR:-$repo_dir/q4-real-histogram-$run_stamp}
while [[ $OUTPUT_DIR != / && $OUTPUT_DIR == */ ]]; do
    OUTPUT_DIR=${OUTPUT_DIR%/}
done

[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "CTX_TOKENS:$CTX_TOKENS" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" \
            "AUDIT_CAPACITY:$AUDIT_CAPACITY" "SKIP_BUILD:$SKIP_BUILD" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT > 0 && STAGE_SPLIT < 43 && CTX_TOKENS > 0 &&
   PREFILL_CHUNK > 0 && AUDIT_CAPACITY > 0 )) || die "invalid capture range"
[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be exactly sm_75"
[[ $SKIP_BUILD == 0 || $SKIP_BUILD == 1 ]] || die "SKIP_BUILD must be 0 or 1"
[[ $CREATE_ARCHIVE == 0 || $CREATE_ARCHIVE == 1 ]] ||
    die "CREATE_ARCHIVE must be 0 or 1"
for tool in awk env find git grep make mv nproc nvidia-smi python3 stat tar tee tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
(( ${#gpu_ids[@]} == 4 )) || die "GPU_DEVICES must contain exactly four devices"
declare -A seen_gpus=()
for gpu in "${gpu_ids[@]}"; do
    [[ $gpu =~ ^[0-9]+$ ]] || die "invalid GPU index: $gpu"
    [[ -z ${seen_gpus[$gpu]+x} ]] || die "duplicate GPU index: $gpu"
    seen_gpus[$gpu]=1
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "GPU $gpu is not SM75 (${cap:-unknown})"
done

if [[ -d $OUTPUT_DIR ]] &&
   [[ -n $(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    die "Q4_HISTOGRAM_DIR already exists and is not empty: $OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
archive_ready=1
current_phase=initialization
finalize() {
    local status=$?
    trap - EXIT
    if [[ $archive_ready == 1 && -d $OUTPUT_DIR ]]; then
        printf 'state=finished\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
            "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >"$OUTPUT_DIR/run-status.txt.tmp"
        mv -f "$OUTPUT_DIR/run-status.txt.tmp" "$OUTPUT_DIR/run-status.txt"
        if [[ $CREATE_ARCHIVE == 1 ]]; then
            if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$OUTPUT_DIR.tar.gz" \
                    "$(basename "$OUTPUT_DIR")"; then
                printf 'Archive to return: %s.tar.gz\n' "$OUTPUT_DIR"
            else
                status=1
            fi
        fi
    fi
    exit "$status"
}
trap finalize EXIT

mapfile -t inherited_ds4_envs < <(
    env | awk -F= '$1 ~ /^DS4_/ { print $1 }' | sort -u
)
clean_prefix=(env)
for name in "${inherited_ds4_envs[@]}"; do clean_prefix+=(-u "$name"); done

if [[ $SKIP_BUILD == 0 ]]; then
    current_phase=build
    make -B -j"$(nproc)" ds4-bench CUDA_ARCH="$CUDA_ARCH" \
        2>&1 | tee "$OUTPUT_DIR/build.log"
fi
[[ -x ./ds4-bench ]] || die "ds4-bench is missing or not executable"

current_phase=manifest
{
    printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'git_commit=%s\ngit_branch=%s\n' \
        "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\ngpu_vram=%s\nstage_split=%s\n' \
        "$GPU_DEVICES" "$GPU_VRAM" "$STAGE_SPLIT"
    printf 'ctx_tokens=%s\nprefill_chunk=%s\naudit_capacity=%s\n' \
        "$CTX_TOKENS" "$PREFILL_CHUNK" "$AUDIT_CAPACITY"
    printf 'model_hashing=disabled\n'
    printf '\n[gpu]\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,ecc.mode.current,driver_version \
        --format=csv
    printf '\n[topology]\n'
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"

current_phase=production-capture
printf 'Capturing exact production Q4 routed-expert counts...\n'
"${clean_prefix[@]}" \
    DS4_CUDA_EP_STAGE_SPLIT="$STAGE_SPLIT" \
    DS4_CUDA_PREFILL_PIPELINE=1 \
    DS4_CUDA_PREFILL_PIPELINE_MB=512 \
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
    DS4_BENCH_ROUTED_QUANT_AUDIT=1 \
    DS4_CUDA_PREFILL_TILE_AUDIT_CSV="$OUTPUT_DIR/tile-audit-wide.csv" \
    DS4_CUDA_PREFILL_TILE_AUDIT_CAPACITY="$AUDIT_CAPACITY" \
    ./ds4-bench \
        -m "$MODEL" \
        --prompt-file "$PROMPT" \
        --cuda --cuda-tensor-parallel \
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
        --ctx-start "$CTX_TOKENS" --ctx-max "$CTX_TOKENS" \
        --ctx-alloc "$((CTX_TOKENS + 1))" --step-incr "$CTX_TOKENS" \
        --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0 \
        --csv "$OUTPUT_DIR/benchmark.csv" \
        2>&1 | tee "$OUTPUT_DIR/benchmark.log"

audit_count=$(grep -c '^ds4: routed-quant-audit layer=' \
    "$OUTPUT_DIR/benchmark.log" || true)
recipe_count=$(grep -Ec \
    '^ds4: routed-quant-audit layer=[0-9]+ gate=q4_k up=q4_k down=q4_k$' \
    "$OUTPUT_DIR/benchmark.log" || true)
[[ $audit_count == 43 && $recipe_count == 43 ]] ||
    die "expected 43 exact full-Q4 recipe records; got $recipe_count/$audit_count"
grep -Fq 'wrote CUDA tile audit' "$OUTPUT_DIR/benchmark.log" ||
    die "benchmark did not report a completed deferred tile audit"
[[ -s $OUTPUT_DIR/tile-audit-wide.csv ]] || die "tile audit CSV is empty"

current_phase=summarize
python3 speed-bench/summarize-q4-real-histogram.py \
    "$OUTPUT_DIR/tile-audit-wide.csv" \
    "$OUTPUT_DIR/tile-plan.csv" \
    "$OUTPUT_DIR/expert-counts.csv" \
    "$OUTPUT_DIR/expert-count-frequency.csv" \
    | tee "$OUTPUT_DIR/summary.txt"
for required in tile-plan.csv expert-counts.csv expert-count-frequency.csv; do
    [[ -s $OUTPUT_DIR/$required ]] || die "missing summary: $required"
done

current_phase=complete
printf 'Real Q4 histogram capture complete: %s\n' "$OUTPUT_DIR"
