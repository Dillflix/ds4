#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
: "${MODEL:?set MODEL to the absolute full-Q4 GGUF path}"
[[ $MODEL == /* && -f $MODEL ]] || die "MODEL must name an existing absolute path"

PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
CTX_START=${CTX_START:-2048}
CTX_MAX=${CTX_MAX:-8192}
STEP_MUL=${STEP_MUL:-2}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
REPEATS=${REPEATS:-4}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${Q8_CACHE_BENEFIT_DIR:-$repo_dir/q8-cache-benefit-ab-$stamp}

[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for v in "$CTX_START" "$CTX_MAX" "$STEP_MUL" "$PREFILL_CHUNK" "$REPEATS" "$SKIP_BUILD"; do
    [[ $v =~ ^[0-9]+$ ]] || die "numeric options must be integers"
done
(( REPEATS >= 2 && REPEATS % 2 == 0 )) || die "REPEATS must be a positive even number"
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] || die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{runs,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
finish() {
    status=$?
    trap - EXIT INT TERM
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" "$status" "$phase" \
        >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        tar -C "$(dirname "$OUTPUT_DIR")" -czf "$OUTPUT_DIR.tar.gz" "$(basename "$OUTPUT_DIR")" || status=1
        printf 'Archive to return: %s.tar.gz\n' "$OUTPUT_DIR"
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM

mapfile -t ds4_env < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${ds4_env[@]}"; do clean+=(-u "$name"); done

if [[ $SKIP_BUILD == 0 ]]; then
    phase=build
    make -B -j"$(nproc)" ds4-bench tests/test_engine_mgpu_placement \
        tests/cuda_long_context_smoke CUDA_ARCH=sm_75 2>&1 | tee "$OUTPUT_DIR/build.log"
    ./tests/test_engine_mgpu_placement >"$OUTPUT_DIR/planner-unit.log" 2>&1 || die "planner unit test failed"
    ./tests/cuda_long_context_smoke >"$OUTPUT_DIR/cuda-smoke.log" 2>&1 || die "CUDA smoke failed"
else
    make -q ds4-bench tests/test_engine_mgpu_placement tests/cuda_long_context_smoke \
        CUDA_ARCH=sm_75 || die "SKIP_BUILD=1 found stale targets"
fi

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\nmodel=%s\nmodel_bytes=%s\nprompt=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" "$MODEL" \
        "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\nstage_split=%s/%s\nrepeats=%s\nmodel_hashing=disabled\n' \
        "$GPU_DEVICES" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))" "$REPEATS"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free --format=csv
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
printf 'repeat\tslot\tvariant\tcsv\tlog\tcache_before\tcache_after\n' >"$OUTPUT_DIR/runs.tsv"

phase=benchmarks
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    if (( repeat % 2 )); then order=(benefit-plan first-use); else order=(first-use benefit-plan); fi
    slot=0
    for variant in "${order[@]}"; do
        slot=$((slot+1))
        stem="$variant-r$repeat"
        csv="$OUTPUT_DIR/runs/$stem.csv"
        log="$OUTPUT_DIR/runs/$stem.log"
        before="$OUTPUT_DIR/runs/$stem.cache-before.csv"
        after="$OUTPUT_DIR/runs/$stem.cache-after.csv"
        extra=()
        [[ $variant == first-use ]] && extra+=(DS4_CUDA_Q8_F16_FIRST_USE=1)
        printf 'Benchmarking %s repeat=%d/%d slot=%d\n' "$variant" "$repeat" "$REPEATS" "$slot"
        "${clean[@]}" "${extra[@]}" \
            "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
            DS4_CUDA_PREFILL_PIPELINE=1 DS4_CUDA_PREFILL_PIPELINE_MB=512 \
            DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
            "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$CTX_START" \
            "DS4_CUDA_Q8_CACHE_PRETIMING_STATE_CSV=$before" \
            "DS4_CUDA_Q8_CACHE_STATE_CSV=$after" \
            ./ds4-bench --cuda --cuda-tensor-parallel \
                --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                --model "$MODEL" --prompt-file "$PROMPT" \
                --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
                --ctx-alloc "$((CTX_MAX+1))" --step-mul "$STEP_MUL" \
                --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0 --csv "$csv" \
                >"$log" 2>&1 || { tail -n 120 "$log" >&2; die "$stem failed"; }
        [[ -s $csv && -s $before && -s $after ]] || die "$stem omitted required evidence"
        cmp -s "$before" "$after" || die "$stem cache changed during timed frontiers"
        if [[ $variant == benefit-plan ]]; then
            grep -Fq 'q8 fp16 benefit plan registered' "$log" || die "$stem did not register a plan"
            grep -Eq 'q8 fp16 benefit plan materialized [0-9]+/[0-9]+.*T32-q_b=[1-9][0-9]*/[1-9][0-9]* T256-output_b=[1-9][0-9]*/[1-9][0-9]*' "$log" ||
                die "$stem did not materialize T32/T256 candidates"
        else
            grep -Fq 'q8 fp16 planner disabled; using legacy first-use admission' "$log" ||
                die "$stem did not use first-use admission"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$repeat" "$slot" "$variant" \
            "$csv" "$log" "$before" "$after" >>"$OUTPUT_DIR/runs.tsv"
        cat "$csv"
    done
done

phase=summarize
python3 speed-bench/summarize-q8-cache-benefit-ab.py "$OUTPUT_DIR/runs.tsv" "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/summary-stdout.txt"
[[ -s $OUTPUT_DIR/paired-samples.csv && -s $OUTPUT_DIR/summary.txt ]] || die "summary missing"
phase=complete
printf 'Q8 cache benefit-plan A/B complete: %s\n' "$OUTPUT_DIR"
