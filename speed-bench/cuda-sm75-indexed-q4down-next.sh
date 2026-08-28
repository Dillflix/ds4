#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
is_uint() { [[ ${1:-} =~ ^[0-9]+$ ]]; }

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

: "${MODEL:?export MODEL=/absolute/path/to/model.gguf}"
PROMPT=${PROMPT:-$ROOT/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
CTX_START=${CTX_START:-8192}
CTX_MAX=${CTX_MAX:-32768}
CTX_ALLOC=${CTX_ALLOC:-32769}
STEP_MUL=${STEP_MUL:-4}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
PIPELINE_MB=${PIPELINE_MB:-512}
REPEATS=${REPEATS:-2}
HARNESS_REPEATS=${HARNESS_REPEATS:-7}
SKIP_BUILD=${SKIP_BUILD:-0}
RUN_PRODUCTION=${RUN_PRODUCTION:-1}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
RESUME=${RESUME:-0}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/sm75-indexed-q4down-next-$(date -u +%Y%m%dT%H%M%SZ)}

[[ -f $MODEL ]] || die "model not found: $MODEL"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for pair in \
    "STAGE_SPLIT:$STAGE_SPLIT" "CTX_START:$CTX_START" "CTX_MAX:$CTX_MAX" \
    "CTX_ALLOC:$CTX_ALLOC" "STEP_MUL:$STEP_MUL" \
    "PREFILL_CHUNK:$PREFILL_CHUNK" "PIPELINE_MB:$PIPELINE_MB" \
    "REPEATS:$REPEATS" "HARNESS_REPEATS:$HARNESS_REPEATS" \
    "SKIP_BUILD:$SKIP_BUILD" "RUN_PRODUCTION:$RUN_PRODUCTION" \
    "CREATE_ARCHIVE:$CREATE_ARCHIVE" "RESUME:$RESUME"; do
    name=${pair%%:*}; value=${pair#*:}
    is_uint "$value" || die "$name must be an unsigned integer"
done
(( STAGE_SPLIT == 22 )) || die "this experiment requires fixed 22/21 placement"
(( CTX_START > 0 && CTX_MAX >= CTX_START && CTX_ALLOC > CTX_MAX )) ||
    die "invalid context bounds"
(( REPEATS > 0 && HARNESS_REPEATS > 0 )) || die "repeat counts must be positive"
(( SKIP_BUILD <= 1 && RUN_PRODUCTION <= 1 && CREATE_ARCHIVE <= 1 &&
   RESUME <= 1 )) ||
    die "boolean controls must be 0 or 1"

mkdir -p "$OUTPUT_DIR"/{harness,production,provenance}
phase=setup
finish() {
    status=$?
    printf 'status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && echo complete || echo failed)" "$phase" \
        >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        archive="$OUTPUT_DIR.tar.gz"
        tar -C "$(dirname "$OUTPUT_DIR")" -czf "$archive" \
            "$(basename "$OUTPUT_DIR")" || status=1
        printf 'Archive to return: %s\n' "$archive"
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM HUP

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done

phase=build
targets=(ds4-bench tests/cuda_long_context_smoke
         tests/cuda_sm75_profile_harness tests/test_engine_mgpu_placement)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q "${targets[@]}" CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi

phase=regression
./tests/test_engine_mgpu_placement >"$OUTPUT_DIR/placement-tests.log" 2>&1 ||
    die "placement regression failed"
./tests/cuda_long_context_smoke >"$OUTPUT_DIR/cuda-regression.log" 2>&1 || {
    tail -n 160 "$OUTPUT_DIR/cuda-regression.log" >&2
    die "CUDA exact-output regression failed"
}
grep -Fq 'indexed attention 16-head/512-thread versus 8-head/256-thread whole and sharded exact' \
    "$OUTPUT_DIR/cuda-regression.log" || die "indexed-attention exact proof missing"
[[ $(grep -Fc 'tile16-stage8 exact' "$OUTPUT_DIR/cuda-regression.log") == 2 ]] ||
    die "Q4-32 stage8 exact proofs missing"

phase=bounded-harness
for variant in control indexed8; do
    heads8=0; [[ $variant == indexed8 ]] && heads8=1
    "${clean[@]}" DS4_CUDA_INDEXED_HEADS8_SM75=$heads8 \
        DS4_PROFILE_REPEATS=$HARNESS_REPEATS \
        ./tests/cuda_sm75_profile_harness attn-indexed-32k \
        >"$OUTPUT_DIR/harness/attention-$variant.log" 2>&1 ||
        die "bounded indexed-attention $variant failed"
done
for variant in control down-stage8; do
    stage8=0; [[ $variant == down-stage8 ]] && stage8=1
    "${clean[@]}" DS4_CUDA_MOE_Q4_32_DOWN_STAGE8_SM75=$stage8 \
        DS4_PROFILE_REPEATS=$HARNESS_REPEATS \
        ./tests/cuda_sm75_profile_harness sm75-q4-32 \
        >"$OUTPUT_DIR/harness/q4down-$variant.log" 2>&1 ||
        die "bounded Q4-32 down $variant failed"
done
cuobjdump --dump-resource-usage ./tests/cuda_sm75_profile_harness \
    >"$OUTPUT_DIR/harness/resource-usage.txt" 2>&1 || true

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nmodel_hashing=disabled\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\nstage_split=22/21\nctx_start=%s\nctx_max=%s\nctx_alloc=%s\nrepeats=%s\n' \
        "$GPU_DEVICES" "$CTX_START" "$CTX_MAX" "$CTX_ALLOC" "$REPEATS"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,clocks.max.sm,power.limit \
        --format=csv
    printf '\ntopology:\n'; nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

if [[ $RUN_PRODUCTION == 1 ]]; then
    phase=production-ab
    printf 'repeat,slot,variant,csv,log,logits\n' \
        >"$OUTPUT_DIR/production/runs.csv"
    variants=(control indexed8 down-stage8 both)
    for ((repeat=1; repeat<=REPEATS; repeat++)); do
        if (( repeat % 2 == 0 )); then
            order=(both down-stage8 indexed8 control)
        else
            order=(control indexed8 down-stage8 both)
        fi
        declare -A logits_by_variant=()
        slot=0
        for variant in "${order[@]}"; do
            slot=$((slot + 1)); heads8=0; stage8=0
            [[ $variant == indexed8 || $variant == both ]] && heads8=1
            [[ $variant == down-stage8 || $variant == both ]] && stage8=1
            base="$OUTPUT_DIR/production/r${repeat}-s${slot}-$variant"
            logits="$base-logits"
            reusable=0
            if [[ $RESUME == 1 && -s $base.csv && -s $base.log &&
                  -d $logits &&
                  -s $logits/frontier_$(printf '%06d' "$CTX_START").logits.f32 &&
                  -s $logits/frontier_$(printf '%06d' "$CTX_MAX").logits.f32 ]]; then
                reusable=1
            fi
            if [[ $reusable == 1 ]]; then
                printf 'Reusing production A/B repeat=%d/%d slot=%d variant=%s...\n' \
                    "$repeat" "$REPEATS" "$slot" "$variant"
            else
                [[ $base == "$OUTPUT_DIR"/production/r*-s*-* &&
                   $logits == "$OUTPUT_DIR"/production/r*-s*-*-logits ]] ||
                    die "refusing to replace output outside the production run directory"
                rm -rf -- "$logits"
                rm -f -- "$base.csv" "$base.log"
                mkdir -p "$logits"
                printf 'Production candidate A/B repeat=%d/%d slot=%d variant=%s...\n' \
                    "$repeat" "$REPEATS" "$slot" "$variant"
                "${clean[@]}" \
                    DS4_CUDA_EP_STAGE_SPLIT=22 \
                    DS4_CUDA_PREFILL_PIPELINE=1 \
                    "DS4_CUDA_PREFILL_PIPELINE_MB=$PIPELINE_MB" \
                    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
                    DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=2048 \
                    DS4_CUDA_TP_PREFILL_ATTN_HEADS=0 \
                    DS4_CUDA_TP_PREFILL_ATTN_ROWS=1 \
                    DS4_CUDA_TP_PREFILL_ATTN_ROWS_AUDIT=1 \
                    "DS4_CUDA_INDEXED_HEADS8_SM75=$heads8" \
                    "DS4_CUDA_MOE_Q4_32_DOWN_STAGE8_SM75=$stage8" \
                    ./ds4-bench --cuda --cuda-tensor-parallel \
                        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                        --model "$MODEL" --prompt-file "$PROMPT" \
                        --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
                        --ctx-alloc "$CTX_ALLOC" --step-mul "$STEP_MUL" \
                        --prefill-chunk "$PREFILL_CHUNK" --gen-tokens 0 \
                        --csv "$base.csv" --dump-frontier-logits-dir "$logits" \
                        >"$base.log" 2>&1 || {
                            tail -n 200 "$base.log" >&2
                            die "$variant production run failed"
                        }
            fi
            grep -Fq 'materialized 344/344 candidates' "$base.log" ||
                die "$variant did not retain complete dense-F16 admission"
            grep -Fq 'dense-placement=stage-aware-fixed-22-21' "$base.log" ||
                die "$variant missed fixed 22/21 dense placement"
            grep -Fq 'dispatch=split kind=indexed' "$base.log" ||
                die "$variant omitted indexed row splitting"
            if [[ $heads8 == 1 ]]; then
                grep -Fq 'indexed attention candidate selected: 8 heads / 256 threads' \
                    "$base.log" ||
                    die "$variant omitted the indexed8 dispatch"
            fi
            if [[ $stage8 == 1 ]]; then
                grep -Fq 'Q4-32 down candidate selected: tile16 stage8' "$base.log" ||
                    die "$variant omitted the Q4-32 stage8 dispatch"
            fi
            logits_by_variant[$variant]=$logits
            printf '%s,%s,%s,%s,%s,%s\n' "$repeat" "$slot" "$variant" \
                "$base.csv" "$base.log" "$logits" \
                >>"$OUTPUT_DIR/production/runs.csv"
        done
        for variant in indexed8 down-stage8 both; do
            diff -rq "${logits_by_variant[control]}" \
                     "${logits_by_variant[$variant]}" \
                >"$OUTPUT_DIR/production/r${repeat}-$variant-logits.diff" ||
                die "$variant changed production frontier logits"
        done
    done
    python3 speed-bench/summarize-sm75-indexed-q4down-next.py \
        "$OUTPUT_DIR/production/runs.csv" \
        "$OUTPUT_DIR/production/summary.csv"
    column -s, -t <"$OUTPUT_DIR/production/summary.csv" ||
        cat "$OUTPUT_DIR/production/summary.csv"
fi

phase=complete
printf 'SM75 indexed-attention/Q4-down candidate A/B complete: %s\n' "$OUTPUT_DIR"
