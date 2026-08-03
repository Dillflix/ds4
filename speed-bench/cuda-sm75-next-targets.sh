#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run controlled SM75 A/B prefill benchmarks for the next kernel targets.

Required:
  MODEL=/absolute/path/model.gguf
  RECIPE=hybrid|full-q4|stock-q2

Optional:
  PROMPT=/absolute/path/prompt.txt
  PROMPT_MANIFEST=/absolute/path/prompts.tsv  # label<TAB>path
  GPU_DEVICES=0,2,1,3
  GPU_VRAM=auto
  CTX_START=2048
  CTX_MAX=65536
  STEP_MUL=2
  PREFILL_CHUNK=2048
  REPEATS=1
  CUDA_ARCH=sm_75
  SKIP_BUILD=0
  TARGETS_DIR=/absolute/path/output-directory

Every candidate is opt-in and compared with the untouched production
baseline. The combined variant is deliberately last.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
: "${MODEL:?set MODEL to the absolute GGUF path}"
: "${RECIPE:?set RECIPE to hybrid, full-q4, or stock-q2}"
[[ -f $MODEL ]] || die "model not found: $MODEL"
[[ $RECIPE == hybrid || $RECIPE == full-q4 || $RECIPE == stock-q2 ]] ||
    die "RECIPE must be hybrid, full-q4, or stock-q2"

GPU_DEVICES=${GPU_DEVICES:-0,2,1,3}
GPU_VRAM=${GPU_VRAM:-auto}
CTX_START=${CTX_START:-2048}
CTX_MAX=${CTX_MAX:-65536}
STEP_MUL=${STEP_MUL:-2}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
REPEATS=${REPEATS:-1}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
SKIP_BUILD=${SKIP_BUILD:-0}
TARGETS_DIR=${TARGETS_DIR:-$repo_dir/sm75-next-targets-$(date -u +%Y%m%dT%H%M%SZ)}
for item in "CTX_START:$CTX_START" "CTX_MAX:$CTX_MAX" \
            "STEP_MUL:$STEP_MUL" "PREFILL_CHUNK:$PREFILL_CHUNK" \
            "REPEATS:$REPEATS" "SKIP_BUILD:$SKIP_BUILD"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( CTX_START > 0 && CTX_MAX >= CTX_START && STEP_MUL >= 1 &&
   PREFILL_CHUNK > 0 && REPEATS > 0 )) || die "invalid benchmark range"
[[ $SKIP_BUILD == 0 || $SKIP_BUILD == 1 ]] || die "SKIP_BUILD must be 0 or 1"

declare -a prompt_labels=() prompt_paths=()
if [[ -n ${PROMPT_MANIFEST:-} ]]; then
    [[ -f $PROMPT_MANIFEST ]] || die "prompt manifest not found: $PROMPT_MANIFEST"
    while IFS=$'\t' read -r label path extra; do
        [[ -n $label && ${label:0:1} != '#' ]] || continue
        [[ -n $path && -z ${extra:-} ]] || die "invalid prompt manifest row"
        [[ $path == /* ]] || path="$repo_dir/$path"
        [[ -f $path ]] || die "prompt not found: $path"
        [[ $label =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe prompt label: $label"
        prompt_labels+=("$label"); prompt_paths+=("$path")
    done <"$PROMPT_MANIFEST"
else
    prompt=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
    [[ -f $prompt ]] || die "prompt not found: $prompt"
    prompt_labels+=(promessi); prompt_paths+=("$prompt")
fi
(( ${#prompt_paths[@]} > 0 )) || die "prompt suite is empty"

mkdir -p "$TARGETS_DIR"
TARGETS_DIR=$(cd "$TARGETS_DIR" && pwd)

if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" ds4-bench tests/cuda_long_context_smoke \
        CUDA_ARCH="$CUDA_ARCH"
    ./tests/cuda_long_context_smoke 2>&1 | tee "$TARGETS_DIR/correctness.log"
fi

export DS4_CUDA_EP_STAGE_SPLIT=22
unset DS4_CUDA_Q8_MMA_SM75_TOK16 \
      DS4_CUDA_MOE_Q4_GATE_TILE16_SM75 \
      DS4_CUDA_MOE_Q4_GATE_STAGE4_SM75 \
      DS4_CUDA_MOE_IQ2_STAGE6_SM75 \
      DS4_CUDA_MOE_IQ2_STAGE4_SM75 \
      DS4_CUDA_MOE_MIXED_TAIL_TILES \
      DS4_CUDA_MOE_Q2_DOWN_MMA_SM75 \
      DS4_CUDA_NO_Q8_MMA_SM75_TOK16 \
      DS4_CUDA_MOE_NO_Q4_GATE_TILE16_SM75 \
      DS4_CUDA_MOE_NO_IQ2_MMA_SM75 \
      DS4_CUDA_MOE_NO_IQ2_MMA_TILE16_SM75 \
      DS4_CUDA_MOE_NO_MIXED_TAIL_TILES \
      DS4_CUDA_MOE_NO_Q2_DOWN_MMA_SM75
common=(
    --cuda --cuda-tensor-parallel
    --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM"
    --model "$MODEL" --prefill-chunk "$PREFILL_CHUNK"
    --ctx-start "$CTX_START" --ctx-max "$CTX_MAX"
    --step-mul "$STEP_MUL" --gen-tokens 0
)

declare -a names=(baseline)
declare -a envs=("")
case "$RECIPE" in
    full-q4)
        names+=(q8-tok16 q4-tile16-stage6 q4-tile16-stage4 mixed-tails combined)
        envs+=(
            "DS4_CUDA_Q8_MMA_SM75_TOK16=1"
            "DS4_CUDA_MOE_Q4_GATE_TILE16_SM75=1"
            "DS4_CUDA_MOE_Q4_GATE_TILE16_SM75=1 DS4_CUDA_MOE_Q4_GATE_STAGE4_SM75=1"
            "DS4_CUDA_MOE_Q4_GATE_TILE16_SM75=1 DS4_CUDA_MOE_MIXED_TAIL_TILES=1"
            "DS4_CUDA_Q8_MMA_SM75_TOK16=1 DS4_CUDA_MOE_Q4_GATE_TILE16_SM75=1 DS4_CUDA_MOE_MIXED_TAIL_TILES=1"
        )
        ;;
    hybrid)
        names+=(q8-tok16 iq2-stage6 iq2-stage4 mixed-tails combined)
        envs+=(
            "DS4_CUDA_Q8_MMA_SM75_TOK16=1"
            "DS4_CUDA_MOE_IQ2_STAGE6_SM75=1"
            "DS4_CUDA_MOE_IQ2_STAGE4_SM75=1"
            "DS4_CUDA_MOE_MIXED_TAIL_TILES=1"
            "DS4_CUDA_Q8_MMA_SM75_TOK16=1 DS4_CUDA_MOE_IQ2_STAGE6_SM75=1 DS4_CUDA_MOE_MIXED_TAIL_TILES=1"
        )
        ;;
    stock-q2)
        names+=(q8-tok16 iq2-stage6 iq2-stage4 q2-down-mma mixed-tails combined)
        envs+=(
            "DS4_CUDA_Q8_MMA_SM75_TOK16=1"
            "DS4_CUDA_MOE_IQ2_STAGE6_SM75=1"
            "DS4_CUDA_MOE_IQ2_STAGE4_SM75=1"
            "DS4_CUDA_MOE_Q2_DOWN_MMA_SM75=1"
            "DS4_CUDA_MOE_MIXED_TAIL_TILES=1"
            "DS4_CUDA_Q8_MMA_SM75_TOK16=1 DS4_CUDA_MOE_IQ2_STAGE6_SM75=1 DS4_CUDA_MOE_Q2_DOWN_MMA_SM75=1 DS4_CUDA_MOE_MIXED_TAIL_TILES=1"
        )
        ;;
esac

{
    printf 'model=%s\nrecipe=%s\ngit_commit=%s\n' \
        "$MODEL" "$RECIPE" "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
    printf 'gpu_devices=%s\ngpu_vram=%s\nsplit=22/21\n' \
        "$GPU_DEVICES" "$GPU_VRAM"
    printf 'ctx_start=%s\nctx_max=%s\nstep_mul=%s\nprefill_chunk=%s\nrepeats=%s\n' \
        "$CTX_START" "$CTX_MAX" "$STEP_MUL" "$PREFILL_CHUNK" "$REPEATS"
    printf 'date_utc=%s\n\nGPU inventory:\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
} >"$TARGETS_DIR/manifest.txt"

{
    printf 'variant\tenvironment\n'
    for i in "${!names[@]}"; do
        printf '%s\t%s\n' "${names[$i]}" "${envs[$i]:-(production baseline)}"
    done
} >"$TARGETS_DIR/variants.tsv"

for i in "${!names[@]}"; do
    name=${names[$i]}
    variant_env=${envs[$i]}
    for p in "${!prompt_paths[@]}"; do
        label=${prompt_labels[$p]}
        prompt_path=${prompt_paths[$p]}
        for ((r=1; r<=REPEATS; r++)); do
            csv="$TARGETS_DIR/${name}-${label}-r${r}.csv"
            log="$TARGETS_DIR/${name}-${label}-r${r}.log"
            printf 'Benchmarking %s prompt=%s repeat=%d/%d\n' \
                "$name" "$label" "$r" "$REPEATS"
            if [[ -n $variant_env ]]; then
                read -r -a assignments <<<"$variant_env"
                env "${assignments[@]}" ./ds4-bench "${common[@]}" \
                    --prompt-file "$prompt_path" --csv "$csv" 2>&1 | tee "$log"
            else
                ./ds4-bench "${common[@]}" --prompt-file "$prompt_path" \
                    --csv "$csv" 2>&1 | tee "$log"
            fi
        done
    done
done

tar -C "$(dirname "$TARGETS_DIR")" -czf "$TARGETS_DIR.tar.gz" \
    "$(basename "$TARGETS_DIR")"
printf 'SM75 target benchmark complete: %s\nArchive to return: %s.tar.gz\n' \
    "$TARGETS_DIR" "$TARGETS_DIR"
