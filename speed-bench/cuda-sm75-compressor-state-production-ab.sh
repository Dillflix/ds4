#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

MIXED_MODEL=${MIXED_MODEL:-$repo_dir/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf}
ALL43_MODEL=${ALL43_MODEL:-$repo_dir/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf}
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
REQUIRED_POWER_LIMITS_W=${REQUIRED_POWER_LIMITS_W:-250,260,250,250}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
TG_TOKENS=${TG_TOKENS:-256}
EXACT_TOKENS=${EXACT_TOKENS:-16}
PREFILL_CHUNK=2048
PIPELINE_MB=512
CTX_ALLOC=33025
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${COMPRESSOR_STATE_PRODUCTION_AB_DIR:-$repo_dir/sm75-compressor-state-production-ab-$stamp}
layouts=(mixed15 all43)
models=("$MIXED_MODEL" "$ALL43_MODEL")

for model in "${models[@]}"; do
    [[ $model == /* && -f $model ]] || die "model not found at absolute path: $model"
done
[[ $MIXED_MODEL != "$ALL43_MODEL" ]] || die "the model paths must differ"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "this production A/B requires the fixed promessi_sposi prompt"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "require GPU_DEVICES=0,3,1,2, GPU_VRAM=auto, STAGE_SPLIT=22"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ $TG_TOKENS =~ ^[1-9][0-9]*$ && $EXACT_TOKENS =~ ^[1-9][0-9]*$ ]] ||
    die "TG_TOKENS and EXACT_TOKENS must be positive integers"
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"
for tool in awk basename cmp date dirname env find git grep make mkdir mv nproc \
            nvidia-smi sort stat tail tar tee tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
IFS=, read -r -a required_power <<<"$REQUIRED_POWER_LIMITS_W"
(( ${#gpu_ids[@]} == 4 && ${#required_power[@]} == 4 )) ||
    die "GPU and power-limit lists must each contain four entries"
declare -A seen_gpu=()
for gpu in 0 1 2 3; do
    [[ ${gpu_ids[$gpu]} =~ ^[0-9]+$ && -z ${seen_gpu[${gpu_ids[$gpu]}]+x} ]] ||
        die "invalid or duplicate GPU ID: ${gpu_ids[$gpu]}"
    seen_gpu[${gpu_ids[$gpu]}]=1
    cap=$(nvidia-smi -i "${gpu_ids[$gpu]}" --query-gpu=compute_cap \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "GPU ${gpu_ids[$gpu]} is not SM75"
    limit=$(nvidia-smi -i "$gpu" --query-gpu=power.limit \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    awk -v actual="$limit" -v expected="${required_power[$gpu]}" \
        'BEGIN {exit !((actual+0)==(expected+0))}' ||
        die "physical GPU $gpu power limit is ${limit:-unknown} W, expected ${required_power[$gpu]} W"
done

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{runs,summary,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
finish() {
    local status=$? archive="$OUTPUT_DIR.tar.gz" partial="$OUTPUT_DIR.tar.gz.partial.$$"
    trap - EXIT INT TERM HUP
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && mv -f -- "$partial" "$archive"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
            printf 'error: could not create %s\n' "$archive" >&2
        fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM HUP

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done
printf '%s\n' "${inherited_ds4[@]:-}" >"$OUTPUT_DIR/provenance/cleared-ds4-env.txt"

production_env=(
    "${clean[@]}"
    DS4_CUDA_EP_STAGE_SPLIT=22
    DS4_CUDA_PREFILL_PIPELINE=1
    "DS4_CUDA_PREFILL_PIPELINE_MB=$PIPELINE_MB"
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK"
    DS4_BENCH_UNTIMED_WARMUP_TOKENS=512
    DS4_CUDA_NO_TP_PREFILL_ATTN_ROWS_PAIRS=0
    DS4_CUDA_TP_PREFILL_INDEXER_ROWS_PAIRS=0,1
    DS4_CUDA_TP_PREFILL_INDEXER_ROWS_AUDIT=1
    DS4_CUDA_COMPRESSOR_PAIR_STATE_STORE_AUDIT=1
    DS4_BENCH_ROUTED_QUANT_AUDIT=1
)

phase=build
if [[ $SKIP_BUILD == 0 ]]; then
    make -j"$(nproc)" ds4-bench CUDA_ARCH=sm_75 2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q ds4-bench CUDA_ARCH=sm_75 || die "SKIP_BUILD=1 found a stale ds4-bench"
fi

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'mixed_model=%s\nall43_model=%s\nprompt=%s\n' \
        "$MIXED_MODEL" "$ALL43_MODEL" "$PROMPT"
    printf 'gpu_devices=%s\npower_limits_w=%s\nstage_split=22/21\n' \
        "$GPU_DEVICES" "$REQUIRED_POWER_LIMITS_W"
    printf 'contexts=512,4096,32768\nrepeats=1\ncomparison=reference-vs-fused-pair-state-store\n'
    printf 'tg_tokens=%s\nexact_tokens=%s\n' "$TG_TOKENS" "$EXACT_TOKENS"
    nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,serial,power.limit,memory.total,compute_cap \
        --format=csv
    printf '\ntopology:\n'
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

capture_gpu_health() {
    local output=$1 partial="$1.partial.$$" gpu
    : >"$partial"
    for gpu in "${gpu_ids[@]}"; do
        nvidia-smi -i "$gpu" --query-gpu=index,pci.bus_id,uuid,power.limit \
            --format=csv,noheader,nounits >>"$partial" 2>&1 || {
                mv -- "$partial" "$output"; return 1;
            }
    done
    mv -- "$partial" "$output"
}

validate_health() {
    local base=$1
    [[ -s $base.pre-gpu.csv && -s $base.post-gpu.csv ]] &&
        ! grep -Eq 'ERR!|Unknown Error|GPU is lost' \
            "$base.pre-gpu.csv" "$base.post-gpu.csv" &&
        cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$base.pre-gpu.csv" &&
        cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$base.post-gpu.csv"
}

validate_throughput() {
    local base=$1
    awk -F, '
        NR==1 {header=($1=="ctx_tokens" && $4=="gen_tokens" &&
                       $8=="gen_steady_tps"); next}
        NR==2 {rows++; a=($1==512 && ($4+0)==tg && ($8+0)>0); next}
        NR==3 {rows++; b=($1==4096 && ($4+0)==tg && ($8+0)>0); next}
        NR==4 {rows++; c=($1==32768 && ($4+0)==tg && ($8+0)>0); next}
        NR>4 {rows++}
        END {exit !(header && rows==3 && a && b && c)}
    ' tg="$TG_TOKENS" "$base.csv"
}

validate_exact() {
    local base=$1 context=$2 logits=$3 token file
    awk -F, -v ctx="$context" -v tg="$EXACT_TOKENS" '
        NR==1 {header=($1=="ctx_tokens" && $4=="gen_tokens" &&
                       $8=="gen_steady_tps"); next}
        NR==2 {rows++; ok=($1==ctx && ($4+0)==tg && ($8+0)>0); next}
        NR>2 {rows++}
        END {exit !(header && rows==1 && ok)}
    ' "$base.csv" || return 1
    for ((token=1; token<=EXACT_TOKENS; token++)); do
        printf -v file 'frontier_%06d.decode_%06d.logits.f32' "$context" "$token"
        [[ -s $logits/$file ]] || return 1
    done
}

validate_topology() {
    local log=$1 require_row_split=${2:-1} marker route
    for marker in 'CUDA EP forced pipeline split 22/21' \
                  't256-placement=stage-aware' \
                  'dense-placement=stage-aware-fixed-22-21' \
                  'materialized 344/344 candidates'; do
        grep -Fq "$marker" "$log" || return 1
    done
    for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
        grep -Fq "$route" "$log" || return 1
    done
    ! grep -Fq 'required but unavailable' "$log" || return 1
    if [[ $require_row_split == 1 ]]; then
        ! grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" || return 1
        grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" || return 1
        grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" || return 1
        grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log" || return 1
    fi
}

validate_selector() {
    local variant=$1 log=$2 width ratio marker
    for spec in 256:4 512:128 1024:4; do
        width=${spec%%:*}; ratio=${spec#*:}
        marker="SM75 compressor pair/state fusion selected width=$width ratio=$ratio"
        if [[ $variant == control ]]; then
            ! grep -Fq "$marker" "$log" || return 1
        else
            [[ $(grep -Fc "$marker" "$log") == 1 ]] || return 1
        fi
    done
}

run_arm() {
    local model=$1 variant=$2 tokens=$3 start=$4 max=$5 base=$6 logits=${7:-} rc=0
    local -a selector cmd
    if [[ $variant == control ]]; then
        selector=(DS4_CUDA_DISABLE_COMPRESSOR_PAIR_STATE_STORE=1)
    else
        selector=(DS4_CUDA_ENABLE_COMPRESSOR_PAIR_STATE_STORE=1)
    fi
    capture_gpu_health "$base.pre-gpu.csv" || return 1
    cmd=("${production_env[@]}" "${selector[@]}" ./ds4-bench \
        --cuda --cuda-tensor-parallel \
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
        --model "$model" --prompt-file "$PROMPT" \
        --ctx-start "$start" --ctx-max "$max" --ctx-alloc "$CTX_ALLOC" \
        --step-mul 8 --prefill-chunk "$PREFILL_CHUNK" --gen-tokens "$tokens")
    [[ -z $logits ]] || cmd+=(--dump-decode-logits-dir "$logits")
    cmd+=(--csv "$base.csv")
    "${cmd[@]}" >"$base.log" 2>&1 || rc=$?
    capture_gpu_health "$base.post-gpu.csv" || return 1
    return "$rc"
}

phase=production-ab
capture_gpu_health "$OUTPUT_DIR/initial-gpu.csv" || die "could not capture initial GPU health"
printf 'layout\tvariant\tcsv\tlog\tlogits\n' >"$OUTPUT_DIR/runs.tsv"
for index in 0 1; do
    layout=${layouts[$index]}; model=${models[$index]}
    if (( index % 2 == 0 )); then
        variants=(control fused)
    else
        variants=(fused control)
    fi
    for variant in "${variants[@]}"; do
        base="$OUTPUT_DIR/runs/$layout-$variant"
        printf 'Compressor/state production A/B model=%s variant=%s...\n' "$layout" "$variant"
        run_arm "$model" "$variant" "$TG_TOKENS" 512 32768 "$base" || {
            tail -n 200 "$base.log" >&2 || true
            die "$layout $variant production run failed"
        }
        validate_health "$base" || die "$layout $variant GPU health changed"
        validate_throughput "$base" || die "$layout $variant decode throughput output is incomplete"
        validate_topology "$base.log" || die "$layout $variant production topology validation failed"
        validate_selector "$variant" "$base.log" || die "$layout $variant fusion dispatch validation failed"
        printf '%s\t%s\t%s\t%s\t%s\n' "$layout" "$variant" \
            "$base.csv" "$base.log" "-" >>"$OUTPUT_DIR/runs.tsv"
    done
    for context in 512 4096 32768; do
        for variant in control fused; do
            base="$OUTPUT_DIR/runs/$layout-$variant-exact-pp$context"
            logits="$base-logits"
            mkdir -p "$logits"
            printf 'Exact compressor/state decode logits model=%s variant=%s PP=%s...\n' \
                "$layout" "$variant" "$context"
            run_arm "$model" "$variant" "$EXACT_TOKENS" "$context" "$context" \
                "$base" "$logits" || {
                tail -n 200 "$base.log" >&2 || true
                die "$layout $variant PP=$context exact run failed"
            }
            validate_health "$base" || die "$layout $variant PP=$context GPU health changed"
            validate_exact "$base" "$context" "$logits" ||
                die "$layout $variant PP=$context exact output is incomplete"
            # A fixed PP=512 exact run is below the row-split dispatch threshold.
            # Validate the physical/pipeline topology here; the 512..32768 timing
            # arm above separately proves the production row-split policy.
            validate_topology "$base.log" 0 ||
                die "$layout $variant PP=$context topology validation failed"
            validate_selector "$variant" "$base.log" ||
                die "$layout $variant PP=$context fusion dispatch validation failed"
        done
        for ((token=1; token<=EXACT_TOKENS; token++)); do
            printf -v name 'frontier_%06d.decode_%06d.logits.f32' "$context" "$token"
            cmp -s "$OUTPUT_DIR/runs/$layout-control-exact-pp$context-logits/$name" \
                   "$OUTPUT_DIR/runs/$layout-fused-exact-pp$context-logits/$name" ||
                die "$layout fused output diverged at PP=$context decode token $token"
        done
    done
done

phase=summarize
{
    printf '# SM75 four-GPU compressor projection/state-store A/B\n\n'
    printf '| Model | Context | Control decode tok/s | Fused decode tok/s | Speedup |\n'
    printf '| --- | ---: | ---: | ---: | ---: |\n'
    for layout in "${layouts[@]}"; do
        awk -F, -v layout="$layout" '
            NR==FNR {if (FNR>1) control[$1]=$8; next}
            FNR>1 {printf "| %s | %s | %.3f | %.3f | %.6fx |\n", layout,$1,control[$1],$8,$8/control[$1]}
        ' "$OUTPUT_DIR/runs/$layout-control.csv" "$OUTPUT_DIR/runs/$layout-fused.csv"
    done
    printf '\nAll %s decode logits at all three frontiers were byte-identical for both models.\n' "$EXACT_TOKENS"
} | tee "$OUTPUT_DIR/summary/report.md"

phase=complete
printf 'SM75 compressor projection/state-store production A/B complete: %s\n' "$OUTPUT_DIR"
