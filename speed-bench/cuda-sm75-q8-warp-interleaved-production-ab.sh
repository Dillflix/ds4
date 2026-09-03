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
TG_TOKENS=${TG_TOKENS:-256}
EXACT_TOKENS=${EXACT_TOKENS:-16}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
PIPELINE_MB=${PIPELINE_MB:-512}
INTERLEAVED_CACHE_MB=${INTERLEAVED_CACHE_MB:-1024}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
CTX_ALLOC=33025
VOCAB_SIZE=129280
LOGITS_BYTES=$((VOCAB_SIZE * 4))
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${Q8_WARP_INTERLEAVED_PRODUCTION_AB_DIR:-$repo_dir/sm75-q8-warp-interleaved-production-ab-$stamp}
layouts=(mixed15 all43)
models=("$MIXED_MODEL" "$ALL43_MODEL")

for model in "${models[@]}"; do
    [[ $model == /* && -f $model ]] ||
        die "model not found at absolute path: $model"
done
[[ $MIXED_MODEL != "$ALL43_MODEL" ]] || die "model paths must differ"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "this A/B requires the fixed promessi_sposi prompt"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "require GPU_DEVICES=0,3,1,2 GPU_VRAM=auto STAGE_SPLIT=22"
for item in "TG_TOKENS:$TG_TOKENS" "EXACT_TOKENS:$EXACT_TOKENS" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" "PIPELINE_MB:$PIPELINE_MB" \
            "INTERLEAVED_CACHE_MB:$INTERLEAVED_CACHE_MB"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
done
(( TG_TOKENS == 256 && EXACT_TOKENS == 16 && PREFILL_CHUNK == 2048 &&
   PIPELINE_MB == 512 && INTERLEAVED_CACHE_MB >= 768 )) ||
    die "require TG=256 EXACT=16 PREFILL_CHUNK=2048 PIPELINE_MB=512 cache>=768 MiB"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] ||
        die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset"

for tool in awk basename cmp date dirname env find git grep make mkdir mv \
            nproc nvidia-smi python3 sha256sum sort stat tail tar tee tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
IFS=, read -r -a required_power <<<"$REQUIRED_POWER_LIMITS_W"
(( ${#gpu_ids[@]} == 4 && ${#required_power[@]} == 4 )) ||
    die "GPU and power lists must each contain four entries"
declare -A seen_gpu=()
for gpu in "${gpu_ids[@]}"; do
    [[ $gpu =~ ^[0-9]+$ && -z ${seen_gpu[$gpu]+x} ]] ||
        die "invalid or duplicate GPU ID: $gpu"
    seen_gpu[$gpu]=1
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "GPU $gpu is ${cap:-unknown}, not SM75"
    limit=$(nvidia-smi -i "$gpu" --query-gpu=power.limit \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    awk -v actual="$limit" -v expected="${required_power[$gpu]}" \
        'BEGIN {exit !((actual+0)==(expected+0))}' ||
        die "GPU $gpu power is ${limit:-unknown} W, expected ${required_power[$gpu]} W"
done

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{runs,exact,summary,provenance}
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
    "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT"
    DS4_CUDA_PREFILL_PIPELINE=1
    "DS4_CUDA_PREFILL_PIPELINE_MB=$PIPELINE_MB"
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK"
    DS4_BENCH_UNTIMED_WARMUP_TOKENS=512
    DS4_CUDA_NO_TP_PREFILL_ATTN_ROWS_PAIRS=0
    DS4_CUDA_TP_PREFILL_INDEXER_ROWS_PAIRS=0,1
    DS4_CUDA_TP_PREFILL_INDEXER_ROWS_AUDIT=1
    DS4_CUDA_NO_MOE_Q32_DECODE_GRAPH=1
    DS4_CUDA_NO_MOE_Q32_DECODE_SPLIT=1
    DS4_CUDA_NO_MOE_Q32_DECODE_FUSED_LOWREG=1
    DS4_CUDA_MOE_Q4_32_DECODE_MAPPING=tile32-mma
    DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING=tile32
    DS4_BENCH_ROUTED_QUANT_AUDIT=1
)

phase=build
targets=(ds4-bench tests/cuda_long_context_smoke tests/cuda_sm75_q8_warp_interleaved)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q "${targets[@]}" CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi
"${clean[@]}" ./tests/cuda_long_context_smoke \
    >"$OUTPUT_DIR/smoke.log" 2>&1 || {
        tail -n 220 "$OUTPUT_DIR/smoke.log" >&2
        die "CUDA regression failed"
    }
grep -Fq 'SM75 warp-interleaved Q8 engine T32 production default exact' "$OUTPUT_DIR/smoke.log" ||
    die "interleaved engine regression marker missing"
"${clean[@]}" ./tests/cuda_sm75_q8_warp_interleaved --correctness-only \
    >"$OUTPUT_DIR/interleaved-correctness.log" 2>&1 || {
        tail -n 220 "$OUTPUT_DIR/interleaved-correctness.log" >&2
        die "interleaved Q8 regression failed"
    }
grep -Fq 'harness_status=ok' "$OUTPUT_DIR/interleaved-correctness.log" ||
    die "interleaved Q8 success marker missing"

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'mixed_model=%s\nmixed_model_bytes=%s\n' \
        "$MIXED_MODEL" "$(stat -c %s "$MIXED_MODEL")"
    printf 'all43_model=%s\nall43_model_bytes=%s\n' \
        "$ALL43_MODEL" "$(stat -c %s "$ALL43_MODEL")"
    printf 'prompt=%s\nprompt_sha256=%s\n' "$PROMPT" \
        "$(sha256sum "$PROMPT" | awk '{print $1}')"
    printf 'gpu_devices=%s\nrequired_power_limits_w=%s\nstage_split=22/21\n' \
        "$GPU_DEVICES" "$REQUIRED_POWER_LIMITS_W"
    printf 'candidate_scope=single-token-q8-t32-1024x32768\n'
    printf 'candidate_cache_mib_per_device=%s\n' "$INTERLEAVED_CACHE_MB"
    printf 'throughput_repeats=1\nfrontiers=512,4096,32768\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,serial,power.limit,memory.total,compute_cap \
        --format=csv
    printf '\n[topology]\n'; nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

capture_gpu_health() {
    local output=$1 partial="$1.partial.$$" gpu
    : >"$partial"
    for gpu in "${gpu_ids[@]}"; do
        nvidia-smi -i "$gpu" --query-gpu=index,pci.bus_id,uuid,serial,power.limit \
            --format=csv,noheader,nounits >>"$partial" 2>&1 || {
                mv -- "$partial" "$output"; return 1;
            }
    done
    mv -- "$partial" "$output"
}

validate_gpu_health() {
    local base=$1
    [[ -s $base.pre-gpu.csv && -s $base.post-gpu.csv ]] &&
        ! grep -Eq 'ERR!|Unknown Error|GPU is lost' \
            "$base.pre-gpu.csv" "$base.post-gpu.csv" &&
        cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$base.pre-gpu.csv" &&
        cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$base.post-gpu.csv"
}

validate_csv() {
    local csv=$1 tokens=$2
    awk -F, -v tg="$tokens" '
        NR==1 {ok=($1=="ctx_tokens" && $3=="prefill_tps" &&
                  $4=="gen_tokens" && $8=="gen_steady_tps"); next}
        NR>1 {rows++; if (($1==512 || $1==4096 || $1==32768) &&
                         $4==tg && ($3+0)>0 && ($8+0)>0) seen[$1]++}
        END {exit !(ok && rows==3 && seen[512]==1 && seen[4096]==1 &&
                    seen[32768]==1)}
    ' "$csv"
}

validate_dispatch() {
    local variant=$1 log=$2
    if [[ $variant == control ]]; then
        ! grep -Fq 'SM75 warp-interleaved Q8 summary' "$log"
    else
        awk '
            /SM75 warp-interleaved Q8 summary/ {
                seen++; calls=fills=fallbacks=-1
                for (i=1; i<=NF; i++) {
                    split($i,a,"=")
                    if (a[1]=="calls") calls=a[2]+0
                    if (a[1]=="fills") fills=a[2]+0
                    if (a[1]=="fallbacks") fallbacks=a[2]+0
                }
                if (calls<fills || fills<43 || fills>86 || fallbacks!=0) bad=1
            }
            END {exit !(seen==1 && !bad)}
        ' "$log"
        [[ $(grep -Ec 'SM75 warp-interleaved Q8 cache fill .* in=1024 out=32768$' \
              "$log") -ge 43 ]]
    fi
}

validate_common_log() {
    local layout=$1 log=$2 marker
    for marker in 'CUDA EP forced pipeline split 22/21' \
                  'dense-placement=stage-aware-fixed-22-21' \
                  'SM75 routed Q32 layout enabled' \
                  'CUDA decode TP enabled'; do
        grep -Fq "$marker" "$log" || return 1
    done
    grep -Fq 'prefill attention row split pair-scoped disable: logical-pairs=0' \
        "$log" || return 1
    grep -Fq 'prefill indexer row split pair policy: enabled-pairs=0,1 disabled-pairs=none' \
        "$log" || return 1
    ! grep -Fq 'required but unavailable' "$log" || return 1
    if [[ $layout == mixed15 ]]; then
        grep -Fq 'SM75 Q4-32 decode gate/up mapping=tile32-mma (production default)' \
            "$log" || return 1
    fi
}

validate_logits() {
    local dir=$1 expected=$((3 * EXACT_TOKENS))
    [[ $(find "$dir" -maxdepth 1 -type f -name '*.f32' | wc -l) == "$expected" ]] ||
        return 1
    python3 - "$dir" "$LOGITS_BYTES" "$EXACT_TOKENS" <<'PY'
import array, math, pathlib, sys
root = pathlib.Path(sys.argv[1]); nbytes = int(sys.argv[2]); ntok = int(sys.argv[3])
for context in (512, 4096, 32768):
    for token in range(1, ntok + 1):
        path = root / f"frontier_{context:06d}.decode_{token:06d}.logits.f32"
        if not path.is_file() or path.stat().st_size != nbytes:
            raise SystemExit(f"missing or invalid logits: {path}")
        values = array.array("f")
        with path.open("rb") as handle: values.fromfile(handle, nbytes // 4)
        if not all(map(math.isfinite, values)) or not any(v != 0 for v in values):
            raise SystemExit(f"invalid logits: {path}")
PY
}

run_engine() {
    local model=$1 variant=$2 tokens=$3 base=$4 logits=${5:-} rc=0
    local -a candidate=() cmd
    if [[ $variant == control ]]; then
        candidate=(DS4_CUDA_Q8_WARP_INTERLEAVED_T32_DECODE=0)
    else
        candidate=(
            "DS4_CUDA_Q8_WARP_INTERLEAVED_CACHE_MB=$INTERLEAVED_CACHE_MB"
            DS4_CUDA_Q8_WARP_INTERLEAVED_AUDIT=1
        )
    fi
    capture_gpu_health "$base.pre-gpu.csv" || return 1
    cmd=("${production_env[@]}" "${candidate[@]}" ./ds4-bench
        --cuda --cuda-tensor-parallel --gpu-devices "$GPU_DEVICES"
        --gpu-vram "$GPU_VRAM" --model "$model" --prompt-file "$PROMPT"
        --ctx-start 512 --ctx-max 32768 --ctx-alloc "$CTX_ALLOC"
        --step-mul 8 --prefill-chunk "$PREFILL_CHUNK" --gen-tokens "$tokens")
    [[ -z $logits ]] || cmd+=(--dump-decode-logits-dir "$logits")
    cmd+=(--csv "$base.csv")
    "${cmd[@]}" >"$base.log" 2>&1 || rc=$?
    capture_gpu_health "$base.post-gpu.csv" || return 1
    return "$rc"
}

validate_run() {
    local layout=$1 variant=$2 tokens=$3 base=$4 logits=${5:-}
    validate_gpu_health "$base" && validate_csv "$base.csv" "$tokens" &&
        validate_common_log "$layout" "$base.log" &&
        validate_dispatch "$variant" "$base.log" || return 1
    [[ -z $logits ]] || validate_logits "$logits"
}

phase=production-ab
capture_gpu_health "$OUTPUT_DIR/initial-gpu.csv" ||
    die "could not capture initial GPU state"
printf 'layout\tvariant\tcsv\tlog\n' >"$OUTPUT_DIR/runs.tsv"
for i in 0 1; do
    layout=${layouts[$i]}; model=${models[$i]}
    for variant in control interleaved; do
        base="$OUTPUT_DIR/runs/$layout-$variant"
        printf 'Warp-interleaved Q8 production A/B model=%s variant=%s...\n' \
            "$layout" "$variant"
        run_engine "$model" "$variant" "$TG_TOKENS" "$base" || {
            tail -n 240 "$base.log" >&2 || true
            die "$layout $variant throughput run failed"
        }
        validate_run "$layout" "$variant" "$TG_TOKENS" "$base" ||
            die "$layout $variant throughput validation failed"
        printf '%s\t%s\t%s\t%s\n' "$layout" "$variant" "$base.csv" \
            "$base.log" >>"$OUTPUT_DIR/runs.tsv"
    done
done

phase=exact-logits
for i in 0 1; do
    layout=${layouts[$i]}; model=${models[$i]}
    for variant in control interleaved; do
        base="$OUTPUT_DIR/exact/$layout-$variant"
        logits="$base-logits"; mkdir -p "$logits"
        printf 'Exact warp-interleaved Q8 logits model=%s variant=%s...\n' \
            "$layout" "$variant"
        run_engine "$model" "$variant" "$EXACT_TOKENS" "$base" "$logits" || {
            tail -n 240 "$base.log" >&2 || true
            die "$layout $variant exact run failed"
        }
        validate_run "$layout" "$variant" "$EXACT_TOKENS" "$base" "$logits" ||
            die "$layout $variant exact validation failed"
    done
    for context in 512 4096 32768; do
        for ((token=1; token<=EXACT_TOKENS; token++)); do
            printf -v file 'frontier_%06d.decode_%06d.logits.f32' "$context" "$token"
            cmp -s "$OUTPUT_DIR/exact/$layout-control-logits/$file" \
                   "$OUTPUT_DIR/exact/$layout-interleaved-logits/$file" ||
                die "$layout interleaved output diverged at $file"
        done
    done
done

phase=summarize
{
    printf '# SM75 four-GPU warp-interleaved Q8 decode A/B\n\n'
    printf '| Model | Context | Control tok/s | Interleaved tok/s | Speedup |\n'
    printf '| --- | ---: | ---: | ---: | ---: |\n'
    for layout in "${layouts[@]}"; do
        for context in 512 4096 32768; do
            control=$(awk -F, -v c="$context" '$1==c {print $8}' \
                "$OUTPUT_DIR/runs/$layout-control.csv")
            candidate=$(awk -F, -v c="$context" '$1==c {print $8}' \
                "$OUTPUT_DIR/runs/$layout-interleaved.csv")
            speedup=$(awk -v a="$control" -v b="$candidate" \
                'BEGIN {printf "%.6f", b/a}')
            printf '| %s | %s | %.3f | %.3f | %sx |\n' \
                "$layout" "$context" "$control" "$candidate" "$speedup"
        done
    done
    printf '\nAll %s decode logits at all three frontiers were byte-identical for both models.\n' \
        "$EXACT_TOKENS"
} | tee "$OUTPUT_DIR/summary/summary.md"

printf 'bit_exact=true\nmodels=mixed15,all43\nfrontiers=512,4096,32768\n' \
    >"$OUTPUT_DIR/exact/verification.txt"
phase=complete
printf 'SM75 warp-interleaved Q8 production A/B complete: %s\n' "$OUTPUT_DIR"
