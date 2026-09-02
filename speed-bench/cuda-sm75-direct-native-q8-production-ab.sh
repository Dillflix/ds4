#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run a one-shot four-GPU production A/B for direct native-Q8 activation
production. Both arms use the same SM75 native Q4/Q3A4 consumers. The
canonical arm writes row-major Q8_K and launches an in-place native pack;
the direct arm writes the native Q8_K record in the quantizer and omits that
pack. Both the routed-MoE input boundary (Q4-32 or Q3A4 gate/up) and the
intermediate boundary (Q4-32 down) are required to dispatch exclusively.

Optional environment:
  MIXED_MODEL=...   default: gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf
  ALL43_MODEL=...   default: gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf
  PROMPT=...        default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  REQUIRED_POWER_LIMITS_W=250,260,250,250
  REPEATS=3
  TG_TOKENS=256
  EXACT_TOKENS=16
  WARMUP_TOKENS=512
  PREFILL_CHUNK=2048
  PIPELINE_MB=512
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  DIRECT_NATIVE_Q8_PRODUCTION_AB_DIR=...

The fixed frontiers are 512, 4096, and 32768. Pair-0 prefill attention row
splitting remains disabled while pair-0 and pair-1 prefill indexer splitting
remain enabled. Failed GPU runs are archived and never resumed.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

MIXED_MODEL=${MIXED_MODEL:-$repo_dir/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf}
ALL43_MODEL=${ALL43_MODEL:-$repo_dir/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf}
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
REQUIRED_POWER_LIMITS_W=${REQUIRED_POWER_LIMITS_W:-250,260,250,250}
REPEATS=${REPEATS:-3}
TG_TOKENS=${TG_TOKENS:-256}
EXACT_TOKENS=${EXACT_TOKENS:-16}
WARMUP_TOKENS=${WARMUP_TOKENS:-512}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
PIPELINE_MB=${PIPELINE_MB:-512}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
CTX_ALLOC=33025
VOCAB_SIZE=129280
LOGITS_BYTES=$((VOCAB_SIZE * 4))
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${DIRECT_NATIVE_Q8_PRODUCTION_AB_DIR:-$repo_dir/sm75-direct-native-q8-production-ab-$stamp}
layouts=(mixed15 all43)
models=("$MIXED_MODEL" "$ALL43_MODEL")
contexts=(512 4096 32768)

for model in "${models[@]}"; do
    [[ $model == /* && -f $model ]] ||
        die "model not found at absolute path: $model"
done
[[ $MIXED_MODEL != "$ALL43_MODEL" ]] || die "the two model paths must differ"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "this production A/B requires the fixed promessi_sposi prompt"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "require GPU_DEVICES=0,3,1,2, GPU_VRAM=auto, STAGE_SPLIT=22"
for item in "REPEATS:$REPEATS" "TG_TOKENS:$TG_TOKENS" \
            "EXACT_TOKENS:$EXACT_TOKENS" "WARMUP_TOKENS:$WARMUP_TOKENS" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" "PIPELINE_MB:$PIPELINE_MB"; do
    name=${item%%:*}
    value=${item#*:}
    [[ $value =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
done
(( REPEATS >= 3 && TG_TOKENS == 256 && EXACT_TOKENS == 16 &&
   WARMUP_TOKENS == 512 && PREFILL_CHUNK == 2048 && PIPELINE_MB == 512 )) ||
    die "require repeats>=3, tg_tokens=256, exact_tokens=16, warmup=512, prefill_chunk=2048, pipeline_mb=512"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"

for tool in awk basename cmp date dirname env find git grep make mkdir mv \
            nproc nvidia-smi python3 sha256sum sort stat tail tar tee tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
IFS=, read -r -a required_power <<<"$REQUIRED_POWER_LIMITS_W"
(( ${#gpu_ids[@]} == 4 && ${#required_power[@]} == 4 )) ||
    die "GPU_DEVICES and REQUIRED_POWER_LIMITS_W must each contain four values"
declare -A seen_gpu=()
for gpu in "${gpu_ids[@]}"; do
    [[ $gpu =~ ^[0-9]+$ && -z ${seen_gpu[$gpu]+x} ]] ||
        die "invalid or duplicate GPU ID: $gpu"
    seen_gpu[$gpu]=1
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ $cap == 7.5 ]] ||
        die "GPU $gpu is compute capability ${cap:-unknown}, not SM75"
    [[ ${required_power[$gpu]} =~ ^[0-9]+([.][0-9]+)?$ ]] ||
        die "invalid required power limit for physical GPU $gpu"
    limit=$(nvidia-smi -i "$gpu" --query-gpu=power.limit \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    awk -v actual="$limit" -v expected="${required_power[$gpu]}" \
        'BEGIN {exit !((actual+0)==(expected+0))}' ||
        die "physical GPU $gpu power limit is ${limit:-unknown} W, expected ${required_power[$gpu]} W"
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
    "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$WARMUP_TOKENS"
    DS4_CUDA_NO_TP_PREFILL_ATTN_ROWS_PAIRS=0
    DS4_CUDA_TP_PREFILL_INDEXER_ROWS_PAIRS=0,1
    DS4_CUDA_TP_PREFILL_INDEXER_ROWS_AUDIT=1
    DS4_CUDA_NO_MOE_Q32_DECODE_GRAPH=1
    DS4_CUDA_NO_MOE_Q32_DECODE_SPLIT=1
    DS4_CUDA_NO_MOE_Q32_DECODE_FUSED_LOWREG=1
    DS4_CUDA_MOE_Q4_32_DECODE_MAPPING=tile32-mma
    DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING=tile32
    DS4_CUDA_MOE_DIRECT_NATIVE_Q8_AUDIT=1
    DS4_BENCH_ROUTED_QUANT_AUDIT=1
)

phase=build
targets=(ds4-bench tests/cuda_long_context_smoke)
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
        die "byte-exact CUDA regression failed"
    }
for marker in \
    'SM75 direct native Q8 opt-in selector exact' \
    'SM75 Q4-32 direct native Q8 producer + Q4-32 down consumer exact/reuse' \
    'SM75 Q3A4 direct native Q8 producer + Q4-32 down consumer exact/reuse' \
    'SM75 Q4-32 tile32-mma gate/up + tile32 down production defaults' \
    'SM75 Q3A4 tile32-dp4a-k4-prefetch2 production default' \
    'cuda long-context regression: OK'; do
    grep -Fq "$marker" "$OUTPUT_DIR/smoke.log" ||
        die "CUDA regression marker missing: $marker"
done

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'mixed_model=%s\nmixed_model_bytes=%s\n' \
        "$MIXED_MODEL" "$(stat -c %s "$MIXED_MODEL")"
    printf 'all43_model=%s\nall43_model_bytes=%s\n' \
        "$ALL43_MODEL" "$(stat -c %s "$ALL43_MODEL")"
    printf 'model_hashing=disabled\nprompt=%s\nprompt_sha256=%s\n' \
        "$PROMPT" "$(sha256sum "$PROMPT" | awk '{print $1}')"
    printf 'gpu_devices=%s\ngpu_vram=%s\nrequired_power_limits_w=%s\n' \
        "$GPU_DEVICES" "$GPU_VRAM" "$REQUIRED_POWER_LIMITS_W"
    printf 'stage_split=22/21\ncontexts=512,4096,32768\nrepeats=%s\n' "$REPEATS"
    printf 'tg_tokens=%s\nexact_tokens=%s\nprefill_chunk=%s\npipeline_mb=%s\n' \
        "$TG_TOKENS" "$EXACT_TOKENS" "$PREFILL_CHUNK" "$PIPELINE_MB"
    printf 'pair0_prefill_attention_rows=disabled\npair1_prefill_attention_rows=enabled\n'
    printf 'pair0_prefill_indexer_rows=enabled\npair1_prefill_indexer_rows=enabled\n'
    printf 'comparison=canonical-q8k-plus-pack-vs-direct-native-q8k\n'
    printf '\n[gpu inventory]\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,serial,power.limit,memory.total,compute_cap \
        --format=csv
    printf '\n[topology]\n'
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

capture_gpu_health() {
    local output=$1 partial="$1.partial.$$" gpu
    : >"$partial"
    for gpu in "${gpu_ids[@]}"; do
        nvidia-smi -i "$gpu" \
            --query-gpu=index,pci.bus_id,uuid,serial,power.limit \
            --format=csv,noheader,nounits >>"$partial" 2>&1 || {
                mv -- "$partial" "$output"
                return 1
            }
    done
    mv -- "$partial" "$output"
}

validate_gpu_health_pair() {
    local base=$1
    [[ -s $base.pre-gpu.csv && -s $base.post-gpu.csv ]] || return 1
    ! grep -Eq 'ERR!|Unknown Error|GPU is lost' \
        "$base.pre-gpu.csv" "$base.post-gpu.csv" || return 1
    cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$base.pre-gpu.csv" &&
        cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$base.post-gpu.csv"
}

validate_layout() {
    local layout=$1 log=$2 expected_count expected_layers
    if [[ $layout == mixed15 ]]; then
        expected_count=15
        expected_layers=6,8,10,12,14,16,18,20,30,32,34,36,38,40,42
    else
        expected_count=43
        expected_layers=$(printf '%s,' {0..42})
        expected_layers=${expected_layers%,}
    fi
    awk -v expected_count="$expected_count" -v expected_layers="$expected_layers" '
        /routed-quant-audit/ {
            seen++; layer=gate=up=down=""
            for (i=1; i<=NF; i++) {
                split($i,a,"=")
                if (a[1]=="layer") layer=a[2]
                if (a[1]=="gate") gate=a[2]
                if (a[1]=="up") up=a[2]
                if (a[1]=="down") down=a[2]
            }
            if (layer !~ /^[0-9]+$/ || layer<0 || layer>42 || layer_seen[layer]++) bad=1
            if (gate=="sm75_q3a4") {
                if (up!="sm75_q3a4" || down!="sm75_q4_32") bad=1
                q3++; layers=layers (layers ? "," : "") layer
            } else if (gate!="sm75_q4_32" || up!="sm75_q4_32" || down!="sm75_q4_32") bad=1
        }
        END {
            for (i=0; i<43; i++) if (layer_seen[i]!=1) bad=1
            exit !(seen==43 && q3==expected_count && layers==expected_layers && !bad)
        }
    ' "$log"
}

validate_common_log() {
    local layout=$1 log=$2 marker route
    for marker in 'CUDA EP forced pipeline split 22/21' \
                  'dense-placement=stage-aware-fixed-22-21' \
                  'materialized 344/344 candidates' \
                  'SM75 routed Q32 layout enabled' \
                  'CUDA decode TP enabled' \
                  'SM75 native F16 indexer cache and streaming WMMA64 enabled'; do
        grep -Fq "$marker" "$log" || return 1
    done
    for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
        grep -Fq "$route" "$log" || return 1
    done
    grep -Fq 'prefill attention row split pair-scoped disable: logical-pairs=0' \
        "$log" || return 1
    grep -Fq 'prefill indexer row split pair policy: enabled-pairs=0,1 disabled-pairs=none' \
        "$log" || return 1
    grep -Fq 'CUDA TP cache mirror policy: attention-pair-mask=0x2 index-pair-mask=0x3' \
        "$log" || return 1
    ! grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
        return 1
    grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
        return 1
    grep -Eq 'prefill indexer row audit event=complete .*home_tier=0 .*selected_mode=gather-home' \
        "$log" || return 1
    grep -Eq 'prefill indexer row audit event=complete .*home_tier=1 .*selected_mode=partner-local' \
        "$log" || return 1
    grep -Eq 'CUDA T32 f16-output fused summary: local=0 partner=[1-9][0-9]*' \
        "$log" || return 1
    grep -Fxq 'ds4: SM75 Q3A4 decode gate/up mapping=tile32-dp4a-k4-prefetch2 (production default)' \
        "$log" || return 1
    grep -Fxq 'ds4: SM75 Q4-32 down decode mapping=tile32-int4 (production default)' \
        "$log" || return 1
    if [[ $layout == mixed15 ]]; then
        grep -Fxq 'ds4: SM75 Q4-32 decode gate/up mapping=tile32-mma (production default)' \
            "$log" || return 1
    else
        ! grep -Fq 'SM75 Q4-32 decode gate/up mapping=' "$log" || return 1
    fi
    ! grep -Fq 'required but unavailable' "$log" || return 1
    ! grep -Fq 'SM75 Q32 owned decode CUDA Graph enabled' "$log" || return 1
    validate_layout "$layout" "$log"
}

validate_direct_audit() {
    local variant=$1 log=$2
    awk -v variant="$variant" '
        /SM75 direct native Q8 audit boundary=/ {
            delete v
            for (i=1; i<=NF; i++) {
                split($i,a,"=")
                if (a[1] ~ /^(boundary|enabled|canonical-prefill-calls|direct-prefill-calls|canonical-prefill-blocks|direct-prefill-blocks|canonical-decode-calls|direct-decode-calls|canonical-decode-blocks|direct-decode-blocks)$/)
                    v[a[1]]=a[2]
            }
            b=v["boundary"]
            if (b!="input" && b!="mid") bad=1
            seen[b]++
            if (variant=="canonical") {
                if (v["enabled"]!=0 || v["canonical-prefill-calls"]+0<=0 ||
                    v["canonical-prefill-blocks"]+0<=0 || v["canonical-decode-calls"]+0<=0 ||
                    v["canonical-decode-blocks"]+0<=0 || v["direct-prefill-calls"]+0!=0 ||
                    v["direct-prefill-blocks"]+0!=0 || v["direct-decode-calls"]+0!=0 ||
                    v["direct-decode-blocks"]+0!=0) bad=1
            } else {
                if (v["enabled"]!=1 || v["direct-prefill-calls"]+0<=0 ||
                    v["direct-prefill-blocks"]+0<=0 || v["direct-decode-calls"]+0<=0 ||
                    v["direct-decode-blocks"]+0<=0 || v["canonical-prefill-calls"]+0!=0 ||
                    v["canonical-prefill-blocks"]+0!=0 || v["canonical-decode-calls"]+0!=0 ||
                    v["canonical-decode-blocks"]+0!=0) bad=1
            }
        }
        END {exit !(seen["input"]==1 && seen["mid"]==1 && !bad)}
    ' "$log" || return 1
    if [[ $variant == canonical ]]; then
        ! grep -Fq 'SM75 direct native Q8 producer selected' "$log"
    else
        [[ $(grep -Fc 'SM75 direct native Q8 producer selected' "$log") == 1 ]]
    fi
}

validate_q8_state_files() {
    local base=$1 plan="$1.q8-plan.csv" bindings="$1.q8-bindings.csv"
    [[ -s $plan && -s $bindings ]] || return 1
    [[ $(wc -l <"$plan") == 345 && $(wc -l <"$bindings") == 345 ]] ||
        return 1
    awk -F, '
        NR==1 {header=($1=="sequence" && $2=="label" && $3=="consumer_device" && $7=="resident_device" && $13=="status"); next}
        {rows++; if (($7+0)<0 || ($13!="home" && $13!="partner")) bad=1}
        END {exit !(header && rows==344 && !bad)}
    ' "$plan" || return 1
    awk -F, '
        NR==1 {header=($1=="consumer_device" && $2=="resident_device" && $3=="partner_offload" && $12=="label" && $15=="live"); next}
        {rows++; if (($13+0)<=0 || ($14+0)<=0 || $15!=1) bad=1}
        END {exit !(header && rows==344 && !bad)}
    ' "$bindings"
}

canonical_q8_bindings() {
    awk -F, 'BEGIN {OFS=","} NR>1 {print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12}' "$1" | sort
}

validate_q8_plan_equal() {
    local left=$1 right=$2 marker left_line right_line
    for marker in 'CUDA q8 fp16 benefit plan candidates=' \
                  'CUDA q8 fp16 stage-aware 22/21 planner selected ' \
                  'CUDA q8 fp16 benefit plan materialized '; do
        [[ $(grep -Fc "$marker" "$left.log") == 1 &&
           $(grep -Fc "$marker" "$right.log") == 1 ]] || return 1
        left_line=$(grep -F "$marker" "$left.log") || return 1
        right_line=$(grep -F "$marker" "$right.log") || return 1
        [[ $left_line == "$right_line" ]] || return 1
    done
    validate_q8_state_files "$left" && validate_q8_state_files "$right" ||
        return 1
    cmp -s "$left.q8-plan.csv" "$right.q8-plan.csv" || return 1
    cmp -s <(canonical_q8_bindings "$left.q8-bindings.csv") \
           <(canonical_q8_bindings "$right.q8-bindings.csv")
}

validate_csv() {
    local csv=$1 tokens=$2
    awk -F, -v tg="$tokens" '
        NR==1 {header=($1=="ctx_tokens" && $3=="prefill_tps" &&
                       $4=="gen_tokens" && $8=="gen_steady_tps"); next}
        NR==2 {rows++; a=($1==512 && $4==tg && ($3+0)>0 && ($8+0)>0); next}
        NR==3 {rows++; b=($1==4096 && $4==tg && ($3+0)>0 && ($8+0)>0); next}
        NR==4 {rows++; c=($1==32768 && $4==tg && ($3+0)>0 && ($8+0)>0); next}
        NR>4 {rows++}
        END {exit !(header && rows==3 && a && b && c)}
    ' "$csv"
}

validate_logits() {
    local dir=$1 expected=$((3 * EXACT_TOKENS))
    [[ $(find "$dir" -maxdepth 1 -type f -name '*.f32' | wc -l) == "$expected" ]] ||
        return 1
    python3 - "$dir" "$LOGITS_BYTES" "$EXACT_TOKENS" <<'PY'
import array
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected_bytes = int(sys.argv[2])
tokens = int(sys.argv[3])
for context in (512, 4096, 32768):
    for token in range(1, tokens + 1):
        path = root / f"frontier_{context:06d}.decode_{token:06d}.logits.f32"
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise SystemExit(f"missing or invalid logits: {path}")
        values = array.array("f")
        with path.open("rb") as handle:
            values.fromfile(handle, expected_bytes // 4)
        if not all(math.isfinite(value) for value in values):
            raise SystemExit(f"non-finite logits: {path}")
        if not any(value != 0.0 for value in values):
            raise SystemExit(f"all-zero logits: {path}")
PY
}

variant_env() {
    if [[ $1 == canonical ]]; then
        printf 'DS4_CUDA_NO_MOE_DIRECT_NATIVE_Q8=1\n'
    else
        printf 'DS4_CUDA_MOE_DIRECT_NATIVE_Q8=1\n'
    fi
}

run_engine() {
    local layout=$1 model=$2 variant=$3 tokens=$4 base=$5 logits=${6:-} rc=0
    local -a selector cmd
    mapfile -t selector < <(variant_env "$variant")
    capture_gpu_health "$base.pre-gpu.csv" || return 1
    cmd=("${production_env[@]}" "${selector[@]}"
        "DS4_CUDA_Q8_PLAN_AUDIT_CSV=$base.q8-plan.csv"
        "DS4_CUDA_Q8_BINDING_STATE_CSV=$base.q8-bindings.csv"
        ./ds4-bench --cuda --cuda-tensor-parallel
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM"
        --model "$model" --prompt-file "$PROMPT"
        --ctx-start 512 --ctx-max 32768 --ctx-alloc "$CTX_ALLOC"
        --step-mul 8 --prefill-chunk "$PREFILL_CHUNK"
        --gen-tokens "$tokens")
    [[ -z $logits ]] || cmd+=(--dump-decode-logits-dir "$logits")
    cmd+=(--csv "$base.csv")
    "${cmd[@]}" >"$base.log" 2>&1 || rc=$?
    capture_gpu_health "$base.post-gpu.csv" || return 1
    return "$rc"
}

validate_run() {
    local layout=$1 variant=$2 tokens=$3 base=$4 logits=${5:-}
    validate_gpu_health_pair "$base" &&
        validate_q8_state_files "$base" &&
        validate_csv "$base.csv" "$tokens" &&
        validate_common_log "$layout" "$base.log" &&
        validate_direct_audit "$variant" "$base.log" || return 1
    [[ -z $logits ]] || validate_logits "$logits"
}

phase=production-ab
capture_gpu_health "$OUTPUT_DIR/initial-gpu.csv" ||
    die "could not capture initial four-GPU identity and power limits"
printf 'model_layout\trepeat\tvariant\tcsv\tlog\n' >"$OUTPUT_DIR/runs.tsv"
for model_index in 0 1; do
    layout=${layouts[$model_index]}
    model=${models[$model_index]}
    layout_reference=
    for ((repeat=1; repeat<=REPEATS; repeat++)); do
        if (( repeat % 2 )); then
            variants=(canonical direct)
        else
            variants=(direct canonical)
        fi
        slot=0
        for variant in "${variants[@]}"; do
            slot=$((slot + 1))
            base="$OUTPUT_DIR/runs/$layout-r${repeat}-$variant"
            printf 'Direct native-Q8 production A/B model=%s repeat=%d/%d slot=%d/2 variant=%s...\n' \
                "$layout" "$repeat" "$REPEATS" "$slot" "$variant"
            run_engine "$layout" "$model" "$variant" "$TG_TOKENS" "$base" || {
                tail -n 240 "$base.log" >&2 || true
                die "$layout $variant throughput run failed"
            }
            validate_run "$layout" "$variant" "$TG_TOKENS" "$base" ||
                die "$layout $variant throughput run failed production-path validation"
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$layout" "$repeat" "$variant" "$base.csv" "$base.log" \
                >>"$OUTPUT_DIR/runs.tsv"
        done
        canonical="$OUTPUT_DIR/runs/$layout-r${repeat}-canonical"
        direct="$OUTPUT_DIR/runs/$layout-r${repeat}-direct"
        validate_q8_plan_equal "$canonical" "$direct" ||
            die "$layout repeat $repeat changed the dense-Q8 plan between arms"
        if [[ -z $layout_reference ]]; then
            layout_reference=$canonical
        else
            validate_q8_plan_equal "$layout_reference" "$canonical" ||
                die "$layout repeat $repeat changed the production dense-Q8 plan"
        fi
    done
done

phase=summarize
python3 speed-bench/summarize-sm75-direct-native-q8-production-ab.py \
    "$OUTPUT_DIR/runs.tsv" "$OUTPUT_DIR/summary" |
    tee "$OUTPUT_DIR/summary-stdout.txt"

phase=exact-logits
printf 'model_layout\tvariant\tcsv\tlog\tlogits\n' >"$OUTPUT_DIR/exact/runs.tsv"
for model_index in 0 1; do
    layout=${layouts[$model_index]}
    model=${models[$model_index]}
    for variant in canonical direct; do
        base="$OUTPUT_DIR/exact/$layout-$variant"
        logits="$base-logits"
        mkdir -p "$logits"
        printf 'Exact direct native-Q8 logits model=%s variant=%s (all frontiers)...\n' \
            "$layout" "$variant"
        run_engine "$layout" "$model" "$variant" "$EXACT_TOKENS" "$base" "$logits" || {
            tail -n 240 "$base.log" >&2 || true
            die "$layout $variant exact-logit run failed"
        }
        validate_run "$layout" "$variant" "$EXACT_TOKENS" "$base" "$logits" ||
            die "$layout $variant exact-logit run failed production-path validation"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$layout" "$variant" "$base.csv" "$base.log" "$logits" \
            >>"$OUTPUT_DIR/exact/runs.tsv"
    done
    validate_q8_plan_equal "$OUTPUT_DIR/exact/$layout-canonical" \
        "$OUTPUT_DIR/exact/$layout-direct" ||
        die "$layout exact arms changed the dense-Q8 plan"
    validate_q8_plan_equal "$OUTPUT_DIR/runs/$layout-r1-canonical" \
        "$OUTPUT_DIR/exact/$layout-canonical" ||
        die "$layout exact phase changed the production dense-Q8 plan"
    for context in "${contexts[@]}"; do
        for ((token=1; token<=EXACT_TOKENS; token++)); do
            printf -v file 'frontier_%06d.decode_%06d.logits.f32' \
                "$context" "$token"
            cmp -s "$OUTPUT_DIR/exact/$layout-canonical-logits/$file" \
                   "$OUTPUT_DIR/exact/$layout-direct-logits/$file" ||
                die "$layout direct native-Q8 diverged at $file"
        done
    done
done

printf 'bit_exact=true\nmodels=mixed15,all43\nfrontiers=512,4096,32768\n' \
    >"$OUTPUT_DIR/exact/verification.txt"
printf 'decode_tokens_per_frontier=%s\ninput_boundary_exclusive=true\nmid_boundary_exclusive=true\n' \
    "$EXACT_TOKENS" >>"$OUTPUT_DIR/exact/verification.txt"
{
    printf '\n## Exact-output verification\n\n'
    printf 'For both mixed15 and all43, all %s decode logits at PP512, PP4096, and PP32768 were byte-identical.\n' \
        "$EXACT_TOKENS"
    printf 'Both the routed-MoE input and Q4-down intermediate boundaries dispatched exclusively through the selected producer.\n'
} >>"$OUTPUT_DIR/summary/summary.md"

phase=complete
printf 'SM75 dual-model direct native-Q8 production A/B complete: %s\n' "$OUTPUT_DIR"
