#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Qualify one canonical/tagged-native model pair across production prefill and
steady decode while measuring exact output and peak four-GPU VRAM.

Required environment:
  MODEL_LAYOUT=mixed15|all43
  CANONICAL_MODEL=/absolute/path/to/canonical.gguf
  NATIVE_MODEL=/absolute/path/to/sm75-native-q8.gguf

Optional environment:
  PROMPT=...                         default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  REQUIRED_POWER_LIMITS_W=250,260,250,250
  TG_TOKENS=256
  EXACT_TOKENS=16
  TELEMETRY_INTERVAL_MS=200
  CASE_TIMEOUT_SECONDS=1800
  MIN_NATIVE_VRAM_SAVING_MIB=3500
  MIN_THROUGHPUT_RATIO=0.95
  NATIVE_SMOKE_ONLY=0              1: run only native PP512/TG16
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  NATIVE_Q8_GGUF_PRODUCTION_AB_DIR=...

This is intentionally one-shot. It never creates, overwrites, resumes, or
deletes either model. Use deepseek4-quantize --repack-sm75-native-q8 before
running it.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

MODEL_LAYOUT=${MODEL_LAYOUT:-}
CANONICAL_MODEL=${CANONICAL_MODEL:-}
NATIVE_MODEL=${NATIVE_MODEL:-}
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
REQUIRED_POWER_LIMITS_W=${REQUIRED_POWER_LIMITS_W:-250,260,250,250}
TG_TOKENS=${TG_TOKENS:-256}
EXACT_TOKENS=${EXACT_TOKENS:-16}
TELEMETRY_INTERVAL_MS=${TELEMETRY_INTERVAL_MS:-200}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-1800}
MIN_NATIVE_VRAM_SAVING_MIB=${MIN_NATIVE_VRAM_SAVING_MIB:-3500}
MIN_THROUGHPUT_RATIO=${MIN_THROUGHPUT_RATIO:-0.95}
NATIVE_SMOKE_ONLY=${NATIVE_SMOKE_ONLY:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
PREFILL_CHUNK=2048
PIPELINE_MB=512
CTX_ALLOC=33025
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_kind=production-ab
[[ $NATIVE_SMOKE_ONLY == 1 ]] && run_kind=native-smoke
OUTPUT_DIR=${NATIVE_Q8_GGUF_PRODUCTION_AB_DIR:-$repo_dir/sm75-native-q8-gguf-$MODEL_LAYOUT-$run_kind-$stamp}

[[ $MODEL_LAYOUT == mixed15 || $MODEL_LAYOUT == all43 ]] ||
    die "MODEL_LAYOUT must be mixed15 or all43"
for model in "$CANONICAL_MODEL" "$NATIVE_MODEL"; do
    [[ $model == /* && -f $model ]] ||
        die "model not found at absolute path: ${model:-<unset>}"
done
[[ $CANONICAL_MODEL != "$NATIVE_MODEL" ]] ||
    die "CANONICAL_MODEL and NATIVE_MODEL must differ"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "this production A/B requires the fixed promessi_sposi prompt"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "require GPU_DEVICES=0,3,1,2, GPU_VRAM=auto, STAGE_SPLIT=22"
for value in "$TG_TOKENS" "$EXACT_TOKENS" "$TELEMETRY_INTERVAL_MS" \
             "$CASE_TIMEOUT_SECONDS" "$MIN_NATIVE_VRAM_SAVING_MIB"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || die "positive integer required, got: $value"
done
awk -v ratio="$MIN_THROUGHPUT_RATIO" \
    'BEGIN {exit !(ratio+0>0 && ratio+0<=1)}' ||
    die "MIN_THROUGHPUT_RATIO must be in (0,1]"
for flag in NATIVE_SMOKE_ONLY SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"
for tool in awk cmp date env find git grep make mkdir mv nproc nvidia-smi \
            python3 sort stat tail tar tee timeout tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
IFS=, read -r -a required_power <<<"$REQUIRED_POWER_LIMITS_W"
(( ${#gpu_ids[@]} == 4 && ${#required_power[@]} == 4 )) ||
    die "GPU and power-limit lists must each contain four entries"
declare -A seen_gpu=()
for slot in 0 1 2 3; do
    gpu=${gpu_ids[$slot]}
    [[ $gpu =~ ^[0-9]+$ && -z ${seen_gpu[$gpu]+x} ]] ||
        die "invalid or duplicate GPU ID: $gpu"
    seen_gpu[$gpu]=1
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "GPU $gpu is compute capability ${cap:-unknown}, not SM75"
done
for gpu in 0 1 2 3; do
    limit=$(nvidia-smi -i "$gpu" --query-gpu=power.limit \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    awk -v actual="$limit" -v expected="${required_power[$gpu]}" \
        'BEGIN {exit !((actual+0)==(expected+0))}' ||
        die "physical GPU $gpu power limit is ${limit:-unknown} W, expected ${required_power[$gpu]} W"
done

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{runs,exact,telemetry,summary,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
telemetry_pid=
stop_telemetry() {
    if [[ -n ${telemetry_pid:-} ]]; then
        kill "$telemetry_pid" 2>/dev/null || true
        wait "$telemetry_pid" 2>/dev/null || true
        telemetry_pid=
    fi
}
finish() {
    local status=$? archive="$OUTPUT_DIR.tar.gz" partial="$OUTPUT_DIR.tar.gz.partial.$$"
    trap - EXIT INT TERM HUP
    stop_telemetry
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
    DS4_CUDA_Q8_WARP_INTERLEAVED_AUDIT=1
    DS4_BENCH_ROUTED_QUANT_AUDIT=1
)

phase=build
if [[ $SKIP_BUILD == 0 ]]; then
    make -j"$(nproc)" ds4-bench CUDA_ARCH=sm_75 2>&1 |
        tee "$OUTPUT_DIR/build-engine.log"
    make -C gguf-tools -j"$(nproc)" deepseek4-quantize 2>&1 |
        tee "$OUTPUT_DIR/build-converter.log"
else
    make -q ds4-bench CUDA_ARCH=sm_75 || die "SKIP_BUILD=1 found a stale ds4-bench"
    make -C gguf-tools -q deepseek4-quantize ||
        die "SKIP_BUILD=1 found a stale dense-Q8 converter"
fi

phase=inventory
gguf-tools/deepseek4-quantize \
    --repack-sm75-native-q8 "$CANONICAL_MODEL" --dry-run \
    >"$OUTPUT_DIR/provenance/canonical-native-plan.txt" 2>&1 ||
    die "canonical model failed the exact 129-tensor native-Q8 inventory"
grep -Fq 'converted_tensors: 129' \
    "$OUTPUT_DIR/provenance/canonical-native-plan.txt" ||
    die "canonical native-Q8 inventory did not contain 129 tensors"

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model_layout=%s\ncanonical_model=%s\ncanonical_model_bytes=%s\n' \
        "$MODEL_LAYOUT" "$CANONICAL_MODEL" "$(stat -c %s "$CANONICAL_MODEL")"
    printf 'native_model=%s\nnative_model_bytes=%s\nprompt=%s\n' \
        "$NATIVE_MODEL" "$(stat -c %s "$NATIVE_MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\npower_limits_w=%s\nstage_split=22/21\n' \
        "$GPU_DEVICES" "$REQUIRED_POWER_LIMITS_W"
    printf 'contexts=%s\ntg_tokens=%s\nexact_tokens=%s\nnative_smoke_only=%s\n' \
        "$([[ $NATIVE_SMOKE_ONLY == 1 ]] && printf 512 || printf 512,4096,32768)" \
        "$TG_TOKENS" "$EXACT_TOKENS" "$NATIVE_SMOKE_ONLY"
    printf 'pair0_prefill_attention_rows=disabled\npair1_prefill_attention_rows=enabled\n'
    printf 'pair0_prefill_indexer_rows=enabled\npair1_prefill_indexer_rows=enabled\n'
    printf 'minimum_native_vram_saving_mib=%s\nminimum_throughput_ratio=%s\n' \
        "$MIN_NATIVE_VRAM_SAVING_MIB" "$MIN_THROUGHPUT_RATIO"
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
        timeout 20s nvidia-smi -i "$gpu" \
            --query-gpu=index,pci.bus_id,uuid,power.limit \
            --format=csv,noheader,nounits >>"$partial" 2>&1 || {
                mv -- "$partial" "$output"
                return 1
            }
    done
    mv -- "$partial" "$output"
}

validate_health() {
    local base=$1
    [[ -s $base.pre-gpu.csv && -s $base.post-gpu.csv ]] &&
        ! grep -Eq 'ERR!|Unknown Error|GPU is lost|GPU Unavailable' \
            "$base.pre-gpu.csv" "$base.post-gpu.csv" &&
        cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$base.pre-gpu.csv" &&
        cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$base.post-gpu.csv"
}

start_telemetry() {
    local output=$1
    nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.free,utilization.gpu,power.draw \
        --format=csv,noheader,nounits -lms "$TELEMETRY_INTERVAL_MS" \
        >"$output" 2>&1 &
    telemetry_pid=$!
}

validate_csv() {
    local csv=$1 expected_tokens=$2 expected_contexts=$3
    awk -F, -v tg="$expected_tokens" -v expected="$expected_contexts" '
        BEGIN {
            n_expected=split(expected, list, ",")
            for (i=1; i<=n_expected; i++) wanted[list[i]+0]=1
        }
        NR==1 {header=($1=="ctx_tokens" && $3=="prefill_tps" &&
                       $4=="gen_tokens" && $8=="gen_steady_tps"); next}
        NR>1 {
            rows++; ctx=$1+0
            if (!wanted[ctx] || seen[ctx]++ ||
                ($3+0)<=0 || $4!=tg || ($8+0)<=0) bad=1
        }
        END {
            for (ctx in wanted) if (!seen[ctx]) bad=1
            exit !(header && rows==n_expected && !bad)
        }
    ' "$csv"
}

validate_topology() {
    local arm=$1 log=$2 kind=$3 marker route required materialized
    materialized='materialized 344/344 candidates'
    required='required-native=0/0'
    if [[ $arm == native ]]; then
        # Tagged T32 owns one complete native-Q8 home residency and uses one
        # complete home-local F16 prefill binding. Pair attention splitting is
        # later and must not introduce a T32 partner projection.
        required='required-native=129/129'
    fi
    for marker in 'CUDA EP forced pipeline split 22/21' \
                  't256-placement=stage-aware' \
                  'dense-placement=stage-aware-fixed-22-21' \
                  "$materialized"; do
        grep -Fq "$marker" "$log" || return 1
    done
    grep -Eq "${required}([[:space:]]|$)" "$log" || return 1
    for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
        grep -Fq "$route" "$log" || return 1
    done
    ! grep -Fq 'required native-GGUF execution binding unavailable' "$log" || return 1
    ! grep -Fq 'tagged native dense-Q8 startup rejected' "$log" || return 1
    if [[ $kind != smoke ]]; then
        ! grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" || return 1
        grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" || return 1
        grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" || return 1
        grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log" || return 1
    fi
    if [[ $arm == native ]]; then
        ! grep -Fq 'CUDA q8 partner execution enabled: home tier 0 device 0 -> partner tier 2 device 1 (attn_q_b, ' "$log" || return 1
        ! grep -Fq 'CUDA q8 partner execution enabled: home tier 1 device 3 -> partner tier 3 device 2 (attn_q_b, ' "$log" || return 1
        ! grep -Fq 'CUDA T32 projection-only pair split enabled:' "$log" || return 1
    fi
}

validate_representation() {
    local arm=$1 log=$2
    if [[ $arm == canonical ]]; then
        ! grep -Fq 'tagged SM75 dense-Q8 GGUF installed' "$log" &&
            ! grep -Fq 'native-GGUF Q8 device=' "$log"
        return
    fi
    [[ $(grep -Fc 'tagged SM75 dense-Q8 GGUF installed through ordinary single-owner residency' "$log") == 1 ]] || return 1
    [[ $(grep -Fc 'tagged native dense-Q8:' "$log") == 4 ]] || return 1
    [[ $(grep -Fc 'native-GGUF Q8 device=' "$log") == 4 ]] || return 1
    [[ $(grep -Fc 'additional-allocation=0' "$log") == 4 ]] || return 1
    ! grep -Fq 'SM75 warp-interleaved Q8 cache fill' "$log" || return 1
    ! grep -Fq 'deferred native-primary Q8' "$log" || return 1
    ! grep -Fq 'canonical sources remain resident through prefill' "$log" || return 1
    ! grep -Fq 'registered native Q8 unavailable' "$log" || return 1
    ! grep -Fq 'registered native attention' "$log" || return 1
}

validate_exact_inventory() {
    local logits=$1 context token file
    for context in 512 4096 32768; do
        printf -v file 'frontier_%06d.logits.f32' "$context"
        [[ -s $logits/$file ]] || return 1
        for ((token=1; token<=EXACT_TOKENS; token++)); do
            printf -v file 'frontier_%06d.decode_%06d.logits.f32' \
                "$context" "$token"
            [[ -s $logits/$file ]] || return 1
        done
    done
    [[ $(find "$logits" -maxdepth 1 -type f -name '*.f32' | wc -l) == \
       $((3 * (EXACT_TOKENS + 1))) ]]
}

run_case() {
    local arm=$1 model=$2 kind=$3 tokens=$4 base=$5 logits=${6:-}
    local contexts=${7:-512,4096,32768} ctx_max=32768
    local rc=0 telemetry="$OUTPUT_DIR/telemetry/$arm-$kind.csv"
    local -a cmd
    [[ $contexts == 512 ]] && ctx_max=512
    capture_gpu_health "$base.pre-gpu.csv" || return 1
    start_telemetry "$telemetry"
    cmd=("${production_env[@]}"
        "DS4_CUDA_Q8_PLAN_AUDIT_CSV=$base.q8-plan.csv"
        "DS4_CUDA_Q8_BINDING_STATE_CSV=$base.q8-bindings.csv"
        ./ds4-bench --cuda --cuda-tensor-parallel
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM"
        --model "$model" --prompt-file "$PROMPT"
        --ctx-start 512 --ctx-max "$ctx_max" --ctx-alloc "$CTX_ALLOC"
        --step-mul 8 --prefill-chunk "$PREFILL_CHUNK"
        --gen-tokens "$tokens" --csv "$base.csv")
    if [[ -n $logits ]]; then
        cmd+=(--dump-frontier-logits-dir "$logits"
              --dump-decode-logits-dir "$logits")
    fi
    timeout --signal=TERM --kill-after=30s "${CASE_TIMEOUT_SECONDS}s" \
        "${cmd[@]}" >"$base.log" 2>&1 || rc=$?
    stop_telemetry
    capture_gpu_health "$base.post-gpu.csv" || return 1
    [[ $rc == 0 ]] || return "$rc"
    [[ -s $telemetry && $(wc -l <"$telemetry") -ge 4 ]] || return 1
    validate_health "$base" &&
        validate_csv "$base.csv" "$tokens" "$contexts" &&
        validate_topology "$arm" "$base.log" "$kind" &&
        validate_representation "$arm" "$base.log"
}

phase=production
capture_gpu_health "$OUTPUT_DIR/initial-gpu.csv" ||
    die "could not capture initial four-GPU health"
declare -A model_by_arm=(
    [canonical]="$CANONICAL_MODEL"
    [native]="$NATIVE_MODEL"
)
if [[ $NATIVE_SMOKE_ONLY == 1 ]]; then
    base="$OUTPUT_DIR/runs/native-smoke"
    printf 'Native-Q8 GGUF PP512 smoke model=%s arm=native...\n' \
        "$MODEL_LAYOUT"
    run_case native "$NATIVE_MODEL" smoke 16 "$base" '' 512 || {
        tail -n 240 "$base.log" >&2 || true
        die "native PP512 smoke failed validation"
    }
    phase=complete
    printf 'SM75 tagged native-Q8 GGUF %s PP512 smoke complete: %s\n' \
        "$MODEL_LAYOUT" "$OUTPUT_DIR"
    exit 0
fi
for arm in canonical native; do
    base="$OUTPUT_DIR/runs/$arm"
    printf 'Native-Q8 GGUF production performance model=%s arm=%s...\n' \
        "$MODEL_LAYOUT" "$arm"
    run_case "$arm" "${model_by_arm[$arm]}" performance \
        "$TG_TOKENS" "$base" || {
        tail -n 240 "$base.log" >&2 || true
        die "$arm production performance run failed validation"
    }
done

phase=exact
for arm in canonical native; do
    base="$OUTPUT_DIR/exact/$arm"
    logits="$base-logits"
    mkdir -p "$logits"
    printf 'Native-Q8 GGUF exact prefill/decode model=%s arm=%s...\n' \
        "$MODEL_LAYOUT" "$arm"
    run_case "$arm" "${model_by_arm[$arm]}" exact \
        "$EXACT_TOKENS" "$base" "$logits" || {
        tail -n 240 "$base.log" >&2 || true
        die "$arm exact prefill/decode run failed validation"
    }
    validate_exact_inventory "$logits" ||
        die "$arm exact logit inventory is incomplete"
done

expected="$OUTPUT_DIR/exact/expected-files.txt"
find "$OUTPUT_DIR/exact/canonical-logits" -maxdepth 1 -type f \
    -name '*.f32' -printf '%f\n' | sort >"$expected"
find "$OUTPUT_DIR/exact/native-logits" -maxdepth 1 -type f \
    -name '*.f32' -printf '%f\n' | sort >"$OUTPUT_DIR/exact/native-files.txt"
cmp -s "$expected" "$OUTPUT_DIR/exact/native-files.txt" ||
    die "canonical and native exact-output inventories differ"
while IFS= read -r file; do
    cmp -s "$OUTPUT_DIR/exact/canonical-logits/$file" \
           "$OUTPUT_DIR/exact/native-logits/$file" ||
        die "native GGUF diverged at $file"
done <"$expected"

phase=summarize
summarize_telemetry() {
    local input=$1 output=$2
    awk -F, '
        function trim(v) {gsub(/^[[:space:]]+|[[:space:]]+$/, "", v); return v}
        {
            gpu=trim($2); used=trim($4)+0
            if (gpu !~ /^[0-9]+$/ || used<0) next
            count[gpu]++; if (used>peak[gpu]) peak[gpu]=used
            sample=int(valid/4); valid++
            key=sample SUBSEP gpu
            if (!seen[key]++) {total[sample]+=used; members[sample]++}
        }
        END {
            for (gpu in count) printf "gpu,%s,peak_mib,%.0f,samples,%d\n", gpu,peak[gpu],count[gpu]
            for (sample in total) if (members[sample]==4 && total[sample]>aggregate) aggregate=total[sample]
            printf "aggregate,peak_mib,%.0f\n",aggregate
        }
    ' "$input" | sort >"$output"
}
for arm in canonical native; do
    summarize_telemetry "$OUTPUT_DIR/telemetry/$arm-performance.csv" \
        "$OUTPUT_DIR/summary/$arm-vram.csv"
done
canonical_peak=$(awk -F, '$1=="aggregate" {print $3}' \
    "$OUTPUT_DIR/summary/canonical-vram.csv")
native_peak=$(awk -F, '$1=="aggregate" {print $3}' \
    "$OUTPUT_DIR/summary/native-vram.csv")
[[ -n $canonical_peak && -n $native_peak ]] || die "peak VRAM summary is incomplete"
vram_saving=$(awk -v canonical="$canonical_peak" -v native="$native_peak" \
    'BEGIN {printf "%.0f", canonical-native}')
(( vram_saving >= MIN_NATIVE_VRAM_SAVING_MIB )) ||
    die "native peak VRAM saving is ${vram_saving} MiB; required at least ${MIN_NATIVE_VRAM_SAVING_MIB} MiB"

awk -F, -v minimum="$MIN_THROUGHPUT_RATIO" '
    NR==FNR {if (FNR>1) {pp[$1]=$3; tg[$1]=$8}; next}
    FNR>1 {
        if (($3+0)/(pp[$1]+0)<minimum || ($8+0)/(tg[$1]+0)<minimum) bad=1
    }
    END {exit bad}
' "$OUTPUT_DIR/runs/canonical.csv" "$OUTPUT_DIR/runs/native.csv" ||
    die "native model regressed a prefill or decode frontier below the accepted throughput ratio"

{
    printf '# SM75 tagged native-Q8 GGUF production qualification\n\n'
    printf 'Model layout: `%s`  \n' "$MODEL_LAYOUT"
    printf 'Exactness: all prefill frontier logits and all %s decode logits at PP512, PP4096, and PP32768 are byte-identical.  \n' "$EXACT_TOKENS"
    printf 'Peak aggregate VRAM: canonical %.0f MiB; native %.0f MiB; saving %.0f MiB.  \n\n' \
        "$canonical_peak" "$native_peak" "$vram_saving"
    printf '| Context | Canonical prefill tok/s | Native prefill tok/s | Prefill ratio | Canonical decode tok/s | Native decode tok/s | Decode ratio |\n'
    printf '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    awk -F, '
        NR==FNR {if (FNR>1) {pp[$1]=$3; tg[$1]=$8}; next}
        FNR>1 {printf "| %s | %.3f | %.3f | %.6fx | %.3f | %.3f | %.6fx |\n",$1,pp[$1],$3,$3/pp[$1],tg[$1],$8,$8/tg[$1]}
    ' "$OUTPUT_DIR/runs/canonical.csv" "$OUTPUT_DIR/runs/native.csv"
    printf '\nRuntime repack/replacement allocations: zero. Stable pair-0-attention-off, pair-0-indexer-on topology retained.\n'
} | tee "$OUTPUT_DIR/summary/report.md"

printf 'bit_exact=true\nmodel_layout=%s\nfrontiers=512,4096,32768\n' \
    "$MODEL_LAYOUT" >"$OUTPUT_DIR/summary/acceptance.txt"
printf 'decode_tokens_per_frontier=%s\ncanonical_peak_vram_mib=%s\nnative_peak_vram_mib=%s\n' \
    "$EXACT_TOKENS" "$canonical_peak" "$native_peak" \
    >>"$OUTPUT_DIR/summary/acceptance.txt"
printf 'native_peak_vram_saving_mib=%s\nminimum_throughput_ratio=%s\n' \
    "$vram_saving" "$MIN_THROUGHPUT_RATIO" \
    >>"$OUTPUT_DIR/summary/acceptance.txt"
printf 'runtime_repack=false\nruntime_replacement=false\nadditional_native_q8_allocation_bytes=0\n' \
    >>"$OUTPUT_DIR/summary/acceptance.txt"

phase=complete
printf 'SM75 tagged native-Q8 GGUF %s production A/B complete: %s\n' \
    "$MODEL_LAYOUT" "$OUTPUT_DIR"
