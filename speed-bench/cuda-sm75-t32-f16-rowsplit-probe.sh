#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run one production-shaped 32K prefill process with T32 FP16 partner results
and attention/indexer row splitting enabled on both tensor-parallel pairs.

This is a crash probe, not a throughput A/B. It changes only the pair-0
row-split suppression used by the stable T32 production A/B and deliberately
does not support resume.

Optional environment:
  MODEL_LAYOUT=mixed15|all43   default: mixed15
  MIXED_MODEL=...             default: gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf
  ALL43_MODEL=...             default: gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf
  PROMPT=...                  default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  REQUIRED_POWER_LIMITS_W=250,260,250,250
                              physical CUDA indices 0,1,2,3
  CASE_TIMEOUT_SECONDS=900
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  T32_F16_ROWSPLIT_PROBE_DIR=...

The runner requires the current 344/344 stage-aware plan, all 43 T32
projections resident on partners, direct peer routes, both attention row
splits, both corresponding prefill indexer row splits, and fused FP16 results.
It records physical bus IDs, UUIDs, serials, and power limits before and after
the process. A GPU-loss run is expected to leave a partial archive.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
MODEL_LAYOUT=${MODEL_LAYOUT:-mixed15}
MIXED_MODEL=${MIXED_MODEL:-$repo_dir/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf}
ALL43_MODEL=${ALL43_MODEL:-$repo_dir/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf}
case "$MODEL_LAYOUT" in
    mixed15) MODEL=$MIXED_MODEL ;;
    all43) MODEL=$ALL43_MODEL ;;
    *) die "MODEL_LAYOUT must be mixed15 or all43" ;;
esac
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
REQUIRED_POWER_LIMITS_W=${REQUIRED_POWER_LIMITS_W:-250,260,250,250}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-900}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
PREFILL_CHUNK=2048
PIPELINE_MB=512
CTX_ALLOC=33025
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${T32_F16_ROWSPLIT_PROBE_DIR:-$repo_dir/sm75-t32-f16-rowsplit-probe-$stamp}

[[ $MODEL == /* && -f $MODEL ]] || die "model not found at absolute path: $MODEL"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "this production probe requires the fixed promessi_sposi prompt"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "require GPU_DEVICES=0,3,1,2, GPU_VRAM=auto, STAGE_SPLIT=22"
[[ $CASE_TIMEOUT_SECONDS =~ ^[1-9][0-9]*$ ]] ||
    die "CASE_TIMEOUT_SECONDS must be a positive integer"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"
for tool in awk basename cat cmp date dirname env git grep make mkdir mv nproc \
            nvidia-smi python3 rm sort stat stdbuf tail tar tee timeout tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done

IFS=, read -r -a required_power <<<"$REQUIRED_POWER_LIMITS_W"
(( ${#required_power[@]} == 4 )) ||
    die "REQUIRED_POWER_LIMITS_W must contain physical GPU 0,1,2,3 limits"
for gpu in 0 1 2 3; do
    [[ ${required_power[$gpu]} =~ ^[0-9]+([.][0-9]+)?$ ]] ||
        die "invalid required power limit for physical GPU $gpu"
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "GPU $gpu is compute capability ${cap:-unknown}, not SM75"
    limit=$(nvidia-smi -i "$gpu" --query-gpu=power.limit \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    awk -v actual="$limit" -v expected="${required_power[$gpu]}" \
        'BEGIN {exit !((actual+0)==(expected+0))}' ||
        die "physical GPU $gpu power limit is ${limit:-unknown} W, expected ${required_power[$gpu]} W"
done

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{provenance,logits}
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
            rm -f -- "$partial" "$archive"
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
    DS4_BENCH_ROUTED_QUANT_AUDIT=1
    DS4_CUDA_TP_PREFILL_ATTN_ROWS=1
    DS4_CUDA_TP_PREFILL_INDEXER_ROWS_PAIRS=0,1
    DS4_CUDA_TP_PREFILL_ATTN_ROWS_AUDIT=1
    DS4_CUDA_T32_F16_FUSED=1
)

phase=build
targets=(ds4-bench tests/test_engine_mgpu_placement)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q "${targets[@]}" CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi
"${clean[@]}" ./tests/test_engine_mgpu_placement \
    >"$OUTPUT_DIR/placement-tests.log" 2>&1 || {
        tail -n 180 "$OUTPUT_DIR/placement-tests.log" >&2
        die "placement regression tests failed"
    }

capture_gpu_health() {
    local output=$1 partial="$1.partial.$$" gpu
    : >"$partial"
    for gpu in 0 1 2 3; do
        nvidia-smi -i "$gpu" \
            --query-gpu=index,pci.bus_id,uuid,serial,power.limit \
            --format=csv,noheader,nounits >>"$partial" 2>&1 || {
                mv -- "$partial" "$output"
                return 1
            }
    done
    mv -- "$partial" "$output"
}

phase=manifest
capture_gpu_health "$OUTPUT_DIR/initial-gpu.csv" ||
    die "could not capture the initial four-GPU identity and power limits"
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model_layout=%s\nmodel=%s\nmodel_bytes=%s\n' \
        "$MODEL_LAYOUT" "$MODEL" "$(stat -c %s "$MODEL")"
    printf 'prompt=%s\ngpu_devices=%s\nstage_split=22/21\n' \
        "$PROMPT" "$GPU_DEVICES"
    printf 'context=32768\nprefill_chunk=%s\npipeline_mb=%s\nctx_alloc=%s\n' \
        "$PREFILL_CHUNK" "$PIPELINE_MB" "$CTX_ALLOC"
    printf 't32_f16_fused=enabled\npair0_prefill_attention_rows=enabled\n'
    printf 'pair1_prefill_attention_rows=enabled\nrequired_power_limits_w=%s\n' \
        "$REQUIRED_POWER_LIMITS_W"
    cat "$OUTPUT_DIR/initial-gpu.csv"
    printf '\ntopology:\n'
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

validate_layout() {
    local log=$1 expected_count expected_layers
    if [[ $MODEL_LAYOUT == mixed15 ]]; then
        expected_count=15
        expected_layers=6,8,10,12,14,16,18,20,30,32,34,36,38,40,42
    else
        expected_count=43
        expected_layers=$(printf '%s,' {0..42}); expected_layers=${expected_layers%,}
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

validate_plan() {
    local plan=$1 bindings=$2
    [[ -s $plan && -s $bindings ]] || return 1
    [[ $(wc -l <"$plan") == 345 && $(wc -l <"$bindings") == 345 ]] || return 1
    awk -F, '
        NR==1 {header=($1=="sequence" && $2=="label" && $3=="consumer_device" && $7=="resident_device" && $13=="status"); next}
        {
            rows++
            if (($7+0)<0 || ($13!="home" && $13!="partner")) bad=1
            if ($2 ~ /\.attn_q_b\.weight$/) {
                t32++
                if ($13!="partner") bad=1
                if ($3==0 && $7==1) pair0++
                else if ($3==3 && $7==2) pair1++
                else bad=1
            }
        }
        END {exit !(header && rows==344 && t32==43 && pair0==22 && pair1==21 && !bad)}
    ' "$plan" || return 1
    awk -F, '
        NR==1 {header=($1=="consumer_device" && $2=="resident_device" && $3=="partner_offload" && $12=="label" && $15=="live"); next}
        {
            rows++
            if (($13+0)<=0 || ($14+0)<=0 || $15!=1) bad=1
            if ($12 ~ /\.attn_q_b\.weight$/) {
                t32++
                if ($3!=1) bad=1
            }
        }
        END {exit !(header && rows==344 && t32==43 && !bad)}
    ' "$bindings"
}

validate_log() {
    local log=$1 marker route line local_calls=0 partner_calls=0 total_partner=0 f16_calls=0
    for marker in 'CUDA EP forced pipeline split 22/21' \
                  't256-placement=stage-aware' \
                  'dense-placement=stage-aware-fixed-22-21' \
                  'materialized 344/344 candidates' \
                  'SM75 routed Q32 layout enabled' \
                  'SM75 native F16 indexer cache and streaming WMMA64 enabled'; do
        grep -Fq "$marker" "$log" || {
            printf 'validation: missing production marker: %s\n' "$marker" >&2
            return 1
        }
    done
    for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
        grep -Fq "$route" "$log" || {
            printf 'validation: missing direct route: %s\n' "$route" >&2
            return 1
        }
    done
    ! grep -Fq 'required but unavailable' "$log" || return 1
    ! grep -Fq 'prefill attention row split pair-scoped disable:' "$log" || return 1
    [[ $(grep -Fc 'CUDA prefill attention query-row split enabled:' "$log") == 2 ]] || return 1
    grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" || return 1
    grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" || return 1
    [[ $(grep -Fc 'CUDA prefill indexer score/top-k row split enabled:' "$log") == 2 ]] || return 1
    grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" || return 1
    grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log" || return 1
    [[ $(grep -Fc 'CUDA prefill attention row audit dispatch=split ' "$log") -gt 0 ]] || return 1
    validate_layout "$log" || return 1

    line=$(grep -F 'CUDA T32 f16-output fused summary:' "$log" | tail -n 1 || true)
    if [[ $line =~ local=([0-9]+)[[:space:]]partner=([0-9]+) ]]; then
        local_calls=${BASH_REMATCH[1]}; partner_calls=${BASH_REMATCH[2]}
    fi
    line=$(grep -F 'CUDA q8 fp16 partner summary:' "$log" | tail -n 1 || true)
    if [[ $line =~ calls=([0-9]+).*f16-result-calls=([0-9]+) ]]; then
        total_partner=${BASH_REMATCH[1]}; f16_calls=${BASH_REMATCH[2]}
    fi
    if (( local_calls != 0 || partner_calls == 0 ||
          total_partner < partner_calls || f16_calls != partner_calls )); then
        printf 'validation: fused T32 counters local=%s partner=%s total-partner=%s f16-results=%s\n' \
            "$local_calls" "$partner_calls" "$total_partner" "$f16_calls" >&2
        return 1
    fi
}

phase=all-pairs-fused-32k
printf 'state=started\nphase=%s\ndate_utc=%s\n' "$phase" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$OUTPUT_DIR/case.started"
capture_gpu_health "$OUTPUT_DIR/case.pre-gpu.csv" ||
    die "could not capture pre-run GPU state"

set +e
"${production_env[@]}" \
    "DS4_CUDA_Q8_PLAN_AUDIT_CSV=$OUTPUT_DIR/q8-plan.csv" \
    "DS4_CUDA_Q8_BINDING_STATE_CSV=$OUTPUT_DIR/q8-bindings.csv" \
    timeout --signal=INT --kill-after=30s "${CASE_TIMEOUT_SECONDS}s" \
    stdbuf -oL -eL ./ds4-bench --cuda --cuda-tensor-parallel \
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
        --model "$MODEL" --prompt-file "$PROMPT" \
        --ctx-start 32768 --ctx-max 32768 --ctx-alloc "$CTX_ALLOC" \
        --step-mul 8 --prefill-chunk "$PREFILL_CHUNK" \
        --gen-tokens 0 --csv "$OUTPUT_DIR/result.csv" \
        --dump-frontier-logits-dir "$OUTPUT_DIR/logits" \
        >"$OUTPUT_DIR/case.log" 2>&1
run_status=$?
set -e
printf 'status=%s\ndate_utc=%s\n' "$run_status" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$OUTPUT_DIR/case.result"
capture_gpu_health "$OUTPUT_DIR/case.post-gpu.csv" ||
    die "GPU health capture failed after the all-pairs fused process"
(( run_status == 0 )) || {
    tail -n 240 "$OUTPUT_DIR/case.log" >&2 || true
    die "all-pairs fused 32K process failed with status $run_status"
}
cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$OUTPUT_DIR/case.pre-gpu.csv" &&
    cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$OUTPUT_DIR/case.post-gpu.csv" ||
    die "GPU identity or power limit changed during the process"
validate_plan "$OUTPUT_DIR/q8-plan.csv" "$OUTPUT_DIR/q8-bindings.csv" ||
    die "dense-Q8 plan or binding state is not the required 344/344 partner-T32 plan"
validate_log "$OUTPUT_DIR/case.log" ||
    die "all-pairs fused process failed production-path validation"
awk -F, '
    NR==1 {header=($1=="ctx_tokens" && $3=="prefill_tps"); next}
    NR==2 {rows++; valid=($1==32768 && ($3+0)>0); next}
    NR>2 {rows++}
    END {exit !(header && rows==1 && valid)}
' "$OUTPUT_DIR/result.csv" || die "32K benchmark result is incomplete"
[[ -s "$OUTPUT_DIR/logits/frontier_032768.logits.f32" &&
   -s "$OUTPUT_DIR/logits/frontier_032768.logits.json" ]] ||
    die "32K frontier logits are incomplete"

phase=complete
printf 'SM75 T32 FP16 all-pairs row-split crash probe passed: %s\n' "$OUTPUT_DIR"
