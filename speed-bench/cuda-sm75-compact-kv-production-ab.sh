#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Qualify the exact 736-byte compressed-attention KV cache against the shipping
F32 cache on one tagged-native all43 model.  Both arms use the same model,
stable four-GPU topology, and 256K context allocation.  The run measures
prefill/decode throughput, byte-exact logits, peak aggregate VRAM, and GPU
health.

Required environment:
  MODEL=/absolute/path/to/SM75-Native-Q8-all43.gguf

Optional environment:
  PROMPT=...                         default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  REQUIRED_POWER_LIMITS_W=250,260,250,250
  CTX_ALLOC=262145
  TG_TOKENS=256
  EXACT_TOKENS=16
  TELEMETRY_INTERVAL_MS=200
  CASE_TIMEOUT_SECONDS=1800
  MIN_COMPACT_VRAM_SAVING_MIB=12000
  MIN_THROUGHPUT_RATIO=0.95
  DIAGNOSTIC_PACK_AUDIT=0           1: F32/compact PP512 one-token exact A/B,
                                    with compact per-layer packed-row checks
  DIAGNOSTIC_PREFILL_ISOLATION=0    1: PP4096 exact F32, compact-hybrid, and
                                    compact-direct three-arm comparison
  DIAGNOSTIC_DECODE_ISOLATION=0     1: compact PP512 one-token runs only,
                                    first without and then with snapshot
  DIAGNOSTIC_DECODE_PROFILE=0       1: Nsight Systems capture of the second
                                    PP512 decode token for F32 and compact
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  COMPACT_KV_PRODUCTION_AB_DIR=...

The compact format is opt-in.  This script does not alter or create the model.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

MODEL=${MODEL:-}
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
REQUIRED_POWER_LIMITS_W=${REQUIRED_POWER_LIMITS_W:-250,260,250,250}
CTX_ALLOC=${CTX_ALLOC:-262145}
TG_TOKENS=${TG_TOKENS:-256}
EXACT_TOKENS=${EXACT_TOKENS:-16}
TELEMETRY_INTERVAL_MS=${TELEMETRY_INTERVAL_MS:-200}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-1800}
MIN_COMPACT_VRAM_SAVING_MIB=${MIN_COMPACT_VRAM_SAVING_MIB:-12000}
MIN_THROUGHPUT_RATIO=${MIN_THROUGHPUT_RATIO:-0.95}
DIAGNOSTIC_PACK_AUDIT=${DIAGNOSTIC_PACK_AUDIT:-0}
DIAGNOSTIC_PREFILL_ISOLATION=${DIAGNOSTIC_PREFILL_ISOLATION:-0}
DIAGNOSTIC_DECODE_ISOLATION=${DIAGNOSTIC_DECODE_ISOLATION:-0}
DIAGNOSTIC_DECODE_PROFILE=${DIAGNOSTIC_DECODE_PROFILE:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
PREFILL_CHUNK=2048
PIPELINE_MB=512
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${COMPACT_KV_PRODUCTION_AB_DIR:-$repo_dir/sm75-compact-kv-all43-production-ab-$stamp}

[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name an existing absolute all43 tagged-native GGUF"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "this production A/B requires the fixed promessi_sposi prompt"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "require GPU_DEVICES=0,3,1,2, GPU_VRAM=auto, STAGE_SPLIT=22"
for value in "$CTX_ALLOC" "$TG_TOKENS" "$EXACT_TOKENS" \
             "$TELEMETRY_INTERVAL_MS" "$CASE_TIMEOUT_SECONDS" \
             "$MIN_COMPACT_VRAM_SAVING_MIB"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] ||
        die "positive integer required, got: $value"
done
(( CTX_ALLOC >= 262145 )) ||
    die "CTX_ALLOC must be at least 262145 for the 256K residency gate"
awk -v ratio="$MIN_THROUGHPUT_RATIO" \
    'BEGIN {exit !(ratio+0>0 && ratio+0<=1)}' ||
    die "MIN_THROUGHPUT_RATIO must be in (0,1]"
for flag in DIAGNOSTIC_PACK_AUDIT DIAGNOSTIC_PREFILL_ISOLATION \
            DIAGNOSTIC_DECODE_ISOLATION \
            DIAGNOSTIC_DECODE_PROFILE \
            SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
(( DIAGNOSTIC_PACK_AUDIT + DIAGNOSTIC_PREFILL_ISOLATION +
   DIAGNOSTIC_DECODE_ISOLATION +
   DIAGNOSTIC_DECODE_PROFILE <= 1 )) ||
    die "select at most one diagnostic mode"
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"
for tool in awk cmp date env find git grep make mkdir mv nproc nvidia-smi \
            sort stat tail tar tee timeout tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
if [[ $DIAGNOSTIC_DECODE_PROFILE == 1 ]]; then
    command -v nsys >/dev/null 2>&1 || die "nsys not found"
fi

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
mkdir -p "$OUTPUT_DIR"/{runs,exact,telemetry,summary,provenance,nsys}
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
    DS4_BENCH_PHASE_TRACE=1
    DS4_SESSION_LIFECYCLE_TRACE=1
    DS4_SESSION_IO_TRACE=1
)

phase=build
if [[ $SKIP_BUILD == 0 ]]; then
    make -j"$(nproc)" ds4-bench tests/test_engine_mgpu_placement \
        CUDA_ARCH=sm_75 2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q ds4-bench tests/test_engine_mgpu_placement CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale production binaries"
fi
"${clean[@]}" DS4_CUDA_ATTN_COMP_CACHE=sm75-compact \
    ./tests/test_engine_mgpu_placement \
    >"$OUTPUT_DIR/placement-test.log" 2>&1 || {
        tail -n 200 "$OUTPUT_DIR/placement-test.log" >&2 || true
        die "compact-cache placement/accounting regression failed"
    }

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\npower_limits_w=%s\nstage_split=22/21\n' \
        "$GPU_DEVICES" "$REQUIRED_POWER_LIMITS_W"
    printf 'contexts=512,4096,32768\nctx_alloc=%s\n' "$CTX_ALLOC"
    printf 'tg_tokens=%s\nexact_tokens=%s\n' "$TG_TOKENS" "$EXACT_TOKENS"
    printf 'control_cache=f32\ncandidate_cache=sm75-compact-exact\n'
    printf 'minimum_compact_vram_saving_mib=%s\nminimum_throughput_ratio=%s\n' \
        "$MIN_COMPACT_VRAM_SAVING_MIB" "$MIN_THROUGHPUT_RATIO"
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
    local csv=$1 expected_tokens=$2
    awk -F, -v tg="$expected_tokens" '
        NR==1 {header=($1=="ctx_tokens" && $3=="prefill_tps" &&
                       $4=="gen_tokens" && $8=="gen_steady_tps"); next}
        NR>1 {
            rows++; ctx=$1+0
            if ((ctx!=512 && ctx!=4096 && ctx!=32768) || seen[ctx]++ ||
                ($3+0)<=0 || $4!=tg || ($8+0)<=0) bad=1
        }
        END {exit !(header && rows==3 && seen[512] && seen[4096] &&
                    seen[32768] && !bad)}
    ' "$csv"
}

validate_topology() {
    local log=$1 marker route
    for marker in 'CUDA EP forced pipeline split 22/21' \
                  't256-placement=stage-aware' \
                  'dense-placement=stage-aware-fixed-22-21' \
                  'materialized 344/344 candidates'; do
        grep -Fq "$marker" "$log" || return 1
    done
    grep -Eq 'required-native=129/129([[:space:]]|$)' "$log" || return 1
    for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
        grep -Fq "$route" "$log" || return 1
    done
    ! grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" || return 1
    grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" || return 1
    grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" || return 1
    grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log" || return 1
    ! grep -Fq 'CUDA T32 projection-only pair split enabled:' "$log" || return 1
    ! grep -Fq 'required native-GGUF execution binding unavailable' "$log" || return 1
}

validate_selector() {
    local arm=$1 log=$2
    if [[ $arm == f32 ]]; then
        grep -Fq 'compressed-attention cache format=f32 row-bytes=2048' "$log" &&
            ! grep -Fq 'compact attention hybrid summary:' "$log"
    elif [[ $arm == compact-direct ]]; then
        grep -Fq 'compressed-attention cache format=sm75-compact-exact row-bytes=736' "$log" &&
            ! grep -Fq 'compact attention hybrid summary:' "$log" &&
            ! grep -Fq 'requested compressed-attention cache format' "$log"
    else
        grep -Fq 'compressed-attention cache format=sm75-compact-exact row-bytes=736' "$log" &&
            grep -Eq 'SM75 compact attention hybrid summary: calls=[1-9][0-9]* ' "$log" &&
            ! grep -Fq 'requested compressed-attention cache format' "$log" &&
            ! grep -Fq 'compact attention hybrid scratch allocation failed' "$log"
    fi
}

validate_ctx512_selector() {
    local arm=$1 log=$2
    if [[ $arm == f32 ]]; then
        grep -Fq 'compressed-attention cache format=f32 row-bytes=2048' "$log"
    else
        grep -Fq \
            'compressed-attention cache format=sm75-compact-exact row-bytes=736' \
            "$log"
    fi
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
    local arm=$1 kind=$2 tokens=$3 base=$4 logits=${5:-}
    local ctx_max=${6:-32768}
    local rc=0 telemetry="$OUTPUT_DIR/telemetry/$arm-$kind.csv"
    local format=f32
    local -a audit_env=() cmd
    [[ $arm == compact* ]] && format=sm75-compact
    if [[ $arm == compact* && ($kind == exact || $kind == pack-audit) ]]; then
        audit_env+=(DS4_CUDA_COMPACT_ATTN_PACK_AUDIT=1)
    fi
    if [[ $arm == compact-direct ]]; then
        audit_env+=(DS4_CUDA_NO_ATTN_COMPACT_HYBRID=1)
    fi
    capture_gpu_health "$base.pre-gpu.csv" || return 1
    start_telemetry "$telemetry"
    cmd=("${production_env[@]}" "${audit_env[@]}"
        "DS4_CUDA_ATTN_COMP_CACHE=$format"
        ./ds4-bench --cuda --cuda-tensor-parallel
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM"
        --model "$MODEL" --prompt-file "$PROMPT"
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
    if [[ $ctx_max == 512 ]]; then
        validate_health "$base" &&
            awk -F, -v tg="$tokens" '
                NR==1 {header=($1=="ctx_tokens" && $3=="prefill_tps" &&
                               $4=="gen_tokens" && $5=="gen_tps" &&
                               $6=="gen_first_ms" &&
                               $7=="gen_steady_tokens" &&
                               $8=="gen_steady_tps"); next}
                NR==2 {
                    row=($1==512 && ($3+0)>0 && $4==tg &&
                         ($5+0)>0 && ($6+0)>0 &&
                         (tg==1 ? (($7+0)==0 && ($8+0)==0) :
                                  (($7+0)>0 && ($8+0)>0)))
                }
                END {exit !(header && NR==2 && row)}
            ' "$base.csv" &&
            validate_ctx512_selector "$arm" "$base.log"
    else
        validate_health "$base" && validate_csv "$base.csv" "$tokens" &&
            validate_topology "$base.log" &&
            validate_selector "$arm" "$base.log"
    fi
}

run_decode_isolation_case() {
    local mode=$1 base="$OUTPUT_DIR/runs/compact-$1" rc=0
    local telemetry="$OUTPUT_DIR/telemetry/compact-$1.csv"
    local -a snapshot_env=() cmd
    [[ $mode == snapshot ]] &&
        snapshot_env+=(DS4_BENCH_FORCE_FRONTIER_SNAPSHOT=1)
    [[ $mode == nosnapshot ]] &&
        snapshot_env+=(DS4_BENCH_DISABLE_SNAPSHOT=1)

    capture_gpu_health "$base.pre-gpu.csv" || return 1
    start_telemetry "$telemetry"
    cmd=("${production_env[@]}"
        DS4_CUDA_ATTN_COMP_CACHE=sm75-compact
        DS4_BENCH_CRASH_TRACE=1
        DS4_SESSION_DECODE_TRACE=1
        DS4_CUDA_COMPACT_ATTN_TRACE=1
        DS4_CUDA_COMPACT_ATTN_SYNC_TRACE=1
        "DS4_BENCH_PROGRESS_JOURNAL=$base.progress.csv"
        "${snapshot_env[@]}"
        ./ds4-bench --cuda --cuda-tensor-parallel
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM"
        --model "$MODEL" --prompt-file "$PROMPT"
        --ctx-start 512 --ctx-max 512 --ctx-alloc "$CTX_ALLOC"
        --step-mul 8 --prefill-chunk "$PREFILL_CHUNK"
        --gen-tokens 1 --csv "$base.csv")
    timeout --signal=TERM --kill-after=30s "${CASE_TIMEOUT_SECONDS}s" \
        "${cmd[@]}" >"$base.log" 2>&1 || rc=$?
    stop_telemetry
    capture_gpu_health "$base.post-gpu.csv" || return 1
    printf '%s\n' "$rc" >"$base.exit-status.txt"

    validate_health "$base" || return 1
    [[ -s $telemetry && $(wc -l <"$telemetry") -ge 4 ]] || return 1
    if [[ $rc == 0 ]]; then
        awk -F, '
            NR==1 {header=($1=="ctx_tokens" && $3=="prefill_tps" &&
                           $4=="gen_tokens" && $8=="gen_steady_tps"); next}
            NR==2 {row=($1==512 && ($3+0)>0 && $4==1)}
            END {exit !(header && NR==2 && row)}
        ' "$base.csv" || return 1
        grep -Fq 'ds4-bench: completed first decode eval at frontier 512' \
            "$base.log" || return 1
    fi
    if [[ $mode == snapshot ]]; then
        grep -Fq 'ds4-bench: completed snapshot at frontier 512' \
            "$base.log" || return 1
    else
        ! grep -Fq 'ds4-bench: starting snapshot at frontier 512' \
            "$base.log" || return 1
    fi
    return "$rc"
}

run_decode_profile_case() {
    local arm=$1 base="$OUTPUT_DIR/nsys/$1-pp512-decode" rc=0 format=f32
    local telemetry="$OUTPUT_DIR/telemetry/$1-decode-profile.csv"
    local profile_tmp="$OUTPUT_DIR/nsys/tmp-$1"
    local -a cmd
    [[ $arm == compact ]] && format=sm75-compact
    mkdir -p "$profile_tmp"

    capture_gpu_health "$base.pre-gpu.csv" || return 1
    start_telemetry "$telemetry"
    cmd=("${production_env[@]}"
        "TMPDIR=$profile_tmp"
        "DS4_CUDA_ATTN_COMP_CACHE=$format"
        DS4_NSYS_CAPTURE_DECODE_SKIP=1
        DS4_NSYS_CAPTURE_DECODE_TOKENS=1
        nsys profile --force-overwrite=true --sample=none --cpuctxsw=none
        --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi
        --capture-range-end=stop --output="$base"
        ./ds4-bench --cuda --cuda-tensor-parallel
        --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM"
        --model "$MODEL" --prompt-file "$PROMPT"
        --ctx-start 512 --ctx-max 512 --ctx-alloc "$CTX_ALLOC"
        --step-mul 8 --prefill-chunk "$PREFILL_CHUNK"
        --gen-tokens 2 --csv "$base-benchmark.csv")
    timeout --signal=TERM --kill-after=30s "${CASE_TIMEOUT_SECONDS}s" \
        "${cmd[@]}" >"$base.log" 2>&1 || rc=$?
    stop_telemetry
    capture_gpu_health "$base.post-gpu.csv" || return 1
    [[ $rc == 0 ]] || return "$rc"
    validate_health "$base" || return 1
    validate_ctx512_selector "$arm" "$base.log" || return 1
    [[ -s $base.nsys-rep && -s $base-benchmark.csv ]] || return 1
    nsys stats --report cuda_gpu_kern_sum --format csv "$base.nsys-rep" \
        >"$base-cuda-gpu-kern-sum.csv" \
        2>"$base-cuda-gpu-kern-sum.log" || return 1
    [[ -s $base-cuda-gpu-kern-sum.csv ]] || return 1
    nsys stats --report cuda_api_sum --format csv "$base.nsys-rep" \
        >"$base-cuda-api-sum.csv" 2>"$base-cuda-api-sum.log" || return 1
    [[ -s $base-cuda-api-sum.csv ]] || return 1
    if grep -Eq ',cudaMalloc(Host)?$' "$base-cuda-api-sum.csv"; then
        printf 'error: %s warmed decode capture still contains a CUDA allocation\n' \
            "$arm" >&2
        return 1
    fi
    if [[ $arm == compact ]]; then
        grep -Fq \
            'compact exact-decode stages reserved during graph setup: tiers=2 bytes-per-tier=2097152' \
            "$base.log" || return 1
        grep -Eq 'SM75 compact exact score split summary: calls=[1-9][0-9]* materialized=[1-9][0-9]*' \
            "$base.log" || return 1
    fi
}

if [[ $DIAGNOSTIC_PACK_AUDIT == 1 ]]; then
    phase=pack-audit
    capture_gpu_health "$OUTPUT_DIR/initial-gpu.csv" ||
        die "could not capture initial four-GPU health"
    for arm in f32 compact; do
        base="$OUTPUT_DIR/exact/$arm-pack-audit"
        logits="$OUTPUT_DIR/exact/$arm-pack-audit-logits"
        mkdir -p "$logits"
        printf 'Compact-KV PP512 production pack audit arm=%s...\n' "$arm"
        run_case "$arm" pack-audit 1 "$base" "$logits" 512 || {
            tail -n 240 "$base.log" >&2 || true
            die "$arm PP512 production pack audit failed"
        }
        [[ -s $logits/frontier_000512.logits.f32 &&
           -s $logits/frontier_000512.decode_000001.logits.f32 ]] ||
            die "$arm PP512 production pack audit logit inventory is incomplete"
        if [[ $arm == compact ]]; then
            grep -Eq 'SM75 compact exact score split summary: calls=[1-9][0-9]* materialized=[1-9][0-9]*' \
                "$base.log" ||
                die "compact PP512 production pack audit missed materialized exact score-split dispatch"
        fi
    done
    for file in frontier_000512.logits.f32 \
                frontier_000512.decode_000001.logits.f32; do
        cmp -s "$OUTPUT_DIR/exact/f32-pack-audit-logits/$file" \
               "$OUTPUT_DIR/exact/compact-pack-audit-logits/$file" ||
            die "compact PP512 production pack audit diverged at $file"
    done
    phase=finished
    printf 'Compact-KV PP512 production exact A/B pack audit passed: %s\n' \
        "$OUTPUT_DIR"
    exit 0
fi

if [[ $DIAGNOSTIC_PREFILL_ISOLATION == 1 ]]; then
    phase=prefill-isolation
    capture_gpu_health "$OUTPUT_DIR/initial-gpu.csv" ||
        die "could not capture initial four-GPU health"
    for arm in f32 compact compact-direct; do
        base="$OUTPUT_DIR/exact/$arm-prefill-isolation"
        logits="$base-logits"
        mkdir -p "$logits"
        printf 'Compact-KV PP4096 prefill isolation arm=%s...\n' "$arm"
        run_case "$arm" exact 1 "$base" "$logits" 4096 || {
            tail -n 240 "$base.log" >&2 || true
            die "$arm PP4096 prefill isolation run failed"
        }
        [[ -s $logits/frontier_000512.logits.f32 &&
           -s $logits/frontier_004096.logits.f32 ]] ||
            die "$arm PP4096 prefill isolation inventory is incomplete"
    done
    for arm in compact compact-direct; do
        cmp -s \
            "$OUTPUT_DIR/exact/f32-prefill-isolation-logits/frontier_000512.logits.f32" \
            "$OUTPUT_DIR/exact/$arm-prefill-isolation-logits/frontier_000512.logits.f32" ||
            die "$arm diverged before historical compact rows were consumed"
    done
    hybrid_exact=false
    direct_exact=false
    arms_equal=false
    cmp -s \
        "$OUTPUT_DIR/exact/f32-prefill-isolation-logits/frontier_004096.logits.f32" \
        "$OUTPUT_DIR/exact/compact-prefill-isolation-logits/frontier_004096.logits.f32" &&
        hybrid_exact=true
    cmp -s \
        "$OUTPUT_DIR/exact/f32-prefill-isolation-logits/frontier_004096.logits.f32" \
        "$OUTPUT_DIR/exact/compact-direct-prefill-isolation-logits/frontier_004096.logits.f32" &&
        direct_exact=true
    cmp -s \
        "$OUTPUT_DIR/exact/compact-prefill-isolation-logits/frontier_004096.logits.f32" \
        "$OUTPUT_DIR/exact/compact-direct-prefill-isolation-logits/frontier_004096.logits.f32" &&
        arms_equal=true
    {
        printf 'mode=pp4096-prefill-three-arm\n'
        printf 'pack_roundtrip_audit=passed\n'
        printf 'hybrid_vs_f32_bit_exact=%s\n' "$hybrid_exact"
        printf 'direct_vs_f32_bit_exact=%s\n' "$direct_exact"
        printf 'hybrid_vs_direct_bit_exact=%s\n' "$arms_equal"
        printf 'acceptance_evidence=no\n'
    } | tee "$OUTPUT_DIR/summary/prefill-isolation.txt"
    phase=finished
    printf 'Compact-KV PP4096 prefill isolation complete: %s\n' "$OUTPUT_DIR"
    exit 0
fi

if [[ $DIAGNOSTIC_DECODE_ISOLATION == 1 ]]; then
    phase=decode-isolation
    capture_gpu_health "$OUTPUT_DIR/initial-gpu.csv" ||
        die "could not capture initial four-GPU health"
    printf 'Compact-KV decode isolation arm=nosnapshot...\n'
    nosnapshot_rc=0
    run_decode_isolation_case nosnapshot || nosnapshot_rc=$?
    if ! capture_gpu_health "$OUTPUT_DIR/between-isolation-arms-gpu.csv"; then
        die "GPU health was lost after compact nosnapshot isolation arm"
    fi
    printf 'Compact-KV decode isolation arm=snapshot...\n'
    snapshot_rc=0
    run_decode_isolation_case snapshot || snapshot_rc=$?
    printf 'mode=compact-pp512-first-token\nnosnapshot_exit_status=%s\nsnapshot_exit_status=%s\nacceptance_evidence=no\n' \
        "$nosnapshot_rc" "$snapshot_rc" \
        >"$OUTPUT_DIR/summary/decode-isolation.txt"
    if [[ $nosnapshot_rc != 0 || $snapshot_rc != 0 ]]; then
        die "compact PP512 decode isolation reproduced a failure"
    fi
    phase=finished
    printf 'Compact-KV PP512 decode isolation completed without failure: %s\n' \
        "$OUTPUT_DIR"
    exit 0
fi

if [[ $DIAGNOSTIC_DECODE_PROFILE == 1 ]]; then
    phase=decode-profile
    capture_gpu_health "$OUTPUT_DIR/initial-gpu.csv" ||
        die "could not capture initial four-GPU health"
    for arm in f32 compact; do
        printf 'Compact-KV PP512 one-token Nsight Systems profile arm=%s...\n' \
            "$arm"
        run_decode_profile_case "$arm" || {
            tail -n 240 "$OUTPUT_DIR/nsys/$arm-pp512-decode.log" >&2 || true
            die "$arm PP512 decode profile failed"
        }
    done
    phase=finished
    printf 'Compact-KV PP512 decode profiles complete: %s\n' "$OUTPUT_DIR"
    exit 0
fi

phase=production
capture_gpu_health "$OUTPUT_DIR/initial-gpu.csv" ||
    die "could not capture initial four-GPU health"
for arm in f32 compact; do
    base="$OUTPUT_DIR/runs/$arm"
    printf 'Compact-KV production performance model=all43 arm=%s...\n' "$arm"
    run_case "$arm" performance "$TG_TOKENS" "$base" || {
        tail -n 240 "$base.log" >&2 || true
        die "$arm production performance run failed validation"
    }
done

phase=exact
for arm in f32 compact; do
    base="$OUTPUT_DIR/exact/$arm"
    logits="$base-logits"
    mkdir -p "$logits"
    printf 'Compact-KV exact prefill/decode model=all43 arm=%s...\n' "$arm"
    run_case "$arm" exact "$EXACT_TOKENS" "$base" "$logits" || {
        tail -n 240 "$base.log" >&2 || true
        die "$arm exact prefill/decode run failed validation"
    }
    validate_exact_inventory "$logits" ||
        die "$arm exact logit inventory is incomplete"
done

expected="$OUTPUT_DIR/exact/expected-files.txt"
find "$OUTPUT_DIR/exact/f32-logits" -maxdepth 1 -type f \
    -name '*.f32' -printf '%f\n' | sort >"$expected"
find "$OUTPUT_DIR/exact/compact-logits" -maxdepth 1 -type f \
    -name '*.f32' -printf '%f\n' | sort >"$OUTPUT_DIR/exact/compact-files.txt"
cmp -s "$expected" "$OUTPUT_DIR/exact/compact-files.txt" ||
    die "F32 and compact exact-output inventories differ"
while IFS= read -r file; do
    cmp -s "$OUTPUT_DIR/exact/f32-logits/$file" \
           "$OUTPUT_DIR/exact/compact-logits/$file" ||
        die "compact cache diverged at $file"
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
for arm in f32 compact; do
    summarize_telemetry "$OUTPUT_DIR/telemetry/$arm-performance.csv" \
        "$OUTPUT_DIR/summary/$arm-vram.csv"
done
f32_peak=$(awk -F, '$1=="aggregate" {print $3}' \
    "$OUTPUT_DIR/summary/f32-vram.csv")
compact_peak=$(awk -F, '$1=="aggregate" {print $3}' \
    "$OUTPUT_DIR/summary/compact-vram.csv")
[[ -n $f32_peak && -n $compact_peak ]] || die "peak VRAM summary is incomplete"
vram_saving=$(awk -v control="$f32_peak" -v candidate="$compact_peak" \
    'BEGIN {printf "%.0f", control-candidate}')
(( vram_saving >= MIN_COMPACT_VRAM_SAVING_MIB )) ||
    die "compact peak VRAM saving is ${vram_saving} MiB; required at least ${MIN_COMPACT_VRAM_SAVING_MIB} MiB"

awk -F, -v minimum="$MIN_THROUGHPUT_RATIO" '
    NR==FNR {if (FNR>1) {pp[$1]=$3; tg[$1]=$8}; next}
    FNR>1 {
        if (($3+0)/(pp[$1]+0)<minimum || ($8+0)/(tg[$1]+0)<minimum) bad=1
    }
    END {exit bad}
' "$OUTPUT_DIR/runs/f32.csv" "$OUTPUT_DIR/runs/compact.csv" ||
    die "compact cache regressed a prefill or decode frontier below the accepted throughput ratio"

{
    printf '# SM75 exact compact-attention KV production qualification\n\n'
    printf 'Exactness: all prefill frontier logits and all %s decode logits at PP512, PP4096, and PP32768 are byte-identical.  \n' "$EXACT_TOKENS"
    printf '256K allocation peak aggregate VRAM: F32 %.0f MiB; compact %.0f MiB; saving %.0f MiB.  \n\n' \
        "$f32_peak" "$compact_peak" "$vram_saving"
    printf '| Context | F32 prefill tok/s | Compact prefill tok/s | Prefill ratio | F32 decode tok/s | Compact decode tok/s | Decode ratio |\n'
    printf '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    awk -F, '
        NR==FNR {if (FNR>1) {pp[$1]=$3; tg[$1]=$8}; next}
        FNR>1 {printf "| %s | %.3f | %.3f | %.6fx | %.3f | %.3f | %.6fx |\n",$1,pp[$1],$3,$3/pp[$1],tg[$1],$8,$8/tg[$1]}
    ' "$OUTPUT_DIR/runs/f32.csv" "$OUTPUT_DIR/runs/compact.csv"
    printf '\nThe model, ordinary single-owner weight residency, and stable pair policy are identical in both arms.\n'
} | tee "$OUTPUT_DIR/summary/report.md"

printf 'bit_exact=true\nmodel_layout=all43\nfrontiers=512,4096,32768\n' \
    >"$OUTPUT_DIR/summary/acceptance.txt"
printf 'context_allocation=%s\ndecode_tokens_per_frontier=%s\n' \
    "$CTX_ALLOC" "$EXACT_TOKENS" >>"$OUTPUT_DIR/summary/acceptance.txt"
printf 'f32_peak_vram_mib=%s\ncompact_peak_vram_mib=%s\ncompact_peak_vram_saving_mib=%s\n' \
    "$f32_peak" "$compact_peak" "$vram_saving" \
    >>"$OUTPUT_DIR/summary/acceptance.txt"
printf 'minimum_throughput_ratio=%s\ncompact_cache_default=false\n' \
    "$MIN_THROUGHPUT_RATIO" >>"$OUTPUT_DIR/summary/acceptance.txt"

phase=complete
printf 'SM75 exact compact-attention KV all43 production A/B complete: %s\n' \
    "$OUTPUT_DIR"
