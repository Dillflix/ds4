#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the known production 32K failure reproducer with physical GPU0 in a home
role. CUDA Graph execution is explicitly disabled. The default is the first
TG16 case only because the observed failure occurs during prefill, before
decode. Flushed, non-synchronizing pipeline breadcrumbs identify the last
completed cross-stage handoff.

Required environment:
  MODEL=/absolute/path/to/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf

Optional environment:
  PROMPT=...                    default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  PP_TOKENS=32768
  TG_LEVELS=16
  REPEATS=1
  TELEMETRY_INTERVAL_MS=500
  POST_CASE_SETTLE_SECONDS=5
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  DECODE_CRASH_ISOLATION_DIR=/absolute/output/directory

If a host reset prevents archive creation, retain the unarchived output
directory. fault-breadcrumbs.log records the last pipeline boundary reached;
active-case.txt, run-journal.tsv, telemetry/, and health/ retain supporting
evidence.
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
PP_TOKENS=${PP_TOKENS:-32768}
TG_LEVELS=${TG_LEVELS:-16}
REPEATS=${REPEATS:-1}
TELEMETRY_INTERVAL_MS=${TELEMETRY_INTERVAL_MS:-500}
POST_CASE_SETTLE_SECONDS=${POST_CASE_SETTLE_SECONDS:-5}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${DECODE_CRASH_ISOLATION_DIR:-$repo_dir/sm75-decode-crash-isolation-$stamp}

[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name the existing absolute tagged SM75 model"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "this isolation requires the fixed promessi_sposi prompt"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "require GPU_DEVICES=0,3,1,2, GPU_VRAM=auto, STAGE_SPLIT=22"
for item in "PP_TOKENS:$PP_TOKENS" "REPEATS:$REPEATS" \
            "TELEMETRY_INTERVAL_MS:$TELEMETRY_INTERVAL_MS" \
            "POST_CASE_SETTLE_SECONDS:$POST_CASE_SETTLE_SECONDS" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( PP_TOKENS == 32768 && REPEATS >= 1 &&
   TELEMETRY_INTERVAL_MS >= 100 && POST_CASE_SETTLE_SECONDS <= 60 )) ||
    die "invalid production decode-isolation configuration"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] ||
        die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical IDs remain stable"

IFS=, read -r -a tg_levels <<<"$TG_LEVELS"
(( ${#tg_levels[@]} >= 1 )) || die "TG_LEVELS selected no cases"
previous=0
for tg in "${tg_levels[@]}"; do
    [[ $tg =~ ^[0-9]+$ ]] || die "invalid TG level: $tg"
    (( tg > previous && tg <= 256 )) ||
        die "TG_LEVELS must be unique, strictly increasing, and no greater than 256"
    previous=$tg
done

for tool in awk basename cat cmp cp date dirname env find git grep kill make \
            mkdir mv nproc nvidia-smi python3 rm sleep sort stat sync tail tar \
            tee timeout tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
git diff --quiet && git diff --cached --quiet ||
    die "tracked source changes are present; test an exact committed tree"
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{health,production,provenance,telemetry}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
printf 'Diagnostic directory: %s\n' "$OUTPUT_DIR"

phase=initialization
telemetry_pid=
telemetry_watch_pid=
active_case_pid=
stop_active_case() {
    if [[ -n ${active_case_pid:-} ]]; then
        local pid=$active_case_pid
        active_case_pid=
        kill -TERM "$pid" >/dev/null 2>&1 || true
        for _ in {1..20}; do
            kill -0 "$pid" >/dev/null 2>&1 || return 0
            sleep 0.1
        done
        kill -KILL "$pid" >/dev/null 2>&1 || true
    fi
}
stop_telemetry() {
    if [[ -n ${telemetry_watch_pid:-} ]]; then
        local watch_pid=$telemetry_watch_pid
        telemetry_watch_pid=
        kill "$watch_pid" >/dev/null 2>&1 || true
    fi
    if [[ -n ${telemetry_pid:-} ]]; then
        local pid=$telemetry_pid
        telemetry_pid=
        kill "$pid" >/dev/null 2>&1 || true
        for _ in {1..20}; do
            kill -0 "$pid" >/dev/null 2>&1 || return 0
            sleep 0.05
        done
        kill -9 "$pid" >/dev/null 2>&1 || true
    fi
}
finish() {
    status=$?
    trap - EXIT INT TERM HUP
    stop_active_case
    stop_telemetry
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    breadcrumb_all="$OUTPUT_DIR/fault-breadcrumbs.all.log"
    breadcrumb_tail="$OUTPUT_DIR/fault-breadcrumbs.log"
    : >"$breadcrumb_all"
    shopt -s nullglob
    for log in "$OUTPUT_DIR"/production/*.log; do
        grep -H -F 'ds4: prefill fault breadcrumb event=' "$log" \
            >>"$breadcrumb_all" || true
    done
    shopt -u nullglob
    if [[ -s $breadcrumb_all ]]; then
        tail -n 160 "$breadcrumb_all" >"$breadcrumb_tail"
    else
        rm -f -- "$breadcrumb_all" "$breadcrumb_tail"
    fi
    sync "$OUTPUT_DIR/run-status.txt" 2>/dev/null || sync
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        archive="$OUTPUT_DIR.tar.gz"
        partial="$archive.partial.$$"
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && [[ -s $partial ]] &&
                tar -tzf "$partial" >/dev/null && mv -- "$partial" "$archive"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
            rm -f -- "$partial"
            printf 'error: failed to create archive %s\n' "$archive" >&2
        fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM HUP

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
(( ${#gpu_ids[@]} == 4 )) || die "GPU_DEVICES must contain exactly four IDs"
declare -A seen_gpu=()
for gpu in "${gpu_ids[@]}"; do
    [[ $gpu =~ ^[0-9]+$ && -z ${seen_gpu[$gpu]:-} ]] ||
        die "invalid or duplicate physical GPU index: $gpu"
    seen_gpu[$gpu]=1
done

gpu_health_snapshot() {
    local path=$1
    {
        printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        timeout --kill-after=5s 20s nvidia-smi -L
        timeout --kill-after=5s 20s nvidia-smi --query-gpu=index,pci.bus_id,uuid,pstate,temperature.gpu,power.draw,power.limit,memory.used,memory.free,utilization.gpu,compute_cap \
            --format=csv
        printf '\ntopology:\n'
        timeout --kill-after=5s 20s nvidia-smi topo -m
    } >"$path" 2>&1 || return 1
    [[ $(grep -c '^GPU [0-9]:' "$path") == 4 ]] || return 1
    ! grep -Eiq 'ERR!|GPU is lost|Unknown Error|Unable to determine' "$path"
}

record_case() {
    local repeat=$1 slot=$2 tg=$3 status=$4
    local now
    now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%s\t%s\t%s\t%s\t%s\n' "$now" "$repeat" "$slot" "$tg" "$status" \
        >>"$OUTPUT_DIR/run-journal.tsv"
    printf 'timestamp_utc=%s\nrepeat=%s\nslot=%s\npp_tokens=%s\ntg_tokens=%s\ngraph=disabled\nstatus=%s\n' \
        "$now" "$repeat" "$slot" "$PP_TOKENS" "$tg" "$status" \
        >"$OUTPUT_DIR/active-case.txt"
    sync "$OUTPUT_DIR/run-journal.tsv" "$OUTPUT_DIR/active-case.txt" \
        2>/dev/null || sync
}

start_telemetry() {
    local output=$1
    stdbuf -oL -eL nvidia-smi --query-gpu=timestamp,index,pci.bus_id,pstate,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.free \
        --format=csv -lms "$TELEMETRY_INTERVAL_MS" >"$output" 2>&1 &
    telemetry_pid=$!
}

start_telemetry_watch() {
    local output=$1 marker=$2 case_pid=$3
    (
        while kill -0 "$telemetry_pid" >/dev/null 2>&1; do
            if tail -n 16 "$output" 2>/dev/null |
                    grep -Eiq 'GPU is lost|GPU requires reset|Unknown Error|ERR!|Unable to determine'; then
                printf 'telemetry-watch: lost-device response detected; stopping nvidia-smi loop\n' \
                    >>"$output"
                printf 'timestamp_utc=%s\nstatus=lost-device-detected\ncase_pid=%s\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$case_pid" >"$marker"
                sync "$marker" 2>/dev/null || sync
                kill "$telemetry_pid" >/dev/null 2>&1 || true
                kill -TERM "$case_pid" >/dev/null 2>&1 || true
                for _ in {1..20}; do
                    kill -0 "$case_pid" >/dev/null 2>&1 || exit 0
                    sleep 0.1
                done
                kill -KILL "$case_pid" >/dev/null 2>&1 || true
                exit 0
            fi
            sleep 0.1
        done
    ) &
    telemetry_watch_pid=$!
}

phase=build
targets=(ds4-bench tests/cuda_long_context_smoke)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log"
    "${clean[@]}" ./tests/cuda_long_context_smoke \
        >"$OUTPUT_DIR/smoke.log" 2>&1 || {
            tail -n 200 "$OUTPUT_DIR/smoke.log" >&2
            die "CUDA long-context smoke failed"
        }
else
    make -q "${targets[@]}" CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi

phase=manifest
gpu_health_snapshot "$OUTPUT_DIR/health/initial.log" ||
    die "initial four-GPU health check failed"
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nmodel_hashing=disabled\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\ngpu_vram=%s\nstage_split=%s/%s\n' \
        "$GPU_DEVICES" "$GPU_VRAM" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))"
    printf 'pp_tokens=%s\ntg_levels=%s\nrepeats=%s\n' \
        "$PP_TOKENS" "$TG_LEVELS" "$REPEATS"
    printf 'decode_graph=explicitly-disabled\nattention_rows=production-enabled\n'
    printf 'prefill_fault_breadcrumbs=enabled-no-added-cuda-sync\n'
    printf 'dense_placement=stage-aware-fixed-22-21\n'
    printf 'partner_classes=automatic:t32,t256,shared_down\n'
    printf 'telemetry_interval_ms=%s\npost_case_settle_seconds=%s\n' \
        "$TELEMETRY_INTERVAL_MS" "$POST_CASE_SETTLE_SECONDS"
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
printf 'timestamp_utc\trepeat\tslot\ttg_tokens\tstatus\n' \
    >"$OUTPUT_DIR/run-journal.tsv"
printf 'repeat\tslot\tpp_tokens\ttg_tokens\tprefill_tps\tgen_tps\tfirst_ms\tsteady_tps\tsteady_ms_per_token\tcsv\tlog\ttelemetry\tprogress\tplan\tbindings\n' \
    >"$OUTPUT_DIR/production/runs.tsv"

validate_log() {
    local log=$1 tg=$2 line calls
    for marker in 'CUDA EP forced pipeline split 22/21' \
                  'dense-placement=stage-aware-fixed-22-21' \
                  'CUDA q8 fp16 benefit plan materialized 344/344 candidates' \
                  'SM75 routed Q32 layout enabled' \
                  'SM75 native F16 indexer cache and streaming WMMA64 enabled'; do
        grep -Fq "$marker" "$log" || return 1
    done
    [[ $(grep -Fc 'CUDA prefill attention query-row split enabled:' "$log") == 2 ]] ||
        return 1
    ! grep -Fq 'SM75 Q32 owned decode CUDA Graph enabled' "$log" || return 1
    ! grep -Fq 'SM75 Q32 decode graph audit' "$log" || return 1
    ! grep -Fq 'required but unavailable' "$log" || return 1
    ! grep -Eiq 'illegal memory|GPU is lost|Unknown Error|CUDA .* failed' "$log" ||
        return 1
    line=$(grep -F 'CUDA q8 fp16 partner summary:' "$log" | tail -n 1 || true)
    [[ $line =~ partner[[:space:]]summary:[[:space:]]calls=([0-9]+) ]] || return 1
    calls=${BASH_REMATCH[1]}
    (( calls > tg ))
}

validate_csv() {
    local csv=$1 tg=$2
    awk -F, -v pp="$PP_TOKENS" -v tg="$tg" '
        NR==1 {header=($1=="ctx_tokens" && $3=="prefill_tps" &&
                       $4=="gen_tokens" && $8=="gen_steady_tps"); next}
        NR==2 {rows++; good=($1==pp && $4==tg && ($3+0)>0 && ($5+0)>0 &&
                             ($8+0)>0); next}
        NR>2 {rows++}
        END {exit !(header && rows==1 && good)}
    ' "$csv"
}

phase=production-decode-control
total_cases=$((REPEATS * ${#tg_levels[@]}))
slot=0
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    for tg in "${tg_levels[@]}"; do
        slot=$((slot + 1))
        tag="r${repeat}-s${slot}-pp${PP_TOKENS}-tg${tg}-control"
        base="$OUTPUT_DIR/production/$tag"
        telemetry="$OUTPUT_DIR/telemetry/$tag.csv"
        lost_marker="$OUTPUT_DIR/telemetry/$tag-lost-device.txt"
        progress="$base-progress.csv"
        plan="$base-plan.csv"
        bindings="$base-bindings.csv"
        pre_health="$OUTPUT_DIR/health/$tag-pre.log"
        post_health="$OUTPUT_DIR/health/$tag-post.log"
        ctx_alloc=$((PP_TOKENS + tg + 1))

        printf 'Decode control repeat=%d/%d slot=%d/%d PP=%d TG=%d...\n' \
            "$repeat" "$REPEATS" "$slot" "$total_cases" "$PP_TOKENS" "$tg"
        record_case "$repeat" "$slot" "$tg" checking-pre-health
        gpu_health_snapshot "$pre_health" || {
            record_case "$repeat" "$slot" "$tg" failed-pre-health
            die "GPU health check failed before TG$tg"
        }
        record_case "$repeat" "$slot" "$tg" starting
        start_telemetry "$telemetry"
        "${clean[@]}" \
                "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
                DS4_CUDA_PREFILL_PIPELINE=1 \
                DS4_CUDA_PREFILL_PIPELINE_MB=512 \
                DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
                DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=2048 \
                DS4_CUDA_TP_PREFILL_ATTN_ROWS=1 \
                DS4_CUDA_TP_ATTN_HEADS=0 \
                DS4_CUDA_TP_DECODE_INDEXER_ROWS=0 \
                DS4_CUDA_NO_MOE_Q32_DECODE_GRAPH=1 \
                DS4_CUDA_PREFILL_FAULT_BREADCRUMBS=1 \
                DS4_BENCH_UNTIMED_WARMUP_TOKENS=512 \
                "DS4_BENCH_PROGRESS_JOURNAL=$progress" \
                "DS4_CUDA_Q8_PLAN_AUDIT_CSV=$plan" \
                "DS4_CUDA_Q8_BINDING_STATE_CSV=$bindings" \
                ./ds4-bench --cuda --cuda-tensor-parallel \
                    --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                    --model "$MODEL" --prompt-file "$PROMPT" \
                    --ctx-start "$PP_TOKENS" --ctx-max "$PP_TOKENS" \
                    --ctx-alloc "$ctx_alloc" --step-incr "$PP_TOKENS" \
                    --prefill-chunk 2048 --gen-tokens "$tg" \
                    --csv "$base.csv" >"$base.log" 2>&1 &
        active_case_pid=$!
        start_telemetry_watch "$telemetry" "$lost_marker" "$active_case_pid"
        if wait "$active_case_pid"; then
            run_status=0
        else
            run_status=$?
        fi
        active_case_pid=
        stop_telemetry
        if (( run_status != 0 )); then
            record_case "$repeat" "$slot" "$tg" "failed-exit-$run_status"
            tail -n 240 "$base.log" >&2 || true
            die "TG$tg control failed with exit $run_status"
        fi
        validate_csv "$base.csv" "$tg" || {
            record_case "$repeat" "$slot" "$tg" failed-csv-validation
            die "TG$tg produced an invalid benchmark CSV"
        }
        validate_log "$base.log" "$tg" || {
            record_case "$repeat" "$slot" "$tg" failed-path-validation
            tail -n 240 "$base.log" >&2 || true
            die "TG$tg did not preserve the production graph-disabled path"
        }
        [[ -s $plan && $(wc -l <"$plan") == 345 ]] ||
            die "TG$tg omitted its 344-entry placement plan"
        [[ -s $bindings && $(wc -l <"$bindings") == 345 ]] ||
            die "TG$tg omitted its 344-entry binding inventory"
        [[ -s $telemetry && $(grep -c . "$telemetry") -ge 2 ]] ||
            die "TG$tg omitted usable telemetry"
        [[ -s $progress ]] || die "TG$tg omitted its progress journal"
        record_case "$repeat" "$slot" "$tg" settling
        sleep "$POST_CASE_SETTLE_SECONDS"
        gpu_health_snapshot "$post_health" || {
            record_case "$repeat" "$slot" "$tg" failed-post-health
            die "GPU health check failed after TG$tg"
        }

        IFS=, read -r ctx prefill_tokens prefill_tps gen_tokens gen_tps \
            first_ms steady_tokens steady_tps kv_bytes < <(tail -n 1 "$base.csv")
        steady_ms=$(awk -v tps="$steady_tps" 'BEGIN {printf "%.6f", 1000/tps}')
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$repeat" "$slot" "$PP_TOKENS" "$tg" "$prefill_tps" \
            "$gen_tps" "$first_ms" "$steady_tps" "$steady_ms" \
            "$base.csv" "$base.log" "$telemetry" "$progress" "$plan" "$bindings" \
            >>"$OUTPUT_DIR/production/runs.tsv"
        record_case "$repeat" "$slot" "$tg" completed
    done
done

python3 speed-bench/summarize-sm75-decode-crash-isolation.py "$OUTPUT_DIR"
cat "$OUTPUT_DIR/production/summary.csv"
phase=complete
printf 'SM75 32K decode control isolation complete: %s\n' "$OUTPUT_DIR"
