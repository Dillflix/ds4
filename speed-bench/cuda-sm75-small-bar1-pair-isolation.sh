#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Isolate the mixed/small-BAR1 NVLink pair without changing the other pair.
Each arm preserves partner-resident weights and partner projection work in the
current 22/21 four-GPU production path at PP32768/TG256.

Required environment:
  MODEL=/absolute/path/to/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf

Optional environment:
  PROMPT=...                         default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  SMALL_BAR1_PAIR=0                 logical home tier; physical 0<->1 here
  VARIANTS=attention-off,production
  PP_TOKENS=32768
  TG_TOKENS=256
  REPEATS=1
  REQUIRED_POWER_LIMIT_W=250
  TELEMETRY_INTERVAL_MS=500
  POST_CASE_SETTLE_SECONDS=5
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  RESUME=0
  SMALL_BAR1_ISOLATION_DIR=/absolute/output/directory

The transport/scheduling diagnostic arms run before the known full-production
reproducer. They preserve arithmetic work but can change its timing envelope.
If a GPU loss interrupts the shell, reboot, set RESUME=1 and reuse the printed
SMALL_BAR1_ISOLATION_DIR. The incomplete arm is retained as failed evidence
and the next arm runs; it is not silently retried.
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
SMALL_BAR1_PAIR=${SMALL_BAR1_PAIR:-0}
VARIANTS=${VARIANTS:-attention-off,production}
PP_TOKENS=${PP_TOKENS:-32768}
TG_TOKENS=${TG_TOKENS:-256}
REPEATS=${REPEATS:-1}
REQUIRED_POWER_LIMIT_W=${REQUIRED_POWER_LIMIT_W:-250}
TELEMETRY_INTERVAL_MS=${TELEMETRY_INTERVAL_MS:-500}
POST_CASE_SETTLE_SECONDS=${POST_CASE_SETTLE_SECONDS:-5}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
RESUME=${RESUME:-0}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${SMALL_BAR1_ISOLATION_DIR:-$repo_dir/sm75-small-bar1-pair-isolation-$stamp}

[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name the existing absolute tagged SM75 model"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "this isolation requires the fixed promessi_sposi prompt"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "require GPU_DEVICES=0,3,1,2, GPU_VRAM=auto, STAGE_SPLIT=22"
for item in "SMALL_BAR1_PAIR:$SMALL_BAR1_PAIR" "PP_TOKENS:$PP_TOKENS" \
            "TG_TOKENS:$TG_TOKENS" "REPEATS:$REPEATS" \
            "REQUIRED_POWER_LIMIT_W:$REQUIRED_POWER_LIMIT_W" \
            "TELEMETRY_INTERVAL_MS:$TELEMETRY_INTERVAL_MS" \
            "POST_CASE_SETTLE_SECONDS:$POST_CASE_SETTLE_SECONDS" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE" \
            "RESUME:$RESUME"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( SMALL_BAR1_PAIR == 0 && PP_TOKENS == 32768 && TG_TOKENS == 256 &&
   REPEATS >= 1 && REQUIRED_POWER_LIMIT_W == 250 &&
   TELEMETRY_INTERVAL_MS >= 100 &&
   POST_CASE_SETTLE_SECONDS <= 60 )) ||
    die "invalid fixed production isolation configuration"
for flag in SKIP_BUILD CREATE_ARCHIVE RESUME; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] ||
        die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical IDs remain stable"

IFS=, read -r -a variants <<<"$VARIANTS"
(( ${#variants[@]} >= 1 )) || die "VARIANTS selected no arms"
declare -A seen_variants=()
for variant in "${variants[@]}"; do
    case "$variant" in
        attention-off|partner-bounce|bounce-indexer-off|partner-serialized|indexer-off|production) ;;
        *) die "unknown variant: $variant" ;;
    esac
    [[ -z ${seen_variants[$variant]:-} ]] || die "duplicate variant: $variant"
    seen_variants[$variant]=1
done

for tool in awk basename cat cmp cp date dirname env find git grep kill make \
            mkdir mv nproc nvidia-smi python3 rm sleep sort stat stdbuf sync \
            tail tar tee timeout tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
git diff --quiet && git diff --cached --quiet ||
    die "tracked source changes are present; test an exact committed tree"

if [[ $RESUME == 0 ]]; then
    [[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"/{health,nvlink,production,provenance,telemetry}
else
    [[ -d $OUTPUT_DIR ]] || die "resume directory not found: $OUTPUT_DIR"
    for subdir in health nvlink production provenance telemetry; do
        mkdir -p "$OUTPUT_DIR/$subdir"
    done
fi
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
printf 'Diagnostic directory: %s\n' "$OUTPUT_DIR"
printf 'Resume with: export SMALL_BAR1_ISOLATION_DIR=%q; RESUME=1 ...\n' \
    "$OUTPUT_DIR"
if [[ $RESUME == 1 && -s $OUTPUT_DIR/manifest.txt ]]; then
    grep -Fxq "git_commit=$(git rev-parse HEAD)" "$OUTPUT_DIR/manifest.txt" ||
        die "resume commit differs from the original isolation"
    grep -Fxq "model=$MODEL" "$OUTPUT_DIR/manifest.txt" ||
        die "resume model differs from the original isolation"
    grep -Fxq "variants=$VARIANTS" "$OUTPUT_DIR/manifest.txt" ||
        die "resume variant order differs from the original isolation"
fi

phase=initialization
telemetry_pid=
telemetry_watch_pid=
nvlink_pid=
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
        kill "$telemetry_watch_pid" >/dev/null 2>&1 || true
        telemetry_watch_pid=
    fi
    for name in telemetry_pid nvlink_pid; do
        local pid=${!name:-}
        [[ -n $pid ]] || continue
        printf -v "$name" '%s' ''
        kill "$pid" >/dev/null 2>&1 || true
        for _ in {1..20}; do
            kill -0 "$pid" >/dev/null 2>&1 || break
            sleep 0.05
        done
        kill -KILL "$pid" >/dev/null 2>&1 || true
    done
}
write_summary() {
    python3 speed-bench/summarize-sm75-small-bar1-pair-isolation.py \
        "$OUTPUT_DIR" >/dev/null 2>&1 || true
}
finish() {
    status=$?
    trap - EXIT INT TERM HUP
    stop_active_case
    stop_telemetry
    write_summary
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    sync "$OUTPUT_DIR/run-status.txt" 2>/dev/null || sync
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        archive="$OUTPUT_DIR.tar.gz"
        partial="$archive.partial.$$"
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && [[ -s $partial ]] &&
                tar -tzf "$partial" >/dev/null && mv -f -- "$partial" "$archive"; then
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
(( ${#gpu_ids[@]} == 4 )) || die "GPU_DEVICES must contain four IDs"

validate_power_limits() {
    local gpu limit
    for gpu in "${gpu_ids[@]}"; do
        limit=$(timeout --kill-after=5s 20s nvidia-smi -i "$gpu" \
            --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null |
            awk 'NR==1 {print $1}') || return 1
        awk -v actual="$limit" -v required="$REQUIRED_POWER_LIMIT_W" \
            'BEGIN {exit !(actual + 0.0 == required + 0.0)}' || return 1
    done
}

gpu_health_snapshot() {
    local path=$1
    {
        printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        timeout --kill-after=5s 20s nvidia-smi -L
        timeout --kill-after=5s 20s nvidia-smi \
            --query-gpu=index,pci.bus_id,uuid,serial,pstate,temperature.gpu,power.draw,power.limit,memory.used,memory.free,utilization.gpu \
            --format=csv
        printf '\ntopology:\n'
        timeout --kill-after=5s 20s nvidia-smi topo -m
    } >"$path" 2>&1 || return 1
    [[ $(grep -c '^GPU [0-9]:' "$path") == 4 ]] || return 1
    ! grep -Eiq 'ERR!|GPU is lost|Unknown Error|Unable to determine' "$path"
}

start_telemetry() {
    local output=$1
    stdbuf -oL -eL nvidia-smi \
        --query-gpu=timestamp,index,pci.bus_id,pstate,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.free \
        --format=csv -lms "$TELEMETRY_INTERVAL_MS" >"$output" 2>&1 &
    telemetry_pid=$!
}

start_nvlink_telemetry() {
    local output=$1 interval_sec
    interval_sec=$(awk -v ms="$TELEMETRY_INTERVAL_MS" 'BEGIN {printf "%.3f", ms/1000}')
    (
        while :; do
            printf 'snapshot_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
            for gpu in "${gpu_ids[@]}"; do
                printf 'gpu=%s counter=0\n' "$gpu"
                timeout --kill-after=2s 5s nvidia-smi nvlink -i "$gpu" -g 0 \
                    2>&1 || printf 'counter_status=unsupported-or-unavailable\n'
                printf 'gpu=%s counter=1\n' "$gpu"
                timeout --kill-after=2s 5s nvidia-smi nvlink -i "$gpu" -g 1 \
                    2>&1 || printf 'counter_status=unsupported-or-unavailable\n'
            done
            sleep "$interval_sec"
        done
    ) >"$output" 2>&1 &
    nvlink_pid=$!
}

start_telemetry_watch() {
    local output=$1 marker=$2 case_pid=$3
    (
        while kill -0 "$telemetry_pid" >/dev/null 2>&1; do
            local watch_status=
            if tail -n 24 "$output" 2>/dev/null |
                    grep -Eiq 'GPU is lost|GPU requires reset|Unknown Error|ERR!|Unable to determine'; then
                watch_status=lost-device-detected
            elif tail -n 24 "$output" 2>/dev/null |
                    awk -F, -v required="$REQUIRED_POWER_LIMIT_W" '
                        {
                            index_field=$2
                            limit=$7
                            gsub(/^[[:space:]]+|[[:space:]]+$/, "", index_field)
                            gsub(/^[[:space:]]+|[[:space:]]+$/, "", limit)
                            sub(/[[:space:]]+W$/, "", limit)
                            if (index_field ~ /^[0-9]+$/ &&
                                limit ~ /^[0-9]+([.][0-9]+)?$/ &&
                                limit + 0.0 != required + 0.0) found=1
                        }
                        END {exit !found}
                    '; then
                watch_status=power-limit-drift
            fi
            if [[ -n $watch_status ]]; then
                printf 'telemetry-watch: %s detected\n' "$watch_status" >>"$output"
                printf 'timestamp_utc=%s\nstatus=%s\ncase_pid=%s\nrequired_power_limit_w=%s\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$watch_status" \
                    "$case_pid" "$REQUIRED_POWER_LIMIT_W" >"$marker"
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

last_progress_fields() {
    local progress=$1
    if [[ -s $progress ]]; then
        awk -F, 'NR>1 {phase=$3; event=$4; current=$5; total=$6}
            END {printf "%s\t%s\t%s\t%s", phase, event, current, total}' \
            "$progress"
    else
        printf 'not-started\tno-progress-journal\t0\t0'
    fi
}

write_result() {
    local result=$1 variant=$2 status=$3 exit_status=$4 progress=$5 log=$6
    local fields last_phase last_event last_current last_total
    fields=$(last_progress_fields "$progress")
    IFS=$'\t' read -r last_phase last_event last_current last_total <<<"$fields"
    local q8_begin q8_complete row_begin row_complete
    q8_begin=$(grep -Fc 'q8 partner transfer audit event=begin' "$log" 2>/dev/null || true)
    q8_complete=$(grep -Fc 'q8 partner transfer audit event=complete' "$log" 2>/dev/null || true)
    row_begin=$(grep -Fc 'decode indexer row audit event=begin' "$log" 2>/dev/null || true)
    row_complete=$(grep -Fc 'decode indexer row audit event=complete' "$log" 2>/dev/null || true)
    {
        printf 'variant=%s\nstatus=%s\nexit_status=%s\n' \
            "$variant" "$status" "$exit_status"
        printf 'last_phase=%s\nlast_event=%s\nlast_current=%s\nlast_total=%s\n' \
            "$last_phase" "$last_event" "$last_current" "$last_total"
        printf 'q8_transfer_begin_checkpoints=%s\nq8_transfer_complete_checkpoints=%s\n' \
            "$q8_begin" "$q8_complete"
        printf 'indexer_row_begin_events=%s\nindexer_row_complete_events=%s\n' \
            "$row_begin" "$row_complete"
    } >"$result"
    sync "$result" 2>/dev/null || sync
}

phase=build
targets=(ds4-bench tests/cuda_long_context_smoke tests/test_engine_mgpu_placement)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log"
    "${clean[@]}" ./tests/test_engine_mgpu_placement \
        >"$OUTPUT_DIR/placement-tests.log" 2>&1 || {
            tail -n 200 "$OUTPUT_DIR/placement-tests.log" >&2
            die "CPU placement tests failed"
        }
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
validate_power_limits ||
    die "all four selected GPUs must be fixed at ${REQUIRED_POWER_LIMIT_W} W"
if [[ $RESUME == 0 || ! -s $OUTPUT_DIR/manifest.txt ]]; then
    {
        printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
            "$(git branch --show-current)"
        printf 'model=%s\nmodel_bytes=%s\nprompt=%s\n' \
            "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
        printf 'gpu_devices=%s\ngpu_vram=%s\nstage_split=%s/%s\n' \
            "$GPU_DEVICES" "$GPU_VRAM" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))"
        printf 'logical_pair_0=physical_0_physical_1\n'
        printf 'logical_pair_1=physical_3_physical_2\n'
        printf 'small_bar1_pair=%s\nvariants=%s\n' "$SMALL_BAR1_PAIR" "$VARIANTS"
        printf 'pp_tokens=%s\ntg_tokens=%s\nrepeats=%s\n' \
            "$PP_TOKENS" "$TG_TOKENS" "$REPEATS"
        printf 'required_power_limit_w=%s\n' "$REQUIRED_POWER_LIMIT_W"
        printf 'partner_work_retained=yes\ndecode_indexer_pair_fallback=home\n'
        printf 'host_bounce_scope=q8_partner_activation_and_result_only\n'
        printf 'serialization_scope=q8_partner_projection_pair_only\n'
        printf 'q8_transfer_audit=begin_complete_64-call-checkpoints\n'
        printf 'indexer_transfer_audit=every-dispatch-begin-complete\n'
        printf 'external_nvlink_counters=best-effort-nvidia-smi-counter0-counter1\n'
    } >"$OUTPUT_DIR/manifest.txt"
    git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
    git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
    nvidia-smi -q >"$OUTPUT_DIR/provenance/nvidia-smi-q.txt" 2>&1 || true
    nvidia-smi topo -m >"$OUTPUT_DIR/provenance/topology.txt" 2>&1 || true
    printf 'timestamp_utc\tvariant\trepeat\tstatus\texit_status\tlast_phase\tlast_event\n' \
        >"$OUTPUT_DIR/run-journal.tsv"
fi

validate_admission_retained() {
    local bindings=$1
    awk -F, '
        NR==1 {
            for (i=1;i<=NF;i++) h[$i]=i
            next
        }
        $(h["consumer_device"])==0 && $(h["resident_device"])==1 &&
        $(h["partner_offload"])==1 {found++}
        END {exit !(found>0)}
    ' "$bindings"
}

log_line_has() {
    local log=$1 first=$2 second=$3
    awk -v first="$first" -v second="$second" '
        index($0, first) && index($0, second) { found=1 }
        END { exit !found }
    ' "$log"
}

validate_success_path() {
    local variant=$1 log=$2 bindings=$3
    validate_admission_retained "$bindings" || return 1
    grep -Fq 'CUDA q8 fp16 benefit plan materialized 344/344 candidates' "$log" ||
        return 1
    grep -Fq 'CUDA EP forced pipeline split 22/21' "$log" || return 1
    grep -Fq 'SM75 Q3A4 decode gate/up mapping=tile32-dp4a-k4 (production default)' \
        "$log" || return 1
    ! grep -Eiq 'illegal memory|GPU is lost|Unknown Error|CUDA .* failed' "$log" ||
        return 1
    grep -Fq 'q8 partner transfer audit event=begin home_tier=0' "$log" ||
        return 1
    grep -Fq 'q8 partner transfer audit event=begin home_tier=1' "$log" ||
        return 1
    case "$variant" in
        attention-off)
            ! grep -Fq 'partner transport override for logical pair 0' "$log" ||
                return 1
            ! grep -Fq 'partner scheduling override for logical pair 0' "$log" ||
                return 1
            grep -Fq 'prefill attention row split pair-scoped disable: logical-pairs=0' \
                "$log" || return 1
            ! grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log" ||
                return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        production)
            ! grep -Fq 'partner transport override for logical pair 0' "$log" ||
                return 1
            ! grep -Fq 'partner scheduling override for logical pair 0' "$log" ||
                return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        partner-bounce)
            grep -Fq 'partner transport override for logical pair 0: pinned-host-bounce' \
                "$log" || return 1
            ! grep -Fq 'partner scheduling override for logical pair 0' "$log" ||
                return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            ;;
        bounce-indexer-off)
            grep -Fq 'partner transport override for logical pair 0: pinned-host-bounce' \
                "$log" || return 1
            grep -Fq 'decode indexer row split pair-scoped disable: logical-pairs=0' "$log" ||
                return 1
            if log_line_has "$log" 'decode indexer row audit event=begin' \
                    'home_tier=0 partner_tier=2'; then
                return 1
            fi
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        partner-serialized)
            grep -Fq 'partner scheduling override for logical pair 0: projection-serialized' \
                "$log" || return 1
            ! grep -Fq 'partner transport override for logical pair 0' "$log" ||
                return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            ;;
        indexer-off)
            ! grep -Fq 'partner transport override for logical pair 0' "$log" ||
                return 1
            grep -Fq 'decode indexer row split pair-scoped disable: logical-pairs=0' "$log" ||
                return 1
            if log_line_has "$log" 'decode indexer row audit event=begin' \
                    'home_tier=0 partner_tier=2'; then
                return 1
            fi
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
    esac
}

ctx_alloc=$((PP_TOKENS + TG_TOKENS + 1))
slot=0
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    for variant in "${variants[@]}"; do
        slot=$((slot + 1))
        tag="r${repeat}-s${slot}-${variant}"
        base="$OUTPUT_DIR/production/$tag"
        started="$base.started"
        result="$base.result"
        progress="$base-progress.csv"
        log="$base.log"
        csv="$base.csv"
        plan="$base-plan.csv"
        bindings="$base-bindings.csv"
        allocations="$base-allocations.csv"
        memory="$base-memory.csv"
        telemetry="$OUTPUT_DIR/telemetry/$tag.csv"
        nvlink="$OUTPUT_DIR/nvlink/$tag.log"
        watch_marker="$OUTPUT_DIR/telemetry/$tag-watch-event.txt"
        pre_health="$OUTPUT_DIR/health/$tag-pre.log"
        post_health="$OUTPUT_DIR/health/$tag-post.log"

        if [[ -s $result ]]; then
            printf 'Reusing completed evidence for variant=%s repeat=%d...\n' \
                "$variant" "$repeat"
            continue
        fi
        if [[ $RESUME == 1 && -s $started ]]; then
            if [[ -s $csv && $(wc -l <"$csv") == 2 && -s $bindings ]] &&
                    [[ $(wc -l <"$bindings") == 345 ]] &&
                    validate_success_path "$variant" "$log" "$bindings"; then
                printf 'Recovering completed prior arm: variant=%s repeat=%d...\n' \
                    "$variant" "$repeat"
                write_result "$result" "$variant" passed 0 "$progress" "$log"
                fields=$(last_progress_fields "$progress")
                IFS=$'\t' read -r lp le lc lt <<<"$fields"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                    recovered-passed 0 "$lp" "$le" \
                    >>"$OUTPUT_DIR/run-journal.tsv"
                sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
                continue
            fi
            printf 'Retaining interrupted prior arm as failure: variant=%s repeat=%d\n' \
                "$variant" "$repeat"
            write_result "$result" "$variant" interrupted-prior-run 125 \
                "$progress" "$log"
            fields=$(last_progress_fields "$progress")
            IFS=$'\t' read -r lp le lc lt <<<"$fields"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                interrupted-prior-run 125 "$lp" "$le" \
                >>"$OUTPUT_DIR/run-journal.tsv"
            continue
        fi

        phase="production-$variant"
        printf 'Pair isolation repeat=%d/%d variant=%s PP=%d TG=%d...\n' \
            "$repeat" "$REPEATS" "$variant" "$PP_TOKENS" "$TG_TOKENS"
        gpu_health_snapshot "$pre_health" ||
            die "GPU health failed before variant=$variant"
        validate_power_limits ||
            die "power limit changed before variant=$variant; require ${REQUIRED_POWER_LIMIT_W} W"
        printf 'timestamp_utc=%s\nvariant=%s\nrepeat=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" >"$started"
        sync "$started" 2>/dev/null || sync

        variant_env=()
        case "$variant" in
            attention-off)
                variant_env+=("DS4_CUDA_NO_TP_PREFILL_ATTN_ROWS_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            partner-bounce)
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            bounce-indexer-off)
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_NO_TP_DECODE_INDEXER_ROWS_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            partner-serialized)
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_SERIALIZE_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            indexer-off)
                variant_env+=("DS4_CUDA_NO_TP_DECODE_INDEXER_ROWS_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            production) ;;
        esac

        start_telemetry "$telemetry"
        start_nvlink_telemetry "$nvlink"
        "${clean[@]}" "${variant_env[@]}" \
                "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
                DS4_CUDA_PREFILL_PIPELINE=1 \
                DS4_CUDA_PREFILL_PIPELINE_MB=512 \
                DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
                DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=2048 \
                DS4_CUDA_TP_PREFILL_ATTN_ROWS=1 \
                DS4_CUDA_TP_PREFILL_ATTN_ROWS_AUDIT=1 \
                DS4_CUDA_NO_MOE_Q32_DECODE_GRAPH=1 \
                DS4_CUDA_PREFILL_FAULT_BREADCRUMBS=1 \
                DS4_CUDA_Q8_F16_PARTNER_TRANSFER_AUDIT=1 \
                DS4_CUDA_TP_DECODE_INDEXER_ROWS_AUDIT=1 \
                DS4_BENCH_UNTIMED_WARMUP_TOKENS=512 \
                "DS4_BENCH_PROGRESS_JOURNAL=$progress" \
                "DS4_CUDA_Q8_PLAN_AUDIT_CSV=$plan" \
                "DS4_CUDA_Q8_BINDING_STATE_CSV=$bindings" \
                "DS4_CUDA_Q8_ALLOCATION_STATE_CSV=$allocations" \
                "DS4_CUDA_MEMORY_STATE_CSV=$memory" \
                ./ds4-bench --cuda --cuda-tensor-parallel \
                    --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                    --model "$MODEL" --prompt-file "$PROMPT" \
                    --ctx-start "$PP_TOKENS" --ctx-max "$PP_TOKENS" \
                    --ctx-alloc "$ctx_alloc" --step-incr "$PP_TOKENS" \
                    --prefill-chunk 2048 --gen-tokens "$TG_TOKENS" \
                    --csv "$csv" >"$log" 2>&1 &
        active_case_pid=$!
        start_telemetry_watch "$telemetry" "$watch_marker" "$active_case_pid"
        if wait "$active_case_pid"; then run_status=0; else run_status=$?; fi
        active_case_pid=
        if (( run_status == 0 )) && ! validate_power_limits; then
            run_status=126
            if [[ ! -s $watch_marker ]]; then
                printf 'timestamp_utc=%s\nstatus=post-run-power-limit-drift\nrequired_power_limit_w=%s\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                    "$REQUIRED_POWER_LIMIT_W" >"$watch_marker"
                sync "$watch_marker" 2>/dev/null || sync
            fi
        fi
        stop_telemetry
        gpu_health_snapshot "$post_health" || true

        fields=$(last_progress_fields "$progress")
        IFS=$'\t' read -r lp le lc lt <<<"$fields"
        if (( run_status != 0 )); then
            write_result "$result" "$variant" failed "$run_status" "$progress" "$log"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                failed "$run_status" "$lp" "$le" >>"$OUTPUT_DIR/run-journal.tsv"
            sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
            tail -n 240 "$log" >&2 || true
            die "variant=$variant failed at phase=$lp event=$le"
        fi
        [[ -s $csv && $(wc -l <"$csv") == 2 ]] ||
            die "variant=$variant produced an invalid benchmark CSV"
        [[ -s $bindings ]] || die "variant=$variant omitted its binding inventory"
        if [[ $(wc -l <"$bindings") != 345 ]]; then
            die "variant=$variant omitted its 344-entry binding inventory"
        fi
        validate_success_path "$variant" "$log" "$bindings" ||
            die "variant=$variant failed production-path validation"
        write_result "$result" "$variant" passed 0 "$progress" "$log"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
            passed 0 "$lp" "$le" >>"$OUTPUT_DIR/run-journal.tsv"
        sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
        sleep "$POST_CASE_SETTLE_SECONDS"
    done
done

write_summary
phase=complete
[[ -s $OUTPUT_DIR/summary.md ]] && cat "$OUTPUT_DIR/summary.md"
printf 'SM75 small-BAR1 pair isolation complete: %s\n' "$OUTPUT_DIR"
