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
  VARIANTS=attention-off,production  scheduling matrix is explicit and fixed
  VARIANTS=attention-host-bounce     host-stage pair-0 attention-owned copies
  VARIANTS=attention-q8-host-bounce  host-stage pair-0 attention and Q8 copies
  VARIANTS=attention-q8-phase-audit  same cut plus pair-0 Q8 phase checkpoints
  VARIANTS=attention-q8-targeted-phase-audit  phase-audit layer-14 attn_output_b only
  VARIANTS=attention-q8-l14-l15-phase-audit   cumulative layer-14/layer-15 audit
  ATTN_PHASE_AUDIT_LAYER=17        one production row-split dispatch only
  ATTN_PHASE_AUDIT_POS=512
  ATTN_END_FENCE_LAYER=21          one end-only production completion fence
  ATTN_END_FENCE_POS=512
  ATTN_ROW_BOUNDARY_END_LAYER=17   combined row-boundary audit: prior row layer
  ATTN_ROW_BOUNDARY_ENTRY_LAYER=18 combined row-boundary audit: next row layer
  ATTN_ROW_BOUNDARY_POS=512
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
SMALL_BAR1_ISOLATION_DIR. The incomplete arm is retained without silently
retrying it. It counts as a failed arm only when a durable lost-device watch
record or an unhealthy post-run GPU snapshot corroborates device loss.
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
Q8_TARGET_BINDING_LABEL=tensor:blk.14.attn_output_b.weight
Q8_TARGET_WEIGHT_OFFSET=143571266304
Q8_TARGET_PASSED_LABEL=attn_output_b
Q8_TARGET_TOKENS=512
Q8_TARGET_IN_DIM=8192
Q8_TARGET_OUT_DIM=4096
Q8_TARGET_WEIGHT_BYTES=35651584
Q8_TARGET_TRANSFER_BYTES=8388608
Q8_TARGET_RESULT_BYTES=8388608
Q8_WINDOW_L15_BINDING_LABEL=tensor:blk.15.attn_output_b.weight
Q8_WINDOW_L15_WEIGHT_OFFSET=143723876608
Q8_WINDOW_TARGETS="${Q8_TARGET_BINDING_LABEL}@${Q8_TARGET_WEIGHT_OFFSET},${Q8_WINDOW_L15_BINDING_LABEL}@${Q8_WINDOW_L15_WEIGHT_OFFSET}"
VARIANTS=${VARIANTS:-attention-off,production}
ATTN_PHASE_AUDIT_LAYER=${ATTN_PHASE_AUDIT_LAYER:-17}
ATTN_PHASE_AUDIT_POS=${ATTN_PHASE_AUDIT_POS:-512}
ATTN_END_FENCE_LAYER=${ATTN_END_FENCE_LAYER:-21}
ATTN_END_FENCE_POS=${ATTN_END_FENCE_POS:-512}
ATTN_ROW_BOUNDARY_END_LAYER=${ATTN_ROW_BOUNDARY_END_LAYER:-17}
ATTN_ROW_BOUNDARY_ENTRY_LAYER=${ATTN_ROW_BOUNDARY_ENTRY_LAYER:-18}
ATTN_ROW_BOUNDARY_POS=${ATTN_ROW_BOUNDARY_POS:-512}
PP_TOKENS=${PP_TOKENS:-32768}
TG_TOKENS=${TG_TOKENS:-256}
Q8_TARGET_EXPECTED_SEQUENCES=$((PP_TOKENS / 512 + 1))
Q8_WINDOW_EXPECTED_TOTAL_SEQUENCES=$((Q8_TARGET_EXPECTED_SEQUENCES * 2))
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
            "ATTN_PHASE_AUDIT_LAYER:$ATTN_PHASE_AUDIT_LAYER" \
            "ATTN_PHASE_AUDIT_POS:$ATTN_PHASE_AUDIT_POS" \
            "ATTN_END_FENCE_LAYER:$ATTN_END_FENCE_LAYER" \
            "ATTN_END_FENCE_POS:$ATTN_END_FENCE_POS" \
            "ATTN_ROW_BOUNDARY_END_LAYER:$ATTN_ROW_BOUNDARY_END_LAYER" \
            "ATTN_ROW_BOUNDARY_ENTRY_LAYER:$ATTN_ROW_BOUNDARY_ENTRY_LAYER" \
            "ATTN_ROW_BOUNDARY_POS:$ATTN_ROW_BOUNDARY_POS" \
            "REQUIRED_POWER_LIMIT_W:$REQUIRED_POWER_LIMIT_W" \
            "TELEMETRY_INTERVAL_MS:$TELEMETRY_INTERVAL_MS" \
            "POST_CASE_SETTLE_SECONDS:$POST_CASE_SETTLE_SECONDS" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE" \
            "RESUME:$RESUME"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( SMALL_BAR1_PAIR == 0 && PP_TOKENS == 32768 && TG_TOKENS == 256 &&
   ATTN_PHASE_AUDIT_LAYER < STAGE_SPLIT &&
   ATTN_PHASE_AUDIT_POS < PP_TOKENS &&
   ATTN_PHASE_AUDIT_POS % 512 == 0 &&
   ATTN_END_FENCE_LAYER < STAGE_SPLIT &&
   ATTN_END_FENCE_POS < PP_TOKENS &&
   ATTN_END_FENCE_POS % 512 == 0 &&
   ATTN_ROW_BOUNDARY_END_LAYER == 17 &&
   ATTN_ROW_BOUNDARY_ENTRY_LAYER == 18 &&
   ATTN_ROW_BOUNDARY_POS == 512 &&
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
attention_copy_matrix_requested=0
for variant in "${variants[@]}"; do
    case "$variant" in
        attention-off|attention-host-bounce|attention-q8-host-bounce|attention-q8-phase-audit|attention-q8-targeted-phase-audit|attention-q8-l14-l15-phase-audit|attention-query-dst|attention-gather-dst|attention-both-dst|attention-phase-audit|attention-end-fence|attention-row-boundary-audit|partner-bounce|bounce-indexer-off|partner-serialized|indexer-off|production) ;;
        *) die "unknown variant: $variant" ;;
    esac
    [[ -z ${seen_variants[$variant]:-} ]] || die "duplicate variant: $variant"
    seen_variants[$variant]=1
    case "$variant" in
        attention-host-bounce|attention-q8-host-bounce|attention-q8-phase-audit|attention-q8-targeted-phase-audit|attention-q8-l14-l15-phase-audit|attention-query-dst|attention-gather-dst|attention-both-dst)
            attention_copy_matrix_requested=1
            ;;
    esac
done
if (( attention_copy_matrix_requested )) && [[ $SKIP_BUILD != 0 ]]; then
    die "attention copy diagnostic arms require SKIP_BUILD=0 so every reboot repeats the fixed CUDA/P2P preflight"
fi

for tool in awk basename cat cmp cp date dirname env find git grep kill make \
            mkdir mv nproc nvidia-smi python3 rm sleep sort stat stdbuf sync \
            tail tar tee timeout tr wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
git diff --quiet && git diff --cached --quiet ||
    die "tracked source changes are present; test an exact committed tree"

if [[ $RESUME == 0 ]]; then
    [[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"/{health,production,provenance,telemetry}
else
    [[ -d $OUTPUT_DIR ]] || die "resume directory not found: $OUTPUT_DIR"
    for subdir in health production provenance telemetry; do
        mkdir -p "$OUTPUT_DIR/$subdir"
    done
fi
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
printf 'Diagnostic directory: %s\n' "$OUTPUT_DIR"
printf 'Resume with: export SMALL_BAR1_ISOLATION_DIR=%q; RESUME=1 ...\n' \
    "$OUTPUT_DIR"
if [[ $RESUME == 1 ]]; then
    [[ -s $OUTPUT_DIR/manifest.txt ]] ||
        die "resume manifest is missing or empty"
    grep -Fxq "git_commit=$(git rev-parse HEAD)" "$OUTPUT_DIR/manifest.txt" ||
        die "resume commit differs from the original isolation"
    grep -Fxq "model=$MODEL" "$OUTPUT_DIR/manifest.txt" ||
        die "resume model differs from the original isolation"
    grep -Fxq "variants=$VARIANTS" "$OUTPUT_DIR/manifest.txt" ||
        die "resume variant order differs from the original isolation"
    grep -Fxq "small_bar1_pair=$SMALL_BAR1_PAIR" "$OUTPUT_DIR/manifest.txt" ||
        die "resume small-BAR1 pair differs from the original isolation"
    grep -Fxq "pp_tokens=$PP_TOKENS" "$OUTPUT_DIR/manifest.txt" ||
        die "resume prefill length differs from the original isolation"
    grep -Fxq "tg_tokens=$TG_TOKENS" "$OUTPUT_DIR/manifest.txt" ||
        die "resume decode length differs from the original isolation"
    grep -Fxq "repeats=$REPEATS" "$OUTPUT_DIR/manifest.txt" ||
        die "resume repeat count differs from the original isolation"
    grep -Fxq "required_power_limit_w=$REQUIRED_POWER_LIMIT_W" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume power limit differs from the original isolation"
    grep -Fxq "attention_phase_audit_layer=$ATTN_PHASE_AUDIT_LAYER" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume attention phase-audit layer differs from the original isolation"
    grep -Fxq "attention_phase_audit_pos=$ATTN_PHASE_AUDIT_POS" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume attention phase-audit position differs from the original isolation"
    grep -Fxq "attention_end_fence_layer=$ATTN_END_FENCE_LAYER" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume attention end-fence layer differs from the original isolation"
    grep -Fxq "attention_end_fence_pos=$ATTN_END_FENCE_POS" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume attention end-fence position differs from the original isolation"
    grep -Fxq "attention_row_boundary_end_layer=$ATTN_ROW_BOUNDARY_END_LAYER" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume attention row-boundary end layer differs from the original isolation"
    grep -Fxq "attention_row_boundary_entry_layer=$ATTN_ROW_BOUNDARY_ENTRY_LAYER" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume attention row-boundary entry layer differs from the original isolation"
    grep -Fxq "attention_row_boundary_pos=$ATTN_ROW_BOUNDARY_POS" \
        "$OUTPUT_DIR/manifest.txt" ||
        die "resume attention row-boundary position differs from the original isolation"
fi

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
        wait "$watch_pid" >/dev/null 2>&1 || true
    fi
    for name in telemetry_pid; do
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

assert_no_compute_processes() {
    local listing
    listing=$(timeout --kill-after=5s 20s nvidia-smi \
        --query-compute-apps=pid,process_name \
        --format=csv,noheader,nounits 2>&1) || {
        printf '%s\n' "$listing" >&2
        return 1
    }
    if [[ -n ${listing//[[:space:]]/} ]]; then
        printf 'unexpected GPU compute process(es) before arm:\n%s\n' \
            "$listing" >&2
        return 1
    fi
}

list_foreign_compute_processes() {
    local allowed_pid=$1 listing
    listing=$(timeout --kill-after=5s 20s nvidia-smi \
        --query-compute-apps=pid,process_name \
        --format=csv,noheader,nounits 2>/dev/null) || return 1
    printf '%s\n' "$listing" | awk -F, -v allowed="$allowed_pid" '
        {
            pid=$1
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", pid)
            if (pid ~ /^[0-9]+$/ && pid != allowed) print $0
        }
    '
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
    gpu_health_snapshot_is_healthy "$path"
}

gpu_health_snapshot_is_healthy() {
    local path=$1
    [[ -s $path ]] || return 1
    [[ $(grep -c '^GPU [0-9]:' "$path") == 4 ]] || return 1
    ! grep -Eiq \
        'ERR!|GPU is lost|Unknown Error|Unable to determine|GPU Unavailable|Critical Xid' \
        "$path"
}

gpu_health_snapshot_is_unhealthy() {
    local path=$1
    [[ -s $path ]] || return 1
    grep -Eiq \
        'ERR!|GPU is lost|Unknown Error|Unable to determine|GPU Unavailable|Critical Xid' \
        "$path"
}

start_telemetry() {
    local output=$1
    stdbuf -oL -eL nvidia-smi \
        --query-gpu=timestamp,index,pci.bus_id,pstate,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.free \
        --format=csv -lms "$TELEMETRY_INTERVAL_MS" >"$output" 2>&1 &
    telemetry_pid=$!
}

start_telemetry_watch() {
    local output=$1 marker=$2 case_pid=$3
    (
        while kill -0 "$telemetry_pid" >/dev/null 2>&1; do
            local watch_status= foreign_processes= lost_devices= recent_tail=
            recent_tail=$(tail -n 24 "$output" 2>/dev/null || true)
            if grep -Eiq 'GPU is lost|GPU requires reset|GPU Unavailable|Critical Xid|Unknown Error|ERR!|Unable to determine' \
                    <<<"$recent_tail"; then
                watch_status=lost-device-detected
                lost_devices=$(awk -F, '
                    function add(idx, bus, key) {
                        key=idx "@" bus
                        if (!seen[key]++) {
                            if (out != "") out=out ","
                            out=out key
                        }
                    }
                    {
                        lower=tolower($0)
                        if (lower !~ /gpu is lost|gpu requires reset|gpu unavailable|critical xid|unknown error|err!|unable to determine/) next
                        idx=$2; bus=$3
                        gsub(/^[[:space:]]+|[[:space:]]+$/, "", idx)
                        gsub(/^[[:space:]]+|[[:space:]]+$/, "", bus)
                        if (idx ~ /^[0-9]+$/) {
                            add(idx, bus)
                            next
                        }
                        if (index($0, "00000000:02:00.0")) add("0", "00000000:02:00.0")
                        if (index($0, "00000000:03:00.0")) add("1", "00000000:03:00.0")
                        if (index($0, "00000000:81:00.0")) add("2", "00000000:81:00.0")
                        if (index($0, "00000000:82:00.0")) add("3", "00000000:82:00.0")
                    }
                    END {print out}
                ' <<<"$recent_tail")
                [[ -n $lost_devices ]] || lost_devices=unknown
            elif printf '%s\n' "$recent_tail" |
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
            elif foreign_processes=$(list_foreign_compute_processes "$case_pid") &&
                    [[ -n ${foreign_processes//[[:space:]]/} ]]; then
                watch_status=foreign-compute-process
            fi
            if [[ -n $watch_status ]]; then
                printf 'telemetry-watch: %s detected\n' "$watch_status" >>"$output"
                printf 'timestamp_utc=%s\nstatus=%s\ncase_pid=%s\nrequired_power_limit_w=%s\nlost_devices=%s\nforeign_processes=%q\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$watch_status" \
                    "$case_pid" "$REQUIRED_POWER_LIMIT_W" "$lost_devices" \
                    "$foreign_processes" >"$marker"
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
targets=(ds4-bench tests/cuda_long_context_smoke tests/test_engine_mgpu_placement \
         tests/test_gpu_xdev)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log"
    "${clean[@]}" ./tests/test_engine_mgpu_placement \
        >"$OUTPUT_DIR/placement-tests.log" 2>&1 || {
            tail -n 200 "$OUTPUT_DIR/placement-tests.log" >&2
            die "CPU placement tests failed"
        }
    assert_no_compute_processes ||
        die "foreign GPU compute process present before CUDA smoke tests"
    "${clean[@]}" ./tests/cuda_long_context_smoke \
        >"$OUTPUT_DIR/smoke.log" 2>&1 || {
            tail -n 200 "$OUTPUT_DIR/smoke.log" >&2
            die "CUDA long-context smoke failed"
        }
    assert_no_compute_processes ||
        die "foreign GPU compute process present before ordered-copy tests"
    "${clean[@]}" ./tests/test_gpu_xdev ordered-dst-copy \
        >"$OUTPUT_DIR/ordered-dst-copy-tests.log" 2>&1 || {
            tail -n 200 "$OUTPUT_DIR/ordered-dst-copy-tests.log" >&2
            die "ordered destination-stream peer-copy tests failed"
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
        printf 'attention_phase_audit_layer=%s\nattention_phase_audit_pos=%s\n' \
            "$ATTN_PHASE_AUDIT_LAYER" "$ATTN_PHASE_AUDIT_POS"
        printf 'attention_end_fence_layer=%s\nattention_end_fence_pos=%s\n' \
            "$ATTN_END_FENCE_LAYER" "$ATTN_END_FENCE_POS"
        printf 'attention_row_boundary_end_layer=%s\nattention_row_boundary_entry_layer=%s\n' \
            "$ATTN_ROW_BOUNDARY_END_LAYER" "$ATTN_ROW_BOUNDARY_ENTRY_LAYER"
        printf 'attention_row_boundary_pos=%s\n' "$ATTN_ROW_BOUNDARY_POS"
        printf 'pp_tokens=%s\ntg_tokens=%s\nrepeats=%s\n' \
            "$PP_TOKENS" "$TG_TOKENS" "$REPEATS"
        printf 'required_power_limit_w=%s\n' "$REQUIRED_POWER_LIMIT_W"
        printf 'partner_work_retained=yes\ndecode_indexer_pair_fallback=home\n'
        printf 'host_bounce_scope=q8_partner_activation_and_result_only\n'
        printf 'serialization_scope=q8_partner_projection_pair_only\n'
        printf 'attention_copy_scheduling_scope=pair0_query_and_gather_independent\n'
        printf 'attention_copy_scheduling_transport=ordered_direct_peer_no_fallback\n'
        printf 'attention_host_bounce_scope=pair0_attention_owned_copies_only\n'
        printf 'attention_q8_host_bounce_scope=pair0_attention_owned_and_q8_partner_copies\n'
        printf 'attention_q8_phase_audit_scope=pair0_q8_partner_phase_checkpoints_and_compute_sync\n'
        printf 'attention_q8_targeted_phase_audit_binding=%s\n' "$Q8_TARGET_BINDING_LABEL"
        printf 'attention_q8_targeted_phase_audit_weight_offset=%s\n' "$Q8_TARGET_WEIGHT_OFFSET"
        printf 'attention_q8_targeted_phase_audit_passed_label=%s\n' "$Q8_TARGET_PASSED_LABEL"
        printf 'attention_q8_targeted_phase_audit_shape=%sx%sx%s\n' \
            "$Q8_TARGET_TOKENS" "$Q8_TARGET_IN_DIM" "$Q8_TARGET_OUT_DIM"
        printf 'attention_q8_targeted_phase_audit_transfer_bytes=%s\n' \
            "$Q8_TARGET_TRANSFER_BYTES"
        printf 'attention_q8_targeted_phase_audit_result_bytes=%s\n' \
            "$Q8_TARGET_RESULT_BYTES"
        printf 'attention_q8_targeted_phase_audit_expected_sequences=%s\n' \
            "$Q8_TARGET_EXPECTED_SEQUENCES"
        printf 'attention_q8_l14_l15_phase_audit_targets=%s\n' "$Q8_WINDOW_TARGETS"
        printf 'attention_q8_l14_l15_phase_audit_expected_weight_bytes=%s\n' \
            "$Q8_TARGET_WEIGHT_BYTES"
        printf 'attention_q8_l14_l15_phase_audit_expected_per_binding=%s\n' \
            "$Q8_TARGET_EXPECTED_SEQUENCES"
        printf 'attention_q8_l14_l15_phase_audit_expected_total_sequences=%s\n' \
            "$Q8_WINDOW_EXPECTED_TOTAL_SEQUENCES"
        printf 'attention_q8_l14_l15_phase_audit_target_preflight=exact-partner-tuples-against-materialized-binding-inventory\n'
        printf 'attention_copy_scheduling_preflight=build-smoke-ordered-copy-every-run\n'
        printf 'q8_transfer_audit=begin_complete_64-call-checkpoints\n'
        printf 'indexer_transfer_audit=every-dispatch-begin-complete\n'
        printf 'external_nvlink_counters=disabled-no-external-compute-workload\n'
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

validate_full_production_load() {
    local csv=$1
    awk -F, '
        NR==1 {
            for (i=1;i<=NF;i++) if ($i=="prefill_tps") col=i
            next
        }
        NR==2 {ok=(col>0 && $col>=500.0)}
        END {exit !ok}
    ' "$csv"
}

validate_attention_copy_schedule() {
    local log=$1 home=$2 partner=$3 query_schedule=$4 gather_schedule=$5
    grep -Eq \
        "prefill attention row audit dispatch=split .*home=${home} partner=${partner} .*query_copy_stream=${query_schedule} gather_copy_stream=${gather_schedule}" \
        "$log"
}

validate_attention_copy_transport() {
    local log=$1 home=$2 partner=$3 query_transport=$4 gather_transport=$5
    grep -Eq \
        "prefill attention row audit dispatch=split .*home=${home} partner=${partner} .*query_copy_transport=${query_transport} gather_copy_transport=${gather_transport}" \
        "$log"
}

validate_success_path() {
    local variant=$1 log=$2 bindings=$3 csv=$4 require_load=${5:-1}
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
            ! grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log" ||
                return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        attention-host-bounce)
            ! grep -Fq 'partner transport override for logical pair 0' "$log" ||
                return 1
            ! grep -Fq 'partner scheduling override for logical pair 0' "$log" ||
                return 1
            log_line_has "$log" 'q8 partner transfer audit event=begin home_tier=0' \
                'transport=peer' || return 1
            log_line_has "$log" 'q8 partner transfer audit event=begin home_tier=1' \
                'transport=peer' || return 1
            grep -Fq 'prefill attention row transport override for logical pair 0: pinned-host-bounce' \
                "$log" || return 1
            grep -Eq 'prefill attention host-bounce checkpoint event=complete .*pos=512 tokens=512 home=0 partner=2' \
                "$log" || return 1
            for cache_class in raw attn-comp index; do
                grep -Fq "prefill attention cache mirror transport=host-bounce home_tier=0 partner_tier=2 class=$cache_class event=complete" \
                    "$log" || return 1
            done
            ! grep -Fq 'prefill attention cache mirror transport=host-bounce home_tier=1' \
                "$log" || return 1
            ! grep -Fq 'prefill attention row split pair-scoped disable' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log" ||
                return 1
            validate_attention_copy_schedule "$log" 0 2 source source || return 1
            validate_attention_copy_schedule "$log" 1 3 source source || return 1
            validate_attention_copy_transport "$log" 0 2 host-bounce host-bounce ||
                return 1
            validate_attention_copy_transport "$log" 1 3 peer peer || return 1
            ! grep -Eq 'prefill attention row audit dispatch=split .*home=0 partner=2 .*query_copy_transport=peer|prefill attention row audit dispatch=split .*home=0 partner=2 .*gather_copy_transport=peer' \
                "$log" || return 1
            ! grep -Eq 'prefill attention row audit dispatch=split .*home=1 partner=3 .*query_copy_transport=host-bounce|prefill attention row audit dispatch=split .*home=1 partner=3 .*gather_copy_transport=host-bounce' \
                "$log" || return 1
            grep -Eq 'prefill attention row audit dispatch=split kind=indexed .*home=0 partner=2 .*topk_copy_transport=partner-local' \
                "$log" || return 1
            grep -Eq 'prefill attention row audit dispatch=split kind=indexed .*home=1 partner=3 .*topk_copy_transport=partner-local' \
                "$log" || return 1
            ! grep -Eq 'prefill attention row audit dispatch=split kind=indexed .*home=[01] partner=[23] .*topk_copy_transport=(peer|host-bounce)' \
                "$log" || return 1
            ! grep -Fq 'ordered destination-stream peer copy unavailable' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row phase audit event=' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row entry fence event=' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row end fence event=' "$log" ||
                return 1
            if (( require_load )); then
                validate_full_production_load "$csv" || return 1
            fi
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        attention-q8-host-bounce|attention-q8-phase-audit|attention-q8-targeted-phase-audit|attention-q8-l14-l15-phase-audit)
            grep -Fq 'partner transport override for logical pair 0: pinned-host-bounce' \
                "$log" || return 1
            ! grep -Fq 'partner scheduling override for logical pair 0' "$log" ||
                return 1
            grep -Fq 'prefill attention row transport override for logical pair 0: pinned-host-bounce' \
                "$log" || return 1

            # Calls=1 proves that each transport was selected during the
            # untimed 512-token warmup.  The later fixed counts are reached
            # only after the measured 32K prefill has begun for this model and
            # placement, so a warmup-only survival cannot satisfy this arm.
            grep -Eq 'q8 partner transfer audit event=begin home_tier=0 partner_tier=2 calls=1 .*transport=host-bounce serialized=no' \
                "$log" || return 1
            grep -Eq 'q8 partner transfer audit event=complete home_tier=0 partner_tier=2 calls=1 ' \
                "$log" || return 1
            grep -Eq 'q8 partner transfer audit event=begin home_tier=0 partner_tier=2 calls=128 .*transport=host-bounce serialized=no' \
                "$log" || return 1
            grep -Eq 'q8 partner transfer audit event=complete home_tier=0 partner_tier=2 calls=128 ' \
                "$log" || return 1
            grep -Eq 'q8 partner transfer audit event=begin home_tier=1 partner_tier=3 calls=1 .*transport=peer serialized=no' \
                "$log" || return 1
            grep -Eq 'q8 partner transfer audit event=complete home_tier=1 partner_tier=3 calls=1 ' \
                "$log" || return 1
            grep -Eq 'q8 partner transfer audit event=begin home_tier=1 partner_tier=3 calls=64 .*transport=peer serialized=no' \
                "$log" || return 1
            grep -Eq 'q8 partner transfer audit event=complete home_tier=1 partner_tier=3 calls=64 ' \
                "$log" || return 1
            ! grep -Eq 'q8 partner transfer audit event=begin home_tier=0 .*transport=peer' \
                "$log" || return 1
            ! grep -Eq 'q8 partner transfer audit event=begin home_tier=1 .*transport=host-bounce' \
                "$log" || return 1

            grep -Eq 'prefill attention host-bounce checkpoint event=complete .*pos=512 tokens=512 home=0 partner=2' \
                "$log" || return 1
            for cache_class in raw attn-comp index; do
                grep -Fq "prefill attention cache mirror transport=host-bounce home_tier=0 partner_tier=2 class=$cache_class event=complete" \
                    "$log" || return 1
            done
            ! grep -Fq 'prefill attention cache mirror transport=host-bounce home_tier=1' \
                "$log" || return 1
            ! grep -Fq 'prefill attention row split pair-scoped disable' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log" ||
                return 1
            validate_attention_copy_schedule "$log" 0 2 source source || return 1
            validate_attention_copy_schedule "$log" 1 3 source source || return 1
            validate_attention_copy_transport "$log" 0 2 host-bounce host-bounce ||
                return 1
            validate_attention_copy_transport "$log" 1 3 peer peer || return 1
            ! grep -Eq 'prefill attention row audit dispatch=split .*home=0 partner=2 .*query_copy_transport=peer|prefill attention row audit dispatch=split .*home=0 partner=2 .*gather_copy_transport=peer' \
                "$log" || return 1
            ! grep -Eq 'prefill attention row audit dispatch=split .*home=1 partner=3 .*query_copy_transport=host-bounce|prefill attention row audit dispatch=split .*home=1 partner=3 .*gather_copy_transport=host-bounce' \
                "$log" || return 1
            grep -Eq 'prefill attention row audit dispatch=split kind=indexed .*home=0 partner=2 .*topk_copy_transport=partner-local' \
                "$log" || return 1
            grep -Eq 'prefill attention row audit dispatch=split kind=indexed .*home=1 partner=3 .*topk_copy_transport=partner-local' \
                "$log" || return 1
            ! grep -Eq 'prefill attention row audit dispatch=split kind=indexed .*home=[01] partner=[23] .*topk_copy_transport=(peer|host-bounce)' \
                "$log" || return 1
            ! grep -Fq 'ordered destination-stream peer copy unavailable' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row phase audit event=' "$log" || return 1
            ! grep -Fq 'prefill attention row entry fence event=' "$log" || return 1
            ! grep -Fq 'prefill attention row end fence event=' "$log" || return 1
            if (( require_load )); then
                validate_full_production_load "$csv" || return 1
            fi
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        attention-query-dst|attention-gather-dst|attention-both-dst)
            ! grep -Fq 'prefill attention row split pair-scoped disable' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log" ||
                return 1
            local query_schedule=source gather_schedule=source
            case "$variant" in
                attention-query-dst) query_schedule=destination ;;
                attention-gather-dst) gather_schedule=destination ;;
                attention-both-dst)
                    query_schedule=destination
                    gather_schedule=destination
                    ;;
            esac
            validate_attention_copy_schedule "$log" 0 2 \
                "$query_schedule" "$gather_schedule" || return 1
            validate_attention_copy_schedule "$log" 1 3 source source || return 1
            ! grep -Fq 'ordered destination-stream peer copy unavailable' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row phase audit event=' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row entry fence event=' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row end fence event=' "$log" ||
                return 1
            if (( require_load )); then
                validate_full_production_load "$csv" || return 1
            fi
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        attention-phase-audit)
            ! grep -Fq 'prefill attention row split pair-scoped disable' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
                return 1
            for audit_phase in query-copy partner-attention home-attention result-gather; do
                log_line_has "$log" \
                    "prefill attention row phase audit event=complete phase=$audit_phase " \
                    "layer=$ATTN_PHASE_AUDIT_LAYER pos=$ATTN_PHASE_AUDIT_POS tokens=512 home=0 partner=2" ||
                    return 1
            done
            ! grep -Eq 'prefill attention row phase audit event=(submit-failed|device-switch-failed|sync-failed)' \
                "$log" || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        attention-end-fence)
            ! grep -Fq 'prefill attention row split pair-scoped disable' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
                return 1
            for fence_target in partner home pair; do
                log_line_has "$log" \
                    "prefill attention row end fence event=complete target=$fence_target " \
                    "layer=$ATTN_END_FENCE_LAYER pos=$ATTN_END_FENCE_POS tokens=512 home=0 partner=2" ||
                    return 1
            done
            ! grep -Eq 'prefill attention row end fence event=(submit-failed|device-switch-failed|sync-failed|failed)' \
                "$log" || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=0 partner_tier=2' || return 1
            log_line_has "$log" 'decode indexer row audit event=complete' \
                'home_tier=1 partner_tier=3' || return 1
            ;;
        attention-row-boundary-audit)
            ! grep -Fq 'prefill attention row split pair-scoped disable' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
                return 1
            python3 speed-bench/summarize-sm75-small-bar1-pair-isolation.py \
                --validate-attention-row-boundary-log "$log" \
                "$ATTN_ROW_BOUNDARY_END_LAYER" \
                "$ATTN_ROW_BOUNDARY_ENTRY_LAYER" \
                "$ATTN_ROW_BOUNDARY_POS" || return 1
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
            ! grep -Fq 'prefill attention row split pair-scoped disable' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 0 ' "$log" ||
                return 1
            grep -Fq 'prefill indexer score/top-k row split enabled: tier 1 ' "$log" ||
                return 1
            validate_attention_copy_schedule "$log" 0 2 source source || return 1
            validate_attention_copy_schedule "$log" 1 3 source source || return 1
            ! grep -Fq 'ordered destination-stream peer copy unavailable' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row phase audit event=' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row entry fence event=' "$log" ||
                return 1
            ! grep -Fq 'prefill attention row end fence event=' "$log" ||
                return 1
            if (( require_load )); then
                validate_full_production_load "$csv" || return 1
            fi
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
    if [[ $variant == attention-q8-l14-l15-phase-audit ]]; then
        grep -Fq \
            'CUDA q8 partner phase-audit target preflight validated 2 exact partner tuples against ' \
            "$log" || return 1
        python3 speed-bench/summarize-sm75-small-bar1-pair-isolation.py \
            --validate-q8-l14-l15-log "$log" \
            "$Q8_TARGET_EXPECTED_SEQUENCES" || return 1
    fi
    if [[ $variant == attention-q8-phase-audit ||
          $variant == attention-q8-targeted-phase-audit ]]; then
        # Require every observed pair-0 sequence to have an identical binding
        # identity and strict phase order.  The targeted arm must contain all
        # 65 calls established by the broad audit: one 512-token warmup plus
        # 64 measured 512-token microbatches. Any pair-1 marker, failure,
        # incomplete chain, or missing target call invalidates the arm. The
        # case validation above independently proves pair 0 bounced and pair 1
        # retained direct transport.
        local required_binding= required_offset= required_passed=
        local required_tokens= required_in= required_out=
        local required_transfer= required_result= required_sequences=
        if [[ $variant == attention-q8-targeted-phase-audit ]]; then
            required_binding=$Q8_TARGET_BINDING_LABEL
            required_offset=$Q8_TARGET_WEIGHT_OFFSET
            required_passed=$Q8_TARGET_PASSED_LABEL
            required_tokens=$Q8_TARGET_TOKENS
            required_in=$Q8_TARGET_IN_DIM
            required_out=$Q8_TARGET_OUT_DIM
            required_transfer=$Q8_TARGET_TRANSFER_BYTES
            required_result=$Q8_TARGET_RESULT_BYTES
            required_sequences=$Q8_TARGET_EXPECTED_SEQUENCES
        fi
        awk -v required_binding="$required_binding" \
            -v required_offset="$required_offset" \
            -v required_passed="$required_passed" \
            -v required_tokens="$required_tokens" \
            -v required_in="$required_in" \
            -v required_out="$required_out" \
            -v required_transfer="$required_transfer" \
            -v required_result="$required_result" \
            -v required_sequences="$required_sequences" '
            /ds4: CUDA q8 partner phase audit sequence=/ {
                sequence=event=stage=binding=passed=offset=home=partner=""
                tokens=in_dim=out_dim=transfer=result=""
                for (i=1; i<=NF; i++) {
                    split($i, field, "=")
                    if (field[1]=="sequence") sequence=field[2]
                    else if (field[1]=="event") event=field[2]
                    else if (field[1]=="stage") stage=field[2]
                    else if (field[1]=="binding_label") binding=field[2]
                    else if (field[1]=="passed_label") passed=field[2]
                    else if (field[1]=="weight_offset") offset=field[2]
                    else if (field[1]=="home_tier") home=field[2]
                    else if (field[1]=="partner_tier") partner=field[2]
                    else if (field[1]=="tokens") tokens=field[2]
                    else if (field[1]=="in") in_dim=field[2]
                    else if (field[1]=="out") out_dim=field[2]
                    else if (field[1]=="transfer_bytes") transfer=field[2]
                    else if (field[1]=="result_bytes") result=field[2]
                }
                if (home==1 || event ~ /-failed$/) bad=1
                if (home!=0) next
                if (partner!=2 || sequence !~ /^[0-9]+$/ ||
                    binding=="" || binding=="unavailable" ||
                    passed=="" || passed=="unavailable" ||
                    offset !~ /^[0-9]+$/) {
                    bad=1
                    next
                }
                if ((required_binding!="" && binding!=required_binding) ||
                    (required_offset!="" && offset!=required_offset) ||
                    (required_passed!="" && passed!=required_passed) ||
                    (required_tokens!="" && tokens!=required_tokens) ||
                    (required_in!="" && in_dim!=required_in) ||
                    (required_out!="" && out_dim!=required_out) ||
                    (required_transfer!="" && transfer!=required_transfer) ||
                    (required_result!="" && result!=required_result)) {
                    bad=1
                    next
                }
                key=sequence SUBSEP binding SUBSEP passed SUBSEP offset
                if (event=="begin" && stage=="activation-prepare") {
                    if (state[key]!=0) bad=1
                    else state[key]=1
                } else if (event=="activation-complete" &&
                           stage=="activation-copy") {
                    if (state[key]!=1) bad=1
                    else state[key]=2
                } else if (event=="pre-compute-sync-begin" &&
                           stage=="pre-compute-sync") {
                    if (state[key]!=2) bad=1
                    else state[key]=3
                } else if (event=="pre-compute-complete" &&
                           stage=="pre-compute-sync") {
                    if (state[key]!=3) bad=1
                    else state[key]=4
                } else if (event=="compute-submitted" && stage=="compute") {
                    if (state[key]!=4) bad=1
                    else state[key]=5
                } else if (event=="compute-complete" &&
                           stage=="compute-sync") {
                    if (state[key]!=5) bad=1
                    else state[key]=6
                } else if (event=="result-complete" &&
                           stage=="result-gather") {
                    if (state[key]!=6) bad=1
                    else state[key]=7
                }
            }
            END {
                observed=complete=0
                for (key in state) {
                    observed++
                    if (state[key]==7) complete++
                    else bad=1
                }
                if (required_sequences!="" &&
                    observed!=required_sequences) bad=1
                exit (bad || complete==0)
            }
        ' "$log" || return 1
    fi
}

validate_completed_path() {
    local variant=$1 log=$2 bindings=$3 csv=$4
    case "$variant" in
        attention-q8-host-bounce|attention-q8-phase-audit|attention-q8-targeted-phase-audit|attention-q8-l14-l15-phase-audit)
            validate_success_path "$variant" "$log" "$bindings" "$csv" 0
            ;;
        *)
            validate_success_path "$variant" "$log" "$bindings" "$csv"
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
                    [[ ! -e $watch_marker ]] &&
                    gpu_health_snapshot_is_healthy "$post_health"; then
                if validate_completed_path \
                        "$variant" "$log" "$bindings" "$csv"; then
                    recovered_status=passed
                    recovered_exit=0
                    case "$variant" in
                        attention-q8-host-bounce|attention-q8-phase-audit|attention-q8-targeted-phase-audit|attention-q8-l14-l15-phase-audit)
                            if ! validate_full_production_load "$csv"; then
                                recovered_status=inconclusive-underloaded
                                recovered_exit=128
                            fi
                            ;;
                    esac
                    printf 'Recovering completed prior arm: variant=%s repeat=%d status=%s...\n' \
                        "$variant" "$repeat" "$recovered_status"
                    write_result "$result" "$variant" "$recovered_status" \
                        "$recovered_exit" "$progress" "$log"
                    fields=$(last_progress_fields "$progress")
                    IFS=$'\t' read -r lp le lc lt <<<"$fields"
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                        "recovered-$recovered_status" "$recovered_exit" "$lp" "$le" \
                        >>"$OUTPUT_DIR/run-journal.tsv"
                    sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
                    continue
                fi
                printf 'Retaining completed prior arm as validation-failed: variant=%s repeat=%d\n' \
                    "$variant" "$repeat"
                write_result "$result" "$variant" validation-failed 127 \
                    "$progress" "$log"
                fields=$(last_progress_fields "$progress")
                IFS=$'\t' read -r lp le lc lt <<<"$fields"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                    recovered-validation-failed 127 "$lp" "$le" \
                    >>"$OUTPUT_DIR/run-journal.tsv"
                sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
                continue
            fi
            retained_status=interrupted-prior-run
            retained_exit=125
            if { [[ -s $watch_marker ]] &&
                    grep -Fxq 'status=lost-device-detected' "$watch_marker"; } ||
                    gpu_health_snapshot_is_unhealthy "$post_health"; then
                retained_status=interrupted-prior-run-device-loss
                retained_exit=124
                printf 'Retaining interrupted prior arm as corroborated device-loss failure: variant=%s repeat=%d\n' \
                    "$variant" "$repeat"
            else
                printf 'Retaining interrupted prior arm as incomplete evidence: variant=%s repeat=%d\n' \
                    "$variant" "$repeat"
            fi
            write_result "$result" "$variant" "$retained_status" "$retained_exit" \
                "$progress" "$log"
            fields=$(last_progress_fields "$progress")
            IFS=$'\t' read -r lp le lc lt <<<"$fields"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                "$retained_status" "$retained_exit" "$lp" "$le" \
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
        assert_no_compute_processes ||
            die "foreign GPU compute process present before variant=$variant"
        printf 'timestamp_utc=%s\nvariant=%s\nrepeat=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" >"$started"
        sync "$started" 2>/dev/null || sync

        variant_env=()
        case "$variant" in
            attention-off)
                variant_env+=("DS4_CUDA_NO_TP_PREFILL_ATTN_ROWS_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-host-bounce)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-q8-host-bounce)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-q8-phase-audit)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-q8-targeted-phase-audit)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_BINDING_LABEL=$Q8_TARGET_BINDING_LABEL")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_WEIGHT_OFFSET=$Q8_TARGET_WEIGHT_OFFSET")
                ;;
            attention-q8-l14-l15-phase-audit)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_HOST_BOUNCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_TARGETS=$Q8_WINDOW_TARGETS")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_EXPECTED_WEIGHT_BYTES=$Q8_TARGET_WEIGHT_BYTES")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_EXPECTED_IN_DIM=$Q8_TARGET_IN_DIM")
                variant_env+=("DS4_CUDA_Q8_F16_PARTNER_PHASE_AUDIT_EXPECTED_OUT_DIM=$Q8_TARGET_OUT_DIM")
                ;;
            attention-query-dst)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_QUERY_DST_STREAM_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-gather-dst)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_GATHER_DST_STREAM_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-both-dst)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_QUERY_DST_STREAM_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_GATHER_DST_STREAM_PAIRS=$SMALL_BAR1_PAIR")
                ;;
            attention-phase-audit)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_PHASE_AUDIT_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_PHASE_AUDIT_LAYER=$ATTN_PHASE_AUDIT_LAYER")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_PHASE_AUDIT_POS=$ATTN_PHASE_AUDIT_POS")
                ;;
            attention-end-fence)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_END_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_END_FENCE_LAYER=$ATTN_END_FENCE_LAYER")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_END_FENCE_POS=$ATTN_END_FENCE_POS")
                ;;
            attention-row-boundary-audit)
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_END_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_END_FENCE_LAYER=$ATTN_ROW_BOUNDARY_END_LAYER")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_END_FENCE_POS=$ATTN_ROW_BOUNDARY_POS")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_ENTRY_FENCE_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_ENTRY_FENCE_LAYER=$ATTN_ROW_BOUNDARY_ENTRY_LAYER")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_ENTRY_FENCE_POS=$ATTN_ROW_BOUNDARY_POS")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_PHASE_AUDIT_PAIRS=$SMALL_BAR1_PAIR")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_PHASE_AUDIT_LAYER=$ATTN_ROW_BOUNDARY_ENTRY_LAYER")
                variant_env+=("DS4_CUDA_TP_PREFILL_ATTN_PHASE_AUDIT_POS=$ATTN_ROW_BOUNDARY_POS")
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
        stop_telemetry
        if (( run_status == 0 )) && [[ -e $watch_marker ]]; then
            run_status=123
        fi
        if (( run_status == 0 )) && ! validate_power_limits; then
            run_status=126
            if [[ ! -s $watch_marker ]]; then
                printf 'timestamp_utc=%s\nstatus=post-run-power-limit-drift\nrequired_power_limit_w=%s\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                    "$REQUIRED_POWER_LIMIT_W" >"$watch_marker"
                sync "$watch_marker" 2>/dev/null || sync
            fi
        fi
        post_health_status=0
        gpu_health_snapshot "$post_health" || post_health_status=124
        if (( run_status == 0 && post_health_status != 0 )); then
            run_status=$post_health_status
        fi

        fields=$(last_progress_fields "$progress")
        IFS=$'\t' read -r lp le lc lt <<<"$fields"
        if (( run_status != 0 )); then
            watch_status=
            if [[ -s $watch_marker ]]; then
                watch_status=$(awk -F= '$1=="status" {print $2; exit}' "$watch_marker")
            fi
            result_status=run-failed-unverified
            if [[ $watch_status == lost-device-detected ]] ||
                    gpu_health_snapshot_is_unhealthy "$post_health"; then
                result_status=failed-device-loss
            elif [[ -n $watch_status ]]; then
                result_status=environment-invalid
            fi
            write_result "$result" "$variant" "$result_status" "$run_status" \
                "$progress" "$log"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                "$result_status" "$run_status" "$lp" "$le" \
                >>"$OUTPUT_DIR/run-journal.tsv"
            sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
            tail -n 240 "$log" >&2 || true
            die "variant=$variant ended with status=$result_status at phase=$lp event=$le"
        fi
        [[ -s $csv && $(wc -l <"$csv") == 2 ]] ||
            die "variant=$variant produced an invalid benchmark CSV"
        [[ -s $bindings ]] || die "variant=$variant omitted its binding inventory"
        if [[ $(wc -l <"$bindings") != 345 ]]; then
            die "variant=$variant omitted its 344-entry binding inventory"
        fi
        if ! validate_completed_path "$variant" "$log" "$bindings" "$csv"; then
            write_result "$result" "$variant" validation-failed 127 \
                "$progress" "$log"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                validation-failed 127 "$lp" "$le" \
                >>"$OUTPUT_DIR/run-journal.tsv"
            sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
            die "variant=$variant failed production-path validation"
        fi
        case "$variant" in
            attention-q8-host-bounce|attention-q8-phase-audit|attention-q8-targeted-phase-audit|attention-q8-l14-l15-phase-audit)
                if ! validate_full_production_load "$csv"; then
                    write_result "$result" "$variant" inconclusive-underloaded 128 \
                        "$progress" "$log"
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$variant" "$repeat" \
                        inconclusive-underloaded 128 "$lp" "$le" \
                        >>"$OUTPUT_DIR/run-journal.tsv"
                    sync "$OUTPUT_DIR/run-journal.tsv" 2>/dev/null || sync
                    printf 'error: variant=%s completed below 500 prefill tok/s; evidence is inconclusive\n' \
                        "$variant" >&2
                    exit 128
                fi
                ;;
        esac
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
