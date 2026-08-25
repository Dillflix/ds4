#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Test whether lowering GDDR6 clocks releases board-power headroom for SM75 Q4.
The bounded production-shaped Q4 harness is used; no GGUF is opened.

Optional environment:
  DEVICES=1,2,3              GPU1 is the healthy passive-board control
  SCENARIOS=q4-early,q4-late
  TRIALS=3
  REPEATS=100
  SAMPLE_MS=100
  TARGET_SM_CLOCK=1620
  MEMORY_CLOCKS=              comma-separated explicit clocks; default discovers
                              common supported clocks and samples the upper range
  MAX_CLOCK_POINTS=5
  MIN_MEMORY_PERCENT=60       ignore low-power-state clocks below this % of max
  SETTLE_SECONDS=1
  USE_SUDO=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  MEMORY_SWEEP_DIR=/absolute/output/directory

The script requires otherwise-idle GPUs. It enables persistence, applies runtime
SM/memory locks, and resets both clock domains plus original persistence settings
on success, failure, or interruption. It never changes the power limit.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
DEVICES=${DEVICES:-1,2,3}
SCENARIOS=${SCENARIOS:-q4-early,q4-late}
TRIALS=${TRIALS:-3}
REPEATS=${REPEATS:-100}
SAMPLE_MS=${SAMPLE_MS:-100}
TARGET_SM_CLOCK=${TARGET_SM_CLOCK:-1620}
MEMORY_CLOCKS=${MEMORY_CLOCKS:-}
MAX_CLOCK_POINTS=${MAX_CLOCK_POINTS:-5}
MIN_MEMORY_PERCENT=${MIN_MEMORY_PERCENT:-60}
SETTLE_SECONDS=${SETTLE_SECONDS:-1}
USE_SUDO=${USE_SUDO:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${MEMORY_SWEEP_DIR:-$repo_dir/sm75-memory-clock-$stamp}

for item in "TRIALS:$TRIALS" "REPEATS:$REPEATS" "SAMPLE_MS:$SAMPLE_MS" \
            "TARGET_SM_CLOCK:$TARGET_SM_CLOCK" \
            "MAX_CLOCK_POINTS:$MAX_CLOCK_POINTS" \
            "MIN_MEMORY_PERCENT:$MIN_MEMORY_PERCENT" \
            "SETTLE_SECONDS:$SETTLE_SECONDS" "USE_SUDO:$USE_SUDO" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( TRIALS >= 2 && REPEATS >= 20 && REPEATS <= 200 &&
   SAMPLE_MS >= 50 && SAMPLE_MS <= 1000 && TARGET_SM_CLOCK > 0 &&
   MAX_CLOCK_POINTS >= 2 && MAX_CLOCK_POINTS <= 10 &&
   MIN_MEMORY_PERCENT >= 40 && MIN_MEMORY_PERCENT <= 100 &&
   SETTLE_SECONDS <= 30 )) || die "invalid sweep setting"
for flag in USE_SUDO SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
IFS=',' read -r -a devices <<<"$DEVICES"
IFS=',' read -r -a scenarios <<<"$SCENARIOS"
(( ${#devices[@]} >= 2 && ${#scenarios[@]} > 0 )) || die "empty device/scenario list"
declare -A seen_devices=()
for gpu in "${devices[@]}"; do
    [[ $gpu =~ ^[0-9]+$ ]] || die "invalid GPU index: $gpu"
    [[ -z ${seen_devices[$gpu]:-} ]] || die "duplicate GPU index: $gpu"
    seen_devices[$gpu]=1
done
for scenario in "${scenarios[@]}"; do
    [[ $scenario == q4-early || $scenario == q4-late ]] ||
        die "unsupported Q4 scenario: $scenario"
done
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{logs,telemetry,settings,supported-clocks,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

for command in nvidia-smi python3 tar awk sort; do
    command -v "$command" >/dev/null 2>&1 || die "$command not found"
done
if [[ $USE_SUDO == 1 ]]; then
    privilege=(sudo)
    sudo -v
else
    privilege=()
fi

phase=initialization
sampler_pid=
clocks_applied=0
cleanup_sampler() {
    if [[ -n ${sampler_pid:-} ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
        sampler_pid=
    fi
}
trim() {
    local value=$1
    value=${value#"${value%%[![:space:]]*}"}
    value=${value%"${value##*[![:space:]]}"}
    printf '%s' "$value"
}
restore_settings() {
    (( clocks_applied == 1 )) || return 0
    set +e
    for gpu in "${devices[@]}"; do
        "${privilege[@]}" nvidia-smi -i "$gpu" -rgc >/dev/null
        "${privilege[@]}" nvidia-smi -i "$gpu" -rmc >/dev/null
    done
    while IFS=, read -r gpu persistence; do
        gpu=$(trim "$gpu"); persistence=$(trim "$persistence")
        [[ -n $gpu ]] || continue
        if [[ $persistence == Disabled ]]; then
            "${privilege[@]}" nvidia-smi -i "$gpu" -pm 0 >/dev/null
        fi
    done <"$OUTPUT_DIR/provenance/original-settings.csv"
    clocks_applied=0
    set -e
}
finish() {
    status=$?
    trap - EXIT INT TERM
    cleanup_sampler
    restore_settings || status=1
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        tar -C "$(dirname "$OUTPUT_DIR")" -czf "$OUTPUT_DIR.tar.gz" \
            "$(basename "$OUTPUT_DIR")" || status=1
        printf 'Archive to return: %s.tar.gz\n' "$OUTPUT_DIR"
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM

phase=preflight
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
for gpu in "${devices[@]}"; do
    (( gpu < gpu_count )) || die "GPU$gpu does not exist (count=$gpu_count)"
done
active_processes=$(nvidia-smi --query-compute-apps=pid,process_name \
    --format=csv,noheader 2>/dev/null || true)
[[ -z $(trim "$active_processes") ]] ||
    die "compute processes are active; stop them before changing clocks: $active_processes"

if [[ $SKIP_BUILD == 0 ]]; then
    phase=build
    make -B -j"$(nproc)" tests/cuda_sm75_profile_harness CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q tests/cuda_sm75_profile_harness CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found a stale harness"
fi
[[ -x ./tests/cuda_sm75_profile_harness ]] || die "profile harness is missing"

phase=inventory
nvidia-smi -q >"$OUTPUT_DIR/nvidia-smi-q-before.txt"
nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,serial,vbios_version,persistence_mode,pstate,clocks.current.sm,clocks.max.sm,clocks.current.memory,power.draw,power.limit,power.max_limit,temperature.gpu \
    --format=csv >"$OUTPUT_DIR/inventory.csv"
nvidia-smi --query-gpu=index,persistence_mode --format=csv,noheader \
    | awk -F, -v wanted="$DEVICES" '
        BEGIN { n=split(wanted,a,","); for(i=1;i<=n;i++) keep[a[i]]=1 }
        { gpu=$1; gsub(/[[:space:]]/,"",gpu); if(keep[gpu]) print $0 }
      ' >"$OUTPUT_DIR/provenance/original-settings.csv"
nvidia-smi topo -m >"$OUTPUT_DIR/topology.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
for gpu in "${devices[@]}"; do
    nvidia-smi -i "$gpu" -q -d SUPPORTED_CLOCKS \
        >"$OUTPUT_DIR/supported-clocks/gpu$gpu.txt"
done

declare -a memory_clocks=()
if [[ -n $MEMORY_CLOCKS ]]; then
    IFS=',' read -r -a requested_clocks <<<"$MEMORY_CLOCKS"
    for clock in "${requested_clocks[@]}"; do
        [[ $clock =~ ^[0-9]+$ && $clock -gt 0 ]] || die "invalid memory clock: $clock"
        memory_clocks+=("$clock")
    done
    mapfile -t memory_clocks < <(printf '%s\n' "${memory_clocks[@]}" | sort -nru)
else
    declare -A clock_counts=()
    for gpu in "${devices[@]}"; do
        mapfile -t gpu_clocks < <(
            awk -F: '/^[[:space:]]*Memory[[:space:]]*:/ {
                value=$2; gsub(/[^0-9]/,"",value); if(value != "") print value
            }' "$OUTPUT_DIR/supported-clocks/gpu$gpu.txt" | sort -nru
        )
        ((${#gpu_clocks[@]} > 0)) || die "no supported memory clocks reported for GPU$gpu"
        for clock in "${gpu_clocks[@]}"; do
            clock_counts[$clock]=$(( ${clock_counts[$clock]:-0} + 1 ))
        done
    done
    mapfile -t reported_clocks < <(printf '%s\n' "${!clock_counts[@]}" | sort -nr)
    declare -a common_clocks=()
    for clock in "${reported_clocks[@]}"; do
        [[ ${clock_counts[$clock]} -eq ${#devices[@]} ]] || continue
        common_clocks+=("$clock")
    done
    ((${#common_clocks[@]} > 0)) || die "no memory-clock intersection"
    maximum_clock=${common_clocks[0]}
    declare -a eligible_clocks=()
    for clock in "${common_clocks[@]}"; do
        (( clock * 100 >= maximum_clock * MIN_MEMORY_PERCENT )) || continue
        eligible_clocks+=("$clock")
    done
    if ((${#eligible_clocks[@]} < 2)); then
        phase=not-applicable
        printf '%s\n' "${common_clocks[@]}" >"$OUTPUT_DIR/common-memory-clocks.txt"
        printf '%s\n' "${eligible_clocks[@]}" >"$OUTPUT_DIR/selected-memory-clocks.txt"
        {
            printf 'outcome=not-applicable\n'
            printf 'common_memory_clocks_mhz=%s\n' \
                "$(IFS=,; printf '%s' "${common_clocks[*]}")"
            printf 'performance_eligible_memory_clocks_mhz=%s\n' \
                "$(IFS=,; printf '%s' "${eligible_clocks[*]}")"
            printf 'reason=The boards expose fewer than two common memory states above %s%% of maximum.\n' \
                "$MIN_MEMORY_PERCENT"
            printf 'conclusion=No bounded memory-clock power-headroom sweep is available; the low-power state also constrains graphics clocks.\n'
        } | tee "$OUTPUT_DIR/analysis.txt"
        printf 'date_utc=%s\ngit_commit=%s\ndevices=%s\nscenarios=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
            "$DEVICES" "$SCENARIOS" >"$OUTPUT_DIR/manifest.txt"
        phase=complete
        printf 'SM75 Q4 memory-clock sweep is not applicable on these boards: %s\n' \
            "$OUTPUT_DIR"
        exit 0
    fi
    if ((${#eligible_clocks[@]} <= MAX_CLOCK_POINTS)); then
        memory_clocks=("${eligible_clocks[@]}")
    else
        declare -A selected=()
        for ((point=0; point<MAX_CLOCK_POINTS; point++)); do
            index=$(( point * (${#eligible_clocks[@]} - 1) / (MAX_CLOCK_POINTS - 1) ))
            selected[${eligible_clocks[$index]}]=1
        done
        mapfile -t memory_clocks < <(printf '%s\n' "${!selected[@]}" | sort -nr)
    fi
fi
((${#memory_clocks[@]} >= 2)) || die "the sweep requires at least two memory clocks"
printf '%s\n' "${memory_clocks[@]}" >"$OUTPUT_DIR/selected-memory-clocks.txt"

printf 'date_utc=%s\ngit_commit=%s\ndevices=%s\nscenarios=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
    "$DEVICES" "$SCENARIOS" >"$OUTPUT_DIR/manifest.txt"
printf 'trials=%s\nrepeats=%s\nsample_ms=%s\ntarget_sm_clock=%s\n' \
    "$TRIALS" "$REPEATS" "$SAMPLE_MS" "$TARGET_SM_CLOCK" \
    >>"$OUTPUT_DIR/manifest.txt"
printf 'memory_clocks=%s\nmax_clock_points=%s\nmin_memory_percent=%s\n' \
    "$(IFS=,; printf '%s' "${memory_clocks[*]}")" "$MAX_CLOCK_POINTS" \
    "$MIN_MEMORY_PERCENT" >>"$OUTPUT_DIR/manifest.txt"

query_help=$(nvidia-smi --help-query-gpu 2>/dev/null || true)
telemetry_query='timestamp,index,pstate,utilization.gpu,utilization.memory,clocks.current.graphics,clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu'
for field in \
    clocks_event_reasons.sw_power_cap \
    clocks_event_reasons.sw_thermal_slowdown \
    clocks_event_reasons.hw_slowdown \
    clocks_event_reasons.hw_thermal_slowdown \
    clocks_event_reasons.hw_power_brake_slowdown \
    clocks_event_reasons.sync_boost \
    clocks_throttle_reasons.sw_power_cap \
    clocks_throttle_reasons.sw_thermal_slowdown \
    clocks_throttle_reasons.hw_slowdown \
    clocks_throttle_reasons.hw_thermal_slowdown \
    clocks_throttle_reasons.hw_power_brake_slowdown \
    clocks_throttle_reasons.sync_boost; do
    if grep -Fq "$field" <<<"$query_help"; then telemetry_query+=",$field"; fi
done
printf '%s\n' "$telemetry_query" >"$OUTPUT_DIR/telemetry-query.txt"
start_sampler() {
    local path=$1
    nvidia-smi "--query-gpu=$telemetry_query" --format=csv -lms "$SAMPLE_MS" \
        >"$path" &
    sampler_pid=$!
}
validate_log() {
    local path=$1
    grep -Fq 'output_validation=exact-zero' "$path" || die "$path failed exactness"
    grep -Fq 'harness_status=ok' "$path" || die "$path failed its harness"
    grep -Eq '^timed_per_call_ms=[0-9]' "$path" || die "$path lacks timing"
}

phase=apply-clocks
clocks_applied=1
for gpu in "${devices[@]}"; do
    "${privilege[@]}" nvidia-smi -i "$gpu" -pm 1 >/dev/null
done
printf 'trial\tslot\tscenario\tdevice\trequested_sm_clock_mhz\trequested_memory_clock_mhz\tlog\ttelemetry\n' \
    >"$OUTPUT_DIR/runs.tsv"
slot=0
for ((trial=1; trial<=TRIALS; trial++)); do
    if (( trial % 2 )); then
        trial_clocks=("${memory_clocks[@]}")
        trial_devices=("${devices[@]}")
    else
        trial_clocks=()
        trial_devices=()
        for ((i=${#memory_clocks[@]}-1; i>=0; i--)); do trial_clocks+=("${memory_clocks[$i]}"); done
        for ((i=${#devices[@]}-1; i>=0; i--)); do trial_devices+=("${devices[$i]}"); done
    fi
    for memory_clock in "${trial_clocks[@]}"; do
        phase="apply-memory-$memory_clock-trial-$trial"
        for gpu in "${devices[@]}"; do
            "${privilege[@]}" nvidia-smi -i "$gpu" \
                -lmc "$memory_clock,$memory_clock" >/dev/null
            "${privilege[@]}" nvidia-smi -i "$gpu" \
                -lgc "$TARGET_SM_CLOCK,$TARGET_SM_CLOCK" >/dev/null
        done
        (( SETTLE_SECONDS == 0 )) || sleep "$SETTLE_SECONDS"
        nvidia-smi --query-gpu=index,pstate,clocks.current.sm,clocks.current.memory,power.draw,power.limit \
            --format=csv >"$OUTPUT_DIR/settings/memory-$memory_clock-trial-$trial.csv"
        for scenario in "${scenarios[@]}"; do
            for gpu in "${trial_devices[@]}"; do
                slot=$((slot + 1))
                stem="memory-$memory_clock-$scenario-gpu$gpu-t$trial"
                log="$OUTPUT_DIR/logs/$stem.log"
                sample="$OUTPUT_DIR/telemetry/$stem.csv"
                printf 'memory=%s MHz trial=%s scenario=%s GPU=%s...\n' \
                    "$memory_clock" "$trial" "$scenario" "$gpu"
                start_sampler "$sample"
                set +e
                CUDA_VISIBLE_DEVICES="$gpu" DS4_PROFILE_REPEATS="$REPEATS" \
                    ./tests/cuda_sm75_profile_harness "$scenario" >"$log" 2>&1
                rc=$?
                set -e
                cleanup_sampler
                (( rc == 0 )) || { tail -n 100 "$log" >&2 || true; die "$stem failed ($rc)"; }
                validate_log "$log"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$trial" "$slot" "$scenario" "$gpu" "$TARGET_SM_CLOCK" \
                    "$memory_clock" "$log" "$sample" >>"$OUTPUT_DIR/runs.tsv"
            done
        done
    done
done

phase=restore-clocks
restore_settings
phase=summarize
python3 speed-bench/summarize-sm75-memory-clock-sweep.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/summary-stdout.txt"
for required in samples.csv summary.csv decisions.csv analysis.txt; do
    [[ -s $OUTPUT_DIR/$required ]] || die "summary omitted $required"
done
phase=complete
printf 'SM75 Q4 memory-clock sweep complete: %s\n' "$OUTPUT_DIR"
