#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Diagnose per-board and per-NVLink-pair Q4 behavior on all four SM75 GPUs.
The test uses the bounded production-shaped Q4 harness; it never opens a GGUF.

Optional environment:
  SCENARIOS=q4-early,q4-late
  PAIR_SCENARIO=q4-late
  TRIALS=3
  REPEATS=100
  SAMPLE_MS=100
  RUN_PAIR_CONCURRENT=1
  NORMALIZE=1               also run a reversible common-clock A/B
  TARGET_SM_CLOCK=1620      diagnostic common clock, not a production default
  RAISE_POWER_LIMIT=0       if 1, temporarily use each board's reported maximum
  USE_SUDO=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  Q4_CLOCK_DIR=/absolute/output/directory

Normalization records the original persistence and power settings, applies a
common locked SM clock only for the second phase, and restores clocks, power
limits, and persistence on success, failure, or interruption.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
SCENARIOS=${SCENARIOS:-q4-early,q4-late}
PAIR_SCENARIO=${PAIR_SCENARIO:-q4-late}
TRIALS=${TRIALS:-3}
REPEATS=${REPEATS:-100}
SAMPLE_MS=${SAMPLE_MS:-100}
RUN_PAIR_CONCURRENT=${RUN_PAIR_CONCURRENT:-1}
NORMALIZE=${NORMALIZE:-1}
TARGET_SM_CLOCK=${TARGET_SM_CLOCK:-1620}
RAISE_POWER_LIMIT=${RAISE_POWER_LIMIT:-0}
USE_SUDO=${USE_SUDO:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${Q4_CLOCK_DIR:-$repo_dir/sm75-q4-clock-$stamp}

for item in "TRIALS:$TRIALS" "REPEATS:$REPEATS" "SAMPLE_MS:$SAMPLE_MS" \
            "RUN_PAIR_CONCURRENT:$RUN_PAIR_CONCURRENT" "NORMALIZE:$NORMALIZE" \
            "TARGET_SM_CLOCK:$TARGET_SM_CLOCK" \
            "RAISE_POWER_LIMIT:$RAISE_POWER_LIMIT" "USE_SUDO:$USE_SUDO" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( TRIALS >= 2 && REPEATS >= 20 && REPEATS <= 100 &&
   SAMPLE_MS >= 50 && SAMPLE_MS <= 1000 && TARGET_SM_CLOCK > 0 )) ||
    die "invalid trial/repeat/sample/clock setting"
for flag in RUN_PAIR_CONCURRENT NORMALIZE RAISE_POWER_LIMIT USE_SUDO \
            SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
IFS=',' read -r -a scenarios <<<"$SCENARIOS"
(( ${#scenarios[@]} > 0 )) || die "SCENARIOS is empty"
for scenario in "${scenarios[@]}" "$PAIR_SCENARIO"; do
    [[ $scenario == q4-early || $scenario == q4-late ]] ||
        die "unsupported Q4 scenario: $scenario"
done
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{logs,telemetry,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
command -v python3 >/dev/null 2>&1 || die "python3 not found"
command -v tar >/dev/null 2>&1 || die "tar not found"
if [[ $USE_SUDO == 1 ]]; then privilege=(sudo); else privilege=(); fi

phase=initialization
sampler_pid=
normalization_applied=0
cleanup_sampler() {
    if [[ -n ${sampler_pid:-} ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
        sampler_pid=
    fi
}
trim() { local value=$1; value=${value#"${value%%[![:space:]]*}"}; value=${value%"${value##*[![:space:]]}"}; printf '%s' "$value"; }
restore_settings() {
    (( normalization_applied == 1 )) || return 0
    set +e
    for gpu in 0 1 2 3; do "${privilege[@]}" nvidia-smi -i "$gpu" -rgc >/dev/null; done
    while IFS=, read -r gpu persistence power; do
        gpu=$(trim "$gpu"); persistence=$(trim "$persistence"); power=$(trim "$power")
        [[ -n $gpu ]] || continue
        "${privilege[@]}" nvidia-smi -i "$gpu" -pl "$power" >/dev/null
        if [[ $persistence == Disabled ]]; then
            "${privilege[@]}" nvidia-smi -i "$gpu" -pm 0 >/dev/null
        fi
    done <"$OUTPUT_DIR/provenance/original-settings.csv"
    normalization_applied=0
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
nvidia-smi -q -x >"$OUTPUT_DIR/nvidia-smi-q-before.xml"
inventory_query='index,name,pci.bus_id,uuid,serial,vbios_version,persistence_mode,pstate,clocks.current.sm,clocks.max.sm,clocks.current.memory,power.draw,power.limit,power.default_limit,power.max_limit,temperature.gpu,ecc.mode.current,driver_version'
nvidia-smi "--query-gpu=$inventory_query" --format=csv \
    >"$OUTPUT_DIR/inventory.csv"
nvidia-smi --query-gpu=index,persistence_mode,power.limit \
    --format=csv,noheader,nounits \
    >"$OUTPUT_DIR/provenance/original-settings.csv"
nvidia-smi topo -m >"$OUTPUT_DIR/topology.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
printf 'date_utc=%s\ngit_commit=%s\nscenarios=%s\npair_scenario=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
    "$SCENARIOS" "$PAIR_SCENARIO" >"$OUTPUT_DIR/manifest.txt"
printf 'trials=%s\nrepeats=%s\nsample_ms=%s\nnormalize=%s\ntarget_sm_clock=%s\nraise_power_limit=%s\n' \
    "$TRIALS" "$REPEATS" "$SAMPLE_MS" "$NORMALIZE" \
    "$TARGET_SM_CLOCK" "$RAISE_POWER_LIMIT" >>"$OUTPUT_DIR/manifest.txt"

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

printf 'mode\tscope\ttrial\tslot\tscenario\tdevice\tlog\ttelemetry\n' \
    >"$OUTPUT_DIR/runs.tsv"
slot=0
run_single() {
    local mode=$1 trial=$2 scenario=$3 gpu=$4
    slot=$((slot + 1))
    local stem="$mode-single-$scenario-gpu$gpu-t$trial"
    local log="$OUTPUT_DIR/logs/$stem.log"
    local sample="$OUTPUT_DIR/telemetry/$stem.csv"
    printf '%s single trial=%s scenario=%s GPU=%s...\n' \
        "$mode" "$trial" "$scenario" "$gpu"
    start_sampler "$sample"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" DS4_PROFILE_REPEATS="$REPEATS" \
        ./tests/cuda_sm75_profile_harness "$scenario" >"$log" 2>&1
    local rc=$?
    set -e
    cleanup_sampler
    (( rc == 0 )) || { tail -n 100 "$log" >&2 || true; die "$stem failed ($rc)"; }
    validate_log "$log"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$mode" single "$trial" "$slot" "$scenario" "$gpu" "$log" "$sample" \
        >>"$OUTPUT_DIR/runs.tsv"
}
run_pair() {
    local mode=$1 trial=$2 gpu_a=$3 gpu_b=$4
    slot=$((slot + 1))
    local scope="pair${gpu_a}${gpu_b}"
    local stem="$mode-$scope-$PAIR_SCENARIO-t$trial"
    local log_a="$OUTPUT_DIR/logs/$stem-gpu$gpu_a.log"
    local log_b="$OUTPUT_DIR/logs/$stem-gpu$gpu_b.log"
    local sample="$OUTPUT_DIR/telemetry/$stem.csv"
    printf '%s concurrent %s trial=%s scenario=%s...\n' \
        "$mode" "$scope" "$trial" "$PAIR_SCENARIO"
    start_sampler "$sample"
    CUDA_VISIBLE_DEVICES="$gpu_a" DS4_PROFILE_REPEATS="$REPEATS" \
        ./tests/cuda_sm75_profile_harness "$PAIR_SCENARIO" >"$log_a" 2>&1 &
    local pid_a=$!
    CUDA_VISIBLE_DEVICES="$gpu_b" DS4_PROFILE_REPEATS="$REPEATS" \
        ./tests/cuda_sm75_profile_harness "$PAIR_SCENARIO" >"$log_b" 2>&1 &
    local pid_b=$!
    set +e
    wait "$pid_a"; local rc_a=$?
    wait "$pid_b"; local rc_b=$?
    set -e
    cleanup_sampler
    (( rc_a == 0 && rc_b == 0 )) || {
        tail -n 100 "$log_a" >&2 || true; tail -n 100 "$log_b" >&2 || true
        die "$stem failed ($rc_a/$rc_b)"
    }
    validate_log "$log_a"; validate_log "$log_b"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$mode" "$scope" "$trial" "$slot" "$PAIR_SCENARIO" "$gpu_a" \
        "$log_a" "$sample" >>"$OUTPUT_DIR/runs.tsv"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$mode" "$scope" "$trial" "$slot" "$PAIR_SCENARIO" "$gpu_b" \
        "$log_b" "$sample" >>"$OUTPUT_DIR/runs.tsv"
}
run_mode() {
    local mode=$1
    phase="$mode-tests"
    for ((trial=1; trial<=TRIALS; trial++)); do
        if (( trial % 2 )); then devices=(0 1 2 3); else devices=(3 2 1 0); fi
        for scenario in "${scenarios[@]}"; do
            for gpu in "${devices[@]}"; do run_single "$mode" "$trial" "$scenario" "$gpu"; done
        done
        if [[ $RUN_PAIR_CONCURRENT == 1 ]]; then
            if (( trial % 2 )); then
                run_pair "$mode" "$trial" 0 1; run_pair "$mode" "$trial" 2 3
            else
                run_pair "$mode" "$trial" 2 3; run_pair "$mode" "$trial" 0 1
            fi
        fi
    done
}

run_mode baseline
if [[ $NORMALIZE == 1 ]]; then
    phase=apply-normalization
    normalization_applied=1
    for gpu in 0 1 2 3; do
        "${privilege[@]}" nvidia-smi -i "$gpu" -pm 1 >/dev/null
    done
    if [[ $RAISE_POWER_LIMIT == 1 ]]; then
        while IFS=, read -r gpu max_power; do
            gpu=$(trim "$gpu"); max_power=$(trim "$max_power")
            "${privilege[@]}" nvidia-smi -i "$gpu" -pl "$max_power" >/dev/null
        done < <(nvidia-smi --query-gpu=index,power.max_limit \
                    --format=csv,noheader,nounits)
    fi
    for gpu in 0 1 2 3; do
        "${privilege[@]}" nvidia-smi -i "$gpu" \
            -lgc "$TARGET_SM_CLOCK,$TARGET_SM_CLOCK" >/dev/null
    done
    nvidia-smi -q >"$OUTPUT_DIR/nvidia-smi-q-normalized.txt"
    run_mode normalized
    phase=restore-normalization
    restore_settings
fi

phase=summarize
python3 speed-bench/summarize-sm75-q4-clock-audit.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/summary-stdout.txt"
for required in samples.csv summary.csv analysis.txt; do
    [[ -s $OUTPUT_DIR/$required ]] || die "summary omitted $required"
done
phase=complete
printf 'SM75 four-GPU Q4 clock audit complete: %s\n' "$OUTPUT_DIR"
