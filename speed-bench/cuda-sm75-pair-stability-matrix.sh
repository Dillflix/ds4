#!/usr/bin/env bash
set -uo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Run a symmetric, model-independent stability matrix on both physical NVLink
pairs. No sudo and no nvidia-smi polling occur while a load case is active.

Optional environment:
  PAIRS="2,3 0,1"              healthy-reference pair first
  SCENARIOS="residency compute p2p combined"
  DURATION_SECONDS=60
  CASE_GRACE_SECONDS=60        hard timeout is duration plus this grace
  RESIDENT_MIB=43008           per GPU; leaves roughly 5 GiB for work buffers
  COPY_MIB=128                 each of two copy buffers per GPU
  GEMM_N=4096
  GEMMS_PER_ROUND=4
  COPIES_PER_ROUND=4
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  PAIR_STABILITY_DIR=/absolute/output/directory

The matrix stops at the first failed, timed-out, or unhealthy case. Durable
started/result/progress records survive a host reset and identify the boundary.
EOF
}

[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
PAIRS=${PAIRS:-"2,3 0,1"}
SCENARIOS=${SCENARIOS:-"residency compute p2p combined"}
DURATION_SECONDS=${DURATION_SECONDS:-60}
CASE_GRACE_SECONDS=${CASE_GRACE_SECONDS:-60}
RESIDENT_MIB=${RESIDENT_MIB:-43008}
COPY_MIB=${COPY_MIB:-128}
GEMM_N=${GEMM_N:-4096}
GEMMS_PER_ROUND=${GEMMS_PER_ROUND:-4}
COPIES_PER_ROUND=${COPIES_PER_ROUND:-4}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${PAIR_STABILITY_DIR:-$repo_dir/sm75-pair-stability-$stamp}
CASE_TIMEOUT_SECONDS=$((DURATION_SECONDS + CASE_GRACE_SECONDS))

for item in "DURATION_SECONDS:$DURATION_SECONDS" \
            "CASE_GRACE_SECONDS:$CASE_GRACE_SECONDS" \
            "RESIDENT_MIB:$RESIDENT_MIB" \
            "COPY_MIB:$COPY_MIB" "GEMM_N:$GEMM_N" \
            "GEMMS_PER_ROUND:$GEMMS_PER_ROUND" \
            "COPIES_PER_ROUND:$COPIES_PER_ROUND" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( DURATION_SECONDS >= 5 && DURATION_SECONDS <= 3600 &&
   CASE_GRACE_SECONDS >= 15 && CASE_GRACE_SECONDS <= 300 &&
   RESIDENT_MIB >= 1024 && RESIDENT_MIB <= 45000 &&
   COPY_MIB >= 16 && COPY_MIB <= 1024 &&
   GEMM_N >= 1024 && GEMM_N <= 8192 &&
   GEMMS_PER_ROUND >= 1 && GEMMS_PER_ROUND <= 64 &&
   COPIES_PER_ROUND >= 1 && COPIES_PER_ROUND <= 64 )) ||
    die "invalid pair-stability configuration"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] ||
        die "$flag must be 0 or 1"
done
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical IDs remain stable"

declare -A allowed_scenario=(
    [residency]=1 [compute]=1 [p2p]=1 [combined]=1
)
scenario_count=0
for scenario in $SCENARIOS; do
    [[ ${allowed_scenario[$scenario]:-0} == 1 ]] ||
        die "unknown scenario: $scenario"
    scenario_count=$((scenario_count + 1))
done
(( scenario_count > 0 )) || die "SCENARIOS is empty"

pair_count=0
declare -A seen_pair=()
for pair in $PAIRS; do
    [[ $pair =~ ^[0-9]+,[0-9]+$ ]] || die "invalid pair: $pair"
    IFS=, read -r first second <<<"$pair"
    [[ $first != "$second" ]] || die "pair repeats one device: $pair"
    canonical=$first,$second
    (( first > second )) && canonical=$second,$first
    [[ -z ${seen_pair[$canonical]:-} ]] || die "duplicate pair: $pair"
    seen_pair[$canonical]=1
    pair_count=$((pair_count + 1))
done
(( pair_count == 2 )) || die "PAIRS must contain exactly two distinct pairs"

for tool in awk basename cat date dirname env git grep make mkdir mv nproc \
            nvidia-smi rm stat stdbuf sync tail tar tee timeout wc; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
git diff --quiet && git diff --cached --quiet ||
    die "tracked source changes are present; test an exact committed tree"
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{health,logs,progress,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
printf 'Diagnostic directory: %s\n' "$OUTPUT_DIR"

finish() {
    status=$?
    trap - EXIT INT TERM HUP
    printf 'exit_status=%s\n' "$status" >"$OUTPUT_DIR/run-status.txt"
    sync "$OUTPUT_DIR/run-status.txt" 2>/dev/null || sync
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        archive="$OUTPUT_DIR.tar.gz"
        partial="$archive.partial.$$"
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && [[ -s $partial ]] &&
                tar -tzf "$partial" >/dev/null && mv -- "$partial" "$archive"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            rm -f -- "$partial"
            printf 'error: failed to create archive %s\n' "$archive" >&2
            status=1
        fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT TERM HUP

if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" tests/cuda_sm75_pair_stability CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log" || die "build failed"
else
    make -q tests/cuda_sm75_pair_stability CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found a stale pair-stability harness"
fi

timeout --kill-after=5s 20s nvidia-smi -L \
    >"$OUTPUT_DIR/health/initial.log" 2>&1 ||
    die "initial GPU health query failed"
[[ $(grep -c '^GPU [0-9]:' "$OUTPUT_DIR/health/initial.log") == 4 ]] ||
    die "initial GPU inventory does not contain four devices"
timeout --kill-after=5s 20s nvidia-smi topo -m \
    >"$OUTPUT_DIR/health/topology.log" 2>&1 ||
    die "initial GPU topology query failed"

{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'pairs=%s\nscenarios=%s\nduration_seconds=%s\ncase_grace_seconds=%s\n' \
        "$PAIRS" "$SCENARIOS" "$DURATION_SECONDS" \
        "$CASE_GRACE_SECONDS"
    printf 'resident_mib_per_gpu=%s\ncopy_mib_per_buffer=%s\n' \
        "$RESIDENT_MIB" "$COPY_MIB"
    printf 'gemm_n=%s\ngemms_per_round=%s\ncopies_per_round=%s\n' \
        "$GEMM_N" "$GEMMS_PER_ROUND" "$COPIES_PER_ROUND"
    printf 'model=none\ntelemetry_during_load=disabled\nsudo=not-used\n'
    timeout --kill-after=5s 20s nvidia-smi \
        --query-gpu=index,pci.bus_id,uuid,serial \
        --format=csv,noheader
} >"$OUTPUT_DIR/manifest.txt" || die "failed to record the initial manifest"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"
printf 'timestamp_utc\tscenario\tpair\tstatus\texit_code\tlog\tprogress\thealth\n' \
    >"$OUTPUT_DIR/results.tsv"

clean=(env)
while IFS='=' read -r name _; do
    [[ $name == DS4_* ]] && clean+=(-u "$name")
done < <(env)

slot=0
stop=0
for scenario in $SCENARIOS; do
    for pair in $PAIRS; do
        slot=$((slot + 1))
        tag="s${slot}-${scenario}-${pair//,/-}"
        started="$OUTPUT_DIR/$tag.started"
        result="$OUTPUT_DIR/$tag.result"
        log="$OUTPUT_DIR/logs/$tag.log"
        progress="$OUTPUT_DIR/progress/$tag.csv"
        health="$OUTPUT_DIR/health/$tag-post.log"
        printf 'timestamp_utc=%s\nscenario=%s\nphysical_pair=%s\nstatus=starting\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$scenario" "$pair" \
            >"$started"
        sync "$started" 2>/dev/null || sync
        printf 'Pair stability slot=%d scenario=%s physical-pair=%s...\n' \
            "$slot" "$scenario" "$pair"

        "${clean[@]}" \
            "CUDA_VISIBLE_DEVICES=$pair" \
            "DS4_PAIR_STRESS_SECONDS=$DURATION_SECONDS" \
            "DS4_PAIR_STRESS_RESIDENT_MIB=$RESIDENT_MIB" \
            "DS4_PAIR_STRESS_COPY_MIB=$COPY_MIB" \
            "DS4_PAIR_STRESS_GEMM_N=$GEMM_N" \
            "DS4_PAIR_STRESS_GEMMS_PER_ROUND=$GEMMS_PER_ROUND" \
            "DS4_PAIR_STRESS_COPIES_PER_ROUND=$COPIES_PER_ROUND" \
            "DS4_PAIR_STRESS_JOURNAL=$progress" \
            timeout --foreground --kill-after=15s "${CASE_TIMEOUT_SECONDS}s" \
            stdbuf -oL -eL ./tests/cuda_sm75_pair_stability "$scenario" \
            >"$log" 2>&1
        status=$?
        timeout --kill-after=5s 20s nvidia-smi -L >"$health" 2>&1
        health_status=$?
        if (( status == 0 && health_status == 0 )) &&
                grep -Fq 'harness_status=ok' "$log" &&
                [[ $(grep -c '^GPU [0-9]:' "$health") == 4 ]]; then
            state=passed
            mv "$started" "$OUTPUT_DIR/$tag.complete"
        else
            state=failed
            mv "$started" "$OUTPUT_DIR/$tag.failed"
            stop=1
        fi
        printf 'status=%s\nexit_code=%s\nhealth_exit_code=%s\n' \
            "$state" "$status" "$health_status" >"$result"
        sync "$result" 2>/dev/null || sync
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$scenario" "$pair" \
            "$state" "$status" "$log" "$progress" "$health" \
            >>"$OUTPUT_DIR/results.tsv"
        (( stop == 0 )) || break
    done
    (( stop == 0 )) || break
done

column -t -s $'\t' "$OUTPUT_DIR/results.tsv" 2>/dev/null ||
    cat "$OUTPUT_DIR/results.tsv"
(( stop == 0 )) || die "pair stability matrix stopped at its first failure"
printf 'SM75 pair stability matrix complete: %s\n' "$OUTPUT_DIR"
