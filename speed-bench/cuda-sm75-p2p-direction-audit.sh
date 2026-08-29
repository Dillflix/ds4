#!/usr/bin/env bash
set -uo pipefail

die() {
    echo "error: $*" >&2
    exit 1
}

PAIRS=${PAIRS:-"1,0 0,1"}
SCENARIO=${SCENARIO:-t32}
TOKENS=${TOKENS:-512}
REPEAT_LEVELS=${REPEAT_LEVELS:-1,64,256,512,1536}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-300}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"$PWD/sm75-p2p-direction-audit-$(date -u +%Y%m%dT%H%M%SZ)"}

[[ $SCENARIO == t32 || $SCENARIO == t256 ]] ||
    die "SCENARIO must be t32 or t256"
[[ $TOKENS =~ ^[0-9]+$ ]] && (( TOKENS >= 2 && TOKENS <= 2048 )) ||
    die "TOKENS must be 2..2048"
[[ $TIMEOUT_SECONDS =~ ^[0-9]+$ ]] && (( TIMEOUT_SECONDS > 0 )) ||
    die "TIMEOUT_SECONDS must be positive"
[[ $SKIP_BUILD == 0 || $SKIP_BUILD == 1 ]] ||
    die "SKIP_BUILD must be 0 or 1"
[[ $CREATE_ARCHIVE == 0 || $CREATE_ARCHIVE == 1 ]] ||
    die "CREATE_ARCHIVE must be 0 or 1"

IFS=',' read -r -a levels <<< "$REPEAT_LEVELS"
(( ${#levels[@]} > 0 )) || die "REPEAT_LEVELS is empty"
for level in "${levels[@]}"; do
    [[ $level =~ ^[0-9]+$ ]] && (( level >= 1 && level <= 65536 )) ||
        die "invalid repeat level: $level"
done

pair_count=0
for pair in $PAIRS; do
    [[ $pair =~ ^[0-9]+,[0-9]+$ ]] || die "invalid pair: $pair"
    IFS=',' read -r home partner <<< "$pair"
    [[ $home != "$partner" ]] || die "pair devices must differ: $pair"
    pair_count=$((pair_count + 1))
done
(( pair_count > 0 )) || die "PAIRS is empty"

mkdir -p "$OUTPUT_DIR"

if [[ $SKIP_BUILD == 0 ]]; then
    make -j"$(nproc)" tests/test_gpu_xdev CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log" || die "build failed"
else
    make -q tests/test_gpu_xdev CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found a stale tests/test_gpu_xdev"
fi

target="q8-partner-${SCENARIO}-profile"
if [[ $SCENARIO == t32 ]]; then
    in_dim=1024
    out_dim=32768
else
    in_dim=8192
    out_dim=4096
fi
activation_bytes=$((TOKENS * in_dim * 4))
result_bytes=$((TOKENS * out_dim * 4))

if [[ ! -f $OUTPUT_DIR/manifest.txt ]]; then
    {
        printf 'scenario=%s\n' "$SCENARIO"
        printf 'tokens=%s\n' "$TOKENS"
        printf 'repeat_levels=%s\n' "$REPEAT_LEVELS"
        printf 'pairs=%s\n' "$PAIRS"
        printf 'activation_bytes_per_call=%s\n' "$activation_bytes"
        printf 'result_bytes_per_call=%s\n' "$result_bytes"
        nvidia-smi --query-gpu=index,pci.bus_id,uuid,serial \
            --format=csv,noheader
    } > "$OUTPUT_DIR/manifest.txt"
else
    grep -Fxq "scenario=$SCENARIO" "$OUTPUT_DIR/manifest.txt" &&
    grep -Fxq "tokens=$TOKENS" "$OUTPUT_DIR/manifest.txt" &&
    grep -Fxq "repeat_levels=$REPEAT_LEVELS" "$OUTPUT_DIR/manifest.txt" &&
    grep -Fxq "pairs=$PAIRS" "$OUTPUT_DIR/manifest.txt" ||
        die "OUTPUT_DIR belongs to a different audit configuration"
fi

if [[ ! -f $OUTPUT_DIR/summary.tsv ]]; then
    printf 'level\tpair\tstate\tstatus\tactivation_bytes\tresult_bytes\tlog\n' \
        > "$OUTPUT_DIR/summary.tsv"
fi

clean_env=(env)
while IFS='=' read -r name value; do
    case "$name" in
        DS4_*) clean_env+=(-u "$name") ;;
    esac
done < <(env)

stop=0
for level in "${levels[@]}"; do
    for pair in $PAIRS; do
        tag=${pair//,/-}
        prefix="$OUTPUT_DIR/${SCENARIO}-r${level}-${tag}"
        started="$prefix.started"
        complete="$prefix.complete"
        log="$prefix.log"
        result="$prefix.result"

        if [[ -f $complete ]]; then
            echo "Reusing completed scenario=$SCENARIO repeats=$level pair=$pair"
            continue
        fi
        if [[ -f $started ]]; then
            echo "Previous run was interrupted: scenario=$SCENARIO repeats=$level pair=$pair"
            stop=1
            break
        fi
        if [[ -f $prefix.failed ]]; then
            echo "Previous run failed: scenario=$SCENARIO repeats=$level pair=$pair"
            stop=1
            break
        fi

        total_activation=$((activation_bytes * level))
        total_result=$((result_bytes * level))
        printf 'scenario=%s\nrepeats=%s\nlogical_home,logical_partner=%s\n' \
            "$SCENARIO" "$level" "$pair" > "$started"
        sync "$started" 2>/dev/null || true

        echo "Running scenario=$SCENARIO repeats=$level pair=$pair "
        echo "  activation=$total_activation bytes result=$total_result bytes"

        "${clean_env[@]}" \
            CUDA_VISIBLE_DEVICES="$pair" \
            DS4_XDEV_PROFILE_TOKENS="$TOKENS" \
            DS4_XDEV_PROFILE_REPEATS="$level" \
            DS4_XDEV_PROFILE_PROGRESS=64 \
            timeout "${TIMEOUT_SECONDS}s" \
            stdbuf -oL -eL \
            ./tests/test_gpu_xdev "$target" \
            2>&1 | tee "$log"
        status=${PIPESTATUS[0]}

        printf 'status=%s\nactivation_bytes=%s\nresult_bytes=%s\n' \
            "$status" "$total_activation" "$total_result" > "$result"
        nvidia-smi -L > "$prefix.nvidia-smi.log" 2>&1 || true

        if (( status == 0 )); then
            mv "$started" "$complete"
            state=passed
        else
            mv "$started" "$prefix.failed"
            state=failed
            stop=1
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$level" "$pair" "$state" "$status" \
            "$total_activation" "$total_result" "$log" \
            >> "$OUTPUT_DIR/summary.tsv"
        (( stop == 0 )) || break
    done
    (( stop == 0 )) || break
done

column -t -s $'\t' "$OUTPUT_DIR/summary.tsv" ||
    cat "$OUTPUT_DIR/summary.tsv"

if [[ $CREATE_ARCHIVE == 1 ]]; then
    tar -C "$(dirname "$OUTPUT_DIR")" -czf "$OUTPUT_DIR.tar.gz" \
        "$(basename "$OUTPUT_DIR")"
    echo "Archive: $OUTPUT_DIR.tar.gz"
fi

if (( stop != 0 )); then
    echo "Audit stopped at the first interrupted or failed level."
fi
