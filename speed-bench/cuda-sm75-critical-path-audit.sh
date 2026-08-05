#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Capture the bounded DS4 SM75 prefill critical path and attribute the observed
stage imbalance to either layer range or physical GPU/pair.

The audit performs exactly two 2048-token Nsight Systems captures:
  current: 0,3,1,2  (stage 0 on GPU 0, stage 1 on GPU 3)
  swapped: 3,0,2,1  (stage 0 on GPU 3, stage 1 on GPU 0)

It also runs the same synthetic production-shaped Q4/Q8 work on physical GPUs
0 and 3. Timeline annotations are opt-in NVTX ranges and add no CUDA events,
stream waits, or synchronization.

Required environment:
  MODEL=/absolute/path/to/full-Q4.gguf

Optional environment:
  PROMPT=...                         fixed prompt file
  CURRENT_DEVICES=0,3,1,2
  SWAPPED_DEVICES=3,0,2,1
  GPU_VRAM=auto
  STAGE_SPLIT=22
  PROFILE_TOKENS=2048
  PREFILL_CHUNK=2048
  PIPELINE_MB=512
  HARNESS_SCENARIOS=q4-early,q4-late,q8-q-b,q8-attn
  HARNESS_TRIALS=3
  HARNESS_REPEATS=20
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  CRITICAL_PATH_DIR=/absolute/output/directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

: "${MODEL:?set MODEL to the absolute full-Q4 GGUF path}"
[[ $MODEL == /* && -f $MODEL ]] || die "MODEL must name an existing absolute file"
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
CURRENT_DEVICES=${CURRENT_DEVICES:-0,3,1,2}
SWAPPED_DEVICES=${SWAPPED_DEVICES:-3,0,2,1}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
PROFILE_TOKENS=${PROFILE_TOKENS:-2048}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
PIPELINE_MB=${PIPELINE_MB:-512}
HARNESS_SCENARIOS=${HARNESS_SCENARIOS:-q4-early,q4-late,q8-q-b,q8-attn}
HARNESS_TRIALS=${HARNESS_TRIALS:-3}
HARNESS_REPEATS=${HARNESS_REPEATS:-20}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${CRITICAL_PATH_DIR:-$repo_dir/sm75-critical-path-$stamp}

[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "PROFILE_TOKENS:$PROFILE_TOKENS" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" "PIPELINE_MB:$PIPELINE_MB" \
            "HARNESS_TRIALS:$HARNESS_TRIALS" \
            "HARNESS_REPEATS:$HARNESS_REPEATS" "SKIP_BUILD:$SKIP_BUILD" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}
    value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT > 0 && STAGE_SPLIT < 43 )) || die "STAGE_SPLIT must be in 1..42"
(( PROFILE_TOKENS >= 1024 && PREFILL_CHUNK >= PROFILE_TOKENS &&
   PIPELINE_MB > 0 && PIPELINE_MB < PROFILE_TOKENS )) ||
    die "PROFILE_TOKENS must exceed PIPELINE_MB and fit PREFILL_CHUNK"
(( HARNESS_TRIALS >= 2 && HARNESS_REPEATS >= 5 )) ||
    die "use at least two harness trials and five timed repeats"
[[ $SKIP_BUILD == 0 || $SKIP_BUILD == 1 ]] || die "SKIP_BUILD must be 0 or 1"
[[ $CREATE_ARCHIVE == 0 || $CREATE_ARCHIVE == 1 ]] ||
    die "CREATE_ARCHIVE must be 0 or 1"
[[ $CURRENT_DEVICES == 0,3,1,2 ]] ||
    die "CURRENT_DEVICES must remain 0,3,1,2 for the fixed 2x2 attribution"
[[ $SWAPPED_DEVICES == 3,0,2,1 ]] ||
    die "SWAPPED_DEVICES must remain 3,0,2,1 for the fixed 2x2 attribution"
command -v nsys >/dev/null 2>&1 || die "Nsight Systems CLI (nsys) not found"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
command -v python3 >/dev/null 2>&1 || die "python3 not found"
command -v tar >/dev/null 2>&1 || die "tar not found"

IFS=',' read -r -a scenarios <<<"$HARNESS_SCENARIOS"
(( ${#scenarios[@]} > 0 )) || die "HARNESS_SCENARIOS is empty"
for scenario in "${scenarios[@]}"; do
    case "$scenario" in
        q4-early|q4-late|q8-q-b|q8-attn|q8-shared|q8-out-b) ;;
        *) die "unsupported harness scenario: $scenario" ;;
    esac
done

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{nsys,harness,telemetry,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
sampler_pid=
cleanup_sampler() {
    if [[ -n ${sampler_pid:-} ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
        sampler_pid=
    fi
}
finish() {
    status=$?
    trap - EXIT INT TERM
    cleanup_sampler
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

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done

if [[ $SKIP_BUILD == 0 ]]; then
    phase=build
    make -B -j"$(nproc)" ds4-bench tests/cuda_sm75_profile_harness \
        CUDA_ARCH=sm_75 2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q ds4-bench tests/cuda_sm75_profile_harness CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi
[[ -x ./ds4-bench && -x ./tests/cuda_sm75_profile_harness ]] ||
    die "required binaries are missing"

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nmodel_hashing=disabled\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'current_devices=%s\nswapped_devices=%s\nstage_split=%s/%s\n' \
        "$CURRENT_DEVICES" "$SWAPPED_DEVICES" "$STAGE_SPLIT" \
        "$((43-STAGE_SPLIT))"
    printf 'profile_tokens=%s\nprefill_chunk=%s\npipeline_mb=%s\n' \
        "$PROFILE_TOKENS" "$PREFILL_CHUNK" "$PIPELINE_MB"
    printf 'q8_partner_classes=t256\nharness_scenarios=%s\n' "$HARNESS_SCENARIOS"
    printf 'harness_trials=%s\nharness_repeats=%s\n' \
        "$HARNESS_TRIALS" "$HARNESS_REPEATS"
    printf '\n[gpu inventory]\n'
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.free,\
clocks.current.sm,clocks.current.memory,pstate,power.limit,ecc.mode.current,driver_version \
        --format=csv
    printf '\n[gpu topology]\n'
    nvidia-smi topo -m
    printf '\n[nsight systems]\n'
    nsys --version
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

printf 'label\tdevices\tsqlite\n' >"$OUTPUT_DIR/trace-map.tsv"

capture_trace() {
    local label=$1 devices=$2
    local base="$OUTPUT_DIR/nsys/$label"
    phase="nsight-$label"
    printf 'Nsight Systems critical-path capture: %s devices=%s...\n' \
        "$label" "$devices"
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,\
memory.used,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory,pstate \
        --format=csv -lms 100 >"$OUTPUT_DIR/telemetry/$label.csv" &
    sampler_pid=$!
    set +e
    "${clean[@]}" \
        "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
        DS4_CUDA_PREFILL_PIPELINE=1 \
        "DS4_CUDA_PREFILL_PIPELINE_MB=$PIPELINE_MB" \
        DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
        DS4_CUDA_Q8_F16_PARTNER_CLASSES=t256 \
        DS4_CUDA_CRITICAL_PATH_NVTX=1 \
        DS4_CUDA_PREFILL_AUDIT=1 \
        "DS4_BENCH_UNTIMED_WARMUP_TOKENS=$PROFILE_TOKENS" \
        DS4_NSYS_CAPTURE_PREFILL=1 \
        nsys profile --force-overwrite=true --sample=none --cpuctxsw=none \
            --trace=cuda,nvtx,osrt,cublas --capture-range=cudaProfilerApi \
            --capture-range-end=stop --output="$base" \
            ./ds4-bench --cuda --cuda-tensor-parallel \
                --gpu-devices "$devices" --gpu-vram "$GPU_VRAM" \
                --model "$MODEL" --prompt-file "$PROMPT" \
                --ctx-start "$PROFILE_TOKENS" --ctx-max "$PROFILE_TOKENS" \
                --ctx-alloc "$((PROFILE_TOKENS+1))" \
                --step-incr "$PROFILE_TOKENS" --prefill-chunk "$PREFILL_CHUNK" \
                --gen-tokens 0 --csv "$base-benchmark.csv" \
                >"$base.log" 2>&1
    local rc=$?
    set -e
    cleanup_sampler
    if (( rc != 0 )); then
        tail -n 180 "$base.log" >&2 || true
        die "Nsight Systems capture failed for $label (exit $rc)"
    fi
    [[ -s $base.nsys-rep && -s $base-benchmark.csv ]] ||
        die "$label omitted its Nsight report or benchmark CSV"
    grep -Fq "CUDA EP forced pipeline split $STAGE_SPLIT/$((43-STAGE_SPLIT))" \
        "$base.log" || die "$label did not use the requested stage split"
    grep -Fq 'partner-classes=t256' "$base.log" ||
        die "$label did not use the fixed T256 partner policy"
    for route in '0->2 DIRECT' '2->0 DIRECT' '1->3 DIRECT' '3->1 DIRECT'; do
        grep -Fq "$route" "$base.log" ||
            die "$label lacks validated direct NVLink route $route"
    done
    nsys export --type sqlite --force-overwrite=true \
        --output "$base.sqlite" "$base.nsys-rep" \
        >"$base-export.log" 2>&1 || {
            cat "$base-export.log" >&2
            die "could not export $label to SQLite"
        }
    [[ -s $base.sqlite ]] || die "$label SQLite export is empty"
    for report in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum \
                  cuda_gpu_trace nvtx_sum nvtx_gpu_proj_sum; do
        nsys stats --report "$report" --format csv "$base.nsys-rep" \
            >"$base-$report.csv" 2>"$base-$report.log" || true
    done
    printf '%s\t%s\t%s\n' "$label" "$devices" "$base.sqlite" \
        >>"$OUTPUT_DIR/trace-map.tsv"
}

capture_trace current "$CURRENT_DEVICES"
capture_trace swapped "$SWAPPED_DEVICES"

phase=same-work-harness
printf 'trial\tslot\tscenario\tdevice\tlog\n' >"$OUTPUT_DIR/harness-runs.tsv"
for ((trial=1; trial<=HARNESS_TRIALS; trial++)); do
    if (( trial % 2 )); then devices=(0 3); else devices=(3 0); fi
    slot=0
    for device in "${devices[@]}"; do
        if (( trial % 2 )); then
            first=0; last=${#scenarios[@]}; step=1
        else
            first=$((${#scenarios[@]}-1)); last=-1; step=-1
        fi
        for ((idx=first; idx!=last; idx+=step)); do
            scenario=${scenarios[$idx]}
            slot=$((slot + 1))
            log="$OUTPUT_DIR/harness/$scenario-gpu$device-t$trial.log"
            printf 'Same-work harness trial=%d/%d slot=%d scenario=%s GPU=%d...\n' \
                "$trial" "$HARNESS_TRIALS" "$slot" "$scenario" "$device"
            "${clean[@]}" CUDA_VISIBLE_DEVICES="$device" \
                "DS4_PROFILE_REPEATS=$HARNESS_REPEATS" \
                ./tests/cuda_sm75_profile_harness "$scenario" \
                >"$log" 2>&1 || {
                    tail -n 120 "$log" >&2 || true
                    die "same-work harness failed for $scenario on GPU $device"
                }
            grep -Fq 'harness_status=ok' "$log" ||
                die "$log lacks exact harness validation"
            grep -Eq '^timed_per_call_ms=[0-9]' "$log" ||
                die "$log lacks a timed result"
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$trial" "$slot" "$scenario" "$device" "$log" \
                >>"$OUTPUT_DIR/harness-runs.tsv"
        done
    done
done

phase=summarize
python3 speed-bench/summarize-cuda-critical-path.py "$OUTPUT_DIR" \
    | tee "$OUTPUT_DIR/summary-stdout.txt"
for required in operation-attribution.csv stage-microbatch-device.csv \
                stage-device-summary.csv layer-device-summary.csv \
                partner-projection-summary.csv handoff-device-summary.csv \
                trace-summary.csv \
                same-work-gpu-summary.csv analysis.txt; do
    [[ -s $OUTPUT_DIR/$required ]] || die "summary omitted $required"
done

phase=complete
printf 'SM75 critical-path attribution audit complete: %s\n' "$OUTPUT_DIR"
