#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run a four-arm SM75 cross-device production isolation while holding the native
F16 indexer cache and streaming64 scorer constant:

  neither        partner T256 execution off, attention row split off
  partner-only   partner T256 execution on,  attention row split off
  rows-only      partner T256 execution off, attention row split on
  both           partner T256 execution on,  attention row split on

The arms run in that fixed safety order. Before every arm, the runner persists
the active-arm identity and starts one-second per-GPU telemetry. If the host
resets, the unarchived output directory retains the last active arm and the
telemetry collected before failure.

Required environment:
  MODEL=/absolute/path/to/tagged-SM75-mixed-Q4-IQ2.gguf

Optional environment:
  PROMPT=...                 default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  CTX_START=2048
  CTX_MAX=16384
  CTX_ALLOC=262273
  REPEATS=1
  TELEMETRY_INTERVAL_MS=1000
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  XDEV_ISOLATION_DIR=/absolute/output/directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
MODEL=${MODEL:-}
[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name the existing absolute tagged mixed-Q4/IQ2 model"
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
CTX_START=${CTX_START:-2048}
CTX_MAX=${CTX_MAX:-16384}
CTX_ALLOC=${CTX_ALLOC:-262273}
REPEATS=${REPEATS:-1}
TELEMETRY_INTERVAL_MS=${TELEMETRY_INTERVAL_MS:-1000}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${XDEV_ISOLATION_DIR:-$repo_dir/sm75-xdev-feature-isolation-$stamp}

[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "CTX_START:$CTX_START" \
            "CTX_MAX:$CTX_MAX" "CTX_ALLOC:$CTX_ALLOC" \
            "REPEATS:$REPEATS" \
            "TELEMETRY_INTERVAL_MS:$TELEMETRY_INTERVAL_MS" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT > 0 && STAGE_SPLIT < 43 &&
   CTX_START > 0 && CTX_MAX >= CTX_START && CTX_ALLOC > CTX_MAX &&
   REPEATS >= 1 && TELEMETRY_INTERVAL_MS >= 100 )) ||
    die "invalid stage/context/repeat/telemetry configuration"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done

expected_contexts=()
ctx=$CTX_START
while (( ctx <= CTX_MAX )); do
    expected_contexts+=("$ctx")
    (( ctx > CTX_MAX / 2 )) && break
    ctx=$((ctx * 2))
done
[[ ${expected_contexts[-1]} == "$CTX_MAX" ]] ||
    die "CTX_MAX must be reachable from CTX_START by doubling"
expected_frontier_files=$((2 * ${#expected_contexts[@]}))

for tool in awk basename cat cmp date dirname env find git grep kill make \
            mkdir mv nproc nvidia-smi python3 rm sleep sort stat sync tail tar \
            tee tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
git diff --quiet && git diff --cached --quiet ||
    die "tracked source changes are present; test an exact committed tree"
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{production,provenance,telemetry}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
printf 'Diagnostic directory: %s\n' "$OUTPUT_DIR"

phase=initialization
telemetry_pid=
stop_telemetry() {
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
    stop_telemetry
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
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
    compute_cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    [[ $compute_cap == 7.5 ]] ||
        die "physical GPU $gpu has compute capability ${compute_cap:-unknown}; SM75 is required"
done

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

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nmodel_hashing=disabled\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\nstage_split=%s/%s\n' \
        "$GPU_DEVICES" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))"
    printf 'ctx_start=%s\nctx_max=%s\nctx_alloc=%s\nrepeats=%s\n' \
        "$CTX_START" "$CTX_MAX" "$CTX_ALLOC" "$REPEATS"
    printf 'indexer_cache=native-f16\nindexer_scorer=streaming64\n'
    printf 'arm_order=neither,partner-only,rows-only,both\n'
    printf 'telemetry_interval_ms=%s\n' "$TELEMETRY_INTERVAL_MS"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,compute_cap \
        --format=csv
    printf '\nbar1_inventory:\n'
    for gpu in "${gpu_ids[@]}"; do
        bar1_total=$(nvidia-smi -i "$gpu" -q 2>/dev/null | awk '
            /^[[:space:]]*BAR1 Memory Usage/ { in_bar1 = 1; next }
            in_bar1 && /^[[:space:]]*Total[[:space:]]*:/ {
                print $3 " " $4
                exit
            }
        ' || true)
        printf 'gpu=%s bar1_total=%s\n' "$gpu" "${bar1_total:-unavailable}"
    done
    printf '\ntopology:\n'
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

printf 'timestamp_utc\trepeat\tslot\tvariant\tpartner\trows\tstatus\n' \
    >"$OUTPUT_DIR/run-journal.tsv"

start_telemetry() {
    local output=$1
    nvidia-smi \
        --query-gpu=timestamp,index,pci.bus_id,pstate,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.free \
        --format=csv -lms "$TELEMETRY_INTERVAL_MS" >"$output" 2>&1 &
    telemetry_pid=$!
}

record_arm_state() {
    local repeat=$1 slot=$2 variant=$3 partner=$4 rows=$5 status=$6
    local now
    now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$now" "$repeat" "$slot" "$variant" "$partner" "$rows" "$status" \
        >>"$OUTPUT_DIR/run-journal.tsv"
    printf 'timestamp_utc=%s\nrepeat=%s\nslot=%s\nvariant=%s\npartner=%s\nrows=%s\nstatus=%s\n' \
        "$now" "$repeat" "$slot" "$variant" "$partner" "$rows" "$status" \
        >"$OUTPUT_DIR/active-arm.txt"
    sync "$OUTPUT_DIR/run-journal.tsv" "$OUTPUT_DIR/active-arm.txt" \
        2>/dev/null || sync
}

audit_launches() {
    local log=$1 dispatch=$2 line
    line=$(grep -F "ds4: CUDA indexer score audit dispatch=$dispatch " "$log" || true)
    [[ $(printf '%s\n' "$line" | grep -c .) == 1 ]] ||
        die "expected one $dispatch audit record in $log"
    line=${line##*launches=}
    [[ $line =~ ^[0-9]+$ ]] || die "invalid $dispatch audit count in $log"
    printf '%s\n' "$line"
}

validate_variant_log() {
    local variant=$1 log=$2 partner=$3 rows=$4 count dispatch line calls
    grep -Fq "CUDA EP forced pipeline split $STAGE_SPLIT/$((43-STAGE_SPLIT))" \
        "$log" || die "$variant did not use the fixed production split"
    grep -Fq 'SM75 native F16 indexer cache and streaming WMMA64 enabled' \
        "$log" || die "$variant did not keep native F16 streaming64 enabled"
    grep -Fq 't256-placement=balanced' "$log" ||
        die "$variant missed balanced T256 placement"
    grep -Fq 'CUDA q8 fp16 benefit plan materialized 344/344 candidates' "$log" ||
        die "$variant did not preserve complete dense-FP16 admission"
    ! grep -Fq 'required but unavailable' "$log" ||
        die "$variant encountered a forbidden row-split fallback"

    count=$(audit_launches "$log" streaming64-native)
    (( count > 0 )) || die "$variant did not dispatch streaming64-native"
    for dispatch in wmma128-f16-native wmma128 wmma64 wmma32 wmma16 generic; do
        count=$(audit_launches "$log" "$dispatch")
        (( count == 0 )) || die "$variant unexpectedly dispatched $dispatch"
    done

    if [[ $partner == 1 ]]; then
        grep -Fq 'CUDA q8 partner execution enabled:' "$log" ||
            die "$variant did not enable partner execution"
        line=$(grep -F 'CUDA q8 fp16 partner summary:' "$log" | tail -n 1 || true)
        [[ -n $line ]] || die "$variant omitted the partner execution summary"
        calls=${line##*calls=}; calls=${calls%% *}
        [[ $calls =~ ^[0-9]+$ ]] && (( calls > 0 )) ||
            die "$variant reported an invalid partner call count"
        ! grep -Fq 'partner execution-only diagnostic:' "$log" ||
            die "$variant unexpectedly suppressed partner execution"
    else
        grep -Fq 'partner execution-only diagnostic:' "$log" ||
            die "$variant did not expose suppressed partner calls"
        ! grep -Fq 'CUDA q8 fp16 partner summary:' "$log" ||
            die "$variant unexpectedly executed partner projections"
        ! grep -Fq 'CUDA q8 partner execution enabled:' "$log" ||
            die "$variant unexpectedly enabled partner execution"
    fi

    if [[ $rows == 1 ]]; then
        [[ $(grep -Fc 'CUDA prefill attention query-row split enabled:' "$log") == 2 ]] ||
            die "$variant did not enable both attention row-split pairs"
        grep -Fq 'dispatch=split kind=indexed' "$log" ||
            die "$variant omitted indexed row-split dispatches"
        grep -Fq 'dispatch=split kind=mixed' "$log" ||
            die "$variant omitted mixed row-split dispatches"
    else
        ! grep -Fq 'CUDA prefill attention query-row split enabled:' "$log" ||
            die "$variant unexpectedly enabled attention row splitting"
        ! grep -Fq 'dispatch=split' "$log" ||
            die "$variant unexpectedly dispatched attention row splitting"
    fi
}

common_env=(
    "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT"
    DS4_CUDA_PREFILL_PIPELINE=1
    DS4_CUDA_PREFILL_PIPELINE_MB=512
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=2048
    DS4_CUDA_TP_PREFILL_ATTN_HEADS=0
    DS4_CUDA_TP_PREFILL_ATTN_ROWS_AUDIT=1
    DS4_CUDA_INDEXER_SCORE_AUDIT=1
)

variants=(neither partner-only rows-only both)
phase=production-isolation
printf 'repeat,slot,variant,partner,rows,csv,log,logits,telemetry\n' \
    >"$OUTPUT_DIR/production/runs.csv"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    no_partner_logits=
    partner_logits=
    slot=0
    for variant in "${variants[@]}"; do
        slot=$((slot + 1))
        partner=0; rows=0
        [[ $variant == partner-only || $variant == both ]] && partner=1
        [[ $variant == rows-only || $variant == both ]] && rows=1
        mode_env=("DS4_CUDA_TP_PREFILL_ATTN_ROWS=$rows")
        if [[ $partner == 0 ]]; then
            mode_env+=(DS4_CUDA_NO_Q8_F16_PARTNER_EXECUTION=1)
        fi
        base="$OUTPUT_DIR/production/r${repeat}-s${slot}-$variant"
        logits="$base-logits"
        telemetry="$OUTPUT_DIR/telemetry/r${repeat}-s${slot}-$variant.csv"
        mkdir -p "$logits"
        printf 'Cross-device isolation repeat=%d/%d slot=%d/4 variant=%s...\n' \
            "$repeat" "$REPEATS" "$slot" "$variant"
        record_arm_state "$repeat" "$slot" "$variant" "$partner" "$rows" starting
        start_telemetry "$telemetry"
        if "${clean[@]}" "${mode_env[@]}" "${common_env[@]}" \
                ./ds4-bench --cuda --cuda-tensor-parallel \
                    --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                    --model "$MODEL" --prompt-file "$PROMPT" \
                    --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
                    --ctx-alloc "$CTX_ALLOC" --step-mul 2 \
                    --prefill-chunk 2048 --gen-tokens 0 --csv "$base.csv" \
                    --dump-frontier-logits-dir "$logits" \
                    >"$base.log" 2>&1; then
            run_status=0
        else
            run_status=$?
        fi
        stop_telemetry
        if (( run_status != 0 )); then
            record_arm_state "$repeat" "$slot" "$variant" "$partner" "$rows" \
                "failed-exit-$run_status"
            tail -n 260 "$base.log" >&2 || true
            die "$variant production run failed with exit $run_status"
        fi
        [[ -s $telemetry && $(grep -c . "$telemetry") -ge 2 ]] ||
            die "$variant did not produce usable GPU telemetry"
        [[ -s $base.csv ]] || die "$variant omitted benchmark CSV"
        validate_variant_log "$variant" "$base.log" "$partner" "$rows"

        mapfile -t candidate_files < <(find "$logits" -maxdepth 1 \
            -type f -printf '%f\n' | sort)
        [[ ${#candidate_files[@]} == "$expected_frontier_files" ]] ||
            die "repeat $repeat $variant frontier-logit inventory is incomplete"
        case $variant in
            neither)
                no_partner_logits=$logits
                no_partner_files=("${candidate_files[@]}")
                ;;
            partner-only)
                partner_logits=$logits
                partner_files=("${candidate_files[@]}")
                [[ "${no_partner_files[*]}" == "${partner_files[*]}" ]] ||
                    die "repeat $repeat partner arithmetic changed the logits inventory"
                partner_arithmetic=identical
                for file in "${no_partner_files[@]}"; do
                    if ! cmp -s "$no_partner_logits/$file" "$partner_logits/$file"; then
                        partner_arithmetic=different
                        break
                    fi
                done
                printf 'repeat=%s\nneither_vs_partner_only=%s\ncomparison_gate=informational\n' \
                    "$repeat" "$partner_arithmetic" \
                    >"$OUTPUT_DIR/production/r${repeat}-partner-arithmetic.txt"
                ;;
            rows-only)
                [[ "${no_partner_files[*]}" == "${candidate_files[*]}" ]] ||
                    die "repeat $repeat rows-only frontier-logit inventory differs"
                for file in "${no_partner_files[@]}"; do
                    cmp -s "$no_partner_logits/$file" "$logits/$file" ||
                        die "repeat $repeat rows-only frontier logits differ: $file"
                done
                ;;
            both)
                [[ "${partner_files[*]}" == "${candidate_files[*]}" ]] ||
                    die "repeat $repeat both frontier-logit inventory differs"
                for file in "${partner_files[@]}"; do
                    cmp -s "$partner_logits/$file" "$logits/$file" ||
                        die "repeat $repeat both frontier logits differ: $file"
                done
                ;;
        esac
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$repeat" "$slot" "$variant" "$partner" "$rows" \
            "$base.csv" "$base.log" "$logits" "$telemetry" \
            >>"$OUTPUT_DIR/production/runs.csv"
        record_arm_state "$repeat" "$slot" "$variant" "$partner" "$rows" completed
    done
done

python3 - "$OUTPUT_DIR" "$CTX_START" "$CTX_MAX" <<'PY'
import csv, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
start, maximum = map(int, sys.argv[2:])
expected, ctx = [], start
while ctx <= maximum:
    expected.append(ctx)
    ctx *= 2
runs = list(csv.DictReader((root / "production/runs.csv").open()))
variants = ["neither", "partner-only", "rows-only", "both"]
values, paired = {}, {}
for run in runs:
    rows = list(csv.DictReader(pathlib.Path(run["csv"]).open()))
    contexts = [int(row["ctx_tokens"]) for row in rows]
    if contexts != expected:
        raise SystemExit(f"frontier mismatch for {run['variant']}: {contexts}")
    for row in rows:
        key = (run["variant"], int(row["ctx_tokens"]))
        value = float(row["prefill_tps"])
        values.setdefault(key, []).append(value)
        paired[(int(run["repeat"]), *key)] = value
n = max(int(run["repeat"]) for run in runs)
with (root / "production/summary.csv").open("w", newline="") as handle:
    out = csv.writer(handle)
    out.writerow(["ctx_tokens", "neither_tps", "partner_only_tps",
                  "rows_only_tps", "both_tps", "partner_vs_neither",
                  "rows_vs_neither", "both_vs_neither",
                  "interaction_ratio", "row_split_logits"])
    for ctx in expected:
        medians = {v: statistics.median(values[(v, ctx)]) for v in variants}
        ratios = {}
        for arm in variants[1:]:
            ratios[arm] = statistics.median(
                paired[(r, arm, ctx)] / paired[(r, "neither", ctx)]
                for r in range(1, n + 1))
        interaction = statistics.median(
            paired[(r, "both", ctx)] * paired[(r, "neither", ctx)] /
            (paired[(r, "partner-only", ctx)] * paired[(r, "rows-only", ctx)])
            for r in range(1, n + 1))
        out.writerow([ctx, *(f"{medians[v]:.3f}" for v in variants),
                      f"{ratios['partner-only']:.6f}",
                      f"{ratios['rows-only']:.6f}",
                      f"{ratios['both']:.6f}", f"{interaction:.6f}",
                      "bit-exact-within-partner-state"])
PY
cat "$OUTPUT_DIR/production/summary.csv"

phase=complete
printf 'SM75 cross-device feature isolation complete: %s\n' "$OUTPUT_DIR"
