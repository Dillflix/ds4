#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the three-arm SM75 indexer production isolation:

  legacy-f32-wmma128       persistent F32 K cache, shipping WMMA128
  native-f16-wmma128       persistent F16 K cache, F16-input WMMA128
  native-f16-streaming64   persistent F16 K cache, streaming WMMA64

The run preserves the balanced dense-FP16 allocation but suppresses partner
projection execution and attention row splitting. Every arm must expose its
requested cache/dispatch path and produce byte-identical frontier logits.

Required environment:
  MODEL=/absolute/path/to/tagged-SM75-mixed-Q4-IQ2.gguf

Optional environment:
  PROMPT=...              default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  CTX_START=2048
  CTX_MAX=16384
  CTX_ALLOC=262273
  REPEATS=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  INDEXER_CONTROL_DIR=/absolute/output/directory
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
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${INDEXER_CONTROL_DIR:-$repo_dir/sm75-indexer-native-controls-$stamp}

[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "CTX_START:$CTX_START" \
            "CTX_MAX:$CTX_MAX" "CTX_ALLOC:$CTX_ALLOC" \
            "REPEATS:$REPEATS" "SKIP_BUILD:$SKIP_BUILD" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT > 0 && STAGE_SPLIT < 43 &&
   CTX_START > 0 && CTX_MAX >= CTX_START && CTX_ALLOC > CTX_MAX &&
   REPEATS >= 1 )) || die "invalid stage/context/repeat configuration"
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

for tool in awk basename cat cmp date dirname env find git grep make mkdir mv \
            nproc nvidia-smi python3 rm sort stat tail tar tee tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
git diff --quiet && git diff --cached --quiet ||
    die "tracked source changes are present; test an exact committed tree"
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{harness,production,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
finish() {
    status=$?
    trap - EXIT INT TERM HUP
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
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
targets=(ds4-bench tests/cuda_sm75_profile_harness
         tests/cuda_long_context_smoke tests/test_engine_mgpu_placement)
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
"${clean[@]}" ./tests/cuda_long_context_smoke \
    >"$OUTPUT_DIR/cuda-exactness.log" 2>&1 || {
        tail -n 220 "$OUTPUT_DIR/cuda-exactness.log" >&2
        die "CUDA exactness tests failed"
    }

phase=bounded-indexer-exactness
for operands in materialized streaming64; do
    if [[ $operands == materialized ]]; then
        tile=128
    else
        tile=64
    fi
    for n_comp in 8192 8191; do
        log="$OUTPUT_DIR/harness/$operands-ncomp-$n_comp.log"
        "${clean[@]}" CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" \
            DS4_PROFILE_INDEXER_TILE="$tile" \
            DS4_PROFILE_INDEXER_TOPK=monolithic \
            DS4_PROFILE_INDEXER_OPERANDS="$operands" \
            DS4_PROFILE_INDEXER_N_COMP="$n_comp" \
            DS4_PROFILE_REPEATS=0 \
            ./tests/cuda_sm75_profile_harness indexer-32k \
            >"$log" 2>&1 || {
                tail -n 180 "$log" >&2
                die "$operands exactness harness failed at n_comp=$n_comp"
            }
        for marker in 'score_validation=bit-exact' \
                      'topk_validation=exact-order-and-set' \
                      'harness_status=ok'; do
            grep -Fq "$marker" "$log" ||
                die "$operands n_comp=$n_comp lacks $marker"
        done
    done
done

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
    printf 'attention_row_split=disabled\npartner_projection_execution=suppressed\n'
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

common_env=(
    "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT"
    DS4_CUDA_PREFILL_PIPELINE=1
    DS4_CUDA_PREFILL_PIPELINE_MB=512
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    DS4_CUDA_Q8_T256_PLACEMENT=balanced
    DS4_CUDA_Q8_F16_PARTNER_CLASSES=t256
    DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=2048
    DS4_CUDA_NO_Q8_F16_PARTNER_EXECUTION=1
    DS4_CUDA_TP_PREFILL_ATTN_HEADS=0
    DS4_CUDA_TP_PREFILL_ATTN_ROWS=0
    DS4_CUDA_INDEXER_SCORE_AUDIT=1
)

variant_env() {
    case $1 in
        legacy-f32-wmma128)
            printf '%s\0' DS4_CUDA_NO_INDEXER_NATIVE_F16_CACHE=1
            ;;
        native-f16-wmma128)
            printf '%s\0' DS4_CUDA_NO_INDEXER_STREAMING64=1
            ;;
        native-f16-streaming64)
            ;;
        *) die "unknown variant: $1" ;;
    esac
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
    local variant=$1 log=$2 expected marker count dispatch
    case $variant in
        legacy-f32-wmma128)
            expected=wmma128
            marker=
            ! grep -Fq 'SM75 native F16 indexer cache' "$log" ||
                die "$variant unexpectedly enabled the native F16 cache"
            ;;
        native-f16-wmma128)
            expected=wmma128-f16-native
            marker='SM75 native F16 indexer cache with F16-input WMMA128 enabled'
            ;;
        native-f16-streaming64)
            expected=streaming64-native
            marker='SM75 native F16 indexer cache and streaming WMMA64 enabled'
            ;;
    esac
    [[ -z $marker ]] || grep -Fq "$marker" "$log" ||
        die "$variant did not expose its requested cache/scorer path"
    grep -Fq "CUDA EP forced pipeline split $STAGE_SPLIT/$((43-STAGE_SPLIT))" \
        "$log" || die "$variant did not use the fixed production split"
    grep -Fq 't256-placement=balanced' "$log" ||
        die "$variant missed balanced T256 placement"
    grep -Fq 'CUDA q8 fp16 benefit plan materialized 344/344 candidates' "$log" ||
        die "$variant did not preserve complete dense-FP16 admission"
    ! grep -Fq 'CUDA prefill attention query-row split enabled:' "$log" ||
        die "$variant unexpectedly enabled attention row splitting"
    count=$(audit_launches "$log" "$expected")
    (( count > 0 )) || die "$variant did not dispatch $expected"
    for dispatch in streaming64-native wmma128-f16-native wmma128 wmma64 wmma32 wmma16 generic; do
        [[ $dispatch == "$expected" ]] && continue
        count=$(audit_launches "$log" "$dispatch")
        (( count == 0 )) || die "$variant unexpectedly dispatched $dispatch"
    done
}

variants=(legacy-f32-wmma128 native-f16-wmma128 native-f16-streaming64)
phase=production-controls
printf 'repeat,slot,variant,csv,log,logits\n' >"$OUTPUT_DIR/production/runs.csv"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    rotation=$(((repeat - 1) % 3))
    order=("${variants[@]:rotation}" "${variants[@]:0:rotation}")
    slot=0
    for variant in "${order[@]}"; do
        slot=$((slot + 1))
        mapfile -d '' -t mode_env < <(variant_env "$variant")
        base="$OUTPUT_DIR/production/r${repeat}-s${slot}-$variant"
        logits="$base-logits"
        mkdir -p "$logits"
        printf 'Indexer control repeat=%d/%d slot=%d/3 variant=%s...\n' \
            "$repeat" "$REPEATS" "$slot" "$variant"
        "${clean[@]}" "${mode_env[@]}" "${common_env[@]}" \
            ./ds4-bench --cuda --cuda-tensor-parallel \
                --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                --model "$MODEL" --prompt-file "$PROMPT" \
                --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
                --ctx-alloc "$CTX_ALLOC" --step-mul 2 \
                --prefill-chunk 2048 --gen-tokens 0 --csv "$base.csv" \
                --dump-frontier-logits-dir "$logits" \
                >"$base.log" 2>&1 || {
                    tail -n 240 "$base.log" >&2
                    die "$variant production run failed"
                }
        [[ -s $base.csv ]] || die "$variant omitted benchmark CSV"
        validate_variant_log "$variant" "$base.log"
        printf '%s,%s,%s,%s,%s,%s\n' "$repeat" "$slot" "$variant" \
            "$base.csv" "$base.log" "$logits" >>"$OUTPUT_DIR/production/runs.csv"
    done

    baseline_logits=$(awk -F, -v r="$repeat" \
        '$1 == r && $3 == "legacy-f32-wmma128" {print $6}' \
        "$OUTPUT_DIR/production/runs.csv")
    mapfile -t baseline_files < <(find "$baseline_logits" -maxdepth 1 \
        -type f -printf '%f\n' | sort)
    [[ ${#baseline_files[@]} == "$expected_frontier_files" ]] ||
        die "repeat $repeat legacy frontier-logit inventory is incomplete"
    for variant in native-f16-wmma128 native-f16-streaming64; do
        candidate_logits=$(awk -F, -v r="$repeat" -v v="$variant" \
            '$1 == r && $3 == v {print $6}' "$OUTPUT_DIR/production/runs.csv")
        mapfile -t candidate_files < <(find "$candidate_logits" -maxdepth 1 \
            -type f -printf '%f\n' | sort)
        [[ "${baseline_files[*]}" == "${candidate_files[*]}" ]] ||
            die "repeat $repeat $variant frontier-logit inventory differs"
        for file in "${baseline_files[@]}"; do
            cmp -s "$baseline_logits/$file" "$candidate_logits/$file" ||
                die "repeat $repeat $variant frontier logits differ: $file"
        done
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
base = "legacy-f32-wmma128"
arms = ["native-f16-wmma128", "native-f16-streaming64"]
with (root / "production/summary.csv").open("w", newline="") as handle:
    out = csv.writer(handle)
    out.writerow(["ctx_tokens", "legacy_f32_wmma128_tps",
                  "native_f16_wmma128_tps", "native_f16_streaming64_tps",
                  "f16_wmma128_vs_legacy", "streaming64_vs_legacy",
                  "frontier_logits"])
    for ctx in expected:
        medians = {v: statistics.median(values[(v, ctx)])
                   for v in [base, *arms]}
        ratios = {}
        for arm in arms:
            ratios[arm] = statistics.median(
                paired[(r, arm, ctx)] / paired[(r, base, ctx)]
                for r in range(1, n + 1))
        out.writerow([ctx, *(f"{medians[v]:.3f}" for v in [base, *arms]),
                      f"{ratios[arms[0]]:.6f}", f"{ratios[arms[1]]:.6f}",
                      "bit-exact"])
PY
cat "$OUTPUT_DIR/production/summary.csv"

phase=complete
printf 'SM75 indexer native-control isolation complete: %s\n' "$OUTPUT_DIR"
