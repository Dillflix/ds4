#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run a fail-closed production A/B of the shipping SM75 indexer WMMA128 score
kernel versus the existing exact WMMA64 kernel.

Required environment:
  MODEL=/absolute/path/to/tagged-SM75-mixed-Q4-IQ2.gguf

Optional environment:
  PROMPT=...                 default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  CTX_START=2048
  CTX_MAX=32768
  CTX_ALLOC=262273
  REPEATS=3
  RUN_HARNESS=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  INDEXER_WMMA64_AB_DIR=/absolute/output/directory

Both variants use the production balanced-T256 and mirrored-KV attention
paths. Every score launch is audited, and every paired frontier logit file
must be byte-identical.
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
CTX_MAX=${CTX_MAX:-32768}
CTX_ALLOC=${CTX_ALLOC:-262273}
REPEATS=${REPEATS:-3}
RUN_HARNESS=${RUN_HARNESS:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${INDEXER_WMMA64_AB_DIR:-$repo_dir/sm75-indexer-wmma64-production-$stamp}

[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "CTX_START:$CTX_START" \
            "CTX_MAX:$CTX_MAX" "CTX_ALLOC:$CTX_ALLOC" \
            "REPEATS:$REPEATS" "RUN_HARNESS:$RUN_HARNESS" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT > 0 && STAGE_SPLIT < 43 &&
   CTX_START >= 2048 && CTX_START % 2048 == 0 &&
   CTX_MAX >= CTX_START && CTX_MAX % 2048 == 0 &&
   CTX_ALLOC > CTX_MAX && REPEATS >= 2 )) || die "invalid benchmark bounds"
[[ $CTX_START == 2048 && $CTX_MAX == 32768 && $CTX_ALLOC == 262273 ]] ||
    die "the production gate requires CTX_START=2048 CTX_MAX=32768 CTX_ALLOC=262273"
ctx_ratio=$((CTX_MAX / CTX_START))
(( CTX_MAX % CTX_START == 0 &&
   (ctx_ratio & (ctx_ratio - 1)) == 0 )) ||
    die "CTX_MAX/CTX_START must be an integer power of two"
for flag in RUN_HARNESS SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
for tool in awk basename cat cmp date dirname env find git grep make mkdir \
            nproc nvidia-smi python3 sort stat tail tar tee tr; do
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
        tar -C "$(dirname "$OUTPUT_DIR")" -czf "$archive" \
            "$(basename "$OUTPUT_DIR")" || status=1
        printf 'Archive to return: %s\n' "$archive"
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM HUP

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done

phase=build
targets=(ds4-bench tests/cuda_sm75_profile_harness tests/test_engine_mgpu_placement)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q "${targets[@]}" CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi
./tests/test_engine_mgpu_placement \
    >"$OUTPUT_DIR/placement-tests.log" 2>&1 || {
        tail -n 160 "$OUTPUT_DIR/placement-tests.log" >&2
        die "placement regression tests failed"
    }

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nmodel_hashing=disabled\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\nstage_split=%s/%s\nctx_start=%s\nctx_max=%s\nctx_alloc=%s\nrepeats=%s\n' \
        "$GPU_DEVICES" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))" \
        "$CTX_START" "$CTX_MAX" "$CTX_ALLOC" "$REPEATS"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,compute_cap \
        --format=csv
    printf '\ntopology:\n'
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
(( ${#gpu_ids[@]} == 4 )) || die "GPU_DEVICES must contain exactly four IDs"
declare -A seen_gpu=()
for gpu in "${gpu_ids[@]}"; do
    [[ $gpu =~ ^[0-9]+$ ]] || die "invalid physical GPU index: $gpu"
    [[ -z ${seen_gpu[$gpu]:-} ]] || die "GPU_DEVICES repeats physical GPU $gpu"
    seen_gpu[$gpu]=1
    compute_cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    [[ $compute_cap == 7.5 ]] ||
        die "physical GPU $gpu has compute capability ${compute_cap:-unknown}; SM75 is required"
done

if [[ $RUN_HARNESS == 1 ]]; then
    phase=bounded-exact-harness
    for tile in 128 64; do
        log="$OUTPUT_DIR/harness/wmma$tile.log"
        "${clean[@]}" CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" \
            DS4_PROFILE_INDEXER_TILE="$tile" \
            DS4_PROFILE_INDEXER_TOPK=monolithic \
            DS4_PROFILE_REPEATS=0 \
            ./tests/cuda_sm75_profile_harness indexer-32k \
            >"$log" 2>&1 || {
                tail -n 160 "$log" >&2
                die "WMMA$tile bounded indexer harness failed"
            }
        grep -Fq 'score_validation=bit-exact' "$log" ||
            die "WMMA$tile score output was not bit-exact"
        grep -Fq 'topk_validation=exact-order-and-set' "$log" ||
            die "WMMA$tile top-k output was not exact"
        grep -Fq 'harness_status=ok' "$log" ||
            die "WMMA$tile harness success marker is missing"
    done
fi

audit_count() {
    local log=$1 dispatch=$2 line count
    count=$(grep -Fc \
        "ds4: CUDA indexer score audit dispatch=$dispatch " "$log" || true)
    [[ $count == 1 ]] ||
        die "expected exactly one $dispatch audit summary in $log; found $count"
    line=$(grep -F \
        "ds4: CUDA indexer score audit dispatch=$dispatch " "$log")
    count=${line##*launches=}
    [[ $count =~ ^[0-9]+$ ]] ||
        die "invalid $dispatch audit count in $log: $count"
    printf '%s\n' "$count"
}

phase=production-ab
printf 'repeat,slot,variant,dispatches,direct_one_dispatches,csv,log,logits\n' \
    >"$OUTPUT_DIR/production/runs.csv"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    if (( repeat % 2 )); then
        variants=(wmma128 wmma64)
    else
        variants=(wmma64 wmma128)
    fi
    slot=0
    for variant in "${variants[@]}"; do
        slot=$((slot + 1))
        base="$OUTPUT_DIR/production/r${repeat}-s${slot}-$variant"
        logits="$base-logits"
        mkdir -p "$logits"
        variant_env=()
        expected_dispatch=$variant
        if [[ $variant == wmma64 ]]; then
            variant_env+=(DS4_CUDA_NO_INDEXER_WMMA128=1)
        fi
        printf 'Production indexer A/B repeat=%d/%d slot=%d variant=%s...\n' \
            "$repeat" "$REPEATS" "$slot" "$variant"
        "${clean[@]}" \
            "${variant_env[@]}" \
            DS4_CUDA_NO_INDEXER_NATIVE_F16_CACHE=1 \
            "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
            DS4_CUDA_PREFILL_PIPELINE=1 \
            DS4_CUDA_PREFILL_PIPELINE_MB=512 \
            DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
            DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=2048 \
            DS4_CUDA_TP_PREFILL_ATTN_HEADS=0 \
            DS4_CUDA_TP_PREFILL_ATTN_ROWS=1 \
            DS4_CUDA_INDEXER_SCORE_AUDIT=1 \
            ./ds4-bench --cuda --cuda-tensor-parallel \
                --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                --model "$MODEL" --prompt-file "$PROMPT" \
                --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
                --ctx-alloc "$CTX_ALLOC" --step-mul 2 \
                --prefill-chunk 2048 --gen-tokens 0 --csv "$base.csv" \
                --dump-frontier-logits-dir "$logits" \
                >"$base.log" 2>&1 || {
                    tail -n 200 "$base.log" >&2
                    die "$variant production run failed"
                }
        [[ -s $base.csv ]] || die "$variant omitted benchmark CSV"
        grep -Fq 't256-placement=balanced' "$base.log" ||
            die "$variant missed balanced T256 placement"
        dispatches=$(audit_count "$base.log" "$expected_dispatch")
        (( dispatches > 0 )) || die "$variant did not dispatch $expected_dispatch"
        direct_dispatches=$(audit_count "$base.log" direct-one)
        for unexpected in streaming64-native wmma128-f16-native wmma128 wmma64 wmma32 wmma16 generic; do
            [[ $unexpected == "$expected_dispatch" ]] && continue
            unexpected_count=$(audit_count "$base.log" "$unexpected")
            (( unexpected_count == 0 )) ||
                die "$variant dispatched unexpected $unexpected $unexpected_count times"
        done
        printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$repeat" "$slot" "$variant" "$dispatches" "$direct_dispatches" \
            "$base.csv" "$base.log" "$logits" \
            >>"$OUTPUT_DIR/production/runs.csv"
    done

    dispatch128=$(awk -F, -v r="$repeat" \
        '$1 == r && $3 == "wmma128" {print $4}' \
        "$OUTPUT_DIR/production/runs.csv")
    dispatch64=$(awk -F, -v r="$repeat" \
        '$1 == r && $3 == "wmma64" {print $4}' \
        "$OUTPUT_DIR/production/runs.csv")
    [[ $dispatch128 == "$dispatch64" ]] ||
        die "repeat $repeat dispatch counts differ: $dispatch128 versus $dispatch64"
    direct128=$(awk -F, -v r="$repeat" \
        '$1 == r && $3 == "wmma128" {print $5}' \
        "$OUTPUT_DIR/production/runs.csv")
    direct64=$(awk -F, -v r="$repeat" \
        '$1 == r && $3 == "wmma64" {print $5}' \
        "$OUTPUT_DIR/production/runs.csv")
    [[ $direct128 == "$direct64" ]] ||
        die "repeat $repeat direct-one counts differ: $direct128 versus $direct64"

    logits128=$(awk -F, -v r="$repeat" \
        '$1 == r && $3 == "wmma128" {print $8}' \
        "$OUTPUT_DIR/production/runs.csv")
    logits64=$(awk -F, -v r="$repeat" \
        '$1 == r && $3 == "wmma64" {print $8}' \
        "$OUTPUT_DIR/production/runs.csv")
    mapfile -t files128 < <(find "$logits128" -maxdepth 1 -type f -printf '%f\n' | sort)
    mapfile -t files64 < <(find "$logits64" -maxdepth 1 -type f -printf '%f\n' | sort)
    expected_files=(
        frontier_002048.logits.f32 frontier_002048.logits.json
        frontier_004096.logits.f32 frontier_004096.logits.json
        frontier_008192.logits.f32 frontier_008192.logits.json
        frontier_016384.logits.f32 frontier_016384.logits.json
        frontier_032768.logits.f32 frontier_032768.logits.json
    )
    [[ "${files128[*]}" == "${expected_files[*]}" ]] ||
        die "repeat $repeat WMMA128 logits inventory is incomplete"
    [[ "${files64[*]}" == "${expected_files[*]}" ]] ||
        die "repeat $repeat WMMA64 logits inventory is incomplete"
    [[ "${files128[*]}" == "${files64[*]}" ]] ||
        die "repeat $repeat logits inventory differs"
    for file in "${files128[@]}"; do
        cmp -s "$logits128/$file" "$logits64/$file" ||
            die "repeat $repeat logits differ: $file"
    done
done

for variant in wmma128 wmma64; do
    mapfile -t stable_counts < <(awk -F, -v variant="$variant" \
        '$3 == variant {print $4}' "$OUTPUT_DIR/production/runs.csv" | sort -u)
    [[ ${#stable_counts[@]} == 1 ]] ||
        die "$variant dispatch count changed across repeats: ${stable_counts[*]}"
done

python3 - "$OUTPUT_DIR" <<'PY'
import csv
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
runs = list(csv.DictReader((root / "production/runs.csv").open()))
values, paired = {}, {}
expected_contexts = [2048, 4096, 8192, 16384, 32768]
for run in runs:
    run_contexts = []
    with pathlib.Path(run["csv"]).open() as handle:
        for row in csv.DictReader(handle):
            ctx = int(row["ctx_tokens"])
            tps = float(row["prefill_tps"])
            run_contexts.append(ctx)
            values.setdefault((run["variant"], ctx), []).append(tps)
            paired[(int(run["repeat"]), run["variant"], ctx)] = tps
    if run_contexts != expected_contexts:
        raise SystemExit(
            f'{run["variant"]} repeat {run["repeat"]} frontier mismatch: '
            f'{run_contexts} != {expected_contexts}'
        )
contexts = sorted({ctx for _, ctx in values})
if contexts != expected_contexts:
    raise SystemExit(f"frontier inventory mismatch: {contexts} != {expected_contexts}")
n_repeats = max(int(run["repeat"]) for run in runs)
with (root / "production/summary.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["ctx_tokens", "wmma128_median_tps", "wmma64_median_tps",
                     "paired_median_speedup", "change_pct", "logits"])
    for ctx in contexts:
        ratios = [paired[(repeat, "wmma64", ctx)] /
                  paired[(repeat, "wmma128", ctx)]
                  for repeat in range(1, n_repeats + 1)]
        speedup = statistics.median(ratios)
        writer.writerow([
            ctx,
            f'{statistics.median(values[("wmma128", ctx)]):.3f}',
            f'{statistics.median(values[("wmma64", ctx)]):.3f}',
            f'{speedup:.6f}',
            f'{(speedup - 1.0) * 100.0:.3f}',
            "bit-exact",
        ])
PY
cat "$OUTPUT_DIR/production/summary.csv"

phase=complete
printf 'SM75 production indexer WMMA64 A/B complete: %s\n' "$OUTPUT_DIR"
