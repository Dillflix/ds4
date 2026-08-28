#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run an exact-output production sweep for SM75 mirrored-KV query-row split.

Required environment:
  MODEL=/absolute/path/to/tagged-SM75-mixed-Q4-IQ2.gguf

Optional environment:
  PROMPT=...                 default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  CTX_START=2048
  CTX_MAX=32768
  CTX_TOKENS=             compatibility shorthand setting start=max
  CTX_ALLOC=CTX_MAX+1           preserves complete dense-F16 admission
  REPEATS=3
  RUN_HARNESS=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  ROWSPLIT_PRODUCTION_DIR=/absolute/output/directory

The candidate is fail-closed for indexed and nonzero-prefix mixed attention.
Every split call is audited. Initial static-mixed/raw-only work is outside the
first candidate scope and remains unchanged in both arms.
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
CTX_START=${CTX_START:-${CTX_TOKENS:-2048}}
CTX_MAX=${CTX_MAX:-${CTX_TOKENS:-32768}}
CTX_ALLOC=${CTX_ALLOC:-$((CTX_MAX + 1))}
REPEATS=${REPEATS:-3}
RUN_HARNESS=${RUN_HARNESS:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${ROWSPLIT_PRODUCTION_DIR:-$repo_dir/sm75-attention-rowsplit-production-$stamp}

[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "CTX_START:$CTX_START" \
            "CTX_MAX:$CTX_MAX" \
            "CTX_ALLOC:$CTX_ALLOC" "REPEATS:$REPEATS" \
            "RUN_HARNESS:$RUN_HARNESS" "SKIP_BUILD:$SKIP_BUILD" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT == 22 &&
   CTX_START >= 2048 && CTX_START % 2048 == 0 &&
   CTX_MAX >= CTX_START && CTX_MAX % 2048 == 0 &&
   CTX_ALLOC > CTX_MAX && REPEATS >= 2 )) || die "invalid benchmark bounds"
ctx_ratio=$((CTX_MAX / CTX_START))
(( CTX_MAX % CTX_START == 0 &&
   (ctx_ratio & (ctx_ratio - 1)) == 0 )) ||
    die "CTX_MAX/CTX_START must be an integer power of two"
for flag in RUN_HARNESS SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
for tool in awk basename cat cmp date dirname env find git grep make mkdir nproc \
            nvidia-smi python3 sort stat tail tar tee; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
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
targets=(ds4-bench tests/cuda_attention_rowsplit_xdev tests/test_engine_mgpu_placement)
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

if [[ $RUN_HARNESS == 1 ]]; then
    phase=bounded-exact-harness
    for pair in "${gpu_ids[0]},${gpu_ids[2]}" "${gpu_ids[1]},${gpu_ids[3]}"; do
        name=${pair/,/-}
        log="$OUTPUT_DIR/harness/pair-$name.log"
        : >"$log"
        for scenario in mixed indexed; do
            CUDA_VISIBLE_DEVICES=$pair DS4_ROWSPLIT_REPEATS=3 \
                ./tests/cuda_attention_rowsplit_xdev "$scenario" \
                >>"$log" 2>&1 || {
                    tail -n 160 "$log" >&2
                    die "row-split $scenario harness failed for physical pair $pair"
                }
        done
        [[ $(grep -Fc 'validation=bit-exact-nonzero' \
            "$log") == 2 ]] ||
            die "pair $pair did not validate both mixed and indexed paths"
    done
fi

phase=production-ab
printf 'repeat,slot,variant,csv,log,logits\n' >"$OUTPUT_DIR/production/runs.csv"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    if (( repeat % 2 )); then variants=(home rows); else variants=(rows home); fi
    slot=0
    for variant in "${variants[@]}"; do
        slot=$((slot + 1))
        rows=0; [[ $variant == rows ]] && rows=1
        base="$OUTPUT_DIR/production/r${repeat}-s${slot}-$variant"
        logits="$base-logits"
        mkdir -p "$logits"
        printf 'Production attention A/B repeat=%d/%d slot=%d variant=%s...\n' \
            "$repeat" "$REPEATS" "$slot" "$variant"
        "${clean[@]}" \
            "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
            DS4_CUDA_PREFILL_PIPELINE=1 \
            DS4_CUDA_PREFILL_PIPELINE_MB=512 \
            DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
            DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=2048 \
            DS4_CUDA_TP_PREFILL_ATTN_HEADS=0 \
            "DS4_CUDA_TP_PREFILL_ATTN_ROWS=$rows" \
            DS4_CUDA_TP_PREFILL_ATTN_ROWS_AUDIT=1 \
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
        grep -Fq 't256-placement=stage-aware' "$base.log" ||
            die "$variant missed fixed-22/21 stage-aware dense placement"
        grep -Fq 'dense-placement=stage-aware-fixed-22-21' "$base.log" ||
            die "$variant missed the fixed-22/21 dense placement marker"
        grep -Fq 'materialized 344/344 candidates' "$base.log" ||
            die "$variant did not retain complete dense-F16 admission"
        ! grep -Fq 'required but unavailable' "$base.log" ||
            die "$variant encountered a forbidden row-split fallback"
        if [[ $variant == rows ]]; then
            grep -Fq 'dispatch=split kind=indexed' "$base.log" ||
                die "row candidate omitted indexed split dispatch"
            grep -Fq 'dispatch=split kind=mixed' "$base.log" ||
                die "row candidate omitted mixed split dispatch"
            [[ $(grep -Fc 'CUDA prefill indexer score/top-k row split enabled:' \
                    "$base.log") == 2 ]] ||
                die "row candidate did not split indexer score/top-k on both pairs"
        else
            ! grep -Fq 'dispatch=split' "$base.log" ||
                die "home control unexpectedly dispatched row splitting"
            ! grep -Fq 'CUDA prefill indexer score/top-k row split enabled:' \
                    "$base.log" ||
                die "home control unexpectedly split indexer score/top-k"
        fi
        printf '%s,%s,%s,%s,%s,%s\n' "$repeat" "$slot" "$variant" \
            "$base.csv" "$base.log" "$logits" \
            >>"$OUTPUT_DIR/production/runs.csv"
    done

    home_logits=$(awk -F, -v r="$repeat" '$1 == r && $3 == "home" {print $6}' \
        "$OUTPUT_DIR/production/runs.csv")
    rows_logits=$(awk -F, -v r="$repeat" '$1 == r && $3 == "rows" {print $6}' \
        "$OUTPUT_DIR/production/runs.csv")
    mapfile -t home_files < <(find "$home_logits" -maxdepth 1 -type f -printf '%f\n' | sort)
    mapfile -t rows_files < <(find "$rows_logits" -maxdepth 1 -type f -printf '%f\n' | sort)
    [[ ${#home_files[@]} -gt 0 && "${home_files[*]}" == "${rows_files[*]}" ]] ||
        die "repeat $repeat logits inventory differs"
    for file in "${home_files[@]}"; do
        cmp -s "$home_logits/$file" "$rows_logits/$file" ||
            die "repeat $repeat logits differ: $file"
    done
done

python3 - "$OUTPUT_DIR" <<'PY'
import csv, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
runs = list(csv.DictReader((root / "production/runs.csv").open()))
vals, paired = {}, {}
for run in runs:
    for row in csv.DictReader(pathlib.Path(run["csv"]).open()):
        ctx = int(row["ctx_tokens"])
        tps = float(row["prefill_tps"])
        vals.setdefault((run["variant"], ctx), []).append(tps)
        paired[(int(run["repeat"]), run["variant"], ctx)] = tps
contexts = sorted({ctx for _, ctx in vals})
n_repeats = max(int(x["repeat"]) for x in runs)
with (root / "production/summary.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ctx_tokens", "home_median_tps", "rows_median_tps",
                "paired_median_speedup", "change_pct", "logits"])
    for ctx in contexts:
        ratios = [paired[(r, "rows", ctx)] / paired[(r, "home", ctx)]
                  for r in range(1, n_repeats + 1)]
        speed = statistics.median(ratios)
        w.writerow([ctx,
                    f'{statistics.median(vals[("home", ctx)]):.3f}',
                    f'{statistics.median(vals[("rows", ctx)]):.3f}',
                    f'{speed:.6f}', f'{(speed - 1.0) * 100.0:.3f}',
                    "bit-exact"])
PY
cat "$OUTPUT_DIR/production/summary.csv"

phase=complete
printf 'SM75 production attention row-split A/B complete: %s\n' "$OUTPUT_DIR"
