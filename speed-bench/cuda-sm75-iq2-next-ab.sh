#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Measure the accepted SM75 IQ2 residual-tail policy:
  Production tail4 rollback versus one tail8 for residuals 1..8.

Required environment:
  MODEL=/absolute/path/to/tagged-SM75-mixed-Q4-IQ2.gguf

Optional environment:
  PROMPT=...                 default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  CTX_START=2048
  CTX_MAX=8192
  CTX_ALLOC=262273
  REPEATS=3
  PROFILE_GPU=0
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  IQ2_AUDIT_DIR=/absolute/output/directory
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
CTX_MAX=${CTX_MAX:-8192}
CTX_ALLOC=${CTX_ALLOC:-262273}
REPEATS=${REPEATS:-3}
PROFILE_GPU=${PROFILE_GPU:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${IQ2_AUDIT_DIR:-$repo_dir/sm75-iq2-next-$stamp}

[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "CTX_START:$CTX_START" \
            "CTX_MAX:$CTX_MAX" "CTX_ALLOC:$CTX_ALLOC" \
            "REPEATS:$REPEATS" "PROFILE_GPU:$PROFILE_GPU" \
            "SKIP_BUILD:$SKIP_BUILD" \
            "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT > 0 && STAGE_SPLIT < 43 && CTX_START >= 2048 &&
   CTX_MAX >= CTX_START && CTX_ALLOC > CTX_MAX && REPEATS >= 2 )) ||
    die "invalid benchmark bounds"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
for tool in awk basename cat date dirname env git grep id make mkdir nproc \
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
targets=(ds4-bench tests/cuda_sm75_profile_harness)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        2>&1 | tee "$OUTPUT_DIR/build.log"
else
    make -q "${targets[@]}" CUDA_ARCH=sm_75 ||
        die "SKIP_BUILD=1 found stale targets"
fi

phase=manifest
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model=%s\nmodel_bytes=%s\nmodel_hashing=disabled\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\nstage_split=%s/%s\nctx=%s..%s\nctx_alloc=%s\n' \
        "$GPU_DEVICES" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))" \
        "$CTX_START" "$CTX_MAX" "$CTX_ALLOC"
    printf 'production_repeats=%s\n' "$REPEATS"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,compute_cap \
        --format=csv
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

phase=tail-exactness
for scenario in hybrid-iq2-q4-early hybrid-iq2-q4-late; do
    log="$OUTPUT_DIR/harness/$scenario-tail8-exact.log"
    printf 'IQ2 tail exactness harness: %s...\n' "$scenario"
    "${clean[@]}" CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        DS4_PROFILE_SCALAR_TARGET=iq2-tile16 DS4_PROFILE_SCALAR=1 \
        DS4_PROFILE_IQ2_TAIL8_ALL=1 \
        DS4_PROFILE_REPEATS=0 \
        ./tests/cuda_sm75_profile_harness "$scenario" >"$log" 2>&1 || {
            tail -n 120 "$log" >&2; die "$scenario tail8 exactness failed";
        }
    grep -Fq 'iq2_candidate_reference_validation=bit-exact' "$log" ||
        die "$scenario tail8 omitted baseline comparison"
done

phase=production-tail-ab
printf 'repeat,slot,variant,csv,log\n' >"$OUTPUT_DIR/production/runs.csv"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    if (( repeat % 2 )); then variants=(tail4 tail8); else variants=(tail8 tail4); fi
    slot=0
    for variant in "${variants[@]}"; do
        slot=$((slot + 1)); tail8=0; [[ $variant == tail8 ]] && tail8=1
        base="$OUTPUT_DIR/production/r${repeat}-s${slot}-$variant"
        printf 'Production IQ2 tail A/B repeat=%d/%d slot=%d variant=%s...\n' \
            "$repeat" "$REPEATS" "$slot" "$variant"
        "${clean[@]}" \
            "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
            DS4_CUDA_PREFILL_PIPELINE=1 \
            DS4_CUDA_PREFILL_PIPELINE_MB=512 \
            DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
            DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=2048 \
            "DS4_CUDA_MOE_IQ2_TAIL8_ALL_SM75=$tail8" \
            ./ds4-bench --cuda --cuda-tensor-parallel \
                --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                --model "$MODEL" --prompt-file "$PROMPT" \
                --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" \
                --ctx-alloc "$CTX_ALLOC" --step-mul 2 \
                --prefill-chunk 2048 --gen-tokens 0 --csv "$base.csv" \
                >"$base.log" 2>&1 || { tail -n 180 "$base.log" >&2; die "$variant failed"; }
        [[ -s $base.csv ]] || die "$variant omitted benchmark CSV"
        grep -Fq 't256-placement=balanced' "$base.log" || die "$variant missed balanced T256"
        if [[ $variant == tail8 ]]; then
            grep -Fq 'SM75 IQ2 residual 1..8 tail8 enabled' "$base.log" ||
                die "tail8 production policy did not dispatch"
        else
            ! grep -Fq 'SM75 IQ2 residual 1..8 tail8 enabled' "$base.log" ||
                die "tail4 rollback dispatched tail8"
        fi
        printf '%s,%s,%s,%s,%s\n' "$repeat" "$slot" "$variant" \
            "$base.csv" "$base.log" >>"$OUTPUT_DIR/production/runs.csv"
    done
done

python3 - "$OUTPUT_DIR" <<'PY'
import csv, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
runs = list(csv.DictReader((root / "production/runs.csv").open()))
values = {}
paired = {}
for run in runs:
    rows = list(csv.DictReader(pathlib.Path(run["csv"]).open()))
    for row in rows:
        ctx = int(row["ctx_tokens"]); tps = float(row["prefill_tps"])
        values.setdefault((run["variant"], ctx), []).append(tps)
        paired[(int(run["repeat"]), run["variant"], ctx)] = tps
contexts = sorted({ctx for _, ctx in values})
with (root / "production/summary.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["ctx_tokens", "tail4_median_tps", "tail8_median_tps", "paired_median_speedup", "change_pct"])
    for ctx in contexts:
        ratios = [paired[(r, "tail8", ctx)] / paired[(r, "tail4", ctx)] for r in range(1, max(int(x["repeat"]) for x in runs)+1)]
        speed = statistics.median(ratios)
        w.writerow([ctx, f'{statistics.median(values[("tail4",ctx)]):.3f}', f'{statistics.median(values[("tail8",ctx)]):.3f}', f'{speed:.6f}', f'{(speed-1)*100:.3f}'])
PY
cat "$OUTPUT_DIR/production/summary.csv"

phase=complete
printf 'SM75 IQ2 tail-policy audit complete: %s\n' "$OUTPUT_DIR"
