#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run a four-GPU, production-shaped 32K prefill A/B for the next SM75-native
prefill bundle:

  * quantize FFN input once on the owner and transfer packed native Q8_K;
  * consume only active owner-local routed slots;
  * stage 16 indexed-attention rows per barrier; and
  * emit the selected top-512 indices in exact attention order inside top-k.

Optional environment:
  MODEL_LAYOUT=all43|mixed15  default: all43
  MIXED_MODEL=...             default: gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf
  ALL43_MODEL=...             default: gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf
  PROMPT=...                  default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  REQUIRED_POWER_LIMITS_W=250,260,250,250
  VARIANTS=control,candidate  comma-separated subset/order of:
                              control,candidate,native-q8-transfer,
                              compact-routed-slots,attention-row-tile16,
                              fused-indexer-selection
  REPEATS=1
  CASE_TIMEOUT_SECONDS=900
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  PREFILL_NATIVE_PIPELINE_AB_DIR=...

Every arm retains the stable topology: pair-0 attention row splitting disabled,
pair-1 enabled. `candidate` enables all four independently rollbackable
native-prefill selectors; the individually named arms enable exactly one.
Frontier logits must be byte-identical to `control`.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
MODEL_LAYOUT=${MODEL_LAYOUT:-all43}
MIXED_MODEL=${MIXED_MODEL:-$repo_dir/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4-50.gguf}
ALL43_MODEL=${ALL43_MODEL:-$repo_dir/gguf/ds4/DeepSeek-V4-Flash-0731-SM75-Q3A4-All-Q4-32-Down.gguf}
case "$MODEL_LAYOUT" in
    all43) MODEL=$ALL43_MODEL ;;
    mixed15) MODEL=$MIXED_MODEL ;;
    *) die "MODEL_LAYOUT must be all43 or mixed15" ;;
esac
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
REQUIRED_POWER_LIMITS_W=${REQUIRED_POWER_LIMITS_W:-250,260,250,250}
VARIANTS=${VARIANTS:-control,candidate}
REPEATS=${REPEATS:-1}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-900}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
PREFILL_CHUNK=2048
PIPELINE_MB=512
CTX_ALLOC=33025
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${PREFILL_NATIVE_PIPELINE_AB_DIR:-$repo_dir/sm75-prefill-native-pipeline-ab-$MODEL_LAYOUT-$stamp}

[[ $MODEL == /* && -f $MODEL ]] || die "model not found at absolute path: $MODEL"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
cmp -s "$PROMPT" "$repo_dir/speed-bench/promessi_sposi.txt" ||
    die "this A/B requires the fixed promessi_sposi prompt"
[[ $GPU_DEVICES == 0,3,1,2 && $GPU_VRAM == auto && $STAGE_SPLIT == 22 ]] ||
    die "require GPU_DEVICES=0,3,1,2, GPU_VRAM=auto, STAGE_SPLIT=22"
[[ $REPEATS =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be a positive integer"
[[ $CASE_TIMEOUT_SECONDS =~ ^[1-9][0-9]*$ ]] ||
    die "CASE_TIMEOUT_SECONDS must be a positive integer"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
IFS=, read -r -a requested_variants <<<"$VARIANTS"
(( ${#requested_variants[@]} >= 2 )) ||
    die "VARIANTS must contain control and at least one candidate arm"
declare -A variant_seen=()
has_control=0
for variant in "${requested_variants[@]}"; do
    case "$variant" in
        control) has_control=1 ;;
        candidate|native-q8-transfer|compact-routed-slots|attention-row-tile16|fused-indexer-selection) ;;
        *) die "unknown VARIANTS arm: ${variant:-<empty>}" ;;
    esac
    [[ -z ${variant_seen[$variant]+x} ]] ||
        die "VARIANTS contains duplicate arm: $variant"
    variant_seen[$variant]=1
done
(( has_control )) || die "VARIANTS must include control"
[[ -z ${CUDA_VISIBLE_DEVICES:-} ]] ||
    die "CUDA_VISIBLE_DEVICES must be unset so physical GPU IDs remain stable"
for tool in awk basename cmp date dirname env find git grep make mkdir mv nproc \
            nvidia-smi python3 rm sort stat stdbuf tail tar tee timeout tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done

IFS=, read -r -a gpu_ids <<<"$GPU_DEVICES"
IFS=, read -r -a required_power <<<"$REQUIRED_POWER_LIMITS_W"
(( ${#gpu_ids[@]} == 4 && ${#required_power[@]} == 4 )) ||
    die "GPU_DEVICES and REQUIRED_POWER_LIMITS_W must each contain four values"
for gpu in 0 1 2 3; do
    cap=$(nvidia-smi -i "$gpu" --query-gpu=compute_cap \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ $cap == 7.5 ]] || die "GPU $gpu is compute capability ${cap:-unknown}, not SM75"
    limit=$(nvidia-smi -i "$gpu" --query-gpu=power.limit \
        --format=csv,noheader,nounits | tr -d '[:space:]')
    awk -v actual="$limit" -v expected="${required_power[$gpu]}" \
        'BEGIN {exit !((actual+0)==(expected+0))}' ||
        die "GPU $gpu power limit is ${limit:-unknown} W, expected ${required_power[$gpu]} W"
done

[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{runs,summary,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
finish() {
    local status=$? archive="$OUTPUT_DIR.tar.gz" partial="$OUTPUT_DIR.tar.gz.partial.$$"
    trap - EXIT INT TERM HUP
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                "$(basename "$OUTPUT_DIR")" && mv -f -- "$partial" "$archive"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
            rm -f -- "$partial" "$archive"
            printf 'error: could not create %s\n' "$archive" >&2
        fi
    fi
    exit "$status"
}
trap finish EXIT
trap 'phase=interrupted; exit 130' INT TERM HUP

mapfile -t inherited_ds4 < <(env | awk -F= '$1 ~ /^DS4_/ {print $1}' | sort -u)
clean=(env)
for name in "${inherited_ds4[@]}"; do clean+=(-u "$name"); done
printf '%s\n' "${inherited_ds4[@]:-}" >"$OUTPUT_DIR/provenance/cleared-ds4-env.txt"

phase=build
targets=(ds4-bench tests/cuda_long_context_smoke tests/test_engine_mgpu_placement)
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
    >"$OUTPUT_DIR/cuda-long-context.log" 2>&1 || {
        tail -n 220 "$OUTPUT_DIR/cuda-long-context.log" >&2
        die "CUDA exactness regression failed"
    }
for marker in \
    'shared-native-Q8/compact-slot owned prefill exact' \
    'fused top-k + ascending attention index selection exact' \
    'tile8/tile16 whole and tile16-sharded exact'; do
    grep -Fq "$marker" "$OUTPUT_DIR/cuda-long-context.log" ||
        die "CUDA exactness marker missing: $marker"
done

capture_gpu_health() {
    local output=$1 partial="$1.partial.$$" gpu
    : >"$partial"
    for gpu in 0 1 2 3; do
        nvidia-smi -i "$gpu" \
            --query-gpu=index,pci.bus_id,uuid,serial,power.limit \
            --format=csv,noheader,nounits >>"$partial" 2>&1 || {
                mv -- "$partial" "$output"
                return 1
            }
    done
    mv -- "$partial" "$output"
}

phase=manifest
capture_gpu_health "$OUTPUT_DIR/initial-gpu.csv" ||
    die "could not capture initial GPU state"
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'model_layout=%s\nmodel=%s\nmodel_bytes=%s\nprompt=%s\n' \
        "$MODEL_LAYOUT" "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\nstage_split=22/21\ncontext=32768\nrepeats=%s\n' \
        "$GPU_DEVICES" "$REPEATS"
    printf 'pair0_attention_rows=disabled\npair1_attention_rows=enabled\n'
    printf 'variants=%s\n' "$VARIANTS"
    cat "$OUTPUT_DIR/initial-gpu.csv"
    printf '\ntopology:\n'
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

production_env=(
    "${clean[@]}"
    DS4_CUDA_EP_STAGE_SPLIT=22
    DS4_CUDA_PREFILL_PIPELINE=1
    "DS4_CUDA_PREFILL_PIPELINE_MB=$PIPELINE_MB"
    DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1
    "DS4_CUDA_Q8_F16_PARTNER_MAX_TOKENS=$PREFILL_CHUNK"
    DS4_BENCH_UNTIMED_WARMUP_TOKENS=512
    DS4_CUDA_NO_TP_PREFILL_ATTN_ROWS_PAIRS=0
)

set_selector_flags() {
    native_q8=0
    compact_slots=0
    row_tile16=0
    fused_indexer=0
    case "$1" in
        control) ;;
        candidate)
            native_q8=1
            compact_slots=1
            row_tile16=1
            fused_indexer=1
            ;;
        native-q8-transfer) native_q8=1 ;;
        compact-routed-slots) compact_slots=1 ;;
        attention-row-tile16) row_tile16=1 ;;
        fused-indexer-selection) fused_indexer=1 ;;
    esac
}

validate_selector_marker() {
    local expected=$1 marker=$2 log=$3
    if [[ $expected == 1 ]]; then
        grep -Fq "$marker" "$log" || {
            printf 'validation: enabled selector marker missing: %s\n' \
                "$marker" >&2
            return 1
        }
    else
        ! grep -Fq "$marker" "$log" || {
            printf 'validation: disabled selector marker present: %s\n' \
                "$marker" >&2
            return 1
        }
    fi
}

validate_log() {
    local log=$1 marker
    for marker in 'CUDA EP forced pipeline split 22/21' \
                  'dense-placement=stage-aware-fixed-22-21' \
                  'materialized 344/344 candidates'; do
        grep -Fq "$marker" "$log" || {
            printf 'validation: missing production marker: %s\n' "$marker" >&2
            return 1
        }
    done
    ! grep -Fq 'required but unavailable' "$log" || return 1
    ! grep -Fq 'prefill attention query-row split enabled: tier 0 ' "$log" || return 1
    grep -Fq 'prefill attention query-row split enabled: tier 1 ' "$log" || return 1
    validate_selector_marker "$native_q8" \
        'SM75 shared native Q8 routed activation selected' "$log" || return 1
    validate_selector_marker "$native_q8" \
        'SM75 direct native Q8 producer selected for routed MoE activations' \
        "$log" || return 1
    validate_selector_marker "$compact_slots" \
        'SM75 compact owner-local routed slots selected' "$log" || return 1
    validate_selector_marker "$fused_indexer" \
        'SM75 fused indexer selection/index-order selected' "$log" || return 1
    validate_selector_marker "$row_tile16" \
        'SM75 indexed prefill attention row tile16 selected' "$log" || return 1
}

phase=production-ab
printf 'repeat,slot,variant,csv,log,logits\n' >"$OUTPUT_DIR/runs/index.csv"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
    variants=("${requested_variants[@]}")
    if (( repeat % 2 == 0 )); then
        variants=()
        for ((i=${#requested_variants[@]}-1; i>=0; i--)); do
            variants+=("${requested_variants[i]}")
        done
    fi
    slot=0
    for variant in "${variants[@]}"; do
        slot=$((slot + 1))
        set_selector_flags "$variant"
        base="$OUTPUT_DIR/runs/r${repeat}-s${slot}-$variant"
        logits="$base-logits"
        mkdir -p "$logits"
        printf 'SM75 native prefill A/B repeat=%d/%d slot=%d variant=%s...\n' \
            "$repeat" "$REPEATS" "$slot" "$variant"
        capture_gpu_health "$base.pre-gpu.csv" || die "pre-run GPU health capture failed"
        set +e
        "${production_env[@]}" \
            "DS4_CUDA_PREFILL_NATIVE_Q8_TRANSFER=$native_q8" \
            "DS4_CUDA_PREFILL_COMPACT_ROUTED_SLOTS=$compact_slots" \
            "DS4_CUDA_PREFILL_ATTN_ROW_TILE16=$row_tile16" \
            "DS4_CUDA_PREFILL_FUSED_INDEXER_SELECTION=$fused_indexer" \
            timeout --signal=INT --kill-after=30s "${CASE_TIMEOUT_SECONDS}s" \
            stdbuf -oL -eL ./ds4-bench --cuda --cuda-tensor-parallel \
                --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
                --model "$MODEL" --prompt-file "$PROMPT" \
                --ctx-start 32768 --ctx-max 32768 --ctx-alloc "$CTX_ALLOC" \
                --step-mul 8 --prefill-chunk "$PREFILL_CHUNK" \
                --gen-tokens 0 --csv "$base.csv" \
                --dump-frontier-logits-dir "$logits" \
                >"$base.log" 2>&1
        run_status=$?
        set -e
        capture_gpu_health "$base.post-gpu.csv" || die "post-run GPU health capture failed"
        (( run_status == 0 )) || {
            tail -n 240 "$base.log" >&2 || true
            die "$variant run failed with status $run_status"
        }
        cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$base.pre-gpu.csv" &&
            cmp -s "$OUTPUT_DIR/initial-gpu.csv" "$base.post-gpu.csv" ||
            die "GPU identity or power limits changed during $variant"
        [[ -s $base.csv ]] || die "$variant omitted benchmark CSV"
        validate_log "$base.log" || die "$variant production validation failed"
        awk -F, 'NR>1 && $1==32768 && ($3+0)>0 {ok=1} END {exit !ok}' \
            "$base.csv" || die "$variant omitted a successful 32K result"
        printf '%s,%s,%s,%s,%s,%s\n' "$repeat" "$slot" "$variant" \
            "$base.csv" "$base.log" "$logits" >>"$OUTPUT_DIR/runs/index.csv"
    done

    control_logits=$(awk -F, -v r="$repeat" '$1==r && $3=="control" {print $6}' \
        "$OUTPUT_DIR/runs/index.csv")
    mapfile -t control_files < <(find "$control_logits" -maxdepth 1 -type f -printf '%f\n' | sort)
    (( ${#control_files[@]} > 0 )) || die "repeat $repeat control logits are empty"
    for variant in "${requested_variants[@]}"; do
        [[ $variant != control ]] || continue
        candidate_logits=$(awk -F, -v r="$repeat" -v v="$variant" \
            '$1==r && $3==v {print $6}' "$OUTPUT_DIR/runs/index.csv")
        mapfile -t candidate_files < <(find "$candidate_logits" -maxdepth 1 -type f -printf '%f\n' | sort)
        [[ "${control_files[*]}" == "${candidate_files[*]}" ]] ||
            die "repeat $repeat $variant logits inventory differs"
        for file in "${control_files[@]}"; do
            cmp -s "$control_logits/$file" "$candidate_logits/$file" ||
                die "repeat $repeat $variant logits differ: $file"
        done
    done
done

python3 - "$OUTPUT_DIR" <<'PY'
import csv, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
runs = list(csv.DictReader((root / "runs/index.csv").open()))
rates = {}
paired = {}
variant_order = []
for run in runs:
    rows = list(csv.DictReader(pathlib.Path(run["csv"]).open()))
    row = next(x for x in rows if int(x["ctx_tokens"]) == 32768)
    value = float(row["prefill_tps"])
    variant = run["variant"]
    if variant not in rates:
        rates[variant] = []
        variant_order.append(variant)
    rates[variant].append(value)
    paired[(int(run["repeat"]), variant)] = value
repeats = range(1, max(x[0] for x in paired) + 1)
layout = next(x.split("=", 1)[1] for x in
              (root / "manifest.txt").read_text().splitlines()
              if x.startswith("model_layout="))
with (root / "summary/summary.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model_layout", "ctx_tokens", "variant", "median_tps",
                "paired_median_speedup_vs_control", "change_pct",
                "logits"])
    for variant in variant_order:
        ratios = [paired[(r, variant)] / paired[(r, "control")]
                  for r in repeats]
        speed = statistics.median(ratios)
        w.writerow([layout, 32768, variant,
                    f'{statistics.median(rates[variant]):.3f}',
                    f'{speed:.6f}', f'{(speed - 1.0) * 100.0:.3f}',
                    "reference" if variant == "control" else "byte-identical"])
PY
cat "$OUTPUT_DIR/summary/summary.csv"

phase=complete
printf 'SM75 native prefill pipeline A/B complete: %s\n' "$OUTPUT_DIR"
