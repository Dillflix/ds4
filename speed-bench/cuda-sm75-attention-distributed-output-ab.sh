#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

MODEL=${MODEL:-}
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
CTX_TOKENS=${CTX_TOKENS:-32768}
CTX_ALLOC=${CTX_ALLOC:-$((CTX_TOKENS + 1))}
CASE_TIMEOUT_SECONDS=${CASE_TIMEOUT_SECONDS:-1800}
MIN_THROUGHPUT_RATIO=${MIN_THROUGHPUT_RATIO:-0.90}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${ATTENTION_DISTRIBUTED_OUTPUT_AB_DIR:-$repo_dir/sm75-attention-distributed-output-ab-$stamp}

[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name the existing tagged SM75 native-Q8 GGUF"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "CTX_TOKENS:$CTX_TOKENS" \
            "CTX_ALLOC:$CTX_ALLOC" "CASE_TIMEOUT_SECONDS:$CASE_TIMEOUT_SECONDS" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
(( STAGE_SPLIT == 22 && CTX_TOKENS >= 2048 &&
   CTX_TOKENS % 2048 == 0 && CTX_ALLOC > CTX_TOKENS &&
   CASE_TIMEOUT_SECONDS >= 60 )) || die "invalid benchmark bounds"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] ||
        die "$flag must be 0 or 1"
done
[[ $MIN_THROUGHPUT_RATIO =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    die "MIN_THROUGHPUT_RATIO must be numeric"
for tool in awk cmp date env find git grep make mkdir nproc nvidia-smi \
            python3 sort stat tail tar timeout; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{production,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
telemetry_pid=
stop_telemetry() {
    if [[ -n ${telemetry_pid:-} ]]; then
        kill "$telemetry_pid" 2>/dev/null || true
        wait "$telemetry_pid" 2>/dev/null || true
        telemetry_pid=
    fi
}
finish() {
    status=$?
    trap - EXIT INT TERM HUP
    stop_telemetry
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
targets=(ds4-bench tests/test_engine_mgpu_placement)
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "${targets[@]}" CUDA_ARCH=sm_75 \
        >"$OUTPUT_DIR/build.log" 2>&1 || {
            tail -n 200 "$OUTPUT_DIR/build.log" >&2
            die "build failed"
        }
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
    printf 'model=%s\nmodel_bytes=%s\nprompt=%s\n' \
        "$MODEL" "$(stat -c %s "$MODEL")" "$PROMPT"
    printf 'gpu_devices=%s\nstage_split=%s/%s\nctx_tokens=%s\n' \
        "$GPU_DEVICES" "$STAGE_SPLIT" "$((43-STAGE_SPLIT))" "$CTX_TOKENS"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,power.limit \
        --format=csv
    printf '\ntopology:\n'
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt"
git status --short >"$OUTPUT_DIR/provenance/git-status.txt"
git diff --stat >"$OUTPUT_DIR/provenance/git-diff-stat.txt"

phase=production-ab
printf 'variant,csv,log,logits,telemetry\n' >"$OUTPUT_DIR/production/runs.csv"
for variant in gather output; do
    output_rows=0; [[ $variant == output ]] && output_rows=1
    base="$OUTPUT_DIR/production/$variant"
    logits="$base-logits"
    telemetry="$base-telemetry.csv"
    mkdir -p "$logits"
    printf 'SM75 distributed attention output A/B variant=%s...\n' "$variant"
    nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,utilization.gpu,power.draw \
        --format=csv,noheader,nounits -lms 200 >"$telemetry" 2>&1 &
    telemetry_pid=$!
    set +e
    timeout --signal=TERM --kill-after=30 "$CASE_TIMEOUT_SECONDS" \
        "${clean[@]}" \
        "DS4_CUDA_EP_STAGE_SPLIT=$STAGE_SPLIT" \
        DS4_CUDA_PREFILL_PIPELINE=1 \
        DS4_CUDA_PREFILL_PIPELINE_MB=512 \
        DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE=1 \
        DS4_CUDA_TP_PREFILL_ATTN_HEADS=0 \
        "DS4_CUDA_TP_PREFILL_ATTN_ROWS_OUTPUT=$output_rows" \
        ./ds4-bench --cuda --cuda-tensor-parallel \
            --gpu-devices "$GPU_DEVICES" --gpu-vram "$GPU_VRAM" \
            --model "$MODEL" --prompt-file "$PROMPT" \
            --ctx-start "$CTX_TOKENS" --ctx-max "$CTX_TOKENS" \
            --ctx-alloc "$CTX_ALLOC" --step-mul 2 \
            --prefill-chunk 2048 --gen-tokens 0 --csv "$base.csv" \
            --dump-frontier-logits-dir "$logits" \
            >"$base.log" 2>&1
    status=$?
    set -e
    stop_telemetry
    if (( status != 0 )); then
        tail -n 240 "$base.log" >&2
        die "$variant production run failed with status $status"
    fi
    [[ -s $base.csv ]] || die "$variant omitted benchmark CSV"
    grep -Fq 'dense-placement=stage-aware-fixed-22-21' "$base.log" ||
        die "$variant missed fixed 22/21 dense placement"
    grep -Fq 'tagged SM75 dense-Q8 GGUF installed through ordinary single-owner residency' \
        "$base.log" || die "$variant did not load the tagged native-Q8 model"
    ! grep -Fq 'required native-GGUF execution binding unavailable' "$base.log" ||
        die "$variant missed a required native-GGUF execution binding"
    grep -Fq 'tier 1 rows ' "$base.log" ||
        die "$variant did not exercise stable logical pair 1"
    ! grep -Fq 'tier 0 rows ' "$base.log" ||
        die "$variant unexpectedly enabled unstable logical pair 0"
    if [[ $variant == output ]]; then
        grep -Fq 'rows retained through inverse RoPE and output A:' "$base.log" ||
            die "candidate missed distributed-output dispatch"
        grep -Fq 'required-native=150' "$base.log" ||
            die "candidate did not materialize all 21 partner A bindings"
    else
        ! grep -Fq 'rows retained through inverse RoPE and output A:' "$base.log" ||
            die "control unexpectedly dispatched distributed output"
        grep -Fq 'required-native=129' "$base.log" ||
            die "control did not retain the established 129 required bindings"
    fi
    printf '%s,%s,%s,%s,%s\n' "$variant" "$base.csv" "$base.log" \
        "$logits" "$telemetry" >>"$OUTPUT_DIR/production/runs.csv"
done

phase=exactness
mapfile -t gather_files < <(find "$OUTPUT_DIR/production/gather-logits" \
    -maxdepth 1 -type f -printf '%f\n' | sort)
mapfile -t output_files < <(find "$OUTPUT_DIR/production/output-logits" \
    -maxdepth 1 -type f -printf '%f\n' | sort)
[[ ${#gather_files[@]} -gt 0 && "${gather_files[*]}" == "${output_files[*]}" ]] ||
    die "logits inventory differs"
for file in "${gather_files[@]}"; do
    cmp -s "$OUTPUT_DIR/production/gather-logits/$file" \
           "$OUTPUT_DIR/production/output-logits/$file" ||
        die "logits differ: $file"
done

phase=summary
python3 - "$OUTPUT_DIR" "$MIN_THROUGHPUT_RATIO" <<'PY'
import csv, pathlib, sys
root = pathlib.Path(sys.argv[1])
minimum = float(sys.argv[2])
def row(name):
    rows = list(csv.DictReader((root / "production" / f"{name}.csv").open()))
    if len(rows) != 1:
        raise SystemExit(f"error: expected one {name} CSV row, found {len(rows)}")
    return rows[0]
gather = row("gather")
output = row("output")
gtps = float(gather["prefill_tps"])
otps = float(output["prefill_tps"])
ratio = otps / gtps
with (root / "summary.txt").open("w") as f:
    f.write(f"gather_prefill_tps={gtps:.6f}\n")
    f.write(f"output_prefill_tps={otps:.6f}\n")
    f.write(f"output_over_gather={ratio:.6f}\n")
    f.write("logits=bit-exact\n")
print((root / "summary.txt").read_text(), end="")
if ratio < minimum:
    raise SystemExit(
        f"error: distributed output ratio {ratio:.6f} is below {minimum:.6f}")
PY

phase=complete
printf 'SM75 distributed attention output A/B complete: %s\n' "$OUTPUT_DIR"
