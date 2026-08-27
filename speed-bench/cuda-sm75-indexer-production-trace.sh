#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Capture paired, unsynchronized 32K production traces for the exact SM75
indexer WMMA128 and WMMA64 score kernels.

Required environment:
  MODEL=/absolute/path/to/tagged-SM75-mixed-Q4-IQ2.gguf

Optional environment:
  PROMPT=...                    default: speed-bench/promessi_sposi.txt
  GPU_DEVICES=0,3,1,2
  GPU_VRAM=auto
  STAGE_SPLIT=22
  CTX_ALLOC=262273
  PREFILL_CHUNK=2048
  PIPELINE_MB=512
  PROFILE_GPU=0
  PROFILE_PARTNER_GPU=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  INDEXER_TRACE_DIR=/absolute/output/directory

The bounded exact harness and the prior uninstrumented throughput A/B remain
the performance and correctness gates. This trace answers only whether the
isolated WMMA64 kernel gain transfers to production score launches and, if it
does, whether it shortens the pipeline critical path.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
MODEL=${MODEL:-}
PROMPT=${PROMPT:-$repo_dir/speed-bench/promessi_sposi.txt}
GPU_DEVICES=${GPU_DEVICES:-0,3,1,2}
GPU_VRAM=${GPU_VRAM:-auto}
STAGE_SPLIT=${STAGE_SPLIT:-22}
CTX_ALLOC=${CTX_ALLOC:-262273}
PREFILL_CHUNK=${PREFILL_CHUNK:-2048}
PIPELINE_MB=${PIPELINE_MB:-512}
PROFILE_GPU=${PROFILE_GPU:-0}
PROFILE_PARTNER_GPU=${PROFILE_PARTNER_GPU:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${INDEXER_TRACE_DIR:-$repo_dir/sm75-indexer-production-trace-$stamp}

[[ $MODEL == /* && -f $MODEL ]] ||
    die "MODEL must name the existing absolute tagged mixed-Q4/IQ2 model"
[[ -f $PROMPT ]] || die "prompt not found: $PROMPT"
for item in "STAGE_SPLIT:$STAGE_SPLIT" "CTX_ALLOC:$CTX_ALLOC" \
            "PREFILL_CHUNK:$PREFILL_CHUNK" "PIPELINE_MB:$PIPELINE_MB" \
            "PROFILE_GPU:$PROFILE_GPU" \
            "PROFILE_PARTNER_GPU:$PROFILE_PARTNER_GPU" \
            "SKIP_BUILD:$SKIP_BUILD" "CREATE_ARCHIVE:$CREATE_ARCHIVE"; do
    name=${item%%:*}; value=${item#*:}
    [[ $value =~ ^[0-9]+$ ]] || die "$name must be an integer"
done
[[ $CTX_ALLOC == 262273 && $PREFILL_CHUNK == 2048 && $PIPELINE_MB == 512 ]] ||
    die "require CTX_ALLOC=262273, PREFILL_CHUNK=2048, and PIPELINE_MB=512"
for flag in SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
for tool in bash basename cmp date dirname find git grep mkdir python3 stat tar; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
git diff --quiet && git diff --cached --quiet ||
    die "tracked source changes are present; test an exact committed tree"
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
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

run_variant() {
    local variant=$1 tile=$2 build_mode=$3
    local target="$OUTPUT_DIR/$variant"
    printf 'Capturing unsynchronized 32K production trace: %s...\n' "$variant"
    MODEL="$MODEL" PROMPT="$PROMPT" GPU_DEVICES="$GPU_DEVICES" \
    GPU_VRAM="$GPU_VRAM" STAGE_SPLIT="$STAGE_SPLIT" \
    PROFILE_TOKENS=32768 CTX_ALLOC="$CTX_ALLOC" \
    PREFILL_CHUNK="$PREFILL_CHUNK" PIPELINE_MB="$PIPELINE_MB" \
    PROFILE_GPU="$PROFILE_GPU" PROFILE_PARTNER_GPU="$PROFILE_PARTNER_GPU" \
    INDEXER_SCORE_TILE="$tile" RUN_NCU=0 RUN_ATTENTION_NCU=0 \
    NCU_USE_SUDO=0 SKIP_BUILD="$build_mode" CREATE_ARCHIVE=0 \
    COMBINED_PROFILE_DIR="$target" \
        bash ./speed-bench/cuda-sm75-native-q4-t256-profile.sh
}

phase=wmma128-trace
run_variant wmma128 128 "$SKIP_BUILD"
phase=wmma64-trace
run_variant wmma64 64 1

phase=paired-integrity
for variant in wmma128 wmma64; do
    [[ -s $OUTPUT_DIR/$variant/run-status.txt ]] ||
        die "$variant trace lacks run status"
    grep -Fqx 'state=finished' "$OUTPUT_DIR/$variant/run-status.txt" ||
        die "$variant trace did not finish"
done
for suffix in f32 json; do
    cmp -s \
        "$OUTPUT_DIR/wmma128/nsys/frontier-logits/frontier_032768.logits.$suffix" \
        "$OUTPUT_DIR/wmma64/nsys/frontier-logits/frontier_032768.logits.$suffix" ||
        die "WMMA128 and WMMA64 32K frontier logits differ ($suffix)"
done
count128=$(grep -F 'ds4: CUDA indexer score audit dispatch=wmma128 ' \
    "$OUTPUT_DIR/wmma128/nsys/combined.log")
count128=${count128##*launches=}
count64=$(grep -F 'ds4: CUDA indexer score audit dispatch=wmma64 ' \
    "$OUTPUT_DIR/wmma64/nsys/combined.log")
count64=${count64##*launches=}
[[ $count128 =~ ^[0-9]+$ && $count128 == "$count64" ]] ||
    die "paired trace dispatch counts differ: $count128 versus $count64"

phase=summary
python3 speed-bench/summarize-sm75-indexer-production-trace.py "$OUTPUT_DIR"

phase=complete
for required in summary.md comparison.json trace-comparison.csv \
                score-device-stage.csv; do
    [[ -s $OUTPUT_DIR/$required ]] || die "missing final evidence: $required"
done
printf 'SM75 indexer production trace comparison complete: %s\n' "$OUTPUT_DIR"
