#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${HF_DIR:-}" ]]; then
    echo "error: set HF_DIR to the DeepSeek-V4-Flash-0731 HF snapshot" >&2
    exit 1
fi
if [[ -z "${IMATRIX:-}" ]]; then
    echo "error: set IMATRIX to the routed-MoE imatrix .dat file" >&2
    exit 1
fi

QUALITY_LAYERS="${QUALITY_LAYERS:-3,21,36}"
QUALITY_EXPERTS="${QUALITY_EXPERTS:-0,127,255}"
QUALITY_PARTS="${QUALITY_PARTS:-w1,w2,w3}"
QUALITY_ROWS="${QUALITY_ROWS:-32}"
SKIP_BUILD="${SKIP_BUILD:-0}"
CREATE_ARCHIVE="${CREATE_ARCHIVE:-1}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
QUALITY_DIR="${QUALITY_DIR:-$ROOT/sm75-q3-q4-real-quality-$STAMP}"
CSV="$QUALITY_DIR/real-weight-quality.csv"
LOG="$QUALITY_DIR/real-weight-quality.log"
ARCHIVE="${QUALITY_DIR}.tar.gz"

mkdir -p "$QUALITY_DIR"

archive_result() {
    local status=$?
    {
        printf 'exit_status=%s\n' "$status"
        printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || true)"
        printf 'git_branch=%s\n' "$(git branch --show-current 2>/dev/null || true)"
        printf 'hf_dir=%s\n' "$HF_DIR"
        printf 'imatrix=%s\n' "$IMATRIX"
        printf 'quality_layers=%s\n' "$QUALITY_LAYERS"
        printf 'quality_experts=%s\n' "$QUALITY_EXPERTS"
        printf 'quality_parts=%s\n' "$QUALITY_PARTS"
        printf 'quality_rows=%s\n' "$QUALITY_ROWS"
        printf 'model_hashing=disabled\n'
        printf 'full_model_quantized=false\n'
        printf 'production_dispatch_changed=false\n'
    } > "$QUALITY_DIR/metadata.txt"
    git status --short > "$QUALITY_DIR/git-status.txt" 2>&1 || true
    git diff --binary > "$QUALITY_DIR/tracked-working-tree.patch" 2>&1 || true
    if [[ "$CREATE_ARCHIVE" == 1 ]]; then
        tar -czf "$ARCHIVE" -C "$(dirname "$QUALITY_DIR")" "$(basename "$QUALITY_DIR")"
        printf 'Archive to return: %s\n' "$ARCHIVE"
    fi
    exit "$status"
}
trap archive_result EXIT

[[ -f "$HF_DIR/model.safetensors.index.json" ]] ||
    { echo "error: HF snapshot is missing model.safetensors.index.json: $HF_DIR" >&2; exit 1; }
[[ -f "$IMATRIX" ]] ||
    { echo "error: imatrix not found: $IMATRIX" >&2; exit 1; }
[[ "$QUALITY_ROWS" =~ ^[1-9][0-9]*$ ]] ||
    { echo "error: QUALITY_ROWS must be a positive integer" >&2; exit 1; }

if [[ "$SKIP_BUILD" != 1 ]]; then
    make -C gguf-tools -B deepseek4-quantize test-quants-experimental
else
    [[ -x gguf-tools/deepseek4-quantize && -x gguf-tools/test-quants-experimental ]] ||
        { echo "error: SKIP_BUILD=1 requires both quantizer and test binary" >&2; exit 1; }
    if find gguf-tools -maxdepth 1 \
        \( -name 'deepseek4-quantize.c' -o -name 'quants.c' -o \
           -name 'quants.h' -o -name 'test_quants_experimental.c' -o \
           -name 'Makefile' \) \
        -newer gguf-tools/deepseek4-quantize -print -quit | grep -q .; then
        echo "error: SKIP_BUILD=1 found a stale quantizer" >&2
        exit 1
    fi
fi

gguf-tools/test-quants-experimental |
    tee "$QUALITY_DIR/experimental-quant-tests.log"

gguf-tools/deepseek4-quantize \
    --hf "$HF_DIR" \
    --imatrix "$IMATRIX" \
    --sm75-q3-q4-quality "$CSV" \
    --quality-layers "$QUALITY_LAYERS" \
    --quality-experts "$QUALITY_EXPERTS" \
    --quality-parts "$QUALITY_PARTS" \
    --quality-rows "$QUALITY_ROWS" \
    2>&1 | tee "$LOG"

python3 - "$CSV" "$QUALITY_DIR/summary.md" \
    "$QUALITY_LAYERS" "$QUALITY_EXPERTS" "$QUALITY_PARTS" <<'PY'
import csv
import math
import sys
from pathlib import Path

csv_path, summary_path, layers_text, experts_text, parts_text = sys.argv[1:]
rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
formats = ("q4_K", "q3_K", "sm75_q3_32", "sm75_q4_32")
layers = [x for x in layers_text.split(",") if x]
experts = [x for x in experts_text.split(",") if x]
parts = [x for x in parts_text.split(",") if x]
expected_tensors = len(layers) * len(experts) * len(parts)

tensor_rows = [r for r in rows if r["scope"] == "tensor"]
aggregate = {r["format"]: r for r in rows if r["scope"] == "aggregate"}
if len(tensor_rows) != expected_tensors * len(formats):
    raise SystemExit(
        f"expected {expected_tensors * len(formats)} tensor rows, got {len(tensor_rows)}"
    )
if set(aggregate) != set(formats):
    raise SystemExit(f"aggregate formats mismatch: {sorted(aggregate)}")
for row in rows:
    for field in ("nrmse", "weighted_nrmse", "max_abs"):
        value = float(row[field])
        if not math.isfinite(value) or value < 0:
            raise SystemExit(f"invalid {field}: {row}")

q4 = float(aggregate["q4_K"]["weighted_nrmse"])
lines = [
    "# SM75 Q3/Q4 real-weight quality",
    "",
    f"Sampled {expected_tensors} routed-expert tensors.",
    "",
    "| Format | Bits/weight | NRMSE | Imatrix-weighted NRMSE | Weighted / Q4_K | Max abs |",
    "|---|---:|---:|---:|---:|---:|",
]
for name in formats:
    row = aggregate[name]
    weighted = float(row["weighted_nrmse"])
    lines.append(
        f"| {name} | {float(row['bits_per_weight']):.3f} | "
        f"{float(row['nrmse']):.8f} | {weighted:.8f} | "
        f"{weighted / q4:.5f} | {float(row['max_abs']):.8g} |"
    )
lines += [
    "",
    "This is a bounded real-weight/imatrix error audit. It does not alter GGUF",
    "type registration or production dispatch, and it is not an end-to-end",
    "model-quality score.",
    "",
]
Path(summary_path).write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
PY

echo "SM75 Q3/Q4 real-weight quality audit complete: $QUALITY_DIR"
