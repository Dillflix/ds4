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

QUALITY_PRESET="${QUALITY_PRESET:-screen}"
case "$QUALITY_PRESET" in
    screen) DEFAULT_QUALITY_LAYERS="3,21,36" ;;
    full) DEFAULT_QUALITY_LAYERS="3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42" ;;
    *) echo "error: QUALITY_PRESET must be screen or full" >&2; exit 1 ;;
esac
QUALITY_LAYERS="${QUALITY_LAYERS:-$DEFAULT_QUALITY_LAYERS}"
QUALITY_EXPERTS="${QUALITY_EXPERTS:-0,127,255}"
QUALITY_PARTS="${QUALITY_PARTS:-w1,w2,w3}"
QUALITY_ROWS="${QUALITY_ROWS:-32}"
SKIP_BUILD="${SKIP_BUILD:-0}"
CREATE_ARCHIVE="${CREATE_ARCHIVE:-1}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
QUALITY_DIR="${QUALITY_DIR:-$ROOT/sm75-routed-quant-quality-$QUALITY_PRESET-$STAMP}"
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
        printf 'quality_preset=%s\n' "$QUALITY_PRESET"
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
    --sm75-routed-quality "$CSV" \
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
all_formats = (
    "q4_K", "q3_K", "sm75_q3_32", "sm75_q4_32", "iq2_xxs",
    "sm75_iq3_32", "sm75_q3a_32_4", "sm75_q3a_32_6",
    "sm75_q3q4_32_25", "sm75_q3q4_32_50",
    "sm75_q4a_32_4", "sm75_q4a_32_5", "sm75_q5_32",
    "sm75_q2q3_32_50", "sm75_q2q3_32_75",
)
whole_model_formats = tuple(x for x in all_formats if x != "iq2_xxs")
gate_formats = all_formats
down_formats = whole_model_formats
layers = [x for x in layers_text.split(",") if x]
experts = [x for x in experts_text.split(",") if x]
parts = [x for x in parts_text.split(",") if x]
expected_tensors = len(layers) * len(experts) * len(parts)
expected_tensor_rows = len(layers) * len(experts) * sum(
    len(gate_formats if part in ("w1", "w3") else down_formats)
    for part in parts
)

tensor_rows = [r for r in rows if r["scope"] == "tensor"]
aggregate = {r["format"]: r for r in rows if r["scope"] == "aggregate"}
role_rows = {(r["part"], r["format"]): r for r in rows if r["scope"] == "role"}
if len(tensor_rows) != expected_tensor_rows:
    raise SystemExit(
        f"expected {expected_tensor_rows} tensor rows, got {len(tensor_rows)}"
    )
if set(aggregate) != set(whole_model_formats):
    raise SystemExit(f"aggregate formats mismatch: {sorted(aggregate)}")
expected_roles = (
    {(role, name) for role in ("gate", "up", "gate_up") for name in gate_formats}
    | {("down", name) for name in down_formats}
)
if set(role_rows) != expected_roles:
    raise SystemExit(f"role rows mismatch: {sorted(role_rows)}")
for row in rows:
    for field in ("nrmse", "weighted_nrmse", "max_abs",
                  "source_energy", "weighted_source_energy"):
        value = float(row[field])
        if not math.isfinite(value) or value < 0:
            raise SystemExit(f"invalid {field}: {row}")

def pareto_names(role, formats):
    points = []
    for name in formats:
        row = role_rows[(role, name)]
        points.append((name, float(row["bits_per_weight"]),
                       float(row["weighted_nrmse"])))
    keep = set()
    for name, bits, error in points:
        dominated = any(
            other_bits <= bits and other_error <= error and
            (other_bits < bits or other_error < error)
            for other_name, other_bits, other_error in points
            if other_name != name
        )
        if not dominated:
            keep.add(name)
    return keep

def add_role_table(lines, title, role, formats):
    q4 = float(role_rows[(role, "q4_K")]["weighted_nrmse"])
    pareto = pareto_names(role, formats)
    lines += [
        f"## {title}", "",
        "| Format | Bits/weight | NRMSE | Imatrix-weighted NRMSE | Weighted / Q4_K | Max abs | Pareto |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for name in sorted(formats, key=lambda n: (
            float(role_rows[(role, n)]["bits_per_weight"]), n)):
        row = role_rows[(role, name)]
        weighted = float(row["weighted_nrmse"])
        lines.append(
            f"| {name} | {float(row['bits_per_weight']):.5f} | "
            f"{float(row['nrmse']):.8f} | {weighted:.8f} | "
            f"{weighted / q4:.5f} | {float(row['max_abs']):.8g} | "
            f"{'yes' if name in pareto else ''} |"
        )
    lines.append("")

lines = [
    "# SM75 routed-quant real-weight quality sweep", "",
    f"Sampled {expected_tensors} routed-expert tensors across "
    f"{len(layers)} layers, {len(experts)} experts, and {len(parts)} parts.", "",
]
add_role_table(lines, "Gate (w1)", "gate", gate_formats)
add_role_table(lines, "Up (w3)", "up", gate_formats)
add_role_table(lines, "Combined gate/up (w1 + w3)", "gate_up", gate_formats)
add_role_table(lines, "Down (w2)", "down", down_formats)

# Same format for gate and up, independently selectable format for down.
recipes = []
for gate_name in gate_formats:
    gate = role_rows[("gate_up", gate_name)]
    for down_name in down_formats:
        down = role_rows[("down", down_name)]
        bits = (2.0 * float(gate["bits_per_weight"]) +
                float(down["bits_per_weight"])) / 3.0
        weighted_sse = float(gate["weighted_sse"]) + float(down["weighted_sse"])
        weighted_energy = (float(gate["weighted_source_energy"]) +
                           float(down["weighted_source_energy"]))
        error = math.sqrt(weighted_sse / weighted_energy)
        recipes.append((gate_name, down_name, bits, error))
recipe_pareto = []
for recipe in recipes:
    _, _, bits, error = recipe
    if not any(
        obits <= bits and oerror <= error and
        (obits < bits or oerror < error)
        for _, _, obits, oerror in recipes
    ):
        recipe_pareto.append(recipe)
recipe_pareto.sort(key=lambda r: (r[2], r[3]))
lines += [
    "## Role-aware routed recipe Pareto frontier", "",
    "Gate and up share one format; down is selected independently. Bits/weight "
    "assumes equal-sized w1, w2, and w3 matrices.", "",
    "| Gate/up format | Down format | Routed bits/weight | Combined weighted NRMSE |",
    "|---|---|---:|---:|",
]
for gate_name, down_name, bits, error in recipe_pareto:
    lines.append(f"| {gate_name} | {down_name} | {bits:.5f} | {error:.8f} |")
lines += [
    "",
    "IQ2_XXS remains a gate/up control only. IQ2-down and Q2_K are intentionally excluded.",
    "",
    "This sweep measures real-weight reconstruction under expert-specific imatrix weights. "
    "It does not register GGUF types, alter production dispatch, or substitute for the "
    "end-to-end model quality suite.", "",
]
Path(summary_path).write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
PY

echo "SM75 routed-quant real-weight quality audit complete: $QUALITY_DIR"
