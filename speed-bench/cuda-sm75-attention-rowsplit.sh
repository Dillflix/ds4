#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PROFILE_GPU=${PROFILE_GPU:-0}
PROFILE_PARTNER_GPU=${PROFILE_PARTNER_GPU:-1}
REPEATS=${REPEATS:-5}
MAKE_JOBS=${MAKE_JOBS:-$(nproc)}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/sm75-attention-rowsplit-$STAMP}
TARGET=tests/cuda_attention_rowsplit_xdev

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be an integer"
[[ $PROFILE_PARTNER_GPU =~ ^[0-9]+$ ]] ||
    die "PROFILE_PARTNER_GPU must be an integer"
[[ $PROFILE_GPU != "$PROFILE_PARTNER_GPU" ]] ||
    die "PROFILE_GPU and PROFILE_PARTNER_GPU must differ"
[[ $REPEATS =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be positive"

mkdir -p "$OUTPUT_DIR"
if [[ $SKIP_BUILD == 0 ]]; then
    make -j"$MAKE_JOBS" "$TARGET" CUDA_ARCH=sm_75
else
    [[ -x $TARGET ]] || die "SKIP_BUILD=1 but $TARGET is missing"
    for source in tests/cuda_attention_rowsplit_xdev.c ds4_cuda.cu \
                  ds4_gpu.h ds4_gpu_mgpu.h Makefile; do
        [[ $TARGET -nt $source ]] ||
            die "SKIP_BUILD=1 found stale target relative to $source"
    done
fi

{
    printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'git_dirty=%s\n' "$(git status --porcelain | grep -q . && printf true || printf false)"
    printf 'profile_gpu=%s\nprofile_partner_gpu=%s\nrepeats=%s\n' \
        "$PROFILE_GPU" "$PROFILE_PARTNER_GPU" "$REPEATS"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader
    nvidia-smi topo -m
} >"$OUTPUT_DIR/manifest.txt" 2>&1

run_case() {
    local scenario=$1
    printf 'Running exact %s 32K attention row-split A/B on physical GPUs %s,%s...\n' \
        "$scenario" "$PROFILE_GPU" "$PROFILE_PARTNER_GPU"
    CUDA_VISIBLE_DEVICES="$PROFILE_GPU,$PROFILE_PARTNER_GPU" \
    DS4_ROWSPLIT_REPEATS="$REPEATS" \
        "$TARGET" "$scenario" \
        >"$OUTPUT_DIR/$scenario.txt" \
        2> >(tee "$OUTPUT_DIR/$scenario.log" >&2)
    grep -q '^validation=bit-exact-nonzero$' "$OUTPUT_DIR/$scenario.txt" ||
        die "$scenario did not report bit-exact non-zero validation"
    cat "$OUTPUT_DIR/$scenario.txt"
}

run_case mixed
run_case indexed
run_case indexed-pipeline

python3 - "$OUTPUT_DIR" <<'PY'
import pathlib, sys

root = pathlib.Path(sys.argv[1])
rows = []
for scenario in ("mixed", "indexed"):
    values = {}
    for line in (root / f"{scenario}.txt").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    rows.append((scenario, values))

lines = [
    "# SM75 32K attention query-row split",
    "",
    "Both candidates must pass bit-exact non-zero validation against the shipping single-GPU kernel.",
    "",
    "| Path | Baseline ms | Peer-read ms | Peer speedup | Mirrored-KV ms | Mirror speedup |",
    "|---|---:|---:|---:|---:|---:|",
]
for name, v in rows:
    lines.append(
        f"| {name} | {float(v['baseline_ms']):.3f} | "
        f"{float(v['peer_read_ms']):.3f} | {float(v['peer_read_speedup']):.3f}x | "
        f"{float(v['mirrored_kv_ms']):.3f} | {float(v['mirrored_kv_speedup']):.3f}x |"
    )
values = {}
for line in (root / "indexed-pipeline.txt").read_text().splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        values[key] = value
lines.extend([
    "",
    "## Complete indexed score -> top-k -> attention chain",
    "",
    "The transfer-inclusive result copies the partner indexer query, indexer weights, "
    "and attention query on every repeat, then gathers the partner attention output. "
    "Persistent index/raw/compressed caches are mirrored outside the timed region.",
    "",
    "| Baseline ms | Mirrored compute ms | Compute speedup | Transfer-inclusive ms | Transfer-inclusive speedup |",
    "|---:|---:|---:|---:|---:|",
    f"| {float(values['baseline_ms']):.3f} | "
    f"{float(values['mirrored_compute_ms']):.3f} | "
    f"{float(values['mirrored_compute_speedup']):.3f}x | "
    f"{float(values['transfer_inclusive_ms']):.3f} | "
    f"{float(values['transfer_inclusive_speedup']):.3f}x |",
])
(root / "summary.md").write_text("\n".join(lines) + "\n")
PY

cat "$OUTPUT_DIR/summary.md"
if [[ $CREATE_ARCHIVE == 1 ]]; then
    tar -C "$(dirname "$OUTPUT_DIR")" -czf "$OUTPUT_DIR.tar.gz" \
        "$(basename "$OUTPUT_DIR")"
    printf 'Archive to return: %s.tar.gz\n' "$OUTPUT_DIR"
fi
