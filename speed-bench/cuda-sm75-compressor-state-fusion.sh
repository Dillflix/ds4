#!/usr/bin/env bash
set -euo pipefail

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
TIMING_ROUNDS=${TIMING_ROUNDS:-9}
TIMING_REPEATS=${TIMING_REPEATS:-25}
RUN_SANITIZER=${RUN_SANITIZER:-1}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${COMPRESSOR_STATE_FUSION_DIR:-$repo_dir/sm75-compressor-state-fusion-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $TIMING_ROUNDS =~ ^[1-9][0-9]*$ ]] || die "TIMING_ROUNDS must be positive"
[[ $TIMING_REPEATS =~ ^[1-9][0-9]*$ ]] || die "TIMING_REPEATS must be positive"
[[ $RUN_SANITIZER == 0 || $RUN_SANITIZER == 1 ]] || die "RUN_SANITIZER must be 0 or 1"
[[ $CREATE_ARCHIVE == 0 || $CREATE_ARCHIVE == 1 ]] || die "CREATE_ARCHIVE must be 0 or 1"

tools=(c++filt cuobjdump git grep make nproc nvidia-smi python3 tar)
(( RUN_SANITIZER == 0 )) || tools+=(compute-sanitizer)
for tool in "${tools[@]}"; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done

compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] ||
    die "physical GPU $PROFILE_GPU has compute capability ${compute_cap:-unknown}; SM75 is required"
[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/results" "$OUTPUT_DIR/sanitizer"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

current_phase=initialization
finalize() {
    local status=$?
    trap - EXIT
    printf 'state=finished\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
        "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        local archive="$OUTPUT_DIR.tar.gz"
        tar -C "$(dirname "$OUTPUT_DIR")" -czf "$archive" \
            "$(basename "$OUTPUT_DIR")" || status=1
        printf 'Archive to return: %s\n' "$archive"
    fi
    exit "$status"
}
trap finalize EXIT

{
    printf 'date_utc=%s\nrepo=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
    printf 'git_commit=%s\ngit_branch=%s\n' "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'profile_gpu=%s\ncuda_arch=%s\n' "$PROFILE_GPU" "$CUDA_ARCH"
    printf 'timing_rounds=%s\ntiming_repeats=%s\nrun_sanitizer=%s\n' \
        "$TIMING_ROUNDS" "$TIMING_REPEATS" "$RUN_SANITIZER"
    printf '\n[gpu]\n'
    nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,name,pci.bus_id,memory.total,memory.free,driver_version \
        --format=csv
    printf '\n[git status]\n'; git status --short
} >"$OUTPUT_DIR/manifest.txt"

harness=tests/cuda_sm75_compressor_state_fusion
current_phase=build
nvccflags=${EVIDENCE_NVCCFLAGS:-"-O3 -g -lineinfo --use_fast_math -arch=$CUDA_ARCH -Xcompiler -march=native -Xcompiler -pthread -Xptxas -v"}
set +e
make -B -j"$(nproc)" "$harness" CUDA_ARCH="$CUDA_ARCH" \
    NVCCFLAGS="$nvccflags" >"$OUTPUT_DIR/build.log" 2>&1
build_rc=$?
set -e
if (( build_rc != 0 )); then
    tail -n 160 "$OUTPUT_DIR/build.log" >&2 || true
    die "compressor state-fusion evidence build failed"
fi
[[ -x $harness ]] || die "CUDA harness binary is missing"

current_phase=resource-audit
cuobjdump --list-elf "$harness" >"$OUTPUT_DIR/elf-list.txt" 2>&1
grep -Eq '\.sm_75\.cubin([[:space:]]|$)' "$OUTPUT_DIR/elf-list.txt" ||
    die "harness does not contain an sm_75 cubin"
cuobjdump --dump-sass "$harness" | c++filt >"$OUTPUT_DIR/sass.demangled.txt" 2>&1
c++filt <"$OUTPUT_DIR/build.log" >"$OUTPUT_DIR/build.demangled.log"

python3 - "$OUTPUT_DIR/sass.demangled.txt" "$OUTPUT_DIR/build.demangled.log" \
        "$OUTPUT_DIR/resource-summary.csv" <<'PY'
import csv
import re
import sys

sass_path, build_path, output_path = sys.argv[1:]
needle = "matmul_f16_pair_compressor_store_ordered_chunks_kernel("

with open(sass_path, encoding="utf-8", errors="replace") as handle:
    sections, current = {}, None
    for line in handle:
        match = re.search(r"Function\s*:\s*(.*\S)", line)
        if match:
            current = match.group(1)
            sections.setdefault(current, [])
        if current is not None:
            sections[current].append(line)

with open(build_path, encoding="utf-8", errors="replace") as handle:
    properties, current = {}, None
    for line in handle:
        match = re.search(r"Function properties for\s+(.*\S)", line)
        if match:
            current = match.group(1)
            properties.setdefault(current, {})
            continue
        if current is None:
            continue
        match = re.search(
            r"(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*"
            r"(\d+) bytes spill loads", line)
        if match:
            properties[current].update(zip(
                ("stack", "spill_stores", "spill_loads"), map(int, match.groups())))
        match = re.search(r"Used\s+(\d+) registers", line)
        if match:
            properties[current]["registers"] = int(match.group(1))

sass_matches = [(name, body) for name, body in sections.items() if needle in name]
ptxas_matches = [(name, values) for name, values in properties.items() if needle in name]
if len(sass_matches) != 1:
    raise SystemExit(f"expected one fused-kernel SASS match, got {len(sass_matches)}")
if len(ptxas_matches) != 1:
    raise SystemExit(f"expected one fused-kernel PTXAS match, got {len(ptxas_matches)}")
sass_name, body = sass_matches[0]
ptxas_name, values = ptxas_matches[0]
missing = {"stack", "spill_stores", "spill_loads", "registers"} - values.keys()
if missing:
    raise SystemExit(f"incomplete PTXAS evidence: {sorted(missing)}")
text = "".join(body)
row = {
    "kernel": "compressor-pair-state-store",
    "registers": values["registers"],
    "stack_frame_bytes": values["stack"],
    "spill_store_bytes": values["spill_stores"],
    "spill_load_bytes": values["spill_loads"],
    "sass_ldl": len(re.findall(r"\bLDL(?:\.|\b)", text)),
    "sass_stl": len(re.findall(r"\bSTL(?:\.|\b)", text)),
    "sass_symbol": sass_name,
    "ptxas_symbol": ptxas_name,
}
with open(output_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=row.keys())
    writer.writeheader()
    writer.writerow(row)
if any(row[key] for key in (
        "stack_frame_bytes", "spill_store_bytes", "spill_load_bytes",
        "sass_ldl", "sass_stl")):
    raise SystemExit(f"fused-kernel resource gate failed: {row}")
print("validated fused kernel: zero stack/spills and zero SASS local traffic")
PY
cat "$OUTPUT_DIR/resource-summary.csv"

current_phase=exact-and-timing
run_case() {
        local width=$1 phase=$2 ape=$3 pos=$4 rounds=$5 repeats=$6
        local stem="width${width}-${phase}-${ape}-pos${pos}"
        printf 'Compressor fusion width=%s phase=%s ape=%s pos=%s...\n' \
            "$width" "$phase" "$ape" "$pos"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            TIMING_ROUNDS="$rounds" TIMING_REPEATS="$repeats" \
            ./$harness "$width" "$phase" "$ape" "$pos" \
            >"$OUTPUT_DIR/results/$stem.log" 2>&1 || {
                tail -n 120 "$OUTPUT_DIR/results/$stem.log" >&2 || true
                die "width=$width phase=$phase ape=$ape pos=$pos run failed"
            }
        grep -q '^validation=byte-exact-nonzero$' \
            "$OUTPUT_DIR/results/$stem.log" ||
            die "$stem exact marker missing"
        grep -q '^canaries=passed$' \
            "$OUTPUT_DIR/results/$stem.log" ||
            die "$stem canary marker missing"
}
for width in 256 1024; do
    for ape in f16 f32; do
        run_case "$width" nonemit "$ape" 0 "$TIMING_ROUNDS" "$TIMING_REPEATS"
        run_case "$width" nonemit "$ape" 1 1 1
        run_case "$width" nonemit "$ape" 2 1 1
        run_case "$width" emit "$ape" 3 "$TIMING_ROUNDS" "$TIMING_REPEATS"
        run_case "$width" nonemit "$ape" 4 1 1
        run_case "$width" emit "$ape" 7 1 1
    done
done
for ape in f16 f32; do
    run_case 512 nonemit "$ape" 0 "$TIMING_ROUNDS" "$TIMING_REPEATS"
    for pos in 1 63 126 128; do
        run_case 512 nonemit "$ape" "$pos" 1 1
    done
    run_case 512 emit "$ape" 127 "$TIMING_ROUNDS" "$TIMING_REPEATS"
    run_case 512 emit "$ape" 255 1 1
done

if [[ $RUN_SANITIZER == 1 ]]; then
    current_phase=compute-sanitizer
    for case in '256 nonemit f32 2' '256 emit f16 3' \
                '512 nonemit f16 128' '512 emit f32 127' \
                '1024 nonemit f16 1' '1024 emit f32 7'; do
        read -r width phase ape pos <<<"$case"
        printf 'Compute Sanitizer width=%s phase=%s ape=%s pos=%s...\n' \
            "$width" "$phase" "$ape" "$pos"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" TIMING_ROUNDS=1 TIMING_REPEATS=1 \
            compute-sanitizer --tool memcheck --error-exitcode=1 \
            ./$harness "$width" "$phase" "$ape" "$pos" \
            >"$OUTPUT_DIR/sanitizer/width${width}-${phase}-${ape}-pos${pos}.log" 2>&1 || {
                tail -n 160 "$OUTPUT_DIR/sanitizer/width${width}-${phase}-${ape}-pos${pos}.log" >&2 || true
                die "Compute Sanitizer failed for width=$width phase=$phase ape=$ape pos=$pos"
            }
    done
fi

current_phase=summary
python3 - "$OUTPUT_DIR/results" "$OUTPUT_DIR/timing-summary.csv" <<'PY'
import csv
import pathlib
import sys

source, output = map(pathlib.Path, sys.argv[1:])
fields = [
    "width", "head_dim", "ratio", "ape_type", "phase", "pos",
    "stage_control_median_ms", "stage_fused_median_ms", "stage_fused_speedup",
    "chain_control_median_ms", "chain_fused_median_ms", "chain_fused_speedup",
]
rows = []
for path in sorted(source.glob("*.log")):
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    missing = [field for field in fields if field not in values]
    if missing:
        raise SystemExit(f"{path}: missing timing fields {missing}")
    rows.append({field: values[field] for field in fields})
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY
cat "$OUTPUT_DIR/timing-summary.csv"
printf 'SM75 compressor projection/state-store fusion experiment complete: %s\n' "$OUTPUT_DIR"
