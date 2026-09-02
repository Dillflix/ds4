#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Build and test the diagnostic SM75 exact two-score-pass indexed-decode
attention candidates (one, four, or eight heads per CTA). H1 is a diagnostic
control; H4/H8 remain the grouped optimization candidates.

This does not enable a production default.  It requires byte-exact output for
the actual 32K decode shape and hostile fixtures, enforces runtime resource and
occupancy gates, optionally runs Compute Sanitizer, records per-symbol SASS,
and optionally times the reference/H1/H4/H8 paths.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  TIMING_ROUNDS=9
  TIMING_REPEATS=100
  RUN_SANITIZER=1
  CREATE_ARCHIVE=1
  ATTN_STREAMING_EXACT_DIR=/absolute/output/directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
TIMING_REPEATS=${TIMING_REPEATS:-100}
TIMING_ROUNDS=${TIMING_ROUNDS:-9}
RUN_SANITIZER=${RUN_SANITIZER:-1}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${ATTN_STREAMING_EXACT_DIR:-$repo_dir/sm75-attention-streaming-exact-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be sm_75"
[[ $TIMING_REPEATS =~ ^[0-9]+$ ]] && (( TIMING_REPEATS <= 10000 )) ||
    die "TIMING_REPEATS must be 0..10000"
[[ $TIMING_ROUNDS =~ ^[0-9]+$ ]] && (( TIMING_ROUNDS <= 101 )) ||
    die "TIMING_ROUNDS must be 0..101"
(( (TIMING_ROUNDS == 0) == (TIMING_REPEATS == 0) )) ||
    die "TIMING_ROUNDS and TIMING_REPEATS must both be zero or nonzero"
[[ $RUN_SANITIZER == 0 || $RUN_SANITIZER == 1 ]] ||
    die "RUN_SANITIZER must be 0 or 1"
[[ $CREATE_ARCHIVE == 0 || $CREATE_ARCHIVE == 1 ]] ||
    die "CREATE_ARCHIVE must be 0 or 1"

tools=(c++filt cuobjdump env git grep make nproc nvidia-smi python3 tar)
for tool in "${tools[@]}"; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
if [[ $RUN_SANITIZER == 1 ]]; then
    command -v compute-sanitizer >/dev/null 2>&1 ||
        die "RUN_SANITIZER=1 but compute-sanitizer was not found"
fi

compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] ||
    die "physical GPU $PROFILE_GPU has compute capability ${compute_cap:-unknown}; SM75 is required"

[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

current_phase=initialization
finalize() {
    local status=$?
    trap - EXIT
    if [[ -d $OUTPUT_DIR ]]; then
        printf 'state=finished\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
            "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >"$OUTPUT_DIR/run-status.txt"
        if [[ $CREATE_ARCHIVE == 1 ]]; then
            local archive="$OUTPUT_DIR.tar.gz"
            local partial="$archive.partial"
            rm -f -- "$partial"
            if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                    "$(basename "$OUTPUT_DIR")" &&
               mv -f -- "$partial" "$archive"; then
                printf 'Archive to return: %s\n' "$archive"
            else
                status=1
                rm -f -- "$partial"
            fi
        fi
    fi
    exit "$status"
}
trap finalize EXIT

{
    printf 'date_utc=%s\nrepo=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
    printf 'git_commit=%s\ngit_branch=%s\n' \
        "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'scope=diagnostic-exact-two-score-pass-indexed-decode\n'
    printf 'production_default=unchanged\n'
    printf 'profile_gpu=%s\ncuda_arch=%s\ntiming_rounds=%s\ntiming_repeats=%s\n' \
        "$PROFILE_GPU" "$CUDA_ARCH" "$TIMING_ROUNDS" "$TIMING_REPEATS"
    printf 'run_sanitizer=%s\n' "$RUN_SANITIZER"
    printf '\n[gpu]\n'
    nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,name,pci.bus_id,memory.total,memory.free,driver_version \
        --format=csv
    printf '\n[git status]\n'
    git status --short
} >"$OUTPUT_DIR/manifest.txt"

harness=tests/cuda_attention_streaming_exact
current_phase=forced-build
evidence_nvccflags=${EVIDENCE_NVCCFLAGS:-"-O3 -g -lineinfo --use_fast_math -arch=$CUDA_ARCH -Xcompiler -march=native -Xcompiler -pthread -Xptxas -v"}
set +e
make -B -j"$(nproc)" "$harness" CUDA_ARCH="$CUDA_ARCH" \
    NVCCFLAGS="$evidence_nvccflags" >"$OUTPUT_DIR/build.log" 2>&1
build_rc=$?
set -e
if (( build_rc != 0 )); then
    tail -n 160 "$OUTPUT_DIR/build.log" >&2 || true
    die "SM75 exact-streaming evidence build failed"
fi
[[ -x $harness ]] || die "exact-streaming harness is missing after build"
c++filt <"$OUTPUT_DIR/build.log" >"$OUTPUT_DIR/build.demangled.log"

current_phase=exact-and-resource-gates
printf 'Exact/resource regression: shipping reference versus diagnostic H1 and H4/H8...\n'
env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
    DS4_STREAMING_EXACT_TIMING_ROUNDS=0 \
    DS4_STREAMING_EXACT_TIMING_REPEATS=0 \
    ./$harness >"$OUTPUT_DIR/exact.log" 2>&1 || {
        tail -n 180 "$OUTPUT_DIR/exact.log" >&2 || true
        die "exact/resource regression failed"
    }
grep -q '^resources,group=1,' "$OUTPUT_DIR/exact.log" ||
    die "H1 diagnostic runtime resource identity is missing"
grep -q '^resources,group=4,' "$OUTPUT_DIR/exact.log" ||
    die "H4 runtime resource identity is missing"
grep -q '^resources,group=8,' "$OUTPUT_DIR/exact.log" ||
    die "H8 runtime resource identity is missing"
grep -q '^fixture,case=production-four-gpu-shard-after-32k,class=production,pos=32768,allocated_n_raw=128,raw_cap=2304,raw_start=385,effective_raw_count=128,n_comp=8192,top_k=512,window=128,ratio=4,n_head=32,head_dim=512$' \
    "$OUTPUT_DIR/exact.log" || die "actual four-GPU first-token-after-32K fixture identity is missing"
grep -q '^fixture,case=stress-max-768-rows,class=stress-only,' \
    "$OUTPUT_DIR/exact.log" || die "explicit 768-row stress fixture identity is missing"
grep -q '^fixture,case=selected-score-tie-after-32k-h32,class=adversarial,' \
    "$OUTPUT_DIR/exact.log" || die "selected-score-tie fixture identity is missing"
grep -q '^exact streaming indexed-decode experiment: OK$' \
    "$OUTPUT_DIR/exact.log" || die "exact-regression completion marker is missing"
grep '^resources,' "$OUTPUT_DIR/exact.log" >"$OUTPUT_DIR/runtime-resources.csv"

current_phase=sass-capture
cuobjdump --list-elf "$harness" >"$OUTPUT_DIR/elf-list.txt" 2>&1
grep -Eq '\.sm_75\.cubin([[:space:]]|$)' "$OUTPUT_DIR/elf-list.txt" ||
    die "harness does not contain an sm_75 cubin"
cuobjdump --dump-sass "$harness" | c++filt \
    >"$OUTPUT_DIR/sass.demangled.txt" 2>&1
python3 - "$OUTPUT_DIR/sass.demangled.txt" \
        "$OUTPUT_DIR/build.demangled.log" \
        "$OUTPUT_DIR/sass-symbol-diagnostics.csv" <<'PY'
import csv
import re
import sys

source, build_source, output = sys.argv[1:]
sections = {}
current = None
with open(source, encoding="utf-8", errors="replace") as handle:
    for line in handle:
        match = re.search(r"Function\s*:\s*(.*\S)", line)
        if match:
            current = match.group(1)
            sections.setdefault(current, [])
        if current is not None:
            sections[current].append(line)

with open(build_source, encoding="utf-8", errors="replace") as handle:
    build = handle.read()

rows = []
for group in (4, 8):
    token = rf"(?:{group}(?:u)?|\(unsigned int\){group})"
    pattern = re.compile(
        rf"attention_indexed_mixed_decode_streaming_exact_kernel<{token}>")
    matches = [(name, body) for name, body in sections.items()
               if pattern.search(name)]
    if len(matches) != 1:
        raise SystemExit(
            f"expected one demangled SASS section for H{group}, got {len(matches)}")
    name, body = matches[0]
    text = "".join(body)
    ptx_pattern = re.compile(
        rf"Function properties for .*attention_indexed_mixed_decode_streaming_exact_kernel<{token}>.*?"
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads.*?"
        r"Used (\d+) registers",
        re.DOTALL)
    ptx_matches = ptx_pattern.findall(build)
    if len(ptx_matches) != 1:
        raise SystemExit(
            f"expected one PTXAS resource record for H{group}, got {len(ptx_matches)}")
    stack, spill_stores, spill_loads, registers = map(int, ptx_matches[0])
    sass_ldl = len(re.findall(r"\bLDL(?:\.|\b)", text))
    sass_stl = len(re.findall(r"\bSTL(?:\.|\b)", text))
    rows.append({
        "heads_per_group": group,
        "ptxas_registers": registers,
        "ptxas_stack_frame_bytes": stack,
        "ptxas_spill_store_bytes": spill_stores,
        "ptxas_spill_load_bytes": spill_loads,
        "sass_ldl": sass_ldl,
        "sass_stl": sass_stl,
        "symbol": name,
        "resource_gate": "pass" if not (stack or spill_stores or spill_loads or sass_ldl or sass_stl) else "fail",
    })
    if stack or spill_stores or spill_loads or sass_ldl or sass_stl:
        raise SystemExit(
            f"H{group} has stack/spill/local traffic: stack={stack} "
            f"spill_store={spill_stores} spill_load={spill_loads} "
            f"LDL={sass_ldl} STL={sass_stl}")

with open(output, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
PY
cat "$OUTPUT_DIR/sass-symbol-diagnostics.csv"

if [[ $RUN_SANITIZER == 1 ]]; then
    current_phase=compute-sanitizer
    printf 'Compute Sanitizer: exact/adversarial H1/H4/H8 regression...\n'
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        DS4_STREAMING_EXACT_TIMING_ROUNDS=0 \
        DS4_STREAMING_EXACT_TIMING_REPEATS=0 \
        compute-sanitizer --tool memcheck --error-exitcode 99 \
        ./$harness >"$OUTPUT_DIR/memcheck.log" 2>&1 || {
            tail -n 180 "$OUTPUT_DIR/memcheck.log" >&2 || true
            die "Compute Sanitizer failed"
        }
    grep -q 'ERROR SUMMARY: 0 errors' "$OUTPUT_DIR/memcheck.log" ||
        die "Compute Sanitizer did not report zero errors"
fi

if (( TIMING_REPEATS > 0 )); then
    current_phase=timing
    printf 'Paired timing shipping versus diagnostic H1 and H4/H8 at the four-GPU H32 shape...\n'
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        DS4_STREAMING_EXACT_TIMING_ROUNDS="$TIMING_ROUNDS" \
        DS4_STREAMING_EXACT_TIMING_REPEATS="$TIMING_REPEATS" \
        ./$harness >"$OUTPUT_DIR/timing.log" 2>&1 || {
            tail -n 180 "$OUTPUT_DIR/timing.log" >&2 || true
            die "timed exact-streaming run failed"
        }
    grep '^timing,' "$OUTPUT_DIR/timing.log" \
        >"$OUTPUT_DIR/timing-summary.csv"
    [[ $(wc -l <"$OUTPUT_DIR/timing-summary.csv") -eq 3 ]] ||
        die "timing run did not report paired H1, H4, and H8 comparisons"
    cat "$OUTPUT_DIR/timing-summary.csv"
fi

current_phase=complete
printf 'SM75 exact two-score-pass indexed-decode experiment complete: %s\n' \
    "$OUTPUT_DIR"
