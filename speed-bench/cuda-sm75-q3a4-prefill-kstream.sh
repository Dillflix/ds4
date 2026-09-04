#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Build, validate, and benchmark the opt-in SM75 Q3A4 prefill tile16/K-streaming
kernel. The bounded harness uses the production Q3A4 shape and routed-expert
histogram without opening a GGUF. It is a promotion gate, not full-model
production evidence.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  TIMING_ROUNDS=7
  TIMING_REPEATS=10
  RUN_SANITIZER=1
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  Q3A4_PREFILL_KSTREAM_DIR=/absolute/new/output/path
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
TIMING_ROUNDS=${TIMING_ROUNDS:-7}
TIMING_REPEATS=${TIMING_REPEATS:-10}
RUN_SANITIZER=${RUN_SANITIZER:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${Q3A4_PREFILL_KSTREAM_DIR:-$repo_dir/sm75-q3a4-prefill-kstream-$(date -u +%Y%m%dT%H%M%SZ)}
smoke=tests/cuda_long_context_smoke
profile=tests/cuda_sm75_profile_harness

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a GPU index"
[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be sm_75"
[[ $TIMING_ROUNDS =~ ^[1-9][0-9]*$ ]] || die "TIMING_ROUNDS must be positive"
[[ $TIMING_REPEATS =~ ^[1-9][0-9]*$ ]] || die "TIMING_REPEATS must be positive"
(( TIMING_REPEATS <= 100 )) || die "TIMING_REPEATS must not exceed 100"
for flag in RUN_SANITIZER SKIP_BUILD CREATE_ARCHIVE; do
    [[ ${!flag} == 0 || ${!flag} == 1 ]] || die "$flag must be 0 or 1"
done
for tool in awk basename cuobjdump date dirname env git grep make mkdir nproc \
            nvidia-smi python3 sha256sum sort tail tar tee tr; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] ||
    die "physical GPU $PROFILE_GPU is not SM75 (${compute_cap:-unknown})"
free_mib=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $free_mib =~ ^[0-9]+$ && $free_mib -ge 4096 ]] ||
    die "at least 4096 MiB free VRAM is required"
[[ ! -e $OUTPUT_DIR && ! -e $OUTPUT_DIR.tar.gz ]] ||
    die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{provenance,validation,timing,structure}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

phase=initialization
finish() {
    local status=$? archive="$OUTPUT_DIR.tar.gz"
    trap - EXIT INT TERM HUP
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$phase" >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$archive" \
                "$(basename "$OUTPUT_DIR")"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            status=1
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

{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'profile_gpu=%s\ncompute_capability=%s\nfree_mib=%s\n' \
        "$PROFILE_GPU" "$compute_cap" "$free_mib"
    printf 'scenario=sm75-q3a4\ntiming_rounds=%s\ntiming_repeats=%s\n' \
        "$TIMING_ROUNDS" "$TIMING_REPEATS"
    printf 'candidate=tile16-kstream\ncandidate_default=off\n'
    printf 'full_model_loaded=false\nprimary_followup_model=all43\n'
    printf '\n[gpu]\n'
    nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,name,pci.bus_id,memory.total,memory.free,driver_version \
        --format=csv
    printf '\n[git status]\n'; git status --short
} >"$OUTPUT_DIR/manifest.txt"
git diff --no-ext-diff --binary HEAD -- ds4_cuda.cu \
    ds4_cuda_sm75_q32_native.inc.cu tests/cuda_long_context_smoke.c \
    tests/cuda_sm75_profile_harness.c \
    speed-bench/cuda-sm75-q3a4-prefill-kstream.sh \
    >"$OUTPUT_DIR/provenance/working-tree.patch" || true

phase=build
evidence_flags="-O3 -g -lineinfo --use_fast_math -arch=$CUDA_ARCH -Xcompiler -march=native -Xcompiler -pthread -Xptxas -v"
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "$smoke" "$profile" CUDA_ARCH="$CUDA_ARCH" \
        NVCCFLAGS="$evidence_flags" >"$OUTPUT_DIR/build.log" 2>&1 || {
            tail -n 180 "$OUTPUT_DIR/build.log" >&2 || true
            die "Q3A4 evidence build failed"
        }
else
    make -q "$smoke" "$profile" CUDA_ARCH="$CUDA_ARCH" ||
        die "SKIP_BUILD=1 found stale targets"
fi
sha256sum "$smoke" "$profile" >"$OUTPUT_DIR/provenance/binary-sha256.txt"

phase=structure
cuobjdump --list-elf "$profile" >"$OUTPUT_DIR/structure/elf-list.txt" 2>&1
grep -Eq '\.sm_75\.cubin([[:space:]]|$)' "$OUTPUT_DIR/structure/elf-list.txt" ||
    die "profile harness does not contain an sm_75 cubin"
cuobjdump --dump-resource-usage "$profile" \
    >"$OUTPUT_DIR/structure/resource-usage.txt" 2>&1
cuobjdump --dump-sass "$profile" >"$OUTPUT_DIR/structure/sass.txt" 2>&1
python3 - "$OUTPUT_DIR/structure/sass.txt" \
        "$OUTPUT_DIR/structure/candidate-sass.csv" <<'PY'
import csv, re, sys
lines = open(sys.argv[1], encoding="utf-8", errors="replace").readlines()
sections, current = {}, None
for line in lines:
    match = re.search(r"Function\s*:\s*(\S+)", line)
    if match:
        current = match.group(1)
        sections[current] = []
    if current is not None:
        sections[current].append(line)
matches = [(name, "".join(body)) for name, body in sections.items()
           if "q3a4_tile16_kstream" in name]
if len(matches) != 1:
    raise SystemExit(f"expected one tile16/K-stream kernel, found {len(matches)}")
name, body = matches[0]
row = {"kernel": name,
       "imma": len(re.findall(r"\bIMMA(?:\.|\b)", body)),
       "ldl": len(re.findall(r"\bLDL(?:\.|\b)", body)),
       "stl": len(re.findall(r"\bSTL(?:\.|\b)", body))}
if row["imma"] == 0:
    raise SystemExit("candidate has no IMMA instructions")
if row["ldl"] or row["stl"]:
    raise SystemExit(f"candidate spills/local memory: {row}")
with open(sys.argv[2], "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=row.keys())
    writer.writeheader(); writer.writerow(row)
print(f"candidate SASS validated: IMMA={row['imma']} LDL=0 STL=0")
PY

phase=exactness
"${clean[@]}" CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
    ./$smoke >"$OUTPUT_DIR/validation/exact.log" 2>&1 || {
        tail -n 180 "$OUTPUT_DIR/validation/exact.log" >&2 || true
        die "Q3A4 exactness smoke failed"
    }
grep -Fq 'SM75 Q3A4 gate/up + Q4-32 down production 16/8/4 prefill/direct-decode, tile16-K-stream exact' \
    "$OUTPUT_DIR/validation/exact.log" || die "candidate exactness marker missing"

phase=sanitizer
if [[ $RUN_SANITIZER == 1 ]]; then
    command -v compute-sanitizer >/dev/null 2>&1 ||
        die "RUN_SANITIZER=1 but compute-sanitizer was not found"
    "${clean[@]}" CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        compute-sanitizer --tool memcheck --error-exitcode 3 ./$smoke \
        >"$OUTPUT_DIR/validation/memcheck.log" 2>&1 || {
            tail -n 180 "$OUTPUT_DIR/validation/memcheck.log" >&2 || true
            die "Q3A4 compute-sanitizer pass failed"
        }
    grep -Eq 'ERROR SUMMARY: 0 errors' "$OUTPUT_DIR/validation/memcheck.log" ||
        die "compute-sanitizer did not report zero errors"
else
    printf 'skipped: RUN_SANITIZER=0\n' >"$OUTPUT_DIR/validation/memcheck.log"
fi

phase=timing
samples="$OUTPUT_DIR/timing/samples.csv"
printf 'round,slot,variant,timed_repeats,timed_total_ms,timed_per_call_ms\n' \
    >"$samples"
run_sample() {
    local round=$1 slot=$2 variant=$3 enabled=$4
    local log="$OUTPUT_DIR/timing/r${round}-${slot}-${variant}.log"
    "${clean[@]}" CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        DS4_PROFILE_REPEATS="$TIMING_REPEATS" \
        DS4_PROFILE_Q3A4_TILE16_KSTREAM="$enabled" \
        ./$profile sm75-q3a4 >"$log" 2>&1 || {
            tail -n 140 "$log" >&2 || true
            die "Q3A4 timing failed: $variant round $round"
        }
    grep -Fxq 'harness_status=ok' "$log" || die "timing success marker missing"
    grep -Fxq "q3a4_tile16_kstream=$enabled" "$log" ||
        die "timing selector mismatch"
    if [[ $enabled == 1 ]]; then
        grep -Fq 'Q3A4 prefill candidate selected: tile16 K-streaming' "$log" ||
            die "candidate did not dispatch"
    else
        ! grep -Fq 'Q3A4 prefill candidate selected: tile16 K-streaming' "$log" ||
            die "control dispatched the candidate"
    fi
    local repeats total per_call
    repeats=$(awk -F= '$1=="timed_repeats"{v=$2} END{print v}' "$log")
    total=$(awk -F= '$1=="timed_total_ms"{v=$2} END{print v}' "$log")
    per_call=$(awk -F= '$1=="timed_per_call_ms"{v=$2} END{print v}' "$log")
    [[ $repeats == "$TIMING_REPEATS" && $total =~ ^[0-9]+([.][0-9]+)?$ &&
       $per_call =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid timing in $log"
    printf '%s,%s,%s,%s,%s,%s\n' "$round" "$slot" "$variant" \
        "$repeats" "$total" "$per_call" >>"$samples"
}
for ((round=0; round<TIMING_ROUNDS; round++)); do
    if (( round % 2 == 0 )); then
        run_sample "$round" 1 control 0
        run_sample "$round" 2 tile16-kstream 1
    else
        run_sample "$round" 1 tile16-kstream 1
        run_sample "$round" 2 control 0
    fi
done
python3 - "$samples" "$OUTPUT_DIR/timing/summary.csv" <<'PY' |
        tee "$OUTPUT_DIR/timing/summary.txt"
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
values = {"control": [], "tile16-kstream": []}
for row in rows:
    values[row["variant"]].append(float(row["timed_per_call_ms"]))
if not values["control"] or len(values["control"]) != len(values["tile16-kstream"]):
    raise SystemExit("unbalanced timing samples")
control = statistics.median(values["control"])
candidate = statistics.median(values["tile16-kstream"])
result = {"samples_per_variant": len(values["control"]),
          "control_median_ms": f"{control:.6f}",
          "tile16_kstream_median_ms": f"{candidate:.6f}",
          "speedup_x": f"{control / candidate:.6f}",
          "candidate_change_pct": f"{(candidate / control - 1.0) * 100.0:.3f}"}
with open(sys.argv[2], "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=result.keys())
    writer.writeheader(); writer.writerow(result)
print("Q3A4 tile16/K-stream: "
      f"{control:.3f} ms -> {candidate:.3f} ms ({control / candidate:.3f}x)")
PY

phase=complete
printf 'bounded_gate=passed\nfull_model_promotion=not_yet_tested\n' \
    >"$OUTPUT_DIR/acceptance.txt"
printf 'SM75 Q3A4 prefill tile16/K-streaming bounded gate complete: %s\n' \
    "$OUTPUT_DIR"
