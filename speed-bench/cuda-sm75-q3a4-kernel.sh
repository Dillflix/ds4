#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Build, validate, and benchmark the opt-in SM75 Q3A4 gate/up pair evaluator.
The bounded harness uses a captured production expert histogram and does not
open a GGUF. Baseline and candidate remain separate kernel instantiations.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  TIMING_ROUNDS=4
  TIMING_REPEATS=10
  RUN_SANITIZER=0
  RUN_NCU=1
  NCU_USE_SUDO=0
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  Q3A4_KERNEL_DIR=/absolute/new/output/path
  EVIDENCE_NVCCFLAGS=...
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
script_rel=speed-bench/cuda-sm75-q3a4-kernel.sh
summary_rel=speed-bench/summarize-sm75-q3a4-kernel.py
validator_rel=speed-bench/validate-ncu-capture.py
smoke_rel=tests/cuda_long_context_smoke
profile_rel=tests/cuda_sm75_profile_harness

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
TIMING_ROUNDS=${TIMING_ROUNDS:-4}
TIMING_REPEATS=${TIMING_REPEATS:-10}
RUN_SANITIZER=${RUN_SANITIZER:-0}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${Q3A4_KERNEL_DIR:-$repo_dir/sm75-q3a4-kernel-$stamp}

BASE_REGEX='moe_gate_up_mid_sm75_q32_tile8_kernel.*Lb1ELb0E'
CANDIDATE_REGEX='moe_gate_up_mid_sm75_q32_tile8_kernel.*Lb1ELb1E'

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a GPU index"
[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be exactly sm_75"
[[ $TIMING_ROUNDS =~ ^[1-9][0-9]*$ ]] ||
    die "TIMING_ROUNDS must be a positive integer"
(( 10#$TIMING_ROUNDS % 2 == 0 )) || die "TIMING_ROUNDS must be even"
[[ $TIMING_REPEATS =~ ^[1-9][0-9]*$ ]] ||
    die "TIMING_REPEATS must be a positive integer"
(( 10#$TIMING_REPEATS <= 100 )) || die "TIMING_REPEATS must not exceed 100"
for flag in RUN_SANITIZER RUN_NCU NCU_USE_SUDO SKIP_BUILD CREATE_ARCHIVE; do
    value=${!flag}
    [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
for tool in make nproc python3 cuobjdump nvidia-smi sha256sum tar; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
[[ -f $summary_rel ]] || die "missing $summary_rel"
[[ -f $validator_rel ]] || die "missing $validator_rel"
[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{provenance,smoke,timing,ncu}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] ||
    die "physical GPU $PROFILE_GPU is not SM75 (${compute_cap:-unknown})"
free_mib=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $free_mib =~ ^[0-9]+$ ]] || die "could not read free VRAM"
(( free_mib >= 4096 )) || die "at least 4096 MiB free VRAM is required"

phase=initialization
caught_signal=
take_ownership() {
    if [[ $NCU_USE_SUDO == 1 ]] && command -v sudo >/dev/null 2>&1; then
        sudo -n chown -R -- "$(id -u):$(id -g)" "$OUTPUT_DIR" \
            >/dev/null 2>&1 || true
    fi
}
finalize() {
    local status=$?
    trap - EXIT INT TERM HUP
    take_ownership
    local state=failed
    [[ -n $caught_signal ]] && state=interrupted
    (( status != 0 )) || state=complete
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\nsignal=%s\n' \
        "$state" "$status" "$phase" "${caught_signal:-none}" \
        >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        local archive=$OUTPUT_DIR.tar.gz
        local partial=$OUTPUT_DIR.tar.gz.partial.$$
        if [[ ! -e $archive ]] &&
                tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                    "$(basename "$OUTPUT_DIR")" &&
                mv -- "$partial" "$archive"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            rm -f -- "$partial"
            printf 'warning: could not create archive for %s\n' \
                "$OUTPUT_DIR" >&2
            status=1
        fi
    fi
    exit "$status"
}
trap finalize EXIT
trap 'caught_signal=INT; exit 130' INT
trap 'caught_signal=TERM; exit 143' TERM
trap 'caught_signal=HUP; exit 129' HUP

{
    printf 'date_utc=%s\nrepo=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
    printf 'git_commit=%s\ngit_branch=%s\n' \
        "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'profile_gpu=%s\ncompute_capability=%s\nfree_mib=%s\n' \
        "$PROFILE_GPU" "$compute_cap" "$free_mib"
    printf 'timing_rounds=%s\ntiming_repeats=%s\n' \
        "$TIMING_ROUNDS" "$TIMING_REPEATS"
    printf 'candidate_env=DS4_CUDA_MOE_Q3A4_PAIR_FUSED_SM75\n'
    printf 'candidate_default=false\nfull_model_loaded=false\n'
    printf '\n[gpu]\n'
    nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,name,pci.bus_id,memory.total,memory.free,driver_version \
        --format=csv
    printf '\n[git status]\n'; git status --short
    printf '\n[toolchain]\n'
    "${NVCC:-/usr/local/cuda/bin/nvcc}" --version
} >"$OUTPUT_DIR/manifest.txt" 2>&1

for path in "$script_rel" "$summary_rel" "$validator_rel" ds4_cuda.cu \
        ds4_cuda_sm75_q32_native.inc.cu tests/cuda_long_context_smoke.c \
        tests/cuda_sm75_profile_harness.c Makefile; do
    cp -- "$path" "$OUTPUT_DIR/provenance/"
    sha256sum "$path"
done >"$OUTPUT_DIR/provenance/source-sha256.txt"
git diff --no-ext-diff --binary HEAD -- ds4_cuda.cu \
    ds4_cuda_sm75_q32_native.inc.cu tests/cuda_long_context_smoke.c \
    tests/cuda_sm75_profile_harness.c "$script_rel" "$summary_rel" \
    >"$OUTPUT_DIR/provenance/tracked-working-tree.patch" || true

phase=build
evidence_flags=${EVIDENCE_NVCCFLAGS:-"-O3 -g -lineinfo --use_fast_math -arch=$CUDA_ARCH -Xcompiler -march=native -Xcompiler -pthread -Xptxas -v"}
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "$smoke_rel" "$profile_rel" \
        CUDA_ARCH="$CUDA_ARCH" NVCCFLAGS="$evidence_flags" \
        >"$OUTPUT_DIR/build.log" 2>&1 || {
            tail -n 160 "$OUTPUT_DIR/build.log" >&2 || true
            die "Q3A4 evidence build failed"
        }
else
    set +e
    make -q "$smoke_rel" "$profile_rel" CUDA_ARCH="$CUDA_ARCH" \
        >"$OUTPUT_DIR/build.log" 2>&1
    query_rc=$?
    set -e
    (( query_rc == 0 )) ||
        die "SKIP_BUILD=1 rejected stale targets; rerun with SKIP_BUILD=0"
fi
[[ -x $smoke_rel && -x $profile_rel ]] ||
    die "required CUDA binaries are missing"
sha256sum "$smoke_rel" "$profile_rel" \
    >"$OUTPUT_DIR/provenance/binary-sha256.txt"

phase=structure
cuobjdump --list-elf "$profile_rel" >"$OUTPUT_DIR/elf-list.txt" 2>&1
grep -Eq '\.sm_75\.cubin([[:space:]]|$)' "$OUTPUT_DIR/elf-list.txt" ||
    die "profile harness does not contain an sm_75 cubin"
cuobjdump --dump-resource-usage "$profile_rel" \
    >"$OUTPUT_DIR/resource-usage.txt" 2>&1
cuobjdump --dump-sass "$profile_rel" >"$OUTPUT_DIR/sass.txt" 2>&1
python3 - "$OUTPUT_DIR/sass.txt" "$BASE_REGEX" "$CANDIDATE_REGEX" \
        "$OUTPUT_DIR/sass-summary.csv" <<'PY'
import csv
import re
import sys

source, base_pattern, candidate_pattern, output = sys.argv[1:]
patterns = {
    "baseline": re.compile(base_pattern),
    "pair-fused": re.compile(candidate_pattern),
}
with open(source, encoding="utf-8", errors="replace") as handle:
    lines = handle.readlines()
sections, current = {}, None
for line in lines:
    match = re.search(r"Function\s*:\s*(\S+)", line)
    if match:
        current = match.group(1)
        sections.setdefault(current, [])
    if current is not None:
        sections[current].append(line)
rows = []
for variant, pattern in patterns.items():
    matches = [(name, "".join(body)) for name, body in sections.items()
               if pattern.search(name)]
    if not matches:
        raise SystemExit(f"no SASS function matched {variant}: {pattern.pattern}")
    for name, body in matches:
        row = {
            "variant": variant,
            "kernel": name,
            "imma": len(re.findall(r"\bIMMA(?:\.|\b)", body)),
            "ldl": len(re.findall(r"\bLDL(?:\.|\b)", body)),
            "stl": len(re.findall(r"\bSTL(?:\.|\b)", body)),
        }
        if row["imma"] == 0:
            raise SystemExit(f"{name} has no IMMA instructions")
        if row["ldl"] or row["stl"]:
            raise SystemExit(
                f"{name} uses local memory: LDL={row['ldl']} STL={row['stl']}")
        rows.append(row)
with open(output, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"validated {len(rows)} Q3A4 SASS functions")
PY

phase=exact
env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
    DS4_CUDA_MOE_Q3A4_PAIR_FUSED_SM75=0 \
    "./$smoke_rel" >"$OUTPUT_DIR/smoke/exact.log" 2>&1 || {
        tail -n 160 "$OUTPUT_DIR/smoke/exact.log" >&2 || true
        die "Q3A4 production exactness smoke failed"
    }
grep -Fq 'SM75 Q3A4 gate/up + Q4-32 down production 16/8/4 prefill/direct-decode, pair-fused exact' \
    "$OUTPUT_DIR/smoke/exact.log" ||
    die "exactness smoke omitted the Q3A4 pair-fused proof"

phase=sanitizer
if [[ $RUN_SANITIZER == 1 ]]; then
    command -v compute-sanitizer >/dev/null 2>&1 ||
        die "RUN_SANITIZER=1 but compute-sanitizer was not found"
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        compute-sanitizer --tool memcheck --error-exitcode 3 \
        "./$smoke_rel" >"$OUTPUT_DIR/smoke/memcheck.log" 2>&1 || {
            tail -n 160 "$OUTPUT_DIR/smoke/memcheck.log" >&2 || true
            die "Q3A4 compute-sanitizer pass failed"
        }
    grep -Eq 'ERROR SUMMARY: 0 errors' "$OUTPUT_DIR/smoke/memcheck.log" ||
        die "compute-sanitizer did not report zero errors"
else
    printf 'skipped: RUN_SANITIZER=0\n' \
        >"$OUTPUT_DIR/smoke/memcheck.log"
fi

phase=timing
samples=$OUTPUT_DIR/timing/samples.csv
printf 'round,slot,variant,timed_repeats,timed_total_ms,timed_per_call_ms\n' \
    >"$samples"
run_sample() {
    local round=$1 slot=$2 variant=$3 enabled=$4
    local log=$OUTPUT_DIR/timing/r${round}-${slot}-${variant}.log
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        DS4_PROFILE_REPEATS="$TIMING_REPEATS" \
        DS4_CUDA_MOE_Q3A4_PAIR_FUSED_SM75="$enabled" \
        "./$profile_rel" sm75-q3a4 >"$log" 2>&1 || {
            tail -n 120 "$log" >&2 || true
            die "Q3A4 timing sample failed: $variant round $round"
        }
    grep -Fxq 'harness_status=ok' "$log" ||
        die "Q3A4 timing harness omitted success"
    local repeats total per_call
    repeats=$(awk -F= '$1=="timed_repeats"{v=$2} END{print v}' "$log")
    total=$(awk -F= '$1=="timed_total_ms"{v=$2} END{print v}' "$log")
    per_call=$(awk -F= '$1=="timed_per_call_ms"{v=$2} END{print v}' "$log")
    [[ $repeats == "$TIMING_REPEATS" ]] ||
        die "unexpected repeat count in $log"
    [[ $total =~ ^[0-9]+([.][0-9]+)?$ &&
       $per_call =~ ^[0-9]+([.][0-9]+)?$ ]] ||
        die "invalid internal timing in $log"
    printf '%s,%s,%s,%s,%s,%s\n' \
        "$round" "$slot" "$variant" "$repeats" "$total" "$per_call" \
        >>"$samples"
}
for ((round=0; round<TIMING_ROUNDS; round++)); do
    if (( round % 2 == 0 )); then
        run_sample "$round" 1 baseline 0
        run_sample "$round" 2 pair-fused 1
    else
        run_sample "$round" 1 pair-fused 1
        run_sample "$round" 2 baseline 0
    fi
done
python3 "$summary_rel" "$samples" "$OUTPUT_DIR/timing/summary.csv" \
    "$TIMING_ROUNDS" | tee "$OUTPUT_DIR/timing/summary.txt"

phase=nsight-compute
if [[ $RUN_NCU == 1 ]]; then
    command -v ncu >/dev/null 2>&1 || die "RUN_NCU=1 but ncu was not found"
    ncu_bin=$(command -v ncu)
    if [[ $NCU_USE_SUDO == 1 ]]; then
        command -v sudo >/dev/null 2>&1 || die "sudo not found"
        sudo -v
    fi
    metrics=$(
        printf '%s,' \
            gpu__time_duration.sum \
            launch__registers_per_thread \
            launch__shared_mem_per_block \
            launch__occupancy_limit_blocks \
            launch__occupancy_limit_registers \
            launch__occupancy_limit_shared_mem \
            launch__occupancy_limit_warps \
            sm__warps_active.avg.pct_of_peak_sustained_active \
            smsp__warps_eligible.avg.per_cycle_active \
            sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_elapsed \
            l1tex__t_sector_hit_rate.pct \
            lts__t_sector_hit_rate.pct \
            dram__bytes.sum.per_second \
            smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio \
            smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
    )
    metrics=${metrics%,}
    profile_one() {
        local variant=$1 enabled=$2 regex=$3
        local base=$OUTPUT_DIR/ncu/$variant
        local -a command
        if [[ $NCU_USE_SUDO == 1 ]]; then
            command=(sudo -E "$ncu_bin")
        else
            command=("$ncu_bin")
        fi
        set +e
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            DS4_PROFILE_REPEATS=0 \
            DS4_CUDA_MOE_Q3A4_PAIR_FUSED_SM75="$enabled" \
            "${command[@]}" --config-file off --devices 0 \
                --target-processes application-only \
                --kernel-name-base function \
                --kernel-name "regex:$regex" \
                --launch-count 1 --replay-mode kernel \
                --cache-control none --clock-control none \
                --force-overwrite --export "$base" \
                --metrics "$metrics" --disable-extra-suffixes \
                "./$profile_rel" sm75-q3a4 >"$base.log" 2>&1
        rc=$?
        set -e
        (( rc == 0 )) || {
            tail -n 120 "$base.log" >&2 || true
            die "Nsight Compute failed for $variant"
        }
        [[ -s $base.ncu-rep ]] || die "missing Nsight report for $variant"
        if [[ $NCU_USE_SUDO == 1 ]]; then
            sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
        fi
        "$ncu_bin" --config-file off --import "$base.ncu-rep" \
            --csv --page raw >"$base.csv" 2>"$base-import.log"
        python3 "$validator_rel" "$base.csv" "$regex" 0 \
            --process cuda_sm75_profile_harness >"$base-validation.txt"
    }
    profile_one baseline 0 "$BASE_REGEX"
    profile_one pair-fused 1 "$CANDIDATE_REGEX"
else
    printf 'skipped: RUN_NCU=0\n' >"$OUTPUT_DIR/ncu/README.txt"
fi

phase=complete
printf 'q3a4_pair_fused_evidence_status=ok\n' \
    >"$OUTPUT_DIR/acceptance.txt"
printf 'SM75 Q3A4 kernel evidence complete: %s\n' "$OUTPUT_DIR"
