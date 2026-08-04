#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Build and validate the production SM75 scalar-slot kernels without loading a
GGUF. Existing CUDA smoke/profile binaries supply correctness and bounded
production early/late routing.

Optional environment:
  PROFILE_GPU=0          physical SM75 GPU to expose as logical device 0
  CUDA_ARCH=sm_75        must remain sm_75
  TIMING_ROUNDS=2        even balanced samples per variant; pilot default
  TIMING_REPEATS=10      production calls timed inside each sample (1..100)
  RUN_SANITIZER=1        run memcheck when compute-sanitizer is installed
  RUN_NCU=0              collect exact-kernel NCU reports (expensive)
  NCU_USE_SUDO=0         use sudo -E for restricted performance counters
  CREATE_ARCHIVE=1       archive complete or partial evidence after the run
  SCALAR_SLOTS_DIR=...   new output directory (must not already exist)
  EVIDENCE_NVCCFLAGS=... override production flags plus -Xptxas -v
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
script_rel=speed-bench/cuda-sm75-production-scalar.sh
smoke_rel=tests/cuda_long_context_smoke
profile_rel=tests/cuda_sm75_profile_harness
validator_rel=speed-bench/validate-ncu-capture.py

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
TIMING_ROUNDS=${TIMING_ROUNDS:-2}
TIMING_REPEATS=${TIMING_REPEATS:-10}
RUN_SANITIZER=${RUN_SANITIZER:-1}
RUN_NCU=${RUN_NCU:-0}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${SCALAR_SLOTS_DIR:-$repo_dir/sm75-production-scalar-$run_stamp}

# The production implementation uses a bool template specialization, not a
# renamed function. cuobjdump prints Lb0E/Lb1E in mangled names while NCU's
# demangled name prints false/true, so each regex deliberately accepts both.
Q4_GATE_BASE_REGEX='moe_gate_up_mid_q4K_tile8_mma_kernel.*(Lb0E|false)'
Q4_GATE_SCALAR_REGEX='moe_gate_up_mid_q4K_tile8_mma_kernel.*(Lb1ELj256E|true[^>]*256)'
Q4_DOWN_BASE_REGEX='moe_down_q4K_tile16_mma_sm75_kernel.*(Lb0E|false)'
Q4_DOWN_SCALAR_REGEX='moe_down_q4K_tile16_mma_sm75_kernel.*(Lb1ELj256E|true[^>]*256)'
IQ2_TILE8_BASE_REGEX='moe_gate_up_mid_iq2_tile8_mma_sm75_kernel.*(Lb0E|false)'
IQ2_TILE8_SCALAR_REGEX='moe_gate_up_mid_iq2_tile8_mma_sm75_kernel.*(Lb1E|true)'
IQ2_TILE16_BASE_REGEX='moe_gate_up_mid_iq2_tile16_mma_sm75_kernel.*(Lb0E|false)'
IQ2_TILE16_SCALAR_REGEX='moe_gate_up_mid_iq2_tile16_mma_sm75_kernel.*(Lb1E|true)'

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a GPU index"
[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be exactly sm_75"
[[ $TIMING_ROUNDS =~ ^[1-9][0-9]*$ ]] || die "TIMING_ROUNDS must be positive"
(( 10#$TIMING_ROUNDS % 2 == 0 )) || die "TIMING_ROUNDS must be even"
[[ $TIMING_REPEATS =~ ^[1-9][0-9]*$ ]] || die "TIMING_REPEATS must be positive"
(( 10#$TIMING_REPEATS <= 100 )) || die "TIMING_REPEATS must not exceed 100"
for flag in RUN_SANITIZER RUN_NCU NCU_USE_SUDO CREATE_ARCHIVE; do
    value=${!flag}; [[ $value == 0 || $value == 1 ]] || die "$flag must be 0 or 1"
done
for tool in make nproc python3 cuobjdump nvidia-smi sha256sum tar; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/{provenance,smoke,timing,ncu}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

current_phase=initialization
finalize() {
    local status=$?
    trap - EXIT
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
        "$([[ $status == 0 ]] && printf finished || printf failed)" \
        "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
    if [[ $CREATE_ARCHIVE == 1 ]]; then
        if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$OUTPUT_DIR.tar.gz" \
                "$(basename "$OUTPUT_DIR")"; then
            printf 'archive=%s\n' "$OUTPUT_DIR.tar.gz"
        else
            printf 'warning: could not archive evidence directory: %s\n' \
                "$OUTPUT_DIR" >&2
        fi
    fi
    exit "$status"
}
on_signal() {
    local signal=$1 status=$2
    current_phase="interrupted-$signal"
    exit "$status"
}
trap finalize EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] || die "GPU $PROFILE_GPU is not SM75 (${compute_cap:-unknown})"
free_mib=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $free_mib =~ ^[0-9]+$ ]] || die "could not read free VRAM"
(( free_mib >= 4096 )) || die "at least 4096 MiB free VRAM is required"

{
    printf 'date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'git_commit=%s\ngit_branch=%s\n' \
        "$(git rev-parse HEAD 2>/dev/null || printf unknown)" \
        "$(git branch --show-current 2>/dev/null || printf unknown)"
    printf 'cuda_arch=%s\nprofile_gpu_physical=%s\ncompute_capability=%s\n' \
        "$CUDA_ARCH" "$PROFILE_GPU" "$compute_cap"
    printf 'free_mib_at_preflight=%s\ntiming_rounds=%s\ntiming_repeats=%s\n' \
        "$free_mib" "$TIMING_ROUNDS" "$TIMING_REPEATS"
    printf 'run_sanitizer=%s\nrun_ncu=%s\n' "$RUN_SANITIZER" "$RUN_NCU"
    printf 'timing_scope=production_calls_after_correctness_warmup\n'
    printf 'timing_order=balanced_base_scalar\n'
    printf '\n[git status]\n'; git status --short 2>/dev/null || true
    printf '\n[gpu]\n'; nvidia-smi -i "$PROFILE_GPU" 2>&1
    printf '\n[nvcc]\n'; "${NVCC:-/usr/local/cuda/bin/nvcc}" --version 2>&1 || true
    printf '\n[cuobjdump]\n'; cuobjdump --version 2>&1 || true
    printf '\n[compute-sanitizer]\n'; compute-sanitizer --version 2>&1 || true
    printf '\n[ncu]\n'; ncu --version 2>&1 || true
} >"$OUTPUT_DIR/manifest.txt"
for path in "$script_rel" ds4_cuda.cu tests/cuda_long_context_smoke.c \
        tests/cuda_sm75_profile_harness.c Makefile "$validator_rel"; do
    sha256sum "$path"
done >"$OUTPUT_DIR/provenance/sha256.txt"

current_phase=build
evidence_nvccflags=${EVIDENCE_NVCCFLAGS:-"-O3 -g -lineinfo --use_fast_math -arch=$CUDA_ARCH -Xcompiler -march=native -Xcompiler -pthread -Xptxas -v"}
set +e
make -B -j"$(nproc)" "$smoke_rel" "$profile_rel" \
    CUDA_ARCH="$CUDA_ARCH" NVCCFLAGS="$evidence_nvccflags" \
    >"$OUTPUT_DIR/build.log" 2>&1
build_rc=$?
set -e
if (( build_rc != 0 )); then
    tail -n 120 "$OUTPUT_DIR/build.log" >&2 || true
    die "production evidence build failed"
fi
[[ -x $smoke_rel && -x $profile_rel ]] || die "required CUDA binaries are missing"

current_phase=sass-ptxas
cuobjdump --list-elf "$profile_rel" >"$OUTPUT_DIR/elf-list.txt" 2>&1
grep -Eq '\.sm_75\.cubin([[:space:]]|$)' "$OUTPUT_DIR/elf-list.txt" ||
    die "profile harness does not contain an sm_75 cubin"
cuobjdump --dump-resource-usage "$profile_rel" >"$OUTPUT_DIR/resource-usage.txt" 2>&1
cuobjdump --dump-sass "$profile_rel" >"$OUTPUT_DIR/sass.txt" 2>&1

python3 - "$OUTPUT_DIR/sass.txt" "$OUTPUT_DIR/build.log" \
        "$OUTPUT_DIR/sass-summary.csv" "$OUTPUT_DIR/ptxas-summary.csv" \
        "$Q4_GATE_BASE_REGEX" "$Q4_GATE_SCALAR_REGEX" \
        "$Q4_DOWN_BASE_REGEX" "$Q4_DOWN_SCALAR_REGEX" \
        "$IQ2_TILE8_BASE_REGEX" "$IQ2_TILE8_SCALAR_REGEX" \
        "$IQ2_TILE16_BASE_REGEX" "$IQ2_TILE16_SCALAR_REGEX" <<'PY'
import csv, re, sys

sass_path, build_path, sass_out, ptxas_out, *patterns = sys.argv[1:]
names = ("q4-gate", "q4-down", "iq2-gate-tile8", "iq2-gate-tile16")
specs = []
for index, target in enumerate(names):
    specs.extend(((target, "base", re.compile(patterns[index * 2])),
                  (target, "scalar", re.compile(patterns[index * 2 + 1]))))

with open(sass_path, encoding="utf-8", errors="replace") as handle:
    lines = handle.readlines()
sections, current = {}, None
for line in lines:
    match = re.search(r"Function\s*:\s*(\S+)", line)
    if match:
        current = match.group(1); sections.setdefault(current, [])
    if current is not None: sections[current].append(line)

sass_rows = []
for target, variant, pattern in specs:
    matches = [(name, text) for name, text in sections.items() if pattern.search(name)]
    if not matches: raise SystemExit(f"no SASS match for {target}/{variant}: {pattern.pattern}")
    for name, text in matches:
        body = "".join(text)
        sass_rows.append({"target": target, "variant": variant, "kernel": name,
                          "imma": len(re.findall(r"\bIMMA(?:\.|\b)", body)),
                          "ffma": len(re.findall(r"\bFFMA\.FTZ(?:\.|\b)", body)),
                          "ldl": len(re.findall(r"\bLDL(?:\.|\b)", body)),
                          "stl": len(re.findall(r"\bSTL(?:\.|\b)", body))})
with open(sass_out, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=sass_rows[0].keys())
    writer.writeheader(); writer.writerows(sass_rows)

with open(build_path, encoding="utf-8", errors="replace") as handle:
    build = handle.readlines()
properties, current = {}, None
for line in build:
    match = re.search(r"Function properties for\s+(\S+)", line)
    if match:
        current = match.group(1); properties.setdefault(current, {}); continue
    if current is None: continue
    match = re.search(r"(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*(\d+) bytes spill loads", line)
    if match:
        properties[current].update(zip(("stack", "spill_stores", "spill_loads"), map(int, match.groups())))
    match = re.search(r"Used\s+(\d+) registers", line)
    if match: properties[current]["registers"] = int(match.group(1))

ptxas_rows = []
for target, variant, pattern in specs:
    matches = [(name, values) for name, values in properties.items() if pattern.search(name)]
    if not matches: raise SystemExit(f"no PTXAS match for {target}/{variant}: {pattern.pattern}")
    for name, values in matches:
        required = {"stack", "spill_stores", "spill_loads", "registers"}
        if not required.issubset(values): raise SystemExit(f"incomplete PTXAS evidence for {name}: {values}")
        ptxas_rows.append({"target": target, "variant": variant, "kernel": name, **values})
with open(ptxas_out, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=ptxas_rows[0].keys())
    writer.writeheader(); writer.writerows(ptxas_rows)

failures = []
for row in sass_rows:
    if row["variant"] == "scalar" and (row["ldl"] or row["stl"]):
        failures.append(f"{row['kernel']}: LDL={row['ldl']} STL={row['stl']}")
for row in ptxas_rows:
    if row["variant"] == "scalar" and (row["stack"] or row["spill_stores"] or row["spill_loads"]):
        failures.append(f"{row['kernel']}: stack={row['stack']} stores={row['spill_stores']} loads={row['spill_loads']}")
for target in names:
    base = [row for row in sass_rows if row["target"] == target and row["variant"] == "base"]
    scalar = [row for row in sass_rows if row["target"] == target and row["variant"] == "scalar"]
    if len(base) != len(scalar): failures.append(f"{target}: function counts {len(base)} != {len(scalar)}")
    base_imma = sum(row["imma"] for row in base)
    scalar_imma = sum(row["imma"] for row in scalar)
    if base_imma == 0 or scalar_imma == 0:
        failures.append(f"{target}: expected nonzero IMMA instructions (base={base_imma}, scalar={scalar_imma})")
    if base_imma != scalar_imma:
        failures.append(f"{target}: aggregate IMMA count differs")
    if target.startswith("iq2-"):
        base_ffma = sum(row["ffma"] for row in base)
        scalar_ffma = sum(row["ffma"] for row in scalar)
        missing_ffma = [row["kernel"] for row in scalar if row["ffma"] == 0]
        if missing_ffma:
            failures.extend(
                f"{kernel}: scalar IQ2 specialization has no FFMA.FTZ"
                for kernel in missing_ffma)
        if base_ffma == 0 or scalar_ffma < base_ffma:
            failures.append(
                f"{target}: fused slot updates missing "
                f"(base FFMA.FTZ={base_ffma}, scalar FFMA.FTZ={scalar_ffma})")
if failures: raise SystemExit("scalar structural gate failed:\n  " + "\n  ".join(failures))
print(f"validated {len(sass_rows)} SASS and {len(ptxas_rows)} PTXAS records")
PY

scalar_envs=(DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75
             DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75
             DS4_CUDA_MOE_Q4_GATE_SCALAR_CTA_SM75
             DS4_CUDA_MOE_Q4_DOWN_SCALAR_CTA_SM75
             DS4_CUDA_MOE_IQ2_SCALAR_SM75)
run_clean() {
    local -a command=(env)
    for name in "${scalar_envs[@]}"; do command+=(-u "$name"); done
    command+=(-u DS4_PROFILE_SCALAR_TARGET -u DS4_PROFILE_SCALAR
              -u DS4_PROFILE_Q4_GATE_CTA_THREADS
              -u DS4_PROFILE_Q4_DOWN_CTA_THREADS
              -u DS4_PROFILE_REPEATS CUDA_VISIBLE_DEVICES="$PROFILE_GPU")
    "${command[@]}" "$@"
}

# Keep the production path fixed for both halves of every profile A/B.  In
# particular, IQ2 tile8 is selected for both base and scalar samples; only the
# production scalar switch changes between them.
run_profile_variant() {
    local target=$1 variant=$2 repeats=$3; shift 3
    local -a command=(env)
    for name in "${scalar_envs[@]}"; do command+=(-u "$name"); done
    command+=(-u DS4_PROFILE_SCALAR_TARGET
              -u DS4_PROFILE_SCALAR
              -u DS4_PROFILE_Q4_GATE_CTA_THREADS
              -u DS4_PROFILE_Q4_DOWN_CTA_THREADS
              -u DS4_PROFILE_REPEATS
              CUDA_VISIBLE_DEVICES="$PROFILE_GPU"
              DS4_PROFILE_SCALAR_TARGET="$target"
              DS4_PROFILE_REPEATS="$repeats")
    case "$target" in
        q4-gate|q4-down|iq2-tile8|iq2-tile16) ;;
        *) die "unknown profile target: $target" ;;
    esac
    if [[ $variant == scalar ]]; then
        command+=(DS4_PROFILE_SCALAR=1)
    elif [[ $variant == base ]]; then
        command+=(DS4_PROFILE_SCALAR=0)
    else
        die "unknown profile variant: $variant"
    fi
    "${command[@]}" "$@"
}

current_phase=smoke
smoke_log="$OUTPUT_DIR/smoke/base-vs-scalar-exact.log"
run_clean "./$smoke_rel" >"$smoke_log" 2>&1 || {
    tail -n 120 "$smoke_log" >&2 || true; die "base-versus-scalar CUDA smoke failed"; }
grep -Fxq 'cuda long-context regression: OK' "$smoke_log" ||
    die "CUDA smoke omitted its success marker"
for marker in 'sm75 q4 gate tile8 scalar exact' \
              'sm75 q4 down tile16 scalar exact' \
              'sm75 iq2 moe tile16 scalar exact' \
              'sm75 iq2 moe tile8 scalar exact' \
              'sm75 iq2 moe stage6 scalar exact' \
              'sm75 iq2 moe stage4 scalar exact' \
              'sm75 iq2 moe mixed-tail tile8 scalar exact' \
              'sm75 q4 mixed-tail tile8 scalar exact'; do
    grep -Fq "$marker" "$smoke_log" || die "CUDA smoke omitted: $marker"
done

current_phase=sanitizer
if [[ $RUN_SANITIZER == 1 ]] && command -v compute-sanitizer >/dev/null 2>&1; then
    run_clean compute-sanitizer --tool memcheck --error-exitcode 3 \
        "./$smoke_rel" >"$OUTPUT_DIR/memcheck.log" 2>&1 || {
        tail -n 160 "$OUTPUT_DIR/memcheck.log" >&2 || true; die "memcheck failed"; }
    grep -Eq 'ERROR SUMMARY: 0 errors' "$OUTPUT_DIR/memcheck.log" ||
        die "memcheck did not report a clean error summary"
else
    printf 'skipped: RUN_SANITIZER=%s compute-sanitizer=%s\n' \
        "$RUN_SANITIZER" "$(command -v compute-sanitizer || true)" >"$OUTPUT_DIR/memcheck.log"
fi

current_phase=timing
samples="$OUTPUT_DIR/timing/internal-samples.csv"
printf 'target,scenario,round,sample_slot,variant,timed_repeats,timed_total_ms,timed_per_call_ms,status\n' >"$samples"
targets=(q4-gate q4-down iq2-tile16 iq2-tile8)
run_timing_sample() {
    local target=$1 scenario=$2 round=$3 slot=$4 variant=$5
    local log="$OUTPUT_DIR/timing/$target-$scenario-r$round-$slot-$variant.log"
    run_profile_variant "$target" "$variant" "$TIMING_REPEATS" \
        "./$profile_rel" "$scenario" >"$log" 2>&1 || {
        tail -n 100 "$log" >&2 || true; die "timing failed for $target/$scenario/$variant"; }
    grep -Fxq 'harness_status=ok' "$log" || die "timing harness omitted success"
    python3 - "$log" "$samples" "$target" "$scenario" "$round" "$slot" \
            "$variant" "$TIMING_REPEATS" <<'PY'
import sys
log_path, csv_path, target, scenario, round_, slot, variant, expected = sys.argv[1:]
values = {}
with open(log_path, encoding="utf-8", errors="replace") as handle:
    for line in handle:
        key, separator, value = line.strip().partition("=")
        if separator and key in {"timed_repeats", "timed_total_ms", "timed_per_call_ms"}:
            values[key] = value
if set(values) != {"timed_repeats", "timed_total_ms", "timed_per_call_ms"}:
    raise SystemExit(f"incomplete internal timing in {log_path}: {values}")
if values["timed_repeats"] != expected:
    raise SystemExit(f"unexpected repeat count in {log_path}: {values['timed_repeats']}")
if float(values["timed_total_ms"]) <= 0 or float(values["timed_per_call_ms"]) <= 0:
    raise SystemExit(f"non-positive internal timing in {log_path}: {values}")
with open(csv_path, "a", encoding="utf-8", newline="") as handle:
    handle.write(",".join((target, scenario, round_, slot, variant,
                           values["timed_repeats"], values["timed_total_ms"],
                           values["timed_per_call_ms"], "ok")) + "\n")
PY
}
for ((round=0; round<TIMING_ROUNDS; round++)); do
    if (( round % 2 == 0 )); then variants=(base scalar); else variants=(scalar base); fi
    for target in "${targets[@]}"; do
        if [[ $target == iq2-* ]]; then scenarios=(q2-early q2-late); else scenarios=(q4-early q4-late); fi
        for scenario in "${scenarios[@]}"; do
            for slot in 0 1; do
                run_timing_sample "$target" "$scenario" "$round" "$slot" "${variants[$slot]}"
            done
        done
    done
done
python3 - "$samples" "$OUTPUT_DIR/timing/internal-summary.csv" "$TIMING_ROUNDS" <<'PY'
import csv, statistics, sys
from collections import defaultdict
source, output, expected_text = sys.argv[1:]
expected = int(expected_text)
groups = defaultdict(list)
with open(source, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        groups[(row["target"], row["scenario"], row["variant"])].append(
            float(row["timed_per_call_ms"]))
rows=[]
target_scenarios = sorted({key[:2] for key in groups})
def median_absolute_deviation(values):
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)

def cv_percent(values):
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean * 100.0

for target, scenario in target_scenarios:
    base = groups[(target, scenario, "base")]
    scalar = groups[(target, scenario, "scalar")]
    if len(base) != expected or len(scalar) != expected:
        raise SystemExit(f"unbalanced samples for {target}/{scenario}: "
                         f"base={len(base)} scalar={len(scalar)} expected={expected}")
    base_median, scalar_median = statistics.median(base), statistics.median(scalar)
    rows.append({"target": target, "scenario": scenario,
                 "samples_per_variant": expected,
                 "base_median_ms": base_median,
                 "scalar_median_ms": scalar_median,
                 "base_over_scalar_speedup_x": base_median / scalar_median,
                 "scalar_over_base_ratio": scalar_median / base_median,
                 "scalar_change_pct": (scalar_median / base_median - 1.0) * 100.0,
                 "base_mad_ms": median_absolute_deviation(base),
                 "scalar_mad_ms": median_absolute_deviation(scalar),
                 "base_cv_pct": cv_percent(base),
                 "scalar_cv_pct": cv_percent(scalar),
                 "base_min_ms": min(base), "base_max_ms": max(base),
                 "scalar_min_ms": min(scalar), "scalar_max_ms": max(scalar)})
with open(output,"w",newline="",encoding="utf-8") as handle:
    writer=csv.DictWriter(handle,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
PY

current_phase=nsight
if [[ $RUN_NCU == 1 ]]; then
    command -v ncu >/dev/null 2>&1 || die "RUN_NCU=1 but ncu was not found"
    ncu_bin=$(command -v ncu); ncu_command=("$ncu_bin")
    if [[ $NCU_USE_SUDO == 1 ]]; then command -v sudo >/dev/null || die "sudo not found"; sudo -v; ncu_command=(sudo -E "$ncu_bin"); fi
    desired_metric_names=(
        gpu__time_duration.sum
        launch__registers_per_thread
        launch__shared_mem_per_block
        launch__occupancy_limit_blocks
        launch__occupancy_limit_registers
        launch__occupancy_limit_shared_mem
        launch__occupancy_limit_warps
        sm__warps_active.avg.pct_of_peak_sustained_active
        smsp__warps_eligible.avg.per_cycle_active
        sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_elapsed
        l1tex__throughput.avg.pct_of_peak_sustained_active
        l1tex__t_sector_hit_rate.pct
        lts__throughput.avg.pct_of_peak_sustained_elapsed
        lts__t_sector_hit_rate.pct
        lts__t_sectors_lookup_miss.sum
        dram__bytes.sum
        dram__bytes.sum.per_second
        smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
        smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
        smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
        dram__bytes_read.sum
        dram__bytes_write.sum
        l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
        l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
        l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum
        l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum
        l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
        l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
        lts__t_sectors_op_read.sum
        lts__t_sectors_op_write.sum
        smsp__inst_executed_pipe_ipa.sum
        smsp__inst_executed.sum
        smsp__sass_thread_inst_executed_op_integer_pred_on.sum
        smsp__sass_thread_inst_executed_op_memory_pred_on.sum
    )
    # Tool-generated gpu__/launch__ metrics can be collectable even when an
    # NCU version omits them from --query-metrics, so retain the proven Turing
    # core as required and discover only the extended counters.
    required_metric_names=(
        gpu__time_duration.sum
        launch__registers_per_thread
        launch__shared_mem_per_block
        launch__occupancy_limit_blocks
        launch__occupancy_limit_registers
        launch__occupancy_limit_shared_mem
        launch__occupancy_limit_warps
        sm__warps_active.avg.pct_of_peak_sustained_active
        sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_elapsed
        l1tex__t_sector_hit_rate.pct
        lts__t_sector_hit_rate.pct
        dram__bytes.sum.per_second
        smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
        smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
    )
    query_args=(--config-file off --devices 0 --query-metrics)
    ncu_help=$($ncu_bin --help 2>/dev/null || true)
    if grep -Fq -- '--query-metrics-mode' <<<"$ncu_help"; then
        query_args+=(--query-metrics-mode all)
    fi
    metric_query_rc=0
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        "${ncu_command[@]}" "${query_args[@]}" \
        >"$OUTPUT_DIR/ncu/available-metrics.raw.txt" \
        2>"$OUTPUT_DIR/ncu/available-metrics-query.log" || metric_query_rc=$?
    if (( metric_query_rc == 0 )); then
        grep -Eo '[[:alnum:]_]+__[[:alnum:]_.]+' \
            "$OUTPUT_DIR/ncu/available-metrics.raw.txt" | sort -u \
            >"$OUTPUT_DIR/ncu/available-metric-names.txt" || true
    else
        : >"$OUTPUT_DIR/ncu/available-metric-names.txt"
        printf 'optional metric discovery failed with exit %s\n' \
            "$metric_query_rc" >>"$OUTPUT_DIR/ncu/available-metrics-query.log"
    fi
    selected_metric_names=("${required_metric_names[@]}")
    : >"$OUTPUT_DIR/ncu/metrics-unavailable.txt"
    for metric in "${desired_metric_names[@]}"; do
        required=0
        for core_metric in "${required_metric_names[@]}"; do
            [[ $metric != "$core_metric" ]] || { required=1; break; }
        done
        (( required == 0 )) || continue
        if grep -Fxq -- "$metric" "$OUTPUT_DIR/ncu/available-metric-names.txt"; then
            selected_metric_names+=("$metric")
        else
            printf '%s\n' "$metric" >>"$OUTPUT_DIR/ncu/metrics-unavailable.txt"
        fi
    done
    printf '%s\n' "${desired_metric_names[@]}" >"$OUTPUT_DIR/ncu/metrics-desired.txt"
    printf '%s\n' "${required_metric_names[@]}" >"$OUTPUT_DIR/ncu/metrics-required.txt"
    printf '%s\n' "${selected_metric_names[@]}" >"$OUTPUT_DIR/ncu/metrics-selected.txt"
    metrics=$(IFS=,; printf '%s' "${selected_metric_names[*]}")
    validate_required_metrics() {
        python3 - "$1" "${required_metric_names[@]}" <<'PY'
import csv, math, sys
path, *metrics = sys.argv[1:]
with open(path, newline="", encoding="utf-8-sig") as handle:
    rows = [row for row in csv.DictReader(handle) if (row.get("ID") or "").strip()]
if len(rows) != 1:
    raise SystemExit(f"expected one Nsight row in {path}, found {len(rows)}")
row = rows[0]
for metric in metrics:
    if metric not in row:
        raise SystemExit(f"missing required Nsight metric {metric} in {path}")
    text = (row[metric] or "").strip().replace(",", "")
    try:
        value = float(text)
    except ValueError:
        raise SystemExit(f"non-numeric required metric {metric}={text!r} in {path}")
    if not math.isfinite(value):
        raise SystemExit(f"non-finite required metric {metric}={text!r} in {path}")
PY
    }
    for target in "${targets[@]}"; do
        if [[ $target == iq2-* ]]; then scenarios=(q2-early q2-late); else scenarios=(q4-early q4-late); fi
        case "$target" in
            q4-gate) regexes=("$Q4_GATE_BASE_REGEX" "$Q4_GATE_SCALAR_REGEX") ;;
            q4-down) regexes=("$Q4_DOWN_BASE_REGEX" "$Q4_DOWN_SCALAR_REGEX") ;;
            iq2-tile16) regexes=("$IQ2_TILE16_BASE_REGEX" "$IQ2_TILE16_SCALAR_REGEX") ;;
            iq2-tile8) regexes=("$IQ2_TILE8_BASE_REGEX" "$IQ2_TILE8_SCALAR_REGEX") ;;
        esac
        for scenario in "${scenarios[@]}"; do
            for index in 0 1; do
                variant=$([[ $index == 0 ]] && printf base || printf scalar); regex=${regexes[$index]}
                base="$OUTPUT_DIR/ncu/$target-$scenario-$variant"
                set +e
                run_profile_variant "$target" "$variant" 0 \
                    "${ncu_command[@]}" --config-file off \
                    --target-processes application-only --devices 0 --kernel-name-base function \
                    --kernel-name "regex:$regex" --launch-count 1 --replay-mode kernel \
                    --cache-control none --clock-control none --force-overwrite --export "$base" \
                    --metrics "$metrics" --disable-extra-suffixes "./$profile_rel" "$scenario" \
                    >"$base.log" 2>&1
                rc=$?; set -e
                (( rc == 0 )) || { tail -n 120 "$base.log" >&2 || true; die "NCU failed"; }
                [[ -s $base.ncu-rep ]] || die "missing NCU report: $base.ncu-rep"
                [[ $NCU_USE_SUDO == 0 ]] || sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
                "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw >"$base.csv" 2>"$base-import.log"
                python3 "$validator_rel" "$base.csv" "$regex" 0 --process cuda_sm75_profile_harness >"$base-validation.txt"
                validate_required_metrics "$base.csv"
            done
        done
    done
else
    printf 'skipped: RUN_NCU=0\n' >"$OUTPUT_DIR/ncu/README.txt"
fi

current_phase=final-validation
[[ -s $OUTPUT_DIR/sass-summary.csv && -s $OUTPUT_DIR/ptxas-summary.csv ]] || die "structural evidence is incomplete"
[[ $(wc -l <"$samples") -eq $((1 + TIMING_ROUNDS * 16)) ]] || die "unexpected timing sample count"
printf 'scalar_slot_evidence_status=ok\n' >"$OUTPUT_DIR/acceptance.txt"
printf 'SM75 production scalar-slot evidence completed: %s\n' "$OUTPUT_DIR"
