#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Test the SM75 one-token Q4-32 and Q3A4 low-register gate/up split.

The experiment compares the shipping fused gate/up kernel against two
independent projection kernels plus an exact combine kernel. It requires:
  * byte-exact non-zero production-intermediate regression for both formats;
  * zero PTXAS stack/spill bytes and zero SASS LDL/STL for every split kernel;
  * no more than 128 registers/thread (the kernels request two 256-thread
    CTAs/SM with __launch_bounds__);
  * production-owned-call timing inclusive of quantization, both extra
    launches, intermediate traffic, combine, mid quantization, and down.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  TIMING_ROUNDS=7
  TIMING_REPEATS=20
  RUN_NCU=1
  NCU_USE_SUDO=0
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  Q32_LOWREG_DIR=/absolute/output/directory
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
TIMING_REPEATS=${TIMING_REPEATS:-20}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${Q32_LOWREG_DIR:-$repo_dir/sm75-decode-q32-lowreg-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $TIMING_ROUNDS =~ ^[1-9][0-9]*$ ]] || die "TIMING_ROUNDS must be positive"
[[ $TIMING_REPEATS =~ ^[1-9][0-9]*$ ]] || die "TIMING_REPEATS must be positive"
[[ $RUN_NCU == 0 || $RUN_NCU == 1 ]] || die "RUN_NCU must be 0 or 1"
[[ $NCU_USE_SUDO == 0 || $NCU_USE_SUDO == 1 ]] || die "NCU_USE_SUDO must be 0 or 1"
[[ $SKIP_BUILD == 0 || $SKIP_BUILD == 1 ]] || die "SKIP_BUILD must be 0 or 1"
[[ $CREATE_ARCHIVE == 0 || $CREATE_ARCHIVE == 1 ]] || die "CREATE_ARCHIVE must be 0 or 1"

tools=(c++filt cuobjdump env git grep make nproc nvidia-smi python3 tar)
(( RUN_NCU == 0 )) || tools+=(ncu)
for tool in "${tools[@]}"; do command -v "$tool" >/dev/null 2>&1 || die "$tool not found"; done

compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] ||
    die "physical GPU $PROFILE_GPU has compute capability ${compute_cap:-unknown}; SM75 is required"

[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/smoke" "$OUTPUT_DIR/timing" "$OUTPUT_DIR/ncu"
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
            tar -C "$(dirname "$OUTPUT_DIR")" -czf "$archive" \
                "$(basename "$OUTPUT_DIR")" || status=1
            printf 'Archive to return: %s\n' "$archive"
        fi
    fi
    exit "$status"
}
trap finalize EXIT

{
    printf 'date_utc=%s\nrepo=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
    printf 'git_commit=%s\ngit_branch=%s\n' "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'profile_gpu=%s\ncuda_arch=%s\n' "$PROFILE_GPU" "$CUDA_ARCH"
    printf 'timing_rounds=%s\ntiming_repeats=%s\nrun_ncu=%s\n' \
        "$TIMING_ROUNDS" "$TIMING_REPEATS" "$RUN_NCU"
    printf '\n[gpu]\n'
    nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,name,pci.bus_id,memory.total,memory.free,driver_version \
        --format=csv
    printf '\n[git status]\n'; git status --short
} >"$OUTPUT_DIR/manifest.txt"

smoke=tests/cuda_long_context_smoke
harness=tests/cuda_sm75_decode_weight_profile
current_phase=build
if [[ $SKIP_BUILD == 0 ]]; then
    evidence_nvccflags=${EVIDENCE_NVCCFLAGS:-"-O3 -g -lineinfo --use_fast_math -arch=$CUDA_ARCH -Xcompiler -march=native -Xcompiler -pthread -Xptxas -v"}
    set +e
    make -B -j"$(nproc)" "$smoke" "$harness" CUDA_ARCH="$CUDA_ARCH" \
        NVCCFLAGS="$evidence_nvccflags" >"$OUTPUT_DIR/build.log" 2>&1
    build_rc=$?
    set -e
    if (( build_rc != 0 )); then
        tail -n 160 "$OUTPUT_DIR/build.log" >&2 || true
        die "SM75 low-register evidence build failed"
    fi
else
    die "SKIP_BUILD=1 cannot provide fresh PTXAS evidence; use SKIP_BUILD=0"
fi
[[ -x $smoke && -x $harness ]] || die "required CUDA binaries are missing"

current_phase=exact-regression
printf 'Running byte-exact Q4-32/Q3A4 production regression...\n'
env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$smoke \
    >"$OUTPUT_DIR/smoke/cuda-long-context.log" 2>&1 || {
        tail -n 160 "$OUTPUT_DIR/smoke/cuda-long-context.log" >&2 || true
        die "byte-exact CUDA regression failed"
    }
grep -q 'SM75 Q4-32 split gate/up.*exact/reuse' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" || die "Q4-32 split exact marker missing"
grep -q 'SM75 Q3A4 split gate/up.*exact/reuse' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" || die "Q3A4 split exact marker missing"

for scenario in q4-32-gate-up-split q3a4-gate-up-split; do
    printf 'Split-path smoke: %s...\n' "$scenario"
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$harness "$scenario" \
        >"$OUTPUT_DIR/smoke/$scenario.log" 2>&1 || {
            tail -n 120 "$OUTPUT_DIR/smoke/$scenario.log" >&2 || true
            die "$scenario smoke failed"
        }
    grep -q '^q32_split=enabled$' "$OUTPUT_DIR/smoke/$scenario.log" ||
        die "$scenario omitted split dispatch"
    grep -q '^output_validation=exact-zero$' "$OUTPUT_DIR/smoke/$scenario.log" ||
        die "$scenario output validation missing"
done

current_phase=resource-audit
cuobjdump --list-elf "$harness" >"$OUTPUT_DIR/elf-list.txt" 2>&1
grep -Eq '\.sm_75\.cubin([[:space:]]|$)' "$OUTPUT_DIR/elf-list.txt" ||
    die "decode harness does not contain an sm_75 cubin"
cuobjdump --dump-sass "$harness" | c++filt >"$OUTPUT_DIR/sass.demangled.txt" 2>&1
c++filt <"$OUTPUT_DIR/build.log" >"$OUTPUT_DIR/build.demangled.log"

python3 - "$OUTPUT_DIR/sass.demangled.txt" "$OUTPUT_DIR/build.demangled.log" \
        "$OUTPUT_DIR/resource-summary.csv" <<'PY'
import csv, re, sys

sass_path, build_path, output_path = sys.argv[1:]
targets = {
    "q4-gate": r"moe_gate_up_mid_decode_sm75_q32_projection_owned_kernel<false, false>",
    "q4-up": r"moe_gate_up_mid_decode_sm75_q32_projection_owned_kernel<false, true>",
    "q3a4-gate": r"moe_gate_up_mid_decode_sm75_q32_projection_owned_kernel<true, false>",
    "q3a4-up": r"moe_gate_up_mid_decode_sm75_q32_projection_owned_kernel<true, true>",
    "combine": r"moe_gate_up_mid_decode_sm75_q32_combine_owned_kernel",
}

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
                ("stack", "spill_stores", "spill_loads"),
                map(int, match.groups())))
        match = re.search(r"Used\s+(\d+) registers", line)
        if match:
            properties[current]["registers"] = int(match.group(1))

rows = []
for label, expression in targets.items():
    pattern = re.compile(expression)
    sass_matches = [(name, body) for name, body in sections.items()
                    if pattern.search(name)]
    ptxas_matches = [(name, values) for name, values in properties.items()
                     if pattern.search(name)]
    if len(sass_matches) != 1:
        raise SystemExit(f"expected one SASS match for {label}, got {len(sass_matches)}")
    if len(ptxas_matches) != 1:
        raise SystemExit(f"expected one PTXAS match for {label}, got {len(ptxas_matches)}")
    sass_name, body = sass_matches[0]
    ptxas_name, values = ptxas_matches[0]
    missing = {"stack", "spill_stores", "spill_loads", "registers"} - values.keys()
    if missing:
        raise SystemExit(f"incomplete PTXAS evidence for {label}: {sorted(missing)}")
    text = "".join(body)
    ldl = len(re.findall(r"\bLDL(?:\.|\b)", text))
    stl = len(re.findall(r"\bSTL(?:\.|\b)", text))
    registers = values["registers"]
    allocated = ((registers + 7) // 8) * 8
    rows.append({
        "kernel": label, "registers": registers,
        "allocated_registers": allocated,
        "stack_frame_bytes": values["stack"],
        "spill_store_bytes": values["spill_stores"],
        "spill_load_bytes": values["spill_loads"],
        "sass_ldl": ldl, "sass_stl": stl,
        "two_ctas_sm_register_gate": "pass" if allocated <= 128 else "fail",
        "sass_symbol": sass_name, "ptxas_symbol": ptxas_name,
    })

with open(output_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

failures = []
for row in rows:
    if row["stack_frame_bytes"] or row["spill_store_bytes"] or row["spill_load_bytes"]:
        failures.append(
            f'{row["kernel"]}: stack={row["stack_frame_bytes"]} '
            f'spill stores={row["spill_store_bytes"]} loads={row["spill_load_bytes"]}')
    if row["sass_ldl"] or row["sass_stl"]:
        failures.append(
            f'{row["kernel"]}: SASS LDL={row["sass_ldl"]} STL={row["sass_stl"]}')
    if row["allocated_registers"] > 128:
        failures.append(
            f'{row["kernel"]}: allocated registers={row["allocated_registers"]} > 128')
if failures:
    raise SystemExit("low-register structural gate failed:\n  " + "\n  ".join(failures))
print("validated split kernels: zero spills/local traffic, <=128 registers/thread")
PY
cat "$OUTPUT_DIR/resource-summary.csv"

current_phase=inclusive-timing
for format in q4-32 q3a4; do
    scenario="$format-gate-up-ab"
    printf 'Inclusive production-call timing: %s...\n' "$format"
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        TIMING_ROUNDS="$TIMING_ROUNDS" TIMING_REPEATS="$TIMING_REPEATS" \
        ./$harness "$scenario" >"$OUTPUT_DIR/timing/$format.log" 2>&1 || {
            tail -n 120 "$OUTPUT_DIR/timing/$format.log" >&2 || true
            die "$format inclusive timing failed"
        }
    grep -q '^timing_scope=production-owned-call-inclusive$' \
        "$OUTPUT_DIR/timing/$format.log" || die "$format timing scope missing"
    grep -q '^split_speedup=' "$OUTPUT_DIR/timing/$format.log" ||
        die "$format timing result missing"
done

{
    printf 'format,control_median_ms,split_median_ms,split_speedup\n'
    for format in q4-32 q3a4; do
        log="$OUTPUT_DIR/timing/$format.log"
        control=$(grep '^control_median_ms=' "$log" | cut -d= -f2)
        split=$(grep '^split_median_ms=' "$log" | cut -d= -f2)
        speedup=$(grep '^split_speedup=' "$log" | cut -d= -f2)
        printf '%s,%s,%s,%s\n' "$format" "$control" "$split" "$speedup"
    done
} | tee "$OUTPUT_DIR/timing-summary.csv"

if [[ $RUN_NCU == 1 ]]; then
    current_phase=nsight-compute
    ncu_bin=$(command -v ncu)
    ncu_command=("$ncu_bin")
    if [[ $NCU_USE_SUDO == 1 ]]; then
        command -v sudo >/dev/null 2>&1 || die "sudo not found"
        sudo -v
        ncu_command=(sudo -E "$ncu_bin")
    fi
    metrics='gpu__time_duration.sum,launch__registers_per_thread,launch__shared_mem_per_block,launch__block_size,launch__grid_size,sm__warps_active.avg.pct_of_peak_sustained_active'
    profile_kernel() {
        local label=$1 scenario=$2 regex=$3 skip=$4 base rc=0
        base="$OUTPUT_DIR/ncu/$label"
        printf 'Nsight Compute: %s...\n' "$label"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            "${ncu_command[@]}" --config-file off --target-processes application-only \
                --devices 0 --kernel-name-base function --kernel-name "regex:$regex" \
                --launch-skip "$skip" --launch-count 1 --replay-mode kernel \
                --cache-control all --clock-control none --metrics "$metrics" \
                --disable-extra-suffixes --force-overwrite --export "$base" \
                ./$harness "$scenario" >"$base.log" 2>&1 || rc=$?
        if (( rc != 0 )) || grep -Eq '==ERROR==|No kernels were profiled' "$base.log"; then
            tail -n 120 "$base.log" >&2 || true
            die "Nsight Compute failed for $label"
        fi
        [[ -s $base.ncu-rep ]] || die "Nsight report missing for $label"
        if [[ $NCU_USE_SUDO == 1 ]]; then
            sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
        fi
        "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
            >"$base.csv" 2>"$base-import.log" || die "Nsight import failed for $label"
        python3 speed-bench/validate-ncu-capture.py "$base.csv" "$regex" 0 \
            --process cuda_sm75_decode_weight_profile --block-size 256 \
            >"$base-validation.txt" 2>&1 || {
                cat "$base-validation.txt" >&2 || true
                die "Nsight validation failed for $label"
            }
    }
    projection='moe_gate_up_mid_decode_sm75_q32_projection_owned_kernel.*'
    combine='moe_gate_up_mid_decode_sm75_q32_combine_owned_kernel.*'
    for format in q4-32 q3a4; do
        scenario="$format-gate-up-split"
        profile_kernel "$format-gate" "$scenario" "$projection" 0
        profile_kernel "$format-up" "$scenario" "$projection" 1
        profile_kernel "$format-combine" "$scenario" "$combine" 0
    done
fi

current_phase=complete
printf 'SM75 Q32 low-register split experiment complete: %s\n' "$OUTPUT_DIR"
