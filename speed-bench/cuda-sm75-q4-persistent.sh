#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
SKIP_BUILD=${SKIP_BUILD:-0}
BENCH_ROUNDS=${BENCH_ROUNDS:-3}
BENCH_LAUNCHES=${BENCH_LAUNCHES:-10}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-1}
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${PERSISTENT_DIR:-$repo_dir/sm75-q4-persistent-$run_stamp}
ARCHIVE_PATH=$OUTPUT_DIR.tar.gz

gate_bin=tests/cuda_sm75_q4_gate_up_native
validator=speed-bench/validate-ncu-capture.py
baseline=native-aw-warp16-scalar-consumer
candidates=(native-aw-persistent-seq16 native-aw-persistent-ws16)

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
for value in SKIP_BUILD RUN_NCU NCU_USE_SUDO; do
    [[ ${!value} == 0 || ${!value} == 1 ]] || die "$value must be 0 or 1"
done
[[ $BENCH_ROUNDS =~ ^[1-9][0-9]*$ ]] || die "BENCH_ROUNDS must be positive"
[[ $BENCH_LAUNCHES =~ ^[1-9][0-9]*$ ]] || die "BENCH_LAUNCHES must be positive"
[[ $OUTPUT_DIR == /* ]] || die "PERSISTENT_DIR must be absolute"
[[ ! -e $OUTPUT_DIR && ! -e $ARCHIVE_PATH ]] ||
    die "output already exists: $OUTPUT_DIR"
for tool in nvidia-smi cuobjdump c++filt python3 tar make; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
mkdir -p "$OUTPUT_DIR"/{build,correctness,benchmark,resources,ncu,provenance}
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
ARCHIVE_PATH=$OUTPUT_DIR.tar.gz

phase=initialization
finalize() {
    local status=$? partial=$ARCHIVE_PATH.partial.$$
    trap - EXIT INT TERM HUP
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
        "$([[ $status == 0 && $phase == complete ]] && printf complete || printf failed)" \
        "$status" "$phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
    if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
            "$(basename "$OUTPUT_DIR")"; then
        mv -- "$partial" "$ARCHIVE_PATH"
        printf 'Archive to return: %s\n' "$ARCHIVE_PATH"
    else
        rm -f -- "$partial"
        status=1
        printf 'error: archive creation failed\n' >&2
    fi
    exit "$status"
}
trap finalize EXIT
trap 'phase=interrupted-INT; exit 130' INT
trap 'phase=interrupted-TERM; exit 143' TERM
trap 'phase=interrupted-HUP; exit 129' HUP

cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $cap == 7.5 ]] ||
    die "GPU $PROFILE_GPU is compute capability ${cap:-unknown}, not 7.5"

cp -- "$0" tests/cuda_sm75_q4_gate_up_native.cu \
    tests/cuda_sm75_q4_down_native.cu \
    tests/cuda_sm75_native_q4_histograms.h Makefile "$validator" \
    "$OUTPUT_DIR/provenance/"
git diff --no-ext-diff --binary HEAD -- \
    tests/cuda_sm75_q4_gate_up_native.cu "$0" Makefile \
    >"$OUTPUT_DIR/provenance/working-tree.patch" || true
{
    printf 'date_utc=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(git rev-parse HEAD)" \
        "$(git branch --show-current)"
    printf 'profile_gpu=%s\ncompute_capability=%s\ncuda_arch=%s\n' \
        "$PROFILE_GPU" "$cap" "$CUDA_ARCH"
    printf 'scenarios=real-early,real-late\n'
    printf 'shipping_reference=%s\n' "$baseline"
    printf 'candidate_seq16=512-row-cta,sequential-gate-up,direct-native-a\n'
    printf 'candidate_ws16=512-row-cta,8-gate-plus-8-up-warps,direct-native-a\n'
    printf 'production_dispatch_modified=false\nmodel_required=false\n'
    printf '\n[gpu]\n'
    nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,name,pci.bus_id,memory.total,memory.free,driver_version \
        --format=csv
    printf '\n[git status]\n'; git status --short
} >"$OUTPUT_DIR/manifest.txt"

phase=build
if [[ $SKIP_BUILD == 0 ]]; then
    make -B -j"$(nproc)" "$gate_bin" CUDA_ARCH="$CUDA_ARCH" \
        2>&1 | tee "$OUTPUT_DIR/build/build.log"
else
    [[ -x $gate_bin ]] || die "SKIP_BUILD=1 but gate harness is missing"
    make -q "$gate_bin" CUDA_ARCH="$CUDA_ARCH" ||
        die "SKIP_BUILD=1 rejected a stale gate harness"
fi
cuobjdump --list-elf "$gate_bin" | grep -Eqi 'sm_?75' ||
    die "gate harness has no SM75 cubin"

phase=resource-validation
cuobjdump --dump-sass "$gate_bin" | c++filt \
    >"$OUTPUT_DIR/resources/gate.sass"
cuobjdump --dump-resource-usage "$gate_bin" | c++filt \
    >"$OUTPUT_DIR/resources/gate.resources"
for kernel in \
    sm75_q4_gate_up_native_aw_persistent_seq16_kernel \
    sm75_q4_gate_up_native_aw_persistent_ws16_kernel; do
    awk -v wanted="$kernel" '
        /Function : / { emit = index($0, wanted) != 0 }
        emit { print }
    ' "$OUTPUT_DIR/resources/gate.sass" \
        >"$OUTPUT_DIR/resources/$kernel.sass"
    [[ -s $OUTPUT_DIR/resources/$kernel.sass ]] || die "$kernel SASS missing"
    grep -Eq 'IMMA\.8832\.U4\.U4' \
        "$OUTPUT_DIR/resources/$kernel.sass" || die "$kernel omitted U4/U4 MMA"
    grep -Eq 'IMMA\.8832\.S4\.U4' \
        "$OUTPUT_DIR/resources/$kernel.sass" || die "$kernel omitted S4/U4 MMA"
    ! grep -Eq '(^|[[:space:]])(LDL|STL)([[:space:].]|$)' \
        "$OUTPUT_DIR/resources/$kernel.sass" || die "$kernel uses local memory"
done
python3 - "$OUTPUT_DIR/resources/gate.resources" \
        "$OUTPUT_DIR/resources/persistent-resources.csv" <<'PY'
import csv, pathlib, re, sys
source = pathlib.Path(sys.argv[1]).read_text(errors="replace").splitlines()
targets = (
    "sm75_q4_gate_up_native_aw_persistent_seq16_kernel",
    "sm75_q4_gate_up_native_aw_persistent_ws16_kernel",
)
rows = []
for target in targets:
    matches = []
    for i, line in enumerate(source):
        if "Function" not in line or target not in line:
            continue
        record = next((x for x in source[i + 1:i + 8]
                       if re.match(r"\s*REG:", x)), "")
        values = {k: int(v) for k, v in
                  re.findall(r"\b(REG|STACK|SHARED|LOCAL):(\d+)", record)}
        if set(values) == {"REG", "STACK", "SHARED", "LOCAL"}:
            matches.append(values)
    if len(matches) != 1:
        raise SystemExit(f"expected one resource record for {target}: {matches}")
    values = matches[0]
    if values["REG"] > 128:
        raise SystemExit(f"{target} exceeds 128 registers: {values}")
    if values["STACK"] or values["LOCAL"]:
        raise SystemExit(f"{target} uses local memory: {values}")
    if values["SHARED"] >= 16384:
        raise SystemExit(f"{target} exceeds 16 KiB shared: {values}")
    rows.append({"kernel": target, **values})
with pathlib.Path(sys.argv[2]).open("w", newline="") as handle:
    writer = csv.DictWriter(handle,
        fieldnames=("kernel", "REG", "STACK", "SHARED", "LOCAL"))
    writer.writeheader(); writer.writerows(rows)
print("validated persistent Gate resource ceilings", rows)
PY

phase=correctness
for scenario in real-early real-late; do
    CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$gate_bin --device 0 \
        --scenario "$scenario" --correctness-only \
        >"$OUTPUT_DIR/correctness/gate-$scenario.log" 2>&1 || {
            tail -n 120 "$OUTPUT_DIR/correctness/gate-$scenario.log" >&2
            die "persistent Gate exactness failed: $scenario"
        }
    grep -q '^correctness_status=ok$' \
        "$OUTPUT_DIR/correctness/gate-$scenario.log" ||
        die "persistent Gate exactness marker missing: $scenario"
done

phase=benchmark
for scenario in real-early real-late; do
    CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$gate_bin --device 0 \
        --scenario "$scenario" --benchmark-only \
        --rounds "$BENCH_ROUNDS" --launches "$BENCH_LAUNCHES" \
        >"$OUTPUT_DIR/benchmark/gate-$scenario.log" 2>&1
done
python3 - "$OUTPUT_DIR/benchmark" "$baseline" "${candidates[@]}" <<'PY'
import csv, pathlib, sys
root = pathlib.Path(sys.argv[1]); baseline = sys.argv[2]
candidates = sys.argv[3:]
rows = []
for scenario in ("real-early", "real-late"):
    lines = (root / f"gate-{scenario}.log").read_text().splitlines()
    begin = lines.index("scenario,variant,median_total_ms,median_us_per_launch,relative_speed")
    table = {}
    for line in lines[begin + 1:]:
        if line == "median_summary_end": break
        fields = line.split(",")
        if len(fields) == 5: table[fields[1]] = float(fields[3])
    base = table[baseline]
    for candidate in candidates:
        value = table[candidate]
        rows.append({"scenario": scenario, "baseline": baseline,
                     "candidate": candidate, "baseline_us": base,
                     "candidate_us": value, "speedup": base / value,
                     "duration_gate": "pass" if value < base else "fail"})
with (root / "persistent-comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
for row in rows: print(row)
PY

if [[ $RUN_NCU == 1 ]]; then
    phase=nsight-compute
    command -v ncu >/dev/null 2>&1 || die "ncu not found"
    ncu_bin=$(command -v ncu)
    ncu_cmd=("$ncu_bin")
    if [[ $NCU_USE_SUDO == 1 ]]; then
        command -v sudo >/dev/null 2>&1 || die "sudo not found"
        sudo -v
        ncu_cmd=(sudo -E "$ncu_bin")
    fi
    sections=(--section SpeedOfLight --section LaunchStats --section Occupancy
        --section SchedulerStats --section WarpStateStats
        --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis)
    profile_one() {
        local scenario=$1 variant=$2 kernel=$3 label=$4
        local base="$OUTPUT_DIR/ncu/$scenario-$label" rc=0
        printf 'Nsight Compute: %s %s...\n' "$scenario" "$label"
        CUDA_VISIBLE_DEVICES="$PROFILE_GPU" "${ncu_cmd[@]}" --config-file off \
            --target-processes application-only --devices 0 \
            --kernel-name-base function --kernel-name "regex:^$kernel$" \
            --launch-skip 1 --launch-count 1 --replay-mode kernel \
            --cache-control none --clock-control none --force-overwrite \
            --export "$base" "${sections[@]}" ./$gate_bin --device 0 \
            --scenario "$scenario" --profile "$variant" --launches 1 \
            >"$base.log" 2>&1 || rc=$?
        (( rc == 0 )) || { tail -n 100 "$base.log" >&2; die "NCU failed: $label"; }
        [[ -s $base.ncu-rep ]] || die "NCU report missing: $label"
        if [[ $NCU_USE_SUDO == 1 ]]; then
            sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
        fi
        "$ncu_bin" --config-file off --import "$base.ncu-rep" \
            --csv --page raw >"$base.csv" 2>"$base-import.log"
        python3 "$validator" "$base.csv" "^$kernel$" 0 \
            --process cuda_sm75_q4_gate_up_native --block-size 512 \
            >"$base-validation.txt" 2>&1 || {
                cat "$base-validation.txt" >&2
                die "wrong NCU kernel: $label"
            }
    }
    for scenario in real-early real-late; do
        profile_one "$scenario" "$baseline" \
            sm75_q4_gate_up_native_aw_warp16_scalar_kernel baseline
        profile_one "$scenario" native-aw-persistent-seq16 \
            sm75_q4_gate_up_native_aw_persistent_seq16_kernel persistent-seq16
        profile_one "$scenario" native-aw-persistent-ws16 \
            sm75_q4_gate_up_native_aw_persistent_ws16_kernel persistent-ws16
    done
fi

phase=complete
printf 'SM75 persistent tile16 Gate audit complete: %s\n' "$OUTPUT_DIR"
