#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Sweep single-launch, fused low-register Q3A4 and Q4-32 decode gate/up.

Q3A4 is evaluated first at partial-unroll factors 1, 2, and 4.  Its measured
ranking determines the order of the Q4-32 pass, but Q4-32 still evaluates all
three variants so a format-specific result is not silently discarded.

Every candidate must pass byte-exact non-zero regression.  PTXAS/SASS resource
evidence records spills, local traffic, and the two-CTA/SM register gate.
Inclusive timing covers the complete production-owned call.  Nsight Compute
captures only the control and best structurally eligible candidate per format.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  TIMING_ROUNDS=7
  TIMING_REPEATS=20
  RUN_NCU=1
  NCU_USE_SUDO=0
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  Q32_FUSED_LOWREG_DIR=/absolute/output/directory
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
OUTPUT_DIR=${Q32_FUSED_LOWREG_DIR:-$repo_dir/sm75-decode-q32-fused-lowreg-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $TIMING_ROUNDS =~ ^[1-9][0-9]*$ ]] || die "TIMING_ROUNDS must be positive"
[[ $TIMING_REPEATS =~ ^[1-9][0-9]*$ ]] || die "TIMING_REPEATS must be positive"
for value in "$RUN_NCU" "$NCU_USE_SUDO" "$SKIP_BUILD" "$CREATE_ARCHIVE"; do
    [[ $value == 0 || $value == 1 ]] || die "binary options must be 0 or 1"
done

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
        tail -n 180 "$OUTPUT_DIR/build.log" >&2 || true
        die "SM75 fused-lowreg evidence build failed"
    fi
else
    die "SKIP_BUILD=1 cannot provide fresh PTXAS evidence; use SKIP_BUILD=0"
fi
[[ -x $smoke && -x $harness ]] || die "required CUDA binaries are missing"

current_phase=exact-regression
printf 'Running byte-exact Q3A4/Q4-32 fused-lowreg regression...\n'
env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$smoke \
    >"$OUTPUT_DIR/smoke/cuda-long-context.log" 2>&1 || {
        tail -n 180 "$OUTPUT_DIR/smoke/cuda-long-context.log" >&2 || true
        die "byte-exact CUDA regression failed"
    }
grep -q 'SM75 Q3A4 fused-lowreg u1/u2/u4' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" || die "Q3A4 exact marker missing"
grep -q 'SM75 Q4-32 fused-lowreg u1/u2/u4' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" || die "Q4-32 exact marker missing"

for format in q3a4 q4-32; do
    for unroll in 1 2 4; do
        scenario="$format-gate-up-fused-u$unroll"
        printf 'Fused-path smoke: %s...\n' "$scenario"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$harness "$scenario" \
            >"$OUTPUT_DIR/smoke/$scenario.log" 2>&1 || {
                tail -n 120 "$OUTPUT_DIR/smoke/$scenario.log" >&2 || true
                die "$scenario smoke failed"
            }
        grep -q "^q32_fused_lowreg_unroll=$unroll$" \
            "$OUTPUT_DIR/smoke/$scenario.log" || die "$scenario omitted candidate dispatch"
        grep -q '^output_validation=exact-zero$' "$OUTPUT_DIR/smoke/$scenario.log" ||
            die "$scenario output validation missing"
    done
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
targets = {}
for fmt, flag in (("q3a4", "true"), ("q4-32", "false")):
    targets[f"{fmt}-control"] = (
        rf"moe_gate_up_mid_decode_sm75_q32_owned_kernel<{flag}>", False)
    for unroll in (1, 2, 4):
        targets[f"{fmt}-u{unroll}"] = (
            rf"moe_gate_up_mid_decode_sm75_q32_fused_lowreg_owned_kernel<"
            rf"{flag},\s*(?:{unroll}u|{unroll})>", True)

def sections(path, marker):
    result, current = {}, None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = re.search(marker, line)
            if match:
                current = match.group(1)
                result.setdefault(current, [] if "Function\\s" in marker else {})
                continue
            if current is not None and isinstance(result[current], list):
                result[current].append(line)
            elif current is not None:
                match = re.search(
                    r"(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*"
                    r"(\d+) bytes spill loads", line)
                if match:
                    result[current].update(zip(
                        ("stack", "spill_stores", "spill_loads"), map(int, match.groups())))
                match = re.search(r"Used\s+(\d+) registers", line)
                if match:
                    result[current]["registers"] = int(match.group(1))
    return result

sass = sections(sass_path, r"Function\s*:\s*(.*\S)")
ptxas = sections(build_path, r"Function properties for\s+(.*\S)")
rows = []
for label, (expression, candidate) in targets.items():
    pattern = re.compile(expression)
    sm = [(name, body) for name, body in sass.items() if pattern.search(name)]
    pm = [(name, values) for name, values in ptxas.items() if pattern.search(name)]
    if len(sm) != 1 or len(pm) != 1:
        raise SystemExit(
            f"kernel identity failure for {label}: SASS={len(sm)} PTXAS={len(pm)}")
    sass_name, body = sm[0]
    ptxas_name, values = pm[0]
    missing = {"stack", "spill_stores", "spill_loads", "registers"} - values.keys()
    if missing:
        raise SystemExit(f"incomplete PTXAS evidence for {label}: {sorted(missing)}")
    text = "".join(body)
    ldl = len(re.findall(r"\bLDL(?:\.|\b)", text))
    stl = len(re.findall(r"\bSTL(?:\.|\b)", text))
    registers = values["registers"]
    allocated = ((registers + 7) // 8) * 8
    eligible = (not candidate or (
        values["stack"] == values["spill_stores"] == values["spill_loads"] == 0
        and ldl == stl == 0 and allocated <= 128))
    rows.append({
        "kernel": label, "candidate": "yes" if candidate else "no",
        "registers": registers, "allocated_registers": allocated,
        "stack_frame_bytes": values["stack"],
        "spill_store_bytes": values["spill_stores"],
        "spill_load_bytes": values["spill_loads"],
        "sass_ldl": ldl, "sass_stl": stl,
        "two_ctas_sm_register_gate": "pass" if allocated <= 128 else "fail",
        "structural_eligible": "yes" if eligible else "no",
        "sass_symbol": sass_name, "ptxas_symbol": ptxas_name,
    })

with open(output_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
print("recorded control plus six fused-lowreg resource variants")
PY
cat "$OUTPUT_DIR/resource-summary.csv"

timing_csv="$OUTPUT_DIR/timing-summary.csv"
printf 'format,unroll,control_median_ms,candidate_median_ms,candidate_speedup\n' >"$timing_csv"
time_candidate() {
    local format=$1 unroll=$2 scenario="$1-gate-up-fused-u$2-ab" log
    log="$OUTPUT_DIR/timing/$format-u$unroll.log"
    printf 'Inclusive timing: %s u%s...\n' "$format" "$unroll"
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        TIMING_ROUNDS="$TIMING_ROUNDS" TIMING_REPEATS="$TIMING_REPEATS" \
        ./$harness "$scenario" >"$log" 2>&1 || {
            tail -n 120 "$log" >&2 || true; die "$format u$unroll timing failed";
        }
    grep -q '^timing_scope=production-owned-call-inclusive$' "$log" ||
        die "$format u$unroll timing scope missing"
    local control candidate speedup
    control=$(grep '^control_median_ms=' "$log" | cut -d= -f2)
    candidate=$(grep '^candidate_median_ms=' "$log" | cut -d= -f2)
    speedup=$(grep '^candidate_speedup=' "$log" | cut -d= -f2)
    printf '%s,%s,%s,%s,%s\n' "$format" "$unroll" \
        "$control" "$candidate" "$speedup" >>"$timing_csv"
}

current_phase=q3a4-inclusive-timing
for unroll in 1 2 4; do time_candidate q3a4 "$unroll"; done

python3 - "$timing_csv" "$OUTPUT_DIR/resource-summary.csv" \
        "$OUTPUT_DIR/q3a4-ranking.csv" <<'PY'
import csv, sys
timing_path, resource_path, output_path = sys.argv[1:]
resources = {row["kernel"]: row for row in csv.DictReader(open(resource_path))}
rows = [row for row in csv.DictReader(open(timing_path)) if row["format"] == "q3a4"]
for row in rows:
    row["structural_eligible"] = resources[f'q3a4-u{row["unroll"]}']["structural_eligible"]
rows.sort(key=lambda r: (r["structural_eligible"] == "yes", float(r["candidate_speedup"])), reverse=True)
with open(output_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
print("Q3A4 measured ranking: " + ",".join("u" + row["unroll"] for row in rows))
PY
cat "$OUTPUT_DIR/q3a4-ranking.csv"

mapfile -t q4_order < <(python3 - "$OUTPUT_DIR/q3a4-ranking.csv" <<'PY'
import csv, sys
for row in csv.DictReader(open(sys.argv[1])):
    print(row["unroll"])
PY
)
[[ ${#q4_order[@]} -eq 3 ]] || die "Q3A4 ranking did not produce three Q4 variants"
current_phase=q4-32-inclusive-timing
for unroll in "${q4_order[@]}"; do time_candidate q4-32 "$unroll"; done
cat "$timing_csv"

python3 - "$timing_csv" "$OUTPUT_DIR/resource-summary.csv" \
        "$OUTPUT_DIR/decision-summary.csv" <<'PY'
import csv, sys
timing_path, resource_path, output_path = sys.argv[1:]
resources = {row["kernel"]: row for row in csv.DictReader(open(resource_path))}
timings = list(csv.DictReader(open(timing_path)))
out = []
for fmt in ("q3a4", "q4-32"):
    rows = [row for row in timings if row["format"] == fmt]
    for row in rows:
        row["eligible"] = resources[f'{fmt}-u{row["unroll"]}']["structural_eligible"]
    eligible = [row for row in rows if row["eligible"] == "yes"]
    best = max(eligible, key=lambda r: float(r["candidate_speedup"])) if eligible else None
    speedup = float(best["candidate_speedup"]) if best else 0.0
    out.append({
        "format": fmt,
        "best_eligible_unroll": best["unroll"] if best else "none",
        "best_speedup": f"{speedup:.9g}",
        "result": "win" if speedup > 1.0 else ("regression" if best else "no-eligible-candidate"),
    })
with open(output_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=out[0].keys())
    writer.writeheader(); writer.writerows(out)
PY
cat "$OUTPUT_DIR/decision-summary.csv"

if [[ $RUN_NCU == 1 ]]; then
    current_phase=nsight-compute
    ncu_bin=$(command -v ncu)
    ncu_command=("$ncu_bin")
    if [[ $NCU_USE_SUDO == 1 ]]; then
        command -v sudo >/dev/null 2>&1 || die "sudo not found"
        sudo -v
        ncu_command=(sudo -E "$ncu_bin")
    fi
    metrics='gpu__time_duration.sum,launch__registers_per_thread,launch__shared_mem_per_block,launch__block_size,launch__grid_size,launch__waves_per_multiprocessor,sm__warps_active.avg.pct_of_peak_sustained_active'
    profile_kernel() {
        local label=$1 scenario=$2 regex=$3 base rc=0
        base="$OUTPUT_DIR/ncu/$label"
        printf 'Nsight Compute: %s...\n' "$label"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            "${ncu_command[@]}" --config-file off --target-processes application-only \
                --devices 0 --kernel-name-base function --kernel-name "regex:$regex" \
                --launch-count 1 --replay-mode kernel --cache-control all \
                --clock-control none --metrics "$metrics" --disable-extra-suffixes \
                --force-overwrite --export "$base" ./$harness "$scenario" \
                >"$base.log" 2>&1 || rc=$?
        if (( rc != 0 )) || grep -Eq '==ERROR==|No kernels were profiled' "$base.log"; then
            tail -n 120 "$base.log" >&2 || true; die "Nsight Compute failed for $label";
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
                cat "$base-validation.txt" >&2 || true; die "Nsight validation failed for $label";
            }
    }
    while IFS=, read -r format best speedup result; do
        [[ $format != format ]] || continue
        [[ $best != none ]] || continue
        profile_kernel "$format-control" "$format-gate-up" \
            'moe_gate_up_mid_decode_sm75_q32_owned_kernel.*'
        profile_kernel "$format-fused-u$best" "$format-gate-up-fused-u$best" \
            'moe_gate_up_mid_decode_sm75_q32_fused_lowreg_owned_kernel.*'
    done <"$OUTPUT_DIR/decision-summary.csv"
fi

current_phase=complete
printf 'SM75 Q32 fused-lowreg sweep complete: %s\n' "$OUTPUT_DIR"
