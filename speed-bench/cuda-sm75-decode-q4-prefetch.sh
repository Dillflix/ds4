#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Audit software-prefetch depths 1 and 2 against the current production SM75
Q4-32 decode kernels, independently for gate/up, down owned_slots, and down
owned_packed. Depth 0 is always the control. This is a bounded screen, not a
production promotion test.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  TIMING_ROUNDS=9
  TIMING_REPEATS=25
  RUN_SANITIZER=1
  RUN_NCU=1
  NCU_USE_SUDO=0
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  Q4_PREFETCH_DIR=/absolute/output/directory
USAGE
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
TIMING_ROUNDS=${TIMING_ROUNDS:-9}
TIMING_REPEATS=${TIMING_REPEATS:-25}
RUN_SANITIZER=${RUN_SANITIZER:-1}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${Q4_PREFETCH_DIR:-$repo_dir/sm75-decode-q4-prefetch-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $TIMING_ROUNDS =~ ^[1-9][0-9]*$ ]] && (( TIMING_ROUNDS % 2 == 1 )) ||
    die "TIMING_ROUNDS must be a positive odd integer"
[[ $TIMING_REPEATS =~ ^[1-9][0-9]*$ ]] || die "TIMING_REPEATS must be positive"
for value in "$RUN_SANITIZER" "$RUN_NCU" "$NCU_USE_SUDO" \
             "$SKIP_BUILD" "$CREATE_ARCHIVE"; do
    [[ $value == 0 || $value == 1 ]] || die "binary options must be 0 or 1"
done
[[ $SKIP_BUILD == 0 ]] ||
    die "SKIP_BUILD=1 cannot provide fresh specialization/resource evidence"

tools=(awk c++filt cat cuobjdump cut date dirname env git grep id make mkdir
       mv nproc nvidia-smi python3 rm tail tar tr)
(( RUN_NCU == 0 )) || tools+=(ncu)
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
mkdir -p "$OUTPUT_DIR/smoke" "$OUTPUT_DIR/timing" \
         "$OUTPUT_DIR/sanitizer" "$OUTPUT_DIR/ncu"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

current_phase=initialization
finalize() {
    local status=$?
    trap - EXIT INT TERM HUP
    if [[ -d $OUTPUT_DIR ]]; then
        printf 'state=%s\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
            "$([[ $status == 0 ]] && printf finished || printf failed)" \
            "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >"$OUTPUT_DIR/run-status.txt"
        if [[ $CREATE_ARCHIVE == 1 ]]; then
            local archive="$OUTPUT_DIR.tar.gz" partial="$OUTPUT_DIR.tar.gz.partial.$$"
            if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                    "$(basename "$OUTPUT_DIR")" && mv -f -- "$partial" "$archive"; then
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
trap 'current_phase=interrupted; exit 130' INT TERM HUP

{
    printf 'date_utc=%s\nrepo=%s\ngit_commit=%s\ngit_branch=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir" \
        "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'scope=q4-32-prefetch-depth-0-1-2\n'
    printf 'gate_control=tile32-mma\ndown_control=tile32-int4\n'
    printf 'profile_gpu=%s\ncuda_arch=%s\ntiming_rounds=%s\ntiming_repeats=%s\n' \
        "$PROFILE_GPU" "$CUDA_ARCH" "$TIMING_ROUNDS" "$TIMING_REPEATS"
    printf 'run_sanitizer=%s\nrun_ncu=%s\n\n[gpu]\n' "$RUN_SANITIZER" "$RUN_NCU"
    nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,name,pci.bus_id,memory.total,memory.free,driver_version \
        --format=csv
    printf '\n[git status]\n'; git status --short
} >"$OUTPUT_DIR/manifest.txt"

smoke=tests/cuda_long_context_smoke
harness=tests/cuda_sm75_decode_weight_profile
families=(gate slots packed)
depths=(0 1 2)

scenario_for() {
    local family=$1 depth=$2 suffix=""
    (( depth == 0 )) || suffix="-prefetch$depth"
    case "$family" in
        gate) printf 'q4-32-gate-up-tile32-mma%s\n' "$suffix" ;;
        slots) printf 'q4-32-down-slots-tile32%s\n' "$suffix" ;;
        packed) printf 'q4-32-down-packed-tile32%s\n' "$suffix" ;;
        *) return 1 ;;
    esac
}

validate_audit() {
    local family=$1 depth=$2 mode=$3 log=$4
    awk -v family="$family" -v depth="$depth" -v mode="$mode" '
        /SM75 Q4-32 decode mapping audit/ && family=="gate" {
            seen++; delete v
            for (i=1; i<=NF; i++) {split($i,a,"="); v[a[1]]=a[2]+0}
            good=(v["control"]==0 && v["hwarp16"]==0 &&
                  v["tile32-dp4a"]==0 && v["tile32-mma"]>0 &&
                  v["pf0"]+v["pf1"]+v["pf2"]==v["tile32-mma"])
            if (mode=="single") {
                good=good && v["tile32-mma"]==1
                for (d=0; d<=2; d++) good=good && v["pf" d]==(d==depth)
            } else {
                good=good && depth>0 && v["pf0"]>0 && v["pf" depth]>0
                for (d=1; d<=2; d++) if (d!=depth) good=good && v["pf" d]==0
            }
        }
        /SM75 Q4-32 down decode mapping audit/ && family!="gate" {
            seen++; delete v
            for (i=1; i<=NF; i++) {split($i,a,"="); v[a[1]]=a[2]+0}
            wanted=(family=="slots" ? v["tile32-slots"] : v["tile32-packed"])
            other=(family=="slots" ? v["tile32-packed"] : v["tile32-slots"])
            good=(v["control"]==0 && v["tile32"]==wanted && wanted>0 &&
                  other==0 && v["pf0"]+v["pf1"]+v["pf2"]==wanted)
            if (mode=="single") {
                good=good && wanted==1
                for (d=0; d<=2; d++) good=good && v["pf" d]==(d==depth)
            } else {
                good=good && depth>0 && v["pf0"]>0 && v["pf" depth]>0
                for (d=1; d<=2; d++) if (d!=depth) good=good && v["pf" d]==0
            }
        }
        END {exit !(seen==1 && good)}
    ' "$log"
}

current_phase=build
evidence_nvccflags=${EVIDENCE_NVCCFLAGS:-"-O3 -g -lineinfo --use_fast_math -arch=$CUDA_ARCH -Xcompiler -march=native -Xcompiler -pthread -Xptxas -v"}
set +e
make -B -j"$(nproc)" "$smoke" "$harness" CUDA_ARCH="$CUDA_ARCH" \
    NVCCFLAGS="$evidence_nvccflags" >"$OUTPUT_DIR/build.log" 2>&1
build_rc=$?
set -e
if (( build_rc != 0 )); then
    tail -n 200 "$OUTPUT_DIR/build.log" >&2 || true
    die "SM75 Q4-32 prefetch evidence build failed"
fi
[[ -x $smoke && -x $harness ]] || die "required CUDA binaries are missing"

current_phase=byte-exact-regression
printf 'Running nonzero byte-exact Q4-32 prefetch regression...\n'
env -u DS4_CUDA_MOE_Q4_32_DECODE_PREFETCH_DEPTH \
    -u DS4_CUDA_MOE_Q4_32_DOWN_DECODE_PREFETCH_DEPTH \
    CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$smoke \
    >"$OUTPUT_DIR/smoke/cuda-long-context.log" 2>&1 || {
        tail -n 200 "$OUTPUT_DIR/smoke/cuda-long-context.log" >&2 || true
        die "byte-exact CUDA regression failed"
    }
grep -Fq 'SM75 Q4-32 tile32-mma prefetch-depth 1/2 nonzero exact' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" || die "gate/up prefetch exact marker missing"
grep -Fq 'SM75 Q4-32 down tile32 packed-INT4 prefetch-depth 1/2 slots/packed exact' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" || die "down prefetch exact marker missing"
grep -Fxq 'cuda long-context regression: OK' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" || die "exact regression completion marker missing"

current_phase=smoke
for family in "${families[@]}"; do
    for depth in "${depths[@]}"; do
        scenario=$(scenario_for "$family" "$depth")
        log="$OUTPUT_DIR/smoke/$family-pf$depth.log"
        printf 'Production-shaped smoke: %s prefetch-depth %s...\n' "$family" "$depth"
        audit_env=DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING_AUDIT
        [[ $family != gate ]] || audit_env=DS4_CUDA_MOE_Q4_32_DECODE_MAPPING_AUDIT
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" "$audit_env=1" \
            ./$harness "$scenario" >"$log" 2>&1 || {
                tail -n 120 "$log" >&2 || true
                die "$family depth $depth smoke failed"
            }
        selector=q4_32_decode_prefetch_depth
        [[ $family == gate ]] || selector=q4_32_down_decode_prefetch_depth
        grep -Fxq "$selector=$depth" "$log" ||
            die "$family depth $depth selector evidence missing"
        grep -Fxq 'output_validation=exact-zero' "$log" ||
            die "$family depth $depth output validation missing"
        grep -Fxq 'harness_status=ok' "$log" || die "$family depth $depth harness marker missing"
        validate_audit "$family" "$depth" single "$log" ||
            die "$family depth $depth dispatch audit failed"
    done
done

if [[ $RUN_SANITIZER == 1 ]]; then
    current_phase=compute-sanitizer
    for family in "${families[@]}"; do
        scenario=$(scenario_for "$family" 2)
        log="$OUTPUT_DIR/sanitizer/$family-pf2.log"
        printf 'Compute Sanitizer: %s prefetch-depth 2...\n' "$family"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" compute-sanitizer \
            --tool memcheck --error-exitcode 97 ./$harness "$scenario" \
            >"$log" 2>&1 || {
                tail -n 160 "$log" >&2 || true
                die "$family depth-2 sanitizer failed"
            }
        grep -Fq 'ERROR SUMMARY: 0 errors' "$log" ||
            die "$family depth-2 sanitizer success marker missing"
    done
fi

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
for family, stem, shared in (
    ("gate", r"moe_gate_up_mid_decode_sm75_q4_32_tile32_owned_kernel<true,", 4096),
    ("slots", r"moe_down_sm75_q4_32_tile32_owned_slots_kernel<", 1024),
    ("packed", r"moe_down_sm75_q4_32_tile32_owned_packed_kernel<", 1152),
):
    for depth in range(3):
        targets[f"{family}-pf{depth}"] = (
            stem + rf"\s*{depth}u?>", family, depth, shared)

def sections(path, marker, sass):
    result, current = {}, None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = re.search(marker, line)
            if match:
                current = match.group(1)
                result.setdefault(current, [] if sass else {})
                continue
            if current is None:
                continue
            if sass:
                result[current].append(line)
            else:
                match = re.search(
                    r"(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*"
                    r"(\d+) bytes spill loads", line)
                if match:
                    result[current].update(zip(
                        ("stack", "spill_stores", "spill_loads"),
                        map(int, match.groups())))
                match = re.search(r"Used\s+(\d+) registers", line)
                if match:
                    result[current]["registers"] = int(match.group(1))
                match = re.search(
                    r"(\d+) bytes (?:smem(?:\[\d+\])?|shared memory)", line)
                if match:
                    result[current]["shared"] = int(match.group(1))
    return result

sass = sections(sass_path, r"Function\s*:\s*(.*\S)", True)
ptxas = sections(build_path, r"Function properties for\s+(.*\S)", False)
rows = []
for label, (expression, family, depth, expected_shared) in targets.items():
    pattern = re.compile(expression)
    sm = [(name, body) for name, body in sass.items() if pattern.search(name)]
    pm = [(name, values) for name, values in ptxas.items() if pattern.search(name)]
    if len(sm) != 1 or len(pm) != 1:
        raise SystemExit(f"kernel identity failure for {label}: SASS={len(sm)} PTXAS={len(pm)}")
    sass_name, body = sm[0]
    ptxas_name, values = pm[0]
    missing = {"stack", "spill_stores", "spill_loads", "registers", "shared"} - values.keys()
    if missing:
        raise SystemExit(f"incomplete PTXAS evidence for {label}: {sorted(missing)}")
    text = "".join(body)
    ldl = len(re.findall(r"\bLDL(?:\.|\b)", text))
    stl = len(re.findall(r"\bSTL(?:\.|\b)", text))
    atom = len(re.findall(r"\b(?:ATOM|RED)(?:\.|\b)", text))
    imma = len(re.findall(r"\bIMMA[^\n]*8832", text, re.I))
    u4s4 = len(re.findall(r"\bIMMA[^\n]*8832[^\n]*U4[^\n]*S4", text, re.I))
    s4s4 = len(re.findall(r"\bIMMA[^\n]*8832[^\n]*S4[^\n]*S4", text, re.I))
    idp4a = len(re.findall(r"\bIDP\.4A(?:\.|\b)", text))
    allocated = ((values["registers"] + 7) // 8) * 8
    if any((values["stack"], values["spill_stores"], values["spill_loads"], ldl, stl, atom)):
        raise SystemExit(f"{label}: stack/spill/local/atomic traffic is nonzero")
    if allocated > 128 or 128 * allocated * 2 > 65536:
        raise SystemExit(f"{label}: {allocated} allocated registers fail the two-CTA/SM gate")
    if values["shared"] != expected_shared:
        raise SystemExit(f"{label}: shared memory {values['shared']} != {expected_shared}")
    if imma == 0 or u4s4 == 0 or s4s4 == 0 or idp4a != 0:
        raise SystemExit(
            f"{label}: packed-INT4 identity failed: IMMA={imma} U4.S4={u4s4} "
            f"S4.S4={s4s4} IDP.4A={idp4a}")
    rows.append({
        "variant": label, "family": family, "prefetch_depth": depth,
        "block_size": 128, "registers": values["registers"],
        "allocated_registers": allocated, "shared_memory_bytes": values["shared"],
        "stack_frame_bytes": values["stack"], "spill_store_bytes": values["spill_stores"],
        "spill_load_bytes": values["spill_loads"], "sass_ldl": ldl, "sass_stl": stl,
        "sass_atom_red": atom, "sass_imma_8832": imma,
        "sass_imma_u4_s4": u4s4, "sass_imma_s4_s4": s4s4,
        "sass_idp4a": idp4a, "two_ctas_sm_register_gate": "pass",
        "resource_gate": "pass", "sass_symbol": sass_name, "ptxas_symbol": ptxas_name,
    })
with open(output_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
print("validated nine Q4-32 prefetch specialization identities and hard resource gates")
PY
cat "$OUTPUT_DIR/resource-summary.csv"

current_phase=inclusive-timing
printf 'family,prefetch_depth,control_median_ms,candidate_median_ms,candidate_speedup\n' \
    >"$OUTPUT_DIR/timing-summary.csv"
for family in "${families[@]}"; do
    for depth in 1 2; do
        scenario=$(scenario_for "$family" "$depth")-ab
        log="$OUTPUT_DIR/timing/$family-pf$depth.log"
        audit_env=DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING_AUDIT
        [[ $family != gate ]] || audit_env=DS4_CUDA_MOE_Q4_32_DECODE_MAPPING_AUDIT
        printf 'Inclusive production-owned-call timing: %s prefetch-depth %s...\n' "$family" "$depth"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" "$audit_env=1" \
            TIMING_ROUNDS="$TIMING_ROUNDS" TIMING_REPEATS="$TIMING_REPEATS" \
            ./$harness "$scenario" >"$log" 2>&1 || {
                tail -n 140 "$log" >&2 || true
                die "$family depth $depth timing failed"
            }
        grep -Fxq 'timing_scope=production-owned-call-inclusive' "$log" ||
            die "$family depth $depth timing scope missing"
        validate_audit "$family" "$depth" paired "$log" ||
            die "$family depth $depth paired dispatch audit failed"
        control=$(grep '^control_median_ms=' "$log" | cut -d= -f2)
        candidate=$(grep '^candidate_median_ms=' "$log" | cut -d= -f2)
        speedup=$(grep '^candidate_speedup=' "$log" | cut -d= -f2)
        [[ -n $control && -n $candidate && -n $speedup ]] ||
            die "$family depth $depth timing summary is incomplete"
        printf '%s,%s,%s,%s,%s\n' "$family" "$depth" "$control" "$candidate" "$speedup" \
            >>"$OUTPUT_DIR/timing-summary.csv"
    done
done
cat "$OUTPUT_DIR/timing-summary.csv"

if [[ $RUN_NCU == 1 ]]; then
    current_phase=nsight-compute
    ncu_bin=$(command -v ncu)
    ncu_command=("$ncu_bin")
    if [[ $NCU_USE_SUDO == 1 ]]; then
        command -v sudo >/dev/null 2>&1 || die "sudo not found"
        sudo -v
        ncu_command=(sudo -E "$ncu_bin")
    fi
    metrics=(gpu__time_duration.sum dram__bytes.sum dram__bytes.sum.per_second
        smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
        smsp__warps_eligible.avg.per_cycle_active launch__block_size
        launch__registers_per_thread launch__shared_mem_per_block)
    metric_csv=$(IFS=,; printf '%s' "${metrics[*]}")
    printf '%s\n' "${metrics[@]}" >"$OUTPUT_DIR/ncu/metrics-required.txt"
    for family in "${families[@]}"; do
        for depth in "${depths[@]}"; do
            scenario=$(scenario_for "$family" "$depth")
            case "$family" in
                gate)
                    stem=moe_gate_up_mid_decode_sm75_q4_32_tile32_owned_kernel
                    regex="$stem.*<true, ${depth}u?>.*"
                    ;;
                slots)
                    stem=moe_down_sm75_q4_32_tile32_owned_slots_kernel
                    regex="$stem.*<${depth}u?>.*"
                    ;;
                packed)
                    stem=moe_down_sm75_q4_32_tile32_owned_packed_kernel
                    regex="$stem.*<${depth}u?>.*"
                    ;;
            esac
            base="$OUTPUT_DIR/ncu/$family-pf$depth"
            printf 'Nsight Compute: %s prefetch-depth %s...\n' "$family" "$depth"
            rc=0
            env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" "${ncu_command[@]}" \
                --config-file off --target-processes application-only --devices 0 \
                --kernel-name-base function --kernel-name "regex:$regex" \
                --launch-count 1 --replay-mode kernel --cache-control all \
                --clock-control none --metrics "$metric_csv" \
                --disable-extra-suffixes --force-overwrite --export "$base" \
                ./$harness "$scenario" >"$base.log" 2>&1 || rc=$?
            if (( rc != 0 )) || grep -Eq \
                    '==ERROR==|No kernels were profiled|Failed to (profile|create report)' "$base.log"; then
                tail -n 120 "$base.log" >&2 || true
                die "Nsight Compute failed for $family depth $depth"
            fi
            [[ -s $base.ncu-rep ]] || die "Nsight report missing for $family depth $depth"
            if [[ $NCU_USE_SUDO == 1 ]]; then
                sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
            fi
            "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
                >"$base.csv" 2>"$base-import.log" ||
                die "Nsight import failed for $family depth $depth"
            python3 speed-bench/validate-ncu-capture.py "$base.csv" "$regex" 0 \
                --process cuda_sm75_decode_weight_profile --block-size 128 \
                >"$base-validation.txt" 2>&1 || {
                    cat "$base-validation.txt" >&2 || true
                    die "Nsight validation failed for $family depth $depth"
                }
            cat "$base-validation.txt"
        done
    done

    python3 - "$OUTPUT_DIR/ncu" "$OUTPUT_DIR/ncu-summary.csv" <<'PY'
import csv, math, pathlib, sys
ncu_dir, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
required = [line.strip() for line in
            (ncu_dir / "metrics-required.txt").read_text().splitlines()
            if line.strip()]
out = []
for family in ("gate", "slots", "packed"):
    for depth in range(3):
        variant = f"{family}-pf{depth}"
        rows = list(csv.DictReader(open(
            ncu_dir / f"{variant}.csv", newline="", encoding="utf-8-sig")))
        data = [row for row in rows if (row.get("ID") or "").strip()]
        units = [row for row in rows if not (row.get("ID") or "").strip()]
        if len(data) != 1:
            raise SystemExit(f"{variant}: expected one NCU row, got {len(data)}")
        unit = units[-1] if units else {}
        for metric in required:
            value = (data[0].get(metric) or "").strip()
            if not value or value.lower() in {"n/a", "not available"}:
                raise SystemExit(f"{variant}: metric {metric} is unavailable")
            try:
                number = float(value.replace(",", ""))
            except ValueError:
                raise SystemExit(f"{variant}: non-numeric {metric}: {value!r}")
            if not math.isfinite(number):
                raise SystemExit(f"{variant}: non-finite {metric}: {value!r}")
            if metric == "gpu__time_duration.sum" and number <= 0.0:
                raise SystemExit(f"{variant}: duration is not positive")
            out.append({"family": family, "prefetch_depth": depth,
                        "metric": metric, "unit": (unit.get(metric) or "").strip(),
                        "value": value})
with open(output, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=out[0].keys())
    writer.writeheader(); writer.writerows(out)
PY
    cat "$OUTPUT_DIR/ncu-summary.csv"
fi

current_phase=complete
printf 'SM75 Q4-32 prefetch-depth audit complete: %s\n' "$OUTPUT_DIR"
