#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Compare bounded SM75-native Q4-32 one-token gate/up mappings.

The unchanged one-row/warp control is compared with:
  hwarp16      two independent rows per warp;
  tile32-dp4a  one warp follows one native eight-row tile, signed DP4A;
  tile32-mma   the same tile mapping using packed m8n8k32 INT4 MMA.

Every candidate is audit-only/default-off.  The run requires a deterministic
nonzero, two-input, production-K exact test through both the mid boundary and
Q4-32 down output.  It also rejects spills/local traffic, ATOM/RED, more than
128 allocated registers, and a failure of the two-CTA/SM register gate.
Packed MMA must contain both U4.S4 and S4.S4 IMMA.8832 opcodes; DP4A must
contain IDP.4A.  Timing includes the complete owned routed call.

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
  Q4_GATE_NATIVE_DIR=/absolute/output/directory
EOF
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
RUN_NCU=${RUN_NCU:-1}
RUN_SANITIZER=${RUN_SANITIZER:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${Q4_GATE_NATIVE_DIR:-$repo_dir/sm75-decode-q4-gate-native-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $TIMING_ROUNDS =~ ^[1-9][0-9]*$ ]] || die "TIMING_ROUNDS must be positive"
(( TIMING_ROUNDS % 2 == 1 )) || die "TIMING_ROUNDS must be odd"
[[ $TIMING_REPEATS =~ ^[1-9][0-9]*$ ]] || die "TIMING_REPEATS must be positive"
for value in "$RUN_NCU" "$RUN_SANITIZER" "$NCU_USE_SUDO" "$SKIP_BUILD" "$CREATE_ARCHIVE"; do
    [[ $value == 0 || $value == 1 ]] || die "binary options must be 0 or 1"
done

# This audit owns the Q4 gate/up mapping.  A caller's stale down-candidate
# environment must not change the inclusive control/candidate comparison.
unset DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING
unset DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING_AUDIT

tools=(c++filt cuobjdump env git grep make nproc nvidia-smi python3 tar)
(( RUN_NCU == 0 )) || tools+=(ncu)
(( RUN_SANITIZER == 0 )) || tools+=(compute-sanitizer)
for tool in "${tools[@]}"; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
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
    printf 'scope=single-gpu-q4-32-audit-only-decode-mapping\n'
    printf 'production_default=control\n'
    printf 'timing_rounds=%s\ntiming_repeats=%s\nrun_sanitizer=%s\nrun_ncu=%s\n' \
        "$TIMING_ROUNDS" "$TIMING_REPEATS" "$RUN_SANITIZER" "$RUN_NCU"
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
        die "SM75 Q4-32 native gate/up evidence build failed"
    fi
else
    die "SKIP_BUILD=1 cannot provide fresh PTXAS evidence; use SKIP_BUILD=0"
fi
[[ -x $smoke && -x $harness ]] || die "required CUDA binaries are missing"

current_phase=byte-exact-regression
printf 'Running nonzero byte-exact Q4-32 16-record production regression...\n'
env -u DS4_CUDA_MOE_Q4_32_DECODE_MAPPING \
    -u DS4_CUDA_MOE_Q4_32_DECODE_MAPPING_AUDIT \
    CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$smoke \
    >"$OUTPUT_DIR/smoke/cuda-long-context.log" 2>&1 || {
        tail -n 180 "$OUTPUT_DIR/smoke/cuda-long-context.log" >&2 || true
        die "byte-exact CUDA regression failed"
    }
grep -q 'SM75 Q4-32 hwarp16/tile32-dp4a/tile32-mma gate/up.*nonzero exact/reuse' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" ||
    die "Q4-32 native nonzero exact marker missing"
grep -q 'SM75 Q4-32 audit mapping selector exact/default-off' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" ||
    die "Q4-32 default-off selector marker missing"
grep -q 'SM75 Q4-32 signed DP4A byte packing exact' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" ||
    die "Q4-32 signed DP4A packing marker missing"
grep -q 'SM75 Q4-32 signed-zero gate probe exact' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" ||
    die "Q4-32 signed-zero exactness marker missing"

declare -A scenario=(
    [control]=q4-32-gate-up
    [hwarp16]=q4-32-gate-up-hwarp16
    [tile32-dp4a]=q4-32-gate-up-tile32-dp4a
    [tile32-mma]=q4-32-gate-up-tile32-mma
)
for variant in control hwarp16 tile32-dp4a tile32-mma; do
    name=${scenario[$variant]}
    printf 'Production-shaped resource smoke: %s...\n' "$variant"
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$harness "$name" \
        >"$OUTPUT_DIR/smoke/$variant.log" 2>&1 || {
            tail -n 120 "$OUTPUT_DIR/smoke/$variant.log" >&2 || true
            die "$variant smoke failed"
        }
    grep -q '^output_validation=exact-zero$' "$OUTPUT_DIR/smoke/$variant.log" ||
        die "$variant synthetic output validation missing"
    grep -q '^harness_status=ok$' "$OUTPUT_DIR/smoke/$variant.log" ||
        die "$variant harness success marker missing"
done
grep -q '^q4_32_decode_mapping=0$' "$OUTPUT_DIR/smoke/control.log" ||
    die "control omitted its dispatch"
grep -q '^q4_32_decode_mapping=1$' "$OUTPUT_DIR/smoke/hwarp16.log" ||
    die "hwarp16 omitted its dispatch"
grep -q '^q4_32_decode_mapping=2$' "$OUTPUT_DIR/smoke/tile32-dp4a.log" ||
    die "tile32-dp4a omitted its dispatch"
grep -q '^q4_32_decode_mapping=3$' "$OUTPUT_DIR/smoke/tile32-mma.log" ||
    die "tile32-mma omitted its dispatch"

if [[ $RUN_SANITIZER == 1 ]]; then
    current_phase=compute-sanitizer
    for variant in hwarp16 tile32-dp4a tile32-mma; do
        log="$OUTPUT_DIR/sanitizer/$variant.log"
        printf 'Compute Sanitizer: %s...\n' "$variant"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            compute-sanitizer --tool memcheck --error-exitcode=99 \
                --leak-check full ./$harness "${scenario[$variant]}" \
                >"$log" 2>&1 || {
                    tail -n 160 "$log" >&2 || true
                    die "Compute Sanitizer failed for $variant"
                }
        grep -q 'ERROR SUMMARY: 0 errors' "$log" || {
            tail -n 160 "$log" >&2 || true
            die "Compute Sanitizer did not report a clean $variant run"
        }
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
targets = {
    "control": (r"moe_gate_up_mid_decode_sm75_q32_owned_kernel<false>", False, 256, 0),
    "hwarp16": (r"moe_gate_up_mid_decode_sm75_q4_32_hwarp16_owned_kernel", True, 256, 0),
    "tile32-dp4a": (r"moe_gate_up_mid_decode_sm75_q4_32_tile32_owned_kernel<false>", True, 128, 4096),
    "tile32-mma": (r"moe_gate_up_mid_decode_sm75_q4_32_tile32_owned_kernel<true>", True, 128, 4096),
}

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
                match = re.search(r"(\d+) bytes smem", line)
                if match:
                    result[current]["static_shared"] = int(match.group(1))
    return result

sass = sections(sass_path, r"Function\s*:\s*(.*\S)", True)
ptxas = sections(build_path, r"Function properties for\s+(.*\S)", False)
rows = []
for label, (expression, candidate, block_size, expected_shared) in targets.items():
    pattern = re.compile(expression)
    sm = [(name, body) for name, body in sass.items() if pattern.search(name)]
    pm = [(name, values) for name, values in ptxas.items() if pattern.search(name)]
    if len(sm) != 1 or len(pm) != 1:
        raise SystemExit(
            f"kernel identity failure for {label}: SASS={len(sm)} PTXAS={len(pm)}")
    sass_name, body = sm[0]
    ptxas_name, values = pm[0]
    values.setdefault("static_shared", 0)
    missing = {"stack", "spill_stores", "spill_loads", "registers"} - values.keys()
    if missing:
        raise SystemExit(f"incomplete PTXAS evidence for {label}: {sorted(missing)}")
    text = "".join(body)
    ldl = len(re.findall(r"\bLDL(?:\.|\b)", text))
    stl = len(re.findall(r"\bSTL(?:\.|\b)", text))
    atom_red = len(re.findall(r"\b(?:ATOM|RED)(?:\.|\b)", text))
    idp4a = len(re.findall(r"\bIDP\.4A(?:\.|\b)", text))
    imma = len(re.findall(r"\bIMMA(?:\.|\b)", text))
    imma_8832 = len(re.findall(r"\bIMMA[^\n]*8832", text, re.I))
    imma_u4s4 = len(re.findall(
        r"\bIMMA[^\n]*8832[^\n]*U4[^\n]*S4", text, re.I))
    imma_s4s4 = len(re.findall(
        r"\bIMMA[^\n]*8832[^\n]*S4[^\n]*S4", text, re.I))
    registers = values["registers"]
    allocated = ((registers + 7) // 8) * 8
    register_two_cta = block_size * allocated * 2 <= 65536
    shared_two_cta = values["static_shared"] * 2 <= 65536
    if candidate and any((values["stack"], values["spill_stores"],
                          values["spill_loads"], ldl, stl, atom_red)):
        raise SystemExit(
            f"{label}: stack/spill/local/atomic traffic: stack={values['stack']} "
            f"spill_stores={values['spill_stores']} spill_loads={values['spill_loads']} "
            f"LDL={ldl} STL={stl} ATOM/RED={atom_red}")
    if candidate and allocated > 128:
        raise SystemExit(f"{label}: {allocated} allocated registers exceed 128")
    if candidate and not register_two_cta:
        raise SystemExit(f"{label}: register allocation cannot sustain two CTAs/SM")
    if candidate and not shared_two_cta:
        raise SystemExit(f"{label}: shared allocation cannot sustain two CTAs/SM")
    if values["static_shared"] != expected_shared:
        raise SystemExit(
            f"{label}: static shared memory {values['static_shared']} != "
            f"expected {expected_shared}")
    if label == "tile32-dp4a" and (idp4a == 0 or imma != 0):
        raise SystemExit(
            f"tile32-dp4a arithmetic identity failed: IDP.4A={idp4a} IMMA={imma}")
    if label == "tile32-mma":
        if idp4a != 0:
            raise SystemExit(f"tile32-mma unexpectedly contains IDP.4A={idp4a}")
        if imma == 0 or imma_8832 == 0 or imma_u4s4 == 0 or imma_s4s4 == 0:
            raise SystemExit(
                "tile32-mma must contain IMMA.8832 U4.S4 and S4.S4: "
                f"IMMA={imma} K32={imma_8832} U4.S4={imma_u4s4} "
                f"S4.S4={imma_s4s4}")
    rows.append({
        "variant": label, "candidate": "yes" if candidate else "no",
        "block_size": block_size, "registers": registers,
        "allocated_registers": allocated,
        "static_shared_memory_bytes": values["static_shared"],
        "expected_static_shared_memory_bytes": expected_shared,
        "stack_frame_bytes": values["stack"],
        "spill_store_bytes": values["spill_stores"],
        "spill_load_bytes": values["spill_loads"],
        "sass_ldl": ldl, "sass_stl": stl, "sass_atom_red": atom_red,
        "sass_idp4a": idp4a, "sass_imma": imma,
        "sass_imma_8832": imma_8832,
        "sass_imma_u4_s4": imma_u4s4,
        "sass_imma_s4_s4": imma_s4s4,
        "two_ctas_sm_register_gate": "pass" if register_two_cta else "fail",
        "two_ctas_sm_shared_gate": "pass" if shared_two_cta else "fail",
        "resource_gate": "pass" if not candidate or (
            allocated <= 128 and register_two_cta and shared_two_cta) else "fail",
        "sass_symbol": sass_name, "ptxas_symbol": ptxas_name,
    })

with open(output_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
print("validated four Q4-32 kernel identities, arithmetic identities, and resource gates")
PY
cat "$OUTPUT_DIR/resource-summary.csv"

timing_csv="$OUTPUT_DIR/timing-summary.csv"
printf 'variant,control_median_ms,candidate_median_ms,candidate_speedup\n' >"$timing_csv"
declare -A timing_scenario=(
    [hwarp16]=q4-32-gate-up-hwarp16-ab
    [tile32-dp4a]=q4-32-gate-up-tile32-dp4a-ab
    [tile32-mma]=q4-32-gate-up-tile32-mma-ab
)
declare -A timing_kind=(
    [hwarp16]=q4-32-hwarp16
    [tile32-dp4a]=q4-32-tile32-dp4a
    [tile32-mma]=q4-32-tile32-mma
)
current_phase=inclusive-timing
for variant in hwarp16 tile32-dp4a tile32-mma; do
    log="$OUTPUT_DIR/timing/$variant.log"
    printf 'Inclusive production-owned-call timing: %s...\n' "$variant"
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        TIMING_ROUNDS="$TIMING_ROUNDS" TIMING_REPEATS="$TIMING_REPEATS" \
        ./$harness "${timing_scenario[$variant]}" >"$log" 2>&1 || {
            tail -n 120 "$log" >&2 || true; die "$variant timing failed";
        }
    grep -q '^timing_scope=production-owned-call-inclusive$' "$log" ||
        die "$variant timing scope missing"
    grep -Fqx "candidate_kind=${timing_kind[$variant]}" "$log" ||
        die "$variant timing compared the wrong candidate"
    control=$(grep '^control_median_ms=' "$log" | cut -d= -f2)
    candidate=$(grep '^candidate_median_ms=' "$log" | cut -d= -f2)
    speedup=$(grep '^candidate_speedup=' "$log" | cut -d= -f2)
    printf '%s,%s,%s,%s\n' "$variant" "$control" "$candidate" "$speedup" \
        >>"$timing_csv"
done
cat "$timing_csv"

python3 - "$timing_csv" "$OUTPUT_DIR/decision-summary.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
rows.sort(key=lambda r: float(r["candidate_speedup"]), reverse=True)
with open(sys.argv[2], "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
print("Q4-32 measured ranking: " + ",".join(r["variant"] for r in rows))
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
    # These metrics are already known to collect on the target Turing setup.
    # Nsight's query output can omit generated gpu__/launch__ names even when
    # direct --metrics collection succeeds, so discovery only adds optional
    # metrics; it never vetoes this core set.
    required_metrics=(
        gpu__time_duration.sum
        dram__bytes.sum.per_second
        lts__t_sector_hit_rate.pct
        smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
        smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
        sm__warps_active.avg.pct_of_peak_sustained_active
        launch__block_size
        launch__registers_per_thread
        launch__shared_mem_per_block
        launch__occupancy_limit_blocks
        launch__occupancy_limit_registers
        launch__occupancy_limit_shared_mem
        launch__occupancy_limit_warps
    )
    optional_metrics=(
        dram__bytes.avg.pct_of_peak_sustained_elapsed
        dram__bytes_read.sum
        dram__bytes_write.sum
        smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
        l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
        smsp__warps_eligible.avg.per_cycle_active
        smsp__sass_thread_inst_executed_op_integer_pred_on.sum
        smsp__sass_thread_inst_executed_op_memory_pred_on.sum
        launch__grid_size
        launch__waves_per_multiprocessor
    )
    available_raw="$OUTPUT_DIR/ncu/available-metrics.raw.txt"
    available_names="$OUTPUT_DIR/ncu/available-metric-names.txt"
    query_args=(--config-file off --devices 0 --query-metrics)
    ncu_help=$("$ncu_bin" --help 2>/dev/null || true)
    grep -Fq -- '--query-metrics-mode' <<<"$ncu_help" &&
        query_args+=(--query-metrics-mode all)
    metric_query_rc=0
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        "${ncu_command[@]}" "${query_args[@]}" \
        >"$available_raw" 2>"$OUTPUT_DIR/ncu/available-metrics-query.log" ||
        metric_query_rc=$?
    if (( metric_query_rc == 0 )); then
        grep -Eo '[[:alnum:]_]+__[[:alnum:]_.]+' "$available_raw" | sort -u \
            >"$available_names" || true
    else
        : >"$available_names"
        printf 'optional metric discovery failed with exit %s; collecting core metrics only\n' \
            "$metric_query_rc" >>"$OUTPUT_DIR/ncu/available-metrics-query.log"
    fi
    metrics=("${required_metrics[@]}")
    : >"$OUTPUT_DIR/ncu/metrics-unavailable.txt"
    for metric in "${optional_metrics[@]}"; do
        if grep -Fxq -- "$metric" "$available_names"; then
            metrics+=("$metric")
        else
            printf '%s\n' "$metric" >>"$OUTPUT_DIR/ncu/metrics-unavailable.txt"
        fi
    done
    printf '%s\n' "${required_metrics[@]}" \
        >"$OUTPUT_DIR/ncu/metrics-required.txt"
    printf '%s\n' "${metrics[@]}" >"$OUTPUT_DIR/ncu/metrics-selected.txt"
    metric_csv=$(IFS=,; printf '%s' "${metrics[*]}")
    profile_kernel() {
        local label=$1 regex block base rc=0
        case "$label" in
            control)
                regex='moe_gate_up_mid_decode_sm75_q32_owned_kernel.*'; block=256 ;;
            hwarp16)
                regex='moe_gate_up_mid_decode_sm75_q4_32_hwarp16_owned_kernel.*'; block=256 ;;
            tile32-dp4a|tile32-mma)
                regex='moe_gate_up_mid_decode_sm75_q4_32_tile32_owned_kernel.*'; block=128 ;;
        esac
        base="$OUTPUT_DIR/ncu/$label"
        printf 'Nsight Compute: %s...\n' "$label"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            "${ncu_command[@]}" --config-file off \
                --target-processes application-only --devices 0 \
                --kernel-name-base function --kernel-name "regex:$regex" \
                --launch-count 1 --replay-mode kernel --cache-control all \
                --clock-control none --metrics "$metric_csv" \
                --disable-extra-suffixes --force-overwrite --export "$base" \
                ./$harness "${scenario[$label]}" >"$base.log" 2>&1 || rc=$?
        if (( rc != 0 )) ||
           grep -Eq '==ERROR==|No kernels were profiled|Failed to (profile|create report)' "$base.log"; then
            tail -n 120 "$base.log" >&2 || true
            die "Nsight Compute failed for $label"
        fi
        [[ -s $base.ncu-rep ]] || die "Nsight report missing for $label"
        if [[ $NCU_USE_SUDO == 1 ]]; then
            sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
        fi
        "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
            >"$base.csv" 2>"$base-import.log" ||
            die "Nsight import failed for $label"
        python3 speed-bench/validate-ncu-capture.py "$base.csv" "$regex" 0 \
            --process cuda_sm75_decode_weight_profile --block-size "$block" \
            >"$base-validation.txt" 2>&1 || {
                cat "$base-validation.txt" >&2 || true
                die "Nsight validation failed for $label"
            }
        cat "$base-validation.txt"
    }
    for variant in control hwarp16 tile32-dp4a tile32-mma; do
        profile_kernel "$variant"
    done
fi

current_phase=complete
printf 'SM75 Q4-32 native gate/up decode experiment complete: %s\n' "$OUTPUT_DIR"
