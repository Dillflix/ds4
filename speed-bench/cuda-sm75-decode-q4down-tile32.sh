#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Audit the SM75 Q4-32 one-token down tile32 packed-INT4 candidates.

The production owned_slots and owned_packed quarter-warp kernels are retained
as controls. The candidates use m8n8k32 packed-INT4 MMA across one native
eight-row Q4-32 tile and retain all eight K256 float leaves so the existing
4/2/1 reduction is byte-exact. The packed fixture includes a real two-slot
prefix-pair exact add.

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
  Q4DOWN_TILE32_DIR=/absolute/output/directory
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
RUN_SANITIZER=${RUN_SANITIZER:-1}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${Q4DOWN_TILE32_DIR:-$repo_dir/sm75-decode-q4down-tile32-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $TIMING_ROUNDS =~ ^[1-9][0-9]*$ ]] || die "TIMING_ROUNDS must be positive"
(( TIMING_ROUNDS % 2 == 1 )) || die "TIMING_ROUNDS must be odd"
[[ $TIMING_REPEATS =~ ^[1-9][0-9]*$ ]] || die "TIMING_REPEATS must be positive"
for value in "$RUN_SANITIZER" "$RUN_NCU" "$NCU_USE_SUDO" "$SKIP_BUILD" "$CREATE_ARCHIVE"; do
    [[ $value == 0 || $value == 1 ]] || die "binary options must be 0 or 1"
done

# This audit owns the Q4 down mapping.  A caller's stale gate/up-candidate
# environment must not change either ownership-mode comparison.
unset DS4_CUDA_MOE_Q4_32_DECODE_MAPPING
unset DS4_CUDA_MOE_Q4_32_DECODE_MAPPING_AUDIT

tools=(c++filt cuobjdump env git grep make nproc nvidia-smi python3 tar)
(( RUN_SANITIZER == 0 )) || tools+=(compute-sanitizer)
(( RUN_NCU == 0 )) || tools+=(ncu)
for tool in "${tools[@]}"; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
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
    printf 'scope=audit-only-q4-32-down-decode-tile32-packed-int4\n'
    printf 'production_default=control\nmidq_blocks=8\n'
    printf 'candidate_block_size=128\ncandidate_static_shared_bytes=1024\n'
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
        die "SM75 Q4-32 down tile32 evidence build failed"
    fi
else
    die "SKIP_BUILD=1 cannot provide fresh PTXAS evidence; use SKIP_BUILD=0"
fi
[[ -x $smoke && -x $harness ]] || die "required CUDA binaries are missing"

current_phase=byte-exact-regression
printf 'Running nonzero byte-exact Q4-32 down slots/packed regression...\n'
env -u DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING \
    CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$smoke \
    >"$OUTPUT_DIR/smoke/cuda-long-context.log" 2>&1 || {
        tail -n 200 "$OUTPUT_DIR/smoke/cuda-long-context.log" >&2 || true
        die "byte-exact CUDA regression failed"
    }
grep -Fq 'Q4-32 down tile32 packed-INT4 owned_slots nonzero poison-overwrite exact' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" ||
    die "owned_slots nonzero exact marker missing"
grep -Fq 'Q4-32 down tile32 packed-INT4 owned_packed masks 000..111 poison-overwrite exact' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" ||
    die "owned_packed mask sweep exact marker missing"
grep -Fq 'Q4-32 down tile32 packed-INT4 mask111 prefix-pair second-cycle exact' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" ||
    die "owned_packed mask111 second-cycle marker missing"
grep -Fq 'Q4-32 down tile32 signed-zero boundary exact' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" ||
    die "signed-zero boundary marker missing"

declare -A scenario=(
    [slots-control]=q4-32-down-slots
    [slots-tile32]=q4-32-down-slots-tile32
    [packed-control]=q4-32-down-packed
    [packed-tile32]=q4-32-down-packed-tile32
)
for variant in slots-control slots-tile32 packed-control packed-tile32; do
    printf 'Production-shaped smoke: %s...\n' "$variant"
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$harness "${scenario[$variant]}" \
        >"$OUTPUT_DIR/smoke/$variant.log" 2>&1 || {
            tail -n 120 "$OUTPUT_DIR/smoke/$variant.log" >&2 || true
            die "$variant smoke failed"
        }
    grep -q '^midq_blocks=8$' "$OUTPUT_DIR/smoke/$variant.log" ||
        die "$variant did not exercise eight down records"
    grep -q '^output_validation=exact-zero$' "$OUTPUT_DIR/smoke/$variant.log" ||
        die "$variant synthetic output validation missing"
    grep -q '^harness_status=ok$' "$OUTPUT_DIR/smoke/$variant.log" ||
        die "$variant harness success marker missing"
done
for variant in slots-tile32 packed-tile32; do
    grep -q '^q4_32_down_decode_mapping=1$' "$OUTPUT_DIR/smoke/$variant.log" ||
        die "$variant omitted tile32 dispatch"
    grep -q 'mapping=tile32-int4 (audit candidate)' \
        "$OUTPUT_DIR/smoke/$variant.log" || die "$variant audit marker missing"
done
grep -q '^down_output_kind=owned_packed-prefix-pair$' \
    "$OUTPUT_DIR/smoke/packed-tile32.log" ||
    die "packed smoke omitted prefix-pair fixture"

if [[ $RUN_SANITIZER == 1 ]]; then
    current_phase=compute-sanitizer
    for variant in slots-tile32 packed-tile32; do
        printf 'Compute Sanitizer: %s...\n' "$variant"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" compute-sanitizer \
            --tool memcheck --error-exitcode 97 \
            ./$harness "${scenario[$variant]}" \
            >"$OUTPUT_DIR/smoke/$variant-memcheck.log" 2>&1 || {
                tail -n 160 "$OUTPUT_DIR/smoke/$variant-memcheck.log" >&2 || true
                die "$variant Compute Sanitizer failed"
            }
        grep -Fq 'ERROR SUMMARY: 0 errors' \
            "$OUTPUT_DIR/smoke/$variant-memcheck.log" ||
            die "$variant sanitizer success marker missing"
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
    "slots-control": (r"moe_down_sm75_q4_32_owned_slots_kernel", False, 256),
    "packed-control": (r"moe_down_sm75_q4_32_owned_packed_kernel", False, 256),
    "slots-tile32": (r"moe_down_sm75_q4_32_tile32_owned_slots_kernel", True, 128),
    "packed-tile32": (r"moe_down_sm75_q4_32_tile32_owned_packed_kernel", True, 128),
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
                match = re.search(
                    r"(\d+) bytes (?:smem(?:\[\d+\])?|shared memory)", line)
                if match:
                    result[current]["shared_memory"] = int(match.group(1))
    return result

sass = sections(sass_path, r"Function\s*:\s*(.*\S)", True)
ptxas = sections(build_path, r"Function properties for\s+(.*\S)", False)
rows = []
for label, (expression, candidate, block_size) in targets.items():
    pattern = re.compile(expression)
    sm = [(name, body) for name, body in sass.items() if pattern.search(name)]
    pm = [(name, values) for name, values in ptxas.items() if pattern.search(name)]
    if len(sm) != 1 or len(pm) != 1:
        raise SystemExit(
            f"kernel identity failure for {label}: SASS={len(sm)} PTXAS={len(pm)}")
    sass_name, body = sm[0]
    ptxas_name, values = pm[0]
    missing = {"stack", "spill_stores", "spill_loads", "registers"} \
        - values.keys()
    if missing:
        raise SystemExit(f"incomplete PTXAS evidence for {label}: {sorted(missing)}")
    text = "".join(body)
    ldl = len(re.findall(r"\bLDL(?:\.|\b)", text))
    stl = len(re.findall(r"\bSTL(?:\.|\b)", text))
    atom = len(re.findall(r"\bATOM(?:\.|\b)", text))
    red = len(re.findall(r"\bRED(?:\.|\b)", text))
    imma32 = len(re.findall(r"IMMA[^\s]*8832|IMMA\.8832", text))
    u4s4 = len(re.findall(r"IMMA[^\s]*8832[^\s]*U4[^\s]*S4", text, re.I))
    s4s4 = len(re.findall(r"IMMA[^\s]*8832[^\s]*S4[^\s]*S4", text, re.I))
    idp4a = len(re.findall(r"\bIDP\.4A(?:\.|\b)", text))
    allocated = ((values["registers"] + 7) // 8) * 8
    if candidate:
        if any((values["stack"], values["spill_stores"],
                values["spill_loads"], ldl, stl)):
            raise SystemExit(f"{label}: stack/spill/local traffic is nonzero")
        if allocated > 64:
            raise SystemExit(
                f"{label}: {allocated} allocated registers exceed the 64-register gate")
        if values.get("shared_memory", 0) != 1024:
            raise SystemExit(
                f"{label}: expected exactly 1024 static shared bytes, got "
                f"{values.get('shared_memory', 0)}")
        if imma32 == 0 or u4s4 == 0 or s4s4 == 0:
            raise SystemExit(
                f"{label}: missing m8n8k32 U4.S4/S4.S4 packed-INT4 MMA "
                f"(IMMA32={imma32} U4.S4={u4s4} S4.S4={s4s4})")
        if atom or red:
            raise SystemExit(
                f"{label}: unexpected atomic/reduction SASS ATOM={atom} RED={red}")
        if idp4a:
            raise SystemExit(f"{label}: unexpectedly uses DP4A instead of native INT4 MMA")
    rows.append({
        "variant": label, "candidate": "yes" if candidate else "no",
        "block_size": block_size, "registers": values["registers"],
        "allocated_registers": allocated,
        # PTXAS commonly omits a zero-smem field for the scalar controls.
        "static_shared_bytes": values.get("shared_memory", 0),
        "stack_frame_bytes": values["stack"],
        "spill_store_bytes": values["spill_stores"],
        "spill_load_bytes": values["spill_loads"],
        "sass_ldl": ldl, "sass_stl": stl,
        "sass_imma_8832": imma32, "sass_imma_u4_s4": u4s4,
        "sass_imma_s4_s4": s4s4, "sass_idp4a": idp4a,
        "sass_atom": atom, "sass_red": red,
        "resource_gate": "pass" if candidate else "control",
        "sass_symbol": sass_name, "ptxas_symbol": ptxas_name,
    })
with open(output_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
print("validated Q4-32 down controls and packed-INT4 tile32 resource gates")
PY
cat "$OUTPUT_DIR/resource-summary.csv"

current_phase=inclusive-timing
printf 'output_kind,control_median_ms,candidate_median_ms,candidate_speedup\n' \
    >"$OUTPUT_DIR/timing-summary.csv"
declare -A timing_scenario=(
    [slots]=q4-32-down-slots-tile32-ab
    [packed]=q4-32-down-packed-tile32-ab
)
for kind in slots packed; do
    log="$OUTPUT_DIR/timing/$kind.log"
    printf 'Inclusive production-owned-call timing: %s...\n' "$kind"
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" TIMING_ROUNDS="$TIMING_ROUNDS" \
        DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING_AUDIT=1 \
        TIMING_REPEATS="$TIMING_REPEATS" \
        ./$harness "${timing_scenario[$kind]}" >"$log" 2>&1 || {
            tail -n 140 "$log" >&2 || true
            die "$kind timing failed"
        }
    grep -q '^timing_scope=production-owned-call-inclusive$' "$log" ||
        die "$kind timing scope missing"
    grep -q "^candidate_kind=q4-32-down-tile32-int4-$kind$" "$log" ||
        die "$kind timing compared the wrong candidate"
    expected_control=$((1 + TIMING_ROUNDS * TIMING_REPEATS))
    expected_candidate=$((2 + TIMING_ROUNDS * TIMING_REPEATS))
    grep -Fq "SM75 Q4-32 down decode mapping audit control=$expected_control tile32=$expected_candidate" \
        "$log" || die "$kind timing dispatch counters are incomplete"
    grep -q 'mapping=tile32-int4 (audit candidate)' "$log" ||
        die "$kind timing omitted tile32 dispatch marker"
    control=$(grep '^control_median_ms=' "$log" | cut -d= -f2)
    candidate=$(grep '^candidate_median_ms=' "$log" | cut -d= -f2)
    speedup=$(grep '^candidate_speedup=' "$log" | cut -d= -f2)
    printf '%s,%s,%s,%s\n' "$kind" "$control" "$candidate" "$speedup" \
        >>"$OUTPUT_DIR/timing-summary.csv"
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
    # These generated and suffixed metric names can be absent from
    # --query-metrics even though direct collection succeeds. Request the
    # known SM75 core set directly; discovery is only for optional metrics.
    required_metrics=(
        gpu__time_duration.sum
        dram__bytes.sum.per_second
        lts__t_sector_hit_rate.pct
        smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
        l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
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
        printf 'optional metric discovery failed with exit %s; collecting required metrics only\n' \
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
        local variant=$1 regex block base rc=0
        case "$variant" in
            slots-control)
                regex='moe_down_sm75_q4_32_owned_slots_kernel.*'; block=256 ;;
            packed-control)
                regex='moe_down_sm75_q4_32_owned_packed_kernel.*'; block=256 ;;
            slots-tile32)
                regex='moe_down_sm75_q4_32_tile32_owned_slots_kernel.*'; block=128 ;;
            packed-tile32)
                regex='moe_down_sm75_q4_32_tile32_owned_packed_kernel.*'; block=128 ;;
        esac
        base="$OUTPUT_DIR/ncu/$variant"
        printf 'Nsight Compute: %s...\n' "$variant"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            "${ncu_command[@]}" --config-file off \
                --target-processes application-only --devices 0 \
                --kernel-name-base function --kernel-name "regex:$regex" \
                --launch-count 1 --replay-mode kernel --cache-control all \
                --clock-control none --metrics "$metric_csv" \
                --disable-extra-suffixes --force-overwrite --export "$base" \
                ./$harness "${scenario[$variant]}" >"$base.log" 2>&1 || rc=$?
        if (( rc != 0 )) ||
           grep -Eq '==ERROR==|No kernels were profiled|Failed to (profile|create report)' "$base.log"; then
            tail -n 120 "$base.log" >&2 || true
            die "Nsight Compute failed for $variant"
        fi
        [[ -s $base.ncu-rep ]] || die "Nsight report missing for $variant"
        if [[ $NCU_USE_SUDO == 1 ]]; then
            sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
        fi
        "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
            >"$base.csv" 2>"$base-import.log" ||
            die "Nsight import failed for $variant"
        python3 speed-bench/validate-ncu-capture.py "$base.csv" "$regex" 0 \
            --process cuda_sm75_decode_weight_profile --block-size "$block" \
            >"$base-validation.txt" 2>&1 || {
                cat "$base-validation.txt" >&2 || true
                die "Nsight validation failed for $variant"
            }
        cat "$base-validation.txt"
    }
    for variant in slots-control slots-tile32 packed-control packed-tile32; do
        profile_kernel "$variant"
    done
fi

current_phase=complete
printf 'SM75 Q4-32 down tile32 audit complete: %s\n' "$OUTPUT_DIR"
