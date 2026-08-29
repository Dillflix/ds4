#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Audit exact in-CTA K2/K4 parallelism for the production SM75 Q3A4 decode
tile32-DP4A kernel. K1 remains the production control and Q4-32 is unchanged.

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
  Q3A4_KSPLIT_DIR=/absolute/output/directory
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
OUTPUT_DIR=${Q3A4_KSPLIT_DIR:-$repo_dir/sm75-decode-q3a4-ksplit-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $TIMING_ROUNDS =~ ^[1-9][0-9]*$ ]] && (( TIMING_ROUNDS % 2 == 1 )) ||
    die "TIMING_ROUNDS must be a positive odd integer"
[[ $TIMING_REPEATS =~ ^[1-9][0-9]*$ ]] || die "TIMING_REPEATS must be positive"
for value in "$RUN_SANITIZER" "$RUN_NCU" "$NCU_USE_SUDO" \
             "$SKIP_BUILD" "$CREATE_ARCHIVE"; do
    [[ $value == 0 || $value == 1 ]] || die "binary options must be 0 or 1"
done

tools=(c++filt cuobjdump env git grep make nproc nvidia-smi python3 tar)
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
            local partial_archive="$archive.partial"
            rm -f -- "$partial_archive"
            if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial_archive" \
                    "$(basename "$OUTPUT_DIR")" &&
               mv -f -- "$partial_archive" "$archive"; then
                printf 'Archive to return: %s\n' "$archive"
            else
                status=1
                rm -f -- "$partial_archive"
                printf 'state=finished\nexit_status=1\nlast_phase=%s\ndate_utc=%s\n' \
                    "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                    >"$OUTPUT_DIR/run-status.txt"
            fi
        fi
    fi
    exit "$status"
}
trap finalize EXIT

{
    printf 'date_utc=%s\nrepo=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
    printf 'git_commit=%s\ngit_branch=%s\n' "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'scope=q3a4-only-exact-in-cta-k1-k2-k4\n'
    printf 'q4_32=unchanged\nproduction_default=k1\n'
    printf 'profile_gpu=%s\ncuda_arch=%s\n' "$PROFILE_GPU" "$CUDA_ARCH"
    printf 'timing_rounds=%s\ntiming_repeats=%s\n' "$TIMING_ROUNDS" "$TIMING_REPEATS"
    printf 'run_sanitizer=%s\nrun_ncu=%s\n' "$RUN_SANITIZER" "$RUN_NCU"
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
        die "SM75 Q3A4 K-split evidence build failed"
    fi
else
    die "SKIP_BUILD=1 cannot provide fresh PTXAS evidence; use SKIP_BUILD=0"
fi
[[ -x $smoke && -x $harness ]] || die "required CUDA binaries are missing"

current_phase=byte-exact-regression
printf 'Running nonzero byte-exact Q3A4 K1/K2/K4 regression...\n'
env -u DS4_CUDA_MOE_Q32_DECODE_FUSED_LOWREG \
    -u DS4_CUDA_NO_MOE_Q32_DECODE_FUSED_LOWREG \
    -u DS4_CUDA_MOE_Q3A4_DECODE_MAPPING \
    -u DS4_CUDA_NO_MOE_Q3A4_DECODE_MAPPING \
    -u DS4_CUDA_MOE_Q3A4_DECODE_KSPLIT \
    CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$smoke \
    >"$OUTPUT_DIR/smoke/cuda-long-context.log" 2>&1 || {
        tail -n 180 "$OUTPUT_DIR/smoke/cuda-long-context.log" >&2 || true
        die "byte-exact CUDA regression failed"
    }
grep -q 'SM75 Q3A4 tile32-dp4a K1/K2/K4 in-CTA gate/up and owned decode exact/reuse' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" ||
    die "Q3A4 K-split exact marker missing"
grep -q 'SM75 Q3A4 tile32-dp4a production default' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" ||
    die "K1 production-default assertion missing"
grep -q 'SM75 Q3A4 K1/K2/K4 environment selector exact' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" ||
    die "Q3A4 K-split environment-selector regression marker missing"

declare -A scenario=(
    [k1]=q3a4-gate-up-tile32-dp4a
    [k2]=q3a4-gate-up-tile32-dp4a-k2
    [k4]=q3a4-gate-up-tile32-dp4a-k4
)
for variant in k1 k2 k4; do
    printf 'Synthetic owned-call smoke: %s...\n' "$variant"
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        DS4_CUDA_MOE_Q3A4_DECODE_MAPPING_AUDIT=1 \
        ./$harness "${scenario[$variant]}" \
        >"$OUTPUT_DIR/smoke/$variant.log" 2>&1 || {
            tail -n 120 "$OUTPUT_DIR/smoke/$variant.log" >&2 || true
            die "$variant smoke failed"
        }
    grep -q '^output_validation=exact-zero$' "$OUTPUT_DIR/smoke/$variant.log" ||
        die "$variant output validation missing"
    grep -q '^q3a4_decode_mapping=3$' "$OUTPUT_DIR/smoke/$variant.log" ||
        die "$variant omitted tile32-DP4A dispatch"
    expected=${variant#k}
    grep -q "^q3a4_decode_ksplit=$expected$" "$OUTPUT_DIR/smoke/$variant.log" ||
        die "$variant K-split marker missing"
    case "$variant" in
        k1) dispatch_marker='mapping=tile32-dp4a (production default)' ;;
        k2) dispatch_marker='mapping=tile32-dp4a-k2' ;;
        k4) dispatch_marker='mapping=tile32-dp4a-k4' ;;
    esac
    grep -Fq -- "$dispatch_marker" "$OUTPUT_DIR/smoke/$variant.log" ||
        die "$variant requested state did not reach the CUDA dispatch"
    case "$variant" in
        k1) audit_pattern='tile32-dp4a=1 k1=1 k2=0 k4=0' ;;
        k2) audit_pattern='tile32-dp4a=1 k1=0 k2=1 k4=0' ;;
        k4) audit_pattern='tile32-dp4a=1 k1=0 k2=0 k4=1' ;;
    esac
    grep -Fq -- "$audit_pattern" "$OUTPUT_DIR/smoke/$variant.log" ||
        die "$variant aggregate dispatch audit is impure"
    grep -q '^harness_status=ok$' "$OUTPUT_DIR/smoke/$variant.log" ||
        die "$variant harness success marker missing"
done

if [[ $RUN_SANITIZER == 1 ]]; then
    current_phase=memcheck
    printf 'Compute Sanitizer: K4 largest-CTA smoke...\n'
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        compute-sanitizer --tool memcheck --error-exitcode 3 \
        ./$harness "${scenario[k4]}" >"$OUTPUT_DIR/memcheck.log" 2>&1 || {
            tail -n 160 "$OUTPUT_DIR/memcheck.log" >&2 || true
            die "K4 compute-sanitizer memcheck failed"
        }
    grep -q 'ERROR SUMMARY: 0 errors' "$OUTPUT_DIR/memcheck.log" ||
        die "compute-sanitizer did not report a clean error summary"
else
    printf 'skipped explicitly: RUN_SANITIZER=0\n' >"$OUTPUT_DIR/memcheck.log"
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
    "k1": (r"moe_gate_up_mid_decode_sm75_q3a4_tile32_owned_kernel<true>", 128, 128),
    "k2": (r"moe_gate_up_mid_decode_sm75_q3a4_tile32_ksplit_owned_kernel<[^>]*2[^>]*>", 256, 64),
    "k4": (r"moe_gate_up_mid_decode_sm75_q3a4_tile32_ksplit_owned_kernel<[^>]*4[^>]*>", 512, 64),
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
for label, (expression, block_size, register_limit) in targets.items():
    pattern = re.compile(expression)
    sm = [(name, body) for name, body in sass.items() if pattern.search(name)]
    pm = [(name, values) for name, values in ptxas.items() if pattern.search(name)]
    if len(sm) != 1 or len(pm) != 1:
        raise SystemExit(
            f"kernel identity failure for {label}: SASS={len(sm)} PTXAS={len(pm)}")
    sass_name, body = sm[0]
    ptxas_name, values = pm[0]
    missing = {
        "stack", "spill_stores", "spill_loads", "registers",
        "shared_memory",
    } - values.keys()
    if missing:
        raise SystemExit(f"incomplete PTXAS evidence for {label}: {sorted(missing)}")
    text = "".join(body)
    ldl = len(re.findall(r"\bLDL(?:\.|\b)", text))
    stl = len(re.findall(r"\bSTL(?:\.|\b)", text))
    idp4a = len(re.findall(r"\bIDP\.4A(?:\.|\b)", text))
    atom = len(re.findall(r"\b(?:ATOM|RED)(?:\.|\b)", text))
    registers = values["registers"]
    allocated = ((registers + 7) // 8) * 8
    if any((values["stack"], values["spill_stores"],
            values["spill_loads"], ldl, stl)):
        raise SystemExit(
            f"{label}: stack/spill/local traffic: stack={values['stack']} "
            f"spill_stores={values['spill_stores']} "
            f"spill_loads={values['spill_loads']} LDL={ldl} STL={stl}")
    if idp4a == 0:
        raise SystemExit(f"{label}: SASS contains no IDP.4A instruction")
    if atom:
        raise SystemExit(f"{label}: unexpected atomic/reduction SASS count={atom}")
    if allocated > register_limit:
        raise SystemExit(
            f"{label}: allocated registers {allocated} exceed {register_limit}")
    if values["shared_memory"] != 4096:
        raise SystemExit(
            f"{label}: expected exactly 4096 bytes shared memory, "
            f"got {values['shared_memory']}")
    rows.append({
        "variant": label, "block_size": block_size,
        "registers": registers, "allocated_registers": allocated,
        "register_limit": register_limit,
        "shared_memory_bytes": values["shared_memory"],
        "stack_frame_bytes": values["stack"],
        "spill_store_bytes": values["spill_stores"],
        "spill_load_bytes": values["spill_loads"],
        "sass_ldl": ldl, "sass_stl": stl, "sass_idp4a": idp4a,
        "sass_atom_red": atom, "resource_gate": "pass",
        "sass_symbol": sass_name, "ptxas_symbol": ptxas_name,
    })
with open(output_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
print("validated K1/K2/K4 resource identities and hard gates")
PY
cat "$OUTPUT_DIR/resource-summary.csv"

current_phase=inclusive-timing
printf 'variant,k1_median_ms,candidate_median_ms,candidate_speedup\n' \
    >"$OUTPUT_DIR/timing-summary.csv"
for variant in k2 k4; do
    log="$OUTPUT_DIR/timing/$variant.log"
    printf 'Inclusive production-owned-call timing: K1 vs %s...\n' "$variant"
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        DS4_CUDA_MOE_Q3A4_DECODE_MAPPING_AUDIT=1 \
        TIMING_ROUNDS="$TIMING_ROUNDS" TIMING_REPEATS="$TIMING_REPEATS" \
        ./$harness "${scenario[$variant]}-ab" >"$log" 2>&1 || {
            tail -n 120 "$log" >&2 || true
            die "$variant inclusive timing failed"
        }
    grep -q '^timing_scope=production-owned-call-inclusive$' "$log" ||
        die "$variant timing scope missing"
    grep -q "^candidate_kind=q3a4-tile32-dp4a-$variant$" "$log" ||
        die "$variant timing compared the wrong dispatch"
    grep -Fq -- "mapping=tile32-dp4a-$variant" "$log" ||
        die "$variant timing candidate did not reach the CUDA dispatch"
    case "$variant" in
        k2) timing_audit_pattern='tile32-dp4a=[1-9][0-9]* k1=[1-9][0-9]* k2=[1-9][0-9]* k4=0' ;;
        k4) timing_audit_pattern='tile32-dp4a=[1-9][0-9]* k1=[1-9][0-9]* k2=0 k4=[1-9][0-9]*' ;;
    esac
    grep -Eq -- "$timing_audit_pattern" "$log" ||
        die "$variant timing audit omitted K1/candidate work or ran the other candidate"
    control=$(grep '^control_median_ms=' "$log" | cut -d= -f2)
    candidate=$(grep '^candidate_median_ms=' "$log" | cut -d= -f2)
    speedup=$(grep '^candidate_speedup=' "$log" | cut -d= -f2)
    printf '%s,%s,%s,%s\n' "$variant" "$control" "$candidate" "$speedup" \
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
    # These metrics are known to be collectable on the target Turing setup.
    # Nsight's query output can omit tool-generated gpu__/launch__ names even
    # when direct --metrics collection succeeds, so discovery only selects
    # optional metrics.
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

    for variant in k1 k2 k4; do
        case "$variant" in
            k1) regex='moe_gate_up_mid_decode_sm75_q3a4_tile32_owned_kernel<true>'; block=128 ;;
            k2) regex='moe_gate_up_mid_decode_sm75_q3a4_tile32_ksplit_owned_kernel.*'; block=256 ;;
            k4) regex='moe_gate_up_mid_decode_sm75_q3a4_tile32_ksplit_owned_kernel.*'; block=512 ;;
        esac
        base="$OUTPUT_DIR/ncu/$variant"
        printf 'Nsight Compute: %s...\n' "$variant"
        rc=0
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
    done

    python3 - "$OUTPUT_DIR/ncu" "$OUTPUT_DIR/ncu-summary.csv" <<'PY'
import csv, math, pathlib, sys
ncu_dir, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
required = [line.strip() for line in
            (ncu_dir / "metrics-required.txt").read_text().splitlines()
            if line.strip()]
metrics = [
    "gpu__time_duration.sum", "dram__bytes.sum.per_second",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
    "lts__t_sector_hit_rate.pct",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_memory_pred_on.sum",
    "launch__registers_per_thread", "launch__shared_mem_per_block",
    "launch__block_size", "launch__grid_size",
    "launch__occupancy_limit_blocks", "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_warps",
    "launch__waves_per_multiprocessor",
]
out = []
for variant in ("k1", "k2", "k4"):
    rows = list(csv.DictReader(open(
        ncu_dir / f"{variant}.csv", newline="", encoding="utf-8-sig")))
    data = [row for row in rows if (row.get("ID") or "").strip()]
    units = [row for row in rows if not (row.get("ID") or "").strip()]
    if len(data) != 1:
        raise SystemExit(
            f"{variant}: expected one NCU kernel row, got {len(data)}")
    unit = units[-1] if units else {}
    for metric in required:
        value = (data[0].get(metric) or "").strip()
        if not value or value.lower() in {"n/a", "not available"}:
            raise SystemExit(f"{variant}: required metric {metric} has no value")
        try:
            number = float(value.replace(",", ""))
        except ValueError:
            raise SystemExit(
                f"{variant}: required metric {metric} is non-numeric: {value!r}")
        if not math.isfinite(number):
            raise SystemExit(
                f"{variant}: required metric {metric} is non-finite: {value!r}")
    for metric in metrics:
        out.append({
            "variant": variant, "metric": metric,
            "unit": (unit.get(metric) or "").strip(),
            "value": (data[0].get(metric) or "").strip(),
        })
with open(output, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=out[0].keys())
    writer.writeheader(); writer.writerows(out)
PY
    cat "$OUTPUT_DIR/ncu-summary.csv"
fi

current_phase=complete
printf 'SM75 Q3A4 exact in-CTA K-split audit complete: %s\n' "$OUTPUT_DIR"
