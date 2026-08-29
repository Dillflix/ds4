#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Compare SM75-native Q3A4 one-token gate/up mappings.

The shipping one-row/warp kernel is compared with:
  fused-u2   proven low-register one-row/warp control;
  hwarp16    two Q3A4 rows per warp, 16 K256 records per half warp;
  tile32     one warp follows one native eight-row tile;
  tile32-dp4a  the identical tile mapping with exact signed DP4A.

Q4-32 is not varied. Every Q3A4 candidate must pass the real 16-record
byte-exact regression and a production-shaped dispatch smoke. Timing includes
the complete owned routed call through Q4-32 down. PTXAS, SASS, and focused
Nsight evidence are captured independently for every mapping.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  TIMING_ROUNDS=9
  TIMING_REPEATS=25
  RUN_NCU=1
  NCU_USE_SUDO=0
  SKIP_BUILD=0
  CREATE_ARCHIVE=1
  Q3A4_NATIVE_DIR=/absolute/output/directory
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
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${Q3A4_NATIVE_DIR:-$repo_dir/sm75-decode-q3a4-native-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $TIMING_ROUNDS =~ ^[1-9][0-9]*$ ]] || die "TIMING_ROUNDS must be positive"
[[ $TIMING_REPEATS =~ ^[1-9][0-9]*$ ]] || die "TIMING_REPEATS must be positive"
for value in "$RUN_NCU" "$NCU_USE_SUDO" "$SKIP_BUILD" "$CREATE_ARCHIVE"; do
    [[ $value == 0 || $value == 1 ]] || die "binary options must be 0 or 1"
done

tools=(c++filt cuobjdump env git grep make nproc nvidia-smi python3 tar)
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
    printf 'scope=single-gpu-q3a4-local-kernel-selection\n'
    printf 'production_decision_requires=two-gpu-ffn-envelope-ab\n'
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
        die "SM75 Q3A4 native evidence build failed"
    fi
else
    die "SKIP_BUILD=1 cannot provide fresh PTXAS evidence; use SKIP_BUILD=0"
fi
[[ -x $smoke && -x $harness ]] || die "required CUDA binaries are missing"

current_phase=byte-exact-regression
printf 'Running byte-exact Q3A4 16-record production regression...\n'
env -u DS4_CUDA_MOE_Q32_DECODE_FUSED_LOWREG \
    -u DS4_CUDA_NO_MOE_Q32_DECODE_FUSED_LOWREG \
    -u DS4_CUDA_MOE_Q3A4_DECODE_MAPPING \
    -u DS4_CUDA_NO_MOE_Q3A4_DECODE_MAPPING \
    CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$smoke \
    >"$OUTPUT_DIR/smoke/cuda-long-context.log" 2>&1 || {
        tail -n 180 "$OUTPUT_DIR/smoke/cuda-long-context.log" >&2 || true
        die "byte-exact CUDA regression failed"
    }
grep -q 'SM75 Q3A4 hwarp16/tile32/dp4a gate/up' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" ||
    die "Q3A4 native exact marker missing"
grep -q 'SM75 Q3A4 DP4A byte packing exact' \
    "$OUTPUT_DIR/smoke/cuda-long-context.log" ||
    die "Q3A4 DP4A packing exact marker missing"

declare -A scenario=(
    [control]=q3a4-gate-up
    [fused-u2]=q3a4-gate-up-fused-u2
    [hwarp16]=q3a4-gate-up-hwarp16
    [tile32]=q3a4-gate-up-tile32
    [tile32-dp4a]=q3a4-gate-up-tile32-dp4a
)
for variant in control fused-u2 hwarp16 tile32 tile32-dp4a; do
    name=${scenario[$variant]}
    printf 'Production-shaped smoke: %s...\n' "$variant"
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$harness "$name" \
        >"$OUTPUT_DIR/smoke/$variant.log" 2>&1 || {
            tail -n 120 "$OUTPUT_DIR/smoke/$variant.log" >&2 || true
            die "$variant smoke failed"
        }
    grep -q '^output_validation=exact-zero$' "$OUTPUT_DIR/smoke/$variant.log" ||
        die "$variant output validation missing"
    grep -q '^harness_status=ok$' "$OUTPUT_DIR/smoke/$variant.log" ||
        die "$variant harness success marker missing"
done
grep -q '^q32_fused_lowreg_unroll=2$' "$OUTPUT_DIR/smoke/fused-u2.log" ||
    die "fused-u2 omitted its dispatch"
grep -q '^q3a4_decode_mapping=1$' "$OUTPUT_DIR/smoke/hwarp16.log" ||
    die "hwarp16 omitted its dispatch"
grep -q '^q3a4_decode_mapping=2$' "$OUTPUT_DIR/smoke/tile32.log" ||
    die "tile32 omitted its dispatch"
grep -q '^q3a4_decode_mapping=3$' "$OUTPUT_DIR/smoke/tile32-dp4a.log" ||
    die "tile32-dp4a omitted its dispatch"

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
    "control": (r"moe_gate_up_mid_decode_sm75_q32_owned_kernel<true>", False, 256, False),
    "fused-u2": (r"moe_gate_up_mid_decode_sm75_q32_fused_lowreg_owned_kernel<true,\s*(?:2u|2)>", True, 256, False),
    "hwarp16": (r"moe_gate_up_mid_decode_sm75_q3a4_hwarp16_owned_kernel", True, 256, False),
    "tile32": (r"moe_gate_up_mid_decode_sm75_q3a4_tile32_owned_kernel<false>", True, 128, False),
    "tile32-dp4a": (r"moe_gate_up_mid_decode_sm75_q3a4_tile32_owned_kernel<true>", True, 128, True),
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
    return result

sass = sections(sass_path, r"Function\s*:\s*(.*\S)", True)
ptxas = sections(build_path, r"Function properties for\s+(.*\S)", False)
rows = []
for label, (expression, candidate, block_size, require_dp4a) in targets.items():
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
    idp4a = len(re.findall(r"\bIDP\.4A(?:\.|\b)", text))
    registers = values["registers"]
    allocated = ((registers + 7) // 8) * 8
    if candidate and any((values["stack"], values["spill_stores"],
                          values["spill_loads"], ldl, stl)):
        raise SystemExit(
            f"{label}: candidate has stack/spill/local traffic: "
            f"stack={values['stack']} spill_stores={values['spill_stores']} "
            f"spill_loads={values['spill_loads']} LDL={ldl} STL={stl}")
    if require_dp4a and idp4a == 0:
        raise SystemExit("tile32-dp4a: SASS contains no IDP.4A instruction")
    eligible = (not candidate or (
        values["stack"] == values["spill_stores"] == values["spill_loads"] == 0
        and ldl == stl == 0 and allocated <= 128))
    rows.append({
        "variant": label, "candidate": "yes" if candidate else "no",
        "block_size": block_size, "registers": registers,
        "allocated_registers": allocated,
        "stack_frame_bytes": values["stack"],
        "spill_store_bytes": values["spill_stores"],
        "spill_load_bytes": values["spill_loads"],
        "sass_ldl": ldl, "sass_stl": stl, "sass_idp4a": idp4a,
        "resource_eligible": "yes" if eligible else "no",
        "sass_symbol": sass_name, "ptxas_symbol": ptxas_name,
    })

with open(output_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
print("validated five Q3A4 kernel identities and resource records")
PY
cat "$OUTPUT_DIR/resource-summary.csv"

timing_csv="$OUTPUT_DIR/timing-summary.csv"
printf 'variant,control_median_ms,candidate_median_ms,candidate_speedup\n' >"$timing_csv"
declare -A timing_scenario=(
    [fused-u2]=q3a4-gate-up-fused-u2-ab
    [hwarp16]=q3a4-gate-up-hwarp16-ab
    [tile32]=q3a4-gate-up-tile32-ab
    [tile32-dp4a]=q3a4-gate-up-tile32-dp4a-ab
)
current_phase=inclusive-timing
for variant in fused-u2 hwarp16 tile32 tile32-dp4a; do
    name=${timing_scenario[$variant]}
    log="$OUTPUT_DIR/timing/$variant.log"
    printf 'Inclusive production-shaped timing: %s...\n' "$variant"
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        TIMING_ROUNDS="$TIMING_ROUNDS" TIMING_REPEATS="$TIMING_REPEATS" \
        ./$harness "$name" >"$log" 2>&1 || {
            tail -n 120 "$log" >&2 || true; die "$variant timing failed";
        }
    grep -q '^timing_scope=production-owned-call-inclusive$' "$log" ||
        die "$variant timing scope missing"
    control=$(grep '^control_median_ms=' "$log" | cut -d= -f2)
    candidate=$(grep '^candidate_median_ms=' "$log" | cut -d= -f2)
    speedup=$(grep '^candidate_speedup=' "$log" | cut -d= -f2)
    printf '%s,%s,%s,%s\n' "$variant" "$control" "$candidate" "$speedup" \
        >>"$timing_csv"
done
cat "$timing_csv"

python3 - "$timing_csv" "$OUTPUT_DIR/resource-summary.csv" \
        "$OUTPUT_DIR/decision-summary.csv" <<'PY'
import csv, sys
timing_path, resource_path, output_path = sys.argv[1:]
resources = {r["variant"]: r for r in csv.DictReader(open(resource_path))}
rows = list(csv.DictReader(open(timing_path)))
for row in rows:
    row["resource_eligible"] = resources[row["variant"]]["resource_eligible"]
rows.sort(key=lambda r: (
    r["resource_eligible"] == "yes", float(r["candidate_speedup"])), reverse=True)
with open(output_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
print("Q3A4 measured ranking: " + ",".join(r["variant"] for r in rows))
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
    required_metrics=(
        gpu__time_duration.sum
        dram__bytes.sum.per_second
        dram__bytes.avg.pct_of_peak_sustained_elapsed
        lts__t_sector_hit_rate.pct
        smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
        l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
        smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
        smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
        sm__warps_active.avg.pct_of_peak_sustained_active
    )
    desired_metrics=(
        "${required_metrics[@]}"
        dram__bytes_read.sum
        dram__bytes_write.sum
        smsp__warps_eligible.avg.per_cycle_active
        smsp__sass_thread_inst_executed_op_integer_pred_on.sum
        smsp__sass_thread_inst_executed_op_memory_pred_on.sum
        launch__registers_per_thread
        launch__shared_mem_per_block
        launch__block_size
        launch__grid_size
        launch__waves_per_multiprocessor
    )
    available_raw="$OUTPUT_DIR/ncu/available-metrics.raw.txt"
    available_names="$OUTPUT_DIR/ncu/available-metric-names.txt"
    query_args=(--config-file off --devices 0 --query-metrics)
    ncu_help=$("$ncu_bin" --help 2>/dev/null || true)
    grep -Fq -- '--query-metrics-mode' <<<"$ncu_help" &&
        query_args+=(--query-metrics-mode all)
    if env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            "${ncu_command[@]}" "${query_args[@]}" \
            >"$available_raw" 2>"$OUTPUT_DIR/ncu/available-metrics-query.log"; then
        grep -Eo '[[:alnum:]_]+__[[:alnum:]_.]+' "$available_raw" | sort -u \
            >"$available_names" || true
        for metric in "${required_metrics[@]}"; do
            grep -Fxq -- "$metric" "$available_names" ||
                die "required Nsight metric is unavailable: $metric"
        done
        metrics=()
        : >"$OUTPUT_DIR/ncu/metrics-unavailable.txt"
        for metric in "${desired_metrics[@]}"; do
            if grep -Fxq -- "$metric" "$available_names"; then
                metrics+=("$metric")
            else
                printf '%s\n' "$metric" >>"$OUTPUT_DIR/ncu/metrics-unavailable.txt"
            fi
        done
    else
        printf 'warning: metric discovery failed; requesting the core set only\n' >&2
        metrics=("${required_metrics[@]}")
    fi
    printf '%s\n' "${metrics[@]}" >"$OUTPUT_DIR/ncu/metrics-selected.txt"
    metric_csv=$(IFS=,; printf '%s' "${metrics[*]}")
    profile_kernel() {
        local label=$1 name=${scenario[$1]} regex block base rc=0
        case "$label" in
            control)
                regex='moe_gate_up_mid_decode_sm75_q32_owned_kernel.*'; block=256 ;;
            fused-u2)
                regex='moe_gate_up_mid_decode_sm75_q32_fused_lowreg_owned_kernel.*'; block=256 ;;
            hwarp16)
                regex='moe_gate_up_mid_decode_sm75_q3a4_hwarp16_owned_kernel.*'; block=256 ;;
            tile32)
                regex='moe_gate_up_mid_decode_sm75_q3a4_tile32_owned_kernel.*'; block=128 ;;
            tile32-dp4a)
                regex='moe_gate_up_mid_decode_sm75_q3a4_tile32_owned_kernel.*'; block=128 ;;
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
                ./$harness "$name" >"$base.log" 2>&1 || rc=$?
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
    for variant in control fused-u2 hwarp16 tile32 tile32-dp4a; do
        profile_kernel "$variant"
    done

    python3 - "$OUTPUT_DIR/ncu" "$OUTPUT_DIR/ncu-summary.csv" <<'PY'
import csv, pathlib, sys
ncu_dir, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
metrics = [
    "gpu__time_duration.sum", "dram__bytes.sum.per_second",
    "dram__bytes.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
    "lts__t_sector_hit_rate.pct",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_memory_pred_on.sum",
    "launch__registers_per_thread", "launch__shared_mem_per_block",
    "launch__block_size", "launch__grid_size",
    "launch__waves_per_multiprocessor",
]
out = []
for variant in ("control", "fused-u2", "hwarp16", "tile32", "tile32-dp4a"):
    rows = list(csv.DictReader(open(ncu_dir / f"{variant}.csv", newline="", encoding="utf-8-sig")))
    data = [r for r in rows if (r.get("ID") or "").strip()]
    units = [r for r in rows if not (r.get("ID") or "").strip()]
    if len(data) != 1:
        raise SystemExit(f"{variant}: expected one NCU kernel row, got {len(data)}")
    unit = units[-1] if units else {}
    for metric in metrics:
        out.append({"variant": variant, "metric": metric,
                    "unit": (unit.get(metric) or "").strip(),
                    "value": (data[0].get(metric) or "").strip()})
with open(output, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=out[0].keys())
    writer.writeheader(); writer.writerows(out)
PY
    cat "$OUTPUT_DIR/ncu-summary.csv"
fi

current_phase=complete
printf 'SM75 Q3A4 native decode experiment complete: %s\n' "$OUTPUT_DIR"
