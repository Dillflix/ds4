#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Build, validate, benchmark, disassemble, and optionally profile the bounded
SM75 Q3_K/Q3-32/Q4-32 arithmetic and layout experiment. No GGUF is opened and
no production dispatch is changed.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  EXACT_CASES=256
  BENCH_REPEATS=128
  BENCH_LAUNCHES=20
  BENCH_ROUNDS=9
  STREAM_WEIGHT_CASES=16384
  PROFILE_REPEATS=128
  SKIP_BUILD=0
  RUN_SANITIZER=1
  RUN_NCU=1
  NCU_USE_SUDO=0
  CREATE_ARCHIVE=1
  Q3_Q4_32_DIR=/absolute/output/directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

script_rel=speed-bench/cuda-sm75-q3-q4-32.sh
source_rel=tests/cuda_sm75_q3_q4_32.cu
binary_rel=tests/cuda_sm75_q3_q4_32
design_rel=SM75_Q3_Q4_32_DESIGN.md
makefile_rel=Makefile
validator_rel=speed-bench/validate-ncu-capture.py

variants=(q4k-native-control q3k-k16-u8 sm75-q3-32 sm75-q4-32)
kernels=(
    sm75_q4k_native_control_kernel
    sm75_q3k_k16_u8_kernel
    sm75_q3_32_kernel
    sm75_q4_32_kernel
)

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
EXACT_CASES=${EXACT_CASES:-256}
BENCH_REPEATS=${BENCH_REPEATS:-128}
BENCH_LAUNCHES=${BENCH_LAUNCHES:-20}
BENCH_ROUNDS=${BENCH_ROUNDS:-9}
STREAM_WEIGHT_CASES=${STREAM_WEIGHT_CASES:-16384}
PROFILE_REPEATS=${PROFILE_REPEATS:-128}
SKIP_BUILD=${SKIP_BUILD:-0}
RUN_SANITIZER=${RUN_SANITIZER:-1}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${Q3_Q4_32_DIR:-$repo_dir/sm75-q3-q4-32-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be exactly sm_75"
[[ $OUTPUT_DIR == /* ]] || die "Q3_Q4_32_DIR must be an absolute path"
for name in EXACT_CASES BENCH_REPEATS BENCH_LAUNCHES BENCH_ROUNDS \
        STREAM_WEIGHT_CASES PROFILE_REPEATS; do
    value=${!name}
    [[ $value =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer"
done
(( STREAM_WEIGHT_CASES >= 8192 )) ||
    die "STREAM_WEIGHT_CASES must be at least 8192 to exceed RTX 8000 L2"
for name in SKIP_BUILD RUN_SANITIZER RUN_NCU NCU_USE_SUDO CREATE_ARCHIVE; do
    value=${!name}
    [[ $value == 0 || $value == 1 ]] || die "$name must be 0 or 1"
done

for command in nvidia-smi cuobjdump sha256sum stat tar; do
    command -v "$command" >/dev/null 2>&1 || die "$command not found"
done
[[ -f $source_rel ]] || die "missing $source_rel"
[[ -f $design_rel ]] || die "missing $design_rel"
[[ -f $validator_rel ]] || die "missing $validator_rel"
compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] ||
    die "physical GPU $PROFILE_GPU has compute capability ${compute_cap:-unknown}; SM75 is required"

[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
ARCHIVE_PATH=$OUTPUT_DIR.tar.gz
[[ $CREATE_ARCHIVE == 0 || ! -e $ARCHIVE_PATH ]] ||
    die "archive path already exists: $ARCHIVE_PATH"
mkdir -p "$OUTPUT_DIR/ncu" "$OUTPUT_DIR/provenance" "$OUTPUT_DIR/sass-kernels"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
ARCHIVE_PATH=$OUTPUT_DIR.tar.gz
current_phase=initialization
caught_signal=

take_output_ownership() {
    if [[ $NCU_USE_SUDO == 1 ]] && command -v sudo >/dev/null 2>&1; then
        sudo -n chown -R -- "$(id -u):$(id -g)" "$OUTPUT_DIR" \
            >/dev/null 2>&1 || true
    fi
}

write_status() {
    local state=$1 status=$2 archive=$3
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\nsignal=%s\narchive_status=%s\ndate_utc=%s\n' \
        "$state" "$status" "$current_phase" "${caught_signal:-none}" \
        "$archive" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$OUTPUT_DIR/run-status.txt"
}

finalize() {
    local status=$?
    trap - EXIT INT TERM HUP
    if [[ -d $OUTPUT_DIR ]]; then
        take_output_ownership
        local state=failed
        [[ -n $caught_signal ]] && state=interrupted
        if (( status == 0 )) && [[ $current_phase == complete ]]; then state=complete; fi
        if [[ $CREATE_ARCHIVE == 1 ]]; then
            local partial=$ARCHIVE_PATH.partial.$$
            write_status "$state" "$status" created
            if [[ ! -e $ARCHIVE_PATH ]] &&
                    tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial" \
                        "$(basename "$OUTPUT_DIR")" &&
                    mv -- "$partial" "$ARCHIVE_PATH"; then
                printf 'Archive to return: %s\n' "$ARCHIVE_PATH"
            else
                rm -f -- "$partial"
                status=1
                write_status failed "$status" failed
                printf 'error: could not create archive: %s\n' "$ARCHIVE_PATH" >&2
            fi
        else
            write_status "$state" "$status" disabled
        fi
    fi
    exit "$status"
}
trap finalize EXIT
trap 'caught_signal=INT; exit 130' INT
trap 'caught_signal=TERM; exit 143' TERM
trap 'caught_signal=HUP; exit 129' HUP

cp -- "$script_rel" "$source_rel" "$design_rel" "$makefile_rel" \
    "$validator_rel" \
    "$OUTPUT_DIR/provenance/"
sha256sum "$script_rel" "$source_rel" "$design_rel" "$makefile_rel" \
    "$validator_rel" \
    >"$OUTPUT_DIR/provenance/source-sha256.txt"
git diff --no-ext-diff --binary HEAD -- \
    "$script_rel" "$source_rel" "$design_rel" "$makefile_rel" \
    "$validator_rel" \
    >"$OUTPUT_DIR/provenance/tracked-working-tree.patch" || true

capture_gpu_state() {
    local label=$1
    {
        printf '\n[%s] date_utc=%s\n' "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        nvidia-smi -i "$PROFILE_GPU" \
            --query-gpu=index,pstate,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,power.limit,utilization.gpu,memory.used,memory.free \
            --format=csv
    } >>"$OUTPUT_DIR/gpu-state.log" 2>>"$OUTPUT_DIR/gpu-state-errors.log" || true
}

{
    printf 'date_utc=%s\nrepo=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
    printf 'git_commit=%s\ngit_branch=%s\n' "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'profile_gpu=%s\ncuda_arch=%s\ncompute_capability=%s\n' \
        "$PROFILE_GPU" "$CUDA_ARCH" "$compute_cap"
    printf 'exact_cases=%s\nbench_repeats=%s\nbench_launches=%s\nbench_rounds=%s\n' \
        "$EXACT_CASES" "$BENCH_REPEATS" "$BENCH_LAUNCHES" "$BENCH_ROUNDS"
    printf 'hot_weight_cases=16\nstream_weight_cases=%s\nprofile_repeats=%s\n' \
        "$STREAM_WEIGHT_CASES" "$PROFILE_REPEATS"
    printf 'full_model_loaded=false\nproduction_dispatch_changed=false\n'
    printf '\n[gpu]\n'
    nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,name,pci.bus_id,memory.total,memory.free,ecc.mode.current,driver_version \
        --format=csv
    printf '\n[source sha256]\n'; cat "$OUTPUT_DIR/provenance/source-sha256.txt"
    printf '\n[toolchain]\n'; nvcc --version 2>/dev/null || true
    cuobjdump --version 2>/dev/null || true
    ncu --version 2>/dev/null || true
    compute-sanitizer --version 2>/dev/null || true
} >"$OUTPUT_DIR/manifest.txt"
capture_gpu_state initialization

current_phase=build
if [[ $SKIP_BUILD == 0 ]]; then
    make -B "$binary_rel" CUDA_ARCH="$CUDA_ARCH" \
        >"$OUTPUT_DIR/build.log" 2>&1 || {
            tail -n 160 "$OUTPUT_DIR/build.log" >&2 || true
            die "SM75 Q3/Q4-32 harness build failed"
        }
else
    set +e
    make -q "$binary_rel" CUDA_ARCH="$CUDA_ARCH" \
        >"$OUTPUT_DIR/build.log" 2>&1
    query_rc=$?
    set -e
    [[ $query_rc == 0 ]] ||
        die "SKIP_BUILD=1 rejected a stale or invalid harness; rerun with SKIP_BUILD=0"
    [[ -x $binary_rel && $binary_rel -nt $source_rel && $binary_rel -nt $makefile_rel ]] ||
        die "SKIP_BUILD=1 rejected a missing or stale harness"
    printf 'build skipped after make -q and mtime validation\n' >>"$OUTPUT_DIR/build.log"
fi
[[ -x $binary_rel ]] || die "$binary_rel is missing"
sha256sum "$binary_rel" >"$OUTPUT_DIR/provenance/binary-sha256.txt"

current_phase=sass
cuobjdump --list-elf "$binary_rel" >"$OUTPUT_DIR/elf-list.txt" 2>&1
grep -Eq '\.sm_75\.cubin([[:space:]]|$)' "$OUTPUT_DIR/elf-list.txt" ||
    die "harness binary does not contain an sm_75 cubin"
cuobjdump --dump-resource-usage "$binary_rel" \
    >"$OUTPUT_DIR/resource-usage.txt" 2>&1 ||
    die "cuobjdump could not report kernel resource usage"
[[ -s $OUTPUT_DIR/resource-usage.txt ]] ||
    die "cuobjdump produced an empty resource-usage report"
cuobjdump --dump-sass "$binary_rel" >"$OUTPUT_DIR/sass.txt" 2>&1
printf 'variant,kernel,imma_total,imma_8816,imma_8832,s8_u8,u4_u4,s4_u4,u4_s4,s4_s4,ldl,stl,lop3,prmt,shf,bfe,iadd3\n' \
    >"$OUTPUT_DIR/sass-summary.csv"
for i in "${!variants[@]}"; do
    variant=${variants[$i]}
    kernel=${kernels[$i]}
    section=$OUTPUT_DIR/sass-kernels/$variant.sass.txt
    awk -v wanted="$kernel" '
        /Function : / {
            name=$0; sub(/^.*Function :[[:space:]]*/, "", name)
            sub(/[[:space:]]*$/, "", name); emit=name == wanted
        }
        emit { print }
    ' "$OUTPUT_DIR/sass.txt" >"$section"
    [[ -s $section ]] || die "SASS contains no function section for $kernel"
    imma=$(grep -Ec 'IMMA' "$section" || true)
    i16=$(grep -Ec 'IMMA[^[:space:]]*8816|IMMA\.8816' "$section" || true)
    i32=$(grep -Ec 'IMMA[^[:space:]]*8832|IMMA\.8832' "$section" || true)
    s8u8=$(grep -Eic 'IMMA[^[:space:]]*8816[^[:space:]]*S8[^[:space:]]*U8' "$section" || true)
    u4u4=$(grep -Eic 'IMMA[^[:space:]]*8832[^[:space:]]*U4[^[:space:]]*U4' "$section" || true)
    s4u4=$(grep -Eic 'IMMA[^[:space:]]*8832[^[:space:]]*S4[^[:space:]]*U4' "$section" || true)
    u4s4=$(grep -Eic 'IMMA[^[:space:]]*8832[^[:space:]]*U4[^[:space:]]*S4' "$section" || true)
    s4s4=$(grep -Eic 'IMMA[^[:space:]]*8832[^[:space:]]*S4[^[:space:]]*S4' "$section" || true)
    ldl=$(grep -Ec '(^|[[:space:]])LDL([[:space:].]|$)' "$section" || true)
    stl=$(grep -Ec '(^|[[:space:]])STL([[:space:].]|$)' "$section" || true)
    lop3=$(grep -Ec 'LOP3' "$section" || true)
    prmt=$(grep -Ec 'PRMT' "$section" || true)
    shf=$(grep -Ec 'SHF' "$section" || true)
    bfe=$(grep -Ec 'BFE' "$section" || true)
    iadd3=$(grep -Ec 'IADD3' "$section" || true)
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$variant" "$kernel" "$imma" "$i16" "$i32" "$s8u8" \
        "$u4u4" "$s4u4" "$u4s4" "$s4s4" "$ldl" "$stl" \
        "$lop3" "$prmt" "$shf" "$bfe" "$iadd3" \
        >>"$OUTPUT_DIR/sass-summary.csv"
    case $variant in
        q4k-native-control|sm75-q3-32)
            (( u4u4 > 0 && s4u4 > 0 && i16 == 0 &&
               imma == u4u4 + s4u4 )) ||
                die "$kernel is not exclusively the required U4xU4/S4xU4 K32 pair" ;;
        q3k-k16-u8)
            (( s8u8 > 0 && i32 == 0 && imma == s8u8 )) ||
                die "$kernel is not exclusively S8xU8 K16 IMMA" ;;
        sm75-q4-32)
            (( u4s4 > 0 && s4s4 > 0 && i16 == 0 &&
               imma == u4s4 + s4s4 )) ||
                 die "$kernel is not exclusively the required U4xS4/S4xS4 K32 pair" ;;
    esac
    (( ldl == 0 && stl == 0 )) ||
        die "$kernel uses local-memory instructions (LDL=$ldl STL=$stl)"
    ! grep -Eq 'IDP[.]4A|DP4A' "$section" ||
        die "$kernel unexpectedly contains DP4A instead of the required MMA path"
done

current_phase=correctness
capture_gpu_state pre-correctness
env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
    "$binary_rel" --device 0 --cases "$EXACT_CASES" --correctness-only \
        >"$OUTPUT_DIR/correctness.log" 2>&1 || {
            cat "$OUTPUT_DIR/correctness.log" >&2 || true
            die "Q3/Q4-32 exactness or round-trip gate failed"
        }
for marker in canonical_fixture_status unpack_status layout_status \
        roundtrip_status exact_status harness_status; do
    grep -q "^${marker}=ok$" "$OUTPUT_DIR/correctness.log" ||
        die "correctness harness did not report ${marker}=ok"
done
capture_gpu_state post-correctness

current_phase=sanitizer
if [[ $RUN_SANITIZER == 1 ]]; then
    command -v compute-sanitizer >/dev/null 2>&1 ||
        die "RUN_SANITIZER=1 but compute-sanitizer was not found"
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        compute-sanitizer --tool memcheck --error-exitcode 3 \
            "$binary_rel" --device 0 --cases 16 --correctness-only \
            >"$OUTPUT_DIR/memcheck.log" 2>&1 || {
                tail -n 160 "$OUTPUT_DIR/memcheck.log" >&2 || true
                die "compute-sanitizer memcheck failed"
            }
else
    printf 'skipped explicitly: RUN_SANITIZER=0\n' >"$OUTPUT_DIR/memcheck.log"
fi

current_phase=benchmark
capture_gpu_state pre-benchmark
env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
    "$binary_rel" --device 0 --benchmark-only \
        --bench-cases "$STREAM_WEIGHT_CASES" --rounds "$BENCH_ROUNDS" \
        --repeats "$BENCH_REPEATS" --launches "$BENCH_LAUNCHES" \
        >"$OUTPUT_DIR/benchmark.log" 2>&1 || {
            cat "$OUTPUT_DIR/benchmark.log" >&2 || true
            die "Q3/Q4-32 benchmark failed"
        }
[[ $(grep -c '^benchmark_summary_begin$' "$OUTPUT_DIR/benchmark.log") -eq 2 &&
   $(grep -c '^benchmark_summary_end$' "$OUTPUT_DIR/benchmark.log") -eq 2 ]] ||
    die "benchmark output does not contain exactly two complete summaries"
grep -q '^benchmark_layout_status=ok$' "$OUTPUT_DIR/benchmark.log" ||
    die "benchmark harness did not validate every native layout"
grep -q '^harness_status=ok$' "$OUTPUT_DIR/benchmark.log" ||
    die "benchmark harness did not report harness_status=ok"
for label in hot streamed; do
    awk -F, -v wanted="$label" '
        /^benchmark_summary_begin$/ { summary=1; next }
        /^benchmark_summary_end$/ { summary=0 }
        summary && /^mode,variant,/ { if (!header) { print; header=1 }; next }
        summary && $1 == wanted { print }
    ' "$OUTPUT_DIR/benchmark.log" >"$OUTPUT_DIR/benchmark-$label.csv"
    [[ $(wc -l <"$OUTPUT_DIR/benchmark-$label.csv") -eq 5 ]] ||
        die "$label benchmark CSV does not contain all four variants"
done
python3 - "$OUTPUT_DIR/benchmark-hot.csv" \
        "$OUTPUT_DIR/benchmark-streamed.csv" <<'PY' ||
    die "benchmark summaries failed semantic validation"
import csv
import math
import pathlib
import sys

expected = {
    "q4k-native-control", "q3k-k16-u8", "sm75-q3-32", "sm75-q4-32"
}
tile_bytes = {
    "q4k-native-control": 1152,
    "q3k-k16-u8": 880,
    "sm75-q3-32": 832,
    "sm75-q4-32": 1088,
}
numeric = (
    "weight_cases", "weight_footprint_bytes", "median_ms", "min_ms",
    "max_ms", "us_per_launch", "relative_speed", "logical_tmac_per_s",
)
for raw_path in sys.argv[1:]:
    path = pathlib.Path(raw_path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4:
        raise SystemExit(f"{path}: expected four rows, found {len(rows)}")
    names = [row.get("variant", "") for row in rows]
    if set(names) != expected or len(names) != len(set(names)):
        raise SystemExit(f"{path}: invalid variant inventory: {names}")
    wanted_mode = "streamed" if "streamed" in path.name else "hot"
    for row in rows:
        if row.get("mode") != wanted_mode:
            raise SystemExit(f"{path}: invalid mode {row.get('mode')!r}")
        for field in numeric:
            try:
                value = float(row[field])
            except (KeyError, ValueError) as error:
                raise SystemExit(f"{path}: invalid {field}: {error}")
            if not math.isfinite(value) or value <= 0:
                raise SystemExit(f"{path}: non-positive/non-finite {field}: {value}")
        cases = int(row["weight_cases"])
        footprint = int(row["weight_footprint_bytes"])
        expected_footprint = cases * tile_bytes[row["variant"]]
        if footprint != expected_footprint:
            raise SystemExit(
                f"{path}: footprint {footprint} != {expected_footprint} "
                f"for {row['variant']}"
            )
PY
capture_gpu_state post-benchmark

if [[ $RUN_NCU == 1 ]]; then
    current_phase=nsight
    command -v ncu >/dev/null 2>&1 || die "RUN_NCU=1 but ncu was not found"
    ncu_bin=$(command -v ncu)
    ncu_command=("$ncu_bin")
    if [[ $NCU_USE_SUDO == 1 ]]; then
        command -v sudo >/dev/null 2>&1 || die "sudo not found"
        sudo -v
        ncu_command=(sudo -E "$ncu_bin")
    fi
    desired_metric_names=(
        gpu__time_duration.sum
        launch__block_size
        launch__registers_per_thread
        launch__shared_mem_per_block
        launch__occupancy_limit_blocks
        launch__occupancy_limit_registers
        launch__occupancy_limit_shared_mem
        launch__occupancy_limit_warps
        sm__warps_active.avg.pct_of_peak_sustained_active
        smsp__warps_eligible.avg.per_cycle_active
        sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_elapsed
        sm__inst_executed_pipe_ipa.avg.pct_of_peak_sustained_elapsed
        smsp__inst_executed_pipe_ipa.sum
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
        l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum
        l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum
        l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
        l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
        l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum
        lts__t_sectors_op_read.sum
        lts__t_sectors_op_write.sum
        lts__t_bytes.sum
        lts__t_sectors_op_atom.sum
        sm__inst_executed_pipe_tensor.sum
        smsp__inst_executed.sum
        smsp__inst_executed_pipe_tensor.sum
        smsp__sass_thread_inst_executed_op_integer_pred_on.sum
        smsp__sass_thread_inst_executed_op_memory_pred_on.sum
    )
    # These have already been collectable on the target Turing/Nsight setup.
    # Tool-generated gpu__/launch__ metrics are not always listed by
    # --query-metrics, so collect and validate them rather than treating that
    # listing as authoritative.
    required_metric_names=(
        gpu__time_duration.sum
        launch__block_size
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
    available_raw=$OUTPUT_DIR/ncu/available-metrics.raw.txt
    available_names=$OUTPUT_DIR/ncu/available-metric-names.txt
    query_log=$OUTPUT_DIR/ncu/available-metrics-query.log
    query_args=(--config-file off --devices 0 --query-metrics)
    ncu_help=$("$ncu_bin" --help 2>/dev/null || true)
    if grep -Fq -- '--query-metrics-mode' <<<"$ncu_help"; then
        query_args+=(--query-metrics-mode all)
    fi
    query_rc=0
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        "${ncu_command[@]}" "${query_args[@]}" \
        >"$available_raw" 2>"$query_log" ||
        query_rc=$?
    take_output_ownership
    if (( query_rc == 0 )); then
        grep -Eo '[[:alnum:]_]+__[[:alnum:]_.]+' "$available_raw" |
            sort -u >"$available_names" || true
    else
        : >"$available_names"
        printf 'optional metric discovery failed with exit %s\n' "$query_rc" \
            >>"$query_log"
    fi
    metric_names=("${required_metric_names[@]}")
    optional_selected_metric_names=()
    : >"$OUTPUT_DIR/ncu/metrics-unavailable.txt"
    : >"$OUTPUT_DIR/ncu/optional-metrics-selected.txt"
    for candidate in "${desired_metric_names[@]}"; do
        required=0
        for required_metric in "${required_metric_names[@]}"; do
            if [[ $candidate == "$required_metric" ]]; then
                required=1
                break
            fi
        done
        (( required == 0 )) || continue
        if grep -Fxq -- "$candidate" "$available_names"; then
            metric_names+=("$candidate")
            optional_selected_metric_names+=("$candidate")
            printf '%s\n' "$candidate" \
                >>"$OUTPUT_DIR/ncu/optional-metrics-selected.txt"
        else
            printf '%s\n' "$candidate" \
                >>"$OUTPUT_DIR/ncu/metrics-unavailable.txt"
        fi
    done
    metrics=$(IFS=,; printf '%s' "${metric_names[*]}")
    printf '%s\n' "${desired_metric_names[@]}" \
        >"$OUTPUT_DIR/ncu/desired-metrics.txt"
    printf '%s\n' "${required_metric_names[@]}" \
        >"$OUTPUT_DIR/ncu/required-metrics.txt"
    printf '%s\n' "${metric_names[@]}" >"$OUTPUT_DIR/ncu/selected-metrics.txt"

    validate_ncu_metric_value() {
        local csv_path=$1 metric=$2
        python3 - "$csv_path" "$metric" <<'PY'
import csv
import math
import sys

path, metric = sys.argv[1:]
with open(path, newline="", encoding="utf-8-sig") as handle:
    rows = csv.reader(handle)
    try:
        header = next(rows)
    except StopIteration:
        raise SystemExit("empty Nsight CSV")
    try:
        id_column = header.index("ID")
        metric_column = header.index(metric)
    except ValueError as error:
        raise SystemExit(f"missing Nsight CSV column: {error}")
    data_rows = [
        row for row in rows
        if len(row) > max(id_column, metric_column) and row[id_column].strip()
    ]
    if len(data_rows) != 1:
        raise SystemExit(
            f"expected one nonempty-ID Nsight row, found {len(data_rows)}"
        )
    value = data_rows[0][metric_column].strip()
    if not value or value.lower() in {"n/a", "not available"}:
        raise SystemExit(f"metric {metric} has no value")
    try:
        number = float(value.replace(",", ""))
    except ValueError:
        raise SystemExit(f"metric {metric} is non-numeric: {value!r}")
    if not math.isfinite(number):
        raise SystemExit(f"metric {metric} is non-finite: {value!r}")
PY
    }
    for i in "${!variants[@]}"; do
        variant=${variants[$i]}
        kernel=${kernels[$i]}
        base=$OUTPUT_DIR/ncu/$variant
        printf 'Nsight Compute: %s...\n' "$variant"
        rc=0
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            "${ncu_command[@]}" --config-file off \
                --target-processes application-only --devices 0 \
                --kernel-name-base function --kernel-name "$kernel" \
                --launch-count 1 --replay-mode kernel --cache-control none \
                --clock-control none --force-overwrite --export "$base" \
                --metrics "$metrics" --disable-extra-suffixes \
                "$binary_rel" --device 0 --profile "$variant" \
                    --bench-cases "$STREAM_WEIGHT_CASES" \
                    --repeats "$PROFILE_REPEATS" \
                >"$base.log" 2>&1 || rc=$?
        take_output_ownership
        if (( rc != 0 )); then
            tail -n 160 "$base.log" >&2 || true
            die "Nsight Compute failed for $variant (exit $rc)"
        fi
        grep -Eq '==ERROR==|No kernels were profiled|Failed to (profile|create report)' \
            "$base.log" && die "Nsight Compute captured no valid kernel for $variant"
        grep -Fxq 'profile_status=ok' "$base.log" ||
            die "profile harness omitted profile_status=ok for $variant"
        grep -Fxq 'harness_status=ok' "$base.log" ||
            die "profile harness omitted harness_status=ok for $variant"
        [[ -s $base.ncu-rep ]] || die "missing Nsight report for $variant"
        "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
            >"$base.csv" 2>"$base-import.log" ||
            die "could not import Nsight report for $variant"
        [[ -s $base.csv ]] || die "empty Nsight CSV for $variant"
        python3 "$validator_rel" "$base.csv" "$kernel" 0 \
            --process cuda_sm75_q3_q4_32 --block-size 256 \
            >"$base-validation.txt" 2>&1 || {
                cat "$base-validation.txt" >&2 || true
                die "Nsight capture identity validation failed for $variant"
            }
        for metric in "${required_metric_names[@]}"; do
            validate_ncu_metric_value "$base.csv" "$metric" ||
                die "Nsight report has no value for required metric $metric ($variant)"
        done
    done
fi

current_phase=complete
capture_gpu_state final
printf 'SM75 Q3/Q4-32 experiment complete: %s\n' "$OUTPUT_DIR"
