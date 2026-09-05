#!/usr/bin/env bash
set -euo pipefail

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

PROFILE_GPU=${PROFILE_GPU:-0}
ROWS=${ROWS:-8192}
TOKENS=${TOKENS:-32}
TIMING_ROUNDS=${TIMING_ROUNDS:-7}
TIMING_REPEATS=${TIMING_REPEATS:-25}
RUN_SANITIZER=${RUN_SANITIZER:-1}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-1}
SKIP_BUILD=${SKIP_BUILD:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
CUDA_ARCH=${CUDA_ARCH:-sm_75}

for value in "$PROFILE_GPU" "$ROWS" "$TOKENS" "$TIMING_ROUNDS" "$TIMING_REPEATS"; do
    [[ $value =~ ^[0-9]+$ ]] || die "numeric settings must be nonnegative integers"
done
for value in "$RUN_SANITIZER" "$RUN_NCU" "$NCU_USE_SUDO" \
             "$SKIP_BUILD" "$CREATE_ARCHIVE"; do
    [[ $value == 0 || $value == 1 ]] || die "boolean settings must be 0 or 1"
done
(( ROWS >= 512 && TOKENS > 0 && TIMING_ROUNDS > 0 && TIMING_REPEATS > 0 )) ||
    die "ROWS must be at least 512; TOKENS, TIMING_ROUNDS, and TIMING_REPEATS must be positive"
[[ $CUDA_ARCH == sm_75 ]] || die "this experiment requires CUDA_ARCH=sm_75"

for tool in make cuobjdump nvidia-smi; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool was not found"
done
if (( CREATE_ARCHIVE )); then
    command -v tar >/dev/null 2>&1 || die "CREATE_ARCHIVE=1 but tar was not found"
fi
if (( RUN_SANITIZER )); then
    command -v compute-sanitizer >/dev/null 2>&1 ||
        die "RUN_SANITIZER=1 but compute-sanitizer was not found"
fi
if (( RUN_NCU )); then
    command -v ncu >/dev/null 2>&1 || die "RUN_NCU=1 but ncu was not found"
    command -v python3 >/dev/null 2>&1 ||
        die "RUN_NCU=1 but python3 was not found"
fi

target=tests/cuda_sm75_compact_attention_kv
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${COMPACT_ATTENTION_KV_DIR:-"$PWD/sm75-compact-attention-kv-$timestamp"}
[[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

archive_on_exit() {
    local rc=$?
    local archive="$OUTPUT_DIR.tar.gz"
    trap - EXIT
    if (( CREATE_ARCHIVE )) && [[ -d $OUTPUT_DIR ]]; then
        if tar -czf "$archive" -C "$(dirname "$OUTPUT_DIR")" \
                "$(basename "$OUTPUT_DIR")"; then
            printf 'Archive to return: %s\n' "$archive"
        else
            printf 'warning: failed to create archive: %s\n' "$archive" >&2
        fi
    fi
    exit "$rc"
}
trap archive_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

git rev-parse HEAD >"$OUTPUT_DIR/git-head.txt"
git status --short >"$OUTPUT_DIR/git-status.txt"
nvidia-smi -i "$PROFILE_GPU" -q >"$OUTPUT_DIR/nvidia-smi-q.txt"

if (( ! SKIP_BUILD )); then
    make -B -j"$(nproc)" "$target" CUDA_ARCH="$CUDA_ARCH" \
        >"$OUTPUT_DIR/build.log" 2>&1 || {
            tail -n 200 "$OUTPUT_DIR/build.log" >&2
            die "build failed"
        }
else
    [[ -x $target ]] || die "$target is missing; rerun with SKIP_BUILD=0"
    printf 'skipped explicitly: SKIP_BUILD=1\n' >"$OUTPUT_DIR/build.log"
fi

grep -E 'Function properties|bytes stack frame|bytes spill (stores|loads)' \
    "$OUTPUT_DIR/build.log" >"$OUTPUT_DIR/whole-binary-ptxas-resources.txt" || true

cuobjdump --dump-resource-usage "$target" >"$OUTPUT_DIR/resource-usage.txt"
cuobjdump --dump-sass "$target" >"$OUTPUT_DIR/sass.txt"
printf 'scope=whole-binary-diagnostic-not-acceptance-gate\nwhole_binary_sass_ldl=%s\nwhole_binary_sass_stl=%s\n' \
    "$(grep -Ec '(^|[[:space:]])LDL([.[:space:]]|$)' "$OUTPUT_DIR/sass.txt" || true)" \
    "$(grep -Ec '(^|[[:space:]])STL([.[:space:]]|$)' "$OUTPUT_DIR/sass.txt" || true)" \
    >"$OUTPUT_DIR/local-traffic.txt"

"./$target" --device "$PROFILE_GPU" --rows 769 --tokens 2 --rounds 1 --repeats 1 \
    >"$OUTPUT_DIR/adversarial-smoke.log" 2>&1
grep -q '^harness_status=ok$' "$OUTPUT_DIR/adversarial-smoke.log" ||
    die "adversarial exactness smoke failed"
grep -q '^all_candidates_bit_exact=1$' "$OUTPUT_DIR/adversarial-smoke.log" ||
    die "one or more prototype smoke outputs were not exact"

if (( RUN_SANITIZER )); then
    compute-sanitizer --tool memcheck --error-exitcode 97 \
        "./$target" --device "$PROFILE_GPU" --rows 769 --tokens 2 --rounds 1 --repeats 1 \
        >"$OUTPUT_DIR/memcheck.log" 2>&1 || {
            tail -n 200 "$OUTPUT_DIR/memcheck.log" >&2
            die "compute-sanitizer failed"
        }
else
    printf 'skipped explicitly: RUN_SANITIZER=0\n' >"$OUTPUT_DIR/memcheck.log"
fi

"./$target" --device "$PROFILE_GPU" --rows "$ROWS" \
    --tokens "$TOKENS" \
    --rounds "$TIMING_ROUNDS" --repeats "$TIMING_REPEATS" \
    >"$OUTPUT_DIR/timing.log" 2>&1
grep -q '^harness_status=ok$' "$OUTPUT_DIR/timing.log" ||
    die "timed exactness run failed"
grep -q '^all_candidates_bit_exact=1$' "$OUTPUT_DIR/timing.log" ||
    die "one or more timed prototype outputs were not exact"

if (( RUN_NCU )); then
    mkdir -p "$OUTPUT_DIR/ncu"
    ncu_bin=$(command -v ncu)
    ncu_command=("$ncu_bin")
    if (( NCU_USE_SUDO )); then
        command -v sudo >/dev/null 2>&1 || die "NCU_USE_SUDO=1 but sudo was not found"
        sudo -v
        ncu_command=(sudo -E "$ncu_bin")
    fi

    required_metrics=(
        gpu__time_duration.sum
        dram__bytes.sum.per_second
        dram__bytes.avg.pct_of_peak_sustained_elapsed
        lts__t_sector_hit_rate.pct
        smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
        smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio
        sm__warps_active.avg.pct_of_peak_sustained_active
    )
    # Nsight 2026.3 exposes launch metadata in captures but may omit it from
    # --query-metrics. Request it directly rather than falsely rejecting the
    # installed profiler during discovery.
    launch_metrics=(
        launch__registers_per_thread
        launch__block_size
        launch__grid_size
        launch__waves_per_multiprocessor
    )
    desired_metrics=(
        "${required_metrics[@]}"
        dram__bytes_read.sum
        dram__bytes_write.sum
        l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
        l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
        l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
        l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio
        smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
        smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct
        smsp__sass_thread_inst_executed_op_integer_pred_on.sum
        smsp__sass_thread_inst_executed_op_conversion_pred_on.sum
        smsp__sass_thread_inst_executed_op_memory_pred_on.sum
        smsp__inst_executed.sum
        smsp__warps_eligible.avg.per_cycle_active
        launch__shared_mem_per_block
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
        metrics=("${launch_metrics[@]}")
        : >"$OUTPUT_DIR/ncu/metrics-unavailable.txt"
        for metric in "${desired_metrics[@]}"; do
            if grep -Fxq -- "$metric" "$available_names"; then
                metrics+=("$metric")
            else
                printf '%s\n' "$metric" >>"$OUTPUT_DIR/ncu/metrics-unavailable.txt"
            fi
        done
    else
        printf 'warning: metric discovery failed; requesting the required set only\n' >&2
        metrics=("${launch_metrics[@]}" "${required_metrics[@]}")
    fi
    printf '%s\n' "${metrics[@]}" >"$OUTPUT_DIR/ncu/metrics-selected.txt"
    metric_csv=$(IFS=,; printf '%s' "${metrics[*]}")
    regex='compact_materialize_selected_kernel.*'
    base="$OUTPUT_DIR/ncu/materialization"
    rc=0
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        "${ncu_command[@]}" --config-file off --verbose \
        --target-processes application-only --devices 0 \
        --kernel-name-base function --kernel-name "regex:$regex" \
        --launch-count 1 --replay-mode kernel --cache-control all \
        --clock-control none --metrics "$metric_csv" \
        --disable-extra-suffixes --force-overwrite --export "$base" \
        "./$target" --device 0 --rows "$ROWS" --tokens "$TOKENS" \
        --rounds 1 --repeats 1 >"$base.log" 2>&1 || rc=$?
    if (( rc != 0 )) ||
       grep -Eq '==ERROR==|No kernels were profiled|Failed to (profile|create report)' \
            "$base.log"; then
        tail -n 200 "$base.log" >&2 || true
        die "Nsight Compute materialization capture failed"
    fi
    [[ -s $base.ncu-rep ]] || die "missing materialization Nsight report"
    if (( NCU_USE_SUDO )); then
        sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
    fi
    "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
        >"$base.csv" 2>"$base-import.log" ||
        die "could not import materialization Nsight report"
    # The materializer visits the selected top-k set, not every persistent
    # cache row: one CTA for each of 512 selected rows per active token.
    expected_grid=$((512 * TOKENS))
    python3 speed-bench/validate-ncu-capture.py \
        "$base.csv" "$regex" 0 \
        --process cuda_sm75_compact_attention_kv \
        --block-size 128 --grid-size "$expected_grid" \
        >"$base-validation.txt" 2>&1 || {
            cat "$base-validation.txt" >&2 || true
            die "Nsight materialization capture validation failed"
        }
    cat "$base-validation.txt"
    python3 - "$base.csv" "$OUTPUT_DIR/ncu/materialization-summary.csv" <<'PY'
import csv, sys
source, destination = sys.argv[1:]
rows = list(csv.DictReader(open(source, newline="", encoding="utf-8-sig")))
data = [row for row in rows if (row.get("ID") or "").strip()]
units = [row for row in rows if not (row.get("ID") or "").strip()]
if len(data) != 1:
    raise SystemExit(f"expected one captured materialization kernel, got {len(data)}")
unit = units[-1] if units else {}
with open(destination, "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(("metric", "unit", "value"))
    for metric, value in data[0].items():
        if "__" not in metric or not (value or "").strip():
            continue
        writer.writerow((metric, (unit.get(metric) or "").strip(), value.strip()))
PY
    printf '%s\n' 'Materialization Nsight metrics:'
    column -s, -t "$OUTPUT_DIR/ncu/materialization-summary.csv" 2>/dev/null ||
        cat "$OUTPUT_DIR/ncu/materialization-summary.csv"
fi

printf 'SM75 exact compact-attention KV throughput prototypes complete: %s\n' "$OUTPUT_DIR"
tail -n 45 "$OUTPUT_DIR/timing.log"
