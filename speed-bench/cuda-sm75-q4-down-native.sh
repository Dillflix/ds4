#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Build, validate, benchmark, disassemble, and profile the model-free SM75 Q4_K
down-projection native-packing harness.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  BENCH_ROUNDS=10
  BENCH_LAUNCHES=20
  PROFILE_SCENARIO=early    early or late
  SKIP_BUILD=0
  RUN_SANITIZER=1          run memcheck when compute-sanitizer is installed
  RUN_NCU=1
  NCU_USE_SUDO=0           use sudo -E for metric discovery and collection
  RESUME_NCU=0             reuse a completed failed run and run only NCU
  Q4_DOWN_NATIVE_DIR=/absolute/output/directory

The harness is synthetic and never opens a model. An archive is created on
success, failure, or interruption after the output directory is initialized.
RESUME_NCU=1 requires an existing explicit Q4_DOWN_NATIVE_DIR whose original
run reached the Nsight phase. Completed evidence is validated, never replaced.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

script_rel=speed-bench/cuda-sm75-q4-down-native.sh
source_rel=tests/cuda_sm75_q4_down_native.cu
binary_rel=tests/cuda_sm75_q4_down_native
makefile_rel=Makefile
engine_rel=ds4_cuda.cu
workflow_rel=.github/workflows/cuda-quantizer-build.yml
readme_rel=speed-bench/README.md
validator_rel=speed-bench/validate-ncu-capture.py

benchmark_variants=(standard native-w native-aw-consumer native-aw-combined pack-a)
profile_variants=(standard native-w native-aw-consumer pack-a)
profile_labels=(baseline native-w native-aw activation-pack)
# Skip matching warmups so the one captured launch is inside the harness's
# explicitly marked profile loop. Pack-A has one additional matching setup
# launch before run_profile().
profile_launch_skips=(1 1 1 2)
profile_kernels=(
    sm75_q4_down_standard_kernel
    sm75_q4_down_native_w_kernel
    sm75_q4_down_native_aw_kernel
    sm75_q4_down_pack_a_kernel
)
sass_labels=(standard native-w native-aw pack-a pack-a-inplace pack-w quantize-q8k quantize-native-q8k)
sass_kernels=(
    sm75_q4_down_standard_kernel
    sm75_q4_down_native_w_kernel
    sm75_q4_down_native_aw_kernel
    sm75_q4_down_pack_a_kernel
    sm75_q4_down_pack_a_inplace_kernel
    sm75_q4_down_pack_w_kernel
    sm75_q4_down_quantize_q8k_kernel
    sm75_q4_down_quantize_native_q8k_kernel
)

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
BENCH_ROUNDS=${BENCH_ROUNDS:-10}
BENCH_LAUNCHES=${BENCH_LAUNCHES:-20}
PROFILE_SCENARIO=${PROFILE_SCENARIO:-early}
SKIP_BUILD=${SKIP_BUILD:-0}
RUN_SANITIZER=${RUN_SANITIZER:-1}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
RESUME_NCU=${RESUME_NCU:-0}
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR=${Q4_DOWN_NATIVE_DIR:-$repo_dir/sm75-q4-down-native-$RUN_STAMP}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $CUDA_ARCH == sm_75 ]] || die "CUDA_ARCH must be exactly sm_75"
[[ $BENCH_ROUNDS =~ ^[1-9][0-9]*$ ]] || die "BENCH_ROUNDS must be a positive integer"
(( 10#$BENCH_ROUNDS % 10 == 0 )) ||
    die "BENCH_ROUNDS must be a multiple of 10 to balance five variants across sample positions"
[[ $BENCH_LAUNCHES =~ ^[1-9][0-9]*$ ]] || die "BENCH_LAUNCHES must be a positive integer"
[[ $PROFILE_SCENARIO == early || $PROFILE_SCENARIO == late ]] ||
    die "PROFILE_SCENARIO must be early or late"
for value_name in SKIP_BUILD RUN_SANITIZER RUN_NCU NCU_USE_SUDO RESUME_NCU; do
    value=${!value_name}
    [[ $value == 0 || $value == 1 ]] || die "$value_name must be 0 or 1"
done
[[ $OUTPUT_DIR == /* ]] || die "Q4_DOWN_NATIVE_DIR must be an absolute path"
if [[ $RESUME_NCU == 1 ]]; then
    [[ -n ${Q4_DOWN_NATIVE_DIR:-} ]] ||
        die "RESUME_NCU=1 requires an explicit Q4_DOWN_NATIVE_DIR"
    [[ $RUN_NCU == 1 ]] || die "RESUME_NCU=1 requires RUN_NCU=1"
fi

command -v tar >/dev/null 2>&1 || die "tar not found; archiving is mandatory"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum not found"
command -v stat >/dev/null 2>&1 || die "stat not found"
command -v python3 >/dev/null 2>&1 || die "python3 not found"
command -v cmp >/dev/null 2>&1 || die "cmp not found"
[[ -f $source_rel ]] || die "$source_rel not found"
[[ -f $makefile_rel ]] || die "$makefile_rel not found"
[[ -f $engine_rel ]] || die "$engine_rel not found"
[[ -f $workflow_rel ]] || die "$workflow_rel not found"
[[ -f $readme_rel ]] || die "$readme_rel not found"
[[ -f $validator_rel ]] || die "$validator_rel not found"
[[ -x $script_rel ]] || die "$script_rel is not executable; commit it with mode 100755"
git_status_before=$(git status --short --untracked-files=all)
if [[ $RESUME_NCU == 1 ]]; then
    [[ -d $OUTPUT_DIR ]] || die "resume output directory not found: $OUTPUT_DIR"
    OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
    RUN_DIR=$OUTPUT_DIR/resume-$RUN_STAMP
    [[ ! -e $RUN_DIR ]] || die "resume evidence path already exists: $RUN_DIR"
    ARCHIVE_PATH=$OUTPUT_DIR.resume-$RUN_STAMP.tar.gz
    [[ ! -e $ARCHIVE_PATH ]] || die "resume archive path already exists: $ARCHIVE_PATH"
    mkdir -p "$RUN_DIR/provenance" "$RUN_DIR/ncu"
else
    [[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
    ARCHIVE_PATH=$OUTPUT_DIR.tar.gz
    [[ ! -e $ARCHIVE_PATH ]] || die "archive path already exists: $ARCHIVE_PATH"
    mkdir -p "$OUTPUT_DIR/provenance" "$OUTPUT_DIR/sass-kernels" "$OUTPUT_DIR/ncu"
    OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
    RUN_DIR=$OUTPUT_DIR
    ARCHIVE_PATH=$OUTPUT_DIR.tar.gz
fi
NCU_DIR=$RUN_DIR/ncu

current_phase=$([[ $RESUME_NCU == 1 ]] && printf resume-initialization || printf initialization)
caught_signal=
benchmark_telemetry_pid=

stop_benchmark_telemetry() {
    if [[ -n ${benchmark_telemetry_pid:-} ]]; then
        kill "$benchmark_telemetry_pid" >/dev/null 2>&1 || true
        wait "$benchmark_telemetry_pid" >/dev/null 2>&1 || true
        benchmark_telemetry_pid=
    fi
}

start_benchmark_telemetry() {
    local scenario=$1
    (
        printf 'date_utc,pstate,temperature_c,graphics_clock_mhz,sm_clock_mhz,memory_clock_mhz,power_w,power_limit_w,gpu_util_pct,memory_util_pct\n'
        while :; do
            printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
            nvidia-smi -i "$PROFILE_GPU" \
                --query-gpu=pstate,temperature.gpu,clocks.current.graphics,clocks.current.sm,clocks.current.memory,power.draw,power.limit,utilization.gpu,utilization.memory \
                --format=csv,noheader,nounits || true
            sleep 0.2
        done
    ) >"$OUTPUT_DIR/benchmark-$scenario-telemetry.csv" \
        2>"$OUTPUT_DIR/benchmark-$scenario-telemetry-errors.log" &
    benchmark_telemetry_pid=$!
}

take_output_ownership() {
    if [[ $NCU_USE_SUDO == 1 ]] && command -v sudo >/dev/null 2>&1; then
        sudo -n chown -R -- "$(id -u):$(id -g)" "$RUN_DIR" \
            >/dev/null 2>&1 || true
    fi
}

write_run_status() {
    local state=$1 status=$2 archive_status=$3
    printf 'state=%s\nexit_status=%s\nlast_phase=%s\nsignal=%s\narchive_status=%s\ndate_utc=%s\n' \
        "$state" "$status" "$current_phase" "${caught_signal:-none}" \
        "$archive_status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"$RUN_DIR/run-status.txt"
}

finalize() {
    local status=$?
    trap - EXIT INT TERM HUP
    stop_benchmark_telemetry
    if [[ -d $RUN_DIR ]]; then
        take_output_ownership
        local state=failed
        if [[ -n $caught_signal ]]; then
            state=interrupted
        elif (( status == 0 )) && [[ $current_phase == complete ]]; then
            state=complete
        fi
        local partial_archive=$ARCHIVE_PATH.partial.$$
        write_run_status "$state" "$status" created
        if [[ ! -e $ARCHIVE_PATH ]] &&
                tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial_archive" \
                    "$(basename "$OUTPUT_DIR")" &&
                mv -- "$partial_archive" "$ARCHIVE_PATH"; then
            printf 'Archive to return: %s\n' "$ARCHIVE_PATH"
        else
            rm -f -- "$partial_archive"
            rm -f -- "$ARCHIVE_PATH"
            status=1
            write_run_status failed "$status" failed
            printf 'error: could not create archive: %s\n' "$ARCHIVE_PATH" >&2
        fi
    fi
    exit "$status"
}
trap finalize EXIT
trap 'caught_signal=INT; exit 130' INT
trap 'caught_signal=TERM; exit 143' TERM
trap 'caught_signal=HUP; exit 129' HUP

cp -- "$script_rel" "$source_rel" "$makefile_rel" "$engine_rel" \
    "$workflow_rel" "$readme_rel" "$validator_rel" "$RUN_DIR/provenance/"
sha256sum "$script_rel" "$source_rel" "$makefile_rel" "$engine_rel" \
    "$workflow_rel" "$readme_rel" "$validator_rel" \
    >"$RUN_DIR/provenance/source-sha256.txt"
git diff --no-ext-diff --binary HEAD -- \
    "$script_rel" "$source_rel" "$makefile_rel" "$engine_rel" \
    "$workflow_rel" "$readme_rel" "$validator_rel" \
    >"$RUN_DIR/provenance/tracked-working-tree.patch" || true
git diff --no-ext-diff --no-index /dev/null "$script_rel" \
    >"$RUN_DIR/provenance/untracked-script.patch" 2>/dev/null || true
git diff --no-ext-diff --no-index /dev/null "$source_rel" \
    >"$RUN_DIR/provenance/untracked-source.patch" 2>/dev/null || true

gpu_state_log=$RUN_DIR/gpu-state.log
gpu_state_error_log=$RUN_DIR/gpu-state-errors.log
capture_gpu_state() {
    local label=$1
    {
        printf '\n[%s] date_utc=%s\n' "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        nvidia-smi -i "$PROFILE_GPU" \
            --query-gpu=index,pstate,temperature.gpu,clocks.current.graphics,clocks.current.sm,clocks.current.memory,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used,memory.free \
            --format=csv
    } >>"$gpu_state_log" 2>>"$gpu_state_error_log" ||
        printf '[%s] GPU telemetry query failed\n' "$label" \
            >>"$gpu_state_error_log"
}

current_phase=preflight
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
command -v cuobjdump >/dev/null 2>&1 || die "cuobjdump not found"
compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] ||
    die "physical GPU $PROFILE_GPU has compute capability ${compute_cap:-unknown}; SM75 is required"

{
    printf 'date_utc=%s\nrepo=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
    printf 'git_commit=%s\ngit_branch=%s\n' "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'profile_gpu_physical=%s\nprofile_gpu_logical=0\n' "$PROFILE_GPU"
    printf 'cuda_arch=%s\ncompute_capability=%s\n' "$CUDA_ARCH" "$compute_cap"
    printf 'bench_rounds=%s\nbench_launches=%s\nprofile_scenario=%s\n' \
        "$BENCH_ROUNDS" "$BENCH_LAUNCHES" "$PROFILE_SCENARIO"
    printf 'model_required=false\nmodel_opened=false\nsynthetic_harness=true\n'
    printf 'script_mode=%s\nsource_mode=%s\nmakefile_mode=%s\nengine_mode=%s\nworkflow_mode=%s\nreadme_mode=%s\nvalidator_mode=%s\n' \
        "$(stat -c %a "$script_rel")" "$(stat -c %a "$source_rel")" \
        "$(stat -c %a "$makefile_rel")" "$(stat -c %a "$engine_rel")" \
        "$(stat -c %a "$workflow_rel")" "$(stat -c %a "$readme_rel")" \
        "$(stat -c %a "$validator_rel")"
    printf '\n[gpu]\n'
    nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,name,pci.bus_id,memory.total,memory.free,ecc.mode.current,driver_version \
        --format=csv
    printf '\n[git status before output creation]\n%s\n' "${git_status_before:-clean}"
    printf '\n[source sha256]\n'; cat "$RUN_DIR/provenance/source-sha256.txt"
    printf '\n[toolchain]\n'
    command -v nvcc || true
    nvcc --version 2>/dev/null || true
    cuobjdump --version 2>/dev/null || true
    ncu --version 2>/dev/null || true
    compute-sanitizer --version 2>/dev/null || true
    if [[ $RESUME_NCU == 1 ]]; then
        printf '\n[resume]\nresume_ncu=true\noriginal_output=%s\nresume_output=%s\n' \
            "$OUTPUT_DIR" "$RUN_DIR"
    fi
} >"$RUN_DIR/manifest.txt"
capture_gpu_state preflight

manifest_value() {
    local file=$1 key=$2
    awk -F= -v wanted="$key" '$1 == wanted { sub(/^[^=]*=/, ""); print; exit }' "$file"
}

verify_recorded_hash() {
    local hash_file=$1 path=$2 label=$3
    local expected actual
    expected=$(awk -v wanted="$path" '$2 == wanted { print $1; exit }' "$hash_file")
    [[ $expected =~ ^[[:xdigit:]]{64}$ ]] ||
        die "resume evidence has no recorded $label hash for $path"
    actual=$(sha256sum "$path" | awk '{ print $1 }')
    [[ $actual == "$expected" ]] ||
        die "resume rejected changed $label: $path"
    printf '%s_sha256=%s\n' "$label" "$actual"
}

if [[ $RESUME_NCU == 1 ]]; then
    current_phase=resume-validation-header
    original_status=$OUTPUT_DIR/run-status.txt
    original_manifest=$OUTPUT_DIR/manifest.txt
    original_source_hashes=$OUTPUT_DIR/provenance/source-sha256.txt
    original_binary_hashes=$OUTPUT_DIR/provenance/binary-sha256.txt
    for required_file in "$original_status" "$original_manifest" \
            "$original_source_hashes" "$original_binary_hashes"; do
        [[ -s $required_file ]] || die "resume evidence is missing: $required_file"
    done
    [[ $(manifest_value "$original_status" state) == failed ]] ||
        die "resume requires an original failed run"
    original_phase=$(manifest_value "$original_status" last_phase)
    [[ $original_phase == nsight-preflight || $original_phase == nsight ]] ||
        die "resume requires an original failure in nsight-preflight or nsight, found: ${original_phase:-unknown}"
    [[ $(manifest_value "$original_manifest" profile_gpu_physical) == "$PROFILE_GPU" ]] ||
        die "resume PROFILE_GPU differs from the original run"
    [[ $(manifest_value "$original_manifest" cuda_arch) == "$CUDA_ARCH" ]] ||
        die "resume CUDA_ARCH differs from the original run"
    [[ $(manifest_value "$original_manifest" compute_capability) == "$compute_cap" ]] ||
        die "resume compute capability differs from the original run"
    [[ $(manifest_value "$original_manifest" bench_rounds) == "$BENCH_ROUNDS" ]] ||
        die "resume BENCH_ROUNDS differs from the original run"
    [[ $(manifest_value "$original_manifest" bench_launches) == "$BENCH_LAUNCHES" ]] ||
        die "resume BENCH_LAUNCHES differs from the original run"
    [[ $(manifest_value "$original_manifest" profile_scenario) == "$PROFILE_SCENARIO" ]] ||
        die "resume PROFILE_SCENARIO differs from the original run"
    {
        printf 'original_state=failed\noriginal_last_phase=%s\n' "$original_phase"
        verify_recorded_hash "$original_source_hashes" "$source_rel" source
        verify_recorded_hash "$original_source_hashes" "$makefile_rel" makefile
        verify_recorded_hash "$original_binary_hashes" "$binary_rel" binary
    } >"$RUN_DIR/resume-validation.log"
    set +e
    make -q "$binary_rel" CUDA_ARCH="$CUDA_ARCH" \
        >>"$RUN_DIR/resume-validation.log" 2>&1
    make_query_rc=$?
    set -e
    [[ $make_query_rc == 0 ]] ||
        die "resume rejected a stale or invalid harness build (make -q exit $make_query_rc)"
    [[ -x $binary_rel && $binary_rel -nt $source_rel && $binary_rel -nt $makefile_rel ]] ||
        die "resume rejected a missing or stale harness binary"
    printf 'build_freshness=validated\n' >>"$RUN_DIR/resume-validation.log"
fi

current_phase=$([[ $RESUME_NCU == 1 ]] && printf resume-validation-build || printf build)
if [[ $RESUME_NCU == 1 ]]; then
    :
elif [[ $SKIP_BUILD == 0 ]]; then
    make -B "$binary_rel" CUDA_ARCH="$CUDA_ARCH" \
        >"$OUTPUT_DIR/build.log" 2>&1 || {
            tail -n 120 "$OUTPUT_DIR/build.log" >&2 || true
            die "SM75 Q4-down native harness build failed"
        }
else
    set +e
    make -q "$binary_rel" CUDA_ARCH="$CUDA_ARCH" \
        >"$OUTPUT_DIR/build.log" 2>&1
    make_query_rc=$?
    set -e
    case $make_query_rc in
        0) ;;
        1) die "SKIP_BUILD=1 rejected a stale harness; rerun with SKIP_BUILD=0" ;;
        *) die "could not validate the skipped harness build; rerun with SKIP_BUILD=0" ;;
    esac
    [[ $binary_rel -nt $source_rel && $binary_rel -nt $makefile_rel ]] ||
        die "SKIP_BUILD=1 rejected a harness older than its source or Makefile"
    printf 'build skipped after make -q and mtime validation\n' \
        >>"$OUTPUT_DIR/build.log"
fi
[[ -x $binary_rel ]] || die "$binary_rel is missing; rerun with SKIP_BUILD=0"
sha256sum "$binary_rel" >"$RUN_DIR/provenance/binary-sha256.txt"
{
    printf '\n[binary]\n'
    cat "$RUN_DIR/provenance/binary-sha256.txt"
    stat -c 'mode=%a size=%s mtime=%y file=%n' "$binary_rel"
} >>"$RUN_DIR/manifest.txt"

current_phase=$([[ $RESUME_NCU == 1 ]] && printf resume-validation-sass || printf sass)
if [[ $RESUME_NCU == 1 ]]; then
    for required_file in "$OUTPUT_DIR/elf-list.txt" "$OUTPUT_DIR/sass.txt" \
            "$OUTPUT_DIR/sass-summary.csv" "$OUTPUT_DIR/sass-relevant.txt"; do
        [[ -s $required_file ]] || die "resume SASS evidence is missing: $required_file"
    done
    grep -Eq '\.sm_75\.cubin([[:space:]]|$)' "$OUTPUT_DIR/elf-list.txt" ||
        die "resume SASS evidence has no sm_75 cubin"
    for i in "${!sass_labels[@]}"; do
        label=${sass_labels[$i]}
        kernel=${sass_kernels[$i]}
        kernel_sass=$OUTPUT_DIR/sass-kernels/$label.sass.txt
        [[ -s $kernel_sass ]] || die "resume SASS evidence is missing: $kernel_sass"
        grep -Fq "Function : $kernel" "$kernel_sass" ||
            die "resume SASS evidence does not contain $kernel"
    done
    python3 - "$OUTPUT_DIR/sass-summary.csv" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="", encoding="utf-8-sig") as handle:
    rows = {row["label"]: row for row in csv.DictReader(handle)}
required = {
    "standard", "native-w", "native-aw", "pack-a", "pack-a-inplace",
    "pack-w", "quantize-q8k", "quantize-native-q8k",
}
if set(rows) != required:
    raise SystemExit(f"unexpected SASS summary labels: {sorted(rows)}")

def count(label, field):
    try:
        return int(rows[label][field])
    except (KeyError, ValueError) as error:
        raise SystemExit(f"invalid SASS summary {label}/{field}: {error}")

if count("standard", "imma_8816") <= 0:
    raise SystemExit("standard kernel has no recorded m8n8k16 IMMA")
for label in ("native-w", "native-aw"):
    if count(label, "imma_8832") <= 0:
        raise SystemExit(f"{label} has no recorded m8n8k32 IMMA")
    if count(label, "s4_u4") <= 0 or count(label, "u4_u4") <= 0:
        raise SystemExit(f"{label} is missing a packed INT4 operand form")
for label in (
    "standard", "native-w", "native-aw", "pack-a-inplace",
    "quantize-q8k", "quantize-native-q8k",
):
    if count(label, "ldl") != 0 or count(label, "stl") != 0:
        raise SystemExit(f"{label} has recorded local-memory traffic")
PY
    printf 'sass_evidence=validated\n' >>"$RUN_DIR/resume-validation.log"
else
    cuobjdump --list-elf "$binary_rel" >"$OUTPUT_DIR/elf-list.txt" 2>&1
    grep -Eq '\.sm_75\.cubin([[:space:]]|$)' "$OUTPUT_DIR/elf-list.txt" ||
        die "harness binary does not contain an sm_75 cubin"
    cuobjdump --dump-resource-usage "$binary_rel" \
        >"$OUTPUT_DIR/resource-usage.txt" 2>&1 || true
    cuobjdump --dump-sass "$binary_rel" >"$OUTPUT_DIR/sass.txt" 2>&1
    printf 'label,kernel,total_imma,imma_8816,imma_8832,s4_u4,u4_u4,lop3,prmt,shf,bfe,iadd3,ldg,stg,ldl,stl\n' \
        >"$OUTPUT_DIR/sass-summary.csv"
    : >"$OUTPUT_DIR/sass-relevant.txt"
    for i in "${!sass_labels[@]}"; do
    label=${sass_labels[$i]}
    kernel=${sass_kernels[$i]}
    kernel_sass=$OUTPUT_DIR/sass-kernels/$label.sass.txt
    awk -v wanted="$kernel" '
        /Function : / {
            current = $0
            sub(/^.*Function :[[:space:]]*/, "", current)
            sub(/[[:space:]]*$/, "", current)
            emit = current == wanted
        }
        emit { print }
    ' "$OUTPUT_DIR/sass.txt" >"$kernel_sass"
    [[ -s $kernel_sass ]] || die "SASS contains no function section for $kernel"

    total_imma=$(grep -Ec 'IMMA' "$kernel_sass" || true)
    imma_8816=$(grep -Ec 'IMMA[^[:space:]]*8816|IMMA\.8816' "$kernel_sass" || true)
    imma_8832=$(grep -Ec 'IMMA[^[:space:]]*8832|IMMA\.8832' "$kernel_sass" || true)
    s4_u4=$(grep -Eic 'IMMA[^[:space:]]*S4[^[:space:]]*U4' "$kernel_sass" || true)
    u4_u4=$(grep -Eic 'IMMA[^[:space:]]*U4[^[:space:]]*U4' "$kernel_sass" || true)
    lop3=$(grep -Ec 'LOP3' "$kernel_sass" || true)
    prmt=$(grep -Ec 'PRMT' "$kernel_sass" || true)
    shf=$(grep -Ec 'SHF' "$kernel_sass" || true)
    bfe=$(grep -Ec 'BFE' "$kernel_sass" || true)
    iadd3=$(grep -Ec 'IADD3' "$kernel_sass" || true)
    ldg=$(grep -Ec 'LDG' "$kernel_sass" || true)
    stg=$(grep -Ec 'STG' "$kernel_sass" || true)
    ldl=$(grep -Ec '(^|[[:space:]])LDL([[:space:].]|$)' "$kernel_sass" || true)
    stl=$(grep -Ec '(^|[[:space:]])STL([[:space:].]|$)' "$kernel_sass" || true)
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$label" "$kernel" "$total_imma" "$imma_8816" "$imma_8832" \
        "$s4_u4" "$u4_u4" "$lop3" "$prmt" "$shf" "$bfe" "$iadd3" \
        "$ldg" "$stg" "$ldl" "$stl" \
        >>"$OUTPUT_DIR/sass-summary.csv"
    {
        printf '\n===== %s (%s) =====\n' "$label" "$kernel"
        grep -E 'Function :|IMMA|LOP3|PRMT|SHF|BFE|IADD3|LDG|STG' \
            "$kernel_sass" || true
    } >>"$OUTPUT_DIR/sass-relevant.txt"

    if [[ $label == standard ]]; then
        (( imma_8816 > 0 )) || die "$kernel contains no 8x8x16 IMMA instruction"
    elif [[ $label == native-w || $label == native-aw ]]; then
        (( imma_8832 > 0 )) || die "$kernel contains no 8x8x32 IMMA instruction"
        (( s4_u4 > 0 )) || die "$kernel contains no S4 x U4 IMMA instruction"
        (( u4_u4 > 0 )) || die "$kernel contains no U4 x U4 IMMA instruction"
    fi
    if [[ $label == standard || $label == native-w || $label == native-aw ||
          $label == pack-a-inplace || $label == quantize-q8k ||
          $label == quantize-native-q8k ]]; then
        (( ldl == 0 && stl == 0 )) ||
            die "$kernel contains local-memory load/store spill indicators"
        fi
    done
fi

current_phase=$([[ $RESUME_NCU == 1 ]] && printf resume-validation-correctness || printf correctness)
if [[ $RESUME_NCU == 0 ]]; then
    capture_gpu_state pre-correctness
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        ./tests/cuda_sm75_q4_down_native --device 0 --correctness-only \
            >"$OUTPUT_DIR/correctness.log" 2>&1 || {
                cat "$OUTPUT_DIR/correctness.log" >&2 || true
                die "Q4-down native correctness failed"
            }
fi
[[ -s $OUTPUT_DIR/correctness.log ]] || die "correctness evidence is missing"
for marker in activation_pack_validation=exact \
        direct_native_quantizer_validation=byte_exact \
        output_validation=bit_exact \
        unowned_output_poison_validation=exact correctness_status=ok \
        harness_status=ok; do
    grep -Fxq "$marker" "$OUTPUT_DIR/correctness.log" ||
        die "correctness harness omitted required marker: $marker"
done
if [[ $RESUME_NCU == 0 ]]; then
    capture_gpu_state post-correctness
else
    printf 'correctness_evidence=validated\n' >>"$RUN_DIR/resume-validation.log"
fi

current_phase=$([[ $RESUME_NCU == 1 ]] && printf resume-validation-sanitizer || printf sanitizer)
if [[ $RESUME_NCU == 1 ]]; then
    [[ -s $OUTPUT_DIR/memcheck.log ]] || die "sanitizer evidence is missing"
    if grep -Fq 'ERROR SUMMARY: 0 errors' "$OUTPUT_DIR/memcheck.log"; then
        printf 'sanitizer_evidence=clean\n' >>"$RUN_DIR/resume-validation.log"
    elif grep -Eq '^skipped: RUN_SANITIZER=(0|1) compute-sanitizer=' \
            "$OUTPUT_DIR/memcheck.log"; then
        printf 'sanitizer_evidence=explicitly_skipped\n' \
            >>"$RUN_DIR/resume-validation.log"
    else
        die "resume sanitizer evidence is neither clean nor explicitly skipped"
    fi
else
    capture_gpu_state pre-sanitizer
    if [[ $RUN_SANITIZER == 1 ]] && command -v compute-sanitizer >/dev/null 2>&1; then
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        compute-sanitizer --tool memcheck --error-exitcode 3 \
            ./tests/cuda_sm75_q4_down_native --device 0 --correctness-only \
            >"$OUTPUT_DIR/memcheck.log" 2>&1 || {
                tail -n 120 "$OUTPUT_DIR/memcheck.log" >&2 || true
                die "compute-sanitizer memcheck failed"
            }
    grep -Fq 'ERROR SUMMARY: 0 errors' "$OUTPUT_DIR/memcheck.log" ||
        die "compute-sanitizer did not report a clean error summary"
    else
        printf 'skipped: RUN_SANITIZER=%s compute-sanitizer=%s\n' \
            "$RUN_SANITIZER" "$(command -v compute-sanitizer || true)" \
            >"$OUTPUT_DIR/memcheck.log"
    fi
    capture_gpu_state post-sanitizer
fi

validate_benchmark() {
    local scenario=$1 log=$2 samples=$3 summary=$4
    grep -Fxq 'benchmark_status=ok' "$log" ||
        die "$scenario benchmark omitted benchmark_status=ok"
    grep -Fxq 'harness_status=ok' "$log" ||
        die "$scenario benchmark omitted harness_status=ok"
    for marker in direct_native_quantizer_benchmark_begin=1 \
            direct_native_quantizer_post_timing_exactness=byte-exact \
            direct_native_quantizer_post_timing_canaries=ok \
            direct_native_quantizer_benchmark_end=1; do
        grep -Fxq "$marker" "$log" ||
            die "$scenario benchmark omitted required marker: $marker"
    done
    if ! python3 - "$log" "$BENCH_ROUNDS" "$BENCH_LAUNCHES" <<'PY'
import math
import re
import sys

path, rounds_text, iterations_text = sys.argv[1:]
with open(path, encoding="utf-8-sig") as handle:
    lines = [line.rstrip("\n") for line in handle]
values = {}
for line in lines:
    if "=" in line:
        key, value = line.split("=", 1)
        values[key] = value

expected_ints = {
    "direct_native_quantizer_rounds": int(rounds_text),
    "direct_native_quantizer_iterations_per_sample": int(iterations_text),
    "quantizer_rows_per_iteration": 3072,
    "quantizer_blocks_per_row": 8,
    "quantizer_blocks_per_iteration": 24576,
    "canonical_quantize_pack_launches_per_iteration": 2,
    "direct_native_quantize_launches_per_iteration": 1,
    "quantizer_input_bytes_per_iteration": 25165824,
    "native_q8_record_bytes": 292,
    "canonical_intermediate_global_bytes_per_block": 584,
    "canonical_intermediate_global_bytes_per_iteration": 14352384,
    "native_q8_output_bytes_per_iteration": 7176192,
    "canonical_nominal_global_bytes_per_iteration": 46694400,
    "direct_native_nominal_global_bytes_per_iteration": 32342016,
    "direct_native_intermediate_global_bytes_per_block": 0,
}
for key, expected in expected_ints.items():
    try:
        actual = int(values[key])
    except (KeyError, ValueError) as error:
        raise SystemExit(f"invalid direct-quantizer field {key}: {error}")
    if actual != expected:
        raise SystemExit(f"unexpected {key}: {actual}, expected {expected}")
if values.get("direct_native_quantizer_scope") != \
        "one-production-shaped-activation-surface":
    raise SystemExit("direct quantizer reported an unexpected benchmark scope")
if values.get("direct_native_quantizer_consumer_excluded") != "1":
    raise SystemExit("direct quantizer benchmark did not exclude the consumer")

def positive(key):
    try:
        value = float(values[key])
    except (KeyError, ValueError) as error:
        raise SystemExit(f"invalid direct-quantizer field {key}: {error}")
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"non-positive/non-finite {key}: {value}")
    return value

canonical = positive("canonical_quantize_pack_median_total_ms")
direct = positive("direct_native_quantize_median_total_ms")
speedup = positive("direct_native_quantize_speedup")
if not math.isclose(speedup, canonical / direct,
                    rel_tol=2e-6, abs_tol=2e-6):
    raise SystemExit("direct-native quantizer speedup is inconsistent")
sample_paths = [line for line in lines if re.fullmatch(
    r"direct_native_quantizer_round_\d+_slot_\d+_path="
    r"(?:canonical-quantize-pack|direct-native-quantize)", line)]
if len(sample_paths) != 2 * int(rounds_text):
    raise SystemExit("direct-native quantizer sample count is incomplete")
PY
    then
        die "$scenario direct-native quantizer benchmark validation failed"
    fi
    grep -Fxq 'scenario,round,sample_slot,variant,total_ms,us_per_launch,relative_speed' \
        "$log" || die "$scenario benchmark omitted the sample CSV header"
    grep -Fxq 'scenario,variant,median_total_ms,median_us_per_launch,relative_speed' \
        "$log" || die "$scenario benchmark omitted the median CSV header"

    awk '
        /^scenario,round,sample_slot,variant,total_ms,us_per_launch,relative_speed$/ {
            emit = 1; print; next
        }
        /^scenario,variant,median_total_ms,median_us_per_launch,relative_speed$/ {
            emit = 0
        }
        emit && /^[^,]+,[^,]+,[^,]+,[^,]+,/ { print }
    ' "$log" >"$samples"
    awk '
        /^scenario,variant,median_total_ms,median_us_per_launch,relative_speed$/ {
            emit = 1; print; next
        }
        emit && /^[^,]+,[^,]+,/ { print }
    ' "$log" >"$summary"

    if ! python3 - "$samples" "$summary" "$scenario" "$BENCH_ROUNDS" \
            "$BENCH_LAUNCHES" "${benchmark_variants[@]}" <<'PY'
import csv
import math
import statistics
import sys

samples_path, summary_path, scenario, rounds_text, launches_text, *variants = sys.argv[1:]
rounds = int(rounds_text)
launches = int(launches_text)
variant_set = set(variants)
slots = set(range(1, len(variants) + 1))

def positive(row, field):
    try:
        value = float(row[field])
    except (KeyError, ValueError):
        raise SystemExit(f"invalid {field}: {row!r}")
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"non-positive/non-finite {field}: {row!r}")
    return value

with open(samples_path, newline="", encoding="utf-8-sig") as handle:
    sample_rows = list(csv.DictReader(handle))
expected_rows = rounds * len(variants)
if len(sample_rows) != expected_rows:
    raise SystemExit(f"expected {expected_rows} samples, found {len(sample_rows)}")

by_round = {r: [] for r in range(1, rounds + 1)}
slot_counts = {(variant, slot): 0 for variant in variants for slot in slots}
values = {variant: [] for variant in variants}
for row in sample_rows:
    if row.get("scenario") != scenario or row.get("variant") not in variant_set:
        raise SystemExit(f"unexpected scenario/variant: {row!r}")
    try:
        round_no = int(row["round"])
        slot = int(row["sample_slot"])
    except (KeyError, ValueError):
        raise SystemExit(f"invalid round/slot: {row!r}")
    if round_no not in by_round or slot not in slots:
        raise SystemExit(f"out-of-range round/slot: {row!r}")
    total_ms = positive(row, "total_ms")
    us_per_launch = positive(row, "us_per_launch")
    relative_speed = positive(row, "relative_speed")
    expected_us = 1000.0 * total_ms / launches
    if not math.isclose(us_per_launch, expected_us, rel_tol=2e-6, abs_tol=2e-6):
        raise SystemExit(f"inconsistent per-launch timing: {row!r}")
    by_round[round_no].append((row["variant"], slot, total_ms, relative_speed))
    slot_counts[(row["variant"], slot)] += 1
    values[row["variant"]].append(total_ms)

for round_no, rows in by_round.items():
    if {row[0] for row in rows} != variant_set or {row[1] for row in rows} != slots:
        raise SystemExit(f"round {round_no} is not a variant/slot permutation")
    standard_ms = next(row[2] for row in rows if row[0] == "standard")
    for variant, _slot, total_ms, relative_speed in rows:
        if not math.isclose(relative_speed, standard_ms / total_ms,
                            rel_tol=2e-6, abs_tol=2e-6):
            raise SystemExit(f"round {round_no} has inconsistent speed for {variant}")

expected_per_slot = rounds // len(variants)
for key, count in slot_counts.items():
    if count != expected_per_slot:
        raise SystemExit(f"unbalanced sample positions for {key}: {count}")

with open(summary_path, newline="", encoding="utf-8-sig") as handle:
    summary_rows = list(csv.DictReader(handle))
if len(summary_rows) != len(variants):
    raise SystemExit(f"expected {len(variants)} summary rows, found {len(summary_rows)}")
seen = set()
summary_values = {}
for row in summary_rows:
    variant = row.get("variant")
    if row.get("scenario") != scenario or variant not in variant_set or variant in seen:
        raise SystemExit(f"unexpected/duplicate summary row: {row!r}")
    seen.add(variant)
    median_ms = positive(row, "median_total_ms")
    median_us = positive(row, "median_us_per_launch")
    speed = positive(row, "relative_speed")
    expected_median = statistics.median(values[variant])
    if not math.isclose(median_ms, expected_median, rel_tol=2e-6, abs_tol=2e-6):
        raise SystemExit(f"incorrect median for {variant}")
    if not math.isclose(median_us, 1000.0 * median_ms / launches,
                        rel_tol=2e-6, abs_tol=2e-6):
        raise SystemExit(f"incorrect median per-launch time for {variant}")
    summary_values[variant] = (median_ms, speed)
standard_median = summary_values["standard"][0]
for variant, (median_ms, speed) in summary_values.items():
    if not math.isclose(speed, standard_median / median_ms,
                        rel_tol=2e-6, abs_tol=2e-6):
        raise SystemExit(f"incorrect median speed for {variant}")
PY
    then
        die "$scenario benchmark CSV validation failed"
    fi
}

for scenario in early late; do
    current_phase=$([[ $RESUME_NCU == 1 ]] && printf 'resume-validation-benchmark-%s' "$scenario" || printf 'benchmark-%s' "$scenario")
    log=$OUTPUT_DIR/benchmark-$scenario.log
    if [[ $RESUME_NCU == 1 ]]; then
        [[ -s $log ]] || die "$scenario benchmark log is missing"
        [[ -s $OUTPUT_DIR/benchmark-$scenario-samples.csv ]] ||
            die "$scenario benchmark sample CSV is missing"
        [[ -s $OUTPUT_DIR/benchmark-$scenario-summary.csv ]] ||
            die "$scenario benchmark summary CSV is missing"
        validate_benchmark "$scenario" "$log" \
            "$RUN_DIR/benchmark-$scenario-revalidated-samples.csv" \
            "$RUN_DIR/benchmark-$scenario-revalidated-summary.csv"
        cmp -s "$OUTPUT_DIR/benchmark-$scenario-samples.csv" \
            "$RUN_DIR/benchmark-$scenario-revalidated-samples.csv" ||
            die "$scenario recorded sample CSV differs from its benchmark log"
        cmp -s "$OUTPUT_DIR/benchmark-$scenario-summary.csv" \
            "$RUN_DIR/benchmark-$scenario-revalidated-summary.csv" ||
            die "$scenario recorded summary CSV differs from its benchmark log"
        [[ -s $OUTPUT_DIR/benchmark-$scenario-telemetry.csv ]] ||
            die "$scenario benchmark telemetry is missing"
        [[ $(wc -l <"$OUTPUT_DIR/benchmark-$scenario-telemetry.csv") -ge 2 ]] ||
            die "$scenario benchmark telemetry contains no samples"
        printf 'benchmark_%s_evidence=validated\n' "$scenario" \
            >>"$RUN_DIR/resume-validation.log"
    else
        capture_gpu_state pre-benchmark-$scenario
        start_benchmark_telemetry "$scenario"
        benchmark_rc=0
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            ./tests/cuda_sm75_q4_down_native --device 0 --benchmark-only \
                --scenario "$scenario" --rounds "$BENCH_ROUNDS" \
                --launches "$BENCH_LAUNCHES" >"$log" 2>&1 || benchmark_rc=$?
        stop_benchmark_telemetry
        if (( benchmark_rc != 0 )); then
            cat "$log" >&2 || true
            die "$scenario randomized/rotated benchmark failed (exit $benchmark_rc)"
        fi
        [[ $(wc -l <"$OUTPUT_DIR/benchmark-$scenario-telemetry.csv") -ge 2 ]] ||
            die "$scenario benchmark telemetry contains no samples"
        validate_benchmark "$scenario" "$log" \
            "$OUTPUT_DIR/benchmark-$scenario-samples.csv" \
            "$OUTPUT_DIR/benchmark-$scenario-summary.csv"
        capture_gpu_state post-benchmark-$scenario
    fi
done
if [[ $RESUME_NCU == 1 ]]; then
    printf 'resume_validation_status=ok\n' >>"$RUN_DIR/resume-validation.log"
fi

if [[ $RUN_NCU == 1 ]]; then
    current_phase=nsight-preflight
    command -v ncu >/dev/null 2>&1 || die "RUN_NCU=1 but ncu was not found"
    ncu_bin=$(command -v ncu)
    [[ $ncu_bin == /* && -x $ncu_bin ]] || die "cannot resolve ncu executable: $ncu_bin"
    ncu_command=("$ncu_bin")
    if [[ $NCU_USE_SUDO == 1 ]]; then
        command -v sudo >/dev/null 2>&1 || die "sudo not found"
        sudo -v
        ncu_command=(sudo -E "$ncu_bin")
    fi

    desired_metric_names=(
        gpu__time_duration.sum
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
        lts__t_sectors_op_atom.sum
        lts__t_bytes.sum
        sm__inst_executed_pipe_ipa.sum
        sm__inst_executed_pipe_tensor.sum
        smsp__inst_executed.sum
        smsp__inst_executed_pipe_tensor.sum
        smsp__sass_thread_inst_executed_op_integer_pred_on.sum
        smsp__sass_thread_inst_executed_op_memory_pred_on.sum
    )

    # These metrics are known to be collectable on the target Turing setup and
    # are validated from the imported report below.  Do not use
    # --query-metrics as an availability oracle for them: depending on the NCU
    # version, its default collection omits tool-generated gpu__/launch__
    # metrics even though --metrics can collect them.
    required_metric_names=(
        gpu__time_duration.sum
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
    available_metrics_raw=$NCU_DIR/available-metrics.raw.txt
    available_metrics=$NCU_DIR/available-metric-names.txt
    available_metrics_log=$NCU_DIR/available-metrics-query.log
    query_args=(--config-file off --devices 0 --query-metrics)
    # Capture help before testing it.  With pipefail, `ncu --help | grep -q`
    # can report false when grep exits at the match and ncu receives SIGPIPE.
    ncu_help=$("$ncu_bin" --help 2>/dev/null || true)
    if grep -Fq -- '--query-metrics-mode' <<<"$ncu_help"; then
        query_args+=(--query-metrics-mode all)
    fi
    # Metric discovery intentionally uses the identical sudo/non-sudo command
    # as collection so restricted-counter hosts see one consistent privilege path.
    metric_query_rc=0
    env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
        "${ncu_command[@]}" "${query_args[@]}" >"$available_metrics_raw" \
        2>"$available_metrics_log" || metric_query_rc=$?
    take_output_ownership
    if (( metric_query_rc == 0 )); then
        grep -Eo '[[:alnum:]_]+__[[:alnum:]_.]+' "$available_metrics_raw" |
            sort -u >"$available_metrics" || true
    else
        : >"$available_metrics"
        printf 'optional metric discovery failed with exit %s; collecting required metrics only\n' \
            "$metric_query_rc" >>"$available_metrics_log"
    fi

    selected_metric_names=("${required_metric_names[@]}")
    optional_selected_metric_names=()
    : >"$NCU_DIR/metrics-unavailable.txt"
    for metric in "${desired_metric_names[@]}"; do
        metric_is_required=0
        for required_metric in "${required_metric_names[@]}"; do
            if [[ $metric == "$required_metric" ]]; then
                metric_is_required=1
                break
            fi
        done
        if (( metric_is_required == 1 )); then
            continue
        fi
        if grep -Fxq -- "$metric" "$available_metrics"; then
            selected_metric_names+=("$metric")
            optional_selected_metric_names+=("$metric")
        else
            printf '%s\n' "$metric" >>"$NCU_DIR/metrics-unavailable.txt"
        fi
    done
    printf '%s\n' "${desired_metric_names[@]}" >"$NCU_DIR/metrics-desired.txt"
    printf '%s\n' "${required_metric_names[@]}" >"$NCU_DIR/metrics-required.txt"
    : >"$NCU_DIR/metrics-optional-selected.txt"
    if (( ${#optional_selected_metric_names[@]} > 0 )); then
        printf '%s\n' "${optional_selected_metric_names[@]}" \
            >"$NCU_DIR/metrics-optional-selected.txt"
    fi
    printf '%s\n' "${selected_metric_names[@]}" >"$NCU_DIR/metrics-selected.txt"
    metrics=$(IFS=,; printf '%s' "${selected_metric_names[*]}")

    validate_ncu_metric_value() {
        local csv=$1 metric=$2
        python3 - "$csv" "$metric" <<'PY'
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

    current_phase=nsight
    capture_gpu_state pre-nsight
    for i in "${!profile_variants[@]}"; do
        variant=${profile_variants[$i]}
        label=${profile_labels[$i]}
        kernel=${profile_kernels[$i]}
        launch_skip=${profile_launch_skips[$i]}
        base=$NCU_DIR/$PROFILE_SCENARIO-$label
        printf 'Nsight Compute: %s %s...\n' "$PROFILE_SCENARIO" "$label"
        rc=0
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            "${ncu_command[@]}" --config-file off \
                --target-processes application-only --devices 0 \
                --kernel-name-base function --kernel-name "$kernel" \
                --launch-skip "$launch_skip" --launch-count 1 \
                --replay-mode kernel --cache-control none \
                --clock-control none --force-overwrite --export "$base" \
                --metrics "$metrics" --disable-extra-suffixes \
                ./tests/cuda_sm75_q4_down_native --device 0 \
                    --scenario "$PROFILE_SCENARIO" --profile "$variant" \
                >"$base.log" 2>&1 || rc=$?
        take_output_ownership
        if (( rc != 0 )); then
            tail -n 120 "$base.log" >&2 || true
            grep -q ERR_NVGPUCTRPERM "$base.log" &&
                printf 'error: rerun with NCU_USE_SUDO=1 for restricted counters\n' >&2 || true
            die "Nsight Compute failed for $label (exit $rc)"
        fi
        grep -Eq '==ERROR==|No kernels were profiled|Failed to (profile|create report)' \
            "$base.log" && die "Nsight Compute captured no valid kernel for $label"
        grep -Fxq 'profile_status=ok' "$base.log" ||
            die "profile harness omitted profile_status=ok for $label"
        grep -Fxq 'harness_status=ok' "$base.log" ||
            die "profile harness omitted harness_status=ok for $label"
        [[ -s $base.ncu-rep ]] || die "missing Nsight report for $label"
        "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
            >"$base.csv" 2>"$base-import.log" ||
            die "could not import Nsight report for $label"
        [[ -s $base.csv ]] || die "empty Nsight CSV for $label"
        grep -q "$kernel" "$base.csv" ||
            die "Nsight report does not contain expected kernel $kernel"
        python3 "$validator_rel" "$base.csv" "$kernel" 0 \
            --process cuda_sm75_q4_down_native \
            >"$base-validation.txt" 2>&1 || {
                cat "$base-validation.txt" >&2 || true
                die "Nsight capture identity validation failed for $label"
            }
        for metric in "${required_metric_names[@]}"; do
            validate_ncu_metric_value "$base.csv" "$metric" ||
                die "Nsight report has no value for required metric $metric ($label)"
        done
        capture_gpu_state post-nsight-$label
    done
fi

current_phase=final-telemetry
capture_gpu_state final
current_phase=complete
printf 'SM75 Q4-down native audit complete: %s\n' "$OUTPUT_DIR"
