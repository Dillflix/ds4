#!/usr/bin/env bash
# HOST-ONLY: builds an interposer and exercises a fake provider, never CUDA/GPU.
set -euo pipefail
umask 077
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
trace_cuda_root=${TRACE_CUDA_ROOT:-/usr/local/cuda}
trace_include="$trace_cuda_root/include"
compiler=${CXX:-g++}
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { printf '%s\n' 'error: Linux x86-64 required' >&2; exit 2; }
for program in "$compiler" python3 readelf sha256sum mktemp tar timeout; do command -v "$program" >/dev/null; done
[[ -r "$trace_include/cuda_runtime_api.h" && -r "$trace_include/cublas_v2.h" ]] || { printf '%s\n' 'error: installed CUDA13.2/cuBLAS headers required' >&2; exit 2; }
[[ -z ${LD_PRELOAD:-} && -z ${LD_AUDIT:-} ]] || { printf '%s\n' 'error: start host check without inherited preload/audit libraries' >&2; exit 2; }
output=$(mktemp -d "$PWD/sm75-runtime-contract-build.XXXXXX")
printf 'Host-only build directory: %s\n' "$output"
finish() {
    build_status=$?
    trap - EXIT
    printf 'build_exit_status=%s\ngpu_workload_executed=0\nexcluded_test_fixture=symlink-log.jsonl (intentional rejected symlink)\n' "$build_status" > "$output/status.txt"
    # Preserve failed qualifications too; no delete/cleanup of original evidence.
    if tar -C "$(dirname "$output")" --exclude="$(basename "$output")/symlink-log.jsonl" \
        -czf "$output.tar.gz" "$(basename "$output")"; then
        printf 'Archive to return: %s.tar.gz\n' "$output"
    else
        printf 'error: archive creation failed; preserved directory: %s\n' "$output" >&2
        [[ $build_status != 0 ]] || build_status=2
    fi
    exit "$build_status"
}
trap finish EXIT
mkdir "$output/source"
cp -- "$repo/speed-bench/sm75-runtime-contract-trace.cpp" "$repo/speed-bench/check-sm75-runtime-trace-headers.py" \
    "$repo/speed-bench/analyze-sm75-runtime-contract.py" \
    "$repo/speed-bench/build-sm75-runtime-contract-trace.sh" "$repo/tests/sm75-runtime-contract-mock.cpp" \
    "$repo/tests/sm75-runtime-contract-mock-main.cpp" "$repo/tests/verify_sm75_runtime_contract_mock.py" "$output/source/"
sha256sum "$output/source/"* > "$output/source.sha256"
python3 -I "$output/source/check-sm75-runtime-trace-headers.py" --include "$trace_include" > "$output/header-check.json" 2> "$output/header-check.log"
"$compiler" --version > "$output/compiler.txt" 2>&1
flags=(-std=c++17 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations -fPIC -I "$trace_include")
"$compiler" "${flags[@]}" -fvisibility=hidden -fvisibility-inlines-hidden -shared -Wl,-z,defs,-z,relro,-z,now "$output/source/sm75-runtime-contract-trace.cpp" \
    -ldl -pthread -o "$output/libsm75-runtime-contract-trace.so" > "$output/build.log" 2>&1 || { cat "$output/build.log" >&2; exit 2; }
"$compiler" "${flags[@]}" -shared -Wl,-soname,libsm75-runtime-contract-mock.so "$output/source/sm75-runtime-contract-mock.cpp" \
    -o "$output/libsm75-runtime-contract-mock.so" >> "$output/build.log" 2>&1
"$compiler" "${flags[@]}" "$output/source/sm75-runtime-contract-mock-main.cpp" -L "$output" \
    -lsm75-runtime-contract-mock -Wl,-rpath,'$ORIGIN' -o "$output/host-mock" >> "$output/build.log" 2>&1
readelf -d "$output/libsm75-runtime-contract-trace.so" > "$output/tracer-dynamic.txt"
readelf -d "$output/libsm75-runtime-contract-mock.so" > "$output/mock-dynamic.txt"
readelf -d "$output/host-mock" > "$output/mock-main-dynamic.txt"
readelf --dyn-syms --wide "$output/libsm75-runtime-contract-trace.so" > "$output/tracer-exports.txt"
for api in ds4_runtime_trace_abi_version cudaMalloc cudaFree cudaDeviceSynchronize cublasGemmEx cublasGemmStridedBatchedEx \
    __cudaRegisterFunction __cudaUnregisterFatBinary __cudaGetKernel __cudaLaunchKernel __cudaPushCallConfiguration __cudaPopCallConfiguration; do
    grep -Eq "GLOBAL[[:space:]]+DEFAULT[[:space:]]+[0-9]+[[:space:]]+$api$" "$output/tracer-exports.txt" || {
        printf 'error: missing visible interposer export %s\n' "$api" >&2; exit 2;
    }
done
# Fail before mock execution if any real CUDA library was linked accidentally.
if grep -E 'NEEDED.*\[(libcuda[.]|libcudart[.]|libcublas|libnvidia)' "$output/"*dynamic.txt; then
    printf '%s\n' 'error: unexpected real GPU-library dependency' >&2; exit 2
fi
timeout --signal=TERM --kill-after=2 30 env DS4_RUNTIME_TRACE_LOG="$output/load-only.jsonl" LD_PRELOAD="$output/libsm75-runtime-contract-trace.so" \
    /usr/bin/true > "$output/load-only.log" 2>&1
timeout --signal=TERM --kill-after=2 30 env DS4_RUNTIME_TRACE_LOG="$output/abi-only.jsonl" python3 -I -c \
    'import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); assert lib.ds4_runtime_trace_abi_version()==1' \
    "$output/libsm75-runtime-contract-trace.so" > "$output/abi-only.log" 2>&1
# Validate fail-closed path handling with a CPU process, never the reproducer.
if timeout --signal=TERM --kill-after=2 30 env DS4_RUNTIME_TRACE_LOG="$output/load-only.jsonl" LD_PRELOAD="$output/libsm75-runtime-contract-trace.so" /usr/bin/true > "$output/reject-existing.log" 2>&1; then
    printf '%s\n' 'error: trace unexpectedly reused an existing log' >&2; exit 2
else
    [[ $? == 125 ]] || exit 2
fi
ln -s "$output/load-only.jsonl" "$output/symlink-log.jsonl"
if timeout --signal=TERM --kill-after=2 30 env DS4_RUNTIME_TRACE_LOG="$output/symlink-log.jsonl" LD_PRELOAD="$output/libsm75-runtime-contract-trace.so" /usr/bin/true > "$output/reject-symlink.log" 2>&1; then
    printf '%s\n' 'error: trace unexpectedly accepted a symlink log' >&2; exit 2
else
    [[ $? == 125 ]] || exit 2
fi
timeout --signal=TERM --kill-after=2 30 env DS4_RUNTIME_TRACE_LOG="$output/mock.jsonl" LD_PRELOAD="$output/libsm75-runtime-contract-trace.so" \
    "$output/host-mock" > "$output/mock.log" 2>&1
if timeout --signal=TERM --kill-after=2 30 env DS4_TRACE_MOCK_FAIL_POINTER_QUERY=1 DS4_RUNTIME_TRACE_LOG="$output/mock-query-failure.jsonl" \
    LD_PRELOAD="$output/libsm75-runtime-contract-trace.so" "$output/host-mock" > "$output/mock-query-failure.log" 2>&1; then
    printf '%s\n' 'error: failed metadata query did not stop before GEMM' >&2; exit 2
else
    [[ $? == 125 ]] || exit 2
fi
python3 -I "$output/source/analyze-sm75-runtime-contract.py" "$output/mock.jsonl" > "$output/mock-analysis.json" 2> "$output/mock-analysis.log"
python3 -I "$output/source/verify_sm75_runtime_contract_mock.py" "$output" > "$output/mock-verification.json" 2> "$output/mock-verification.log"
sha256sum "$output/libsm75-runtime-contract-trace.so" > "$output/tracer.sha256"
printf '%s\n' 'host_only_build_and_mock=passed' 'gpu_workload_executed=0' \
    "Trace library: $output/libsm75-runtime-contract-trace.so" \
    "Evidence directory: $output"
