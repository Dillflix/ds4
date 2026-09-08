#!/usr/bin/env bash
# Tests the actual wrapper using CPU-only commands and a synthetic log.
# No CUDA executable, driver utility, compiler, or elevated command is run.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fixture_dir=$repo_dir/tests/fixtures/token-row-runner
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/token-row-runner.XXXXXX")
# Retain failure evidence as well as passing fixtures; no recursive deletion.
trap 'printf "Runner test artifacts: %s\n" "$test_dir"' EXIT
mkdir -p "$test_dir/repo/speed-bench" "$test_dir/repo/tests" "$test_dir/bin"
cp "$repo_dir/speed-bench/cuda-sm75-token-row-arithmetic.sh" "$test_dir/repo/speed-bench/"
cp "$fixture_dir/mock.sh" "$test_dir/repo/tests/cuda_sm75_token_row_arithmetic"
chmod +x "$test_dir/repo/tests/cuda_sm75_token_row_arithmetic"
for tool in make git sudo journalctl nvidia-smi compute-sanitizer python3; do
    cp "$fixture_dir/mock.sh" "$test_dir/bin/$tool"
    chmod +x "$test_dir/bin/$tool"
done

run_case() {
    local name=$1 expected=$2 launches=$3 instrumented=$4
    shift 4
    local case_dir=$test_dir/$name mock_name=${name#initcheck-} status
    mock_name=${mock_name#synccheck-}
    mock_name=${mock_name#racecheck-}
    mock_name=${mock_name#capture-}
    mkdir -p "$case_dir"
    : >"$case_dir/trace"
    set +e
    env PATH="$test_dir/bin:$PATH" MOCK_CASE="$mock_name" MOCK_EXPECT_TOOL=memcheck \
        MOCK_TRACE="$case_dir/trace" MOCK_STARTED="$case_dir/started" \
        MOCK_PASSED_LOG="$fixture_dir/passed.log" \
        CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 DS4_UNRELATED_TEST_EXPERIMENT=1 \
        DS4_TOKEN_ROW_ARITHMETIC_SANITIZER_SMOKE=1 \
        DS4_TOKEN_ROW_ARITHMETIC_STOP_AFTER_Q_B=1 \
        PROFILE_GPU=1 DIAGNOSTIC_SCOPE=output-b-production103-no-row-owned \
        SANITIZER_ONLY=1 RUN_SANITIZER=1 SKIP_BUILD=1 CREATE_ARCHIVE=1 \
        SANITIZER_TOOL=memcheck \
        CAPTURE_FAILURE_CONTEXT=0 CASE_TIMEOUT_SECONDS=600 \
        NSYS_CAPTURE=0 NSYS_PREFLIGHT_ONLY=0 NSYS_QUALIFICATION_ARCHIVE= \
        RUNTIME_CONTRACT_TRACE_LIBRARY= RUNTIME_CONTRACT_TRACE_SHA256= \
        B_TIMING_ROUNDS=7 B_TIMING_REPEATS=10 B_TIMING_WARMUPS=3 \
        OUTPUT_B_PRODUCTION103_CALLS=1024 OUTPUT_B_PRODUCTION103_BATCH=10 \
        TOKEN_ROW_ARITHMETIC_DIR="$case_dir/output" "$@" \
        bash "$test_dir/repo/speed-bench/cuda-sm75-token-row-arithmetic.sh" \
        >"$case_dir/console.log" 2>&1
    status=$?
    set -e
    if [[ $status != "$expected" ||
          $(grep -cx application-launch "$case_dir/trace" || true) != "$launches" ||
          $(grep -cx instrumented-launch "$case_dir/trace" || true) != "$instrumented" ]]; then
        printf 'FAIL %s: status=%s\n' "$name" "$status" >&2
        cat "$case_dir/console.log" "$case_dir/trace" >&2
        exit 1
    fi
    if (( launches )); then
        [[ -s $case_dir/output.tar.gz && -e $case_dir/output/health/kernel.log ]]
        [[ -s $case_dir/output/health/post-gpu.csv ]]
        tar -tzf "$case_dir/output.tar.gz" | grep 'output/run-status.txt' >/dev/null
        local diagnostic_status=0
        [[ $mock_name != application-fault ]] || diagnostic_status=134
        [[ $mock_name != memcheck-error ]] || diagnostic_status=99
        [[ $mock_name != timeout ]] || diagnostic_status=124
        grep -Fxq "diagnostic_exit_status=$diagnostic_status" "$case_dir/output/run-status.txt"
    elif [[ $name == ns-integrated-preflight ]]; then
        [[ -s $case_dir/output.tar.gz ]]
        grep -Fxq 'nsys-preflight-no-launch' "$case_dir/trace"
        grep -Fxq 'last_phase=nsys-preflight-complete' "$case_dir/output/run-status.txt"
        grep -Fxq 'diagnostic_exit_status=0' "$case_dir/output/run-status.txt"
    elif [[ $mock_name == preflight ]]; then
        [[ -s $case_dir/output.tar.gz ]]
        grep -Fxq 'diagnostic_exit_status=2' "$case_dir/output/run-status.txt"
        tar -tzf "$case_dir/output.tar.gz" | grep 'output/failure-context/summary.json' >/dev/null
    elif [[ $mock_name == stale-target ]]; then
        [[ -s $case_dir/output.tar.gz ]]
        [[ $(<"$case_dir/trace") == make ]]
        grep -Fxq 'diagnostic_exit_status=not-run' "$case_dir/output/run-status.txt"
    else
        [[ ! -e $case_dir/output && ! -s $case_dir/trace ]]
    fi
    printf 'PASS %s\n' "$name"
}

run_case clean 0 1 1
grep -Fxq 'uninstrumented_runs=0' "$test_dir/clean/output/manifest.txt"
grep -Fxq 'sanitizer_smoke=0' "$test_dir/clean/output/manifest.txt"
[[ -s $test_dir/clean/output/provenance/compute-sanitizer-version.txt ]]
[[ -s $test_dir/clean/output/provenance/diagnostic-sha256.txt ]]
grep -Fq 'compute-sanitizer --tool memcheck --error-exitcode=99' \
    "$test_dir/clean/output/provenance/diagnostic-command.txt"
run_case ordinary 0 1 0 SANITIZER_ONLY=0 RUN_SANITIZER=0
run_case legacy 0 2 1 SANITIZER_ONLY=0 RUN_SANITIZER=1
run_case summary-prefix 0 1 1
run_case application-fault 1 1 1
run_case memcheck-error 1 1 1
run_case missing-summary 1 1 1
run_case contradictory-summary 1 1 1
run_case incomplete 1 1 1
run_case kernel-fault 1 1 1
run_case health-fault 1 1 1
run_case timeout 1 1 1 CASE_TIMEOUT_SECONDS=1
run_case sanitizer-disabled 1 0 0 RUN_SANITIZER=0
run_case wrong-scope 1 0 0 DIAGNOSTIC_SCOPE=full
run_case invalid-flag 1 0 0 SANITIZER_ONLY=2
run_case stale-target 1 0 0
run_case initcheck-clean 0 1 1 SANITIZER_TOOL=initcheck MOCK_EXPECT_TOOL=initcheck
grep -Fxq 'sanitizer_tool=initcheck' "$test_dir/initcheck-clean/output/manifest.txt"
grep -Fxq 'execution_mode=initcheck-full-selected-scope' "$test_dir/initcheck-clean/output/manifest.txt"
grep -Fq 'compute-sanitizer --tool initcheck --error-exitcode=99' \
    "$test_dir/initcheck-clean/output/provenance/diagnostic-command.txt"
grep -Fxq 'instrumentation-tool=initcheck' "$test_dir/initcheck-clean/trace"
for failure in application-fault memcheck-error missing-summary contradictory-summary \
    incomplete kernel-fault health-fault; do
    run_case "initcheck-$failure" 1 1 1 SANITIZER_TOOL=initcheck MOCK_EXPECT_TOOL=initcheck
done
run_case initcheck-timeout 1 1 1 SANITIZER_TOOL=initcheck MOCK_EXPECT_TOOL=initcheck CASE_TIMEOUT_SECONDS=1
run_case invalid-tool 1 0 0 SANITIZER_TOOL=unknown
run_case initcheck-ordinary-rejected 1 0 0 SANITIZER_TOOL=initcheck SANITIZER_ONLY=0
run_case initcheck-disabled 1 0 0 SANITIZER_TOOL=initcheck RUN_SANITIZER=0
run_case synccheck-clean 0 1 1 SANITIZER_TOOL=synccheck MOCK_EXPECT_TOOL=synccheck
grep -Fxq 'sanitizer_tool=synccheck' "$test_dir/synccheck-clean/output/manifest.txt"
grep -Fxq 'execution_mode=synccheck-full-selected-scope' "$test_dir/synccheck-clean/output/manifest.txt"
grep -Fxq 'uninstrumented_runs=0' "$test_dir/synccheck-clean/output/manifest.txt"
grep -Fxq 'sanitizer_smoke=0' "$test_dir/synccheck-clean/output/manifest.txt"
grep -Fq 'compute-sanitizer --tool synccheck --error-exitcode=99' \
    "$test_dir/synccheck-clean/output/provenance/diagnostic-command.txt"
grep -Fxq 'instrumentation-tool=synccheck' "$test_dir/synccheck-clean/trace"
for failure in application-fault memcheck-error missing-summary contradictory-summary \
    incomplete kernel-fault health-fault; do
    run_case "synccheck-$failure" 1 1 1 SANITIZER_TOOL=synccheck MOCK_EXPECT_TOOL=synccheck
done
run_case synccheck-timeout 1 1 1 SANITIZER_TOOL=synccheck MOCK_EXPECT_TOOL=synccheck CASE_TIMEOUT_SECONDS=1
run_case synccheck-ordinary-rejected 1 0 0 SANITIZER_TOOL=synccheck SANITIZER_ONLY=0
run_case synccheck-disabled 1 0 0 SANITIZER_TOOL=synccheck RUN_SANITIZER=0
run_case synccheck-wrong-scope 1 0 0 SANITIZER_TOOL=synccheck DIAGNOSTIC_SCOPE=full
run_case synccheck-stale-target 1 0 0 SANITIZER_TOOL=synccheck
run_case racecheck-clean 0 1 1 SANITIZER_TOOL=racecheck MOCK_EXPECT_TOOL=racecheck
grep -Fxq 'sanitizer_tool=racecheck' "$test_dir/racecheck-clean/output/manifest.txt"
grep -Fxq 'execution_mode=racecheck-full-selected-scope' "$test_dir/racecheck-clean/output/manifest.txt"
grep -Fxq 'uninstrumented_runs=0' "$test_dir/racecheck-clean/output/manifest.txt"
grep -Fxq 'sanitizer_smoke=0' "$test_dir/racecheck-clean/output/manifest.txt"
grep -Fq 'compute-sanitizer --tool racecheck --error-exitcode=99 --racecheck-report analysis' \
    "$test_dir/racecheck-clean/output/provenance/diagnostic-command.txt"
grep -Fxq 'instrumentation-tool=racecheck' "$test_dir/racecheck-clean/trace"
! grep -Fq 'ERROR SUMMARY:' "$test_dir/racecheck-clean/output/diagnostic.log"
run_case racecheck-summary-prefix 0 1 1 SANITIZER_TOOL=racecheck MOCK_EXPECT_TOOL=racecheck
for failure in application-fault memcheck-error missing-summary contradictory-summary \
    incomplete kernel-fault health-fault error-summary-only warning-summary error-summary \
    contradictory-error-summary warning-body error-body fatal-body; do
    run_case "racecheck-$failure" 1 1 1 SANITIZER_TOOL=racecheck MOCK_EXPECT_TOOL=racecheck
done
run_case racecheck-timeout 1 1 1 SANITIZER_TOOL=racecheck MOCK_EXPECT_TOOL=racecheck CASE_TIMEOUT_SECONDS=1
run_case racecheck-ordinary-rejected 1 0 0 SANITIZER_TOOL=racecheck SANITIZER_ONLY=0
run_case racecheck-disabled 1 0 0 SANITIZER_TOOL=racecheck RUN_SANITIZER=0
run_case racecheck-wrong-scope 1 0 0 SANITIZER_TOOL=racecheck DIAGNOSTIC_SCOPE=full
run_case racecheck-stale-target 1 0 0 SANITIZER_TOOL=racecheck
run_case capture-clean 0 1 0 CAPTURE_FAILURE_CONTEXT=1 SANITIZER_ONLY=0 RUN_SANITIZER=0
run_case capture-application-fault 1 1 0 CAPTURE_FAILURE_CONTEXT=1 SANITIZER_ONLY=0 RUN_SANITIZER=0
run_case capture-preflight 1 0 0 CAPTURE_FAILURE_CONTEXT=1 SANITIZER_ONLY=0 RUN_SANITIZER=0
for name in capture-clean capture-application-fault; do
    [[ $(grep -cx python3 "$test_dir/$name/trace") == 1 ]]
    [[ -s $test_dir/$name/output/provenance/diagnostic-sha256.txt ]]
    tar -tzf "$test_dir/$name/output.tar.gz" | grep 'output/failure-context/summary.json' >/dev/null
done
for setting in CAPTURE_FAILURE_CONTEXT=2 PROFILE_GPU=0 DIAGNOSTIC_SCOPE=q-b \
    RUN_SANITIZER=1 SKIP_BUILD=0 CREATE_ARCHIVE=0 OUTPUT_B_PRODUCTION103_CALLS=2048 \
    OUTPUT_B_PRODUCTION103_BATCH=20 B_TIMING_ROUNDS=8 B_TIMING_REPEATS=11 \
    B_TIMING_WARMUPS=4 CASE_TIMEOUT_SECONDS=601; do
    run_case "capture-reject-${setting%%=*}" 1 0 0 CAPTURE_FAILURE_CONTEXT=1 \
        SANITIZER_ONLY=0 RUN_SANITIZER=0 "$setting"
done
trace_hash=1111111111111111111111111111111111111111111111111111111111111111
run_case capture-runtime-clean 0 1 0 CAPTURE_FAILURE_CONTEXT=1 SANITIZER_ONLY=0 RUN_SANITIZER=0 \
    RUNTIME_CONTRACT_TRACE_LIBRARY=/mock/tracer.so RUNTIME_CONTRACT_TRACE_SHA256="$trace_hash"
grep -Fxq 'execution_mode=instrumented-runtime-contract' "$test_dir/capture-runtime-clean/output/manifest.txt"
grep -Fxq 'uninstrumented_runs=0' "$test_dir/capture-runtime-clean/output/manifest.txt"
grep -Fxq 'runtime-contract-option' "$test_dir/capture-runtime-clean/trace"
run_case trace-without-capture 1 0 0 SANITIZER_ONLY=0 RUN_SANITIZER=0 \
    RUNTIME_CONTRACT_TRACE_LIBRARY=/mock/tracer.so RUNTIME_CONTRACT_TRACE_SHA256="$trace_hash"
run_case trace-without-hash 1 0 0 CAPTURE_FAILURE_CONTEXT=1 SANITIZER_ONLY=0 RUN_SANITIZER=0 \
    RUNTIME_CONTRACT_TRACE_LIBRARY=/mock/tracer.so
run_case trace-without-library 1 0 0 CAPTURE_FAILURE_CONTEXT=1 SANITIZER_ONLY=0 RUN_SANITIZER=0 \
    RUNTIME_CONTRACT_TRACE_SHA256="$trace_hash"
run_case trace-bad-hash 1 0 0 CAPTURE_FAILURE_CONTEXT=1 SANITIZER_ONLY=0 RUN_SANITIZER=0 \
    RUNTIME_CONTRACT_TRACE_LIBRARY=/mock/tracer.so RUNTIME_CONTRACT_TRACE_SHA256=invalid
run_case capture-nsys-clean 0 1 0 CAPTURE_FAILURE_CONTEXT=1 SANITIZER_ONLY=0 RUN_SANITIZER=0 \
    NSYS_CAPTURE=1 NSYS_QUALIFICATION_ARCHIVE=/mock/qualified.tar.gz
grep -Fxq 'execution_mode=instrumented-nsys-frozen-gpu1' "$test_dir/capture-nsys-clean/output/manifest.txt"
grep -Fxq 'nsys-capture-option' "$test_dir/capture-nsys-clean/trace"
run_case nsys-without-capture 1 0 0 SANITIZER_ONLY=0 RUN_SANITIZER=0 \
    NSYS_CAPTURE=1 NSYS_QUALIFICATION_ARCHIVE=/mock/qualified.tar.gz
run_case nsys-without-receipt 1 0 0 CAPTURE_FAILURE_CONTEXT=1 SANITIZER_ONLY=0 RUN_SANITIZER=0 NSYS_CAPTURE=1
run_case nsys-mixed-trace 1 0 0 CAPTURE_FAILURE_CONTEXT=1 SANITIZER_ONLY=0 RUN_SANITIZER=0 \
    NSYS_CAPTURE=1 NSYS_QUALIFICATION_ARCHIVE=/mock/qualified.tar.gz \
    RUNTIME_CONTRACT_TRACE_LIBRARY=/mock/tracer.so RUNTIME_CONTRACT_TRACE_SHA256="$trace_hash"
run_case nsys-orphan-receipt 1 0 0 NSYS_QUALIFICATION_ARCHIVE=/mock/qualified.tar.gz
run_case nsys-bad-flag 1 0 0 NSYS_CAPTURE=2
run_case ns-integrated-preflight 0 0 0 CAPTURE_FAILURE_CONTEXT=1 SANITIZER_ONLY=0 RUN_SANITIZER=0 \
    NSYS_CAPTURE=1 NSYS_PREFLIGHT_ONLY=1 NSYS_QUALIFICATION_ARCHIVE=/mock/qualified.tar.gz
run_case nsys-preflight-orphan 1 0 0 NSYS_PREFLIGHT_ONLY=1
run_case nsys-preflight-bad-flag 1 0 0 NSYS_PREFLIGHT_ONLY=2
printf 'All 91 CPU-only runner cases passed. No GPU validation performed.\n'
