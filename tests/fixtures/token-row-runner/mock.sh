#!/usr/bin/env bash
# CPU-only command doubles. Never forwards to a real GPU or build command.
set -euo pipefail
tool=${0##*/}
printf '%s\n' "$tool" >>"$MOCK_TRACE"
case $tool in
    make) [[ $MOCK_CASE != stale-target ]] ;;
    git)
        case ${1:-} in
            rev-parse) printf 'mock-commit\n' ;;
            branch) printf 'mock-branch\n' ;;
        esac
        ;;
    sudo) exit 1 ;; # Exercise unprivileged journal capture; never calls sudo.
    python3)
        # Test runner-to-helper wiring only. The real helper has its own Python
        # tests; this double cannot execute any OS/GPU collection commands.
        [[ $1 == */capture-sm75-gpu1-failure.py && $2 == --output &&
           $4 == --executable && $5 == ./tests/cuda_sm75_token_row_arithmetic &&
           $6 == --case-timeout ]]
        mkdir -p "$3"
        printf '{"collection":"mock-partial-evidence"}\n' >"$3/summary.json"
        [[ $MOCK_CASE != preflight ]] || exit 2
        shift 7
        if [[ ${1:-} == --runtime-trace-library ]]; then
            [[ $2 == /mock/tracer.so && $3 == --runtime-trace-sha256 &&
               $4 == 1111111111111111111111111111111111111111111111111111111111111111 ]]
            printf 'runtime-contract-option\n' >>"$MOCK_TRACE"
            shift 4
        fi
        [[ $1 == -- ]]
        shift
        MOCK_CAPTURE_MODE=1 "$@"
        ;;
    journalctl)
        if [[ $MOCK_CASE == kernel-fault ]]; then
            printf 'NVRM: Xid (PCI:0000:03:00): 79, GPU has fallen off the bus.\n'
        fi
        ;;
    nvidia-smi)
        if [[ $MOCK_CASE == health-fault && -f $MOCK_STARTED ]]; then
            printf 'GPU health unavailable\n'
            exit 1
        fi
        printf '1, 0000:03:00.0, MOCK-GPU, 1, 49151, 260\n'
        ;;
    compute-sanitizer)
        if [[ ${1:-} == --version ]]; then
            printf 'Mock Compute Sanitizer (CPU-only runner test)\n'
            exit 0
        fi
        printf 'instrumented-launch\n' >>"$MOCK_TRACE"
        [[ $1 == --tool && $2 == "$MOCK_EXPECT_TOOL" && $3 == --error-exitcode=99 ]]
        printf 'instrumentation-tool=%s\n' "$2" >>"$MOCK_TRACE"
        shift 3
        if [[ $MOCK_EXPECT_TOOL == racecheck ]]; then
            [[ $1 == --racecheck-report && $2 == analysis ]]
            shift 2
        fi
        set +e
        MOCK_INSTRUMENTED=1 "$@"
        status=$?
        set -e
        if [[ $MOCK_CASE == missing-summary ]]; then
            exit "$status"
        elif [[ $MOCK_CASE == memcheck-error ]]; then
            printf '========= ERROR SUMMARY: 1 error\n'
            exit 99
        fi
        if [[ $MOCK_EXPECT_TOOL == racecheck ]]; then
            case $MOCK_CASE in
                error-summary-only)
                    printf '========= ERROR SUMMARY: 0 errors\n'
                    exit "$status" ;;
                warning-summary)
                    printf '========= RACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)\n'
                    exit "$status" ;;
                error-summary)
                    printf '========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)\n'
                    exit "$status" ;;
                contradictory-summary)
                    printf '========= RACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)\n' ;;
                contradictory-error-summary)
                    printf '========= ERROR SUMMARY: 1 error\n' ;;
                warning-body)
                    printf '========= Warning: Race reported between Write and Read\n' ;;
                error-body)
                    printf '========= Error: Race reported between Write and Read\n' ;;
                fatal-body)
                    printf '========= FATAL: Hazard analysis incomplete\n' ;;
            esac
            if [[ $MOCK_CASE == summary-prefix ]]; then
                printf '======== RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n'
            else
                printf '========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n'
            fi
            exit "$status"
        fi
        if [[ $MOCK_CASE == contradictory-summary ]]; then
            printf '========= ERROR SUMMARY: 1 error\n'
        fi
        if [[ $MOCK_CASE == summary-prefix ]]; then
            printf '======== ERROR SUMMARY: 0 errors\n'
        else
            printf '========= ERROR SUMMARY: 0 errors\n'
        fi
        exit "$status"
        ;;
    cuda_sm75_token_row_arithmetic)
        printf 'application-launch\n' >>"$MOCK_TRACE"
        : >"$MOCK_STARTED"
        [[ ${CUDA_VISIBLE_DEVICES:-} == 1 && ${CUDA_DEVICE_ORDER:-} == PCI_BUS_ID ]]
        [[ ! -v CUDA_LAUNCH_BLOCKING && ! -v DS4_UNRELATED_TEST_EXPERIMENT ]]
        [[ ! -v DS4_TOKEN_ROW_ARITHMETIC_STOP_AFTER_Q_B ]]
        [[ ${DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_PRODUCTION103_NO_ROW_OWNED:-} == 1 ]]
        [[ ${DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_PRODUCTION103_CALLS:-} == 1024 ]]
        [[ ${DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_PRODUCTION103_BATCH:-} == 10 ]]
        if [[ $MOCK_CASE == ordinary || ${MOCK_CAPTURE_MODE:-0} == 1 ]]; then
            [[ ! -v MOCK_INSTRUMENTED && ! -v DS4_TOKEN_ROW_ARITHMETIC_SANITIZER_SMOKE ]]
        elif [[ $MOCK_CASE == legacy && ! -v MOCK_INSTRUMENTED ]]; then
            [[ ! -v DS4_TOKEN_ROW_ARITHMETIC_SANITIZER_SMOKE ]]
        elif [[ $MOCK_CASE == legacy ]]; then
            [[ ${DS4_TOKEN_ROW_ARITHMETIC_SANITIZER_SMOKE:-} == 1 ]]
            printf 'diagnostic_scope=q-b-only\nharness_status=ok\n'
            exit 0
        else
            [[ ${MOCK_INSTRUMENTED:-} == 1 && ! -v DS4_TOKEN_ROW_ARITHMETIC_SANITIZER_SMOKE ]]
        fi
        if [[ $MOCK_CASE == application-fault ]]; then
            printf 'mock application failure before final transition\n'
            exit 134
        elif [[ $MOCK_CASE == timeout ]]; then
            # Sleep only in this mock, to exercise the real bounded timeout.
            exec sleep 5
        elif [[ $MOCK_CASE == incomplete ]]; then
            sed '/^suffix_transition_phase=algo103-half1-256-complete$/d' "$MOCK_PASSED_LOG"
        else
            cat "$MOCK_PASSED_LOG"
        fi
        ;;
    *) printf 'Unexpected mock tool: %s\n' "$tool" >&2; exit 1 ;;
esac
