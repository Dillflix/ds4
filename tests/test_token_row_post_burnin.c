/* CPU-only dispatch test for the real helper used by the CUDA replay.
 * Linux: cc -O2 -std=c99 -D_GNU_SOURCE -ffunction-sections -fdata-sections
 *   -I. tests/test_token_row_post_burnin.c -Wl,--gc-sections -lm -o /tmp/test-post
 * MSVC: cl /O2 /Gy /std:c11 /I. tests/test_token_row_post_burnin.c
 *   /link /OPT:REF
 * Unused GPU diagnostic sections are discarded; no CUDA runtime is linked.
 */
#include <assert.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
static int test_setenv(const char *name, const char *value, int overwrite) {
    if (!overwrite && getenv(name)) return 0;
    return _putenv_s(name, value);
}
static int test_unsetenv(const char *name) { return _putenv_s(name, ""); }
#define setenv test_setenv
#define unsetenv test_unsetenv
#define CLOCK_MONOTONIC 0
int clock_gettime(int clock_id, struct timespec *value);
#endif

#define main static token_row_gpu_diagnostic_unused_main
#include "cuda_sm75_token_row_arithmetic.c"
#undef main

struct ds4_gpu_tensor { int tag; };
static ds4_gpu_tensor test_out[2], test_low[2], test_heads[2];
static const unsigned char test_model[16] = {0};
static char trace[16];
static unsigned trace_count, call_count, fail_call;
static int fail_sync;

static void trace_op(char op) {
    assert(trace_count + 1u < sizeof(trace));
    trace[trace_count++] = op;
    trace[trace_count] = '\0';
}

static void check_b_args(ds4_gpu_tensor *out, const ds4_gpu_tensor *low,
                         const void *model, uint64_t model_size,
                         uint64_t offset, uint32_t rows) {
    assert(call_count < 2u);
    assert(out == &test_out[call_count]);
    assert(low == &test_low[call_count]);
    assert(model == test_model && model_size == sizeof(test_model));
    assert(offset == 8u && rows == 256u);
    const char *algo = getenv("DS4_CUDA_ATTN_OUTPUT_B_F16_GEMM_ALGO_DIAGNOSTIC");
    assert(algo && strcmp(algo, "103") == 0);
}

int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, ds4_gpu_tensor *group_tmp,
        ds4_gpu_tensor *low_tmp, const void *model, uint64_t model_size,
        uint64_t a_offset, uint64_t b_offset, uint64_t group_dim,
        uint64_t rank, uint32_t groups, uint64_t out_dim,
        const ds4_gpu_tensor *heads, uint32_t rows) {
    check_b_args(out, low, model, model_size, b_offset, rows);
    assert(!group_tmp && !low_tmp && a_offset == 4u);
    assert(group_dim == 4096u && rank == 1024u && groups == 8u);
    assert(out_dim == 4096u && heads == &test_heads[call_count]);
    const char *a_only = getenv("DS4_CUDA_OUTPUT_A_CANONICAL_ONLY_LOCAL_DIAGNOSTIC");
    if (a_only) assert(strcmp(a_only, "1") == 0);
    trace_op('A');
    if (!a_only) trace_op('B');
    call_count++;
    return !fail_call || call_count != fail_call;
}

int ds4_gpu_attention_output_q8_batch_b_tensor(
        ds4_gpu_tensor *out, const void *model, uint64_t model_size,
        uint64_t b_offset, uint64_t low_dim, uint64_t out_dim,
        const ds4_gpu_tensor *low, uint32_t rows) {
    check_b_args(out, low, model, model_size, b_offset, rows);
    assert(low_dim == 8192u && out_dim == 4096u);
    assert(!getenv("DS4_CUDA_OUTPUT_A_CANONICAL_ONLY_LOCAL_DIAGNOSTIC"));
    trace_op('B');
    call_count++;
    return !fail_call || call_count != fail_call;
}

int ds4_gpu_synchronize(void) {
    assert(call_count == 2u);
    trace_op('S');
    return !fail_sync;
}

static void run_case(const char *mode, const char *expected_trace,
                     unsigned failing_call, int failing_sync, int expected_ok) {
    trace_count = call_count = 0u;
    trace[0] = '\0';
    fail_call = failing_call;
    fail_sync = failing_sync;
    const int ok = launch_post_burnin_pair(
        mode, &test_out[0], &test_out[1], &test_low[0], &test_low[1],
        &test_heads[0], &test_heads[1], test_model, sizeof(test_model), 4u, 8u);
    assert(ok == expected_ok);
    assert(strcmp(trace, expected_trace) == 0);
    assert(!getenv("DS4_CUDA_ATTN_OUTPUT_B_F16_GEMM_ALGO_DIAGNOSTIC"));
    assert(!getenv("DS4_CUDA_OUTPUT_A_CANONICAL_ONLY_LOCAL_DIAGNOSTIC"));
}

int main(void) {
    unsetenv("DS4_CUDA_ATTN_OUTPUT_B_F16_GEMM_ALGO_DIAGNOSTIC");
    unsetenv("DS4_CUDA_OUTPUT_A_CANONICAL_ONLY_LOCAL_DIAGNOSTIC");
    run_case("ab", "ABABS", 0u, 0, 1);
    run_case("a", "AAS", 0u, 0, 1);
    run_case("b", "BBS", 0u, 0, 1);
    run_case("bad", "", 0u, 0, 0);
    run_case("a", "A", 1u, 0, 0);
    run_case("a", "AA", 2u, 0, 0);
    run_case("b", "B", 1u, 0, 0);
    run_case("ab", "ABAB", 2u, 0, 0);
    run_case("a", "AAS", 0u, 1, 0);
    puts("post-burn-in dispatch tests passed (9 cases; no CUDA runtime)");
    return 0;
}
