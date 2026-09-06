#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Bounded single-GPU diagnostic for the arithmetic shape change made by the
 * token-row prototype.  Production dispatch is not modified: every boundary
 * is invoked explicitly once as N=512 and once as two ordered N=256 views. */
#define N_TOK 512u
#define HALF_TOK (N_TOK / 2u)
#define IN_DIM 1024u
#define N_HEAD 64u
#define HEAD_DIM 512u
#define Q_DIM ((uint64_t)N_HEAD * HEAD_DIM)
#define N_ROT 64u
#define N_GROUP 8u
#define GROUP_DIM 4096u
#define RANK 1024u
#define LOW_DIM ((uint64_t)N_GROUP * RANK)
#define OUT_DIM 4096u
#define N_COMP 128u
#define ATTN_WINDOW 128u
#define ATTN_RATIO 4u
#define POS0 0u
#define ORIG_CTX 65536u
#define ROPE_BASE 160000.0f
#define ROPE_SCALE 0.0625f
#define ROPE_EXT 1.0f
#define BETA_FAST 32.0f
#define BETA_SLOW 1.0f
#define EPSILON 1.0e-6f

static const int b_algorithms[] = {
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    99,
    100, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115,
};

typedef struct {
    uint64_t mismatches;
    uint64_t first;
    double max_abs;
    double rmse;
} diff_metrics;

static uint16_t float_to_half_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16u) & 0x8000u;
    const uint32_t exponent = (bits >> 23u) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu)
        return (uint16_t)(sign | (mantissa ? 0x7e00u : 0x7c00u));
    int32_t half_exponent = (int32_t)exponent - 127 + 15;
    if (half_exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return (uint16_t)sign;
        mantissa |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - half_exponent);
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((UINT32_C(1) << shift) - 1u);
        const uint32_t halfway = UINT32_C(1) << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u)))
            rounded++;
        return (uint16_t)(sign | rounded);
    }
    uint32_t rounded = mantissa >> 13u;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        rounded++;
        if (rounded == 0x400u) {
            rounded = 0u;
            half_exponent++;
            if (half_exponent >= 31)
                return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)half_exponent << 10u) | rounded);
}

static void build_q8_rows(unsigned char *dst, uint64_t rows,
                          uint64_t columns, uint64_t seed) {
    const uint64_t blocks = columns / 32u;
    for (uint64_t row = 0u; row < rows; row++) {
        for (uint64_t block = 0u; block < blocks; block++) {
            unsigned char *packed = dst + (row * blocks + block) * 34u;
            const float scale =
                (float)(1u + ((row * 11u + block * 7u + seed) % 13u)) /
                256.0f;
            const uint16_t scale_bits = float_to_half_bits(scale);
            packed[0] = (unsigned char)(scale_bits & 0xffu);
            packed[1] = (unsigned char)(scale_bits >> 8u);
            for (uint64_t lane = 0u; lane < 32u; lane++) {
                const uint64_t column = block * 32u + lane;
                const int value = (int)((row * 19u + column * 23u +
                    block * 29u + (row >> 3u) * 5u + seed) % 127u) - 63;
                packed[2u + lane] = (unsigned char)(int8_t)value;
            }
        }
    }
}

static diff_metrics compare_f32(const float *reference,
                                const float *candidate, uint64_t count) {
    diff_metrics result = {0u, UINT64_MAX, 0.0, 0.0};
    double squared = 0.0;
    for (uint64_t i = 0u; i < count; i++) {
        uint32_t a, b;
        memcpy(&a, reference + i, sizeof(a));
        memcpy(&b, candidate + i, sizeof(b));
        if (a != b) {
            if (result.first == UINT64_MAX) result.first = i;
            result.mismatches++;
        }
        const double delta = (double)reference[i] - (double)candidate[i];
        const double abs_delta = fabs(delta);
        if (abs_delta > result.max_abs) result.max_abs = abs_delta;
        squared += delta * delta;
    }
    result.rmse = count ? sqrt(squared / (double)count) : 0.0;
    return result;
}

static diff_metrics compare_u16(const uint16_t *reference,
                                const uint16_t *candidate, uint64_t count) {
    diff_metrics result = {0u, UINT64_MAX, 0.0, 0.0};
    for (uint64_t i = 0u; i < count; i++) {
        if (reference[i] != candidate[i]) {
            if (result.first == UINT64_MAX) result.first = i;
            result.mismatches++;
        }
    }
    return result;
}

static void report_diff(const char *boundary, const char *type,
                        uint64_t count, diff_metrics diff) {
    printf("boundary=%s,type=%s,values=%llu,status=%s,bit_mismatches=%llu,"
           "first_index=%lld,max_abs=%.9g,rmse=%.9g\n",
           boundary, type, (unsigned long long)count,
           diff.mismatches ? "different" : "exact",
           (unsigned long long)diff.mismatches,
           diff.first == UINT64_MAX ? -1ll : (long long)diff.first,
           diff.max_abs, diff.rmse);
}

static float rope_attention_factor(void) {
    return 1.0f / (1.0f + 0.1f * logf(1.0f / ROPE_SCALE));
}

static int launch_q_b(ds4_gpu_tensor *out, ds4_gpu_tensor *half,
                      const unsigned char *model, uint64_t model_bytes,
                      uint64_t q_b_offset, const ds4_gpu_tensor *input,
                      uint32_t n_tokens, uint32_t pos0) {
    return ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        out, half, model, model_bytes, q_b_offset, IN_DIM, Q_DIM, input,
        n_tokens, N_HEAD, HEAD_DIM, N_ROT, pos0, ORIG_CTX, false,
        ROPE_BASE, ROPE_SCALE, ROPE_EXT, rope_attention_factor(),
        BETA_FAST, BETA_SLOW, EPSILON);
}

static int launch_output(ds4_gpu_tensor *out, ds4_gpu_tensor *low,
                         const unsigned char *model, uint64_t model_bytes,
                         uint64_t out_a_offset, uint64_t out_b_offset,
                         const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    return ds4_gpu_attention_output_q8_batch_tensor(
        out, low, NULL, NULL, model, model_bytes, out_a_offset, out_b_offset,
        GROUP_DIM, RANK, N_GROUP, OUT_DIM, heads, n_tokens);
}

static int compare_double(const void *a, const void *b) {
    const double lhs = *(const double *)a;
    const double rhs = *(const double *)b;
    return (lhs > rhs) - (lhs < rhs);
}

static double monotonic_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return -1.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static uint32_t positive_env_u32(const char *name, uint32_t fallback,
                                 uint32_t maximum) {
    const char *text = getenv(name);
    if (!text || !text[0]) return fallback;
    char *end = NULL;
    const unsigned long parsed = strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0ul || parsed > maximum) {
        fprintf(stderr, "error: %s must be an integer in [1,%u]\n",
                name, maximum);
        return 0u;
    }
    return (uint32_t)parsed;
}

static void select_b_algorithm(int algorithm) {
    if (algorithm < 0) {
        (void)unsetenv("DS4_CUDA_ATTN_OUTPUT_B_F16_GEMM_ALGO_DIAGNOSTIC");
        return;
    }
    char text[32];
    snprintf(text, sizeof(text), "%d", algorithm);
    (void)setenv("DS4_CUDA_ATTN_OUTPUT_B_F16_GEMM_ALGO_DIAGNOSTIC", text, 1);
}

static int time_output_b(double *median_ms, ds4_gpu_tensor *out,
                         const unsigned char *model, uint64_t model_bytes,
                         uint64_t out_b_offset, const ds4_gpu_tensor *low,
                         uint32_t n_tokens, int algorithm, uint32_t rounds,
                         uint32_t repeats, uint32_t warmups) {
    double *samples = (double *)malloc((size_t)rounds * sizeof(*samples));
    if (!samples) return 0;
    select_b_algorithm(algorithm);
    for (uint32_t i = 0u; i < warmups; i++) {
        if (!ds4_gpu_attention_output_q8_batch_b_tensor(
                out, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
                low, n_tokens)) {
            free(samples);
            return 0;
        }
    }
    if (!ds4_gpu_synchronize()) {
        free(samples);
        return 0;
    }
    for (uint32_t round = 0u; round < rounds; round++) {
        const double begin = monotonic_seconds();
        if (begin < 0.0) {
            free(samples);
            return 0;
        }
        for (uint32_t repeat = 0u; repeat < repeats; repeat++) {
            if (!ds4_gpu_attention_output_q8_batch_b_tensor(
                    out, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
                    low, n_tokens)) {
                free(samples);
                return 0;
            }
        }
        if (!ds4_gpu_synchronize()) {
            free(samples);
            return 0;
        }
        const double end = monotonic_seconds();
        if (end < begin) {
            free(samples);
            return 0;
        }
        samples[round] = (end - begin) * 1000.0 / (double)repeats;
    }
    qsort(samples, rounds, sizeof(*samples), compare_double);
    *median_ms = (rounds & 1u)
        ? samples[rounds / 2u]
        : 0.5 * (samples[rounds / 2u - 1u] + samples[rounds / 2u]);
    free(samples);
    return 1;
}

int main(void) {
    const uint64_t q_b_bytes = Q_DIM * (IN_DIM / 32u) * 34u;
    const uint64_t sinks_offset = q_b_bytes;
    const uint64_t sinks_bytes = N_HEAD * sizeof(float);
    const uint64_t out_a_offset = sinks_offset + sinks_bytes;
    const uint64_t out_a_bytes = LOW_DIM * (GROUP_DIM / 32u) * 34u;
    const uint64_t out_b_offset = out_a_offset + out_a_bytes;
    const uint64_t out_b_bytes = OUT_DIM * (LOW_DIM / 32u) * 34u;
    const uint64_t model_bytes = out_b_offset + out_b_bytes;
    const uint64_t input_count = (uint64_t)N_TOK * IN_DIM;
    const uint64_t q_count = (uint64_t)N_TOK * Q_DIM;
    const uint64_t heads_count = q_count;
    const uint64_t low_count = (uint64_t)N_TOK * LOW_DIM;
    const uint64_t out_count = (uint64_t)N_TOK * OUT_DIM;
    const uint64_t raw_count = (uint64_t)N_TOK * HEAD_DIM;
    const uint64_t comp_count = (uint64_t)N_COMP * HEAD_DIM;
    const uint64_t input_bytes = input_count * sizeof(float);
    const uint64_t q_bytes = q_count * sizeof(float);
    const uint64_t q_half_bytes = q_count * sizeof(uint16_t);
    const uint64_t heads_bytes = heads_count * sizeof(float);
    const uint64_t low_bytes = low_count * sizeof(float);
    const uint64_t out_bytes = out_count * sizeof(float);
    const uint64_t input_half_bytes = input_bytes / 2u;
    const uint64_t q_row_half_bytes = q_bytes / 2u;
    const uint64_t qh_row_half_bytes = q_half_bytes / 2u;
    const uint64_t heads_row_half_bytes = heads_bytes / 2u;
    const uint64_t low_row_half_bytes = low_bytes / 2u;
    const uint64_t out_row_half_bytes = out_bytes / 2u;

    unsigned char *model = NULL;
    float *input_host = NULL, *reference = NULL, *candidate = NULL;
    float *actual_low_host = NULL;
    float *combined_full_host = NULL, *combined_split_host = NULL;
    float *shipping_b_host = NULL;
    float *raw_host = NULL, *comp_host = NULL;
    uint16_t *reference_half = NULL, *candidate_half = NULL;
    ds4_gpu_tensor *input = NULL, *q_full = NULL, *q_split = NULL;
    ds4_gpu_tensor *qh_full = NULL, *qh_split = NULL;
    ds4_gpu_tensor *heads_full = NULL, *heads_split = NULL;
    ds4_gpu_tensor *low_full = NULL, *low_split = NULL;
    ds4_gpu_tensor *out_full = NULL, *out_split = NULL;
    ds4_gpu_tensor *raw = NULL, *comp = NULL;
    ds4_gpu_tensor *input0 = NULL, *input1 = NULL;
    ds4_gpu_tensor *q0 = NULL, *q1 = NULL, *qh0 = NULL, *qh1 = NULL;
    ds4_gpu_tensor *q_ref0 = NULL, *q_ref1 = NULL;
    ds4_gpu_tensor *heads0 = NULL, *heads1 = NULL;
    ds4_gpu_tensor *heads_ref0 = NULL, *heads_ref1 = NULL;
    ds4_gpu_tensor *low0 = NULL, *low1 = NULL, *out0 = NULL, *out1 = NULL;
    ds4_gpu_tensor *low_ref0 = NULL, *low_ref1 = NULL;
    int initialized = 0;
    int status = 1;

    model = (unsigned char *)malloc((size_t)model_bytes);
    input_host = (float *)malloc((size_t)input_bytes);
    reference = (float *)malloc((size_t)q_bytes);
    candidate = (float *)malloc((size_t)q_bytes);
    reference_half = (uint16_t *)malloc((size_t)q_half_bytes);
    candidate_half = (uint16_t *)malloc((size_t)q_half_bytes);
    raw_host = (float *)malloc((size_t)raw_count * sizeof(float));
    comp_host = (float *)malloc((size_t)comp_count * sizeof(float));
    if (!model || !input_host || !reference || !candidate ||
        !reference_half || !candidate_half || !raw_host || !comp_host) {
        fprintf(stderr, "error: host allocation failed\n");
        goto cleanup;
    }
    /* q_count is four times low_count, leaving room in the comparison slabs
     * for one real A output and one combined-output snapshot per arm. */
    actual_low_host = reference + low_count;
    combined_full_host = reference + 2u * low_count;
    combined_split_host = candidate + 2u * low_count;
    shipping_b_host = reference + 2u * low_count + out_count;

    build_q8_rows(model, Q_DIM, IN_DIM, 17u);
    for (uint32_t h = 0u; h < N_HEAD; h++) {
        const float sink = (float)((int)(h % 11u) - 5) / 32.0f;
        memcpy(model + sinks_offset + (uint64_t)h * sizeof(float),
               &sink, sizeof(sink));
    }
    build_q8_rows(model + out_a_offset, LOW_DIM, GROUP_DIM, 37u);
    build_q8_rows(model + out_b_offset, OUT_DIM, LOW_DIM, 53u);
    for (uint64_t i = 0u; i < input_count; i++) {
        const int value = (int)((i * 29u + (i >> 5u) * 17u +
            (i / IN_DIM) * 7u + 23u) % 257u) - 128;
        input_host[i] = (float)value / 128.0f;
    }
    for (uint64_t i = 0u; i < raw_count; i++) {
        const int value = (int)((i * 13u + (i >> 5u) * 17u + 37u) % 257u) - 128;
        raw_host[i] = (float)value / 2048.0f;
    }
    for (uint64_t i = 0u; i < comp_count; i++) {
        const int value = (int)((i * 31u + (i >> 4u) * 7u + 41u) % 263u) - 131;
        comp_host[i] = (float)value / 2048.0f;
    }

    (void)setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    (void)setenv("DS4_CUDA_Q8_F16_CACHE_MB", "512", 1);
    (void)setenv("DS4_CUDA_Q8_F16_CACHE_RESERVE_MB", "1", 1);
    (void)setenv("DS4_CUDA_NO_Q8_F32_CACHE", "1", 1);
    (void)setenv("DS4_CUDA_ATTENTION_OUTPUT_PRELOAD", "1", 1);
    (void)setenv("DS4_CUDA_NO_TF32", "1", 1);
    (void)setenv("DS4_CUDA_T32_F16_FUSED", "1", 1);
    (void)unsetenv("DS4_CUDA_NO_T32_F16_FUSED");
    (void)unsetenv("DS4_CUDA_NO_Q8_F16_CACHE");
    (void)unsetenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE");
    (void)unsetenv("DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A");
    (void)unsetenv("DS4_CUDA_NO_CUBLAS_ATTENTION");
    (void)unsetenv("DS4_CUDA_NO_WINDOW_ATTENTION");
    (void)unsetenv("DS4_CUDA_T32_F16_GEMM_ALGO_DIAGNOSTIC");
    (void)unsetenv("DS4_CUDA_ATTN_OUTPUT_B_F16_GEMM_ALGO_DIAGNOSTIC");

    if (!ds4_gpu_init()) {
        fprintf(stderr, "error: CUDA initialization failed\n");
        goto cleanup;
    }
    initialized = 1;
    if (!ds4_gpu_set_model_map(model, model_bytes) ||
        !ds4_gpu_cache_q8_f16_range_on_device(
            model, model_bytes, 0u, q_b_bytes, IN_DIM, Q_DIM, 0,
            "attn_q_b") ||
        !ds4_gpu_cache_q8_f16_range_on_device(
            model, model_bytes, out_a_offset, out_a_bytes,
            GROUP_DIM, LOW_DIM, 0, "attn_output_a") ||
        !ds4_gpu_cache_q8_f16_range_on_device(
            model, model_bytes, out_b_offset, out_b_bytes,
            LOW_DIM, OUT_DIM, 0, "attn_output_b")) {
        fprintf(stderr, "error: model/cache installation failed\n");
        goto cleanup;
    }

    input = ds4_gpu_tensor_alloc(input_bytes);
    q_full = ds4_gpu_tensor_alloc(q_bytes);
    q_split = ds4_gpu_tensor_alloc(q_bytes);
    qh_full = ds4_gpu_tensor_alloc(q_half_bytes);
    qh_split = ds4_gpu_tensor_alloc(q_half_bytes);
    heads_full = ds4_gpu_tensor_alloc(heads_bytes);
    heads_split = ds4_gpu_tensor_alloc(heads_bytes);
    low_full = ds4_gpu_tensor_alloc(low_bytes);
    low_split = ds4_gpu_tensor_alloc(low_bytes);
    out_full = ds4_gpu_tensor_alloc(out_bytes);
    out_split = ds4_gpu_tensor_alloc(out_bytes);
    raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    if (!input || !q_full || !q_split || !qh_full || !qh_split ||
        !heads_full || !heads_split || !low_full || !low_split ||
        !out_full || !out_split || !raw || !comp ||
        !ds4_gpu_tensor_write(input, 0u, input_host, input_bytes) ||
        !ds4_gpu_tensor_write(raw, 0u, raw_host,
                              raw_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(comp, 0u, comp_host,
                              comp_count * sizeof(float))) {
        fprintf(stderr, "error: device allocation/setup failed\n");
        goto cleanup;
    }

    input0 = ds4_gpu_tensor_view(input, 0u, input_half_bytes);
    input1 = ds4_gpu_tensor_view(input, input_half_bytes, input_half_bytes);
    q0 = ds4_gpu_tensor_view(q_split, 0u, q_row_half_bytes);
    q1 = ds4_gpu_tensor_view(q_split, q_row_half_bytes, q_row_half_bytes);
    qh0 = ds4_gpu_tensor_view(qh_split, 0u, qh_row_half_bytes);
    qh1 = ds4_gpu_tensor_view(qh_split, qh_row_half_bytes, qh_row_half_bytes);
    q_ref0 = ds4_gpu_tensor_view(q_full, 0u, q_row_half_bytes);
    q_ref1 = ds4_gpu_tensor_view(q_full, q_row_half_bytes, q_row_half_bytes);
    heads0 = ds4_gpu_tensor_view(heads_split, 0u, heads_row_half_bytes);
    heads1 = ds4_gpu_tensor_view(heads_split, heads_row_half_bytes,
                                 heads_row_half_bytes);
    heads_ref0 = ds4_gpu_tensor_view(heads_full, 0u, heads_row_half_bytes);
    heads_ref1 = ds4_gpu_tensor_view(heads_full, heads_row_half_bytes,
                                     heads_row_half_bytes);
    low0 = ds4_gpu_tensor_view(low_split, 0u, low_row_half_bytes);
    low1 = ds4_gpu_tensor_view(low_split, low_row_half_bytes,
                               low_row_half_bytes);
    out0 = ds4_gpu_tensor_view(out_split, 0u, out_row_half_bytes);
    out1 = ds4_gpu_tensor_view(out_split, out_row_half_bytes,
                               out_row_half_bytes);
    low_ref0 = ds4_gpu_tensor_view(low_full, 0u, low_row_half_bytes);
    low_ref1 = ds4_gpu_tensor_view(low_full, low_row_half_bytes,
                                   low_row_half_bytes);
    if (!input0 || !input1 || !q0 || !q1 || !qh0 || !qh1 || !q_ref0 ||
        !q_ref1 || !heads0 ||
        !heads1 || !heads_ref0 || !heads_ref1 || !low0 || !low1 || !out0 ||
        !out1 || !low_ref0 || !low_ref1) {
        fprintf(stderr, "error: tensor-view construction failed\n");
        goto cleanup;
    }

    printf("scenario=sm75-token-row-arithmetic-full512-vs-row256x2\n"
           "scope=bounded-single-gpu-production-shapes\n"
           "production_default_changed=0\n"
           "n_tokens=%u\nhalf_tokens=%u\nq_dim=%llu\nlow_dim=%llu\n"
           "output_dim=%u\nf16_cache_bytes=%llu\n",
           N_TOK, HALF_TOK, (unsigned long long)Q_DIM,
           (unsigned long long)LOW_DIM, OUT_DIM,
           (unsigned long long)(2u * (Q_DIM * IN_DIM +
               LOW_DIM * GROUP_DIM + OUT_DIM * LOW_DIM)));

    /* q_b: identical rows and weights, only the cuBLAS N dimension and
     * positional row origin differ. */
    if (!launch_q_b(q_full, qh_full, model, model_bytes, 0u, input,
                    N_TOK, POS0) ||
        !launch_q_b(q0, qh0, model, model_bytes, 0u, input0,
                    HALF_TOK, POS0) ||
        !launch_q_b(q1, qh1, model, model_bytes, 0u, input1,
                    HALF_TOK, POS0 + HALF_TOK) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(qh_full, 0u, reference_half, q_half_bytes) ||
        !ds4_gpu_tensor_read(qh_split, 0u, candidate_half, q_half_bytes) ||
        !ds4_gpu_tensor_read(q_full, 0u, reference, q_bytes) ||
        !ds4_gpu_tensor_read(q_split, 0u, candidate, q_bytes)) {
        fprintf(stderr, "error: q_b boundary runtime failed\n");
        goto cleanup;
    }
    const diff_metrics qh_diff = compare_u16(
        reference_half, candidate_half, q_count);
    const diff_metrics q_diff = compare_f32(reference, candidate, q_count);
    report_diff("q-b-f16-projection-full512-vs-row256x2", "f16", q_count,
                qh_diff);
    report_diff("q-b-rms-rope-full512-vs-row256x2", "f32", q_count, q_diff);

    /* Match the current internal token-row candidate as well: control supplies
     * its persistent q_half, while each candidate call asks the helper to use
     * its internal temporary q_half. */
    if (!launch_q_b(q0, NULL, model, model_bytes, 0u, input0,
                    HALF_TOK, POS0) ||
        !launch_q_b(q1, NULL, model, model_bytes, 0u, input1,
                    HALF_TOK, POS0 + HALF_TOK) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(q_split, 0u, candidate, q_bytes)) {
        fprintf(stderr, "error: internal token-row q_b scratch runtime failed\n");
        goto cleanup;
    }
    const diff_metrics q_scratch_diff = compare_f32(
        reference, candidate, q_count);
    report_diff("q-b-control-persistent-vs-internal-row-scratch256x2", "f32",
                q_count, q_scratch_diff);
    if (getenv("DS4_TOKEN_ROW_ARITHMETIC_STOP_AFTER_Q_B") ||
        getenv("DS4_TOKEN_ROW_ARITHMETIC_SANITIZER_SMOKE")) {
        printf("diagnostic_scope=q-b-only\n");
        printf("boundary=static-mixed-attention-shipping-vs-full-range512,status=skipped-by-q-b-scope\n");
        printf("boundary=static-mixed-attention-full-range512-vs-row256x2,status=skipped-by-q-b-scope\n");
        printf("boundary=inverse-rope-full512-vs-row256x2,status=skipped-by-q-b-scope\n");
        printf("boundary=output-a-full512-vs-row256x2,status=skipped-by-q-b-scope\n");
        printf("boundary=output-a-plus-b-full512-vs-row256x2,status=skipped-by-q-b-scope\n");
        printf("boundary=output-b-from-identical-actual-a-low-full512-vs-row256x2,status=skipped-by-q-b-scope\n");
        printf("boundary=output-a-to-b-chain-reference,status=skipped-by-q-b-scope\n");
        printf("boundary=output-b-structured-control-full512-vs-row256x2,status=skipped-by-q-b-scope\n");
        printf("diagnostic_conclusion=%s\n",
               qh_diff.mismatches ? "first-divergence-q-b-f16-projection" :
               q_diff.mismatches ? "first-divergence-q-b-postprocess" :
               q_scratch_diff.mismatches ?
                   "first-divergence-q-b-internal-scratch-shape" :
               "q-b-boundaries-bit-exact");
        printf("harness_status=ok\n");
        status = 0;
        goto cleanup;
    }

    /* Static mixed attention sees the same full q_b result and the same full
     * current/compressed KV tensors in every arm.  First distinguish the
     * shipping full wrapper from a 512-row launch of the range kernel.  Then
     * keep the range kernel fixed and isolate its 512 versus 256+256 extent. */
    if (!ds4_gpu_attention_prefill_static_mixed_heads_tensor(
            heads_full, model, model_bytes, sinks_offset, q_full, raw, comp,
            0u, N_TOK, N_COMP, ATTN_WINDOW, ATTN_RATIO, N_HEAD, HEAD_DIM) ||
        !ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(
            heads_split, model, model_bytes, sinks_offset, q_full, raw, comp,
            0u, 0u, N_TOK, N_TOK, N_COMP, ATTN_WINDOW, ATTN_RATIO,
            N_HEAD, HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads_full, 0u, reference, heads_bytes) ||
        !ds4_gpu_tensor_read(heads_split, 0u, candidate, heads_bytes)) {
        fprintf(stderr, "error: static-mixed full-range boundary runtime failed\n");
        goto cleanup;
    }
    const diff_metrics attention_wrapper_diff = compare_f32(
        reference, candidate, heads_count);
    report_diff("static-mixed-attention-shipping-vs-full-range512", "f32",
                heads_count, attention_wrapper_diff);

    /* Save the full-range result as the baseline before its output tensor is
     * overwritten by the two ordered rectangular launches. */
    memcpy(reference, candidate, (size_t)heads_bytes);
    if (!ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(
            heads0, model, model_bytes, sinks_offset, q_ref0, raw, comp, 0u,
            0u, HALF_TOK, N_TOK, N_COMP, ATTN_WINDOW, ATTN_RATIO,
            N_HEAD, HEAD_DIM) ||
        !ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(
            heads1, model, model_bytes, sinks_offset, q_ref1, raw, comp, 0u,
            HALF_TOK, HALF_TOK, N_TOK, N_COMP, ATTN_WINDOW, ATTN_RATIO,
            N_HEAD, HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads_split, 0u, candidate, heads_bytes)) {
        fprintf(stderr, "error: static-mixed row-extent boundary runtime failed\n");
        goto cleanup;
    }
    const diff_metrics attention_extent_diff = compare_f32(
        reference, candidate, heads_count);
    report_diff("static-mixed-attention-full-range512-vs-row256x2", "f32",
                heads_count, attention_extent_diff);

    /* Inverse RoPE is pointwise, but explicitly verify that changing the row
     * launch extent and pos0 preserves its ordered arithmetic.  Reset the
     * split input to the identical full-reference attention output first. */
    if (!ds4_gpu_tensor_write(heads_full, 0u, reference, heads_bytes) ||
        !ds4_gpu_tensor_write(heads_split, 0u, reference, heads_bytes) ||
        !ds4_gpu_rope_tail_tensor(heads_full, N_TOK, N_HEAD, HEAD_DIM, N_ROT,
            POS0, ORIG_CTX, true, ROPE_BASE, ROPE_SCALE, ROPE_EXT,
            rope_attention_factor(), BETA_FAST, BETA_SLOW) ||
        !ds4_gpu_rope_tail_tensor(heads0, HALF_TOK, N_HEAD, HEAD_DIM, N_ROT,
            POS0, ORIG_CTX, true, ROPE_BASE, ROPE_SCALE, ROPE_EXT,
            rope_attention_factor(), BETA_FAST, BETA_SLOW) ||
        !ds4_gpu_rope_tail_tensor(heads1, HALF_TOK, N_HEAD, HEAD_DIM, N_ROT,
            POS0 + HALF_TOK, ORIG_CTX, true, ROPE_BASE, ROPE_SCALE, ROPE_EXT,
            rope_attention_factor(), BETA_FAST, BETA_SLOW) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads_full, 0u, reference, heads_bytes) ||
        !ds4_gpu_tensor_read(heads_split, 0u, candidate, heads_bytes)) {
        fprintf(stderr, "error: inverse-RoPE boundary runtime failed\n");
        goto cleanup;
    }
    const diff_metrics rope_diff = compare_f32(
        reference, candidate, heads_count);
    report_diff("inverse-rope-full512-vs-row256x2", "f32", heads_count,
                rope_diff);

    /* Feed the identical full-reference heads into both output shapes.  low
     * exposes output A, while out exposes output B; this keeps an upstream
     * difference from being misattributed to either GEMM. */
    if (!launch_output(out_full, low_full, model, model_bytes, out_a_offset,
                       out_b_offset, heads_full, N_TOK) ||
        !launch_output(out0, low0, model, model_bytes, out_a_offset,
                       out_b_offset, heads_ref0, HALF_TOK) ||
        !launch_output(out1, low1, model, model_bytes, out_a_offset,
                       out_b_offset, heads_ref1, HALF_TOK) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(low_full, 0u, reference, low_bytes) ||
        !ds4_gpu_tensor_read(low_split, 0u, candidate, low_bytes)) {
        fprintf(stderr, "error: output-A boundary runtime failed\n");
        goto cleanup;
    }
    const diff_metrics out_a_diff = compare_f32(
        reference, candidate, low_count);
    report_diff("output-a-full512-vs-row256x2", "f32", low_count,
                out_a_diff);
    memcpy(actual_low_host, reference, (size_t)low_bytes);

    if (!ds4_gpu_tensor_read(out_full, 0u, reference, out_bytes) ||
        !ds4_gpu_tensor_read(out_split, 0u, candidate, out_bytes)) {
        fprintf(stderr, "error: output-B boundary readback failed\n");
        goto cleanup;
    }
    const diff_metrics out_b_diff = compare_f32(
        reference, candidate, out_count);
    report_diff("output-a-plus-b-full512-vs-row256x2", "f32", out_count,
                out_b_diff);
    memcpy(combined_full_host, reference, (size_t)out_bytes);
    memcpy(combined_split_host, candidate, (size_t)out_bytes);

    /* Re-run B only after the exact A result has been synchronized and copied
     * identically into both low tensors.  This distinguishes row-count-sensitive
     * B arithmetic from an A-producer/B-consumer ordering defect in the wrapper.
     * The two additional comparisons identify which composed arm, if either,
     * differs from its synchronized B reference. */
    if (!ds4_gpu_tensor_write(low_full, 0u, actual_low_host, low_bytes) ||
        !ds4_gpu_tensor_write(low_split, 0u, actual_low_host, low_bytes) ||
        !ds4_gpu_attention_output_q8_batch_b_tensor(
            out_full, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
            low_full, N_TOK) ||
        !ds4_gpu_attention_output_q8_batch_b_tensor(
            out0, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
            low_ref0, HALF_TOK) ||
        !ds4_gpu_attention_output_q8_batch_b_tensor(
            out1, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
            low_ref1, HALF_TOK) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(out_full, 0u, reference, out_bytes) ||
        !ds4_gpu_tensor_read(out_split, 0u, candidate, out_bytes)) {
        fprintf(stderr, "error: actual-A-low output-B boundary runtime failed\n");
        goto cleanup;
    }
    const diff_metrics out_b_actual_diff = compare_f32(
        reference, candidate, out_count);
    const diff_metrics out_b_full_chain_diff = compare_f32(
        combined_full_host, reference, out_count);
    const diff_metrics out_b_split_chain_diff = compare_f32(
        combined_split_host, candidate, out_count);
    memcpy(shipping_b_host, reference, (size_t)out_bytes);
    report_diff("output-b-from-identical-actual-a-low-full512-vs-row256x2",
                "f32", out_count, out_b_actual_diff);
    report_diff("output-a-plus-b-full512-vs-synchronized-b-full512", "f32",
                out_count, out_b_full_chain_diff);
    report_diff("output-a-plus-b-row256x2-vs-synchronized-b-row256x2", "f32",
                out_count, out_b_split_chain_diff);

    int exact_b_algorithm = -1;
    int exact_b_algorithms[sizeof(b_algorithms) / sizeof(b_algorithms[0])];
    size_t exact_b_algorithm_count = 0u;
    for (size_t ai = 0u;
         ai < sizeof(b_algorithms) / sizeof(b_algorithms[0]); ai++) {
        const int algorithm = b_algorithms[ai];
        char text[32];
        snprintf(text, sizeof(text), "%d", algorithm);
        (void)setenv("DS4_CUDA_ATTN_OUTPUT_B_F16_GEMM_ALGO_DIAGNOSTIC",
                     text, 1);
        if (!ds4_gpu_attention_output_q8_batch_b_tensor(
                out_full, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
                low_full, N_TOK)) {
            printf("b_algorithm=%d,status=full-unsupported\n", algorithm);
            continue;
        }
        if (!ds4_gpu_attention_output_q8_batch_b_tensor(
                out0, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
                low_ref0, HALF_TOK) ||
            !ds4_gpu_attention_output_q8_batch_b_tensor(
                out1, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
                low_ref1, HALF_TOK)) {
            printf("b_algorithm=%d,status=row256-unsupported\n", algorithm);
            continue;
        }
        if (!ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(out_full, 0u, reference, out_bytes) ||
            !ds4_gpu_tensor_read(out_split, 0u, candidate, out_bytes)) {
            fprintf(stderr, "error: B algorithm %d readback failed\n",
                    algorithm);
            goto cleanup;
        }
        const diff_metrics full_vs_split = compare_f32(
            reference, candidate, out_count);
        const diff_metrics shipping_vs_full = compare_f32(
            shipping_b_host, reference, out_count);
        const diff_metrics shipping_vs_split = compare_f32(
            shipping_b_host, candidate, out_count);
        printf("b_algorithm=%d,status=ok,full_vs_row256x2_mismatches=%llu,"
               "full_vs_row256x2_max_abs=%.9g,"
               "shipping_vs_full_mismatches=%llu,"
               "shipping_vs_full_max_abs=%.9g,"
               "shipping_vs_row256x2_mismatches=%llu,"
               "shipping_vs_row256x2_max_abs=%.9g\n",
               algorithm,
               (unsigned long long)full_vs_split.mismatches,
               full_vs_split.max_abs,
               (unsigned long long)shipping_vs_full.mismatches,
               shipping_vs_full.max_abs,
               (unsigned long long)shipping_vs_split.mismatches,
               shipping_vs_split.max_abs);
        if (full_vs_split.mismatches == 0u &&
            shipping_vs_full.mismatches == 0u &&
            shipping_vs_split.mismatches == 0u) {
            if (exact_b_algorithm < 0) exact_b_algorithm = algorithm;
            exact_b_algorithms[exact_b_algorithm_count++] = algorithm;
        }
    }
    (void)unsetenv("DS4_CUDA_ATTN_OUTPUT_B_F16_GEMM_ALGO_DIAGNOSTIC");
    printf("first_shipping_exact_b_algorithm=%d\n", exact_b_algorithm);
    printf("b_algorithm_conclusion=%s\n",
           exact_b_algorithm < 0 ?
               "no-legacy-algorithm-preserved-shipping-output" :
               "explicit-algorithm-preserved-shipping-output");

    const uint32_t timing_rounds = positive_env_u32(
        "B_TIMING_ROUNDS", 7u, 99u);
    const uint32_t timing_repeats = positive_env_u32(
        "B_TIMING_REPEATS", 10u, 1000u);
    const uint32_t timing_warmups = positive_env_u32(
        "B_TIMING_WARMUPS", 3u, 100u);
    if (!timing_rounds || !timing_repeats || !timing_warmups) goto cleanup;
    printf("b_timing_scope=bounded-host-wall-clock-batched-launches\n"
           "b_timing_rounds=%u\nb_timing_repeats=%u\n"
           "b_timing_warmups=%u\n",
           timing_rounds, timing_repeats, timing_warmups);
    double shipping_full_ms = 0.0, shipping_half_ms = 0.0;
    if (!time_output_b(&shipping_full_ms, out_full, model, model_bytes,
                       out_b_offset, low_full, N_TOK, -1, timing_rounds,
                       timing_repeats, timing_warmups) ||
        !time_output_b(&shipping_half_ms, out0, model, model_bytes,
                       out_b_offset, low_ref0, HALF_TOK, -1, timing_rounds,
                       timing_repeats, timing_warmups)) {
        fprintf(stderr, "error: shipping output-B timing failed\n");
        goto cleanup;
    }
    printf("b_timing=shipping-default,rows=%u,median_ms=%.9g\n",
           N_TOK, shipping_full_ms);
    printf("b_timing=shipping-default,rows=%u,median_ms=%.9g\n",
           HALF_TOK, shipping_half_ms);

    int fastest_exact_b_algorithm = -1;
    double fastest_exact_b_half_ms = HUGE_VAL;
    for (size_t i = 0u; i < exact_b_algorithm_count; i++) {
        const int algorithm = exact_b_algorithms[i];
        double full_ms = 0.0, half_ms = 0.0;
        if (!time_output_b(&full_ms, out_full, model, model_bytes,
                           out_b_offset, low_full, N_TOK, algorithm,
                           timing_rounds, timing_repeats, timing_warmups) ||
            !time_output_b(&half_ms, out0, model, model_bytes,
                           out_b_offset, low_ref0, HALF_TOK, algorithm,
                           timing_rounds, timing_repeats, timing_warmups)) {
            fprintf(stderr, "error: exact output-B algorithm %d timing failed\n",
                    algorithm);
            goto cleanup;
        }
        printf("b_timing=explicit,algorithm=%d,rows=%u,median_ms=%.9g,"
               "shipping_full_over_arm=%.9g\n",
               algorithm, N_TOK, full_ms, shipping_full_ms / full_ms);
        printf("b_timing=explicit,algorithm=%d,rows=%u,median_ms=%.9g,"
               "shipping_full_over_parallel_half_envelope=%.9g\n",
               algorithm, HALF_TOK, half_ms, shipping_full_ms / half_ms);
        if (half_ms < fastest_exact_b_half_ms) {
            fastest_exact_b_algorithm = algorithm;
            fastest_exact_b_half_ms = half_ms;
        }
    }
    select_b_algorithm(-1);
    printf("fastest_shipping_exact_b_algorithm=%d\n",
           fastest_exact_b_algorithm);
    printf("fastest_shipping_exact_b_half_median_ms=%.9g\n",
           fastest_exact_b_half_ms);
    printf("fastest_shipping_exact_b_projected_parallel_speedup=%.9g\n",
           fastest_exact_b_algorithm < 0 ? 0.0 :
               shipping_full_ms / fastest_exact_b_half_ms);

    /* Exercise the dedicated production-candidate entry point itself.  Its
     * two calls must reproduce both the already-exact A result and shipping
     * DEFAULT's full-row A+B result; testing only the diagnostic environment
     * selector would not prove that the graph-facing helper is wired safely. */
    if (!ds4_gpu_attention_output_q8_batch_row_owned_sm75_tensor(
            out0, low0, NULL, NULL, model, model_bytes, out_a_offset,
            out_b_offset, GROUP_DIM, RANK, N_GROUP, OUT_DIM, heads_ref0,
            HALF_TOK) ||
        !ds4_gpu_attention_output_q8_batch_row_owned_sm75_tensor(
            out1, low1, NULL, NULL, model, model_bytes, out_a_offset,
            out_b_offset, GROUP_DIM, RANK, N_GROUP, OUT_DIM, heads_ref1,
            HALF_TOK) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(low_split, 0u, candidate, low_bytes)) {
        fprintf(stderr, "error: dedicated row-owned output helper failed\n");
        goto cleanup;
    }
    const diff_metrics row_owned_low_diff = compare_f32(
        actual_low_host, candidate, low_count);
    report_diff("output-a-shipping-full512-vs-row-owned-algo3-256x2",
                "f32", low_count, row_owned_low_diff);
    if (!ds4_gpu_tensor_read(out_split, 0u, candidate, out_bytes)) {
        fprintf(stderr, "error: dedicated row-owned output readback failed\n");
        goto cleanup;
    }
    const diff_metrics row_owned_output_diff = compare_f32(
        shipping_b_host, candidate, out_count);
    report_diff("output-a-plus-b-shipping-full512-vs-row-owned-algo3-256x2",
                "f32", out_count, row_owned_output_diff);
    if (row_owned_low_diff.mismatches || row_owned_output_diff.mismatches) {
        fprintf(stderr,
                "error: dedicated row-owned output helper failed exactness\n");
        goto cleanup;
    }

    /* Keep the original structured B-only fixture as a control.  Its exactness
     * is not sufficient to clear B for arbitrary A-produced inputs. */
    for (uint64_t i = 0u; i < low_count; i++) {
        const int value = (int)((i * 43u + (i >> 6u) * 17u + 61u) % 509u) - 254;
        actual_low_host[i] = (float)value / 4096.0f;
    }
    if (!ds4_gpu_tensor_write(low_full, 0u, actual_low_host, low_bytes) ||
        !ds4_gpu_tensor_write(low_split, 0u, actual_low_host, low_bytes) ||
        !ds4_gpu_attention_output_q8_batch_b_tensor(
            out_full, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
            low_full, N_TOK) ||
        !ds4_gpu_attention_output_q8_batch_b_tensor(
            out0, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
            low_ref0, HALF_TOK) ||
        !ds4_gpu_attention_output_q8_batch_b_tensor(
            out1, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
            low_ref1, HALF_TOK) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(out_full, 0u, reference, out_bytes) ||
        !ds4_gpu_tensor_read(out_split, 0u, candidate, out_bytes)) {
        fprintf(stderr, "error: isolated output-B boundary runtime failed\n");
        goto cleanup;
    }
    const diff_metrics out_b_isolated_diff = compare_f32(
        reference, candidate, out_count);
    report_diff("output-b-structured-control-full512-vs-row256x2", "f32",
                out_count, out_b_isolated_diff);

    printf("diagnostic_conclusion=%s\n",
           qh_diff.mismatches ? "first-divergence-q-b-f16-projection" :
           q_diff.mismatches ? "first-divergence-q-b-postprocess" :
           q_scratch_diff.mismatches ?
               "first-divergence-q-b-internal-scratch-shape" :
           attention_wrapper_diff.mismatches ?
               "first-divergence-static-attention-wrapper" :
           attention_extent_diff.mismatches ?
               "first-divergence-static-attention-row-extent" :
           rope_diff.mismatches ? "first-divergence-inverse-rope" :
           out_a_diff.mismatches ? "first-divergence-output-a" :
           row_owned_low_diff.mismatches ?
               "dedicated-row-owned-helper-diverged-at-output-a" :
           row_owned_output_diff.mismatches ?
               "dedicated-row-owned-helper-diverged-at-output-b" :
           out_b_full_chain_diff.mismatches || out_b_split_chain_diff.mismatches ?
               "first-divergence-output-a-to-b-dependency" :
           out_b_actual_diff.mismatches ?
               "first-divergence-output-b-row-extent" :
           out_b_diff.mismatches ? "combined-output-diverged-only" :
           out_b_isolated_diff.mismatches ?
               "structured-output-b-control-diverged" :
           "all-tested-row-arithmetic-boundaries-bit-exact");
    printf("harness_status=ok\n");
    status = 0;

cleanup:
    (void)unsetenv("DS4_CUDA_ATTN_OUTPUT_B_F16_GEMM_ALGO_DIAGNOSTIC");
    ds4_gpu_tensor_free(low_ref1);
    ds4_gpu_tensor_free(low_ref0);
    ds4_gpu_tensor_free(out1);
    ds4_gpu_tensor_free(out0);
    ds4_gpu_tensor_free(low1);
    ds4_gpu_tensor_free(low0);
    ds4_gpu_tensor_free(heads_ref1);
    ds4_gpu_tensor_free(heads_ref0);
    ds4_gpu_tensor_free(heads1);
    ds4_gpu_tensor_free(heads0);
    ds4_gpu_tensor_free(qh1);
    ds4_gpu_tensor_free(qh0);
    ds4_gpu_tensor_free(q_ref1);
    ds4_gpu_tensor_free(q_ref0);
    ds4_gpu_tensor_free(q1);
    ds4_gpu_tensor_free(q0);
    ds4_gpu_tensor_free(input1);
    ds4_gpu_tensor_free(input0);
    ds4_gpu_tensor_free(out_split);
    ds4_gpu_tensor_free(out_full);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(low_split);
    ds4_gpu_tensor_free(low_full);
    ds4_gpu_tensor_free(heads_split);
    ds4_gpu_tensor_free(heads_full);
    ds4_gpu_tensor_free(qh_split);
    ds4_gpu_tensor_free(qh_full);
    ds4_gpu_tensor_free(q_split);
    ds4_gpu_tensor_free(q_full);
    ds4_gpu_tensor_free(input);
    if (initialized) ds4_gpu_cleanup();
    free(candidate_half);
    free(reference_half);
    free(comp_host);
    free(raw_host);
    free(candidate);
    free(reference);
    free(input_host);
    free(model);
    return status;
}
