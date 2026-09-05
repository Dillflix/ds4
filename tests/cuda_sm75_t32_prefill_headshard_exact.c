#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* This is a bounded one-GPU arithmetic-shape diagnostic.  It compares the
 * production 64-head T32 projection with the exact two 32-head calls used by
 * DS4_CUDA_TP_PREFILL_T32_HEADS.  It does not change production selection. */
#define IN_DIM 1024u
#define N_HEAD 64u
#define SHARD_HEADS 32u
#define HEAD_DIM 512u
#define OUT_DIM ((uint64_t)N_HEAD * HEAD_DIM)
#define SHARD_OUT_DIM ((uint64_t)SHARD_HEADS * HEAD_DIM)
#define N_ROT 64u
#define N_TOK 512u
#define POS0 32256u
#define ORIG_CTX 65536u
#define ROPE_BASE 160000.0f
#define ROPE_SCALE 0.0625f
#define ROPE_EXT 1.0f
#define BETA_FAST 32.0f
#define BETA_SLOW 1.0f
#define EPSILON 1.0e-6f

static const int algorithms[] = {
    -1,
    0, 1, 2, 3, 4, 5, 6, 7,
    99,
    100, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115,
};

typedef struct {
    uint64_t mismatches;
    double max_abs;
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

static int build_model(unsigned char **model_out, uint64_t *model_bytes_out,
                       uint64_t *weight_bytes_out) {
    const uint64_t blocks = IN_DIM / 32u;
    const uint64_t weight_bytes = OUT_DIM * blocks * 34u;
    if (weight_bytes > (uint64_t)SIZE_MAX / 2u) return 0;
    unsigned char *model =
        (unsigned char *)malloc((size_t)(weight_bytes * 2u));
    if (!model) return 0;

    for (uint64_t row = 0u; row < OUT_DIM; row++) {
        for (uint64_t block = 0u; block < blocks; block++) {
            unsigned char *packed =
                model + (row * blocks + block) * 34u;
            const float scale =
                (float)(1u + ((row * 3u + block * 5u) % 7u)) / 128.0f;
            const uint16_t scale_bits = float_to_half_bits(scale);
            packed[0] = (unsigned char)(scale_bits & 0xffu);
            packed[1] = (unsigned char)(scale_bits >> 8u);
            for (uint64_t lane = 0u; lane < 32u; lane++) {
                const uint64_t column = block * 32u + lane;
                const int value = (int)((row * 17u + column * 11u +
                    (row >> 4u) * 7u + block * 13u + 19u) % 63u) - 31;
                packed[2u + lane] = (unsigned char)(int8_t)value;
            }
        }
    }
    memcpy(model + weight_bytes, model, (size_t)weight_bytes);
    *model_out = model;
    *model_bytes_out = weight_bytes * 2u;
    *weight_bytes_out = weight_bytes;
    return 1;
}

static diff_metrics compare_f32(const float *lhs, const float *rhs,
                                uint64_t count) {
    diff_metrics result = {0u, 0.0};
    for (uint64_t i = 0u; i < count; i++) {
        uint32_t a_bits, b_bits;
        memcpy(&a_bits, lhs + i, sizeof(a_bits));
        memcpy(&b_bits, rhs + i, sizeof(b_bits));
        if (a_bits != b_bits) result.mismatches++;
        const double delta = fabs((double)lhs[i] - (double)rhs[i]);
        if (delta > result.max_abs) result.max_abs = delta;
    }
    return result;
}

static uint64_t compare_u16(const uint16_t *lhs, const uint16_t *rhs,
                            uint64_t count) {
    uint64_t mismatches = 0u;
    for (uint64_t i = 0u; i < count; i++)
        if (lhs[i] != rhs[i]) mismatches++;
    return mismatches;
}

static float rope_attention_factor(void) {
    return 1.0f / (1.0f + 0.1f * logf(1.0f / ROPE_SCALE));
}

static int launch_full(ds4_gpu_tensor *output, ds4_gpu_tensor *half,
                       const unsigned char *model, uint64_t model_bytes,
                       const ds4_gpu_tensor *input, uint64_t weight_offset) {
    return ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        output, half, model, model_bytes, weight_offset,
        IN_DIM, OUT_DIM, input, N_TOK, N_HEAD, HEAD_DIM, N_ROT,
        POS0, ORIG_CTX, false, ROPE_BASE, ROPE_SCALE, ROPE_EXT,
        rope_attention_factor(), BETA_FAST, BETA_SLOW, EPSILON);
}

static int launch_shards(ds4_gpu_tensor *output,
                         ds4_gpu_tensor *half0, ds4_gpu_tensor *half1,
                         const unsigned char *model, uint64_t model_bytes,
                         uint64_t shard_model_offset,
                         const ds4_gpu_tensor *input,
                         uint64_t shard_weight_bytes) {
    return ds4_gpu_attn_q_b_f16_head_shard_rms_rope_tail_tensor(
               output, half0, model, model_bytes, shard_model_offset,
               IN_DIM, SHARD_OUT_DIM, input, N_TOK, SHARD_HEADS, 0u,
               N_HEAD, HEAD_DIM, N_ROT, POS0, ORIG_CTX, false,
               ROPE_BASE, ROPE_SCALE, ROPE_EXT, rope_attention_factor(),
               BETA_FAST, BETA_SLOW, EPSILON) &&
           ds4_gpu_attn_q_b_f16_head_shard_rms_rope_tail_tensor(
               output, half1, model, model_bytes,
               shard_model_offset + shard_weight_bytes,
               IN_DIM, SHARD_OUT_DIM, input, N_TOK, SHARD_HEADS,
               SHARD_HEADS, N_HEAD, HEAD_DIM, N_ROT, POS0, ORIG_CTX, false,
               ROPE_BASE, ROPE_SCALE, ROPE_EXT, rope_attention_factor(),
               BETA_FAST, BETA_SLOW, EPSILON);
}

int main(void) {
    const uint64_t input_count = (uint64_t)N_TOK * IN_DIM;
    const uint64_t output_count = (uint64_t)N_TOK * OUT_DIM;
    const uint64_t half_count = output_count;
    const uint64_t shard_half_count = (uint64_t)N_TOK * SHARD_OUT_DIM;
    const uint64_t output_bytes = output_count * sizeof(float);
    const uint64_t half_bytes = half_count * sizeof(uint16_t);
    const uint64_t shard_half_bytes = shard_half_count * sizeof(uint16_t);
    unsigned char *model = NULL;
    uint64_t model_bytes = 0u, weight_bytes = 0u;
    float *input_host = NULL, *shipping_output = NULL, *full_output = NULL;
    float *shard_output = NULL;
    uint16_t *shipping_half = NULL, *full_half = NULL;
    uint16_t *shard_half0 = NULL, *shard_half1 = NULL;
    uint16_t *assembled_half = NULL;
    ds4_gpu_tensor *input = NULL, *full = NULL, *sharded = NULL;
    ds4_gpu_tensor *qh_full = NULL, *qh0 = NULL, *qh1 = NULL;
    int initialized = 0, status = 1;

    if (!build_model(&model, &model_bytes, &weight_bytes)) {
        fprintf(stderr, "error: model construction failed\n");
        goto cleanup;
    }
    const uint64_t shard_weight_bytes = weight_bytes / 2u;
    input_host = (float *)malloc((size_t)input_count * sizeof(float));
    shipping_output = (float *)malloc((size_t)output_bytes);
    full_output = (float *)malloc((size_t)output_bytes);
    shard_output = (float *)malloc((size_t)output_bytes);
    shipping_half = (uint16_t *)malloc((size_t)half_bytes);
    full_half = (uint16_t *)malloc((size_t)half_bytes);
    shard_half0 = (uint16_t *)malloc((size_t)shard_half_bytes);
    shard_half1 = (uint16_t *)malloc((size_t)shard_half_bytes);
    assembled_half = (uint16_t *)malloc((size_t)half_bytes);
    if (!input_host || !shipping_output || !full_output || !shard_output ||
        !shipping_half || !full_half || !shard_half0 || !shard_half1 ||
        !assembled_half) {
        fprintf(stderr, "error: host allocation failed\n");
        goto cleanup;
    }
    for (uint64_t i = 0u; i < input_count; i++) {
        const int value = (int)((i * 29u + (i >> 5u) * 17u +
                                 (i / IN_DIM) * 7u + 23u) % 257u) - 128;
        input_host[i] = (float)value / 128.0f;
    }

    (void)setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    (void)setenv("DS4_CUDA_Q8_F16_CACHE_MB", "192", 1);
    (void)setenv("DS4_CUDA_Q8_F16_CACHE_RESERVE_MB", "1", 1);
    (void)setenv("DS4_CUDA_NO_TF32", "1", 1);
    (void)setenv("DS4_CUDA_T32_F16_FUSED", "1", 1);
    (void)unsetenv("DS4_CUDA_NO_T32_F16_FUSED");
    (void)unsetenv("DS4_CUDA_NO_Q8_F16_CACHE");
    (void)unsetenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE");
    (void)unsetenv("DS4_CUDA_T32_F16_GEMM_ALGO_DIAGNOSTIC");

    if (!ds4_gpu_init()) {
        fprintf(stderr, "error: CUDA initialization failed\n");
        goto cleanup;
    }
    initialized = 1;
    if (!ds4_gpu_set_model_map(model, model_bytes) ||
        !ds4_gpu_cache_q8_f16_range(model, model_bytes, 0u, weight_bytes,
                                    IN_DIM, OUT_DIM, "attn_q_b") ||
        !ds4_gpu_cache_q8_f16_range(model, model_bytes, weight_bytes,
                                    shard_weight_bytes, IN_DIM,
                                    SHARD_OUT_DIM, "attn_q_b") ||
        !ds4_gpu_cache_q8_f16_range(model, model_bytes,
                                    weight_bytes + shard_weight_bytes,
                                    shard_weight_bytes, IN_DIM,
                                    SHARD_OUT_DIM, "attn_q_b")) {
        fprintf(stderr, "error: model/cache installation failed\n");
        goto cleanup;
    }

    input = ds4_gpu_tensor_alloc(input_count * sizeof(float));
    full = ds4_gpu_tensor_alloc(output_bytes);
    sharded = ds4_gpu_tensor_alloc(output_bytes);
    qh_full = ds4_gpu_tensor_alloc(half_bytes);
    qh0 = ds4_gpu_tensor_alloc(shard_half_bytes);
    qh1 = ds4_gpu_tensor_alloc(shard_half_bytes);
    if (!input || !full || !sharded || !qh_full || !qh0 || !qh1 ||
        !ds4_gpu_tensor_write(input, 0u, input_host,
                              input_count * sizeof(float))) {
        fprintf(stderr, "error: device allocation/setup failed\n");
        goto cleanup;
    }

    if (!launch_full(full, qh_full, model, model_bytes, input, 0u) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(full, 0u, shipping_output, output_bytes) ||
        !ds4_gpu_tensor_read(qh_full, 0u, shipping_half, half_bytes)) {
        fprintf(stderr, "error: shipping DEFAULT reference failed\n");
        goto cleanup;
    }

    printf("scenario=sm75-t32-prefill-full-vs-two-head-shards\n"
           "scope=bounded-single-gpu-production-shapes\n"
           "production_default_changed=0\n"
           "n_tokens=%u\nin_dim=%u\nout_dim=%llu\n"
           "full_heads=%u\nshard_heads=%u\n"
           "full_weight_bytes=%llu\nf16_cache_bytes=%llu\n",
           N_TOK, IN_DIM, (unsigned long long)OUT_DIM, N_HEAD, SHARD_HEADS,
           (unsigned long long)weight_bytes,
           (unsigned long long)(2u * OUT_DIM * IN_DIM * sizeof(uint16_t)));

    int exact_algorithm = -2;
    size_t algorithm_count = sizeof(algorithms) / sizeof(algorithms[0]);
    if (getenv("DS4_T32_HEADSHARD_SANITIZER_SMOKE"))
        algorithm_count = 1u;
    for (size_t ai = 0u; ai < algorithm_count; ai++) {
        const int algorithm = algorithms[ai];
        char text[32];
        if (algorithm < 0) {
            (void)unsetenv("DS4_CUDA_T32_F16_GEMM_ALGO_DIAGNOSTIC");
        } else {
            snprintf(text, sizeof(text), "%d", algorithm);
            (void)setenv("DS4_CUDA_T32_F16_GEMM_ALGO_DIAGNOSTIC", text, 1);
        }
        if (!launch_full(full, qh_full, model, model_bytes, input, 0u)) {
            printf("algorithm=%d,status=full-unsupported\n", algorithm);
            continue;
        }
        if (!launch_shards(sharded, qh0, qh1, model, model_bytes,
                           weight_bytes, input, shard_weight_bytes)) {
            printf("algorithm=%d,status=shard-unsupported\n", algorithm);
            continue;
        }
        if (!ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(full, 0u, full_output, output_bytes) ||
            !ds4_gpu_tensor_read(sharded, 0u, shard_output, output_bytes) ||
            !ds4_gpu_tensor_read(qh_full, 0u, full_half, half_bytes) ||
            !ds4_gpu_tensor_read(qh0, 0u, shard_half0, shard_half_bytes) ||
            !ds4_gpu_tensor_read(qh1, 0u, shard_half1, shard_half_bytes)) {
            fprintf(stderr, "error: algorithm %d readback failed\n", algorithm);
            goto cleanup;
        }
        for (uint64_t token = 0u; token < N_TOK; token++) {
            memcpy(assembled_half + token * OUT_DIM,
                   shard_half0 + token * SHARD_OUT_DIM,
                   (size_t)shard_half_bytes / N_TOK);
            memcpy(assembled_half + token * OUT_DIM + SHARD_OUT_DIM,
                   shard_half1 + token * SHARD_OUT_DIM,
                   (size_t)shard_half_bytes / N_TOK);
        }
        const uint64_t full_vs_shard_half =
            compare_u16(full_half, assembled_half, half_count);
        const uint64_t shipping_vs_full_half =
            compare_u16(shipping_half, full_half, half_count);
        const diff_metrics full_vs_shard =
            compare_f32(full_output, shard_output, output_count);
        const diff_metrics shipping_vs_full =
            compare_f32(shipping_output, full_output, output_count);
        const diff_metrics shipping_vs_shard =
            compare_f32(shipping_output, shard_output, output_count);
        printf("algorithm=%d,status=ok,"
               "full_vs_shard_half_mismatches=%llu,"
               "shipping_vs_full_half_mismatches=%llu,"
               "full_vs_shard_f32_mismatches=%llu,"
               "full_vs_shard_max_abs=%.9g,"
               "shipping_vs_full_f32_mismatches=%llu,"
               "shipping_vs_full_max_abs=%.9g,"
               "shipping_vs_shard_f32_mismatches=%llu,"
               "shipping_vs_shard_max_abs=%.9g\n",
               algorithm,
               (unsigned long long)full_vs_shard_half,
               (unsigned long long)shipping_vs_full_half,
               (unsigned long long)full_vs_shard.mismatches,
               full_vs_shard.max_abs,
               (unsigned long long)shipping_vs_full.mismatches,
               shipping_vs_full.max_abs,
               (unsigned long long)shipping_vs_shard.mismatches,
               shipping_vs_shard.max_abs);
        if (shipping_vs_shard.mismatches == 0u &&
            shipping_vs_full.mismatches == 0u &&
            full_vs_shard_half == 0u && shipping_vs_full_half == 0u &&
            exact_algorithm == -2) {
            exact_algorithm = algorithm;
        }
    }
    printf("first_shipping_exact_algorithm=%d\n", exact_algorithm);
    printf("diagnostic_conclusion=%s\n",
           exact_algorithm == -2 ?
               "no-legacy-cublas-algorithm-preserved-shipping-exactness" :
               "fixed-cublas-algorithm-candidate-found");
    printf("harness_status=ok\n");
    status = 0;

cleanup:
    (void)unsetenv("DS4_CUDA_T32_F16_GEMM_ALGO_DIAGNOSTIC");
    ds4_gpu_tensor_free(qh1);
    ds4_gpu_tensor_free(qh0);
    ds4_gpu_tensor_free(qh_full);
    ds4_gpu_tensor_free(sharded);
    ds4_gpu_tensor_free(full);
    ds4_gpu_tensor_free(input);
    if (initialized) ds4_gpu_cleanup();
    free(assembled_half);
    free(shard_half1);
    free(shard_half0);
    free(full_half);
    free(shipping_half);
    free(shard_output);
    free(full_output);
    free(shipping_output);
    free(input_host);
    free(model);
    return status;
}
