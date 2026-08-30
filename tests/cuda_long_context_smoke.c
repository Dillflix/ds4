#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void ds4_gpu_test_set_moe_q32_decode_graph(int enabled);
extern void ds4_gpu_test_set_moe_q32_decode_split(int enabled);
extern void ds4_gpu_test_set_moe_q32_decode_fused_lowreg(uint32_t unroll);
extern void ds4_gpu_test_set_moe_q4_32_decode_mapping(uint32_t mapping);
extern uint32_t ds4_gpu_test_get_moe_q4_32_decode_mapping(void);
extern void ds4_gpu_test_set_moe_q3a4_decode_mapping(uint32_t mapping);
extern uint32_t ds4_gpu_test_get_moe_q3a4_decode_mapping(void);
extern void ds4_gpu_test_set_moe_q3a4_decode_ksplit(uint32_t split);
extern uint32_t ds4_gpu_test_get_moe_q3a4_decode_ksplit(void);
extern void ds4_gpu_test_set_moe_q3a4_decode_prefetch_depth(uint32_t depth);
extern uint32_t ds4_gpu_test_get_moe_q3a4_decode_prefetch_depth(void);
extern void ds4_gpu_test_refresh_decode_dispatch_env(void);

static unsigned char *idle_model_map;
static const uint64_t idle_model_bytes = 4096u;

/* A registered model mapping must outlive every kernel that can reference it
 * and must not be freed before ds4_gpu_set_model_map unregisters it. Keep one
 * process-lifetime idle mapping so individual tests can retire temporary
 * model buffers safely without tearing down the CUDA backend. */
static int retire_temporary_model_map(void) {
    return idle_model_map &&
           ds4_gpu_set_model_map(idle_model_map, idle_model_bytes);
}

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static double getenv_seconds(const char *name, double fallback) {
    const char *s = getenv(name);
    if (!s || !s[0]) return fallback;
    char *end = NULL;
    const double v = strtod(s, &end);
    return end != s && v > 0.0 ? v : fallback;
}

static int check_sm75_q8_mma_exact_case(uint32_t in_dim,
                                        uint32_t out_dim,
                                        uint32_t n_tokens,
                                        const char *label) {
    /* The T32 case is sized so its allocation ends on a CUDA page boundary;
     * the second call covers the production T256 reduction. */
    const uint32_t blocks = in_dim / 32u;
    const size_t weight_bytes = (size_t)out_dim * blocks * 34u;
    const size_t x_count = (size_t)n_tokens * in_dim;
    const size_t out_count = (size_t)n_tokens * out_dim;
    unsigned char *model = (unsigned char *)calloc(1, weight_bytes);
    float *x_host = (float *)malloc(x_count * sizeof(float));
    float *reference = (float *)malloc(out_count * sizeof(float));
    float *candidate = (float *)malloc(out_count * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    int rc = 1;
    if (!model || !x_host || !reference || !candidate || !x || !out) goto cleanup;

    for (uint32_t row = 0; row < out_dim; row++) {
        for (uint32_t b = 0; b < blocks; b++) {
            unsigned char *blk = model + ((size_t)row * blocks + b) * 34u;
            const uint16_t d = 0x2400u; /* 1 / 64, exactly representable in f16. */
            memcpy(blk, &d, sizeof(d));
            for (uint32_t i = 0; i < 32u; i++) {
                const int q = (int)((row * 17u + b * 13u + i * 7u) % 127u) - 63;
                blk[2u + i] = (unsigned char)(int8_t)q;
            }
        }
    }
    for (size_t i = 0; i < x_count; i++) {
        const int v = (int)((i * 29u + (i >> 3u) * 11u) % 193u) - 96;
        x_host[i] = (float)v / 101.0f;
    }
    if (!ds4_gpu_tensor_write(x, 0, x_host, x_count * sizeof(float)) ||
        !ds4_gpu_set_model_map(model, weight_bytes)) goto cleanup;
    (void)setenv("DS4_CUDA_NO_Q8_F16_CACHE", "1", 1);
    (void)setenv("DS4_CUDA_NO_Q8_MMA_SM75", "1", 1);
    if (!ds4_gpu_matmul_q8_0_tensor(out, model, weight_bytes, 0,
                                    in_dim, out_dim, x, n_tokens) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(out, 0, reference,
                             out_count * sizeof(float))) goto cleanup;
    (void)unsetenv("DS4_CUDA_NO_Q8_MMA_SM75");
    (void)setenv("DS4_CUDA_Q8_MMA_SM75_TOK16", "1", 1);
    if (!ds4_gpu_matmul_q8_0_tensor(out, model, weight_bytes, 0,
                                    in_dim, out_dim, x, n_tokens) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(out, 0, candidate,
                             out_count * sizeof(float))) goto cleanup;
    if (memcmp(reference, candidate, out_count * sizeof(float)) != 0) {
        size_t first = 0;
        while (first < out_count &&
               memcmp(reference + first, candidate + first, sizeof(float)) == 0) {
            first++;
        }
        fprintf(stderr,
                "sm75 q8 mma mismatch at %zu: reference=%a candidate=%a\n",
                first, (double)reference[first], (double)candidate[first]);
        goto cleanup;
    }
    fprintf(stderr,
            "cuda-regression: sm75 q8 mma tok16 %s exact (%zu values)\n",
            label, out_count);
    rc = 0;

cleanup:
    if (model && !retire_temporary_model_map()) rc = 1;
    (void)unsetenv("DS4_CUDA_NO_Q8_MMA_SM75");
    (void)unsetenv("DS4_CUDA_Q8_MMA_SM75_TOK16");
    (void)unsetenv("DS4_CUDA_NO_Q8_F16_CACHE");
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    free(candidate);
    free(reference);
    free(x_host);
    free(model);
    return rc;
}

static int check_sm75_q8_mma_exact(void) {
    /* T32 retains the page-boundary over-read regression fixture. T256 is
     * the shipping 8192-wide attention-output reduction. */
    if (check_sm75_q8_mma_exact_case(1024u, 1024u, 17u, "T32") != 0)
        return 1;
    return check_sm75_q8_mma_exact_case(8192u, 128u, 17u, "T256");
}

static int check_sm75_iq2_moe_mma_exact(void) {
    const uint32_t n_total_expert = 8;
    /* One selected expert per token isolates every final-output write.  The
     * uneven populations below exercise full tile16 work plus tile8/tile4
     * tails without introducing atomic-order ambiguity. */
    const uint32_t n_expert = 1;
    const uint32_t n_tokens = 128;
    const uint32_t in_dim = 4096;
    /* Production-width routed intermediate: eight Q4_K down blocks exercise
     * every b&7 accumulator slot in both the array and scalar tile16 paths. */
    const uint32_t mid_dim = 2048;
    const uint32_t out_dim = 256;
    const uint32_t gate_blocks = in_dim / 256u;
    const uint32_t down_blocks = mid_dim / 256u;
    const uint64_t gate_row_bytes = (uint64_t)gate_blocks * 66u;
    const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = (uint64_t)down_blocks * 144u;
    const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = gate_expert_bytes * n_total_expert;
    const uint64_t down_offset = up_offset + gate_expert_bytes * n_total_expert;
    const uint64_t model_bytes = down_offset + down_expert_bytes * n_total_expert;
    const uint64_t pair_count = (uint64_t)n_tokens * n_expert;
    const uint64_t mid_count = pair_count * mid_dim;
    const uint64_t x_count = (uint64_t)n_tokens * in_dim;
    const uint64_t out_count = (uint64_t)n_tokens * out_dim;
    /* routed_moe_launch aliases down as the Q8_K input scratch when that is
     * larger than the per-pair down output.  Float input bytes are a simple,
     * type-independent upper bound for the packed Q8_K scratch. */
    const uint64_t down_output_bytes = pair_count * out_dim * sizeof(float);
    const uint64_t x_scratch_bound = x_count * sizeof(float);
    const uint64_t down_storage_bytes =
        down_output_bytes > x_scratch_bound ? down_output_bytes : x_scratch_bound;
    unsigned char *model = (unsigned char *)calloc(1, (size_t)model_bytes);
    float *x_host = (float *)malloc((size_t)x_count * sizeof(float));
    int32_t *selected_host = (int32_t *)malloc((size_t)pair_count * sizeof(int32_t));
    float *weights_host = (float *)malloc((size_t)pair_count * sizeof(float));
    float *reference = (float *)malloc((size_t)mid_count * sizeof(float));
    float *candidate = (float *)malloc((size_t)mid_count * sizeof(float));
    float *down_reference = (float *)malloc((size_t)out_count * sizeof(float));
    float *down_candidate = (float *)malloc((size_t)out_count * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(pair_count * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * out_dim * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(down_storage_bytes);
    int rc = 1;
    if (!model || !x_host || !selected_host || !weights_host ||
        !reference || !candidate || !down_reference || !down_candidate ||
        !x || !selected || !weights || !out ||
        !gate || !up || !mid || !down) goto cleanup;

    for (uint32_t matrix = 0; matrix < 2u; matrix++) {
        unsigned char *base = model + (matrix ? up_offset : gate_offset);
        for (uint32_t e = 0; e < n_total_expert; e++) {
            for (uint32_t row = 0; row < mid_dim; row++) {
                for (uint32_t b = 0; b < gate_blocks; b++) {
                    unsigned char *blk = base + (uint64_t)e * gate_expert_bytes +
                        (uint64_t)row * gate_row_bytes + (uint64_t)b * 66u;
                    const uint16_t d = 0x1800u; /* 1 / 512. */
                    memcpy(blk, &d, sizeof(d));
                    uint16_t *q2 = (uint16_t *)(blk + 2u);
                    for (uint32_t j = 0; j < 8u; j++) {
                        const uint32_t g0 = (e * 31u + row * 7u + b * 13u +
                                             j * 17u + matrix * 19u) & 255u;
                        const uint32_t g1 = (g0 + 37u) & 255u;
                        const uint32_t g2 = (g0 + 79u) & 255u;
                        const uint32_t g3 = (g0 + 131u) & 255u;
                        const uint32_t s0 = (g0 + j) & 127u;
                        const uint32_t s1 = (g1 + j * 3u) & 127u;
                        const uint32_t s2 = (g2 + j * 5u) & 127u;
                        const uint32_t s3 = (g3 + j * 7u) & 127u;
                        const uint32_t ls = (e + row + b + j + matrix) & 15u;
                        const uint32_t aux0 = g0 | (g1 << 8u) |
                                              (g2 << 16u) | (g3 << 24u);
                        const uint32_t aux1 = s0 | (s1 << 7u) |
                                              (s2 << 14u) | (s3 << 21u) |
                                              (ls << 28u);
                        q2[j * 4u + 0u] = (uint16_t)aux0;
                        q2[j * 4u + 1u] = (uint16_t)(aux0 >> 16u);
                        q2[j * 4u + 2u] = (uint16_t)aux1;
                        q2[j * 4u + 3u] = (uint16_t)(aux1 >> 16u);
                    }
                }
            }
        }
    }
    for (uint32_t e = 0; e < n_total_expert; e++) {
        for (uint32_t row = 0; row < out_dim; row++) {
            for (uint32_t b = 0; b < down_blocks; b++) {
                unsigned char *blk = model + down_offset +
                    (uint64_t)e * down_expert_bytes +
                    (uint64_t)row * down_row_bytes + (uint64_t)b * 144u;
                const uint16_t d = 0x2000u;    /* 1 / 128. */
                const uint16_t dmin = 0x1800u; /* 1 / 512. */
                memcpy(blk + 0u, &d, sizeof(d));
                memcpy(blk + 2u, &dmin, sizeof(dmin));
                for (uint32_t i = 0; i < 12u; i++)
                    blk[4u + i] = (unsigned char)(
                        (e * 29u + row * 11u + b * 7u + i * 13u) & 0xffu);
                for (uint32_t i = 0; i < 128u; i++)
                    blk[16u + i] = (unsigned char)(
                        (e * 17u + row * 23u + b * 31u + i * 5u) & 0xffu);
            }
        }
    }
    for (uint64_t i = 0; i < x_count; i++) {
        const int v = (int)((i * 23u + (i >> 5u) * 17u) % 257u) - 128;
        x_host[i] = (float)v / 133.0f;
    }
    /* Counts 25,25,20,20,19,19: two true tile8 tails and six tile4 tails. */
    const uint32_t cuts[6] = {25u, 50u, 70u, 90u, 109u, 128u};
    for (uint32_t t = 0; t < n_tokens; t++) {
        uint32_t e = 0u;
        while (e < 5u && t >= cuts[e]) e++;
        selected_host[t] = (int32_t)e;
        weights_host[t] = (float)((t % 7u) + 1u) / 8.0f;
    }
    if (!ds4_gpu_tensor_write(x, 0, x_host, x_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(selected, 0, selected_host,
                              pair_count * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(weights, 0, weights_host,
                              pair_count * sizeof(float)) ||
        !ds4_gpu_set_model_map(model, model_bytes)) goto cleanup;
    (void)setenv("DS4_CUDA_MOE_WRITE_GATE_UP", "1", 1);
    (void)setenv("DS4_CUDA_MOE_NO_IQ2_MMA_SM75", "1", 1);
    bool mid_is_f16 = false;
    if (!ds4_gpu_routed_moe_batch_tensor(
            out, gate, up, mid, down, model, model_bytes,
            gate_offset, up_offset, down_offset, 16u, 12u,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, selected, weights,
            n_total_expert, n_expert, 10.0f, x, 0u, n_tokens,
            &mid_is_f16, true) || mid_is_f16 || !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(mid, 0, reference,
                             mid_count * sizeof(float))) goto cleanup;
    (void)unsetenv("DS4_CUDA_MOE_NO_IQ2_MMA_SM75");
    if (!ds4_gpu_routed_moe_batch_tensor(
            out, gate, up, mid, down, model, model_bytes,
            gate_offset, up_offset, down_offset, 16u, 12u,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, selected, weights,
            n_total_expert, n_expert, 10.0f, x, 0u, n_tokens,
            &mid_is_f16, true) || mid_is_f16 || !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(mid, 0, candidate,
                             mid_count * sizeof(float))) goto cleanup;
    if (memcmp(reference, candidate, mid_count * sizeof(float)) != 0) {
        uint64_t first = 0;
        while (first < mid_count &&
               memcmp(reference + first, candidate + first, sizeof(float)) == 0) {
            first++;
        }
        fprintf(stderr,
                "sm75 iq2 moe mma mismatch at %llu: reference=%a candidate=%a\n",
                (unsigned long long)first,
                (double)reference[first], (double)candidate[first]);
        goto cleanup;
    }
    fprintf(stderr, "cuda-regression: sm75 iq2 moe tile16 exact (%llu values)\n",
            (unsigned long long)mid_count);

    const char *stage_env[2] = {
        "DS4_CUDA_MOE_IQ2_STAGE6_SM75",
        "DS4_CUDA_MOE_IQ2_STAGE4_SM75"
    };
    const char *stage_name[2] = { "stage6", "stage4" };
    for (uint32_t variant = 0; variant < 2u; variant++) {
        (void)setenv(stage_env[variant], "1", 1);
        if (!ds4_gpu_routed_moe_batch_tensor(
                out, gate, up, mid, down, model, model_bytes,
                gate_offset, up_offset, down_offset, 16u, 12u,
                gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes,
                in_dim, mid_dim, out_dim, selected, weights,
                n_total_expert, n_expert, 10.0f, x, 0u, n_tokens,
                &mid_is_f16, true) || mid_is_f16 || !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(mid, 0, candidate,
                                 mid_count * sizeof(float))) goto cleanup;
        if (memcmp(reference, candidate, mid_count * sizeof(float)) != 0) {
            uint64_t first = 0;
            while (first < mid_count &&
                   memcmp(reference + first, candidate + first,
                          sizeof(float)) == 0) first++;
            fprintf(stderr,
                    "sm75 iq2 %s mismatch at %llu: reference=%a candidate=%a\n",
                    stage_name[variant], (unsigned long long)first,
                    (double)reference[first], (double)candidate[first]);
            goto cleanup;
        }
        fprintf(stderr,
                "cuda-regression: sm75 iq2 moe %s exact (%llu values)\n",
                stage_name[variant], (unsigned long long)mid_count);

        (void)setenv("DS4_CUDA_MOE_IQ2_SCALAR_SM75", "1", 1);
        if (!ds4_gpu_routed_moe_batch_tensor(
                out, gate, up, mid, down, model, model_bytes,
                gate_offset, up_offset, down_offset, 16u, 12u,
                gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes,
                in_dim, mid_dim, out_dim, selected, weights,
                n_total_expert, n_expert, 10.0f, x, 0u, n_tokens,
                &mid_is_f16, true) || mid_is_f16 || !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(mid, 0, candidate,
                                 mid_count * sizeof(float))) goto cleanup;
        (void)setenv("DS4_CUDA_MOE_IQ2_SCALAR_SM75", "0", 1);
        (void)unsetenv(stage_env[variant]);
        if (memcmp(reference, candidate, mid_count * sizeof(float)) != 0) {
            uint64_t first = 0;
            while (first < mid_count &&
                   memcmp(reference + first, candidate + first,
                          sizeof(float)) == 0) first++;
            fprintf(stderr,
                    "sm75 iq2 %s scalar mismatch at %llu: reference=%a candidate=%a\n",
                    stage_name[variant], (unsigned long long)first,
                    (double)reference[first], (double)candidate[first]);
            goto cleanup;
        }
        fprintf(stderr,
                "cuda-regression: sm75 iq2 moe %s scalar exact (%llu values)\n",
                stage_name[variant], (unsigned long long)mid_count);
    }

    (void)setenv("DS4_CUDA_MOE_IQ2_SCALAR_SM75", "1", 1);
    if (!ds4_gpu_routed_moe_batch_tensor(
            out, gate, up, mid, down, model, model_bytes,
            gate_offset, up_offset, down_offset, 16u, 12u,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, selected, weights,
            n_total_expert, n_expert, 10.0f, x, 0u, n_tokens,
            &mid_is_f16, true) || mid_is_f16 || !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(mid, 0, candidate,
                             mid_count * sizeof(float))) goto cleanup;
    if (memcmp(reference, candidate, mid_count * sizeof(float)) != 0) {
        uint64_t first = 0;
        while (first < mid_count &&
               memcmp(reference + first, candidate + first,
                      sizeof(float)) == 0) first++;
        fprintf(stderr,
                "sm75 iq2 tile16 scalar mismatch at %llu: reference=%a candidate=%a\n",
                (unsigned long long)first,
                (double)reference[first], (double)candidate[first]);
        goto cleanup;
    }
    fprintf(stderr,
            "cuda-regression: sm75 iq2 moe tile16 scalar exact (%llu values)\n",
            (unsigned long long)mid_count);
    (void)setenv("DS4_CUDA_MOE_IQ2_SCALAR_SM75", "0", 1);

    (void)setenv("DS4_CUDA_MOE_NO_IQ2_MMA_TILE16_SM75", "1", 1);
    if (!ds4_gpu_routed_moe_batch_tensor(
            out, gate, up, mid, down, model, model_bytes,
            gate_offset, up_offset, down_offset, 16u, 12u,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, selected, weights,
            n_total_expert, n_expert, 10.0f, x, 0u, n_tokens,
            &mid_is_f16, true) || mid_is_f16 || !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(mid, 0, candidate,
                             mid_count * sizeof(float))) goto cleanup;
    if (memcmp(reference, candidate, mid_count * sizeof(float)) != 0) {
        uint64_t first = 0;
        while (first < mid_count &&
               memcmp(reference + first, candidate + first,
                      sizeof(float)) == 0) first++;
        fprintf(stderr,
                "sm75 iq2 tile8 mismatch at %llu: reference=%a candidate=%a\n",
                (unsigned long long)first,
                (double)reference[first], (double)candidate[first]);
        goto cleanup;
    }
    fprintf(stderr,
            "cuda-regression: sm75 iq2 moe tile8 exact (%llu values)\n",
            (unsigned long long)mid_count);

    (void)setenv("DS4_CUDA_MOE_IQ2_SCALAR_SM75", "1", 1);
    if (!ds4_gpu_routed_moe_batch_tensor(
            out, gate, up, mid, down, model, model_bytes,
            gate_offset, up_offset, down_offset, 16u, 12u,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, selected, weights,
            n_total_expert, n_expert, 10.0f, x, 0u, n_tokens,
            &mid_is_f16, true) || mid_is_f16 || !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(mid, 0, candidate,
                             mid_count * sizeof(float))) goto cleanup;
    if (memcmp(reference, candidate, mid_count * sizeof(float)) != 0) {
        uint64_t first = 0;
        while (first < mid_count &&
               memcmp(reference + first, candidate + first,
                      sizeof(float)) == 0) first++;
        fprintf(stderr,
                "sm75 iq2 tile8 scalar mismatch at %llu: reference=%a candidate=%a\n",
                (unsigned long long)first,
                (double)reference[first], (double)candidate[first]);
        goto cleanup;
    }
    fprintf(stderr,
            "cuda-regression: sm75 iq2 moe tile8 scalar exact (%llu values)\n",
            (unsigned long long)mid_count);
    (void)setenv("DS4_CUDA_MOE_IQ2_SCALAR_SM75", "0", 1);
    (void)unsetenv("DS4_CUDA_MOE_NO_IQ2_MMA_TILE16_SM75");

    /* The mixed-tail tile16 route invokes the scalar tile8 specialization on
     * the true 8-slot remainders and the existing tile4 kernel thereafter. */
    (void)setenv("DS4_CUDA_MOE_MIXED_TAIL_TILES", "1", 1);
    (void)setenv("DS4_CUDA_MOE_IQ2_SCALAR_SM75", "1", 1);
    if (!ds4_gpu_routed_moe_batch_tensor(
            out, gate, up, mid, down, model, model_bytes,
            gate_offset, up_offset, down_offset, 16u, 12u,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, selected, weights,
            n_total_expert, n_expert, 10.0f, x, 0u, n_tokens,
            &mid_is_f16, true) || mid_is_f16 || !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(mid, 0, candidate,
                             mid_count * sizeof(float))) goto cleanup;
    if (memcmp(reference, candidate, mid_count * sizeof(float)) != 0) {
        uint64_t first = 0;
        while (first < mid_count &&
               memcmp(reference + first, candidate + first,
                      sizeof(float)) == 0) first++;
        fprintf(stderr,
                "sm75 iq2 mixed-tail scalar mismatch at %llu: reference=%a candidate=%a\n",
                (unsigned long long)first,
                (double)reference[first], (double)candidate[first]);
        goto cleanup;
    }
    fprintf(stderr,
            "cuda-regression: sm75 iq2 moe mixed-tail tile8 scalar exact (%llu values)\n",
            (unsigned long long)mid_count);
    (void)setenv("DS4_CUDA_MOE_IQ2_SCALAR_SM75", "0", 1);
    (void)unsetenv("DS4_CUDA_MOE_MIXED_TAIL_TILES");

    (void)setenv("DS4_CUDA_MOE_NO_Q4_MMA_TILE16_SM75", "1", 1);
    if (!ds4_gpu_routed_moe_batch_tensor(
            out, gate, up, mid, down, model, model_bytes,
            gate_offset, up_offset, down_offset, 16u, 12u,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, selected, weights,
            n_total_expert, n_expert, 10.0f, x, 0u, n_tokens,
            &mid_is_f16, true) || mid_is_f16 || !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(out, 0, down_reference,
                             out_count * sizeof(float))) goto cleanup;
    (void)unsetenv("DS4_CUDA_MOE_NO_Q4_MMA_TILE16_SM75");
    if (!ds4_gpu_routed_moe_batch_tensor(
            out, gate, up, mid, down, model, model_bytes,
            gate_offset, up_offset, down_offset, 16u, 12u,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, selected, weights,
            n_total_expert, n_expert, 10.0f, x, 0u, n_tokens,
            &mid_is_f16, true) || mid_is_f16 || !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(out, 0, down_candidate,
                             out_count * sizeof(float))) goto cleanup;
    if (memcmp(down_reference, down_candidate,
               out_count * sizeof(float)) != 0) {
        uint64_t first = 0;
        while (first < out_count &&
               memcmp(down_reference + first, down_candidate + first,
                      sizeof(float)) == 0) {
            first++;
        }
        fprintf(stderr,
                "sm75 q4 down tile16 mismatch at %llu: reference=%a candidate=%a\n",
                (unsigned long long)first,
                (double)down_reference[first],
                (double)down_candidate[first]);
        goto cleanup;
    }
    fprintf(stderr,
            "cuda-regression: sm75 q4 down tile16 exact (%llu values)\n",
            (unsigned long long)out_count);

    (void)setenv("DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75", "1", 1);
    if (!ds4_gpu_routed_moe_batch_tensor(
            out, gate, up, mid, down, model, model_bytes,
            gate_offset, up_offset, down_offset, 16u, 12u,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, selected, weights,
            n_total_expert, n_expert, 10.0f, x, 0u, n_tokens,
            &mid_is_f16, true) || mid_is_f16 || !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(out, 0, down_reference,
                             out_count * sizeof(float))) goto cleanup;
    if (memcmp(down_candidate, down_reference,
               out_count * sizeof(float)) != 0) {
        uint64_t first = 0;
        while (first < out_count &&
               memcmp(down_candidate + first, down_reference + first,
                      sizeof(float)) == 0) first++;
        fprintf(stderr,
                "sm75 q4 down tile16 scalar mismatch at %llu: reference=%a candidate=%a\n",
                (unsigned long long)first,
                (double)down_candidate[first],
                (double)down_reference[first]);
        goto cleanup;
    }
    fprintf(stderr,
            "cuda-regression: sm75 q4 down tile16 scalar exact (%llu values)\n",
            (unsigned long long)out_count);
    (void)setenv("DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75", "0", 1);
    (void)unsetenv("DS4_CUDA_MOE_MIXED_TAIL_TILES");
    rc = 0;

cleanup:
    if (model && !retire_temporary_model_map()) rc = 1;
    (void)unsetenv("DS4_CUDA_MOE_NO_IQ2_MMA_SM75");
    (void)unsetenv("DS4_CUDA_MOE_NO_IQ2_MMA_TILE16_SM75");
    (void)unsetenv("DS4_CUDA_MOE_IQ2_STAGE6_SM75");
    (void)unsetenv("DS4_CUDA_MOE_IQ2_STAGE4_SM75");
    (void)setenv("DS4_CUDA_MOE_IQ2_SCALAR_SM75", "0", 1);
    (void)unsetenv("DS4_CUDA_MOE_NO_Q4_MMA_TILE16_SM75");
    (void)setenv("DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75", "0", 1);
    (void)unsetenv("DS4_CUDA_MOE_WRITE_GATE_UP");
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(x);
    free(candidate);
    free(reference);
    free(down_candidate);
    free(down_reference);
    free(weights_host);
    free(selected_host);
    free(x_host);
    free(model);
    return rc;
}

static int compare_exact_f32(const char *label, const float *reference,
                             const float *candidate, uint64_t count) {
    if (memcmp(reference, candidate, (size_t)count * sizeof(float)) == 0)
        return 1;
    uint64_t first = 0;
    while (first < count &&
           memcmp(reference + first, candidate + first,
                  sizeof(float)) == 0) first++;
    fprintf(stderr, "%s mismatch at %llu: reference=%a candidate=%a\n",
            label, (unsigned long long)first,
            (double)reference[first], (double)candidate[first]);
    return 0;
}

/* One-expert-per-token hybrid fixture isolates every atomic output, making
 * the Q2 baseline and IMMA result bitwise comparable while uneven expert
 * populations exercise 16/8/4 tails. */
static int check_sm75_q4_q2_next_targets_exact(void) {
    const uint32_t n_total_expert = 8, n_expert = 1, n_tokens = 128;
    const uint32_t in_dim = 4096, mid_dim = 256, out_dim = 256;
    const uint32_t gate_blocks = in_dim / 256u;
    const uint32_t down_blocks = mid_dim / 256u;
    const uint64_t gate_row_bytes = (uint64_t)gate_blocks * 144u;
    const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = (uint64_t)down_blocks * 84u;
    const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;
    const uint64_t gate_offset = 0u;
    const uint64_t up_offset = gate_expert_bytes * n_total_expert;
    const uint64_t down_offset = up_offset + gate_expert_bytes * n_total_expert;
    const uint64_t model_bytes = down_offset + down_expert_bytes * n_total_expert;
    const uint64_t pair_count = (uint64_t)n_tokens * n_expert;
    const uint64_t x_count = (uint64_t)n_tokens * in_dim;
    const uint64_t mid_count = pair_count * mid_dim;
    const uint64_t out_count = (uint64_t)n_tokens * out_dim;
    const uint64_t down_output_bytes = pair_count * out_dim * sizeof(float);
    const uint64_t x_scratch_bound = x_count * sizeof(float);
    const uint64_t down_storage_bytes =
        down_output_bytes > x_scratch_bound ? down_output_bytes : x_scratch_bound;
    unsigned char *model = (unsigned char *)calloc(1, (size_t)model_bytes);
    float *x_host = (float *)malloc((size_t)x_count * sizeof(float));
    int32_t *selected_host = (int32_t *)malloc((size_t)pair_count * sizeof(int32_t));
    float *weights_host = (float *)malloc((size_t)pair_count * sizeof(float));
    float *mid_reference = (float *)malloc((size_t)mid_count * sizeof(float));
    float *mid_candidate = (float *)malloc((size_t)mid_count * sizeof(float));
    float *out_reference = (float *)malloc((size_t)out_count * sizeof(float));
    float *out_candidate = (float *)malloc((size_t)out_count * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(pair_count * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(down_storage_bytes);
    int rc = 1;
    if (!model || !x_host || !selected_host || !weights_host ||
        !mid_reference || !mid_candidate || !out_reference || !out_candidate ||
        !x || !selected || !weights || !out || !gate || !up || !mid || !down)
        goto cleanup;

    for (uint32_t matrix = 0; matrix < 2u; matrix++) {
        unsigned char *base = model + (matrix ? up_offset : gate_offset);
        for (uint32_t e = 0; e < n_total_expert; e++) {
            for (uint32_t row = 0; row < mid_dim; row++) {
                for (uint32_t b = 0; b < gate_blocks; b++) {
                    unsigned char *blk = base + (uint64_t)e * gate_expert_bytes +
                        (uint64_t)row * gate_row_bytes + (uint64_t)b * 144u;
                    const uint16_t d = 0x2000u, dmin = 0x1800u;
                    memcpy(blk, &d, 2u); memcpy(blk + 2u, &dmin, 2u);
                    for (uint32_t i = 0; i < 12u; i++)
                        blk[4u + i] = (unsigned char)(
                            (e * 29u + row * 11u + b * 7u +
                             i * 13u + matrix * 17u) & 0xffu);
                    for (uint32_t i = 0; i < 128u; i++)
                        blk[16u + i] = (unsigned char)(
                            (e * 17u + row * 23u + b * 31u +
                             i * 5u + matrix * 19u) & 0xffu);
                }
            }
        }
    }
    for (uint32_t e = 0; e < n_total_expert; e++) {
        for (uint32_t row = 0; row < out_dim; row++) {
            unsigned char *blk = model + down_offset +
                (uint64_t)e * down_expert_bytes +
                (uint64_t)row * down_row_bytes;
            for (uint32_t i = 0; i < 16u; i++)
                blk[i] = (unsigned char)(
                    ((e + row + i) & 15u) | (((e * 3u + row + i * 5u) & 15u) << 4u));
            for (uint32_t i = 0; i < 64u; i++)
                blk[16u + i] = (unsigned char)(
                    e * 13u + row * 7u + i * 11u);
            const uint16_t d = 0x2000u, dmin = 0x1800u;
            memcpy(blk + 80u, &d, 2u);
            memcpy(blk + 82u, &dmin, 2u);
        }
    }
    for (uint64_t i = 0; i < x_count; i++) {
        const int v = (int)((i * 23u + (i >> 5u) * 17u) % 257u) - 128;
        x_host[i] = (float)v / 133.0f;
    }
    /* Counts 25,25,20,20,19,19 exercise 16+8+4 and 16+4 tails. */
    const uint32_t cuts[6] = {25u, 50u, 70u, 90u, 109u, 128u};
    for (uint32_t t = 0; t < n_tokens; t++) {
        uint32_t e = 0u;
        while (e < 5u && t >= cuts[e]) e++;
        selected_host[t] = (int32_t)e;
        weights_host[t] = (float)((t % 7u) + 1u) / 8.0f;
    }
    if (!ds4_gpu_tensor_write(x, 0, x_host, x_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(selected, 0, selected_host,
                              pair_count * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(weights, 0, weights_host,
                              pair_count * sizeof(float)) ||
        !ds4_gpu_set_model_map(model, model_bytes)) goto cleanup;

    bool mid_is_f16 = false;
#define RUN_Q4_Q2_TARGET(MID_DST, OUT_DST) \
    (ds4_gpu_routed_moe_batch_tensor( \
        out, gate, up, mid, down, model, model_bytes, \
        gate_offset, up_offset, down_offset, 12u, 10u, \
        gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes, \
        in_dim, mid_dim, out_dim, selected, weights, n_total_expert, n_expert, \
        10.0f, x, 0u, n_tokens, &mid_is_f16, true) && !mid_is_f16 && \
     ds4_gpu_synchronize() && \
     ds4_gpu_tensor_read(mid, 0, MID_DST, mid_count * sizeof(float)) && \
     ds4_gpu_tensor_read(out, 0, OUT_DST, out_count * sizeof(float)))

    if (!RUN_Q4_Q2_TARGET(mid_reference, out_reference)) goto cleanup;
    (void)setenv("DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75", "1", 1);
    if (!RUN_Q4_Q2_TARGET(mid_candidate, out_candidate) ||
        !compare_exact_f32("sm75 q4 gate tile8 scalar", mid_reference,
                           mid_candidate, mid_count) ||
        !compare_exact_f32("sm75 q4 gate tile8 scalar output", out_reference,
                           out_candidate, out_count)) goto cleanup;
    (void)setenv("DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75", "0", 1);
    fprintf(stderr, "cuda-regression: sm75 q4 gate tile8 scalar exact\n");

    (void)setenv("DS4_CUDA_MOE_Q4_GATE_TILE16_SM75", "1", 1);
    if (!RUN_Q4_Q2_TARGET(mid_candidate, out_candidate) ||
        !compare_exact_f32("sm75 q4 gate stage6", mid_reference,
                           mid_candidate, mid_count) ||
        !compare_exact_f32("sm75 q4 gate stage6 output", out_reference,
                           out_candidate, out_count)) goto cleanup;
    fprintf(stderr, "cuda-regression: sm75 q4 gate tile16 stage6 exact\n");

    (void)setenv("DS4_CUDA_MOE_Q4_GATE_STAGE4_SM75", "1", 1);
    if (!RUN_Q4_Q2_TARGET(mid_candidate, out_candidate) ||
        !compare_exact_f32("sm75 q4 gate stage4", mid_reference,
                           mid_candidate, mid_count)) goto cleanup;
    (void)unsetenv("DS4_CUDA_MOE_Q4_GATE_STAGE4_SM75");
    fprintf(stderr, "cuda-regression: sm75 q4 gate tile16 stage4 exact\n");

    (void)setenv("DS4_CUDA_MOE_Q2_DOWN_MMA_SM75", "1", 1);
    if (!RUN_Q4_Q2_TARGET(mid_candidate, out_candidate) ||
        !compare_exact_f32("sm75 q2 down mma", out_reference,
                           out_candidate, out_count)) goto cleanup;
    fprintf(stderr, "cuda-regression: sm75 q2 down m8n8k16 exact\n");

    (void)setenv("DS4_CUDA_MOE_MIXED_TAIL_TILES", "1", 1);
    if (!RUN_Q4_Q2_TARGET(mid_candidate, out_candidate) ||
        !compare_exact_f32("sm75 mixed tails mid", mid_reference,
                           mid_candidate, mid_count) ||
        !compare_exact_f32("sm75 mixed tails output", out_reference,
                           out_candidate, out_count)) goto cleanup;
    fprintf(stderr, "cuda-regression: sm75 expert 16/8/4 tails exact\n");

    /* With Q4 tile16 and mixed tails held fixed, this flag changes only the
     * standard-layout Q4 tile8 specialization used for true 8-slot tails. */
    (void)setenv("DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75", "1", 1);
    if (!RUN_Q4_Q2_TARGET(mid_candidate, out_candidate) ||
        !compare_exact_f32("sm75 q4 mixed-tail scalar", mid_reference,
                           mid_candidate, mid_count) ||
        !compare_exact_f32("sm75 q4 mixed-tail scalar output", out_reference,
                           out_candidate, out_count)) goto cleanup;
    (void)setenv("DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75", "0", 1);
    fprintf(stderr,
            "cuda-regression: sm75 q4 mixed-tail tile8 scalar exact\n");
    rc = 0;
#undef RUN_Q4_Q2_TARGET

cleanup:
    if (model && !retire_temporary_model_map()) rc = 1;
    (void)setenv("DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75", "0", 1);
    (void)unsetenv("DS4_CUDA_MOE_Q4_GATE_TILE16_SM75");
    (void)unsetenv("DS4_CUDA_MOE_Q4_GATE_STAGE4_SM75");
    (void)unsetenv("DS4_CUDA_MOE_Q2_DOWN_MMA_SM75");
    (void)unsetenv("DS4_CUDA_MOE_MIXED_TAIL_TILES");
    ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(x);
    free(out_candidate); free(out_reference);
    free(mid_candidate); free(mid_reference);
    free(weights_host); free(selected_host); free(x_host); free(model);
    return rc;
}

static void pack_sm75_native_q4_tensor(unsigned char *dst,
                                        const unsigned char *src,
                                        uint32_t n_experts,
                                        uint32_t n_rows,
                                        uint32_t n_blocks) {
    const uint64_t row_bytes = (uint64_t)n_blocks * 144u;
    const uint64_t expert_bytes = (uint64_t)n_rows * row_bytes;
    for (uint32_t expert = 0; expert < n_experts; expert++) {
        for (uint32_t tile = 0; tile < n_rows / 8u; tile++) {
            for (uint32_t block = 0; block < n_blocks; block++) {
                unsigned char *record = dst +
                    (uint64_t)expert * expert_bytes +
                    ((uint64_t)tile * n_blocks + block) * 8u * 144u;
                for (uint32_t row = 0; row < 8u; row++) {
                    const unsigned char *q4 = src +
                        (uint64_t)expert * expert_bytes +
                        (uint64_t)(tile * 8u + row) * row_bytes +
                        (uint64_t)block * 144u;
                    memcpy(record + row * 16u, q4, 16u);
                }
                uint32_t *mma = (uint32_t *)(record + 8u * 16u);
                for (uint32_t lane = 0; lane < 32u; lane++) {
                    const uint32_t row = lane >> 2u;
                    const uint32_t lane4 = lane & 3u;
                    const unsigned char *qs = src +
                        (uint64_t)expert * expert_bytes +
                        (uint64_t)(tile * 8u + row) * row_bytes +
                        (uint64_t)block * 144u + 16u;
                    for (uint32_t group = 0; group < 8u; group++) {
                        const uint32_t off = (group >> 1u) * 32u + lane4 * 8u;
                        const uint32_t shift = (group & 1u) ? 4u : 0u;
                        uint32_t packed = 0u;
                        for (uint32_t i = 0; i < 4u; i++) {
                            packed |= ((uint32_t)((qs[off + i] >> shift) & 15u))
                                << (4u * i);
                            packed |= ((uint32_t)((qs[off + 4u + i] >> shift) & 15u))
                                << (4u * (i + 4u));
                        }
                        mma[group * 32u + lane] = packed;
                    }
                }
            }
        }
    }
}

static void fill_q4_tensor(unsigned char *base, uint32_t matrix,
                           uint32_t n_experts, uint32_t n_rows,
                           uint32_t n_blocks) {
    const uint64_t row_bytes = (uint64_t)n_blocks * 144u;
    const uint64_t expert_bytes = (uint64_t)n_rows * row_bytes;
    for (uint32_t e = 0; e < n_experts; e++) {
        for (uint32_t row = 0; row < n_rows; row++) {
            for (uint32_t b = 0; b < n_blocks; b++) {
                unsigned char *blk = base + (uint64_t)e * expert_bytes +
                    (uint64_t)row * row_bytes + (uint64_t)b * 144u;
                const uint16_t d = 0x2000u, dmin = 0x1800u;
                memcpy(blk, &d, 2u); memcpy(blk + 2u, &dmin, 2u);
                for (uint32_t i = 0; i < 12u; i++)
                    blk[4u + i] = (unsigned char)(e * 29u + row * 11u +
                        b * 7u + i * 13u + matrix * 17u);
                for (uint32_t i = 0; i < 128u; i++)
                    blk[16u + i] = (unsigned char)(e * 17u + row * 23u +
                        b * 31u + i * 5u + matrix * 19u);
            }
        }
    }
}

static void fill_iq2_tensor(unsigned char *base, uint32_t matrix,
                            uint32_t n_experts, uint32_t n_rows,
                            uint32_t n_blocks) {
    const uint64_t row_bytes = (uint64_t)n_blocks * 66u;
    const uint64_t expert_bytes = (uint64_t)n_rows * row_bytes;
    for (uint32_t e = 0; e < n_experts; e++) {
        for (uint32_t row = 0; row < n_rows; row++) {
            for (uint32_t b = 0; b < n_blocks; b++) {
                unsigned char *blk = base + (uint64_t)e * expert_bytes +
                    (uint64_t)row * row_bytes + (uint64_t)b * 66u;
                blk[0] = 0x00u;
                blk[1] = 0x30u; /* finite FP16 d=0.125 */
                for (uint32_t i = 0; i < 64u; i++)
                    blk[2u + i] = (unsigned char)(e * 17u + row * 13u +
                        b * 29u + i * 7u + matrix * 31u);
            }
        }
    }
}

typedef struct {
    uint16_t d[8];
    uint8_t scales[8][6];
    uint32_t b[8][32];
} test_sm75_q4_32_tile;

typedef struct {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t scales[8][4];
    uint8_t mins[8][4];
    uint16_t low2[8][32];
    uint8_t high[8][32];
} test_sm75_q3a4_tile;

_Static_assert(sizeof(test_sm75_q4_32_tile) == 8u * 136u,
               "test Q4-32 record size");
_Static_assert(sizeof(test_sm75_q3a4_tile) == 8u * 108u,
               "test Q3A4 record size");

static void test_pack_scale6(uint8_t packed[6], uint32_t group, int value) {
    const uint32_t code = (uint32_t)(value + 32) & 63u;
    if (group < 4u) packed[group] |= (uint8_t)(code & 15u);
    else packed[group - 4u] |= (uint8_t)((code & 15u) << 4u);
    packed[4u + (group >> 2u)] |=
        (uint8_t)(((code >> 4u) & 3u) << (2u * (group & 3u)));
}

static void fill_sm75_q4_32_tensor(unsigned char *base, uint32_t matrix,
                                    uint32_t n_experts, uint32_t n_rows,
                                    uint32_t n_blocks) {
    test_sm75_q4_32_tile *tiles = (test_sm75_q4_32_tile *)base;
    for (uint32_t e = 0; e < n_experts; e++) {
        for (uint32_t rt = 0; rt < n_rows / 8u; rt++) {
            for (uint32_t b = 0; b < n_blocks; b++) {
                test_sm75_q4_32_tile *tile = tiles +
                    ((uint64_t)e * (n_rows / 8u) + rt) * n_blocks + b;
                memset(tile, 0, sizeof(*tile));
                for (uint32_t r = 0; r < 8u; r++) {
                    tile->d[r] = 0x2800u; /* 1/32, exact FP16. */
                    for (uint32_t g = 0; g < 8u; g++)
                        test_pack_scale6(tile->scales[r], g,
                            (int)((e + rt + r + b + g + matrix) % 7u) - 3);
                }
                for (uint32_t g = 0; g < 8u; g++) {
                    for (uint32_t lane = 0; lane < 32u; lane++) {
                        uint32_t word = 0u;
                        for (uint32_t i = 0; i < 8u; i++) {
                            const int q = (int)((e * 3u + rt * 5u + b * 7u +
                                g * 11u + lane * 13u + i * 17u +
                                matrix * 19u) % 16u) - 8;
                            word |= ((uint32_t)q & 15u) << (4u * i);
                        }
                        tile->b[g][lane] = word;
                    }
                }
            }
        }
    }
}

/* Make expert 0 / row 0 an adversarial signed-zero probe.  Every gate leaf
 * is mathematically zero with a positive weight scale; the input fixture
 * below forces a negative Q8_K scale in every block, so raw leaves are -0.
 * The control's +0 accumulator and offset-16 inactive-lane add normalize
 * them to +0.  Up uses one strictly positive block so the final SiLU product
 * retains the gate zero's sign. */
static void force_sm75_q4_32_signed_zero_probe(
        unsigned char *gate_base, unsigned char *up_base,
        uint32_t n_blocks) {
    test_sm75_q4_32_tile *gate = (test_sm75_q4_32_tile *)gate_base;
    test_sm75_q4_32_tile *up = (test_sm75_q4_32_tile *)up_base;
    for (uint32_t b = 0; b < n_blocks; b++) {
        test_sm75_q4_32_tile *gw = gate + b;
        test_sm75_q4_32_tile *uw = up + b;
        gw->d[0] = 0x2800u;
        uw->d[0] = 0x2800u;
        memset(gw->scales[0], 0, sizeof(gw->scales[0]));
        memset(uw->scales[0], 0, sizeof(uw->scales[0]));
        for (uint32_t g = 0; g < 8u; g++) {
            test_pack_scale6(gw->scales[0], g, 1);
            test_pack_scale6(uw->scales[0], g, 1);
            for (uint32_t lane4 = 0; lane4 < 4u; lane4++) {
                gw->b[g][lane4] = 0u;
                uw->b[g][lane4] = b == 0u ? 0x11111111u : 0u;
            }
        }
    }
}

static void fill_sm75_q3a4_tensor(unsigned char *base, uint32_t matrix,
                                   uint32_t n_experts, uint32_t n_rows,
                                   uint32_t n_blocks) {
    test_sm75_q3a4_tile *tiles = (test_sm75_q3a4_tile *)base;
    for (uint32_t e = 0; e < n_experts; e++) {
        for (uint32_t rt = 0; rt < n_rows / 8u; rt++) {
            for (uint32_t b = 0; b < n_blocks; b++) {
                test_sm75_q3a4_tile *tile = tiles +
                    ((uint64_t)e * (n_rows / 8u) + rt) * n_blocks + b;
                memset(tile, 0, sizeof(*tile));
                for (uint32_t r = 0; r < 8u; r++) {
                    tile->d[r] = 0x2800u;
                    tile->dmin[r] = 0x2400u;
                    for (uint32_t g = 0; g < 8u; g++) {
                        const uint8_t scale =
                            (uint8_t)(1u + (e + rt + r + g + matrix) % 7u);
                        const uint8_t minv =
                            (uint8_t)((e + b + r + 2u * g + matrix) % 5u);
                        tile->scales[r][g >> 1u] |=
                            (uint8_t)(scale << (4u * (g & 1u)));
                        tile->mins[r][g >> 1u] |=
                            (uint8_t)(minv << (4u * (g & 1u)));
                    }
                }
                for (uint32_t g = 0; g < 8u; g++) {
                    for (uint32_t lane = 0; lane < 32u; lane++) {
                        uint16_t low = 0u;
                        uint8_t high = 0u;
                        for (uint32_t i = 0; i < 8u; i++) {
                            const uint32_t q = (e * 3u + rt * 5u + b * 7u +
                                g * 11u + lane * 13u + i * 17u +
                                matrix * 19u) & 7u;
                            low |= (uint16_t)((q & 3u) << (2u * i));
                            high |= (uint8_t)(((q >> 2u) & 1u) << i);
                        }
                        tile->low2[g][lane] = low;
                        tile->high[g][lane] = high;
                    }
                }
            }
        }
    }
}

/* Compare the real sorted 16/8/4 prefill dispatcher with the independent
 * direct-decode dispatcher, token by token. This covers production offsets,
 * expert selection, native records, Q4-32 down, and both supported gate/up
 * formats without treating an isolated harness kernel as production proof. */
static int check_sm75_q32_production_exact_case(uint32_t gate_type,
                                                const char *label) {
    const uint32_t n_total = 8u, n_expert = 1u, n_tokens = 32u;
    const uint32_t in_dim = 256u, mid_dim = 256u, out_dim = 256u;
    const uint32_t in_blocks = 1u, mid_blocks = 1u;
    const uint64_t gate_block = gate_type == 43u ? 108u : 136u;
    const uint64_t gate_row = gate_block * in_blocks;
    const uint64_t gate_expert = (uint64_t)mid_dim * gate_row;
    const uint64_t down_row = 136u * mid_blocks;
    const uint64_t down_expert = (uint64_t)out_dim * down_row;
    const uint64_t gate_off = 0u;
    const uint64_t up_off = gate_expert * n_total;
    const uint64_t down_off = up_off + gate_expert * n_total;
    const uint64_t model_bytes = down_off + down_expert * n_total;
    const uint64_t x_count = (uint64_t)n_tokens * in_dim;
    const uint64_t mid_count = (uint64_t)n_tokens * mid_dim;
    const uint64_t out_count = (uint64_t)n_tokens * out_dim;
    const uint64_t scratch_bytes = x_count * sizeof(float) >
        out_count * sizeof(float) ? x_count * sizeof(float) :
                                   out_count * sizeof(float);
    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    float *xh = (float *)malloc((size_t)x_count * sizeof(float));
    int32_t *selh = (int32_t *)malloc(n_tokens * sizeof(int32_t));
    float *wh = (float *)malloc(n_tokens * sizeof(float));
    float *mid_ref = (float *)malloc((size_t)mid_count * sizeof(float));
    float *mid_got = (float *)malloc((size_t)mid_count * sizeof(float));
    float *out_ref = (float *)malloc((size_t)out_count * sizeof(float));
    float *out_got = (float *)malloc((size_t)out_count * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(n_tokens * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(n_tokens * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(scratch_bytes);
    int rc = 1;
    if (!model || !xh || !selh || !wh || !mid_ref || !mid_got ||
        !out_ref || !out_got || !x || !selected || !weights || !out ||
        !gate || !up || !mid || !down) goto cleanup;
    if (gate_type == 43u) {
        fill_sm75_q3a4_tensor(model + gate_off, 0u, n_total,
                              mid_dim, in_blocks);
        fill_sm75_q3a4_tensor(model + up_off, 1u, n_total,
                              mid_dim, in_blocks);
    } else {
        fill_sm75_q4_32_tensor(model + gate_off, 0u, n_total,
                               mid_dim, in_blocks);
        fill_sm75_q4_32_tensor(model + up_off, 1u, n_total,
                               mid_dim, in_blocks);
    }
    fill_sm75_q4_32_tensor(model + down_off, 2u, n_total,
                           out_dim, mid_blocks);
    for (uint64_t i = 0; i < x_count; i++)
        xh[i] = (float)((int)((i * 23u + (i >> 3u) * 17u) % 193u) - 96) /
            101.0f;
    /* Exact 16/8/4/4 expert populations force every production tile size. */
    const uint32_t cuts[4] = {16u, 24u, 28u, 32u};
    for (uint32_t t = 0; t < n_tokens; t++) {
        uint32_t e = 0u;
        while (e < 3u && t >= cuts[e]) e++;
        selh[t] = (int32_t)e;
        wh[t] = (float)((t % 7u) + 1u) / 8.0f;
    }
    ds4_gpu_set_routed_q4_layout(DS4_TENSOR_LAYOUT_SM75_Q4_32 |
                                  DS4_TENSOR_LAYOUT_SM75_Q3A4);
    if (!ds4_gpu_set_model_map(model, model_bytes)) goto cleanup;
    bool mid_is_f16 = false;
#define RUN_Q32(NTOK, MID_DST, OUT_DST) \
    (mid_is_f16 = false, \
     ds4_gpu_routed_moe_batch_tensor( \
        out, gate, up, mid, down, model, model_bytes, \
        gate_off, up_off, down_off, gate_type, 42u, \
        gate_expert, gate_row, down_expert, down_row, \
        in_dim, mid_dim, out_dim, selected, weights, n_total, n_expert, \
        10.0f, x, 0u, (NTOK), &mid_is_f16, true) && !mid_is_f16 && \
     ds4_gpu_synchronize() && \
     ds4_gpu_tensor_read(mid, 0, (MID_DST), \
                         (uint64_t)(NTOK) * mid_dim * sizeof(float)) && \
     ds4_gpu_tensor_read(out, 0, (OUT_DST), \
                         (uint64_t)(NTOK) * out_dim * sizeof(float)))
    for (uint32_t t = 0; t < n_tokens; t++) {
        if (!ds4_gpu_tensor_write(x, 0, xh + (uint64_t)t * in_dim,
                                  (uint64_t)in_dim * sizeof(float)) ||
            !ds4_gpu_tensor_write(selected, 0, selh + t, sizeof(int32_t)) ||
            !ds4_gpu_tensor_write(weights, 0, wh + t, sizeof(float)) ||
            !RUN_Q32(1u, mid_ref + (uint64_t)t * mid_dim,
                     out_ref + (uint64_t)t * out_dim)) goto cleanup;
    }
    if (!ds4_gpu_tensor_write(x, 0, xh, x_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(selected, 0, selh,
                              n_tokens * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(weights, 0, wh,
                              n_tokens * sizeof(float)) ||
        !RUN_Q32(n_tokens, mid_got, out_got) ||
        !compare_exact_f32("SM75 Q32 production prefill mid",
                           mid_ref, mid_got, mid_count) ||
        !compare_exact_f32("SM75 Q32 production prefill output",
                           out_ref, out_got, out_count)) goto cleanup;
    fprintf(stderr,
            "cuda-regression: SM75 %s gate/up + Q4-32 down production "
            "16/8/4 prefill/direct-decode exact\n", label);
    rc = 0;
#undef RUN_Q32

cleanup:
    ds4_gpu_set_routed_q4_layout(0u);
    if (model && !retire_temporary_model_map()) rc = 1;
    ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(x);
    free(out_got); free(out_ref); free(mid_got); free(mid_ref);
    free(wh); free(selh); free(xh); free(model);
    return rc;
}

static int check_sm75_q32_production_exact(void) {
    int rc = check_sm75_q32_production_exact_case(42u, "Q4-32");
    if (check_sm75_q32_production_exact_case(43u, "Q3A4") != 0) rc = 1;
    return rc;
}

static uint32_t test_sm75_q4_sign_extend_nibble_bytes(uint32_t v) {
    v &= 0x0f0f0f0fu;
    const uint32_t sign = v & 0x08080808u;
    return v | (sign << 1u) | (sign << 2u) |
               (sign << 3u) | (sign << 4u);
}

/* Exhaustively protect the signed-Q4 byte expansion and signed-Q8 activation
 * packing used by the Q4-32 tile32-DP4A candidate. */
static int check_sm75_q4_32_dp4a_pack(void) {
    const uint32_t nibble_mask = 0x0f0f0f0fu;
    for (uint32_t byte = 0; byte < 256u; byte++) {
        const uint32_t lo = (byte & 15u) * 0x11111111u;
        const uint32_t hi = ((byte >> 4u) & 15u) * 0x11111111u;
        const uint32_t a_even =
            (lo & nibble_mask) | ((hi & nibble_mask) << 4u);
        const uint32_t a_odd =
            ((lo >> 4u) & nibble_mask) | (hi & 0xf0f0f0f0u);
        int activation = (int)byte;
        if (activation >= 128) activation -= 256;
        for (int q = -8; q <= 7; q++) {
            const uint32_t qw = ((uint32_t)q & 15u) * 0x11111111u;
            const uint32_t w_even =
                test_sm75_q4_sign_extend_nibble_bytes(qw);
            const uint32_t w_odd =
                test_sm75_q4_sign_extend_nibble_bytes(qw >> 4u);
            int got = 0;
            for (uint32_t i = 0; i < 4u; i++) {
                int ae = (int)((a_even >> (8u * i)) & 255u);
                int ao = (int)((a_odd >> (8u * i)) & 255u);
                int we = (int)((w_even >> (8u * i)) & 255u);
                int wo = (int)((w_odd >> (8u * i)) & 255u);
                if (ae >= 128) ae -= 256;
                if (ao >= 128) ao -= 256;
                if (we >= 128) we -= 256;
                if (wo >= 128) wo -= 256;
                got += ae * we + ao * wo;
            }
            if (got != 8 * activation * q) {
                fprintf(stderr,
                        "SM75 Q4-32 DP4A pack mismatch byte=%u q=%d: "
                        "got=%d expected=%d\n",
                        byte, q, got, 8 * activation * q);
                return 1;
            }
        }
    }
    fprintf(stderr,
            "cuda-regression: SM75 Q4-32 signed DP4A byte packing exact\n");
    return 0;
}

/* Exhaustively prove the byte packing used by the tile32 DP4A path.  The
 * activation byte is low_nibble | high_signed_nibble<<4; Q3A4 weights are
 * non-negative 0..7 bytes. */
static int check_sm75_q3a4_dp4a_pack(void) {
    const uint32_t nibble_mask = 0x0f0f0f0fu;
    for (uint32_t byte = 0; byte < 256u; byte++) {
        const uint32_t lo = (byte & 15u) * 0x11111111u;
        const uint32_t hi = ((byte >> 4u) & 15u) * 0x11111111u;
        const uint32_t a_even =
            (lo & nibble_mask) | ((hi & nibble_mask) << 4u);
        const uint32_t a_odd =
            ((lo >> 4u) & nibble_mask) | (hi & 0xf0f0f0f0u);
        int activation = (int)byte;
        if (activation >= 128) activation -= 256;
        for (uint32_t q = 0; q < 8u; q++) {
            const uint32_t qw = q * 0x11111111u;
            const uint32_t w_even = qw & nibble_mask;
            const uint32_t w_odd = (qw >> 4u) & nibble_mask;
            int got = 0;
            for (uint32_t i = 0; i < 4u; i++) {
                int ae = (int)((a_even >> (8u * i)) & 255u);
                int ao = (int)((a_odd >> (8u * i)) & 255u);
                if (ae >= 128) ae -= 256;
                if (ao >= 128) ao -= 256;
                got += ae * (int)((w_even >> (8u * i)) & 255u);
                got += ao * (int)((w_odd >> (8u * i)) & 255u);
            }
            if (got != 8 * activation * (int)q) {
                fprintf(stderr,
                        "SM75 Q3A4 DP4A pack mismatch byte=%u q=%u: "
                        "got=%d expected=%d\n",
                        byte, q, got, 8 * activation * (int)q);
                return 1;
            }
        }
    }
    fprintf(stderr,
            "cuda-regression: SM75 Q3A4 DP4A byte packing exact\n");
    return 0;
}

static int require_nonzero_f32(const char *label, const float *values,
                               uint64_t count) {
    for (uint64_t i = 0; i < count; i++)
        if (values[i] != 0.0f) return 1;
    fprintf(stderr, "%s unexpectedly contains only zero values\n", label);
    return 0;
}

static void fill_sm75_q32_poison_f32(float *values, uint64_t count) {
    const uint32_t poison = 0x7fc1d5a4u;
    for (uint64_t i = 0; i < count; i++)
        memcpy(values + i, &poison, sizeof(poison));
}

static int require_sm75_q32_overwritten_f32(
        const char *label, const float *values, uint64_t count) {
    const uint32_t poison = 0x7fc1d5a4u;
    for (uint64_t i = 0; i < count; i++) {
        uint32_t bits = 0u;
        memcpy(&bits, values + i, sizeof(bits));
        if (bits == poison) {
            fprintf(stderr, "%s retained poison at %llu\n", label,
                    (unsigned long long)i);
            return 0;
        }
    }
    return 1;
}

static int require_sm75_q32_f32_bits(
        const char *label, float value, uint32_t expected) {
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    if (bits == expected) return 1;
    fprintf(stderr, "%s bits=0x%08x expected=0x%08x\n",
            label, bits, expected);
    return 0;
}

static int check_sm75_q3a4_ksplit_env(void) {
    int rc = 1;
    if (setenv("DS4_CUDA_MOE_Q3A4_DECODE_MAPPING", "tile32-dp4a", 1) != 0 ||
        setenv("DS4_CUDA_MOE_Q3A4_DECODE_KSPLIT", "1", 1) != 0 ||
        setenv("DS4_CUDA_MOE_Q3A4_DECODE_PREFETCH_DEPTH", "2", 1) != 0)
        goto cleanup;
    ds4_gpu_test_refresh_decode_dispatch_env();
    if (ds4_gpu_test_get_moe_q3a4_decode_mapping() != 3u ||
        ds4_gpu_test_get_moe_q3a4_decode_ksplit() != 1u ||
        ds4_gpu_test_get_moe_q3a4_decode_prefetch_depth() != 0u)
        goto cleanup;

    if (setenv("DS4_CUDA_MOE_Q3A4_DECODE_KSPLIT", "2", 1) != 0)
        goto cleanup;
    ds4_gpu_test_refresh_decode_dispatch_env();
    if (ds4_gpu_test_get_moe_q3a4_decode_mapping() != 3u ||
        ds4_gpu_test_get_moe_q3a4_decode_ksplit() != 2u ||
        ds4_gpu_test_get_moe_q3a4_decode_prefetch_depth() != 0u)
        goto cleanup;

    if (setenv("DS4_CUDA_MOE_Q3A4_DECODE_KSPLIT", "4", 1) != 0 ||
        setenv("DS4_CUDA_MOE_Q3A4_DECODE_PREFETCH_DEPTH", "0", 1) != 0)
        goto cleanup;
    ds4_gpu_test_refresh_decode_dispatch_env();
    if (ds4_gpu_test_get_moe_q3a4_decode_mapping() != 3u ||
        ds4_gpu_test_get_moe_q3a4_decode_ksplit() != 4u ||
        ds4_gpu_test_get_moe_q3a4_decode_prefetch_depth() != 0u)
        goto cleanup;

    if (setenv("DS4_CUDA_MOE_Q3A4_DECODE_PREFETCH_DEPTH", "2", 1) != 0)
        goto cleanup;
    ds4_gpu_test_refresh_decode_dispatch_env();
    if (ds4_gpu_test_get_moe_q3a4_decode_mapping() != 3u ||
        ds4_gpu_test_get_moe_q3a4_decode_ksplit() != 4u ||
        ds4_gpu_test_get_moe_q3a4_decode_prefetch_depth() != 2u)
        goto cleanup;

    if (setenv("DS4_CUDA_MOE_Q3A4_DECODE_PREFETCH_DEPTH", "1", 1) != 0)
        goto cleanup;
    ds4_gpu_test_refresh_decode_dispatch_env();
    if (ds4_gpu_test_get_moe_q3a4_decode_mapping() != 3u ||
        ds4_gpu_test_get_moe_q3a4_decode_ksplit() != 4u ||
        ds4_gpu_test_get_moe_q3a4_decode_prefetch_depth() != 0u)
        goto cleanup;

    if (setenv("DS4_CUDA_MOE_Q3A4_DECODE_KSPLIT", "3", 1) != 0 ||
        setenv("DS4_CUDA_MOE_Q3A4_DECODE_PREFETCH_DEPTH", "2", 1) != 0)
        goto cleanup;
    ds4_gpu_test_refresh_decode_dispatch_env();
    if (ds4_gpu_test_get_moe_q3a4_decode_mapping() != 3u ||
        ds4_gpu_test_get_moe_q3a4_decode_ksplit() != 4u ||
        ds4_gpu_test_get_moe_q3a4_decode_prefetch_depth() != 0u)
        goto cleanup;

    if (setenv("DS4_CUDA_MOE_Q3A4_DECODE_KSPLIT", "4", 1) != 0 ||
        setenv("DS4_CUDA_MOE_Q3A4_DECODE_PREFETCH_DEPTH", "2", 1) != 0 ||
        setenv("DS4_CUDA_NO_MOE_Q3A4_DECODE_MAPPING", "1", 1) != 0)
        goto cleanup;
    ds4_gpu_test_refresh_decode_dispatch_env();
    if (ds4_gpu_test_get_moe_q3a4_decode_mapping() != 0u ||
        ds4_gpu_test_get_moe_q3a4_decode_ksplit() != 1u ||
        ds4_gpu_test_get_moe_q3a4_decode_prefetch_depth() != 0u)
        goto cleanup;
    rc = 0;

cleanup:
    (void)unsetenv("DS4_CUDA_MOE_Q3A4_DECODE_MAPPING");
    (void)unsetenv("DS4_CUDA_MOE_Q3A4_DECODE_KSPLIT");
    (void)unsetenv("DS4_CUDA_MOE_Q3A4_DECODE_PREFETCH_DEPTH");
    (void)unsetenv("DS4_CUDA_NO_MOE_Q3A4_DECODE_MAPPING");
    ds4_gpu_test_refresh_decode_dispatch_env();
    if (rc == 0 &&
        (ds4_gpu_test_get_moe_q3a4_decode_mapping() != 3u ||
         ds4_gpu_test_get_moe_q3a4_decode_ksplit() != 4u ||
         ds4_gpu_test_get_moe_q3a4_decode_prefetch_depth() != 0u))
        rc = 1;
    if (rc == 0) {
        fputs("cuda-regression: SM75 Q3A4 K1/K2/K4 environment selector exact\n",
              stderr);
        fputs("cuda-regression: SM75 Q3A4 K4 prefetch depth 0/2 "
              "environment selector exact\n",
              stderr);
    } else {
        fputs("cuda-regression: SM75 Q3A4 K-split/prefetch environment "
              "selector failed\n",
              stderr);
    }
    return rc;
}

static int check_sm75_q4_32_mapping_env(void) {
    static const struct {
        const char *name;
        uint32_t expected;
    } cases[] = {
        {"control", 0u},
        {"hwarp16", 1u},
        {"tile32-dp4a", 2u},
        {"tile32-mma", 3u},
        {"invalid", 0u},
    };
    int rc = 0;
    for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (setenv("DS4_CUDA_MOE_Q4_32_DECODE_MAPPING",
                   cases[i].name, 1) != 0) return 1;
        ds4_gpu_test_refresh_decode_dispatch_env();
        if (ds4_gpu_test_get_moe_q4_32_decode_mapping() !=
                cases[i].expected) {
            rc = 1;
            break;
        }
    }
    (void)unsetenv("DS4_CUDA_MOE_Q4_32_DECODE_MAPPING");
    ds4_gpu_test_refresh_decode_dispatch_env();
    if (ds4_gpu_test_get_moe_q4_32_decode_mapping() != 0u) rc = 1;
    fputs(rc == 0
              ? "cuda-regression: SM75 Q4-32 audit mapping selector exact/default-off\n"
              : "cuda-regression: SM75 Q4-32 audit mapping selector failed\n",
          stderr);
    return rc;
}

/* Exercise the exact six-node production CUDA Graph through the real
 * owned-expert API.  The full-expert dispatcher is the independent reference;
 * home slots plus fixed-three partner output are combined exactly as ds4.c
 * does for the two-GPU decode path.  A second input validates executable
 * reuse rather than only graph instantiation. */
static int check_sm75_q32_owned_graph_case(uint32_t gate_type,
                                           const char *label) {
    const uint32_t n_total = 8u, n_expert = 6u;
    /* Use the real decode K shape so Q3A4 native mappings are validated with
     * all 16 K256 records, not merely a one-record short test. */
    const uint32_t in_dim = 4096u, mid_dim = 256u, out_dim = 256u;
    const uint32_t in_blocks = 16u, mid_blocks = 1u;
    const uint64_t gate_block = gate_type == 43u ? 108u : 136u;
    const uint64_t gate_row = gate_block * in_blocks;
    const uint64_t gate_expert = (uint64_t)mid_dim * gate_row;
    const uint64_t down_row = 136u * mid_blocks;
    const uint64_t down_expert = (uint64_t)out_dim * down_row;
    const uint64_t gate_off = 0u;
    const uint64_t up_off = gate_expert * n_total;
    const uint64_t down_off = up_off + gate_expert * n_total;
    const uint64_t model_bytes = down_off + down_expert * n_total;
    const uint64_t slot_bytes = 6ull * out_dim * sizeof(float);
    const uint64_t packed_bytes = 4ull * out_dim * sizeof(float);
    const uint64_t mid_bytes = 6ull * mid_dim * sizeof(float);
    const uint64_t shared_prequant_bytes =
        (uint64_t)(in_dim / 32u) * 32u +
        (uint64_t)(in_dim / 32u) * sizeof(float);
    unsigned char *model = (unsigned char *)malloc((size_t)model_bytes);
    float *xh = (float *)malloc((size_t)in_dim * sizeof(float));
    float *refh = (float *)malloc((size_t)out_dim * sizeof(float));
    float *goth = (float *)malloc((size_t)out_dim * sizeof(float));
    float *mid_refh = (float *)malloc((size_t)mid_bytes);
    float *mid_splith = (float *)malloc((size_t)mid_bytes);
    float *home_refh = (float *)malloc((size_t)slot_bytes);
    float *home_splith = (float *)malloc((size_t)slot_bytes);
    const int32_t selh[6] = {0, 4, 1, 5, 2, 6};
    const float wh[6] = {0.125f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f};
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(sizeof(selh));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(sizeof(wh));
    ds4_gpu_tensor *ref = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *got = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *tmp_out = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(slot_bytes);
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(slot_bytes);
    ds4_gpu_tensor *home_slots = ds4_gpu_tensor_alloc(slot_bytes);
    ds4_gpu_tensor *peer_packed = ds4_gpu_tensor_alloc(packed_bytes);
    ds4_gpu_tensor *shared_prequant =
        ds4_gpu_tensor_alloc(shared_prequant_bytes);
    int rc = 1;
    if (!model || !xh || !refh || !goth || !mid_refh || !mid_splith ||
        !home_refh || !home_splith ||
        !x || !selected || !weights ||
        !ref || !got || !tmp_out || !gate || !up || !mid || !down ||
        !home_slots || !peer_packed || !shared_prequant) goto cleanup;
    if (gate_type == 43u) {
        fill_sm75_q3a4_tensor(model + gate_off, 0u, n_total,
                              mid_dim, in_blocks);
        fill_sm75_q3a4_tensor(model + up_off, 1u, n_total,
                              mid_dim, in_blocks);
    } else {
        fill_sm75_q4_32_tensor(model + gate_off, 0u, n_total,
                               mid_dim, in_blocks);
        fill_sm75_q4_32_tensor(model + up_off, 1u, n_total,
                               mid_dim, in_blocks);
        force_sm75_q4_32_signed_zero_probe(
            model + gate_off, model + up_off, in_blocks);
    }
    fill_sm75_q4_32_tensor(model + down_off, 2u, n_total,
                           out_dim, mid_blocks);
    ds4_gpu_set_routed_q4_layout(DS4_TENSOR_LAYOUT_SM75_Q4_32 |
                                  DS4_TENSOR_LAYOUT_SM75_Q3A4);
    if (!ds4_gpu_set_model_map(model, model_bytes) ||
        !ds4_gpu_tensor_write(selected, 0, selh, sizeof(selh)) ||
        !ds4_gpu_tensor_write(weights, 0, wh, sizeof(wh))) goto cleanup;

    for (uint32_t pass = 0; pass < 2u; pass++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            xh[i] = i < 256u
                ? (float)(1u + (i * 29u + pass * 37u) % 211u) / 211.0f
                : (float)((int)((i * 29u + pass * 37u) % 211u) - 105) /
                    109.0f;
        }
        for (uint32_t b = 1u; b < in_blocks; b++)
            xh[b * 256u] = 2.0f + (float)pass / 16.0f;
        if (!ds4_gpu_tensor_write(x, 0, xh, in_dim * sizeof(float)))
            goto cleanup;

        /* Compare at the gate/up boundary before gate scratch is reused for
         * mid quantization.  This directly proves the two projection kernels
         * plus exact combine preserve every output byte, including +0 for
         * partner-owned slots. */
        ds4_gpu_test_set_moe_q32_decode_graph(0);
        ds4_gpu_test_set_moe_q32_decode_split(0);
        ds4_gpu_test_set_moe_q32_decode_fused_lowreg(0u);
        ds4_gpu_test_set_moe_q4_32_decode_mapping(0u);
        ds4_gpu_test_set_moe_q3a4_decode_mapping(0u);
        if (!ds4_gpu_routed_moe_one_owned_tensor(
                tmp_out, gate, up, mid, down, model, model_bytes,
                gate_off, up_off, down_off, gate_type, 42u,
                gate_expert, gate_row, down_expert, down_row,
                in_dim, mid_dim, out_dim, selected, weights,
                n_total, n_expert, 0u, 4u, 10.0f, x,
                home_slots, false, shared_prequant) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(mid, 0, mid_refh, mid_bytes) ||
            !ds4_gpu_tensor_read(home_slots, 0, home_refh, slot_bytes) ||
            !require_nonzero_f32("SM75 Q32 native reference intermediate",
                                 mid_refh,
                                 (uint64_t)n_expert * mid_dim) ||
            !require_nonzero_f32("SM75 Q32 native reference owned output",
                                 home_refh,
                                 (uint64_t)n_expert * out_dim) ||
            (gate_type == 42u &&
             !require_sm75_q32_f32_bits(
                 "SM75 Q4-32 signed-zero reference mid[0]",
                 mid_refh[0], 0x00000000u)))
            goto cleanup;
        ds4_gpu_test_set_moe_q32_decode_split(1);
        if (!ds4_gpu_routed_moe_one_owned_tensor(
                tmp_out, gate, up, mid, down, model, model_bytes,
                gate_off, up_off, down_off, gate_type, 42u,
                gate_expert, gate_row, down_expert, down_row,
                in_dim, mid_dim, out_dim, selected, weights,
                n_total, n_expert, 0u, 4u, 10.0f, x,
                home_slots, false, shared_prequant) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(mid, 0, mid_splith, mid_bytes) ||
            !ds4_gpu_tensor_read(home_slots, 0, home_splith, slot_bytes) ||
            !compare_exact_f32("SM75 Q32 split gate/up intermediate",
                               mid_refh, mid_splith,
                               (size_t)n_expert * mid_dim) ||
            !compare_exact_f32("SM75 Q32 split owned output",
                               home_refh, home_splith,
                               (size_t)n_expert * out_dim)) goto cleanup;
        ds4_gpu_test_set_moe_q32_decode_split(0);
        for (uint32_t unroll = 1u; unroll <= 4u; unroll *= 2u) {
            ds4_gpu_test_set_moe_q32_decode_fused_lowreg(unroll);
            if (!ds4_gpu_routed_moe_one_owned_tensor(
                    tmp_out, gate, up, mid, down, model, model_bytes,
                    gate_off, up_off, down_off, gate_type, 42u,
                    gate_expert, gate_row, down_expert, down_row,
                    in_dim, mid_dim, out_dim, selected, weights,
                    n_total, n_expert, 0u, 4u, 10.0f, x,
                    home_slots, false, shared_prequant) ||
                !ds4_gpu_synchronize() ||
                !ds4_gpu_tensor_read(mid, 0, mid_splith, mid_bytes) ||
                !ds4_gpu_tensor_read(home_slots, 0, home_splith, slot_bytes) ||
                !compare_exact_f32("SM75 Q32 fused-lowreg gate/up intermediate",
                                   mid_refh, mid_splith,
                                   (size_t)n_expert * mid_dim) ||
                !compare_exact_f32("SM75 Q32 fused-lowreg owned output",
                                   home_refh, home_splith,
                                   (size_t)n_expert * out_dim)) goto cleanup;
        }
        ds4_gpu_test_set_moe_q32_decode_fused_lowreg(0u);
        if (gate_type == 42u) {
            for (uint32_t mapping = 1u; mapping <= 3u; mapping++) {
                ds4_gpu_test_set_moe_q4_32_decode_mapping(mapping);
                fill_sm75_q32_poison_f32(
                    mid_splith, (uint64_t)n_expert * mid_dim);
                fill_sm75_q32_poison_f32(
                    home_splith, (uint64_t)n_expert * out_dim);
                if (!ds4_gpu_tensor_write(mid, 0, mid_splith, mid_bytes) ||
                    !ds4_gpu_tensor_write(home_slots, 0, home_splith,
                                          slot_bytes) ||
                    !ds4_gpu_routed_moe_one_owned_tensor(
                        tmp_out, gate, up, mid, down, model, model_bytes,
                        gate_off, up_off, down_off, gate_type, 42u,
                        gate_expert, gate_row, down_expert, down_row,
                        in_dim, mid_dim, out_dim, selected, weights,
                        n_total, n_expert, 0u, 4u, 10.0f, x,
                        home_slots, false, shared_prequant) ||
                    !ds4_gpu_synchronize() ||
                    !ds4_gpu_tensor_read(mid, 0, mid_splith, mid_bytes) ||
                    !ds4_gpu_tensor_read(home_slots, 0, home_splith,
                                         slot_bytes) ||
                    !require_sm75_q32_overwritten_f32(
                        "SM75 Q4-32 native mapping intermediate",
                        mid_splith, (uint64_t)n_expert * mid_dim) ||
                    !require_sm75_q32_overwritten_f32(
                        "SM75 Q4-32 native mapping owned output",
                        home_splith, (uint64_t)n_expert * out_dim) ||
                    !require_nonzero_f32(
                        "SM75 Q4-32 native mapping intermediate",
                        mid_splith, (uint64_t)n_expert * mid_dim) ||
                    !require_nonzero_f32(
                        "SM75 Q4-32 native mapping owned output",
                        home_splith, (uint64_t)n_expert * out_dim) ||
                    !compare_exact_f32(
                        "SM75 Q4-32 native mapping gate/up intermediate",
                        mid_refh, mid_splith,
                        (size_t)n_expert * mid_dim) ||
                    !compare_exact_f32(
                        "SM75 Q4-32 native mapping owned output",
                        home_refh, home_splith,
                        (size_t)n_expert * out_dim)) goto cleanup;
            }
            ds4_gpu_test_set_moe_q4_32_decode_mapping(0u);
        } else if (gate_type == 43u) {
            ds4_gpu_test_set_moe_q3a4_decode_ksplit(1u);
            for (uint32_t mapping = 1u; mapping <= 3u; mapping++) {
                ds4_gpu_test_set_moe_q3a4_decode_mapping(mapping);
                if (!ds4_gpu_routed_moe_one_owned_tensor(
                        tmp_out, gate, up, mid, down, model, model_bytes,
                        gate_off, up_off, down_off, gate_type, 42u,
                        gate_expert, gate_row, down_expert, down_row,
                        in_dim, mid_dim, out_dim, selected, weights,
                        n_total, n_expert, 0u, 4u, 10.0f, x,
                        home_slots, false, shared_prequant) ||
                    !ds4_gpu_synchronize() ||
                    !ds4_gpu_tensor_read(mid, 0, mid_splith, mid_bytes) ||
                    !ds4_gpu_tensor_read(home_slots, 0, home_splith,
                                         slot_bytes) ||
                    !compare_exact_f32(
                        "SM75 Q3A4 native mapping gate/up intermediate",
                        mid_refh, mid_splith,
                        (size_t)n_expert * mid_dim) ||
                    !compare_exact_f32(
                        "SM75 Q3A4 native mapping owned output",
                        home_refh, home_splith,
                        (size_t)n_expert * out_dim)) goto cleanup;
            }
            /* K2/K4 retain mapping 3 and only redistribute its independent
             * K256 leaves among cooperating warps in the same CTA. */
            ds4_gpu_test_set_moe_q3a4_decode_mapping(3u);
            for (uint32_t ksplit = 2u; ksplit <= 4u; ksplit *= 2u) {
                ds4_gpu_test_set_moe_q3a4_decode_ksplit(ksplit);
                if (!ds4_gpu_routed_moe_one_owned_tensor(
                        tmp_out, gate, up, mid, down, model, model_bytes,
                        gate_off, up_off, down_off, gate_type, 42u,
                        gate_expert, gate_row, down_expert, down_row,
                        in_dim, mid_dim, out_dim, selected, weights,
                        n_total, n_expert, 0u, 4u, 10.0f, x,
                        home_slots, false, shared_prequant) ||
                    !ds4_gpu_synchronize() ||
                    !ds4_gpu_tensor_read(mid, 0, mid_splith, mid_bytes) ||
                    !ds4_gpu_tensor_read(home_slots, 0, home_splith,
                                         slot_bytes) ||
                    !compare_exact_f32(
                        "SM75 Q3A4 K-split gate/up intermediate",
                        mid_refh, mid_splith,
                        (size_t)n_expert * mid_dim) ||
                    !compare_exact_f32(
                        "SM75 Q3A4 K-split owned output",
                        home_refh, home_splith,
                        (size_t)n_expert * out_dim)) goto cleanup;
            }
            /* The K4 prefetch candidates alter only load scheduling.  Run
             * both against the same nonzero production-owned-call reference
             * and require exact intermediate and final outputs. */
            ds4_gpu_test_set_moe_q3a4_decode_ksplit(4u);
            for (uint32_t depth = 1u; depth <= 2u; depth++) {
                ds4_gpu_test_set_moe_q3a4_decode_prefetch_depth(depth);
                if (!ds4_gpu_routed_moe_one_owned_tensor(
                        tmp_out, gate, up, mid, down, model, model_bytes,
                        gate_off, up_off, down_off, gate_type, 42u,
                        gate_expert, gate_row, down_expert, down_row,
                        in_dim, mid_dim, out_dim, selected, weights,
                        n_total, n_expert, 0u, 4u, 10.0f, x,
                        home_slots, false, shared_prequant) ||
                    !ds4_gpu_synchronize() ||
                    !ds4_gpu_tensor_read(mid, 0, mid_splith, mid_bytes) ||
                    !ds4_gpu_tensor_read(home_slots, 0, home_splith,
                                         slot_bytes) ||
                    !compare_exact_f32(
                        "SM75 Q3A4 K4 prefetch gate/up intermediate",
                        mid_refh, mid_splith,
                        (size_t)n_expert * mid_dim) ||
                    !compare_exact_f32(
                        "SM75 Q3A4 K4 prefetch owned output",
                        home_refh, home_splith,
                        (size_t)n_expert * out_dim)) goto cleanup;
            }
            ds4_gpu_test_set_moe_q3a4_decode_prefetch_depth(0u);
            ds4_gpu_test_set_moe_q3a4_decode_ksplit(1u);
            ds4_gpu_test_set_moe_q3a4_decode_mapping(0u);
        }
        if (!ds4_gpu_routed_moe_one_owned_tensor(
                tmp_out, gate, up, mid, down, model, model_bytes,
                gate_off, up_off, down_off, gate_type, 42u,
                gate_expert, gate_row, down_expert, down_row,
                in_dim, mid_dim, out_dim, selected, weights,
                n_total, n_expert, 0u, 4u, 10.0f, x,
                home_slots, false, shared_prequant) ||
            !ds4_gpu_routed_moe_one_owned_tensor(
                tmp_out, gate, up, mid, down, model, model_bytes,
                gate_off, up_off, down_off, gate_type, 42u,
                gate_expert, gate_row, down_expert, down_row,
                in_dim, mid_dim, out_dim, selected, weights,
                n_total, n_expert, 4u, 4u, 10.0f, x,
                peer_packed, true, NULL) ||
            !ds4_gpu_routed_moe_owned_packed_combine_tensor(
                ref, home_slots, peer_packed, selected, out_dim, 4u) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(ref, 0, refh,
                                 out_dim * sizeof(float))) goto cleanup;

        ds4_gpu_test_set_moe_q32_decode_graph(1);
        if (!ds4_gpu_routed_moe_one_owned_tensor(
                tmp_out, gate, up, mid, down, model, model_bytes,
                gate_off, up_off, down_off, gate_type, 42u,
                gate_expert, gate_row, down_expert, down_row,
                in_dim, mid_dim, out_dim, selected, weights,
                n_total, n_expert, 0u, 4u, 10.0f, x,
                home_slots, false, shared_prequant) ||
            !ds4_gpu_routed_moe_one_owned_tensor(
                tmp_out, gate, up, mid, down, model, model_bytes,
                gate_off, up_off, down_off, gate_type, 42u,
                gate_expert, gate_row, down_expert, down_row,
                in_dim, mid_dim, out_dim, selected, weights,
                n_total, n_expert, 4u, 4u, 10.0f, x,
                peer_packed, true, NULL) ||
            !ds4_gpu_routed_moe_owned_packed_combine_tensor(
                got, home_slots, peer_packed, selected, out_dim, 4u) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(got, 0, goth,
                                 out_dim * sizeof(float)) ||
            !compare_exact_f32("SM75 Q32 owned graph output",
                               refh, goth, out_dim)) goto cleanup;
    }
    fprintf(stderr,
            "cuda-regression: SM75 %s split gate/up and owned decode "
            "CUDA Graph exact/reuse\n"
            "cuda-regression: SM75 %s fused-lowreg u1/u2/u4 gate/up "
            "and owned decode exact/reuse\n%s",
            label, label,
            gate_type == 43u
                ? "cuda-regression: SM75 Q3A4 hwarp16/tile32/dp4a gate/up "
                  "and owned decode exact/reuse\n"
                  "cuda-regression: SM75 Q3A4 tile32-dp4a K1/K2/K4 "
                  "in-CTA gate/up and owned decode exact/reuse\n"
                  "cuda-regression: SM75 Q3A4 tile32-dp4a K4 "
                  "prefetch-depth 1/2 nonzero exact\n"
                : "cuda-regression: SM75 Q4-32 hwarp16/tile32-dp4a/"
                  "tile32-mma gate/up and owned decode nonzero exact/reuse\n"
                  "cuda-regression: SM75 Q4-32 signed-zero gate probe exact\n");
    rc = 0;

cleanup:
    ds4_gpu_test_set_moe_q32_decode_split(0);
    ds4_gpu_test_set_moe_q32_decode_fused_lowreg(0u);
    ds4_gpu_test_set_moe_q4_32_decode_mapping(0u);
    ds4_gpu_test_set_moe_q3a4_decode_mapping(0u);
    ds4_gpu_test_set_moe_q3a4_decode_ksplit(1u);
    ds4_gpu_test_set_moe_q3a4_decode_prefetch_depth(0u);
    ds4_gpu_test_set_moe_q32_decode_graph(1);
    ds4_gpu_set_routed_q4_layout(0u);
    if (model && !retire_temporary_model_map()) rc = 1;
    ds4_gpu_tensor_free(shared_prequant);
    ds4_gpu_tensor_free(peer_packed); ds4_gpu_tensor_free(home_slots);
    ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(tmp_out); ds4_gpu_tensor_free(got);
    ds4_gpu_tensor_free(ref); ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(x);
    free(home_splith); free(home_refh); free(mid_splith); free(mid_refh);
    free(goth); free(refh);
    free(xh); free(model);
    return rc;
}

static int check_sm75_q32_owned_graph(void) {
    int rc = check_sm75_q32_owned_graph_case(42u, "Q4-32");
    if (check_sm75_q32_owned_graph_case(43u, "Q3A4") != 0) rc = 1;
    return rc;
}

/* This is the production API, not an isolated microkernel comparison.  It
 * proves exact standard-vs-tagged output for a prefill histogram containing
 * full 16s plus true 8/4 tails and for the direct six-expert decode route. */
static int check_sm75_native_q4_layout_exact(void) {
    const uint32_t n_total = 8u, in_dim = 4096u, mid_dim = 256u,
                   out_dim = 256u, in_blocks = 16u, mid_blocks = 1u;
    const uint32_t n_tokens = 128u, n_expert = 1u;
    const uint64_t gate_row = (uint64_t)in_blocks * 144u;
    const uint64_t gate_expert = (uint64_t)mid_dim * gate_row;
    const uint64_t down_row = (uint64_t)mid_blocks * 144u;
    const uint64_t down_expert = (uint64_t)out_dim * down_row;
    const uint64_t gate_off = 0u;
    const uint64_t up_off = gate_expert * n_total;
    const uint64_t down_off = up_off + gate_expert * n_total;
    const uint64_t model_bytes = down_off + down_expert * n_total;
    const uint64_t pairs = (uint64_t)n_tokens * n_expert;
    const uint64_t x_count = (uint64_t)n_tokens * in_dim;
    const uint64_t mid_count = pairs * mid_dim;
    const uint64_t out_count = (uint64_t)n_tokens * out_dim;
    const uint64_t down_bytes = x_count * sizeof(float) >
        out_count * sizeof(float) ? x_count * sizeof(float) :
                                   out_count * sizeof(float);
    unsigned char *standard = (unsigned char *)calloc(1, (size_t)model_bytes);
    unsigned char *native = (unsigned char *)malloc((size_t)model_bytes);
    float *xh = (float *)malloc((size_t)x_count * sizeof(float));
    int32_t *selh = (int32_t *)malloc((size_t)pairs * sizeof(int32_t));
    float *wh = (float *)malloc((size_t)pairs * sizeof(float));
    float *mid_ref = (float *)malloc((size_t)mid_count * sizeof(float));
    float *mid_got = (float *)malloc((size_t)mid_count * sizeof(float));
    float *up_ref = (float *)malloc((size_t)mid_count * sizeof(float));
    float *up_got = (float *)malloc((size_t)mid_count * sizeof(float));
    float *out_ref = (float *)malloc((size_t)out_count * sizeof(float));
    float *out_got = (float *)malloc((size_t)out_count * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(pairs * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(pairs * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(down_bytes);
    int rc = 1;
    if (!standard || !native || !xh || !selh || !wh || !mid_ref || !mid_got ||
        !up_ref || !up_got ||
        !out_ref || !out_got || !x || !selected || !weights || !out ||
        !gate || !up || !mid || !down) goto cleanup;
    /* Cost-aware residual tiles are the tagged-layout production default.
     * The legacy planner and new K-window kernels remain explicit diagnostics
     * until their exactness and end-to-end transfer are measured. */
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_LEGACY_TILES");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_GATE_STREAM7");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_GATE_FIXED_K16");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_DOWN_COMPACT7");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_GATE_FULL64_FUSED");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_DOWN_WIDE512");
    unsetenv("DS4_CUDA_MOE_WRITE_GATE_UP");
    fill_q4_tensor(standard + gate_off, 0u, n_total, mid_dim, in_blocks);
    fill_q4_tensor(standard + up_off, 1u, n_total, mid_dim, in_blocks);
    fill_q4_tensor(standard + down_off, 2u, n_total, out_dim, mid_blocks);
    memcpy(native, standard, (size_t)model_bytes);
    pack_sm75_native_q4_tensor(native + gate_off, standard + gate_off,
                               n_total, mid_dim, in_blocks);
    pack_sm75_native_q4_tensor(native + up_off, standard + up_off,
                               n_total, mid_dim, in_blocks);
    pack_sm75_native_q4_tensor(native + down_off, standard + down_off,
                               n_total, out_dim, mid_blocks);
    for (uint64_t i = 0; i < x_count; i++)
        xh[i] = (float)((int)((i * 23u + (i >> 5u) * 17u) % 257u) - 128) /
            133.0f;
    /* Per-expert counts 25/23/22/21/19/18 exercise both cost promotions:
     * residual 9 -> tile16 and residuals 5..7 -> tile8. */
    const uint32_t cuts[6] = {25u, 48u, 70u, 91u, 110u, 128u};
    for (uint32_t t = 0; t < n_tokens; t++) {
        uint32_t e = 0u;
        while (e < 5u && t >= cuts[e]) e++;
        selh[t] = (int32_t)e;
        wh[t] = (float)((t % 7u) + 1u) / 8.0f;
    }
    if (!ds4_gpu_tensor_write(x, 0, xh, x_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(selected, 0, selh, pairs * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(weights, 0, wh, pairs * sizeof(float))) goto cleanup;
    bool mid_is_f16 = false;
#define RUN_NATIVE_Q4(MODEL, LAYOUT, NTOK, NEXP, MID_DST, OUT_DST) \
    (ds4_gpu_set_routed_q4_layout(LAYOUT), \
     ds4_gpu_set_model_map((MODEL), model_bytes) && \
     ds4_gpu_routed_moe_batch_tensor( \
        out, gate, up, mid, down, (MODEL), model_bytes, \
        gate_off, up_off, down_off, 12u, 12u, \
        gate_expert, gate_row, down_expert, down_row, \
        in_dim, mid_dim, out_dim, selected, weights, n_total, (NEXP), \
        10.0f, x, 0u, (NTOK), &mid_is_f16, true) && !mid_is_f16 && \
     ds4_gpu_synchronize() && \
     ds4_gpu_tensor_read(mid, 0, (MID_DST), \
                         (uint64_t)(NTOK) * (NEXP) * mid_dim * sizeof(float)) && \
     ds4_gpu_tensor_read(out, 0, (OUT_DST), \
                         (uint64_t)(NTOK) * out_dim * sizeof(float)))
    if (!RUN_NATIVE_Q4(standard, 0u, n_tokens, n_expert,
                       mid_ref, out_ref) ||
        !RUN_NATIVE_Q4(native, DS4_TENSOR_LAYOUT_SM75_NATIVE_Q4,
                       n_tokens, n_expert, mid_got, out_got) ||
        !compare_exact_f32("sm75 native q4 prefill mid", mid_ref, mid_got,
                           mid_count) ||
        !compare_exact_f32("sm75 native q4 prefill output", out_ref, out_got,
                           out_count)) goto cleanup;
    fprintf(stderr,
            "cuda-regression: tagged SM75 native Q4 cost-planner default exact\n");

    static const struct {
        const char *name;
        const char *legacy;
        const char *gate;
        const char *down;
        const char *fixed_k16;
        const char *full64_fused;
        const char *wide512;
    } optimized_cases[] = {
        {"legacy-planner diagnostic", "1", "0", "0", "0", "0", "0"},
        {"gate-stream7", "0", "1", "0", "0", "0", "0"},
        {"down-compact7", "0", "0", "1", "0", "0", "0"},
        {"gate-stream7/down-compact7", "0", "1", "1", "0", "0", "0"},
        {"gate-fixed-k16", "0", "0", "0", "1", "0", "0"},
        {"down-wide512", "0", "0", "0", "0", "0", "1"},
        {"gate-full64-fused", "0", "0", "0", "1", "1", "0"},
        {"gate-full64-fused/down-wide512", "0", "0", "0", "1", "1", "1"},
    };
    for (uint32_t c = 0;
         c < sizeof(optimized_cases) / sizeof(optimized_cases[0]); c++) {
        if (setenv("DS4_CUDA_MOE_NATIVE_Q4_LEGACY_TILES",
                   optimized_cases[c].legacy, 1) != 0 ||
            setenv("DS4_CUDA_MOE_NATIVE_Q4_GATE_STREAM7",
                   optimized_cases[c].gate, 1) != 0 ||
            setenv("DS4_CUDA_MOE_NATIVE_Q4_DOWN_COMPACT7",
                   optimized_cases[c].down, 1) != 0 ||
            setenv("DS4_CUDA_MOE_NATIVE_Q4_GATE_FIXED_K16",
                   optimized_cases[c].fixed_k16, 1) != 0 ||
            setenv("DS4_CUDA_MOE_NATIVE_Q4_GATE_FULL64_FUSED",
                   optimized_cases[c].full64_fused, 1) != 0 ||
            setenv("DS4_CUDA_MOE_NATIVE_Q4_DOWN_WIDE512",
                   optimized_cases[c].wide512, 1) != 0) {
            fprintf(stderr,
                    "cuda-regression: could not select native Q4 next paths\n");
            goto cleanup;
        }
        if (!RUN_NATIVE_Q4(native, DS4_TENSOR_LAYOUT_SM75_NATIVE_Q4,
                           n_tokens, n_expert, mid_got, out_got) ||
            !compare_exact_f32("sm75 native q4 optimized prefill mid",
                               mid_ref, mid_got, mid_count) ||
            !compare_exact_f32("sm75 native q4 optimized prefill output",
                               out_ref, out_got, out_count)) goto cleanup;
        fprintf(stderr, "cuda-regression: tagged SM75 native Q4 %s exact\n",
                optimized_cases[c].name);
    }

    /* The routed-MoE implementation reuses gate->ptr as the mid-Q8 scratch
     * immediately after Gate/Up. Reading it after this API returns would compare
     * standard-Q8 scratch bytes with native-Q8 scratch bytes, not Gate values.
     * Up remains resident. Validate that auxiliary Up is exact and that
     * enabling both auxiliary stores leaves mid/final output bit-exact. */
    if (setenv("DS4_CUDA_MOE_NATIVE_Q4_GATE_STREAM7", "0", 1) != 0 ||
        setenv("DS4_CUDA_MOE_NATIVE_Q4_GATE_FIXED_K16", "1", 1) != 0 ||
        setenv("DS4_CUDA_MOE_NATIVE_Q4_DOWN_COMPACT7", "0", 1) != 0 ||
        setenv("DS4_CUDA_MOE_NATIVE_Q4_GATE_FULL64_FUSED", "1", 1) != 0 ||
        setenv("DS4_CUDA_MOE_NATIVE_Q4_DOWN_WIDE512", "0", 1) != 0 ||
        setenv("DS4_CUDA_MOE_WRITE_GATE_UP", "1", 1) != 0 ||
        !RUN_NATIVE_Q4(standard, 0u, n_tokens, n_expert,
                       mid_ref, out_ref) ||
        !ds4_gpu_tensor_read(up, 0, up_ref, mid_count * sizeof(float)) ||
        !RUN_NATIVE_Q4(native, DS4_TENSOR_LAYOUT_SM75_NATIVE_Q4,
                       n_tokens, n_expert, mid_got, out_got) ||
        !ds4_gpu_tensor_read(up, 0, up_got, mid_count * sizeof(float)) ||
        !compare_exact_f32("sm75 native q4 full64-fused up aux",
                           up_ref, up_got, mid_count) ||
        !compare_exact_f32("sm75 native q4 full64-fused aux mid",
                           mid_ref, mid_got, mid_count) ||
        !compare_exact_f32("sm75 native q4 full64-fused aux output",
                           out_ref, out_got, out_count)) goto cleanup;
    fprintf(stderr,
            "cuda-regression: tagged SM75 native Q4 gate-full64-fused "
            "up-aux/no-perturb exact\n");
    unsetenv("DS4_CUDA_MOE_WRITE_GATE_UP");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_LEGACY_TILES");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_GATE_STREAM7");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_GATE_FIXED_K16");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_DOWN_COMPACT7");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_GATE_FULL64_FUSED");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_DOWN_WIDE512");
    unsetenv("DS4_CUDA_MOE_WRITE_GATE_UP");

    for (uint32_t s = 0; s < 6u; s++) {
        selh[s] = (int32_t)s;
        wh[s] = (float)(s + 1u) / 8.0f;
    }
    if (!ds4_gpu_tensor_write(selected, 0, selh, 6u * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(weights, 0, wh, 6u * sizeof(float)) ||
        !RUN_NATIVE_Q4(standard, 0u, 1u, 6u, mid_ref, out_ref) ||
        !RUN_NATIVE_Q4(native, DS4_TENSOR_LAYOUT_SM75_NATIVE_Q4,
                       1u, 6u, mid_got, out_got) ||
        !compare_exact_f32("sm75 native q4 decode mid", mid_ref, mid_got,
                           6ull * mid_dim) ||
        !compare_exact_f32("sm75 native q4 decode output", out_ref, out_got,
                           out_dim)) goto cleanup;
    fprintf(stderr, "cuda-regression: tagged SM75 native Q4 decode exact\n");

    /* Mixed production layout: IQ2 gate/up bytes remain standard while the
     * Q4 down tensor alone carries the tagged SM75 native transform. */
    const uint64_t iq2_gate_row = (uint64_t)in_blocks * 66u;
    const uint64_t iq2_gate_expert = (uint64_t)mid_dim * iq2_gate_row;
    const uint64_t iq2_up_off = iq2_gate_expert * n_total;
    const uint64_t iq2_down_off = iq2_up_off + iq2_gate_expert * n_total;
    const uint64_t iq2_model_bytes = iq2_down_off + down_expert * n_total;
    memset(standard, 0, (size_t)model_bytes);
    fill_iq2_tensor(standard, 0u, n_total, mid_dim, in_blocks);
    fill_iq2_tensor(standard + iq2_up_off, 1u, n_total, mid_dim, in_blocks);
    fill_q4_tensor(standard + iq2_down_off, 2u, n_total, out_dim, mid_blocks);
    memcpy(native, standard, (size_t)iq2_model_bytes);
    pack_sm75_native_q4_tensor(native + iq2_down_off,
                               standard + iq2_down_off,
                               n_total, out_dim, mid_blocks);
#define RUN_MIXED_Q4_IQ2(MODEL, LAYOUT, MID_DST, OUT_DST) \
    (mid_is_f16 = false, ds4_gpu_set_routed_q4_layout(LAYOUT), \
     ds4_gpu_set_model_map((MODEL), iq2_model_bytes) && \
     ds4_gpu_routed_moe_batch_tensor( \
        out, gate, up, mid, down, (MODEL), iq2_model_bytes, \
        0u, iq2_up_off, iq2_down_off, 16u, 12u, \
        iq2_gate_expert, iq2_gate_row, down_expert, down_row, \
        in_dim, mid_dim, out_dim, selected, weights, n_total, n_expert, \
        10.0f, x, 0u, n_tokens, &mid_is_f16, true) && !mid_is_f16 && \
     ds4_gpu_synchronize() && \
     ds4_gpu_tensor_read(mid, 0, (MID_DST), mid_count * sizeof(float)) && \
     ds4_gpu_tensor_read(out, 0, (OUT_DST), out_count * sizeof(float)))
    unsetenv("DS4_CUDA_MOE_IQ2_TAIL8_ALL_SM75");
    if (!ds4_gpu_tensor_write(selected, 0, selh, pairs * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(weights, 0, wh, pairs * sizeof(float)) ||
        !RUN_MIXED_Q4_IQ2(standard, 0u, mid_ref, out_ref) ||
        !RUN_MIXED_Q4_IQ2(native, DS4_TENSOR_LAYOUT_SM75_NATIVE_Q4,
                          mid_got, out_got) ||
        !compare_exact_f32("tagged IQ2 gate/Q4 down prefill mid",
                           mid_ref, mid_got, mid_count) ||
        !compare_exact_f32("tagged IQ2 gate/Q4 down prefill output",
                           out_ref, out_got, out_count)) goto cleanup;
    fprintf(stderr,
            "cuda-regression: tagged IQ2 tail8-default gate/up + native Q4 "
            "down exact\n");

    if (setenv("DS4_CUDA_MOE_IQ2_TAIL8_ALL_SM75", "0", 1) != 0 ||
        !RUN_MIXED_Q4_IQ2(native, DS4_TENSOR_LAYOUT_SM75_NATIVE_Q4,
                          mid_got, out_got) ||
        !compare_exact_f32("tagged IQ2 tail4 rollback prefill mid",
                           mid_ref, mid_got, mid_count) ||
        !compare_exact_f32("tagged IQ2 tail4 rollback prefill output",
                           out_ref, out_got, out_count)) goto cleanup;
    fprintf(stderr,
            "cuda-regression: tagged IQ2 tail4 rollback exact\n");
    unsetenv("DS4_CUDA_MOE_IQ2_TAIL8_ALL_SM75");

#define RUN_MIXED_Q4_IQ2_OWNED(MODEL, LAYOUT, MID_DST, OUT_DST) \
    (mid_is_f16 = false, \
     ds4_gpu_tensor_write(selected, 0, selh, pairs * sizeof(int32_t)) && \
     ds4_gpu_tensor_write(weights, 0, wh, pairs * sizeof(float)) && \
     (ds4_gpu_set_routed_q4_layout(LAYOUT), 1) && \
     ds4_gpu_set_model_map((MODEL), iq2_model_bytes) && \
     ds4_gpu_routed_moe_batch_owned_tensor( \
        out, gate, up, mid, down, (MODEL), iq2_model_bytes, \
        0u, iq2_up_off, iq2_down_off, 16u, 12u, \
        iq2_gate_expert, iq2_gate_row, down_expert, down_row, \
        in_dim, mid_dim, out_dim, selected, weights, n_total, n_expert, \
        0u, n_total, 10.0f, x, 3u, n_tokens, 0u, &mid_is_f16) && \
     !mid_is_f16 && ds4_gpu_synchronize() && \
     ds4_gpu_tensor_read(mid, 0, (MID_DST), mid_count * sizeof(float)) && \
     ds4_gpu_tensor_read(out, 0, (OUT_DST), out_count * sizeof(float)))
    if (!RUN_MIXED_Q4_IQ2_OWNED(standard, 0u, mid_ref, out_ref) ||
        !RUN_MIXED_Q4_IQ2_OWNED(
            native, DS4_TENSOR_LAYOUT_SM75_NATIVE_Q4, mid_got, out_got) ||
        !compare_exact_f32("tagged owned IQ2 gate/Q4 down prefill mid",
                           mid_ref, mid_got, mid_count) ||
        !compare_exact_f32("tagged owned IQ2 gate/Q4 down prefill output",
                           out_ref, out_got, out_count)) goto cleanup;
    fprintf(stderr,
            "cuda-regression: tagged owned IQ2 gate/up + native Q4 down "
            "exact\n");
#undef RUN_MIXED_Q4_IQ2_OWNED
#undef RUN_MIXED_Q4_IQ2
    rc = 0;
#undef RUN_NATIVE_Q4

cleanup:
    unsetenv("DS4_CUDA_MOE_IQ2_TAIL8_ALL_SM75");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_LEGACY_TILES");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_GATE_STREAM7");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_GATE_FIXED_K16");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_DOWN_COMPACT7");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_GATE_FULL64_FUSED");
    unsetenv("DS4_CUDA_MOE_NATIVE_Q4_DOWN_WIDE512");
    unsetenv("DS4_CUDA_MOE_WRITE_GATE_UP");
    ds4_gpu_set_routed_q4_layout(0u);
    if ((standard || native) && !retire_temporary_model_map()) rc = 1;
    ds4_gpu_tensor_free(down); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(x);
    free(out_got); free(out_ref); free(up_got); free(up_ref);
    free(mid_got); free(mid_ref);
    free(wh); free(selh); free(xh); free(native); free(standard);
    return rc;
}

static int check_large_topk(void) {
    const uint32_t n_comp = 32768;
    const uint32_t n_tokens = 32;
    const uint32_t top_k = 512;
    const uint64_t score_count = (uint64_t)n_comp * n_tokens;
    float *scores_host = (float *)malloc((size_t)score_count * sizeof(float));
    uint32_t *selected_host = (uint32_t *)malloc((size_t)n_tokens * top_k * sizeof(uint32_t));
    if (!scores_host || !selected_host) return 1;

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < n_comp; i++) {
            scores_host[(uint64_t)t * n_comp + i] = (float)i;
        }
    }

    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * top_k * sizeof(uint32_t));
    int rc = 1;
    double elapsed = 0.0;
    if (scores && selected &&
        ds4_gpu_tensor_write(scores, 0, scores_host, score_count * sizeof(float))) {
        /* Exclude one-time CUDA module/kernel setup from the throughput guard. */
        if (!ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) ||
            !ds4_gpu_synchronize()) {
            rc = 1;
            goto cleanup;
        }
        const double t0 = monotonic_seconds();
        if (ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) &&
            ds4_gpu_synchronize()) {
            elapsed = monotonic_seconds() - t0;
            rc = ds4_gpu_tensor_read(selected, 0, selected_host,
                                     (uint64_t)n_tokens * top_k * sizeof(uint32_t)) ? 0 : 1;
        }
    }
    if (rc == 0) {
        for (uint32_t t = 0; t < n_tokens && rc == 0; t++) {
            for (uint32_t i = 0; i < top_k; i++) {
                const uint32_t expected = n_comp - 1u - i;
                const uint32_t got = selected_host[(uint64_t)t * top_k + i];
                if (got != expected) {
                    fprintf(stderr, "top-k mismatch token=%u rank=%u got=%u expected=%u\n",
                            t, i, got, expected);
                    rc = 1;
                    break;
                }
            }
        }
    }
    if (rc == 0) {
        const double max_seconds = getenv_seconds("DS4_CUDA_TOPK_REGRESSION_SEC", 2.0);
        fprintf(stderr, "cuda-regression: top-k n_comp=%u n_tokens=%u elapsed=%.3fs\n",
                n_comp, n_tokens, elapsed);
        if (elapsed > max_seconds) {
            fprintf(stderr, "top-k regression: %.3fs exceeds %.3fs\n", elapsed, max_seconds);
            rc = 1;
        }
    }

cleanup:
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    free(selected_host);
    free(scores_host);
    return rc;
}

static int check_decode_attention_overflow_path(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_raw = 128;
    const uint32_t n_comp = 8100;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;

    float *sinks = (float *)calloc(n_head, sizeof(float));
    float *q_host = (float *)calloc((size_t)q_count, sizeof(float));
    float *raw_host = (float *)calloc((size_t)raw_count, sizeof(float));
    float *comp_host = (float *)calloc((size_t)comp_count, sizeof(float));
    float *heads_host = (float *)calloc((size_t)q_count, sizeof(float));
    if (!sinks || !q_host || !raw_host || !comp_host || !heads_host) return 1;

    for (uint32_t c = 0; c < n_comp; c++) {
        comp_host[(uint64_t)c * head_dim] = 1.0f;
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    int rc = 1;
    if (heads && q && raw && comp &&
        ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp, 0, comp_host, comp_count * sizeof(float)) &&
        ds4_gpu_set_model_map(sinks, n_head * sizeof(float)) &&
        ds4_gpu_attention_decode_heads_tensor(heads,
                                              sinks,
                                              n_head * sizeof(float),
                                              0,
                                              q,
                                              raw,
                                              n_raw,
                                              n_raw,
                                              0,
                                              comp,
                                              0,
                                              n_comp,
                                              NULL,
                                              0,
                                              n_head,
                                              head_dim) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(heads, 0, heads_host, q_count * sizeof(float))) {
        rc = 0;
        for (uint32_t h = 0; h < n_head; h++) {
            const float v = heads_host[(uint64_t)h * head_dim];
            if (v < 0.90f) {
                fprintf(stderr, "attention fallback ignored compressed rows for head=%u value=%f\n",
                        h, (double)v);
                rc = 1;
            }
        }
    }

    if (!retire_temporary_model_map()) rc = 1;
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    free(heads_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    free(sinks);
    return rc;
}

static int check_prefill_attention_head_shards(void) {
    const uint32_t n_tokens = 128;
    const uint32_t n_head = 64;
    const uint32_t half = n_head / 2u;
    const uint32_t head_dim = 512;
    const uint32_t n_rot = 64;
    const uint64_t q_count = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t kv_count = (uint64_t)n_tokens * head_dim;
    float *sinks = (float *)calloc(n_head, sizeof(float));
    float *q_host = (float *)malloc((size_t)q_count * sizeof(float));
    float *kv_host = (float *)malloc((size_t)kv_count * sizeof(float));
    float *reference = (float *)malloc((size_t)q_count * sizeof(float));
    float *candidate = (float *)malloc((size_t)q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(kv_count * sizeof(float));
    ds4_gpu_tensor *full = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *split = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    int rc = 1;
    if (!sinks || !q_host || !kv_host || !reference || !candidate ||
        !q || !kv || !full || !split) goto cleanup;
    for (uint64_t i = 0; i < q_count; i++) {
        q_host[i] = (float)((int)(i * 17u % 101u) - 50) / 127.0f;
    }
    for (uint64_t i = 0; i < kv_count; i++) {
        kv_host[i] = (float)((int)(i * 29u % 113u) - 56) / 139.0f;
    }
    if (!ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(kv, 0, kv_host, kv_count * sizeof(float)) ||
        !ds4_gpu_set_model_map(sinks, n_head * sizeof(float)) ||
        !ds4_gpu_attention_prefill_raw_heads_tensor(
                full, sinks, n_head * sizeof(float), 0, q, kv,
                n_tokens, 128u, n_head, head_dim) ||
        !ds4_gpu_attention_prefill_raw_heads_shard_tensor(
                split, sinks, n_head * sizeof(float), 0, q, kv,
                n_tokens, 128u, 0u, half, n_head, head_dim) ||
        !ds4_gpu_attention_prefill_raw_heads_shard_tensor(
                split, sinks, n_head * sizeof(float), 0, q, kv,
                n_tokens, 128u, half, half, n_head, head_dim) ||
        !ds4_gpu_rope_tail_tensor(full, n_tokens, n_head, head_dim, n_rot,
                                  17u, 0u, true, 10000.0f, 1.0f, 0.0f,
                                  1.0f, 32.0f, 1.0f) ||
        !ds4_gpu_rope_tail_head_range_tensor(
                split, n_tokens, 0u, half, n_head, head_dim, n_rot,
                17u, 0u, true, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f) ||
        !ds4_gpu_rope_tail_head_range_tensor(
                split, n_tokens, half, half, n_head, head_dim, n_rot,
                17u, 0u, true, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(full, 0, reference, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_read(split, 0, candidate, q_count * sizeof(float))) {
        goto cleanup;
    }
    if (memcmp(reference, candidate, (size_t)q_count * sizeof(float)) != 0) {
        uint64_t first = 0;
        while (first < q_count && reference[first] == candidate[first]) first++;
        fprintf(stderr,
                "prefill attention head shard mismatch at %llu: full=%g split=%g\n",
                (unsigned long long)first,
                (double)reference[first], (double)candidate[first]);
        goto cleanup;
    }
    fprintf(stderr,
            "cuda-regression: 64-head prefill split 32/32 exact (%llu values)\n",
            (unsigned long long)q_count);
    rc = 0;

cleanup:
    if (sinks && !retire_temporary_model_map()) rc = 1;
    ds4_gpu_tensor_free(split);
    ds4_gpu_tensor_free(full);
    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(q);
    free(candidate);
    free(reference);
    free(kv_host);
    free(q_host);
    free(sinks);
    return rc;
}

static int check_sm75_indexed_attention_heads8_exact(void) {
    const uint32_t n_tokens = 128u, pos0 = 31744u;
    const uint32_t n_raw = 2304u, raw_cap = 2304u, raw_start = 0u;
    const uint32_t n_comp = 7936u, top_k = 512u;
    const uint32_t window = 2048u, ratio = 4u;
    const uint32_t n_head = 16u, head_dim = 512u;
    const uint64_t q_count = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t raw_count = (uint64_t)raw_cap * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t topk_count = (uint64_t)n_tokens * top_k;
    float *sinks = (float *)calloc(n_head, sizeof(float));
    int32_t *topk_host = (int32_t *)malloc(
        (size_t)topk_count * sizeof(int32_t));
    float *reference = (float *)malloc((size_t)q_count * sizeof(float));
    float *candidate = (float *)malloc((size_t)q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    ds4_gpu_tensor *topk = ds4_gpu_tensor_alloc(
        topk_count * sizeof(int32_t));
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    int rc = 1;
    if (!sinks || !topk_host || !reference || !candidate ||
        !q || !raw || !comp || !topk || !heads) goto cleanup;
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t k = 0; k < top_k; k++) {
            topk_host[(uint64_t)t * top_k + k] =
                (int32_t)((t * 131u + k * 17u) % n_comp);
        }
    }
    if (!ds4_gpu_tensor_fill_f32(q, 0.03125f, q_count) ||
        !ds4_gpu_tensor_fill_f32(raw, 0.015625f, raw_count) ||
        !ds4_gpu_tensor_fill_f32(comp, -0.0078125f, comp_count) ||
        !ds4_gpu_tensor_write(topk, 0, topk_host,
                              topk_count * sizeof(int32_t)) ||
        !ds4_gpu_set_model_map(sinks, n_head * sizeof(float))) goto cleanup;
#define RUN_INDEXED_HEADS8(DST) \
    (ds4_gpu_attention_indexed_mixed_batch_heads_tensor( \
        heads, sinks, n_head * sizeof(float), 0u, q, raw, comp, 0u, topk, \
        n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k, window, \
        ratio, n_head, head_dim) && ds4_gpu_synchronize() && \
     ds4_gpu_tensor_read(heads, 0, (DST), q_count * sizeof(float)))
    (void)setenv("DS4_CUDA_INDEXED_HEADS8_SM75", "0", 1);
    if (!RUN_INDEXED_HEADS8(reference)) goto cleanup;
    (void)unsetenv("DS4_CUDA_INDEXED_HEADS8_SM75");
    if (!RUN_INDEXED_HEADS8(candidate) ||
        !compare_exact_f32("SM75 indexed attention heads8",
                           reference, candidate, q_count)) goto cleanup;
    if (!ds4_gpu_attention_indexed_mixed_batch_heads_shard_tensor(
            heads, sinks, n_head * sizeof(float), 0u, q, raw, comp, 0u,
            topk, n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
            window, ratio, 0u, n_head / 2u, n_head, head_dim) ||
        !ds4_gpu_attention_indexed_mixed_batch_heads_shard_tensor(
            heads, sinks, n_head * sizeof(float), 0u, q, raw, comp, 0u,
            topk, n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
            window, ratio, n_head / 2u, n_head / 2u, n_head, head_dim) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads, 0, candidate,
                             q_count * sizeof(float)) ||
        !compare_exact_f32("SM75 indexed attention heads8 shards",
                           reference, candidate, q_count)) goto cleanup;
    {
        uint64_t nonzero = 0u;
        for (uint64_t i = 0; i < q_count; i++)
            nonzero += candidate[i] != 0.0f;
        if (nonzero == 0u) {
            fprintf(stderr,
                    "SM75 indexed attention heads8 produced only zeros\n");
            goto cleanup;
        }
    }
    fprintf(stderr,
            "cuda-regression: SM75 indexed attention 16-head/512-thread "
            "versus 8-head/256-thread whole and sharded exact (%llu values)\n",
            (unsigned long long)q_count);
    rc = 0;
#undef RUN_INDEXED_HEADS8

cleanup:
    (void)unsetenv("DS4_CUDA_INDEXED_HEADS8_SM75");
    if (sinks && !retire_temporary_model_map()) rc = 1;
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(topk);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    free(candidate);
    free(reference);
    free(topk_host);
    free(sinks);
    return rc;
}

int main(void) {
    /* The regression owns the path and both halves of each new A/B.  Do not
     * let a debug shell silently change tile width, row span, MMA eligibility,
     * staging, or turn an array baseline into a scalar/scalar comparison. */
    (void)unsetenv("DS4_CUDA_MOE_NO_Q4_MMA");
    (void)unsetenv("DS4_CUDA_MOE_NO_Q4_MMA_TILE16");
    (void)unsetenv("DS4_CUDA_MOE_NO_Q4_MMA_TILE16_SM75");
    (void)unsetenv("DS4_CUDA_MOE_NO_Q4_SORTED");
    (void)unsetenv("DS4_CUDA_MOE_Q4_GATE_TILE16_SM75");
    (void)unsetenv("DS4_CUDA_MOE_NO_Q4_GATE_TILE16_SM75");
    (void)unsetenv("DS4_CUDA_MOE_Q4_GATE_STAGE4_SM75");
    (void)unsetenv("DS4_CUDA_MOE_NO_EXPERT_TILES");
    (void)unsetenv("DS4_CUDA_MOE_TILE4");
    (void)unsetenv("DS4_CUDA_MOE_TILE8");
    (void)unsetenv("DS4_CUDA_MOE_NO_IQ2_MMA_SM75");
    (void)unsetenv("DS4_CUDA_MOE_NO_IQ2_MMA_TILE16_SM75");
    (void)unsetenv("DS4_CUDA_MOE_IQ2_STAGE6_SM75");
    (void)unsetenv("DS4_CUDA_MOE_IQ2_STAGE4_SM75");
    (void)unsetenv("DS4_CUDA_MOE_Q2_DOWN_MMA_SM75");
    (void)unsetenv("DS4_CUDA_MOE_NO_Q2_DOWN_MMA_SM75");
    (void)unsetenv("DS4_CUDA_MOE_MIXED_TAIL_TILES");
    (void)unsetenv("DS4_CUDA_MOE_NO_MIXED_TAIL_TILES");
    (void)unsetenv("DS4_CUDA_MOE_NO_Q4_DOWN_ROWSPAN");
    (void)unsetenv("DS4_CUDA_MOE_ATOMIC_DOWN");
    (void)unsetenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN");
    (void)unsetenv("DS4_CUDA_MOE_NO_DOWN_TILE16");
    (void)unsetenv("DS4_CUDA_MOE_NO_DOWN_ROW2048");
    (void)unsetenv("DS4_CUDA_MOE_NO_DOWN_ROW256");
    (void)unsetenv("DS4_CUDA_MOE_NO_DOWN_ROW128");
    (void)unsetenv("DS4_CUDA_MOE_NO_DOWN_ROW64");
    (void)unsetenv("DS4_CUDA_MOE_DOWN_ROW512");
    (void)unsetenv("DS4_CUDA_MOE_DOWN_ROW1024");
    (void)unsetenv("DS4_CUDA_MOE_DOWN_ROW2048");
    (void)unsetenv("DS4_CUDA_MOE_DOWN_ROW256");
    (void)unsetenv("DS4_CUDA_MOE_DOWN_ROW128");
    (void)unsetenv("DS4_CUDA_MOE_DOWN_ROW64");
    (void)unsetenv("DS4_CUDA_MOE_NO_GATE_ROW2048");
    (void)unsetenv("DS4_CUDA_MOE_NO_GATE_ROW256");
    (void)unsetenv("DS4_CUDA_MOE_NO_GATE_ROW128");
    (void)unsetenv("DS4_CUDA_MOE_GATE_ROW1024");
    (void)unsetenv("DS4_CUDA_MOE_GATE_ROW2048");
    (void)unsetenv("DS4_CUDA_MOE_GATE_ROW256");
    (void)unsetenv("DS4_CUDA_MOE_GATE_ROW128");
    (void)setenv("DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75", "0", 1);
    (void)setenv("DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75", "0", 1);
    (void)setenv("DS4_CUDA_MOE_IQ2_SCALAR_SM75", "0", 1);
    (void)setenv("DS4_CUDA_MOE_Q32_DECODE_GRAPH", "1", 1);
    (void)unsetenv("DS4_CUDA_MOE_Q32_DECODE_SPLIT");
    (void)unsetenv("DS4_CUDA_NO_MOE_Q32_DECODE_SPLIT");
    (void)unsetenv("DS4_CUDA_MOE_Q32_DECODE_FUSED_LOWREG");
    (void)unsetenv("DS4_CUDA_NO_MOE_Q32_DECODE_FUSED_LOWREG");
    (void)unsetenv("DS4_CUDA_MOE_Q4_32_DECODE_MAPPING");
    (void)unsetenv("DS4_CUDA_MOE_Q4_32_DECODE_MAPPING_AUDIT");
    (void)unsetenv("DS4_CUDA_MOE_Q3A4_DECODE_MAPPING");
    (void)unsetenv("DS4_CUDA_NO_MOE_Q3A4_DECODE_MAPPING");
    (void)unsetenv("DS4_CUDA_MOE_Q3A4_DECODE_KSPLIT");
    (void)unsetenv("DS4_CUDA_MOE_Q3A4_DECODE_PREFETCH_DEPTH");
    if (check_sm75_q4_32_mapping_env() != 0) return 1;
    if (check_sm75_q3a4_ksplit_env() != 0) return 1;
    if (check_sm75_q4_32_dp4a_pack() != 0) return 1;
    if (check_sm75_q3a4_dp4a_pack() != 0) return 1;
    idle_model_map = (unsigned char *)calloc(1, (size_t)idle_model_bytes);
    if (!idle_model_map) return 1;
    if (!ds4_gpu_init()) {
        free(idle_model_map);
        return 1;
    }
    if (ds4_gpu_test_get_moe_q4_32_decode_mapping() != 0u) {
        fprintf(stderr,
                "cuda-regression: SM75 Q4-32 audit mapping is not default-off\n");
        ds4_gpu_cleanup();
        free(idle_model_map);
        return 1;
    }
    if (ds4_gpu_test_get_moe_q3a4_decode_mapping() != 3u) {
        fprintf(stderr,
                "cuda-regression: SM75 Q3A4 production mapping is not "
                "tile32-dp4a\n");
        ds4_gpu_cleanup();
        free(idle_model_map);
        return 1;
    }
    if (ds4_gpu_test_get_moe_q3a4_decode_ksplit() != 4u) {
        fprintf(stderr,
                "cuda-regression: SM75 Q3A4 production K split is not K4\n");
        ds4_gpu_cleanup();
        free(idle_model_map);
        return 1;
    }
    if (ds4_gpu_test_get_moe_q3a4_decode_prefetch_depth() != 0u) {
        fprintf(stderr,
                "cuda-regression: SM75 Q3A4 production K4 unexpectedly "
                "enables prefetch candidate\n");
        ds4_gpu_cleanup();
        free(idle_model_map);
        return 1;
    }
    fprintf(stderr,
            "cuda-regression: SM75 Q3A4 tile32-dp4a-k4 production default\n");
    if (!retire_temporary_model_map()) {
        ds4_gpu_cleanup();
        free(idle_model_map);
        return 1;
    }
    int rc = check_sm75_q8_mma_exact();
    if (check_sm75_iq2_moe_mma_exact() != 0) rc = 1;
    if (check_sm75_q4_q2_next_targets_exact() != 0) rc = 1;
    if (check_sm75_q32_production_exact() != 0) rc = 1;
    if (check_sm75_q32_owned_graph() != 0) rc = 1;
    if (check_sm75_native_q4_layout_exact() != 0) rc = 1;
    if (check_large_topk() != 0) rc = 1;
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    if (check_prefill_attention_head_shards() != 0) rc = 1;
    if (check_sm75_indexed_attention_heads8_exact() != 0) rc = 1;
    ds4_gpu_cleanup();
    free(idle_model_map);
    idle_model_map = NULL;
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
