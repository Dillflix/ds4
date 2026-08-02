#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

static int check_sm75_q8_mma_exact(void) {
    const uint32_t in_dim = 512;
    const uint32_t out_dim = 64;
    const uint32_t n_tokens = 17;
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
                "sm75 q8 mma mismatch at %zu: reference=%g candidate=%g\n",
                first, (double)reference[first], (double)candidate[first]);
        goto cleanup;
    }
    fprintf(stderr, "cuda-regression: sm75 q8 mma exact (%zu values)\n",
            out_count);
    rc = 0;

cleanup:
    (void)unsetenv("DS4_CUDA_NO_Q8_MMA_SM75");
    (void)unsetenv("DS4_CUDA_NO_Q8_F16_CACHE");
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    free(candidate);
    free(reference);
    free(x_host);
    free(model);
    return rc;
}

static int check_sm75_iq2_moe_mma_exact(void) {
    const uint32_t n_total_expert = 8;
    const uint32_t n_expert = 6;
    const uint32_t n_tokens = 32;
    const uint32_t in_dim = 4096;
    const uint32_t mid_dim = 256;
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
    unsigned char *model = (unsigned char *)calloc(1, (size_t)model_bytes);
    float *x_host = (float *)malloc((size_t)x_count * sizeof(float));
    int32_t *selected_host = (int32_t *)malloc((size_t)pair_count * sizeof(int32_t));
    float *weights_host = (float *)malloc((size_t)pair_count * sizeof(float));
    float *reference = (float *)malloc((size_t)mid_count * sizeof(float));
    float *candidate = (float *)malloc((size_t)mid_count * sizeof(float));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(pair_count * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * out_dim * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(pair_count * out_dim * sizeof(float));
    int rc = 1;
    if (!model || !x_host || !selected_host || !weights_host ||
        !reference || !candidate || !x || !selected || !weights || !out ||
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
    for (uint64_t i = 0; i < x_count; i++) {
        const int v = (int)((i * 23u + (i >> 5u) * 17u) % 257u) - 128;
        x_host[i] = (float)v / 133.0f;
    }
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t s = 0; s < n_expert; s++) {
            selected_host[(uint64_t)t * n_expert + s] =
                (int32_t)((t * 3u + s) % n_total_expert);
            weights_host[(uint64_t)t * n_expert + s] =
                (float)(s + 1u) / 21.0f;
        }
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
                "sm75 iq2 moe mma mismatch at %llu: reference=%g candidate=%g\n",
                (unsigned long long)first,
                (double)reference[first], (double)candidate[first]);
        goto cleanup;
    }
    fprintf(stderr, "cuda-regression: sm75 iq2 moe mma exact (%llu values)\n",
            (unsigned long long)mid_count);
    rc = 0;

cleanup:
    (void)unsetenv("DS4_CUDA_MOE_NO_IQ2_MMA_SM75");
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
    free(weights_host);
    free(selected_host);
    free(x_host);
    free(model);
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

int main(void) {
    if (!ds4_gpu_init()) return 1;
    int rc = check_sm75_q8_mma_exact();
    if (check_sm75_iq2_moe_mma_exact() != 0) rc = 1;
    if (check_large_topk() != 0) rc = 1;
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
