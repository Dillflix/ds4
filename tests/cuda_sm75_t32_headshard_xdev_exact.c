/* Bounded two-GPU exactness audit for the production T32 head-shard protocol.
 * It checks copied input and partner q_b, local-KV/peer-topk indexed attention,
 * inverse RoPE, the compact output-A peer gather, output-B, and finally the
 * complete production-ordered chain without intermediate fences. This is
 * diagnostic-only. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
#define N_GROUP 8u
#define HALF_GROUP 4u
#define GROUP_DIM 4096u
#define OUT_A_RANK 1024u
#define LOW_DIM ((uint64_t)N_GROUP * OUT_A_RANK)
#define HALF_LOW_DIM ((uint64_t)HALF_GROUP * OUT_A_RANK)
#define EMBED_DIM 7168u
#define RAW_CAP 2304u
#define N_RAW 128u
#define RAW_START 385u
#define N_COMP 8192u
#define TOP_K 512u
#define ATTN_WINDOW 128u
#define ATTN_RATIO 4u

#define CHECK(c, m) do { if (!(c)) {                                      \
    fprintf(stderr, "error: %s (line %d)\n", (m), __LINE__); goto cleanup; \
} } while (0)

typedef struct {
    uint64_t mismatches;
    uint64_t first;
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

static diff_metrics compare_f32(const float *a, const float *b,
                                uint64_t count) {
    diff_metrics d = {0u, UINT64_MAX, 0.0};
    for (uint64_t i = 0u; i < count; i++) {
        uint32_t ab, bb;
        memcpy(&ab, a + i, sizeof(ab));
        memcpy(&bb, b + i, sizeof(bb));
        if (ab != bb) {
            if (!d.mismatches) d.first = i;
            d.mismatches++;
        }
        const double delta = fabs((double)a[i] - (double)b[i]);
        if (delta > d.max_abs) d.max_abs = delta;
    }
    return d;
}

static diff_metrics compare_head_half(const float *reference,
                                      const float *candidate,
                                      uint32_t head0) {
    diff_metrics total = {0u, UINT64_MAX, 0.0};
    for (uint32_t token = 0u; token < N_TOK; token++) {
        const uint64_t offset = (uint64_t)token * OUT_DIM +
                                (uint64_t)head0 * HEAD_DIM;
        diff_metrics d = compare_f32(reference + offset, candidate + offset,
                                     SHARD_OUT_DIM);
        if (d.mismatches && !total.mismatches)
            total.first = offset + d.first;
        total.mismatches += d.mismatches;
        if (d.max_abs > total.max_abs) total.max_abs = d.max_abs;
    }
    return total;
}

static diff_metrics compare_row_slice(const float *reference,
                                      uint64_t reference_stride,
                                      const float *candidate,
                                      uint64_t candidate_stride,
                                      uint32_t rows,
                                      uint64_t column0,
                                      uint64_t width) {
    diff_metrics total = {0u, UINT64_MAX, 0.0};
    for (uint32_t row = 0u; row < rows; row++) {
        const uint64_t reference_offset =
            (uint64_t)row * reference_stride + column0;
        const uint64_t candidate_offset =
            (uint64_t)row * candidate_stride;
        diff_metrics d = compare_f32(reference + reference_offset,
                                     candidate + candidate_offset, width);
        if (d.mismatches && !total.mismatches)
            total.first = reference_offset + d.first;
        total.mismatches += d.mismatches;
        if (d.max_abs > total.max_abs) total.max_abs = d.max_abs;
    }
    return total;
}

static void print_diff(const char *boundary, diff_metrics home,
                       diff_metrics partner) {
    printf("boundary=%s,home_mismatches=%llu,partner_mismatches=%llu,"
           "home_first=%lld,partner_first=%lld,max_abs=%.9g\n",
           boundary,
           (unsigned long long)home.mismatches,
           (unsigned long long)partner.mismatches,
           home.first == UINT64_MAX ? -1ll : (long long)home.first,
           partner.first == UINT64_MAX ? -1ll : (long long)partner.first,
           home.max_abs > partner.max_abs ? home.max_abs : partner.max_abs);
}

static float rope_attention_factor(void) {
    return 1.0f / (1.0f + 0.1f * logf(1.0f / ROPE_SCALE));
}

static int sync_tier(int tier) {
    return ds4_gpu_set_current_device(tier) == 0 && ds4_gpu_synchronize();
}

int main(void) {
    const uint64_t input_count = (uint64_t)N_TOK * IN_DIM;
    const uint64_t input_bytes = input_count * sizeof(float);
    const uint64_t q_count = (uint64_t)N_TOK * OUT_DIM;
    const uint64_t q_bytes = q_count * sizeof(float);
    const uint64_t half_q_count = (uint64_t)N_TOK * SHARD_OUT_DIM;
    const uint64_t half_q_bytes = half_q_count * sizeof(uint16_t);
    const uint64_t q8_row_bytes = (IN_DIM / 32u) * 34u;
    const uint64_t q_b_bytes = OUT_DIM * q8_row_bytes;
    const uint64_t q_b_half_bytes = q_b_bytes / 2u;
    const uint64_t q_model_bytes = 2u * q_b_bytes;
    const uint64_t raw_count = (uint64_t)RAW_CAP * HEAD_DIM;
    const uint64_t comp_count = (uint64_t)N_COMP * HEAD_DIM;
    const uint64_t topk_count = (uint64_t)N_TOK * TOP_K;
    const uint64_t low_count = (uint64_t)N_TOK * LOW_DIM;
    const uint64_t half_low_count = (uint64_t)N_TOK * HALF_LOW_DIM;
    const uint64_t a_row_bytes = (GROUP_DIM / 32u) * 34u;
    const uint64_t a_bytes = LOW_DIM * a_row_bytes;
    const uint64_t b_row_bytes = (LOW_DIM / 32u) * 34u;
    const uint64_t b_bytes = EMBED_DIM * b_row_bytes;
    const uint64_t b_offset = a_bytes;
    const uint64_t sinks_offset = b_offset + b_bytes;
    const uint64_t attn_model_bytes = sinks_offset + N_HEAD * sizeof(float);
    const uint64_t output_count = (uint64_t)N_TOK * EMBED_DIM;
    const uint64_t output_bytes = output_count * sizeof(float);
    int initialized = 0, status = 1;

    unsigned char *q_model = NULL, *attn_model = NULL;
    float *input_host = NULL, *raw_host = NULL, *comp_host = NULL;
    int32_t *topk_host = NULL;
    float *ref_host = NULL, *home_host = NULL, *peer_host = NULL;
    float *low_ref_host = NULL, *low_gather_host = NULL;
    float *output_ref_host = NULL, *output_candidate_host = NULL;

    ds4_gpu_tensor input_home = {0}, input_peer = {0};
    ds4_gpu_tensor q_ref = {0}, q_home = {0}, q_peer = {0};
    ds4_gpu_tensor qh_ref = {0}, qh_home = {0}, qh_peer = {0};
    ds4_gpu_tensor raw_home = {0}, raw_peer = {0};
    ds4_gpu_tensor comp_home = {0}, comp_peer = {0}, topk_home = {0};
    ds4_gpu_tensor heads_ref = {0}, heads_home = {0}, heads_peer = {0};
    ds4_gpu_tensor low_ref = {0}, low_home = {0}, low_peer = {0};
    ds4_gpu_tensor low_gather = {0};
    ds4_gpu_tensor output_ref = {0}, output_candidate = {0};
    ds4_gpu_tensor output_scheduled = {0};

    q_model = (unsigned char *)malloc((size_t)q_model_bytes);
    attn_model = (unsigned char *)calloc(1u, (size_t)attn_model_bytes);
    input_host = (float *)malloc((size_t)input_bytes);
    raw_host = (float *)malloc((size_t)raw_count * sizeof(float));
    comp_host = (float *)malloc((size_t)comp_count * sizeof(float));
    topk_host = (int32_t *)malloc((size_t)topk_count * sizeof(int32_t));
    ref_host = (float *)malloc((size_t)q_bytes);
    home_host = (float *)malloc((size_t)q_bytes);
    peer_host = (float *)malloc((size_t)q_bytes);
    low_ref_host = (float *)malloc((size_t)low_count * sizeof(float));
    low_gather_host = (float *)malloc((size_t)low_count * sizeof(float));
    output_ref_host = (float *)malloc((size_t)output_bytes);
    output_candidate_host = (float *)malloc((size_t)output_bytes);
    CHECK(q_model && attn_model && input_host && raw_host && comp_host &&
          topk_host && ref_host && home_host && peer_host && low_ref_host &&
          low_gather_host && output_ref_host && output_candidate_host,
          "host allocations");

    build_q8_rows(q_model, OUT_DIM, IN_DIM, 19u);
    memcpy(q_model + q_b_bytes, q_model, (size_t)q_b_bytes);
    build_q8_rows(attn_model, LOW_DIM, GROUP_DIM, 31u);
    build_q8_rows(attn_model + b_offset, EMBED_DIM, LOW_DIM, 43u);
    for (uint64_t i = 0u; i < input_count; i++)
        input_host[i] = (float)((int)((i * 29u + (i >> 5u) * 17u +
                              (i / IN_DIM) * 7u + 23u) % 257u) - 128) /
                        128.0f;
    for (uint64_t i = 0u; i < raw_count; i++)
        raw_host[i] = (float)((int)((i * 13u + 37u) % 257u) - 128) /
                      2048.0f;
    for (uint64_t i = 0u; i < comp_count; i++)
        comp_host[i] = (float)((int)((i * 31u + 41u) % 263u) - 131) /
                       2048.0f;
    for (uint32_t token = 0u; token < N_TOK; token++)
        for (uint32_t k = 0u; k < TOP_K; k++)
            topk_host[(uint64_t)token * TOP_K + k] =
                (int32_t)(((uint64_t)token * 131u +
                           (uint64_t)k * 17u + 19u) % N_COMP);

    (void)unsetenv("DS4_CUDA_COPY_MODEL");
    (void)setenv("DS4_CUDA_Q8_F16_CACHE_MB", "512", 1);
    (void)setenv("DS4_CUDA_Q8_F16_CACHE_RESERVE_MB", "1", 1);
    (void)setenv("DS4_CUDA_NO_TF32", "1", 1);
    (void)setenv("DS4_CUDA_T32_F16_FUSED", "1", 1);
    (void)unsetenv("DS4_CUDA_NO_T32_F16_FUSED");
    (void)unsetenv("DS4_CUDA_NO_Q8_F16_CACHE");
    (void)unsetenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE");

    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 1;
    CHECK(ds4_gpu_init_multi(&cfg), "two-GPU initialization");
    initialized = 1;
    CHECK(g_gpu_peer_ok[0][1] && g_gpu_peer_ok[1][0],
          "bidirectional peer access required");

    CHECK(ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_set_model_map(q_model, q_model_bytes), "q_b model map");
    /* Multi-GPU F16 materialization deliberately refuses to stage arbitrary
     * host-map bytes: its source must already have the same selective-cache
     * ownership that production placement installs.  Admit the synthetic
     * full/home-half and peer-half sources before asking for F16 views. */
    ds4_tensor_range q_home_sources[] = {
        {0u, q_b_bytes, 0},
        {q_b_bytes, q_b_half_bytes, 0},
    };
    ds4_tensor_range q_peer_source = {
        q_b_bytes + q_b_half_bytes, q_b_half_bytes, 1
    };
    CHECK(ds4_gpu_device_cache_tensors(
              0, q_home_sources,
              (int)(sizeof(q_home_sources) / sizeof(q_home_sources[0]))) == 0 &&
          ds4_gpu_device_cache_tensors(1, &q_peer_source, 1) == 0,
          "q_b selective source caches");
    CHECK(ds4_gpu_cache_q8_f16_range_on_device(
              q_model, q_model_bytes, 0u, q_b_bytes, IN_DIM, OUT_DIM,
              g_gpu[0].device_id, "attn_q_b_full") &&
          ds4_gpu_cache_q8_f16_range_on_device(
              q_model, q_model_bytes, q_b_bytes, q_b_half_bytes,
              IN_DIM, SHARD_OUT_DIM, g_gpu[0].device_id,
              "attn_q_b_head0") &&
          ds4_gpu_cache_q8_f16_range_on_device(
              q_model, q_model_bytes, q_b_bytes + q_b_half_bytes,
              q_b_half_bytes, IN_DIM, SHARD_OUT_DIM, g_gpu[1].device_id,
              "attn_q_b_head1"), "q_b device caches");

#define ALLOC(t, tier, bytes) (ds4_gpu_tensor_alloc_on(&(t), (tier), (bytes)) == 0)
    CHECK(ALLOC(input_home, 0, input_bytes) && ALLOC(input_peer, 1, input_bytes) &&
          ALLOC(q_ref, 0, q_bytes) && ALLOC(q_home, 0, q_bytes) &&
          ALLOC(q_peer, 1, q_bytes) &&
          ALLOC(qh_ref, 0, q_count * sizeof(uint16_t)) &&
          ALLOC(qh_home, 0, half_q_bytes) && ALLOC(qh_peer, 1, half_q_bytes),
          "q_b device allocations");
    CHECK(ds4_gpu_tensor_write(&input_home, 0u, input_host, input_bytes),
          "home input write");
    CHECK(ds4_gpu_tensor_wait_xdev_default(&input_peer, 0) &&
          ds4_gpu_tensor_copy_xdev_default(&input_peer, &input_home,
                                           input_bytes),
          "production input copy");

    CHECK(ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
              &q_ref, &qh_ref, q_model, q_model_bytes, 0u, IN_DIM, OUT_DIM,
              &input_home, N_TOK, N_HEAD, HEAD_DIM, N_ROT, POS0, ORIG_CTX,
              false, ROPE_BASE, ROPE_SCALE, ROPE_EXT,
              rope_attention_factor(), BETA_FAST, BETA_SLOW, EPSILON) &&
          sync_tier(0), "full q_b reference");
    CHECK(ds4_gpu_set_current_device(1) == 0 &&
          ds4_gpu_attn_q_b_f16_head_shard_rms_rope_tail_tensor(
              &q_peer, &qh_peer, q_model, q_model_bytes,
              q_b_bytes + q_b_half_bytes, IN_DIM, SHARD_OUT_DIM,
              &input_peer, N_TOK, SHARD_HEADS, SHARD_HEADS, N_HEAD,
              HEAD_DIM, N_ROT, POS0, ORIG_CTX, false, ROPE_BASE, ROPE_SCALE,
              ROPE_EXT, rope_attention_factor(), BETA_FAST, BETA_SLOW,
              EPSILON) &&
          ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_attn_q_b_f16_head_shard_rms_rope_tail_tensor(
              &q_home, &qh_home, q_model, q_model_bytes, q_b_bytes,
              IN_DIM, SHARD_OUT_DIM, &input_home, N_TOK, SHARD_HEADS, 0u,
              N_HEAD, HEAD_DIM, N_ROT, POS0, ORIG_CTX, false, ROPE_BASE,
              ROPE_SCALE, ROPE_EXT, rope_attention_factor(), BETA_FAST,
              BETA_SLOW, EPSILON) && sync_tier(0) && sync_tier(1),
          "physical q_b shards");
    CHECK(ds4_gpu_tensor_read(&q_ref, 0u, ref_host, q_bytes) &&
          ds4_gpu_tensor_read(&q_home, 0u, home_host, q_bytes) &&
          ds4_gpu_tensor_read(&q_peer, 0u, peer_host, q_bytes),
          "q_b readback");
    diff_metrics q_home_diff = compare_head_half(ref_host, home_host, 0u);
    diff_metrics q_peer_diff = compare_head_half(ref_host, peer_host,
                                                 SHARD_HEADS);
    print_diff("physical-q-b", q_home_diff, q_peer_diff);
    CHECK(!q_home_diff.mismatches && !q_peer_diff.mismatches,
          "physical q_b diverged");

    CHECK(sync_tier(0) && sync_tier(1) &&
          ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_set_model_map(attn_model, attn_model_bytes),
          "attention model map");
    ds4_tensor_range attn_home_sources[] = {
        {0u, a_bytes, 0},
        {b_offset, b_bytes, 0},
        {sinks_offset, N_HEAD * sizeof(float), 0},
    };
    ds4_tensor_range attn_peer_sources[] = {
        {a_bytes / 2u, a_bytes / 2u, 1},
        {sinks_offset, N_HEAD * sizeof(float), 1},
    };
    CHECK(ds4_gpu_device_cache_tensors(
              0, attn_home_sources,
              (int)(sizeof(attn_home_sources) /
                    sizeof(attn_home_sources[0]))) == 0 &&
          ds4_gpu_device_cache_tensors(
              1, attn_peer_sources,
              (int)(sizeof(attn_peer_sources) /
                    sizeof(attn_peer_sources[0]))) == 0,
          "attention selective source caches");
    CHECK(ds4_gpu_cache_q8_f16_range_on_device(
              attn_model, attn_model_bytes, 0u, a_bytes, GROUP_DIM,
              LOW_DIM, g_gpu[0].device_id, "attn_output_a_full") &&
          ds4_gpu_cache_q8_f16_range_on_device(
              attn_model, attn_model_bytes, 0u, a_bytes / 2u, GROUP_DIM,
              HALF_LOW_DIM, g_gpu[0].device_id, "attn_output_a_head0") &&
          ds4_gpu_cache_q8_f16_range_on_device(
              attn_model, attn_model_bytes, a_bytes / 2u, a_bytes / 2u,
              GROUP_DIM, HALF_LOW_DIM, g_gpu[1].device_id,
              "attn_output_a_head1") &&
          ds4_gpu_cache_q8_f16_range_on_device(
              attn_model, attn_model_bytes, b_offset, b_bytes, LOW_DIM,
              EMBED_DIM, g_gpu[0].device_id, "attn_output_b"),
          "attention-output device caches");
    CHECK(ALLOC(raw_home, 0, raw_count * sizeof(float)) &&
          ALLOC(raw_peer, 1, raw_count * sizeof(float)) &&
          ALLOC(comp_home, 0, comp_count * sizeof(float)) &&
          ALLOC(comp_peer, 1, comp_count * sizeof(float)) &&
          ALLOC(topk_home, 0, topk_count * sizeof(int32_t)) &&
          ALLOC(heads_ref, 0, q_bytes) && ALLOC(heads_home, 0, q_bytes) &&
          ALLOC(heads_peer, 1, q_bytes) &&
          ALLOC(low_ref, 0, low_count * sizeof(float)) &&
          ALLOC(low_home, 0, half_low_count * sizeof(float)) &&
          ALLOC(low_peer, 1, half_low_count * sizeof(float)) &&
          ALLOC(low_gather, 0, low_count * sizeof(float)) &&
          ALLOC(output_ref, 0, output_bytes) &&
          ALLOC(output_candidate, 0, output_bytes) &&
          ALLOC(output_scheduled, 0, output_bytes),
          "downstream device allocations");
    CHECK(ds4_gpu_tensor_write(&raw_home, 0u, raw_host,
                               raw_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&raw_peer, 0u, raw_host,
                               raw_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&comp_home, 0u, comp_host,
                               comp_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&comp_peer, 0u, comp_host,
                               comp_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&topk_home, 0u, topk_host,
                               topk_count * sizeof(int32_t)),
          "downstream input writes");

    CHECK(ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
              &heads_ref, attn_model, attn_model_bytes, sinks_offset,
              &q_ref, &raw_home, &comp_home, 0u, &topk_home, N_TOK, POS0,
              N_RAW, RAW_CAP, RAW_START, N_COMP, TOP_K, ATTN_WINDOW,
              ATTN_RATIO, N_HEAD, HEAD_DIM) && sync_tier(0),
          "full attention reference");
    CHECK(ds4_gpu_tensor_wait_xdev_default(&topk_home, 1) &&
          ds4_gpu_set_current_device(1) == 0 &&
          ds4_gpu_attention_indexed_mixed_batch_heads_shard_tensor(
              &heads_peer, attn_model, attn_model_bytes, sinks_offset,
              &q_peer, &raw_peer, &comp_peer, 0u, &topk_home, N_TOK, POS0,
              N_RAW, RAW_CAP, RAW_START, N_COMP, TOP_K, ATTN_WINDOW,
              ATTN_RATIO, SHARD_HEADS, SHARD_HEADS, N_HEAD, HEAD_DIM) &&
          ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_attention_indexed_mixed_batch_heads_shard_tensor(
              &heads_home, attn_model, attn_model_bytes, sinks_offset,
              &q_home, &raw_home, &comp_home, 0u, &topk_home, N_TOK, POS0,
              N_RAW, RAW_CAP, RAW_START, N_COMP, TOP_K, ATTN_WINDOW,
              ATTN_RATIO, 0u, SHARD_HEADS, N_HEAD, HEAD_DIM) &&
          sync_tier(0) && sync_tier(1), "physical attention shards");
    CHECK(ds4_gpu_tensor_read(&heads_ref, 0u, ref_host, q_bytes) &&
          ds4_gpu_tensor_read(&heads_home, 0u, home_host, q_bytes) &&
          ds4_gpu_tensor_read(&heads_peer, 0u, peer_host, q_bytes),
          "attention readback");
    diff_metrics attn_home_diff = compare_head_half(ref_host, home_host, 0u);
    diff_metrics attn_peer_diff = compare_head_half(ref_host, peer_host,
                                                    SHARD_HEADS);
    print_diff("physical-indexed-attention", attn_home_diff, attn_peer_diff);
    CHECK(!attn_home_diff.mismatches && !attn_peer_diff.mismatches,
          "physical indexed attention diverged");

    CHECK(ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_rope_tail_tensor(&heads_ref, N_TOK, N_HEAD, HEAD_DIM, N_ROT,
              POS0, ORIG_CTX, true, ROPE_BASE, ROPE_SCALE, ROPE_EXT,
              rope_attention_factor(), BETA_FAST, BETA_SLOW) &&
          ds4_gpu_rope_tail_head_range_tensor(
              &heads_home, N_TOK, 0u, SHARD_HEADS, N_HEAD, HEAD_DIM, N_ROT,
              POS0, ORIG_CTX, true, ROPE_BASE, ROPE_SCALE, ROPE_EXT,
              rope_attention_factor(), BETA_FAST, BETA_SLOW) &&
          ds4_gpu_set_current_device(1) == 0 &&
          ds4_gpu_rope_tail_head_range_tensor(
              &heads_peer, N_TOK, SHARD_HEADS, SHARD_HEADS, N_HEAD, HEAD_DIM,
              N_ROT, POS0, ORIG_CTX, true, ROPE_BASE, ROPE_SCALE, ROPE_EXT,
              rope_attention_factor(), BETA_FAST, BETA_SLOW) &&
          sync_tier(0) && sync_tier(1), "physical inverse RoPE");
    CHECK(ds4_gpu_tensor_read(&heads_ref, 0u, ref_host, q_bytes) &&
          ds4_gpu_tensor_read(&heads_home, 0u, home_host, q_bytes) &&
          ds4_gpu_tensor_read(&heads_peer, 0u, peer_host, q_bytes),
          "inverse RoPE readback");
    diff_metrics rope_home_diff = compare_head_half(ref_host, home_host, 0u);
    diff_metrics rope_peer_diff = compare_head_half(ref_host, peer_host,
                                                    SHARD_HEADS);
    print_diff("physical-inverse-rope", rope_home_diff, rope_peer_diff);
    CHECK(!rope_home_diff.mismatches && !rope_peer_diff.mismatches,
          "physical inverse RoPE diverged");

    CHECK(ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_attention_output_q8_batch_low_shard_tensor(
              &low_ref, attn_model, attn_model_bytes, 0u, GROUP_DIM,
              OUT_A_RANK, N_GROUP, 0u, N_GROUP, &heads_ref, N_TOK) &&
          ds4_gpu_set_current_device(1) == 0 &&
          ds4_gpu_attention_output_q8_batch_low_shard_tensor(
              &low_peer, attn_model, attn_model_bytes, 0u, GROUP_DIM,
              OUT_A_RANK, N_GROUP, HALF_GROUP, HALF_GROUP, &heads_peer,
              N_TOK) &&
          ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_attention_output_q8_batch_low_shard_tensor(
              &low_home, attn_model, attn_model_bytes, 0u, GROUP_DIM,
              OUT_A_RANK, N_GROUP, 0u, HALF_GROUP, &heads_home, N_TOK) &&
          ds4_gpu_tensor_wait_xdev_default(&low_peer, 0) &&
          ds4_gpu_gather_pair_rows_xdev_tensor(
              &low_gather, &low_home, &low_peer, N_TOK, HALF_LOW_DIM) &&
          sync_tier(0) && sync_tier(1), "physical output-A and gather");
    CHECK(ds4_gpu_tensor_read(&low_ref, 0u, low_ref_host,
                              low_count * sizeof(float)) &&
          ds4_gpu_tensor_read(&low_gather, 0u, low_gather_host,
                              low_count * sizeof(float)),
          "output-A gather readback");
    diff_metrics low_diff = compare_f32(low_ref_host, low_gather_host,
                                        low_count);
    printf("boundary=physical-output-a-gather,mismatches=%llu,first=%lld,"
           "max_abs=%.9g\n", (unsigned long long)low_diff.mismatches,
           low_diff.first == UINT64_MAX ? -1ll : (long long)low_diff.first,
           low_diff.max_abs);
    CHECK(!low_diff.mismatches, "physical output-A gather diverged");

    CHECK(ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_attention_output_q8_batch_b_tensor(
              &output_ref, attn_model, attn_model_bytes, b_offset, LOW_DIM,
              EMBED_DIM, &low_ref, N_TOK) &&
          ds4_gpu_attention_output_q8_batch_b_tensor(
              &output_candidate, attn_model, attn_model_bytes, b_offset,
              LOW_DIM, EMBED_DIM, &low_gather, N_TOK) &&
          sync_tier(0), "physical output-B");
    CHECK(ds4_gpu_tensor_read(&output_ref, 0u, output_ref_host,
                              output_bytes) &&
          ds4_gpu_tensor_read(&output_candidate, 0u, output_candidate_host,
                              output_bytes), "output-B readback");
    diff_metrics output_b_diff = compare_f32(
        output_ref_host, output_candidate_host, output_count);
    printf("boundary=physical-output-b,mismatches=%llu,first=%lld,"
           "max_abs=%.9g\n",
           (unsigned long long)output_b_diff.mismatches,
           output_b_diff.first == UINT64_MAX ? -1ll :
               (long long)output_b_diff.first,
           output_b_diff.max_abs);
    CHECK(!output_b_diff.mismatches, "physical output-B diverged");

    /* Repeat the candidate exactly as production orders it: no host-side
     * synchronization between q_b, attention, inverse RoPE, output-A, peer
     * gather, and output-B.  Only the explicit cross-device dependencies that
     * production uses are present. */
    CHECK(ds4_gpu_tensor_wait_xdev_default(&input_peer, 0) &&
          ds4_gpu_tensor_copy_xdev_default(&input_peer, &input_home,
                                           input_bytes),
          "scheduled input copy");
    CHECK(ds4_gpu_set_current_device(1) == 0 &&
          ds4_gpu_attn_q_b_f16_head_shard_rms_rope_tail_tensor(
              &q_peer, &qh_peer, q_model, q_model_bytes,
              q_b_bytes + q_b_half_bytes, IN_DIM, SHARD_OUT_DIM,
              &input_peer, N_TOK, SHARD_HEADS, SHARD_HEADS, N_HEAD,
              HEAD_DIM, N_ROT, POS0, ORIG_CTX, false, ROPE_BASE, ROPE_SCALE,
              ROPE_EXT, rope_attention_factor(), BETA_FAST, BETA_SLOW,
              EPSILON) &&
          ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_attn_q_b_f16_head_shard_rms_rope_tail_tensor(
              &q_home, &qh_home, q_model, q_model_bytes, q_b_bytes,
              IN_DIM, SHARD_OUT_DIM, &input_home, N_TOK, SHARD_HEADS, 0u,
              N_HEAD, HEAD_DIM, N_ROT, POS0, ORIG_CTX, false, ROPE_BASE,
              ROPE_SCALE, ROPE_EXT, rope_attention_factor(), BETA_FAST,
              BETA_SLOW, EPSILON), "scheduled q_b shards");
    CHECK(ds4_gpu_tensor_wait_xdev_default(&topk_home, 1) &&
          ds4_gpu_set_current_device(1) == 0 &&
          ds4_gpu_attention_indexed_mixed_batch_heads_shard_tensor(
              &heads_peer, attn_model, attn_model_bytes, sinks_offset,
              &q_peer, &raw_peer, &comp_peer, 0u, &topk_home, N_TOK, POS0,
              N_RAW, RAW_CAP, RAW_START, N_COMP, TOP_K, ATTN_WINDOW,
              ATTN_RATIO, SHARD_HEADS, SHARD_HEADS, N_HEAD, HEAD_DIM) &&
          ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_attention_indexed_mixed_batch_heads_shard_tensor(
              &heads_home, attn_model, attn_model_bytes, sinks_offset,
              &q_home, &raw_home, &comp_home, 0u, &topk_home, N_TOK, POS0,
              N_RAW, RAW_CAP, RAW_START, N_COMP, TOP_K, ATTN_WINDOW,
              ATTN_RATIO, 0u, SHARD_HEADS, N_HEAD, HEAD_DIM),
          "scheduled attention shards");
    CHECK(ds4_gpu_set_current_device(1) == 0 &&
          ds4_gpu_rope_tail_head_range_tensor(
              &heads_peer, N_TOK, SHARD_HEADS, SHARD_HEADS, N_HEAD,
              HEAD_DIM, N_ROT, POS0, ORIG_CTX, true, ROPE_BASE, ROPE_SCALE,
              ROPE_EXT, rope_attention_factor(), BETA_FAST, BETA_SLOW) &&
          ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_rope_tail_head_range_tensor(
              &heads_home, N_TOK, 0u, SHARD_HEADS, N_HEAD, HEAD_DIM, N_ROT,
              POS0, ORIG_CTX, true, ROPE_BASE, ROPE_SCALE, ROPE_EXT,
              rope_attention_factor(), BETA_FAST, BETA_SLOW),
          "scheduled inverse RoPE");
    CHECK(ds4_gpu_set_current_device(1) == 0 &&
          ds4_gpu_attention_output_q8_batch_low_shard_tensor(
              &low_peer, attn_model, attn_model_bytes, 0u, GROUP_DIM,
              OUT_A_RANK, N_GROUP, HALF_GROUP, HALF_GROUP, &heads_peer,
              N_TOK) &&
          ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_attention_output_q8_batch_low_shard_tensor(
              &low_home, attn_model, attn_model_bytes, 0u, GROUP_DIM,
              OUT_A_RANK, N_GROUP, 0u, HALF_GROUP, &heads_home, N_TOK) &&
          ds4_gpu_tensor_wait_xdev_default(&low_peer, 0) &&
          ds4_gpu_gather_pair_rows_xdev_tensor(
              &low_gather, &low_home, &low_peer, N_TOK, HALF_LOW_DIM) &&
          ds4_gpu_attention_output_q8_batch_b_tensor(
              &output_scheduled, attn_model, attn_model_bytes, b_offset,
              LOW_DIM, EMBED_DIM, &low_gather, N_TOK) &&
          sync_tier(0) && sync_tier(1),
          "production-ordered head-shard chain");

    CHECK(ds4_gpu_tensor_read(&input_peer, 0u, home_host, input_bytes),
          "scheduled input readback");
    diff_metrics scheduled_input_diff = compare_f32(
        input_host, home_host, input_count);
    printf("boundary=physical-production-ordered-input-copy,mismatches=%llu,"
           "first=%lld,max_abs=%.9g\n",
           (unsigned long long)scheduled_input_diff.mismatches,
           scheduled_input_diff.first == UINT64_MAX ? -1ll :
               (long long)scheduled_input_diff.first,
           scheduled_input_diff.max_abs);

    CHECK(ds4_gpu_tensor_read(&q_ref, 0u, ref_host, q_bytes) &&
          ds4_gpu_tensor_read(&q_home, 0u, home_host, q_bytes) &&
          ds4_gpu_tensor_read(&q_peer, 0u, peer_host, q_bytes),
          "scheduled q_b readback");
    diff_metrics scheduled_q_home_diff =
        compare_head_half(ref_host, home_host, 0u);
    diff_metrics scheduled_q_peer_diff =
        compare_head_half(ref_host, peer_host, SHARD_HEADS);
    print_diff("physical-production-ordered-q-b",
               scheduled_q_home_diff, scheduled_q_peer_diff);

    CHECK(ds4_gpu_tensor_read(&heads_ref, 0u, ref_host, q_bytes) &&
          ds4_gpu_tensor_read(&heads_home, 0u, home_host, q_bytes) &&
          ds4_gpu_tensor_read(&heads_peer, 0u, peer_host, q_bytes),
          "scheduled post-RoPE readback");
    diff_metrics scheduled_heads_home_diff =
        compare_head_half(ref_host, home_host, 0u);
    diff_metrics scheduled_heads_peer_diff =
        compare_head_half(ref_host, peer_host, SHARD_HEADS);
    print_diff("physical-production-ordered-post-rope",
               scheduled_heads_home_diff, scheduled_heads_peer_diff);

    CHECK(ds4_gpu_tensor_read(&low_ref, 0u, low_ref_host,
                              low_count * sizeof(float)) &&
          ds4_gpu_tensor_read(&low_home, 0u, low_gather_host,
                              half_low_count * sizeof(float)),
          "scheduled home output-A readback");
    diff_metrics scheduled_low_home_diff = compare_row_slice(
        low_ref_host, LOW_DIM, low_gather_host, HALF_LOW_DIM, N_TOK, 0u,
        HALF_LOW_DIM);
    CHECK(ds4_gpu_tensor_read(&low_peer, 0u, low_gather_host,
                              half_low_count * sizeof(float)),
          "scheduled peer output-A readback");
    diff_metrics scheduled_low_peer_diff = compare_row_slice(
        low_ref_host, LOW_DIM, low_gather_host, HALF_LOW_DIM, N_TOK,
        HALF_LOW_DIM, HALF_LOW_DIM);
    print_diff("physical-production-ordered-output-a-halves",
               scheduled_low_home_diff, scheduled_low_peer_diff);

    CHECK(ds4_gpu_tensor_read(&low_gather, 0u, low_gather_host,
                              low_count * sizeof(float)),
          "scheduled gather readback");
    diff_metrics scheduled_gather_diff = compare_f32(
        low_ref_host, low_gather_host, low_count);
    printf("boundary=physical-production-ordered-output-a-gather,"
           "mismatches=%llu,first=%lld,max_abs=%.9g\n",
           (unsigned long long)scheduled_gather_diff.mismatches,
           scheduled_gather_diff.first == UINT64_MAX ? -1ll :
               (long long)scheduled_gather_diff.first,
           scheduled_gather_diff.max_abs);

    CHECK(ds4_gpu_tensor_read(&output_scheduled, 0u, output_candidate_host,
                              output_bytes), "scheduled output readback");
    diff_metrics scheduled_diff = compare_f32(
        output_ref_host, output_candidate_host, output_count);
    printf("boundary=physical-production-ordered-output-b,mismatches=%llu,"
           "first=%lld,max_abs=%.9g\n",
           (unsigned long long)scheduled_diff.mismatches,
           scheduled_diff.first == UINT64_MAX ? -1ll :
               (long long)scheduled_diff.first,
           scheduled_diff.max_abs);
    CHECK(!scheduled_input_diff.mismatches &&
          !scheduled_q_home_diff.mismatches &&
          !scheduled_q_peer_diff.mismatches &&
          !scheduled_heads_home_diff.mismatches &&
          !scheduled_heads_peer_diff.mismatches &&
          !scheduled_low_home_diff.mismatches &&
          !scheduled_low_peer_diff.mismatches &&
          !scheduled_gather_diff.mismatches && !scheduled_diff.mismatches,
          "production-ordered head-shard chain diverged");

    printf("output_a_rank=%u\nphysical_pair_protocol=bit-exact\n"
           "production_ordered_protocol=bit-exact\nharness_status=ok\n",
           OUT_A_RANK);
    status = 0;

cleanup:
#define FREE(t) ds4_gpu_tensor_free_in_place(&(t))
    FREE(output_scheduled); FREE(output_candidate); FREE(output_ref);
    FREE(low_gather); FREE(low_peer); FREE(low_home); FREE(low_ref);
    FREE(heads_peer); FREE(heads_home); FREE(heads_ref);
    FREE(topk_home); FREE(comp_peer); FREE(comp_home);
    FREE(raw_peer); FREE(raw_home);
    FREE(qh_peer); FREE(qh_home); FREE(qh_ref);
    FREE(q_peer); FREE(q_home); FREE(q_ref);
    FREE(input_peer); FREE(input_home);
    if (initialized) ds4_gpu_cleanup();
    free(output_candidate_host); free(output_ref_host);
    free(low_gather_host); free(low_ref_host);
    free(peer_host); free(home_host); free(ref_host);
    free(topk_host); free(comp_host); free(raw_host); free(input_host);
    free(attn_model); free(q_model);
    return status;
}
