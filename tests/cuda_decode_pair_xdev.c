/* Bounded SM75 one-token decode experiment for one NVLink pair.
 *
 * indexer: split compressed rows between the home and partner GPU, gather the
 * second score half, then run the unchanged production top-k on the home GPU.
 * The persistent native-F16 index cache is mirrored once; the timed transfer
 * case includes the per-token query and head-weight handoff plus score gather.
 *
 * attention: split the 64 attention heads 32/32, using mirrored raw and
 * compressed KV caches.  The timed transfer case includes the partner query
 * half, top-k, one raw row, one compressed row, and result gather.
 *
 * Both candidates are checked bit-for-bit against the shipping one-GPU
 * one-token paths before timing.  This file is a mechanism harness, not a
 * production dispatch. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);       \
            goto cleanup;                                                   \
        }                                                                   \
    } while (0)

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void borrow_view(ds4_gpu_tensor *view, const ds4_gpu_tensor *base,
                        uint64_t offset, uint64_t bytes) {
    memset(view, 0, sizeof(*view));
    view->ptr = (unsigned char *)base->ptr + offset;
    view->bytes = bytes;
    view->device_id = base->device_id;
}

static uint32_t env_u32(const char *name, uint32_t fallback,
                        uint32_t minimum, uint32_t maximum) {
    const char *value = getenv(name);
    if (!value || !value[0]) return fallback;
    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    if (!end || *end || parsed < minimum || parsed > maximum) return 0u;
    return (uint32_t)parsed;
}

static int sync_tier(int tier) {
    return ds4_gpu_set_current_device(tier) == 0 && ds4_gpu_synchronize();
}

static int init_pair(int *home_physical, int *partner_physical) {
    const uint32_t home = env_u32("DS4_DECODE_HOME_GPU", 0u, 0u,
                                  DS4_MAX_GPUS - 1u);
    const uint32_t partner = env_u32("DS4_DECODE_PARTNER_GPU", 1u, 0u,
                                     DS4_MAX_GPUS - 1u);
    if (home == partner) {
        fprintf(stderr, "error: decode pair requires two distinct GPUs\n");
        return 0;
    }
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = (int)home;
    cfg.device_indices[1] = (int)partner;
    if (!ds4_gpu_init_multi(&cfg)) return 0;
    if (!g_gpu_peer_ok[0][1] || !g_gpu_peer_ok[1][0]) {
        fprintf(stderr,
                "error: physical GPUs %u and %u require bidirectional peer access\n",
                home, partner);
        ds4_gpu_cleanup();
        return 0;
    }
    *home_physical = (int)home;
    *partner_physical = (int)partner;
    return 1;
}

static int compare_f32_bits(const float *reference, const float *candidate,
                            uint64_t count, const char *label) {
    if (memcmp(reference, candidate, (size_t)count * sizeof(float)) == 0) {
        for (uint64_t i = 0; i < count; i++) {
            if (reference[i] != 0.0f && isfinite(reference[i])) return 1;
        }
        fprintf(stderr, "FAIL: %s output is bit-exact but entirely zero\n",
                label);
        return 0;
    }
    for (uint64_t i = 0; i < count; i++) {
        if (memcmp(reference + i, candidate + i, sizeof(float)) != 0) {
            fprintf(stderr,
                    "FAIL: %s[%llu] reference=%a candidate=%a\n",
                    label, (unsigned long long)i,
                    reference[i], candidate[i]);
            return 0;
        }
    }
    return 0;
}

static int compare_u32(const uint32_t *reference, const uint32_t *candidate,
                       uint64_t count, const char *label) {
    if (memcmp(reference, candidate,
               (size_t)count * sizeof(uint32_t)) == 0) return 1;
    for (uint64_t i = 0; i < count; i++) {
        if (reference[i] != candidate[i]) {
            fprintf(stderr,
                    "FAIL: %s[%llu] reference=%u candidate=%u\n",
                    label, (unsigned long long)i,
                    reference[i], candidate[i]);
            return 0;
        }
    }
    return 0;
}

static int run_indexer(void) {
    const uint32_t n_comp = 7936u;
    const uint32_t home_rows = n_comp / 2u;
    const uint32_t partner_rows = n_comp - home_rows;
    const uint32_t n_head = 64u, head_dim = 128u, top_k = 512u;
    const float scale = 0.125f;
    const uint32_t repeats = env_u32("DS4_DECODE_PAIR_REPEATS", 100u,
                                     1u, 10000u);
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t weight_count = n_head;
    const uint64_t cache_count = (uint64_t)n_comp * head_dim;
    const uint64_t home_cache_count = (uint64_t)home_rows * head_dim;
    const uint64_t partner_cache_count =
        (uint64_t)partner_rows * head_dim;
    int ok = 0, initialized = 0, home_physical = -1, partner_physical = -1;

    float *q_host = NULL, *weights_host = NULL, *cache_host = NULL;
    float *reference_host = NULL, *candidate_host = NULL;
    uint32_t *selected_reference_host = NULL, *selected_candidate_host = NULL;
    ds4_gpu_tensor q0 = {0}, weights0 = {0}, cache0_f32 = {0};
    ds4_gpu_tensor cache0_f16 = {0}, reference = {0}, candidate = {0};
    ds4_gpu_tensor selected_reference = {0}, selected_candidate = {0};
    ds4_gpu_tensor q1 = {0}, weights1 = {0}, cache1_f16 = {0};
    ds4_gpu_tensor partner_scores = {0};
    ds4_gpu_tensor home_scores_view = {0}, gathered_scores_view = {0};
    ds4_gpu_tensor home_cache_view = {0}, partner_cache_view = {0};

    if (!repeats) {
        fprintf(stderr, "error: DS4_DECODE_PAIR_REPEATS must be 1..10000\n");
        return 0;
    }
    q_host = (float *)malloc((size_t)q_count * sizeof(float));
    weights_host = (float *)malloc((size_t)weight_count * sizeof(float));
    cache_host = (float *)malloc((size_t)cache_count * sizeof(float));
    reference_host = (float *)malloc((size_t)n_comp * sizeof(float));
    candidate_host = (float *)malloc((size_t)n_comp * sizeof(float));
    selected_reference_host =
        (uint32_t *)malloc((size_t)top_k * sizeof(uint32_t));
    selected_candidate_host =
        (uint32_t *)malloc((size_t)top_k * sizeof(uint32_t));
    CHECK(q_host && weights_host && cache_host && reference_host &&
          candidate_host && selected_reference_host && selected_candidate_host,
          "indexer host allocations");

    for (uint64_t i = 0; i < q_count; i++)
        q_host[i] = (float)((int)((i * 17u + (i >> 7u) * 13u) % 251u) -
                            125) / 191.0f;
    for (uint64_t i = 0; i < weight_count; i++)
        weights_host[i] =
            (float)((int)((i * 19u + 7u) % 97u) - 48) / 113.0f;
    for (uint64_t i = 0; i < cache_count; i++)
        cache_host[i] =
            (float)((int)((i * 23u + (i >> 6u) * 5u) % 257u) - 128) /
            211.0f;

    CHECK(init_pair(&home_physical, &partner_physical),
          "indexer two-GPU initialization");
    initialized = 1;
    CHECK(ds4_gpu_tensor_alloc_on(&q0, 0, q_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&weights0, 0,
                                  weight_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&cache0_f32, 0,
                                  cache_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&cache0_f16, 0,
                                  cache_count * sizeof(uint16_t)) == 0 &&
          ds4_gpu_tensor_alloc_on(&reference, 0,
                                  (uint64_t)n_comp * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&candidate, 0,
                                  (uint64_t)n_comp * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&selected_reference, 0,
                                  (uint64_t)top_k * sizeof(uint32_t)) == 0 &&
          ds4_gpu_tensor_alloc_on(&selected_candidate, 0,
                                  (uint64_t)top_k * sizeof(uint32_t)) == 0 &&
          ds4_gpu_tensor_alloc_on(&q1, 1, q_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&weights1, 1,
                                  weight_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&cache1_f16, 1,
                                  cache_count * sizeof(uint16_t)) == 0 &&
          ds4_gpu_tensor_alloc_on(&partner_scores, 1,
                                  (uint64_t)partner_rows * sizeof(float)) == 0,
          "indexer device allocations");
    CHECK(ds4_gpu_tensor_write(&q0, 0, q_host,
                               q_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&weights0, 0, weights_host,
                               weight_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&cache0_f32, 0, cache_host,
                               cache_count * sizeof(float)) &&
          ds4_gpu_dsv4_indexer_qat_tensor(&q0, n_head, head_dim) &&
          ds4_gpu_dsv4_indexer_qat_tensor(&cache0_f32, n_comp, head_dim) &&
          ds4_gpu_tensor_copy_f32_to_f16(&cache0_f16, 0u, &cache0_f32,
                                         0u, cache_count) &&
          sync_tier(0), "indexer QAT and native-cache materialization");
    CHECK(ds4_gpu_tensor_copy_xdev_default(&q1, &q0,
                                           q_count * sizeof(float)) &&
          ds4_gpu_tensor_copy_xdev_default(&weights1, &weights0,
                                           weight_count * sizeof(float)) &&
          ds4_gpu_tensor_copy_xdev_default(&cache1_f16, &cache0_f16,
                                           cache_count * sizeof(uint16_t)) &&
          sync_tier(0) && sync_tier(1), "indexer persistent mirror setup");

    borrow_view(&home_scores_view, &candidate, 0u,
                (uint64_t)home_rows * sizeof(float));
    borrow_view(&gathered_scores_view, &candidate,
                (uint64_t)home_rows * sizeof(float),
                (uint64_t)partner_rows * sizeof(float));
    borrow_view(&home_cache_view, &cache0_f16, 0u,
                home_cache_count * sizeof(uint16_t));
    borrow_view(&partner_cache_view, &cache1_f16,
                home_cache_count * sizeof(uint16_t),
                partner_cache_count * sizeof(uint16_t));

#define INDEXER_HOME()                                                       \
    ds4_gpu_indexer_score_one_f16_cache_tensor(                              \
        &home_scores_view, &q0, &weights0, &home_cache_view,                 \
        home_rows, n_head, head_dim, scale)
#define INDEXER_PARTNER()                                                    \
    ds4_gpu_indexer_score_one_f16_cache_tensor(                              \
        &partner_scores, &q1, &weights1, &partner_cache_view,                \
        partner_rows, n_head, head_dim, scale)
#define INDEXER_GATHER()                                                     \
    ds4_gpu_tensor_copy_xdev_default(                                        \
        &gathered_scores_view, &partner_scores,                              \
        (uint64_t)partner_rows * sizeof(float))

    CHECK(ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_indexer_score_one_f16_cache_tensor(
              &reference, &q0, &weights0, &cache0_f16,
              n_comp, n_head, head_dim, scale) &&
          ds4_gpu_indexer_topk_tensor(&selected_reference, &reference,
                                      n_comp, 1u, top_k) && sync_tier(0),
          "indexer baseline chain");
    CHECK(ds4_gpu_set_current_device(0) == 0 && INDEXER_HOME() &&
          ds4_gpu_set_current_device(1) == 0 && INDEXER_PARTNER() &&
          INDEXER_GATHER() &&
          ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_indexer_topk_tensor(&selected_candidate, &candidate,
                                      n_comp, 1u, top_k) &&
          sync_tier(0) && sync_tier(1), "indexer split chain");
    CHECK(ds4_gpu_tensor_read(&reference, 0, reference_host,
                              (uint64_t)n_comp * sizeof(float)) &&
          ds4_gpu_tensor_read(&candidate, 0, candidate_host,
                              (uint64_t)n_comp * sizeof(float)) &&
          ds4_gpu_tensor_read(&selected_reference, 0,
                              selected_reference_host,
                              (uint64_t)top_k * sizeof(uint32_t)) &&
          ds4_gpu_tensor_read(&selected_candidate, 0,
                              selected_candidate_host,
                              (uint64_t)top_k * sizeof(uint32_t)),
          "indexer result reads");
    CHECK(compare_f32_bits(reference_host, candidate_host, n_comp,
                           "indexer scores"), "indexer score exactness");
    CHECK(compare_u32(selected_reference_host, selected_candidate_host,
                      top_k, "indexer top-k"), "indexer top-k exactness");

    double start = now_seconds();
    for (uint32_t r = 0; r < repeats; r++) {
        CHECK(ds4_gpu_set_current_device(0) == 0 &&
              ds4_gpu_indexer_score_one_f16_cache_tensor(
                  &reference, &q0, &weights0, &cache0_f16,
                  n_comp, n_head, head_dim, scale) &&
              ds4_gpu_indexer_topk_tensor(&selected_reference, &reference,
                                          n_comp, 1u, top_k),
              "indexer timed baseline chain");
    }
    CHECK(sync_tier(0), "indexer baseline timing sync");
    const double baseline_chain_ms =
        (now_seconds() - start) * 1000.0 / (double)repeats;

    start = now_seconds();
    for (uint32_t r = 0; r < repeats; r++) {
        CHECK(ds4_gpu_set_current_device(0) == 0 && INDEXER_HOME() &&
              ds4_gpu_set_current_device(1) == 0 && INDEXER_PARTNER(),
              "indexer timed split scores");
    }
    CHECK(sync_tier(0) && sync_tier(1), "indexer split score timing sync");
    const double split_score_ms =
        (now_seconds() - start) * 1000.0 / (double)repeats;

    start = now_seconds();
    for (uint32_t r = 0; r < repeats; r++) {
        CHECK(ds4_gpu_set_current_device(0) == 0 && INDEXER_HOME() &&
              ds4_gpu_set_current_device(1) == 0 && INDEXER_PARTNER() &&
              INDEXER_GATHER() &&
              ds4_gpu_set_current_device(0) == 0 &&
              ds4_gpu_indexer_topk_tensor(&selected_candidate, &candidate,
                                          n_comp, 1u, top_k),
              "indexer timed mirrored chain");
    }
    CHECK(sync_tier(0) && sync_tier(1), "indexer mirrored timing sync");
    const double mirrored_chain_ms =
        (now_seconds() - start) * 1000.0 / (double)repeats;

    start = now_seconds();
    for (uint32_t r = 0; r < repeats; r++) {
        CHECK(ds4_gpu_tensor_copy_xdev_default(&q1, &q0,
                                               q_count * sizeof(float)) &&
              ds4_gpu_tensor_copy_xdev_default(&weights1, &weights0,
                                               weight_count * sizeof(float)) &&
              ds4_gpu_set_current_device(0) == 0 && INDEXER_HOME() &&
              ds4_gpu_set_current_device(1) == 0 && INDEXER_PARTNER() &&
              INDEXER_GATHER() &&
              ds4_gpu_set_current_device(0) == 0 &&
              ds4_gpu_indexer_topk_tensor(&selected_candidate, &candidate,
                                          n_comp, 1u, top_k),
              "indexer timed transfer chain");
    }
    CHECK(sync_tier(0) && sync_tier(1), "indexer transfer timing sync");
    const double transfer_chain_ms =
        (now_seconds() - start) * 1000.0 / (double)repeats;

    printf("scenario=decode-indexer-row-split-32k\n");
    printf("validation=bit-exact-nonzero\nrepeats=%u\n", repeats);
    printf("home_physical_gpu=%d\npartner_physical_gpu=%d\n",
           home_physical, partner_physical);
    printf("n_comp=%u\nhome_rows=%u\npartner_rows=%u\ntop_k=%u\n",
           n_comp, home_rows, partner_rows, top_k);
    printf("baseline_chain_ms=%.6f\nsplit_score_ms=%.6f\n"
           "mirrored_chain_ms=%.6f\ntransfer_inclusive_chain_ms=%.6f\n",
           baseline_chain_ms, split_score_ms,
           mirrored_chain_ms, transfer_chain_ms);
    printf("mirrored_chain_speedup=%.6f\n"
           "transfer_inclusive_chain_speedup=%.6f\n",
           baseline_chain_ms / mirrored_chain_ms,
           baseline_chain_ms / transfer_chain_ms);
    printf("persistent_partner_index_cache_bytes=%llu\n",
           (unsigned long long)(cache_count * sizeof(uint16_t)));
    printf("partner_query_bytes=%llu\npartner_weight_bytes=%llu\n"
           "partner_score_gather_bytes=%llu\n",
           (unsigned long long)(q_count * sizeof(float)),
           (unsigned long long)(weight_count * sizeof(float)),
           (unsigned long long)((uint64_t)partner_rows * sizeof(float)));
    ok = 1;

#undef INDEXER_GATHER
#undef INDEXER_PARTNER
#undef INDEXER_HOME

cleanup:
    ds4_gpu_tensor_free_in_place(&partner_scores);
    ds4_gpu_tensor_free_in_place(&cache1_f16);
    ds4_gpu_tensor_free_in_place(&weights1);
    ds4_gpu_tensor_free_in_place(&q1);
    ds4_gpu_tensor_free_in_place(&selected_candidate);
    ds4_gpu_tensor_free_in_place(&selected_reference);
    ds4_gpu_tensor_free_in_place(&candidate);
    ds4_gpu_tensor_free_in_place(&reference);
    ds4_gpu_tensor_free_in_place(&cache0_f16);
    ds4_gpu_tensor_free_in_place(&cache0_f32);
    ds4_gpu_tensor_free_in_place(&weights0);
    ds4_gpu_tensor_free_in_place(&q0);
    if (initialized) ds4_gpu_cleanup();
    free(selected_candidate_host);
    free(selected_reference_host);
    free(candidate_host);
    free(reference_host);
    free(cache_host);
    free(weights_host);
    free(q_host);
    return ok;
}

static int launch_attention_half(
        ds4_gpu_tensor *heads, const void *model, uint64_t model_bytes,
        uint64_t sinks_offset,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw,
        const ds4_gpu_tensor *comp, const ds4_gpu_tensor *topk,
        uint32_t n_tokens, uint32_t pos0, uint32_t n_raw,
        uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp,
        uint32_t top_k, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim) {
    return ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        heads, model, model_bytes, sinks_offset, q, raw, comp, 0u, topk,
        n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
        window, ratio, n_head, head_dim);
}

static int run_attention(void) {
    const uint32_t n_tokens = 1u, pos0 = 32768u;
    const uint32_t n_raw = 2304u, raw_cap = 2304u, raw_start = 0u;
    const uint32_t n_comp = 7936u, top_k = 512u;
    const uint32_t window = 2048u, ratio = 4u;
    const uint32_t n_head = 64u, head_dim = 512u;
    const uint32_t home_heads = 32u, partner_heads = 32u;
    const uint32_t repeats = env_u32("DS4_DECODE_PAIR_REPEATS", 100u,
                                     1u, 10000u);
    const uint64_t row_count = (uint64_t)n_head * head_dim;
    const uint64_t half_count = (uint64_t)partner_heads * head_dim;
    const uint64_t raw_count = (uint64_t)raw_cap * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t model_bytes = (uint64_t)n_head * sizeof(float);
    const uint32_t raw_update_row = pos0 % raw_cap;
    const uint32_t comp_update_row = n_comp - 1u;
    int ok = 0, initialized = 0, home_physical = -1, partner_physical = -1;

    float *model = NULL, *q_host = NULL, *raw_host = NULL, *comp_host = NULL;
    float *reference_host = NULL, *candidate_host = NULL;
    int32_t *topk_host = NULL;
    ds4_gpu_tensor q0 = {0}, raw0 = {0}, comp0 = {0}, topk0 = {0};
    ds4_gpu_tensor reference = {0}, candidate = {0};
    ds4_gpu_tensor q1 = {0}, raw1 = {0}, comp1 = {0}, topk1 = {0};
    ds4_gpu_tensor partner_output = {0};
    ds4_gpu_tensor q0_home_view = {0}, q0_partner_view = {0};
    ds4_gpu_tensor q1_partner_view = {0};
    ds4_gpu_tensor candidate_home_view = {0}, candidate_partner_view = {0};
    ds4_gpu_tensor partner_result_view = {0};
    ds4_gpu_tensor raw0_update = {0}, raw1_update = {0};
    ds4_gpu_tensor comp0_update = {0}, comp1_update = {0};

    if (!repeats) {
        fprintf(stderr, "error: DS4_DECODE_PAIR_REPEATS must be 1..10000\n");
        return 0;
    }
    model = (float *)malloc((size_t)model_bytes);
    q_host = (float *)malloc((size_t)row_count * sizeof(float));
    raw_host = (float *)malloc((size_t)raw_count * sizeof(float));
    comp_host = (float *)malloc((size_t)comp_count * sizeof(float));
    topk_host = (int32_t *)malloc((size_t)top_k * sizeof(int32_t));
    reference_host = (float *)malloc((size_t)row_count * sizeof(float));
    candidate_host = (float *)malloc((size_t)row_count * sizeof(float));
    CHECK(model && q_host && raw_host && comp_host && topk_host &&
          reference_host && candidate_host, "attention host allocations");

    for (uint32_t h = 0; h < n_head; h++)
        model[h] = (float)((int)(h % 9u) - 4) * 0.03125f;
    for (uint64_t i = 0; i < row_count; i++)
        q_host[i] =
            (float)((int)((i * 17u + 3u) % 31u) - 15) / 1024.0f;
    for (uint64_t i = 0; i < raw_count; i++)
        raw_host[i] =
            (float)((int)((i * 13u + 5u) % 29u) - 14) / 512.0f;
    for (uint64_t i = 0; i < comp_count; i++)
        comp_host[i] =
            (float)((int)((i * 11u + 7u) % 27u) - 13) / 512.0f;
    for (uint32_t k = 0; k < top_k; k++)
        topk_host[k] = (int32_t)(((uint64_t)k * 17u + 23u) % n_comp);

    CHECK(init_pair(&home_physical, &partner_physical),
          "attention two-GPU initialization");
    initialized = 1;
    CHECK(ds4_gpu_tensor_alloc_on(&q0, 0, row_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&raw0, 0,
                                  raw_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&comp0, 0,
                                  comp_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&topk0, 0,
                                  (uint64_t)top_k * sizeof(int32_t)) == 0 &&
          ds4_gpu_tensor_alloc_on(&reference, 0,
                                  row_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&candidate, 0,
                                  row_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&q1, 1, row_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&raw1, 1,
                                  raw_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&comp1, 1,
                                  comp_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&topk1, 1,
                                  (uint64_t)top_k * sizeof(int32_t)) == 0 &&
          ds4_gpu_tensor_alloc_on(&partner_output, 1,
                                  row_count * sizeof(float)) == 0,
          "attention device allocations");
    CHECK(ds4_gpu_tensor_write(&q0, 0, q_host,
                               row_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&raw0, 0, raw_host,
                               raw_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&comp0, 0, comp_host,
                               comp_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&topk0, 0, topk_host,
                               (uint64_t)top_k * sizeof(int32_t)) &&
          ds4_gpu_tensor_write(&q1, 0, q_host + half_count,
                               half_count * sizeof(float)) &&
          ds4_gpu_tensor_copy_xdev_default(&raw1, &raw0,
                                           raw_count * sizeof(float)) &&
          ds4_gpu_tensor_copy_xdev_default(&comp1, &comp0,
                                           comp_count * sizeof(float)) &&
          ds4_gpu_tensor_copy_xdev_default(&topk1, &topk0,
                                           (uint64_t)top_k * sizeof(int32_t)) &&
          sync_tier(0) && sync_tier(1), "attention persistent mirror setup");

    CHECK(ds4_gpu_register_model_map_no_copy(model, model_bytes),
          "attention model-map registration");
    ds4_tensor_range sink0 = {0u, model_bytes, home_physical};
    ds4_tensor_range sink1 = {0u, model_bytes, partner_physical};
    CHECK(ds4_gpu_device_cache_tensors(home_physical, &sink0, 1) == 0 &&
          ds4_gpu_device_cache_tensors(partner_physical, &sink1, 1) == 0,
          "attention sink cache on both devices");
    CHECK(sync_tier(0) && sync_tier(1), "attention initial synchronization");

    borrow_view(&q0_home_view, &q0, 0u,
                half_count * sizeof(float));
    borrow_view(&q0_partner_view, &q0,
                (uint64_t)home_heads * head_dim * sizeof(float),
                half_count * sizeof(float));
    borrow_view(&q1_partner_view, &q1, 0u,
                half_count * sizeof(float));
    borrow_view(&candidate_home_view, &candidate, 0u,
                half_count * sizeof(float));
    borrow_view(&candidate_partner_view, &candidate,
                (uint64_t)home_heads * head_dim * sizeof(float),
                half_count * sizeof(float));
    borrow_view(&partner_result_view, &partner_output, 0u,
                half_count * sizeof(float));
    borrow_view(&raw0_update, &raw0,
                (uint64_t)raw_update_row * head_dim * sizeof(float),
                (uint64_t)head_dim * sizeof(float));
    borrow_view(&raw1_update, &raw1,
                (uint64_t)raw_update_row * head_dim * sizeof(float),
                (uint64_t)head_dim * sizeof(float));
    borrow_view(&comp0_update, &comp0,
                (uint64_t)comp_update_row * head_dim * sizeof(float),
                (uint64_t)head_dim * sizeof(float));
    borrow_view(&comp1_update, &comp1,
                (uint64_t)comp_update_row * head_dim * sizeof(float),
                (uint64_t)head_dim * sizeof(float));

#define ATTN_HOME()                                                          \
    launch_attention_half(                                                   \
        &candidate_home_view, model, model_bytes, 0u,                       \
        &q0_home_view, &raw0, &comp0, &topk0,                               \
        n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k, window,   \
        ratio, home_heads, head_dim)
#define ATTN_PARTNER()                                                       \
    launch_attention_half(                                                   \
        &partner_result_view, model, model_bytes,                            \
        (uint64_t)home_heads * sizeof(float),                               \
        &q1_partner_view, &raw1, &comp1, &topk1,                            \
        n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k, window,   \
        ratio, partner_heads, head_dim)
#define ATTN_GATHER()                                                        \
    ds4_gpu_tensor_copy_xdev_default(                                        \
        &candidate_partner_view, &partner_result_view,                       \
        half_count * sizeof(float))

    CHECK(ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
              &reference, model, model_bytes, 0u, &q0, &raw0, &comp0,
              0u, &topk0, n_tokens, pos0, n_raw, raw_cap, raw_start,
              n_comp, top_k, window, ratio, n_head, head_dim) && sync_tier(0),
          "attention baseline launch");
    CHECK(ds4_gpu_set_current_device(0) == 0 && ATTN_HOME() &&
          ds4_gpu_set_current_device(1) == 0 && ATTN_PARTNER() &&
          ATTN_GATHER() && sync_tier(0) && sync_tier(1),
          "attention head-split launch");
    CHECK(ds4_gpu_tensor_read(&reference, 0, reference_host,
                              row_count * sizeof(float)) &&
          ds4_gpu_tensor_read(&candidate, 0, candidate_host,
                              row_count * sizeof(float)),
          "attention result reads");
    CHECK(compare_f32_bits(reference_host, candidate_host, row_count,
                           "indexed attention heads"),
          "attention head-split exactness");

    double start = now_seconds();
    for (uint32_t r = 0; r < repeats; r++) {
        CHECK(ds4_gpu_set_current_device(0) == 0 &&
              ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                  &reference, model, model_bytes, 0u, &q0, &raw0, &comp0,
                  0u, &topk0, n_tokens, pos0, n_raw, raw_cap, raw_start,
                  n_comp, top_k, window, ratio, n_head, head_dim),
              "attention timed baseline");
    }
    CHECK(sync_tier(0), "attention baseline timing sync");
    const double baseline_ms =
        (now_seconds() - start) * 1000.0 / (double)repeats;

    start = now_seconds();
    for (uint32_t r = 0; r < repeats; r++) {
        CHECK(ds4_gpu_set_current_device(0) == 0 && ATTN_HOME() &&
              ds4_gpu_set_current_device(1) == 0 && ATTN_PARTNER(),
              "attention timed split compute");
    }
    CHECK(sync_tier(0) && sync_tier(1), "attention split timing sync");
    const double split_compute_ms =
        (now_seconds() - start) * 1000.0 / (double)repeats;

    start = now_seconds();
    for (uint32_t r = 0; r < repeats; r++) {
        CHECK(ds4_gpu_set_current_device(0) == 0 && ATTN_HOME() &&
              ds4_gpu_set_current_device(1) == 0 && ATTN_PARTNER() &&
              ATTN_GATHER(), "attention timed mirrored chain");
    }
    CHECK(sync_tier(0) && sync_tier(1), "attention mirrored timing sync");
    const double mirrored_ms =
        (now_seconds() - start) * 1000.0 / (double)repeats;

    start = now_seconds();
    for (uint32_t r = 0; r < repeats; r++) {
        CHECK(ds4_gpu_tensor_copy_xdev_default(
                  &q1_partner_view, &q0_partner_view,
                  half_count * sizeof(float)) &&
              ds4_gpu_tensor_copy_xdev_default(
                  &topk1, &topk0, (uint64_t)top_k * sizeof(int32_t)) &&
              ds4_gpu_tensor_copy_xdev_default(
                  &raw1_update, &raw0_update,
                  (uint64_t)head_dim * sizeof(float)) &&
              ds4_gpu_tensor_copy_xdev_default(
                  &comp1_update, &comp0_update,
                  (uint64_t)head_dim * sizeof(float)) &&
              ds4_gpu_set_current_device(0) == 0 && ATTN_HOME() &&
              ds4_gpu_set_current_device(1) == 0 && ATTN_PARTNER() &&
              ATTN_GATHER(), "attention timed transfer chain");
    }
    CHECK(sync_tier(0) && sync_tier(1), "attention transfer timing sync");
    const double transfer_ms =
        (now_seconds() - start) * 1000.0 / (double)repeats;

    printf("scenario=decode-indexed-attention-head-split-32k\n");
    printf("validation=bit-exact-nonzero\nrepeats=%u\n", repeats);
    printf("home_physical_gpu=%d\npartner_physical_gpu=%d\n",
           home_physical, partner_physical);
    printf("home_heads=%u\npartner_heads=%u\nn_comp=%u\ntop_k=%u\n",
           home_heads, partner_heads, n_comp, top_k);
    printf("baseline_ms=%.6f\nsplit_compute_ms=%.6f\n"
           "mirrored_chain_ms=%.6f\ntransfer_inclusive_ms=%.6f\n",
           baseline_ms, split_compute_ms, mirrored_ms, transfer_ms);
    printf("mirrored_chain_speedup=%.6f\n"
           "transfer_inclusive_speedup=%.6f\n",
           baseline_ms / mirrored_ms, baseline_ms / transfer_ms);
    printf("persistent_partner_kv_bytes=%llu\n",
           (unsigned long long)((raw_count + comp_count) * sizeof(float)));
    printf("partner_query_bytes=%llu\npartner_topk_bytes=%llu\n"
           "partner_raw_update_bytes=%llu\n"
           "partner_compressed_update_bytes=%llu\n"
           "partner_output_gather_bytes=%llu\n",
           (unsigned long long)(half_count * sizeof(float)),
           (unsigned long long)((uint64_t)top_k * sizeof(int32_t)),
           (unsigned long long)((uint64_t)head_dim * sizeof(float)),
           (unsigned long long)((uint64_t)head_dim * sizeof(float)),
           (unsigned long long)(half_count * sizeof(float)));
    ok = 1;

#undef ATTN_GATHER
#undef ATTN_PARTNER
#undef ATTN_HOME

cleanup:
    ds4_gpu_tensor_free_in_place(&partner_output);
    ds4_gpu_tensor_free_in_place(&topk1);
    ds4_gpu_tensor_free_in_place(&comp1);
    ds4_gpu_tensor_free_in_place(&raw1);
    ds4_gpu_tensor_free_in_place(&q1);
    ds4_gpu_tensor_free_in_place(&candidate);
    ds4_gpu_tensor_free_in_place(&reference);
    ds4_gpu_tensor_free_in_place(&topk0);
    ds4_gpu_tensor_free_in_place(&comp0);
    ds4_gpu_tensor_free_in_place(&raw0);
    ds4_gpu_tensor_free_in_place(&q0);
    if (initialized) ds4_gpu_cleanup();
    free(candidate_host);
    free(reference_host);
    free(topk_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    free(model);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2 ||
        (strcmp(argv[1], "indexer") && strcmp(argv[1], "attention"))) {
        fprintf(stderr, "Usage: %s indexer|attention\n", argv[0]);
        return 2;
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        fprintf(stderr, "error: at least two visible CUDA devices are required\n");
        return 2;
    }
    return (strcmp(argv[1], "indexer") == 0
                ? run_indexer()
                : run_attention()) ? 0 : 1;
}
