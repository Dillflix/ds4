/* Bounded SM75 two-GPU attention row-split experiment.  This is a harness,
 * not a production dispatch.  It compares the shipping 512-row launch with
 * concurrent 256/256 launches using either peer-read or mirrored partner
 * inputs.  Query rows are independent, so adjusted pos0/n_raw preserves the
 * shipping kernel's arithmetic order and permits bit-exact validation. */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <cuda_runtime.h>
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

static uint32_t get_repeats(void) {
    const char *env = getenv("DS4_ROWSPLIT_REPEATS");
    if (!env || !env[0]) return 5u;
    char *end = NULL;
    unsigned long n = strtoul(env, &end, 10);
    return end && !*end && n >= 1ul && n <= 50ul ? (uint32_t)n : 0u;
}

static int sync_tier(int tier) {
    return ds4_gpu_set_current_device(tier) == 0 && ds4_gpu_synchronize();
}

static int launch_attention(
        int indexed, ds4_gpu_tensor *heads,
        const void *model, uint64_t model_bytes,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw,
        const ds4_gpu_tensor *comp, const ds4_gpu_tensor *topk,
        uint32_t n_tokens, uint32_t pos0, uint32_t n_raw,
        uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp,
        uint32_t top_k, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim) {
    if (indexed) {
        return ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
            heads, model, model_bytes, 0u, q, raw, comp, 0u, topk,
            n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
            window, ratio, n_head, head_dim);
    }
    return ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        heads, model, model_bytes, 0u, q, raw, comp, 0u, NULL, 0u,
        n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, window,
        ratio, n_head, head_dim);
}

static int compare_halves(const float *ref, const float *home,
                          const float *partner, uint64_t half_count,
                          const char *label) {
    const size_t bytes = (size_t)half_count * sizeof(float);
    if (memcmp(ref, home, bytes) == 0 &&
        memcmp(ref + half_count, partner, bytes) == 0) return 1;
    for (uint64_t i = 0; i < half_count; i++) {
        if (memcmp(ref + i, home + i, sizeof(float)) != 0) {
            fprintf(stderr,
                    "FAIL: %s home[%llu] reference=%a candidate=%a\n",
                    label, (unsigned long long)i, ref[i], home[i]);
            return 0;
        }
        if (memcmp(ref + half_count + i, partner + i,
                   sizeof(float)) != 0) {
            fprintf(stderr,
                    "FAIL: %s partner[%llu] reference=%a candidate=%a\n",
                    label, (unsigned long long)i,
                    ref[half_count + i], partner[i]);
            return 0;
        }
    }
    return 0;
}

static int run_case(int indexed) {
    const uint32_t n_tokens = 512u, half_tokens = 256u;
    const uint32_t pos0 = 31744u, partner_pos0 = 32000u;
    const uint32_t n_raw = 2304u, home_n_raw = 2048u;
    const uint32_t raw_cap = 2304u, raw_start = 0u;
    const uint32_t n_comp = 7936u, top_k = 512u;
    const uint32_t window = 2048u, ratio = 4u;
    const uint32_t n_head = 64u, head_dim = 512u;
    const uint64_t row_count = (uint64_t)n_head * head_dim;
    const uint64_t q_count = (uint64_t)n_tokens * row_count;
    const uint64_t half_count = (uint64_t)half_tokens * row_count;
    const uint64_t raw_count = (uint64_t)raw_cap * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t topk_count = (uint64_t)n_tokens * top_k;
    const uint64_t topk_half_count = (uint64_t)half_tokens * top_k;
    const uint64_t model_bytes = (uint64_t)n_head * sizeof(float);
    const uint32_t repeats = get_repeats();
    int ok = 0, initialized = 0;

    float *model = NULL, *q_host = NULL, *raw_host = NULL;
    float *comp_host = NULL, *ref_host = NULL;
    float *home_host = NULL, *partner_host = NULL;
    int32_t *topk_host = NULL;
    ds4_gpu_tensor q0 = {0}, raw0 = {0}, comp0 = {0}, topk0 = {0};
    ds4_gpu_tensor ref = {0}, home = {0}, partner = {0};
    ds4_gpu_tensor q1 = {0}, raw1 = {0}, comp1 = {0}, topk1 = {0};
    ds4_gpu_tensor q_home = {0}, q_peer = {0};
    ds4_gpu_tensor topk_home = {0}, topk_peer = {0};

    if (!repeats) {
        fprintf(stderr,
                "error: DS4_ROWSPLIT_REPEATS must be an integer from 1 to 50\n");
        return 0;
    }
    model = (float *)malloc((size_t)model_bytes);
    q_host = (float *)malloc((size_t)q_count * sizeof(float));
    raw_host = (float *)malloc((size_t)raw_count * sizeof(float));
    comp_host = (float *)malloc((size_t)comp_count * sizeof(float));
    ref_host = (float *)malloc((size_t)q_count * sizeof(float));
    home_host = (float *)malloc((size_t)half_count * sizeof(float));
    partner_host = (float *)malloc((size_t)half_count * sizeof(float));
    if (indexed)
        topk_host = (int32_t *)malloc((size_t)topk_count * sizeof(int32_t));
    CHECK(model && q_host && raw_host && comp_host && ref_host && home_host &&
          partner_host && (!indexed || topk_host), "host allocations");

    for (uint32_t h = 0; h < n_head; h++)
        model[h] = (float)((int)(h % 9u) - 4) * 0.03125f;
    for (uint64_t i = 0; i < q_count; i++)
        q_host[i] = (float)((int)((i * 17u + 3u) % 31u) - 15) /
                    1024.0f;
    for (uint64_t i = 0; i < raw_count; i++)
        raw_host[i] = (float)((int)((i * 13u + 5u) % 29u) - 14) /
                      512.0f;
    for (uint64_t i = 0; i < comp_count; i++)
        comp_host[i] = (float)((int)((i * 11u + 7u) % 27u) - 13) /
                       512.0f;
    if (indexed) {
        for (uint32_t t = 0; t < n_tokens; t++)
            for (uint32_t k = 0; k < top_k; k++)
                topk_host[(uint64_t)t * top_k + k] =
                    (int32_t)(((uint64_t)t * 37u +
                               (uint64_t)k * 17u) % n_comp);
    }

    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 1;
    CHECK(ds4_gpu_init_multi(&cfg), "two-GPU initialization");
    initialized = 1;
    CHECK(g_gpu_peer_ok[0][1] && g_gpu_peer_ok[1][0],
          "bidirectional peer access is required");

    CHECK(ds4_gpu_tensor_alloc_on(&q0, 0, q_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&raw0, 0, raw_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&comp0, 0, comp_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&ref, 0, q_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&home, 0, half_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&partner, 1,
                                  half_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&q1, 1, half_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&raw1, 1, raw_count * sizeof(float)) == 0 &&
          ds4_gpu_tensor_alloc_on(&comp1, 1,
                                  comp_count * sizeof(float)) == 0,
          "device allocations");
    if (indexed) {
        CHECK(ds4_gpu_tensor_alloc_on(&topk0, 0,
                                      topk_count * sizeof(int32_t)) == 0 &&
              ds4_gpu_tensor_alloc_on(&topk1, 1,
                                      topk_half_count * sizeof(int32_t)) == 0,
              "top-k allocations");
    }

    CHECK(ds4_gpu_tensor_write(&q0, 0, q_host,
                               q_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&raw0, 0, raw_host,
                               raw_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&comp0, 0, comp_host,
                               comp_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&q1, 0, q_host + half_count,
                               half_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&raw1, 0, raw_host,
                               raw_count * sizeof(float)) &&
          ds4_gpu_tensor_write(&comp1, 0, comp_host,
                               comp_count * sizeof(float)) &&
          (!indexed ||
           (ds4_gpu_tensor_write(&topk0, 0, topk_host,
                                 topk_count * sizeof(int32_t)) &&
            ds4_gpu_tensor_write(&topk1, 0,
                                 topk_host + topk_half_count,
                                 topk_half_count * sizeof(int32_t)))),
          "device input writes");

    CHECK(ds4_gpu_set_current_device(0) == 0 &&
          ds4_gpu_set_model_map(model, model_bytes), "model-map setup");
    ds4_tensor_range sink0 = {0u, model_bytes, 0};
    ds4_tensor_range sink1 = {0u, model_bytes, 1};
    CHECK(ds4_gpu_device_cache_tensors(0, &sink0, 1) == 0 &&
          ds4_gpu_device_cache_tensors(1, &sink1, 1) == 0,
          "sink cache on both devices");
    CHECK(sync_tier(0) && sync_tier(1), "initial synchronization");

    borrow_view(&q_home, &q0, 0u, half_count * sizeof(float));
    borrow_view(&q_peer, &q0, half_count * sizeof(float),
                half_count * sizeof(float));
    if (indexed) {
        borrow_view(&topk_home, &topk0, 0u,
                    topk_half_count * sizeof(int32_t));
        borrow_view(&topk_peer, &topk0,
                    topk_half_count * sizeof(int32_t),
                    topk_half_count * sizeof(int32_t));
    }

#define LAUNCH_HOME() \
    launch_attention(indexed, &home, model, model_bytes, &q_home, &raw0, \
                     &comp0, indexed ? &topk_home : NULL, half_tokens, pos0, \
                     home_n_raw, raw_cap, raw_start, n_comp, top_k, window, \
                     ratio, n_head, head_dim)
#define LAUNCH_PARTNER(q_, raw_, comp_, topk_) \
    launch_attention(indexed, &partner, model, model_bytes, (q_), (raw_), \
                     (comp_), indexed ? (topk_) : NULL, half_tokens, \
                     partner_pos0, n_raw, raw_cap, raw_start, n_comp, top_k, \
                     window, ratio, n_head, head_dim)

    CHECK(ds4_gpu_set_current_device(0) == 0 &&
          launch_attention(indexed, &ref, model, model_bytes, &q0, &raw0,
                           &comp0, indexed ? &topk0 : NULL, n_tokens, pos0,
                           n_raw, raw_cap, raw_start, n_comp, top_k, window,
                           ratio, n_head, head_dim) && sync_tier(0),
          "baseline launch");
    CHECK(ds4_gpu_tensor_read(&ref, 0, ref_host,
                              q_count * sizeof(float)), "baseline read");

    CHECK(ds4_gpu_set_current_device(0) == 0 && LAUNCH_HOME(),
          "peer home launch");
    CHECK(ds4_gpu_set_current_device(1) == 0 &&
          LAUNCH_PARTNER(&q_peer, &raw0, &comp0, &topk_peer),
          "peer partner launch");
    CHECK(sync_tier(0) && sync_tier(1), "peer split synchronization");
    CHECK(ds4_gpu_tensor_read(&home, 0, home_host,
                              half_count * sizeof(float)) &&
          ds4_gpu_tensor_read(&partner, 0, partner_host,
                              half_count * sizeof(float)), "peer result read");
    CHECK(compare_halves(ref_host, home_host, partner_host, half_count,
                         "peer"), "peer exactness");

    CHECK(ds4_gpu_set_current_device(0) == 0 && LAUNCH_HOME(),
          "mirror home launch");
    CHECK(ds4_gpu_set_current_device(1) == 0 &&
          LAUNCH_PARTNER(&q1, &raw1, &comp1, &topk1),
          "mirror partner launch");
    CHECK(sync_tier(0) && sync_tier(1), "mirror split synchronization");
    CHECK(ds4_gpu_tensor_read(&home, 0, home_host,
                              half_count * sizeof(float)) &&
          ds4_gpu_tensor_read(&partner, 0, partner_host,
                              half_count * sizeof(float)), "mirror result read");
    CHECK(compare_halves(ref_host, home_host, partner_host, half_count,
                         "mirror"), "mirror exactness");

    double start = now_seconds();
    CHECK(ds4_gpu_set_current_device(0) == 0, "baseline timing device");
    for (uint32_t r = 0; r < repeats; r++)
        CHECK(launch_attention(indexed, &ref, model, model_bytes, &q0, &raw0,
                               &comp0, indexed ? &topk0 : NULL, n_tokens,
                               pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
                               window, ratio, n_head, head_dim),
              "baseline timed launch");
    CHECK(sync_tier(0), "baseline timing synchronization");
    const double baseline_ms =
        (now_seconds() - start) * 1000.0 / (double)repeats;

    start = now_seconds();
    for (uint32_t r = 0; r < repeats; r++) {
        CHECK(ds4_gpu_set_current_device(0) == 0 && LAUNCH_HOME(),
              "peer timed home launch");
        CHECK(ds4_gpu_set_current_device(1) == 0 &&
              LAUNCH_PARTNER(&q_peer, &raw0, &comp0, &topk_peer),
              "peer timed partner launch");
    }
    CHECK(sync_tier(0) && sync_tier(1), "peer timing synchronization");
    const double peer_ms =
        (now_seconds() - start) * 1000.0 / (double)repeats;

    start = now_seconds();
    for (uint32_t r = 0; r < repeats; r++) {
        CHECK(ds4_gpu_set_current_device(0) == 0 && LAUNCH_HOME(),
              "mirror timed home launch");
        CHECK(ds4_gpu_set_current_device(1) == 0 &&
              LAUNCH_PARTNER(&q1, &raw1, &comp1, &topk1),
              "mirror timed partner launch");
    }
    CHECK(sync_tier(0) && sync_tier(1), "mirror timing synchronization");
    const double mirror_ms =
        (now_seconds() - start) * 1000.0 / (double)repeats;

    printf("scenario=attention-row-split-%s-32k\n",
           indexed ? "indexed" : "mixed");
    printf("validation=bit-exact-nonzero\nrepeats=%u\n", repeats);
    printf("baseline_ms=%.6f\npeer_read_ms=%.6f\nmirrored_kv_ms=%.6f\n",
           baseline_ms, peer_ms, mirror_ms);
    printf("peer_read_speedup=%.6f\nmirrored_kv_speedup=%.6f\n",
           baseline_ms / peer_ms, baseline_ms / mirror_ms);
    printf("partner_q_bytes=%llu\npartner_attention_output_bytes=%llu\n",
           (unsigned long long)(half_count * sizeof(float)),
           (unsigned long long)(half_count * sizeof(float)));
    printf("persistent_partner_kv_bytes=%llu\n",
           (unsigned long long)((raw_count + comp_count) * sizeof(float)));
    ok = 1;

#undef LAUNCH_PARTNER
#undef LAUNCH_HOME

cleanup:
    ds4_gpu_tensor_free_in_place(&topk1);
    ds4_gpu_tensor_free_in_place(&comp1);
    ds4_gpu_tensor_free_in_place(&raw1);
    ds4_gpu_tensor_free_in_place(&q1);
    ds4_gpu_tensor_free_in_place(&partner);
    ds4_gpu_tensor_free_in_place(&home);
    ds4_gpu_tensor_free_in_place(&ref);
    ds4_gpu_tensor_free_in_place(&topk0);
    ds4_gpu_tensor_free_in_place(&comp0);
    ds4_gpu_tensor_free_in_place(&raw0);
    ds4_gpu_tensor_free_in_place(&q0);
    if (initialized) ds4_gpu_cleanup();
    free(topk_host);
    free(partner_host);
    free(home_host);
    free(ref_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    free(model);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2 ||
        (strcmp(argv[1], "mixed") && strcmp(argv[1], "indexed"))) {
        fprintf(stderr, "Usage: %s mixed|indexed\n", argv[0]);
        return 2;
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        fprintf(stderr, "error: two visible CUDA devices are required\n");
        return 2;
    }
    return run_case(strcmp(argv[1], "indexed") == 0) ? 0 : 1;
}
