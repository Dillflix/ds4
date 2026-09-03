#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GUARD_FLOATS 32u

extern void ds4_gpu_test_set_compressor_pair_state_store(int enabled);
extern void ds4_gpu_test_set_compressor_pair_state_store_disabled(int disabled);
extern void ds4_gpu_test_set_compressor_pair_state_store_reference(int reference);
extern void ds4_gpu_test_set_compressor_projection_staged_small(int enabled);
extern int ds4_gpu_test_compressor_pair_state_store_resources(
        int *registers, int *static_shared_bytes, int *local_bytes,
        int *max_threads_per_block, int *active_blocks_per_sm);

typedef struct {
    ds4_gpu_tensor *base;
    ds4_gpu_tensor *view;
    float *initial;
    uint64_t payload_count;
} guarded_tensor;

static unsigned char *model_storage;

static uint32_t env_u32(const char *name, uint32_t fallback) {
    const char *text = getenv(name);
    if (!text || !text[0]) return fallback;
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (end == text || *end || value == 0u || value > UINT32_MAX) return fallback;
    return (uint32_t)value;
}

static int compare_float(const void *lhs, const void *rhs) {
    const float a = *(const float *)lhs;
    const float b = *(const float *)rhs;
    return (a > b) - (a < b);
}

static uint16_t patterned_half(uint64_t i, uint32_t salt) {
    static const uint16_t pattern[] = {
        0x3000u, 0xb400u, 0x3800u, 0xb800u,
        0x3a00u, 0xba00u, 0x3400u, 0xb000u,
    };
    return pattern[(i * 5u + salt) & 7u];
}

static float initial_float(uint64_t i, uint32_t salt) {
    const int v = (int)((i * 37u + salt * 19u) % 251u) - 125;
    return (float)v / 64.0f;
}

static guarded_tensor guarded_alloc(uint64_t payload_count, uint32_t salt) {
    guarded_tensor result;
    memset(&result, 0, sizeof(result));
    const uint64_t total = payload_count + 2u * GUARD_FLOATS;
    if (total > SIZE_MAX / sizeof(float)) return result;
    result.initial = (float *)malloc((size_t)total * sizeof(float));
    if (!result.initial) return result;
    for (uint64_t i = 0; i < total; i++) result.initial[i] = initial_float(i, salt);
    result.base = ds4_gpu_tensor_alloc(total * sizeof(float));
    if (!result.base ||
        !ds4_gpu_tensor_write(result.base, 0u, result.initial,
                              total * sizeof(float))) {
        ds4_gpu_tensor_free(result.base);
        free(result.initial);
        memset(&result, 0, sizeof(result));
        return result;
    }
    result.view = ds4_gpu_tensor_view(
            result.base, GUARD_FLOATS * sizeof(float),
            payload_count * sizeof(float));
    if (!result.view) {
        ds4_gpu_tensor_free(result.base);
        free(result.initial);
        memset(&result, 0, sizeof(result));
        return result;
    }
    result.payload_count = payload_count;
    return result;
}

static void guarded_free(guarded_tensor *tensor) {
    if (!tensor) return;
    ds4_gpu_tensor_free(tensor->view);
    ds4_gpu_tensor_free(tensor->base);
    free(tensor->initial);
    memset(tensor, 0, sizeof(*tensor));
}

static int guarded_read(const guarded_tensor *tensor, float **host_out,
                        uint64_t *total_out) {
    const uint64_t total = tensor->payload_count + 2u * GUARD_FLOATS;
    float *host = (float *)malloc((size_t)total * sizeof(float));
    if (!host || !ds4_gpu_tensor_read(
            tensor->base, 0u, host, total * sizeof(float))) {
        free(host);
        return 0;
    }
    *host_out = host;
    *total_out = total;
    return 1;
}

static int guards_unchanged(const guarded_tensor *tensor, const float *host,
                            const char *label) {
    const uint64_t suffix = GUARD_FLOATS + tensor->payload_count;
    if (memcmp(host, tensor->initial, GUARD_FLOATS * sizeof(float)) != 0 ||
        memcmp(host + suffix, tensor->initial + suffix,
               GUARD_FLOATS * sizeof(float)) != 0) {
        fprintf(stderr, "error: %s canary changed\n", label);
        return 0;
    }
    return 1;
}

static int byte_equal_guarded(const guarded_tensor *reference,
                              const guarded_tensor *candidate,
                              const char *label, int require_nonzero) {
    float *ref = NULL;
    float *cand = NULL;
    uint64_t ref_total = 0, cand_total = 0;
    int ok = guarded_read(reference, &ref, &ref_total) &&
             guarded_read(candidate, &cand, &cand_total);
    if (!ok || ref_total != cand_total ||
        memcmp(ref, cand, (size_t)ref_total * sizeof(float)) != 0) {
        fprintf(stderr, "error: %s differs from control\n", label);
        ok = 0;
    }
    if (ok) {
        ok = guards_unchanged(reference, ref, label) &&
             guards_unchanged(candidate, cand, label);
    }
    if (ok && require_nonzero) {
        const float *payload = ref + GUARD_FLOATS;
        int nonzero = 0;
        for (uint64_t i = 0; i < reference->payload_count; i++) {
            nonzero |= payload[i] != 0.0f;
            if (!isfinite(payload[i])) {
                fprintf(stderr, "error: %s contains non-finite value\n", label);
                ok = 0;
                break;
            }
        }
        if (ok && !nonzero) {
            fprintf(stderr, "error: %s unexpectedly contains only zeros\n", label);
            ok = 0;
        }
    }
    free(cand);
    free(ref);
    return ok;
}

static int payload_unchanged(const guarded_tensor *tensor, const char *label) {
    float *host = NULL;
    uint64_t total = 0;
    int ok = guarded_read(tensor, &host, &total);
    if (!ok || total != tensor->payload_count + 2u * GUARD_FLOATS) {
        fprintf(stderr, "error: could not read %s\n", label);
        free(host);
        return 0;
    }
    ok = guards_unchanged(tensor, host, label);
    if (ok && memcmp(host + GUARD_FLOATS,
                     tensor->initial + GUARD_FLOATS,
                     (size_t)tensor->payload_count * sizeof(float)) != 0) {
        fprintf(stderr, "error: non-emitting %s payload changed\n", label);
        ok = 0;
    }
    free(host);
    return ok;
}

static int run_control_stage(
        ds4_gpu_tensor *out_kv, ds4_gpu_tensor *out_score,
        ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score,
        const ds4_gpu_tensor *x, uint64_t model_bytes,
        uint64_t score_offset, uint64_t ape_offset,
        uint32_t head_dim, uint32_t ratio, uint32_t ape_type,
        uint32_t pos) {
    return ds4_gpu_matmul_f16_pair_tensor(
                   out_kv, out_score, model_storage, model_bytes,
                   0u, score_offset, 4096u, ratio == 4u ? 2u * head_dim : head_dim,
                   x, 1u) &&
           ds4_gpu_compressor_store_batch_tensor(
                   out_kv, out_score, state_kv, state_score,
                   model_storage, model_bytes, ape_offset, ape_type,
                   head_dim, ratio, pos, 1u);
}

static int run_candidate_stage(
        ds4_gpu_tensor *out_kv, ds4_gpu_tensor *out_score,
        ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score,
        const ds4_gpu_tensor *x, uint64_t model_bytes,
        uint64_t score_offset, uint64_t ape_offset,
        uint32_t head_dim, uint32_t ratio, uint32_t ape_type,
        uint32_t pos) {
    const uint32_t width = ratio == 4u ? 2u * head_dim : head_dim;
    return ds4_gpu_matmul_f16_pair_compressor_store_tensor(
            out_kv, out_score, state_kv, state_score,
            model_storage, model_bytes, 0u, score_offset, ape_offset, ape_type,
            4096u, width, x, ratio, pos) == 1;
}

static int run_control(
        ds4_gpu_tensor *out_kv, ds4_gpu_tensor *out_score,
        ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score,
        ds4_gpu_tensor *comp, const ds4_gpu_tensor *x,
        uint64_t model_bytes, uint64_t score_offset, uint64_t ape_offset,
        uint64_t norm_offset, uint32_t head_dim, uint32_t ratio,
        uint32_t ape_type,
        uint32_t pos) {
    if (!run_control_stage(out_kv, out_score, state_kv, state_score, x,
                           model_bytes, score_offset, ape_offset,
                           head_dim, ratio, ape_type, pos) ||
        !ds4_gpu_compressor_update_tensor(
                   out_kv, out_score, state_kv, state_score, comp,
                   model_storage, model_bytes, ape_offset, ape_type,
                   norm_offset, 0u, head_dim, ratio, pos, 0u, 64u, 32768u,
                   10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f,
                   1.0e-6f, true)) {
        return 0;
    }
    if ((pos + 1u) % ratio != 0u) return 1;
    return head_dim == 128u
        ? ds4_gpu_dsv4_indexer_qat_tensor(comp, 1u, head_dim)
        : ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp, 1u, head_dim, 64u);
}

static int run_candidate(
        ds4_gpu_tensor *out_kv, ds4_gpu_tensor *out_score,
        ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score,
        ds4_gpu_tensor *comp, const ds4_gpu_tensor *x,
        uint64_t model_bytes, uint64_t score_offset, uint64_t ape_offset,
        uint64_t norm_offset, uint32_t head_dim, uint32_t ratio,
        uint32_t ape_type,
        uint32_t pos) {
    if (!run_candidate_stage(out_kv, out_score, state_kv, state_score, x,
                             model_bytes, score_offset, ape_offset,
                             head_dim, ratio, ape_type, pos) ||
        !ds4_gpu_compressor_update_tensor(
                out_kv, out_score, state_kv, state_score, comp,
                model_storage, model_bytes, ape_offset, ape_type,
                norm_offset, 0u, head_dim, ratio, pos, 0u, 64u, 32768u,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f,
                1.0e-6f, true)) {
        return 0;
    }
    if ((pos + 1u) % ratio != 0u) return 1;
    return head_dim == 128u
        ? ds4_gpu_dsv4_indexer_qat_tensor(comp, 1u, head_dim)
        : ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp, 1u, head_dim, 64u);
}

static int time_variant(
        int candidate, int full_chain,
        ds4_gpu_tensor *out_kv, ds4_gpu_tensor *out_score,
        ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score,
        ds4_gpu_tensor *comp, const ds4_gpu_tensor *x,
        uint64_t model_bytes, uint64_t score_offset, uint64_t ape_offset,
        uint64_t norm_offset, uint32_t head_dim, uint32_t ratio,
        uint32_t ape_type,
        uint32_t pos, uint32_t repeats, float *elapsed_ms) {
    ds4_gpu_timer *timer = ds4_gpu_timer_create();
    if (!timer || !ds4_gpu_timer_record_start(timer)) {
        ds4_gpu_timer_free(timer);
        return 0;
    }
    for (uint32_t i = 0; i < repeats; i++) {
        const int ok = full_chain
            ? (candidate
                ? run_candidate(out_kv, out_score, state_kv, state_score, comp, x,
                                model_bytes, score_offset, ape_offset, norm_offset,
                                head_dim, ratio, ape_type, pos)
                : run_control(out_kv, out_score, state_kv, state_score, comp, x,
                              model_bytes, score_offset, ape_offset, norm_offset,
                              head_dim, ratio, ape_type, pos))
            : (candidate
                ? run_candidate_stage(out_kv, out_score, state_kv, state_score, x,
                                      model_bytes, score_offset, ape_offset,
                                      head_dim, ratio, ape_type, pos)
                : run_control_stage(out_kv, out_score, state_kv, state_score, x,
                                    model_bytes, score_offset, ape_offset,
                                    head_dim, ratio, ape_type, pos));
        if (!ok) {
            ds4_gpu_timer_free(timer);
            return 0;
        }
    }
    int ok = ds4_gpu_timer_record_end(timer) &&
             ds4_gpu_timer_elapsed_ms(timer, elapsed_ms);
    ds4_gpu_timer_free(timer);
    if (ok) *elapsed_ms /= (float)repeats;
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        fprintf(stderr,
                "usage: %s 256|512|1024 nonemit|emit [f16|f32] [pos]\n",
                argv[0]);
        return 2;
    }
    const uint32_t width = (uint32_t)strtoul(argv[1], NULL, 10);
    const uint32_t ratio = width == 512u ? 128u : 4u;
    const uint32_t head_dim = ratio == 4u ? width / 2u : width;
    if (width != 256u && width != 512u && width != 1024u) {
        fprintf(stderr, "error: width must be 256, 512, or 1024\n");
        return 2;
    }
    const int emit = strcmp(argv[2], "emit") == 0;
    if (!emit && strcmp(argv[2], "nonemit") != 0) {
        fprintf(stderr, "error: phase must be nonemit or emit\n");
        return 2;
    }
    const uint32_t ape_type = argc >= 4 && strcmp(argv[3], "f32") == 0 ? 0u : 1u;
    if (argc >= 4 && strcmp(argv[3], "f16") != 0 &&
        strcmp(argv[3], "f32") != 0) {
        fprintf(stderr, "error: APE type must be f16 or f32\n");
        return 2;
    }
    const uint32_t pos = argc >= 5 ? (uint32_t)strtoul(argv[4], NULL, 10)
                                   : (emit ? ratio - 1u : 0u);
    if ((((pos + 1u) % ratio) == 0u) != emit) {
        fprintf(stderr, "error: phase and pos disagree\n");
        return 2;
    }
    const uint32_t state_rows = ratio == 4u ? 2u * ratio : ratio;
    const uint64_t weight_elems = (uint64_t)4096u * width;
    const uint64_t weight_bytes = weight_elems * sizeof(uint16_t);
    const uint64_t score_offset = weight_bytes;
    const uint64_t ape_offset = 2u * weight_bytes;
    const uint64_t ape_elems = (uint64_t)ratio * width;
    const uint64_t ape_elem_bytes = ape_type == 1u ? sizeof(uint16_t) : sizeof(float);
    const uint64_t norm_offset = ape_offset + ape_elems * ape_elem_bytes;
    const uint64_t model_bytes = norm_offset + (uint64_t)head_dim * sizeof(float);
    if (model_bytes > SIZE_MAX) return 1;

    model_storage = (unsigned char *)malloc((size_t)model_bytes);
    float *x_host = (float *)malloc(4096u * sizeof(float));
    if (!model_storage || !x_host) {
        fprintf(stderr, "error: host allocation failed\n");
        return 1;
    }
    uint16_t *wkv = (uint16_t *)(model_storage + 0u);
    uint16_t *wscore = (uint16_t *)(model_storage + score_offset);
    float *norm = (float *)(model_storage + norm_offset);
    for (uint64_t i = 0; i < weight_elems; i++) {
        wkv[i] = patterned_half(i, 1u);
        wscore[i] = patterned_half(i, 3u);
    }
    if (ape_type == 1u) {
        uint16_t *ape = (uint16_t *)(model_storage + ape_offset);
        for (uint64_t i = 0; i < ape_elems; i++) ape[i] = patterned_half(i, 5u);
    } else {
        float *ape = (float *)(model_storage + ape_offset);
        for (uint64_t i = 0; i < ape_elems; i++) ape[i] = initial_float(i, 5u);
    }
    for (uint32_t i = 0; i < head_dim; i++) norm[i] = 0.75f + (float)(i & 7u) / 32.0f;
    for (uint32_t i = 0; i < 4096u; i++) x_host[i] = initial_float(i, 7u) / 8.0f;

    (void)setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    (void)setenv("DS4_CUDA_ENABLE_COMPRESSOR_PAIR_STATE_STORE", "1", 1);
    (void)unsetenv("DS4_CUDA_DISABLE_COMPRESSOR_PAIR_STATE_STORE");
    (void)unsetenv("DS4_CUDA_NO_F16_PAIR_MATMUL");
    (void)unsetenv("DS4_CUDA_SERIAL_F16_MATMUL");
    (void)unsetenv("DS4_CUDA_SERIAL_ROUTER");
    (void)unsetenv("DS4_CUDA_NO_ORDERED_F16_MATMUL");
    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(model_storage, model_bytes)) {
        fprintf(stderr, "error: CUDA initialization/model copy failed\n");
        return 1;
    }
    int registers = 0, static_shared = 0, local_bytes = 0;
    int max_threads = 0, active_blocks = 0;
    if (!ds4_gpu_test_compressor_pair_state_store_resources(
            &registers, &static_shared, &local_bytes,
            &max_threads, &active_blocks) ||
        registers > 128 || local_bytes != 0 || max_threads < 32 ||
        active_blocks < 2) {
        fprintf(stderr,
                "error: runtime resource gate failed: regs=%d shared=%d "
                "local=%d max_threads=%d active_blocks_sm=%d\n",
                registers, static_shared, local_bytes, max_threads, active_blocks);
        return 1;
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(4096u * sizeof(float));
    guarded_tensor ref_out_kv = guarded_alloc(width, 11u);
    guarded_tensor ref_out_score = guarded_alloc(width, 13u);
    guarded_tensor ref_state_kv = guarded_alloc((uint64_t)state_rows * width, 17u);
    guarded_tensor ref_state_score = guarded_alloc((uint64_t)state_rows * width, 19u);
    guarded_tensor ref_comp = guarded_alloc(head_dim, 23u);
    guarded_tensor cand_out_kv = guarded_alloc(width, 11u);
    guarded_tensor cand_out_score = guarded_alloc(width, 13u);
    guarded_tensor cand_state_kv = guarded_alloc((uint64_t)state_rows * width, 17u);
    guarded_tensor cand_state_score = guarded_alloc((uint64_t)state_rows * width, 19u);
    guarded_tensor cand_comp = guarded_alloc(head_dim, 23u);
    int ok = x && ref_out_kv.view && ref_out_score.view && ref_state_kv.view &&
             ref_state_score.view && ref_comp.view && cand_out_kv.view &&
             cand_out_score.view && cand_state_kv.view &&
             cand_state_score.view && cand_comp.view &&
             ds4_gpu_tensor_write(x, 0u, x_host, 4096u * sizeof(float));
    if (!ok) {
        fprintf(stderr, "error: device tensor setup failed\n");
        goto cleanup;
    }

    /* Production defaults off, and an explicit disable must override the
     * diagnostic enable without touching any destination. */
    ds4_gpu_test_set_compressor_pair_state_store(0);
    int selector_result = ds4_gpu_matmul_f16_pair_compressor_store_tensor(
            cand_out_kv.view, cand_out_score.view,
            cand_state_kv.view, cand_state_score.view,
            model_storage, model_bytes, 0u, score_offset, ape_offset, ape_type,
            4096u, width, x, ratio, pos);
    ok = selector_result == 0 &&
         payload_unchanged(&cand_out_kv, "selector-off-output-kv") &&
         payload_unchanged(&cand_out_score, "selector-off-output-score") &&
         payload_unchanged(&cand_state_kv, "selector-off-state-kv") &&
         payload_unchanged(&cand_state_score, "selector-off-state-score");
    ds4_gpu_test_set_compressor_pair_state_store(1);
    ds4_gpu_test_set_compressor_pair_state_store_disabled(1);
    selector_result = ds4_gpu_matmul_f16_pair_compressor_store_tensor(
            cand_out_kv.view, cand_out_score.view,
            cand_state_kv.view, cand_state_score.view,
            model_storage, model_bytes, 0u, score_offset, ape_offset, ape_type,
            4096u, width, x, ratio, pos);
    ok = ok && selector_result == 0 &&
         payload_unchanged(&cand_out_kv, "disable-output-kv") &&
         payload_unchanged(&cand_out_score, "disable-output-score") &&
         payload_unchanged(&cand_state_kv, "disable-state-kv") &&
         payload_unchanged(&cand_state_score, "disable-state-score");
    ds4_gpu_test_set_compressor_pair_state_store_disabled(0);
    ds4_gpu_test_set_compressor_pair_state_store_reference(1);
    selector_result = ds4_gpu_matmul_f16_pair_compressor_store_tensor(
            cand_out_kv.view, cand_out_score.view,
            cand_state_kv.view, cand_state_score.view,
            model_storage, model_bytes, 0u, score_offset, ape_offset, ape_type,
            4096u, width, x, ratio, pos);
    ok = ok && selector_result == 0 &&
         payload_unchanged(&cand_out_kv, "reference-output-kv") &&
         payload_unchanged(&cand_out_score, "reference-output-score") &&
         payload_unchanged(&cand_state_kv, "reference-state-kv") &&
         payload_unchanged(&cand_state_score, "reference-state-score");
    ds4_gpu_test_set_compressor_pair_state_store_reference(0);
    /* Exercise the candidate canonical-staged implementation for both smaller
     * production shapes.  Width 1024 is already the default and is unaffected
     * by this diagnostic-only setter. */
    ds4_gpu_test_set_compressor_projection_staged_small(1);
    if (!ok) goto cleanup;

    ok = run_control(ref_out_kv.view, ref_out_score.view,
                     ref_state_kv.view, ref_state_score.view, ref_comp.view, x,
                     model_bytes, score_offset, ape_offset, norm_offset,
                     head_dim, ratio, ape_type, pos) &&
         run_candidate(cand_out_kv.view, cand_out_score.view,
                       cand_state_kv.view, cand_state_score.view, cand_comp.view, x,
                       model_bytes, score_offset, ape_offset, norm_offset,
                       head_dim, ratio, ape_type, pos) &&
         ds4_gpu_synchronize();
    if (!ok) {
        fprintf(stderr, "error: exact A/B launch failed\n");
        goto cleanup;
    }
    ok = byte_equal_guarded(&ref_out_kv, &cand_out_kv, "projected-kv", 1) &&
         byte_equal_guarded(&ref_out_score, &cand_out_score, "projected-score", 1) &&
         byte_equal_guarded(&ref_state_kv, &cand_state_kv, "state-kv", 1) &&
         byte_equal_guarded(&ref_state_score, &cand_state_score, "state-score", 1) &&
         byte_equal_guarded(&ref_comp, &cand_comp, "compressed-cache-row", 1);
    if (ok && !emit) {
        ok = payload_unchanged(&ref_comp, "control-compressed-cache-row") &&
             payload_unchanged(&cand_comp, "candidate-compressed-cache-row");
    }
    if (!ok) goto cleanup;

    const uint32_t rounds = env_u32("TIMING_ROUNDS", 9u) | 1u;
    const uint32_t repeats = env_u32("TIMING_REPEATS", 25u);
    float *control_ms = (float *)malloc(rounds * sizeof(float));
    float *candidate_ms = (float *)malloc(rounds * sizeof(float));
    if (!control_ms || !candidate_ms) {
        free(candidate_ms);
        free(control_ms);
        ok = 0;
        goto cleanup;
    }
    float stage_control = 0.0f, stage_candidate = 0.0f;
    float chain_control = 0.0f, chain_candidate = 0.0f;
    for (uint32_t scope = 0; scope < 2u && ok; scope++) {
        const int full_chain = scope == 1u;
        for (uint32_t round = 0; round < rounds && ok; round++) {
            const int candidate_first = round & 1u;
            for (uint32_t slot = 0; slot < 2u && ok; slot++) {
                const int candidate = slot == 0u
                    ? candidate_first : !candidate_first;
                float elapsed = 0.0f;
                ok = time_variant(
                        candidate, full_chain,
                        candidate ? cand_out_kv.view : ref_out_kv.view,
                        candidate ? cand_out_score.view : ref_out_score.view,
                        candidate ? cand_state_kv.view : ref_state_kv.view,
                        candidate ? cand_state_score.view : ref_state_score.view,
                        candidate ? cand_comp.view : ref_comp.view,
                        x, model_bytes, score_offset, ape_offset, norm_offset,
                        head_dim, ratio, ape_type, pos, repeats, &elapsed);
                if (candidate) candidate_ms[round] = elapsed;
                else control_ms[round] = elapsed;
            }
        }
        if (!ok) break;
        qsort(control_ms, rounds, sizeof(float), compare_float);
        qsort(candidate_ms, rounds, sizeof(float), compare_float);
        if (full_chain) {
            chain_control = control_ms[rounds / 2u];
            chain_candidate = candidate_ms[rounds / 2u];
        } else {
            stage_control = control_ms[rounds / 2u];
            stage_candidate = candidate_ms[rounds / 2u];
        }
    }
    if (ok) {
        printf("scenario=compressor-projection-state-store\n"
               "validation=byte-exact-nonzero\n"
               "canaries=passed\n"
               "phase=%s\npos=%u\nemit=%s\n"
               "width=%u\nhead_dim=%u\nratio=%u\nape_type=%s\n"
               "runtime_registers=%d\nruntime_static_shared_bytes=%d\n"
               "runtime_local_bytes=%d\nruntime_active_blocks_per_sm=%d\n"
               "timing_rounds=%u\ntiming_repeats=%u\n"
               "stage_scope=paired-projection-plus-state-store\n"
               "stage_control_median_ms=%.9g\n"
               "stage_fused_median_ms=%.9g\n"
               "stage_fused_speedup=%.9g\n"
               "chain_scope=projection-state-update-rope-and-cache-qat-if-emitting\n"
               "chain_control_median_ms=%.9g\n"
               "chain_fused_median_ms=%.9g\n"
               "chain_fused_speedup=%.9g\n",
               emit ? "emit" : "nonemit", pos, emit ? "yes" : "no",
               width, head_dim, ratio, ape_type == 1u ? "f16" : "f32",
               registers, static_shared, local_bytes, active_blocks,
               rounds, repeats,
               stage_control, stage_candidate, stage_control / stage_candidate,
               chain_control, chain_candidate, chain_control / chain_candidate);
    }
    free(candidate_ms);
    free(control_ms);

cleanup:
    guarded_free(&cand_comp);
    guarded_free(&cand_state_score);
    guarded_free(&cand_state_kv);
    guarded_free(&cand_out_score);
    guarded_free(&cand_out_kv);
    guarded_free(&ref_comp);
    guarded_free(&ref_state_score);
    guarded_free(&ref_state_kv);
    guarded_free(&ref_out_score);
    guarded_free(&ref_out_kv);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    free(x_host);
    free(model_storage);
    return ok ? 0 : 1;
}
