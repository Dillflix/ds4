#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GIB (1024ull * 1024ull * 1024ull)
#define PROFILE_FREE_VRAM_MIN (4ull * GIB)
#define PROFILE_TYPE_SM75_Q4_32 42u
#define PROFILE_TYPE_SM75_Q3A4 43u

typedef enum {
    SCENARIO_Q4_32_GATE_UP,
    SCENARIO_Q3A4_GATE_UP,
    SCENARIO_Q8_SINGLE_T32,
    SCENARIO_Q8_PAIR_2048,
    SCENARIO_Q8_PAIR_1024,
    SCENARIO_Q8_KSLICE_T256,
    SCENARIO_Q8_GROUPED_A_HALF,
    SCENARIO_Q8_SHARED_MID,
    SCENARIO_F16_PAIR_256,
    SCENARIO_F16_PAIR_512,
    SCENARIO_F16_PAIR_1024,
} scenario_kind;

typedef struct {
    const char *name;
    const char *family;
    scenario_kind kind;
} scenario_spec;

static const scenario_spec scenarios[] = {
    { "q4-32-gate-up", "routed-q4-32", SCENARIO_Q4_32_GATE_UP },
    { "q3a4-gate-up", "routed-q3a4", SCENARIO_Q3A4_GATE_UP },
    { "q8-single-t32", "dense-q8-single", SCENARIO_Q8_SINGLE_T32 },
    { "q8-pair-2048", "dense-q8-pair", SCENARIO_Q8_PAIR_2048 },
    { "q8-pair-1024", "dense-q8-pair", SCENARIO_Q8_PAIR_1024 },
    { "q8-kslice-t256", "dense-q8-kslice", SCENARIO_Q8_KSLICE_T256 },
    { "q8-grouped-a-half", "dense-q8-grouped-a", SCENARIO_Q8_GROUPED_A_HALF },
    { "q8-shared-mid", "dense-q8-shared", SCENARIO_Q8_SHARED_MID },
    { "f16-pair-256", "dense-f16-pair", SCENARIO_F16_PAIR_256 },
    { "f16-pair-512", "dense-f16-pair", SCENARIO_F16_PAIR_512 },
    { "f16-pair-1024", "dense-f16-pair", SCENARIO_F16_PAIR_1024 },
};

/* CUDA may host-register the map if a forced device copy cannot be made.  Keep
 * the mapping alive until after backend cleanup in either case. */
static unsigned char *model_storage;

static void usage(const char *argv0) {
    fprintf(stderr, "Usage: %s SCENARIO\n\nScenarios:\n", argv0);
    for (size_t i = 0; i < sizeof(scenarios) / sizeof(scenarios[0]); i++)
        fprintf(stderr, "  %s\n", scenarios[i].name);
    fprintf(stderr,
            "\nProfiles exactly one production-shape one-token decode call.\n"
            "The model payload is synthetic zero data; outputs are checked as\n"
            "exact zero after the shipping kernel executes.\n");
}

static const scenario_spec *find_scenario(const char *name) {
    for (size_t i = 0; i < sizeof(scenarios) / sizeof(scenarios[0]); i++) {
        if (strcmp(name, scenarios[i].name) == 0) return &scenarios[i];
    }
    return NULL;
}

static int checked_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0u && b > UINT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int checked_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (b > UINT64_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static int verify_zero_tensor(const ds4_gpu_tensor *tensor, uint64_t count,
                              const char *label) {
    float *host = (float *)malloc((size_t)count * sizeof(float));
    if (!host) {
        fprintf(stderr, "error: %s host validation allocation failed\n", label);
        return 0;
    }
    if (!ds4_gpu_tensor_read(tensor, 0u, host, count * sizeof(float))) {
        fprintf(stderr, "error: %s output readback failed\n", label);
        free(host);
        return 0;
    }
    for (uint64_t i = 0; i < count; i++) {
        if (host[i] != 0.0f || !isfinite(host[i])) {
            fprintf(stderr, "error: %s output[%llu]=%.9g, expected zero\n",
                    label, (unsigned long long)i, host[i]);
            free(host);
            return 0;
        }
    }
    free(host);
    printf("output_values_checked=%llu\noutput_validation=exact-zero\n",
           (unsigned long long)count);
    return 1;
}

static int install_zero_model(uint64_t model_bytes) {
    if (model_bytes == 0u || model_bytes > 3ull * GIB ||
        model_bytes > (uint64_t)SIZE_MAX) {
        fprintf(stderr, "error: invalid model allocation: %llu bytes\n",
                (unsigned long long)model_bytes);
        return 0;
    }
    model_storage = (unsigned char *)calloc(1u, (size_t)model_bytes);
    if (!model_storage) {
        fprintf(stderr, "error: host model allocation failed for %.3f GiB\n",
                (double)model_bytes / (double)GIB);
        return 0;
    }
    if (!ds4_gpu_set_model_map(model_storage, model_bytes) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: device model installation failed\n");
        return 0;
    }
    printf("model_bytes=%llu\n", (unsigned long long)model_bytes);
    return 1;
}

static ds4_gpu_tensor *zero_tensor(uint64_t count, uint64_t elem_bytes) {
    uint64_t bytes;
    if (!checked_mul(count, elem_bytes, &bytes) || bytes == 0u) return NULL;
    ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(bytes);
    if (!tensor) return NULL;
    if (elem_bytes == sizeof(float) &&
        !ds4_gpu_tensor_fill_f32(tensor, 0.0f, count)) {
        ds4_gpu_tensor_free(tensor);
        return NULL;
    }
    return tensor;
}

static ds4_gpu_tensor *input_tensor(uint64_t count) {
    float *host = (float *)malloc((size_t)count * sizeof(float));
    if (!host) return NULL;
    for (uint64_t i = 0; i < count; i++) {
        const int value = (int)((i * 29u + (i >> 3u) * 11u) % 193u) - 96;
        host[i] = (float)value / 101.0f;
    }
    ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(count * sizeof(float));
    if (!tensor || !ds4_gpu_tensor_write(
            tensor, 0u, host, count * sizeof(float))) {
        ds4_gpu_tensor_free(tensor);
        tensor = NULL;
    }
    free(host);
    return tensor;
}

static int run_routed_gate_up(int q3a4) {
    /* Three active home experts plus three partner-owned slots reproduce the
     * per-device production ownership shape.  Only the three addressable home
     * payloads need to be materialized; resident_expert_count is merely the
     * ownership bound used by this kernel and does not alter its launch grid. */
    const uint32_t resident_experts = 3u;
    const uint32_t n_total_experts = 256u;
    const uint32_t n_expert = 6u;
    const uint32_t in_dim = 4096u;
    const uint32_t mid_dim = 2048u;
    const uint32_t out_dim = 4096u;
    const uint64_t gate_block_bytes = q3a4 ? 108u : 136u;
    const uint64_t gate_row_bytes = (in_dim / 256u) * gate_block_bytes;
    const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = (mid_dim / 256u) * 136u;
    const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;
    const uint64_t gate_offset = 0u;
    const uint64_t up_offset = gate_expert_bytes * resident_experts;
    const uint64_t down_offset = up_offset + gate_expert_bytes * resident_experts;
    uint64_t model_bytes;
    if (!checked_add(down_offset,
                     down_expert_bytes * resident_experts, &model_bytes) ||
        !install_zero_model(model_bytes)) return 0;

    const int32_t selected_host[6] = { 0, 1, 2, 128, 129, 130 };
    const float weights_host[6] = {
        1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
        1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
    };
    printf("selected_home_experts=3\nselected_partner_experts=3\n"
           "materialized_home_experts=%u\n", resident_experts);
    ds4_gpu_tensor *x = input_tensor(in_dim);
    ds4_gpu_tensor *selected = zero_tensor(n_expert, sizeof(int32_t));
    ds4_gpu_tensor *weights = zero_tensor(n_expert, sizeof(float));
    ds4_gpu_tensor *out = zero_tensor(out_dim, sizeof(float));
    ds4_gpu_tensor *gate = zero_tensor((uint64_t)n_expert * mid_dim,
                                       sizeof(float));
    ds4_gpu_tensor *up = zero_tensor((uint64_t)n_expert * mid_dim,
                                     sizeof(float));
    ds4_gpu_tensor *mid = zero_tensor((uint64_t)n_expert * mid_dim,
                                      sizeof(float));
    ds4_gpu_tensor *down = zero_tensor((uint64_t)n_expert * out_dim,
                                       sizeof(float));
    int ok = 0;
    if (!x || !selected || !weights || !out || !gate || !up || !mid || !down ||
        !ds4_gpu_tensor_write(selected, 0u, selected_host,
                              sizeof(selected_host)) ||
        !ds4_gpu_tensor_write(weights, 0u, weights_host,
                              sizeof(weights_host))) {
        fprintf(stderr, "error: routed decode tensor setup failed\n");
        goto cleanup;
    }
    ds4_gpu_set_routed_q4_layout(DS4_TENSOR_LAYOUT_SM75_Q4_32 |
                                 DS4_TENSOR_LAYOUT_SM75_Q3A4);
    if (!ds4_gpu_routed_moe_one_owned_tensor(
            out, gate, up, mid, down, model_storage, model_bytes,
            gate_offset, up_offset, down_offset,
            q3a4 ? PROFILE_TYPE_SM75_Q3A4 : PROFILE_TYPE_SM75_Q4_32,
            PROFILE_TYPE_SM75_Q4_32,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, selected, weights,
            n_total_experts, n_expert, 0u, resident_experts, 10.0f, x,
            NULL, false, NULL) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: routed decode production launch failed\n");
        goto cleanup;
    }
    ok = verify_zero_tensor(mid, (uint64_t)n_expert * mid_dim,
                            q3a4 ? "q3a4-gate-up" : "q4-32-gate-up");

cleanup:
    ds4_gpu_set_routed_q4_layout(0u);
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(x);
    return ok;
}

static uint64_t q8_matrix_bytes(uint64_t in_dim, uint64_t out_dim) {
    return out_dim * ((in_dim + 31u) / 32u) * 34u;
}

static int run_q8_single(uint64_t in_dim, uint64_t out_dim) {
    const uint64_t model_bytes = q8_matrix_bytes(in_dim, out_dim);
    if (!install_zero_model(model_bytes)) return 0;
    ds4_gpu_tensor *x = input_tensor(in_dim);
    ds4_gpu_tensor *out = zero_tensor(out_dim, sizeof(float));
    int ok = 0;
    if (!x || !out || !ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
            out, model_storage, model_bytes, 0u, in_dim, out_dim, x, 1u) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: single dense-Q8 decode launch failed\n");
        goto cleanup;
    }
    ok = verify_zero_tensor(out, out_dim, "q8-single-t32");
cleanup:
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    return ok;
}

static int run_q8_pair(uint64_t out_dim) {
    const uint64_t in_dim = 4096u;
    const uint64_t one_bytes = q8_matrix_bytes(in_dim, out_dim);
    const uint64_t model_bytes = 2u * one_bytes;
    if (!install_zero_model(model_bytes)) return 0;
    ds4_gpu_tensor *x = input_tensor(in_dim);
    ds4_gpu_tensor *out0 = zero_tensor(out_dim, sizeof(float));
    ds4_gpu_tensor *out1 = zero_tensor(out_dim, sizeof(float));
    int ok = 0;
    if (!x || !out0 || !out1 ||
        !ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
            out0, out1, model_storage, model_bytes, 0u, one_bytes,
            in_dim, out_dim, out_dim, x, 1u) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: paired dense-Q8 decode launch failed\n");
        goto cleanup;
    }
    ok = verify_zero_tensor(out0, out_dim, "q8-pair-out0") &&
         verify_zero_tensor(out1, out_dim, "q8-pair-out1");
cleanup:
    ds4_gpu_tensor_free(out1);
    ds4_gpu_tensor_free(out0);
    ds4_gpu_tensor_free(x);
    return ok;
}

static int run_q8_kslice(void) {
    const uint64_t full_in_dim = 8192u;
    const uint64_t slice_dim = 4096u;
    const uint64_t out_dim = 4096u;
    const uint64_t model_bytes = q8_matrix_bytes(full_in_dim, out_dim);
    if (!install_zero_model(model_bytes)) return 0;
    ds4_gpu_tensor *x = input_tensor(slice_dim);
    ds4_gpu_tensor *out = zero_tensor(out_dim, sizeof(float));
    int ok = 0;
    if (!x || !out || !ds4_gpu_matmul_q8_0_kslice_rows_tensor(
            out, model_storage, model_bytes, 0u, full_in_dim, out_dim,
            0u, slice_dim, x, 1u) || !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: K-slice dense-Q8 decode launch failed\n");
        goto cleanup;
    }
    ok = verify_zero_tensor(out, out_dim, "q8-kslice-t256");
cleanup:
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    return ok;
}

static int run_q8_grouped_a(void) {
    const uint32_t groups_total = 64u;
    const uint32_t group_count = 32u;
    const uint64_t group_dim = 128u;
    const uint64_t rank = 128u;
    const uint64_t low_dim = (uint64_t)group_count * rank;
    const uint64_t model_bytes =
        q8_matrix_bytes(group_dim, (uint64_t)groups_total * rank);
    if (!install_zero_model(model_bytes)) return 0;
    ds4_gpu_tensor *heads = input_tensor((uint64_t)groups_total * group_dim);
    ds4_gpu_tensor *low = zero_tensor(low_dim, sizeof(float));
    int ok = 0;
    if (!heads || !low || !ds4_gpu_attention_output_q8_batch_low_shard_tensor(
            low, model_storage, model_bytes, 0u, group_dim, rank,
            groups_total, 0u, group_count, heads, 1u) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: grouped-A dense-Q8 decode launch failed\n");
        goto cleanup;
    }
    ok = verify_zero_tensor(low, low_dim, "q8-grouped-a-half");
cleanup:
    ds4_gpu_tensor_free(low);
    ds4_gpu_tensor_free(heads);
    return ok;
}

static int run_q8_shared_mid(void) {
    const uint64_t in_dim = 4096u;
    const uint64_t out_dim = 2048u;
    const uint64_t one_bytes = q8_matrix_bytes(in_dim, out_dim);
    const uint64_t model_bytes = 2u * one_bytes;
    const int32_t selected_host[6] = { 0, 1, 2, 3, 4, 5 };
    if (!install_zero_model(model_bytes)) return 0;
    ds4_gpu_tensor *x = input_tensor(in_dim);
    ds4_gpu_tensor *mid = zero_tensor(out_dim, sizeof(float));
    ds4_gpu_tensor *selected = zero_tensor(6u, sizeof(int32_t));
    int ok = 0;
    if (!x || !mid || !selected ||
        !ds4_gpu_tensor_write(selected, 0u, selected_host,
                              sizeof(selected_host)) ||
        !ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(
            mid, model_storage, model_bytes, 0u, one_bytes,
            in_dim, out_dim, x, 10.0f, selected, NULL, 128u, true) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: shared-mid dense-Q8 decode launch failed\n");
        goto cleanup;
    }
    ok = verify_zero_tensor(mid, out_dim, "q8-shared-mid");
cleanup:
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(x);
    return ok;
}

static int run_f16_pair(uint64_t out_dim) {
    const uint64_t in_dim = 4096u;
    uint64_t one_bytes, model_bytes;
    if (!checked_mul(in_dim * out_dim, sizeof(uint16_t), &one_bytes) ||
        !checked_mul(one_bytes, 2u, &model_bytes) ||
        !install_zero_model(model_bytes)) return 0;
    ds4_gpu_tensor *x = input_tensor(in_dim);
    ds4_gpu_tensor *out0 = zero_tensor(out_dim, sizeof(float));
    ds4_gpu_tensor *out1 = zero_tensor(out_dim, sizeof(float));
    int ok = 0;
    if (!x || !out0 || !out1 || !ds4_gpu_matmul_f16_pair_tensor(
            out0, out1, model_storage, model_bytes, 0u, one_bytes,
            in_dim, out_dim, x, 1u) || !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: F16 compressor-pair decode launch failed\n");
        goto cleanup;
    }
    ok = verify_zero_tensor(out0, out_dim, "f16-pair-out0") &&
         verify_zero_tensor(out1, out_dim, "f16-pair-out1");
cleanup:
    ds4_gpu_tensor_free(out1);
    ds4_gpu_tensor_free(out0);
    ds4_gpu_tensor_free(x);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2 || strcmp(argv[1], "-h") == 0 ||
        strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return argc == 2 ? 0 : 2;
    }
    const scenario_spec *spec = find_scenario(argv[1]);
    if (!spec) {
        fprintf(stderr, "error: unknown scenario: %s\n", argv[1]);
        usage(argv[0]);
        return 2;
    }

    /* Normalize every presence-based switch that can silently change these
     * decode kernels.  DP4A remains enabled because that is the current
     * production default for the packed-Q8 one-token kernels. */
    (void)setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    (void)setenv("DS4_CUDA_NO_Q8_F16_CACHE", "1", 1);
    (void)unsetenv("DS4_CUDA_NO_Q8_DP4A");
    (void)unsetenv("DS4_CUDA_NO_F16_PAIR_MATMUL");
    (void)unsetenv("DS4_CUDA_SERIAL_F16_MATMUL");
    (void)unsetenv("DS4_CUDA_SERIAL_ROUTER");
    (void)unsetenv("DS4_CUDA_NO_ORDERED_F16_MATMUL");
    (void)unsetenv("DS4_CUDA_MOE_Q32_DECODE_GRAPH");
    (void)setenv("DS4_CUDA_NO_MOE_Q32_DECODE_GRAPH", "1", 1);

    printf("scenario=%s\nfamily=%s\nn_tokens=1\n"
           "q8_arithmetic=production-dp4a\nq8_f16_cache=disabled\n"
           "q32_decode_graph=disabled\n",
           spec->name, spec->family);
    if (!ds4_gpu_init()) {
        fprintf(stderr, "error: CUDA backend initialization failed\n");
        return 1;
    }
    const uint64_t free_vram = ds4_gpu_tier_free_vram(0);
    printf("free_vram_at_start_bytes=%llu\n",
           (unsigned long long)free_vram);
    if (free_vram < PROFILE_FREE_VRAM_MIN) {
        fprintf(stderr, "error: at least 4 GiB free VRAM is required\n");
        ds4_gpu_cleanup();
        return 1;
    }

    int ok = 0;
    switch (spec->kind) {
        case SCENARIO_Q4_32_GATE_UP: ok = run_routed_gate_up(0); break;
        case SCENARIO_Q3A4_GATE_UP: ok = run_routed_gate_up(1); break;
        case SCENARIO_Q8_SINGLE_T32: ok = run_q8_single(1024u, 32768u); break;
        case SCENARIO_Q8_PAIR_2048: ok = run_q8_pair(2048u); break;
        case SCENARIO_Q8_PAIR_1024: ok = run_q8_pair(1024u); break;
        case SCENARIO_Q8_KSLICE_T256: ok = run_q8_kslice(); break;
        case SCENARIO_Q8_GROUPED_A_HALF: ok = run_q8_grouped_a(); break;
        case SCENARIO_Q8_SHARED_MID: ok = run_q8_shared_mid(); break;
        case SCENARIO_F16_PAIR_256: ok = run_f16_pair(256u); break;
        case SCENARIO_F16_PAIR_512: ok = run_f16_pair(512u); break;
        case SCENARIO_F16_PAIR_1024: ok = run_f16_pair(1024u); break;
    }
    ds4_gpu_cleanup();
    free(model_storage);
    model_storage = NULL;
    if (ok) printf("harness_status=ok\n");
    return ok ? 0 : 1;
}
