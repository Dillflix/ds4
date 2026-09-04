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

extern void ds4_gpu_test_set_moe_q32_decode_split(int enabled);
extern void ds4_gpu_test_set_moe_q32_decode_fused_lowreg(uint32_t unroll);
extern void ds4_gpu_test_set_moe_q4_32_decode_mapping(uint32_t mapping);
extern void ds4_gpu_test_set_moe_q4_32_decode_prefetch_depth(uint32_t depth);
extern void ds4_gpu_test_set_moe_q3a4_decode_mapping(uint32_t mapping);
extern void ds4_gpu_test_set_moe_q3a4_decode_ksplit(uint32_t split);
extern void ds4_gpu_test_set_moe_q3a4_decode_prefetch_depth(uint32_t depth);
extern void ds4_gpu_test_set_moe_q4_32_down_decode_mapping(uint32_t mapping);
extern void ds4_gpu_test_set_moe_q4_32_down_decode_prefetch_depth(
    uint32_t depth);

typedef enum {
    SCENARIO_Q4_32_GATE_UP,
    SCENARIO_Q4_32_GATE_UP_HWARP16,
    SCENARIO_Q4_32_GATE_UP_TILE32_DP4A,
    SCENARIO_Q4_32_GATE_UP_TILE32_MMA,
    SCENARIO_Q4_32_GATE_UP_TILE32_MMA_PREFETCH1,
    SCENARIO_Q4_32_GATE_UP_TILE32_MMA_PREFETCH2,
    SCENARIO_Q4_32_GATE_UP_HWARP16_AB,
    SCENARIO_Q4_32_GATE_UP_TILE32_DP4A_AB,
    SCENARIO_Q4_32_GATE_UP_TILE32_MMA_AB,
    SCENARIO_Q4_32_GATE_UP_TILE32_MMA_PREFETCH1_AB,
    SCENARIO_Q4_32_GATE_UP_TILE32_MMA_PREFETCH2_AB,
    SCENARIO_Q3A4_GATE_UP,
    SCENARIO_Q4_32_GATE_UP_SPLIT,
    SCENARIO_Q3A4_GATE_UP_SPLIT,
    SCENARIO_Q4_32_GATE_UP_AB,
    SCENARIO_Q3A4_GATE_UP_AB,
    SCENARIO_Q3A4_GATE_UP_FUSED_U1,
    SCENARIO_Q3A4_GATE_UP_FUSED_U2,
    SCENARIO_Q3A4_GATE_UP_FUSED_U4,
    SCENARIO_Q3A4_GATE_UP_FUSED_U1_AB,
    SCENARIO_Q3A4_GATE_UP_FUSED_U2_AB,
    SCENARIO_Q3A4_GATE_UP_FUSED_U4_AB,
    SCENARIO_Q3A4_GATE_UP_HWARP16,
    SCENARIO_Q3A4_GATE_UP_TILE32,
    SCENARIO_Q3A4_GATE_UP_TILE32_DP4A,
    SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K2,
    SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4,
    SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4_PREFETCH1,
    SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4_PREFETCH2,
    SCENARIO_Q3A4_GATE_UP_HWARP16_AB,
    SCENARIO_Q3A4_GATE_UP_TILE32_AB,
    SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_AB,
    SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K2_AB,
    SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4_AB,
    SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4_PREFETCH1_AB,
    SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4_PREFETCH2_AB,
    SCENARIO_Q4_32_GATE_UP_FUSED_U1,
    SCENARIO_Q4_32_GATE_UP_FUSED_U2,
    SCENARIO_Q4_32_GATE_UP_FUSED_U4,
    SCENARIO_Q4_32_GATE_UP_FUSED_U1_AB,
    SCENARIO_Q4_32_GATE_UP_FUSED_U2_AB,
    SCENARIO_Q4_32_GATE_UP_FUSED_U4_AB,
    SCENARIO_Q4_32_DOWN_SLOTS,
    SCENARIO_Q4_32_DOWN_SLOTS_TILE32,
    SCENARIO_Q4_32_DOWN_SLOTS_TILE32_PREFETCH1,
    SCENARIO_Q4_32_DOWN_SLOTS_TILE32_PREFETCH2,
    SCENARIO_Q4_32_DOWN_SLOTS_TILE32_AB,
    SCENARIO_Q4_32_DOWN_SLOTS_TILE32_PREFETCH1_AB,
    SCENARIO_Q4_32_DOWN_SLOTS_TILE32_PREFETCH2_AB,
    SCENARIO_Q4_32_DOWN_PACKED,
    SCENARIO_Q4_32_DOWN_PACKED_TILE32,
    SCENARIO_Q4_32_DOWN_PACKED_TILE32_PREFETCH1,
    SCENARIO_Q4_32_DOWN_PACKED_TILE32_PREFETCH2,
    SCENARIO_Q4_32_DOWN_PACKED_TILE32_AB,
    SCENARIO_Q4_32_DOWN_PACKED_TILE32_PREFETCH1_AB,
    SCENARIO_Q4_32_DOWN_PACKED_TILE32_PREFETCH2_AB,
    SCENARIO_Q8_SINGLE_T32,
    SCENARIO_Q8_PAIR_2048,
    SCENARIO_Q8_PAIR_1024,
    SCENARIO_Q8_KSLICE_T256,
    SCENARIO_Q8_GROUPED_A_HALF,
    SCENARIO_Q8_SHARED_MID,
    SCENARIO_Q8_NATIVE_QUANTIZE,
    SCENARIO_F16_PAIR_256,
    SCENARIO_F16_PAIR_512,
    SCENARIO_F16_PAIR_1024,
    SCENARIO_F16_PAIR_STATE_256,
    SCENARIO_F16_PAIR_STATE_512,
    SCENARIO_F16_PAIR_STATE_1024,
} scenario_kind;

typedef struct {
    const char *name;
    const char *family;
    scenario_kind kind;
} scenario_spec;

static const scenario_spec scenarios[] = {
    { "q4-32-gate-up", "routed-q4-32", SCENARIO_Q4_32_GATE_UP },
    { "q4-32-gate-up-hwarp16", "routed-q4-32-hwarp16",
      SCENARIO_Q4_32_GATE_UP_HWARP16 },
    { "q4-32-gate-up-tile32-dp4a", "routed-q4-32-tile32-dp4a",
      SCENARIO_Q4_32_GATE_UP_TILE32_DP4A },
    { "q4-32-gate-up-tile32-mma", "routed-q4-32-tile32-mma",
      SCENARIO_Q4_32_GATE_UP_TILE32_MMA },
    { "q4-32-gate-up-tile32-mma-prefetch1",
      "routed-q4-32-tile32-mma-prefetch1",
      SCENARIO_Q4_32_GATE_UP_TILE32_MMA_PREFETCH1 },
    { "q4-32-gate-up-tile32-mma-prefetch2",
      "routed-q4-32-tile32-mma-prefetch2",
      SCENARIO_Q4_32_GATE_UP_TILE32_MMA_PREFETCH2 },
    { "q4-32-gate-up-hwarp16-ab", "routed-q4-32-hwarp16-ab",
      SCENARIO_Q4_32_GATE_UP_HWARP16_AB },
    { "q4-32-gate-up-tile32-dp4a-ab", "routed-q4-32-tile32-dp4a-ab",
      SCENARIO_Q4_32_GATE_UP_TILE32_DP4A_AB },
    { "q4-32-gate-up-tile32-mma-ab", "routed-q4-32-tile32-mma-ab",
      SCENARIO_Q4_32_GATE_UP_TILE32_MMA_AB },
    { "q4-32-gate-up-tile32-mma-prefetch1-ab",
      "routed-q4-32-tile32-mma-prefetch1-ab",
      SCENARIO_Q4_32_GATE_UP_TILE32_MMA_PREFETCH1_AB },
    { "q4-32-gate-up-tile32-mma-prefetch2-ab",
      "routed-q4-32-tile32-mma-prefetch2-ab",
      SCENARIO_Q4_32_GATE_UP_TILE32_MMA_PREFETCH2_AB },
    { "q3a4-gate-up", "routed-q3a4", SCENARIO_Q3A4_GATE_UP },
    { "q4-32-gate-up-split", "routed-q4-32-split",
      SCENARIO_Q4_32_GATE_UP_SPLIT },
    { "q3a4-gate-up-split", "routed-q3a4-split",
      SCENARIO_Q3A4_GATE_UP_SPLIT },
    { "q4-32-gate-up-ab", "routed-q4-32-ab",
      SCENARIO_Q4_32_GATE_UP_AB },
    { "q3a4-gate-up-ab", "routed-q3a4-ab",
      SCENARIO_Q3A4_GATE_UP_AB },
    { "q3a4-gate-up-fused-u1", "routed-q3a4-fused-u1",
      SCENARIO_Q3A4_GATE_UP_FUSED_U1 },
    { "q3a4-gate-up-fused-u2", "routed-q3a4-fused-u2",
      SCENARIO_Q3A4_GATE_UP_FUSED_U2 },
    { "q3a4-gate-up-fused-u4", "routed-q3a4-fused-u4",
      SCENARIO_Q3A4_GATE_UP_FUSED_U4 },
    { "q3a4-gate-up-fused-u1-ab", "routed-q3a4-fused-u1-ab",
      SCENARIO_Q3A4_GATE_UP_FUSED_U1_AB },
    { "q3a4-gate-up-fused-u2-ab", "routed-q3a4-fused-u2-ab",
      SCENARIO_Q3A4_GATE_UP_FUSED_U2_AB },
    { "q3a4-gate-up-fused-u4-ab", "routed-q3a4-fused-u4-ab",
      SCENARIO_Q3A4_GATE_UP_FUSED_U4_AB },
    { "q3a4-gate-up-hwarp16", "routed-q3a4-hwarp16",
      SCENARIO_Q3A4_GATE_UP_HWARP16 },
    { "q3a4-gate-up-tile32", "routed-q3a4-tile32",
      SCENARIO_Q3A4_GATE_UP_TILE32 },
    { "q3a4-gate-up-tile32-dp4a", "routed-q3a4-tile32-dp4a",
      SCENARIO_Q3A4_GATE_UP_TILE32_DP4A },
    { "q3a4-gate-up-tile32-dp4a-k2", "routed-q3a4-tile32-dp4a-k2",
      SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K2 },
    { "q3a4-gate-up-tile32-dp4a-k4", "routed-q3a4-tile32-dp4a-k4",
      SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4 },
    { "q3a4-tile32-dp4a-k4", "routed-q3a4-tile32-dp4a-k4",
      SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4 },
    { "q3a4-tile32-dp4a-k4-prefetch1",
      "routed-q3a4-tile32-dp4a-k4-prefetch1",
      SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4_PREFETCH1 },
    { "q3a4-tile32-dp4a-k4-prefetch2",
      "routed-q3a4-tile32-dp4a-k4-prefetch2",
      SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4_PREFETCH2 },
    { "q3a4-gate-up-hwarp16-ab", "routed-q3a4-hwarp16-ab",
      SCENARIO_Q3A4_GATE_UP_HWARP16_AB },
    { "q3a4-gate-up-tile32-ab", "routed-q3a4-tile32-ab",
      SCENARIO_Q3A4_GATE_UP_TILE32_AB },
    { "q3a4-gate-up-tile32-dp4a-ab", "routed-q3a4-tile32-dp4a-ab",
      SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_AB },
    { "q3a4-gate-up-tile32-dp4a-k2-ab",
      "routed-q3a4-tile32-dp4a-k2-ab",
      SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K2_AB },
    { "q3a4-gate-up-tile32-dp4a-k4-ab",
      "routed-q3a4-tile32-dp4a-k4-ab",
      SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4_AB },
    { "q3a4-tile32-dp4a-k4-ab",
      "routed-q3a4-tile32-dp4a-k4-ab",
      SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4_AB },
    { "q3a4-tile32-dp4a-k4-prefetch1-ab",
      "routed-q3a4-tile32-dp4a-k4-prefetch1-ab",
      SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4_PREFETCH1_AB },
    { "q3a4-tile32-dp4a-k4-prefetch2-ab",
      "routed-q3a4-tile32-dp4a-k4-prefetch2-ab",
      SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4_PREFETCH2_AB },
    { "q4-32-gate-up-fused-u1", "routed-q4-32-fused-u1",
      SCENARIO_Q4_32_GATE_UP_FUSED_U1 },
    { "q4-32-gate-up-fused-u2", "routed-q4-32-fused-u2",
      SCENARIO_Q4_32_GATE_UP_FUSED_U2 },
    { "q4-32-gate-up-fused-u4", "routed-q4-32-fused-u4",
      SCENARIO_Q4_32_GATE_UP_FUSED_U4 },
    { "q4-32-gate-up-fused-u1-ab", "routed-q4-32-fused-u1-ab",
      SCENARIO_Q4_32_GATE_UP_FUSED_U1_AB },
    { "q4-32-gate-up-fused-u2-ab", "routed-q4-32-fused-u2-ab",
      SCENARIO_Q4_32_GATE_UP_FUSED_U2_AB },
    { "q4-32-gate-up-fused-u4-ab", "routed-q4-32-fused-u4-ab",
      SCENARIO_Q4_32_GATE_UP_FUSED_U4_AB },
    { "q4-32-down-slots", "routed-q4-32-down-slots",
      SCENARIO_Q4_32_DOWN_SLOTS },
    { "q4-32-down-slots-tile32", "routed-q4-32-down-slots-tile32",
      SCENARIO_Q4_32_DOWN_SLOTS_TILE32 },
    { "q4-32-down-slots-tile32-prefetch1",
      "routed-q4-32-down-slots-tile32-prefetch1",
      SCENARIO_Q4_32_DOWN_SLOTS_TILE32_PREFETCH1 },
    { "q4-32-down-slots-tile32-prefetch2",
      "routed-q4-32-down-slots-tile32-prefetch2",
      SCENARIO_Q4_32_DOWN_SLOTS_TILE32_PREFETCH2 },
    { "q4-32-down-slots-tile32-ab", "routed-q4-32-down-slots-tile32-ab",
      SCENARIO_Q4_32_DOWN_SLOTS_TILE32_AB },
    { "q4-32-down-slots-tile32-prefetch1-ab",
      "routed-q4-32-down-slots-tile32-prefetch1-ab",
      SCENARIO_Q4_32_DOWN_SLOTS_TILE32_PREFETCH1_AB },
    { "q4-32-down-slots-tile32-prefetch2-ab",
      "routed-q4-32-down-slots-tile32-prefetch2-ab",
      SCENARIO_Q4_32_DOWN_SLOTS_TILE32_PREFETCH2_AB },
    { "q4-32-down-packed", "routed-q4-32-down-packed",
      SCENARIO_Q4_32_DOWN_PACKED },
    { "q4-32-down-packed-tile32", "routed-q4-32-down-packed-tile32",
      SCENARIO_Q4_32_DOWN_PACKED_TILE32 },
    { "q4-32-down-packed-tile32-prefetch1",
      "routed-q4-32-down-packed-tile32-prefetch1",
      SCENARIO_Q4_32_DOWN_PACKED_TILE32_PREFETCH1 },
    { "q4-32-down-packed-tile32-prefetch2",
      "routed-q4-32-down-packed-tile32-prefetch2",
      SCENARIO_Q4_32_DOWN_PACKED_TILE32_PREFETCH2 },
    { "q4-32-down-packed-tile32-ab", "routed-q4-32-down-packed-tile32-ab",
      SCENARIO_Q4_32_DOWN_PACKED_TILE32_AB },
    { "q4-32-down-packed-tile32-prefetch1-ab",
      "routed-q4-32-down-packed-tile32-prefetch1-ab",
      SCENARIO_Q4_32_DOWN_PACKED_TILE32_PREFETCH1_AB },
    { "q4-32-down-packed-tile32-prefetch2-ab",
      "routed-q4-32-down-packed-tile32-prefetch2-ab",
      SCENARIO_Q4_32_DOWN_PACKED_TILE32_PREFETCH2_AB },
    { "q8-single-t32", "dense-q8-single", SCENARIO_Q8_SINGLE_T32 },
    { "q8-pair-2048", "dense-q8-pair", SCENARIO_Q8_PAIR_2048 },
    { "q8-pair-1024", "dense-q8-pair", SCENARIO_Q8_PAIR_1024 },
    { "q8-kslice-t256", "dense-q8-kslice", SCENARIO_Q8_KSLICE_T256 },
    { "q8-grouped-a-half", "dense-q8-grouped-a", SCENARIO_Q8_GROUPED_A_HALF },
    { "q8-shared-mid", "dense-q8-shared", SCENARIO_Q8_SHARED_MID },
    { "q8-native-quantize", "routed-native-q8-quantize",
      SCENARIO_Q8_NATIVE_QUANTIZE },
    { "f16-pair-256", "dense-f16-pair", SCENARIO_F16_PAIR_256 },
    { "f16-pair-512", "dense-f16-pair", SCENARIO_F16_PAIR_512 },
    { "f16-pair-1024", "dense-f16-pair", SCENARIO_F16_PAIR_1024 },
    { "f16-pair-state-256", "dense-f16-pair-state",
      SCENARIO_F16_PAIR_STATE_256 },
    { "f16-pair-state-512", "dense-f16-pair-state",
      SCENARIO_F16_PAIR_STATE_512 },
    { "f16-pair-state-1024", "dense-f16-pair-state",
      SCENARIO_F16_PAIR_STATE_1024 },
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

static uint32_t positive_env_u32(const char *name, uint32_t fallback) {
    const char *text = getenv(name);
    if (!text || !text[0]) return fallback;
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (end == text || *end || value == 0u || value > UINT32_MAX)
        return fallback;
    return (uint32_t)value;
}

static int compare_float(const void *lhs, const void *rhs) {
    const float a = *(const float *)lhs;
    const float b = *(const float *)rhs;
    return (a > b) - (a < b);
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

static int run_routed_gate_up_candidate(int q3a4, int split,
                                        uint32_t fused_unroll,
                                        uint32_t q4_32_mapping,
                                        uint32_t q4_32_prefetch_depth,
                                        uint32_t q3a4_mapping,
                                        uint32_t q3a4_ksplit,
                                        uint32_t q3a4_prefetch_depth,
                                        int benchmark) {
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
    ds4_gpu_test_set_moe_q32_decode_split(split);
    ds4_gpu_test_set_moe_q32_decode_fused_lowreg(fused_unroll);
    ds4_gpu_test_set_moe_q4_32_decode_mapping(q4_32_mapping);
    ds4_gpu_test_set_moe_q4_32_decode_prefetch_depth(
        q4_32_prefetch_depth);
    ds4_gpu_test_set_moe_q3a4_decode_mapping(q3a4_mapping);
    ds4_gpu_test_set_moe_q3a4_decode_ksplit(q3a4_ksplit);
    ds4_gpu_test_set_moe_q3a4_decode_prefetch_depth(q3a4_prefetch_depth);
#define RUN_ROUTED_GATE_UP() ds4_gpu_routed_moe_one_owned_tensor( \
            out, gate, up, mid, down, model_storage, model_bytes, \
            gate_offset, up_offset, down_offset, \
            q3a4 ? PROFILE_TYPE_SM75_Q3A4 : PROFILE_TYPE_SM75_Q4_32, \
            PROFILE_TYPE_SM75_Q4_32, \
            gate_expert_bytes, gate_row_bytes, \
            down_expert_bytes, down_row_bytes, \
            in_dim, mid_dim, out_dim, selected, weights, \
            n_total_experts, n_expert, 0u, resident_experts, 10.0f, x, \
            NULL, false, NULL)
    if (!RUN_ROUTED_GATE_UP() || !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: routed decode production launch failed\n");
        goto cleanup;
    }
    printf("q32_split=%s\nq32_fused_lowreg_unroll=%u\n"
           "q4_32_decode_mapping=%u\n"
           "q4_32_decode_prefetch_depth=%u\n"
           "q3a4_decode_mapping=%u\nq3a4_decode_ksplit=%u\n"
           "q3a4_decode_prefetch_depth=%u\n",
           split ? "enabled" : "disabled", fused_unroll, q4_32_mapping,
           q4_32_prefetch_depth,
           q3a4_mapping, q3a4_ksplit, q3a4_prefetch_depth);
    if (benchmark) {
        const uint32_t control_mapping = q3a4_ksplit > 1u ? 3u : 0u;
        const uint32_t control_ksplit =
            q3a4_prefetch_depth != 0u ? 4u : 1u;
        ds4_gpu_test_set_moe_q32_decode_split(0);
        ds4_gpu_test_set_moe_q32_decode_fused_lowreg(0u);
        ds4_gpu_test_set_moe_q4_32_decode_mapping(
            q4_32_prefetch_depth != 0u ? 3u : 0u);
        ds4_gpu_test_set_moe_q4_32_decode_prefetch_depth(0u);
        ds4_gpu_test_set_moe_q3a4_decode_mapping(control_mapping);
        ds4_gpu_test_set_moe_q3a4_decode_ksplit(control_ksplit);
        ds4_gpu_test_set_moe_q3a4_decode_prefetch_depth(0u);
        if (!RUN_ROUTED_GATE_UP() || !ds4_gpu_synchronize()) {
            fprintf(stderr, "error: routed decode control warmup failed\n");
            goto cleanup;
        }
        ds4_gpu_test_set_moe_q32_decode_split(
            fused_unroll == 0u && q4_32_mapping == 0u &&
            q3a4_mapping == 0u);
        ds4_gpu_test_set_moe_q32_decode_fused_lowreg(fused_unroll);
        ds4_gpu_test_set_moe_q4_32_decode_mapping(q4_32_mapping);
        ds4_gpu_test_set_moe_q4_32_decode_prefetch_depth(
            q4_32_prefetch_depth);
        ds4_gpu_test_set_moe_q3a4_decode_mapping(q3a4_mapping);
        ds4_gpu_test_set_moe_q3a4_decode_ksplit(q3a4_ksplit);
        ds4_gpu_test_set_moe_q3a4_decode_prefetch_depth(
            q3a4_prefetch_depth);
        if (!RUN_ROUTED_GATE_UP() || !ds4_gpu_synchronize()) {
            fprintf(stderr, "error: routed decode candidate warmup failed\n");
            goto cleanup;
        }
        const uint32_t rounds = positive_env_u32("TIMING_ROUNDS", 7u);
        const uint32_t repeats = positive_env_u32("TIMING_REPEATS", 20u);
        float *control_ms = (float *)malloc(rounds * sizeof(float));
        float *candidate_ms = (float *)malloc(rounds * sizeof(float));
        ds4_gpu_timer *timer = ds4_gpu_timer_create();
        if (!control_ms || !candidate_ms || !timer) {
            fprintf(stderr, "error: routed decode timing allocation failed\n");
            free(candidate_ms); free(control_ms); ds4_gpu_timer_free(timer);
            goto cleanup;
        }
        for (uint32_t round = 0; round < rounds; round++) {
            for (uint32_t order = 0; order < 2u; order++) {
                const int candidate = ((round & 1u) ^ order) != 0u;
                ds4_gpu_test_set_moe_q32_decode_split(
                    fused_unroll == 0u && q4_32_mapping == 0u &&
                    q3a4_mapping == 0u
                        ? candidate : 0);
                ds4_gpu_test_set_moe_q32_decode_fused_lowreg(
                    candidate ? fused_unroll : 0u);
                ds4_gpu_test_set_moe_q4_32_decode_mapping(
                    candidate ? q4_32_mapping
                              : (q4_32_prefetch_depth != 0u ? 3u : 0u));
                ds4_gpu_test_set_moe_q4_32_decode_prefetch_depth(
                    candidate ? q4_32_prefetch_depth : 0u);
                ds4_gpu_test_set_moe_q3a4_decode_mapping(
                    candidate ? q3a4_mapping : control_mapping);
                ds4_gpu_test_set_moe_q3a4_decode_ksplit(
                    candidate ? q3a4_ksplit : control_ksplit);
                ds4_gpu_test_set_moe_q3a4_decode_prefetch_depth(
                    candidate ? q3a4_prefetch_depth : 0u);
                if (!ds4_gpu_timer_record_start(timer)) goto timing_error;
                for (uint32_t repeat = 0; repeat < repeats; repeat++)
                    if (!RUN_ROUTED_GATE_UP()) goto timing_error;
                if (!ds4_gpu_timer_record_end(timer)) goto timing_error;
                float elapsed = 0.0f;
                if (!ds4_gpu_timer_elapsed_ms(timer, &elapsed))
                    goto timing_error;
                (candidate ? candidate_ms : control_ms)[round] =
                    elapsed / (float)repeats;
            }
        }
        qsort(control_ms, rounds, sizeof(float), compare_float);
        qsort(candidate_ms, rounds, sizeof(float), compare_float);
        const float control_median = control_ms[rounds / 2u];
        const float candidate_median = candidate_ms[rounds / 2u];
        const char *candidate_kind = "split";
        if (q4_32_mapping == 1u)
            candidate_kind = "q4-32-hwarp16";
        else if (q4_32_mapping == 2u)
            candidate_kind = "q4-32-tile32-dp4a";
        else if (q4_32_prefetch_depth == 1u)
            candidate_kind = "q4-32-tile32-mma-prefetch1";
        else if (q4_32_prefetch_depth == 2u)
            candidate_kind = "q4-32-tile32-mma-prefetch2";
        else if (q4_32_mapping == 3u)
            candidate_kind = "q4-32-tile32-mma";
        else if (q3a4_prefetch_depth == 1u)
            candidate_kind = "q3a4-tile32-dp4a-k4-prefetch1";
        else if (q3a4_prefetch_depth == 2u)
            candidate_kind = "q3a4-tile32-dp4a-k4-prefetch2";
        else if (q3a4_ksplit == 2u)
            candidate_kind = "q3a4-tile32-dp4a-k2";
        else if (q3a4_ksplit == 4u)
            candidate_kind = "q3a4-tile32-dp4a-k4";
        else if (q3a4_mapping == 1u)
            candidate_kind = "q3a4-hwarp16";
        else if (q3a4_mapping == 2u)
            candidate_kind = "q3a4-tile32";
        else if (q3a4_mapping == 3u)
            candidate_kind = "q3a4-tile32-dp4a";
        else if (fused_unroll)
            candidate_kind = "fused-lowreg";
        printf("timing_scope=production-owned-call-inclusive\n"
               "timing_rounds=%u\ntiming_repeats=%u\n"
               "candidate_kind=%s\ncontrol_median_ms=%.9g\n"
               "candidate_median_ms=%.9g\ncandidate_speedup=%.9g\n",
               rounds, repeats, candidate_kind,
               control_median, candidate_median,
               control_median / candidate_median);
        if (fused_unroll == 0u && q4_32_mapping == 0u &&
            q3a4_mapping == 0u)
            printf("split_median_ms=%.9g\nsplit_speedup=%.9g\n",
                   candidate_median, control_median / candidate_median);
        ds4_gpu_timer_free(timer);
        free(candidate_ms); free(control_ms);
        ds4_gpu_test_set_moe_q32_decode_split(split);
        ds4_gpu_test_set_moe_q32_decode_fused_lowreg(fused_unroll);
        ds4_gpu_test_set_moe_q4_32_decode_mapping(q4_32_mapping);
        ds4_gpu_test_set_moe_q4_32_decode_prefetch_depth(
            q4_32_prefetch_depth);
        ds4_gpu_test_set_moe_q3a4_decode_mapping(q3a4_mapping);
        ds4_gpu_test_set_moe_q3a4_decode_ksplit(q3a4_ksplit);
        ds4_gpu_test_set_moe_q3a4_decode_prefetch_depth(
            q3a4_prefetch_depth);
        goto timing_done;
timing_error:
        fprintf(stderr, "error: routed decode inclusive timing failed\n");
        ds4_gpu_timer_free(timer);
        free(candidate_ms); free(control_ms);
        goto cleanup;
timing_done:;
    }
    ok = verify_zero_tensor(mid, (uint64_t)n_expert * mid_dim,
                            q3a4 ? "q3a4-gate-up" : "q4-32-gate-up");

cleanup:
    ds4_gpu_test_set_moe_q32_decode_split(0);
    ds4_gpu_test_set_moe_q32_decode_fused_lowreg(0u);
    ds4_gpu_test_set_moe_q4_32_decode_mapping(0u);
    ds4_gpu_test_set_moe_q4_32_decode_prefetch_depth(0u);
    ds4_gpu_test_set_moe_q3a4_decode_mapping(0u);
    ds4_gpu_test_set_moe_q3a4_decode_ksplit(1u);
    ds4_gpu_test_set_moe_q3a4_decode_prefetch_depth(0u);
    ds4_gpu_set_routed_q4_layout(0u);
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(x);
#undef RUN_ROUTED_GATE_UP
    return ok;
}

static int run_routed_gate_up(int q3a4, int split,
                              uint32_t fused_unroll,
                              uint32_t q3a4_mapping,
                              uint32_t q3a4_ksplit, int benchmark) {
    return run_routed_gate_up_candidate(
        q3a4, split, fused_unroll, 0u, 0u,
        q3a4_mapping, q3a4_ksplit, 0u,
        benchmark);
}

static int run_routed_gate_up_q4_mapping(uint32_t mapping,
                                         uint32_t prefetch_depth,
                                         int benchmark) {
    return run_routed_gate_up_candidate(
        0, 0, 0u, mapping, prefetch_depth, 0u, 1u, 0u, benchmark);
}

static int run_q4_32_down_candidate(int packed, uint32_t mapping,
                                    uint32_t prefetch_depth, int benchmark) {
    const uint32_t n_total_experts = 8u;
    const uint32_t n_expert = 6u;
    const uint32_t in_dim = 256u;
    const uint32_t mid_dim = 2048u;
    const uint32_t out_dim = 4096u;
    const uint32_t resident_base = packed ? 4u : 0u;
    const uint32_t resident_count = 4u;
    const uint64_t gate_row_bytes = (in_dim / 256u) * 136u;
    const uint64_t gate_expert_bytes =
        (uint64_t)mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = (mid_dim / 256u) * 136u;
    const uint64_t down_expert_bytes =
        (uint64_t)out_dim * down_row_bytes;
    const uint64_t gate_offset = 0u;
    const uint64_t up_offset = gate_expert_bytes * n_total_experts;
    const uint64_t down_offset =
        up_offset + gate_expert_bytes * n_total_experts;
    const uint64_t output_count = (packed ? 4ull : 6ull) * out_dim;
    const uint64_t slot_scratch_count = 6ull * out_dim;
    const uint64_t mid_count = 6ull * mid_dim;
    const int32_t slots_selected[6] = {0, 1, 4, 2, 3, 5};
    /* Both three-slot groups begin with an owned prefix pair. */
    const int32_t packed_selected[6] = {4, 5, 0, 6, 7, 1};
    const int32_t *selected_host = packed ? packed_selected : slots_selected;
    const float weights_host[6] = {
        0.125f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f,
    };
    uint64_t model_bytes;
    if (!checked_add(down_offset,
                     down_expert_bytes * n_total_experts, &model_bytes) ||
        !install_zero_model(model_bytes)) return 0;

    ds4_gpu_tensor *x = input_tensor(in_dim);
    ds4_gpu_tensor *selected = zero_tensor(n_expert, sizeof(int32_t));
    ds4_gpu_tensor *weights = zero_tensor(n_expert, sizeof(float));
    ds4_gpu_tensor *out = zero_tensor(out_dim, sizeof(float));
    ds4_gpu_tensor *gate = zero_tensor(mid_count, sizeof(float));
    ds4_gpu_tensor *up = zero_tensor(mid_count, sizeof(float));
    ds4_gpu_tensor *mid = zero_tensor(mid_count, sizeof(float));
    ds4_gpu_tensor *down = zero_tensor(slot_scratch_count, sizeof(float));
    ds4_gpu_tensor *down_output = zero_tensor(output_count, sizeof(float));
    int ok = 0;
    if (!x || !selected || !weights || !out || !gate || !up || !mid ||
        !down || !down_output ||
        !ds4_gpu_tensor_write(selected, 0u, selected_host,
                              sizeof(slots_selected)) ||
        !ds4_gpu_tensor_write(weights, 0u, weights_host,
                              sizeof(weights_host))) {
        fprintf(stderr, "error: Q4-32 down tensor setup failed\n");
        goto cleanup;
    }
    ds4_gpu_set_routed_q4_layout(DS4_TENSOR_LAYOUT_SM75_Q4_32 |
                                 DS4_TENSOR_LAYOUT_SM75_Q3A4);
    ds4_gpu_test_set_moe_q32_decode_split(0);
    ds4_gpu_test_set_moe_q32_decode_fused_lowreg(0u);
    ds4_gpu_test_set_moe_q3a4_decode_mapping(0u);
    ds4_gpu_test_set_moe_q3a4_decode_ksplit(1u);
    ds4_gpu_test_set_moe_q3a4_decode_prefetch_depth(0u);
    ds4_gpu_test_set_moe_q4_32_down_decode_mapping(mapping);
    ds4_gpu_test_set_moe_q4_32_down_decode_prefetch_depth(prefetch_depth);

#define RUN_Q4_32_DOWN() ds4_gpu_routed_moe_one_owned_tensor( \
            out, gate, up, mid, down, model_storage, model_bytes, \
            gate_offset, up_offset, down_offset, \
            PROFILE_TYPE_SM75_Q4_32, PROFILE_TYPE_SM75_Q4_32, \
            gate_expert_bytes, gate_row_bytes, \
            down_expert_bytes, down_row_bytes, \
            in_dim, mid_dim, out_dim, selected, weights, \
            n_total_experts, n_expert, resident_base, resident_count, \
            10.0f, x, down_output, packed != 0, NULL)

    if (!RUN_Q4_32_DOWN() || !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: Q4-32 down production launch failed\n");
        goto cleanup;
    }
    printf("down_output_kind=%s\nq4_32_down_decode_mapping=%u\n"
           "q4_32_down_decode_prefetch_depth=%u\n"
           "midq_blocks=8\n",
           packed ? "owned_packed-prefix-pair" : "owned_slots", mapping,
           prefetch_depth);
    if (benchmark) {
        ds4_gpu_test_set_moe_q4_32_down_decode_mapping(
            prefetch_depth != 0u ? 1u : 0u);
        ds4_gpu_test_set_moe_q4_32_down_decode_prefetch_depth(0u);
        if (!RUN_Q4_32_DOWN() || !ds4_gpu_synchronize()) {
            fprintf(stderr, "error: Q4-32 down control warmup failed\n");
            goto cleanup;
        }
        ds4_gpu_test_set_moe_q4_32_down_decode_mapping(1u);
        ds4_gpu_test_set_moe_q4_32_down_decode_prefetch_depth(
            prefetch_depth);
        if (!RUN_Q4_32_DOWN() || !ds4_gpu_synchronize()) {
            fprintf(stderr, "error: Q4-32 down candidate warmup failed\n");
            goto cleanup;
        }
        const uint32_t rounds = positive_env_u32("TIMING_ROUNDS", 9u);
        const uint32_t repeats = positive_env_u32("TIMING_REPEATS", 25u);
        if ((rounds & 1u) == 0u) {
            fprintf(stderr, "error: TIMING_ROUNDS must be odd\n");
            goto cleanup;
        }
        float *control_ms = (float *)malloc(rounds * sizeof(float));
        float *candidate_ms = (float *)malloc(rounds * sizeof(float));
        ds4_gpu_timer *timer = ds4_gpu_timer_create();
        if (!control_ms || !candidate_ms || !timer) {
            fprintf(stderr, "error: Q4-32 down timing allocation failed\n");
            free(candidate_ms); free(control_ms); ds4_gpu_timer_free(timer);
            goto cleanup;
        }
        for (uint32_t round = 0; round < rounds; round++) {
            for (uint32_t order = 0; order < 2u; order++) {
                const int candidate = ((round & 1u) ^ order) != 0u;
                ds4_gpu_test_set_moe_q4_32_down_decode_mapping(
                    candidate ? 1u : (prefetch_depth != 0u ? 1u : 0u));
                ds4_gpu_test_set_moe_q4_32_down_decode_prefetch_depth(
                    candidate ? prefetch_depth : 0u);
                if (!ds4_gpu_timer_record_start(timer)) goto timing_error;
                for (uint32_t repeat = 0; repeat < repeats; repeat++)
                    if (!RUN_Q4_32_DOWN()) goto timing_error;
                if (!ds4_gpu_timer_record_end(timer)) goto timing_error;
                float elapsed = 0.0f;
                if (!ds4_gpu_timer_elapsed_ms(timer, &elapsed))
                    goto timing_error;
                (candidate ? candidate_ms : control_ms)[round] =
                    elapsed / (float)repeats;
            }
        }
        qsort(control_ms, rounds, sizeof(float), compare_float);
        qsort(candidate_ms, rounds, sizeof(float), compare_float);
        const float control_median = control_ms[rounds / 2u];
        const float candidate_median = candidate_ms[rounds / 2u];
        printf("timing_scope=production-owned-call-inclusive\n"
               "timing_rounds=%u\ntiming_repeats=%u\n"
               "candidate_kind=q4-32-down-tile32-int4-%s%s\n"
               "control_median_ms=%.9g\ncandidate_median_ms=%.9g\n"
               "candidate_speedup=%.9g\n",
               rounds, repeats, packed ? "packed" : "slots",
               prefetch_depth == 1u ? "-prefetch1"
                   : (prefetch_depth == 2u ? "-prefetch2" : ""),
               control_median, candidate_median,
               control_median / candidate_median);
        ds4_gpu_timer_free(timer);
        free(candidate_ms); free(control_ms);
        ds4_gpu_test_set_moe_q4_32_down_decode_mapping(mapping);
        ds4_gpu_test_set_moe_q4_32_down_decode_prefetch_depth(
            prefetch_depth);
        goto timing_done;
timing_error:
        fprintf(stderr, "error: Q4-32 down inclusive timing failed\n");
        ds4_gpu_timer_free(timer);
        free(candidate_ms); free(control_ms);
        goto cleanup;
timing_done:;
    }
    ok = verify_zero_tensor(
        down_output, output_count,
        packed ? "q4-32-down-packed" : "q4-32-down-slots");

cleanup:
    ds4_gpu_test_set_moe_q4_32_down_decode_prefetch_depth(0u);
    ds4_gpu_test_set_moe_q4_32_down_decode_mapping(0u);
    ds4_gpu_test_set_moe_q3a4_decode_prefetch_depth(0u);
    ds4_gpu_test_set_moe_q3a4_decode_ksplit(1u);
    ds4_gpu_test_set_moe_q3a4_decode_mapping(0u);
    ds4_gpu_set_routed_q4_layout(0u);
    ds4_gpu_tensor_free(down_output);
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(x);
#undef RUN_Q4_32_DOWN
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
#define RUN_Q8_KSLICE() ds4_gpu_matmul_q8_0_kslice_rows_tensor( \
            out, model_storage, model_bytes, 0u, full_in_dim, out_dim, \
            0u, slice_dim, x, 1u)
    if (!x || !out || !RUN_Q8_KSLICE() || !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: K-slice dense-Q8 decode launch failed\n");
        goto cleanup;
    }
    if (getenv("DS4_Q8_INTERLEAVED_PROFILE_AB") != NULL) {
        (void)setenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_DECODE", "0", 1);
        if (!RUN_Q8_KSLICE() || !ds4_gpu_synchronize()) goto timing_error;
        (void)setenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_DECODE", "1", 1);
        (void)setenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_DIRECT_XQ", "0", 1);
        (void)setenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_K128", "0", 1);
        if (!RUN_Q8_KSLICE() || !ds4_gpu_synchronize()) goto timing_error;
        (void)setenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_DIRECT_XQ", "1", 1);
        (void)setenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_K128", "1", 1);
        if (!RUN_Q8_KSLICE() || !ds4_gpu_synchronize()) goto timing_error;
        const uint32_t rounds = positive_env_u32("TIMING_ROUNDS", 9u);
        const uint32_t repeats = positive_env_u32("TIMING_REPEATS", 100u);
        if ((rounds & 1u) == 0u) {
            fprintf(stderr, "error: TIMING_ROUNDS must be odd\n");
            goto timing_error;
        }
        float *control_ms = (float *)malloc(rounds * sizeof(float));
        float *baseline_ms = (float *)malloc(rounds * sizeof(float));
        float *candidate_ms = (float *)malloc(rounds * sizeof(float));
        ds4_gpu_timer *timer = ds4_gpu_timer_create();
        if (!control_ms || !baseline_ms || !candidate_ms || !timer) {
            free(candidate_ms); free(baseline_ms); free(control_ms);
            ds4_gpu_timer_free(timer);
            goto timing_error;
        }
        for (uint32_t round = 0u; round < rounds; ++round) {
            for (uint32_t order = 0u; order < 3u; ++order) {
                const uint32_t arm = (round + order) % 3u;
                (void)setenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_DECODE",
                             arm == 0u ? "0" : "1", 1);
                (void)setenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_DIRECT_XQ",
                             arm == 2u ? "1" : "0", 1);
                (void)setenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_K128",
                             arm == 2u ? "1" : "0", 1);
                if (!ds4_gpu_timer_record_start(timer)) goto timing_alloc_error;
                for (uint32_t repeat = 0u; repeat < repeats; ++repeat) {
                    if (!RUN_Q8_KSLICE()) goto timing_alloc_error;
                }
                if (!ds4_gpu_timer_record_end(timer)) goto timing_alloc_error;
                float elapsed = 0.0f;
                if (!ds4_gpu_timer_elapsed_ms(timer, &elapsed))
                    goto timing_alloc_error;
                float *samples = arm == 0u ? control_ms :
                    (arm == 1u ? baseline_ms : candidate_ms);
                samples[round] = elapsed / (float)repeats;
            }
        }
        qsort(control_ms, rounds, sizeof(float), compare_float);
        qsort(baseline_ms, rounds, sizeof(float), compare_float);
        qsort(candidate_ms, rounds, sizeof(float), compare_float);
        printf("timing_scope=production-shape-owned-call-inclusive\n"
               "timing_rounds=%u\ntiming_repeats=%u\n"
               "baseline_kind=q8-kslice-warp-interleaved\n"
               "candidate_kind=q8-kslice-warp-interleaved-direct-xq-k128\n"
               "control_median_ms=%.9g\n"
               "baseline_interleaved_median_ms=%.9g\n"
               "candidate_median_ms=%.9g\n"
               "baseline_over_control_speedup=%.9g\n"
               "candidate_over_baseline_speedup=%.9g\n"
               "candidate_speedup=%.9g\n",
               rounds, repeats, control_ms[rounds / 2u],
               baseline_ms[rounds / 2u],
               candidate_ms[rounds / 2u],
               control_ms[rounds / 2u] / baseline_ms[rounds / 2u],
               baseline_ms[rounds / 2u] / candidate_ms[rounds / 2u],
               control_ms[rounds / 2u] / candidate_ms[rounds / 2u]);
        ds4_gpu_timer_free(timer);
        free(candidate_ms); free(baseline_ms); free(control_ms);
        goto timing_done;
timing_alloc_error:
        ds4_gpu_timer_free(timer);
        free(candidate_ms); free(baseline_ms); free(control_ms);
timing_error:
        fprintf(stderr, "error: K-slice interleaved timing failed\n");
        goto cleanup;
timing_done:;
    }
    ok = verify_zero_tensor(out, out_dim, "q8-kslice-t256");
cleanup:
    (void)unsetenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_DECODE");
    (void)unsetenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_DIRECT_XQ");
    (void)unsetenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_B_K128");
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
#undef RUN_Q8_KSLICE
    return ok;
}

static int run_q8_grouped_a(void) {
    const uint32_t groups_total = 8u;
    const uint32_t group_count = 4u;
    const uint64_t group_dim = 4096u;
    const uint64_t rank = 1024u;
    const uint64_t low_dim = (uint64_t)group_count * rank;
    const uint64_t model_bytes =
        q8_matrix_bytes(group_dim, (uint64_t)groups_total * rank);
    if (!install_zero_model(model_bytes)) return 0;
    ds4_gpu_tensor *heads = input_tensor((uint64_t)groups_total * group_dim);
    ds4_gpu_tensor *low = zero_tensor(low_dim, sizeof(float));
    int ok = 0;
#define RUN_Q8_GROUPED_A() ds4_gpu_attention_output_q8_batch_low_shard_tensor( \
            low, model_storage, model_bytes, 0u, group_dim, rank, \
            groups_total, 0u, group_count, heads, 1u)
    if (!heads || !low || !RUN_Q8_GROUPED_A() ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: grouped-A dense-Q8 decode launch failed\n");
        goto cleanup;
    }
    if (getenv("DS4_Q8_INTERLEAVED_PROFILE_AB") != NULL) {
        (void)setenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_A_DECODE", "0", 1);
        if (!RUN_Q8_GROUPED_A() || !ds4_gpu_synchronize()) goto timing_error;
        (void)setenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_A_DECODE", "1", 1);
        (void)setenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_A_DIRECT_XQ", "1", 1);
        (void)setenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_A_K128", "1", 1);
        if (!RUN_Q8_GROUPED_A() || !ds4_gpu_synchronize()) goto timing_error;
        const uint32_t rounds = positive_env_u32("TIMING_ROUNDS", 9u);
        const uint32_t repeats = positive_env_u32("TIMING_REPEATS", 100u);
        if ((rounds & 1u) == 0u) {
            fprintf(stderr, "error: TIMING_ROUNDS must be odd\n");
            goto timing_error;
        }
        float *control_ms = (float *)malloc(rounds * sizeof(float));
        float *candidate_ms = (float *)malloc(rounds * sizeof(float));
        ds4_gpu_timer *timer = ds4_gpu_timer_create();
        if (!control_ms || !candidate_ms || !timer) {
            free(candidate_ms); free(control_ms);
            ds4_gpu_timer_free(timer);
            goto timing_error;
        }
        for (uint32_t round = 0u; round < rounds; ++round) {
            for (uint32_t order = 0u; order < 2u; ++order) {
                const uint32_t arm = (round + order) % 2u;
                (void)setenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_A_DECODE",
                             arm == 0u ? "0" : "1", 1);
                (void)setenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_A_DIRECT_XQ",
                             arm == 1u ? "1" : "0", 1);
                (void)setenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_A_K128",
                             arm == 1u ? "1" : "0", 1);
                if (!ds4_gpu_timer_record_start(timer)) goto timing_alloc_error;
                for (uint32_t repeat = 0u; repeat < repeats; ++repeat) {
                    if (!RUN_Q8_GROUPED_A()) goto timing_alloc_error;
                }
                if (!ds4_gpu_timer_record_end(timer)) goto timing_alloc_error;
                float elapsed = 0.0f;
                if (!ds4_gpu_timer_elapsed_ms(timer, &elapsed))
                    goto timing_alloc_error;
                float *samples = arm == 0u ? control_ms : candidate_ms;
                samples[round] = elapsed / (float)repeats;
            }
        }
        qsort(control_ms, rounds, sizeof(float), compare_float);
        qsort(candidate_ms, rounds, sizeof(float), compare_float);
        printf("timing_scope=production-shape-owned-call-inclusive\n"
               "timing_rounds=%u\ntiming_repeats=%u\n"
               "candidate_kind=q8-grouped-a-warp-interleaved-direct-xq-k128\n"
               "control_median_ms=%.9g\n"
               "candidate_median_ms=%.9g\n"
               "candidate_speedup=%.9g\n",
               rounds, repeats, control_ms[rounds / 2u],
               candidate_ms[rounds / 2u],
               control_ms[rounds / 2u] / candidate_ms[rounds / 2u]);
        ds4_gpu_timer_free(timer);
        free(candidate_ms); free(control_ms);
        goto timing_done;
timing_alloc_error:
        ds4_gpu_timer_free(timer);
        free(candidate_ms); free(control_ms);
timing_error:
        fprintf(stderr, "error: grouped-A interleaved timing failed\n");
        goto cleanup;
timing_done:;
    }
    ok = verify_zero_tensor(low, low_dim, "q8-grouped-a-half");
cleanup:
    (void)unsetenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_A_DECODE");
    (void)unsetenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_A_DIRECT_XQ");
    (void)unsetenv("DS4_CUDA_Q8_WARP_INTERLEAVED_ATTN_A_K128");
    ds4_gpu_tensor_free(low);
    ds4_gpu_tensor_free(heads);
#undef RUN_Q8_GROUPED_A
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

static int run_f16_pair_state(uint32_t width) {
    const uint64_t in_dim = 4096u;
    const uint32_t ratio = width == 512u ? 128u : 4u;
    const uint32_t state_rows = ratio == 4u ? 2u * ratio : ratio;
    uint64_t one_bytes, weights_bytes, ape_bytes, model_bytes;
    if (!checked_mul(in_dim * width, sizeof(uint16_t), &one_bytes) ||
        !checked_mul(one_bytes, 2u, &weights_bytes) ||
        !checked_mul((uint64_t)ratio * width, sizeof(float), &ape_bytes) ||
        !checked_add(weights_bytes, ape_bytes, &model_bytes) ||
        !install_zero_model(model_bytes)) return 0;
    ds4_gpu_tensor *x = input_tensor(in_dim);
    ds4_gpu_tensor *out0 = zero_tensor(width, sizeof(float));
    ds4_gpu_tensor *out1 = zero_tensor(width, sizeof(float));
    ds4_gpu_tensor *state0 = zero_tensor(
        (uint64_t)state_rows * width, sizeof(float));
    ds4_gpu_tensor *state1 = zero_tensor(
        (uint64_t)state_rows * width, sizeof(float));
    int ok = 0;
    const int launched = x && out0 && out1 && state0 && state1
        ? ds4_gpu_matmul_f16_pair_compressor_store_tensor(
              out0, out1, state0, state1,
              model_storage, model_bytes, 0u, one_bytes, weights_bytes,
              0u, in_dim, width, x, ratio, 7u)
        : -1;
    if (launched != 1 || !ds4_gpu_synchronize()) {
        fprintf(stderr,
                "error: fused F16 compressor-pair/state decode launch failed\n");
        goto cleanup;
    }
    ok = verify_zero_tensor(out0, width, "f16-pair-state-out0") &&
         verify_zero_tensor(out1, width, "f16-pair-state-out1") &&
         verify_zero_tensor(state0, (uint64_t)state_rows * width,
                            "f16-pair-state-state0") &&
         verify_zero_tensor(state1, (uint64_t)state_rows * width,
                            "f16-pair-state-state1");
cleanup:
    ds4_gpu_tensor_free(state1);
    ds4_gpu_tensor_free(state0);
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
    (void)unsetenv("DS4_CUDA_DISABLE_COMPRESSOR_PAIR_STATE_STORE");
    (void)unsetenv("DS4_CUDA_MOE_Q32_DECODE_GRAPH");
    (void)setenv("DS4_CUDA_NO_MOE_Q32_DECODE_GRAPH", "1", 1);
    (void)unsetenv("DS4_CUDA_MOE_Q32_DECODE_FUSED_LOWREG");
    (void)unsetenv("DS4_CUDA_NO_MOE_Q32_DECODE_FUSED_LOWREG");
    (void)unsetenv("DS4_CUDA_MOE_Q4_32_DECODE_MAPPING");
    (void)unsetenv("DS4_CUDA_MOE_DIRECT_NATIVE_Q8");
    (void)unsetenv("DS4_CUDA_NO_MOE_DIRECT_NATIVE_Q8");
    /* Preserve mapping-audit switches supplied by evidence runners.  They
     * only add teardown counters and cannot alter kernel selection. */
    (void)unsetenv("DS4_CUDA_MOE_Q3A4_DECODE_MAPPING");
    (void)unsetenv("DS4_CUDA_NO_MOE_Q3A4_DECODE_MAPPING");
    (void)unsetenv("DS4_CUDA_MOE_Q3A4_DECODE_KSPLIT");
    (void)unsetenv("DS4_CUDA_MOE_Q4_32_DOWN_DECODE_MAPPING");

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
        case SCENARIO_Q4_32_GATE_UP:
            ok = run_routed_gate_up(0, 0, 0u, 0u, 1u, 0); break;
        case SCENARIO_Q4_32_GATE_UP_HWARP16:
            ok = run_routed_gate_up_q4_mapping(1u, 0u, 0); break;
        case SCENARIO_Q4_32_GATE_UP_TILE32_DP4A:
            ok = run_routed_gate_up_q4_mapping(2u, 0u, 0); break;
        case SCENARIO_Q4_32_GATE_UP_TILE32_MMA:
            ok = run_routed_gate_up_q4_mapping(3u, 0u, 0); break;
        case SCENARIO_Q4_32_GATE_UP_TILE32_MMA_PREFETCH1:
            ok = run_routed_gate_up_q4_mapping(3u, 1u, 0); break;
        case SCENARIO_Q4_32_GATE_UP_TILE32_MMA_PREFETCH2:
            ok = run_routed_gate_up_q4_mapping(3u, 2u, 0); break;
        case SCENARIO_Q4_32_GATE_UP_HWARP16_AB:
            ok = run_routed_gate_up_q4_mapping(1u, 0u, 1); break;
        case SCENARIO_Q4_32_GATE_UP_TILE32_DP4A_AB:
            ok = run_routed_gate_up_q4_mapping(2u, 0u, 1); break;
        case SCENARIO_Q4_32_GATE_UP_TILE32_MMA_AB:
            ok = run_routed_gate_up_q4_mapping(3u, 0u, 1); break;
        case SCENARIO_Q4_32_GATE_UP_TILE32_MMA_PREFETCH1_AB:
            ok = run_routed_gate_up_q4_mapping(3u, 1u, 1); break;
        case SCENARIO_Q4_32_GATE_UP_TILE32_MMA_PREFETCH2_AB:
            ok = run_routed_gate_up_q4_mapping(3u, 2u, 1); break;
        case SCENARIO_Q3A4_GATE_UP:
            ok = run_routed_gate_up(1, 0, 0u, 0u, 1u, 0); break;
        case SCENARIO_Q4_32_GATE_UP_SPLIT:
            ok = run_routed_gate_up(0, 1, 0u, 0u, 1u, 0); break;
        case SCENARIO_Q3A4_GATE_UP_SPLIT:
            ok = run_routed_gate_up(1, 1, 0u, 0u, 1u, 0); break;
        case SCENARIO_Q4_32_GATE_UP_AB:
            ok = run_routed_gate_up(0, 0, 0u, 0u, 1u, 1); break;
        case SCENARIO_Q3A4_GATE_UP_AB:
            ok = run_routed_gate_up(1, 0, 0u, 0u, 1u, 1); break;
        case SCENARIO_Q3A4_GATE_UP_FUSED_U1:
            ok = run_routed_gate_up(1, 0, 1u, 0u, 1u, 0); break;
        case SCENARIO_Q3A4_GATE_UP_FUSED_U2:
            ok = run_routed_gate_up(1, 0, 2u, 0u, 1u, 0); break;
        case SCENARIO_Q3A4_GATE_UP_FUSED_U4:
            ok = run_routed_gate_up(1, 0, 4u, 0u, 1u, 0); break;
        case SCENARIO_Q3A4_GATE_UP_FUSED_U1_AB:
            ok = run_routed_gate_up(1, 0, 1u, 0u, 1u, 1); break;
        case SCENARIO_Q3A4_GATE_UP_FUSED_U2_AB:
            ok = run_routed_gate_up(1, 0, 2u, 0u, 1u, 1); break;
        case SCENARIO_Q3A4_GATE_UP_FUSED_U4_AB:
            ok = run_routed_gate_up(1, 0, 4u, 0u, 1u, 1); break;
        case SCENARIO_Q3A4_GATE_UP_HWARP16:
            ok = run_routed_gate_up(1, 0, 0u, 1u, 1u, 0); break;
        case SCENARIO_Q3A4_GATE_UP_TILE32:
            ok = run_routed_gate_up(1, 0, 0u, 2u, 1u, 0); break;
        case SCENARIO_Q3A4_GATE_UP_TILE32_DP4A:
            ok = run_routed_gate_up(1, 0, 0u, 3u, 1u, 0); break;
        case SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K2:
            ok = run_routed_gate_up(1, 0, 0u, 3u, 2u, 0); break;
        case SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4:
            ok = run_routed_gate_up(1, 0, 0u, 3u, 4u, 0); break;
        case SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4_PREFETCH1:
            ok = run_routed_gate_up_candidate(
                1, 0, 0u, 0u, 0u, 3u, 4u, 1u, 0); break;
        case SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4_PREFETCH2:
            ok = run_routed_gate_up_candidate(
                1, 0, 0u, 0u, 0u, 3u, 4u, 2u, 0); break;
        case SCENARIO_Q3A4_GATE_UP_HWARP16_AB:
            ok = run_routed_gate_up(1, 0, 0u, 1u, 1u, 1); break;
        case SCENARIO_Q3A4_GATE_UP_TILE32_AB:
            ok = run_routed_gate_up(1, 0, 0u, 2u, 1u, 1); break;
        case SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_AB:
            ok = run_routed_gate_up(1, 0, 0u, 3u, 1u, 1); break;
        case SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K2_AB:
            ok = run_routed_gate_up(1, 0, 0u, 3u, 2u, 1); break;
        case SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4_AB:
            ok = run_routed_gate_up(1, 0, 0u, 3u, 4u, 1); break;
        case SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4_PREFETCH1_AB:
            ok = run_routed_gate_up_candidate(
                1, 0, 0u, 0u, 0u, 3u, 4u, 1u, 1); break;
        case SCENARIO_Q3A4_GATE_UP_TILE32_DP4A_K4_PREFETCH2_AB:
            ok = run_routed_gate_up_candidate(
                1, 0, 0u, 0u, 0u, 3u, 4u, 2u, 1); break;
        case SCENARIO_Q4_32_GATE_UP_FUSED_U1:
            ok = run_routed_gate_up(0, 0, 1u, 0u, 1u, 0); break;
        case SCENARIO_Q4_32_GATE_UP_FUSED_U2:
            ok = run_routed_gate_up(0, 0, 2u, 0u, 1u, 0); break;
        case SCENARIO_Q4_32_GATE_UP_FUSED_U4:
            ok = run_routed_gate_up(0, 0, 4u, 0u, 1u, 0); break;
        case SCENARIO_Q4_32_GATE_UP_FUSED_U1_AB:
            ok = run_routed_gate_up(0, 0, 1u, 0u, 1u, 1); break;
        case SCENARIO_Q4_32_GATE_UP_FUSED_U2_AB:
            ok = run_routed_gate_up(0, 0, 2u, 0u, 1u, 1); break;
        case SCENARIO_Q4_32_GATE_UP_FUSED_U4_AB:
            ok = run_routed_gate_up(0, 0, 4u, 0u, 1u, 1); break;
        case SCENARIO_Q4_32_DOWN_SLOTS:
            ok = run_q4_32_down_candidate(0, 0u, 0u, 0); break;
        case SCENARIO_Q4_32_DOWN_SLOTS_TILE32:
            ok = run_q4_32_down_candidate(0, 1u, 0u, 0); break;
        case SCENARIO_Q4_32_DOWN_SLOTS_TILE32_PREFETCH1:
            ok = run_q4_32_down_candidate(0, 1u, 1u, 0); break;
        case SCENARIO_Q4_32_DOWN_SLOTS_TILE32_PREFETCH2:
            ok = run_q4_32_down_candidate(0, 1u, 2u, 0); break;
        case SCENARIO_Q4_32_DOWN_SLOTS_TILE32_AB:
            ok = run_q4_32_down_candidate(0, 1u, 0u, 1); break;
        case SCENARIO_Q4_32_DOWN_SLOTS_TILE32_PREFETCH1_AB:
            ok = run_q4_32_down_candidate(0, 1u, 1u, 1); break;
        case SCENARIO_Q4_32_DOWN_SLOTS_TILE32_PREFETCH2_AB:
            ok = run_q4_32_down_candidate(0, 1u, 2u, 1); break;
        case SCENARIO_Q4_32_DOWN_PACKED:
            ok = run_q4_32_down_candidate(1, 0u, 0u, 0); break;
        case SCENARIO_Q4_32_DOWN_PACKED_TILE32:
            ok = run_q4_32_down_candidate(1, 1u, 0u, 0); break;
        case SCENARIO_Q4_32_DOWN_PACKED_TILE32_PREFETCH1:
            ok = run_q4_32_down_candidate(1, 1u, 1u, 0); break;
        case SCENARIO_Q4_32_DOWN_PACKED_TILE32_PREFETCH2:
            ok = run_q4_32_down_candidate(1, 1u, 2u, 0); break;
        case SCENARIO_Q4_32_DOWN_PACKED_TILE32_AB:
            ok = run_q4_32_down_candidate(1, 1u, 0u, 1); break;
        case SCENARIO_Q4_32_DOWN_PACKED_TILE32_PREFETCH1_AB:
            ok = run_q4_32_down_candidate(1, 1u, 1u, 1); break;
        case SCENARIO_Q4_32_DOWN_PACKED_TILE32_PREFETCH2_AB:
            ok = run_q4_32_down_candidate(1, 1u, 2u, 1); break;
        case SCENARIO_Q8_SINGLE_T32: ok = run_q8_single(1024u, 32768u); break;
        case SCENARIO_Q8_PAIR_2048: ok = run_q8_pair(2048u); break;
        case SCENARIO_Q8_PAIR_1024: ok = run_q8_pair(1024u); break;
        case SCENARIO_Q8_KSLICE_T256: ok = run_q8_kslice(); break;
        case SCENARIO_Q8_GROUPED_A_HALF: ok = run_q8_grouped_a(); break;
        case SCENARIO_Q8_SHARED_MID: ok = run_q8_shared_mid(); break;
        case SCENARIO_Q8_NATIVE_QUANTIZE:
            ok = run_routed_gate_up(0, 0, 0u, 3u, 1u, 0); break;
        case SCENARIO_F16_PAIR_256: ok = run_f16_pair(256u); break;
        case SCENARIO_F16_PAIR_512: ok = run_f16_pair(512u); break;
        case SCENARIO_F16_PAIR_1024: ok = run_f16_pair(1024u); break;
        case SCENARIO_F16_PAIR_STATE_256:
            ok = run_f16_pair_state(256u); break;
        case SCENARIO_F16_PAIR_STATE_512:
            ok = run_f16_pair_state(512u); break;
        case SCENARIO_F16_PAIR_STATE_1024:
            ok = run_f16_pair_state(1024u); break;
    }
    ds4_gpu_cleanup();
    free(model_storage);
    model_storage = NULL;
    if (ok) printf("harness_status=ok\n");
    return ok ? 0 : 1;
}
