#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    GUARD_BYTES = 256u,
    N_TOKENS = 128u,
    N_SELECTED = 1u,
    N_TOTAL_EXPERT = 16u,
    N_RESIDENT_EXPERT = 8u,
    IN_DIM = 4096u,
    MID_DIM = 2048u,
    QK_K = 256u,
    Q4_K_BYTES = 144u,
    Q8_K_BYTES = 292u
};

static const uint32_t k_populations[N_RESIDENT_EXPERT] = {
    1u, 3u, 4u, 7u, 8u, 9u, 15u, 16u
};
static const uint32_t k_gate_widths[] = {256u, 384u, 512u};
static const uint32_t k_down_widths[] = {256u, 384u, 512u, 640u};
static const uint32_t k_out_dims[] = {504u, 520u, 4096u};
static const uint32_t k_row_spans[] = {512u, 1024u, 2048u};

static unsigned char *g_idle_model_map;
static const uint64_t g_idle_model_bytes = 4096u;

typedef struct {
    const char *name;
    ds4_gpu_tensor *base;
    ds4_gpu_tensor *view;
    uint64_t payload_bytes;
    uint32_t guard_seed;
    float sentinel;
} guarded_tensor;

typedef struct {
    unsigned char *out;
    unsigned char *gate;
    unsigned char *up;
    unsigned char *mid;
    unsigned char *down;
    uint64_t out_bytes;
    uint64_t mid_bytes;
    uint64_t down_bytes;
} run_snapshot;

typedef struct {
    uint32_t out_dim;
    uint64_t gate_row_bytes;
    uint64_t gate_expert_bytes;
    uint64_t down_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t model_bytes;
    uint64_t pair_count;
    uint64_t x_count;
    uint64_t mid_count;
    uint64_t out_count;
    uint64_t down_storage_bytes;
    unsigned char *model;
    float *x_host;
    int32_t *selected_host;
    float *weights_host;
    ds4_gpu_tensor *x;
    ds4_gpu_tensor *selected;
    ds4_gpu_tensor *weights;
    guarded_tensor out;
    guarded_tensor gate;
    guarded_tensor up;
    guarded_tensor mid;
    guarded_tensor down;
} test_fixture;

static int retire_temporary_model_map(void) {
    return g_idle_model_map &&
           ds4_gpu_set_model_map(g_idle_model_map, g_idle_model_bytes);
}

static uint8_t guard_byte(uint32_t seed, uint32_t side, uint32_t offset) {
    uint32_t x = seed ^ (side ? 0xa511e9b3u : 0x63d83595u);
    x ^= offset * 0x9e3779b9u;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return (uint8_t)(x & 0xffu);
}

static void fill_guard_bytes(uint8_t *dst, uint32_t seed, uint32_t side) {
    for (uint32_t i = 0; i < GUARD_BYTES; i++)
        dst[i] = guard_byte(seed, side, i);
}

static int guarded_tensor_alloc(guarded_tensor *tensor,
                                const char *name,
                                uint64_t payload_bytes,
                                uint32_t seed,
                                float sentinel) {
    if (!tensor || !name || payload_bytes == 0u ||
        (payload_bytes & (sizeof(float) - 1u)) != 0u ||
        payload_bytes > UINT64_MAX - 2u * GUARD_BYTES) {
        return 0;
    }
    memset(tensor, 0, sizeof(*tensor));
    tensor->name = name;
    tensor->payload_bytes = payload_bytes;
    tensor->guard_seed = seed;
    tensor->sentinel = sentinel;
    tensor->base = ds4_gpu_tensor_alloc(
        payload_bytes + 2u * (uint64_t)GUARD_BYTES);
    tensor->view = tensor->base
        ? ds4_gpu_tensor_view(tensor->base, GUARD_BYTES, payload_bytes)
        : NULL;
    if (!tensor->base || !tensor->view) {
        ds4_gpu_tensor_free(tensor->view);
        ds4_gpu_tensor_free(tensor->base);
        tensor->view = NULL;
        tensor->base = NULL;
        return 0;
    }
    return 1;
}

static void guarded_tensor_free(guarded_tensor *tensor) {
    if (!tensor) return;
    ds4_gpu_tensor_free(tensor->view);
    ds4_gpu_tensor_free(tensor->base);
    memset(tensor, 0, sizeof(*tensor));
}

static int guarded_tensor_prepare(const guarded_tensor *tensor) {
    uint8_t prefix[GUARD_BYTES];
    uint8_t suffix[GUARD_BYTES];
    fill_guard_bytes(prefix, tensor->guard_seed, 0u);
    fill_guard_bytes(suffix, tensor->guard_seed, 1u);
    if (!ds4_gpu_tensor_fill_f32(
            tensor->view, tensor->sentinel,
            tensor->payload_bytes / sizeof(float)) ||
        !ds4_gpu_tensor_write(tensor->base, 0u, prefix, sizeof(prefix)) ||
        !ds4_gpu_tensor_write(
            tensor->base, GUARD_BYTES + tensor->payload_bytes,
            suffix, sizeof(suffix))) {
        fprintf(stderr, "error: could not initialize guarded tensor %s\n",
                tensor->name);
        return 0;
    }
    return 1;
}

static int guarded_tensor_check(const guarded_tensor *tensor,
                                uint32_t gate_width,
                                uint32_t down_width,
                                uint32_t write_aux) {
    uint8_t actual[GUARD_BYTES];
    uint8_t expected[GUARD_BYTES];
    for (uint32_t side = 0; side < 2u; side++) {
        const uint64_t offset = side
            ? GUARD_BYTES + tensor->payload_bytes
            : 0u;
        fill_guard_bytes(expected, tensor->guard_seed, side);
        if (!ds4_gpu_tensor_read(
                tensor->base, offset, actual, sizeof(actual))) {
            fprintf(stderr,
                    "error: could not read %s %s canary "
                    "(gate=%u down=%u write_aux=%u)\n",
                    tensor->name, side ? "suffix" : "prefix",
                    gate_width, down_width, write_aux);
            return 0;
        }
        if (memcmp(actual, expected, sizeof(actual)) != 0) {
            uint32_t first = 0u;
            while (first < GUARD_BYTES && actual[first] == expected[first])
                first++;
            fprintf(stderr,
                    "error: %s %s canary changed at byte %u: "
                    "expected=0x%02x actual=0x%02x "
                    "(gate=%u down=%u write_aux=%u)\n",
                    tensor->name, side ? "suffix" : "prefix", first,
                    (unsigned)expected[first], (unsigned)actual[first],
                    gate_width, down_width, write_aux);
            return 0;
        }
    }
    return 1;
}

static void q4_k_block_fill(unsigned char *block,
                            uint32_t expert,
                            uint32_t row,
                            uint32_t superblock,
                            uint32_t matrix) {
    const uint16_t d = (uint16_t)(0x1c00u +
        ((expert + row + superblock + matrix) & 3u) * 0x0100u);
    const uint16_t dmin = (uint16_t)(0x1400u +
        ((expert * 3u + row + superblock * 5u + matrix) & 3u) * 0x0100u);
    memcpy(block + 0u, &d, sizeof(d));
    memcpy(block + 2u, &dmin, sizeof(dmin));
    for (uint32_t i = 0; i < 12u; i++) {
        const uint32_t v = expert * 29u + row * 11u + superblock * 7u +
                           i * 13u + matrix * 17u;
        block[4u + i] = (unsigned char)(1u + (v % 255u));
    }
    for (uint32_t i = 0; i < 128u; i++) {
        const uint32_t v = expert * 17u + row * 23u + superblock * 31u +
                           i * 5u + matrix * 19u;
        block[16u + i] = (unsigned char)(1u + (v % 255u));
    }
}

static int snapshot_alloc(run_snapshot *snapshot, const test_fixture *fixture) {
    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->out_bytes = fixture->out.payload_bytes;
    snapshot->mid_bytes = fixture->mid.payload_bytes;
    snapshot->down_bytes = fixture->down.payload_bytes;
    snapshot->out = (unsigned char *)malloc((size_t)snapshot->out_bytes);
    snapshot->gate = (unsigned char *)malloc((size_t)snapshot->mid_bytes);
    snapshot->up = (unsigned char *)malloc((size_t)snapshot->mid_bytes);
    snapshot->mid = (unsigned char *)malloc((size_t)snapshot->mid_bytes);
    snapshot->down = (unsigned char *)malloc((size_t)snapshot->down_bytes);
    if (!snapshot->out || !snapshot->gate || !snapshot->up ||
        !snapshot->mid || !snapshot->down) {
        free(snapshot->down);
        free(snapshot->mid);
        free(snapshot->up);
        free(snapshot->gate);
        free(snapshot->out);
        memset(snapshot, 0, sizeof(*snapshot));
        return 0;
    }
    return 1;
}

static void snapshot_free(run_snapshot *snapshot) {
    if (!snapshot) return;
    free(snapshot->down);
    free(snapshot->mid);
    free(snapshot->up);
    free(snapshot->gate);
    free(snapshot->out);
    memset(snapshot, 0, sizeof(*snapshot));
}

static int snapshot_read(run_snapshot *snapshot, const test_fixture *fixture) {
    return ds4_gpu_tensor_read(
               fixture->out.view, 0u, snapshot->out, snapshot->out_bytes) &&
           ds4_gpu_tensor_read(
               fixture->gate.view, 0u, snapshot->gate, snapshot->mid_bytes) &&
           ds4_gpu_tensor_read(
               fixture->up.view, 0u, snapshot->up, snapshot->mid_bytes) &&
           ds4_gpu_tensor_read(
               fixture->mid.view, 0u, snapshot->mid, snapshot->mid_bytes) &&
           ds4_gpu_tensor_read(
               fixture->down.view, 0u, snapshot->down, snapshot->down_bytes);
}

static int compare_bytes(const char *name,
                         const unsigned char *reference,
                         const unsigned char *candidate,
                         uint64_t bytes,
                         uint32_t out_dim,
                         uint32_t write_aux,
                         uint32_t gate_width,
                         uint32_t down_width) {
    if (memcmp(reference, candidate, (size_t)bytes) == 0) return 1;
    uint64_t first = 0u;
    while (first < bytes && reference[first] == candidate[first]) first++;
    uint32_t reference_word = 0u;
    uint32_t candidate_word = 0u;
    const uint64_t word_offset = first & ~3ull;
    if (word_offset + sizeof(uint32_t) <= bytes) {
        memcpy(&reference_word, reference + word_offset, sizeof(reference_word));
        memcpy(&candidate_word, candidate + word_offset, sizeof(candidate_word));
    }
    fprintf(stderr,
            "error: %s mismatch at byte %llu (word %llu): "
            "reference=0x%08x candidate=0x%08x "
            "dim=%u write_aux=%u gate=%u down=%u\n",
            name, (unsigned long long)first,
            (unsigned long long)(word_offset / sizeof(uint32_t)),
            reference_word, candidate_word,
            out_dim, write_aux, gate_width, down_width);
    return 0;
}

static int snapshot_compare(const run_snapshot *reference,
                            const run_snapshot *candidate,
                            uint32_t out_dim,
                            uint32_t write_aux,
                            uint32_t gate_width,
                            uint32_t down_width) {
    return compare_bytes("out", reference->out, candidate->out,
                         reference->out_bytes, out_dim, write_aux,
                         gate_width, down_width) &&
           compare_bytes("gate", reference->gate, candidate->gate,
                         reference->mid_bytes, out_dim, write_aux,
                         gate_width, down_width) &&
           compare_bytes("up", reference->up, candidate->up,
                         reference->mid_bytes, out_dim, write_aux,
                         gate_width, down_width) &&
           compare_bytes("mid", reference->mid, candidate->mid,
                         reference->mid_bytes, out_dim, write_aux,
                         gate_width, down_width) &&
           compare_bytes("down", reference->down, candidate->down,
                         reference->down_bytes, out_dim, write_aux,
                         gate_width, down_width);
}

static int verify_aux_untouched(const run_snapshot *snapshot,
                                const test_fixture *fixture) {
    const uint32_t expected = 0xc2f68000u; /* -123.25f */
    /* The production path aliases the beginning of gate as its packed Q8_K
     * mid scratch after gate/up.  With auxiliary writes disabled, only the
     * remainder of gate and all of up must retain their sentinels. */
    const uint64_t gate_scratch_bytes = fixture->pair_count *
        (MID_DIM / QK_K) * Q8_K_BYTES;
    for (uint32_t which = 0; which < 2u; which++) {
        const unsigned char *values = which ? snapshot->up : snapshot->gate;
        const char *name = which ? "up" : "gate";
        const uint64_t first_offset = which ? 0u : gate_scratch_bytes;
        for (uint64_t offset = first_offset; offset < snapshot->mid_bytes;
             offset += sizeof(uint32_t)) {
            uint32_t actual = 0u;
            memcpy(&actual, values + offset, sizeof(actual));
            if (actual != expected) {
                fprintf(stderr,
                        "error: write_aux=0 changed %s word %llu: "
                        "expected=0x%08x actual=0x%08x dim=%u\n",
                        name,
                        (unsigned long long)(offset / sizeof(uint32_t)),
                        expected, actual, fixture->out_dim);
                return 0;
            }
        }
    }
    return 1;
}

static int verify_finite_region(const char *name,
                                const unsigned char *bytes,
                                uint64_t count) {
    for (uint64_t index = 0u; index < count; index++) {
        float value = 0.0f;
        memcpy(&value, bytes + index * sizeof(float), sizeof(value));
        if (!isfinite(value)) {
            fprintf(stderr,
                    "error: canonical %s contains non-finite value at %llu\n",
                    name, (unsigned long long)index);
            return 0;
        }
    }
    return 1;
}

static int verify_changed_value(const char *name,
                                const unsigned char *bytes,
                                uint64_t count,
                                uint64_t index,
                                float sentinel,
                                uint32_t token,
                                uint32_t row) {
    float value = 0.0f;
    if (index >= count) {
        fprintf(stderr,
                "error: canonical %s probe is out of range token=%u row=%u\n",
                name, token, row);
        return 0;
    }
    memcpy(&value, bytes + index * sizeof(float), sizeof(value));
    if (!isfinite(value) || value == sentinel) {
        fprintf(stderr,
                "error: canonical %s did no finite work at token=%u row=%u "
                "value=%g\n",
                name, token, row, (double)value);
        return 0;
    }
    return 1;
}

static int verify_snapshot_nonvacuous(const run_snapshot *snapshot,
                                      const test_fixture *fixture,
                                      uint32_t write_aux) {
    const uint32_t expected_resident_pairs = 63u;
    const uint64_t gate_scratch_bytes = fixture->pair_count *
        (MID_DIM / QK_K) * Q8_K_BYTES;
    if (!verify_finite_region("out", snapshot->out, fixture->out_count) ||
        !verify_finite_region("mid", snapshot->mid, fixture->mid_count) ||
        !verify_finite_region(
            "down", snapshot->down,
            (uint64_t)expected_resident_pairs * fixture->out_dim) ||
        (write_aux && !verify_finite_region(
            "up", snapshot->up, fixture->mid_count))) {
        return 0;
    }

    uint32_t resident_pairs = 0u;
    for (uint32_t expert = 0u; expert < N_RESIDENT_EXPERT; expert++) {
        for (uint32_t local = 0u; local < k_populations[expert]; local++) {
            const uint32_t token = resident_pairs++;
            const uint32_t mid_rows[2] = {0u, MID_DIM - 1u};
            const uint32_t out_rows[2] = {0u, fixture->out_dim - 1u};
            for (uint32_t edge = 0u; edge < 2u; edge++) {
                const uint32_t mid_row = mid_rows[edge];
                const uint32_t out_row = out_rows[edge];
                if (!verify_changed_value(
                        "mid", snapshot->mid, fixture->mid_count,
                        (uint64_t)token * MID_DIM + mid_row,
                        fixture->mid.sentinel, token, mid_row) ||
                    !verify_changed_value(
                        "down", snapshot->down, fixture->out_count,
                        (uint64_t)token * fixture->out_dim + out_row,
                        fixture->down.sentinel, token, out_row) ||
                    !verify_changed_value(
                        "out", snapshot->out, fixture->out_count,
                        (uint64_t)token * fixture->out_dim + out_row,
                        fixture->out.sentinel, token, out_row)) {
                    return 0;
                }
                if (write_aux &&
                    !verify_changed_value(
                         "up", snapshot->up, fixture->mid_count,
                         (uint64_t)token * MID_DIM + mid_row,
                         fixture->up.sentinel, token, mid_row)) {
                    return 0;
                }
                const uint64_t gate_index =
                    (uint64_t)token * MID_DIM + mid_row;
                if (write_aux && gate_index * sizeof(float) >= gate_scratch_bytes &&
                    !verify_changed_value(
                        "gate", snapshot->gate, fixture->mid_count,
                        gate_index, fixture->gate.sentinel,
                        token, mid_row)) {
                    return 0;
                }
            }
        }
    }
    printf("wide-cta-nonvacuous: dim=%u write_aux=%u "
           "resident_pairs=%u populations=1|3|4|7|8|9|15|16\n",
           fixture->out_dim, write_aux, resident_pairs);
    return resident_pairs == expected_resident_pairs;
}

static int fixture_outputs_prepare(const test_fixture *fixture) {
    return guarded_tensor_prepare(&fixture->out) &&
           guarded_tensor_prepare(&fixture->gate) &&
           guarded_tensor_prepare(&fixture->up) &&
           guarded_tensor_prepare(&fixture->mid) &&
           guarded_tensor_prepare(&fixture->down);
}

static int fixture_canaries_check(const test_fixture *fixture,
                                  uint32_t gate_width,
                                  uint32_t down_width,
                                  uint32_t write_aux) {
    return guarded_tensor_check(
               &fixture->out, gate_width, down_width, write_aux) &&
           guarded_tensor_check(
               &fixture->gate, gate_width, down_width, write_aux) &&
           guarded_tensor_check(
               &fixture->up, gate_width, down_width, write_aux) &&
           guarded_tensor_check(
               &fixture->mid, gate_width, down_width, write_aux) &&
           guarded_tensor_check(
               &fixture->down, gate_width, down_width, write_aux);
}

static int set_width_environment(uint32_t gate_width,
                                 uint32_t down_width,
                                 uint32_t row_span,
                                 uint32_t write_aux) {
    char gate_value[16];
    char down_value[16];
    snprintf(gate_value, sizeof(gate_value), "%u", gate_width);
    snprintf(down_value, sizeof(down_value), "%u", down_width);
    if (unsetenv("DS4_CUDA_MOE_GATE_ROW1024") != 0 ||
        unsetenv("DS4_CUDA_MOE_GATE_ROW2048") != 0 ||
        unsetenv("DS4_CUDA_MOE_DOWN_ROW1024") != 0 ||
        unsetenv("DS4_CUDA_MOE_DOWN_ROW2048") != 0) {
        return 0;
    }
    if (row_span == 1024u) {
        if (setenv("DS4_CUDA_MOE_GATE_ROW1024", "1", 1) != 0 ||
            setenv("DS4_CUDA_MOE_DOWN_ROW1024", "1", 1) != 0) {
            return 0;
        }
    } else if (row_span == 2048u) {
        if (setenv("DS4_CUDA_MOE_GATE_ROW2048", "1", 1) != 0 ||
            setenv("DS4_CUDA_MOE_DOWN_ROW2048", "1", 1) != 0) {
            return 0;
        }
    } else if (row_span != 512u) {
        return 0;
    }
    return setenv("DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75", "1", 1) == 0 &&
           setenv("DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75", "1", 1) == 0 &&
           setenv("DS4_CUDA_MOE_Q4_GATE_SCALAR_CTA_SM75",
                  gate_value, 1) == 0 &&
           setenv("DS4_CUDA_MOE_Q4_DOWN_SCALAR_CTA_SM75",
                  down_value, 1) == 0 &&
           (write_aux
                ? setenv("DS4_CUDA_MOE_WRITE_GATE_UP", "1", 1) == 0
                : unsetenv("DS4_CUDA_MOE_WRITE_GATE_UP") == 0);
}

static int run_width(test_fixture *fixture,
                     uint32_t gate_width,
                     uint32_t down_width,
                     uint32_t row_span,
                     uint32_t write_aux,
                     run_snapshot *snapshot) {
    if (!set_width_environment(
            gate_width, down_width, row_span, write_aux)) {
        fprintf(stderr, "error: could not configure width environment\n");
        return 0;
    }
    /* The owned production entry point filters these tensors in-place. Each
     * A/B run must therefore restore the original partner assignments. */
    if (!ds4_gpu_tensor_write(
            fixture->selected, 0u, fixture->selected_host,
            fixture->pair_count * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(
            fixture->weights, 0u, fixture->weights_host,
            fixture->pair_count * sizeof(float)) ||
        !fixture_outputs_prepare(fixture)) {
        fprintf(stderr,
                "error: input/output reset failed dim=%u span=%u "
                "gate=%u down=%u\n",
                fixture->out_dim, row_span, gate_width, down_width);
        return 0;
    }

    bool mid_is_f16 = false;
    if (!ds4_gpu_routed_moe_batch_owned_tensor(
            fixture->out.view,
            fixture->gate.view,
            fixture->up.view,
            fixture->mid.view,
            fixture->down.view,
            fixture->model,
            fixture->model_bytes,
            fixture->gate_offset,
            fixture->up_offset,
            fixture->down_offset,
            12u,
            12u,
            fixture->gate_expert_bytes,
            fixture->gate_row_bytes,
            fixture->down_expert_bytes,
            fixture->down_row_bytes,
            IN_DIM,
            MID_DIM,
            fixture->out_dim,
            fixture->selected,
            fixture->weights,
            N_TOTAL_EXPERT,
            N_SELECTED,
            0u,
            N_RESIDENT_EXPERT,
            10.0f,
            fixture->x,
            3u,
            N_TOKENS,
            0u,
            &mid_is_f16) ||
        mid_is_f16 || !ds4_gpu_synchronize()) {
        fprintf(stderr,
                "error: production routed-MoE launch failed "
                "dim=%u span=%u write_aux=%u gate=%u down=%u\n",
                fixture->out_dim, row_span, write_aux,
                gate_width, down_width);
        return 0;
    }
    if (!fixture_canaries_check(
            fixture, gate_width, down_width, write_aux) ||
        !snapshot_read(snapshot, fixture)) {
        fprintf(stderr,
                "error: result collection failed "
                "dim=%u span=%u write_aux=%u gate=%u down=%u\n",
                fixture->out_dim, row_span, write_aux,
                gate_width, down_width);
        return 0;
    }
    if (!write_aux && !verify_aux_untouched(snapshot, fixture)) return 0;
    return 1;
}

static void normalize_environment(void) {
    static const char *const unset_names[] = {
        "DS4_CUDA_MOE_NO_Q4_MMA",
        "DS4_CUDA_MOE_NO_Q4_MMA_TILE16",
        "DS4_CUDA_MOE_NO_Q4_MMA_TILE16_SM75",
        "DS4_CUDA_MOE_NO_Q4_SORTED",
        "DS4_CUDA_MOE_Q4_GATE_TILE16_SM75",
        "DS4_CUDA_MOE_NO_Q4_GATE_TILE16_SM75",
        "DS4_CUDA_MOE_Q4_GATE_STAGE4_SM75",
        "DS4_CUDA_MOE_NO_EXPERT_TILES",
        "DS4_CUDA_MOE_NO_OWNED_SPARSE_BUFFERS",
        "DS4_CUDA_MOE_TILE4",
        "DS4_CUDA_MOE_NO_MIXED_TAIL_TILES",
        "DS4_CUDA_MOE_MIXED_TAIL_TILES",
        "DS4_CUDA_MOE_NO_Q4_DOWN_ROWSPAN",
        "DS4_CUDA_MOE_ATOMIC_DOWN",
        "DS4_CUDA_MOE_NO_ATOMIC_DOWN",
        "DS4_CUDA_MOE_NO_DOWN_TILE16",
        "DS4_CUDA_MOE_NO_DOWN_ROW2048",
        "DS4_CUDA_MOE_NO_DOWN_ROW256",
        "DS4_CUDA_MOE_NO_DOWN_ROW128",
        "DS4_CUDA_MOE_NO_DOWN_ROW64",
        "DS4_CUDA_MOE_NO_GATE_ROW2048",
        "DS4_CUDA_MOE_NO_GATE_ROW256",
        "DS4_CUDA_MOE_NO_GATE_ROW128",
        "DS4_CUDA_MOE_GATE_ROW1024",
        "DS4_CUDA_MOE_GATE_ROW2048",
        "DS4_CUDA_MOE_GATE_ROW256",
        "DS4_CUDA_MOE_GATE_ROW128",
        "DS4_CUDA_MOE_PROFILE",
        "DS4_CUDA_MOE_DOWN_ROW512",
        "DS4_CUDA_MOE_DOWN_ROW1024",
        "DS4_CUDA_MOE_DOWN_ROW2048",
        "DS4_CUDA_MOE_DOWN_ROW256",
        "DS4_CUDA_MOE_DOWN_ROW128",
        "DS4_CUDA_MOE_DOWN_ROW64",
        "DS4_CUDA_MOE_WRITE_GATE_UP",
        "DS4_CUDA_MOE_Q4_GATE_SCALAR_SM75",
        "DS4_CUDA_MOE_Q4_DOWN_SCALAR_SM75",
        "DS4_CUDA_MOE_Q4_GATE_SCALAR_CTA_SM75",
        "DS4_CUDA_MOE_Q4_DOWN_SCALAR_CTA_SM75",
        "DS4_CUDA_MOE_SCALAR_AUDIT"
    };
    for (size_t i = 0; i < sizeof(unset_names) / sizeof(unset_names[0]); i++)
        (void)unsetenv(unset_names[i]);
    (void)setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    (void)setenv("DS4_CUDA_NO_Q8_F16_CACHE", "1", 1);
    (void)setenv("DS4_CUDA_MOE_TILE8", "1", 1);
    (void)setenv("DS4_CUDA_MOE_SCALAR_AUDIT", "1", 1);
}

static int fixture_init(test_fixture *fixture, uint32_t out_dim) {
    memset(fixture, 0, sizeof(*fixture));
    fixture->out_dim = out_dim;
    fixture->gate_row_bytes = (uint64_t)(IN_DIM / QK_K) * Q4_K_BYTES;
    fixture->gate_expert_bytes =
        (uint64_t)MID_DIM * fixture->gate_row_bytes;
    fixture->down_row_bytes = (uint64_t)(MID_DIM / QK_K) * Q4_K_BYTES;
    fixture->down_expert_bytes =
        (uint64_t)out_dim * fixture->down_row_bytes;
    fixture->gate_offset = 0u;
    fixture->up_offset =
        fixture->gate_expert_bytes * N_RESIDENT_EXPERT;
    fixture->down_offset = fixture->up_offset +
        fixture->gate_expert_bytes * N_RESIDENT_EXPERT;
    fixture->model_bytes = fixture->down_offset +
        fixture->down_expert_bytes * N_RESIDENT_EXPERT;
    fixture->pair_count = (uint64_t)N_TOKENS * N_SELECTED;
    fixture->x_count = (uint64_t)N_TOKENS * IN_DIM;
    fixture->mid_count = fixture->pair_count * MID_DIM;
    fixture->out_count = fixture->pair_count * out_dim;
    const uint64_t down_output_bytes =
        fixture->out_count * sizeof(float);
    const uint64_t x_scratch_bytes = fixture->x_count * sizeof(float);
    fixture->down_storage_bytes = down_output_bytes > x_scratch_bytes
        ? down_output_bytes
        : x_scratch_bytes;

    if (fixture->model_bytes > SIZE_MAX ||
        fixture->x_count > SIZE_MAX / sizeof(float) ||
        fixture->pair_count > SIZE_MAX / sizeof(int32_t)) {
        fprintf(stderr, "error: fixture size exceeds host address space\n");
        return 0;
    }
    fixture->model =
        (unsigned char *)malloc((size_t)fixture->model_bytes);
    fixture->x_host =
        (float *)malloc((size_t)fixture->x_count * sizeof(float));
    fixture->selected_host =
        (int32_t *)malloc((size_t)fixture->pair_count * sizeof(int32_t));
    fixture->weights_host =
        (float *)malloc((size_t)fixture->pair_count * sizeof(float));
    fixture->x = ds4_gpu_tensor_alloc(
        fixture->x_count * sizeof(float));
    fixture->selected = ds4_gpu_tensor_alloc(
        fixture->pair_count * sizeof(int32_t));
    fixture->weights = ds4_gpu_tensor_alloc(
        fixture->pair_count * sizeof(float));
    if (!fixture->model || !fixture->x_host || !fixture->selected_host ||
        !fixture->weights_host || !fixture->x || !fixture->selected ||
        !fixture->weights ||
        !guarded_tensor_alloc(
            &fixture->out, "out", fixture->out_count * sizeof(float),
            0x1001u + out_dim, -121.25f) ||
        !guarded_tensor_alloc(
            &fixture->gate, "gate", fixture->mid_count * sizeof(float),
            0x2003u + out_dim, -123.25f) ||
        !guarded_tensor_alloc(
            &fixture->up, "up", fixture->mid_count * sizeof(float),
            0x3007u + out_dim, -123.25f) ||
        !guarded_tensor_alloc(
            &fixture->mid, "mid", fixture->mid_count * sizeof(float),
            0x4009u + out_dim, -125.25f) ||
        !guarded_tensor_alloc(
            &fixture->down, "down", fixture->down_storage_bytes,
            0x500bu + out_dim, -127.25f)) {
        fprintf(stderr, "error: allocation failed for dim=%u fixture\n", out_dim);
        return 0;
    }

    for (uint32_t matrix = 0u; matrix < 2u; matrix++) {
        unsigned char *base = fixture->model +
            (matrix ? fixture->up_offset : fixture->gate_offset);
        for (uint32_t expert = 0u; expert < N_RESIDENT_EXPERT; expert++) {
            for (uint32_t row = 0u; row < MID_DIM; row++) {
                for (uint32_t block = 0u; block < IN_DIM / QK_K; block++) {
                    unsigned char *q4 = base +
                        (uint64_t)expert * fixture->gate_expert_bytes +
                        (uint64_t)row * fixture->gate_row_bytes +
                        (uint64_t)block * Q4_K_BYTES;
                    q4_k_block_fill(q4, expert, row, block, matrix);
                }
            }
        }
    }
    for (uint32_t expert = 0u; expert < N_RESIDENT_EXPERT; expert++) {
        for (uint32_t row = 0u; row < out_dim; row++) {
            for (uint32_t block = 0u; block < MID_DIM / QK_K; block++) {
                unsigned char *q4 = fixture->model + fixture->down_offset +
                    (uint64_t)expert * fixture->down_expert_bytes +
                    (uint64_t)row * fixture->down_row_bytes +
                    (uint64_t)block * Q4_K_BYTES;
                q4_k_block_fill(q4, expert, row, block, 2u);
            }
        }
    }
    for (uint64_t i = 0u; i < fixture->x_count; i++) {
        int value = (int)((i * 23u + (i >> 5u) * 17u) % 257u) - 128;
        if (value == 0) value = 37;
        fixture->x_host[i] = (float)value / 133.0f;
    }
    uint32_t token = 0u;
    for (uint32_t expert = 0u; expert < N_RESIDENT_EXPERT; expert++) {
        for (uint32_t i = 0u; i < k_populations[expert]; i++, token++)
            fixture->selected_host[token] = (int32_t)expert;
    }
    for (; token < N_TOKENS; token++) {
        fixture->selected_host[token] = (int32_t)(
            N_RESIDENT_EXPERT + (token %
                (N_TOTAL_EXPERT - N_RESIDENT_EXPERT)));
    }
    for (uint64_t i = 0u; i < fixture->pair_count; i++)
        fixture->weights_host[i] = (float)((i % 11u) + 1u) / 13.0f;

    if (!ds4_gpu_tensor_write(
            fixture->x, 0u, fixture->x_host,
            fixture->x_count * sizeof(float)) ||
        !ds4_gpu_set_model_map(fixture->model, fixture->model_bytes) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: fixture upload failed for dim=%u\n", out_dim);
        return 0;
    }
    return 1;
}

static void fixture_destroy(test_fixture *fixture) {
    guarded_tensor_free(&fixture->down);
    guarded_tensor_free(&fixture->mid);
    guarded_tensor_free(&fixture->up);
    guarded_tensor_free(&fixture->gate);
    guarded_tensor_free(&fixture->out);
    ds4_gpu_tensor_free(fixture->weights);
    ds4_gpu_tensor_free(fixture->selected);
    ds4_gpu_tensor_free(fixture->x);
    free(fixture->weights_host);
    free(fixture->selected_host);
    free(fixture->x_host);
    free(fixture->model);
    memset(fixture, 0, sizeof(*fixture));
}

static int run_dimension(uint32_t out_dim) {
    test_fixture fixture;
    run_snapshot reference;
    run_snapshot candidate;
    int ok = 0;
    memset(&fixture, 0, sizeof(fixture));
    memset(&reference, 0, sizeof(reference));
    memset(&candidate, 0, sizeof(candidate));
    if (!fixture_init(&fixture, out_dim) ||
        !snapshot_alloc(&reference, &fixture) ||
        !snapshot_alloc(&candidate, &fixture)) {
        goto cleanup;
    }
    const uint64_t checked_values =
        (reference.out_bytes + 3u * reference.mid_bytes +
         reference.down_bytes) / sizeof(float);

    const size_t row_span_count = out_dim == 4096u
        ? sizeof(k_row_spans) / sizeof(k_row_spans[0]) : 1u;
    for (uint32_t write_aux = 0u; write_aux <= 1u; write_aux++) {
        /* Span 512 at CTA 256 is the one canonical output.  Every other
         * segmentation and width is compared with this snapshot so a row
         * coverage bug common to all widths of a wider span cannot pass. */
        if (!run_width(
                &fixture, 256u, 256u, 512u, write_aux, &reference) ||
            !verify_snapshot_nonvacuous(&reference, &fixture, write_aux)) {
            goto cleanup;
        }
        printf("wide-cta-correctness: dim=%u row_span=512 write_aux=%u "
               "gate_cta=256 down_cta=256 canonical-reference "
               "(%llu values)\n",
               out_dim, write_aux, (unsigned long long)checked_values);

        for (size_t span_index = 0u;
             span_index < row_span_count; span_index++) {
            const uint32_t row_span = k_row_spans[span_index];
            for (size_t i = 0u;
                 i < sizeof(k_gate_widths) / sizeof(k_gate_widths[0]); i++) {
                const uint32_t width = k_gate_widths[i];
                if (!run_width(
                        &fixture, width, 256u, row_span, write_aux,
                        &candidate) ||
                    !snapshot_compare(
                        &reference, &candidate, out_dim, write_aux,
                        width, 256u)) {
                    goto cleanup;
                }
                printf("wide-cta-correctness: dim=%u row_span=%u "
                       "write_aux=%u gate_cta=%u down_cta=256 exact "
                       "(%llu values)\n",
                       out_dim, row_span, write_aux, width,
                        (unsigned long long)checked_values);
            }
            for (size_t i = 1u;
                 i < sizeof(k_down_widths) / sizeof(k_down_widths[0]); i++) {
                const uint32_t width = k_down_widths[i];
                if (!run_width(
                        &fixture, 256u, width, row_span, write_aux,
                        &candidate) ||
                    !snapshot_compare(
                        &reference, &candidate, out_dim, write_aux,
                        256u, width)) {
                    goto cleanup;
                }
                printf("wide-cta-correctness: dim=%u row_span=%u "
                       "write_aux=%u gate_cta=256 down_cta=%u exact "
                       "(%llu values)\n",
                       out_dim, row_span, write_aux, width,
                        (unsigned long long)checked_values);
            }
        }
    }
    ok = 1;

cleanup:
    if (fixture.model && !retire_temporary_model_map()) {
        fprintf(stderr, "error: could not retire dim=%u model map\n", out_dim);
        ok = 0;
    }
    snapshot_free(&candidate);
    snapshot_free(&reference);
    fixture_destroy(&fixture);
    return ok;
}

int main(void) {
    normalize_environment();
    g_idle_model_map =
        (unsigned char *)calloc(1u, (size_t)g_idle_model_bytes);
    if (!g_idle_model_map) {
        fprintf(stderr, "error: idle model allocation failed\n");
        return 1;
    }
    if (!ds4_gpu_init()) {
        fprintf(stderr, "error: CUDA backend initialization failed\n");
        free(g_idle_model_map);
        g_idle_model_map = NULL;
        return 1;
    }
    if (!retire_temporary_model_map()) {
        fprintf(stderr, "error: idle model registration failed\n");
        ds4_gpu_cleanup();
        free(g_idle_model_map);
        g_idle_model_map = NULL;
        return 1;
    }

    printf("wide-cta-populations=1,3,4,7,8,9,15,16\n");
    int rc = 0;
    for (size_t i = 0u; i < sizeof(k_out_dims) / sizeof(k_out_dims[0]); i++) {
        if (!run_dimension(k_out_dims[i])) {
            rc = 1;
            break;
        }
    }

    normalize_environment();
    ds4_gpu_cleanup();
    free(g_idle_model_map);
    g_idle_model_map = NULL;
    if (rc == 0) puts("sm75 wide-cta correctness: OK");
    return rc;
}
