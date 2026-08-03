#include "ds4_gpu.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define GIB (1024ull * 1024ull * 1024ull)
#define PROFILE_DEVICE_LIMIT (3ull * GIB)
#define PROFILE_SAFETY_RESERVE (256ull * 1024ull * 1024ull)

typedef enum {
    SCENARIO_MOE_Q4_EARLY,
    SCENARIO_MOE_Q4_LATE,
    SCENARIO_MOE_Q2_EARLY,
    SCENARIO_MOE_Q2_LATE,
    SCENARIO_Q8_Q_B,
    SCENARIO_Q8_ATTN,
    SCENARIO_Q8_SHARED,
    SCENARIO_Q8_OUT_B,
} scenario_kind;

typedef struct {
    const char *name;
    scenario_kind kind;
    uint32_t layer;
    uint32_t owned_pairs;
    uint32_t active_experts;
    uint32_t tile16_count;
    uint32_t in_dim;
    uint32_t out_dim;
} scenario_spec;

static const scenario_spec scenarios[] = {
    {
        "q4-early", SCENARIO_MOE_Q4_EARLY, 3u,
        1879u, 99u, 183u, 4096u, 4096u,
    },
    {
        "q4-late", SCENARIO_MOE_Q4_LATE, 36u,
        2186u, 76u, 189u, 4096u, 4096u,
    },
    {
        "q2-early", SCENARIO_MOE_Q2_EARLY, 3u,
        1879u, 99u, 183u, 4096u, 4096u,
    },
    {
        "q2-late", SCENARIO_MOE_Q2_LATE, 36u,
        2186u, 76u, 189u, 4096u, 4096u,
    },
    {
        "q8-q-b", SCENARIO_Q8_Q_B, 9u,
        0u, 0u, 0u, 1024u, 32768u,
    },
    {
        "q8-attn", SCENARIO_Q8_ATTN, 9u,
        0u, 0u, 0u, 4096u, 1024u,
    },
    {
        "q8-shared", SCENARIO_Q8_SHARED, 8u,
        0u, 0u, 0u, 2048u, 4096u,
    },
    {
        "q8-out-b", SCENARIO_Q8_OUT_B, 9u,
        0u, 0u, 0u, 8192u, 4096u,
    },
};

/* The host mapping must outlive ds4_gpu_cleanup even when a requested device
 * copy fails and the backend falls back to CUDA host registration. */
static unsigned char *model_storage;

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s q4-early|q4-late|q2-early|q2-late|"
            "q8-q-b|q8-attn|q8-shared|q8-out-b\n"
            "\n"
            "A one-process, one-GPU SM75 profiling harness. It never opens a\n"
            "GGUF and caps predicted device state at 3 GiB. Set\n"
            "DS4_PROFILE_AUDIT_CSV to preserve a routed-MoE tile-audit CSV.\n",
            argv0);
}

static const scenario_spec *find_scenario(const char *name) {
    const size_t count = sizeof(scenarios) / sizeof(scenarios[0]);
    for (size_t i = 0; i < count; i++) {
        if (strcmp(name, scenarios[i].name) == 0) return &scenarios[i];
    }
    return NULL;
}

static int checked_add(uint64_t *total, uint64_t bytes) {
    if (UINT64_MAX - *total < bytes) return 0;
    *total += bytes;
    return 1;
}

static int verify_zero_f32(const float *values, uint64_t count,
                           const char *label) {
    for (uint64_t i = 0; i < count; i++) {
        if (!isfinite(values[i]) || values[i] != 0.0f) {
            fprintf(stderr,
                    "error: %s output mismatch at %llu: %.9g (expected 0)\n",
                    label, (unsigned long long)i, (double)values[i]);
            return 0;
        }
    }
    return 1;
}

static int token_has_expert(const int32_t *selected, uint32_t token,
                            uint32_t used, int32_t expert) {
    const uint64_t off = (uint64_t)token * 6u;
    for (uint32_t slot = 0; slot < used; slot++) {
        if (selected[off + slot] == expert) return 1;
    }
    return 0;
}

/* Build a deterministic synthetic top-6 assignment whose owned half matches
 * the aggregate production evidence exactly. The per-expert histogram is not
 * claimed to be the original router histogram: only pair count, active expert
 * count, and tile16 count are constrained by the recorded audit. */
static int build_owned_assignment(const scenario_spec *spec,
                                  int32_t *selected,
                                  float *weights,
                                  uint32_t *tile8_count_out) {
    const uint32_t n_tokens = 512u;
    const uint32_t n_slots = 6u;
    const uint32_t active = spec->active_experts;
    uint32_t *counts = (uint32_t *)calloc(active, sizeof(uint32_t));
    uint32_t *tile_units = (uint32_t *)calloc(active, sizeof(uint32_t));
    uint32_t *fills = (uint32_t *)calloc(n_tokens, sizeof(uint32_t));
    if (!counts || !tile_units || !fills) {
        free(fills); free(tile_units); free(counts);
        return 0;
    }

    for (uint32_t e = 0; e < active; e++) tile_units[e] = 1u;
    uint32_t extra_tiles = spec->tile16_count - active;
    for (uint32_t i = 0; i < extra_tiles; i++) {
        tile_units[(i * 37u + 11u) % active]++;
    }

    uint32_t minimum_pairs = 0u;
    for (uint32_t e = 0; e < active; e++) {
        counts[e] = (tile_units[e] - 1u) * 16u + 1u;
        minimum_pairs += counts[e];
    }
    if (minimum_pairs > spec->owned_pairs ||
        spec->owned_pairs > spec->tile16_count * 16u) {
        fprintf(stderr, "error: impossible routed-MoE aggregate constraints\n");
        free(fills); free(tile_units); free(counts);
        return 0;
    }
    uint32_t remaining = spec->owned_pairs - minimum_pairs;
    uint32_t count_cursor = 0u;
    while (remaining > 0u) {
        uint32_t best = UINT32_MAX;
        for (uint32_t k = 0; k < active; k++) {
            const uint32_t e = (count_cursor + k) % active;
            const uint32_t ceiling = tile_units[e] * 16u;
            if (counts[e] >= ceiling) continue;
            if (best == UINT32_MAX || counts[e] < counts[best]) best = e;
        }
        if (best == UINT32_MAX) {
            fprintf(stderr, "error: aggregate count ceilings exhausted\n");
            free(fills); free(tile_units); free(counts);
            return 0;
        }
        counts[best]++;
        remaining--;
        count_cursor = (best + 1u) % active;
    }

    for (uint64_t i = 0; i < (uint64_t)n_tokens * n_slots; i++) {
        selected[i] = -1;
        weights[i] = 1.0f / 6.0f;
    }

    uint32_t cursor = 0u;
    for (uint32_t e = 0; e < active; e++) {
        for (uint32_t occurrence = 0; occurrence < counts[e]; occurrence++) {
            uint32_t best = UINT32_MAX;
            uint32_t best_fill = n_slots + 1u;
            for (uint32_t k = 0; k < n_tokens; k++) {
                const uint32_t token = (cursor + k) % n_tokens;
                if (fills[token] >= n_slots ||
                    token_has_expert(selected, token, fills[token], (int32_t)e)) {
                    continue;
                }
                if (fills[token] < best_fill) {
                    best = token;
                    best_fill = fills[token];
                    if (best_fill == 0u) break;
                }
            }
            if (best == UINT32_MAX) {
                fprintf(stderr, "error: could not schedule owned expert %u\n", e);
                free(fills); free(tile_units); free(counts);
                return 0;
            }
            selected[(uint64_t)best * n_slots + fills[best]++] = (int32_t)e;
            cursor = (best + 97u) % n_tokens;
        }
    }

    for (uint32_t token = 0; token < n_tokens; token++) {
        while (fills[token] < n_slots) {
            const uint32_t slot = fills[token];
            int32_t expert = -1;
            for (uint32_t attempt = 0; attempt < 128u; attempt++) {
                const int32_t candidate = (int32_t)(128u +
                    ((token * 17u + slot * 31u + attempt) & 127u));
                if (!token_has_expert(selected, token, fills[token], candidate)) {
                    expert = candidate;
                    break;
                }
            }
            if (expert < 0) {
                fprintf(stderr, "error: could not fill partner slots\n");
                free(fills); free(tile_units); free(counts);
                return 0;
            }
            selected[(uint64_t)token * n_slots + fills[token]++] = expert;
        }
    }

    uint32_t pairs = 0u, active_check = 0u, tile16 = 0u, tile8 = 0u;
    for (uint32_t e = 0; e < active; e++) {
        if (counts[e]) active_check++;
        pairs += counts[e];
        tile16 += (counts[e] + 15u) / 16u;
        tile8 += (counts[e] + 7u) / 8u;
    }
    free(fills); free(tile_units); free(counts);
    if (pairs != spec->owned_pairs || active_check != spec->active_experts ||
        tile16 != spec->tile16_count) {
        fprintf(stderr,
                "error: generated aggregate mismatch pairs=%u active=%u tile16=%u\n",
                pairs, active_check, tile16);
        return 0;
    }
    *tile8_count_out = tile8;
    return 1;
}

static int verify_tile_audit(const char *path, const scenario_spec *spec) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "error: cannot read tile audit %s: %s\n",
                path, strerror(errno));
        return 0;
    }
    char line[1024];
    if (!fgets(line, sizeof(line), fp) || !fgets(line, sizeof(line), fp)) {
        fprintf(stderr, "error: tile audit has no data row: %s\n", path);
        fclose(fp);
        return 0;
    }
    fclose(fp);

    unsigned logical = 0, physical = 0, sequence = 0, layer = 0;
    unsigned token_offset = 0, n_tokens = 0, owner_base = 0, owner_count = 0;
    unsigned selected_slots = 0, pair_count = 0, tile_count = 0, slot_count = 0;
    unsigned active_experts = 0, padded_slots = 0;
    char ownership[32];
    double fill = 0.0;
    const int fields = sscanf(
        line,
        "%u,%u,%u,%u,%u,%u,%31[^,],%u,%u,%u,%u,%u,%u,%u,%u,%lf",
        &logical, &physical, &sequence, &layer, &token_offset, &n_tokens,
        ownership, &owner_base, &owner_count, &selected_slots, &pair_count,
        &tile_count, &slot_count, &active_experts, &padded_slots, &fill);
    if (fields != 16) {
        fprintf(stderr, "error: malformed tile-audit row: %s", line);
        return 0;
    }
    const uint32_t expected_padding = spec->tile16_count * 16u - spec->owned_pairs;
    if (layer != spec->layer || token_offset != 0u || n_tokens != 512u ||
        owner_base != 0u || owner_count != 128u ||
        pair_count != spec->owned_pairs || tile_count != spec->tile16_count ||
        slot_count != spec->tile16_count * 16u ||
        active_experts != spec->active_experts || padded_slots != expected_padding) {
        fprintf(stderr,
                "error: tile audit mismatch layer=%u tokens=%u pairs=%u "
                "tile16=%u active=%u padded=%u\n",
                layer, n_tokens, pair_count, tile_count, active_experts,
                padded_slots);
        return 0;
    }
    printf("audit_logical_device=%u\naudit_physical_device=%u\n"
           "audit_pair_count=%u\naudit_tile16_count=%u\n"
           "audit_active_experts=%u\naudit_padded_slots=%u\n"
           "audit_fill_pct=%.6f\n",
           logical, physical, pair_count, tile_count, active_experts,
           padded_slots, fill);
    return 1;
}

static int confirm_device_copy(uint64_t free_before, uint64_t free_after,
                               uint64_t model_bytes) {
    const uint64_t used = free_before > free_after ? free_before - free_after : 0u;
    printf("model_device_copy_delta_bytes=%llu\n",
           (unsigned long long)used);
    if (used < model_bytes * 9u / 10u) {
        fprintf(stderr,
                "error: model was not confirmed device-resident "
                "(model=%llu free-vram delta=%llu)\n",
                (unsigned long long)model_bytes,
                (unsigned long long)used);
        return 0;
    }
    return 1;
}

static int run_moe(const scenario_spec *spec, int q2_recipe) {
    const uint32_t n_tokens = 512u;
    const uint32_t n_expert = 6u;
    const uint32_t n_total_expert = 256u;
    const uint32_t resident_experts = 128u;
    const uint32_t in_dim = 4096u;
    const uint32_t mid_dim = 2048u;
    const uint32_t out_dim = 4096u;
    const uint32_t gate_type = q2_recipe ? 16u : 12u;
    const uint32_t down_type = q2_recipe ? 10u : 12u;
    const uint64_t gate_block_bytes = q2_recipe ? 66u : 144u;
    const uint64_t down_block_bytes = q2_recipe ? 84u : 144u;
    const uint64_t gate_row_bytes =
        (uint64_t)(in_dim / 256u) * gate_block_bytes;
    const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes =
        (uint64_t)(mid_dim / 256u) * down_block_bytes;
    const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;
    const uint64_t gate_offset = 0u;
    const uint64_t up_offset = gate_expert_bytes * resident_experts;
    const uint64_t down_offset = up_offset + gate_expert_bytes * resident_experts;
    const uint64_t model_bytes = down_offset + down_expert_bytes * resident_experts;
    const uint64_t pair_count = (uint64_t)n_tokens * n_expert;
    const uint64_t x_count = (uint64_t)n_tokens * in_dim;
    const uint64_t mid_count = pair_count * mid_dim;
    const uint64_t out_count = (uint64_t)n_tokens * out_dim;
    const uint64_t down_count = pair_count * out_dim;
    uint64_t tensor_bytes = 0u;
    if (!checked_add(&tensor_bytes, x_count * sizeof(float)) ||
        !checked_add(&tensor_bytes, pair_count * sizeof(int32_t)) ||
        !checked_add(&tensor_bytes, pair_count * sizeof(float)) ||
        !checked_add(&tensor_bytes, out_count * sizeof(float)) ||
        !checked_add(&tensor_bytes, mid_count * sizeof(float) * 3u) ||
        !checked_add(&tensor_bytes, down_count * sizeof(float))) {
        fprintf(stderr, "error: routed-MoE tensor byte accounting overflow\n");
        return 0;
    }
    const uint64_t predicted = model_bytes + tensor_bytes + PROFILE_SAFETY_RESERVE;
    printf("scenario=%s\nprofile_kind=%s\n"
           "gate_type=%s\ndown_type=%s\n"
           "layer=%u\nn_tokens=%u\nresident_experts=%u\n"
           "model_bytes=%llu\ntensor_bytes=%llu\n"
           "predicted_device_bytes=%llu\ndevice_limit_bytes=%llu\n",
           spec->name,
           q2_recipe ? "iq2_gate_up_q2_down" : "q4_gate_up_q4_down",
           q2_recipe ? "iq2_xxs" : "q4_k",
           q2_recipe ? "q2_k" : "q4_k",
           spec->layer, n_tokens, resident_experts,
           (unsigned long long)model_bytes,
           (unsigned long long)tensor_bytes,
           (unsigned long long)predicted,
           (unsigned long long)PROFILE_DEVICE_LIMIT);
    if (predicted > PROFILE_DEVICE_LIMIT) {
        fprintf(stderr,
                "error: predicted routed-MoE state exceeds the 3 GiB ceiling\n");
        return 0;
    }

    unsigned char *model = (unsigned char *)calloc(1, (size_t)model_bytes);
    model_storage = model;
    float *x_host = (float *)malloc((size_t)x_count * sizeof(float));
    int32_t *selected_host = (int32_t *)malloc((size_t)pair_count * sizeof(int32_t));
    float *weights_host = (float *)malloc((size_t)pair_count * sizeof(float));
    float *out_host = (float *)malloc((size_t)out_count * sizeof(float));
    ds4_gpu_tensor *x = NULL, *selected = NULL, *weights = NULL;
    ds4_gpu_tensor *out = NULL, *gate = NULL, *up = NULL, *mid = NULL, *down = NULL;
    int ok = 0;
    int audit_started = 0;
    char default_audit[256];
    const char *audit_path = getenv("DS4_PROFILE_AUDIT_CSV");
    int remove_audit = 0;
    uint32_t tile8_count = 0u;
    if (!audit_path || !audit_path[0]) {
        snprintf(default_audit, sizeof(default_audit),
                 "/tmp/ds4-sm75-profile-audit-%ld.csv", (long)getpid());
        audit_path = default_audit;
        remove_audit = 1;
    }
    if (!model || !x_host || !selected_host || !weights_host || !out_host) {
        fprintf(stderr, "error: host allocation failed for routed-MoE harness\n");
        goto cleanup;
    }
    for (uint64_t i = 0; i < x_count; i++) {
        const int value = (int)((i * 23u + (i >> 5u) * 17u) % 257u) - 128;
        x_host[i] = (float)value / 133.0f;
    }
    if (!build_owned_assignment(spec, selected_host, weights_host,
                                &tile8_count)) {
        goto cleanup;
    }
    printf("synthetic_tile8_count=%u\nrecorded_owned_pairs=%u\n"
           "recorded_active_experts=%u\nrecorded_tile16_count=%u\n",
           tile8_count, spec->owned_pairs, spec->active_experts,
           spec->tile16_count);

    x = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    selected = ds4_gpu_tensor_alloc(pair_count * sizeof(int32_t));
    weights = ds4_gpu_tensor_alloc(pair_count * sizeof(float));
    out = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    gate = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    up = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    mid = ds4_gpu_tensor_alloc(mid_count * sizeof(float));
    down = ds4_gpu_tensor_alloc(down_count * sizeof(float));
    if (!x || !selected || !weights || !out || !gate || !up || !mid || !down) {
        fprintf(stderr,
                "error: device tensor allocation failed for routed-MoE harness\n");
        goto cleanup;
    }
    if (!ds4_gpu_tensor_write(x, 0, x_host, x_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(selected, 0, selected_host,
                              pair_count * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(weights, 0, weights_host,
                              pair_count * sizeof(float))) {
        fprintf(stderr, "error: routed-MoE input upload failed\n");
        goto cleanup;
    }
    const uint64_t free_before = ds4_gpu_tier_free_vram(0);
    if (!ds4_gpu_set_model_map(model, model_bytes) || !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: routed-MoE model device copy failed\n");
        goto cleanup;
    }
    const uint64_t free_after = ds4_gpu_tier_free_vram(0);
    if (!confirm_device_copy(free_before, free_after, model_bytes)) goto cleanup;
    if (!ds4_gpu_prefill_tile_audit_begin(8u)) {
        fprintf(stderr, "error: could not start routed-MoE tile audit\n");
        goto cleanup;
    }
    audit_started = 1;
    bool mid_is_f16 = false;
    if (!ds4_gpu_routed_moe_batch_owned_tensor(
            out, gate, up, mid, down, model, model_bytes,
            gate_offset, up_offset, down_offset, gate_type, down_type,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim,
            selected, weights, n_total_expert, n_expert,
            0u, resident_experts, 10.0f, x,
            spec->layer, n_tokens, 0u, &mid_is_f16) ||
        mid_is_f16 || !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: routed-MoE production kernel launch failed\n");
        goto cleanup;
    }
    if (!ds4_gpu_prefill_tile_audit_write_csv(audit_path) ||
        !verify_tile_audit(audit_path, spec)) {
        goto cleanup;
    }
    if (!ds4_gpu_tensor_read(out, 0, out_host, out_count * sizeof(float)) ||
        !verify_zero_f32(out_host, out_count, spec->name)) {
        goto cleanup;
    }
    printf("output_values_checked=%llu\noutput_validation=exact-zero\n",
           (unsigned long long)out_count);
    ok = 1;

cleanup:
    if (audit_started) ds4_gpu_prefill_tile_audit_end();
    ds4_gpu_tensor_free(down);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(x);
    free(out_host);
    free(weights_host);
    free(selected_host);
    free(x_host);
    if (remove_audit) (void)unlink(audit_path);
    return ok;
}

static int run_q8(const scenario_spec *spec) {
    const uint32_t n_tokens = 512u;
    const uint64_t blocks = spec->in_dim / 32u;
    const uint64_t model_bytes = (uint64_t)spec->out_dim * blocks * 34u;
    const uint64_t x_count = (uint64_t)n_tokens * spec->in_dim;
    const uint64_t out_count = (uint64_t)n_tokens * spec->out_dim;
    const uint64_t tensor_bytes = (x_count + out_count) * sizeof(float);
    const uint64_t predicted = model_bytes + tensor_bytes + PROFILE_SAFETY_RESERVE;
    printf("scenario=%s\nprofile_kind=dense_q8_sm75\nlayer=%u\n"
           "n_tokens=%u\nin_dim=%u\nout_dim=%u\nq8_blocks=%llu\n"
           "model_bytes=%llu\ntensor_bytes=%llu\n"
           "predicted_device_bytes=%llu\ndevice_limit_bytes=%llu\n",
           spec->name, spec->layer, n_tokens, spec->in_dim, spec->out_dim,
           (unsigned long long)blocks,
           (unsigned long long)model_bytes,
           (unsigned long long)tensor_bytes,
           (unsigned long long)predicted,
           (unsigned long long)PROFILE_DEVICE_LIMIT);
    if (predicted > PROFILE_DEVICE_LIMIT) {
        fprintf(stderr, "error: predicted Q8 state exceeds the 3 GiB ceiling\n");
        return 0;
    }

    unsigned char *model = (unsigned char *)calloc(1, (size_t)model_bytes);
    model_storage = model;
    float *x_host = (float *)malloc((size_t)x_count * sizeof(float));
    float *out_host = (float *)malloc((size_t)out_count * sizeof(float));
    ds4_gpu_tensor *x = NULL, *out = NULL;
    int ok = 0;
    if (!model || !x_host || !out_host) {
        fprintf(stderr, "error: host allocation failed for Q8 harness\n");
        goto cleanup;
    }
    for (uint64_t i = 0; i < x_count; i++) {
        const int value = (int)((i * 29u + (i >> 3u) * 11u) % 193u) - 96;
        x_host[i] = (float)value / 101.0f;
    }
    x = ds4_gpu_tensor_alloc(x_count * sizeof(float));
    out = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    if (!x || !out || !ds4_gpu_tensor_write(x, 0, x_host,
                                             x_count * sizeof(float))) {
        fprintf(stderr, "error: Q8 device allocation/input upload failed\n");
        goto cleanup;
    }
    const uint64_t free_before = ds4_gpu_tier_free_vram(0);
    if (!ds4_gpu_set_model_map(model, model_bytes) || !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: Q8 model device copy failed\n");
        goto cleanup;
    }
    const uint64_t free_after = ds4_gpu_tier_free_vram(0);
    if (!confirm_device_copy(free_before, free_after, model_bytes)) goto cleanup;
    if (!ds4_gpu_matmul_q8_0_tensor(out, model, model_bytes, 0u,
                                    spec->in_dim, spec->out_dim,
                                    x, n_tokens) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: Q8 production kernel launch failed\n");
        goto cleanup;
    }
    if (!ds4_gpu_tensor_read(out, 0, out_host, out_count * sizeof(float)) ||
        !verify_zero_f32(out_host, out_count, spec->name)) {
        goto cleanup;
    }
    printf("output_values_checked=%llu\noutput_validation=exact-zero\n",
           (unsigned long long)out_count);
    ok = 1;

cleanup:
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    free(out_host);
    free(x_host);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        usage(argv[0]);
        return 2;
    }
    const scenario_spec *spec = find_scenario(argv[1]);
    if (!spec) {
        usage(argv[0]);
        return 2;
    }

    /* Presence-based production switches are normalized inside the harness so
     * an inherited debug shell cannot silently select a different kernel. */
    (void)setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    (void)setenv("DS4_CUDA_NO_Q8_F16_CACHE", "1", 1);
    (void)unsetenv("DS4_CUDA_NO_Q8_MMA");
    (void)unsetenv("DS4_CUDA_NO_Q8_MMA_SM75");
    (void)unsetenv("DS4_CUDA_Q8_MMA_SM75_TOK16");
    (void)unsetenv("DS4_CUDA_MOE_NO_Q4_MMA");
    (void)unsetenv("DS4_CUDA_MOE_NO_Q4_MMA_TILE16");
    (void)unsetenv("DS4_CUDA_MOE_NO_Q4_MMA_TILE16_SM75");
    (void)unsetenv("DS4_CUDA_MOE_Q4_GATE_TILE16_SM75");
    (void)unsetenv("DS4_CUDA_MOE_Q4_GATE_STAGE4_SM75");
    (void)unsetenv("DS4_CUDA_MOE_NO_EXPERT_TILES");
    (void)unsetenv("DS4_CUDA_MOE_NO_IQ2_MMA_SM75");
    (void)unsetenv("DS4_CUDA_MOE_NO_IQ2_MMA_TILE16_SM75");
    (void)unsetenv("DS4_CUDA_MOE_IQ2_STAGE6_SM75");
    (void)unsetenv("DS4_CUDA_MOE_IQ2_STAGE4_SM75");
    (void)unsetenv("DS4_CUDA_MOE_Q2_DOWN_MMA_SM75");
    (void)unsetenv("DS4_CUDA_MOE_MIXED_TAIL_TILES");
    (void)unsetenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN");
    (void)unsetenv("DS4_CUDA_MOE_NO_DOWN_TILE16");
    (void)unsetenv("DS4_CUDA_MOE_NO_DOWN_ROW2048");
    (void)unsetenv("DS4_CUDA_MOE_DOWN_ROW1024");
    (void)unsetenv("DS4_CUDA_MOE_DOWN_ROW2048");
    (void)unsetenv("DS4_CUDA_MOE_NO_GATE_ROW2048");
    (void)unsetenv("DS4_CUDA_MOE_GATE_ROW1024");
    (void)unsetenv("DS4_CUDA_MOE_GATE_ROW2048");
    (void)unsetenv("DS4_CUDA_MOE_WRITE_GATE_UP");

    if (!ds4_gpu_init()) {
        fprintf(stderr, "error: CUDA backend initialization failed\n");
        return 1;
    }
    const uint64_t free_at_start = ds4_gpu_tier_free_vram(0);
    printf("free_vram_at_start_bytes=%llu\n",
           (unsigned long long)free_at_start);
    if (free_at_start < PROFILE_DEVICE_LIMIT + GIB) {
        fprintf(stderr,
                "error: harness requires at least 4 GiB free VRAM; found %.2f GiB\n",
                (double)free_at_start / (double)GIB);
        ds4_gpu_cleanup();
        return 1;
    }

    const int q4_moe = spec->kind == SCENARIO_MOE_Q4_EARLY ||
                       spec->kind == SCENARIO_MOE_Q4_LATE;
    const int q2_moe = spec->kind == SCENARIO_MOE_Q2_EARLY ||
                       spec->kind == SCENARIO_MOE_Q2_LATE;
    const int ok = q4_moe ? run_moe(spec, 0) :
                   (q2_moe ? run_moe(spec, 1) : run_q8(spec));
    ds4_gpu_cleanup();
    free(model_storage);
    model_storage = NULL;
    if (ok) printf("harness_status=ok\n");
    return ok ? 0 : 1;
}
