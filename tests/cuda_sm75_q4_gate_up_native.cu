/*
 * Production-shaped SM75 Q4_K gate/up packed-A/W experiment.
 *
 * The down-layout harness already contains byte-validated definitions of
 * BlockQ4K, BlockQ8K, NativeQ8K, NativeWeightTileBlock, the Q8_K quantizer,
 * nibble-plane packer, MMA helpers, guarded allocations, and host edge-case
 * generators.  Include it as an implementation library after renaming its
 * entry point.  This avoids a second subtly different quantization/layout
 * implementation; none of its down consumers are called by this harness.
 * The include is intentional until these experiment-only helpers earn a
 * small common header.
 */
#define main sm75_q4_down_native_embedded_main
#include "cuda_sm75_q4_down_native.cu"
#undef main

enum {
    GU_TOKENS = 512,
    GU_SELECTED = 6,
    GU_PAIR_SLOTS = GU_TOKENS * GU_SELECTED,
    GU_IN_BLOCKS = 16,
    GU_IN_DIM = GU_IN_BLOCKS * QK_K,
    GU_MID_DIM = 2048,
    GU_OUT_TILE = 8,
    GU_OUT_TILES = GU_MID_DIM / GU_OUT_TILE,
    GU_ROW_SPAN = 512,
    GU_TILE_PAIRS = 8,
};

static_assert(GU_PAIR_SLOTS == TOTAL_PAIR_SLOTS,
              "embedded helpers and gate/up pair space must agree");
static_assert(GU_IN_DIM == 4096, "production gate/up input width");
static_assert(GU_MID_DIM % GU_ROW_SPAN == 0, "row-span coverage");
static_assert(GU_MID_DIM / GU_ROW_SPAN == 4,
              "shipping gate/up launch must use four row CTAs");
static_assert((8u * GU_IN_BLOCKS * sizeof(BlockQ8K) +
               (3u * GU_TILE_PAIRS + 1u) * sizeof(uint32_t) + 15u) /
              16u * 16u == 37488u,
              "full shipping route must request 37488 shared bytes");
static_assert(7u * GU_IN_BLOCKS * sizeof(NativeQ8K) +
              2u * GU_TILE_PAIRS * sizeof(uint32_t) == 32768u,
              "stage7 must target the 32 KiB occupancy boundary exactly");

struct GuScenarioSpec {
    const char *name;
    uint32_t layer;
    uint32_t pairs;
    uint32_t active_experts;
    uint32_t recorded_tile16;
    uint32_t tile8;
    uint32_t padded8;
};

/* Only pairs/active/tile16 were recorded in the production audit.  tile8 is
 * derived from the deterministic synthetic histogram below, not asserted to
 * be the original router histogram. */
static const GuScenarioSpec gu_scenarios[] = {
    {"early", 3u, 1879u, 99u, 183u, 282u, 377u},
    {"late", 36u, 2186u, 76u, 189u, 331u, 462u},
};

enum GuVariant {
    GU_STANDARD,
    GU_STANDARD_WARP16,
    GU_NATIVE_W,
    GU_NATIVE_AW_CONSUMER,
    GU_NATIVE_AW_WARP16_CONSUMER,
    GU_NATIVE_AW_STAGE7_CONSUMER,
    GU_NATIVE_AW_COMBINED,
    GU_NATIVE_AW_WARP16_COMBINED,
    GU_NATIVE_AW_STAGE7_COMBINED,
    GU_NATIVE_AW_STAGE7_SCALAR_CONSUMER,
    GU_NATIVE_AW_WARP16_SCALAR_CONSUMER,
    GU_NATIVE_AW_STAGE7_SCALAR_COMBINED,
    GU_NATIVE_AW_WARP16_SCALAR_COMBINED,
    GU_PACK_A,
    GU_VARIANT_COUNT,
};

static const char *const gu_variant_names[GU_VARIANT_COUNT] = {
    "standard",
    "standard-warp16",
    "native-w",
    "native-aw-consumer",
    "native-aw-warp16-consumer",
    "native-aw-stage7-consumer",
    "native-aw-combined",
    "native-aw-warp16-combined",
    "native-aw-stage7-combined",
    "native-aw-stage7-scalar-consumer",
    "native-aw-warp16-scalar-consumer",
    "native-aw-stage7-scalar-combined",
    "native-aw-warp16-scalar-combined",
    "pack-a",
};

struct GuHostMetadata {
    uint32_t *counts;
    uint32_t *offsets;
    uint32_t *sorted_pairs;
    uint32_t *tile_experts;
    uint32_t *tile_starts;
};

struct GuData {
    const GuScenarioSpec *spec;
    GuHostMetadata meta;
    DeviceBuffer standard_gate_w;
    DeviceBuffer standard_up_w;
    DeviceBuffer native_gate_w;
    DeviceBuffer native_up_w;
    DeviceBuffer standard_a;
    DeviceBuffer native_a;
    DeviceBuffer router_weights;
    DeviceBuffer gate_out;
    DeviceBuffer up_out;
    DeviceBuffer mid_out;
    uint32_t *d_sorted_pairs;
    uint32_t *d_offsets;
    uint32_t *d_counts;
    uint32_t *d_tile_total;
    uint32_t *d_tile_experts;
    uint32_t *d_tile_starts;
    uint64_t weight_bytes;
    uint64_t activation_blocks;
    uint64_t output_values;
};

static const GuScenarioSpec *gu_find_scenario(const char *name) {
    for (size_t i = 0; i < sizeof(gu_scenarios) / sizeof(gu_scenarios[0]); i++)
        if (!strcmp(name, gu_scenarios[i].name)) return &gu_scenarios[i];
    return NULL;
}

static GuVariant gu_find_variant(const char *name) {
    for (int i = 0; i < GU_VARIANT_COUNT; i++)
        if (!strcmp(name, gu_variant_names[i])) return (GuVariant)i;
    fprintf(stderr, "error: unknown gate/up profile variant: %s\n", name);
    exit(2);
}

static void gu_free_metadata(GuHostMetadata *m) {
    free(m->tile_starts);
    free(m->tile_experts);
    free(m->sorted_pairs);
    free(m->offsets);
    free(m->counts);
    memset(m, 0, sizeof(*m));
}

/* Reproduce the profile harness's tile16-constrained histogram, then derive
 * its tile8 table.  This preserves the recorded aggregates while labeling
 * the finer tile8 distribution honestly as synthetic. */
static int gu_build_metadata(const GuScenarioSpec *spec, GuHostMetadata *m) {
    memset(m, 0, sizeof(*m));
    const uint32_t active = spec->active_experts;
    uint32_t *tile16_units = (uint32_t *)calloc(active, sizeof(uint32_t));
    m->counts = (uint32_t *)calloc(active, sizeof(uint32_t));
    m->offsets = (uint32_t *)calloc(active + 1u, sizeof(uint32_t));
    m->sorted_pairs = (uint32_t *)malloc((size_t)spec->pairs * sizeof(uint32_t));
    m->tile_experts = (uint32_t *)malloc((size_t)spec->tile8 * sizeof(uint32_t));
    m->tile_starts = (uint32_t *)malloc((size_t)spec->tile8 * sizeof(uint32_t));
    if (!tile16_units || !m->counts || !m->offsets || !m->sorted_pairs ||
        !m->tile_experts || !m->tile_starts) {
        free(tile16_units);
        gu_free_metadata(m);
        return 0;
    }
    for (uint32_t e = 0; e < active; e++) tile16_units[e] = 1u;
    for (uint32_t i = 0; i < spec->recorded_tile16 - active; i++)
        tile16_units[(i * 37u + 11u) % active]++;
    uint32_t minimum = 0;
    for (uint32_t e = 0; e < active; e++) {
        m->counts[e] = (tile16_units[e] - 1u) * 16u + 1u;
        minimum += m->counts[e];
    }
    if (minimum > spec->pairs || spec->pairs > spec->recorded_tile16 * 16u) {
        fprintf(stderr, "error: impossible %s recorded aggregate\n", spec->name);
        free(tile16_units);
        gu_free_metadata(m);
        return 0;
    }
    uint32_t remaining = spec->pairs - minimum;
    uint32_t cursor = 0u;
    while (remaining) {
        uint32_t best = UINT32_MAX;
        for (uint32_t k = 0; k < active; k++) {
            const uint32_t e = (cursor + k) % active;
            if (m->counts[e] >= tile16_units[e] * 16u) continue;
            if (best == UINT32_MAX || m->counts[e] < m->counts[best]) best = e;
        }
        if (best == UINT32_MAX) {
            fprintf(stderr, "error: %s count capacity exhausted\n", spec->name);
            free(tile16_units);
            gu_free_metadata(m);
            return 0;
        }
        m->counts[best]++;
        remaining--;
        cursor = (best + 1u) % active;
    }
    uint32_t pair_cursor = 0u, tile_cursor = 0u;
    uint32_t stride = 97u;
    while (gcd_u32(stride, GU_PAIR_SLOTS) != 1u) stride += 2u;
    for (uint32_t e = 0; e < active; e++) {
        m->offsets[e] = pair_cursor;
        for (uint32_t i = 0; i < m->counts[e]; i++) {
            const uint32_t logical = pair_cursor + i;
            m->sorted_pairs[logical] = (uint32_t)(
                ((uint64_t)logical * stride + 17u) % GU_PAIR_SLOTS);
        }
        const uint32_t tiles = (m->counts[e] + GU_TILE_PAIRS - 1u) /
                               GU_TILE_PAIRS;
        for (uint32_t t = 0; t < tiles; t++) {
            m->tile_experts[tile_cursor] = e;
            m->tile_starts[tile_cursor] = t * GU_TILE_PAIRS;
            tile_cursor++;
        }
        pair_cursor += m->counts[e];
    }
    m->offsets[active] = pair_cursor;
    free(tile16_units);
    if (pair_cursor != spec->pairs || tile_cursor != spec->tile8 ||
        spec->tile8 * GU_TILE_PAIRS - spec->pairs != spec->padded8) {
        fprintf(stderr, "error: %s synthetic tile8 aggregate mismatch\n", spec->name);
        gu_free_metadata(m);
        return 0;
    }
    return 1;
}

/* Gate/up weight records are [expert][8-row output tile][input block].  One
 * record is exactly eight row-major Q4_K blocks, so both matrices remain
 * byte-size-neutral. */
extern "C" __global__ void sm75_q4_gate_up_pack_w_kernel(
        NativeWeightTileBlock *out, const BlockQ4K *in,
        uint32_t n_experts) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint64_t record = (uint64_t)blockIdx.x * WARPS_PER_CTA + warp;
    const uint64_t records = (uint64_t)n_experts * GU_OUT_TILES * GU_IN_BLOCKS;
    if (record >= records) return;
    const uint32_t b = (uint32_t)(record % GU_IN_BLOCKS);
    const uint64_t tile_index = record / GU_IN_BLOCKS;
    const uint32_t out_tile = (uint32_t)(tile_index % GU_OUT_TILES);
    const uint32_t expert = (uint32_t)(tile_index / GU_OUT_TILES);
    const BlockQ4K *base = in +
        ((uint64_t)expert * GU_MID_DIM + out_tile * GU_OUT_TILE) * GU_IN_BLOCKS;
    NativeWeightTileBlock *dst = out + record;
    if (lane < GU_OUT_TILE)
        dst->hdr[lane] = *(const uint4 *)(
            base + (uint64_t)lane * GU_IN_BLOCKS + b);
    const uint32_t row = lane >> 2u;
    const uint32_t lane4 = lane & 3u;
    const BlockQ4K *src = base + (uint64_t)row * GU_IN_BLOCKS + b;
#pragma unroll
    for (uint32_t j = 0; j < Q4_GROUPS; j++) {
        const uint32_t off = (j >> 1u) * 32u + lane4 * 8u;
        const uint32_t q0 = load_u32_unaligned(src->qs + off);
        const uint32_t q1 = load_u32_unaligned(src->qs + off + 4u);
        dst->b[j][lane] = pack_nibbles_8(q0, q1, (j & 1u) ? 4u : 0u);
    }
}

extern "C" __global__ void sm75_q4_gate_up_pack_a_kernel(
        NativeQ8K *out, const BlockQ8K *in, uint64_t n_blocks) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint64_t block = (uint64_t)blockIdx.x * WARPS_PER_CTA + warp;
    if (block >= n_blocks) return;
    const BlockQ8K *src = in + block;
    NativeQ8K *dst = out + block;
    const uint32_t j = lane >> 2u;
    const uint32_t lane4 = lane & 3u;
    const uint32_t off = j * 32u + lane4 * 8u;
    const uint32_t x0 = load_u32_unaligned(src->qs + off);
    const uint32_t x1 = load_u32_unaligned(src->qs + off + 4u);
    dst->low[j][lane4] = pack_nibbles_8(x0, x1, 0u);
    dst->high_signed[j][lane4] = pack_nibbles_8(x0, x1, 4u);
    if (lane == 0u) dst->d = src->d;
    if (lane < QK_K / 16) dst->bsums[lane] = src->bsums[lane];
}

static size_t gu_shared_bytes(int native_a, uint32_t staged_rows,
                              int shipping_route = 0) {
    const size_t activation = (size_t)staged_rows * GU_IN_BLOCKS *
        (native_a ? sizeof(NativeQ8K) : sizeof(BlockQ8K));
    const size_t route_words = shipping_route ?
        3u * GU_TILE_PAIRS + 1u : 2u * GU_TILE_PAIRS;
    return (activation + route_words * sizeof(uint32_t) + 15u) &
           ~(size_t)15u;
}

template <bool NATIVE_W, bool NATIVE_A, uint32_t STAGED_ROWS>
__device__ __forceinline__ static void gu_gate_up_pass(
        const BlockQ4K *standard_gate_w,
        const BlockQ4K *standard_up_w,
        const NativeWeightTileBlock *native_gate_w,
        const NativeWeightTileBlock *native_up_w,
        const BlockQ8K *standard_a,
        const NativeQ8K *native_a,
        const BlockQ8K *staged_standard,
        const NativeQ8K *staged_native,
        const uint32_t *tokens,
        uint32_t np, uint32_t expert, uint32_t row0, uint32_t lane,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint32_t xq_blocks, uint32_t expert_mid_dim,
        float gate[2], float up[2]) {
    const uint32_t mtok = lane >> 2u;
    const uint32_t n0 = (lane & 3u) * 2u;
    const BlockQ8K *sa = NULL;
    const NativeQ8K *na = NULL;
    if (NATIVE_A) {
        na = xq_blocks <= GU_IN_BLOCKS && mtok < STAGED_ROWS
            ? staged_native + (uint64_t)mtok * GU_IN_BLOCKS
            : (mtok < np
               ? native_a + (uint64_t)tokens[mtok] * xq_blocks
               : staged_native);
    } else {
        sa = xq_blocks <= GU_IN_BLOCKS && mtok < STAGED_ROWS
            ? staged_standard + (uint64_t)mtok * GU_IN_BLOCKS
            : (mtok < np
               ? standard_a + (uint64_t)tokens[mtok] * xq_blocks
               : staged_standard);
    }
    const char *gbase = (const char *)standard_gate_w +
        (uint64_t)expert * gate_expert_bytes +
        (uint64_t)row0 * gate_row_bytes;
    const char *ubase = (const char *)standard_up_w +
        (uint64_t)expert * gate_expert_bytes +
        (uint64_t)row0 * gate_row_bytes;
    const uint64_t native_tile =
        (uint64_t)expert * (expert_mid_dim / GU_OUT_TILE) +
        row0 / GU_OUT_TILE;
    float sg0[8] = {0,0,0,0,0,0,0,0};
    float sg1[8] = {0,0,0,0,0,0,0,0};
    float su0[8] = {0,0,0,0,0,0,0,0};
    float su1[8] = {0,0,0,0,0,0,0,0};
    for (uint32_t b = 0; b < xq_blocks; b++) {
        const NativeWeightTileBlock *ng = native_gate_w +
            native_tile * xq_blocks + b;
        const NativeWeightTileBlock *nu = native_up_w +
            native_tile * xq_blocks + b;
        const uint4 gh0 = NATIVE_W ? ng->hdr[n0] :
            *(const uint4 *)(gbase +
                (uint64_t)n0 * gate_row_bytes + b * sizeof(BlockQ4K));
        const uint4 gh1 = NATIVE_W ? ng->hdr[n0 + 1u] :
            *(const uint4 *)(gbase +
                (uint64_t)(n0 + 1u) * gate_row_bytes +
                b * sizeof(BlockQ4K));
        const uint4 uh0 = NATIVE_W ? nu->hdr[n0] :
            *(const uint4 *)(ubase +
                (uint64_t)n0 * gate_row_bytes + b * sizeof(BlockQ4K));
        const uint4 uh1 = NATIVE_W ? nu->hdr[n0 + 1u] :
            *(const uint4 *)(ubase +
                (uint64_t)(n0 + 1u) * gate_row_bytes +
                b * sizeof(BlockQ4K));
        uint32_t gw8[8], uw8[8];
        if (!NATIVE_W) {
            const uint32_t *gqw = (const uint32_t *)(
                ((const BlockQ4K *)(gbase +
                 (uint64_t)(lane >> 2u) * gate_row_bytes) + b)->qs);
            const uint32_t *uqw = (const uint32_t *)(
                ((const BlockQ4K *)(ubase +
                 (uint64_t)(lane >> 2u) * gate_row_bytes) + b)->qs);
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++) {
                gw8[k] = gqw[k * 4u + (lane & 3u)];
                uw8[k] = uqw[k * 4u + (lane & 3u)];
            }
        }
        int gi0 = 0, gi1 = 0, gm0 = 0, gm1 = 0;
        int ui0 = 0, ui1 = 0, um0 = 0, um1 = 0;
#pragma unroll
        for (uint32_t j = 0; j < Q4_GROUPS; j++) {
            int32_t gc0 = 0, gc1 = 0, uc0 = 0, uc1 = 0;
            if (NATIVE_W) {
                uint32_t al, ah;
                if (NATIVE_A) {
                    al = na[b].low[j][lane & 3u];
                    ah = na[b].high_signed[j][lane & 3u];
                } else {
                    const uint32_t off = j * 32u + (lane & 3u) * 8u;
                    const uint32_t x0 = load_u32_unaligned(sa[b].qs + off);
                    const uint32_t x1 = load_u32_unaligned(sa[b].qs + off + 4u);
                    al = pack_nibbles_8(x0, x1, 0u);
                    ah = pack_nibbles_8(x0, x1, 4u);
                }
                int32_t glo0 = 0, glo1 = 0, ghi0 = 0, ghi1 = 0;
                int32_t ulo0 = 0, ulo1 = 0, uhi0 = 0, uhi1 = 0;
                const uint32_t gb = ng->b[j][lane];
                const uint32_t ub = nu->b[j][lane];
                mma_m8n8k32_u4_u4(glo0, glo1, al, gb);
                mma_m8n8k32_s4_u4(ghi0, ghi1, ah, gb);
                mma_m8n8k32_u4_u4(ulo0, ulo1, al, ub);
                mma_m8n8k32_s4_u4(uhi0, uhi1, ah, ub);
                gc0 = glo0 + 16 * ghi0;
                gc1 = glo1 + 16 * ghi1;
                uc0 = ulo0 + 16 * uhi0;
                uc1 = ulo1 + 16 * uhi1;
            } else {
                const int shift = (j & 1u) ? 4 : 0;
#pragma unroll
                for (uint32_t h = 0; h < 2u; h++) {
                    const uint32_t off = j * 32u + h * 16u +
                                         (lane & 3u) * 4u;
                    const uint32_t av = *(const uint32_t *)(sa[b].qs + off);
                    const uint32_t gbv =
                        (gw8[(j >> 1u) * 2u + h] >> shift) & 0x0f0f0f0fu;
                    const uint32_t ubv =
                        (uw8[(j >> 1u) * 2u + h] >> shift) & 0x0f0f0f0fu;
                    mma_m8n8k16_s8(gc0, gc1, av, gbv);
                    mma_m8n8k16_s8(uc0, uc1, av, ubv);
                }
            }
            uint8_t sc, mn;
            q4_get_scale_min(j, (const uint8_t *)&gh0.y, &sc, &mn);
            gi0 += (int)sc * gc0;
            const int16_t *bsums = NATIVE_A ? na[b].bsums : sa[b].bsums;
            const int bsum = (int)bsums[2u * j] + (int)bsums[2u * j + 1u];
            gm0 += (int)mn * bsum;
            q4_get_scale_min(j, (const uint8_t *)&gh1.y, &sc, &mn);
            gi1 += (int)sc * gc1;
            gm1 += (int)mn * bsum;
            q4_get_scale_min(j, (const uint8_t *)&uh0.y, &sc, &mn);
            ui0 += (int)sc * uc0;
            um0 += (int)mn * bsum;
            q4_get_scale_min(j, (const uint8_t *)&uh1.y, &sc, &mn);
            ui1 += (int)sc * uc1;
            um1 += (int)mn * bsum;
        }
        const float yd = NATIVE_A ? na[b].d : sa[b].d;
        const uint32_t slot = b & 7u;
        sg0[slot] += yd * f16_to_f32((uint16_t)(gh0.x & 0xffffu)) *
                     (float)gi0 - yd * f16_to_f32((uint16_t)(gh0.x >> 16u)) *
                     (float)gm0;
        sg1[slot] += yd * f16_to_f32((uint16_t)(gh1.x & 0xffffu)) *
                     (float)gi1 - yd * f16_to_f32((uint16_t)(gh1.x >> 16u)) *
                     (float)gm1;
        su0[slot] += yd * f16_to_f32((uint16_t)(uh0.x & 0xffffu)) *
                     (float)ui0 - yd * f16_to_f32((uint16_t)(uh0.x >> 16u)) *
                     (float)um0;
        su1[slot] += yd * f16_to_f32((uint16_t)(uh1.x & 0xffffu)) *
                     (float)ui1 - yd * f16_to_f32((uint16_t)(uh1.x >> 16u)) *
                     (float)um1;
    }
    float a0 = sg0[0] + sg0[4], a1 = sg0[1] + sg0[5];
    float a2 = sg0[2] + sg0[6], a3 = sg0[3] + sg0[7];
    gate[0] = (a0 + a2) + (a1 + a3);
    a0 = sg1[0] + sg1[4]; a1 = sg1[1] + sg1[5];
    a2 = sg1[2] + sg1[6]; a3 = sg1[3] + sg1[7];
    gate[1] = (a0 + a2) + (a1 + a3);
    a0 = su0[0] + su0[4]; a1 = su0[1] + su0[5];
    a2 = su0[2] + su0[6]; a3 = su0[3] + su0[7];
    up[0] = (a0 + a2) + (a1 + a3);
    a0 = su1[0] + su1[4]; a1 = su1[1] + su1[5];
    a2 = su1[2] + su1[6]; a3 = su1[3] + su1[7];
    up[1] = (a0 + a2) + (a1 + a3);
}

/* Experimental packed-A/W pass with no dynamically indexed float arrays.
 * b is warp-uniform, so every lane takes the same slot switch case.  Each
 * scalar retains the shipping slot=b&7 update order and final grouping. */
template <uint32_t STAGED_ROWS>
__device__ __forceinline__ static void gu_gate_up_pass_scalar(
        const NativeWeightTileBlock *native_gate_w,
        const NativeWeightTileBlock *native_up_w,
        const NativeQ8K *native_a,
        const NativeQ8K *staged_native,
        const uint32_t *tokens,
        uint32_t np, uint32_t expert, uint32_t row0, uint32_t lane,
        uint32_t xq_blocks, uint32_t expert_mid_dim,
        float gate[2], float up[2]) {
    const uint32_t mtok = lane >> 2u;
    const uint32_t n0 = (lane & 3u) * 2u;
    const NativeQ8K *na = xq_blocks <= GU_IN_BLOCKS && mtok < STAGED_ROWS
        ? staged_native + (uint64_t)mtok * GU_IN_BLOCKS
        : (mtok < np
           ? native_a + (uint64_t)tokens[mtok] * xq_blocks
           : staged_native);
    const uint64_t native_tile =
        (uint64_t)expert * (expert_mid_dim / GU_OUT_TILE) +
        row0 / GU_OUT_TILE;
#define GU_SCALAR_SLOT_DECL(S) \
    float sg0_##S = 0.0f, sg1_##S = 0.0f; \
    float su0_##S = 0.0f, su1_##S = 0.0f
    GU_SCALAR_SLOT_DECL(0);
    GU_SCALAR_SLOT_DECL(1);
    GU_SCALAR_SLOT_DECL(2);
    GU_SCALAR_SLOT_DECL(3);
    GU_SCALAR_SLOT_DECL(4);
    GU_SCALAR_SLOT_DECL(5);
    GU_SCALAR_SLOT_DECL(6);
    GU_SCALAR_SLOT_DECL(7);
#undef GU_SCALAR_SLOT_DECL
    for (uint32_t b = 0; b < xq_blocks; b++) {
        const NativeWeightTileBlock *ng = native_gate_w +
            native_tile * xq_blocks + b;
        const NativeWeightTileBlock *nu = native_up_w +
            native_tile * xq_blocks + b;
        const uint4 gh0 = ng->hdr[n0];
        const uint4 gh1 = ng->hdr[n0 + 1u];
        const uint4 uh0 = nu->hdr[n0];
        const uint4 uh1 = nu->hdr[n0 + 1u];
        int gi0 = 0, gi1 = 0, gm0 = 0, gm1 = 0;
        int ui0 = 0, ui1 = 0, um0 = 0, um1 = 0;
#pragma unroll
        for (uint32_t j = 0; j < Q4_GROUPS; j++) {
            const uint32_t al = na[b].low[j][lane & 3u];
            const uint32_t ah = na[b].high_signed[j][lane & 3u];
            const uint32_t gb = ng->b[j][lane];
            const uint32_t ub = nu->b[j][lane];
            int32_t glo0 = 0, glo1 = 0, ghi0 = 0, ghi1 = 0;
            int32_t ulo0 = 0, ulo1 = 0, uhi0 = 0, uhi1 = 0;
            mma_m8n8k32_u4_u4(glo0, glo1, al, gb);
            mma_m8n8k32_s4_u4(ghi0, ghi1, ah, gb);
            mma_m8n8k32_u4_u4(ulo0, ulo1, al, ub);
            mma_m8n8k32_s4_u4(uhi0, uhi1, ah, ub);
            const int32_t gc0 = glo0 + 16 * ghi0;
            const int32_t gc1 = glo1 + 16 * ghi1;
            const int32_t uc0 = ulo0 + 16 * uhi0;
            const int32_t uc1 = ulo1 + 16 * uhi1;
            uint8_t sc, mn;
            q4_get_scale_min(j, (const uint8_t *)&gh0.y, &sc, &mn);
            gi0 += (int)sc * gc0;
            const int bsum = (int)na[b].bsums[2u * j] +
                             (int)na[b].bsums[2u * j + 1u];
            gm0 += (int)mn * bsum;
            q4_get_scale_min(j, (const uint8_t *)&gh1.y, &sc, &mn);
            gi1 += (int)sc * gc1;
            gm1 += (int)mn * bsum;
            q4_get_scale_min(j, (const uint8_t *)&uh0.y, &sc, &mn);
            ui0 += (int)sc * uc0;
            um0 += (int)mn * bsum;
            q4_get_scale_min(j, (const uint8_t *)&uh1.y, &sc, &mn);
            ui1 += (int)sc * uc1;
            um1 += (int)mn * bsum;
        }
        const float yd = na[b].d;
        const float vg0 =
            yd * f16_to_f32((uint16_t)(gh0.x & 0xffffu)) * (float)gi0 -
            yd * f16_to_f32((uint16_t)(gh0.x >> 16u)) * (float)gm0;
        const float vg1 =
            yd * f16_to_f32((uint16_t)(gh1.x & 0xffffu)) * (float)gi1 -
            yd * f16_to_f32((uint16_t)(gh1.x >> 16u)) * (float)gm1;
        const float vu0 =
            yd * f16_to_f32((uint16_t)(uh0.x & 0xffffu)) * (float)ui0 -
            yd * f16_to_f32((uint16_t)(uh0.x >> 16u)) * (float)um0;
        const float vu1 =
            yd * f16_to_f32((uint16_t)(uh1.x & 0xffffu)) * (float)ui1 -
            yd * f16_to_f32((uint16_t)(uh1.x >> 16u)) * (float)um1;
#define GU_SCALAR_SLOT_CASE(S) \
        case S: \
            sg0_##S += vg0; sg1_##S += vg1; \
            su0_##S += vu0; su1_##S += vu1; \
            break
        switch (b & 7u) {
            GU_SCALAR_SLOT_CASE(0);
            GU_SCALAR_SLOT_CASE(1);
            GU_SCALAR_SLOT_CASE(2);
            GU_SCALAR_SLOT_CASE(3);
            GU_SCALAR_SLOT_CASE(4);
            GU_SCALAR_SLOT_CASE(5);
            GU_SCALAR_SLOT_CASE(6);
            GU_SCALAR_SLOT_CASE(7);
        }
#undef GU_SCALAR_SLOT_CASE
    }
    float a0 = sg0_0 + sg0_4, a1 = sg0_1 + sg0_5;
    float a2 = sg0_2 + sg0_6, a3 = sg0_3 + sg0_7;
    gate[0] = (a0 + a2) + (a1 + a3);
    a0 = sg1_0 + sg1_4; a1 = sg1_1 + sg1_5;
    a2 = sg1_2 + sg1_6; a3 = sg1_3 + sg1_7;
    gate[1] = (a0 + a2) + (a1 + a3);
    a0 = su0_0 + su0_4; a1 = su0_1 + su0_5;
    a2 = su0_2 + su0_6; a3 = su0_3 + su0_7;
    up[0] = (a0 + a2) + (a1 + a3);
    a0 = su1_0 + su1_4; a1 = su1_1 + su1_5;
    a2 = su1_2 + su1_6; a3 = su1_3 + su1_7;
    up[1] = (a0 + a2) + (a1 + a3);
}

#define GU_KERNEL_ARGS \
    float *gate_out, float *up_out, float *mid_out, \
    const BlockQ4K *standard_gate_w, const BlockQ4K *standard_up_w, \
    const NativeWeightTileBlock *native_gate_w, \
    const NativeWeightTileBlock *native_up_w, \
    const BlockQ8K *standard_a, const NativeQ8K *native_a, \
    const uint32_t *sorted_pairs, const uint32_t *offsets, \
    const uint32_t *counts, const uint32_t *tile_total, \
    const uint32_t *tile_experts, const uint32_t *tile_starts, \
    const float *router_weights, uint64_t gate_expert_bytes, \
    uint64_t gate_row_bytes, uint32_t xq_blocks, uint32_t expert_mid_dim, \
    uint32_t n_expert, uint32_t write_aux, float clamp

template <bool NATIVE_W, bool NATIVE_A, uint32_t STAGED_ROWS,
          uint32_t THREADS, bool SHIPPING_ROUTE, bool SCALAR_SLOTS>
__device__ __forceinline__ static void gu_consumer_body(
        unsigned char *shared, GU_KERNEL_ARGS) {
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t expert = tile_experts[tile];
    const uint32_t local_start = tile_starts[tile];
    const size_t a_bytes = (size_t)STAGED_ROWS * GU_IN_BLOCKS *
        (NATIVE_A ? sizeof(NativeQ8K) : sizeof(BlockQ8K));
    uint32_t *route = (uint32_t *)(shared + a_bytes);
    uint32_t *s_pair = route;
    uint32_t *s_tok = s_pair + GU_TILE_PAIRS;
    uint32_t *s_slot = SHIPPING_ROUTE ? s_tok + GU_TILE_PAIRS : NULL;
    uint32_t *s_np = SHIPPING_ROUTE ? s_slot + GU_TILE_PAIRS : NULL;
    if (threadIdx.x == 0u) {
        const uint32_t count = counts[expert];
        const uint32_t np = local_start < count ?
            ((count - local_start) < GU_TILE_PAIRS ?
             count - local_start : GU_TILE_PAIRS) : 0u;
        if (!SHIPPING_ROUTE) s_pair[GU_TILE_PAIRS - 1u] = 0u;
        for (uint32_t p = 0; p < np; p++) {
            const uint32_t local = local_start + p;
            const uint32_t pair = sorted_pairs[offsets[expert] + local];
            s_pair[p] = pair;
            s_tok[p] = pair / n_expert;
            if (SHIPPING_ROUTE) s_slot[p] = pair - s_tok[p] * n_expert;
        }
        if (SHIPPING_ROUTE) {
            *s_np = np;
        } else {
            /* Pair ids use only 12 bits.  Compact stage7 broadcasts np in the
             * otherwise-unused high byte of its last pair word. */
            s_pair[GU_TILE_PAIRS - 1u] =
                (s_pair[GU_TILE_PAIRS - 1u] & 0x00ffffffu) | (np << 24u);
        }
    }
    __syncthreads();
    const uint32_t active_np = SHIPPING_ROUTE ?
        *s_np : s_pair[GU_TILE_PAIRS - 1u] >> 24u;
    if (xq_blocks <= GU_IN_BLOCKS) {
        const uint32_t staged_rows = active_np < STAGED_ROWS ?
            active_np : STAGED_ROWS;
        const uint32_t block_words =
            (uint32_t)(sizeof(BlockQ8K) / sizeof(uint32_t));
        const uint32_t words_per_tok = xq_blocks * block_words;
        for (uint32_t i = threadIdx.x; i < staged_rows * words_per_tok;
             i += blockDim.x) {
            const uint32_t p = i / words_per_tok;
            const uint32_t word = i - p * words_per_tok;
            const uint32_t dst = p * GU_IN_BLOCKS * block_words + word;
            if (NATIVE_A) {
                ((uint32_t *)shared)[dst] = ((const uint32_t *)(native_a +
                    (uint64_t)s_tok[p] * xq_blocks))[word];
            } else {
                ((uint32_t *)shared)[dst] = ((const uint32_t *)(standard_a +
                    (uint64_t)s_tok[p] * xq_blocks))[word];
            }
        }
        __syncthreads();
    }
    const BlockQ8K *staged_standard = (const BlockQ8K *)shared;
    const NativeQ8K *staged_native = (const NativeQ8K *)shared;
    const uint32_t warps = THREADS / 32u;
    const uint32_t rows_per_iteration = warps * GU_OUT_TILE;
    for (uint32_t rr = 0; rr < GU_ROW_SPAN / rows_per_iteration; rr++) {
        const uint32_t row0 = blockIdx.x * GU_ROW_SPAN +
            rr * rows_per_iteration + warp * GU_OUT_TILE;
        if (row0 >= expert_mid_dim) continue;
        float gate[2], up[2];
        if (SCALAR_SLOTS) {
            gu_gate_up_pass_scalar<STAGED_ROWS>(
                native_gate_w, native_up_w, native_a, staged_native,
                s_tok, active_np, expert, row0, lane,
                xq_blocks, expert_mid_dim, gate, up);
        } else {
            gu_gate_up_pass<NATIVE_W, NATIVE_A, STAGED_ROWS>(
                standard_gate_w, standard_up_w, native_gate_w, native_up_w,
                standard_a, native_a,
                staged_standard, staged_native, s_tok, active_np,
                expert, row0, lane,
                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim,
                gate, up);
        }
        const uint32_t p = lane >> 2u;
        if (p >= active_np) continue;
#pragma unroll
        for (uint32_t e = 0; e < 2u; e++) {
            const uint32_t row = row0 + (lane & 3u) * 2u + e;
            if (row >= expert_mid_dim) continue;
            float g = gate[e], u = up[e];
            if (clamp > 1.0e-6f) {
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            const uint32_t pair = SHIPPING_ROUTE ?
                s_pair[p] : s_pair[p] & 0x00ffffffu;
            const uint64_t out = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[out] = g;
                up_out[out] = u;
            }
            const uint32_t slot = SHIPPING_ROUTE ? s_slot[p] :
                pair - s_tok[p] * n_expert;
            mid_out[out] = (g / (1.0f + expf(-g))) * u *
                router_weights[(uint64_t)s_tok[p] * n_expert + slot];
        }
    }
}

#define DEFINE_GU_KERNEL(NAME, NW, NA, STAGED, THREADS, SHIPPING_ROUTE, SCALAR) \
extern "C" __global__ void NAME(GU_KERNEL_ARGS) { \
    extern __shared__ __align__(16) unsigned char shared[]; \
    gu_consumer_body<NW, NA, STAGED, THREADS, SHIPPING_ROUTE, SCALAR>( \
        shared, gate_out, up_out, \
        mid_out, standard_gate_w, standard_up_w, native_gate_w, native_up_w, \
        standard_a, native_a, sorted_pairs, offsets, counts, tile_total, \
        tile_experts, tile_starts, router_weights, gate_expert_bytes, \
        gate_row_bytes, xq_blocks, expert_mid_dim, n_expert, \
        write_aux, clamp); \
}

DEFINE_GU_KERNEL(sm75_q4_gate_up_standard_kernel,
                 false, false, 8, 256, true, false)
DEFINE_GU_KERNEL(sm75_q4_gate_up_standard_warp16_kernel,
                 false, false, 8, 512, true, false)
DEFINE_GU_KERNEL(sm75_q4_gate_up_native_w_kernel,
                 true, false, 8, 256, true, false)
DEFINE_GU_KERNEL(sm75_q4_gate_up_native_aw_kernel,
                 true, true, 8, 256, true, false)
DEFINE_GU_KERNEL(sm75_q4_gate_up_native_aw_warp16_kernel,
                 true, true, 8, 512, true, false)
DEFINE_GU_KERNEL(sm75_q4_gate_up_native_aw_stage7_kernel,
                 true, true, 7, 256, false, false)
DEFINE_GU_KERNEL(sm75_q4_gate_up_native_aw_stage7_scalar_kernel,
                 true, true, 7, 256, false, true)
DEFINE_GU_KERNEL(sm75_q4_gate_up_native_aw_warp16_scalar_kernel,
                 true, true, 8, 512, true, true)

#undef DEFINE_GU_KERNEL
#undef GU_KERNEL_ARGS

static void gu_fill_weights(BlockQ4K *weights,
                            const GuScenarioSpec *spec, uint32_t matrix) {
    uint32_t state = 0x243f6a88u ^ spec->layer ^ (matrix * 0x9e3779b9u);
    const uint64_t blocks = (uint64_t)spec->active_experts *
        GU_MID_DIM * GU_IN_BLOCKS;
    for (uint64_t index = 0; index < blocks; index++) {
        BlockQ4K *w = weights + index;
        const uint32_t mode = (uint32_t)((index + matrix) & 7u);
        w->d = host_f32_to_f16(0.001953125f * (float)(1u + mode));
        w->dmin = (index & 1u) ?
            host_f32_to_f16(0.0009765625f *
                            (float)(1u + ((mode + matrix) & 3u))) : 0u;
        for (uint32_t i = 0; i < sizeof(w->scales); i++)
            w->scales[i] = (uint8_t)rng_next(&state);
        for (uint32_t i = 0; i < sizeof(w->qs); i++) {
            if (mode == 0u) w->qs[i] = 0xf0u;
            else if (mode == 1u) w->qs[i] = 0x0fu;
            else if (mode == 2u) w->qs[i] = 0x00u;
            else if (mode == 3u) w->qs[i] = 0xffu;
            else w->qs[i] = (uint8_t)rng_next(&state);
        }
    }
}

static void gu_fill_input(float *values, const GuScenarioSpec *spec) {
    uint32_t state = 0x85a308d3u ^ spec->layer;
    for (uint32_t token = 0; token < GU_TOKENS; token++) {
        for (uint32_t b = 0; b < GU_IN_BLOCKS; b++) {
            float *row = values + (uint64_t)token * GU_IN_DIM + b * QK_K;
            const bool negative_max = ((token + b) & 1u) != 0u;
            for (uint32_t i = 0; i < QK_K; i++) {
                const int raw = (int)(rng_next(&state) % 255u) - 127;
                row[i] = (float)raw / 127.0f;
            }
            row[0] = negative_max ? -2.0f : 2.0f;
            row[1] = negative_max ? 1.984375f : -1.984375f;
            row[2] = -2.0f;
            row[3] = 127.0f / 64.0f;
        }
    }
}

static int gu_validate_weight_pack_samples(
        const BlockQ4K *standard, const GuScenarioSpec *spec,
        const NativeWeightTileBlock *native_device, const char *matrix) {
    const uint64_t records = (uint64_t)spec->active_experts *
        GU_OUT_TILES * GU_IN_BLOCKS;
    const uint32_t samples = 96u;
    for (uint32_t s = 0; s < samples; s++) {
        const uint64_t record = s == 0u ? 0u :
            (s == 1u ? records - 1u :
             ((uint64_t)s * 0x9e3779b97f4a7c15ull) % records);
        NativeWeightTileBlock got;
        cuda_die(cudaMemcpy(&got, native_device + record, sizeof(got),
                            cudaMemcpyDeviceToHost),
                 "copy gate/up native-W sample");
        const uint32_t b = (uint32_t)(record % GU_IN_BLOCKS);
        const uint64_t tile_index = record / GU_IN_BLOCKS;
        const uint32_t out_tile = (uint32_t)(tile_index % GU_OUT_TILES);
        const uint32_t expert = (uint32_t)(tile_index / GU_OUT_TILES);
        const BlockQ4K *base = standard +
            ((uint64_t)expert * GU_MID_DIM + out_tile * GU_OUT_TILE) *
            GU_IN_BLOCKS;
        for (uint32_t row = 0; row < GU_OUT_TILE; row++) {
            if (memcmp(&got.hdr[row],
                       base + (uint64_t)row * GU_IN_BLOCKS + b,
                       sizeof(uint4))) {
                fprintf(stderr,
                        "error: %s native-W header mismatch record=%llu row=%u\n",
                        matrix, (unsigned long long)record, row);
                return 0;
            }
        }
        for (uint32_t j = 0; j < Q4_GROUPS; j++) {
            for (uint32_t lane = 0; lane < 32u; lane++) {
                const uint32_t row = lane >> 2u;
                const uint32_t lane4 = lane & 3u;
                const BlockQ4K *src = base +
                    (uint64_t)row * GU_IN_BLOCKS + b;
                const uint32_t off = (j >> 1u) * 32u + lane4 * 8u;
                const uint32_t q0 = load_u32_unaligned(src->qs + off);
                const uint32_t q1 = load_u32_unaligned(src->qs + off + 4u);
                const uint32_t expected = pack_nibbles_8(
                    q0, q1, (j & 1u) ? 4u : 0u);
                if (got.b[j][lane] != expected) {
                    fprintf(stderr,
                            "error: %s native-W payload mismatch record=%llu "
                            "j=%u lane=%u\n", matrix,
                            (unsigned long long)record, j, lane);
                    return 0;
                }
            }
        }
    }
    printf("%s_weight_pack_records_byte_validated=%u\n", matrix, samples);
    return 1;
}

static int gu_validate_activation_pack(const GuData *d) {
    const size_t bytes = (size_t)d->activation_blocks * sizeof(BlockQ8K);
    BlockQ8K *standard = (BlockQ8K *)malloc(bytes);
    NativeQ8K *native = (NativeQ8K *)malloc(bytes);
    if (!standard || !native) {
        fprintf(stderr, "error: gate/up activation validation allocation failed\n");
        free(native); free(standard);
        return 0;
    }
    cuda_die(cudaMemcpy(standard, d->standard_a.ptr, bytes,
                        cudaMemcpyDeviceToHost), "copy gate/up standard A");
    cuda_die(cudaMemcpy(native, d->native_a.ptr, bytes,
                        cudaMemcpyDeviceToHost), "copy gate/up native A");
    uint64_t negative_d = 0u, positive_d = 0u;
    for (uint64_t block = 0; block < d->activation_blocks; block++) {
        if (memcmp(&standard[block].d, &native[block].d, sizeof(float)) ||
            memcmp(standard[block].bsums, native[block].bsums,
                   sizeof(standard[block].bsums))) {
            fprintf(stderr, "error: native-A metadata mismatch block=%llu\n",
                    (unsigned long long)block);
            free(native); free(standard);
            return 0;
        }
        negative_d += standard[block].d < 0.0f;
        positive_d += standard[block].d > 0.0f;
        for (uint32_t j = 0; j < Q4_GROUPS; j++) {
            for (uint32_t lane4 = 0; lane4 < 4u; lane4++) {
                const uint32_t off = j * 32u + lane4 * 8u;
                const uint32_t x0 = load_u32_unaligned(standard[block].qs + off);
                const uint32_t x1 = load_u32_unaligned(standard[block].qs + off + 4u);
                if (native[block].low[j][lane4] != pack_nibbles_8(x0, x1, 0u) ||
                    native[block].high_signed[j][lane4] !=
                        pack_nibbles_8(x0, x1, 4u)) {
                    fprintf(stderr,
                            "error: native-A plane mismatch block=%llu j=%u lane=%u\n",
                            (unsigned long long)block, j, lane4);
                    free(native); free(standard);
                    return 0;
                }
            }
        }
        for (uint32_t g = 0; g < QK_K / 16; g++) {
            int sum = 0;
            for (uint32_t i = 0; i < 16u; i++)
                sum += standard[block].qs[g * 16u + i];
            if (sum != standard[block].bsums[g]) {
                fprintf(stderr, "error: Q8_K bsum mismatch block=%llu group=%u\n",
                        (unsigned long long)block, g);
                free(native); free(standard);
                return 0;
            }
        }
    }
    if (!negative_d || !positive_d) {
        fprintf(stderr, "error: Q8_K corpus omitted a d sign\n");
        free(native); free(standard);
        return 0;
    }
    printf("activation_blocks_byte_validated=%llu\n"
           "q8_negative_d_blocks=%llu\nq8_positive_d_blocks=%llu\n"
           "activation_pack_validation=exact\n",
           (unsigned long long)d->activation_blocks,
           (unsigned long long)negative_d,
           (unsigned long long)positive_d);
    free(native); free(standard);
    return 1;
}

static void gu_cleanup(GuData *d) {
    if (d->d_tile_starts) cudaFree(d->d_tile_starts);
    if (d->d_tile_experts) cudaFree(d->d_tile_experts);
    if (d->d_tile_total) cudaFree(d->d_tile_total);
    if (d->d_counts) cudaFree(d->d_counts);
    if (d->d_offsets) cudaFree(d->d_offsets);
    if (d->d_sorted_pairs) cudaFree(d->d_sorted_pairs);
    free_guarded(&d->mid_out);
    free_guarded(&d->up_out);
    free_guarded(&d->gate_out);
    free_guarded(&d->router_weights);
    free_guarded(&d->native_a);
    free_guarded(&d->standard_a);
    free_guarded(&d->native_up_w);
    free_guarded(&d->native_gate_w);
    free_guarded(&d->standard_up_w);
    free_guarded(&d->standard_gate_w);
    gu_free_metadata(&d->meta);
    memset(d, 0, sizeof(*d));
}

static int gu_setup(GuData *d, const GuScenarioSpec *spec) {
    memset(d, 0, sizeof(*d));
    d->spec = spec;
    if (!gu_build_metadata(spec, &d->meta)) return 0;
    d->weight_bytes = (uint64_t)spec->active_experts * GU_MID_DIM *
        GU_IN_BLOCKS * sizeof(BlockQ4K);
    d->activation_blocks = (uint64_t)GU_TOKENS * GU_IN_BLOCKS;
    d->output_values = (uint64_t)GU_PAIR_SLOTS * GU_MID_DIM;
    const uint64_t activation_bytes =
        d->activation_blocks * sizeof(BlockQ8K);
    const uint64_t output_bytes = d->output_values * sizeof(float);
    const uint64_t records = (uint64_t)spec->active_experts *
        GU_OUT_TILES * GU_IN_BLOCKS;
    const size_t input_bytes = (size_t)GU_TOKENS * GU_IN_DIM * sizeof(float);
    const size_t router_bytes = (size_t)GU_PAIR_SLOTS * sizeof(float);
    printf("scenario=%s\nlayer=%u\npair_count=%u\nactive_experts=%u\n"
           "recorded_tile16_count=%u\nsynthetic_tile8_count=%u\n"
           "synthetic_tile8_padded_slots=%u\n"
           "synthetic_tile8_histogram_caveat=true\n"
           "consumer_grid_x=4\nconsumer_grid_y_capacity=512\n"
           "consumer_excess_ctas_early_return=true\n"
           "input_dim=%u\nmid_dim=%u\nrow_span=%u\n"
           "standard_gate_weight_bytes=%llu\nstandard_up_weight_bytes=%llu\n"
           "native_gate_weight_bytes=%llu\nnative_up_weight_bytes=%llu\n"
           "activation_bytes=%llu\noutput_bytes_per_tensor=%llu\n",
           spec->name, spec->layer, spec->pairs, spec->active_experts,
           spec->recorded_tile16, spec->tile8, spec->padded8,
           GU_IN_DIM, GU_MID_DIM, GU_ROW_SPAN,
           (unsigned long long)d->weight_bytes,
           (unsigned long long)d->weight_bytes,
           (unsigned long long)d->weight_bytes,
           (unsigned long long)d->weight_bytes,
           (unsigned long long)activation_bytes,
           (unsigned long long)output_bytes);

    BlockQ4K *host_weights = (BlockQ4K *)malloc((size_t)d->weight_bytes);
    float *host_input = (float *)malloc(input_bytes);
    float *host_router = (float *)malloc(router_bytes);
    float *device_input = NULL;
    int ok = 0;
    if (!host_weights || !host_input || !host_router) {
        fprintf(stderr, "error: gate/up host allocation failed\n");
        goto done;
    }
    if (!alloc_guarded(&d->standard_gate_w, d->weight_bytes) ||
        !alloc_guarded(&d->standard_up_w, d->weight_bytes) ||
        !alloc_guarded(&d->native_gate_w, d->weight_bytes) ||
        !alloc_guarded(&d->native_up_w, d->weight_bytes) ||
        !alloc_guarded(&d->standard_a, activation_bytes) ||
        !alloc_guarded(&d->native_a, activation_bytes) ||
        !alloc_guarded(&d->router_weights, router_bytes) ||
        !alloc_guarded(&d->gate_out, output_bytes) ||
        !alloc_guarded(&d->up_out, output_bytes) ||
        !alloc_guarded(&d->mid_out, output_bytes)) goto done;

    gu_fill_weights(host_weights, spec, 0u);
    cuda_die(cudaMemcpy(d->standard_gate_w.ptr, host_weights,
                        (size_t)d->weight_bytes, cudaMemcpyHostToDevice),
             "copy standard gate weights");
    sm75_q4_gate_up_pack_w_kernel<<<
        (unsigned int)((records + WARPS_PER_CTA - 1u) / WARPS_PER_CTA),
        THREADS_PER_CTA>>>(
        (NativeWeightTileBlock *)d->native_gate_w.ptr,
        (const BlockQ4K *)d->standard_gate_w.ptr, spec->active_experts);
    cuda_die(cudaGetLastError(), "launch native gate-W pack");
    cuda_die(cudaDeviceSynchronize(), "synchronize native gate-W pack");
    if (!gu_validate_weight_pack_samples(
            host_weights, spec,
            (const NativeWeightTileBlock *)d->native_gate_w.ptr, "gate"))
        goto done;

    gu_fill_weights(host_weights, spec, 1u);
    cuda_die(cudaMemcpy(d->standard_up_w.ptr, host_weights,
                        (size_t)d->weight_bytes, cudaMemcpyHostToDevice),
             "copy standard up weights");
    sm75_q4_gate_up_pack_w_kernel<<<
        (unsigned int)((records + WARPS_PER_CTA - 1u) / WARPS_PER_CTA),
        THREADS_PER_CTA>>>(
        (NativeWeightTileBlock *)d->native_up_w.ptr,
        (const BlockQ4K *)d->standard_up_w.ptr, spec->active_experts);
    cuda_die(cudaGetLastError(), "launch native up-W pack");
    cuda_die(cudaDeviceSynchronize(), "synchronize native up-W pack");
    if (!gu_validate_weight_pack_samples(
            host_weights, spec,
            (const NativeWeightTileBlock *)d->native_up_w.ptr, "up"))
        goto done;
    printf("weight_pack_validation=exact\n");

    gu_fill_input(host_input, spec);
    for (uint32_t token = 0; token < GU_TOKENS; token++) {
        for (uint32_t slot = 0; slot < GU_SELECTED; slot++) {
            const uint32_t i = token * GU_SELECTED + slot;
            host_router[i] = 0.0625f +
                (float)((token * 13u + slot * 7u) & 31u) / 256.0f;
        }
    }
    cuda_die(cudaMalloc((void **)&device_input, input_bytes),
             "allocate gate/up float input");
    cuda_die(cudaMemcpy(device_input, host_input, input_bytes,
                        cudaMemcpyHostToDevice), "copy gate/up float input");
    sm75_q4_down_quantize_q8k_kernel<<<
        dim3(GU_IN_BLOCKS, GU_TOKENS), THREADS_PER_CTA>>>(
        (BlockQ8K *)d->standard_a.ptr, device_input, GU_IN_DIM, GU_TOKENS);
    cuda_die(cudaGetLastError(), "launch gate/up Q8_K quantizer");
    sm75_q4_gate_up_pack_a_kernel<<<
        (unsigned int)((d->activation_blocks + WARPS_PER_CTA - 1u) /
                       WARPS_PER_CTA), THREADS_PER_CTA>>>(
        (NativeQ8K *)d->native_a.ptr,
        (const BlockQ8K *)d->standard_a.ptr, d->activation_blocks);
    cuda_die(cudaGetLastError(), "launch gate/up native-A pack");
    cuda_die(cudaDeviceSynchronize(), "synchronize gate/up activation setup");
    if (!gu_validate_activation_pack(d)) goto done;
    cuda_die(cudaMemcpy(d->router_weights.ptr, host_router, router_bytes,
                        cudaMemcpyHostToDevice), "copy router weights");

#define GU_ALLOC_COPY(NAME, HOST, COUNT) do { \
    cuda_die(cudaMalloc((void **)&d->NAME, (size_t)(COUNT) * sizeof(uint32_t)), \
             "allocate " #NAME); \
    cuda_die(cudaMemcpy(d->NAME, HOST, (size_t)(COUNT) * sizeof(uint32_t), \
                        cudaMemcpyHostToDevice), "copy " #NAME); \
} while (0)
    GU_ALLOC_COPY(d_sorted_pairs, d->meta.sorted_pairs, spec->pairs);
    GU_ALLOC_COPY(d_offsets, d->meta.offsets, spec->active_experts + 1u);
    GU_ALLOC_COPY(d_counts, d->meta.counts, spec->active_experts);
    GU_ALLOC_COPY(d_tile_experts, d->meta.tile_experts, spec->tile8);
    GU_ALLOC_COPY(d_tile_starts, d->meta.tile_starts, spec->tile8);
    GU_ALLOC_COPY(d_tile_total, &spec->tile8, 1u);
#undef GU_ALLOC_COPY
    ok = validate_canary(&d->standard_gate_w, "standard-gate-W") &&
         validate_canary(&d->standard_up_w, "standard-up-W") &&
         validate_canary(&d->native_gate_w, "native-gate-W") &&
         validate_canary(&d->native_up_w, "native-up-W") &&
         validate_canary(&d->standard_a, "standard-A") &&
         validate_canary(&d->native_a, "native-A") &&
         validate_canary(&d->router_weights, "router-weights");
done:
    if (device_input) cudaFree(device_input);
    free(host_router);
    free(host_input);
    free(host_weights);
    if (!ok) gu_cleanup(d);
    return ok;
}

static int gu_is_combined(GuVariant variant) {
    return variant == GU_NATIVE_AW_COMBINED ||
           variant == GU_NATIVE_AW_WARP16_COMBINED ||
           variant == GU_NATIVE_AW_STAGE7_COMBINED ||
           variant == GU_NATIVE_AW_STAGE7_SCALAR_COMBINED ||
           variant == GU_NATIVE_AW_WARP16_SCALAR_COMBINED;
}

static GuVariant gu_consumer_variant(GuVariant variant) {
    switch (variant) {
        case GU_NATIVE_AW_COMBINED: return GU_NATIVE_AW_CONSUMER;
        case GU_NATIVE_AW_WARP16_COMBINED:
            return GU_NATIVE_AW_WARP16_CONSUMER;
        case GU_NATIVE_AW_STAGE7_COMBINED:
            return GU_NATIVE_AW_STAGE7_CONSUMER;
        case GU_NATIVE_AW_STAGE7_SCALAR_COMBINED:
            return GU_NATIVE_AW_STAGE7_SCALAR_CONSUMER;
        case GU_NATIVE_AW_WARP16_SCALAR_COMBINED:
            return GU_NATIVE_AW_WARP16_SCALAR_CONSUMER;
        default: return variant;
    }
}

static void gu_launch(const GuData *d, GuVariant requested, uint32_t write_aux) {
    if (requested == GU_PACK_A || gu_is_combined(requested)) {
        sm75_q4_gate_up_pack_a_kernel<<<
            (unsigned int)((d->activation_blocks + WARPS_PER_CTA - 1u) /
                           WARPS_PER_CTA), THREADS_PER_CTA>>>(
            (NativeQ8K *)d->native_a.ptr,
            (const BlockQ8K *)d->standard_a.ptr, d->activation_blocks);
        cuda_die(cudaGetLastError(), "launch timed gate/up native-A pack");
        if (requested == GU_PACK_A) return;
    }
    const GuVariant variant = gu_consumer_variant(requested);
    const dim3 grid(GU_MID_DIM / GU_ROW_SPAN, GU_TOKENS, 1u);
#define GU_LAUNCH_ARGS \
    (float *)d->gate_out.ptr, (float *)d->up_out.ptr, \
    (float *)d->mid_out.ptr, \
    (const BlockQ4K *)d->standard_gate_w.ptr, \
    (const BlockQ4K *)d->standard_up_w.ptr, \
    (const NativeWeightTileBlock *)d->native_gate_w.ptr, \
    (const NativeWeightTileBlock *)d->native_up_w.ptr, \
    (const BlockQ8K *)d->standard_a.ptr, \
    (const NativeQ8K *)d->native_a.ptr, d->d_sorted_pairs, d->d_offsets, \
    d->d_counts, d->d_tile_total, d->d_tile_experts, d->d_tile_starts, \
    (const float *)d->router_weights.ptr, \
    (uint64_t)GU_MID_DIM * GU_IN_BLOCKS * sizeof(BlockQ4K), \
    (uint64_t)GU_IN_BLOCKS * sizeof(BlockQ4K), \
    GU_IN_BLOCKS, GU_MID_DIM, GU_SELECTED, write_aux, 1.25f
    switch (variant) {
        case GU_STANDARD:
            sm75_q4_gate_up_standard_kernel<<<
                grid, 256, gu_shared_bytes(0, 8, 1)>>>(GU_LAUNCH_ARGS);
            break;
        case GU_STANDARD_WARP16:
            sm75_q4_gate_up_standard_warp16_kernel<<<
                grid, 512, gu_shared_bytes(0, 8, 1)>>>(GU_LAUNCH_ARGS);
            break;
        case GU_NATIVE_W:
            sm75_q4_gate_up_native_w_kernel<<<
                grid, 256, gu_shared_bytes(0, 8, 1)>>>(GU_LAUNCH_ARGS);
            break;
        case GU_NATIVE_AW_CONSUMER:
            sm75_q4_gate_up_native_aw_kernel<<<
                grid, 256, gu_shared_bytes(1, 8, 1)>>>(GU_LAUNCH_ARGS);
            break;
        case GU_NATIVE_AW_WARP16_CONSUMER:
            sm75_q4_gate_up_native_aw_warp16_kernel<<<
                grid, 512, gu_shared_bytes(1, 8, 1)>>>(GU_LAUNCH_ARGS);
            break;
        case GU_NATIVE_AW_STAGE7_CONSUMER:
            sm75_q4_gate_up_native_aw_stage7_kernel<<<
                grid, 256, gu_shared_bytes(1, 7)>>>(GU_LAUNCH_ARGS);
            break;
        case GU_NATIVE_AW_STAGE7_SCALAR_CONSUMER:
            sm75_q4_gate_up_native_aw_stage7_scalar_kernel<<<
                grid, 256, gu_shared_bytes(1, 7)>>>(GU_LAUNCH_ARGS);
            break;
        case GU_NATIVE_AW_WARP16_SCALAR_CONSUMER:
            sm75_q4_gate_up_native_aw_warp16_scalar_kernel<<<
                grid, 512, gu_shared_bytes(1, 8, 1)>>>(GU_LAUNCH_ARGS);
            break;
        default:
            fprintf(stderr, "error: variant has no consumer: %s\n",
                    gu_variant_names[variant]);
            exit(2);
    }
#undef GU_LAUNCH_ARGS
    cuda_die(cudaGetLastError(), "launch Q4 gate/up consumer");
}

static int gu_validate_output_tensor(
        const GuData *d, const float *expected, const float *actual,
        const char *variant, const char *tensor) {
    uint8_t owned[GU_PAIR_SLOTS] = {0};
    for (uint32_t i = 0; i < d->spec->pairs; i++)
        owned[d->meta.sorted_pairs[i]] = 1u;
    for (uint32_t pair = 0; pair < GU_PAIR_SLOTS; pair++) {
        for (uint32_t row = 0; row < GU_MID_DIM; row++) {
            const uint64_t index = (uint64_t)pair * GU_MID_DIM + row;
            uint32_t actual_bits;
            memcpy(&actual_bits, actual + index, sizeof(actual_bits));
            if (!owned[pair]) {
                if (actual_bits != 0xffffffffu) {
                    fprintf(stderr,
                            "error: %s %s wrote unowned pair=%u row=%u bits=%08x\n",
                            variant, tensor, pair, row, actual_bits);
                    return 0;
                }
                continue;
            }
            if (!isfinite(actual[index])) {
                fprintf(stderr,
                        "error: %s %s left owned pair=%u row=%u nonfinite\n",
                        variant, tensor, pair, row);
                return 0;
            }
            if (expected) {
                uint32_t expected_bits;
                memcpy(&expected_bits, expected + index, sizeof(expected_bits));
                if (expected_bits != actual_bits) {
                    fprintf(stderr,
                            "error: %s mismatch tensor=%s pair=%u row=%u "
                            "expected=%.9g[%08x] actual=%.9g[%08x]\n",
                            variant, tensor, pair, row,
                            (double)expected[index], expected_bits,
                            (double)actual[index], actual_bits);
                    return 0;
                }
            }
        }
    }
    return 1;
}

static void gu_poison_outputs(GuData *d) {
    const size_t bytes = (size_t)d->output_values * sizeof(float);
    cuda_die(cudaMemset(d->gate_out.ptr, 0xff, bytes), "poison gate output");
    cuda_die(cudaMemset(d->up_out.ptr, 0xff, bytes), "poison up output");
    cuda_die(cudaMemset(d->mid_out.ptr, 0xff, bytes), "poison mid output");
}

static void gu_copy_outputs(const GuData *d, float *all) {
    const size_t bytes = (size_t)d->output_values * sizeof(float);
    cuda_die(cudaMemcpy(all, d->gate_out.ptr, bytes, cudaMemcpyDeviceToHost),
             "copy gate output");
    cuda_die(cudaMemcpy(all + d->output_values, d->up_out.ptr, bytes,
                        cudaMemcpyDeviceToHost), "copy up output");
    cuda_die(cudaMemcpy(all + 2u * d->output_values, d->mid_out.ptr, bytes,
                        cudaMemcpyDeviceToHost), "copy mid output");
}

static int gu_run_correctness(GuData *d) {
    const uint64_t all_values = 3u * d->output_values;
    if (all_values > SIZE_MAX / sizeof(float)) return 0;
    float *expected = (float *)malloc((size_t)all_values * sizeof(float));
    float *actual = (float *)malloc((size_t)all_values * sizeof(float));
    if (!expected || !actual) {
        fprintf(stderr, "error: gate/up output validation allocation failed\n");
        free(actual); free(expected);
        return 0;
    }
    gu_poison_outputs(d);
    gu_launch(d, GU_STANDARD, 1u);
    cuda_die(cudaDeviceSynchronize(), "synchronize shipping gate/up reference");
    gu_copy_outputs(d, expected);
    const char *tensor_names[] = {"gate", "up", "mid"};
    for (uint32_t t = 0; t < 3u; t++) {
        if (!gu_validate_output_tensor(
                d, NULL, expected + (uint64_t)t * d->output_values,
                gu_variant_names[GU_STANDARD], tensor_names[t])) {
            free(actual); free(expected);
            return 0;
        }
    }
    const GuVariant checked[] = {
        GU_STANDARD_WARP16,
        GU_NATIVE_W,
        GU_NATIVE_AW_CONSUMER,
        GU_NATIVE_AW_WARP16_CONSUMER,
        GU_NATIVE_AW_STAGE7_CONSUMER,
        GU_NATIVE_AW_COMBINED,
        GU_NATIVE_AW_WARP16_COMBINED,
        GU_NATIVE_AW_STAGE7_COMBINED,
        GU_NATIVE_AW_STAGE7_SCALAR_CONSUMER,
        GU_NATIVE_AW_WARP16_SCALAR_CONSUMER,
        GU_NATIVE_AW_STAGE7_SCALAR_COMBINED,
        GU_NATIVE_AW_WARP16_SCALAR_COMBINED,
    };
    for (uint32_t v = 0; v < sizeof(checked) / sizeof(checked[0]); v++) {
        gu_poison_outputs(d);
        gu_launch(d, checked[v], 1u);
        cuda_die(cudaDeviceSynchronize(), "synchronize gate/up correctness");
        gu_copy_outputs(d, actual);
        for (uint32_t t = 0; t < 3u; t++) {
            if (!gu_validate_output_tensor(
                    d, expected + (uint64_t)t * d->output_values,
                    actual + (uint64_t)t * d->output_values,
                    gu_variant_names[checked[v]], tensor_names[t])) {
                free(actual); free(expected);
                return 0;
            }
        }
    }
    const int canaries_ok =
        validate_canary(&d->standard_gate_w, "standard-gate-W") &&
        validate_canary(&d->standard_up_w, "standard-up-W") &&
        validate_canary(&d->native_gate_w, "native-gate-W") &&
        validate_canary(&d->native_up_w, "native-up-W") &&
        validate_canary(&d->standard_a, "standard-A") &&
        validate_canary(&d->native_a, "native-A") &&
        validate_canary(&d->router_weights, "router-weights") &&
        validate_canary(&d->gate_out, "gate-output") &&
        validate_canary(&d->up_out, "up-output") &&
        validate_canary(&d->mid_out, "mid-output");
    printf("gate_values_compared=%llu\nup_values_compared=%llu\n"
           "mid_values_compared=%llu\n"
           "unowned_output_poison_validation=exact\n"
           "output_reference=harness-standard-production-shaped\n"
           "output_validation=bit_exact\ncorrectness_canaries=%s\n"
           "correctness_status=%s\n",
           (unsigned long long)d->spec->pairs * GU_MID_DIM,
           (unsigned long long)d->spec->pairs * GU_MID_DIM,
           (unsigned long long)d->spec->pairs * GU_MID_DIM,
           canaries_ok ? "ok" : "failed", canaries_ok ? "ok" : "failed");
    free(actual); free(expected);
    return canaries_ok;
}

template <typename Kernel>
static void gu_print_resource(const char *name, Kernel kernel,
                              int threads, size_t dynamic_shared) {
    cudaFuncAttributes attr;
    cuda_die(cudaFuncGetAttributes(&attr, kernel), "query gate/up resource");
    int blocks = 0;
    cuda_die(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &blocks, kernel, threads, dynamic_shared),
              "query gate/up occupancy");
    int device = 0, max_threads_per_sm = 0;
    cuda_die(cudaGetDevice(&device), "query resource-report device");
    cuda_die(cudaDeviceGetAttribute(
                 &max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor,
                 device),
             "query maximum threads per SM");
    const int active_warps = blocks * (threads / 32);
    const double theoretical_occupancy = max_threads_per_sm > 0 ?
        100.0 * (double)(blocks * threads) / (double)max_threads_per_sm : 0.0;
    printf("resource_%s_threads=%d\n"
           "resource_%s_registers_per_thread=%d\n"
           "resource_%s_static_shared_bytes=%zu\n"
           "resource_%s_dynamic_shared_bytes=%zu\n"
           "resource_%s_local_bytes_per_thread=%zu\n"
           "resource_%s_active_blocks_per_sm=%d\n"
           "resource_%s_active_warps_per_sm=%d\n"
           "resource_%s_theoretical_occupancy_percent=%.2f\n",
           name, threads, name, attr.numRegs, name, attr.sharedSizeBytes,
           name, dynamic_shared, name, attr.localSizeBytes, name, blocks,
           name, active_warps, name, theoretical_occupancy);
}

static void gu_print_resources(void) {
    gu_print_resource("standard",
        sm75_q4_gate_up_standard_kernel, 256, gu_shared_bytes(0, 8, 1));
    gu_print_resource("standard_warp16",
        sm75_q4_gate_up_standard_warp16_kernel, 512,
        gu_shared_bytes(0, 8, 1));
    gu_print_resource("native_w",
        sm75_q4_gate_up_native_w_kernel, 256, gu_shared_bytes(0, 8, 1));
    gu_print_resource("native_aw",
        sm75_q4_gate_up_native_aw_kernel, 256, gu_shared_bytes(1, 8, 1));
    gu_print_resource("native_aw_warp16",
        sm75_q4_gate_up_native_aw_warp16_kernel, 512,
        gu_shared_bytes(1, 8, 1));
    gu_print_resource("native_aw_stage7",
        sm75_q4_gate_up_native_aw_stage7_kernel, 256, gu_shared_bytes(1, 7));
    gu_print_resource("native_aw_stage7_scalar",
        sm75_q4_gate_up_native_aw_stage7_scalar_kernel, 256,
        gu_shared_bytes(1, 7));
    gu_print_resource("native_aw_warp16_scalar",
        sm75_q4_gate_up_native_aw_warp16_scalar_kernel, 512,
        gu_shared_bytes(1, 8, 1));
    gu_print_resource("pack_a", sm75_q4_gate_up_pack_a_kernel, 256, 0u);
    gu_print_resource("pack_w", sm75_q4_gate_up_pack_w_kernel, 256, 0u);
    printf("resource_standard_binary_identity=false\n"
           "resource_standard_reference=production-shaped-reimplementation\n");
    printf("resource_spill_note=verify_local_spills_with_ptxas_and_cuobjdump\n");
}

static int gu_compare_double(const void *a, const void *b) {
    const double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static int gu_run_benchmark(GuData *d, uint32_t rounds,
                            uint32_t launches) {
    double *samples = (double *)calloc(
        (size_t)GU_VARIANT_COUNT * rounds, sizeof(double));
    double *sorted = (double *)malloc((size_t)rounds * sizeof(double));
    if (!samples || !sorted) {
        fprintf(stderr, "error: gate/up benchmark allocation failed\n");
        free(sorted); free(samples);
        return 0;
    }
    uint32_t base_order[GU_VARIANT_COUNT];
    for (uint32_t i = 0; i < GU_VARIANT_COUNT; i++) base_order[i] = i;
    uint32_t seed = 0xc001d00du ^ d->spec->layer;
    for (uint32_t i = GU_VARIANT_COUNT - 1u; i > 0u; i--) {
        const uint32_t j = rng_next(&seed) % (i + 1u);
        const uint32_t tmp = base_order[i];
        base_order[i] = base_order[j];
        base_order[j] = tmp;
    }
    cudaEvent_t start, stop;
    cuda_die(cudaEventCreate(&start), "create gate/up benchmark start");
    cuda_die(cudaEventCreate(&stop), "create gate/up benchmark stop");
    for (int v = 0; v < GU_VARIANT_COUNT; v++)
        gu_launch(d, (GuVariant)v, 0u);
    cuda_die(cudaDeviceSynchronize(), "synchronize gate/up warmup");
    float fastest_consumer_ms = 0.0f;
    for (int v = 0; v < GU_VARIANT_COUNT; v++) {
        if (v == GU_PACK_A) continue;
        cuda_die(cudaEventRecord(start), "record calibration start");
        gu_launch(d, (GuVariant)v, 0u);
        cuda_die(cudaEventRecord(stop), "record calibration stop");
        cuda_die(cudaEventSynchronize(stop), "synchronize calibration");
        float ms = 0.0f;
        cuda_die(cudaEventElapsedTime(&ms, start, stop),
                 "measure calibration");
        if (ms > 0.0f && (fastest_consumer_ms == 0.0f ||
                          ms < fastest_consumer_ms))
            fastest_consumer_ms = ms;
    }
    if (fastest_consumer_ms <= 0.0f) {
        fprintf(stderr, "error: consumer calibration returned no duration\n");
        cudaEventDestroy(stop); cudaEventDestroy(start);
        free(sorted); free(samples);
        return 0;
    }
    uint32_t sample_launches = (uint32_t)ceil(100.0 / fastest_consumer_ms);
    if (sample_launches < launches) sample_launches = launches;
    printf("benchmark_random_seed=%u\n"
           "benchmark_position_balance_period=%u\n"
           "benchmark_warmup_launches_per_variant=1\n"
           "benchmark_requested_launches=%u\n"
           "benchmark_effective_launches=%u\n"
           "benchmark_consumer_sample_target_ms=100\n"
           "benchmark_write_aux=0\ncorrectness_write_aux=1\n"
           "activation_quantization_common_and_excluded=1\n"
           "weight_layout_pack_offline_and_excluded=1\n"
           "combined_event_includes_pack_a_and_consumer=1\n"
           "pack_only_relative_not_consumer_comparable=1\n",
           0xc001d00du ^ d->spec->layer, GU_VARIANT_COUNT,
           launches, sample_launches);
    printf("benchmark_samples_begin\n"
           "scenario,round,sample_slot,variant,total_ms,us_per_launch,relative_speed\n");
    for (uint32_t round = 0; round < rounds; round++) {
        uint32_t order[GU_VARIANT_COUNT];
        for (uint32_t slot = 0; slot < GU_VARIANT_COUNT; slot++)
            order[slot] = base_order[(slot + round) % GU_VARIANT_COUNT];
        for (uint32_t slot = 0; slot < GU_VARIANT_COUNT; slot++) {
            const GuVariant variant = (GuVariant)order[slot];
            cuda_die(cudaEventRecord(start), "record gate/up benchmark start");
            for (uint32_t i = 0; i < sample_launches; i++)
                gu_launch(d, variant, 0u);
            cuda_die(cudaEventRecord(stop), "record gate/up benchmark stop");
            cuda_die(cudaEventSynchronize(stop), "sync gate/up benchmark sample");
            float ms = 0.0f;
            cuda_die(cudaEventElapsedTime(&ms, start, stop),
                     "measure gate/up benchmark sample");
            samples[(size_t)variant * rounds + round] = ms;
        }
        const double standard =
            samples[(size_t)GU_STANDARD * rounds + round];
        for (uint32_t slot = 0; slot < GU_VARIANT_COUNT; slot++) {
            const GuVariant variant = (GuVariant)order[slot];
            const double ms = samples[(size_t)variant * rounds + round];
            printf("%s,%u,%u,%s,%.6f,%.6f,%.6f\n",
                   d->spec->name, round + 1u, slot + 1u,
                   gu_variant_names[variant], ms,
                   1000.0 * ms / sample_launches,
                   standard / ms);
        }
    }
    printf("benchmark_samples_end\nmedian_summary_begin\n"
           "scenario,variant,median_total_ms,median_us_per_launch,relative_speed\n");
    double medians[GU_VARIANT_COUNT];
    for (int v = 0; v < GU_VARIANT_COUNT; v++) {
        memcpy(sorted, samples + (size_t)v * rounds,
               (size_t)rounds * sizeof(double));
        qsort(sorted, rounds, sizeof(double), gu_compare_double);
        medians[v] = (rounds & 1u) ? sorted[rounds / 2u] :
            0.5 * (sorted[rounds / 2u - 1u] + sorted[rounds / 2u]);
    }
    for (int v = 0; v < GU_VARIANT_COUNT; v++) {
        printf("%s,%s,%.6f,%.6f,%.6f\n",
               d->spec->name, gu_variant_names[v], medians[v],
               1000.0 * medians[v] / sample_launches,
               medians[GU_STANDARD] / medians[v]);
    }
    printf("median_summary_end\nbenchmark_status=ok\n");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    free(sorted); free(samples);
    return 1;
}

static int gu_run_profile(GuData *d, GuVariant variant,
                          uint32_t launches) {
    gu_launch(d, variant, 0u);
    cuda_die(cudaDeviceSynchronize(), "synchronize gate/up profile warmup");
    printf("profile_variant=%s\nprofile_warmup_launches=1\n"
           "profile_capture_launches=%u\nprofile_capture_begin=1\n",
           gu_variant_names[variant], launches);
    fflush(stdout);
    for (uint32_t i = 0; i < launches; i++) gu_launch(d, variant, 0u);
    cuda_die(cudaDeviceSynchronize(), "synchronize gate/up profile capture");
    printf("profile_capture_end=1\nprofile_status=ok\n");
    return 1;
}

static void gu_usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [--device N] [--scenario early|late] MODE [OPTIONS]\n"
        "\nModes:\n"
        "  --correctness-only\n"
        "  --benchmark-only [--rounds N] [--launches N]\n"
        "  --profile VARIANT [--launches N]\n"
        "  --resources-only\n"
        "\nVariants:\n", argv0);
    for (int i = 0; i < GU_VARIANT_COUNT; i++)
        fprintf(stderr, "  %s\n", gu_variant_names[i]);
    fprintf(stderr,
        "\nCorrectness defaults to early+late. Benchmark/profile default "
        "to early. No model is opened.\n");
}

int main(int argc, char **argv) {
    enum GuMode { GU_ALL, GU_CORRECTNESS, GU_BENCHMARK, GU_PROFILE,
                  GU_RESOURCES };
    GuMode mode = GU_ALL;
    int mode_explicit = 0, device = 0;
    const GuScenarioSpec *selected = NULL;
    GuVariant profile_variant = GU_STANDARD;
    uint32_t rounds = GU_VARIANT_COUNT, launches = 3u;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            device = parse_device(argv[++i]);
        } else if (!strcmp(argv[i], "--scenario") && i + 1 < argc) {
            selected = gu_find_scenario(argv[++i]);
            if (!selected) { gu_usage(argv[0]); return 2; }
        } else if (!strcmp(argv[i], "--correctness-only")) {
            if (mode_explicit++) { gu_usage(argv[0]); return 2; }
            mode = GU_CORRECTNESS;
        } else if (!strcmp(argv[i], "--benchmark-only")) {
            if (mode_explicit++) { gu_usage(argv[0]); return 2; }
            mode = GU_BENCHMARK;
        } else if (!strcmp(argv[i], "--profile") && i + 1 < argc) {
            if (mode_explicit++) { gu_usage(argv[0]); return 2; }
            mode = GU_PROFILE;
            profile_variant = gu_find_variant(argv[++i]);
        } else if (!strcmp(argv[i], "--resources-only")) {
            if (mode_explicit++) { gu_usage(argv[0]); return 2; }
            mode = GU_RESOURCES;
        } else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) {
            rounds = parse_u32(argv[++i], "rounds");
        } else if (!strcmp(argv[i], "--launches") && i + 1 < argc) {
            launches = parse_u32(argv[++i], "launches");
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            gu_usage(argv[0]);
            return 0;
        } else {
            gu_usage(argv[0]);
            return 2;
        }
    }
    cuda_die(cudaSetDevice(device), "select gate/up CUDA device");
    cudaDeviceProp prop;
    cuda_die(cudaGetDeviceProperties(&prop, device),
             "query gate/up CUDA device");
    if (prop.major != 7 || prop.minor != 5) {
        fprintf(stderr,
                "error: gate/up harness requires compute capability 7.5; got %d.%d\n",
                prop.major, prop.minor);
        return 1;
    }
    printf("harness=sm75-q4-gate-up-native\ndevice=%d\ndevice_name=%s\n"
           "compute_capability=%d.%d\nmodel_required=false\n"
           "model_opened=false\n"
           "exact_reference=harness-standard-production-shaped\n",
           device, prop.name, prop.major, prop.minor);
    if (!run_adversarial_host_gates(
            mode != GU_PROFILE && mode != GU_RESOURCES)) return 1;
    gu_print_resources();
    if (mode == GU_RESOURCES) {
        printf("resource_status=ok\nharness_status=ok\n");
        return 0;
    }
    const size_t begin = selected ?
        (size_t)(selected - gu_scenarios) : 0u;
    const size_t end = selected ? begin + 1u :
        (mode == GU_CORRECTNESS || mode == GU_ALL ?
         sizeof(gu_scenarios) / sizeof(gu_scenarios[0]) : 1u);
    int ok = 1;
    for (size_t s = begin; s < end && ok; s++) {
        GuData data;
        if (!gu_setup(&data, &gu_scenarios[s])) return 1;
        if (mode == GU_CORRECTNESS || mode == GU_ALL)
            ok = gu_run_correctness(&data);
        if (ok && (mode == GU_BENCHMARK || mode == GU_ALL))
            ok = gu_run_benchmark(&data, rounds, launches);
        if (ok && mode == GU_PROFILE)
            ok = gu_run_profile(&data, profile_variant, launches);
        gu_cleanup(&data);
    }
    if (ok) printf("harness_status=ok\n");
    return ok ? 0 : 1;
}
