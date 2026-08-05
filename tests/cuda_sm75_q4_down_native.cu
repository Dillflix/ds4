/*
 * Production-shaped SM75 Q4_K down-layout experiment.
 *
 * This executable is intentionally isolated from ds4's runtime dispatcher.
 * It compares the shipping SM75 tile16 arithmetic with two size-neutral
 * packed-INT4 layouts:
 *
 *   standard             row-major Q4_K + row-major Q8_K
 *   native-w             MMA-lane-native Q4_K + row-major Q8_K
 *   native-aw-consumer   MMA-lane-native Q4_K + nibble-plane Q8_K
 *   native-aw-combined   Q8_K pack plus native-aw consumer per launch
 *   pack-a               Q8_K pack only
 *
 * The Q4_K headers, packed scale/min metadata, signed Q8_K scale, bsums min
 * correction, b-slot accumulation, and final float reduction tree match the
 * production source.  This file is built without --use_fast_math, so an
 * exact result means bit-exact against this harness's standard kernel, not a
 * claim about a separately built production binary.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_sm75_native_q4_histograms.h"

enum {
    QK_K = 256,
    Q4_GROUPS = 8,
    MIDQ_BLOCKS_MAX = 8,
    MID_DIM = QK_K * MIDQ_BLOCKS_MAX,
    OUT_DIM = 4096,
    OUT_TILE = 8,
    OUT_TILES = OUT_DIM / OUT_TILE,
    TILE_PAIRS = 16,
    ROW_SPAN = 512,
    WARPS_PER_CTA = 8,
    THREADS_PER_CTA = 256,
    TOTAL_PAIR_SLOTS = 512 * 6,
};

struct BlockQ4K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
};

struct BlockQ8K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};

/* One record replaces eight row-major Q4_K blocks without changing bytes.
 * hdr[row] preserves d/dmin/scales verbatim. b[group][lane] is exactly the
 * m8n8k32 B fragment owned by that lane for the 8-row output tile. */
struct __align__(16) NativeWeightTileBlock {
    uint4 hdr[OUT_TILE];
    uint32_t b[Q4_GROUPS][32];
};

/* Size-neutral per-row Q8_K representation. Each 32-value group contains
 * four packed u4 words for the low plane and four packed s4 words for the
 * signed high plane. d and bsums retain their original bits. */
struct NativeQ8K {
    float d;
    uint32_t low[Q4_GROUPS][4];
    uint32_t high_signed[Q4_GROUPS][4];
    int16_t bsums[QK_K / 16];
};

static_assert(sizeof(BlockQ4K) == 144, "Q4_K block must remain 144 bytes");
static_assert(sizeof(BlockQ8K) == 292, "Q8_K block must remain 292 bytes");
static_assert(sizeof(NativeQ8K) == 292,
              "native Q8_K must not acquire 292->304 padding");
static_assert(alignof(NativeQ8K) == 4,
              "native Q8_K alignment must remain size-neutral");
static_assert(sizeof(NativeWeightTileBlock) == 8 * sizeof(BlockQ4K),
              "native Q4_K tile must be size-neutral");

struct ScenarioSpec {
    const char *name;
    uint32_t layer;
    uint32_t pairs;
    uint32_t active_experts;
    uint32_t tiles;
    uint32_t padded;
    const uint16_t *expert_counts;
};

/* Recorded home-half production aggregates from the fixed prompt audit. */
static const ScenarioSpec kScenarios[] = {
    {"early", 3u, 1879u, 99u, 183u, 1049u, NULL},
    {"late", 36u, 2186u, 76u, 189u, 838u, NULL},
    /* Cost-planner tile16 populations from the exact production histogram.
     * Residual 8/4 routes are deliberately excluded from this experiment. */
    {"real-early", 3u, 1617u, 39u, 106u, 79u,
     ds4_native_q4_early_counts},
    {"real-late", 36u, 1957u, 23u, 126u, 59u,
     ds4_native_q4_late_counts},
};

enum Variant {
    VAR_STANDARD,
    VAR_NATIVE_W,
    VAR_NATIVE_AW_CONSUMER,
    VAR_NATIVE_AW_COMBINED,
    VAR_PACK_A,
    VAR_NATIVE_AW_NSPLIT4,
    VAR_NATIVE_AW_NSPLIT8,
    VARIANT_COUNT,
};

static const char *const kVariantNames[VARIANT_COUNT] = {
    "standard",
    "native-w",
    "native-aw-consumer",
    "native-aw-combined",
    "pack-a",
    "native-aw-nsplit4",
    "native-aw-nsplit8",
};

static void cuda_die(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return;
    fprintf(stderr, "error: %s: %s\n", what, cudaGetErrorString(err));
    exit(2);
}

__host__ __device__ __forceinline__ static uint32_t load_u32_unaligned(
        const void *ptr) {
    uint32_t value;
#if defined(__CUDA_ARCH__)
    value = *(const uint32_t *)ptr;
#else
    memcpy(&value, ptr, sizeof(value));
#endif
    return value;
}

__host__ __device__ __forceinline__ static uint32_t pack_nibbles_8(
        uint32_t lo_words, uint32_t hi_words, uint32_t shift) {
    uint32_t out = 0;
#pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        out |= ((lo_words >> (8u * i + shift)) & 0x0fu) << (4u * i);
        out |= ((hi_words >> (8u * i + shift)) & 0x0fu)
            << (4u * (i + 4u));
    }
    return out;
}

__device__ __forceinline__ static void mma_m8n8k16_s8(
        int32_t &c0, int32_t &c1, uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
#else
    (void)c0; (void)c1; (void)a; (void)b;
#endif
}

__device__ __forceinline__ static void mma_m8n8k32_u4_u4(
        int32_t &c0, int32_t &c1, uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile(
        "mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32 "
        "{%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
#else
    (void)c0; (void)c1; (void)a; (void)b;
#endif
}

__device__ __forceinline__ static void mma_m8n8k32_s4_u4(
        int32_t &c0, int32_t &c1, uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile(
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32 "
        "{%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
#else
    (void)c0; (void)c1; (void)a; (void)b;
#endif
}

__device__ __forceinline__ static float f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

__host__ __device__ __forceinline__ static void q4_get_scale_min(
        uint32_t j, const uint8_t *scales, uint8_t *d, uint8_t *m) {
    if (j < 4u) {
        *d = scales[j] & 63u;
        *m = scales[j + 4u] & 63u;
    } else {
        *d = (scales[j + 4u] & 0x0fu) |
             ((scales[j - 4u] >> 6u) << 4u);
        *m = (scales[j + 4u] >> 4u) |
             ((scales[j] >> 6u) << 4u);
    }
}

/* Common production Q8_K quantizer, retained here so signed d, extrema and
 * bsums are exercised rather than supplying synthetic pre-quantized bytes. */
extern "C" __global__ void sm75_q4_down_quantize_q8k_kernel(
        BlockQ8K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    const uint32_t b = blockIdx.x;
    const uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * QK_K;
    BlockQ8K *yb = out + (uint64_t)row * (in_dim / QK_K) + b;
    __shared__ float abs_part[QK_K];
    __shared__ float val_part[QK_K];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    const uint32_t tid = threadIdx.x;
    const float v = xr[tid];
    abs_part[tid] = fabsf(v);
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    if (abs_part[0] == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        yb->qs[tid] = 0;
        if (tid < QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    int qv = (int)lrintf(iscale_s * xr[tid]);
    qv = qv > 127 ? 127 : (qv < -128 ? -128 : qv);
    yb->qs[tid] = (int8_t)qv;
    __syncthreads();
    if (tid < QK_K / 16) {
        int sum = 0;
#pragma unroll
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}

/* Eight warps convert eight Q8_K blocks per CTA. No standard and native A
 * copies are staged together in shared memory; this is a pure global layout
 * transform and is timed separately and in the combined variant. */
extern "C" __global__ void sm75_q4_down_pack_a_kernel(
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
    if (lane == 0) dst->d = src->d;
    if (lane < QK_K / 16) dst->bsums[lane] = src->bsums[lane];
}

/* Offline, size-neutral weight-layout builder. Its cost is deliberately not
 * included in inference timing: production would generate/store this layout
 * during quantization. */
extern "C" __global__ void sm75_q4_down_pack_w_kernel(
        NativeWeightTileBlock *out, const BlockQ4K *in,
        uint32_t n_experts, uint32_t midq_blocks) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint64_t record = (uint64_t)blockIdx.x * WARPS_PER_CTA + warp;
    const uint64_t records =
        (uint64_t)n_experts * OUT_TILES * midq_blocks;
    if (record >= records) return;
    const uint32_t b = (uint32_t)(record % midq_blocks);
    const uint64_t tile_index = record / midq_blocks;
    const uint32_t out_tile = (uint32_t)(tile_index % OUT_TILES);
    const uint32_t expert = (uint32_t)(tile_index / OUT_TILES);
    const uint64_t row_blocks = (uint64_t)OUT_DIM * midq_blocks;
    const BlockQ4K *base = in + (uint64_t)expert * row_blocks +
        (uint64_t)out_tile * OUT_TILE * midq_blocks;
    NativeWeightTileBlock *dst = out + record;
    if (lane < OUT_TILE) dst->hdr[lane] =
        *(const uint4 *)(base + (uint64_t)lane * midq_blocks + b);
    const uint32_t row = lane >> 2u;
    const uint32_t lane4 = lane & 3u;
    const BlockQ4K *src = base + (uint64_t)row * midq_blocks + b;
#pragma unroll
    for (uint32_t j = 0; j < Q4_GROUPS; j++) {
        const uint32_t off = (j >> 1u) * 32u + lane4 * 8u;
        const uint32_t q0 = load_u32_unaligned(src->qs + off);
        const uint32_t q1 = load_u32_unaligned(src->qs + off + 4u);
        dst->b[j][lane] = pack_nibbles_8(
            q0, q1, (j & 1u) ? 4u : 0u);
    }
}

template <int V>
__device__ __forceinline__ static void q4_down_consumer_body(
        float *down_out,
        const BlockQ4K *standard_w,
        const NativeWeightTileBlock *native_w,
        const BlockQ8K *standard_a,
        const NativeQ8K *native_a,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t expert = tile_experts[tile];
    const uint32_t local_start = tile_starts[tile];
    enum { WORDS_PER_Q8 = sizeof(BlockQ8K) / sizeof(uint32_t) };
    __shared__ uint32_t sxq_words
        [TILE_PAIRS * MIDQ_BLOCKS_MAX * WORDS_PER_Q8];
    __shared__ uint32_t s_pair[TILE_PAIRS], s_np;
    if (threadIdx.x == 0) {
        uint32_t np = 0;
        for (; np < TILE_PAIRS; np++) {
            const uint32_t local_pair = local_start + np;
            if (local_pair >= counts[expert]) break;
            s_pair[np] = sorted_pairs[offsets[expert] + local_pair];
        }
        s_np = np;
    }
    __syncthreads();
    const uint32_t np = s_np;
    const uint32_t words_per_tok = midq_blocks * WORDS_PER_Q8;
    for (uint32_t i = threadIdx.x;
         i < TILE_PAIRS * words_per_tok;
         i += blockDim.x) {
        const uint32_t p = i / words_per_tok;
        const uint32_t w = i - p * words_per_tok;
        const uint32_t pair = p < np ? s_pair[p] : 0u;
        const uint32_t *src = V == VAR_NATIVE_AW_CONSUMER
            ? (const uint32_t *)(native_a +
                (uint64_t)pair * midq_blocks)
            : (const uint32_t *)(standard_a +
                (uint64_t)pair * midq_blocks);
        sxq_words[p * words_per_tok + w] = p < np ? src[w] : 0u;
    }
    __syncthreads();

    const BlockQ8K *sxq_standard = (const BlockQ8K *)sxq_words;
    const NativeQ8K *sxq_native = (const NativeQ8K *)sxq_words;
    const uint32_t mtok_a = lane >> 2u;
    const uint32_t mtok_b = mtok_a + 8u;
    const uint32_t n0 = (lane & 3u) * 2u;
    const uint64_t standard_row_blocks = (uint64_t)out_dim * midq_blocks;
    for (uint32_t rr = 0; rr < ROW_SPAN / 64u; rr++) {
        const uint32_t row0 =
            blockIdx.x * ROW_SPAN + rr * 64u + warp * OUT_TILE;
        if (row0 >= out_dim) continue;
        float s0[8] = {0,0,0,0,0,0,0,0};
        float s1[8] = {0,0,0,0,0,0,0,0};
        float s2[8] = {0,0,0,0,0,0,0,0};
        float s3[8] = {0,0,0,0,0,0,0,0};
#pragma unroll
        for (uint32_t b = 0; b < MIDQ_BLOCKS_MAX; b++) {
            if (b >= midq_blocks) continue;
            uint4 hdr0, hdr1;
            uint32_t w8[8];
            const NativeWeightTileBlock *native_block = NULL;
            if (V == VAR_STANDARD) {
                const BlockQ4K *wrow = standard_w +
                    (uint64_t)expert * standard_row_blocks +
                    (uint64_t)row0 * midq_blocks;
                hdr0 = *(const uint4 *)(wrow + (uint64_t)n0 * midq_blocks + b);
                hdr1 = *(const uint4 *)(wrow +
                    (uint64_t)(n0 + 1u) * midq_blocks + b);
                const uint32_t *wqw = (const uint32_t *)(
                    wrow[(uint64_t)(lane >> 2u) * midq_blocks + b].qs);
#pragma unroll
                for (uint32_t k = 0; k < 8u; k++)
                    w8[k] = wqw[k * 4u + (lane & 3u)];
            } else {
                native_block = native_w +
                    (((uint64_t)expert * OUT_TILES + row0 / OUT_TILE) *
                     midq_blocks + b);
                hdr0 = native_block->hdr[n0];
                hdr1 = native_block->hdr[n0 + 1u];
            }

            const BlockQ8K *xa = sxq_standard +
                (uint64_t)mtok_a * midq_blocks + b;
            const BlockQ8K *xb = sxq_standard +
                (uint64_t)mtok_b * midq_blocks + b;
            const NativeQ8K *na = sxq_native +
                (uint64_t)mtok_a * midq_blocks + b;
            const NativeQ8K *nb = sxq_native +
                (uint64_t)mtok_b * midq_blocks + b;
            int ia0 = 0, ia1 = 0, ib0 = 0, ib1 = 0;
            int ma0 = 0, ma1 = 0, mb0 = 0, mb1 = 0;
#pragma unroll
            for (uint32_t j = 0; j < Q4_GROUPS; j++) {
                int32_t ca0 = 0, ca1 = 0, cb0 = 0, cb1 = 0;
                if (V == VAR_STANDARD) {
                    const int shift = (j & 1u) ? 4 : 0;
#pragma unroll
                    for (uint32_t h = 0; h < 2u; h++) {
                        const uint32_t off = j * 32u + h * 16u +
                            (lane & 3u) * 4u;
                        const uint32_t aa =
                            load_u32_unaligned(xa->qs + off);
                        const uint32_t ab =
                            load_u32_unaligned(xb->qs + off);
                        const uint32_t ww =
                            (w8[(j >> 1u) * 2u + h] >> shift) &
                            0x0f0f0f0fu;
                        mma_m8n8k16_s8(ca0, ca1, aa, ww);
                        mma_m8n8k16_s8(cb0, cb1, ab, ww);
                    }
                } else {
                    const uint32_t bf = native_block->b[j][lane];
                    uint32_t al, ah, bl, bh;
                    if (V == VAR_NATIVE_AW_CONSUMER) {
                        al = na->low[j][lane & 3u];
                        ah = na->high_signed[j][lane & 3u];
                        bl = nb->low[j][lane & 3u];
                        bh = nb->high_signed[j][lane & 3u];
                    } else {
                        const uint32_t off = j * 32u +
                            (lane & 3u) * 8u;
                        const uint32_t a0 =
                            load_u32_unaligned(xa->qs + off);
                        const uint32_t a1 =
                            load_u32_unaligned(xa->qs + off + 4u);
                        const uint32_t b0 =
                            load_u32_unaligned(xb->qs + off);
                        const uint32_t b1 =
                            load_u32_unaligned(xb->qs + off + 4u);
                        al = pack_nibbles_8(a0, a1, 0u);
                        ah = pack_nibbles_8(a0, a1, 4u);
                        bl = pack_nibbles_8(b0, b1, 0u);
                        bh = pack_nibbles_8(b0, b1, 4u);
                    }
                    int32_t cal0 = 0, cal1 = 0, cah0 = 0, cah1 = 0;
                    int32_t cbl0 = 0, cbl1 = 0, cbh0 = 0, cbh1 = 0;
                    mma_m8n8k32_u4_u4(cal0, cal1, al, bf);
                    mma_m8n8k32_s4_u4(cah0, cah1, ah, bf);
                    mma_m8n8k32_u4_u4(cbl0, cbl1, bl, bf);
                    mma_m8n8k32_s4_u4(cbh0, cbh1, bh, bf);
                    ca0 = cal0 + 16 * cah0;
                    ca1 = cal1 + 16 * cah1;
                    cb0 = cbl0 + 16 * cbh0;
                    cb1 = cbl1 + 16 * cbh1;
                }
                uint8_t sc0, sm0, sc1, sm1;
                q4_get_scale_min(j, (const uint8_t *)&hdr0.y, &sc0, &sm0);
                q4_get_scale_min(j, (const uint8_t *)&hdr1.y, &sc1, &sm1);
                const int bs_a = V == VAR_NATIVE_AW_CONSUMER
                    ? (int)na->bsums[2u * j] +
                      (int)na->bsums[2u * j + 1u]
                    : (int)xa->bsums[2u * j] +
                      (int)xa->bsums[2u * j + 1u];
                const int bs_b = V == VAR_NATIVE_AW_CONSUMER
                    ? (int)nb->bsums[2u * j] +
                      (int)nb->bsums[2u * j + 1u]
                    : (int)xb->bsums[2u * j] +
                      (int)xb->bsums[2u * j + 1u];
                ia0 += (int)sc0 * ca0;
                ia1 += (int)sc1 * ca1;
                ib0 += (int)sc0 * cb0;
                ib1 += (int)sc1 * cb1;
                ma0 += (int)sm0 * bs_a;
                ma1 += (int)sm1 * bs_a;
                mb0 += (int)sm0 * bs_b;
                mb1 += (int)sm1 * bs_b;
            }
            const float yd_a = V == VAR_NATIVE_AW_CONSUMER ? na->d : xa->d;
            const float yd_b = V == VAR_NATIVE_AW_CONSUMER ? nb->d : xb->d;
            const float d0 = f16_to_f32((uint16_t)(hdr0.x & 0xffffu));
            const float m0 = f16_to_f32((uint16_t)(hdr0.x >> 16u));
            const float d1 = f16_to_f32((uint16_t)(hdr1.x & 0xffffu));
            const float m1 = f16_to_f32((uint16_t)(hdr1.x >> 16u));
            s0[b] += yd_a * d0 * (float)ia0 -
                     yd_a * m0 * (float)ma0;
            s1[b] += yd_a * d1 * (float)ia1 -
                     yd_a * m1 * (float)ma1;
            s2[b] += yd_b * d0 * (float)ib0 -
                     yd_b * m0 * (float)mb0;
            s3[b] += yd_b * d1 * (float)ib1 -
                     yd_b * m1 * (float)mb1;
        }
        float out4[4];
        float a0 = s0[0] + s0[4], a1 = s0[1] + s0[5];
        float a2 = s0[2] + s0[6], a3 = s0[3] + s0[7];
        out4[0] = (a0 + a2) + (a1 + a3);
        a0 = s1[0] + s1[4]; a1 = s1[1] + s1[5];
        a2 = s1[2] + s1[6]; a3 = s1[3] + s1[7];
        out4[1] = (a0 + a2) + (a1 + a3);
        a0 = s2[0] + s2[4]; a1 = s2[1] + s2[5];
        a2 = s2[2] + s2[6]; a3 = s2[3] + s2[7];
        out4[2] = (a0 + a2) + (a1 + a3);
        a0 = s3[0] + s3[4]; a1 = s3[1] + s3[5];
        a2 = s3[2] + s3[6]; a3 = s3[3] + s3[7];
        out4[3] = (a0 + a2) + (a1 + a3);
#pragma unroll
        for (uint32_t e = 0; e < 4u; e++) {
            const uint32_t p = e < 2u ? mtok_a : mtok_b;
            const uint32_t row = row0 + n0 + (e & 1u);
            if (p < np && row < out_dim)
                down_out[(uint64_t)s_pair[p] * out_dim + row] = out4[e];
        }
    }
    (void)n_expert;
}

extern "C" __global__ void sm75_q4_down_standard_kernel(
        float *out, const BlockQ4K *standard_w,
        const NativeWeightTileBlock *native_w,
        const BlockQ8K *standard_a, const NativeQ8K *native_a,
        const uint32_t *sorted_pairs, const uint32_t *offsets,
        const uint32_t *counts, const uint32_t *tile_total,
        const uint32_t *tile_experts, const uint32_t *tile_starts,
        uint32_t midq_blocks, uint32_t out_dim, uint32_t n_expert) {
    q4_down_consumer_body<VAR_STANDARD>(
        out, standard_w, native_w, standard_a, native_a, sorted_pairs,
        offsets, counts, tile_total, tile_experts, tile_starts,
        midq_blocks, out_dim, n_expert);
}

extern "C" __global__ void sm75_q4_down_native_w_kernel(
        float *out, const BlockQ4K *standard_w,
        const NativeWeightTileBlock *native_w,
        const BlockQ8K *standard_a, const NativeQ8K *native_a,
        const uint32_t *sorted_pairs, const uint32_t *offsets,
        const uint32_t *counts, const uint32_t *tile_total,
        const uint32_t *tile_experts, const uint32_t *tile_starts,
        uint32_t midq_blocks, uint32_t out_dim, uint32_t n_expert) {
    q4_down_consumer_body<VAR_NATIVE_W>(
        out, standard_w, native_w, standard_a, native_a, sorted_pairs,
        offsets, counts, tile_total, tile_experts, tile_starts,
        midq_blocks, out_dim, n_expert);
}

extern "C" __global__ void sm75_q4_down_native_aw_kernel(
        float *out, const BlockQ4K *standard_w,
        const NativeWeightTileBlock *native_w,
        const BlockQ8K *standard_a, const NativeQ8K *native_a,
        const uint32_t *sorted_pairs, const uint32_t *offsets,
        const uint32_t *counts, const uint32_t *tile_total,
        const uint32_t *tile_experts, const uint32_t *tile_starts,
        uint32_t midq_blocks, uint32_t out_dim, uint32_t n_expert) {
    q4_down_consumer_body<VAR_NATIVE_AW_CONSUMER>(
        out, standard_w, native_w, standard_a, native_a, sorted_pairs,
        offsets, counts, tile_total, tile_experts, tile_starts,
        midq_blocks, out_dim, n_expert);
}

/* Compact N-split experiment.  One CTA owns one real 16-route expert tile
 * and one output-row macro tile.  Only one 256-K activation slab is staged
 * at a time (16 * 292 = 4672 bytes); WARPS independent warps own disjoint
 * native 8-row weight tiles.  Every packed weight fragment feeds both route
 * halves before the next fragment is loaded. */
#define NS_DOWN_SLOT4_DECL(S) \
    float ns0_##S=0.0f,ns1_##S=0.0f,ns2_##S=0.0f,ns3_##S=0.0f
#define NS_DOWN_SLOT4_SET(S,V0,V1,V2,V3) \
    case S: ns0_##S=(V0);ns1_##S=(V1);ns2_##S=(V2);ns3_##S=(V3);break
#define NS_DOWN_REDUCE(P,O) do { \
    const float _a0=P##_0+P##_4,_a1=P##_1+P##_5; \
    const float _a2=P##_2+P##_6,_a3=P##_3+P##_7; \
    (O)=(_a0+_a2)+(_a1+_a3); \
} while (0)

template <uint32_t WARPS>
__global__ static void sm75_q4_down_native_aw_nsplit_kernel(
        float *down_out, const NativeWeightTileBlock *native_w,
        const NativeQ8K *native_a,
        const uint32_t *sorted_pairs, const uint32_t *offsets,
        const uint32_t *counts, const uint32_t *tile_total,
        const uint32_t *tile_experts, const uint32_t *tile_starts,
        uint32_t midq_blocks, uint32_t out_dim) {
    static_assert(WARPS == 4u || WARPS == 8u,
                  "N-split sweep is intentionally bounded to 4/8 warps");
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t expert = tile_experts[tile];
    const uint32_t local_start = tile_starts[tile];
    __shared__ NativeQ8K sxq[TILE_PAIRS];
    __shared__ uint32_t s_pair[TILE_PAIRS], s_np;
    if (threadIdx.x == 0u) {
        const uint32_t count = counts[expert];
        const uint32_t remaining = local_start < count ?
            count - local_start : 0u;
        const uint32_t np = remaining < TILE_PAIRS ? remaining : TILE_PAIRS;
        for (uint32_t p = 0; p < np; p++)
            s_pair[p] = sorted_pairs[offsets[expert] + local_start + p];
        s_np = np;
    }
    __syncthreads();
    const uint32_t np = s_np;
    const uint32_t p0 = lane >> 2u, p1 = p0 + 8u;
    const bool have0 = p0 < np, have1 = p1 < np;
    const uint32_t row0 = blockIdx.x * (WARPS * OUT_TILE) +
                          warp * OUT_TILE;
    const bool row_active = row0 < out_dim;
    const uint32_t compute_row0 = row_active ? row0 : 0u;
    const uint32_t n0 = (lane & 3u) * 2u;
    NS_DOWN_SLOT4_DECL(0); NS_DOWN_SLOT4_DECL(1);
    NS_DOWN_SLOT4_DECL(2); NS_DOWN_SLOT4_DECL(3);
    NS_DOWN_SLOT4_DECL(4); NS_DOWN_SLOT4_DECL(5);
    NS_DOWN_SLOT4_DECL(6); NS_DOWN_SLOT4_DECL(7);
    enum { WORDS_PER_NATIVE_Q8 = sizeof(NativeQ8K) / sizeof(uint32_t) };
    for (uint32_t b = 0; b < midq_blocks; b++) {
        for (uint32_t i = threadIdx.x;
             i < TILE_PAIRS * WORDS_PER_NATIVE_Q8;
             i += blockDim.x) {
            const uint32_t p = i / WORDS_PER_NATIVE_Q8;
            const uint32_t word = i - p * WORDS_PER_NATIVE_Q8;
            const uint32_t pair = p < np ? s_pair[p] : 0u;
            const uint32_t value = p < np ?
                ((const uint32_t *)(native_a +
                    (uint64_t)pair * midq_blocks + b))[word] : 0u;
            ((uint32_t *)&sxq[p])[word] = value;
        }
        __syncthreads();
        float v00=0.0f,v01=0.0f,v10=0.0f,v11=0.0f;
        if (row_active) {
            const NativeWeightTileBlock *w = native_w +
                (((uint64_t)expert * OUT_TILES +
                  compute_row0 / OUT_TILE) * midq_blocks + b);
            const uint4 h0 = w->hdr[n0], h1 = w->hdr[n0 + 1u];
            const NativeQ8K *a0 = &sxq[p0], *a1 = &sxq[p1];
            int i00=0,i01=0,i10=0,i11=0;
            int m00=0,m01=0,m10=0,m11=0;
#pragma unroll
            for (uint32_t j = 0; j < Q4_GROUPS; j++) {
                const uint32_t wf = w->b[j][lane];
                int32_t l00=0,l01=0,h00=0,h01=0;
                int32_t l10=0,l11=0,h10=0,h11=0;
                mma_m8n8k32_u4_u4(l00,l01,a0->low[j][lane&3u],wf);
                mma_m8n8k32_s4_u4(h00,h01,a0->high_signed[j][lane&3u],wf);
                mma_m8n8k32_u4_u4(l10,l11,a1->low[j][lane&3u],wf);
                mma_m8n8k32_s4_u4(h10,h11,a1->high_signed[j][lane&3u],wf);
                const int c00=l00+16*h00,c01=l01+16*h01;
                const int c10=l10+16*h10,c11=l11+16*h11;
                const int bs0=have0?
                    (int)a0->bsums[2u*j]+(int)a0->bsums[2u*j+1u]:0;
                const int bs1=have1?
                    (int)a1->bsums[2u*j]+(int)a1->bsums[2u*j+1u]:0;
                uint8_t sc0,mn0,sc1,mn1;
                q4_get_scale_min(j,(const uint8_t *)&h0.y,&sc0,&mn0);
                q4_get_scale_min(j,(const uint8_t *)&h1.y,&sc1,&mn1);
                i00+=(int)sc0*c00;i01+=(int)sc1*c01;
                i10+=(int)sc0*c10;i11+=(int)sc1*c11;
                m00+=(int)mn0*bs0;m01+=(int)mn1*bs0;
                m10+=(int)mn0*bs1;m11+=(int)mn1*bs1;
            }
            const float d0=f16_to_f32((uint16_t)h0.x);
            const float z0=f16_to_f32((uint16_t)(h0.x>>16u));
            const float d1=f16_to_f32((uint16_t)h1.x);
            const float z1=f16_to_f32((uint16_t)(h1.x>>16u));
            const float yd0=have0?a0->d:0.0f,yd1=have1?a1->d:0.0f;
            v00=yd0*d0*(float)i00-yd0*z0*(float)m00;
            v01=yd0*d1*(float)i01-yd0*z1*(float)m01;
            v10=yd1*d0*(float)i10-yd1*z0*(float)m10;
            v11=yd1*d1*(float)i11-yd1*z1*(float)m11;
        }
        switch (b & 7u) {
            NS_DOWN_SLOT4_SET(0,v00,v01,v10,v11);
            NS_DOWN_SLOT4_SET(1,v00,v01,v10,v11);
            NS_DOWN_SLOT4_SET(2,v00,v01,v10,v11);
            NS_DOWN_SLOT4_SET(3,v00,v01,v10,v11);
            NS_DOWN_SLOT4_SET(4,v00,v01,v10,v11);
            NS_DOWN_SLOT4_SET(5,v00,v01,v10,v11);
            NS_DOWN_SLOT4_SET(6,v00,v01,v10,v11);
            NS_DOWN_SLOT4_SET(7,v00,v01,v10,v11);
        }
        __syncthreads();
    }
    float r0,r1,r2,r3;
    NS_DOWN_REDUCE(ns0,r0); NS_DOWN_REDUCE(ns1,r1);
    NS_DOWN_REDUCE(ns2,r2); NS_DOWN_REDUCE(ns3,r3);
    if (row_active) {
        if (have0) {
            const uint32_t row = row0 + n0;
            if (row < out_dim)
                down_out[(uint64_t)s_pair[p0]*out_dim+row]=r0;
            if (row+1u < out_dim)
                down_out[(uint64_t)s_pair[p0]*out_dim+row+1u]=r1;
        }
        if (have1) {
            const uint32_t row = row0 + n0;
            if (row < out_dim)
                down_out[(uint64_t)s_pair[p1]*out_dim+row]=r2;
            if (row+1u < out_dim)
                down_out[(uint64_t)s_pair[p1]*out_dim+row+1u]=r3;
        }
    }
}

#undef NS_DOWN_REDUCE
#undef NS_DOWN_SLOT4_SET
#undef NS_DOWN_SLOT4_DECL

struct HostMetadata {
    uint32_t *counts;
    uint32_t *offsets;
    uint32_t *sorted_pairs;
    uint32_t *tile_experts;
    uint32_t *tile_starts;
};

struct DeviceBuffer {
    uint8_t *base;
    uint8_t *ptr;
    uint64_t bytes;
};

struct ScenarioData {
    const ScenarioSpec *spec;
    HostMetadata host_meta;
    DeviceBuffer standard_w;
    DeviceBuffer native_w;
    DeviceBuffer standard_a;
    DeviceBuffer native_a;
    DeviceBuffer output;
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

enum { CANARY_BYTES = 4096 };

static uint32_t rng_next(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static uint32_t gcd_u32(uint32_t a, uint32_t b) {
    while (b) {
        const uint32_t r = a % b;
        a = b;
        b = r;
    }
    return a;
}

static uint16_t host_f32_to_f16(float value) {
    const __half h = __float2half_rn(value);
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

static const ScenarioSpec *find_scenario(const char *name) {
    for (size_t i = 0; i < sizeof(kScenarios) / sizeof(kScenarios[0]); i++)
        if (strcmp(name, kScenarios[i].name) == 0) return &kScenarios[i];
    return NULL;
}

static Variant find_variant(const char *name) {
    for (int i = 0; i < VARIANT_COUNT; i++)
        if (strcmp(name, kVariantNames[i]) == 0) return (Variant)i;
    fprintf(stderr, "error: unknown profile variant: %s\n", name);
    exit(2);
}

static int alloc_guarded(DeviceBuffer *buf, uint64_t bytes) {
    memset(buf, 0, sizeof(*buf));
    if (bytes > SIZE_MAX - 2u * CANARY_BYTES) return 0;
    cudaError_t err = cudaMalloc((void **)&buf->base,
                                 (size_t)bytes + 2u * CANARY_BYTES);
    if (err != cudaSuccess) {
        fprintf(stderr, "error: cudaMalloc(%llu): %s\n",
                (unsigned long long)bytes, cudaGetErrorString(err));
        return 0;
    }
    buf->ptr = buf->base + CANARY_BYTES;
    buf->bytes = bytes;
    cuda_die(cudaMemset(buf->base, 0xa5,
                        (size_t)bytes + 2u * CANARY_BYTES),
             "initialize guarded buffer");
    return 1;
}

static void free_guarded(DeviceBuffer *buf) {
    if (buf->base) cudaFree(buf->base);
    memset(buf, 0, sizeof(*buf));
}

static int validate_canary(const DeviceBuffer *buf, const char *name) {
    uint8_t guard[CANARY_BYTES];
    cuda_die(cudaMemcpy(guard, buf->base, CANARY_BYTES,
                        cudaMemcpyDeviceToHost), "copy leading canary");
    for (uint32_t i = 0; i < CANARY_BYTES; i++) {
        if (guard[i] != 0xa5u) {
            fprintf(stderr, "error: %s leading canary changed at %u\n",
                    name, i);
            return 0;
        }
    }
    cuda_die(cudaMemcpy(guard, buf->ptr + buf->bytes, CANARY_BYTES,
                        cudaMemcpyDeviceToHost), "copy trailing canary");
    for (uint32_t i = 0; i < CANARY_BYTES; i++) {
        if (guard[i] != 0xa5u) {
            fprintf(stderr, "error: %s trailing canary changed at %u\n",
                    name, i);
            return 0;
        }
    }
    return 1;
}

static void free_host_metadata(HostMetadata *m) {
    free(m->tile_starts);
    free(m->tile_experts);
    free(m->sorted_pairs);
    free(m->offsets);
    free(m->counts);
    memset(m, 0, sizeof(*m));
}

/* Reconstruct an expert-major tile table with the exact observed aggregate.
 * Every declared active expert owns at least one pair; the deterministic
 * affine permutation prevents the activation rows within a tile from being
 * artificially contiguous in pair order. */
static int build_metadata(const ScenarioSpec *spec, HostMetadata *m) {
    memset(m, 0, sizeof(*m));
    const uint32_t active = spec->active_experts;
    uint32_t *tile_units = (uint32_t *)calloc(active, sizeof(uint32_t));
    m->counts = (uint32_t *)calloc(active, sizeof(uint32_t));
    m->offsets = (uint32_t *)calloc(active + 1u, sizeof(uint32_t));
    m->sorted_pairs = (uint32_t *)malloc(
        (size_t)spec->pairs * sizeof(uint32_t));
    m->tile_experts = (uint32_t *)malloc(
        (size_t)spec->tiles * sizeof(uint32_t));
    m->tile_starts = (uint32_t *)malloc(
        (size_t)spec->tiles * sizeof(uint32_t));
    if (!tile_units || !m->counts || !m->offsets || !m->sorted_pairs ||
        !m->tile_experts || !m->tile_starts) {
        free(tile_units);
        free_host_metadata(m);
        return 0;
    }
    if (spec->expert_counts) {
        /* Preserve the exact per-expert production population for every
         * tile16 handled by the cost planner.  Experts containing only 8/4
         * residual work are intentionally omitted and surviving expert ids
         * are compacted; id values do not affect the kernel geometry. */
        uint32_t compact = 0u;
        for (uint32_t source = 0; source < 128u; source++) {
            const uint32_t count = spec->expert_counts[source];
            const uint32_t remainder = count & 15u;
            const uint32_t candidate = (count & ~15u) +
                (remainder > 8u ? remainder : 0u);
            if (!candidate) continue;
            if (compact >= active) break;
            m->counts[compact] = candidate;
            tile_units[compact] = (candidate + 15u) / 16u;
            compact++;
        }
        if (compact != active) {
            fprintf(stderr, "error: %s real histogram active mismatch\n",
                    spec->name);
            free(tile_units);
            free_host_metadata(m);
            return 0;
        }
    } else {
        for (uint32_t e = 0; e < active; e++) tile_units[e] = 1u;
        for (uint32_t i = 0; i < spec->tiles - active; i++)
            tile_units[(i * 37u + 11u) % active]++;
        uint32_t minimum = 0;
        for (uint32_t e = 0; e < active; e++) {
            m->counts[e] = (tile_units[e] - 1u) * TILE_PAIRS + 1u;
            minimum += m->counts[e];
        }
        if (minimum > spec->pairs ||
            spec->pairs > spec->tiles * TILE_PAIRS) {
            fprintf(stderr, "error: impossible %s aggregate\n", spec->name);
            free(tile_units);
            free_host_metadata(m);
            return 0;
        }
        uint32_t remaining = spec->pairs - minimum;
        uint32_t cursor = 0;
        while (remaining) {
            uint32_t best = UINT32_MAX;
            for (uint32_t k = 0; k < active; k++) {
                const uint32_t e = (cursor + k) % active;
                if (m->counts[e] >= tile_units[e] * TILE_PAIRS) continue;
                if (best == UINT32_MAX || m->counts[e] < m->counts[best])
                    best = e;
            }
            if (best == UINT32_MAX) {
                fprintf(stderr, "error: %s pair capacities exhausted\n",
                        spec->name);
                free(tile_units);
                free_host_metadata(m);
                return 0;
            }
            m->counts[best]++;
            remaining--;
            cursor = (best + 1u) % active;
        }
    }
    uint32_t stride = 97u;
    while (gcd_u32(stride, TOTAL_PAIR_SLOTS) != 1u) stride += 2u;
    uint32_t pair_cursor = 0, tile_cursor = 0;
    for (uint32_t e = 0; e < active; e++) {
        m->offsets[e] = pair_cursor;
        for (uint32_t i = 0; i < m->counts[e]; i++) {
            const uint32_t logical = pair_cursor + i;
            m->sorted_pairs[logical] =
                (uint32_t)(((uint64_t)logical * stride + 17u) %
                           TOTAL_PAIR_SLOTS);
        }
        for (uint32_t t = 0; t < tile_units[e]; t++) {
            m->tile_experts[tile_cursor] = e;
            m->tile_starts[tile_cursor] = t * TILE_PAIRS;
            tile_cursor++;
        }
        pair_cursor += m->counts[e];
    }
    m->offsets[active] = pair_cursor;
    free(tile_units);
    if (pair_cursor != spec->pairs || tile_cursor != spec->tiles ||
        spec->tiles * TILE_PAIRS - spec->pairs != spec->padded) {
        fprintf(stderr, "error: %s metadata aggregate mismatch\n", spec->name);
        free_host_metadata(m);
        return 0;
    }
    return 1;
}

static void fill_weights(BlockQ4K *weights, const ScenarioSpec *spec) {
    uint32_t state = 0x6a09e667u ^ spec->layer;
    const uint64_t blocks = (uint64_t)spec->active_experts *
        OUT_DIM * MIDQ_BLOCKS_MAX;
    for (uint64_t index = 0; index < blocks; index++) {
        BlockQ4K *w = weights + index;
        const uint32_t mode = (uint32_t)(index & 7u);
        w->d = host_f32_to_f16(0.001953125f * (float)(1u + mode));
        w->dmin = (index & 1u) ?
            host_f32_to_f16(0.0009765625f * (float)(1u + (mode & 3u))) : 0u;
        for (uint32_t i = 0; i < sizeof(w->scales); i++)
            w->scales[i] = (uint8_t)rng_next(&state);
        for (uint32_t i = 0; i < sizeof(w->qs); i++) {
            if (mode == 0u) w->qs[i] = 0xf0u;       /* even=0, odd=15 */
            else if (mode == 1u) w->qs[i] = 0x0fu;  /* even=15, odd=0 */
            else if (mode == 2u) w->qs[i] = 0x00u;
            else if (mode == 3u) w->qs[i] = 0xffu;
            else w->qs[i] = (uint8_t)rng_next(&state);
        }
    }
}

static void fill_activations(float *values, const ScenarioSpec *spec) {
    uint32_t state = 0xbb67ae85u ^ spec->layer;
    for (uint32_t pair = 0; pair < TOTAL_PAIR_SLOTS; pair++) {
        for (uint32_t b = 0; b < MIDQ_BLOCKS_MAX; b++) {
            float *row = values + (uint64_t)pair * MID_DIM + b * QK_K;
            const int negative_max = ((pair + b) & 1u) != 0u;
            for (uint32_t i = 0; i < QK_K; i++) {
                const int raw = (int)(rng_next(&state) % 255u) - 127;
                row[i] = (float)raw / 127.0f;
            }
            row[0] = negative_max ? -2.0f : 2.0f;
            row[1] = negative_max ? 1.984375f : -1.984375f;
            row[2] = -128.0f / 64.0f;
            row[3] = 127.0f / 64.0f;
        }
    }
}

static int validate_weight_pack_samples(
        const BlockQ4K *standard, const ScenarioSpec *spec,
        const NativeWeightTileBlock *native_device) {
    const uint64_t records = (uint64_t)spec->active_experts *
        OUT_TILES * MIDQ_BLOCKS_MAX;
    const uint32_t samples = 192u;
    for (uint32_t s = 0; s < samples; s++) {
        const uint64_t record = s == 0 ? 0u :
            (s == 1 ? records - 1u :
             ((uint64_t)s * 0x9e3779b97f4a7c15ull) % records);
        NativeWeightTileBlock got;
        cuda_die(cudaMemcpy(&got, native_device + record, sizeof(got),
                            cudaMemcpyDeviceToHost),
                 "copy native weight sample");
        const uint32_t b = (uint32_t)(record % MIDQ_BLOCKS_MAX);
        const uint64_t tile_index = record / MIDQ_BLOCKS_MAX;
        const uint32_t out_tile = (uint32_t)(tile_index % OUT_TILES);
        const uint32_t expert = (uint32_t)(tile_index / OUT_TILES);
        const BlockQ4K *base = standard +
            ((uint64_t)expert * OUT_DIM + out_tile * OUT_TILE) *
            MIDQ_BLOCKS_MAX;
        for (uint32_t row = 0; row < OUT_TILE; row++) {
            if (memcmp(&got.hdr[row],
                       base + (uint64_t)row * MIDQ_BLOCKS_MAX + b,
                       sizeof(uint4)) != 0) {
                fprintf(stderr,
                        "error: native-W header mismatch record=%llu row=%u\n",
                        (unsigned long long)record, row);
                return 0;
            }
        }
        for (uint32_t j = 0; j < Q4_GROUPS; j++) {
            for (uint32_t lane = 0; lane < 32u; lane++) {
                const uint32_t row = lane >> 2u;
                const uint32_t lane4 = lane & 3u;
                const BlockQ4K *src = base +
                    (uint64_t)row * MIDQ_BLOCKS_MAX + b;
                const uint32_t off = (j >> 1u) * 32u + lane4 * 8u;
                const uint32_t q0 = load_u32_unaligned(src->qs + off);
                const uint32_t q1 = load_u32_unaligned(src->qs + off + 4u);
                const uint32_t expected = pack_nibbles_8(
                    q0, q1, (j & 1u) ? 4u : 0u);
                if (got.b[j][lane] != expected) {
                    fprintf(stderr,
                            "error: native-W payload mismatch record=%llu "
                            "j=%u lane=%u expected=%08x got=%08x\n",
                            (unsigned long long)record, j, lane,
                            expected, got.b[j][lane]);
                    return 0;
                }
            }
        }
    }
    printf("weight_pack_records_byte_validated=%u\n"
           "weight_pack_validation=exact\n", samples);
    return 1;
}

static int validate_activation_pack(const ScenarioData *d) {
    const size_t bytes = (size_t)d->activation_blocks * sizeof(BlockQ8K);
    BlockQ8K *standard = (BlockQ8K *)malloc(bytes);
    NativeQ8K *native = (NativeQ8K *)malloc(bytes);
    if (!standard || !native) {
        fprintf(stderr, "error: activation validation allocation failed\n");
        free(native); free(standard);
        return 0;
    }
    cuda_die(cudaMemcpy(standard, d->standard_a.ptr, bytes,
                        cudaMemcpyDeviceToHost),
             "copy standard Q8_K activations");
    cuda_die(cudaMemcpy(native, d->native_a.ptr, bytes,
                        cudaMemcpyDeviceToHost),
             "copy native Q8_K activations");
    uint64_t negative_d = 0, positive_d = 0;
    for (uint64_t block = 0; block < d->activation_blocks; block++) {
        if (memcmp(&standard[block].d, &native[block].d, sizeof(float)) != 0 ||
            memcmp(standard[block].bsums, native[block].bsums,
                   sizeof(standard[block].bsums)) != 0) {
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
                const uint32_t x0 = load_u32_unaligned(
                    standard[block].qs + off);
                const uint32_t x1 = load_u32_unaligned(
                    standard[block].qs + off + 4u);
                const uint32_t expected_low = pack_nibbles_8(x0, x1, 0u);
                const uint32_t expected_high = pack_nibbles_8(x0, x1, 4u);
                if (native[block].low[j][lane4] != expected_low ||
                    native[block].high_signed[j][lane4] != expected_high) {
                    fprintf(stderr,
                            "error: native-A plane mismatch block=%llu "
                            "j=%u lane4=%u\n",
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
            if (standard[block].bsums[g] != sum) {
                fprintf(stderr,
                        "error: Q8_K bsum mismatch block=%llu group=%u\n",
                        (unsigned long long)block, g);
                free(native); free(standard);
                return 0;
            }
        }
    }
    if (!negative_d || !positive_d) {
        fprintf(stderr,
                "error: activation corpus did not exercise both signs of d\n");
        free(native); free(standard);
        return 0;
    }
    printf("activation_blocks_byte_validated=%llu\n"
           "q8_negative_d_blocks=%llu\nq8_positive_d_blocks=%llu\n"
           "activation_pack_validation=exact\n",
           (unsigned long long)d->activation_blocks,
           (unsigned long long)negative_d,
           (unsigned long long)positive_d);
    free(native);
    free(standard);
    return 1;
}

static void cleanup_scenario(ScenarioData *d) {
    if (d->d_tile_starts) cudaFree(d->d_tile_starts);
    if (d->d_tile_experts) cudaFree(d->d_tile_experts);
    if (d->d_tile_total) cudaFree(d->d_tile_total);
    if (d->d_counts) cudaFree(d->d_counts);
    if (d->d_offsets) cudaFree(d->d_offsets);
    if (d->d_sorted_pairs) cudaFree(d->d_sorted_pairs);
    free_guarded(&d->output);
    free_guarded(&d->native_a);
    free_guarded(&d->standard_a);
    free_guarded(&d->native_w);
    free_guarded(&d->standard_w);
    free_host_metadata(&d->host_meta);
    memset(d, 0, sizeof(*d));
}

static int setup_scenario(ScenarioData *d, const ScenarioSpec *spec) {
    memset(d, 0, sizeof(*d));
    d->spec = spec;
    if (!build_metadata(spec, &d->host_meta)) return 0;
    d->weight_bytes = (uint64_t)spec->active_experts * OUT_DIM *
        MIDQ_BLOCKS_MAX * sizeof(BlockQ4K);
    d->activation_blocks = (uint64_t)TOTAL_PAIR_SLOTS * MIDQ_BLOCKS_MAX;
    d->output_values = (uint64_t)TOTAL_PAIR_SLOTS * OUT_DIM;
    const uint64_t activation_bytes =
        d->activation_blocks * sizeof(BlockQ8K);
    const uint64_t output_bytes = d->output_values * sizeof(float);
    const uint64_t weight_records =
        (uint64_t)spec->active_experts * OUT_TILES * MIDQ_BLOCKS_MAX;
    const size_t input_bytes =
        (size_t)TOTAL_PAIR_SLOTS * MID_DIM * sizeof(float);
    printf("scenario=%s\nlayer=%u\npair_count=%u\ntotal_pair_slots=%u\n"
           "active_experts=%u\n"
           "tile16_count=%u\npadded_slots=%u\nmidq_blocks=%u\n"
           "mid_dim=%u\nout_dim=%u\nrow_span=%u\n"
           "standard_weight_bytes=%llu\nnative_weight_bytes=%llu\n"
           "activation_bytes=%llu\noutput_bytes=%llu\n",
           spec->name, spec->layer, spec->pairs, TOTAL_PAIR_SLOTS,
           spec->active_experts,
           spec->tiles, spec->padded, MIDQ_BLOCKS_MAX, MID_DIM, OUT_DIM,
           ROW_SPAN, (unsigned long long)d->weight_bytes,
           (unsigned long long)d->weight_bytes,
           (unsigned long long)activation_bytes,
           (unsigned long long)output_bytes);

    BlockQ4K *host_weights =
        (BlockQ4K *)malloc((size_t)d->weight_bytes);
    float *host_activations = (float *)malloc(
        (size_t)TOTAL_PAIR_SLOTS * MID_DIM * sizeof(float));
    float *device_activations = NULL;
    int ok = 0;
    if (!host_weights || !host_activations) {
        fprintf(stderr, "error: scenario host allocation failed\n");
        goto done;
    }
    fill_weights(host_weights, spec);
    fill_activations(host_activations, spec);
    if (!alloc_guarded(&d->standard_w, d->weight_bytes) ||
        !alloc_guarded(&d->native_w, d->weight_bytes) ||
        !alloc_guarded(&d->standard_a, activation_bytes) ||
        !alloc_guarded(&d->native_a, activation_bytes) ||
        !alloc_guarded(&d->output, output_bytes)) {
        goto done;
    }
    cuda_die(cudaMemcpy(d->standard_w.ptr, host_weights,
                        (size_t)d->weight_bytes, cudaMemcpyHostToDevice),
             "copy standard Q4_K weights");
    sm75_q4_down_pack_w_kernel<<<
        (unsigned int)((weight_records + WARPS_PER_CTA - 1u) /
                       WARPS_PER_CTA), THREADS_PER_CTA>>>(
        (NativeWeightTileBlock *)d->native_w.ptr,
        (const BlockQ4K *)d->standard_w.ptr,
        spec->active_experts, MIDQ_BLOCKS_MAX);
    cuda_die(cudaGetLastError(), "launch native-W pack");
    cuda_die(cudaDeviceSynchronize(), "synchronize native-W pack");
    if (!validate_weight_pack_samples(
            host_weights, spec,
            (const NativeWeightTileBlock *)d->native_w.ptr)) goto done;

    cuda_die(cudaMalloc((void **)&device_activations, input_bytes),
             "allocate activation floats");
    cuda_die(cudaMemcpy(device_activations, host_activations, input_bytes,
                        cudaMemcpyHostToDevice),
             "copy activation floats");
    sm75_q4_down_quantize_q8k_kernel<<<
        dim3(MIDQ_BLOCKS_MAX, TOTAL_PAIR_SLOTS), THREADS_PER_CTA>>>(
        (BlockQ8K *)d->standard_a.ptr, device_activations,
        MID_DIM, TOTAL_PAIR_SLOTS);
    cuda_die(cudaGetLastError(), "launch Q8_K quantizer");
    sm75_q4_down_pack_a_kernel<<<
        (unsigned int)((d->activation_blocks + WARPS_PER_CTA - 1u) /
                       WARPS_PER_CTA), THREADS_PER_CTA>>>(
        (NativeQ8K *)d->native_a.ptr,
        (const BlockQ8K *)d->standard_a.ptr,
        d->activation_blocks);
    cuda_die(cudaGetLastError(), "launch native-A pack");
    cuda_die(cudaDeviceSynchronize(), "synchronize activation setup");
    if (!validate_activation_pack(d)) goto done;

#define ALLOC_COPY(member, host, count) do { \
    cuda_die(cudaMalloc((void **)&d->member, \
                        (size_t)(count) * sizeof(uint32_t)), \
             "allocate metadata " #member); \
    cuda_die(cudaMemcpy(d->member, (host), \
                        (size_t)(count) * sizeof(uint32_t), \
                        cudaMemcpyHostToDevice), \
             "copy metadata " #member); \
} while (0)
    ALLOC_COPY(d_sorted_pairs, d->host_meta.sorted_pairs, spec->pairs);
    ALLOC_COPY(d_offsets, d->host_meta.offsets, spec->active_experts + 1u);
    ALLOC_COPY(d_counts, d->host_meta.counts, spec->active_experts);
    ALLOC_COPY(d_tile_experts, d->host_meta.tile_experts, spec->tiles);
    ALLOC_COPY(d_tile_starts, d->host_meta.tile_starts, spec->tiles);
#undef ALLOC_COPY
    cuda_die(cudaMalloc((void **)&d->d_tile_total, sizeof(uint32_t)),
             "allocate tile total");
    cuda_die(cudaMemcpy(d->d_tile_total, &spec->tiles, sizeof(uint32_t),
                        cudaMemcpyHostToDevice), "copy tile total");
    ok = validate_canary(&d->standard_w, "standard-W") &&
         validate_canary(&d->native_w, "native-W") &&
         validate_canary(&d->standard_a, "standard-A") &&
         validate_canary(&d->native_a, "native-A") &&
         validate_canary(&d->output, "output");
    if (ok) printf("setup_canaries=ok\n");

done:
    if (device_activations) cudaFree(device_activations);
    free(host_activations);
    free(host_weights);
    if (!ok) cleanup_scenario(d);
    return ok;
}

static void launch_variant(const ScenarioData *d, Variant variant,
                           cudaStream_t stream) {
    if (variant == VAR_PACK_A || variant == VAR_NATIVE_AW_COMBINED) {
        sm75_q4_down_pack_a_kernel<<<
            (unsigned int)((d->activation_blocks + WARPS_PER_CTA - 1u) /
                           WARPS_PER_CTA), THREADS_PER_CTA, 0, stream>>>(
            (NativeQ8K *)d->native_a.ptr,
            (const BlockQ8K *)d->standard_a.ptr,
            d->activation_blocks);
        cuda_die(cudaGetLastError(), "launch timed native-A pack");
        if (variant == VAR_PACK_A) return;
    }
    if (variant == VAR_NATIVE_AW_NSPLIT4 ||
        variant == VAR_NATIVE_AW_NSPLIT8) {
        const uint32_t warps = variant == VAR_NATIVE_AW_NSPLIT4 ? 4u : 8u;
        const dim3 ngrid((OUT_DIM + warps * OUT_TILE - 1u) /
                         (warps * OUT_TILE), d->spec->tiles, 1u);
#define NSPLIT_ARGS \
        (float *)d->output.ptr, \
        (const NativeWeightTileBlock *)d->native_w.ptr, \
        (const NativeQ8K *)d->native_a.ptr, d->d_sorted_pairs, \
        d->d_offsets, d->d_counts, d->d_tile_total, d->d_tile_experts, \
        d->d_tile_starts, MIDQ_BLOCKS_MAX, OUT_DIM
        if (variant == VAR_NATIVE_AW_NSPLIT4)
            sm75_q4_down_native_aw_nsplit_kernel<4><<<
                ngrid, 128, 0, stream>>>(NSPLIT_ARGS);
        else
            sm75_q4_down_native_aw_nsplit_kernel<8><<<
                ngrid, 256, 0, stream>>>(NSPLIT_ARGS);
#undef NSPLIT_ARGS
        cuda_die(cudaGetLastError(), "launch Q4 down N-split consumer");
        return;
    }
    const dim3 grid((OUT_DIM + ROW_SPAN - 1u) / ROW_SPAN,
                    d->spec->tiles, 1u);
#define CONSUMER_ARGS \
    (float *)d->output.ptr, \
    (const BlockQ4K *)d->standard_w.ptr, \
    (const NativeWeightTileBlock *)d->native_w.ptr, \
    (const BlockQ8K *)d->standard_a.ptr, \
    (const NativeQ8K *)d->native_a.ptr, \
    d->d_sorted_pairs, d->d_offsets, d->d_counts, d->d_tile_total, \
    d->d_tile_experts, d->d_tile_starts, MIDQ_BLOCKS_MAX, OUT_DIM, \
    d->spec->active_experts
    if (variant == VAR_STANDARD) {
        sm75_q4_down_standard_kernel<<<grid, THREADS_PER_CTA, 0, stream>>>(
            CONSUMER_ARGS);
    } else if (variant == VAR_NATIVE_W) {
        sm75_q4_down_native_w_kernel<<<grid, THREADS_PER_CTA, 0, stream>>>(
            CONSUMER_ARGS);
    } else {
        sm75_q4_down_native_aw_kernel<<<grid, THREADS_PER_CTA, 0, stream>>>(
            CONSUMER_ARGS);
    }
#undef CONSUMER_ARGS
    cuda_die(cudaGetLastError(), "launch Q4 down consumer");
}

static int compare_output_bits(const float *expected, const float *actual,
                               uint64_t count, const char *variant) {
    for (uint64_t i = 0; i < count; i++) {
        uint32_t eb, ab;
        memcpy(&eb, expected + i, sizeof(eb));
        memcpy(&ab, actual + i, sizeof(ab));
        if (eb != ab) {
            fprintf(stderr,
                    "error: output mismatch variant=%s index=%llu "
                    "expected=%.9g[%08x] actual=%.9g[%08x]\n",
                    variant, (unsigned long long)i,
                    (double)expected[i], eb, (double)actual[i], ab);
            return 0;
        }
    }
    return 1;
}

static int validate_production_output(
        const ScenarioData *d, const float *expected, const float *actual,
        const char *variant) {
    uint8_t owned[TOTAL_PAIR_SLOTS] = {0};
    for (uint32_t i = 0; i < d->spec->pairs; i++)
        owned[d->host_meta.sorted_pairs[i]] = 1u;
    for (uint32_t pair = 0; pair < TOTAL_PAIR_SLOTS; pair++) {
        for (uint32_t row = 0; row < OUT_DIM; row++) {
            const uint64_t index = (uint64_t)pair * OUT_DIM + row;
            uint32_t actual_bits;
            memcpy(&actual_bits, actual + index, sizeof(actual_bits));
            if (!owned[pair]) {
                if (actual_bits != 0xffffffffu) {
                    fprintf(stderr,
                            "error: %s wrote unowned pair=%u row=%u bits=%08x\n",
                            variant, pair, row, actual_bits);
                    return 0;
                }
                continue;
            }
            if (!isfinite(actual[index])) {
                fprintf(stderr,
                        "error: %s left owned output uninitialized pair=%u "
                        "row=%u\n", variant, pair, row);
                return 0;
            }
            if (expected) {
                uint32_t expected_bits;
                memcpy(&expected_bits, expected + index,
                       sizeof(expected_bits));
                if (actual_bits != expected_bits) {
                    fprintf(stderr,
                            "error: output mismatch variant=%s pair=%u row=%u "
                            "expected=%.9g[%08x] actual=%.9g[%08x]\n",
                            variant, pair, row, (double)expected[index],
                            expected_bits, (double)actual[index], actual_bits);
                    return 0;
                }
            }
        }
    }
    return 1;
}

static int run_correctness(ScenarioData *d) {
    const size_t output_bytes =
        (size_t)d->output_values * sizeof(float);
    float *expected = (float *)malloc(output_bytes);
    float *actual = (float *)malloc(output_bytes);
    if (!expected || !actual) {
        fprintf(stderr, "error: output validation allocation failed\n");
        free(actual); free(expected);
        return 0;
    }
    cuda_die(cudaMemset(d->output.ptr, 0xff, output_bytes),
             "poison standard output");
    launch_variant(d, VAR_STANDARD, 0);
    cuda_die(cudaDeviceSynchronize(), "synchronize standard output");
    cuda_die(cudaMemcpy(expected, d->output.ptr, output_bytes,
                        cudaMemcpyDeviceToHost), "copy standard output");
    if (!validate_production_output(
            d, NULL, expected, kVariantNames[VAR_STANDARD])) {
        free(actual); free(expected);
        return 0;
    }
    const Variant checked[] = {
        VAR_NATIVE_W, VAR_NATIVE_AW_CONSUMER, VAR_NATIVE_AW_COMBINED,
        VAR_NATIVE_AW_NSPLIT4, VAR_NATIVE_AW_NSPLIT8,
    };
    for (uint32_t v = 0; v < sizeof(checked) / sizeof(checked[0]); v++) {
        cuda_die(cudaMemset(d->output.ptr, 0xff, output_bytes),
                 "poison native output");
        launch_variant(d, checked[v], 0);
        cuda_die(cudaDeviceSynchronize(), "synchronize native output");
        cuda_die(cudaMemcpy(actual, d->output.ptr, output_bytes,
                            cudaMemcpyDeviceToHost), "copy native output");
        if (!validate_production_output(
                d, expected, actual, kVariantNames[checked[v]])) {
            free(actual); free(expected);
            return 0;
        }
    }
    const int canaries_ok =
        validate_canary(&d->standard_w, "standard-W") &&
        validate_canary(&d->native_w, "native-W") &&
        validate_canary(&d->standard_a, "standard-A") &&
        validate_canary(&d->native_a, "native-A") &&
        validate_canary(&d->output, "output");
    printf("output_values_compared=%llu\n"
           "output_allocation_values=%llu\n"
           "unowned_output_poison_validation=exact\n"
           "output_reference=harness-standard-no-fast-math\n"
           "output_validation=bit_exact\n"
           "correctness_canaries=%s\n"
           "correctness_status=%s\n",
           (unsigned long long)d->spec->pairs * OUT_DIM,
           (unsigned long long)d->output_values,
           canaries_ok ? "ok" : "failed",
           canaries_ok ? "ok" : "failed");
    free(actual);
    free(expected);
    return canaries_ok;
}

static int compare_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static int run_benchmark(ScenarioData *d, uint32_t rounds,
                         uint32_t launches) {
    double *samples = (double *)calloc(
        (size_t)VARIANT_COUNT * rounds, sizeof(double));
    double *sorted = (double *)malloc((size_t)rounds * sizeof(double));
    if (!samples || !sorted) {
        fprintf(stderr, "error: benchmark sample allocation failed\n");
        free(sorted); free(samples);
        return 0;
    }
    cudaEvent_t start, stop;
    cuda_die(cudaEventCreate(&start), "create benchmark start event");
    cuda_die(cudaEventCreate(&stop), "create benchmark stop event");
    for (int v = 0; v < VARIANT_COUNT; v++) launch_variant(d, (Variant)v, 0);
    cuda_die(cudaDeviceSynchronize(), "synchronize benchmark warmup");
    printf("benchmark_warmup_launches_per_variant=1\n"
           "activation_quantization_common_and_excluded=1\n"
           "weight_layout_pack_offline_and_excluded=1\n"
           "combined_event_includes_pack_a_and_consumer=1\n"
           "pack_only_relative_not_consumer_comparable=1\n");
    printf("benchmark_samples_begin\n");
    printf("scenario,round,sample_slot,variant,total_ms,us_per_launch,relative_speed\n");
    for (uint32_t round = 0; round < rounds; round++) {
        const uint32_t start_variant = round % VARIANT_COUNT;
        uint32_t order[VARIANT_COUNT];
        for (uint32_t slot = 0; slot < VARIANT_COUNT; slot++) {
            const uint32_t step = (round & 1u)
                ? (VARIANT_COUNT - slot) % VARIANT_COUNT : slot;
            order[slot] = (start_variant + step) % VARIANT_COUNT;
            const Variant v = (Variant)order[slot];
            cuda_die(cudaEventRecord(start), "record benchmark start");
            for (uint32_t i = 0; i < launches; i++) launch_variant(d, v, 0);
            cuda_die(cudaEventRecord(stop), "record benchmark stop");
            cuda_die(cudaEventSynchronize(stop),
                     "synchronize benchmark sample");
            float ms = 0.0f;
            cuda_die(cudaEventElapsedTime(&ms, start, stop),
                     "measure benchmark sample");
            samples[(size_t)v * rounds + round] = ms;
        }
        const double standard_ms = samples[(size_t)VAR_STANDARD * rounds + round];
        for (uint32_t slot = 0; slot < VARIANT_COUNT; slot++) {
            const Variant v = (Variant)order[slot];
            const double ms = samples[(size_t)v * rounds + round];
            printf("%s,%u,%u,%s,%.6f,%.6f,%.6f\n",
                   d->spec->name, round + 1u, slot + 1u,
                   kVariantNames[v], ms, 1000.0 * ms / launches,
                   standard_ms / ms);
        }
    }
    printf("benchmark_samples_end\n");
    double medians[VARIANT_COUNT];
    for (int v = 0; v < VARIANT_COUNT; v++) {
        memcpy(sorted, samples + (size_t)v * rounds,
               (size_t)rounds * sizeof(double));
        qsort(sorted, rounds, sizeof(double), compare_double);
        medians[v] = (rounds & 1u)
            ? sorted[rounds / 2u]
            : 0.5 * (sorted[rounds / 2u - 1u] + sorted[rounds / 2u]);
    }
    printf("median_summary_begin\n");
    printf("scenario,variant,median_total_ms,median_us_per_launch,relative_speed\n");
    for (int v = 0; v < VARIANT_COUNT; v++) {
        printf("%s,%s,%.6f,%.6f,%.6f\n",
               d->spec->name, kVariantNames[v], medians[v],
               1000.0 * medians[v] / launches,
               medians[VAR_STANDARD] / medians[v]);
    }
    printf("median_summary_end\nbenchmark_status=ok\n");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    free(sorted);
    free(samples);
    return 1;
}

static int run_profile(ScenarioData *d, Variant variant,
                       uint32_t launches) {
    launch_variant(d, variant, 0);
    cuda_die(cudaDeviceSynchronize(), "synchronize profile warmup");
    printf("profile_variant=%s\nprofile_warmup_launches=1\n"
           "profile_capture_launches=%u\nprofile_capture_begin=1\n",
           kVariantNames[variant], launches);
    fflush(stdout);
    for (uint32_t i = 0; i < launches; i++) launch_variant(d, variant, 0);
    cuda_die(cudaDeviceSynchronize(), "synchronize profile capture");
    printf("profile_capture_end=1\nprofile_status=ok\n");
    return 1;
}

template <typename Kernel>
static void print_one_resource(const char *name, Kernel kernel,
                               int threads = THREADS_PER_CTA) {
    cudaFuncAttributes attr;
    cuda_die(cudaFuncGetAttributes(&attr, kernel), "query kernel attributes");
    int blocks = 0;
    cuda_die(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &blocks, kernel, threads, 0),
             "query kernel occupancy");
    printf("resource_%s_registers_per_thread=%d\n"
           "resource_%s_static_shared_bytes=%zu\n"
           "resource_%s_local_bytes_per_thread=%zu\n"
           "resource_%s_max_threads_per_block=%d\n"
           "resource_%s_active_blocks_per_sm=%d\n",
           name, attr.numRegs, name, attr.sharedSizeBytes,
           name, attr.localSizeBytes, name, attr.maxThreadsPerBlock,
           name, blocks);
}

static void print_resources(void) {
    print_one_resource("standard", sm75_q4_down_standard_kernel);
    print_one_resource("native_w", sm75_q4_down_native_w_kernel);
    print_one_resource("native_aw", sm75_q4_down_native_aw_kernel);
    print_one_resource("native_aw_nsplit4",
        sm75_q4_down_native_aw_nsplit_kernel<4>, 128);
    print_one_resource("native_aw_nsplit8",
        sm75_q4_down_native_aw_nsplit_kernel<8>, 256);
    print_one_resource("pack_a", sm75_q4_down_pack_a_kernel);
    printf("resource_spill_note=local_bytes_are_runtime_metadata;verify_spills_with_ptxas_or_ncu\n");
}

static void q4_set_scale_min(uint32_t j, uint8_t *scales,
                             uint8_t d, uint8_t m) {
    memset(scales, 0, 12u);
    if (j < 4u) {
        scales[j] = d & 63u;
        scales[j + 4u] = m & 63u;
    } else {
        scales[j + 4u] = (d & 0x0fu) | ((m & 0x0fu) << 4u);
        scales[j - 4u] = (uint8_t)((d >> 4u) << 6u);
        scales[j] = (uint8_t)((m >> 4u) << 6u);
    }
}

static int run_adversarial_host_gates(int run_gpu_pack_gate) {
    uint8_t scales[12];
    for (uint32_t j = 0; j < Q4_GROUPS; j++) {
        for (uint32_t d = 0; d < 64u; d++) {
            for (uint32_t m = 0; m < 64u; m++) {
                q4_set_scale_min(j, scales, (uint8_t)d, (uint8_t)m);
                uint8_t got_d, got_m;
                q4_get_scale_min(j, scales, &got_d, &got_m);
                if (got_d != d || got_m != m) {
                    fprintf(stderr,
                            "error: scale/min gate j=%u d=%u m=%u got=%u/%u\n",
                            j, d, m, got_d, got_m);
                    return 0;
                }
            }
        }
    }
    if (!run_gpu_pack_gate) {
        printf("scale_min_decode_cases=%u\n"
               "scale_min_decode_validation=exhaustive\n"
               "adversarial_gpu_pack_gate=skipped-for-profile\n"
               "adversarial_host_gates=ok\n",
               Q4_GROUPS * 64u * 64u);
        return 1;
    }
    BlockQ8K standard = {};
    standard.d = -0.03125f;
    for (uint32_t i = 0; i < QK_K; i++) {
        static const int8_t edge[] =
            {-128, -127, -16, -15, -1, 0, 1, 15, 16, 126, 127};
        standard.qs[i] = edge[i % (sizeof(edge) / sizeof(edge[0]))];
    }
    for (uint32_t g = 0; g < QK_K / 16; g++) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; i++)
            sum += standard.qs[g * 16u + i];
        standard.bsums[g] = (int16_t)sum;
    }
    BlockQ8K *d_standard = NULL;
    NativeQ8K *d_native = NULL;
    NativeQ8K native;
    cuda_die(cudaMalloc((void **)&d_standard, sizeof(standard)),
             "allocate adversarial standard A");
    cuda_die(cudaMalloc((void **)&d_native, sizeof(native)),
             "allocate adversarial native A");
    cuda_die(cudaMemcpy(d_standard, &standard, sizeof(standard),
                        cudaMemcpyHostToDevice),
             "copy adversarial standard A");
    sm75_q4_down_pack_a_kernel<<<1, THREADS_PER_CTA>>>(
        d_native, d_standard, 1u);
    cuda_die(cudaGetLastError(), "launch adversarial native-A pack");
    cuda_die(cudaMemcpy(&native, d_native, sizeof(native),
                        cudaMemcpyDeviceToHost),
             "copy adversarial native A");
    cudaFree(d_native);
    cudaFree(d_standard);
    if (memcmp(&native.d, &standard.d, sizeof(float)) ||
        memcmp(native.bsums, standard.bsums, sizeof(standard.bsums))) {
        fprintf(stderr, "error: adversarial native-A metadata mismatch\n");
        return 0;
    }
    for (uint32_t j = 0; j < Q4_GROUPS; j++) {
        for (uint32_t lane4 = 0; lane4 < 4u; lane4++) {
            const uint32_t off = j * 32u + lane4 * 8u;
            const uint32_t x0 = load_u32_unaligned(standard.qs + off);
            const uint32_t x1 = load_u32_unaligned(standard.qs + off + 4u);
            if (native.low[j][lane4] != pack_nibbles_8(x0, x1, 0u) ||
                native.high_signed[j][lane4] !=
                    pack_nibbles_8(x0, x1, 4u)) {
                fprintf(stderr, "error: adversarial native-A plane mismatch\n");
                return 0;
            }
        }
    }
    printf("scale_min_decode_cases=%u\n"
           "scale_min_decode_validation=exhaustive\n"
           "q8_extrema_validation=-128,-127,0,127\n"
           "q8_bsum_validation=exact\n"
           "adversarial_host_gates=ok\n",
           Q4_GROUPS * 64u * 64u);
    return 1;
}

static void q4_encode_all_scale_min(BlockQ4K *w,
                                    const uint8_t sc[Q4_GROUPS],
                                    const uint8_t mn[Q4_GROUPS]) {
    memset(w->scales, 0, sizeof(w->scales));
    for (uint32_t j = 0; j < 4u; j++) {
        w->scales[j] = (uint8_t)((sc[j] & 63u) |
            (((sc[j + 4u] >> 4u) & 3u) << 6u));
        w->scales[j + 4u] = (uint8_t)((mn[j] & 63u) |
            (((mn[j + 4u] >> 4u) & 3u) << 6u));
        w->scales[j + 8u] = (uint8_t)((sc[j + 4u] & 15u) |
            ((mn[j + 4u] & 15u) << 4u));
    }
}

static void q4_set_value(BlockQ4K *w, uint32_t j,
                         uint32_t k, uint8_t value) {
    uint8_t *byte = w->qs + (j >> 1u) * 32u + k;
    if (j & 1u) *byte = (uint8_t)((*byte & 0x0fu) |
                                  ((value & 15u) << 4u));
    else *byte = (uint8_t)((*byte & 0xf0u) | (value & 15u));
}

static void fill_micro_weight(BlockQ4K *w, uint32_t row, uint32_t b) {
    memset(w, 0, sizeof(*w));
    w->d = host_f32_to_f16(0.00390625f * (float)(1u + ((row + b) & 3u)));
    w->dmin = ((row + b) & 1u)
        ? host_f32_to_f16(0.001953125f * (float)(1u + (b & 1u))) : 0u;
    uint8_t sc[Q4_GROUPS], mn[Q4_GROUPS];
    for (uint32_t j = 0; j < Q4_GROUPS; j++) {
        sc[j] = (uint8_t)((row * 17u + b * 11u + j * 9u) & 63u);
        mn[j] = (uint8_t)((row * 7u + b * 19u + j * 13u) & 63u);
    }
    sc[0] = 0u; sc[4] = 63u;
    mn[1] = 0u; mn[5] = 63u;
    q4_encode_all_scale_min(w, sc, mn);
    for (uint32_t j = 0; j < Q4_GROUPS; j++) {
        for (uint32_t k = 0; k < 32u; k++) {
            uint8_t value;
            if (j == 0u) value = 0u;
            else if (j == 1u) value = 15u;
            else if (j == 2u) value = (k & 1u) ? 15u : 0u;
            else if (j == 3u) value = (k & 1u) ? 0u : 15u;
            else value = (uint8_t)((row * 3u + b * 5u + j * 7u +
                                    k * 11u) & 15u);
            q4_set_value(w, j, k, value);
        }
    }
}

static void build_micro_native_weight(const BlockQ4K *standard,
                                      uint32_t blocks,
                                      NativeWeightTileBlock *native) {
    for (uint32_t b = 0; b < blocks; b++) {
        NativeWeightTileBlock *dst = native + b;
        memset(dst, 0, sizeof(*dst));
        for (uint32_t row = 0; row < OUT_TILE; row++)
            memcpy(&dst->hdr[row], standard + row * blocks + b,
                   sizeof(uint4));
        for (uint32_t j = 0; j < Q4_GROUPS; j++) {
            for (uint32_t lane = 0; lane < 32u; lane++) {
                const uint32_t row = lane >> 2u;
                const uint32_t lane4 = lane & 3u;
                const BlockQ4K *src = standard + row * blocks + b;
                const uint32_t off = (j >> 1u) * 32u + lane4 * 8u;
                const uint32_t q0 = load_u32_unaligned(src->qs + off);
                const uint32_t q1 = load_u32_unaligned(src->qs + off + 4u);
                dst->b[j][lane] = pack_nibbles_8(
                    q0, q1, (j & 1u) ? 4u : 0u);
            }
        }
    }
}

static void fill_micro_activation(BlockQ8K *a, uint32_t pair, uint32_t b) {
    static const int8_t edge[] = {
        -128, -127, -113, -17, -16, -15, -9, -8, -1, 0, 1,
        7, 8, 15, 16, 63, 126, 127,
    };
    a->d = ((pair + b) & 1u) ? -0.03125f : 0.015625f;
    for (uint32_t i = 0; i < QK_K; i++)
        a->qs[i] = edge[(pair * 7u + b * 11u + i) %
            (sizeof(edge) / sizeof(edge[0]))];
    for (uint32_t g = 0; g < QK_K / 16; g++) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; i++) sum += a->qs[g * 16u + i];
        a->bsums[g] = (int16_t)sum;
    }
}

static void launch_micro_consumer(
        Variant variant, float *out,
        const BlockQ4K *standard_w,
        const NativeWeightTileBlock *native_w,
        const BlockQ8K *standard_a, const NativeQ8K *native_a,
        const uint32_t *pairs, const uint32_t *offsets,
        const uint32_t *counts, const uint32_t *tile_total,
        const uint32_t *tile_experts, const uint32_t *tile_starts,
        uint32_t blocks) {
#define MICRO_ARGS \
    out, standard_w, native_w, standard_a, native_a, pairs, offsets, \
    counts, tile_total, tile_experts, tile_starts, blocks, OUT_TILE, 1u
    if (variant == VAR_STANDARD)
        sm75_q4_down_standard_kernel<<<1, THREADS_PER_CTA>>>(MICRO_ARGS);
    else if (variant == VAR_NATIVE_W)
        sm75_q4_down_native_w_kernel<<<1, THREADS_PER_CTA>>>(MICRO_ARGS);
    else
        sm75_q4_down_native_aw_kernel<<<1, THREADS_PER_CTA>>>(MICRO_ARGS);
#undef MICRO_ARGS
    cuda_die(cudaGetLastError(), "launch adversarial consumer");
}

static int run_adversarial_gpu_cases(void) {
    const uint32_t populations[] = {16u, 8u, 4u};
    BlockQ4K host_w[OUT_TILE * MIDQ_BLOCKS_MAX];
    NativeWeightTileBlock host_nw[MIDQ_BLOCKS_MAX];
    BlockQ8K host_a[TILE_PAIRS * MIDQ_BLOCKS_MAX];
    float expected[TILE_PAIRS * OUT_TILE];
    float actual[TILE_PAIRS * OUT_TILE];
    DeviceBuffer standard_w = {}, native_w = {}, standard_a = {};
    DeviceBuffer native_a = {}, output = {};
    uint32_t *d_pairs = NULL, *d_offsets = NULL, *d_counts = NULL;
    uint32_t *d_tile_total = NULL, *d_tile_experts = NULL;
    uint32_t *d_tile_starts = NULL;
    uint32_t pair_ids[TILE_PAIRS], offsets[2] = {0u, 0u};
    const uint32_t one = 1u, zero = 0u;
    int ok = 0;
    if (!alloc_guarded(&standard_w, sizeof(host_w)) ||
        !alloc_guarded(&native_w, sizeof(host_nw)) ||
        !alloc_guarded(&standard_a, sizeof(host_a)) ||
        !alloc_guarded(&native_a,
                       TILE_PAIRS * MIDQ_BLOCKS_MAX * sizeof(NativeQ8K)) ||
        !alloc_guarded(&output, sizeof(expected))) goto done;
    cuda_die(cudaMalloc((void **)&d_pairs,
                        TILE_PAIRS * sizeof(uint32_t)),
             "allocate adversarial pairs");
    cuda_die(cudaMalloc((void **)&d_offsets, 2u * sizeof(uint32_t)),
             "allocate adversarial offsets");
    cuda_die(cudaMalloc((void **)&d_counts, sizeof(uint32_t)),
             "allocate adversarial counts");
    cuda_die(cudaMalloc((void **)&d_tile_total, sizeof(uint32_t)),
             "allocate adversarial tile total");
    cuda_die(cudaMalloc((void **)&d_tile_experts, sizeof(uint32_t)),
             "allocate adversarial tile experts");
    cuda_die(cudaMalloc((void **)&d_tile_starts, sizeof(uint32_t)),
             "allocate adversarial tile starts");
    for (uint32_t i = 0; i < TILE_PAIRS; i++) pair_ids[i] = i;
    cuda_die(cudaMemcpy(d_pairs, pair_ids, sizeof(pair_ids),
                        cudaMemcpyHostToDevice), "copy adversarial pairs");
    cuda_die(cudaMemcpy(d_tile_total, &one, sizeof(one),
                        cudaMemcpyHostToDevice), "copy adversarial tile total");
    cuda_die(cudaMemcpy(d_tile_experts, &zero, sizeof(zero),
                        cudaMemcpyHostToDevice), "copy adversarial expert");
    cuda_die(cudaMemcpy(d_tile_starts, &zero, sizeof(zero),
                        cudaMemcpyHostToDevice), "copy adversarial tile start");
    for (uint32_t blocks = 1u; blocks <= MIDQ_BLOCKS_MAX; blocks++) {
        for (uint32_t row = 0; row < OUT_TILE; row++)
            for (uint32_t b = 0; b < blocks; b++)
                fill_micro_weight(&host_w[row * blocks + b], row, b);
        build_micro_native_weight(host_w, blocks, host_nw);
        cuda_die(cudaMemcpy(standard_w.ptr, host_w,
                            OUT_TILE * blocks * sizeof(BlockQ4K),
                            cudaMemcpyHostToDevice),
                 "copy adversarial standard weights");
        cuda_die(cudaMemcpy(native_w.ptr, host_nw,
                            blocks * sizeof(NativeWeightTileBlock),
                            cudaMemcpyHostToDevice),
                 "copy adversarial native weights");
        for (uint32_t pi = 0;
             pi < sizeof(populations) / sizeof(populations[0]); pi++) {
            const uint32_t np = populations[pi];
            for (uint32_t p = 0; p < np; p++)
                for (uint32_t b = 0; b < blocks; b++)
                    fill_micro_activation(&host_a[p * blocks + b], p, b);
            cuda_die(cudaMemcpy(standard_a.ptr, host_a,
                                (size_t)np * blocks * sizeof(BlockQ8K),
                                cudaMemcpyHostToDevice),
                     "copy adversarial standard activations");
            sm75_q4_down_pack_a_kernel<<<
                (np * blocks + WARPS_PER_CTA - 1u) / WARPS_PER_CTA,
                THREADS_PER_CTA>>>(
                (NativeQ8K *)native_a.ptr,
                (const BlockQ8K *)standard_a.ptr,
                (uint64_t)np * blocks);
            cuda_die(cudaGetLastError(), "launch adversarial native-A pack");
            offsets[1] = np;
            cuda_die(cudaMemcpy(d_offsets, offsets, sizeof(offsets),
                                cudaMemcpyHostToDevice),
                     "copy adversarial offsets");
            cuda_die(cudaMemcpy(d_counts, &np, sizeof(np),
                                cudaMemcpyHostToDevice),
                     "copy adversarial counts");
            cuda_die(cudaMemset(output.ptr, 0xff,
                                (size_t)np * OUT_TILE * sizeof(float)),
                     "poison adversarial standard output");
            launch_micro_consumer(
                VAR_STANDARD, (float *)output.ptr,
                (const BlockQ4K *)standard_w.ptr,
                (const NativeWeightTileBlock *)native_w.ptr,
                (const BlockQ8K *)standard_a.ptr,
                (const NativeQ8K *)native_a.ptr,
                d_pairs, d_offsets, d_counts, d_tile_total,
                d_tile_experts, d_tile_starts, blocks);
            cuda_die(cudaDeviceSynchronize(),
                     "synchronize adversarial standard");
            cuda_die(cudaMemcpy(expected, output.ptr,
                                (size_t)np * OUT_TILE * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "copy adversarial standard output");
            for (uint32_t i = 0; i < np * OUT_TILE; i++) {
                if (!isfinite(expected[i])) {
                    fprintf(stderr,
                            "error: adversarial standard output remained "
                            "poisoned blocks=%u np=%u index=%u\n",
                            blocks, np, i);
                    goto done;
                }
            }
            const Variant variants[] = {
                VAR_NATIVE_W, VAR_NATIVE_AW_CONSUMER,
            };
            for (uint32_t vi = 0;
                 vi < sizeof(variants) / sizeof(variants[0]); vi++) {
                cuda_die(cudaMemset(output.ptr, 0xff,
                                    (size_t)np * OUT_TILE * sizeof(float)),
                         "poison adversarial native output");
                launch_micro_consumer(
                    variants[vi], (float *)output.ptr,
                    (const BlockQ4K *)standard_w.ptr,
                    (const NativeWeightTileBlock *)native_w.ptr,
                    (const BlockQ8K *)standard_a.ptr,
                    (const NativeQ8K *)native_a.ptr,
                    d_pairs, d_offsets, d_counts, d_tile_total,
                    d_tile_experts, d_tile_starts, blocks);
                cuda_die(cudaDeviceSynchronize(),
                         "synchronize adversarial native");
                cuda_die(cudaMemcpy(actual, output.ptr,
                                    (size_t)np * OUT_TILE * sizeof(float),
                                    cudaMemcpyDeviceToHost),
                         "copy adversarial native output");
                if (!compare_output_bits(expected, actual,
                                         (uint64_t)np * OUT_TILE,
                                         kVariantNames[variants[vi]])) {
                    fprintf(stderr,
                            "error: adversarial case blocks=%u np=%u\n",
                            blocks, np);
                    goto done;
                }
            }
        }
    }
    ok = validate_canary(&standard_w, "adversarial-standard-W") &&
         validate_canary(&native_w, "adversarial-native-W") &&
         validate_canary(&standard_a, "adversarial-standard-A") &&
         validate_canary(&native_a, "adversarial-native-A") &&
         validate_canary(&output, "adversarial-output");
    if (ok) {
        printf("adversarial_midq_blocks=1,2,3,4,5,6,7,8\n"
               "adversarial_tile_populations=16,8,4\n"
               "adversarial_gpu_output_cases=24\n"
               "adversarial_gpu_reference=harness-standard-no-fast-math\n"
               "adversarial_gpu_validation=bit_exact\n"
               "adversarial_gpu_canaries=ok\n");
    }

done:
    if (d_tile_starts) cudaFree(d_tile_starts);
    if (d_tile_experts) cudaFree(d_tile_experts);
    if (d_tile_total) cudaFree(d_tile_total);
    if (d_counts) cudaFree(d_counts);
    if (d_offsets) cudaFree(d_offsets);
    if (d_pairs) cudaFree(d_pairs);
    free_guarded(&output);
    free_guarded(&native_a);
    free_guarded(&standard_a);
    free_guarded(&native_w);
    free_guarded(&standard_w);
    return ok;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [--device N] [--scenario early|late|real-early|real-late] MODE [OPTIONS]\n"
        "\n"
        "Modes:\n"
        "  --correctness-only\n"
        "  --benchmark-only [--rounds N] [--launches N]\n"
        "  --profile standard|native-w|native-aw-consumer|"
        "native-aw-combined|pack-a|native-aw-nsplit4|native-aw-nsplit8 "
        "[--launches N]\n"
        "\n"
        "Without --scenario, correctness runs early and late; benchmark and\n"
        "profile default to early. Without a mode, correctness and benchmark\n"
        "run for the selected scenarios.\n", argv0);
}

static uint32_t parse_u32(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno || !end || *end || value == 0 || value > UINT32_MAX) {
        fprintf(stderr, "error: invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint32_t)value;
}

static int parse_device(const char *text) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || !end || *end || value < 0 || value > INT32_MAX) {
        fprintf(stderr, "error: invalid device: %s\n", text);
        exit(2);
    }
    return (int)value;
}

int main(int argc, char **argv) {
    enum Mode { MODE_ALL, MODE_CORRECTNESS, MODE_BENCHMARK, MODE_PROFILE };
    Mode mode = MODE_ALL;
    int mode_explicit = 0;
    int device = 0;
    const ScenarioSpec *selected = NULL;
    Variant profile_variant = VAR_STANDARD;
    uint32_t rounds = 10u, launches = 3u;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            device = parse_device(argv[++i]);
        } else if (!strcmp(argv[i], "--scenario") && i + 1 < argc) {
            selected = find_scenario(argv[++i]);
            if (!selected) { usage(argv[0]); return 2; }
        } else if (!strcmp(argv[i], "--correctness-only")) {
            if (mode_explicit++) { usage(argv[0]); return 2; }
            mode = MODE_CORRECTNESS;
        } else if (!strcmp(argv[i], "--benchmark-only")) {
            if (mode_explicit++) { usage(argv[0]); return 2; }
            mode = MODE_BENCHMARK;
        } else if (!strcmp(argv[i], "--profile") && i + 1 < argc) {
            if (mode_explicit++) { usage(argv[0]); return 2; }
            mode = MODE_PROFILE;
            profile_variant = find_variant(argv[++i]);
        } else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) {
            rounds = parse_u32(argv[++i], "rounds");
        } else if (!strcmp(argv[i], "--launches") && i + 1 < argc) {
            launches = parse_u32(argv[++i], "launches");
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    cuda_die(cudaSetDevice(device), "select CUDA device");
    cudaDeviceProp prop;
    cuda_die(cudaGetDeviceProperties(&prop, device), "query CUDA device");
    if (prop.major != 7 || prop.minor != 5) {
        fprintf(stderr,
                "error: SM75 harness requires compute capability 7.5; got %d.%d\n",
                prop.major, prop.minor);
        return 1;
    }
    printf("harness=sm75-q4-down-native\ndevice=%d\ndevice_name=%s\n"
           "compute_capability=%d.%d\nfast_math=disabled\n",
           device, prop.name, prop.major, prop.minor);
    if (!run_adversarial_host_gates(mode != MODE_PROFILE)) return 1;
    if (mode != MODE_PROFILE && !run_adversarial_gpu_cases()) return 1;
    print_resources();
    const size_t begin = selected ?
        (size_t)(selected - kScenarios) : 0u;
    const size_t end = selected ? begin + 1u :
        (mode == MODE_CORRECTNESS || mode == MODE_ALL ?
         2u : 1u);
    int ok = 1;
    for (size_t s = begin; s < end && ok; s++) {
        ScenarioData data;
        if (!setup_scenario(&data, &kScenarios[s])) return 1;
        if (mode == MODE_CORRECTNESS || mode == MODE_ALL)
            ok = run_correctness(&data);
        if (ok && (mode == MODE_BENCHMARK || mode == MODE_ALL))
            ok = run_benchmark(&data, rounds, launches);
        if (ok && mode == MODE_PROFILE)
            ok = run_profile(&data, profile_variant, launches);
        cleanup_scenario(&data);
    }
    if (ok) printf("harness_status=ok\n");
    return ok ? 0 : 1;
}
