/*
 * Bounded SM75 packed-INT4 MMA experiment.
 *
 * This is deliberately independent of the inference engine.  It is a
 * microarchitectural comparison, not a production-kernel or end-to-end
 * benchmark.  Each warp computes a 16-token x 8-output Q4_K integer tile over
 * K=256, reusing every B fragment across two m8n8 operations.  Eight
 * independent warps run in each 256-thread CTA.
 *
 *   i8-standard       current-style Q4 expansion + 2 x m8n8k16 per A tile
 *   i4-mixed-standard standard Q8/Q4 bytes packed in registers +
 *                     m8n8k32 u4xu4 and s4xu4
 *   i4-mixed-group32-w standard Q8 + same-row block-local Q4 group32 layout
 *   i4-mixed-native-w standard Q8 bytes + prepacked MMA-native Q4 fragments
 *   i4-mixed-native-aw prepacked Q8 and Q4 MMA fragments
 *   i4-u4-corrected   all-u4 activation decomposition with exact -128*sum(w)
 *
 * The mixed path uses the exact two's-complement identity
 *
 *     a_s8 = low_u4(a) + 16 * high_s4(a)
 *
 * and therefore requires no weight-sum correction.  The all-u4 path is kept
 * as a measured fallback/reference:
 *
 *     a_s8 = low_u4(a + 128) + 16 * high_u4(a + 128) - 128.
 *
 * Only the scaled integer Q4_K sum is replaced here.  Q4_K's d/dmin float
 * finish and the existing Q8_K bsums-based min correction are unchanged.
 */

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    MMA_M = 8,
    A_TILES = 2,
    TILE_M = MMA_M * A_TILES,
    TILE_N = 8,
    TILE_K = 256,
    GROUP_K = 32,
    N_GROUPS = TILE_K / GROUP_K,
    WARP_SIZE_ = 32,
    WARPS_PER_CTA = 8,
    THREADS_PER_CTA = WARP_SIZE_ * WARPS_PER_CTA,
    HOT_ACTIVATION_CASES = 16,
};

struct StandardCase {
    int8_t q8[TILE_M][TILE_K];
    /* Standard Q4_K payload order: each 32-byte group stores two adjacent
     * 32-value groups in its low/high nibbles. */
    uint8_t q4[TILE_N][TILE_K / 2];
    uint8_t scales[TILE_N][N_GROUPS];
};

struct NativeCase {
    /* [8-row tile][group][lane], in exact m8n8k32 fragment order. Two nibble
     * planes represent Q8 without increasing its value payload. */
    uint32_t a_low[A_TILES][N_GROUPS][WARP_SIZE_];
    uint32_t a_high_signed[A_TILES][N_GROUPS][WARP_SIZE_];
    uint32_t a_offset_low[A_TILES][N_GROUPS][WARP_SIZE_];
    uint32_t a_offset_high[A_TILES][N_GROUPS][WARP_SIZE_];
    /* Same-size, block-local layout: one contiguous packed 16-byte segment
     * per output row and k32 group. Row/block offsets remain conventional. */
    uint8_t b_group32[TILE_N][N_GROUPS][GROUP_K / 2];
    /* More aggressive tile8 x k32 layout in exact B-fragment lane order. */
    uint32_t b[N_GROUPS][WARP_SIZE_];
    int16_t weight_sums[TILE_N][N_GROUPS];
    uint8_t scales[TILE_N][N_GROUPS];
};

static_assert(sizeof(((StandardCase *)0)->q8) == 4096, "Q8 tile payload");
static_assert(sizeof(((StandardCase *)0)->q4) == 1024, "Q4 tile payload");
static_assert(sizeof(((NativeCase *)0)->a_low) +
              sizeof(((NativeCase *)0)->a_high_signed) == 4096,
              "MMA-native Q8 nibble planes must be size-neutral");
static_assert(sizeof(((NativeCase *)0)->b) == 1024,
              "MMA-native Q4 fragments must be size-neutral");
static_assert(sizeof(((NativeCase *)0)->b_group32) == 1024,
              "block-local group32 Q4 must be size-neutral");

enum Variant {
    VAR_I8_STANDARD = 0,
    VAR_I4_MIXED_STANDARD,
    VAR_I4_MIXED_GROUP32_W,
    VAR_I4_MIXED_NATIVE_W,
    VAR_I4_MIXED_NATIVE_AW,
    VAR_I4_U4_CORRECTED_NATIVE_AW,
    VARIANT_COUNT,
};

static const char *const variant_names[VARIANT_COUNT] = {
    "i8-standard",
    "i4-mixed-standard",
    "i4-mixed-group32-w",
    "i4-mixed-native-w",
    "i4-mixed-native-aw",
    "i4-u4-corrected-native-aw",
};

struct TileI32 {
    /* [A tile 0 col 0, A tile 0 col 1, A tile 1 col 0, A tile 1 col 1]. */
    int32_t v[2 * A_TILES];
};

static void die_cuda(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return;
    fprintf(stderr, "error: %s: %s\n", what, cudaGetErrorString(err));
    exit(2);
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

__host__ __device__ __forceinline__ static uint32_t pack_nibbles_8(
        uint32_t lo_words, uint32_t hi_words, uint32_t shift) {
    uint32_t out = 0;
#pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        out |= ((lo_words >> (8u * i + shift)) & 0x0fu) << (4u * i);
        out |= ((hi_words >> (8u * i + shift)) & 0x0fu) << (4u * (i + 4u));
    }
    return out;
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

__device__ __forceinline__ static void preload_standard_q4_i8(
        const StandardCase *weights, uint32_t lane, uint32_t words[N_GROUPS]) {
    const uint32_t col = lane >> 2u;
    const uint32_t lane4 = lane & 3u;
#pragma unroll
    for (uint32_t k = 0; k < N_GROUPS; k++) {
        /* This is the same cooperative word ownership used by the production
         * SM75 Q4 kernel. Each word is reused for both of its nibble groups. */
        words[k] = load_u32_unaligned(
            &weights->q4[col][k * 16u + lane4 * 4u]);
    }
}

__device__ __forceinline__ static void preload_standard_q4_i4(
        const StandardCase *weights, uint32_t lane,
        uint32_t word0[N_GROUPS / 2], uint32_t word1[N_GROUPS / 2]) {
    const uint32_t col = lane >> 2u;
    const uint32_t lane4 = lane & 3u;
#pragma unroll
    for (uint32_t pair = 0; pair < N_GROUPS / 2; pair++) {
        const uint32_t qoff = pair * GROUP_K + lane4 * 8u;
        word0[pair] = load_u32_unaligned(&weights->q4[col][qoff]);
        word1[pair] = load_u32_unaligned(&weights->q4[col][qoff + 4u]);
    }
}

__device__ __forceinline__ static void standard_a_fragments(
        const StandardCase *activation, uint32_t a_tile, uint32_t j, uint32_t lane,
        uint32_t *low, uint32_t *high_signed) {
    const uint32_t row = a_tile * MMA_M + (lane >> 2u);
    const uint32_t k = j * GROUP_K + (lane & 3u) * 8u;
    const uint32_t x0 = load_u32_unaligned(&activation->q8[row][k]);
    const uint32_t x1 = load_u32_unaligned(&activation->q8[row][k + 4u]);
    *low = pack_nibbles_8(x0, x1, 0u);
    *high_signed = pack_nibbles_8(x0, x1, 4u);
}

__device__ __forceinline__ static uint32_t standard_b_fragment_preloaded(
        const uint32_t word0[N_GROUPS / 2],
        const uint32_t word1[N_GROUPS / 2], uint32_t j) {
    const uint32_t pair = j >> 1u;
    return pack_nibbles_8(
        word0[pair], word1[pair], (j & 1u) ? 4u : 0u);
}

__device__ __forceinline__ static TileI32 compute_i8_standard(
        const StandardCase *activation, const StandardCase *weights,
        uint32_t lane) {
    const uint32_t lane4 = lane & 3u;
    const uint32_t col0 = lane4 * 2u;
    uint32_t words[N_GROUPS];
    preload_standard_q4_i8(weights, lane, words);
    TileI32 out = {};
#pragma unroll
    for (uint32_t j = 0; j < N_GROUPS; j++) {
        int32_t c[2 * A_TILES] = {};
        const uint32_t shift = (j & 1u) ? 4u : 0u;
#pragma unroll
        for (uint32_t h = 0; h < 2; h++) {
            const uint32_t k = j * GROUP_K + h * 16u + lane4 * 4u;
            const uint32_t b =
                (words[(j >> 1u) * 2u + h] >> shift) & 0x0f0f0f0fu;
#pragma unroll
            for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
                const uint32_t row = a_tile * MMA_M + (lane >> 2u);
                const uint32_t a = load_u32_unaligned(&activation->q8[row][k]);
                mma_m8n8k16_s8(c[2u * a_tile], c[2u * a_tile + 1u], a, b);
            }
        }
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            out.v[2u * a_tile] +=
                (int32_t)weights->scales[col0][j] * c[2u * a_tile];
            out.v[2u * a_tile + 1u] +=
                (int32_t)weights->scales[col0 + 1u][j] * c[2u * a_tile + 1u];
        }
    }
    return out;
}

__device__ __forceinline__ static TileI32 compute_i4_mixed_standard(
        const StandardCase *activation, const StandardCase *weights,
        uint32_t lane) {
    const uint32_t col0 = (lane & 3u) * 2u;
    uint32_t word0[N_GROUPS / 2], word1[N_GROUPS / 2];
    preload_standard_q4_i4(weights, lane, word0, word1);
    TileI32 out = {};
#pragma unroll
    for (uint32_t j = 0; j < N_GROUPS; j++) {
        const uint32_t b = standard_b_fragment_preloaded(word0, word1, j);
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            uint32_t a_low, a_high;
            standard_a_fragments(activation, a_tile, j, lane, &a_low, &a_high);
            int32_t low0 = 0, low1 = 0, high0 = 0, high1 = 0;
            mma_m8n8k32_u4_u4(low0, low1, a_low, b);
            mma_m8n8k32_s4_u4(high0, high1, a_high, b);
            const int32_t c0 = low0 + 16 * high0;
            const int32_t c1 = low1 + 16 * high1;
            out.v[2u * a_tile] += (int32_t)weights->scales[col0][j] * c0;
            out.v[2u * a_tile + 1u] +=
                (int32_t)weights->scales[col0 + 1u][j] * c1;
        }
    }
    return out;
}

__device__ __forceinline__ static TileI32 compute_i4_mixed_native_w(
        const StandardCase *activation, const NativeCase *weights, uint32_t lane) {
    const uint32_t col0 = (lane & 3u) * 2u;
    TileI32 out = {};
#pragma unroll
    for (uint32_t j = 0; j < N_GROUPS; j++) {
        const uint32_t b = weights->b[j][lane];
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            uint32_t a_low, a_high;
            standard_a_fragments(activation, a_tile, j, lane, &a_low, &a_high);
            int32_t low0 = 0, low1 = 0, high0 = 0, high1 = 0;
            mma_m8n8k32_u4_u4(low0, low1, a_low, b);
            mma_m8n8k32_s4_u4(high0, high1, a_high, b);
            const int32_t c0 = low0 + 16 * high0;
            const int32_t c1 = low1 + 16 * high1;
            out.v[2u * a_tile] += (int32_t)weights->scales[col0][j] * c0;
            out.v[2u * a_tile + 1u] +=
                (int32_t)weights->scales[col0 + 1u][j] * c1;
        }
    }
    return out;
}

__device__ __forceinline__ static TileI32 compute_i4_mixed_group32_w(
        const StandardCase *activation, const NativeCase *weights, uint32_t lane) {
    const uint32_t lane4 = lane & 3u;
    const uint32_t col0 = lane4 * 2u;
    const uint32_t b_col = lane >> 2u;
    TileI32 out = {};
#pragma unroll
    for (uint32_t j = 0; j < N_GROUPS; j++) {
        const uint32_t b = load_u32_unaligned(
            &weights->b_group32[b_col][j][lane4 * 4u]);
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            uint32_t a_low, a_high;
            standard_a_fragments(activation, a_tile, j, lane, &a_low, &a_high);
            int32_t low0 = 0, low1 = 0, high0 = 0, high1 = 0;
            mma_m8n8k32_u4_u4(low0, low1, a_low, b);
            mma_m8n8k32_s4_u4(high0, high1, a_high, b);
            const int32_t c0 = low0 + 16 * high0;
            const int32_t c1 = low1 + 16 * high1;
            out.v[2u * a_tile] += (int32_t)weights->scales[col0][j] * c0;
            out.v[2u * a_tile + 1u] +=
                (int32_t)weights->scales[col0 + 1u][j] * c1;
        }
    }
    return out;
}

__device__ __forceinline__ static TileI32 compute_i4_mixed_native_aw(
        const NativeCase *activation, const NativeCase *weights,
        uint32_t lane) {
    const uint32_t col0 = (lane & 3u) * 2u;
    TileI32 out = {};
#pragma unroll
    for (uint32_t j = 0; j < N_GROUPS; j++) {
        const uint32_t b = weights->b[j][lane];
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            int32_t low0 = 0, low1 = 0, high0 = 0, high1 = 0;
            mma_m8n8k32_u4_u4(
                low0, low1, activation->a_low[a_tile][j][lane], b);
            mma_m8n8k32_s4_u4(
                high0, high1, activation->a_high_signed[a_tile][j][lane], b);
            const int32_t c0 = low0 + 16 * high0;
            const int32_t c1 = low1 + 16 * high1;
            out.v[2u * a_tile] += (int32_t)weights->scales[col0][j] * c0;
            out.v[2u * a_tile + 1u] +=
                (int32_t)weights->scales[col0 + 1u][j] * c1;
        }
    }
    return out;
}

__device__ __forceinline__ static TileI32 compute_i4_u4_corrected_native_aw(
        const NativeCase *activation, const NativeCase *weights,
        uint32_t lane) {
    const uint32_t col0 = (lane & 3u) * 2u;
    TileI32 out = {};
#pragma unroll
    for (uint32_t j = 0; j < N_GROUPS; j++) {
        const uint32_t b = weights->b[j][lane];
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            int32_t low0 = 0, low1 = 0, high0 = 0, high1 = 0;
            mma_m8n8k32_u4_u4(
                low0, low1, activation->a_offset_low[a_tile][j][lane], b);
            mma_m8n8k32_u4_u4(
                high0, high1, activation->a_offset_high[a_tile][j][lane], b);
            const int32_t c0 = low0 + 16 * high0 -
                128 * (int32_t)weights->weight_sums[col0][j];
            const int32_t c1 = low1 + 16 * high1 -
                128 * (int32_t)weights->weight_sums[col0 + 1u][j];
            out.v[2u * a_tile] += (int32_t)weights->scales[col0][j] * c0;
            out.v[2u * a_tile + 1u] +=
                (int32_t)weights->scales[col0 + 1u][j] * c1;
        }
    }
    return out;
}

template <Variant V>
__device__ __forceinline__ static TileI32 compute_variant(
        const StandardCase *standard_activation,
        const StandardCase *standard_weights,
        const NativeCase *native_activation,
        const NativeCase *native_weights,
        uint32_t lane) {
    if (V == VAR_I8_STANDARD) {
        return compute_i8_standard(standard_activation, standard_weights, lane);
    } else if (V == VAR_I4_MIXED_STANDARD) {
        return compute_i4_mixed_standard(
            standard_activation, standard_weights, lane);
    } else if (V == VAR_I4_MIXED_GROUP32_W) {
        return compute_i4_mixed_group32_w(
            standard_activation, native_weights, lane);
    } else if (V == VAR_I4_MIXED_NATIVE_W) {
        return compute_i4_mixed_native_w(
            standard_activation, native_weights, lane);
    } else if (V == VAR_I4_MIXED_NATIVE_AW) {
        return compute_i4_mixed_native_aw(
            native_activation, native_weights, lane);
    } else {
        return compute_i4_u4_corrected_native_aw(
            native_activation, native_weights, lane);
    }
}

template <Variant V>
__device__ __forceinline__ static void run_kernel_body(
        const StandardCase *standard, const NativeCase *native,
        int32_t *exact, uint32_t *checksums,
        uint32_t repeats, uint32_t activation_case_count,
        uint32_t weight_case_count) {
    const uint32_t lane = threadIdx.x & (WARP_SIZE_ - 1u);
    const uint32_t warp = threadIdx.x / WARP_SIZE_;
    const uint64_t work_id =
        (uint64_t)blockIdx.x * WARPS_PER_CTA + (uint64_t)warp;
    const uint64_t work_count =
        (uint64_t)gridDim.x * WARPS_PER_CTA;
    uint32_t check = 0x9e3779b9u ^
        ((uint32_t)work_id * 0x85ebca6bu + lane);
    TileI32 result = {};
    uint32_t activation_ci = (uint32_t)(work_id % activation_case_count);
    uint32_t weight_ci = (uint32_t)(work_id % weight_case_count);
    const uint32_t weight_step = (uint32_t)(work_count % weight_case_count);
    const uint32_t weight_wrap = weight_case_count - weight_step;
    for (uint32_t r = 0; r < repeats; r++) {
        /* Activations deliberately remain in a 16-case hot set. The grid-wide
         * repeat stride walks the independently sized weight set in slabs.
         * Incremental wrapping avoids putting integer division in the loop. */
        result = compute_variant<V>(
            standard + activation_ci, standard + weight_ci,
            native + activation_ci, native + weight_ci, lane);
        check = (check << 5) | (check >> 27);
        check ^= (uint32_t)result.v[0] + 0x7f4a7c15u * r;
        check ^= (uint32_t)result.v[1] * 0x27d4eb2du;
        check ^= (uint32_t)result.v[2] * 0x165667b1u;
        check ^= (uint32_t)result.v[3] * 0xd3a2646cu;
        activation_ci++;
        if (activation_ci == activation_case_count) activation_ci = 0;
        weight_ci = weight_ci >= weight_wrap
            ? weight_ci - weight_wrap : weight_ci + weight_step;
    }
    if (work_id == 0) {
        const uint32_t col0 = (lane & 3u) * 2u;
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            const uint32_t row = a_tile * MMA_M + (lane >> 2u);
            exact[row * TILE_N + col0] = result.v[2u * a_tile];
            exact[row * TILE_N + col0 + 1u] = result.v[2u * a_tile + 1u];
        }
    }
    checksums[work_id * WARP_SIZE_ + lane] = check;
}

extern "C" __global__ void sm75_q4_i8_standard_kernel(
        const StandardCase *standard, const NativeCase *native,
        int32_t *exact, uint32_t *checksums,
        uint32_t repeats, uint32_t activation_case_count,
        uint32_t weight_case_count) {
    (void)native;
    run_kernel_body<VAR_I8_STANDARD>(
        standard, native, exact, checksums, repeats,
        activation_case_count, weight_case_count);
}

extern "C" __global__ void sm75_q4_i4_mixed_standard_kernel(
        const StandardCase *standard, const NativeCase *native,
        int32_t *exact, uint32_t *checksums,
        uint32_t repeats, uint32_t activation_case_count,
        uint32_t weight_case_count) {
    (void)native;
    run_kernel_body<VAR_I4_MIXED_STANDARD>(
        standard, native, exact, checksums, repeats,
        activation_case_count, weight_case_count);
}

extern "C" __global__ void sm75_q4_i4_mixed_native_w_kernel(
        const StandardCase *standard, const NativeCase *native,
        int32_t *exact, uint32_t *checksums,
        uint32_t repeats, uint32_t activation_case_count,
        uint32_t weight_case_count) {
    run_kernel_body<VAR_I4_MIXED_NATIVE_W>(
        standard, native, exact, checksums, repeats,
        activation_case_count, weight_case_count);
}

extern "C" __global__ void sm75_q4_i4_mixed_group32_w_kernel(
        const StandardCase *standard, const NativeCase *native,
        int32_t *exact, uint32_t *checksums,
        uint32_t repeats, uint32_t activation_case_count,
        uint32_t weight_case_count) {
    run_kernel_body<VAR_I4_MIXED_GROUP32_W>(
        standard, native, exact, checksums, repeats,
        activation_case_count, weight_case_count);
}

extern "C" __global__ void sm75_q4_i4_mixed_native_aw_kernel(
        const StandardCase *standard, const NativeCase *native,
        int32_t *exact, uint32_t *checksums,
        uint32_t repeats, uint32_t activation_case_count,
        uint32_t weight_case_count) {
    (void)standard;
    run_kernel_body<VAR_I4_MIXED_NATIVE_AW>(
        standard, native, exact, checksums, repeats,
        activation_case_count, weight_case_count);
}

extern "C" __global__ void sm75_q4_i4_u4_corrected_native_aw_kernel(
        const StandardCase *standard, const NativeCase *native,
        int32_t *exact, uint32_t *checksums,
        uint32_t repeats, uint32_t activation_case_count,
        uint32_t weight_case_count) {
    (void)standard;
    run_kernel_body<VAR_I4_U4_CORRECTED_NATIVE_AW>(
        standard, native, exact, checksums, repeats,
        activation_case_count, weight_case_count);
}

static uint32_t rng_next(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static uint8_t standard_weight(const StandardCase *input,
                               uint32_t col, uint32_t k) {
    const uint32_t j = k / GROUP_K;
    const uint32_t in_group = k % GROUP_K;
    const uint8_t byte = input->q4[col][(j >> 1u) * GROUP_K + in_group];
    return (byte >> ((j & 1u) ? 4u : 0u)) & 0x0fu;
}

static void set_standard_weight(StandardCase *input,
                                uint32_t col, uint32_t k, uint8_t value) {
    const uint32_t j = k / GROUP_K;
    const uint32_t in_group = k % GROUP_K;
    uint8_t *byte = &input->q4[col][(j >> 1u) * GROUP_K + in_group];
    if (j & 1u) *byte = (uint8_t)((*byte & 0x0fu) | ((value & 0x0fu) << 4u));
    else          *byte = (uint8_t)((*byte & 0xf0u) | (value & 0x0fu));
}

static uint32_t host_pack8(const uint8_t values[8]) {
    uint32_t out = 0;
    for (uint32_t i = 0; i < 8; i++) out |= (uint32_t)(values[i] & 0x0fu) << (4u * i);
    return out;
}

static void build_native(const StandardCase *standard, NativeCase *native) {
    memset(native, 0, sizeof(*native));
    memcpy(native->scales, standard->scales, sizeof(native->scales));
    for (uint32_t j = 0; j < N_GROUPS; j++) {
        for (uint32_t lane = 0; lane < WARP_SIZE_; lane++) {
            const uint32_t group = lane >> 2u;
            const uint32_t lane4 = lane & 3u;
            uint8_t b[8];
            for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
                uint8_t a_low[8], a_high[8], a_off_low[8], a_off_high[8];
                for (uint32_t i = 0; i < 8; i++) {
                    const int8_t av = standard->q8[a_tile * MMA_M + group]
                        [j * GROUP_K + lane4 * 8u + i];
                    const uint8_t raw = (uint8_t)av;
                    const uint8_t offset = (uint8_t)((int)av + 128);
                    a_low[i] = raw & 0x0fu;
                    a_high[i] = raw >> 4u;
                    a_off_low[i] = offset & 0x0fu;
                    a_off_high[i] = offset >> 4u;
                }
                native->a_low[a_tile][j][lane] = host_pack8(a_low);
                native->a_high_signed[a_tile][j][lane] = host_pack8(a_high);
                native->a_offset_low[a_tile][j][lane] = host_pack8(a_off_low);
                native->a_offset_high[a_tile][j][lane] = host_pack8(a_off_high);
            }
            for (uint32_t i = 0; i < 8; i++) {
                b[i] = standard_weight(
                    standard, group, j * GROUP_K + lane4 * 8u + i);
            }
            native->b[j][lane] = host_pack8(b);
        }
        for (uint32_t col = 0; col < TILE_N; col++) {
            int sum = 0;
            for (uint32_t i = 0; i < GROUP_K; i += 2u) {
                const uint8_t w0 = standard_weight(
                    standard, col, j * GROUP_K + i);
                const uint8_t w1 = standard_weight(
                    standard, col, j * GROUP_K + i + 1u);
                native->b_group32[col][j][i / 2u] =
                    (uint8_t)(w0 | (w1 << 4u));
                sum += (int)w0 + (int)w1;
            }
            native->weight_sums[col][j] = (int16_t)sum;
        }
    }
}

static void fill_case(StandardCase *input, uint32_t case_id) {
    static const int edge_a[16] = {
        -128, -127, -113, -17, -16, -15, -9, -8,
        -1, 0, 1, 7, 8, 15, 16, 127,
    };
    static const uint8_t edge_w[8] = {0, 1, 7, 8, 9, 14, 15, 3};
    memset(input, 0, sizeof(*input));
    uint32_t state = 0x243f6a88u ^ (case_id * 0x9e3779b9u);
    for (uint32_t row = 0; row < TILE_M; row++) {
        for (uint32_t k = 0; k < TILE_K; k++) {
            int value;
            if (case_id == 0) value = 0;
            else if (case_id == 1) value = 127;
            else if (case_id == 2) value = -128;
            else if (case_id == 3) value = -1;
            else if (case_id == 4) value = ((row + k) & 1u) ? 127 : -128;
            else if (case_id == 5) value = edge_a[(row * 5u + k) & 15u];
            else if (case_id == 6)
                value = (int)((row * 29u + k * 17u + (k >> 5u) * 11u) & 0xffu) - 128;
            else if (case_id == 7)
                value = (k & 2u) ? ((row & 1u) ? -1 : 15)
                                 : ((row & 1u) ? -16 : -128);
            else value = (int)(rng_next(&state) & 0xffu) - 128;
            input->q8[row][k] = (int8_t)value;
        }
    }
    for (uint32_t col = 0; col < TILE_N; col++) {
        for (uint32_t k = 0; k < TILE_K; k++) {
            uint8_t value;
            if (case_id == 0) value = 0;
            else if (case_id <= 3) value = 15;
            else if (case_id == 4) value = ((col + k) & 1u) ? 15u : 0u;
            else if (case_id == 5) value = edge_w[(col * 3u + k) & 7u];
            else if (case_id == 6) value =
                (uint8_t)((col * 13u + k * 7u + (k >> 5u) * 5u) & 0x0fu);
            else if (case_id == 7) value = 0;
            else value = (uint8_t)(rng_next(&state) & 0x0fu);
            set_standard_weight(input, col, k, value);
        }
        for (uint32_t j = 0; j < N_GROUPS; j++) {
            uint8_t scale;
            if (case_id == 0) scale = 0;
            else if (case_id <= 4 || case_id == 7) scale = 63u;
            else if (case_id == 5) scale = edge_w[(col + j) & 7u] * 4u + 3u;
            else if (case_id == 6) scale =
                (uint8_t)(1u + (col * 11u + j * 7u) % 63u);
            else scale = (uint8_t)(rng_next(&state) & 63u);
            input->scales[col][j] = scale;
        }
    }
}

static void reference(const StandardCase *input, int32_t out[TILE_M][TILE_N]) {
    for (uint32_t row = 0; row < TILE_M; row++) {
        for (uint32_t col = 0; col < TILE_N; col++) {
            int32_t total = 0;
            for (uint32_t j = 0; j < N_GROUPS; j++) {
                int32_t dot = 0;
                for (uint32_t i = 0; i < GROUP_K; i++) {
                    dot += (int32_t)input->q8[row][j * GROUP_K + i] *
                           (int32_t)standard_weight(input, col, j * GROUP_K + i);
                }
                total += (int32_t)input->scales[col][j] * dot;
            }
            out[row][col] = total;
        }
    }
}

static Variant parse_variant(const char *name) {
    for (int i = 0; i < VARIANT_COUNT; i++)
        if (strcmp(name, variant_names[i]) == 0) return (Variant)i;
    fprintf(stderr, "error: unknown variant: %s\n", name);
    fprintf(stderr, "variants:");
    for (int i = 0; i < VARIANT_COUNT; i++) fprintf(stderr, " %s", variant_names[i]);
    fprintf(stderr, "\n");
    exit(2);
}

static void launch_variant(Variant variant, uint32_t blocks, uint32_t repeats,
                           uint32_t activation_case_count,
                           uint32_t weight_case_count,
                           const StandardCase *standard, const NativeCase *native,
                           int32_t *exact, uint32_t *checksums, cudaStream_t stream) {
    dim3 grid(blocks), block(THREADS_PER_CTA);
    switch (variant) {
    case VAR_I8_STANDARD:
        sm75_q4_i8_standard_kernel<<<grid, block, 0, stream>>>(
            standard, native, exact, checksums, repeats,
            activation_case_count, weight_case_count); break;
    case VAR_I4_MIXED_STANDARD:
        sm75_q4_i4_mixed_standard_kernel<<<grid, block, 0, stream>>>(
            standard, native, exact, checksums, repeats,
            activation_case_count, weight_case_count); break;
    case VAR_I4_MIXED_GROUP32_W:
        sm75_q4_i4_mixed_group32_w_kernel<<<grid, block, 0, stream>>>(
            standard, native, exact, checksums, repeats,
            activation_case_count, weight_case_count); break;
    case VAR_I4_MIXED_NATIVE_W:
        sm75_q4_i4_mixed_native_w_kernel<<<grid, block, 0, stream>>>(
            standard, native, exact, checksums, repeats,
            activation_case_count, weight_case_count); break;
    case VAR_I4_MIXED_NATIVE_AW:
        sm75_q4_i4_mixed_native_aw_kernel<<<grid, block, 0, stream>>>(
            standard, native, exact, checksums, repeats,
            activation_case_count, weight_case_count); break;
    case VAR_I4_U4_CORRECTED_NATIVE_AW:
        sm75_q4_i4_u4_corrected_native_aw_kernel<<<grid, block, 0, stream>>>(
            standard, native, exact, checksums, repeats,
            activation_case_count, weight_case_count); break;
    default: abort();
    }
}

static int run_correctness(uint32_t n_cases,
                           StandardCase *h_standard, NativeCase *h_native,
                           StandardCase *d_standard, NativeCase *d_native,
                           int32_t *d_exact, uint32_t *d_checksums) {
    int32_t expected[TILE_M][TILE_N];
    int32_t actual[TILE_M][TILE_N];
    for (uint32_t case_id = 0; case_id < n_cases; case_id++) {
        fill_case(h_standard, case_id);
        build_native(h_standard, h_native);
        reference(h_standard, expected);
        die_cuda(cudaMemcpy(d_standard, h_standard, sizeof(*h_standard), cudaMemcpyHostToDevice),
                 "copy standard case");
        die_cuda(cudaMemcpy(d_native, h_native, sizeof(*h_native), cudaMemcpyHostToDevice),
                 "copy native case");
        for (int v = 0; v < VARIANT_COUNT; v++) {
            launch_variant((Variant)v, 1, 1, 1, 1, d_standard, d_native,
                           d_exact, d_checksums, 0);
            die_cuda(cudaGetLastError(), "launch exactness kernel");
            die_cuda(cudaMemcpy(actual, d_exact, sizeof(actual), cudaMemcpyDeviceToHost),
                     "copy exactness output");
            for (uint32_t row = 0; row < TILE_M; row++) {
                for (uint32_t col = 0; col < TILE_N; col++) {
                    if (actual[row][col] != expected[row][col]) {
                        fprintf(stderr,
                            "exactness failure: case=%u variant=%s row=%u col=%u "
                            "expected=%d actual=%d\n",
                            case_id, variant_names[v], row, col,
                            expected[row][col], actual[row][col]);
                        return 1;
                    }
                }
            }
        }
    }
    printf("exact_cases=%u\nexact_variants=%d\nexact_status=ok\n",
           n_cases, VARIANT_COUNT);
    return 0;
}

static int compare_double(const void *lhs, const void *rhs) {
    const double a = *(const double *)lhs;
    const double b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static void run_benchmark(uint32_t blocks, uint32_t repeats, uint32_t launches,
                          uint32_t rounds, uint32_t weight_case_count,
                          uint32_t allocated_case_count,
                          StandardCase *h_standard, NativeCase *h_native,
                          StandardCase *d_standard, NativeCase *d_native,
                          int32_t *d_exact, uint32_t *d_checksums) {
    for (uint32_t i = 0; i < allocated_case_count; i++) {
        fill_case(h_standard + i, 0x5a17u + i);
        build_native(h_standard + i, h_native + i);
    }
    die_cuda(cudaMemcpy(d_standard, h_standard,
                        (size_t)allocated_case_count * sizeof(*h_standard),
                        cudaMemcpyHostToDevice),
             "copy benchmark standard case");
    die_cuda(cudaMemcpy(d_native, h_native,
                        (size_t)allocated_case_count * sizeof(*h_native),
                        cudaMemcpyHostToDevice),
             "copy benchmark native case");

    cudaEvent_t start, stop;
    die_cuda(cudaEventCreate(&start), "create start event");
    die_cuda(cudaEventCreate(&stop), "create stop event");
    double *samples = (double *)malloc(
        (size_t)VARIANT_COUNT * rounds * sizeof(*samples));
    double *sorted = (double *)malloc((size_t)rounds * sizeof(*sorted));
    if (!samples || !sorted) {
        fprintf(stderr, "error: benchmark sample allocation failed\n");
        exit(2);
    }

    for (int v = 0; v < VARIANT_COUNT; v++) {
        for (int warm = 0; warm < 3; warm++)
            launch_variant((Variant)v, blocks, repeats,
                           HOT_ACTIVATION_CASES, weight_case_count,
                           d_standard, d_native,
                           d_exact, d_checksums, 0);
    }
    die_cuda(cudaDeviceSynchronize(), "benchmark warmup");

    for (uint32_t round = 0; round < rounds; round++) {
        const uint32_t start_slot = round % VARIANT_COUNT;
        for (uint32_t slot = 0; slot < VARIANT_COUNT; slot++) {
            const uint32_t offset = (round & 1u)
                ? (VARIANT_COUNT - slot) % VARIANT_COUNT : slot;
            const uint32_t v = (start_slot + offset) % VARIANT_COUNT;
            die_cuda(cudaEventRecord(start), "record start");
            for (uint32_t i = 0; i < launches; i++)
                launch_variant((Variant)v, blocks, repeats,
                               HOT_ACTIVATION_CASES, weight_case_count,
                               d_standard, d_native,
                               d_exact, d_checksums, 0);
            die_cuda(cudaEventRecord(stop), "record stop");
            die_cuda(cudaEventSynchronize(stop), "synchronize benchmark");
            float ms = 0.0f;
            die_cuda(cudaEventElapsedTime(&ms, start, stop), "measure benchmark");
            samples[(size_t)v * rounds + round] = (double)ms;
        }
    }

    const double logical_macs = (double)launches * blocks *
                                WARPS_PER_CTA * repeats *
                                TILE_M * TILE_N * TILE_K;
    printf("benchmark_samples_begin\n");
    printf("sample_round,sample_slot,variant,total_ms,us_per_launch,"
           "relative_speed,logical_tmac_per_s\n");
    for (uint32_t round = 0; round < rounds; round++) {
        const double baseline_ms = samples[round];
        const uint32_t start_slot = round % VARIANT_COUNT;
        for (uint32_t slot = 0; slot < VARIANT_COUNT; slot++) {
            const uint32_t offset = (round & 1u)
                ? (VARIANT_COUNT - slot) % VARIANT_COUNT : slot;
            const uint32_t v = (start_slot + offset) % VARIANT_COUNT;
            const double ms = samples[(size_t)v * rounds + round];
            printf("%u,%u,%s,%.6f,%.6f,%.6f,%.6f\n",
                   round + 1u, slot + 1u, variant_names[v], ms,
                   1000.0 * ms / launches, baseline_ms / ms,
                   logical_macs / (ms / 1000.0) / 1.0e12);
        }
    }
    printf("benchmark_samples_end\n");

    double medians[VARIANT_COUNT], minimums[VARIANT_COUNT], maximums[VARIANT_COUNT];
    for (int v = 0; v < VARIANT_COUNT; v++) {
        memcpy(sorted, samples + (size_t)v * rounds,
               (size_t)rounds * sizeof(*sorted));
        qsort(sorted, rounds, sizeof(*sorted), compare_double);
        minimums[v] = sorted[0];
        maximums[v] = sorted[rounds - 1u];
        medians[v] = (rounds & 1u)
            ? sorted[rounds / 2u]
            : 0.5 * (sorted[rounds / 2u - 1u] + sorted[rounds / 2u]);
    }
    printf("benchmark_summary_begin\n");
    printf("benchmark_summary_statistic=median\n");
    printf("variant,total_ms,min_ms,max_ms,us_per_launch,relative_speed,"
           "logical_tmac_per_s\n");
    for (int v = 0; v < VARIANT_COUNT; v++) {
        const double ms = medians[v];
        printf("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
               variant_names[v], ms, minimums[v], maximums[v],
               1000.0 * ms / launches, medians[0] / ms,
               logical_macs / (ms / 1000.0) / 1.0e12);
    }
    printf("benchmark_summary_end\n");

    free(sorted);
    free(samples);
    die_cuda(cudaEventDestroy(start), "destroy start event");
    die_cuda(cudaEventDestroy(stop), "destroy stop event");
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [--device N] [--cases N] [--blocks N] [--repeats N] "
        "[--launches N] [--bench-cases N] [--rounds N] "
        "[--correctness-only | --benchmark-only | "
        "--profile VARIANT]\n", argv0);
}

static uint32_t parse_u32_allow_zero(const char *text, const char *name) {
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (!text[0] || !end || *end || value > 0xfffffffful) {
        fprintf(stderr, "error: invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint32_t)value;
}

static uint32_t parse_u32(const char *text, const char *name) {
    const uint32_t value = parse_u32_allow_zero(text, name);
    if (value == 0) {
        fprintf(stderr, "error: %s must be greater than zero\n", name);
        exit(2);
    }
    return value;
}

int main(int argc, char **argv) {
    int device = 0;
    uint32_t cases = 256;
    uint32_t blocks = 0;
    uint32_t repeats = 128;
    uint32_t launches = 20;
    uint32_t bench_cases = 16;
    uint32_t rounds = 9;
    int do_correctness = 1, do_benchmark = 1;
    int profile = 0;
    Variant profile_variant = VAR_I8_STANDARD;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            device = (int)parse_u32_allow_zero(argv[++i], "device");
        else if (strcmp(argv[i], "--cases") == 0 && i + 1 < argc)
            cases = parse_u32(argv[++i], "cases");
        else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc)
            blocks = parse_u32(argv[++i], "blocks");
        else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc)
            repeats = parse_u32(argv[++i], "repeats");
        else if (strcmp(argv[i], "--launches") == 0 && i + 1 < argc)
            launches = parse_u32(argv[++i], "launches");
        else if (strcmp(argv[i], "--bench-cases") == 0 && i + 1 < argc)
            bench_cases = parse_u32(argv[++i], "bench-cases");
        else if (strcmp(argv[i], "--rounds") == 0 && i + 1 < argc)
            rounds = parse_u32(argv[++i], "rounds");
        else if (strcmp(argv[i], "--correctness-only") == 0) {
            do_correctness = 1; do_benchmark = 0;
        } else if (strcmp(argv[i], "--benchmark-only") == 0) {
            do_correctness = 0; do_benchmark = 1;
        } else if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            profile = 1; do_correctness = 0; do_benchmark = 0;
            profile_variant = parse_variant(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); return 0;
        } else {
            usage(argv[0]); return 2;
        }
    }

    die_cuda(cudaSetDevice(device), "select CUDA device");
    cudaDeviceProp prop;
    die_cuda(cudaGetDeviceProperties(&prop, device), "query CUDA device");
    if (prop.major != 7 || prop.minor != 5) {
        fprintf(stderr, "error: %s is sm_%d%d; this experiment requires sm_75\n",
                prop.name, prop.major, prop.minor);
        return 2;
    }
    if (!blocks) blocks = (uint32_t)prop.multiProcessorCount * 2u;
    const uint32_t allocated_cases = bench_cases > HOT_ACTIVATION_CASES
        ? bench_cases : HOT_ACTIVATION_CASES;
    printf("device=%d\ndevice_name=%s\ncompute_capability=%d.%d\nsm_count=%d\n",
           device, prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("threads_per_cta=%d\nwarps_per_cta=%d\ntokens_per_warp=%d\n",
           THREADS_PER_CTA, WARPS_PER_CTA, TILE_M);
    printf("hot_activation_cases=%d\nweight_cases=%u\nbenchmark_rounds=%u\n",
           HOT_ACTIVATION_CASES, bench_cases, rounds);
    printf("q8_value_bytes_standard=%zu\nq8_value_bytes_native=%zu\n",
           sizeof(((StandardCase *)0)->q8),
           sizeof(((NativeCase *)0)->a_low) + sizeof(((NativeCase *)0)->a_high_signed));
    printf("q4_value_bytes_standard=%zu\nq4_value_bytes_native=%zu\n",
           sizeof(((StandardCase *)0)->q4), sizeof(((NativeCase *)0)->b));
    printf("q4_group32_value_bytes=%zu\nq4_u4_weight_sum_bytes=%zu\n"
           "q4_u4_effective_block_bytes=%u\nq4_standard_block_bytes=%u\n",
           sizeof(((NativeCase *)0)->b_group32),
           sizeof(((NativeCase *)0)->weight_sums),
           160u, 144u);

    StandardCase *h_standard =
        (StandardCase *)malloc((size_t)allocated_cases * sizeof(StandardCase));
    NativeCase *h_native =
        (NativeCase *)malloc((size_t)allocated_cases * sizeof(NativeCase));
    if (!h_standard || !h_native) {
        fprintf(stderr, "error: host allocation failed\n");
        return 2;
    }
    StandardCase *d_standard = NULL;
    NativeCase *d_native = NULL;
    int32_t *d_exact = NULL;
    uint32_t *d_checksums = NULL;
    die_cuda(cudaMalloc(&d_standard,
                        (size_t)allocated_cases * sizeof(*d_standard)),
             "allocate standard cases");
    die_cuda(cudaMalloc(&d_native,
                        (size_t)allocated_cases * sizeof(*d_native)),
             "allocate native cases");
    die_cuda(cudaMalloc(&d_exact, TILE_M * TILE_N * sizeof(int32_t)), "allocate exact output");
    die_cuda(cudaMalloc(&d_checksums,
                        (uint64_t)blocks * WARPS_PER_CTA *
                            WARP_SIZE_ * sizeof(uint32_t)),
             "allocate checksums");

    int rc = 0;
    if (do_correctness)
        rc = run_correctness(cases, h_standard, h_native, d_standard, d_native,
                             d_exact, d_checksums);
    if (!rc && do_benchmark) {
        printf("benchmark_blocks=%u\nbenchmark_repeats=%u\nbenchmark_launches=%u\n"
               "benchmark_weight_cases=%u\nbenchmark_rounds=%u\n",
               blocks, repeats, launches, bench_cases, rounds);
        run_benchmark(blocks, repeats, launches, rounds, bench_cases,
                      allocated_cases, h_standard, h_native,
                      d_standard, d_native, d_exact, d_checksums);
    }
    if (!rc && profile) {
        for (uint32_t i = 0; i < allocated_cases; i++) {
            fill_case(h_standard + i, 0x5a17u + i);
            build_native(h_standard + i, h_native + i);
        }
        die_cuda(cudaMemcpy(d_standard, h_standard,
                            (size_t)allocated_cases * sizeof(*h_standard),
                            cudaMemcpyHostToDevice),
                 "copy profile standard case");
        die_cuda(cudaMemcpy(d_native, h_native,
                            (size_t)allocated_cases * sizeof(*h_native),
                            cudaMemcpyHostToDevice),
                 "copy profile native case");
        printf("profile_variant=%s\nprofile_blocks=%u\nprofile_repeats=%u\n"
               "profile_weight_cases=%u\n",
               variant_names[profile_variant], blocks, repeats, bench_cases);
        launch_variant(profile_variant, blocks, repeats,
                       HOT_ACTIVATION_CASES, bench_cases,
                       d_standard, d_native,
                       d_exact, d_checksums, 0);
        die_cuda(cudaGetLastError(), "launch profile kernel");
        die_cuda(cudaDeviceSynchronize(), "synchronize profile kernel");
        printf("profile_status=ok\n");
    }

    cudaFree(d_checksums);
    cudaFree(d_exact);
    cudaFree(d_native);
    cudaFree(d_standard);
    free(h_native);
    free(h_standard);
    if (!rc) printf("harness_status=ok\n");
    return rc;
}
