/*
 * Bounded SM75 arithmetic/layout harness for the routed-quant performance
 * screen.  It covers the standard Q3_K/Q4_K controls plus the compact K32
 * candidates that survived the real-weight quality gate.
 *
 * This file intentionally contains no quantizer.  Opaque fp16 headers are
 * carried only to prove the byte layouts; all checked arithmetic is integer.
 * One warp computes M16xN8xK256 and eight warps run in each CTA.
 */

#include <cuda_runtime.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 750
#error "cuda_sm75_q3_q4_32 must be compiled exclusively for sm_75"
#endif

enum {
    MMA_M = 8,
    A_TILES = 2,
    TILE_M = MMA_M * A_TILES,
    TILE_N = 8,
    TILE_K = 256,
    K16 = 16,
    K32 = 32,
    K16_GROUPS = TILE_K / K16,
    K32_GROUPS = TILE_K / K32,
    WARP_SIZE_ = 32,
    WARPS_PER_CTA = 8,
    THREADS_PER_CTA = WARP_SIZE_ * WARPS_PER_CTA,
    HOT_ACTIVATION_CASES = 16,
    HOT_WEIGHT_CASES = 16,
};

struct Q4KBlock {
    uint16_t d_bits;
    uint16_t dmin_bits;
    uint8_t scales_mins[12];
    uint8_t qs[128];
};

/* Exact GGML Q3_K field order and K16 semantics. */
struct Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d_bits;
};

struct SM75Q3_32Block {
    uint16_t d_bits;
    uint8_t scales[6];
    uint8_t qs[96];
};

struct SM75Q4_32Block {
    uint16_t d_bits;
    uint8_t scales[6];
    uint8_t qs[128];
};

struct __attribute__((packed)) SM75Q2Q3_50Block {
    uint16_t d_bits;
    uint8_t scales[6];
    uint8_t low2[64];
    uint8_t promoted_mask;
    uint8_t high[16];
};

struct __attribute__((packed)) SM75Q2Q3_75Block {
    uint16_t d_bits;
    uint8_t scales[6];
    uint8_t low2[64];
    uint8_t promoted_mask;
    uint8_t high[24];
};

struct SM75Q3A4Block {
    uint16_t d_bits;
    uint16_t dmin_bits;
    uint8_t scales[4];
    uint8_t mins[4];
    uint8_t qs[96];
};

struct SM75Q3A6Block {
    uint16_t d_bits;
    uint16_t dmin_bits;
    uint8_t scales[6];
    uint8_t mins[6];
    uint8_t qs[96];
};

struct __attribute__((packed)) SM75Q3Q4_50Block {
    uint16_t d_bits;
    uint8_t scales[6];
    uint8_t low3[96];
    uint8_t promoted_mask;
    uint8_t high[16];
};

struct SM75Q5_32Block {
    uint16_t d_bits;
    uint8_t scales[6];
    uint8_t qs[160];
};

struct alignas(16) NativeQ4KHeader {
    uint16_t d_bits;
    uint16_t dmin_bits;
    uint8_t scales_mins[12];
};

struct alignas(16) NativeQ4KTile {
    /* Exact production header and B-fragment ordering. */
    NativeQ4KHeader hdr[TILE_N];
    uint32_t b[K32_GROUPS][WARP_SIZE_];
};

struct alignas(16) NativeQ3KTile {
    uint16_t d_bits[TILE_N];
    uint8_t scales[TILE_N][12];
    /* Every K16 payload is low2[one byte/lane] + high[one nibble/lane]. */
    uint8_t low2[K16_GROUPS][WARP_SIZE_];
    uint8_t high[K16_GROUPS][WARP_SIZE_ / 2];
};

struct alignas(16) NativeQ3_32Tile {
    uint16_t d_bits[TILE_N];
    uint8_t scales[TILE_N][6];
    /* One lane owns eight values: sixteen low bits plus eight high bits. */
    uint16_t low2[K32_GROUPS][WARP_SIZE_];
    uint8_t high[K32_GROUPS][WARP_SIZE_];
};

struct alignas(16) NativeQ4_32Tile {
    uint16_t d_bits[TILE_N];
    uint8_t scales[TILE_N][6];
    uint32_t b[K32_GROUPS][WARP_SIZE_];
};

/* Q2/Q3 and Q3/Q4 use a sparse high plane.  The masks are per output row;
 * promoted groups remain compact and are found with popcount at execution.
 * These types deliberately have no alignas attribute because their exact
 * eight-row sizes are not multiples of 16.  cudaMalloc aligns the first tile;
 * the alternating alignment of streamed tiles is part of the real format
 * cost rather than hidden padding. */
struct NativeQ2Q3_50Tile {
    uint16_t d_bits[TILE_N];
    uint8_t scales[TILE_N][6];
    uint8_t promoted_mask[TILE_N];
    uint16_t low2[K32_GROUPS][WARP_SIZE_];
    uint8_t high[TILE_N][16];
};

struct NativeQ2Q3_75Tile {
    uint16_t d_bits[TILE_N];
    uint8_t scales[TILE_N][6];
    uint8_t promoted_mask[TILE_N];
    uint16_t low2[K32_GROUPS][WARP_SIZE_];
    uint8_t high[TILE_N][24];
};

struct alignas(16) NativeQ3A4Tile {
    uint16_t d_bits[TILE_N];
    uint16_t dmin_bits[TILE_N];
    uint8_t scales[TILE_N][4];
    uint8_t mins[TILE_N][4];
    uint16_t low2[K32_GROUPS][WARP_SIZE_];
    uint8_t high[K32_GROUPS][WARP_SIZE_];
};

struct alignas(16) NativeQ3A6Tile {
    uint16_t d_bits[TILE_N];
    uint16_t dmin_bits[TILE_N];
    uint8_t scales[TILE_N][6];
    uint8_t mins[TILE_N][6];
    uint16_t low2[K32_GROUPS][WARP_SIZE_];
    uint8_t high[K32_GROUPS][WARP_SIZE_];
};

struct NativeQ3Q4_50Tile {
    uint16_t d_bits[TILE_N];
    uint8_t scales[TILE_N][6];
    uint8_t promoted_mask[TILE_N];
    uint16_t low2[K32_GROUPS][WARP_SIZE_];
    uint8_t high3[K32_GROUPS][WARP_SIZE_];
    uint8_t high4_biased[TILE_N][16];
};

struct alignas(16) NativeQ5_32Tile {
    uint16_t d_bits[TILE_N];
    uint8_t scales[TILE_N][6];
    uint32_t low4[K32_GROUPS][WARP_SIZE_];
    uint8_t high[K32_GROUPS][WARP_SIZE_];
};

struct ActivationCase {
    int8_t q8[TILE_M][TILE_K];
    int16_t bsums[TILE_M][K16_GROUPS];
};

static_assert(sizeof(Q4KBlock) == 144, "canonical Q4_K block must be 144 bytes");
static_assert(sizeof(Q3KBlock) == 110, "canonical Q3_K block must be 110 bytes");
static_assert(sizeof(SM75Q3_32Block) == 104, "canonical Q3-32 block must be 104 bytes");
static_assert(sizeof(SM75Q4_32Block) == 136, "canonical Q4-32 block must be 136 bytes");
static_assert(sizeof(SM75Q2Q3_50Block) == 89, "canonical Q2/Q3-50 block size");
static_assert(sizeof(SM75Q2Q3_75Block) == 97, "canonical Q2/Q3-75 block size");
static_assert(sizeof(SM75Q3A4Block) == 108, "canonical Q3A4 block size");
static_assert(sizeof(SM75Q3A6Block) == 112, "canonical Q3A6 block size");
static_assert(sizeof(SM75Q3Q4_50Block) == 121, "canonical Q3/Q4-50 block size");
static_assert(sizeof(SM75Q5_32Block) == 168, "canonical Q5-32 block size");
static_assert(offsetof(Q3KBlock, d_bits) == 108, "canonical Q3_K fp16 offset");
static_assert(offsetof(SM75Q3_32Block, qs) == 8, "canonical Q3-32 payload offset");
static_assert(offsetof(SM75Q4_32Block, qs) == 8, "canonical Q4-32 payload offset");
static_assert(sizeof(NativeQ4KTile) == 1152, "native Q4_K tile must be 1152 bytes");
static_assert(alignof(NativeQ4KTile) == 16, "native Q4_K tile alignment");
static_assert(sizeof(NativeQ4KHeader) == 16, "native Q4_K row header must be 16 bytes");
static_assert(offsetof(NativeQ4KHeader, dmin_bits) == 2,
              "native Q4_K dmin offset");
static_assert(offsetof(NativeQ4KHeader, scales_mins) == 4,
              "native Q4_K scales/mins offset");
static_assert(offsetof(NativeQ4KTile, b) == 128,
              "native Q4_K MMA payload offset");
static_assert(sizeof(NativeQ3KTile) == 880, "native Q3_K tile must be 880 bytes");
static_assert(alignof(NativeQ3KTile) == 16, "native Q3_K tile alignment");
static_assert(offsetof(NativeQ3KTile, low2) == 112,
              "native Q3_K low-plane offset");
static_assert(offsetof(NativeQ3KTile, high) == 624,
              "native Q3_K high-plane offset");
static_assert(sizeof(NativeQ3_32Tile) == 832, "native Q3-32 tile must be 832 bytes");
static_assert(alignof(NativeQ3_32Tile) == 16, "native Q3-32 tile alignment");
static_assert(offsetof(NativeQ3_32Tile, low2) == 64,
              "native Q3-32 low-plane offset");
static_assert(offsetof(NativeQ3_32Tile, high) == 576,
              "native Q3-32 high-plane offset");
static_assert(sizeof(NativeQ4_32Tile) == 1088, "native Q4-32 tile must be 1088 bytes");
static_assert(alignof(NativeQ4_32Tile) == 16, "native Q4-32 tile alignment");
static_assert(offsetof(NativeQ4_32Tile, b) == 64,
              "native Q4-32 MMA payload offset");
static_assert(sizeof(NativeQ4KTile) == TILE_N * sizeof(Q4KBlock),
              "native Q4_K tile must be size-neutral");
static_assert(sizeof(NativeQ3KTile) == TILE_N * sizeof(Q3KBlock),
              "native Q3_K tile must be size-neutral");
static_assert(sizeof(NativeQ3_32Tile) == TILE_N * sizeof(SM75Q3_32Block),
              "native Q3-32 tile must be size-neutral");
static_assert(sizeof(NativeQ4_32Tile) == TILE_N * sizeof(SM75Q4_32Block),
              "native Q4-32 tile must be size-neutral");
static_assert(sizeof(((NativeQ3KTile *)0)->low2) +
              sizeof(((NativeQ3KTile *)0)->high) == 768,
              "native Q3_K packed values must remain three bits/weight");
static_assert(sizeof(((NativeQ3_32Tile *)0)->low2) +
              sizeof(((NativeQ3_32Tile *)0)->high) == 768,
              "native Q3-32 packed values must remain three bits/weight");
static_assert(sizeof(((NativeQ4_32Tile *)0)->b) == 1024,
              "native Q4-32 value payload");
static_assert(sizeof(((NativeQ4KTile *)0)->b) == 1024,
              "native Q4_K value payload");
static_assert(sizeof(NativeQ2Q3_50Tile) == 712, "native Q2/Q3-50 tile size");
static_assert(sizeof(NativeQ2Q3_75Tile) == 776, "native Q2/Q3-75 tile size");
static_assert(sizeof(NativeQ3A4Tile) == 864, "native Q3A4 tile size");
static_assert(sizeof(NativeQ3A6Tile) == 896, "native Q3A6 tile size");
static_assert(sizeof(NativeQ3Q4_50Tile) == 968, "native Q3/Q4-50 tile size");
static_assert(sizeof(NativeQ5_32Tile) == 1344, "native Q5-32 tile size");
static_assert(sizeof(NativeQ2Q3_50Tile) ==
              TILE_N * sizeof(SM75Q2Q3_50Block),
              "native Q2/Q3-50 must be size-neutral");
static_assert(sizeof(NativeQ2Q3_75Tile) ==
              TILE_N * sizeof(SM75Q2Q3_75Block),
              "native Q2/Q3-75 must be size-neutral");
static_assert(sizeof(NativeQ3A4Tile) == TILE_N * sizeof(SM75Q3A4Block),
              "native Q3A4 must be size-neutral");
static_assert(sizeof(NativeQ3A6Tile) == TILE_N * sizeof(SM75Q3A6Block),
              "native Q3A6 must be size-neutral");
static_assert(sizeof(NativeQ3Q4_50Tile) ==
              TILE_N * sizeof(SM75Q3Q4_50Block),
              "native Q3/Q4-50 must be size-neutral");
static_assert(sizeof(NativeQ5_32Tile) == TILE_N * sizeof(SM75Q5_32Block),
              "native Q5-32 must be size-neutral");

enum Variant {
    VAR_Q4K_NATIVE_CONTROL = 0,
    VAR_Q3K_K16_U8,
    VAR_SM75_Q3_32,
    VAR_SM75_Q4_32,
    VAR_SM75_Q2Q3_50,
    VAR_SM75_Q2Q3_75,
    VAR_SM75_Q3A4,
    VAR_SM75_Q3A6,
    VAR_SM75_Q3Q4_50,
    VAR_SM75_Q5_32,
    VARIANT_COUNT,
};

static const char *const variant_names[VARIANT_COUNT] = {
    "q4k-native-control",
    "q3k-k16-u8",
    "sm75-q3-32",
    "sm75-q4-32",
    "sm75-q2q3-32-50",
    "sm75-q2q3-32-75",
    "sm75-q3a-32-4",
    "sm75-q3a-32-6",
    "sm75-q3q4-32-50",
    "sm75-q5-32",
};

struct CanonicalWeights {
    Q4KBlock q4k[TILE_N];
    Q3KBlock q3k[TILE_N];
    SM75Q3_32Block q3_32[TILE_N];
    SM75Q4_32Block q4_32[TILE_N];
    SM75Q2Q3_50Block q2q3_50[TILE_N];
    SM75Q2Q3_75Block q2q3_75[TILE_N];
    SM75Q3A4Block q3a4[TILE_N];
    SM75Q3A6Block q3a6[TILE_N];
    SM75Q3Q4_50Block q3q4_50[TILE_N];
    SM75Q5_32Block q5_32[TILE_N];
};

struct HostWeights {
    NativeQ4KTile *q4k;
    NativeQ3KTile *q3k;
    NativeQ3_32Tile *q3_32;
    NativeQ4_32Tile *q4_32;
    NativeQ2Q3_50Tile *q2q3_50;
    NativeQ2Q3_75Tile *q2q3_75;
    NativeQ3A4Tile *q3a4;
    NativeQ3A6Tile *q3a6;
    NativeQ3Q4_50Tile *q3q4_50;
    NativeQ5_32Tile *q5_32;
};

struct DeviceWeights {
    NativeQ4KTile *q4k;
    NativeQ3KTile *q3k;
    NativeQ3_32Tile *q3_32;
    NativeQ4_32Tile *q4_32;
    NativeQ2Q3_50Tile *q2q3_50;
    NativeQ2Q3_75Tile *q2q3_75;
    NativeQ3A4Tile *q3a4;
    NativeQ3A6Tile *q3a6;
    NativeQ3Q4_50Tile *q3q4_50;
    NativeQ5_32Tile *q5_32;
};

struct TileI32 {
    /* [M tile 0 col 0/1, M tile 1 col 0/1] for this lane.  Q4_K keeps
     * its affine scale and minimum chains separate, as production must. */
    int32_t v[2 * A_TILES];
    int32_t correction[2 * A_TILES];
};

__host__ __device__ __forceinline__ static int32_t tile_value(
        const TileI32 &tile, uint32_t i) {
    return tile.v[i] - tile.correction[i];
}

static void die_cuda(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return;
    fprintf(stderr, "error: %s: %s\n", what, cudaGetErrorString(err));
    exit(2);
}

static uint32_t rng_next(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static uint8_t rotate_mask8(uint8_t mask, uint32_t shift) {
    shift &= 7u;
    return shift ? (uint8_t)((mask << shift) | (mask >> (8u - shift))) : mask;
}

__host__ __device__ __forceinline__ static int32_t decode_q3k_scale(
        const uint8_t packed[12], uint32_t j) {
    const uint32_t low = j < 8u
        ? packed[j] & 0x0fu : packed[j - 8u] >> 4u;
    const uint32_t high =
        (packed[8u + (j & 3u)] >> (2u * (j >> 2u))) & 3u;
    return (int32_t)(low | (high << 4u)) - 32;
}

static void encode_q3k_scales(const int8_t scales[K16_GROUPS],
                              uint8_t packed[12]) {
    memset(packed, 0, 12);
    for (uint32_t j = 0; j < K16_GROUPS; j++) {
        const uint32_t code = (uint32_t)((int32_t)scales[j] + 32);
        if (j < 8u) packed[j] |= (uint8_t)(code & 0x0fu);
        else packed[j - 8u] |= (uint8_t)((code & 0x0fu) << 4u);
        packed[8u + (j & 3u)] |=
            (uint8_t)((code >> 4u) << (2u * (j >> 2u)));
    }
}

__host__ __device__ __forceinline__ static int32_t decode_scale6_8(
        const uint8_t packed[6], uint32_t j) {
    const uint32_t low = j < 4u
        ? packed[j] & 0x0fu : packed[j - 4u] >> 4u;
    const uint32_t high = (packed[4u + (j >> 2u)] >>
                           (2u * (j & 3u))) & 3u;
    return (int32_t)(low | (high << 4u)) - 32;
}

static void encode_scale6_8(const int8_t scales[K32_GROUPS],
                            uint8_t packed[6]) {
    memset(packed, 0, 6);
    for (uint32_t j = 0; j < K32_GROUPS; j++) {
        const uint32_t code = (uint32_t)((int32_t)scales[j] + 32);
        if (j < 4u) packed[j] |= (uint8_t)(code & 0x0fu);
        else packed[j - 4u] |= (uint8_t)((code & 0x0fu) << 4u);
        packed[4u + (j >> 2u)] |=
            (uint8_t)((code >> 4u) << (2u * (j & 3u)));
    }
}

__host__ __device__ __forceinline__ static uint32_t unpack_bits(
        const uint8_t *packed, uint32_t index, uint32_t bits) {
    const uint32_t bit = index * bits;
    const uint32_t byte = bit >> 3u;
    const uint32_t shift = bit & 7u;
    uint32_t value = packed[byte] >> shift;
    if (shift + bits > 8u)
        value |= (uint32_t)packed[byte + 1u] << (8u - shift);
    return value & ((1u << bits) - 1u);
}

static void pack_bits(const uint8_t *values, uint32_t count, uint32_t bits,
                      uint8_t *packed) {
    const size_t bytes = ((size_t)count * bits + 7u) / 8u;
    memset(packed, 0, bytes);
    for (uint32_t i = 0; i < count; i++) {
        const uint32_t bit = i * bits;
        const uint32_t byte = bit >> 3u;
        const uint32_t shift = bit & 7u;
        const uint32_t value = values[i] & ((1u << bits) - 1u);
        packed[byte] |= (uint8_t)(value << shift);
        if (shift + bits > 8u)
            packed[byte + 1u] |= (uint8_t)(value >> (8u - shift));
    }
}

__host__ __device__ __forceinline__ static uint32_t popcount8(uint8_t value) {
    uint32_t count = 0;
    for (; value; value &= (uint8_t)(value - 1u)) count++;
    return count;
}

__host__ __device__ __forceinline__ static uint32_t decode_unsigned_field8(
        const uint8_t *packed, uint32_t bits, uint32_t group) {
    return unpack_bits(packed, group, bits);
}

static void encode_unsigned_fields8(const uint8_t values[K32_GROUPS],
                                    uint32_t bits, uint8_t *packed) {
    pack_bits(values, K32_GROUPS, bits, packed);
}

__host__ __device__ __forceinline__ static void decode_q4k_scale_min(
        const uint8_t packed[12], uint32_t j, uint32_t *scale,
        uint32_t *minimum) {
    if (j < 4u) {
        *scale = packed[j] & 63u;
        *minimum = packed[j + 4u] & 63u;
    } else {
        *scale = (packed[j + 4u] & 0x0fu) |
                 ((uint32_t)(packed[j - 4u] >> 6u) << 4u);
        *minimum = (packed[j + 4u] >> 4u) |
                   ((uint32_t)(packed[j] >> 6u) << 4u);
    }
}

static void encode_q4k_scale_mins(const uint8_t scales[K32_GROUPS],
                                  const uint8_t mins[K32_GROUPS],
                                  uint8_t packed[12]) {
    memset(packed, 0, 12);
    for (uint32_t j = 0; j < 4; j++) {
        packed[j] = scales[j] & 63u;
        packed[j + 4u] = mins[j] & 63u;
    }
    for (uint32_t j = 4; j < K32_GROUPS; j++) {
        packed[j + 4u] = (uint8_t)((scales[j] & 0x0fu) |
                                    ((mins[j] & 0x0fu) << 4u));
        packed[j - 4u] |= (uint8_t)((scales[j] >> 4u) << 6u);
        packed[j] |= (uint8_t)((mins[j] >> 4u) << 6u);
    }
}

static uint8_t canonical_q4k_value(const Q4KBlock *block, uint32_t k) {
    const uint32_t j = k / K32;
    const uint8_t byte = block->qs[(j >> 1u) * K32 + k % K32];
    return (byte >> (4u * (j & 1u))) & 0x0fu;
}

static void set_canonical_q4k_value(Q4KBlock *block, uint32_t k,
                                    uint8_t value) {
    const uint32_t j = k / K32;
    uint8_t *byte = &block->qs[(j >> 1u) * K32 + k % K32];
    if (j & 1u) *byte = (uint8_t)((*byte & 0x0fu) | ((value & 15u) << 4u));
    else *byte = (uint8_t)((*byte & 0xf0u) | (value & 15u));
}

static int8_t canonical_q3k_value(const Q3KBlock *block, uint32_t k) {
    const uint32_t qindex = (k / 128u) * 32u + k % 32u;
    const uint32_t shift = 2u * ((k % 128u) / 32u);
    const uint32_t low = (block->qs[qindex] >> shift) & 3u;
    const uint32_t high = (block->hmask[k % 32u] >> (k / 32u)) & 1u;
    return (int8_t)((int32_t)low - (high ? 0 : 4));
}

static void set_canonical_q3k_value(Q3KBlock *block, uint32_t k,
                                    int8_t value) {
    const uint32_t u = (uint32_t)((int32_t)value + 4);
    const uint32_t qindex = (k / 128u) * 32u + k % 32u;
    const uint32_t shift = 2u * ((k % 128u) / 32u);
    const uint8_t mask = (uint8_t)(3u << shift);
    block->qs[qindex] = (uint8_t)((block->qs[qindex] & ~mask) |
                                  ((u & 3u) << shift));
    const uint8_t hbit = (uint8_t)(1u << (k / 32u));
    if (u & 4u) block->hmask[k % 32u] |= hbit;
    else block->hmask[k % 32u] &= (uint8_t)~hbit;
}

static uint8_t canonical_q3_32_code(const SM75Q3_32Block *block,
                                    uint32_t k) {
    const uint32_t low = (block->qs[k / 4u] >> (2u * (k & 3u))) & 3u;
    const uint32_t high =
        (block->qs[64u + k / 8u] >> (k & 7u)) & 1u;
    return (uint8_t)(low | (high << 2u));
}

static void set_canonical_q3_32_code(SM75Q3_32Block *block, uint32_t k,
                                     uint8_t code) {
    const uint32_t shift = 2u * (k & 3u);
    const uint8_t mask = (uint8_t)(3u << shift);
    block->qs[k / 4u] = (uint8_t)((block->qs[k / 4u] & ~mask) |
                                  ((code & 3u) << shift));
    const uint8_t hbit = (uint8_t)(1u << (k & 7u));
    if (code & 4u) block->qs[64u + k / 8u] |= hbit;
    else block->qs[64u + k / 8u] &= (uint8_t)~hbit;
}

static int8_t canonical_q4_32_value(const SM75Q4_32Block *block,
                                    uint32_t k) {
    const uint8_t byte = block->qs[k / 2u];
    const uint32_t nibble = (byte >> (4u * (k & 1u))) & 15u;
    return (int8_t)(nibble < 8u ? (int32_t)nibble : (int32_t)nibble - 16);
}

static void set_canonical_q4_32_value(SM75Q4_32Block *block, uint32_t k,
                                      int8_t value) {
    const uint8_t code = (uint8_t)value & 15u;
    uint8_t *byte = &block->qs[k / 2u];
    if (k & 1u) *byte = (uint8_t)((*byte & 0x0fu) | (code << 4u));
    else *byte = (uint8_t)((*byte & 0xf0u) | code);
}

static uint32_t sparse_high_bit(const uint8_t *high, uint8_t promoted_mask,
                                uint32_t group, uint32_t index_in_group) {
    if (!(promoted_mask & (1u << group))) return 0u;
    const uint32_t ordinal = popcount8(
        (uint8_t)(promoted_mask & ((1u << group) - 1u)));
    return (high[ordinal * 4u + index_in_group / 8u] >>
            (index_in_group & 7u)) & 1u;
}

static int8_t canonical_q2q3_value(const uint8_t *low2,
                                   uint8_t promoted_mask,
                                   const uint8_t *high, uint32_t k) {
    const uint32_t group = k / K32;
    uint32_t code = unpack_bits(low2, k, 2u);
    if (promoted_mask & (1u << group))
        code |= sparse_high_bit(high, promoted_mask, group, k & 31u) << 2u;
    const int32_t bias = (promoted_mask & (1u << group)) ? 4 : 2;
    return (int8_t)((int32_t)code - bias);
}

static int8_t canonical_q3q4_value(const SM75Q3Q4_50Block *block,
                                   uint32_t k) {
    const uint32_t group = k / K32;
    uint32_t code = unpack_bits(block->low3, k, 3u);
    if (!(block->promoted_mask & (1u << group)))
        return (int8_t)((int32_t)code - 4);
    code |= sparse_high_bit(block->high, block->promoted_mask,
                            group, k & 31u) << 3u;
    return (int8_t)(code < 8u ? (int32_t)code : (int32_t)code - 16);
}

static uint8_t canonical_affine_value(const uint8_t qs[96], uint32_t k) {
    return (uint8_t)unpack_bits(qs, k, 3u);
}

static int8_t canonical_q5_value(const SM75Q5_32Block *block, uint32_t k) {
    return (int8_t)((int32_t)unpack_bits(block->qs, k, 5u) - 16);
}

static void encode_q4k_block(Q4KBlock *block, const uint8_t values[TILE_K],
                             const uint8_t scales[K32_GROUPS],
                             const uint8_t mins[K32_GROUPS], uint16_t d_bits,
                             uint16_t dmin_bits) {
    memset(block, 0, sizeof(*block));
    block->d_bits = d_bits;
    block->dmin_bits = dmin_bits;
    encode_q4k_scale_mins(scales, mins, block->scales_mins);
    for (uint32_t k = 0; k < TILE_K; k++)
        set_canonical_q4k_value(block, k, values[k]);
}

static void encode_q3k_block(Q3KBlock *block, const int8_t values[TILE_K],
                             const int8_t scales[K16_GROUPS], uint16_t d_bits) {
    memset(block, 0, sizeof(*block));
    block->d_bits = d_bits;
    encode_q3k_scales(scales, block->scales);
    for (uint32_t k = 0; k < TILE_K; k++)
        set_canonical_q3k_value(block, k, values[k]);
}

static void encode_q3_32_block(SM75Q3_32Block *block,
                               const int8_t values[TILE_K],
                               const int8_t scales[K32_GROUPS],
                               uint16_t d_bits) {
    memset(block, 0, sizeof(*block));
    block->d_bits = d_bits;
    encode_scale6_8(scales, block->scales);
    for (uint32_t k = 0; k < TILE_K; k++)
        set_canonical_q3_32_code(block, k,
            (uint8_t)((int32_t)values[k] + 4));
}

static void encode_q4_32_block(SM75Q4_32Block *block,
                               const int8_t values[TILE_K],
                               const int8_t scales[K32_GROUPS],
                               uint16_t d_bits) {
    memset(block, 0, sizeof(*block));
    block->d_bits = d_bits;
    encode_scale6_8(scales, block->scales);
    for (uint32_t k = 0; k < TILE_K; k++)
        set_canonical_q4_32_value(block, k, values[k]);
}

template <typename Block>
static void encode_q2q3_block(Block *block, const int8_t values[TILE_K],
                              const int8_t scales[K32_GROUPS],
                              uint8_t promoted_mask, uint16_t d_bits) {
    memset(block, 0, sizeof(*block));
    block->d_bits = d_bits;
    block->promoted_mask = promoted_mask;
    encode_scale6_8(scales, block->scales);
    uint8_t low[TILE_K];
    uint32_t ordinal = 0;
    for (uint32_t group = 0; group < K32_GROUPS; group++) {
        const bool promoted = (promoted_mask & (1u << group)) != 0;
        for (uint32_t i = 0; i < K32; i++) {
            const uint32_t k = group * K32 + i;
            const uint32_t code = (uint32_t)((int32_t)values[k] +
                                              (promoted ? 4 : 2));
            low[k] = (uint8_t)(code & 3u);
            if (promoted && (code & 4u))
                block->high[ordinal * 4u + i / 8u] |=
                    (uint8_t)(1u << (i & 7u));
        }
        if (promoted) ordinal++;
    }
    pack_bits(low, TILE_K, 2u, block->low2);
}

template <typename Block>
static void encode_q3a_block(Block *block, const uint8_t values[TILE_K],
                             const uint8_t scales[K32_GROUPS],
                             const uint8_t mins[K32_GROUPS],
                             uint32_t metadata_bits, uint16_t d_bits,
                             uint16_t dmin_bits) {
    memset(block, 0, sizeof(*block));
    block->d_bits = d_bits;
    block->dmin_bits = dmin_bits;
    encode_unsigned_fields8(scales, metadata_bits, block->scales);
    encode_unsigned_fields8(mins, metadata_bits, block->mins);
    pack_bits(values, TILE_K, 3u, block->qs);
}

static void encode_q3q4_50_block(SM75Q3Q4_50Block *block,
                                 const int8_t values[TILE_K],
                                 const int8_t scales[K32_GROUPS],
                                 uint8_t promoted_mask, uint16_t d_bits) {
    memset(block, 0, sizeof(*block));
    block->d_bits = d_bits;
    block->promoted_mask = promoted_mask;
    encode_scale6_8(scales, block->scales);
    uint8_t low[TILE_K];
    uint32_t ordinal = 0;
    for (uint32_t group = 0; group < K32_GROUPS; group++) {
        const bool promoted = (promoted_mask & (1u << group)) != 0;
        for (uint32_t i = 0; i < K32; i++) {
            const uint32_t k = group * K32 + i;
            const uint32_t code = promoted
                ? ((uint32_t)(uint8_t)values[k] & 15u)
                : (uint32_t)((int32_t)values[k] + 4);
            low[k] = (uint8_t)(code & 7u);
            if (promoted && (code & 8u))
                block->high[ordinal * 4u + i / 8u] |=
                    (uint8_t)(1u << (i & 7u));
        }
        if (promoted) ordinal++;
    }
    pack_bits(low, TILE_K, 3u, block->low3);
}

static void encode_q5_32_block(SM75Q5_32Block *block,
                               const int8_t values[TILE_K],
                               const int8_t scales[K32_GROUPS],
                               uint16_t d_bits) {
    memset(block, 0, sizeof(*block));
    block->d_bits = d_bits;
    encode_scale6_8(scales, block->scales);
    uint8_t codes[TILE_K];
    for (uint32_t k = 0; k < TILE_K; k++)
        codes[k] = (uint8_t)((int32_t)values[k] + 16);
    pack_bits(codes, TILE_K, 5u, block->qs);
}

static uint32_t host_pack_nibbles8(const uint8_t values[8]) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < 8; i++)
        result |= (uint32_t)(values[i] & 15u) << (4u * i);
    return result;
}

static void repack_q4k_native(const Q4KBlock canonical[TILE_N],
                              NativeQ4KTile *native) {
    memset(native, 0, sizeof(*native));
    for (uint32_t col = 0; col < TILE_N; col++) {
        native->hdr[col].d_bits = canonical[col].d_bits;
        native->hdr[col].dmin_bits = canonical[col].dmin_bits;
        memcpy(native->hdr[col].scales_mins, canonical[col].scales_mins, 12);
    }
    for (uint32_t j = 0; j < K32_GROUPS; j++) {
        for (uint32_t lane = 0; lane < WARP_SIZE_; lane++) {
            const uint32_t col = lane >> 2u;
            const uint32_t k0 = j * K32 + (lane & 3u) * 8u;
            uint8_t values[8];
            for (uint32_t i = 0; i < 8; i++)
                values[i] = canonical_q4k_value(&canonical[col], k0 + i);
            native->b[j][lane] = host_pack_nibbles8(values);
        }
    }
}

static void repack_q3k_native(const Q3KBlock canonical[TILE_N],
                              NativeQ3KTile *native) {
    memset(native, 0, sizeof(*native));
    for (uint32_t col = 0; col < TILE_N; col++) {
        native->d_bits[col] = canonical[col].d_bits;
        memcpy(native->scales[col], canonical[col].scales, 12);
    }
    for (uint32_t j = 0; j < K16_GROUPS; j++) {
        for (uint32_t lane = 0; lane < WARP_SIZE_; lane++) {
            const uint32_t col = lane >> 2u;
            const uint32_t k0 = j * K16 + (lane & 3u) * 4u;
            uint8_t low = 0, high = 0;
            for (uint32_t i = 0; i < 4; i++) {
                const uint32_t u =
                    (uint32_t)((int32_t)canonical_q3k_value(
                        &canonical[col], k0 + i) + 4);
                low |= (uint8_t)((u & 3u) << (2u * i));
                high |= (uint8_t)(((u >> 2u) & 1u) << i);
            }
            native->low2[j][lane] = low;
            native->high[j][lane >> 1u] |=
                (uint8_t)(high << (4u * (lane & 1u)));
        }
    }
}

static void repack_q3_32_native(const SM75Q3_32Block canonical[TILE_N],
                                NativeQ3_32Tile *native) {
    memset(native, 0, sizeof(*native));
    for (uint32_t col = 0; col < TILE_N; col++) {
        native->d_bits[col] = canonical[col].d_bits;
        memcpy(native->scales[col], canonical[col].scales, 6);
    }
    for (uint32_t j = 0; j < K32_GROUPS; j++) {
        for (uint32_t lane = 0; lane < WARP_SIZE_; lane++) {
            const uint32_t col = lane >> 2u;
            const uint32_t k0 = j * K32 + (lane & 3u) * 8u;
            uint16_t low = 0;
            uint8_t high = 0;
            for (uint32_t i = 0; i < 8; i++) {
                const uint32_t u = canonical_q3_32_code(
                    &canonical[col], k0 + i);
                low |= (uint16_t)((u & 3u) << (2u * i));
                high |= (uint8_t)(((u >> 2u) & 1u) << i);
            }
            native->low2[j][lane] = low;
            native->high[j][lane] = high;
        }
    }
}

static void repack_q4_32_native(const SM75Q4_32Block canonical[TILE_N],
                                NativeQ4_32Tile *native) {
    memset(native, 0, sizeof(*native));
    for (uint32_t col = 0; col < TILE_N; col++) {
        native->d_bits[col] = canonical[col].d_bits;
        memcpy(native->scales[col], canonical[col].scales, 6);
    }
    for (uint32_t j = 0; j < K32_GROUPS; j++) {
        for (uint32_t lane = 0; lane < WARP_SIZE_; lane++) {
            const uint32_t col = lane >> 2u;
            const uint32_t k0 = j * K32 + (lane & 3u) * 8u;
            uint8_t values[8];
            for (uint32_t i = 0; i < 8; i++)
                values[i] = (uint8_t)canonical_q4_32_value(
                    &canonical[col], k0 + i) & 15u;
            native->b[j][lane] = host_pack_nibbles8(values);
        }
    }
}

template <typename Canonical, typename Native>
static void repack_q2q3_native(const Canonical canonical[TILE_N],
                               Native *native) {
    memset(native, 0, sizeof(*native));
    for (uint32_t col = 0; col < TILE_N; col++) {
        native->d_bits[col] = canonical[col].d_bits;
        memcpy(native->scales[col], canonical[col].scales, 6);
        native->promoted_mask[col] = canonical[col].promoted_mask;
        memcpy(native->high[col], canonical[col].high,
               sizeof(native->high[col]));
    }
    for (uint32_t group = 0; group < K32_GROUPS; group++) {
        for (uint32_t lane = 0; lane < WARP_SIZE_; lane++) {
            const uint32_t col = lane >> 2u;
            const uint32_t k0 = group * K32 + (lane & 3u) * 8u;
            uint16_t low = 0;
            for (uint32_t i = 0; i < 8; i++)
                low |= (uint16_t)(unpack_bits(canonical[col].low2,
                                              k0 + i, 2u) << (2u * i));
            native->low2[group][lane] = low;
        }
    }
}

template <typename Canonical, typename Native>
static void repack_q3a_native(const Canonical canonical[TILE_N],
                              Native *native) {
    memset(native, 0, sizeof(*native));
    for (uint32_t col = 0; col < TILE_N; col++) {
        native->d_bits[col] = canonical[col].d_bits;
        native->dmin_bits[col] = canonical[col].dmin_bits;
        memcpy(native->scales[col], canonical[col].scales,
               sizeof(native->scales[col]));
        memcpy(native->mins[col], canonical[col].mins,
               sizeof(native->mins[col]));
    }
    for (uint32_t group = 0; group < K32_GROUPS; group++) {
        for (uint32_t lane = 0; lane < WARP_SIZE_; lane++) {
            const uint32_t col = lane >> 2u;
            const uint32_t k0 = group * K32 + (lane & 3u) * 8u;
            uint16_t low = 0;
            uint8_t high = 0;
            for (uint32_t i = 0; i < 8; i++) {
                const uint32_t code = unpack_bits(canonical[col].qs,
                                                  k0 + i, 3u);
                low |= (uint16_t)((code & 3u) << (2u * i));
                high |= (uint8_t)(((code >> 2u) & 1u) << i);
            }
            native->low2[group][lane] = low;
            native->high[group][lane] = high;
        }
    }
}

static void repack_q3q4_50_native(
        const SM75Q3Q4_50Block canonical[TILE_N],
        NativeQ3Q4_50Tile *native) {
    memset(native, 0, sizeof(*native));
    for (uint32_t col = 0; col < TILE_N; col++) {
        native->d_bits[col] = canonical[col].d_bits;
        memcpy(native->scales[col], canonical[col].scales, 6);
        native->promoted_mask[col] = canonical[col].promoted_mask;
    }
    for (uint32_t group = 0; group < K32_GROUPS; group++) {
        for (uint32_t lane = 0; lane < WARP_SIZE_; lane++) {
            const uint32_t col = lane >> 2u;
            const bool promoted =
                (canonical[col].promoted_mask & (1u << group)) != 0;
            const uint32_t k0 = group * K32 + (lane & 3u) * 8u;
            uint16_t low = 0;
            uint8_t high3 = 0, high4 = 0;
            for (uint32_t i = 0; i < 8; i++) {
                const int32_t q = canonical_q3q4_value(
                    &canonical[col], k0 + i);
                const uint32_t biased = (uint32_t)(q + (promoted ? 8 : 4));
                low |= (uint16_t)((biased & 3u) << (2u * i));
                high3 |= (uint8_t)(((biased >> 2u) & 1u) << i);
                high4 |= (uint8_t)(((biased >> 3u) & 1u) << i);
            }
            native->low2[group][lane] = low;
            native->high3[group][lane] = high3;
            if (promoted) {
                const uint32_t ordinal = popcount8((uint8_t)(
                    canonical[col].promoted_mask & ((1u << group) - 1u)));
                native->high4_biased[col][ordinal * 4u + (lane & 3u)] =
                    high4;
            }
        }
    }
}

static void repack_q5_32_native(const SM75Q5_32Block canonical[TILE_N],
                                NativeQ5_32Tile *native) {
    memset(native, 0, sizeof(*native));
    for (uint32_t col = 0; col < TILE_N; col++) {
        native->d_bits[col] = canonical[col].d_bits;
        memcpy(native->scales[col], canonical[col].scales, 6);
    }
    for (uint32_t group = 0; group < K32_GROUPS; group++) {
        for (uint32_t lane = 0; lane < WARP_SIZE_; lane++) {
            const uint32_t col = lane >> 2u;
            const uint32_t k0 = group * K32 + (lane & 3u) * 8u;
            uint8_t low[8];
            uint8_t high = 0;
            for (uint32_t i = 0; i < 8; i++) {
                const uint32_t code = (uint32_t)(
                    (int32_t)canonical_q5_value(&canonical[col], k0 + i) + 16);
                low[i] = (uint8_t)(code & 15u);
                high |= (uint8_t)(((code >> 4u) & 1u) << i);
            }
            native->low4[group][lane] = host_pack_nibbles8(low);
            native->high[group][lane] = high;
        }
    }
}

static uint8_t native_q4k_value(const NativeQ4KTile *native,
                                uint32_t col, uint32_t k) {
    const uint32_t lane = col * 4u + (k % K32) / 8u;
    return (native->b[k / K32][lane] >> (4u * (k & 7u))) & 15u;
}

static int8_t native_q3k_value(const NativeQ3KTile *native,
                               uint32_t col, uint32_t k) {
    const uint32_t lane = col * 4u + (k % K16) / 4u;
    const uint32_t i = k & 3u;
    const uint32_t low = (native->low2[k / K16][lane] >> (2u * i)) & 3u;
    const uint32_t high = (native->high[k / K16][lane >> 1u] >>
                           (4u * (lane & 1u) + i)) & 1u;
    return (int8_t)((int32_t)(low | (high << 2u)) - 4);
}

static int8_t native_q3_32_value(const NativeQ3_32Tile *native,
                                 uint32_t col, uint32_t k) {
    const uint32_t lane = col * 4u + (k % K32) / 8u;
    const uint32_t i = k & 7u;
    const uint32_t low = (native->low2[k / K32][lane] >> (2u * i)) & 3u;
    const uint32_t high = (native->high[k / K32][lane] >> i) & 1u;
    return (int8_t)((int32_t)(low | (high << 2u)) - 4);
}

static int8_t native_q4_32_value(const NativeQ4_32Tile *native,
                                 uint32_t col, uint32_t k) {
    const uint32_t lane = col * 4u + (k % K32) / 8u;
    const uint32_t code =
        (native->b[k / K32][lane] >> (4u * (k & 7u))) & 15u;
    return (int8_t)(code < 8u ? (int32_t)code : (int32_t)code - 16);
}

template <typename Native>
static int8_t native_q2q3_value(const Native *native, uint32_t col,
                                uint32_t k) {
    const uint32_t group = k / K32;
    const uint32_t lane = col * 4u + (k % K32) / 8u;
    const uint32_t i = k & 7u;
    uint32_t code = (native->low2[group][lane] >> (2u * i)) & 3u;
    const bool promoted =
        (native->promoted_mask[col] & (1u << group)) != 0;
    if (promoted) {
        const uint32_t ordinal = popcount8((uint8_t)(
            native->promoted_mask[col] & ((1u << group) - 1u)));
        code |= ((native->high[col][ordinal * 4u + (lane & 3u)] >> i) &
                 1u) << 2u;
    }
    return (int8_t)((int32_t)code - (promoted ? 4 : 2));
}

template <typename Native>
static uint8_t native_q3a_value(const Native *native, uint32_t col,
                                uint32_t k) {
    const uint32_t group = k / K32;
    const uint32_t lane = col * 4u + (k % K32) / 8u;
    const uint32_t i = k & 7u;
    const uint32_t low = (native->low2[group][lane] >> (2u * i)) & 3u;
    const uint32_t high = (native->high[group][lane] >> i) & 1u;
    return (uint8_t)(low | (high << 2u));
}

static int8_t native_q3q4_50_value(const NativeQ3Q4_50Tile *native,
                                   uint32_t col, uint32_t k) {
    const uint32_t group = k / K32;
    const uint32_t lane = col * 4u + (k % K32) / 8u;
    const uint32_t i = k & 7u;
    const bool promoted =
        (native->promoted_mask[col] & (1u << group)) != 0;
    uint32_t biased =
        ((native->low2[group][lane] >> (2u * i)) & 3u) |
        (((native->high3[group][lane] >> i) & 1u) << 2u);
    if (promoted) {
        const uint32_t ordinal = popcount8((uint8_t)(
            native->promoted_mask[col] & ((1u << group) - 1u)));
        biased |= ((native->high4_biased[col][ordinal * 4u +
                    (lane & 3u)] >> i) & 1u) << 3u;
    }
    return (int8_t)((int32_t)biased - (promoted ? 8 : 4));
}

static int8_t native_q5_32_value(const NativeQ5_32Tile *native,
                                 uint32_t col, uint32_t k) {
    const uint32_t group = k / K32;
    const uint32_t lane = col * 4u + (k % K32) / 8u;
    const uint32_t i = k & 7u;
    const uint32_t code =
        ((native->low4[group][lane] >> (4u * i)) & 15u) |
        (((native->high[group][lane] >> i) & 1u) << 4u);
    return (int8_t)((int32_t)code - 16);
}

static void unpack_q4k_native(const NativeQ4KTile *native,
                              Q4KBlock canonical[TILE_N]) {
    memset(canonical, 0, sizeof(Q4KBlock) * TILE_N);
    for (uint32_t col = 0; col < TILE_N; col++) {
        canonical[col].d_bits = native->hdr[col].d_bits;
        canonical[col].dmin_bits = native->hdr[col].dmin_bits;
        memcpy(canonical[col].scales_mins, native->hdr[col].scales_mins, 12);
        for (uint32_t k = 0; k < TILE_K; k++)
            set_canonical_q4k_value(&canonical[col], k,
                                    native_q4k_value(native, col, k));
    }
}

static void unpack_q3k_native(const NativeQ3KTile *native,
                              Q3KBlock canonical[TILE_N]) {
    memset(canonical, 0, sizeof(Q3KBlock) * TILE_N);
    for (uint32_t col = 0; col < TILE_N; col++) {
        canonical[col].d_bits = native->d_bits[col];
        memcpy(canonical[col].scales, native->scales[col], 12);
        for (uint32_t k = 0; k < TILE_K; k++)
            set_canonical_q3k_value(&canonical[col], k,
                                    native_q3k_value(native, col, k));
    }
}

static void unpack_q3_32_native(const NativeQ3_32Tile *native,
                                SM75Q3_32Block canonical[TILE_N]) {
    memset(canonical, 0, sizeof(SM75Q3_32Block) * TILE_N);
    for (uint32_t col = 0; col < TILE_N; col++) {
        canonical[col].d_bits = native->d_bits[col];
        memcpy(canonical[col].scales, native->scales[col], 6);
        for (uint32_t k = 0; k < TILE_K; k++)
            set_canonical_q3_32_code(&canonical[col], k,
                (uint8_t)((int32_t)native_q3_32_value(native, col, k) + 4));
    }
}

static void unpack_q4_32_native(const NativeQ4_32Tile *native,
                                SM75Q4_32Block canonical[TILE_N]) {
    memset(canonical, 0, sizeof(SM75Q4_32Block) * TILE_N);
    for (uint32_t col = 0; col < TILE_N; col++) {
        canonical[col].d_bits = native->d_bits[col];
        memcpy(canonical[col].scales, native->scales[col], 6);
        for (uint32_t k = 0; k < TILE_K; k++)
            set_canonical_q4_32_value(&canonical[col], k,
                                      native_q4_32_value(native, col, k));
    }
}

template <typename Canonical, typename Native>
static void unpack_q2q3_native(const Native *native,
                               Canonical canonical[TILE_N]) {
    memset(canonical, 0, sizeof(Canonical) * TILE_N);
    for (uint32_t col = 0; col < TILE_N; col++) {
        canonical[col].d_bits = native->d_bits[col];
        memcpy(canonical[col].scales, native->scales[col], 6);
        canonical[col].promoted_mask = native->promoted_mask[col];
        memcpy(canonical[col].high, native->high[col],
               sizeof(canonical[col].high));
        uint8_t low[TILE_K];
        for (uint32_t k = 0; k < TILE_K; k++) {
            const uint32_t group = k / K32;
            const uint32_t lane = col * 4u + (k % K32) / 8u;
            low[k] = (uint8_t)((native->low2[group][lane] >>
                                (2u * (k & 7u))) & 3u);
        }
        pack_bits(low, TILE_K, 2u, canonical[col].low2);
    }
}

template <typename Canonical, typename Native>
static void unpack_q3a_native(const Native *native,
                              Canonical canonical[TILE_N]) {
    memset(canonical, 0, sizeof(Canonical) * TILE_N);
    for (uint32_t col = 0; col < TILE_N; col++) {
        canonical[col].d_bits = native->d_bits[col];
        canonical[col].dmin_bits = native->dmin_bits[col];
        memcpy(canonical[col].scales, native->scales[col],
               sizeof(canonical[col].scales));
        memcpy(canonical[col].mins, native->mins[col],
               sizeof(canonical[col].mins));
        uint8_t codes[TILE_K];
        for (uint32_t k = 0; k < TILE_K; k++)
            codes[k] = native_q3a_value(native, col, k);
        pack_bits(codes, TILE_K, 3u, canonical[col].qs);
    }
}

static void unpack_q3q4_50_native(const NativeQ3Q4_50Tile *native,
                                  SM75Q3Q4_50Block canonical[TILE_N]) {
    memset(canonical, 0, sizeof(SM75Q3Q4_50Block) * TILE_N);
    for (uint32_t col = 0; col < TILE_N; col++) {
        canonical[col].d_bits = native->d_bits[col];
        memcpy(canonical[col].scales, native->scales[col], 6);
        canonical[col].promoted_mask = native->promoted_mask[col];
        uint8_t low[TILE_K];
        uint32_t ordinal = 0;
        for (uint32_t group = 0; group < K32_GROUPS; group++) {
            const bool promoted =
                (canonical[col].promoted_mask & (1u << group)) != 0;
            for (uint32_t i = 0; i < K32; i++) {
                const uint32_t k = group * K32 + i;
                const int32_t q = native_q3q4_50_value(native, col, k);
                const uint32_t code = promoted
                    ? ((uint32_t)q & 15u) : (uint32_t)(q + 4);
                low[k] = (uint8_t)(code & 7u);
                if (promoted && (code & 8u))
                    canonical[col].high[ordinal * 4u + i / 8u] |=
                        (uint8_t)(1u << (i & 7u));
            }
            if (promoted) ordinal++;
        }
        pack_bits(low, TILE_K, 3u, canonical[col].low3);
    }
}

static void unpack_q5_32_native(const NativeQ5_32Tile *native,
                                SM75Q5_32Block canonical[TILE_N]) {
    memset(canonical, 0, sizeof(SM75Q5_32Block) * TILE_N);
    for (uint32_t col = 0; col < TILE_N; col++) {
        canonical[col].d_bits = native->d_bits[col];
        memcpy(canonical[col].scales, native->scales[col], 6);
        uint8_t codes[TILE_K];
        for (uint32_t k = 0; k < TILE_K; k++)
            codes[k] = (uint8_t)((int32_t)native_q5_32_value(
                native, col, k) + 16);
        pack_bits(codes, TILE_K, 5u, canonical[col].qs);
    }
}

static int layout_gate(const CanonicalWeights *canonical,
                       const NativeQ4KTile *q4k,
                       const NativeQ3KTile *q3k,
                       const NativeQ3_32Tile *q3_32,
                       const NativeQ4_32Tile *q4_32,
                       const NativeQ2Q3_50Tile *q2q3_50,
                       const NativeQ2Q3_75Tile *q2q3_75,
                       const NativeQ3A4Tile *q3a4,
                       const NativeQ3A6Tile *q3a6,
                       const NativeQ3Q4_50Tile *q3q4_50,
                       const NativeQ5_32Tile *q5_32,
                       uint32_t case_id) {
    for (uint32_t col = 0; col < TILE_N; col++) {
        for (uint32_t k = 0; k < TILE_K; k++) {
            if (canonical_q4k_value(&canonical->q4k[col], k) !=
                    native_q4k_value(q4k, col, k) ||
                canonical_q3k_value(&canonical->q3k[col], k) !=
                    native_q3k_value(q3k, col, k) ||
                (int8_t)((int32_t)canonical_q3_32_code(
                    &canonical->q3_32[col], k) - 4) !=
                    native_q3_32_value(q3_32, col, k) ||
                canonical_q4_32_value(&canonical->q4_32[col], k) !=
                    native_q4_32_value(q4_32, col, k) ||
                canonical_q2q3_value(canonical->q2q3_50[col].low2,
                    canonical->q2q3_50[col].promoted_mask,
                    canonical->q2q3_50[col].high, k) !=
                    native_q2q3_value(q2q3_50, col, k) ||
                canonical_q2q3_value(canonical->q2q3_75[col].low2,
                    canonical->q2q3_75[col].promoted_mask,
                    canonical->q2q3_75[col].high, k) !=
                    native_q2q3_value(q2q3_75, col, k) ||
                canonical_affine_value(canonical->q3a4[col].qs, k) !=
                    native_q3a_value(q3a4, col, k) ||
                canonical_affine_value(canonical->q3a6[col].qs, k) !=
                    native_q3a_value(q3a6, col, k) ||
                canonical_q3q4_value(&canonical->q3q4_50[col], k) !=
                    native_q3q4_50_value(q3q4_50, col, k) ||
                canonical_q5_value(&canonical->q5_32[col], k) !=
                    native_q5_32_value(q5_32, col, k)) {
                fprintf(stderr,
                    "layout decode failure: case=%u col=%u k=%u\n",
                    case_id, col, k);
                return 1;
            }
        }
        for (uint32_t j = 0; j < K16_GROUPS; j++) {
            if (decode_q3k_scale(canonical->q3k[col].scales, j) !=
                decode_q3k_scale(q3k->scales[col], j)) {
                fprintf(stderr, "Q3_K scale decode failure: case=%u\n", case_id);
                return 1;
            }
        }
        for (uint32_t j = 0; j < K32_GROUPS; j++) {
            uint32_t cs, cm, ns, nm;
            decode_q4k_scale_min(canonical->q4k[col].scales_mins,
                                 j, &cs, &cm);
            decode_q4k_scale_min(q4k->hdr[col].scales_mins, j, &ns, &nm);
            if (cs != ns || cm != nm ||
                decode_scale6_8(canonical->q3_32[col].scales, j) !=
                    decode_scale6_8(q3_32->scales[col], j) ||
                decode_scale6_8(canonical->q4_32[col].scales, j) !=
                    decode_scale6_8(q4_32->scales[col], j) ||
                decode_scale6_8(canonical->q2q3_50[col].scales, j) !=
                    decode_scale6_8(q2q3_50->scales[col], j) ||
                decode_scale6_8(canonical->q2q3_75[col].scales, j) !=
                    decode_scale6_8(q2q3_75->scales[col], j) ||
                decode_scale6_8(canonical->q3q4_50[col].scales, j) !=
                    decode_scale6_8(q3q4_50->scales[col], j) ||
                decode_scale6_8(canonical->q5_32[col].scales, j) !=
                    decode_scale6_8(q5_32->scales[col], j) ||
                decode_unsigned_field8(canonical->q3a4[col].scales, 4u, j) !=
                    decode_unsigned_field8(q3a4->scales[col], 4u, j) ||
                decode_unsigned_field8(canonical->q3a4[col].mins, 4u, j) !=
                    decode_unsigned_field8(q3a4->mins[col], 4u, j) ||
                decode_unsigned_field8(canonical->q3a6[col].scales, 6u, j) !=
                    decode_unsigned_field8(q3a6->scales[col], 6u, j) ||
                decode_unsigned_field8(canonical->q3a6[col].mins, 6u, j) !=
                    decode_unsigned_field8(q3a6->mins[col], 6u, j)) {
                fprintf(stderr, "K32 metadata decode failure: case=%u\n", case_id);
                return 1;
            }
        }
    }

    CanonicalWeights roundtrip;
    unpack_q4k_native(q4k, roundtrip.q4k);
    unpack_q3k_native(q3k, roundtrip.q3k);
    unpack_q3_32_native(q3_32, roundtrip.q3_32);
    unpack_q4_32_native(q4_32, roundtrip.q4_32);
    unpack_q2q3_native(q2q3_50, roundtrip.q2q3_50);
    unpack_q2q3_native(q2q3_75, roundtrip.q2q3_75);
    unpack_q3a_native(q3a4, roundtrip.q3a4);
    unpack_q3a_native(q3a6, roundtrip.q3a6);
    unpack_q3q4_50_native(q3q4_50, roundtrip.q3q4_50);
    unpack_q5_32_native(q5_32, roundtrip.q5_32);
    if (memcmp(roundtrip.q4k, canonical->q4k, sizeof(roundtrip.q4k)) != 0 ||
        memcmp(roundtrip.q3k, canonical->q3k, sizeof(roundtrip.q3k)) != 0 ||
        memcmp(roundtrip.q3_32, canonical->q3_32,
               sizeof(roundtrip.q3_32)) != 0 ||
        memcmp(roundtrip.q4_32, canonical->q4_32,
               sizeof(roundtrip.q4_32)) != 0 ||
        memcmp(roundtrip.q2q3_50, canonical->q2q3_50,
               sizeof(roundtrip.q2q3_50)) != 0 ||
        memcmp(roundtrip.q2q3_75, canonical->q2q3_75,
               sizeof(roundtrip.q2q3_75)) != 0 ||
        memcmp(roundtrip.q3a4, canonical->q3a4,
               sizeof(roundtrip.q3a4)) != 0 ||
        memcmp(roundtrip.q3a6, canonical->q3a6,
               sizeof(roundtrip.q3a6)) != 0 ||
        memcmp(roundtrip.q3q4_50, canonical->q3q4_50,
               sizeof(roundtrip.q3q4_50)) != 0 ||
        memcmp(roundtrip.q5_32, canonical->q5_32,
               sizeof(roundtrip.q5_32)) != 0) {
        fprintf(stderr, "canonical/native roundtrip failure: case=%u\n", case_id);
        return 1;
    }
    return 0;
}

static void fill_activation(ActivationCase *activation, uint32_t case_id) {
    static const int8_t edges[16] = {
        -128, -127, -65, -17, -16, -15, -9, -1,
        0, 1, 7, 8, 15, 16, 63, 127,
    };
    uint32_t state = 0x243f6a88u ^ case_id * 0x9e3779b9u;
    for (uint32_t row = 0; row < TILE_M; row++) {
        for (uint32_t k = 0; k < TILE_K; k++) {
            int32_t value;
            if (case_id == 0) value = 0;
            else if (case_id == 1) value = 127;
            else if (case_id == 2) value = -128;
            else if (case_id == 3) value = ((row + k) & 1u) ? 127 : -128;
            else if (case_id == 4) value = edges[(row * 5u + k) & 15u];
            else if (case_id == 5)
                value = (int32_t)((row * 29u + k * 17u +
                                   (k >> 4u) * 11u) & 255u) - 128;
            else value = (int32_t)(rng_next(&state) & 255u) - 128;
            activation->q8[row][k] = (int8_t)value;
        }
        for (uint32_t j = 0; j < K16_GROUPS; j++) {
            int32_t sum = 0;
            for (uint32_t i = 0; i < K16; i++)
                sum += activation->q8[row][j * K16 + i];
            activation->bsums[row][j] = (int16_t)sum;
        }
    }
}

static void fill_weights(CanonicalWeights *weights, uint32_t case_id) {
    static const int8_t q3_edges[8] = {-4, -3, -2, -1, 0, 1, 2, 3};
    static const int8_t q4_edges[16] = {
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
    };
    uint32_t state = 0x13198a2eu ^ case_id * 0x85ebca6bu;
    for (uint32_t col = 0; col < TILE_N; col++) {
        uint8_t q4k_values[TILE_K], q4k_scales[K32_GROUPS];
        uint8_t q4k_mins[K32_GROUPS];
        int8_t q3k_values[TILE_K], q3k_scales[K16_GROUPS];
        int8_t q3_values[TILE_K], q3_scales[K32_GROUPS];
        int8_t q4_values[TILE_K], q4_scales[K32_GROUPS];
        int8_t q2q3_50_values[TILE_K], q2q3_75_values[TILE_K];
        int8_t q3q4_50_values[TILE_K], q5_values[TILE_K];
        uint8_t q3a4_values[TILE_K], q3a6_values[TILE_K];
        uint8_t q3a4_scales[K32_GROUPS], q3a4_mins[K32_GROUPS];
        uint8_t q3a6_scales[K32_GROUPS], q3a6_mins[K32_GROUPS];
        const uint8_t q2q3_50_mask = rotate_mask8(
            0x0fu, col + case_id * 3u);
        const uint8_t q2q3_75_mask = rotate_mask8(
            0x3fu, col * 3u + case_id);
        const uint8_t q3q4_50_mask = rotate_mask8(
            0x33u, col * 5u + case_id);
        for (uint32_t k = 0; k < TILE_K; k++) {
            const uint32_t group = k / K32;
            const bool q2q3_50_promoted =
                (q2q3_50_mask & (1u << group)) != 0;
            const bool q2q3_75_promoted =
                (q2q3_75_mask & (1u << group)) != 0;
            const bool q3q4_50_promoted =
                (q3q4_50_mask & (1u << group)) != 0;
            if (case_id == 0) {
                q4k_values[k] = 0;
                q3k_values[k] = -4;
                q3_values[k] = -4;
                q4_values[k] = -8;
                q2q3_50_values[k] = (int8_t)(q2q3_50_promoted ? -4 : -2);
                q2q3_75_values[k] = (int8_t)(q2q3_75_promoted ? -4 : -2);
                q3a4_values[k] = q3a6_values[k] = 0;
                q3q4_50_values[k] =
                    (int8_t)(q3q4_50_promoted ? -8 : -4);
                q5_values[k] = -16;
            } else if (case_id == 1) {
                q4k_values[k] = 15;
                q3k_values[k] = 3;
                q3_values[k] = 3;
                q4_values[k] = 7;
                q2q3_50_values[k] = (int8_t)(q2q3_50_promoted ? 3 : 1);
                q2q3_75_values[k] = (int8_t)(q2q3_75_promoted ? 3 : 1);
                q3a4_values[k] = q3a6_values[k] = 7;
                q3q4_50_values[k] =
                    (int8_t)(q3q4_50_promoted ? 7 : 3);
                q5_values[k] = 15;
            } else if (case_id == 2) {
                q4k_values[k] = (uint8_t)(((col + k) & 1u) ? 15u : 0u);
                q3k_values[k] = q3_edges[(col * 3u + k) & 7u];
                q3_values[k] = q3_edges[(col * 5u + k) & 7u];
                q4_values[k] = q4_edges[(col * 7u + k) & 15u];
                q2q3_50_values[k] = q2q3_50_promoted
                    ? q3_edges[(col + k) & 7u]
                    : (int8_t)(((col + k) & 3u) - 2);
                q2q3_75_values[k] = q2q3_75_promoted
                    ? q3_edges[(col * 3u + k) & 7u]
                    : (int8_t)(((col * 3u + k) & 3u) - 2);
                q3a4_values[k] = (uint8_t)((col * 5u + k) & 7u);
                q3a6_values[k] = (uint8_t)((col * 7u + k * 3u) & 7u);
                q3q4_50_values[k] = q3q4_50_promoted
                    ? q4_edges[(col * 11u + k) & 15u]
                    : q3_edges[(col * 13u + k) & 7u];
                q5_values[k] = (int8_t)(((col * 17u + k) & 31u) - 16);
            } else if (case_id == 3) {
                q4k_values[k] = (uint8_t)((col * 13u + k * 7u +
                                           (k >> 5u) * 5u) & 15u);
                q3k_values[k] = (int8_t)(((col * 5u + k * 3u) & 7u) - 4);
                q3_values[k] = (int8_t)(((col * 7u + k * 5u) & 7u) - 4);
                q4_values[k] = (int8_t)(((col * 11u + k * 9u) & 15u) - 8);
                q2q3_50_values[k] = q2q3_50_promoted
                    ? (int8_t)(((col * 7u + k * 5u) & 7u) - 4)
                    : (int8_t)(((col * 7u + k * 5u) & 3u) - 2);
                q2q3_75_values[k] = q2q3_75_promoted
                    ? (int8_t)(((col * 5u + k * 11u) & 7u) - 4)
                    : (int8_t)(((col * 5u + k * 11u) & 3u) - 2);
                q3a4_values[k] = (uint8_t)((col * 13u + k * 3u) & 7u);
                q3a6_values[k] = (uint8_t)((col * 3u + k * 13u) & 7u);
                q3q4_50_values[k] = q3q4_50_promoted
                    ? (int8_t)(((col * 19u + k * 7u) & 15u) - 8)
                    : (int8_t)(((col * 19u + k * 7u) & 7u) - 4);
                q5_values[k] = (int8_t)(((col * 23u + k * 17u) & 31u) - 16);
            } else {
                q4k_values[k] = (uint8_t)(rng_next(&state) & 15u);
                q3k_values[k] = (int8_t)((rng_next(&state) & 7u) - 4);
                q3_values[k] = (int8_t)((rng_next(&state) & 7u) - 4);
                q4_values[k] = (int8_t)((rng_next(&state) & 15u) - 8);
                q2q3_50_values[k] = q2q3_50_promoted
                    ? (int8_t)((rng_next(&state) & 7u) - 4)
                    : (int8_t)((rng_next(&state) & 3u) - 2);
                q2q3_75_values[k] = q2q3_75_promoted
                    ? (int8_t)((rng_next(&state) & 7u) - 4)
                    : (int8_t)((rng_next(&state) & 3u) - 2);
                q3a4_values[k] = (uint8_t)(rng_next(&state) & 7u);
                q3a6_values[k] = (uint8_t)(rng_next(&state) & 7u);
                q3q4_50_values[k] = q3q4_50_promoted
                    ? (int8_t)((rng_next(&state) & 15u) - 8)
                    : (int8_t)((rng_next(&state) & 7u) - 4);
                q5_values[k] = (int8_t)((rng_next(&state) & 31u) - 16);
            }
        }
        for (uint32_t j = 0; j < K16_GROUPS; j++) {
            if (case_id == 0) q3k_scales[j] = 0;
            else if (case_id == 1) q3k_scales[j] = (j & 1u) ? 31 : -32;
            else if (case_id < 4)
                q3k_scales[j] = (int8_t)(((col * 13u + j * 17u) & 63u) - 32);
            else q3k_scales[j] = (int8_t)((rng_next(&state) & 63u) - 32);
        }
        for (uint32_t j = 0; j < K32_GROUPS; j++) {
            if (case_id == 0) {
                q4k_scales[j] = q4k_mins[j] = 0;
                q3_scales[j] = q4_scales[j] = 0;
                q3a4_scales[j] = q3a4_mins[j] = 0;
                q3a6_scales[j] = q3a6_mins[j] = 0;
            } else if (case_id == 1) {
                q4k_scales[j] = q4k_mins[j] = 63;
                q3_scales[j] = (j & 1u) ? 31 : -32;
                q4_scales[j] = (j & 1u) ? -32 : 31;
                q3a4_scales[j] = q3a4_mins[j] = 15;
                q3a6_scales[j] = q3a6_mins[j] = 63;
            } else if (case_id < 4) {
                q4k_scales[j] = (uint8_t)((col * 11u + j * 7u) & 63u);
                q4k_mins[j] = (uint8_t)((col * 3u + j * 19u) & 63u);
                q3_scales[j] = (int8_t)(((col * 7u + j * 13u) & 63u) - 32);
                q4_scales[j] = (int8_t)(((col * 17u + j * 5u) & 63u) - 32);
                q3a4_scales[j] = (uint8_t)((col * 5u + j * 11u) & 15u);
                q3a4_mins[j] = (uint8_t)((col * 13u + j * 3u) & 15u);
                q3a6_scales[j] = (uint8_t)((col * 7u + j * 17u) & 63u);
                q3a6_mins[j] = (uint8_t)((col * 19u + j * 5u) & 63u);
            } else {
                q4k_scales[j] = (uint8_t)(rng_next(&state) & 63u);
                q4k_mins[j] = (uint8_t)(rng_next(&state) & 63u);
                q3_scales[j] = (int8_t)((rng_next(&state) & 63u) - 32);
                q4_scales[j] = (int8_t)((rng_next(&state) & 63u) - 32);
                q3a4_scales[j] = (uint8_t)(rng_next(&state) & 15u);
                q3a4_mins[j] = (uint8_t)(rng_next(&state) & 15u);
                q3a6_scales[j] = (uint8_t)(rng_next(&state) & 63u);
                q3a6_mins[j] = (uint8_t)(rng_next(&state) & 63u);
            }
        }
        const uint16_t header = (uint16_t)(0x3c00u ^
            (uint16_t)(case_id * 37u + col * 257u));
        encode_q4k_block(&weights->q4k[col], q4k_values, q4k_scales,
                         q4k_mins, header, (uint16_t)(header ^ 0x55aau));
        encode_q3k_block(&weights->q3k[col], q3k_values, q3k_scales,
                         (uint16_t)(header ^ 0x1111u));
        encode_q3_32_block(&weights->q3_32[col], q3_values, q3_scales,
                           (uint16_t)(header ^ 0x2222u));
        encode_q4_32_block(&weights->q4_32[col], q4_values, q4_scales,
                           (uint16_t)(header ^ 0x3333u));
        encode_q2q3_block(&weights->q2q3_50[col], q2q3_50_values,
                          q3_scales, q2q3_50_mask,
                          (uint16_t)(header ^ 0x4444u));
        encode_q2q3_block(&weights->q2q3_75[col], q2q3_75_values,
                          q3_scales, q2q3_75_mask,
                          (uint16_t)(header ^ 0x5555u));
        encode_q3a_block(&weights->q3a4[col], q3a4_values,
                         q3a4_scales, q3a4_mins, 4u,
                         (uint16_t)(header ^ 0x6666u),
                         (uint16_t)(header ^ 0x7777u));
        encode_q3a_block(&weights->q3a6[col], q3a6_values,
                         q3a6_scales, q3a6_mins, 6u,
                         (uint16_t)(header ^ 0x8888u),
                         (uint16_t)(header ^ 0x9999u));
        encode_q3q4_50_block(&weights->q3q4_50[col], q3q4_50_values,
                             q3_scales, q3q4_50_mask,
                             (uint16_t)(header ^ 0xaaaau));
        encode_q5_32_block(&weights->q5_32[col], q5_values, q4_scales,
                           (uint16_t)(header ^ 0xbbbbu));
    }
}

static void repack_all(const CanonicalWeights *canonical,
                       NativeQ4KTile *q4k, NativeQ3KTile *q3k,
                       NativeQ3_32Tile *q3_32, NativeQ4_32Tile *q4_32,
                       NativeQ2Q3_50Tile *q2q3_50,
                       NativeQ2Q3_75Tile *q2q3_75,
                       NativeQ3A4Tile *q3a4, NativeQ3A6Tile *q3a6,
                       NativeQ3Q4_50Tile *q3q4_50,
                       NativeQ5_32Tile *q5_32) {
    repack_q4k_native(canonical->q4k, q4k);
    repack_q3k_native(canonical->q3k, q3k);
    repack_q3_32_native(canonical->q3_32, q3_32);
    repack_q4_32_native(canonical->q4_32, q4_32);
    repack_q2q3_native(canonical->q2q3_50, q2q3_50);
    repack_q2q3_native(canonical->q2q3_75, q2q3_75);
    repack_q3a_native(canonical->q3a4, q3a4);
    repack_q3a_native(canonical->q3a6, q3a6);
    repack_q3q4_50_native(canonical->q3q4_50, q3q4_50);
    repack_q5_32_native(canonical->q5_32, q5_32);
}

__device__ __forceinline__ static void mma_m8n8k16_s8_u8(
        int32_t &c0, int32_t &c1, uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32 "
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

__device__ __forceinline__ static void mma_m8n8k32_u4_s4(
        int32_t &c0, int32_t &c1, uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile(
        "mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32 "
        "{%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
#else
    (void)c0; (void)c1; (void)a; (void)b;
#endif
}

__device__ __forceinline__ static void mma_m8n8k32_s4_s4(
        int32_t &c0, int32_t &c1, uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile(
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
        "{%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
#else
    (void)c0; (void)c1; (void)a; (void)b;
#endif
}

__device__ __forceinline__ static uint32_t load_q8x4(
        const int8_t *ptr) {
    return *(const uint32_t *)ptr;
}

__device__ __forceinline__ static uint32_t pack_nibbles8_device(
        uint32_t x0, uint32_t x1, uint32_t shift) {
    uint32_t result = 0;
#pragma unroll
    for (uint32_t i = 0; i < 4; i++) {
        result |= ((x0 >> (8u * i + shift)) & 15u) << (4u * i);
        result |= ((x1 >> (8u * i + shift)) & 15u) << (4u * (i + 4u));
    }
    return result;
}

__device__ __forceinline__ static void activation_k32_fragments(
        const ActivationCase *activation, uint32_t a_tile, uint32_t j,
        uint32_t lane, uint32_t *low, uint32_t *high_signed) {
    const uint32_t row = a_tile * MMA_M + (lane >> 2u);
    const uint32_t k = j * K32 + (lane & 3u) * 8u;
    const uint32_t x0 = load_q8x4(&activation->q8[row][k]);
    const uint32_t x1 = load_q8x4(&activation->q8[row][k + 4u]);
    *low = pack_nibbles8_device(x0, x1, 0u);
    *high_signed = pack_nibbles8_device(x0, x1, 4u);
}

__host__ __device__ __forceinline__ static uint32_t dilate_q3k_u8(
        uint8_t low2, uint8_t high4) {
    uint32_t low = low2;
    low = (low | (low << 12u)) & 0x000f000fu;
    low = (low | (low << 6u)) & 0x03030303u;
    uint32_t high = high4 & 15u;
    high = (high | (high << 14u)) & 0x00030003u;
    high = (high | (high << 7u)) & 0x01010101u;
    return low | (high << 2u);
}

__host__ __device__ __forceinline__ static uint32_t dilate_q3_32_u4(
        uint16_t low2, uint8_t high8) {
    uint32_t low = low2;
    low = (low | (low << 8u)) & 0x00ff00ffu;
    low = (low | (low << 4u)) & 0x0f0f0f0fu;
    low = (low | (low << 2u)) & 0x33333333u;
    uint32_t high = high8;
    high = (high | (high << 12u)) & 0x000f000fu;
    high = (high | (high << 6u)) & 0x03030303u;
    high = (high | (high << 3u)) & 0x11111111u;
    return low | (high << 2u);
}

__host__ __device__ __forceinline__ static uint32_t dilate_high1_u4(
        uint8_t high8) {
    uint32_t high = high8;
    high = (high | (high << 12u)) & 0x000f000fu;
    high = (high | (high << 6u)) & 0x03030303u;
    high = (high | (high << 3u)) & 0x11111111u;
    return high;
}

template <typename Native>
__device__ __forceinline__ static uint8_t sparse_high_fragment(
        const Native *weights, uint32_t col, uint32_t group,
        uint32_t segment) {
    if (!(weights->promoted_mask[col] & (1u << group))) return 0u;
    const uint32_t ordinal = popcount8((uint8_t)(
        weights->promoted_mask[col] & ((1u << group) - 1u)));
    return weights->high[col][ordinal * 4u + segment];
}

__device__ __forceinline__ static uint8_t q3q4_high4_fragment(
        const NativeQ3Q4_50Tile *weights, uint32_t col,
        uint32_t group, uint32_t segment) {
    if (!(weights->promoted_mask[col] & (1u << group))) return 0u;
    const uint32_t ordinal = popcount8((uint8_t)(
        weights->promoted_mask[col] & ((1u << group) - 1u)));
    return weights->high4_biased[col][ordinal * 4u + segment];
}

__device__ __forceinline__ static TileI32 compute_q4k_native(
        const ActivationCase *activation, const NativeQ4KTile *weights,
        uint32_t lane) {
    const uint32_t row0 = lane >> 2u;
    const uint32_t col0 = (lane & 3u) * 2u;
    TileI32 out = {};
#pragma unroll
    for (uint32_t j = 0; j < K32_GROUPS; j++) {
        uint32_t scale0, min0, scale1, min1;
        decode_q4k_scale_min(weights->hdr[col0].scales_mins, j,
                             &scale0, &min0);
        decode_q4k_scale_min(weights->hdr[col0 + 1u].scales_mins, j,
                             &scale1, &min1);
        const uint32_t b = weights->b[j][lane];
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            uint32_t a_low, a_high;
            activation_k32_fragments(activation, a_tile, j, lane,
                                     &a_low, &a_high);
            int32_t lo0 = 0, lo1 = 0, hi0 = 0, hi1 = 0;
            mma_m8n8k32_u4_u4(lo0, lo1, a_low, b);
            mma_m8n8k32_s4_u4(hi0, hi1, a_high, b);
            const int32_t dot0 = lo0 + 16 * hi0;
            const int32_t dot1 = lo1 + 16 * hi1;
            const uint32_t row = a_tile * MMA_M + row0;
            const int32_t bsum = (int32_t)activation->bsums[row][2u * j] +
                                 (int32_t)activation->bsums[row][2u * j + 1u];
            out.v[2u * a_tile] += (int32_t)scale0 * dot0;
            out.v[2u * a_tile + 1u] += (int32_t)scale1 * dot1;
            out.correction[2u * a_tile] += (int32_t)min0 * bsum;
            out.correction[2u * a_tile + 1u] += (int32_t)min1 * bsum;
        }
    }
#if defined(__CUDA_ARCH__)
    /* Q4_K has distinct fp16 d/dmin factors in production.  Keep both
     * integer dependency chains live until the block boundary so the control
     * cannot be algebraically collapsed into a cheaper symmetric kernel. */
    asm volatile("" :
        "+r"(out.v[0]), "+r"(out.v[1]), "+r"(out.v[2]), "+r"(out.v[3]),
        "+r"(out.correction[0]), "+r"(out.correction[1]),
        "+r"(out.correction[2]), "+r"(out.correction[3]));
#endif
    return out;
}

__device__ __forceinline__ static TileI32 compute_q3k_k16_u8(
        const ActivationCase *activation, const NativeQ3KTile *weights,
        uint32_t lane) {
    const uint32_t row0 = lane >> 2u;
    const uint32_t col0 = (lane & 3u) * 2u;
    TileI32 out = {};
#pragma unroll
    for (uint32_t j = 0; j < K16_GROUPS; j++) {
        const uint8_t high_byte = weights->high[j][lane >> 1u];
        const uint8_t high4 = (high_byte >> (4u * (lane & 1u))) & 15u;
        const uint32_t b = dilate_q3k_u8(weights->low2[j][lane], high4);
        const int32_t scale0 = decode_q3k_scale(weights->scales[col0], j);
        const int32_t scale1 = decode_q3k_scale(weights->scales[col0 + 1u], j);
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            const uint32_t row = a_tile * MMA_M + row0;
            const uint32_t k = j * K16 + (lane & 3u) * 4u;
            const uint32_t a = load_q8x4(&activation->q8[row][k]);
            int32_t dot0 = 0, dot1 = 0;
            mma_m8n8k16_s8_u8(dot0, dot1, a, b);
            const int32_t correction =
                4 * (int32_t)activation->bsums[row][j];
            out.v[2u * a_tile] += scale0 * (dot0 - correction);
            out.v[2u * a_tile + 1u] += scale1 * (dot1 - correction);
        }
    }
    return out;
}

__device__ __forceinline__ static TileI32 compute_q3_32(
        const ActivationCase *activation, const NativeQ3_32Tile *weights,
        uint32_t lane) {
    const uint32_t row0 = lane >> 2u;
    const uint32_t col0 = (lane & 3u) * 2u;
    TileI32 out = {};
#pragma unroll
    for (uint32_t j = 0; j < K32_GROUPS; j++) {
        const uint32_t b = dilate_q3_32_u4(
            weights->low2[j][lane], weights->high[j][lane]);
        const int32_t scale0 = decode_scale6_8(weights->scales[col0], j);
        const int32_t scale1 = decode_scale6_8(weights->scales[col0 + 1u], j);
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            uint32_t a_low, a_high;
            activation_k32_fragments(activation, a_tile, j, lane,
                                     &a_low, &a_high);
            int32_t lo0 = 0, lo1 = 0, hi0 = 0, hi1 = 0;
            mma_m8n8k32_u4_u4(lo0, lo1, a_low, b);
            mma_m8n8k32_s4_u4(hi0, hi1, a_high, b);
            const uint32_t row = a_tile * MMA_M + row0;
            const int32_t bsum = (int32_t)activation->bsums[row][2u * j] +
                                 (int32_t)activation->bsums[row][2u * j + 1u];
            const int32_t dot0 = lo0 + 16 * hi0 - 4 * bsum;
            const int32_t dot1 = lo1 + 16 * hi1 - 4 * bsum;
            out.v[2u * a_tile] += scale0 * dot0;
            out.v[2u * a_tile + 1u] += scale1 * dot1;
        }
    }
    return out;
}

__device__ __forceinline__ static TileI32 compute_q4_32(
        const ActivationCase *activation, const NativeQ4_32Tile *weights,
        uint32_t lane) {
    const uint32_t col0 = (lane & 3u) * 2u;
    TileI32 out = {};
#pragma unroll
    for (uint32_t j = 0; j < K32_GROUPS; j++) {
        const uint32_t b = weights->b[j][lane];
        const int32_t scale0 = decode_scale6_8(weights->scales[col0], j);
        const int32_t scale1 = decode_scale6_8(weights->scales[col0 + 1u], j);
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            uint32_t a_low, a_high;
            activation_k32_fragments(activation, a_tile, j, lane,
                                     &a_low, &a_high);
            int32_t lo0 = 0, lo1 = 0, hi0 = 0, hi1 = 0;
            mma_m8n8k32_u4_s4(lo0, lo1, a_low, b);
            mma_m8n8k32_s4_s4(hi0, hi1, a_high, b);
            out.v[2u * a_tile] += scale0 * (lo0 + 16 * hi0);
            out.v[2u * a_tile + 1u] += scale1 * (lo1 + 16 * hi1);
        }
    }
    return out;
}


template <typename Native>
__device__ __forceinline__ static TileI32 compute_q2q3(
        const ActivationCase *activation, const Native *weights,
        uint32_t lane) {
    const uint32_t row0 = lane >> 2u;
    const uint32_t col0 = (lane & 3u) * 2u;
    TileI32 out = {};
#pragma unroll
    for (uint32_t j = 0; j < K32_GROUPS; j++) {
        const uint32_t b = dilate_q3_32_u4(
            weights->low2[j][lane], sparse_high_fragment(
                weights, lane >> 2u, j, lane & 3u));
        const int32_t scale0 = decode_scale6_8(weights->scales[col0], j);
        const int32_t scale1 = decode_scale6_8(weights->scales[col0 + 1u], j);
        const int32_t bias0 =
            (weights->promoted_mask[col0] & (1u << j)) ? 4 : 2;
        const int32_t bias1 =
            (weights->promoted_mask[col0 + 1u] & (1u << j)) ? 4 : 2;
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            uint32_t a_low, a_high;
            activation_k32_fragments(activation, a_tile, j, lane,
                                     &a_low, &a_high);
            int32_t lo0 = 0, lo1 = 0, hi0 = 0, hi1 = 0;
            mma_m8n8k32_u4_u4(lo0, lo1, a_low, b);
            mma_m8n8k32_s4_u4(hi0, hi1, a_high, b);
            const uint32_t row = a_tile * MMA_M + row0;
            const int32_t bsum = (int32_t)activation->bsums[row][2u * j] +
                                 (int32_t)activation->bsums[row][2u * j + 1u];
            out.v[2u * a_tile] +=
                scale0 * (lo0 + 16 * hi0 - bias0 * bsum);
            out.v[2u * a_tile + 1u] +=
                scale1 * (lo1 + 16 * hi1 - bias1 * bsum);
        }
    }
    return out;
}

template <typename Native, uint32_t METADATA_BITS>
__device__ __forceinline__ static TileI32 compute_q3a(
        const ActivationCase *activation, const Native *weights,
        uint32_t lane) {
    const uint32_t row0 = lane >> 2u;
    const uint32_t col0 = (lane & 3u) * 2u;
    TileI32 out = {};
#pragma unroll
    for (uint32_t j = 0; j < K32_GROUPS; j++) {
        const uint32_t b = dilate_q3_32_u4(
            weights->low2[j][lane], weights->high[j][lane]);
        const int32_t scale0 = (int32_t)decode_unsigned_field8(
            weights->scales[col0], METADATA_BITS, j);
        const int32_t scale1 = (int32_t)decode_unsigned_field8(
            weights->scales[col0 + 1u], METADATA_BITS, j);
        const int32_t min0 = (int32_t)decode_unsigned_field8(
            weights->mins[col0], METADATA_BITS, j);
        const int32_t min1 = (int32_t)decode_unsigned_field8(
            weights->mins[col0 + 1u], METADATA_BITS, j);
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            uint32_t a_low, a_high;
            activation_k32_fragments(activation, a_tile, j, lane,
                                     &a_low, &a_high);
            int32_t lo0 = 0, lo1 = 0, hi0 = 0, hi1 = 0;
            mma_m8n8k32_u4_u4(lo0, lo1, a_low, b);
            mma_m8n8k32_s4_u4(hi0, hi1, a_high, b);
            const uint32_t row = a_tile * MMA_M + row0;
            const int32_t bsum = (int32_t)activation->bsums[row][2u * j] +
                                 (int32_t)activation->bsums[row][2u * j + 1u];
            out.v[2u * a_tile] += scale0 * (lo0 + 16 * hi0);
            out.v[2u * a_tile + 1u] += scale1 * (lo1 + 16 * hi1);
            out.correction[2u * a_tile] += min0 * bsum;
            out.correction[2u * a_tile + 1u] += min1 * bsum;
        }
    }
    return out;
}

__device__ __forceinline__ static TileI32 compute_q3q4_50(
        const ActivationCase *activation,
        const NativeQ3Q4_50Tile *weights, uint32_t lane) {
    const uint32_t row0 = lane >> 2u;
    const uint32_t col0 = (lane & 3u) * 2u;
    const uint32_t bcol = lane >> 2u;
    TileI32 out = {};
#pragma unroll
    for (uint32_t j = 0; j < K32_GROUPS; j++) {
        uint32_t b = dilate_q3_32_u4(
            weights->low2[j][lane], weights->high3[j][lane]);
        b |= dilate_high1_u4(q3q4_high4_fragment(
            weights, bcol, j, lane & 3u)) << 3u;
        const int32_t scale0 = decode_scale6_8(weights->scales[col0], j);
        const int32_t scale1 = decode_scale6_8(weights->scales[col0 + 1u], j);
        const int32_t bias0 =
            (weights->promoted_mask[col0] & (1u << j)) ? 8 : 4;
        const int32_t bias1 =
            (weights->promoted_mask[col0 + 1u] & (1u << j)) ? 8 : 4;
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            uint32_t a_low, a_high;
            activation_k32_fragments(activation, a_tile, j, lane,
                                     &a_low, &a_high);
            int32_t lo0 = 0, lo1 = 0, hi0 = 0, hi1 = 0;
            mma_m8n8k32_u4_u4(lo0, lo1, a_low, b);
            mma_m8n8k32_s4_u4(hi0, hi1, a_high, b);
            const uint32_t row = a_tile * MMA_M + row0;
            const int32_t bsum = (int32_t)activation->bsums[row][2u * j] +
                                 (int32_t)activation->bsums[row][2u * j + 1u];
            out.v[2u * a_tile] +=
                scale0 * (lo0 + 16 * hi0 - bias0 * bsum);
            out.v[2u * a_tile + 1u] +=
                scale1 * (lo1 + 16 * hi1 - bias1 * bsum);
        }
    }
    return out;
}

__device__ __forceinline__ static TileI32 compute_q5_32(
        const ActivationCase *activation, const NativeQ5_32Tile *weights,
        uint32_t lane) {
    const uint32_t row0 = lane >> 2u;
    const uint32_t col0 = (lane & 3u) * 2u;
    TileI32 out = {};
#pragma unroll
    for (uint32_t j = 0; j < K32_GROUPS; j++) {
        const uint32_t b_low = weights->low4[j][lane];
        const uint32_t b_high = dilate_high1_u4(weights->high[j][lane]);
        const int32_t scale0 = decode_scale6_8(weights->scales[col0], j);
        const int32_t scale1 = decode_scale6_8(weights->scales[col0 + 1u], j);
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            uint32_t a_low, a_high;
            activation_k32_fragments(activation, a_tile, j, lane,
                                     &a_low, &a_high);
            int32_t l0 = 0, l1 = 0, lh0 = 0, lh1 = 0;
            int32_t h0 = 0, h1 = 0, hh0 = 0, hh1 = 0;
            mma_m8n8k32_u4_u4(l0, l1, a_low, b_low);
            mma_m8n8k32_s4_u4(lh0, lh1, a_high, b_low);
            mma_m8n8k32_u4_u4(h0, h1, a_low, b_high);
            mma_m8n8k32_s4_u4(hh0, hh1, a_high, b_high);
            const uint32_t row = a_tile * MMA_M + row0;
            const int32_t bsum = (int32_t)activation->bsums[row][2u * j] +
                                 (int32_t)activation->bsums[row][2u * j + 1u];
            const int32_t dot0 = l0 + 16 * lh0 + 16 * (h0 + 16 * hh0) -
                                 16 * bsum;
            const int32_t dot1 = l1 + 16 * lh1 + 16 * (h1 + 16 * hh1) -
                                 16 * bsum;
            out.v[2u * a_tile] += scale0 * dot0;
            out.v[2u * a_tile + 1u] += scale1 * dot1;
        }
    }
    return out;
}

template <Variant V, typename Weight> struct VariantCompute;

template <> struct VariantCompute<VAR_Q4K_NATIVE_CONTROL, NativeQ4KTile> {
    __device__ __forceinline__ static TileI32 run(
            const ActivationCase *activation, const NativeQ4KTile *weights,
            uint32_t lane) {
        return compute_q4k_native(activation, weights, lane);
    }
};

template <> struct VariantCompute<VAR_Q3K_K16_U8, NativeQ3KTile> {
    __device__ __forceinline__ static TileI32 run(
            const ActivationCase *activation, const NativeQ3KTile *weights,
            uint32_t lane) {
        return compute_q3k_k16_u8(activation, weights, lane);
    }
};

template <> struct VariantCompute<VAR_SM75_Q3_32, NativeQ3_32Tile> {
    __device__ __forceinline__ static TileI32 run(
            const ActivationCase *activation, const NativeQ3_32Tile *weights,
            uint32_t lane) {
        return compute_q3_32(activation, weights, lane);
    }
};

template <> struct VariantCompute<VAR_SM75_Q4_32, NativeQ4_32Tile> {
    __device__ __forceinline__ static TileI32 run(
            const ActivationCase *activation, const NativeQ4_32Tile *weights,
            uint32_t lane) {
        return compute_q4_32(activation, weights, lane);
    }
};

template <> struct VariantCompute<VAR_SM75_Q2Q3_50, NativeQ2Q3_50Tile> {
    __device__ __forceinline__ static TileI32 run(
            const ActivationCase *activation,
            const NativeQ2Q3_50Tile *weights, uint32_t lane) {
        return compute_q2q3(activation, weights, lane);
    }
};

template <> struct VariantCompute<VAR_SM75_Q2Q3_75, NativeQ2Q3_75Tile> {
    __device__ __forceinline__ static TileI32 run(
            const ActivationCase *activation,
            const NativeQ2Q3_75Tile *weights, uint32_t lane) {
        return compute_q2q3(activation, weights, lane);
    }
};

template <> struct VariantCompute<VAR_SM75_Q3A4, NativeQ3A4Tile> {
    __device__ __forceinline__ static TileI32 run(
            const ActivationCase *activation, const NativeQ3A4Tile *weights,
            uint32_t lane) {
        return compute_q3a<NativeQ3A4Tile, 4u>(activation, weights, lane);
    }
};

template <> struct VariantCompute<VAR_SM75_Q3A6, NativeQ3A6Tile> {
    __device__ __forceinline__ static TileI32 run(
            const ActivationCase *activation, const NativeQ3A6Tile *weights,
            uint32_t lane) {
        return compute_q3a<NativeQ3A6Tile, 6u>(activation, weights, lane);
    }
};

template <> struct VariantCompute<VAR_SM75_Q3Q4_50, NativeQ3Q4_50Tile> {
    __device__ __forceinline__ static TileI32 run(
            const ActivationCase *activation,
            const NativeQ3Q4_50Tile *weights, uint32_t lane) {
        return compute_q3q4_50(activation, weights, lane);
    }
};

template <> struct VariantCompute<VAR_SM75_Q5_32, NativeQ5_32Tile> {
    __device__ __forceinline__ static TileI32 run(
            const ActivationCase *activation, const NativeQ5_32Tile *weights,
            uint32_t lane) {
        return compute_q5_32(activation, weights, lane);
    }
};

template <Variant V, typename Weight>
__device__ __forceinline__ static TileI32 compute_variant(
        const ActivationCase *activation, const Weight *weights,
        uint32_t lane) {
    return VariantCompute<V, Weight>::run(activation, weights, lane);
}

template <Variant V, typename Weight>
__device__ __forceinline__ static void run_kernel_body(
        const ActivationCase *activations, const Weight *weights,
        int32_t *exact, uint32_t *checksums, uint32_t repeats,
        uint32_t activation_case_count, uint32_t weight_case_count,
        uint32_t case_offset) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint64_t work_id =
        (uint64_t)blockIdx.x * WARPS_PER_CTA + warp;
    uint32_t activation_ci = (uint32_t)(work_id % activation_case_count);
    uint32_t weight_ci =
        (uint32_t)((work_id + case_offset) % weight_case_count);
    const uint64_t work_count = (uint64_t)gridDim.x * WARPS_PER_CTA;
    uint32_t weight_step = (uint32_t)(work_count % weight_case_count);
    if (!weight_step) weight_step = 1u;
    const uint32_t weight_wrap = weight_case_count - weight_step;
    uint32_t check = 0x9e3779b9u ^
        ((uint32_t)work_id * 0x85ebca6bu + lane);
    TileI32 result = {};
    for (uint32_t r = 0; r < repeats; r++) {
        result = compute_variant<V>(activations + activation_ci,
                                    weights + weight_ci, lane);
        check = (check << 5u) | (check >> 27u);
        check ^= (uint32_t)tile_value(result, 0) + r * 0x7f4a7c15u;
        check ^= (uint32_t)tile_value(result, 1) * 0x27d4eb2du;
        check ^= (uint32_t)tile_value(result, 2) * 0x165667b1u;
        check ^= (uint32_t)tile_value(result, 3) * 0xd3a2646cu;
        if (++activation_ci == activation_case_count) activation_ci = 0;
        weight_ci = weight_ci >= weight_wrap
            ? weight_ci - weight_wrap : weight_ci + weight_step;
    }
    if (work_id == 0) {
        const uint32_t col0 = (lane & 3u) * 2u;
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            const uint32_t row = a_tile * MMA_M + (lane >> 2u);
            exact[row * TILE_N + col0] = tile_value(result, 2u * a_tile);
            exact[row * TILE_N + col0 + 1u] =
                tile_value(result, 2u * a_tile + 1u);
        }
    }
    checksums[work_id * WARP_SIZE_ + lane] = check;
}

/* A bounded fused gate/up shape.  It deliberately uses two disjoint weight
 * cases, retains both complete accumulator chains, and performs the shipping
 * SiLU*up epilogue once per M16xN8 result.  This is still a K256 arithmetic
 * screen rather than the full K4096 expert kernel; its purpose is to expose
 * the format-dependent register/issue cost of retaining both projections. */
template <Variant V, typename Weight>
__device__ __forceinline__ static void run_gate_up_kernel_body(
        const ActivationCase *activations, const Weight *weights,
        int32_t *exact, uint32_t *checksums, uint32_t repeats,
        uint32_t activation_case_count, uint32_t weight_case_count,
        uint32_t case_offset) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint64_t work_id =
        (uint64_t)blockIdx.x * WARPS_PER_CTA + warp;
    uint32_t activation_ci = (uint32_t)(work_id % activation_case_count);
    uint32_t gate_ci =
        (uint32_t)((work_id + case_offset) % weight_case_count);
    const uint32_t up_delta = weight_case_count > 1u
        ? (weight_case_count >> 1u) : 0u;
    const uint64_t work_count = (uint64_t)gridDim.x * WARPS_PER_CTA;
    uint32_t weight_step = (uint32_t)(work_count % weight_case_count);
    if (!weight_step) weight_step = 1u;
    const uint32_t weight_wrap = weight_case_count - weight_step;
    uint32_t check = 0x517cc1b7u ^
        ((uint32_t)work_id * 0x9e3779b9u + lane);
    TileI32 gate = {}, up = {};
    for (uint32_t r = 0; r < repeats; r++) {
        uint32_t up_ci = gate_ci + up_delta;
        if (up_ci >= weight_case_count) up_ci -= weight_case_count;
        gate = compute_variant<V>(activations + activation_ci,
                                  weights + gate_ci, lane);
        up = compute_variant<V>(activations + activation_ci,
                                weights + up_ci, lane);
#pragma unroll
        for (uint32_t i = 0; i < 2u * A_TILES; i++) {
            /* Keep expf in a numerically tame range.  The opaque fp16 block
             * factors are intentionally outside this integer screen. */
            const float g = (float)tile_value(gate, i) * 0x1p-20f;
            const float u = (float)tile_value(up, i) * 0x1p-20f;
            const float mid = (g / (1.0f + expf(-g))) * u;
            check = (check << 7u) | (check >> 25u);
            check ^= __float_as_uint(mid) + i * 0x85ebca6bu;
        }
        if (++activation_ci == activation_case_count) activation_ci = 0;
        gate_ci = gate_ci >= weight_wrap
            ? gate_ci - weight_wrap : gate_ci + weight_step;
    }
    if (work_id == 0) {
        const uint32_t col0 = (lane & 3u) * 2u;
#pragma unroll
        for (uint32_t a_tile = 0; a_tile < A_TILES; a_tile++) {
            const uint32_t row = a_tile * MMA_M + (lane >> 2u);
            exact[row * TILE_N + col0] = tile_value(gate, 2u * a_tile);
            exact[row * TILE_N + col0 + 1u] =
                tile_value(up, 2u * a_tile + 1u);
        }
    }
    checksums[work_id * WARP_SIZE_ + lane] = check;
}

extern "C" __global__ void sm75_q4k_native_control_kernel(
        const ActivationCase *activations, const NativeQ4KTile *weights,
        int32_t *exact, uint32_t *checksums, uint32_t repeats,
        uint32_t activation_case_count, uint32_t weight_case_count,
        uint32_t case_offset) {
    run_kernel_body<VAR_Q4K_NATIVE_CONTROL>(
        activations, weights, exact, checksums, repeats,
        activation_case_count, weight_case_count, case_offset);
}

extern "C" __global__ void sm75_q3k_k16_u8_kernel(
        const ActivationCase *activations, const NativeQ3KTile *weights,
        int32_t *exact, uint32_t *checksums, uint32_t repeats,
        uint32_t activation_case_count, uint32_t weight_case_count,
        uint32_t case_offset) {
    run_kernel_body<VAR_Q3K_K16_U8>(
        activations, weights, exact, checksums, repeats,
        activation_case_count, weight_case_count, case_offset);
}

extern "C" __global__ void sm75_q3_32_kernel(
        const ActivationCase *activations, const NativeQ3_32Tile *weights,
        int32_t *exact, uint32_t *checksums, uint32_t repeats,
        uint32_t activation_case_count, uint32_t weight_case_count,
        uint32_t case_offset) {
    run_kernel_body<VAR_SM75_Q3_32>(
        activations, weights, exact, checksums, repeats,
        activation_case_count, weight_case_count, case_offset);
}

extern "C" __global__ void sm75_q4_32_kernel(
        const ActivationCase *activations, const NativeQ4_32Tile *weights,
        int32_t *exact, uint32_t *checksums, uint32_t repeats,
        uint32_t activation_case_count, uint32_t weight_case_count,
        uint32_t case_offset) {
    run_kernel_body<VAR_SM75_Q4_32>(
        activations, weights, exact, checksums, repeats,
        activation_case_count, weight_case_count, case_offset);
}

#define DEFINE_SCREEN_KERNEL(NAME, VARIANT, TYPE) \
extern "C" __global__ void NAME( \
        const ActivationCase *activations, const TYPE *weights, \
        int32_t *exact, uint32_t *checksums, uint32_t repeats, \
        uint32_t activation_case_count, uint32_t weight_case_count, \
        uint32_t case_offset) { \
    run_kernel_body<VARIANT>( \
        activations, weights, exact, checksums, repeats, \
        activation_case_count, weight_case_count, case_offset); \
}

DEFINE_SCREEN_KERNEL(sm75_q2q3_32_50_kernel, VAR_SM75_Q2Q3_50,
                     NativeQ2Q3_50Tile)
DEFINE_SCREEN_KERNEL(sm75_q2q3_32_75_kernel, VAR_SM75_Q2Q3_75,
                     NativeQ2Q3_75Tile)
DEFINE_SCREEN_KERNEL(sm75_q3a_32_4_kernel, VAR_SM75_Q3A4,
                     NativeQ3A4Tile)
DEFINE_SCREEN_KERNEL(sm75_q3a_32_6_kernel, VAR_SM75_Q3A6,
                     NativeQ3A6Tile)
DEFINE_SCREEN_KERNEL(sm75_q3q4_32_50_kernel, VAR_SM75_Q3Q4_50,
                     NativeQ3Q4_50Tile)
DEFINE_SCREEN_KERNEL(sm75_q5_32_kernel, VAR_SM75_Q5_32,
                     NativeQ5_32Tile)

#undef DEFINE_SCREEN_KERNEL

#define DEFINE_GATE_UP_SCREEN_KERNEL(NAME, VARIANT, TYPE) \
extern "C" __global__ void NAME( \
        const ActivationCase *activations, const TYPE *weights, \
        int32_t *exact, uint32_t *checksums, uint32_t repeats, \
        uint32_t activation_case_count, uint32_t weight_case_count, \
        uint32_t case_offset) { \
    run_gate_up_kernel_body<VARIANT>( \
        activations, weights, exact, checksums, repeats, \
        activation_case_count, weight_case_count, case_offset); \
}

DEFINE_GATE_UP_SCREEN_KERNEL(sm75_q4k_native_control_gate_up_kernel,
    VAR_Q4K_NATIVE_CONTROL, NativeQ4KTile)
DEFINE_GATE_UP_SCREEN_KERNEL(sm75_q3k_k16_u8_gate_up_kernel,
    VAR_Q3K_K16_U8, NativeQ3KTile)
DEFINE_GATE_UP_SCREEN_KERNEL(sm75_q3_32_gate_up_kernel,
    VAR_SM75_Q3_32, NativeQ3_32Tile)
DEFINE_GATE_UP_SCREEN_KERNEL(sm75_q4_32_gate_up_kernel,
    VAR_SM75_Q4_32, NativeQ4_32Tile)
DEFINE_GATE_UP_SCREEN_KERNEL(sm75_q2q3_32_50_gate_up_kernel,
    VAR_SM75_Q2Q3_50, NativeQ2Q3_50Tile)
DEFINE_GATE_UP_SCREEN_KERNEL(sm75_q2q3_32_75_gate_up_kernel,
    VAR_SM75_Q2Q3_75, NativeQ2Q3_75Tile)
DEFINE_GATE_UP_SCREEN_KERNEL(sm75_q3a_32_4_gate_up_kernel,
    VAR_SM75_Q3A4, NativeQ3A4Tile)
DEFINE_GATE_UP_SCREEN_KERNEL(sm75_q3a_32_6_gate_up_kernel,
    VAR_SM75_Q3A6, NativeQ3A6Tile)
DEFINE_GATE_UP_SCREEN_KERNEL(sm75_q3q4_32_50_gate_up_kernel,
    VAR_SM75_Q3Q4_50, NativeQ3Q4_50Tile)
DEFINE_GATE_UP_SCREEN_KERNEL(sm75_q5_32_gate_up_kernel,
    VAR_SM75_Q5_32, NativeQ5_32Tile)

#undef DEFINE_GATE_UP_SCREEN_KERNEL

static void reference_q4k(const ActivationCase *activation,
                          const Q4KBlock weights[TILE_N],
                          int32_t out[TILE_M][TILE_N]) {
    for (uint32_t row = 0; row < TILE_M; row++) {
        for (uint32_t col = 0; col < TILE_N; col++) {
            int32_t total = 0;
            for (uint32_t j = 0; j < K32_GROUPS; j++) {
                uint32_t scale, minimum;
                decode_q4k_scale_min(weights[col].scales_mins, j,
                                     &scale, &minimum);
                int32_t dot = 0, sum = 0;
                for (uint32_t i = 0; i < K32; i++) {
                    const int32_t a = activation->q8[row][j * K32 + i];
                    dot += a * canonical_q4k_value(&weights[col], j * K32 + i);
                    sum += a;
                }
                total += (int32_t)scale * dot - (int32_t)minimum * sum;
            }
            out[row][col] = total;
        }
    }
}

static void reference_q3k(const ActivationCase *activation,
                          const Q3KBlock weights[TILE_N],
                          int32_t out[TILE_M][TILE_N]) {
    for (uint32_t row = 0; row < TILE_M; row++) {
        for (uint32_t col = 0; col < TILE_N; col++) {
            int32_t total = 0;
            for (uint32_t j = 0; j < K16_GROUPS; j++) {
                int32_t dot = 0;
                for (uint32_t i = 0; i < K16; i++)
                    dot += (int32_t)activation->q8[row][j * K16 + i] *
                           (int32_t)canonical_q3k_value(
                               &weights[col], j * K16 + i);
                total += decode_q3k_scale(weights[col].scales, j) * dot;
            }
            out[row][col] = total;
        }
    }
}

static void reference_q3_32(const ActivationCase *activation,
                            const SM75Q3_32Block weights[TILE_N],
                            int32_t out[TILE_M][TILE_N]) {
    for (uint32_t row = 0; row < TILE_M; row++) {
        for (uint32_t col = 0; col < TILE_N; col++) {
            int32_t total = 0;
            for (uint32_t j = 0; j < K32_GROUPS; j++) {
                int32_t dot = 0;
                for (uint32_t i = 0; i < K32; i++) {
                    const int32_t q = (int32_t)canonical_q3_32_code(
                        &weights[col], j * K32 + i) - 4;
                    dot += (int32_t)activation->q8[row][j * K32 + i] * q;
                }
                total += decode_scale6_8(weights[col].scales, j) * dot;
            }
            out[row][col] = total;
        }
    }
}

static void reference_q4_32(const ActivationCase *activation,
                            const SM75Q4_32Block weights[TILE_N],
                            int32_t out[TILE_M][TILE_N]) {
    for (uint32_t row = 0; row < TILE_M; row++) {
        for (uint32_t col = 0; col < TILE_N; col++) {
            int32_t total = 0;
            for (uint32_t j = 0; j < K32_GROUPS; j++) {
                int32_t dot = 0;
                for (uint32_t i = 0; i < K32; i++)
                    dot += (int32_t)activation->q8[row][j * K32 + i] *
                           (int32_t)canonical_q4_32_value(
                               &weights[col], j * K32 + i);
                total += decode_scale6_8(weights[col].scales, j) * dot;
            }
            out[row][col] = total;
        }
    }
}

template <typename Block>
static void reference_q2q3(const ActivationCase *activation,
                           const Block weights[TILE_N],
                           int32_t out[TILE_M][TILE_N]) {
    for (uint32_t row = 0; row < TILE_M; row++) {
        for (uint32_t col = 0; col < TILE_N; col++) {
            int32_t total = 0;
            for (uint32_t j = 0; j < K32_GROUPS; j++) {
                int32_t dot = 0;
                for (uint32_t i = 0; i < K32; i++)
                    dot += (int32_t)activation->q8[row][j * K32 + i] *
                           (int32_t)canonical_q2q3_value(
                               weights[col].low2,
                               weights[col].promoted_mask,
                               weights[col].high, j * K32 + i);
                total += decode_scale6_8(weights[col].scales, j) * dot;
            }
            out[row][col] = total;
        }
    }
}

template <typename Block, uint32_t METADATA_BITS>
static void reference_q3a(const ActivationCase *activation,
                          const Block weights[TILE_N],
                          int32_t out[TILE_M][TILE_N]) {
    for (uint32_t row = 0; row < TILE_M; row++) {
        for (uint32_t col = 0; col < TILE_N; col++) {
            int32_t total = 0;
            for (uint32_t j = 0; j < K32_GROUPS; j++) {
                int32_t dot = 0, sum = 0;
                for (uint32_t i = 0; i < K32; i++) {
                    const int32_t a = activation->q8[row][j * K32 + i];
                    dot += a * canonical_affine_value(
                        weights[col].qs, j * K32 + i);
                    sum += a;
                }
                const int32_t scale = (int32_t)decode_unsigned_field8(
                    weights[col].scales, METADATA_BITS, j);
                const int32_t minimum = (int32_t)decode_unsigned_field8(
                    weights[col].mins, METADATA_BITS, j);
                total += scale * dot - minimum * sum;
            }
            out[row][col] = total;
        }
    }
}

static void reference_q3q4_50(const ActivationCase *activation,
                              const SM75Q3Q4_50Block weights[TILE_N],
                              int32_t out[TILE_M][TILE_N]) {
    for (uint32_t row = 0; row < TILE_M; row++) {
        for (uint32_t col = 0; col < TILE_N; col++) {
            int32_t total = 0;
            for (uint32_t j = 0; j < K32_GROUPS; j++) {
                int32_t dot = 0;
                for (uint32_t i = 0; i < K32; i++)
                    dot += (int32_t)activation->q8[row][j * K32 + i] *
                           (int32_t)canonical_q3q4_value(
                               &weights[col], j * K32 + i);
                total += decode_scale6_8(weights[col].scales, j) * dot;
            }
            out[row][col] = total;
        }
    }
}

static void reference_q5_32(const ActivationCase *activation,
                            const SM75Q5_32Block weights[TILE_N],
                            int32_t out[TILE_M][TILE_N]) {
    for (uint32_t row = 0; row < TILE_M; row++) {
        for (uint32_t col = 0; col < TILE_N; col++) {
            int32_t total = 0;
            for (uint32_t j = 0; j < K32_GROUPS; j++) {
                int32_t dot = 0;
                for (uint32_t i = 0; i < K32; i++)
                    dot += (int32_t)activation->q8[row][j * K32 + i] *
                           (int32_t)canonical_q5_value(
                               &weights[col], j * K32 + i);
                total += decode_scale6_8(weights[col].scales, j) * dot;
            }
            out[row][col] = total;
        }
    }
}

static void reference_variant(Variant variant, const ActivationCase *activation,
                              const CanonicalWeights *weights,
                              int32_t out[TILE_M][TILE_N]) {
    switch (variant) {
    case VAR_Q4K_NATIVE_CONTROL:
        reference_q4k(activation, weights->q4k, out); break;
    case VAR_Q3K_K16_U8:
        reference_q3k(activation, weights->q3k, out); break;
    case VAR_SM75_Q3_32:
        reference_q3_32(activation, weights->q3_32, out); break;
    case VAR_SM75_Q4_32:
        reference_q4_32(activation, weights->q4_32, out); break;
    case VAR_SM75_Q2Q3_50:
        reference_q2q3(activation, weights->q2q3_50, out); break;
    case VAR_SM75_Q2Q3_75:
        reference_q2q3(activation, weights->q2q3_75, out); break;
    case VAR_SM75_Q3A4:
        reference_q3a<SM75Q3A4Block, 4u>(activation, weights->q3a4, out); break;
    case VAR_SM75_Q3A6:
        reference_q3a<SM75Q3A6Block, 6u>(activation, weights->q3a6, out); break;
    case VAR_SM75_Q3Q4_50:
        reference_q3q4_50(activation, weights->q3q4_50, out); break;
    case VAR_SM75_Q5_32:
        reference_q5_32(activation, weights->q5_32, out); break;
    default: abort();
    }
}

static size_t native_tile_bytes(Variant variant) {
    switch (variant) {
    case VAR_Q4K_NATIVE_CONTROL: return sizeof(NativeQ4KTile);
    case VAR_Q3K_K16_U8: return sizeof(NativeQ3KTile);
    case VAR_SM75_Q3_32: return sizeof(NativeQ3_32Tile);
    case VAR_SM75_Q4_32: return sizeof(NativeQ4_32Tile);
    case VAR_SM75_Q2Q3_50: return sizeof(NativeQ2Q3_50Tile);
    case VAR_SM75_Q2Q3_75: return sizeof(NativeQ2Q3_75Tile);
    case VAR_SM75_Q3A4: return sizeof(NativeQ3A4Tile);
    case VAR_SM75_Q3A6: return sizeof(NativeQ3A6Tile);
    case VAR_SM75_Q3Q4_50: return sizeof(NativeQ3Q4_50Tile);
    case VAR_SM75_Q5_32: return sizeof(NativeQ5_32Tile);
    default: abort();
    }
}

static size_t canonical_block_bytes(Variant variant) {
    switch (variant) {
    case VAR_Q4K_NATIVE_CONTROL: return sizeof(Q4KBlock);
    case VAR_Q3K_K16_U8: return sizeof(Q3KBlock);
    case VAR_SM75_Q3_32: return sizeof(SM75Q3_32Block);
    case VAR_SM75_Q4_32: return sizeof(SM75Q4_32Block);
    case VAR_SM75_Q2Q3_50: return sizeof(SM75Q2Q3_50Block);
    case VAR_SM75_Q2Q3_75: return sizeof(SM75Q2Q3_75Block);
    case VAR_SM75_Q3A4: return sizeof(SM75Q3A4Block);
    case VAR_SM75_Q3A6: return sizeof(SM75Q3A6Block);
    case VAR_SM75_Q3Q4_50: return sizeof(SM75Q3Q4_50Block);
    case VAR_SM75_Q5_32: return sizeof(SM75Q5_32Block);
    default: abort();
    }
}

static void launch_variant(Variant variant, uint32_t blocks, uint32_t repeats,
                           uint32_t activation_case_count,
                           uint32_t weight_case_count, uint32_t case_offset,
                           const ActivationCase *activations,
                           const DeviceWeights &weights, int32_t *exact,
                           uint32_t *checksums, cudaStream_t stream) {
    const dim3 grid(blocks), block(THREADS_PER_CTA);
    switch (variant) {
    case VAR_Q4K_NATIVE_CONTROL:
        sm75_q4k_native_control_kernel<<<grid, block, 0, stream>>>(
            activations, weights.q4k, exact, checksums, repeats,
            activation_case_count, weight_case_count, case_offset); break;
    case VAR_Q3K_K16_U8:
        sm75_q3k_k16_u8_kernel<<<grid, block, 0, stream>>>(
            activations, weights.q3k, exact, checksums, repeats,
            activation_case_count, weight_case_count, case_offset); break;
    case VAR_SM75_Q3_32:
        sm75_q3_32_kernel<<<grid, block, 0, stream>>>(
            activations, weights.q3_32, exact, checksums, repeats,
            activation_case_count, weight_case_count, case_offset); break;
    case VAR_SM75_Q4_32:
        sm75_q4_32_kernel<<<grid, block, 0, stream>>>(
            activations, weights.q4_32, exact, checksums, repeats,
            activation_case_count, weight_case_count, case_offset); break;
    case VAR_SM75_Q2Q3_50:
        sm75_q2q3_32_50_kernel<<<grid, block, 0, stream>>>(
            activations, weights.q2q3_50, exact, checksums, repeats,
            activation_case_count, weight_case_count, case_offset); break;
    case VAR_SM75_Q2Q3_75:
        sm75_q2q3_32_75_kernel<<<grid, block, 0, stream>>>(
            activations, weights.q2q3_75, exact, checksums, repeats,
            activation_case_count, weight_case_count, case_offset); break;
    case VAR_SM75_Q3A4:
        sm75_q3a_32_4_kernel<<<grid, block, 0, stream>>>(
            activations, weights.q3a4, exact, checksums, repeats,
            activation_case_count, weight_case_count, case_offset); break;
    case VAR_SM75_Q3A6:
        sm75_q3a_32_6_kernel<<<grid, block, 0, stream>>>(
            activations, weights.q3a6, exact, checksums, repeats,
            activation_case_count, weight_case_count, case_offset); break;
    case VAR_SM75_Q3Q4_50:
        sm75_q3q4_32_50_kernel<<<grid, block, 0, stream>>>(
            activations, weights.q3q4_50, exact, checksums, repeats,
            activation_case_count, weight_case_count, case_offset); break;
    case VAR_SM75_Q5_32:
        sm75_q5_32_kernel<<<grid, block, 0, stream>>>(
            activations, weights.q5_32, exact, checksums, repeats,
            activation_case_count, weight_case_count, case_offset); break;
    default: abort();
    }
}

static void launch_gate_up_variant(
        Variant variant, uint32_t blocks, uint32_t repeats,
        uint32_t activation_case_count, uint32_t weight_case_count,
        uint32_t case_offset, const ActivationCase *activations,
        const DeviceWeights &weights, int32_t *exact,
        uint32_t *checksums, cudaStream_t stream) {
    const dim3 grid(blocks), block(THREADS_PER_CTA);
#define LAUNCH_GATE_UP(KERNEL, FIELD) \
    KERNEL<<<grid, block, 0, stream>>>( \
        activations, weights.FIELD, exact, checksums, repeats, \
        activation_case_count, weight_case_count, case_offset)
    switch (variant) {
    case VAR_Q4K_NATIVE_CONTROL:
        LAUNCH_GATE_UP(sm75_q4k_native_control_gate_up_kernel, q4k); break;
    case VAR_Q3K_K16_U8:
        LAUNCH_GATE_UP(sm75_q3k_k16_u8_gate_up_kernel, q3k); break;
    case VAR_SM75_Q3_32:
        LAUNCH_GATE_UP(sm75_q3_32_gate_up_kernel, q3_32); break;
    case VAR_SM75_Q4_32:
        LAUNCH_GATE_UP(sm75_q4_32_gate_up_kernel, q4_32); break;
    case VAR_SM75_Q2Q3_50:
        LAUNCH_GATE_UP(sm75_q2q3_32_50_gate_up_kernel, q2q3_50); break;
    case VAR_SM75_Q2Q3_75:
        LAUNCH_GATE_UP(sm75_q2q3_32_75_gate_up_kernel, q2q3_75); break;
    case VAR_SM75_Q3A4:
        LAUNCH_GATE_UP(sm75_q3a_32_4_gate_up_kernel, q3a4); break;
    case VAR_SM75_Q3A6:
        LAUNCH_GATE_UP(sm75_q3a_32_6_gate_up_kernel, q3a6); break;
    case VAR_SM75_Q3Q4_50:
        LAUNCH_GATE_UP(sm75_q3q4_32_50_gate_up_kernel, q3q4_50); break;
    case VAR_SM75_Q5_32:
        LAUNCH_GATE_UP(sm75_q5_32_gate_up_kernel, q5_32); break;
    default: abort();
    }
#undef LAUNCH_GATE_UP
}

static uint32_t scalar_q3k_u8(uint8_t low2, uint8_t high4) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < 4; i++) {
        const uint32_t u = ((low2 >> (2u * i)) & 3u) |
                           (((high4 >> i) & 1u) << 2u);
        result |= u << (8u * i);
    }
    return result;
}

static uint32_t scalar_q3_32_u4(uint16_t low2, uint8_t high8) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < 8; i++) {
        const uint32_t u = ((low2 >> (2u * i)) & 3u) |
                           (((high8 >> i) & 1u) << 2u);
        result |= u << (4u * i);
    }
    return result;
}

static int run_canonical_fixture_gate(void) {
    static const int8_t q3k_scales[K16_GROUPS] = {
        -32, -27, -22, -17, -16, -11, -6, -1,
          0,   5,  10,  15,  16,  21, 26, 31,
    };
    static const uint8_t q3k_expected[12] = {
        0x00, 0x55, 0xaa, 0xff, 0x00, 0x55, 0xaa, 0xff,
        0xe4, 0xe4, 0xe4, 0xe4,
    };
    static const int8_t k32_scales[K32_GROUPS] = {
        -32, -27, -22, -17, -16, -11, -6, -1,
    };
    static const uint8_t k32_expected[6] = {
        0x00, 0x55, 0xaa, 0xff, 0x00, 0x55,
    };
    uint8_t packed_q3k[12], packed_k32[6];
    encode_q3k_scales(q3k_scales, packed_q3k);
    encode_scale6_8(k32_scales, packed_k32);
    if (memcmp(packed_q3k, q3k_expected, sizeof(packed_q3k)) != 0 ||
        memcmp(packed_k32, k32_expected, sizeof(packed_k32)) != 0) {
        fprintf(stderr, "canonical six-bit scale fixture failure\n");
        return 1;
    }
    for (uint32_t j = 0; j < K16_GROUPS; j++) {
        if (decode_q3k_scale(packed_q3k, j) != q3k_scales[j]) {
            fprintf(stderr, "canonical Q3_K scale fixture decode failure: j=%u\n", j);
            return 1;
        }
    }
    for (uint32_t j = 0; j < K32_GROUPS; j++) {
        if (decode_scale6_8(packed_k32, j) != k32_scales[j]) {
            fprintf(stderr, "canonical K32 scale fixture decode failure: j=%u\n", j);
            return 1;
        }
    }

    Q3KBlock q3k = {};
    for (uint32_t k = 0; k < TILE_K; k++) {
        if (canonical_q3k_value(&q3k, k) != -4) {
            fprintf(stderr, "canonical Q3_K zero-mask fixture failure: k=%u\n", k);
            return 1;
        }
    }
    memset(q3k.hmask, 0xff, sizeof(q3k.hmask));
    for (uint32_t k = 0; k < TILE_K; k++) {
        if (canonical_q3k_value(&q3k, k) != 0) {
            fprintf(stderr, "canonical Q3_K high-mask fixture failure: k=%u\n", k);
            return 1;
        }
    }
    memset(q3k.qs, 0xff, sizeof(q3k.qs));
    for (uint32_t k = 0; k < TILE_K; k++) {
        if (canonical_q3k_value(&q3k, k) != 3) {
            fprintf(stderr, "canonical Q3_K low/high fixture failure: k=%u\n", k);
            return 1;
        }
    }

    /* Independent full-block standard-Q3_K mapping fixture.  The expected
     * FNV-1a digest was generated from the upstream hmask/qs equations, not
     * through this harness's setter or decoder. */
    Q3KBlock raw_q3k = {};
    for (uint32_t i = 0; i < 32u; i++)
        raw_q3k.hmask[i] = (uint8_t)(i * 29u + 7u);
    for (uint32_t i = 0; i < 64u; i++)
        raw_q3k.qs[i] = (uint8_t)(i * 53u + 11u);
    uint32_t q3k_digest = 2166136261u;
    for (uint32_t k = 0; k < TILE_K; k++) {
        const uint32_t biased =
            (uint32_t)((int32_t)canonical_q3k_value(&raw_q3k, k) + 4);
        q3k_digest = (q3k_digest ^ biased) * 16777619u;
    }
    if (q3k_digest != 0x2159f6c0u) {
        fprintf(stderr,
            "canonical Q3_K full-block fixture failure: expected=2159f6c0 actual=%08x\n",
            q3k_digest);
        return 1;
    }

    SM75Q3_32Block q3_32 = {};
    for (uint32_t k = 0; k < TILE_K; k++) {
        if (canonical_q3_32_code(&q3_32, k) != 0) {
            fprintf(stderr, "canonical Q3-32 zero-code fixture failure: k=%u\n", k);
            return 1;
        }
    }
    memset(q3_32.qs, 0xff, sizeof(q3_32.qs));
    for (uint32_t k = 0; k < TILE_K; k++) {
        if (canonical_q3_32_code(&q3_32, k) != 7) {
            fprintf(stderr, "canonical Q3-32 max-code fixture failure: k=%u\n", k);
            return 1;
        }
    }

    SM75Q4_32Block q4_32 = {};
    memset(q4_32.qs, 0x88, sizeof(q4_32.qs));
    for (uint32_t k = 0; k < TILE_K; k++) {
        if (canonical_q4_32_value(&q4_32, k) != -8) {
            fprintf(stderr, "canonical signed Q4-32 fixture failure: k=%u\n", k);
            return 1;
        }
    }
    printf("canonical_fixture_status=ok\n");
    return 0;
}

static int run_unpack_gate(void) {
    for (uint32_t low = 0; low < 256u; low++) {
        for (uint32_t high = 0; high < 16u; high++) {
            const uint32_t expected = scalar_q3k_u8((uint8_t)low,
                                                    (uint8_t)high);
            const uint32_t actual = dilate_q3k_u8((uint8_t)low,
                                                  (uint8_t)high);
            if (actual != expected) {
                fprintf(stderr,
                    "Q3_K SWAR unpack failure: low=%u high=%u expected=%08x actual=%08x\n",
                    low, high, expected, actual);
                return 1;
            }
        }
    }
    for (uint32_t low = 0; low < 65536u; low++) {
        const uint32_t expected = scalar_q3_32_u4((uint16_t)low, 0);
        const uint32_t actual = dilate_q3_32_u4((uint16_t)low, 0);
        if (actual != expected) {
            fprintf(stderr,
                "Q3-32 SWAR low-plane failure: low=%u expected=%08x actual=%08x\n",
                low, expected, actual);
            return 1;
        }
    }
    for (uint32_t high = 0; high < 256u; high++) {
        const uint32_t expected = scalar_q3_32_u4(0, (uint8_t)high);
        const uint32_t actual = dilate_q3_32_u4(0, (uint8_t)high);
        if (actual != expected) {
            fprintf(stderr,
                "Q3-32 SWAR high-plane failure: high=%u expected=%08x actual=%08x\n",
                high, expected, actual);
            return 1;
        }
    }
    uint32_t state = 0x6a09e667u;
    for (uint32_t sample = 0; sample < 65536u; sample++) {
        const uint16_t low = (uint16_t)rng_next(&state);
        const uint8_t high = (uint8_t)rng_next(&state);
        const uint32_t expected = scalar_q3_32_u4(low, high);
        const uint32_t actual = dilate_q3_32_u4(low, high);
        if (actual != expected) {
            fprintf(stderr,
                "Q3-32 SWAR mixed failure: sample=%u low=%u high=%u expected=%08x actual=%08x\n",
                sample, (uint32_t)low, (uint32_t)high, expected, actual);
            return 1;
        }
    }
    printf("unpack_status=ok\n");
    return 0;
}

static int run_correctness(uint32_t n_cases, ActivationCase *h_activation,
                           const HostWeights &host, ActivationCase *d_activation,
                           const DeviceWeights &device, int32_t *d_exact,
                           uint32_t *d_checksums) {
    if (run_canonical_fixture_gate()) return 1;
    if (run_unpack_gate()) return 1;
    CanonicalWeights canonical;
    int32_t expected[TILE_M][TILE_N];
    int32_t actual[TILE_M][TILE_N];
    for (uint32_t case_id = 0; case_id < n_cases; case_id++) {
        fill_activation(h_activation, case_id);
        fill_weights(&canonical, case_id);
        repack_all(&canonical, host.q4k, host.q3k, host.q3_32, host.q4_32,
                   host.q2q3_50, host.q2q3_75, host.q3a4, host.q3a6,
                   host.q3q4_50, host.q5_32);
        if (layout_gate(&canonical, host.q4k, host.q3k,
                        host.q3_32, host.q4_32, host.q2q3_50,
                        host.q2q3_75, host.q3a4, host.q3a6,
                        host.q3q4_50, host.q5_32, case_id)) return 1;
        die_cuda(cudaMemcpy(d_activation, h_activation, sizeof(*h_activation),
                            cudaMemcpyHostToDevice), "copy correctness activation");
        die_cuda(cudaMemcpy(device.q4k, host.q4k, sizeof(*host.q4k),
                            cudaMemcpyHostToDevice), "copy correctness Q4_K");
        die_cuda(cudaMemcpy(device.q3k, host.q3k, sizeof(*host.q3k),
                            cudaMemcpyHostToDevice), "copy correctness Q3_K");
        die_cuda(cudaMemcpy(device.q3_32, host.q3_32, sizeof(*host.q3_32),
                            cudaMemcpyHostToDevice), "copy correctness Q3-32");
        die_cuda(cudaMemcpy(device.q4_32, host.q4_32, sizeof(*host.q4_32),
                            cudaMemcpyHostToDevice), "copy correctness Q4-32");
        die_cuda(cudaMemcpy(device.q2q3_50, host.q2q3_50,
                            sizeof(*host.q2q3_50), cudaMemcpyHostToDevice),
                 "copy correctness Q2/Q3-50");
        die_cuda(cudaMemcpy(device.q2q3_75, host.q2q3_75,
                            sizeof(*host.q2q3_75), cudaMemcpyHostToDevice),
                 "copy correctness Q2/Q3-75");
        die_cuda(cudaMemcpy(device.q3a4, host.q3a4, sizeof(*host.q3a4),
                            cudaMemcpyHostToDevice), "copy correctness Q3A4");
        die_cuda(cudaMemcpy(device.q3a6, host.q3a6, sizeof(*host.q3a6),
                            cudaMemcpyHostToDevice), "copy correctness Q3A6");
        die_cuda(cudaMemcpy(device.q3q4_50, host.q3q4_50,
                            sizeof(*host.q3q4_50), cudaMemcpyHostToDevice),
                 "copy correctness Q3/Q4-50");
        die_cuda(cudaMemcpy(device.q5_32, host.q5_32, sizeof(*host.q5_32),
                            cudaMemcpyHostToDevice), "copy correctness Q5-32");
        for (int v = 0; v < VARIANT_COUNT; v++) {
            reference_variant((Variant)v, h_activation, &canonical, expected);
            launch_variant((Variant)v, 1, 1, 1, 1, 0, d_activation, device,
                           d_exact, d_checksums, 0);
            die_cuda(cudaGetLastError(), "launch correctness kernel");
            die_cuda(cudaMemcpy(actual, d_exact, sizeof(actual),
                                cudaMemcpyDeviceToHost), "copy correctness output");
            for (uint32_t row = 0; row < TILE_M; row++) {
                for (uint32_t col = 0; col < TILE_N; col++) {
                    if (actual[row][col] != expected[row][col]) {
                        fprintf(stderr,
                            "exactness failure: case=%u variant=%s row=%u col=%u "
                            "expected=%d actual=%d\n", case_id, variant_names[v],
                            row, col, expected[row][col], actual[row][col]);
                        return 1;
                    }
                }
            }
        }
    }
    printf("layout_cases=%u\nlayout_status=ok\nroundtrip_status=ok\n", n_cases);
    printf("exact_cases=%u\nexact_variants=%d\nexact_status=ok\n",
           n_cases, VARIANT_COUNT);
    return 0;
}

static int prepare_benchmark_cases(uint32_t weight_cases,
                                   ActivationCase *h_activations,
                                   const HostWeights &host) {
    for (uint32_t i = 0; i < HOT_ACTIVATION_CASES; i++)
        fill_activation(h_activations + i, 0xa110u + i);
    CanonicalWeights canonical;
    for (uint32_t i = 0; i < weight_cases; i++) {
        fill_weights(&canonical, 0x5a170000u + i);
        repack_all(&canonical, host.q4k + i, host.q3k + i,
                   host.q3_32 + i, host.q4_32 + i,
                   host.q2q3_50 + i, host.q2q3_75 + i,
                   host.q3a4 + i, host.q3a6 + i,
                   host.q3q4_50 + i, host.q5_32 + i);
        if (layout_gate(&canonical, host.q4k + i, host.q3k + i,
                        host.q3_32 + i, host.q4_32 + i,
                        host.q2q3_50 + i, host.q2q3_75 + i,
                        host.q3a4 + i, host.q3a6 + i,
                        host.q3q4_50 + i, host.q5_32 + i,
                        0x5a170000u + i)) return 1;
    }
    return 0;
}

static void copy_benchmark_cases(uint32_t weight_cases,
                                 const ActivationCase *h_activations,
                                 const HostWeights &host,
                                 ActivationCase *d_activations,
                                 const DeviceWeights &device) {
    die_cuda(cudaMemcpy(d_activations, h_activations,
                        sizeof(ActivationCase) * HOT_ACTIVATION_CASES,
                        cudaMemcpyHostToDevice), "copy benchmark activations");
    die_cuda(cudaMemcpy(device.q4k, host.q4k,
                        sizeof(NativeQ4KTile) * (size_t)weight_cases,
                        cudaMemcpyHostToDevice), "copy benchmark Q4_K");
    die_cuda(cudaMemcpy(device.q3k, host.q3k,
                        sizeof(NativeQ3KTile) * (size_t)weight_cases,
                        cudaMemcpyHostToDevice), "copy benchmark Q3_K");
    die_cuda(cudaMemcpy(device.q3_32, host.q3_32,
                        sizeof(NativeQ3_32Tile) * (size_t)weight_cases,
                        cudaMemcpyHostToDevice), "copy benchmark Q3-32");
    die_cuda(cudaMemcpy(device.q4_32, host.q4_32,
                        sizeof(NativeQ4_32Tile) * (size_t)weight_cases,
                        cudaMemcpyHostToDevice), "copy benchmark Q4-32");
    die_cuda(cudaMemcpy(device.q2q3_50, host.q2q3_50,
                        sizeof(NativeQ2Q3_50Tile) * (size_t)weight_cases,
                        cudaMemcpyHostToDevice), "copy benchmark Q2/Q3-50");
    die_cuda(cudaMemcpy(device.q2q3_75, host.q2q3_75,
                        sizeof(NativeQ2Q3_75Tile) * (size_t)weight_cases,
                        cudaMemcpyHostToDevice), "copy benchmark Q2/Q3-75");
    die_cuda(cudaMemcpy(device.q3a4, host.q3a4,
                        sizeof(NativeQ3A4Tile) * (size_t)weight_cases,
                        cudaMemcpyHostToDevice), "copy benchmark Q3A4");
    die_cuda(cudaMemcpy(device.q3a6, host.q3a6,
                        sizeof(NativeQ3A6Tile) * (size_t)weight_cases,
                        cudaMemcpyHostToDevice), "copy benchmark Q3A6");
    die_cuda(cudaMemcpy(device.q3q4_50, host.q3q4_50,
                        sizeof(NativeQ3Q4_50Tile) * (size_t)weight_cases,
                        cudaMemcpyHostToDevice), "copy benchmark Q3/Q4-50");
    die_cuda(cudaMemcpy(device.q5_32, host.q5_32,
                        sizeof(NativeQ5_32Tile) * (size_t)weight_cases,
                        cudaMemcpyHostToDevice), "copy benchmark Q5-32");
}

static int compare_double(const void *lhs, const void *rhs) {
    const double a = *(const double *)lhs;
    const double b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static void run_benchmark_mode(const char *mode, int gate_up,
                               uint32_t weight_cases,
                               uint32_t blocks, uint32_t repeats,
                               uint32_t launches, uint32_t rounds,
                               const ActivationCase *d_activations,
                               const DeviceWeights &device, int32_t *d_exact,
                               uint32_t *d_checksums) {
    double *samples = (double *)malloc(
        (size_t)VARIANT_COUNT * rounds * sizeof(*samples));
    double *sorted = (double *)malloc((size_t)rounds * sizeof(*sorted));
    if (!samples || !sorted) {
        fprintf(stderr, "error: benchmark sample allocation failed\n");
        exit(2);
    }
    cudaEvent_t start, stop;
    die_cuda(cudaEventCreate(&start), "create benchmark start event");
    die_cuda(cudaEventCreate(&stop), "create benchmark stop event");
    for (int v = 0; v < VARIANT_COUNT; v++) {
        for (uint32_t warm = 0; warm < 2; warm++) {
            if (gate_up)
                launch_gate_up_variant(
                    (Variant)v, blocks, repeats, HOT_ACTIVATION_CASES,
                    weight_cases, 0, d_activations, device,
                    d_exact, d_checksums, 0);
            else
                launch_variant((Variant)v, blocks, repeats,
                               HOT_ACTIVATION_CASES, weight_cases, 0,
                               d_activations, device, d_exact,
                               d_checksums, 0);
        }
    }
    die_cuda(cudaDeviceSynchronize(), "benchmark warmup");

    const uint64_t work_per_launch =
        (uint64_t)blocks * WARPS_PER_CTA * repeats;
    for (uint32_t round = 0; round < rounds; round++) {
        const uint32_t first = round % VARIANT_COUNT;
        for (uint32_t slot = 0; slot < VARIANT_COUNT; slot++) {
            const uint32_t offset = (round & 1u)
                ? (VARIANT_COUNT - slot) % VARIANT_COUNT : slot;
            const uint32_t v = (first + offset) % VARIANT_COUNT;
            die_cuda(cudaEventRecord(start), "record benchmark start");
            for (uint32_t launch = 0; launch < launches; launch++) {
                const uint32_t offset = (uint32_t)(
                    ((uint64_t)launch * work_per_launch) % weight_cases);
                if (gate_up)
                    launch_gate_up_variant(
                        (Variant)v, blocks, repeats, HOT_ACTIVATION_CASES,
                        weight_cases, offset, d_activations, device,
                        d_exact, d_checksums, 0);
                else
                    launch_variant((Variant)v, blocks, repeats,
                                   HOT_ACTIVATION_CASES, weight_cases, offset,
                                   d_activations, device, d_exact,
                                   d_checksums, 0);
            }
            die_cuda(cudaEventRecord(stop), "record benchmark stop");
            die_cuda(cudaEventSynchronize(stop), "synchronize benchmark");
            float ms = 0.0f;
            die_cuda(cudaEventElapsedTime(&ms, start, stop),
                     "measure benchmark");
            samples[(size_t)v * rounds + round] = (double)ms;
        }
    }

    const double logical_macs = (double)launches * blocks * WARPS_PER_CTA *
        repeats * TILE_M * TILE_N * TILE_K * (gate_up ? 2.0 : 1.0);
    printf("benchmark_samples_begin\n");
    printf("mode,round,slot,variant,canonical_block_bytes,native_tile_bytes,"
           "weight_cases,weight_footprint_bytes,total_ms,us_per_launch,"
           "relative_speed,logical_tmac_per_s\n");
    for (uint32_t round = 0; round < rounds; round++) {
        const double baseline = samples[round];
        const uint32_t first = round % VARIANT_COUNT;
        for (uint32_t slot = 0; slot < VARIANT_COUNT; slot++) {
            const uint32_t offset = (round & 1u)
                ? (VARIANT_COUNT - slot) % VARIANT_COUNT : slot;
            const uint32_t v = (first + offset) % VARIANT_COUNT;
            const double ms = samples[(size_t)v * rounds + round];
            const size_t tile_bytes = native_tile_bytes((Variant)v);
            printf("%s,%u,%u,%s,%zu,%zu,%u,%llu,%.6f,%.6f,%.6f,%.6f\n",
                   mode, round + 1u, slot + 1u, variant_names[v],
                   canonical_block_bytes((Variant)v), tile_bytes, weight_cases,
                   (unsigned long long)((uint64_t)tile_bytes * weight_cases),
                   ms, 1000.0 * ms / launches, baseline / ms,
                   logical_macs / (ms / 1000.0) / 1.0e12);
        }
    }
    printf("benchmark_samples_end\n");

    printf("benchmark_summary_begin\n");
    printf("mode,variant,weight_cases,weight_footprint_bytes,median_ms,min_ms,"
           "max_ms,us_per_launch,relative_speed,logical_tmac_per_s\n");
    double medians[VARIANT_COUNT];
    double minimums[VARIANT_COUNT];
    double maximums[VARIANT_COUNT];
    for (int v = 0; v < VARIANT_COUNT; v++) {
        memcpy(sorted, samples + (size_t)v * rounds,
               (size_t)rounds * sizeof(*sorted));
        qsort(sorted, rounds, sizeof(*sorted), compare_double);
        minimums[v] = sorted[0];
        maximums[v] = sorted[rounds - 1u];
        medians[v] = (rounds & 1u) ? sorted[rounds / 2u] :
            0.5 * (sorted[rounds / 2u - 1u] + sorted[rounds / 2u]);
    }
    for (int v = 0; v < VARIANT_COUNT; v++) {
        const double ms = medians[v];
        const size_t bytes = native_tile_bytes((Variant)v);
        printf("%s,%s,%u,%llu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
               mode, variant_names[v], weight_cases,
               (unsigned long long)((uint64_t)bytes * weight_cases),
               ms, minimums[v], maximums[v], 1000.0 * ms / launches,
               medians[0] / ms,
               logical_macs / (ms / 1000.0) / 1.0e12);
    }
    printf("benchmark_summary_end\n");
    die_cuda(cudaEventDestroy(stop), "destroy benchmark stop event");
    die_cuda(cudaEventDestroy(start), "destroy benchmark start event");
    free(sorted);
    free(samples);
}

static Variant parse_variant(const char *name) {
    for (int v = 0; v < VARIANT_COUNT; v++)
        if (strcmp(name, variant_names[v]) == 0) return (Variant)v;
    fprintf(stderr, "error: unknown variant: %s\nvariants:", name);
    for (int v = 0; v < VARIANT_COUNT; v++)
        fprintf(stderr, " %s", variant_names[v]);
    fprintf(stderr, "\n");
    exit(2);
}

static uint32_t parse_u32_allow_zero(const char *text, const char *name) {
    char *end = NULL;
    const unsigned long value = strtoul(text, &end, 10);
    if (!text[0] || !end || *end || value > 0xfffffffful) {
        fprintf(stderr, "error: invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint32_t)value;
}

static uint32_t parse_u32(const char *text, const char *name) {
    const uint32_t value = parse_u32_allow_zero(text, name);
    if (!value) {
        fprintf(stderr, "error: %s must be greater than zero\n", name);
        exit(2);
    }
    return value;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [--device N] [--cases N] [--blocks N] [--repeats N] "
        "[--launches N] [--bench-cases N] [--rounds N] "
        "[--correctness-only | --benchmark-only | --profile VARIANT] "
        "[--profile-shape down|gate-up]\n",
        argv0);
}

static void *checked_malloc(size_t bytes, const char *what) {
    void *ptr = malloc(bytes);
    if (!ptr) {
        fprintf(stderr, "error: host allocation failed for %s (%zu bytes)\n",
                what, bytes);
        exit(2);
    }
    return ptr;
}

int main(int argc, char **argv) {
    int device_id = 0;
    uint32_t cases = 256;
    uint32_t blocks = 0;
    uint32_t repeats = 128;
    uint32_t launches = 20;
    uint32_t bench_cases = 0;
    uint32_t rounds = 9;
    int do_correctness = 1, do_benchmark = 1, profile = 0;
    int profile_gate_up = 0;
    Variant profile_variant = VAR_Q4K_NATIVE_CONTROL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            device_id = (int)parse_u32_allow_zero(argv[++i], "device");
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
            do_correctness = 1; do_benchmark = 0; profile = 0;
        } else if (strcmp(argv[i], "--benchmark-only") == 0) {
            do_correctness = 0; do_benchmark = 1; profile = 0;
        } else if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            profile_variant = parse_variant(argv[++i]);
            do_correctness = 0; do_benchmark = 0; profile = 1;
        } else if (strcmp(argv[i], "--profile-shape") == 0 && i + 1 < argc) {
            const char *shape = argv[++i];
            if (strcmp(shape, "down") == 0) profile_gate_up = 0;
            else if (strcmp(shape, "gate-up") == 0) profile_gate_up = 1;
            else {
                fprintf(stderr, "error: invalid profile shape: %s\n", shape);
                return 2;
            }
        } else if (strcmp(argv[i], "--help") == 0 ||
                   strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    die_cuda(cudaSetDevice(device_id), "select CUDA device");
    cudaDeviceProp prop;
    die_cuda(cudaGetDeviceProperties(&prop, device_id), "query CUDA device");
    if (prop.major != 7 || prop.minor != 5) {
        fprintf(stderr, "error: %s is sm_%d%d; this harness requires sm_75\n",
                prop.name, prop.major, prop.minor);
        return 2;
    }
    if (!blocks) blocks = (uint32_t)prop.multiProcessorCount * 2u;
    if (!bench_cases) {
        const uint64_t target = (uint64_t)prop.l2CacheSize * 2u + 1u;
        bench_cases = (uint32_t)((target + sizeof(NativeQ3_32Tile) - 1u) /
                                 sizeof(NativeQ3_32Tile));
        if (!bench_cases) bench_cases = 1;
    }
    if ((do_benchmark || profile) &&
        (uint64_t)bench_cases * sizeof(NativeQ2Q3_50Tile) <=
            (uint64_t)prop.l2CacheSize) {
        fprintf(stderr,
            "error: --bench-cases=%u leaves the smallest format within "
            "L2 (%llu <= %d bytes)\n", bench_cases,
            (unsigned long long)((uint64_t)bench_cases *
                                 sizeof(NativeQ2Q3_50Tile)),
            prop.l2CacheSize);
        return 2;
    }
    const uint64_t work_per_kernel =
        (uint64_t)blocks * WARPS_PER_CTA * repeats;
    if (do_benchmark && work_per_kernel * launches < bench_cases) {
        fprintf(stderr,
            "error: benchmark traversal touches at most %llu weight cases but "
            "--bench-cases is %u; increase blocks, repeats, or launches\n",
            (unsigned long long)(work_per_kernel * launches), bench_cases);
        return 2;
    }
    if (profile && work_per_kernel < bench_cases) {
        fprintf(stderr,
            "error: profile traversal touches at most %llu weight cases but "
            "--bench-cases is %u; increase blocks or repeats\n",
            (unsigned long long)work_per_kernel, bench_cases);
        return 2;
    }
    const uint32_t allocated_weight_cases =
        (do_benchmark || profile) ? bench_cases : 1u;

    printf("device=%d\ndevice_name=%s\ncompute_capability=%d.%d\nsm_count=%d\n",
           device_id, prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("l2_bytes=%d\nthreads_per_cta=%d\nwarps_per_cta=%d\n"
           "tokens_per_warp=%d\n", prop.l2CacheSize, THREADS_PER_CTA,
           WARPS_PER_CTA, TILE_M);
    for (int v = 0; v < VARIANT_COUNT; v++)
        printf("format=%s canonical_block_bytes=%zu native_tile_bytes=%zu\n",
               variant_names[v], canonical_block_bytes((Variant)v),
               native_tile_bytes((Variant)v));

    ActivationCase *h_activations = (ActivationCase *)checked_malloc(
        sizeof(ActivationCase) * HOT_ACTIVATION_CASES, "activations");
    HostWeights host = {
        (NativeQ4KTile *)checked_malloc(
            sizeof(NativeQ4KTile) * (size_t)allocated_weight_cases, "Q4_K tiles"),
        (NativeQ3KTile *)checked_malloc(
            sizeof(NativeQ3KTile) * (size_t)allocated_weight_cases, "Q3_K tiles"),
        (NativeQ3_32Tile *)checked_malloc(
            sizeof(NativeQ3_32Tile) * (size_t)allocated_weight_cases, "Q3-32 tiles"),
        (NativeQ4_32Tile *)checked_malloc(
            sizeof(NativeQ4_32Tile) * (size_t)allocated_weight_cases, "Q4-32 tiles"),
        (NativeQ2Q3_50Tile *)checked_malloc(
            sizeof(NativeQ2Q3_50Tile) * (size_t)allocated_weight_cases,
            "Q2/Q3-50 tiles"),
        (NativeQ2Q3_75Tile *)checked_malloc(
            sizeof(NativeQ2Q3_75Tile) * (size_t)allocated_weight_cases,
            "Q2/Q3-75 tiles"),
        (NativeQ3A4Tile *)checked_malloc(
            sizeof(NativeQ3A4Tile) * (size_t)allocated_weight_cases,
            "Q3A4 tiles"),
        (NativeQ3A6Tile *)checked_malloc(
            sizeof(NativeQ3A6Tile) * (size_t)allocated_weight_cases,
            "Q3A6 tiles"),
        (NativeQ3Q4_50Tile *)checked_malloc(
            sizeof(NativeQ3Q4_50Tile) * (size_t)allocated_weight_cases,
            "Q3/Q4-50 tiles"),
        (NativeQ5_32Tile *)checked_malloc(
            sizeof(NativeQ5_32Tile) * (size_t)allocated_weight_cases,
            "Q5-32 tiles"),
    };
    ActivationCase *d_activations = NULL;
    DeviceWeights device = {};
    int32_t *d_exact = NULL;
    uint32_t *d_checksums = NULL;
    die_cuda(cudaMalloc(&d_activations,
                        sizeof(ActivationCase) * HOT_ACTIVATION_CASES),
             "allocate activations");
    die_cuda(cudaMalloc(&device.q4k, sizeof(NativeQ4KTile) *
                        (size_t)allocated_weight_cases), "allocate Q4_K tiles");
    die_cuda(cudaMalloc(&device.q3k, sizeof(NativeQ3KTile) *
                        (size_t)allocated_weight_cases), "allocate Q3_K tiles");
    die_cuda(cudaMalloc(&device.q3_32, sizeof(NativeQ3_32Tile) *
                        (size_t)allocated_weight_cases), "allocate Q3-32 tiles");
    die_cuda(cudaMalloc(&device.q4_32, sizeof(NativeQ4_32Tile) *
                        (size_t)allocated_weight_cases), "allocate Q4-32 tiles");
    die_cuda(cudaMalloc(&device.q2q3_50, sizeof(NativeQ2Q3_50Tile) *
                        (size_t)allocated_weight_cases),
             "allocate Q2/Q3-50 tiles");
    die_cuda(cudaMalloc(&device.q2q3_75, sizeof(NativeQ2Q3_75Tile) *
                        (size_t)allocated_weight_cases),
             "allocate Q2/Q3-75 tiles");
    die_cuda(cudaMalloc(&device.q3a4, sizeof(NativeQ3A4Tile) *
                        (size_t)allocated_weight_cases), "allocate Q3A4 tiles");
    die_cuda(cudaMalloc(&device.q3a6, sizeof(NativeQ3A6Tile) *
                        (size_t)allocated_weight_cases), "allocate Q3A6 tiles");
    die_cuda(cudaMalloc(&device.q3q4_50, sizeof(NativeQ3Q4_50Tile) *
                        (size_t)allocated_weight_cases),
             "allocate Q3/Q4-50 tiles");
    die_cuda(cudaMalloc(&device.q5_32, sizeof(NativeQ5_32Tile) *
                        (size_t)allocated_weight_cases), "allocate Q5-32 tiles");
    die_cuda(cudaMalloc(&d_exact, TILE_M * TILE_N * sizeof(int32_t)),
             "allocate exact output");
    die_cuda(cudaMalloc(&d_checksums, (uint64_t)blocks * WARPS_PER_CTA *
                        WARP_SIZE_ * sizeof(uint32_t)), "allocate checksums");

    int rc = 0;
    if (do_correctness)
        rc = run_correctness(cases, h_activations, host, d_activations,
                             device, d_exact, d_checksums);
    if (!rc && (do_benchmark || profile)) {
        rc = prepare_benchmark_cases(bench_cases, h_activations, host);
        if (!rc) {
            copy_benchmark_cases(bench_cases, h_activations, host,
                                 d_activations, device);
            printf("benchmark_layout_cases=%u\nbenchmark_layout_status=ok\n",
                   bench_cases);
        }
    }
    if (!rc && do_benchmark) {
        printf("benchmark_blocks=%u\nbenchmark_repeats=%u\n"
               "benchmark_launches=%u\nbenchmark_rounds=%u\n"
               "benchmark_streamed_cases=%u\n",
               blocks, repeats, launches, rounds, bench_cases);
        const uint32_t hot_weight_cases = bench_cases < HOT_WEIGHT_CASES
            ? bench_cases : HOT_WEIGHT_CASES;
        run_benchmark_mode("down-hot", 0, hot_weight_cases,
                           blocks, repeats, launches, rounds,
                           d_activations, device, d_exact, d_checksums);
        run_benchmark_mode("down-streamed", 0, bench_cases,
                           blocks, repeats, launches, rounds,
                           d_activations, device, d_exact, d_checksums);
        run_benchmark_mode("gate-up-hot", 1, hot_weight_cases,
                           blocks, repeats, launches, rounds,
                           d_activations, device, d_exact, d_checksums);
        run_benchmark_mode("gate-up-streamed", 1, bench_cases,
                           blocks, repeats,
                           launches, rounds, d_activations, device,
                           d_exact, d_checksums);
    }
    if (!rc && profile) {
        printf("profile_variant=%s\nprofile_shape=%s\nprofile_blocks=%u\n"
               "profile_repeats=%u\nprofile_weight_cases=%u\n",
               variant_names[profile_variant],
               profile_gate_up ? "gate-up" : "down",
               blocks, repeats, bench_cases);
        /* Deliberately exactly one launch for ncu/nsys process profiling. */
        if (profile_gate_up)
            launch_gate_up_variant(
                profile_variant, blocks, repeats, HOT_ACTIVATION_CASES,
                bench_cases, 0, d_activations, device,
                d_exact, d_checksums, 0);
        else
            launch_variant(profile_variant, blocks, repeats,
                           HOT_ACTIVATION_CASES, bench_cases, 0,
                           d_activations, device, d_exact, d_checksums, 0);
        die_cuda(cudaGetLastError(), "launch profiled kernel");
        die_cuda(cudaDeviceSynchronize(), "synchronize profiled kernel");
        printf("profile_status=ok\n");
    }

    cudaFree(d_checksums);
    cudaFree(d_exact);
    cudaFree(device.q5_32);
    cudaFree(device.q3q4_50);
    cudaFree(device.q3a6);
    cudaFree(device.q3a4);
    cudaFree(device.q2q3_75);
    cudaFree(device.q2q3_50);
    cudaFree(device.q4_32);
    cudaFree(device.q3_32);
    cudaFree(device.q3k);
    cudaFree(device.q4k);
    cudaFree(d_activations);
    free(host.q5_32);
    free(host.q3q4_50);
    free(host.q3a6);
    free(host.q3a4);
    free(host.q2q3_75);
    free(host.q2q3_50);
    free(host.q4_32);
    free(host.q3_32);
    free(host.q3k);
    free(host.q4k);
    free(h_activations);
    if (!rc) printf("harness_status=ok\n");
    return rc;
}
