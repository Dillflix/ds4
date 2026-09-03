/*
 * Bounded SM75 one-token Q8_0 weight-layout experiment.
 *
 * The shipping control is a source-level copy of
 * matmul_q8_0_preq_warp8_kernel.  The candidate changes the word-plane
 * representation of both weights and the already-quantized activation.
 * For every group of 32 canonical Q8_0 weight blocks it stores:
 *
 *     32 x fp16 scale                         64 bytes
 *      8 x (32 lanes x one packed int32)    1024 bytes
 *                                             ---------
 *                                             1088 bytes
 *
 * That is exactly 32 * 34 bytes.  Each warp lane still owns the same Q8_0
 * block, executes the same eight signed DP4A operations, accumulates blocks
 * in the same order, and participates in the same warp reduction.  Weight
 * and activation words become coalesced int32 planes; weight scales become a
 * contiguous half plane.
 *
 * This harness is deliberately disconnected from ds4's model-cache and
 * runtime dispatch.  It allocates equal-sized scratch representations,
 * proves both GPU repacks reversible, requires bit-exact control/candidate
 * output, and reports the one-time weight repack and per-token activation
 * repack separately from steady consumer timing.  Shipping behavior is
 * therefore unchanged.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kQk = 32u;
constexpr uint32_t kWordsPerBlock = 8u;
constexpr uint32_t kBlocksPerGroup = 32u;
constexpr uint32_t kWarpsPerCta = 8u;
constexpr uint32_t kThreads = 256u;
constexpr size_t kCanonicalBlockBytes = 34u;
constexpr size_t kFullGroupBytes = 1088u;
constexpr size_t kActivationFullGroupBytes = 1024u;
constexpr size_t kGuardBytes = 4096u;
constexpr unsigned char kGuardValue = 0xa5u;
constexpr unsigned char kPoisonValue = 0x5au;

static_assert(kFullGroupBytes ==
                  kBlocksPerGroup * kCanonicalBlockBytes,
              "interleaved group must be size-neutral");
static_assert(64u + 8u * 128u == kFullGroupBytes,
              "interleaved full-group planes must total 1088 bytes");
static_assert(kActivationFullGroupBytes == kBlocksPerGroup * kQk,
              "interleaved activation group must be size-neutral");

struct GuardedDeviceBuffer {
    unsigned char *allocation = nullptr;
    unsigned char *data = nullptr;
    size_t bytes = 0u;
};

enum Variant {
    kControl = 0,
    kInterleaved = 1,
    kInterleavedInclusive = 2,
    kVariantCount = 3,
};

const char *const kVariantNames[kVariantCount] = {
    "control-row-major-34b",
    "warp-interleaved-consumer",
    "warp-interleaved-plus-xq-pack",
};

struct Problem {
    uint32_t in_dim = 1024u;
    uint32_t out_dim = 32768u;
    uint32_t blocks = 32u;
    size_t row_bytes = 1088u;
    size_t weight_bytes = 0u;
    GuardedDeviceBuffer canonical_w;
    GuardedDeviceBuffer interleaved_w;
    GuardedDeviceBuffer xq;
    GuardedDeviceBuffer interleaved_xq;
    GuardedDeviceBuffer xscale;
    GuardedDeviceBuffer control_out;
    GuardedDeviceBuffer candidate_out;
    std::vector<unsigned char> host_canonical;
    std::vector<int8_t> host_xq;
    std::vector<float> host_xscale;
};

[[noreturn]] static void fail(const char *message) {
    std::fprintf(stderr, "error: %s\n", message);
    std::exit(2);
}

static void cuda_check(cudaError_t status, const char *what) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "error: %s: %s\n", what,
                 cudaGetErrorString(status));
    std::exit(2);
}

__device__ __forceinline__ int32_t load_i8x4_unaligned(
        const int8_t *ptr) {
    const uint8_t *u = reinterpret_cast<const uint8_t *>(ptr);
    return static_cast<int32_t>(
        static_cast<uint32_t>(u[0]) |
        (static_cast<uint32_t>(u[1]) << 8u) |
        (static_cast<uint32_t>(u[2]) << 16u) |
        (static_cast<uint32_t>(u[3]) << 24u));
}

__device__ __forceinline__ int32_t load_i8x4_aligned(
        const int8_t *ptr) {
    return *reinterpret_cast<const int32_t *>(ptr);
}

__device__ __forceinline__ void store_u32_unaligned(
        unsigned char *ptr, uint32_t value) {
    ptr[0] = static_cast<unsigned char>(value);
    ptr[1] = static_cast<unsigned char>(value >> 8u);
    ptr[2] = static_cast<unsigned char>(value >> 16u);
    ptr[3] = static_cast<unsigned char>(value >> 24u);
}

__device__ __forceinline__ int32_t dot_control(
        const int8_t *weight, const int8_t *activation, uint32_t n) {
    int32_t dot = 0;
    if (n == kQk) {
#pragma unroll
        for (uint32_t word = 0u; word < kWordsPerBlock; ++word) {
            dot = __dp4a(load_i8x4_unaligned(weight + 4u * word),
                         load_i8x4_aligned(activation + 4u * word), dot);
        }
        return dot;
    }
    for (uint32_t i = 0u; i < n; ++i)
        dot += static_cast<int32_t>(weight[i]) *
               static_cast<int32_t>(activation[i]);
    return dot;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

/* One CTA owns one matrix row.  The source is ordinary GGML Q8_0 row-major
 * storage.  The destination has exactly the same byte count. */
__global__ void sm75_q8_0_warp_interleave_pack_kernel(
        unsigned char *dst, const unsigned char *src,
        uint32_t blocks, uint32_t out_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (row >= out_dim || lane >= kBlocksPerGroup) return;
    const size_t row_bytes = static_cast<size_t>(blocks) *
                             kCanonicalBlockBytes;
    const unsigned char *src_row = src + static_cast<size_t>(row) * row_bytes;
    unsigned char *dst_row = dst + static_cast<size_t>(row) * row_bytes;
    const uint32_t groups = (blocks + kBlocksPerGroup - 1u) /
                            kBlocksPerGroup;

    for (uint32_t group = 0u; group < groups; ++group) {
        const uint32_t first = group * kBlocksPerGroup;
        const uint32_t remaining = blocks - first;
        const uint32_t count = remaining < kBlocksPerGroup
            ? remaining : kBlocksPerGroup;
        if (lane >= count) continue;
        const unsigned char *block =
            src_row + static_cast<size_t>(first + lane) *
                          kCanonicalBlockBytes;
        unsigned char *group_base =
            dst_row + static_cast<size_t>(first) * kCanonicalBlockBytes;
        unsigned char *scale_dst = group_base + 2u * lane;
        scale_dst[0] = block[0];
        scale_dst[1] = block[1];
        unsigned char *words = group_base + 2u * count;
#pragma unroll
        for (uint32_t word = 0u; word < kWordsPerBlock; ++word) {
            const uint32_t packed = static_cast<uint32_t>(
                load_i8x4_unaligned(
                    reinterpret_cast<const int8_t *>(block + 2u + 4u * word)));
            store_u32_unaligned(
                words + 4u * (static_cast<size_t>(word) * count + lane),
                packed);
        }
    }
}

/* The quantized activation receives the corresponding size-neutral word
 * planes.  A production implementation should emit this representation
 * directly from its quantizer; this bounded prototype times this extra pack
 * explicitly so consumer-only results cannot hide its cost. */
__global__ void sm75_q8_0_xq_warp_interleave_pack_kernel(
        unsigned char *dst, const int8_t *src, uint32_t blocks) {
    const uint32_t group = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t first = group * kBlocksPerGroup;
    if (first >= blocks) return;
    const uint32_t remaining = blocks - first;
    const uint32_t count = remaining < kBlocksPerGroup
        ? remaining : kBlocksPerGroup;
    if (lane >= count) return;
    const int8_t *source = src + static_cast<size_t>(first + lane) * kQk;
    unsigned char *group_base = dst + static_cast<size_t>(first) * kQk;
#pragma unroll
    for (uint32_t word = 0u; word < kWordsPerBlock; ++word) {
        const uint32_t packed = static_cast<uint32_t>(
            load_i8x4_aligned(source + 4u * word));
        store_u32_unaligned(
            group_base + 4u *
                (static_cast<size_t>(word) * count + lane),
            packed);
    }
}

__global__ void sm75_q8_0_cache_scrub_kernel(uint32_t *data,
                                              size_t words) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                   threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (; index < words; index += stride) {
        const uint32_t value = data[index];
        data[index] = value * 1664525u + 1013904223u +
                      static_cast<uint32_t>(index);
    }
}

__global__ void sm75_q8_0_output_poison_kernel(uint32_t *data,
                                               uint32_t count,
                                               uint32_t bits) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) data[index] = bits;
}

/* Source-level copy of shipping matmul_q8_0_preq_warp8_kernel for n_tok=1. */
__global__ void sm75_q8_0_preq_warp8_control_kernel(
        float *out, const unsigned char *weights,
        const int8_t *xq, const float *xscale,
        uint32_t in_dim, uint32_t out_dim, uint32_t blocks) {
    const uint32_t row = blockIdx.x * kWarpsPerCta +
                         (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const size_t row_bytes = static_cast<size_t>(blocks) *
                             kCanonicalBlockBytes;
    const unsigned char *weight_row =
        weights + static_cast<size_t>(row) * row_bytes;
    float acc = 0.0f;
    for (uint32_t block = lane; block < blocks;
         block += kBlocksPerGroup) {
        const uint32_t begin = block * kQk;
        const uint32_t remaining = in_dim - begin;
        const uint32_t count = remaining < kQk ? remaining : kQk;
        const unsigned char *wb =
            weight_row + static_cast<size_t>(block) *
                             kCanonicalBlockBytes;
        const __half scale = *reinterpret_cast<const __half *>(wb);
        const int32_t dot = dot_control(
            reinterpret_cast<const int8_t *>(wb + 2u),
            xq + static_cast<size_t>(block) * kQk, count);
        acc += __half2float(scale) * xscale[block] *
               static_cast<float>(dot);
    }
    acc = warp_sum(acc);
    if (lane == 0u) out[row] = acc;
}

__device__ __forceinline__ int32_t interleaved_weight_word(
        const unsigned char *words, uint32_t count,
        uint32_t lane, uint32_t word) {
    const unsigned char *ptr =
        words + 4u * (static_cast<size_t>(word) * count + lane);
    if ((reinterpret_cast<uintptr_t>(ptr) & 3u) == 0u)
        return *reinterpret_cast<const int32_t *>(ptr);
    return load_i8x4_unaligned(reinterpret_cast<const int8_t *>(ptr));
}

__device__ __forceinline__ int32_t dot_interleaved(
        const unsigned char *weight_words,
        const unsigned char *activation_words,
        uint32_t group_count, uint32_t lane, uint32_t n) {
    int32_t dot = 0;
    if (n == kQk) {
#pragma unroll
        for (uint32_t word = 0u; word < kWordsPerBlock; ++word) {
            dot = __dp4a(interleaved_weight_word(
                             weight_words, group_count, lane, word),
                         interleaved_weight_word(
                             activation_words, group_count, lane, word),
                         dot);
        }
        return dot;
    }
    for (uint32_t i = 0u; i < n; ++i) {
        const uint32_t weight_word = static_cast<uint32_t>(
            interleaved_weight_word(
                weight_words, group_count, lane, i >> 2u));
        const uint32_t activation_word = static_cast<uint32_t>(
            interleaved_weight_word(
                activation_words, group_count, lane, i >> 2u));
        const int8_t weight_value = static_cast<int8_t>(
            (weight_word >> (8u * (i & 3u))) & 0xffu);
        const int8_t activation_value = static_cast<int8_t>(
            (activation_word >> (8u * (i & 3u))) & 0xffu);
        dot += static_cast<int32_t>(weight_value) *
               static_cast<int32_t>(activation_value);
    }
    return dot;
}

__global__ void sm75_q8_0_preq_warp8_interleaved_kernel(
        float *out, const unsigned char *weights,
        const unsigned char *xq_words, const float *xscale,
        uint32_t in_dim, uint32_t out_dim, uint32_t blocks) {
    const uint32_t row = blockIdx.x * kWarpsPerCta +
                         (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const size_t row_bytes = static_cast<size_t>(blocks) *
                             kCanonicalBlockBytes;
    const unsigned char *weight_row =
        weights + static_cast<size_t>(row) * row_bytes;
    const uint32_t groups = (blocks + kBlocksPerGroup - 1u) /
                            kBlocksPerGroup;
    float acc = 0.0f;
    for (uint32_t group = 0u; group < groups; ++group) {
        const uint32_t first = group * kBlocksPerGroup;
        const uint32_t remaining_blocks = blocks - first;
        const uint32_t count = remaining_blocks < kBlocksPerGroup
            ? remaining_blocks : kBlocksPerGroup;
        if (lane >= count) continue;
        const uint32_t block = first + lane;
        const uint32_t begin = block * kQk;
        const uint32_t remaining_values = in_dim - begin;
        const uint32_t values = remaining_values < kQk
            ? remaining_values : kQk;
        const unsigned char *group_base =
            weight_row + static_cast<size_t>(first) *
                             kCanonicalBlockBytes;
        const __half scale = *reinterpret_cast<const __half *>(
            group_base + 2u * lane);
        const unsigned char *words = group_base + 2u * count;
        const unsigned char *activation_words =
            xq_words + static_cast<size_t>(first) * kQk;
        const int32_t dot = dot_interleaved(
            words, activation_words, count, lane, values);
        acc += __half2float(scale) * xscale[block] *
               static_cast<float>(dot);
    }
    acc = warp_sum(acc);
    if (lane == 0u) out[row] = acc;
}

static GuardedDeviceBuffer allocate_guarded(size_t bytes,
                                             const char *label) {
    if (bytes > SIZE_MAX - 2u * kGuardBytes) fail("allocation overflow");
    GuardedDeviceBuffer buffer;
    buffer.bytes = bytes;
    cudaError_t status = cudaMalloc(
        reinterpret_cast<void **>(&buffer.allocation),
        bytes + 2u * kGuardBytes);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "error: allocate %s (%zu bytes): %s\n", label,
                     bytes, cudaGetErrorString(status));
        std::exit(2);
    }
    buffer.data = buffer.allocation + kGuardBytes;
    cuda_check(cudaMemset(buffer.allocation, kGuardValue,
                          bytes + 2u * kGuardBytes),
               "initialize guarded allocation");
    return buffer;
}

static void release_guarded(GuardedDeviceBuffer *buffer) {
    if (buffer->allocation) cudaFree(buffer->allocation);
    *buffer = GuardedDeviceBuffer{};
}

static bool verify_guard(const GuardedDeviceBuffer &buffer,
                         const char *label) {
    std::vector<unsigned char> guard(2u * kGuardBytes);
    cuda_check(cudaMemcpy(guard.data(), buffer.allocation, kGuardBytes,
                          cudaMemcpyDeviceToHost),
               "read leading canary");
    cuda_check(cudaMemcpy(guard.data() + kGuardBytes,
                          buffer.data + buffer.bytes, kGuardBytes,
                          cudaMemcpyDeviceToHost),
               "read trailing canary");
    for (size_t i = 0u; i < guard.size(); ++i) {
        if (guard[i] != kGuardValue) {
            std::fprintf(stderr,
                         "error: %s canary mismatch at guard byte %zu\n",
                         label, i);
            return false;
        }
    }
    return true;
}

static uint32_t next_random(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return x;
}

static void initialize_weights(Problem *problem) {
    static const uint16_t scales[] = {
        0x2400u, 0x2800u, 0x2c00u, 0x3000u,
        0x3400u, 0x3600u, 0x3800u, 0x3a00u,
    };
    problem->host_canonical.resize(problem->weight_bytes);
    uint32_t random = 0x6d2b79f5u;
    for (uint32_t row = 0u; row < problem->out_dim; ++row) {
        unsigned char *row_ptr = problem->host_canonical.data() +
            static_cast<size_t>(row) * problem->row_bytes;
        for (uint32_t block = 0u; block < problem->blocks; ++block) {
            unsigned char *dst = row_ptr +
                static_cast<size_t>(block) * kCanonicalBlockBytes;
            const uint16_t scale = scales[(row * 17u + block * 5u) & 7u];
            std::memcpy(dst, &scale, sizeof(scale));
            for (uint32_t i = 0u; i < kQk; ++i) {
                const uint32_t value = next_random(&random);
                unsigned char byte = static_cast<unsigned char>(
                    static_cast<int8_t>(static_cast<int>(value % 255u) - 127));
                if (byte == kPoisonValue) byte = kPoisonValue - 1u;
                dst[2u + i] = byte;
            }
        }
    }
}

static void initialize_activation(Problem *problem, uint32_t fixture) {
    problem->host_xq.resize(static_cast<size_t>(problem->blocks) * kQk);
    problem->host_xscale.resize(problem->blocks);
    uint32_t random = 0x9e3779b9u ^ (fixture * 0x85ebca6bu);
    for (uint32_t block = 0u; block < problem->blocks; ++block) {
        problem->host_xscale[block] =
            static_cast<float>(1u + ((block * 3u + fixture) & 7u)) /
            256.0f;
        for (uint32_t i = 0u; i < kQk; ++i) {
            int value = 0;
            if (fixture == 0u) {
                value = static_cast<int>(next_random(&random) % 255u) - 127;
            } else if (fixture == 1u) {
                value = ((block + i) & 1u) ? 127 : -128;
            } else {
                const uint32_t selector = (block * kQk + i) % 17u;
                value = selector == 0u ? -128 :
                        (selector == 1u ? 127 :
                         (selector == 2u ? -1 :
                          (selector == 3u ? 1 : 0)));
            }
            if (static_cast<unsigned char>(value) == kPoisonValue)
                value = static_cast<int>(kPoisonValue) - 1;
            problem->host_xq[static_cast<size_t>(block) * kQk + i] =
                static_cast<int8_t>(value);
        }
    }
    cuda_check(cudaMemcpy(problem->xq.data, problem->host_xq.data(),
                          problem->host_xq.size(), cudaMemcpyHostToDevice),
               "copy quantized activation");
    cuda_check(cudaMemcpy(problem->xscale.data,
                          problem->host_xscale.data(),
                          problem->host_xscale.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               "copy activation scales");
}

static void launch_pack(Problem *problem, cudaStream_t stream = nullptr) {
    sm75_q8_0_warp_interleave_pack_kernel<<<
        problem->out_dim, kBlocksPerGroup, 0u, stream>>>(
            problem->interleaved_w.data, problem->canonical_w.data,
            problem->blocks, problem->out_dim);
    cuda_check(cudaGetLastError(), "launch Q8_0 interleave pack");
}

static void launch_xq_pack(Problem *problem,
                           cudaStream_t stream = nullptr) {
    const uint32_t groups =
        (problem->blocks + kBlocksPerGroup - 1u) / kBlocksPerGroup;
    sm75_q8_0_xq_warp_interleave_pack_kernel<<<
        groups, kBlocksPerGroup, 0u, stream>>>(
            problem->interleaved_xq.data,
            reinterpret_cast<const int8_t *>(problem->xq.data),
            problem->blocks);
    cuda_check(cudaGetLastError(), "launch Q8_0 activation interleave pack");
}

static void launch_variant(Problem *problem, Variant variant,
                           cudaStream_t stream = nullptr) {
    const uint32_t grid =
        (problem->out_dim + kWarpsPerCta - 1u) / kWarpsPerCta;
    if (variant == kControl) {
        sm75_q8_0_preq_warp8_control_kernel<<<grid, kThreads, 0u, stream>>>(
            reinterpret_cast<float *>(problem->control_out.data),
            problem->canonical_w.data,
            reinterpret_cast<const int8_t *>(problem->xq.data),
            reinterpret_cast<const float *>(problem->xscale.data),
            problem->in_dim, problem->out_dim, problem->blocks);
    } else {
        if (variant == kInterleavedInclusive)
            launch_xq_pack(problem, stream);
        sm75_q8_0_preq_warp8_interleaved_kernel<<<
            grid, kThreads, 0u, stream>>>(
                reinterpret_cast<float *>(problem->candidate_out.data),
                problem->interleaved_w.data,
                problem->interleaved_xq.data,
                reinterpret_cast<const float *>(problem->xscale.data),
                problem->in_dim, problem->out_dim, problem->blocks);
    }
    cuda_check(cudaGetLastError(), "launch Q8_0 consumer");
}

static bool verify_interleaved_mapping(const Problem &problem) {
    std::vector<unsigned char> packed(problem.weight_bytes);
    cuda_check(cudaMemcpy(packed.data(), problem.interleaved_w.data,
                          packed.size(), cudaMemcpyDeviceToHost),
               "read interleaved weights");
    if (std::find(packed.begin(), packed.end(), kPoisonValue) != packed.end()) {
        std::fprintf(stderr,
                     "error: weight repack left the poison byte in its output\n");
        return false;
    }
    for (uint32_t row = 0u; row < problem.out_dim; ++row) {
        const unsigned char *canonical = problem.host_canonical.data() +
            static_cast<size_t>(row) * problem.row_bytes;
        const unsigned char *interleaved = packed.data() +
            static_cast<size_t>(row) * problem.row_bytes;
        for (uint32_t first = 0u; first < problem.blocks;
             first += kBlocksPerGroup) {
            const uint32_t count = std::min(
                kBlocksPerGroup, problem.blocks - first);
            const unsigned char *group = interleaved +
                static_cast<size_t>(first) * kCanonicalBlockBytes;
            const unsigned char *words = group + 2u * count;
            for (uint32_t lane = 0u; lane < count; ++lane) {
                const unsigned char *source = canonical +
                    static_cast<size_t>(first + lane) *
                        kCanonicalBlockBytes;
                if (std::memcmp(source, group + 2u * lane, 2u) != 0) {
                    std::fprintf(stderr,
                                 "error: scale mapping mismatch row=%u "
                                 "block=%u\n", row, first + lane);
                    return false;
                }
                for (uint32_t word = 0u; word < kWordsPerBlock; ++word) {
                    const unsigned char *mapped = words + 4u *
                        (static_cast<size_t>(word) * count + lane);
                    if (std::memcmp(source + 2u + 4u * word,
                                    mapped, 4u) != 0) {
                        std::fprintf(stderr,
                                     "error: qword mapping mismatch row=%u "
                                     "block=%u word=%u\n", row,
                                     first + lane, word);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

static bool verify_interleaved_activation(const Problem &problem) {
    std::vector<unsigned char> packed(problem.host_xq.size());
    cuda_check(cudaMemcpy(packed.data(), problem.interleaved_xq.data,
                          packed.size(), cudaMemcpyDeviceToHost),
               "read interleaved activation");
    if (std::find(packed.begin(), packed.end(), kPoisonValue) != packed.end()) {
        std::fprintf(stderr,
                     "error: activation repack left the poison byte in its output\n");
        return false;
    }
    for (uint32_t first = 0u; first < problem.blocks;
         first += kBlocksPerGroup) {
        const uint32_t count = std::min(
            kBlocksPerGroup, problem.blocks - first);
        const unsigned char *group = packed.data() +
            static_cast<size_t>(first) * kQk;
        for (uint32_t lane = 0u; lane < count; ++lane) {
            const int8_t *source = problem.host_xq.data() +
                static_cast<size_t>(first + lane) * kQk;
            for (uint32_t word = 0u; word < kWordsPerBlock; ++word) {
                const unsigned char *mapped = group + 4u *
                    (static_cast<size_t>(word) * count + lane);
                if (std::memcmp(source + 4u * word, mapped, 4u) != 0) {
                    std::fprintf(stderr,
                                 "error: activation qword mapping mismatch "
                                 "block=%u word=%u\n", first + lane, word);
                    return false;
                }
            }
        }
    }
    return true;
}

static uint64_t checksum(const std::vector<float> &values) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (float value : values) {
        uint32_t bits = 0u;
        std::memcpy(&bits, &value, sizeof(bits));
        for (uint32_t byte = 0u; byte < 4u; ++byte) {
            hash ^= static_cast<unsigned char>(bits >> (8u * byte));
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

static bool compare_outputs(Problem *problem, uint32_t fixture,
                            bool print_success) {
    std::vector<float> control(problem->out_dim);
    std::vector<float> candidate(problem->out_dim);
    cuda_check(cudaMemcpy(control.data(), problem->control_out.data,
                          control.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "read control output");
    cuda_check(cudaMemcpy(candidate.data(), problem->candidate_out.data,
                          candidate.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "read candidate output");
    uint32_t nonzero_values = 0u;
    for (uint32_t i = 0u; i < problem->out_dim; ++i) {
        uint32_t control_bits = 0u;
        uint32_t candidate_bits = 0u;
        std::memcpy(&control_bits, &control[i], sizeof(control_bits));
        std::memcpy(&candidate_bits, &candidate[i], sizeof(candidate_bits));
        if (control_bits != candidate_bits || !std::isfinite(control[i])) {
            std::fprintf(stderr,
                         "error: fixture=%u output=%u control=%.9g/0x%08x "
                         "candidate=%.9g/0x%08x\n", fixture, i, control[i],
                         control_bits, candidate[i], candidate_bits);
            return false;
        }
        if ((control_bits & 0x7fffffffu) != 0u) ++nonzero_values;
    }
    if (nonzero_values == 0u) {
        std::fprintf(stderr,
                     "error: fixture=%u produced only exact-zero values\n",
                     fixture);
        return false;
    }
    if (print_success) {
        std::printf("correctness_fixture=%u\n"
                    "correctness_values=%u\n"
                    "correctness_nonzero_values=%u\n"
                    "correctness_checksum=0x%016llx\n"
                    "correctness_output_full_overwrite=pass\n"
                    "correctness_result=bit-exact\n",
                    fixture, problem->out_dim, nonzero_values,
                    static_cast<unsigned long long>(checksum(control)));
    }
    return true;
}

static bool run_correctness(Problem *problem) {
    std::printf("correctness_shape_begin=1\n"
                "correctness_in_dim=%u\n"
                "correctness_out_dim=%u\n"
                "correctness_blocks=%u\n"
                "correctness_weight_bytes=%zu\n",
                problem->in_dim, problem->out_dim, problem->blocks,
                problem->weight_bytes);
    cuda_check(cudaMemset(problem->interleaved_w.data, kPoisonValue,
                          problem->interleaved_w.bytes),
               "poison interleaved weights");
    launch_pack(problem);
    cuda_check(cudaDeviceSynchronize(), "synchronize weight repack");
    if (!verify_interleaved_mapping(*problem)) return false;
    std::printf("repack_roundtrip=byte-exact\n");
    for (uint32_t fixture = 0u; fixture < 3u; ++fixture) {
        initialize_activation(problem, fixture);
        cuda_check(cudaMemset(problem->interleaved_xq.data, kPoisonValue,
                              problem->interleaved_xq.bytes),
                   "poison interleaved activation");
        launch_xq_pack(problem);
        const uint32_t output_blocks =
            (problem->out_dim + kThreads - 1u) / kThreads;
        sm75_q8_0_output_poison_kernel<<<output_blocks, kThreads>>>(
            reinterpret_cast<uint32_t *>(problem->control_out.data),
            problem->out_dim, 0x7fa00001u);
        sm75_q8_0_output_poison_kernel<<<output_blocks, kThreads>>>(
            reinterpret_cast<uint32_t *>(problem->candidate_out.data),
            problem->out_dim, 0x7fa00002u);
        cuda_check(cudaGetLastError(), "poison output payloads");
        launch_variant(problem, kControl);
        launch_variant(problem, kInterleaved);
        cuda_check(cudaDeviceSynchronize(), "synchronize correctness");
        if (!verify_interleaved_activation(*problem)) return false;
        std::printf("activation_repack_roundtrip=byte-exact\n");
        if (!compare_outputs(problem, fixture, true)) return false;
    }
    const bool guards =
        verify_guard(problem->canonical_w, "canonical weights") &&
        verify_guard(problem->interleaved_w, "interleaved weights") &&
        verify_guard(problem->xq, "quantized activation") &&
        verify_guard(problem->interleaved_xq,
                     "interleaved quantized activation") &&
        verify_guard(problem->xscale, "activation scales") &&
        verify_guard(problem->control_out, "control output") &&
        verify_guard(problem->candidate_out, "candidate output");
    std::printf("correctness_canaries=%s\n"
                "correctness_shape_end=1\n",
                guards ? "ok" : "failed");
    return guards;
}

static int compare_double(const void *left, const void *right) {
    const double a = *reinterpret_cast<const double *>(left);
    const double b = *reinterpret_cast<const double *>(right);
    return (a > b) - (a < b);
}

static double median(std::vector<double> values) {
    std::qsort(values.data(), values.size(), sizeof(double), compare_double);
    const size_t mid = values.size() / 2u;
    return (values.size() & 1u) ? values[mid]
                               : 0.5 * (values[mid - 1u] + values[mid]);
}

struct TimingResult {
    std::vector<double> samples[kVariantCount];
    double median_ms[kVariantCount] = {};
};

struct RatioStats {
    double median = 0.0;
    double sd = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

static RatioStats paired_ratio_stats(const std::vector<double> &control,
                                     const std::vector<double> &candidate) {
    std::vector<double> ratios(control.size());
    double mean = 0.0;
    for (size_t i = 0u; i < control.size(); ++i) {
        ratios[i] = control[i] / candidate[i];
        mean += ratios[i];
    }
    mean /= static_cast<double>(ratios.size());
    double sum_square = 0.0;
    for (double ratio : ratios) {
        const double delta = ratio - mean;
        sum_square += delta * delta;
    }
    RatioStats result;
    result.median = median(ratios);
    result.sd = ratios.size() > 1u
        ? std::sqrt(sum_square / static_cast<double>(ratios.size() - 1u))
        : 0.0;
    const auto bounds = std::minmax_element(ratios.begin(), ratios.end());
    result.minimum = *bounds.first;
    result.maximum = *bounds.second;
    return result;
}

static void launch_cache_scrub(GuardedDeviceBuffer *buffer) {
    const size_t words = buffer->bytes / sizeof(uint32_t);
    const uint32_t blocks = static_cast<uint32_t>(std::min<size_t>(
        65535u, (words + kThreads - 1u) / kThreads));
    sm75_q8_0_cache_scrub_kernel<<<blocks, kThreads>>>(
        reinterpret_cast<uint32_t *>(buffer->data), words);
    cuda_check(cudaGetLastError(), "launch cache scrub");
}

static TimingResult run_timing_surface(
        Problem *problem, GuardedDeviceBuffer *cache_scrub,
        uint32_t rounds, uint32_t launches, bool cold) {
    TimingResult result;
    for (uint32_t variant = 0u; variant < kVariantCount; ++variant)
        result.samples[variant].resize(rounds);
    const char *cache_state = cold ? "scrubbed" : "warm";
    std::printf("timing_samples_begin=%s\n"
                "cache_state,round,sample_slot,variant,total_ms,us_per_launch\n",
                cache_state);
    for (uint32_t round = 0u; round < rounds; ++round) {
        Variant order[kVariantCount];
        const uint32_t start_variant = round % kVariantCount;
        for (uint32_t slot = 0u; slot < kVariantCount; ++slot) {
            const uint32_t step = (round & 1u)
                ? (kVariantCount - slot) % kVariantCount : slot;
            order[slot] = static_cast<Variant>(
                (start_variant + step) % kVariantCount);
            const Variant variant = order[slot];
            if (cold) {
                launch_cache_scrub(cache_scrub);
                cuda_check(cudaDeviceSynchronize(),
                           "synchronize cache scrub");
            }
            cudaEvent_t start = nullptr;
            cudaEvent_t stop = nullptr;
            cuda_check(cudaEventCreate(&start), "create timing start event");
            cuda_check(cudaEventCreate(&stop), "create timing stop event");
            cuda_check(cudaEventRecord(start), "record timing start");
            for (uint32_t launch = 0u; launch < launches; ++launch)
                launch_variant(problem, variant);
            cuda_check(cudaEventRecord(stop), "record timing stop");
            cuda_check(cudaEventSynchronize(stop),
                       "synchronize timing sample");
            float elapsed_ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop),
                       "measure timing sample");
            cudaEventDestroy(stop);
            cudaEventDestroy(start);
            result.samples[variant][round] = elapsed_ms;
            std::printf("%s,%u,%u,%s,%.6f,%.6f\n", cache_state,
                        round + 1u, slot + 1u, kVariantNames[variant],
                        static_cast<double>(elapsed_ms),
                        1000.0 * static_cast<double>(elapsed_ms) / launches);
        }
    }
    std::printf("timing_samples_end=%s\n", cache_state);
    for (uint32_t variant = 0u; variant < kVariantCount; ++variant)
        result.median_ms[variant] = median(result.samples[variant]);
    std::printf("timing_summary_begin=%s\n"
                "cache_state,variant,median_total_ms,median_us_per_launch,relative_speed\n",
                cache_state);
    for (uint32_t variant = 0u; variant < kVariantCount; ++variant) {
        std::printf("%s,%s,%.6f,%.6f,%.6f\n", cache_state,
                    kVariantNames[variant], result.median_ms[variant],
                    1000.0 * result.median_ms[variant] / launches,
                    result.median_ms[kControl] /
                        result.median_ms[variant]);
    }
    std::printf("timing_summary_end=%s\n", cache_state);
    for (uint32_t variant = kInterleaved;
         variant < kVariantCount; ++variant) {
        const RatioStats stats = paired_ratio_stats(
            result.samples[kControl], result.samples[variant]);
        std::printf("paired_speedup_cache_%s_variant_%s_median=%.6f\n"
                    "paired_speedup_cache_%s_variant_%s_sd=%.6f\n"
                    "paired_speedup_cache_%s_variant_%s_min=%.6f\n"
                    "paired_speedup_cache_%s_variant_%s_max=%.6f\n",
                    cache_state, kVariantNames[variant], stats.median,
                    cache_state, kVariantNames[variant], stats.sd,
                    cache_state, kVariantNames[variant], stats.minimum,
                    cache_state, kVariantNames[variant], stats.maximum);
    }
    return result;
}

static bool run_benchmark(Problem *problem, uint32_t rounds,
                          uint32_t launches) {
    initialize_activation(problem, 0u);
    launch_pack(problem);
    launch_xq_pack(problem);
    launch_variant(problem, kControl);
    launch_variant(problem, kInterleaved);
    launch_variant(problem, kInterleavedInclusive);
    cuda_check(cudaDeviceSynchronize(), "synchronize benchmark warmup");

    GuardedDeviceBuffer cache_scrub = allocate_guarded(
        64u * 1024u * 1024u, "cache scrub surface");
    cuda_check(cudaMemset(cache_scrub.data, 0x3c, cache_scrub.bytes),
               "initialize cache scrub surface");

    std::printf("benchmark_scope=single-token-prequant-consumer\n"
                "activation_quantization_common_and_excluded=1\n"
                "weight_repack_one_time_and_excluded_from_consumers=1\n"
                "activation_repack_included_only_in_plus-xq-pack=1\n"
                "warm_rounds=%u\n"
                "warm_launches_per_sample=%u\n"
                "scrubbed_rounds=%u\n"
                "scrubbed_launches_per_sample=1\n"
                "cache_scrub_bytes=%zu\n",
                rounds, launches, rounds, cache_scrub.bytes);
    const TimingResult warm = run_timing_surface(
        problem, &cache_scrub, rounds, launches, false);
    const TimingResult cold = run_timing_surface(
        problem, &cache_scrub, rounds, 1u, true);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuda_check(cudaEventCreate(&start), "create repack start event");
    cuda_check(cudaEventCreate(&stop), "create repack stop event");
    std::vector<double> weight_pack_samples(rounds);
    std::vector<double> xq_pack_samples(rounds);
    for (uint32_t round = 0u; round < rounds; ++round) {
        cuda_check(cudaEventRecord(start), "record weight pack start");
        launch_pack(problem);
        cuda_check(cudaEventRecord(stop), "record weight pack stop");
        cuda_check(cudaEventSynchronize(stop), "synchronize weight pack");
        float elapsed_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop),
                   "measure weight pack");
        weight_pack_samples[round] = elapsed_ms;
        cuda_check(cudaEventRecord(start), "record xq pack start");
        launch_xq_pack(problem);
        cuda_check(cudaEventRecord(stop), "record xq pack stop");
        cuda_check(cudaEventSynchronize(stop), "synchronize xq pack");
        cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop),
                   "measure xq pack");
        xq_pack_samples[round] = elapsed_ms;
    }
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    const double weight_pack_ms = median(weight_pack_samples);
    const double xq_pack_ms = median(xq_pack_samples);
    const double control_us = 1000.0 * warm.median_ms[kControl] / launches;
    const double candidate_us =
        1000.0 * warm.median_ms[kInterleaved] / launches;
    const double saved_us = control_us - candidate_us;
    const double break_even = saved_us > 0.0
        ? 1000.0 * weight_pack_ms / saved_us : -1.0;
    const double pack_gib_s =
        (2.0 * static_cast<double>(problem->weight_bytes) /
         (1024.0 * 1024.0 * 1024.0)) / (weight_pack_ms / 1000.0);
    std::printf("weight_repack_median_ms=%.6f\n"
                "weight_repack_nominal_read_write_gib_per_s=%.6f\n"
                "weight_repack_break_even_decode_launches=%.6f\n"
                "xq_repack_median_us=%.6f\n"
                "inclusive_candidate_warm_speedup=%.6f\n"
                "inclusive_candidate_scrubbed_speedup=%.6f\n",
                weight_pack_ms, pack_gib_s, break_even,
                1000.0 * xq_pack_ms,
                warm.median_ms[kControl] /
                    warm.median_ms[kInterleavedInclusive],
                cold.median_ms[kControl] /
                    cold.median_ms[kInterleavedInclusive]);

    launch_variant(problem, kControl);
    launch_variant(problem, kInterleavedInclusive);
    cuda_check(cudaDeviceSynchronize(), "synchronize post-timing exactness");
    const bool exact = compare_outputs(problem, 0u, false);
    const bool guards =
        verify_guard(problem->canonical_w, "canonical weights") &&
        verify_guard(problem->interleaved_w, "interleaved weights") &&
        verify_guard(problem->xq, "quantized activation") &&
        verify_guard(problem->interleaved_xq,
                     "interleaved quantized activation") &&
        verify_guard(problem->xscale, "activation scales") &&
        verify_guard(problem->control_out, "control output") &&
        verify_guard(problem->candidate_out, "candidate output") &&
        verify_guard(cache_scrub, "cache scrub surface");
    std::printf("post_timing_exactness=%s\n"
                "post_timing_canaries=%s\n"
                "benchmark_status=%s\n",
                exact ? "bit-exact" : "failed",
                guards ? "ok" : "failed",
                exact && guards ? "ok" : "failed");
    release_guarded(&cache_scrub);
    return exact && guards;
}

template <typename Kernel>
static void print_resource(const char *name, Kernel kernel,
                           uint32_t threads) {
    cudaFuncAttributes attributes{};
    cuda_check(cudaFuncGetAttributes(&attributes, kernel),
               "query kernel resources");
    int active_blocks = 0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &active_blocks, kernel, threads, 0u),
               "query kernel occupancy");
    std::printf("resource_%s_registers_per_thread=%d\n"
                "resource_%s_static_shared_bytes=%zu\n"
                "resource_%s_local_bytes_per_thread=%zu\n"
                "resource_%s_active_blocks_per_sm=%d\n",
                name, attributes.numRegs,
                name, attributes.sharedSizeBytes,
                name, attributes.localSizeBytes,
                name, active_blocks);
}

static void print_resources() {
    print_resource("control", sm75_q8_0_preq_warp8_control_kernel,
                   kThreads);
    print_resource("interleaved",
                   sm75_q8_0_preq_warp8_interleaved_kernel, kThreads);
    print_resource("repack", sm75_q8_0_warp_interleave_pack_kernel,
                   kBlocksPerGroup);
    print_resource("xq_repack", sm75_q8_0_xq_warp_interleave_pack_kernel,
                   kBlocksPerGroup);
}

static bool run_profile(Problem *problem, Variant variant,
                        uint32_t launches) {
    initialize_activation(problem, 0u);
    launch_pack(problem);
    launch_xq_pack(problem);
    launch_variant(problem, variant);
    cuda_check(cudaDeviceSynchronize(), "synchronize profile warmup");
    std::printf("profile_variant=%s\n"
                "profile_warmup_launches=1\n"
                "profile_capture_launches=%u\n"
                "profile_capture_begin=1\n",
                kVariantNames[variant], launches);
    std::fflush(stdout);
    for (uint32_t launch = 0u; launch < launches; ++launch)
        launch_variant(problem, variant);
    cuda_check(cudaDeviceSynchronize(), "synchronize profile capture");
    std::printf("profile_capture_end=1\nprofile_status=ok\n");
    return true;
}

static Problem setup_problem(uint32_t in_dim, uint32_t out_dim) {
    Problem problem;
    problem.in_dim = in_dim;
    problem.out_dim = out_dim;
    problem.blocks = (problem.in_dim + kQk - 1u) / kQk;
    problem.row_bytes = static_cast<size_t>(problem.blocks) *
                        kCanonicalBlockBytes;
    if (problem.out_dim > SIZE_MAX / problem.row_bytes)
        fail("weight byte count overflow");
    problem.weight_bytes =
        static_cast<size_t>(problem.out_dim) * problem.row_bytes;
    problem.canonical_w = allocate_guarded(problem.weight_bytes,
                                            "canonical weights");
    problem.interleaved_w = allocate_guarded(problem.weight_bytes,
                                              "interleaved weights");
    problem.xq = allocate_guarded(
        static_cast<size_t>(problem.blocks) * kQk, "quantized activation");
    problem.interleaved_xq = allocate_guarded(
        static_cast<size_t>(problem.blocks) * kQk,
        "interleaved quantized activation");
    problem.xscale = allocate_guarded(
        static_cast<size_t>(problem.blocks) * sizeof(float),
        "activation scales");
    problem.control_out = allocate_guarded(
        static_cast<size_t>(problem.out_dim) * sizeof(float),
        "control output");
    problem.candidate_out = allocate_guarded(
        static_cast<size_t>(problem.out_dim) * sizeof(float),
        "candidate output");
    initialize_weights(&problem);
    cuda_check(cudaMemcpy(problem.canonical_w.data,
                          problem.host_canonical.data(),
                          problem.host_canonical.size(),
                          cudaMemcpyHostToDevice),
               "copy canonical weights");
    return problem;
}

static void cleanup_problem(Problem *problem) {
    release_guarded(&problem->candidate_out);
    release_guarded(&problem->control_out);
    release_guarded(&problem->xscale);
    release_guarded(&problem->interleaved_xq);
    release_guarded(&problem->xq);
    release_guarded(&problem->interleaved_w);
    release_guarded(&problem->canonical_w);
}

static uint32_t parse_u32(const char *text, const char *name) {
    errno = 0;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (errno || !end || *end || value == 0u || value > UINT32_MAX) {
        std::fprintf(stderr, "error: invalid %s: %s\n", name, text);
        std::exit(2);
    }
    return static_cast<uint32_t>(value);
}

static int parse_device(const char *text) {
    errno = 0;
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno || !end || *end || value < 0 || value > INT32_MAX) {
        std::fprintf(stderr, "error: invalid device: %s\n", text);
        std::exit(2);
    }
    return static_cast<int>(value);
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
        "Usage: %s [--device N] MODE [OPTIONS]\n\n"
        "Modes:\n"
        "  --correctness-only\n"
        "  --benchmark-only [--rounds N] [--launches N]\n"
        "  --profile control|interleaved [--launches N]\n\n"
        "With no mode, runs correctness and the paired benchmark. The fixed\n"
        "shape is the production one-token T32 projection: 1024 x 32768.\n",
        argv0);
}

}  // namespace

int main(int argc, char **argv) {
    enum Mode { kAll, kCorrectnessOnly, kBenchmarkOnly, kProfile };
    Mode mode = kAll;
    bool mode_seen = false;
    int device = 0;
    uint32_t rounds = 14u;
    uint32_t launches = 100u;
    Variant profile_variant = kControl;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--device") && i + 1 < argc) {
            device = parse_device(argv[++i]);
        } else if (!std::strcmp(argv[i], "--correctness-only")) {
            if (mode_seen) { usage(argv[0]); return 2; }
            mode_seen = true;
            mode = kCorrectnessOnly;
        } else if (!std::strcmp(argv[i], "--benchmark-only")) {
            if (mode_seen) { usage(argv[0]); return 2; }
            mode_seen = true;
            mode = kBenchmarkOnly;
        } else if (!std::strcmp(argv[i], "--profile") && i + 1 < argc) {
            if (mode_seen) { usage(argv[0]); return 2; }
            mode_seen = true;
            mode = kProfile;
            const char *variant = argv[++i];
            if (!std::strcmp(variant, "control")) {
                profile_variant = kControl;
            } else if (!std::strcmp(variant, "interleaved")) {
                profile_variant = kInterleaved;
            } else {
                usage(argv[0]);
                return 2;
            }
        } else if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) {
            rounds = parse_u32(argv[++i], "rounds");
        } else if (!std::strcmp(argv[i], "--launches") && i + 1 < argc) {
            launches = parse_u32(argv[++i], "launches");
        } else if (!std::strcmp(argv[i], "--help") ||
                   !std::strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    cuda_check(cudaSetDevice(device), "select CUDA device");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device),
               "query CUDA device");
    if (properties.major != 7 || properties.minor != 5) {
        std::fprintf(stderr,
                     "error: SM75 harness requires compute capability 7.5; "
                     "got %d.%d\n", properties.major, properties.minor);
        return 1;
    }
    std::printf("harness=sm75-q8-warp-interleaved\n"
                "device=%d\n"
                "device_name=%s\n"
                "compute_capability=%d.%d\n"
                "benchmark_projection_role=single-token-t32\n"
                "benchmark_in_dim=1024\n"
                "benchmark_out_dim=32768\n"
                "benchmark_q8_blocks_per_row=32\n"
                "canonical_bytes_per_32_blocks=%zu\n"
                "interleaved_bytes_per_32_blocks=%zu\n"
                "size_neutral=yes\n"
                "shipping_dispatch_modified=no\n",
                device, properties.name, properties.major, properties.minor,
                kBlocksPerGroup * kCanonicalBlockBytes, kFullGroupBytes);
    print_resources();

    bool ok = true;
    if (mode == kCorrectnessOnly || mode == kAll) {
        struct Shape { uint32_t in_dim; uint32_t out_dim; };
        const Shape shapes[] = {
            {1024u, 32768u}, /* production T32, one full interleave group */
            {2048u, 4096u},  /* two full interleave groups */
            {1055u, 259u},   /* 33 blocks: partial block/group, 2B row skew */
        };
        for (const Shape &shape : shapes) {
            Problem problem = setup_problem(shape.in_dim, shape.out_dim);
            ok = run_correctness(&problem);
            cleanup_problem(&problem);
            if (!ok) break;
        }
    }
    if (ok && (mode == kBenchmarkOnly || mode == kAll)) {
        Problem problem = setup_problem(1024u, 32768u);
        ok = run_benchmark(&problem, rounds, launches);
        cleanup_problem(&problem);
    }
    if (ok && mode == kProfile) {
        Problem problem = setup_problem(1024u, 32768u);
        ok = run_profile(&problem, profile_variant, launches);
        cleanup_problem(&problem);
    }
    if (ok) std::printf("harness_status=ok\n");
    return ok ? 0 : 1;
}
