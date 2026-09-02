/*
 * Bounded exact-codec experiment for DeepSeek V4 Flash compressed-attention
 * KV rows.  Production currently stores the already-E4M3-rounded 448 non-RoPE
 * values and the untouched 64 RoPE values as 512 floats (2048 bytes).  This
 * harness tests a 736-byte row without changing production allocation or
 * dispatch:
 *
 *   7 exact power-of-two F32 scales
 *   448 E4M3 sign/index bytes
 *   64 untouched F32 RoPE values
 *
 * The packer consumes the existing rounded F32 representation and accepts a
 * row only when unpacking is bit-identical.  Non-finite and non-representable
 * rows fail closed.  The consumer A/B includes compact decoding in its timing.
 */

#include <cuda_runtime.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

enum {
    HEAD_DIM = 512,
    N_ROT = 64,
    N_NOPE = HEAD_DIM - N_ROT,
    GROUP = 64,
    N_SCALE = N_NOPE / GROUP,
    THREADS = 64,
    GUARD_BYTES = 256,
    CANARY = 0xa5,
};

struct __align__(32) CompactAttentionKVRow {
    float scale[N_SCALE];
    uint32_t reserved;
    uint8_t code[N_NOPE];
    float rope_f32[N_ROT];
};

static_assert(N_SCALE == 7, "DeepSeek V4 Flash compact KV needs seven scales");
static_assert(sizeof(CompactAttentionKVRow) == 736,
              "compact compressed-attention KV row must be 736 bytes");
static_assert(alignof(CompactAttentionKVRow) == 32,
              "compact compressed-attention KV row must be 32-byte aligned");

enum PackStatus {
    PACK_OK = 0,
    PACK_NONFINITE = 1,
    PACK_UNREPRESENTABLE = 2,
    PACK_BAD_SCALE = 4,
};

static void cuda_die(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return;
    fprintf(stderr, "error: %s: %s\n", what, cudaGetErrorString(err));
    exit(2);
}

__host__ __device__ __forceinline__ static float e4m3fn_value(uint32_t i) {
    const uint32_t exp = (i >> 3) & 15u;
    const uint32_t mant = i & 7u;
    if (exp == 0u) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

__device__ __forceinline__ static float decode_code(uint8_t code, float scale) {
    float value = e4m3fn_value((uint32_t)(code & 0x7fu)) * scale;
    uint32_t bits = __float_as_uint(value);
    bits |= (uint32_t)(code & 0x80u) << 24;
    return __uint_as_float(bits);
}

__device__ __forceinline__ static int exact_code_for(float value, float scale) {
    if (!isfinite(value) || !isfinite(scale) || !(scale > 0.0f)) return -1;
    const uint32_t sign = __float_as_uint(value) >> 31;
    const uint32_t magnitude_bits = __float_as_uint(value) & 0x7fffffffu;
    for (uint32_t i = 0; i <= 126u; i++) {
        const float reconstructed = e4m3fn_value(i) * scale;
        if (__float_as_uint(reconstructed) == magnitude_bits) {
            return (int)(i | (sign << 7));
        }
    }
    return -1;
}

__global__ static void compact_pack_kernel(
        const float *rows,
        CompactAttentionKVRow *compact,
        uint32_t *status,
        uint32_t n_rows) {
    /* Status is produced asynchronously with the row. A future caller must
     * observe completion and PACK_OK before committing or consuming dst; a
     * rejected row may contain partial diagnostic output. */
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (row >= n_rows) return;

    __shared__ float scratch[GROUP];
    CompactAttentionKVRow *dst = compact + row;
    const float *src = rows + (uint64_t)row * HEAD_DIM;
    if (tid == 0u) dst->reserved = 0u;

    for (uint32_t g = 0; g < N_SCALE; g++) {
        const uint32_t d = g * GROUP + tid;
        const float v = src[d];
        if (!isfinite(v)) atomicOr(status + row, (uint32_t)PACK_NONFINITE);
        scratch[tid] = isfinite(v) ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            }
            __syncthreads();
        }
        const float scale = exp2f(ceilf(log2f(
            fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (tid == 0u) {
            dst->scale[g] = scale;
            if (!isfinite(scale) || !(scale > 0.0f)) {
                atomicOr(status + row, (uint32_t)PACK_BAD_SCALE);
            }
        }
        __syncthreads();
        const int code = exact_code_for(v, scale);
        if (code < 0) {
            atomicOr(status + row, (uint32_t)PACK_UNREPRESENTABLE);
            dst->code[d] = 0x7fu;
        } else {
            dst->code[d] = (uint8_t)code;
        }
        __syncthreads();
    }

    for (uint32_t d = tid; d < N_ROT; d += blockDim.x) {
        const float value = src[N_NOPE + d];
        if (!isfinite(value)) {
            atomicOr(status + row, (uint32_t)PACK_NONFINITE);
        }
        dst->rope_f32[d] = value;
    }
}

__global__ static void compact_unpack_kernel(
        const CompactAttentionKVRow *compact,
        float *rows,
        uint32_t n_rows) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (row >= n_rows) return;
    const CompactAttentionKVRow *src = compact + row;
    float *dst = rows + (uint64_t)row * HEAD_DIM;
    for (uint32_t d = tid; d < N_NOPE; d += blockDim.x) {
        dst[d] = decode_code(src->code[d], src->scale[d / GROUP]);
    }
    for (uint32_t d = tid; d < N_ROT; d += blockDim.x) {
        dst[N_NOPE + d] = src->rope_f32[d];
    }
}

__global__ static void f32_consumer_kernel(
        const float *rows, float *out, uint32_t n_rows) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= HEAD_DIM) return;
    float acc = 0.0f;
    for (uint32_t row = 0; row < n_rows; row++) {
        const float weight = (float)((row & 15u) + 1u) * 0.0009765625f;
        acc = __fmaf_rn(rows[(uint64_t)row * HEAD_DIM + d], weight, acc);
    }
    out[d] = acc;
}

__global__ static void compact_consumer_kernel(
        const CompactAttentionKVRow *rows, float *out, uint32_t n_rows) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= HEAD_DIM) return;
    float acc = 0.0f;
    for (uint32_t row = 0; row < n_rows; row++) {
        const CompactAttentionKVRow *src = rows + row;
        const float value = d < N_NOPE
            ? decode_code(src->code[d], src->scale[d / GROUP])
            : src->rope_f32[d - N_NOPE];
        const float weight = (float)((row & 15u) + 1u) * 0.0009765625f;
        acc = __fmaf_rn(value, weight, acc);
    }
    out[d] = acc;
}

static uint32_t rng_state = 0x6d2b79f5u;

static uint32_t rng_u32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static float signed_unit(void) {
    const float magnitude = (float)(rng_u32() & 0x00ffffffu) / 16777215.0f;
    return (rng_u32() & 1u) ? magnitude : -magnitude;
}

static void fill_finite_input(float *rows, uint32_t n_rows) {
    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t g = 0; g < N_SCALE; g++) {
            const int exponent = -18 + (int)((row * 11u + g * 7u) % 80u);
            const float scale = ldexpf(1.0f, exponent);
            for (uint32_t j = 0; j < GROUP; j++) {
                rows[(uint64_t)row * HEAD_DIM + g * GROUP + j] =
                    signed_unit() * 448.0f * scale;
            }
        }
        for (uint32_t d = 0; d < N_ROT; d++) {
            uint32_t bits = rng_u32();
            bits = (bits & 0x807fffffu) | (((bits >> 23) % 240u) << 23);
            float value;
            memcpy(&value, &bits, sizeof(value));
            rows[(uint64_t)row * HEAD_DIM + N_NOPE + d] = value;
        }
    }

    if (n_rows > 0u) {
        memset(rows, 0, HEAD_DIM * sizeof(float));
        for (uint32_t d = 1; d < N_NOPE; d += 2) {
            const uint32_t neg_zero = 0x80000000u;
            memcpy(rows + d, &neg_zero, sizeof(neg_zero));
        }
        for (uint32_t d = 1; d < N_ROT; d += 2) {
            const uint32_t neg_zero = 0x80000000u;
            memcpy(rows + N_NOPE + d, &neg_zero, sizeof(neg_zero));
        }
    }
    if (n_rows > 1u) {
        float *row = rows + HEAD_DIM;
        for (uint32_t g = 0; g < N_SCALE; g++) {
            const float scale = ldexpf(1.0f, (int)g - 12);
            row[g * GROUP] = 448.0f * scale;
            row[g * GROUP + 1u] = -448.0f * scale;
            for (uint32_t j = 2; j < GROUP; j++) {
                const uint32_t i = 1u + ((j * 13u) % 124u);
                row[g * GROUP + j] =
                    0.5f * (e4m3fn_value(i) + e4m3fn_value(i + 1u)) *
                    scale * ((j & 1u) ? -1.0f : 1.0f);
            }
        }
    }
    if (n_rows > 2u) {
        float *row = rows + 2u * HEAD_DIM;
        for (uint32_t g = 0; g < N_SCALE; g++) {
            memset(row + g * GROUP, 0, GROUP * sizeof(float));
            const float scale = ldexpf(1.0f, 119 - (int)(g % 3u));
            row[g * GROUP] = 448.0f * scale;
            row[g * GROUP + 1u] = -448.0f * scale;
            row[g * GROUP + 2u] = FLT_MIN;
            row[g * GROUP + 3u] = -FLT_MIN;
        }
    }
    if (n_rows > 3u) {
        float *row = rows + 3u * HEAD_DIM;
        for (uint32_t g = 0; g < N_SCALE; g++) {
            memset(row + g * GROUP, 0, GROUP * sizeof(float));
            const float original_scale = ldexpf(1.0f, (int)g - 9);
            row[g * GROUP] = 225.0f * original_scale;
            row[g * GROUP + 1u] = -225.0f * original_scale;
        }
    }
}

struct GuardedDeviceBuffer {
    uint8_t *base;
    void *data;
    size_t data_bytes;
};

static GuardedDeviceBuffer guarded_alloc(size_t bytes, const char *what) {
    GuardedDeviceBuffer buffer = {NULL, NULL, bytes};
    cuda_die(cudaMalloc((void **)&buffer.base, bytes + 2u * GUARD_BYTES), what);
    cuda_die(cudaMemset(buffer.base, CANARY, bytes + 2u * GUARD_BYTES),
             "initialize guarded allocation");
    buffer.data = buffer.base + GUARD_BYTES;
    return buffer;
}

static void check_guards(const GuardedDeviceBuffer *buffer, const char *what) {
    uint8_t guard[GUARD_BYTES];
    cuda_die(cudaMemcpy(guard, buffer->base, GUARD_BYTES, cudaMemcpyDeviceToHost),
             "copy leading guard");
    for (uint32_t i = 0; i < GUARD_BYTES; i++) {
        if (guard[i] != CANARY) {
            fprintf(stderr, "error: %s leading canary changed at byte %u\n", what, i);
            exit(2);
        }
    }
    cuda_die(cudaMemcpy(guard,
                        buffer->base + GUARD_BYTES + buffer->data_bytes,
                        GUARD_BYTES, cudaMemcpyDeviceToHost),
             "copy trailing guard");
    for (uint32_t i = 0; i < GUARD_BYTES; i++) {
        if (guard[i] != CANARY) {
            fprintf(stderr, "error: %s trailing canary changed at byte %u\n", what, i);
            exit(2);
        }
    }
}

static int compare_bits(const float *a, const float *b, uint64_t count,
                        const char *what) {
    for (uint64_t i = 0; i < count; i++) {
        uint32_t abits;
        uint32_t bbits;
        memcpy(&abits, a + i, sizeof(abits));
        memcpy(&bbits, b + i, sizeof(bbits));
        if (abits != bbits) {
            fprintf(stderr,
                    "error: %s mismatch index=%llu reference=0x%08x candidate=0x%08x\n",
                    what, (unsigned long long)i, abits, bbits);
            return 0;
        }
    }
    return 1;
}

static int compare_u32_zero(const uint32_t *values, uint32_t count,
                            const char *what) {
    for (uint32_t i = 0; i < count; i++) {
        if (values[i] != 0u) {
            fprintf(stderr, "error: %s row=%u status=0x%x\n", what, i, values[i]);
            return 0;
        }
    }
    return 1;
}

static int has_finite_nonzero(const float *values, uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        if (isfinite(values[i]) && values[i] != 0.0f) return 1;
    }
    return 0;
}

static int float_compare(const void *a, const void *b) {
    const float av = *(const float *)a;
    const float bv = *(const float *)b;
    return av < bv ? -1 : av > bv ? 1 : 0;
}

static float time_one_consumer(
        int compact_variant,
        const float *f32_rows,
        const CompactAttentionKVRow *compact_rows,
        float *out,
        uint32_t n_rows,
        uint32_t repeats,
        cudaEvent_t begin,
        cudaEvent_t end) {
    cuda_die(cudaEventRecord(begin), "record begin event");
    for (uint32_t repeat = 0; repeat < repeats; repeat++) {
        if (compact_variant) {
            compact_consumer_kernel<<<2, 256>>>(compact_rows, out, n_rows);
        } else {
            f32_consumer_kernel<<<2, 256>>>(f32_rows, out, n_rows);
        }
    }
    cuda_die(cudaEventRecord(end), "record end event");
    cuda_die(cudaEventSynchronize(end), "synchronize end event");
    float elapsed = 0.0f;
    cuda_die(cudaEventElapsedTime(&elapsed, begin, end), "elapsed time");
    return elapsed / (float)repeats;
}

static void time_consumers_paired(
        const float *f32_rows,
        const CompactAttentionKVRow *compact_rows,
        float *f32_out,
        float *compact_out,
        uint32_t n_rows,
        uint32_t rounds,
        uint32_t repeats,
        float *f32_median,
        float *compact_median,
        float *paired_speedup_median) {
    float *f32_samples = (float *)malloc((size_t)rounds * sizeof(float));
    float *compact_samples = (float *)malloc((size_t)rounds * sizeof(float));
    float *paired_samples = (float *)malloc((size_t)rounds * sizeof(float));
    if (!f32_samples || !compact_samples || !paired_samples) {
        fprintf(stderr, "error: timing sample allocation failed\n");
        exit(2);
    }
    cudaEvent_t begin;
    cudaEvent_t end;
    cuda_die(cudaEventCreate(&begin), "create begin event");
    cuda_die(cudaEventCreate(&end), "create end event");

    for (uint32_t warmup = 0; warmup < 3u; warmup++) {
        f32_consumer_kernel<<<2, 256>>>(f32_rows, f32_out, n_rows);
        compact_consumer_kernel<<<2, 256>>>(compact_rows, compact_out, n_rows);
    }
    cuda_die(cudaDeviceSynchronize(), "consumer warmup");

    for (uint32_t round = 0; round < rounds; round++) {
        if ((round & 1u) == 0u) {
            f32_samples[round] = time_one_consumer(
                0, f32_rows, compact_rows, f32_out, n_rows, repeats,
                begin, end);
            compact_samples[round] = time_one_consumer(
                1, f32_rows, compact_rows, compact_out, n_rows, repeats,
                begin, end);
        } else {
            compact_samples[round] = time_one_consumer(
                1, f32_rows, compact_rows, compact_out, n_rows, repeats,
                begin, end);
            f32_samples[round] = time_one_consumer(
                0, f32_rows, compact_rows, f32_out, n_rows, repeats,
                begin, end);
        }
        paired_samples[round] = f32_samples[round] / compact_samples[round];
    }
    qsort(f32_samples, rounds, sizeof(float), float_compare);
    qsort(compact_samples, rounds, sizeof(float), float_compare);
    qsort(paired_samples, rounds, sizeof(float), float_compare);
    *f32_median = f32_samples[rounds / 2u];
    *compact_median = compact_samples[rounds / 2u];
    *paired_speedup_median = paired_samples[rounds / 2u];
    cuda_die(cudaEventDestroy(begin), "destroy begin event");
    cuda_die(cudaEventDestroy(end), "destroy end event");
    free(f32_samples);
    free(compact_samples);
    free(paired_samples);
}

static uint32_t parse_u32(const char *text, const char *name) {
    char *end = NULL;
    const unsigned long value = strtoul(text, &end, 10);
    if (!text[0] || !end || *end || value == 0ul || value > UINT32_MAX) {
        fprintf(stderr, "error: invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint32_t)value;
}

static uint32_t parse_device(const char *text) {
    char *end = NULL;
    const unsigned long value = strtoul(text, &end, 10);
    if (!text[0] || !end || *end || value > UINT32_MAX) {
        fprintf(stderr, "error: invalid device: %s\n", text);
        exit(2);
    }
    return (uint32_t)value;
}

int main(int argc, char **argv) {
    uint32_t device = 0;
    uint32_t n_rows = 8192;
    uint32_t rounds = 7;
    uint32_t repeats = 25;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            device = parse_device(argv[++i]);
        } else if (!strcmp(argv[i], "--rows") && i + 1 < argc) {
            n_rows = parse_u32(argv[++i], "rows");
        } else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) {
            rounds = parse_u32(argv[++i], "rounds");
        } else if (!strcmp(argv[i], "--repeats") && i + 1 < argc) {
            repeats = parse_u32(argv[++i], "repeats");
        } else {
            fprintf(stderr,
                    "usage: %s [--device INDEX] [--rows N] [--rounds N] [--repeats N]\n",
                    argv[0]);
            return 2;
        }
    }
    if (n_rows < 4u) {
        fprintf(stderr, "error: at least four rows are required for boundary coverage\n");
        return 2;
    }
    ds4_gpu_config config;
    memset(&config, 0, sizeof(config));
    config.n_gpus = 1;
    config.device_indices[0] = (int)device;
    if (!ds4_gpu_init_multi(&config)) {
        fprintf(stderr, "error: shipping CUDA backend initialization failed\n");
        return 2;
    }
    cuda_die(cudaSetDevice((int)device), "select CUDA device");

    const uint64_t values = (uint64_t)n_rows * HEAD_DIM;
    const size_t f32_bytes = (size_t)values * sizeof(float);
    const size_t compact_bytes = (size_t)n_rows * sizeof(CompactAttentionKVRow);
    const size_t status_bytes = (size_t)n_rows * sizeof(uint32_t);

    float *host_input = (float *)malloc(f32_bytes);
    float *host_reference = (float *)malloc(f32_bytes);
    float *host_unpacked = (float *)malloc(f32_bytes);
    uint32_t *host_status = (uint32_t *)malloc(status_bytes);
    CompactAttentionKVRow *host_compact = NULL;
    cuda_die(cudaMallocHost((void **)&host_compact, compact_bytes),
             "allocate host compact rows");
    if (!host_input || !host_reference || !host_unpacked ||
        !host_status || !host_compact) {
        fprintf(stderr, "error: host allocation failed\n");
        return 2;
    }
    fill_finite_input(host_input, n_rows);

    ds4_gpu_tensor *shipping_rows = ds4_gpu_tensor_alloc(f32_bytes);
    if (!shipping_rows ||
        !ds4_gpu_tensor_write(shipping_rows, 0, host_input, f32_bytes) ||
        !ds4_gpu_dsv4_fp8_kv_quantize_tensor(
            shipping_rows, n_rows, HEAD_DIM, N_ROT) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: shipping compressed-KV quantizer failed\n");
        return 2;
    }

    GuardedDeviceBuffer d_rows = guarded_alloc(f32_bytes, "allocate F32 rows");
    GuardedDeviceBuffer d_unpacked = guarded_alloc(f32_bytes, "allocate unpack rows");
    GuardedDeviceBuffer d_compact = guarded_alloc(compact_bytes, "allocate compact rows");
    GuardedDeviceBuffer d_status = guarded_alloc(status_bytes, "allocate status");
    GuardedDeviceBuffer d_f32_out = guarded_alloc(HEAD_DIM * sizeof(float), "allocate F32 result");
    GuardedDeviceBuffer d_compact_out = guarded_alloc(HEAD_DIM * sizeof(float), "allocate compact result");

    cuda_die(cudaMemset(d_status.data, 0, status_bytes), "clear pack status");
    compact_pack_kernel<<<n_rows, THREADS>>>(
        (const float *)ds4_gpu_tensor_contents(shipping_rows),
        (CompactAttentionKVRow *)d_compact.data,
        (uint32_t *)d_status.data, n_rows);
    compact_unpack_kernel<<<n_rows, THREADS>>>(
        (const CompactAttentionKVRow *)d_compact.data,
        (float *)d_unpacked.data, n_rows);
    cuda_die(cudaDeviceSynchronize(), "codec validation kernels");

    if (!ds4_gpu_tensor_read(shipping_rows, 0, host_reference, f32_bytes)) {
        fprintf(stderr, "error: shipping quantized-row readback failed\n");
        return 2;
    }
    cuda_die(cudaMemcpy(host_unpacked, d_unpacked.data, f32_bytes, cudaMemcpyDeviceToHost),
             "copy unpacked rows");
    cuda_die(cudaMemcpy(host_status, d_status.data, status_bytes,
                        cudaMemcpyDeviceToHost), "copy pack status");
    cuda_die(cudaMemcpy(host_compact, d_compact.data, compact_bytes,
                        cudaMemcpyDeviceToHost), "copy compact rows");
    if (!compare_u32_zero(host_status, n_rows, "finite pack rejected") ||
        !compare_bits(host_reference, host_unpacked, values, "codec exactness")) {
        return 2;
    }
    if (!has_finite_nonzero(host_reference, values)) {
        fprintf(stderr, "error: finite codec validation was degenerate\n");
        return 2;
    }
    uint64_t alternate_scale_count = 0;
    const float floor_scale = exp2f(ceilf(log2f(1.0e-4f / 448.0f)));
    for (uint32_t row = 0; row < n_rows; row++) {
        if (host_compact[row].reserved != 0u) {
            fprintf(stderr, "error: reserved word is nonzero at row=%u\n", row);
            return 2;
        }
        for (uint32_t g = 0; g < N_SCALE; g++) {
            uint32_t compact_bits;
            memcpy(&compact_bits, host_compact[row].scale + g,
                   sizeof(compact_bits));
            int exponent = 0;
            const float mantissa = frexpf(host_compact[row].scale[g], &exponent);
            if (mantissa != 0.5f) {
                fprintf(stderr,
                        "error: recovered scale is not a power of two row=%u group=%u bits=0x%08x\n",
                        row, g, compact_bits);
                return 2;
            }
            if (row == 0u && host_compact[row].scale[g] != floor_scale) {
                fprintf(stderr,
                        "error: scale-floor mismatch group=%u expected=%g candidate=%g\n",
                        g, floor_scale, host_compact[row].scale[g]);
                return 2;
            }
            if (row == 2u) {
                const float upper_scale = ldexpf(1.0f, 119 - (int)(g % 3u));
                const float upper_value = host_reference[
                    (uint64_t)row * HEAD_DIM + g * GROUP];
                if (!isfinite(upper_value) ||
                    host_compact[row].scale[g] != upper_scale) {
                    fprintf(stderr,
                            "error: upper-exponent scale mismatch group=%u expected=%g candidate=%g\n",
                            g, upper_scale, host_compact[row].scale[g]);
                    return 2;
                }
            }
            if (row == 3u) {
                const float original_scale = ldexpf(1.0f, (int)g - 9);
                const float alternate_scale = original_scale * 0.5f;
                const float rounded_boundary = host_reference[
                    (uint64_t)row * HEAD_DIM + g * GROUP];
                if (rounded_boundary != 224.0f * original_scale ||
                    host_compact[row].scale[g] != alternate_scale) {
                    fprintf(stderr,
                            "error: alternate exact scale was not recovered group=%u rounded=%g expected_scale=%g candidate_scale=%g\n",
                            g, rounded_boundary,
                            alternate_scale, host_compact[row].scale[g]);
                    return 2;
                }
                alternate_scale_count++;
            }
        }
    }
    if (alternate_scale_count != N_SCALE) {
        fprintf(stderr,
                "error: alternate exact scale coverage incomplete count=%llu expected=%u\n",
                (unsigned long long)alternate_scale_count, N_SCALE);
        return 2;
    }
    uint32_t rope_negative_zero_bits = 0u;
    memcpy(&rope_negative_zero_bits, host_compact[0].rope_f32 + 1u,
           sizeof(rope_negative_zero_bits));
    if (rope_negative_zero_bits != 0x80000000u) {
        fprintf(stderr, "error: signed zero in untouched RoPE tail was lost\n");
        return 2;
    }

    /* A finite but non-E4M3-grid value must not be rounded a second time. */
    memcpy(host_input, host_reference, f32_bytes);
    memset(host_input, 0, N_NOPE * sizeof(float));
    host_input[0] = nextafterf(1.0f, 2.0f);
    cuda_die(cudaMemcpy(d_rows.data, host_input, f32_bytes, cudaMemcpyHostToDevice),
             "copy nonrepresentable input");
    cuda_die(cudaMemset(d_status.data, 0, status_bytes),
             "clear nonrepresentable status");
    compact_pack_kernel<<<n_rows, THREADS>>>(
        (const float *)d_rows.data,
        (CompactAttentionKVRow *)d_compact.data,
        (uint32_t *)d_status.data, n_rows);
    cuda_die(cudaDeviceSynchronize(), "nonrepresentable pack rejection");
    cuda_die(cudaMemcpy(host_status, d_status.data, status_bytes,
                        cudaMemcpyDeviceToHost), "copy nonrepresentable status");
    if ((host_status[0] & PACK_UNREPRESENTABLE) == 0u) {
        fprintf(stderr, "error: finite non-E4M3 row was not rejected\n");
        return 2;
    }

    /* The current attention path assumes finite activations.  Compact packing
     * makes that implicit contract explicit and rejects non-finite rows rather
     * than inventing an approximate representation. */
    memcpy(host_input, host_reference, f32_bytes);
    host_input[0] = NAN;
    if (n_rows > 1u) host_input[HEAD_DIM + 1u] = INFINITY;
    if (n_rows > 2u) host_input[2u * HEAD_DIM + N_NOPE + 7u] = -INFINITY;
    cuda_die(cudaMemcpy(d_rows.data, host_input, f32_bytes, cudaMemcpyHostToDevice),
             "copy nonfinite input");
    cuda_die(cudaMemset(d_status.data, 0, status_bytes), "clear nonfinite status");
    compact_pack_kernel<<<n_rows, THREADS>>>(
        (const float *)d_rows.data,
        (CompactAttentionKVRow *)d_compact.data,
        (uint32_t *)d_status.data, n_rows);
    cuda_die(cudaDeviceSynchronize(), "nonfinite pack rejection");
    cuda_die(cudaMemcpy(host_status, d_status.data, status_bytes,
                        cudaMemcpyDeviceToHost), "copy nonfinite status");
    if ((host_status[0] & PACK_NONFINITE) == 0u ||
        (n_rows > 1u && (host_status[1] & PACK_NONFINITE) == 0u) ||
        (n_rows > 2u && (host_status[2] & PACK_NONFINITE) == 0u)) {
        fprintf(stderr, "error: nonfinite row was not rejected\n");
        return 2;
    }

    /* Restore the accepted compact rows before measuring consumers. */
    cuda_die(cudaMemcpy(d_rows.data, host_reference, f32_bytes, cudaMemcpyHostToDevice),
             "restore reference rows");
    cuda_die(cudaMemcpy(d_compact.data, host_compact, compact_bytes, cudaMemcpyHostToDevice),
             "restore compact rows");
    f32_consumer_kernel<<<2, 256>>>(
        (const float *)d_rows.data, (float *)d_f32_out.data, n_rows);
    compact_consumer_kernel<<<2, 256>>>(
        (const CompactAttentionKVRow *)d_compact.data,
        (float *)d_compact_out.data, n_rows);
    cuda_die(cudaDeviceSynchronize(), "consumer exactness kernels");
    float host_f32_out[HEAD_DIM];
    float host_compact_out[HEAD_DIM];
    cuda_die(cudaMemcpy(host_f32_out, d_f32_out.data, sizeof(host_f32_out),
                        cudaMemcpyDeviceToHost), "copy F32 consumer result");
    cuda_die(cudaMemcpy(host_compact_out, d_compact_out.data,
                        sizeof(host_compact_out), cudaMemcpyDeviceToHost),
             "copy compact consumer result");
    if (!compare_bits(host_f32_out, host_compact_out, HEAD_DIM,
                      "consumer exactness")) {
        return 2;
    }
    if (!has_finite_nonzero(host_f32_out, HEAD_DIM)) {
        fprintf(stderr, "error: consumer validation was degenerate\n");
        return 2;
    }

    check_guards(&d_rows, "F32 rows");
    check_guards(&d_unpacked, "unpacked rows");
    check_guards(&d_compact, "compact rows");
    check_guards(&d_status, "pack status");
    check_guards(&d_f32_out, "F32 result");
    check_guards(&d_compact_out, "compact result");

    float f32_ms = 0.0f;
    float compact_ms = 0.0f;
    float paired_speedup = 0.0f;
    time_consumers_paired(
        (const float *)d_rows.data,
        (const CompactAttentionKVRow *)d_compact.data,
        (float *)d_f32_out.data, (float *)d_compact_out.data,
        n_rows, rounds, repeats, &f32_ms, &compact_ms, &paired_speedup);
    check_guards(&d_rows, "timed F32 rows");
    check_guards(&d_compact, "timed compact rows");
    check_guards(&d_f32_out, "timed F32 result");
    check_guards(&d_compact_out, "timed compact result");

    printf("scenario=compact-attention-kv-codec\n");
    printf("validation=byte-exact-nonzero-adversarial\n");
    printf("nonfinite_policy=reject-whole-row\n");
    printf("rejected_row_commit_policy=caller-must-observe-pack-ok-before-use\n");
    printf("reference_quantizer=shipping-ds4-cuda-production-flags\n");
    printf("timing_scope=paired-alternating-bounded-diagnostic-not-promotion-evidence\n");
    printf("rows=%u\n", n_rows);
    printf("head_dim=%u\n", HEAD_DIM);
    printf("n_rot=%u\n", N_ROT);
    printf("f32_row_bytes=%zu\n", (size_t)HEAD_DIM * sizeof(float));
    printf("compact_row_bytes=%zu\n", sizeof(CompactAttentionKVRow));
    printf("storage_reduction=%.9f\n",
           1.0 - (double)sizeof(CompactAttentionKVRow) /
                     ((double)HEAD_DIM * sizeof(float)));
    printf("f32_consumer_median_ms=%.9g\n", f32_ms);
    printf("compact_consumer_median_ms=%.9g\n", compact_ms);
    printf("compact_consumer_speedup=%.9g\n", f32_ms / compact_ms);
    printf("paired_consumer_speedup_median=%.9g\n", paired_speedup);
    printf("alternate_exact_scales=%llu\n",
           (unsigned long long)alternate_scale_count);
    printf("canaries=passed\n");
    printf("harness_status=ok\n");

    cudaFree(d_rows.base);
    cudaFree(d_unpacked.base);
    cudaFree(d_compact.base);
    cudaFree(d_status.base);
    cudaFree(d_f32_out.base);
    cudaFree(d_compact_out.base);
    free(host_input);
    free(host_reference);
    free(host_unpacked);
    free(host_status);
    cudaFreeHost(host_compact);
    ds4_gpu_tensor_free(shipping_rows);
    ds4_gpu_cleanup();
    return 0;
}
