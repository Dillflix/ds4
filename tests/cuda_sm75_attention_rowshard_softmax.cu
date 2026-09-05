/*
 * Independent SM75 diagnostic for row-sharded compressed-attention history.
 *
 * Production is not modified.  The control owns one complete compact cache.
 * The candidate owns the same rows exactly once across two separately guarded
 * allocations.  Selected rows deliberately alternate between owners so an
 * implementation cannot accidentally rely on one contiguous Top-K phase.
 *
 * Two candidate semantics are tested:
 *
 *   ordered-local-address: visit the original Top-K sequence and select the
 *     appropriate owner-local allocation for every row.  This must be bit
 *     exact, but is only an address/ownership proof; it is not a distributed
 *     execution schedule.
 *
 *   parallel-partials: each owner computes (max, sum, numerator[512]) for its
 *     selected rows, then a sink merges the two states and the attention sink.
 *     This is the useful distributed protocol.  It is mathematically exact,
 *     but changes floating-point association, so the harness reports rather
 *     than hides any bit drift.  Production integration remains ineligible.
 */

#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    HEAD_DIM = 512,
    N_ROT = 64,
    N_NOPE = HEAD_DIM - N_ROT,
    SCALE_GROUP = 64,
    N_SCALE = N_NOPE / SCALE_GROUP,
    N_HEAD = 64,
    HEADS_PER_BLOCK = 8,
    ROWS_PER_STAGE = 8,
    TOP_K = 512,
    OWNER_TOP_K = TOP_K / 2,
    THREADS = 256,
    STATE_HEADER_FLOATS = 4,
    STATE_LOGICAL_FLOATS = HEAD_DIM + 2,
    STATE_FLOATS = HEAD_DIM + STATE_HEADER_FLOATS,
    GUARD_BYTES = 256,
    CANARY = 0xa5,
};

struct __align__(32) CompactAttentionKVRow {
    float scale[N_SCALE];
    uint32_t reserved;
    uint8_t code[N_NOPE];
    float rope_f32[N_ROT];
};

static_assert(N_SCALE == 7, "compact row scale count changed");
static_assert(sizeof(CompactAttentionKVRow) == 736,
              "compact compressed-attention row must remain 736 bytes");
static_assert((STATE_FLOATS * sizeof(float)) % alignof(float4) == 0,
              "partial-state records must preserve float4 alignment");

static void *host_alloc_aligned(
        size_t alignment, size_t bytes, const char *what) {
    void *data = NULL;
    const int rc = posix_memalign(&data, alignment, bytes);
    if (rc != 0 || !data) {
        fprintf(stderr, "error: %s: %s\n", what,
                rc == 0 ? "allocation returned null" : strerror(rc));
        exit(2);
    }
    return data;
}

struct GuardedBuffer {
    uint8_t *base;
    uint8_t *data;
    size_t bytes;
};

static void cuda_die(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return;
    fprintf(stderr, "error: %s: %s\n", what, cudaGetErrorString(err));
    exit(2);
}

static GuardedBuffer guarded_alloc(size_t bytes, const char *what) {
    GuardedBuffer out = {};
    out.bytes = bytes;
    cuda_die(cudaMalloc((void **)&out.base, bytes + 2u * GUARD_BYTES), what);
    out.data = out.base + GUARD_BYTES;
    cuda_die(cudaMemset(out.base, CANARY, GUARD_BYTES), "initialize leading guard");
    cuda_die(cudaMemset(out.data + bytes, CANARY, GUARD_BYTES),
             "initialize trailing guard");
    return out;
}

static int check_guard(const GuardedBuffer *buffer, const char *name) {
    uint8_t host[GUARD_BYTES];
    cuda_die(cudaMemcpy(host, buffer->base, GUARD_BYTES, cudaMemcpyDeviceToHost),
             "copy leading guard");
    for (uint32_t i = 0; i < GUARD_BYTES; i++) {
        if (host[i] != CANARY) {
            fprintf(stderr, "error: %s leading guard changed at %u\n", name, i);
            return 0;
        }
    }
    cuda_die(cudaMemcpy(host, buffer->data + buffer->bytes, GUARD_BYTES,
                        cudaMemcpyDeviceToHost), "copy trailing guard");
    for (uint32_t i = 0; i < GUARD_BYTES; i++) {
        if (host[i] != CANARY) {
            fprintf(stderr, "error: %s trailing guard changed at %u\n", name, i);
            return 0;
        }
    }
    return 1;
}

__host__ __device__ __forceinline__ static float e4m3fn_value(uint32_t code) {
    const uint32_t exponent = (code >> 3) & 15u;
    const uint32_t mantissa = code & 7u;
    if (exponent == 0u) return (float)mantissa * 0.001953125f;
    return (1.0f + (float)mantissa * 0.125f) *
           exp2f((float)exponent - 7.0f);
}

__device__ __forceinline__ static float decode_code_bits(
        uint8_t code, uint32_t scale_bits) {
    const uint32_t sign = (uint32_t)(code & 0x80u) << 24;
    const uint32_t magnitude = (uint32_t)(code & 0x7fu);
    if (magnitude == 0u) return __uint_as_float(sign);

    const uint32_t scale_exp = (scale_bits >> 23) & 0xffu;
    const uint32_t fp8_exp = magnitude >> 3;
    const uint32_t fp8_mantissa = magnitude & 7u;
    int32_t result_exp;
    uint32_t result_mantissa;
    if (fp8_exp != 0u) {
        result_exp = (int32_t)scale_exp + (int32_t)fp8_exp - 7;
        result_mantissa = fp8_mantissa << 20;
    } else {
        const uint32_t leading = 31u - (uint32_t)__clz(fp8_mantissa);
        result_exp = (int32_t)scale_exp - 9 + (int32_t)leading;
        result_mantissa =
            (fp8_mantissa << (23u - leading)) & 0x7fffffu;
    }
    if (scale_exp != 0u && scale_exp != 0xffu &&
        result_exp > 0 && result_exp < 0xff) {
        return __uint_as_float(
            sign | ((uint32_t)result_exp << 23) | result_mantissa);
    }
    float value = e4m3fn_value(magnitude) * __uint_as_float(scale_bits);
    return __uint_as_float(__float_as_uint(value) | sign);
}

__device__ __forceinline__ static float4 compact_load_float4(
        const CompactAttentionKVRow *row, uint32_t c4, uint32_t lane) {
    const uint32_t d = c4 * 4u;
    uint32_t scale_bits = 0u;
    if (d < N_NOPE && (lane & 15u) == 0u) {
        scale_bits = __float_as_uint(row->scale[d / SCALE_GROUP]);
    }
    scale_bits = __shfl_sync(0xffffffffu, scale_bits, lane & 16u);
    if (d < N_NOPE) {
        const uint32_t codes = *(const uint32_t *)(row->code + d);
        return make_float4(
            decode_code_bits((uint8_t)(codes >> 0), scale_bits),
            decode_code_bits((uint8_t)(codes >> 8), scale_bits),
            decode_code_bits((uint8_t)(codes >> 16), scale_bits),
            decode_code_bits((uint8_t)(codes >> 24), scale_bits));
    }
    const uint32_t rope = d - N_NOPE;
    return make_float4(row->rope_f32[rope + 0u],
                       row->rope_f32[rope + 1u],
                       row->rope_f32[rope + 2u],
                       row->rope_f32[rope + 3u]);
}

__device__ __forceinline__ static float dot4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ static float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ static void online_update(
        float score, float4 k0, float4 k1, float4 k2, float4 k3,
        float *max_score, float *sum,
        float4 *o0, float4 *o1, float4 *o2, float4 *o3) {
    const float next_max = fmaxf(*max_score, score);
    const float old_scale = expf(*max_score - next_max);
    const float row_scale = expf(score - next_max);
    *sum = *sum * old_scale + row_scale;
#define UPDATE4(dst, src) do { \
    (dst).x = (dst).x * old_scale + (src).x * row_scale; \
    (dst).y = (dst).y * old_scale + (src).y * row_scale; \
    (dst).z = (dst).z * old_scale + (src).z * row_scale; \
    (dst).w = (dst).w * old_scale + (src).w * row_scale; \
} while (0)
    UPDATE4(*o0, k0);
    UPDATE4(*o1, k1);
    UPDATE4(*o2, k2);
    UPDATE4(*o3, k3);
#undef UPDATE4
    *max_score = next_max;
}

template <bool OWNER_LOCAL>
__global__ static void ordered_attention_kernel(
        const CompactAttentionKVRow *full_rows,
        const CompactAttentionKVRow *owner0_rows,
        const CompactAttentionKVRow *owner1_rows,
        uint32_t owner1_base,
        const int32_t *topk,
        const float *sinks,
        const float *query,
        float *out,
        uint32_t n_tokens) {
    const uint32_t token = blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = blockIdx.y * HEADS_PER_BLOCK + warp;
    if (token >= n_tokens || head >= N_HEAD) return;

    __shared__ float4 staged[ROWS_PER_STAGE * (HEAD_DIM / 4u)];
    const float4 *q4 = (const float4 *)(
        query + ((uint64_t)token * N_HEAD + head) * HEAD_DIM);
    const float4 q0 = q4[lane + 0u];
    const float4 q1 = q4[lane + 32u];
    const float4 q2 = q4[lane + 64u];
    const float4 q3 = q4[lane + 96u];
    float max_score = -INFINITY;
    float sum = 0.0f;
    float4 o0 = make_float4(0, 0, 0, 0);
    float4 o1 = o0, o2 = o0, o3 = o0;
    const float score_scale = rsqrtf((float)HEAD_DIM);

    for (uint32_t row0 = 0; row0 < TOP_K; row0 += ROWS_PER_STAGE) {
        for (uint32_t off = threadIdx.x;
             off < ROWS_PER_STAGE * (HEAD_DIM / 4u); off += blockDim.x) {
            const uint32_t rr = off / (HEAD_DIM / 4u);
            const uint32_t c4 = off % (HEAD_DIM / 4u);
            const uint32_t global_row =
                (uint32_t)topk[(uint64_t)token * TOP_K + row0 + rr];
            const CompactAttentionKVRow *row;
            if (OWNER_LOCAL) {
                row = global_row < owner1_base
                    ? owner0_rows + global_row
                    : owner1_rows + (global_row - owner1_base);
            } else {
                row = full_rows + global_row;
            }
            staged[off] = compact_load_float4(row, c4, threadIdx.x & 31u);
        }
        __syncthreads();
        for (uint32_t rr = 0; rr < ROWS_PER_STAGE; rr++) {
            const float4 *kv = staged + rr * (HEAD_DIM / 4u);
            const float4 k0 = kv[lane + 0u];
            const float4 k1 = kv[lane + 32u];
            const float4 k2 = kv[lane + 64u];
            const float4 k3 = kv[lane + 96u];
            float score = dot4(q0, k0) + dot4(q1, k1) +
                          dot4(q2, k2) + dot4(q3, k3);
            score = warp_sum(score) * score_scale;
            score = __shfl_sync(0xffffffffu, score, 0);
            online_update(score, k0, k1, k2, k3,
                          &max_score, &sum, &o0, &o1, &o2, &o3);
        }
        __syncthreads();
    }

    const float sink = sinks[head];
    const float next_max = fmaxf(max_score, sink);
    const float old_scale = expf(max_score - next_max);
    const float sink_scale = expf(sink - next_max);
    sum = sum * old_scale + sink_scale;
    const float inv_sum = sum == 0.0f ? 0.0f : 1.0f / sum;
    o0.x *= old_scale * inv_sum; o0.y *= old_scale * inv_sum;
    o0.z *= old_scale * inv_sum; o0.w *= old_scale * inv_sum;
    o1.x *= old_scale * inv_sum; o1.y *= old_scale * inv_sum;
    o1.z *= old_scale * inv_sum; o1.w *= old_scale * inv_sum;
    o2.x *= old_scale * inv_sum; o2.y *= old_scale * inv_sum;
    o2.z *= old_scale * inv_sum; o2.w *= old_scale * inv_sum;
    o3.x *= old_scale * inv_sum; o3.y *= old_scale * inv_sum;
    o3.z *= old_scale * inv_sum; o3.w *= old_scale * inv_sum;
    float4 *dst = (float4 *)(out +
        ((uint64_t)token * N_HEAD + head) * HEAD_DIM);
    dst[lane + 0u] = o0;
    dst[lane + 32u] = o1;
    dst[lane + 64u] = o2;
    dst[lane + 96u] = o3;
}

__global__ static void owner_partial_kernel(
        const CompactAttentionKVRow *local_rows,
        const int32_t *local_topk,
        const float *query,
        float *states,
        uint32_t n_tokens) {
    const uint32_t token = blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = blockIdx.y * HEADS_PER_BLOCK + warp;
    if (token >= n_tokens || head >= N_HEAD) return;

    __shared__ float4 staged[ROWS_PER_STAGE * (HEAD_DIM / 4u)];
    const float4 *q4 = (const float4 *)(
        query + ((uint64_t)token * N_HEAD + head) * HEAD_DIM);
    const float4 q0 = q4[lane + 0u];
    const float4 q1 = q4[lane + 32u];
    const float4 q2 = q4[lane + 64u];
    const float4 q3 = q4[lane + 96u];
    float max_score = -INFINITY;
    float sum = 0.0f;
    float4 o0 = make_float4(0, 0, 0, 0);
    float4 o1 = o0, o2 = o0, o3 = o0;
    const float score_scale = rsqrtf((float)HEAD_DIM);

    for (uint32_t row0 = 0; row0 < OWNER_TOP_K;
         row0 += ROWS_PER_STAGE) {
        const uint32_t nr = OWNER_TOP_K - row0 < ROWS_PER_STAGE
            ? OWNER_TOP_K - row0 : ROWS_PER_STAGE;
        for (uint32_t off = threadIdx.x;
             off < nr * (HEAD_DIM / 4u); off += blockDim.x) {
            const uint32_t rr = off / (HEAD_DIM / 4u);
            const uint32_t c4 = off % (HEAD_DIM / 4u);
            staged[off] = compact_load_float4(
                local_rows + local_topk[(uint64_t)token * OWNER_TOP_K +
                                        row0 + rr],
                c4, threadIdx.x & 31u);
        }
        __syncthreads();
        for (uint32_t rr = 0; rr < nr; rr++) {
            const float4 *kv = staged + rr * (HEAD_DIM / 4u);
            const float4 k0 = kv[lane + 0u];
            const float4 k1 = kv[lane + 32u];
            const float4 k2 = kv[lane + 64u];
            const float4 k3 = kv[lane + 96u];
            float score = dot4(q0, k0) + dot4(q1, k1) +
                          dot4(q2, k2) + dot4(q3, k3);
            score = warp_sum(score) * score_scale;
            score = __shfl_sync(0xffffffffu, score, 0);
            online_update(score, k0, k1, k2, k3,
                          &max_score, &sum, &o0, &o1, &o2, &o3);
        }
        __syncthreads();
    }

    float *state = states +
        ((uint64_t)token * N_HEAD + head) * STATE_FLOATS;
    if (lane == 0u) {
        state[0] = max_score;
        state[1] = sum;
    }
    float4 *numerator = (float4 *)(state + STATE_HEADER_FLOATS);
    numerator[lane + 0u] = o0;
    numerator[lane + 32u] = o1;
    numerator[lane + 64u] = o2;
    numerator[lane + 96u] = o3;
}

__global__ static void merge_partial_states_kernel(
        const float *state0,
        const float *state1,
        const float *sinks,
        float *out,
        uint32_t n_tokens) {
    const uint32_t token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t d = threadIdx.x;
    if (token >= n_tokens || head >= N_HEAD || d >= HEAD_DIM) return;
    const uint64_t state_offset =
        ((uint64_t)token * N_HEAD + head) * STATE_FLOATS;
    const float m0 = state0[state_offset + 0u];
    const float m1 = state1[state_offset + 0u];
    const float l0 = state0[state_offset + 1u];
    const float l1 = state1[state_offset + 1u];
    const float merged_max = fmaxf(m0, m1);
    const float scale0 = l0 == 0.0f ? 0.0f : expf(m0 - merged_max);
    const float scale1 = l1 == 0.0f ? 0.0f : expf(m1 - merged_max);
    float denominator = l0 * scale0 + l1 * scale1;
    const float final_max = fmaxf(merged_max, sinks[head]);
    const float old_scale = expf(merged_max - final_max);
    denominator = denominator * old_scale + expf(sinks[head] - final_max);
    const float numerator =
        (state0[state_offset + STATE_HEADER_FLOATS + d] * scale0 +
         state1[state_offset + STATE_HEADER_FLOATS + d] * scale1) * old_scale;
    out[((uint64_t)token * N_HEAD + head) * HEAD_DIM + d] =
        denominator == 0.0f ? 0.0f : numerator / denominator;
}

enum TimingArm {
    ARM_CONTROL,
    ARM_ORDERED_LOCAL,
    ARM_OWNER0,
    ARM_OWNER1,
    ARM_MERGE,
    ARM_PARALLEL_SEQUENTIAL,
};

static void launch_arm(
        TimingArm arm,
        const CompactAttentionKVRow *full_rows,
        const CompactAttentionKVRow *owner0_rows,
        const CompactAttentionKVRow *owner1_rows,
        uint32_t owner1_base,
        const int32_t *topk,
        const int32_t *owner0_topk,
        const int32_t *owner1_topk,
        const float *sinks,
        const float *query,
        float *state0,
        float *state1,
        float *out,
        uint32_t n_tokens) {
    const dim3 attention_grid(n_tokens, N_HEAD / HEADS_PER_BLOCK, 1u);
    const dim3 merge_grid(n_tokens, N_HEAD, 1u);
    switch (arm) {
        case ARM_CONTROL:
            ordered_attention_kernel<false><<<attention_grid, THREADS>>>(
                full_rows, owner0_rows, owner1_rows, owner1_base,
                topk, sinks, query, out, n_tokens);
            break;
        case ARM_ORDERED_LOCAL:
            ordered_attention_kernel<true><<<attention_grid, THREADS>>>(
                full_rows, owner0_rows, owner1_rows, owner1_base,
                topk, sinks, query, out, n_tokens);
            break;
        case ARM_OWNER0:
            owner_partial_kernel<<<attention_grid, THREADS>>>(
                owner0_rows, owner0_topk, query, state0, n_tokens);
            break;
        case ARM_OWNER1:
            owner_partial_kernel<<<attention_grid, THREADS>>>(
                owner1_rows, owner1_topk, query, state1, n_tokens);
            break;
        case ARM_MERGE:
            merge_partial_states_kernel<<<merge_grid, HEAD_DIM>>>(
                state0, state1, sinks, out, n_tokens);
            break;
        case ARM_PARALLEL_SEQUENTIAL:
            owner_partial_kernel<<<attention_grid, THREADS>>>(
                owner0_rows, owner0_topk, query, state0, n_tokens);
            owner_partial_kernel<<<attention_grid, THREADS>>>(
                owner1_rows, owner1_topk, query, state1, n_tokens);
            merge_partial_states_kernel<<<merge_grid, HEAD_DIM>>>(
                state0, state1, sinks, out, n_tokens);
            break;
    }
}

static int float_compare(const void *a, const void *b) {
    const float av = *(const float *)a;
    const float bv = *(const float *)b;
    return (av > bv) - (av < bv);
}

static float time_arm(
        TimingArm arm,
        const CompactAttentionKVRow *full_rows,
        const CompactAttentionKVRow *owner0_rows,
        const CompactAttentionKVRow *owner1_rows,
        uint32_t owner1_base,
        const int32_t *topk,
        const int32_t *owner0_topk,
        const int32_t *owner1_topk,
        const float *sinks,
        const float *query,
        float *state0,
        float *state1,
        float *out,
        uint32_t n_tokens,
        uint32_t repeats) {
    cudaEvent_t begin, end;
    cuda_die(cudaEventCreate(&begin), "create begin event");
    cuda_die(cudaEventCreate(&end), "create end event");
    cuda_die(cudaEventRecord(begin), "record begin event");
    for (uint32_t i = 0; i < repeats; i++) {
        launch_arm(arm, full_rows, owner0_rows, owner1_rows,
                   owner1_base, topk,
                   owner0_topk, owner1_topk, sinks, query,
                   state0, state1, out, n_tokens);
    }
    cuda_die(cudaEventRecord(end), "record end event");
    cuda_die(cudaEventSynchronize(end), "synchronize end event");
    float elapsed = 0.0f;
    cuda_die(cudaEventElapsedTime(&elapsed, begin, end), "measure elapsed time");
    cuda_die(cudaEventDestroy(begin), "destroy begin event");
    cuda_die(cudaEventDestroy(end), "destroy end event");
    return elapsed / (float)repeats;
}

static float median_arm(
        TimingArm arm,
        const CompactAttentionKVRow *full_rows,
        const CompactAttentionKVRow *owner0_rows,
        const CompactAttentionKVRow *owner1_rows,
        uint32_t owner1_base,
        const int32_t *topk,
        const int32_t *owner0_topk,
        const int32_t *owner1_topk,
        const float *sinks,
        const float *query,
        float *state0,
        float *state1,
        float *out,
        uint32_t n_tokens,
        uint32_t rounds,
        uint32_t repeats) {
    float *samples = (float *)malloc((size_t)rounds * sizeof(float));
    if (!samples) {
        fprintf(stderr, "error: timing allocation failed\n");
        exit(2);
    }
    for (uint32_t round = 0; round < rounds; round++) {
        samples[round] = time_arm(
            arm, full_rows, owner0_rows, owner1_rows,
            owner1_base, topk, owner0_topk, owner1_topk,
            sinks, query,
            state0, state1, out, n_tokens, repeats);
    }
    qsort(samples, rounds, sizeof(float), float_compare);
    const float median = samples[rounds / 2u];
    free(samples);
    return median;
}

static uint32_t parse_u32(const char *text, const char *name) {
    char *end = NULL;
    const unsigned long value = strtoul(text, &end, 10);
    if (!text[0] || !end || *end || value > UINT32_MAX) {
        fprintf(stderr, "error: invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint32_t)value;
}

static void fill_rows(CompactAttentionKVRow *rows, uint32_t n_rows) {
    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t group = 0; group < N_SCALE; group++) {
            rows[row].scale[group] = exp2f((float)((int)(row + group) % 5 - 7));
        }
        rows[row].reserved = 0u;
        for (uint32_t d = 0; d < N_NOPE; d++) {
            uint8_t code = (uint8_t)(1u +
                ((row * 29u + d * 17u + d / 11u) % 126u));
            if ((row + d) & 1u) code |= 0x80u;
            rows[row].code[d] = code;
        }
        for (uint32_t d = 0; d < N_ROT; d++) {
            const int32_t value =
                (int32_t)((row * 13u + d * 7u) % 257u) - 128;
            rows[row].rope_f32[d] = (float)value * 0.0009765625f;
        }
    }
}

static void fill_metadata(
        float *query, float *sinks, int32_t *topk,
        int32_t *owner0_topk, int32_t *owner1_topk,
        uint32_t n_rows, uint32_t n_tokens) {
    const uint32_t owner1_base = n_rows / 2u;
    const uint32_t owner1_count = n_rows - owner1_base;
    for (uint32_t head = 0; head < N_HEAD; head++) {
        sinks[head] = -0.5f + (float)(head % 9u) * 0.03125f;
    }
    const uint64_t query_values =
        (uint64_t)n_tokens * N_HEAD * HEAD_DIM;
    for (uint64_t i = 0; i < query_values; i++) {
        const int32_t value = (int32_t)((i * 37u + i / 19u) % 509u) - 254;
        query[i] = (float)value * 0.00048828125f;
    }
    for (uint32_t token = 0; token < n_tokens; token++) {
        for (uint32_t i = 0; i < TOP_K; i++) {
            const uint32_t local = i / 2u;
            const uint32_t row = (i & 1u)
                ? owner1_base + (token * 19u + local) % owner1_count
                : (token * 17u + local) % owner1_base;
            topk[(uint64_t)token * TOP_K + i] = (int32_t)row;
            if (i & 1u) {
                owner1_topk[(uint64_t)token * OWNER_TOP_K + local] =
                    (int32_t)(row - owner1_base);
            } else {
                owner0_topk[(uint64_t)token * OWNER_TOP_K + local] =
                    (int32_t)row;
            }
        }
    }
}

int main(int argc, char **argv) {
    uint32_t device = 0u;
    uint32_t n_rows = 8192u;
    uint32_t n_tokens = 32u;
    uint32_t rounds = 9u;
    uint32_t repeats = 25u;
    for (int i = 1; i < argc; i++) {
        if (i + 1 >= argc) {
            fprintf(stderr, "error: missing value for %s\n", argv[i]);
            return 2;
        }
        if (!strcmp(argv[i], "--device")) device = parse_u32(argv[++i], "device");
        else if (!strcmp(argv[i], "--rows")) n_rows = parse_u32(argv[++i], "rows");
        else if (!strcmp(argv[i], "--tokens")) n_tokens = parse_u32(argv[++i], "tokens");
        else if (!strcmp(argv[i], "--rounds")) rounds = parse_u32(argv[++i], "rounds");
        else if (!strcmp(argv[i], "--repeats")) repeats = parse_u32(argv[++i], "repeats");
        else {
            fprintf(stderr, "error: unknown argument: %s\n", argv[i]);
            return 2;
        }
    }
    if (n_rows < TOP_K || n_tokens == 0u || rounds == 0u || repeats == 0u) {
        fprintf(stderr, "error: rows >= 512 and positive tokens/rounds/repeats required\n");
        return 2;
    }
    cuda_die(cudaSetDevice((int)device), "select CUDA device");
    cudaDeviceProp properties = {};
    cuda_die(cudaGetDeviceProperties(&properties, (int)device),
             "query CUDA device properties");
    if (properties.major != 7 || properties.minor != 5) {
        fprintf(stderr, "error: device %u is sm_%d%d, expected sm_75\n",
                device, properties.major, properties.minor);
        return 2;
    }

    const uint32_t owner1_base = n_rows / 2u;
    const uint32_t owner0_count = owner1_base;
    const uint32_t owner1_count = n_rows - owner1_base;
    const size_t full_bytes =
        (size_t)n_rows * sizeof(CompactAttentionKVRow);
    const size_t owner0_bytes =
        (size_t)owner0_count * sizeof(CompactAttentionKVRow);
    const size_t owner1_bytes =
        (size_t)owner1_count * sizeof(CompactAttentionKVRow);
    const size_t query_bytes =
        (size_t)n_tokens * N_HEAD * HEAD_DIM * sizeof(float);
    const size_t sink_bytes = N_HEAD * sizeof(float);
    const size_t topk_bytes = (size_t)n_tokens * TOP_K * sizeof(int32_t);
    const size_t owner_topk_bytes =
        (size_t)n_tokens * OWNER_TOP_K * sizeof(int32_t);
    const size_t output_bytes = query_bytes;
    const size_t state_storage_bytes =
        (size_t)n_tokens * N_HEAD * STATE_FLOATS * sizeof(float);
    const size_t state_logical_bytes =
        (size_t)n_tokens * N_HEAD * STATE_LOGICAL_FLOATS * sizeof(float);

    CompactAttentionKVRow *host_rows =
        (CompactAttentionKVRow *)host_alloc_aligned(
            alignof(CompactAttentionKVRow), full_bytes,
            "allocate aligned host cache fixture");
    float *host_query = (float *)malloc(query_bytes);
    float *host_sinks = (float *)malloc(sink_bytes);
    int32_t *host_topk = (int32_t *)malloc(topk_bytes);
    int32_t *host_owner0_topk = (int32_t *)malloc(owner_topk_bytes);
    int32_t *host_owner1_topk = (int32_t *)malloc(owner_topk_bytes);
    float *host_control = (float *)malloc(output_bytes);
    float *host_ordered = (float *)malloc(output_bytes);
    float *host_parallel = (float *)malloc(output_bytes);
    if (!host_query || !host_sinks || !host_topk ||
        !host_owner0_topk || !host_owner1_topk ||
        !host_control || !host_ordered || !host_parallel) {
        fprintf(stderr, "error: host allocation failed\n");
        return 2;
    }
    if ((uintptr_t)host_rows % alignof(CompactAttentionKVRow) != 0u) {
        fprintf(stderr, "error: host cache fixture is not %zu-byte aligned\n",
                alignof(CompactAttentionKVRow));
        return 2;
    }
    fill_rows(host_rows, n_rows);
    fill_metadata(host_query, host_sinks, host_topk,
                  host_owner0_topk, host_owner1_topk, n_rows, n_tokens);

    GuardedBuffer d_full = guarded_alloc(full_bytes, "allocate control cache");
    GuardedBuffer d_owner0 = guarded_alloc(owner0_bytes, "allocate owner0 cache");
    GuardedBuffer d_owner1 = guarded_alloc(owner1_bytes, "allocate owner1 cache");
    GuardedBuffer d_query = guarded_alloc(query_bytes, "allocate query");
    GuardedBuffer d_sinks = guarded_alloc(sink_bytes, "allocate sinks");
    GuardedBuffer d_topk = guarded_alloc(topk_bytes, "allocate top-k");
    GuardedBuffer d_owner0_topk =
        guarded_alloc(owner_topk_bytes, "allocate owner0 local top-k");
    GuardedBuffer d_owner1_topk =
        guarded_alloc(owner_topk_bytes, "allocate owner1 local top-k");
    GuardedBuffer d_state0 =
        guarded_alloc(state_storage_bytes, "allocate owner0 state");
    GuardedBuffer d_state1 =
        guarded_alloc(state_storage_bytes, "allocate owner1 state");
    GuardedBuffer d_control = guarded_alloc(output_bytes, "allocate control output");
    GuardedBuffer d_ordered = guarded_alloc(output_bytes, "allocate ordered output");
    GuardedBuffer d_parallel = guarded_alloc(output_bytes, "allocate parallel output");

    cuda_die(cudaMemcpy(d_full.data, host_rows, full_bytes, cudaMemcpyHostToDevice),
             "copy control cache");
    cuda_die(cudaMemcpy(d_owner0.data, host_rows, owner0_bytes, cudaMemcpyHostToDevice),
             "copy owner0 cache");
    cuda_die(cudaMemcpy(d_owner1.data, host_rows + owner1_base,
                        owner1_bytes, cudaMemcpyHostToDevice),
             "copy owner1 cache");
    cuda_die(cudaMemcpy(d_query.data, host_query, query_bytes, cudaMemcpyHostToDevice),
             "copy query");
    cuda_die(cudaMemcpy(d_sinks.data, host_sinks, sink_bytes, cudaMemcpyHostToDevice),
             "copy sinks");
    cuda_die(cudaMemcpy(d_topk.data, host_topk, topk_bytes, cudaMemcpyHostToDevice),
             "copy top-k");
    cuda_die(cudaMemcpy(d_owner0_topk.data, host_owner0_topk,
                        owner_topk_bytes, cudaMemcpyHostToDevice),
             "copy owner0 local top-k");
    cuda_die(cudaMemcpy(d_owner1_topk.data, host_owner1_topk,
                        owner_topk_bytes, cudaMemcpyHostToDevice),
             "copy owner1 local top-k");

    const CompactAttentionKVRow *full =
        (const CompactAttentionKVRow *)d_full.data;
    const CompactAttentionKVRow *owner0 =
        (const CompactAttentionKVRow *)d_owner0.data;
    const CompactAttentionKVRow *owner1 =
        (const CompactAttentionKVRow *)d_owner1.data;
    const int32_t *topk = (const int32_t *)d_topk.data;
    const int32_t *owner0_topk = (const int32_t *)d_owner0_topk.data;
    const int32_t *owner1_topk = (const int32_t *)d_owner1_topk.data;
    const float *sinks = (const float *)d_sinks.data;
    const float *query = (const float *)d_query.data;
    float *state0 = (float *)d_state0.data;
    float *state1 = (float *)d_state1.data;

    launch_arm(ARM_CONTROL, full, owner0, owner1, owner1_base,
               topk, owner0_topk, owner1_topk, sinks, query, state0, state1,
               (float *)d_control.data, n_tokens);
    launch_arm(ARM_ORDERED_LOCAL, full, owner0, owner1, owner1_base,
               topk, owner0_topk, owner1_topk, sinks, query, state0, state1,
               (float *)d_ordered.data, n_tokens);
    launch_arm(ARM_PARALLEL_SEQUENTIAL, full, owner0, owner1,
               owner1_base, topk, owner0_topk, owner1_topk,
               sinks, query, state0, state1,
               (float *)d_parallel.data, n_tokens);
    cuda_die(cudaDeviceSynchronize(), "run exactness arms");
    cuda_die(cudaMemcpy(host_control, d_control.data, output_bytes,
                        cudaMemcpyDeviceToHost), "copy control output");
    cuda_die(cudaMemcpy(host_ordered, d_ordered.data, output_bytes,
                        cudaMemcpyDeviceToHost), "copy ordered output");
    cuda_die(cudaMemcpy(host_parallel, d_parallel.data, output_bytes,
                        cudaMemcpyDeviceToHost), "copy parallel output");

    const uint64_t output_values =
        (uint64_t)n_tokens * N_HEAD * HEAD_DIM;
    uint64_t ordered_mismatches = 0u;
    uint64_t control_nonfinite = 0u;
    uint64_t control_nonzero = 0u;
    uint64_t parallel_mismatches = 0u;
    uint64_t parallel_nonfinite = 0u;
    double squared_error = 0.0;
    double reference_squared = 0.0;
    float max_abs = 0.0f;
    for (uint64_t i = 0; i < output_values; i++) {
        uint32_t a, b, c;
        memcpy(&a, host_control + i, sizeof(a));
        memcpy(&b, host_ordered + i, sizeof(b));
        memcpy(&c, host_parallel + i, sizeof(c));
        ordered_mismatches += a != b;
        control_nonfinite += !isfinite(host_control[i]);
        control_nonzero += host_control[i] != 0.0f;
        parallel_mismatches += a != c;
        parallel_nonfinite += !isfinite(host_parallel[i]);
        const double error =
            (double)host_parallel[i] - (double)host_control[i];
        squared_error += error * error;
        reference_squared +=
            (double)host_control[i] * (double)host_control[i];
        const float abs_error = fabsf(host_parallel[i] - host_control[i]);
        if (abs_error > max_abs) max_abs = abs_error;
    }
    if (ordered_mismatches != 0u || control_nonfinite != 0u ||
        control_nonzero == 0u || parallel_nonfinite != 0u) {
        fprintf(stderr,
                "error: ownership validation failed: ordered=%llu "
                "control_nonfinite=%llu control_nonzero=%llu "
                "parallel_nonfinite=%llu\n",
                (unsigned long long)ordered_mismatches,
                (unsigned long long)control_nonfinite,
                (unsigned long long)control_nonzero,
                (unsigned long long)parallel_nonfinite);
        return 1;
    }

    GuardedBuffer *buffers[] = {
        &d_full, &d_owner0, &d_owner1, &d_query, &d_sinks, &d_topk,
        &d_owner0_topk, &d_owner1_topk, &d_state0, &d_state1,
        &d_control, &d_ordered, &d_parallel,
    };
    const char *buffer_names[] = {
        "control cache", "owner0 cache", "owner1 cache", "query", "sinks",
        "top-k", "owner0 local top-k", "owner1 local top-k", "owner0 state",
        "owner1 state", "control output", "ordered output", "parallel output",
    };
    for (uint32_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++) {
        if (!check_guard(buffers[i], buffer_names[i])) return 1;
    }

    for (uint32_t i = 0; i < 3u; i++) {
        launch_arm(ARM_PARALLEL_SEQUENTIAL, full, owner0, owner1,
                   owner1_base, topk, owner0_topk, owner1_topk, sinks, query,
                   state0, state1, (float *)d_parallel.data, n_tokens);
    }
    cuda_die(cudaDeviceSynchronize(), "timing warmup");
    const float control_ms = median_arm(
        ARM_CONTROL, full, owner0, owner1, owner1_base,
        topk, owner0_topk, owner1_topk, sinks, query,
        state0, state1, (float *)d_control.data,
        n_tokens, rounds, repeats);
    const float ordered_ms = median_arm(
        ARM_ORDERED_LOCAL, full, owner0, owner1, owner1_base,
        topk, owner0_topk, owner1_topk, sinks, query,
        state0, state1, (float *)d_ordered.data,
        n_tokens, rounds, repeats);
    const float owner0_ms = median_arm(
        ARM_OWNER0, full, owner0, owner1, owner1_base,
        topk, owner0_topk, owner1_topk, sinks, query,
        state0, state1, (float *)d_parallel.data,
        n_tokens, rounds, repeats);
    const float owner1_ms = median_arm(
        ARM_OWNER1, full, owner0, owner1, owner1_base,
        topk, owner0_topk, owner1_topk, sinks, query,
        state0, state1, (float *)d_parallel.data,
        n_tokens, rounds, repeats);
    launch_arm(ARM_OWNER0, full, owner0, owner1, owner1_base,
               topk, owner0_topk, owner1_topk, sinks, query, state0, state1,
               (float *)d_parallel.data, n_tokens);
    launch_arm(ARM_OWNER1, full, owner0, owner1, owner1_base,
               topk, owner0_topk, owner1_topk, sinks, query, state0, state1,
               (float *)d_parallel.data, n_tokens);
    cuda_die(cudaDeviceSynchronize(), "seed merge timing");
    const float merge_ms = median_arm(
        ARM_MERGE, full, owner0, owner1, owner1_base,
        topk, owner0_topk, owner1_topk, sinks, query,
        state0, state1, (float *)d_parallel.data,
        n_tokens, rounds, repeats);
    const float sequential_ms = median_arm(
        ARM_PARALLEL_SEQUENTIAL, full, owner0, owner1,
        owner1_base, topk, owner0_topk, owner1_topk,
        sinks, query, state0, state1,
        (float *)d_parallel.data, n_tokens, rounds, repeats);
    const float projected_envelope_ms =
        fmaxf(owner0_ms, owner1_ms) + merge_ms;

    for (uint32_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++) {
        if (!check_guard(buffers[i], buffer_names[i])) return 1;
    }

    const uint64_t compact_cache_bytes = full_bytes;
    const uint64_t compact_cache_256k_bytes =
        (256ull * 1024ull / 4ull) * sizeof(CompactAttentionKVRow);
    const uint64_t state_handoff_logical_bytes = state_logical_bytes;
    const uint64_t state_handoff_aligned_bytes = state_storage_bytes;
    const uint64_t query_f32_bytes = query_bytes;
    const uint64_t query_f16_bytes = query_bytes / 2u;
    const uint64_t remote_selection_route_bytes = owner_topk_bytes;
    const uint64_t all_owner_selection_bytes = topk_bytes;
    const uint64_t compact_cache_43_layer_256k_bytes =
        compact_cache_256k_bytes * 43ull;
    const uint64_t compact_half_shard_43_layer_256k_bytes =
        compact_cache_43_layer_256k_bytes / 2ull;
    const uint64_t f32_cache_43_layer_256k_bytes =
        (256ull * 1024ull / 4ull) * HEAD_DIM * sizeof(float) * 43ull;
    printf("scenario=sm75-attention-rowshard-softmax\n");
    printf("scope=independent-single-gpu-protocol-emulation-not-production-dispatch\n");
    printf("device=%u\n", device);
    printf("device_name=%s\n", properties.name);
    printf("cache_ownership=two-separately-guarded-contiguous-row-shards\n");
    printf("selection_scope=prepartitioned-global-topk-routing-cost-excluded\n");
    printf("raw_ring_scope=excluded-must-remain-small-and-explicitly-replicated\n");
    printf("harness_control_copy_excluded_from_production_accounting=1\n");
    printf("n_rows=%u\n", n_rows);
    printf("n_tokens=%u\n", n_tokens);
    printf("n_heads=%u\n", N_HEAD);
    printf("heads_per_owner=%u\n", N_HEAD);
    printf("head_partition=none-row-owners-each-evaluate-all-heads\n");
    printf("selected_rows_per_token=%u\n", TOP_K);
    printf("selected_rows_per_owner_per_token=%u\n", TOP_K / 2u);
    printf("compact_row_bytes=%zu\n", sizeof(CompactAttentionKVRow));
    printf("one_authoritative_cache_bytes=%llu\n",
           (unsigned long long)compact_cache_bytes);
    printf("owner0_cache_bytes=%zu\n", owner0_bytes);
    printf("owner1_cache_bytes=%zu\n", owner1_bytes);
    printf("avoided_second_full_cache_bytes=%llu\n",
           (unsigned long long)compact_cache_bytes);
    printf("projected_one_layer_compact_cache_256k_bytes=%llu\n",
           (unsigned long long)compact_cache_256k_bytes);
    printf("projected_43_layer_compact_cache_256k_bytes=%llu\n",
           (unsigned long long)compact_cache_43_layer_256k_bytes);
    printf("projected_per_gpu_43_layer_half_shard_256k_bytes=%llu\n",
           (unsigned long long)compact_half_shard_43_layer_256k_bytes);
    printf("projected_43_layer_f32_cache_256k_bytes=%llu\n",
           (unsigned long long)f32_cache_43_layer_256k_bytes);
    printf("partial_state_payload=max-sum-numerator-f32-512\n");
    printf("partial_state_alignment_padding_floats_per_head=%u\n",
           STATE_HEADER_FLOATS - 2u);
    printf("remote_state_logical_payload_bytes=%llu\n",
           (unsigned long long)state_handoff_logical_bytes);
    printf("remote_state_aligned_transfer_bytes=%llu\n",
           (unsigned long long)state_handoff_aligned_bytes);
    printf("query_replication_f32_bytes=%llu\n",
           (unsigned long long)query_f32_bytes);
    printf("query_replication_f16_bytes=%llu\n",
           (unsigned long long)query_f16_bytes);
    printf("all_owner_selection_list_bytes=%llu\n",
           (unsigned long long)all_owner_selection_bytes);
    printf("remote_selection_route_bytes=%llu\n",
           (unsigned long long)remote_selection_route_bytes);
    printf("minimum_modeled_transport_f32_query_bytes=%llu\n",
           (unsigned long long)(state_handoff_aligned_bytes + query_f32_bytes +
                                remote_selection_route_bytes));
    printf("minimum_modeled_transport_f16_query_bytes=%llu\n",
           (unsigned long long)(state_handoff_aligned_bytes + query_f16_bytes +
                                remote_selection_route_bytes));
    printf("ordered_local_address_bit_mismatches=%llu\n",
           (unsigned long long)ordered_mismatches);
    printf("control_nonfinite=%llu\n",
           (unsigned long long)control_nonfinite);
    printf("control_nonzero=%llu\n",
           (unsigned long long)control_nonzero);
    printf("parallel_partial_bit_mismatches=%llu\n",
           (unsigned long long)parallel_mismatches);
    printf("parallel_partial_nonfinite=%llu\n",
           (unsigned long long)parallel_nonfinite);
    printf("parallel_partial_max_abs=%.9g\n", max_abs);
    printf("parallel_partial_relative_l2=%.9g\n",
           reference_squared == 0.0 ? 0.0 :
           sqrt(squared_error / reference_squared));
    printf("control_full_cache_median_ms=%.9g\n", control_ms);
    printf("ordered_local_address_median_ms=%.9g\n", ordered_ms);
    printf("owner0_partial_median_ms=%.9g\n", owner0_ms);
    printf("owner1_partial_median_ms=%.9g\n", owner1_ms);
    printf("partial_merge_median_ms=%.9g\n", merge_ms);
    printf("single_gpu_sequential_partial_median_ms=%.9g\n", sequential_ms);
    printf("parallel_projected_compute_envelope_before_transport_ms=%.9g\n",
           projected_envelope_ms);
    printf("parallel_projected_compute_speedup_before_transport=%.9g\n",
           control_ms / projected_envelope_ms);
    printf("ordered_address_exactness_eligible=1\n");
    printf("parallel_exactness_eligible=%u\n",
           parallel_mismatches == 0u ? 1u : 0u);
    printf("physical_multi_gpu_validation_required=1\n");
    printf("production_integration_eligible=0\n");
    printf("harness_status=ok\n");

    for (uint32_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++) {
        cudaFree(buffers[i]->base);
    }
    free(host_rows);
    free(host_query);
    free(host_sinks);
    free(host_topk);
    free(host_owner0_topk);
    free(host_owner1_topk);
    free(host_control);
    free(host_ordered);
    free(host_parallel);
    return 0;
}
