/* Production consumers for the explicitly tagged, size-neutral routed
 * Q4-32/Q3A4 layout.  Ordinary GGUF Q4_K/Q3_K tensors never enter here. */

__device__ __forceinline__ static int sm75_q32_scale6(
        const uint8_t packed[6], uint32_t group) {
    const uint32_t low = group < 4u
        ? packed[group] & 15u : packed[group - 4u] >> 4u;
    const uint32_t high = (packed[4u + (group >> 2u)] >>
                           (2u * (group & 3u))) & 3u;
    return (int)(low | (high << 4u)) - 32;
}

__device__ __forceinline__ static uint32_t sm75_q32_u4(
        const uint8_t packed[4], uint32_t group) {
    return (packed[group >> 1u] >> (4u * (group & 1u))) & 15u;
}

__device__ __forceinline__ static uint32_t sm75_q32_dilate_q3(
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

template <bool Q3A> struct sm75_q32_ops;

template <> struct sm75_q32_ops<false> {
    typedef cuda_sm75_q4_32_tile tile_type;
    __device__ __forceinline__ static const tile_type *record(
            const char *base, uint32_t expert, uint32_t row,
            uint32_t rows, uint32_t blocks, uint32_t block) {
        return (const tile_type *)base +
            (((uint64_t)expert * (rows / 8u) + row / 8u) * blocks + block);
    }
    __device__ __forceinline__ static float dot(
            const tile_type *w, uint32_t row8,
            const cuda_sm75_native_q8_K *a) {
        int total = 0;
#pragma unroll
        for (uint32_t group = 0; group < 8u; group++) {
            int sum = 0;
#pragma unroll
            for (uint32_t lane4 = 0; lane4 < 4u; lane4++) {
                const uint32_t qw = w->b[group][row8 * 4u + lane4];
                const uint32_t lo = a->low[group][lane4];
                const uint32_t hi = a->high_signed[group][lane4];
#pragma unroll
                for (uint32_t i = 0; i < 8u; i++) {
                    const int wc = (int)((qw >> (4u * i)) & 15u);
                    const int ws = (wc ^ 8) - 8;
                    const int al = (int)((lo >> (4u * i)) & 15u);
                    const int hc = (int)((hi >> (4u * i)) & 15u);
                    const int ah = (hc ^ 8) - 8;
                    sum += ws * (al + 16 * ah);
                }
            }
            total += sm75_q32_scale6(w->scales[row8], group) * sum;
        }
        return a->d * dev_f16_to_f32(w->d[row8]) * (float)total;
    }
    /* Decode-only scalar form.  Deliberately inhibit the complete unroll used
     * by the fused kernel: keeping one projection live at a time and one
     * packed word live per loop is the resource experiment.  Integer sums and
     * the single final float conversion retain dot()'s exact operation order. */
    __device__ __forceinline__ static float dot_decode_lowreg(
            const tile_type *w, uint32_t row8,
            const cuda_sm75_native_q8_K *a) {
        int total = 0;
#pragma unroll 1
        for (uint32_t group = 0; group < 8u; group++) {
            int sum = 0;
#pragma unroll 1
            for (uint32_t lane4 = 0; lane4 < 4u; lane4++) {
                const uint32_t qw = w->b[group][row8 * 4u + lane4];
                const uint32_t lo = a->low[group][lane4];
                const uint32_t hi = a->high_signed[group][lane4];
#pragma unroll 1
                for (uint32_t i = 0; i < 8u; i++) {
                    const int wc = (int)((qw >> (4u * i)) & 15u);
                    const int ws = (wc ^ 8) - 8;
                    const int al = (int)((lo >> (4u * i)) & 15u);
                    const int hc = (int)((hi >> (4u * i)) & 15u);
                    const int ah = (hc ^ 8) - 8;
                    sum += ws * (al + 16 * ah);
                }
            }
            total += sm75_q32_scale6(w->scales[row8], group) * sum;
        }
        return a->d * dev_f16_to_f32(w->d[row8]) * (float)total;
    }
    __device__ __forceinline__ static void mma_block(
            const tile_type *w, const cuda_sm75_native_q8_K *a,
            uint32_t lane, uint32_t n0, float *v0, float *v1) {
        int i0 = 0, i1 = 0;
#pragma unroll
        for (uint32_t group = 0; group < 8u; group++) {
            const uint32_t b = w->b[group][lane];
            int l0 = 0, l1 = 0, h0 = 0, h1 = 0;
            mma_m8n8k32_u4_s4(l0, l1, a->low[group][lane & 3u], b);
            mma_m8n8k32_s4_s4(h0, h1,
                              a->high_signed[group][lane & 3u], b);
            i0 += sm75_q32_scale6(w->scales[n0], group) *
                  (l0 + 16 * h0);
            i1 += sm75_q32_scale6(w->scales[n0 + 1u], group) *
                  (l1 + 16 * h1);
        }
        *v0 = a->d * dev_f16_to_f32(w->d[n0]) * (float)i0;
        *v1 = a->d * dev_f16_to_f32(w->d[n0 + 1u]) * (float)i1;
    }
};

template <> struct sm75_q32_ops<true> {
    typedef cuda_sm75_q3a4_tile tile_type;
    __device__ __forceinline__ static const tile_type *record(
            const char *base, uint32_t expert, uint32_t row,
            uint32_t rows, uint32_t blocks, uint32_t block) {
        return (const tile_type *)base +
            (((uint64_t)expert * (rows / 8u) + row / 8u) * blocks + block);
    }
    __device__ __forceinline__ static float dot(
            const tile_type *w, uint32_t row8,
            const cuda_sm75_native_q8_K *a) {
        int total = 0, correction = 0;
#pragma unroll
        for (uint32_t group = 0; group < 8u; group++) {
            int sum = 0;
#pragma unroll
            for (uint32_t lane4 = 0; lane4 < 4u; lane4++) {
                const uint32_t lane = row8 * 4u + lane4;
                const uint32_t qw = sm75_q32_dilate_q3(
                    w->low2[group][lane], w->high[group][lane]);
                const uint32_t lo = a->low[group][lane4];
                const uint32_t hi = a->high_signed[group][lane4];
#pragma unroll
                for (uint32_t i = 0; i < 8u; i++) {
                    const int wc = (int)((qw >> (4u * i)) & 15u);
                    const int al = (int)((lo >> (4u * i)) & 15u);
                    const int hc = (int)((hi >> (4u * i)) & 15u);
                    const int ah = (hc ^ 8) - 8;
                    sum += wc * (al + 16 * ah);
                }
            }
            const int bsum = (int)a->bsums[2u * group] +
                             (int)a->bsums[2u * group + 1u];
            total += (int)sm75_q32_u4(w->scales[row8], group) * sum;
            correction += (int)sm75_q32_u4(w->mins[row8], group) * bsum;
        }
        return a->d * (dev_f16_to_f32(w->d[row8]) * (float)total -
                       dev_f16_to_f32(w->dmin[row8]) * (float)correction);
    }
    __device__ __forceinline__ static float dot_decode_lowreg(
            const tile_type *w, uint32_t row8,
            const cuda_sm75_native_q8_K *a) {
        int total = 0, correction = 0;
#pragma unroll 1
        for (uint32_t group = 0; group < 8u; group++) {
            int sum = 0;
#pragma unroll 1
            for (uint32_t lane4 = 0; lane4 < 4u; lane4++) {
                const uint32_t lane = row8 * 4u + lane4;
                const uint32_t qw = sm75_q32_dilate_q3(
                    w->low2[group][lane], w->high[group][lane]);
                const uint32_t lo = a->low[group][lane4];
                const uint32_t hi = a->high_signed[group][lane4];
#pragma unroll 1
                for (uint32_t i = 0; i < 8u; i++) {
                    const int wc = (int)((qw >> (4u * i)) & 15u);
                    const int al = (int)((lo >> (4u * i)) & 15u);
                    const int hc = (int)((hi >> (4u * i)) & 15u);
                    const int ah = (hc ^ 8) - 8;
                    sum += wc * (al + 16 * ah);
                }
            }
            const int bsum = (int)a->bsums[2u * group] +
                             (int)a->bsums[2u * group + 1u];
            total += (int)sm75_q32_u4(w->scales[row8], group) * sum;
            correction += (int)sm75_q32_u4(w->mins[row8], group) * bsum;
        }
        return a->d * (dev_f16_to_f32(w->d[row8]) * (float)total -
                       dev_f16_to_f32(w->dmin[row8]) * (float)correction);
    }
    __device__ __forceinline__ static void mma_block(
            const tile_type *w, const cuda_sm75_native_q8_K *a,
            uint32_t lane, uint32_t n0, float *v0, float *v1) {
        int i0 = 0, i1 = 0, m0 = 0, m1 = 0;
#pragma unroll
        for (uint32_t group = 0; group < 8u; group++) {
            const uint32_t b = sm75_q32_dilate_q3(
                w->low2[group][lane], w->high[group][lane]);
            int l0 = 0, l1 = 0, h0 = 0, h1 = 0;
            mma_m8n8k32_u4_u4(l0, l1, a->low[group][lane & 3u], b);
            mma_m8n8k32_s4_u4(h0, h1,
                              a->high_signed[group][lane & 3u], b);
            const int c0 = l0 + 16 * h0;
            const int c1 = l1 + 16 * h1;
            const int bsum = (int)a->bsums[2u * group] +
                             (int)a->bsums[2u * group + 1u];
            i0 += (int)sm75_q32_u4(w->scales[n0], group) * c0;
            i1 += (int)sm75_q32_u4(w->scales[n0 + 1u], group) * c1;
            m0 += (int)sm75_q32_u4(w->mins[n0], group) * bsum;
            m1 += (int)sm75_q32_u4(w->mins[n0 + 1u], group) * bsum;
        }
        *v0 = a->d * (dev_f16_to_f32(w->d[n0]) * (float)i0 -
                      dev_f16_to_f32(w->dmin[n0]) * (float)m0);
        *v1 = a->d * (dev_f16_to_f32(w->d[n0 + 1u]) * (float)i1 -
                      dev_f16_to_f32(w->dmin[n0 + 1u]) * (float)m1);
    }
};

template <bool Q3A>
__global__ static void moe_gate_up_mid_decode_sm75_q32_kernel(
        float *gate_out, float *up_out, float *mid_out,
        const char *gate_base, const char *up_base,
        const cuda_sm75_native_q8_K *xq,
        const int32_t *selected, const float *weights,
        uint32_t xq_blocks, uint32_t expert_mid_dim,
        uint32_t n_expert, uint32_t write_aux, float clamp) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t ei = selected[(uint64_t)tok * n_expert + slot];
    if (ei < 0) ei = 0;
    const uint32_t expert = (uint32_t)ei;
    const cuda_sm75_native_q8_K *a = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f, up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        gate += sm75_q32_ops<Q3A>::dot(
            sm75_q32_ops<Q3A>::record(gate_base, expert, row,
                                      expert_mid_dim, xq_blocks, b),
            row & 7u, a + b);
        up += sm75_q32_ops<Q3A>::dot(
            sm75_q32_ops<Q3A>::record(up_base, expert, row,
                                      expert_mid_dim, xq_blocks, b),
            row & 7u, a + b);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        if (write_aux) { gate_out[off] = gate; up_out[off] = up; }
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
            weights[(uint64_t)tok * n_expert + slot];
    }
}

template <bool Q3A>
__global__ static void moe_gate_up_mid_decode_sm75_q32_owned_kernel(
        float *mid_out, const char *gate_base, const char *up_base,
        const cuda_sm75_native_q8_K *xq,
        const int32_t *selected, const float *weights,
        uint32_t xq_blocks, uint32_t expert_mid_dim,
        uint32_t expert_base, uint32_t expert_count, float clamp) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t slot = blockIdx.y;
    if (row >= expert_mid_dim || slot >= 6u) return;
    uint32_t expert = 0;
    if (!moe_owned_local_expert(selected[slot], expert_base,
                                expert_count, &expert)) {
        if (lane == 0u)
            mid_out[(uint64_t)slot * expert_mid_dim + row] = 0.0f;
        return;
    }
    float gate = 0.0f, up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        gate += sm75_q32_ops<Q3A>::dot(
            sm75_q32_ops<Q3A>::record(gate_base, expert, row,
                                      expert_mid_dim, xq_blocks, b),
            row & 7u, xq + b);
        up += sm75_q32_ops<Q3A>::dot(
            sm75_q32_ops<Q3A>::record(up_base, expert, row,
                                      expert_mid_dim, xq_blocks, b),
            row & 7u, xq + b);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        mid_out[(uint64_t)slot * expert_mid_dim + row] =
            (gate / (1.0f + expf(-gate))) * up * weights[slot];
    }
}

/* Gate and up have distinct template identities so profiling can account for
 * both launches independently even though their projection arithmetic is
 * intentionally identical. */
template <bool Q3A, bool IS_UP>
__global__ __launch_bounds__(256, 2) static void
moe_gate_up_mid_decode_sm75_q32_projection_owned_kernel(
        float *projection_out, const char *projection_base,
        const cuda_sm75_native_q8_K *xq, const int32_t *selected,
        uint32_t xq_blocks, uint32_t expert_mid_dim,
        uint32_t expert_base, uint32_t expert_count) {
    (void)IS_UP;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t slot = blockIdx.y;
    if (row >= expert_mid_dim || slot >= 6u) return;
    uint32_t expert = 0;
    if (!moe_owned_local_expert(selected[slot], expert_base,
                                expert_count, &expert)) {
        if (lane == 0u)
            projection_out[(uint64_t)slot * expert_mid_dim + row] = 0.0f;
        return;
    }
    float projection = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        projection += sm75_q32_ops<Q3A>::dot_decode_lowreg(
            sm75_q32_ops<Q3A>::record(
                projection_base, expert, row, expert_mid_dim, xq_blocks, b),
            row & 7u, xq + b);
    }
    projection = warp_sum_f32(projection);
    if (lane == 0u)
        projection_out[(uint64_t)slot * expert_mid_dim + row] = projection;
}

__global__ __launch_bounds__(256, 2) static void
moe_gate_up_mid_decode_sm75_q32_combine_owned_kernel(
        float *mid_out, const float *gate_in, const float *up_in,
        const int32_t *selected, const float *weights,
        uint32_t expert_mid_dim, uint32_t expert_base,
        uint32_t expert_count, float clamp) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t count = 6u * expert_mid_dim;
    if (index >= count) return;
    const uint32_t slot = index / expert_mid_dim;
    uint32_t expert = 0;
    if (!moe_owned_local_expert(selected[slot], expert_base,
                                expert_count, &expert)) {
        mid_out[index] = 0.0f;
        return;
    }
    float gate = gate_in[index];
    float up = up_in[index];
    if (clamp > 1.0e-6f) {
        if (gate > clamp) gate = clamp;
        if (up > clamp) up = clamp;
        if (up < -clamp) up = -clamp;
    }
    mid_out[index] = (gate / (1.0f + expf(-gate))) * up * weights[slot];
}

template <uint32_t NSLOT>
__global__ static void moe_down_sm75_q4_32_sum_kernel(
        float *out, const char *down_base,
        const cuda_sm75_native_q8_K *midq, const int32_t *selected,
        uint32_t midq_blocks, uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
#pragma unroll
    for (uint32_t slot = 0; slot < NSLOT; slot++) {
        int32_t ei = selected[slot];
        if (ei < 0) ei = 0;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            acc += sm75_q32_ops<false>::dot(
                sm75_q32_ops<false>::record(
                    down_base, (uint32_t)ei, row, out_dim, midq_blocks, b),
                row & 7u, midq + (uint64_t)slot * midq_blocks + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0u) total += acc;
    }
    if (lane == 0u) out[row] = total;
}

__global__ static void moe_down_sm75_q4_32_pair_kernel(
        float *out, const char *down_base,
        const cuda_sm75_native_q8_K *midq, const int32_t *selected,
        uint32_t midq_blocks, uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    int32_t ei = selected[pair];
    if (ei < 0) ei = 0;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        acc += sm75_q32_ops<false>::dot(
            sm75_q32_ops<false>::record(
                down_base, (uint32_t)ei, row, out_dim, midq_blocks, b),
            row & 7u, midq + (uint64_t)pair * midq_blocks + b);
    }
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0u) out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_down_sm75_q4_32_owned_slots_kernel(
        float *out, const char *down_base,
        const cuda_sm75_native_q8_K *midq, const int32_t *selected,
        uint32_t midq_blocks, uint32_t out_dim,
        uint32_t expert_base, uint32_t expert_count) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t slot = blockIdx.y;
    if (row >= out_dim || slot >= 6u) return;
    uint32_t expert = 0;
    if (!moe_owned_local_expert(selected[slot], expert_base,
                                expert_count, &expert)) {
        if (lane == 0u) out[(uint64_t)slot * out_dim + row] = 0.0f;
        return;
    }
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        acc += sm75_q32_ops<false>::dot(
            sm75_q32_ops<false>::record(
                down_base, expert, row, out_dim, midq_blocks, b),
            row & 7u, midq + (uint64_t)slot * midq_blocks + b);
    }
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0u) out[(uint64_t)slot * out_dim + row] = acc;
}

__global__ static void moe_down_sm75_q4_32_owned_packed_kernel(
        float *out, const char *down_base,
        const cuda_sm75_native_q8_K *midq, const int32_t *selected,
        uint32_t midq_blocks, uint32_t out_dim,
        uint32_t expert_base, uint32_t expert_count) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t packed_slot = blockIdx.y;
    if (row >= out_dim || packed_slot >= 4u) return;
    bool prefix_pair = false;
    const int first = moe_owned_packed_component(
        selected, packed_slot / 2u, packed_slot & 1u,
        expert_base, expert_count, &prefix_pair);
    if (first < 0) {
        if (lane == 0u) out[(uint64_t)packed_slot * out_dim + row] = 0.0f;
        return;
    }
    float total = 0.0f;
    const uint32_t count = prefix_pair ? 2u : 1u;
#pragma unroll
    for (uint32_t i = 0; i < 2u; i++) {
        if (i >= count) break;
        const uint32_t slot = (uint32_t)first + i;
        uint32_t expert = 0;
        if (!moe_owned_local_expert(selected[slot], expert_base,
                                    expert_count, &expert)) continue;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            acc += sm75_q32_ops<false>::dot(
                sm75_q32_ops<false>::record(
                    down_base, expert, row, out_dim, midq_blocks, b),
                row & 7u, midq + (uint64_t)slot * midq_blocks + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0u) total = prefix_pair ? __fadd_rn(total, acc) : acc;
    }
    if (lane == 0u) out[(uint64_t)packed_slot * out_dim + row] = total;
}

#define DS4_SM75_Q32_SLOT4_DECL(S) \
    float s0_##S=0.0f,s1_##S=0.0f,s2_##S=0.0f,s3_##S=0.0f
#define DS4_SM75_Q32_SLOT4_ADD(S,A,B,C,D) \
    case S:s0_##S+=(A);s1_##S+=(B);s2_##S+=(C);s3_##S+=(D);break
#define DS4_SM75_Q32_REDUCE(P,O) do { \
    const float _a0=P##_0+P##_4,_a1=P##_1+P##_5; \
    const float _a2=P##_2+P##_6,_a3=P##_3+P##_7; \
    (O)=(_a0+_a2)+(_a1+_a3); \
} while (0)

template <uint32_t ROW_SPAN, uint32_t TILE_DELTA, bool Q3A>
__global__ static void moe_gate_up_mid_sm75_q32_tile8_kernel(
        float *gate_out, float *up_out, float *mid_out,
        const char *gate_base, const char *up_base,
        const cuda_sm75_native_q8_K *xq,
        const uint32_t *sorted_pairs, const uint32_t *offsets,
        const uint32_t *counts, const uint32_t *tile_total,
        const uint32_t *tile_experts, const uint32_t *tile_starts,
        const float *weights, uint32_t xq_blocks,
        uint32_t expert_mid_dim, uint32_t n_expert,
        uint32_t write_aux, float clamp) {
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t expert = tile_experts[tile];
    const uint32_t start = tile_starts[tile] + TILE_DELTA;
    __shared__ cuda_sm75_native_q8_K sxq[8][16];
    __shared__ uint32_t s_pair[8], s_tok[8], s_slot[8], s_np;
    if (threadIdx.x == 0u) {
        uint32_t np = 0;
        for (; np < 8u; np++) {
            const uint32_t local = start + np;
            if (local >= counts[expert]) break;
            const uint32_t pair = sorted_pairs[offsets[expert] + local];
            s_pair[np] = pair;
            s_tok[np] = pair / n_expert;
            s_slot[np] = pair - s_tok[np] * n_expert;
        }
        s_np = np;
    }
    __syncthreads();
    const uint32_t np = s_np;
    const uint32_t words = sizeof(cuda_sm75_native_q8_K) / 4u;
    const uint32_t words_per_tok = xq_blocks * words;
    for (uint32_t i = threadIdx.x; i < 8u * words_per_tok;
         i += blockDim.x) {
        const uint32_t p = i / words_per_tok;
        const uint32_t w = i - p * words_per_tok;
        ((uint32_t *)sxq[p])[w] = p < np
            ? ((const uint32_t *)(xq + (uint64_t)s_tok[p] * xq_blocks))[w]
            : 0u;
    }
    __syncthreads();
    const uint32_t mtok = lane >> 2u;
    const uint32_t n0 = (lane & 3u) * 2u;
    for (uint32_t rr = 0; rr < ROW_SPAN / 128u; rr++) {
        const uint32_t row0 = blockIdx.x * ROW_SPAN + rr * 128u + warp * 8u;
        if (row0 >= expert_mid_dim) continue;
        DS4_SM75_Q32_SLOT4_DECL(0); DS4_SM75_Q32_SLOT4_DECL(1);
        DS4_SM75_Q32_SLOT4_DECL(2); DS4_SM75_Q32_SLOT4_DECL(3);
        DS4_SM75_Q32_SLOT4_DECL(4); DS4_SM75_Q32_SLOT4_DECL(5);
        DS4_SM75_Q32_SLOT4_DECL(6); DS4_SM75_Q32_SLOT4_DECL(7);
        const uint64_t nt = (uint64_t)expert * (expert_mid_dim / 8u) +
                            row0 / 8u;
        const cuda_sm75_native_q8_K *a = &sxq[mtok][0];
        for (uint32_t b = 0; b < xq_blocks; b++) {
            const typename sm75_q32_ops<Q3A>::tile_type *gw =
                (const typename sm75_q32_ops<Q3A>::tile_type *)gate_base +
                nt * xq_blocks + b;
            const typename sm75_q32_ops<Q3A>::tile_type *uw =
                (const typename sm75_q32_ops<Q3A>::tile_type *)up_base +
                nt * xq_blocks + b;
            float g0, g1, u0, u1;
            sm75_q32_ops<Q3A>::mma_block(gw, a + b, lane, n0, &g0, &g1);
            sm75_q32_ops<Q3A>::mma_block(uw, a + b, lane, n0, &u0, &u1);
            switch (b & 7u) {
                DS4_SM75_Q32_SLOT4_ADD(0,g0,g1,u0,u1);
                DS4_SM75_Q32_SLOT4_ADD(1,g0,g1,u0,u1);
                DS4_SM75_Q32_SLOT4_ADD(2,g0,g1,u0,u1);
                DS4_SM75_Q32_SLOT4_ADD(3,g0,g1,u0,u1);
                DS4_SM75_Q32_SLOT4_ADD(4,g0,g1,u0,u1);
                DS4_SM75_Q32_SLOT4_ADD(5,g0,g1,u0,u1);
                DS4_SM75_Q32_SLOT4_ADD(6,g0,g1,u0,u1);
                DS4_SM75_Q32_SLOT4_ADD(7,g0,g1,u0,u1);
            }
        }
        float gate0,gate1,up0,up1;
        DS4_SM75_Q32_REDUCE(s0,gate0); DS4_SM75_Q32_REDUCE(s1,gate1);
        DS4_SM75_Q32_REDUCE(s2,up0); DS4_SM75_Q32_REDUCE(s3,up1);
        if (mtok < np) {
#pragma unroll
            for (uint32_t e = 0; e < 2u; e++) {
                const uint32_t row = row0 + n0 + e;
                if (row >= expert_mid_dim) continue;
                float g = e ? gate1 : gate0;
                float u = e ? up1 : up0;
                if (clamp > 1.0e-6f) {
                    if (g > clamp) g = clamp;
                    if (u > clamp) u = clamp;
                    if (u < -clamp) u = -clamp;
                }
                const uint64_t off =
                    (uint64_t)s_pair[mtok] * expert_mid_dim + row;
                if (write_aux) { gate_out[off]=g; up_out[off]=u; }
                mid_out[off] = (g / (1.0f + expf(-g))) * u *
                    weights[(uint64_t)s_tok[mtok] * n_expert + s_slot[mtok]];
            }
        }
    }
}

template <uint32_t ROW_SPAN, uint32_t TILE_PAIRS>
__global__ static void moe_down_sm75_q4_32_tile_kernel(
        float *down_out, const char *down_base,
        const cuda_sm75_native_q8_K *midq,
        const uint32_t *sorted_pairs, const uint32_t *offsets,
        const uint32_t *counts, const uint32_t *tile_total,
        const uint32_t *tile_experts, const uint32_t *tile_starts,
        uint32_t midq_blocks, uint32_t out_dim) {
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t expert = tile_experts[tile];
    const uint32_t start = tile_starts[tile];
    __shared__ cuda_sm75_native_q8_K sxq[TILE_PAIRS][8];
    __shared__ uint32_t s_pair[TILE_PAIRS], s_np;
    if (threadIdx.x == 0u) {
        uint32_t np = 0;
        for (; np < TILE_PAIRS; np++) {
            const uint32_t local = start + np;
            if (local >= counts[expert]) break;
            s_pair[np] = sorted_pairs[offsets[expert] + local];
        }
        s_np = np;
    }
    __syncthreads();
    const uint32_t np = s_np;
    const uint32_t words = sizeof(cuda_sm75_native_q8_K) / 4u;
    const uint32_t words_per_pair = midq_blocks * words;
    for (uint32_t i = threadIdx.x; i < TILE_PAIRS * words_per_pair;
         i += blockDim.x) {
        const uint32_t p = i / words_per_pair;
        const uint32_t w = i - p * words_per_pair;
        ((uint32_t *)sxq[p])[w] = p < np
            ? ((const uint32_t *)(midq +
                (uint64_t)s_pair[p] * midq_blocks))[w] : 0u;
    }
    __syncthreads();
    const uint32_t p0 = lane >> 2u;
    const uint32_t p1 = p0 + 8u;
    const uint32_t n0 = (lane & 3u) * 2u;
    const uint32_t row_stride = (blockDim.x >> 5u) * 8u;
    for (uint32_t base = 0; base < ROW_SPAN; base += row_stride) {
        const uint32_t row0 = blockIdx.x * ROW_SPAN + base + warp * 8u;
        if (base + warp * 8u >= ROW_SPAN || row0 >= out_dim) continue;
        DS4_SM75_Q32_SLOT4_DECL(0); DS4_SM75_Q32_SLOT4_DECL(1);
        DS4_SM75_Q32_SLOT4_DECL(2); DS4_SM75_Q32_SLOT4_DECL(3);
        DS4_SM75_Q32_SLOT4_DECL(4); DS4_SM75_Q32_SLOT4_DECL(5);
        DS4_SM75_Q32_SLOT4_DECL(6); DS4_SM75_Q32_SLOT4_DECL(7);
        const uint64_t nt = (uint64_t)expert * (out_dim / 8u) + row0 / 8u;
#pragma unroll
        for (uint32_t b = 0; b < 8u; b++) {
            if (b >= midq_blocks) continue;
            const cuda_sm75_q4_32_tile *w =
                (const cuda_sm75_q4_32_tile *)down_base +
                nt * midq_blocks + b;
            const cuda_sm75_native_q8_K *a0 =
                p0 < TILE_PAIRS ? &sxq[p0][b] : &sxq[0][b];
            const cuda_sm75_native_q8_K *a1 =
                TILE_PAIRS > 8u ? &sxq[p1][b] : a0;
            float v00,v01,v10=0.0f,v11=0.0f;
            sm75_q32_ops<false>::mma_block(w,a0,lane,n0,&v00,&v01);
            if (TILE_PAIRS > 8u)
                sm75_q32_ops<false>::mma_block(w,a1,lane,n0,&v10,&v11);
            switch (b) {
                DS4_SM75_Q32_SLOT4_ADD(0,v00,v01,v10,v11);
                DS4_SM75_Q32_SLOT4_ADD(1,v00,v01,v10,v11);
                DS4_SM75_Q32_SLOT4_ADD(2,v00,v01,v10,v11);
                DS4_SM75_Q32_SLOT4_ADD(3,v00,v01,v10,v11);
                DS4_SM75_Q32_SLOT4_ADD(4,v00,v01,v10,v11);
                DS4_SM75_Q32_SLOT4_ADD(5,v00,v01,v10,v11);
                DS4_SM75_Q32_SLOT4_ADD(6,v00,v01,v10,v11);
                DS4_SM75_Q32_SLOT4_ADD(7,v00,v01,v10,v11);
            }
        }
        float o0,o1,o2,o3;
        DS4_SM75_Q32_REDUCE(s0,o0); DS4_SM75_Q32_REDUCE(s1,o1);
        DS4_SM75_Q32_REDUCE(s2,o2); DS4_SM75_Q32_REDUCE(s3,o3);
        if (p0 < TILE_PAIRS && p0 < np) {
            const uint32_t r0=row0+n0,r1=r0+1u;
            if (r0<out_dim) down_out[(uint64_t)s_pair[p0]*out_dim+r0]=o0;
            if (r1<out_dim) down_out[(uint64_t)s_pair[p0]*out_dim+r1]=o1;
        }
        if (p1 < TILE_PAIRS && p1 < np) {
            const uint32_t r0=row0+n0,r1=r0+1u;
            if (r0<out_dim) down_out[(uint64_t)s_pair[p1]*out_dim+r0]=o2;
            if (r1<out_dim) down_out[(uint64_t)s_pair[p1]*out_dim+r1]=o3;
        }
    }
}

#undef DS4_SM75_Q32_REDUCE
#undef DS4_SM75_Q32_SLOT4_ADD
#undef DS4_SM75_Q32_SLOT4_DECL
