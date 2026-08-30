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
    /* Decode-only partial-unroll forms.  UNROLL is deliberately applied at
     * every fixed loop level: u1 is the minimum-register control, while u2
     * and u4 restore progressively more independent integer work without
     * returning to the fully-unrolled 255-register production kernel.
     * Integer accumulation and the one final float conversion retain dot()'s
     * exact operation order. */
    template <uint32_t UNROLL>
    __device__ __forceinline__ static float dot_decode_partial(
            const tile_type *w, uint32_t row8,
            const cuda_sm75_native_q8_K *a) {
        static_assert(UNROLL == 1u || UNROLL == 2u || UNROLL == 4u,
                      "unsupported Q4-32 decode unroll");
        int total = 0;
#pragma unroll UNROLL
        for (uint32_t group = 0; group < 8u; group++) {
            int sum = 0;
#pragma unroll UNROLL
            for (uint32_t lane4 = 0; lane4 < 4u; lane4++) {
                const uint32_t qw = w->b[group][row8 * 4u + lane4];
                const uint32_t lo = a->low[group][lane4];
                const uint32_t hi = a->high_signed[group][lane4];
#pragma unroll UNROLL
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
    __device__ __forceinline__ static float dot_decode_lowreg(
            const tile_type *w, uint32_t row8,
            const cuda_sm75_native_q8_K *a) {
        return dot_decode_partial<1u>(w, row8, a);
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

    struct mma_group_operands {
        uint32_t b;
        uint32_t lo;
        uint32_t hi;
    };

    template <uint32_t GROUP>
    __device__ __forceinline__ static mma_group_operands mma_load_group(
            const tile_type *w, const cuda_sm75_native_q8_K *a,
            uint32_t lane) {
        static_assert(GROUP < 8u, "Q4-32 MMA group must be in [0, 8)");
        mma_group_operands op;
        op.b = w->b[GROUP][lane];
        op.lo = a->low[GROUP][lane & 3u];
        op.hi = a->high_signed[GROUP][lane & 3u];
        return op;
    }

    template <uint32_t GROUP>
    __device__ __forceinline__ static void mma_consume_group(
            mma_group_operands op, const tile_type *w, uint32_t n0,
            int *i0, int *i1) {
        static_assert(GROUP < 8u, "Q4-32 MMA group must be in [0, 8)");
        int l0 = 0, l1 = 0, h0 = 0, h1 = 0;
        mma_m8n8k32_u4_s4(l0, l1, op.lo, op.b);
        mma_m8n8k32_s4_s4(h0, h1, op.hi, op.b);
        *i0 += sm75_q32_scale6(w->scales[n0], GROUP) *
               (l0 + 16 * h0);
        *i1 += sm75_q32_scale6(w->scales[n0 + 1u], GROUP) *
               (l1 + 16 * h1);
    }

    /* Group-level software pipelines for the packed-INT4 MMA path.  Loads
     * for one or two future groups are issued before the current MMA work,
     * while consume order remains explicitly 0..7.  Integer accumulation
     * and the final float conversion therefore remain byte-identical to
     * mma_block(). */
    template <uint32_t PREFETCH_DEPTH>
    __device__ __forceinline__ static void mma_block_prefetch(
            const tile_type *w, const cuda_sm75_native_q8_K *a,
            uint32_t lane, uint32_t n0, float *v0, float *v1) {
        static_assert(PREFETCH_DEPTH == 1u || PREFETCH_DEPTH == 2u,
                      "Q4-32 MMA prefetch depth must be 1 or 2");
        int i0 = 0, i1 = 0;
        mma_group_operands op0 = mma_load_group<0u>(w, a, lane);
        mma_group_operands op1 = mma_load_group<1u>(w, a, lane);
        if (PREFETCH_DEPTH == 1u) {
            mma_consume_group<0u>(op0, w, n0, &i0, &i1);
            op0 = mma_load_group<2u>(w, a, lane);
            mma_consume_group<1u>(op1, w, n0, &i0, &i1);
            op1 = mma_load_group<3u>(w, a, lane);
            mma_consume_group<2u>(op0, w, n0, &i0, &i1);
            op0 = mma_load_group<4u>(w, a, lane);
            mma_consume_group<3u>(op1, w, n0, &i0, &i1);
            op1 = mma_load_group<5u>(w, a, lane);
            mma_consume_group<4u>(op0, w, n0, &i0, &i1);
            op0 = mma_load_group<6u>(w, a, lane);
            mma_consume_group<5u>(op1, w, n0, &i0, &i1);
            op1 = mma_load_group<7u>(w, a, lane);
            mma_consume_group<6u>(op0, w, n0, &i0, &i1);
            mma_consume_group<7u>(op1, w, n0, &i0, &i1);
        } else {
            mma_group_operands op2 = mma_load_group<2u>(w, a, lane);
            mma_consume_group<0u>(op0, w, n0, &i0, &i1);
            op0 = mma_load_group<3u>(w, a, lane);
            mma_consume_group<1u>(op1, w, n0, &i0, &i1);
            op1 = mma_load_group<4u>(w, a, lane);
            mma_consume_group<2u>(op2, w, n0, &i0, &i1);
            op2 = mma_load_group<5u>(w, a, lane);
            mma_consume_group<3u>(op0, w, n0, &i0, &i1);
            op0 = mma_load_group<6u>(w, a, lane);
            mma_consume_group<4u>(op1, w, n0, &i0, &i1);
            op1 = mma_load_group<7u>(w, a, lane);
            mma_consume_group<5u>(op2, w, n0, &i0, &i1);
            mma_consume_group<6u>(op0, w, n0, &i0, &i1);
            mma_consume_group<7u>(op1, w, n0, &i0, &i1);
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
    template <uint32_t UNROLL>
    __device__ __forceinline__ static float dot_decode_partial(
            const tile_type *w, uint32_t row8,
            const cuda_sm75_native_q8_K *a) {
        static_assert(UNROLL == 1u || UNROLL == 2u || UNROLL == 4u,
                      "unsupported Q3A4 decode unroll");
        int total = 0, correction = 0;
#pragma unroll UNROLL
        for (uint32_t group = 0; group < 8u; group++) {
            int sum = 0;
#pragma unroll UNROLL
            for (uint32_t lane4 = 0; lane4 < 4u; lane4++) {
                const uint32_t lane = row8 * 4u + lane4;
                const uint32_t qw = sm75_q32_dilate_q3(
                    w->low2[group][lane], w->high[group][lane]);
                const uint32_t lo = a->low[group][lane4];
                const uint32_t hi = a->high_signed[group][lane4];
#pragma unroll UNROLL
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
        return dot_decode_partial<1u>(w, row8, a);
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

/* Q3A4-only decode control: the production shape has exactly 16 K256
 * records.  Two independent rows therefore occupy the two 16-lane halves of
 * each warp instead of leaving lanes 16..31 idle.  The dot expression and
 * the 8/4/2/1 float-reduction tree are unchanged. */
__global__ static void moe_gate_up_mid_decode_sm75_q3a4_hwarp16_owned_kernel(
        float *mid_out, const char *gate_base, const char *up_base,
        const cuda_sm75_native_q8_K *xq,
        const int32_t *selected, const float *weights,
        uint32_t xq_blocks, uint32_t expert_mid_dim,
        uint32_t expert_base, uint32_t expert_count, float clamp) {
    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    const uint32_t slot = blockIdx.y;
    if (row >= expert_mid_dim || slot >= 6u) return;
    uint32_t expert = 0;
    if (!moe_owned_local_expert(selected[slot], expert_base,
                                expert_count, &expert)) {
        if (lane16 == 0u)
            mid_out[(uint64_t)slot * expert_mid_dim + row] = 0.0f;
        return;
    }
    float gate = 0.0f, up = 0.0f;
    for (uint32_t b = lane16; b < xq_blocks; b += 16u) {
        gate += sm75_q32_ops<true>::dot_decode_partial<2u>(
            sm75_q32_ops<true>::record(gate_base, expert, row,
                                       expert_mid_dim, xq_blocks, b),
            row & 7u, xq + b);
        up += sm75_q32_ops<true>::dot_decode_partial<2u>(
            sm75_q32_ops<true>::record(up_base, expert, row,
                                       expert_mid_dim, xq_blocks, b),
            row & 7u, xq + b);
    }
    gate = half_warp_sum_f32(gate, lane16);
    up = half_warp_sum_f32(up, lane16);
    if (lane16 == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        mid_out[(uint64_t)slot * expert_mid_dim + row] =
            (gate / (1.0f + expf(-gate))) * up * weights[slot];
    }
}

/* Q4-32 decode half-warp diagnostic.  Like the Q3A4 experiment above, the
 * shipping decode shape has exactly 16 K256 records, so two rows can share a
 * warp without changing either the per-record dot expression or the original
 * 8/4/2/1 floating-point reduction tree.  This remains explicitly selected;
 * tile32 packed-INT4 MMA is the production Q4-32 default. */
__global__ static void moe_gate_up_mid_decode_sm75_q4_32_hwarp16_owned_kernel(
        float *mid_out, const char *gate_base, const char *up_base,
        const cuda_sm75_native_q8_K *xq,
        const int32_t *selected, const float *weights,
        uint32_t xq_blocks, uint32_t expert_mid_dim,
        uint32_t expert_base, uint32_t expert_count, float clamp) {
    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    const uint32_t slot = blockIdx.y;
    if (row >= expert_mid_dim || slot >= 6u) return;
    uint32_t expert = 0;
    if (!moe_owned_local_expert(selected[slot], expert_base,
                                expert_count, &expert)) {
        if (lane16 == 0u)
            mid_out[(uint64_t)slot * expert_mid_dim + row] = 0.0f;
        return;
    }
    float gate = 0.0f, up = 0.0f;
    for (uint32_t b = lane16; b < xq_blocks; b += 16u) {
        gate += sm75_q32_ops<false>::dot_decode_partial<2u>(
            sm75_q32_ops<false>::record(gate_base, expert, row,
                                        expert_mid_dim, xq_blocks, b),
            row & 7u, xq + b);
        up += sm75_q32_ops<false>::dot_decode_partial<2u>(
            sm75_q32_ops<false>::record(up_base, expert, row,
                                        expert_mid_dim, xq_blocks, b),
            row & 7u, xq + b);
    }
    /* The control warp performs an offset-16 add against a +0 lane before
     * its 8/4/2/1 tree.  Preserve that signed-zero normalization even though
     * this half-warp mapping has no offset-16 stage. */
    gate = __fadd_rn(0.0f, gate);
    up = __fadd_rn(0.0f, up);
    gate = half_warp_sum_f32(gate, lane16);
    up = half_warp_sum_f32(up, lane16);
    if (lane16 == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        mid_out[(uint64_t)slot * expert_mid_dim + row] =
            (gate / (1.0f + expf(-gate))) * up * weights[slot];
    }
}

__device__ __forceinline__ static uint32_t
sm75_q4_32_sign_extend_nibble_bytes(uint32_t v) {
    v &= 0x0f0f0f0fu;
    const uint32_t sign = v & 0x08080808u;
    return v | (sign << 1u) | (sign << 2u) |
               (sign << 3u) | (sign << 4u);
}

/* Native Q4-32 one-token tile mappings.  A warp follows one 8-row weight
 * tile.  The DP4A diagnostic expands four signed Q4 nibbles to four
 * signed bytes per instruction.  The packed-MMA specialization feeds the
 * native Q4 words directly to Turing m8n8k32 INT4 MMA and replicates the one
 * decode activation across MMA's eight M rows; lanes 0..3 retain the single
 * unique row. It is the production default. Both mappings stage all 16 K256
 * leaves and reproduce the control warp-sum tree exactly before
 * SiLU/multiply/weight combine. */
template <bool PACKED_MMA, uint32_t PREFETCH_DEPTH = 0u>
__global__ __launch_bounds__(128, 4) static void
moe_gate_up_mid_decode_sm75_q4_32_tile32_owned_kernel(
        float *mid_out, const char *gate_base, const char *up_base,
        const cuda_sm75_native_q8_K *xq,
        const int32_t *selected, const float *weights,
        uint32_t xq_blocks, uint32_t expert_mid_dim,
        uint32_t expert_base, uint32_t expert_count, float clamp) {
    static_assert(PREFETCH_DEPTH <= 2u,
                  "Q4-32 gate/up prefetch depth must be 0, 1, or 2");
    static_assert(PACKED_MMA || PREFETCH_DEPTH == 0u,
                  "Q4-32 prefetch applies only to packed MMA");
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t row8 = lane >> 2u;
    const uint32_t lane4 = lane & 3u;
    const uint32_t row0 = blockIdx.x * 32u + warp * 8u;
    const uint32_t slot = blockIdx.y;
    __shared__ float gate_part[4][16][8];
    __shared__ float up_part[4][16][8];

    if (xq_blocks != 16u) return;
    if (slot >= 6u || row0 >= expert_mid_dim) return;
    uint32_t expert = 0;
    if (!moe_owned_local_expert(selected[slot], expert_base,
                                expert_count, &expert)) {
        if (lane < 8u && row0 + lane < expert_mid_dim)
            mid_out[(uint64_t)slot * expert_mid_dim + row0 + lane] = 0.0f;
        return;
    }

    for (uint32_t b = 0; b < xq_blocks; b++) {
        const cuda_sm75_q4_32_tile *gate_w =
            sm75_q32_ops<false>::record(gate_base, expert, row0,
                                        expert_mid_dim, xq_blocks, b);
        const cuda_sm75_q4_32_tile *up_w =
            sm75_q32_ops<false>::record(up_base, expert, row0,
                                        expert_mid_dim, xq_blocks, b);
        const cuda_sm75_native_q8_K *a = xq + b;

        if (PACKED_MMA) {
            /* Every lane supplies its native A/B fragment.  Making A depend
             * only on lane&3 duplicates the same decode row in all eight M
             * positions.  The first M row (lanes 0..3) owns the unique
             * result, two output columns per lane. */
            float gate0, gate1, up0, up1;
            const uint32_t n0 = lane4 * 2u;
            if (PREFETCH_DEPTH == 1u) {
                sm75_q32_ops<false>::mma_block_prefetch<1u>(
                    gate_w, a, lane, n0, &gate0, &gate1);
                sm75_q32_ops<false>::mma_block_prefetch<1u>(
                    up_w, a, lane, n0, &up0, &up1);
            } else if (PREFETCH_DEPTH == 2u) {
                sm75_q32_ops<false>::mma_block_prefetch<2u>(
                    gate_w, a, lane, n0, &gate0, &gate1);
                sm75_q32_ops<false>::mma_block_prefetch<2u>(
                    up_w, a, lane, n0, &up0, &up1);
            } else {
                sm75_q32_ops<false>::mma_block(
                    gate_w, a, lane, n0, &gate0, &gate1);
                sm75_q32_ops<false>::mma_block(
                    up_w, a, lane, n0, &up0, &up1);
            }
            if (lane < 4u) {
                gate_part[warp][b][n0] = __fadd_rn(0.0f, gate0);
                gate_part[warp][b][n0 + 1u] = __fadd_rn(0.0f, gate1);
                up_part[warp][b][n0] = __fadd_rn(0.0f, up0);
                up_part[warp][b][n0 + 1u] = __fadd_rn(0.0f, up1);
            }
        } else {
            int gate_total = 0, up_total = 0;
#pragma unroll 1
            for (uint32_t group = 0; group < 8u; group++) {
                const uint32_t gate_qw = gate_w->b[group][lane];
                const uint32_t up_qw = up_w->b[group][lane];
                const uint32_t lo = a->low[group][lane4];
                const uint32_t hi = a->high_signed[group][lane4];
                const uint32_t nibble_mask = 0x0f0f0f0fu;
                const uint32_t a_even =
                    (lo & nibble_mask) | ((hi & nibble_mask) << 4u);
                const uint32_t a_odd =
                    ((lo >> 4u) & nibble_mask) | (hi & 0xf0f0f0f0u);
                const uint32_t gate_even =
                    sm75_q4_32_sign_extend_nibble_bytes(gate_qw);
                const uint32_t gate_odd =
                    sm75_q4_32_sign_extend_nibble_bytes(gate_qw >> 4u);
                const uint32_t up_even =
                    sm75_q4_32_sign_extend_nibble_bytes(up_qw);
                const uint32_t up_odd =
                    sm75_q4_32_sign_extend_nibble_bytes(up_qw >> 4u);
                int gate_sum = __dp4a((int)a_even, (int)gate_even, 0);
                gate_sum = __dp4a(
                    (int)a_odd, (int)gate_odd, gate_sum);
                int up_sum = __dp4a((int)a_even, (int)up_even, 0);
                up_sum = __dp4a((int)a_odd, (int)up_odd, up_sum);
                const int gate1 =
                    __shfl_sync(0xffffffffu, gate_sum, 1, 4);
                const int gate2 =
                    __shfl_sync(0xffffffffu, gate_sum, 2, 4);
                const int gate3 =
                    __shfl_sync(0xffffffffu, gate_sum, 3, 4);
                const int up1 = __shfl_sync(0xffffffffu, up_sum, 1, 4);
                const int up2 = __shfl_sync(0xffffffffu, up_sum, 2, 4);
                const int up3 = __shfl_sync(0xffffffffu, up_sum, 3, 4);
                if (lane4 == 0u) {
                    const int gate_group =
                        ((gate_sum + gate1) + gate2) + gate3;
                    const int up_group = ((up_sum + up1) + up2) + up3;
                    gate_total += sm75_q32_scale6(
                        gate_w->scales[row8], group) * gate_group;
                    up_total += sm75_q32_scale6(
                        up_w->scales[row8], group) * up_group;
                }
            }
            if (lane4 == 0u) {
                const float gate_leaf = a->d *
                    dev_f16_to_f32(gate_w->d[row8]) * (float)gate_total;
                const float up_leaf = a->d *
                    dev_f16_to_f32(up_w->d[row8]) * (float)up_total;
                gate_part[warp][b][row8] =
                    __fadd_rn(0.0f, gate_leaf);
                up_part[warp][b][row8] = __fadd_rn(0.0f, up_leaf);
            }
        }
    }
    __syncwarp();

    /* The control's first offset-16 stage adds an inactive-lane +0 to every
     * live leaf.  Normalize the staged values explicitly, then perform its
     * offset 8, offset 4, and four-lane offset 2/1 tree. */
    const float gate0 =
        __fadd_rn(gate_part[warp][lane4][row8], 0.0f);
    const float gate8 =
        __fadd_rn(gate_part[warp][lane4 + 8u][row8], 0.0f);
    const float gate4 =
        __fadd_rn(gate_part[warp][lane4 + 4u][row8], 0.0f);
    const float gate12 =
        __fadd_rn(gate_part[warp][lane4 + 12u][row8], 0.0f);
    const float up0 = __fadd_rn(up_part[warp][lane4][row8], 0.0f);
    const float up8 =
        __fadd_rn(up_part[warp][lane4 + 8u][row8], 0.0f);
    const float up4 =
        __fadd_rn(up_part[warp][lane4 + 4u][row8], 0.0f);
    const float up12 =
        __fadd_rn(up_part[warp][lane4 + 12u][row8], 0.0f);
    float gate = __fadd_rn(
        __fadd_rn(gate0, gate8), __fadd_rn(gate4, gate12));
    float up = __fadd_rn(
        __fadd_rn(up0, up8), __fadd_rn(up4, up12));
    gate = __fadd_rn(gate, __shfl_down_sync(0xffffffffu, gate, 2, 4));
    up = __fadd_rn(up, __shfl_down_sync(0xffffffffu, up, 2, 4));
    gate = __fadd_rn(gate, __shfl_down_sync(0xffffffffu, gate, 1, 4));
    up = __fadd_rn(up, __shfl_down_sync(0xffffffffu, up, 1, 4));
    if (lane4 == 0u && row0 + row8 < expert_mid_dim) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        mid_out[(uint64_t)slot * expert_mid_dim + row0 + row8] =
            (gate / (1.0f + expf(-gate))) * up * weights[slot];
    }
}

/* Native-layout Q3A4 decode mapping.  A warp follows one 8-row Q3A4 tile
 * instead of following one row through 16 tile records:
 *
 *   lane = row-in-tile * 4 + activation-lane
 *
 * This makes low2/high weight loads contiguous across all 32 lanes and
 * broadcasts each activation word to the eight output rows that reuse it.
 * Four integer lane fragments are gathered in lane order, then the 16 K256
 * float contributions are staged and reduced with the production warp tree.
 * Gate and up stay fused and share the activation traversal. */
template <bool USE_DP4A>
__global__ __launch_bounds__(128, 4) static void
moe_gate_up_mid_decode_sm75_q3a4_tile32_owned_kernel(
        float *mid_out, const char *gate_base, const char *up_base,
        const cuda_sm75_native_q8_K *xq,
        const int32_t *selected, const float *weights,
        uint32_t xq_blocks, uint32_t expert_mid_dim,
        uint32_t expert_base, uint32_t expert_count, float clamp) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t row8 = lane >> 2u;
    const uint32_t lane4 = lane & 3u;
    const uint32_t row0 = blockIdx.x * 32u + warp * 8u;
    const uint32_t slot = blockIdx.y;
    /* block-major scratch makes the eight row-leader stores contiguous.  The
     * final four-lane reduction also addresses all 32 banks exactly once. */
    __shared__ float gate_part[4][16][8];
    __shared__ float up_part[4][16][8];

    if (slot >= 6u || row0 >= expert_mid_dim) return;
    uint32_t expert = 0;
    if (!moe_owned_local_expert(selected[slot], expert_base,
                                expert_count, &expert)) {
        if (lane < 8u && row0 + lane < expert_mid_dim)
            mid_out[(uint64_t)slot * expert_mid_dim + row0 + lane] = 0.0f;
        return;
    }

    /* Dispatch specializes this mapping for the production 16-record K. */
    for (uint32_t b = 0; b < xq_blocks; b++) {
        const cuda_sm75_q3a4_tile *gate_w =
            sm75_q32_ops<true>::record(gate_base, expert, row0,
                                       expert_mid_dim, xq_blocks, b);
        const cuda_sm75_q3a4_tile *up_w =
            sm75_q32_ops<true>::record(up_base, expert, row0,
                                       expert_mid_dim, xq_blocks, b);
        const cuda_sm75_native_q8_K *a = xq + b;
        int gate_total = 0, up_total = 0;
        int gate_correction = 0, up_correction = 0;
#pragma unroll 1
        for (uint32_t group = 0; group < 8u; group++) {
            const uint32_t gate_qw = sm75_q32_dilate_q3(
                gate_w->low2[group][lane], gate_w->high[group][lane]);
            const uint32_t up_qw = sm75_q32_dilate_q3(
                up_w->low2[group][lane], up_w->high[group][lane]);
            const uint32_t lo = a->low[group][lane4];
            const uint32_t hi = a->high_signed[group][lane4];
            int gate_sum = 0, up_sum = 0;
            if (USE_DP4A) {
                /* low | high<<4 is the exact signed Q8 byte.  Q3A4 codes are
                 * 0..7, so signed/signed DP4A preserves the dot product. */
                const uint32_t nibble_mask = 0x0f0f0f0fu;
                const uint32_t a_even =
                    (lo & nibble_mask) | ((hi & nibble_mask) << 4u);
                const uint32_t a_odd =
                    ((lo >> 4u) & nibble_mask) | (hi & 0xf0f0f0f0u);
                const uint32_t gate_even = gate_qw & nibble_mask;
                const uint32_t gate_odd =
                    (gate_qw >> 4u) & nibble_mask;
                const uint32_t up_even = up_qw & nibble_mask;
                const uint32_t up_odd = (up_qw >> 4u) & nibble_mask;
                gate_sum = __dp4a((int)a_even, (int)gate_even, 0);
                gate_sum = __dp4a((int)a_odd, (int)gate_odd, gate_sum);
                up_sum = __dp4a((int)a_even, (int)up_even, 0);
                up_sum = __dp4a((int)a_odd, (int)up_odd, up_sum);
            } else {
#pragma unroll
                for (uint32_t i = 0; i < 8u; i++) {
                    const int al = (int)((lo >> (4u * i)) & 15u);
                    const int hc = (int)((hi >> (4u * i)) & 15u);
                    const int av = al + 16 * ((hc ^ 8) - 8);
                    gate_sum +=
                        (int)((gate_qw >> (4u * i)) & 15u) * av;
                    up_sum += (int)((up_qw >> (4u * i)) & 15u) * av;
                }
            }
            const int gate1 = __shfl_sync(0xffffffffu, gate_sum, 1, 4);
            const int gate2 = __shfl_sync(0xffffffffu, gate_sum, 2, 4);
            const int gate3 = __shfl_sync(0xffffffffu, gate_sum, 3, 4);
            const int up1 = __shfl_sync(0xffffffffu, up_sum, 1, 4);
            const int up2 = __shfl_sync(0xffffffffu, up_sum, 2, 4);
            const int up3 = __shfl_sync(0xffffffffu, up_sum, 3, 4);
            if (lane4 == 0u) {
                const int gate_group =
                    ((gate_sum + gate1) + gate2) + gate3;
                const int up_group = ((up_sum + up1) + up2) + up3;
                const int bsum = (int)a->bsums[2u * group] +
                                 (int)a->bsums[2u * group + 1u];
                gate_total += (int)sm75_q32_u4(
                    gate_w->scales[row8], group) * gate_group;
                up_total += (int)sm75_q32_u4(
                    up_w->scales[row8], group) * up_group;
                gate_correction += (int)sm75_q32_u4(
                    gate_w->mins[row8], group) * bsum;
                up_correction += (int)sm75_q32_u4(
                    up_w->mins[row8], group) * bsum;
            }
        }
        if (lane4 == 0u) {
            gate_part[warp][b][row8] = a->d * (
                dev_f16_to_f32(gate_w->d[row8]) * (float)gate_total -
                dev_f16_to_f32(gate_w->dmin[row8]) *
                    (float)gate_correction);
            up_part[warp][b][row8] = a->d * (
                dev_f16_to_f32(up_w->d[row8]) * (float)up_total -
                dev_f16_to_f32(up_w->dmin[row8]) *
                    (float)up_correction);
        }
    }
    __syncwarp();

    /* Reproduce the production 16-value reduction tree exactly:
     * offset 8, then offset 4, then subgroup offsets 2 and 1. */
    float gate = __fadd_rn(
        __fadd_rn(gate_part[warp][lane4][row8],
                  gate_part[warp][lane4 + 8u][row8]),
        __fadd_rn(gate_part[warp][lane4 + 4u][row8],
                  gate_part[warp][lane4 + 12u][row8]));
    float up = __fadd_rn(
        __fadd_rn(up_part[warp][lane4][row8],
                  up_part[warp][lane4 + 8u][row8]),
        __fadd_rn(up_part[warp][lane4 + 4u][row8],
                  up_part[warp][lane4 + 12u][row8]));
    gate = __fadd_rn(gate, __shfl_down_sync(0xffffffffu, gate, 2, 4));
    up = __fadd_rn(up, __shfl_down_sync(0xffffffffu, up, 2, 4));
    gate = __fadd_rn(gate, __shfl_down_sync(0xffffffffu, gate, 1, 4));
    up = __fadd_rn(up, __shfl_down_sync(0xffffffffu, up, 1, 4));
    if (lane4 == 0u && row0 + row8 < expert_mid_dim) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        mid_out[(uint64_t)slot * expert_mid_dim + row0 + row8] =
            (gate / (1.0f + expf(-gate))) * up * weights[slot];
    }
}

/* Q3A4-only exact in-CTA K-split candidates.  K_SPLIT warps cooperate on
 * each native eight-row tile while retaining the K1 kernel's complete
 * 16-leaf table.  That is intentional: after the CTA barrier the split-zero
 * warps execute the identical explicit floating-point tree, so moving an
 * independent K256 record to another warp cannot change accumulation order.
 *
 * K2 uses 8 warps (four row groups x two K partitions); K4 uses 16.  Both
 * launch-bound specializations target all 32 resident SM75 warps without
 * changing the 32-row CTA span or the 4 KiB shared-memory footprint. */
template <uint32_t K_SPLIT>
__global__ __launch_bounds__(128u * K_SPLIT, 8u / K_SPLIT) static void
moe_gate_up_mid_decode_sm75_q3a4_tile32_ksplit_owned_kernel(
        float *mid_out, const char *gate_base, const char *up_base,
        const cuda_sm75_native_q8_K *xq,
        const int32_t *selected, const float *weights,
        uint32_t xq_blocks, uint32_t expert_mid_dim,
        uint32_t expert_base, uint32_t expert_count, float clamp) {
    static_assert(K_SPLIT == 2u || K_SPLIT == 4u,
                  "Q3A4 decode K split must be 2 or 4");
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t split = warp >> 2u;
    const uint32_t row_group = warp & 3u;
    const uint32_t row8 = lane >> 2u;
    const uint32_t lane4 = lane & 3u;
    const uint32_t block_row0 = blockIdx.x * 32u;
    const uint32_t row0 = block_row0 + row_group * 8u;
    const uint32_t slot = blockIdx.y;
    __shared__ float gate_part[4][16][8];
    __shared__ float up_part[4][16][8];

    /* These exits are CTA-uniform.  Row-group tails may not return before the
     * block barrier below; they simply skip their record work and final store. */
    if (slot >= 6u || block_row0 >= expert_mid_dim || xq_blocks != 16u)
        return;
    uint32_t expert = 0;
    if (!moe_owned_local_expert(selected[slot], expert_base,
                                expert_count, &expert)) {
        if (split == 0u && lane < 8u && row0 + lane < expert_mid_dim)
            mid_out[(uint64_t)slot * expert_mid_dim + row0 + lane] = 0.0f;
        return;
    }

    const bool row_group_valid = row0 < expert_mid_dim;
    if (row_group_valid) {
        for (uint32_t b = split; b < 16u; b += K_SPLIT) {
            const cuda_sm75_q3a4_tile *gate_w =
                sm75_q32_ops<true>::record(gate_base, expert, row0,
                                           expert_mid_dim, xq_blocks, b);
            const cuda_sm75_q3a4_tile *up_w =
                sm75_q32_ops<true>::record(up_base, expert, row0,
                                           expert_mid_dim, xq_blocks, b);
            const cuda_sm75_native_q8_K *a = xq + b;
            int gate_total = 0, up_total = 0;
            int gate_correction = 0, up_correction = 0;
#pragma unroll 1
            for (uint32_t group = 0; group < 8u; group++) {
                const uint32_t gate_qw = sm75_q32_dilate_q3(
                    gate_w->low2[group][lane], gate_w->high[group][lane]);
                const uint32_t up_qw = sm75_q32_dilate_q3(
                    up_w->low2[group][lane], up_w->high[group][lane]);
                const uint32_t lo = a->low[group][lane4];
                const uint32_t hi = a->high_signed[group][lane4];
                const uint32_t nibble_mask = 0x0f0f0f0fu;
                const uint32_t a_even =
                    (lo & nibble_mask) | ((hi & nibble_mask) << 4u);
                const uint32_t a_odd =
                    ((lo >> 4u) & nibble_mask) | (hi & 0xf0f0f0f0u);
                const uint32_t gate_even = gate_qw & nibble_mask;
                const uint32_t gate_odd =
                    (gate_qw >> 4u) & nibble_mask;
                const uint32_t up_even = up_qw & nibble_mask;
                const uint32_t up_odd = (up_qw >> 4u) & nibble_mask;
                int gate_sum = __dp4a((int)a_even, (int)gate_even, 0);
                gate_sum = __dp4a((int)a_odd, (int)gate_odd, gate_sum);
                int up_sum = __dp4a((int)a_even, (int)up_even, 0);
                up_sum = __dp4a((int)a_odd, (int)up_odd, up_sum);
                const int gate1 =
                    __shfl_sync(0xffffffffu, gate_sum, 1, 4);
                const int gate2 =
                    __shfl_sync(0xffffffffu, gate_sum, 2, 4);
                const int gate3 =
                    __shfl_sync(0xffffffffu, gate_sum, 3, 4);
                const int up1 = __shfl_sync(0xffffffffu, up_sum, 1, 4);
                const int up2 = __shfl_sync(0xffffffffu, up_sum, 2, 4);
                const int up3 = __shfl_sync(0xffffffffu, up_sum, 3, 4);
                if (lane4 == 0u) {
                    const int gate_group =
                        ((gate_sum + gate1) + gate2) + gate3;
                    const int up_group = ((up_sum + up1) + up2) + up3;
                    const int bsum = (int)a->bsums[2u * group] +
                                     (int)a->bsums[2u * group + 1u];
                    gate_total += (int)sm75_q32_u4(
                        gate_w->scales[row8], group) * gate_group;
                    up_total += (int)sm75_q32_u4(
                        up_w->scales[row8], group) * up_group;
                    gate_correction += (int)sm75_q32_u4(
                        gate_w->mins[row8], group) * bsum;
                    up_correction += (int)sm75_q32_u4(
                        up_w->mins[row8], group) * bsum;
                }
            }
            if (lane4 == 0u) {
                gate_part[row_group][b][row8] = a->d * (
                    dev_f16_to_f32(gate_w->d[row8]) * (float)gate_total -
                    dev_f16_to_f32(gate_w->dmin[row8]) *
                        (float)gate_correction);
                up_part[row_group][b][row8] = a->d * (
                    dev_f16_to_f32(up_w->d[row8]) * (float)up_total -
                    dev_f16_to_f32(up_w->dmin[row8]) *
                        (float)up_correction);
            }
        }
    }
    __syncthreads();

    if (split != 0u || !row_group_valid) return;
    /* Keep this tree byte-for-byte aligned with the production K1 kernel. */
    float gate = __fadd_rn(
        __fadd_rn(gate_part[row_group][lane4][row8],
                  gate_part[row_group][lane4 + 8u][row8]),
        __fadd_rn(gate_part[row_group][lane4 + 4u][row8],
                  gate_part[row_group][lane4 + 12u][row8]));
    float up = __fadd_rn(
        __fadd_rn(up_part[row_group][lane4][row8],
                  up_part[row_group][lane4 + 8u][row8]),
        __fadd_rn(up_part[row_group][lane4 + 4u][row8],
                  up_part[row_group][lane4 + 12u][row8]));
    gate = __fadd_rn(gate, __shfl_down_sync(0xffffffffu, gate, 2, 4));
    up = __fadd_rn(up, __shfl_down_sync(0xffffffffu, up, 2, 4));
    gate = __fadd_rn(gate, __shfl_down_sync(0xffffffffu, gate, 1, 4));
    up = __fadd_rn(up, __shfl_down_sync(0xffffffffu, up, 1, 4));
    if (lane4 == 0u && row0 + row8 < expert_mid_dim) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        mid_out[(uint64_t)slot * expert_mid_dim + row0 + row8] =
            (gate / (1.0f + expf(-gate))) * up * weights[slot];
    }
}

/* K4-only software-pipelined weight-stream candidates.  The operand bundle
 * contains only the two dilated weight words and the matching activation
 * words for one Q3A4 group.  Depth 1 keeps the current and next group live;
 * depth 2 keeps the current and next two groups live.  The consume calls stay
 * explicitly ordered from group 0 through group 7, so all integer totals and
 * corrections are accumulated in exactly the production K4 order. */
struct sm75_q3a4_k4_group_operands {
    uint32_t gate_qw;
    uint32_t up_qw;
    uint32_t lo;
    uint32_t hi;
};

template <uint32_t GROUP>
__device__ __forceinline__ static sm75_q3a4_k4_group_operands
sm75_q3a4_k4_load_group(
        const cuda_sm75_q3a4_tile *gate_w,
        const cuda_sm75_q3a4_tile *up_w,
        const cuda_sm75_native_q8_K *a,
        uint32_t lane, uint32_t lane4) {
    static_assert(GROUP < 8u, "Q3A4 group must be in [0, 8)");
    sm75_q3a4_k4_group_operands operands;
    operands.gate_qw = sm75_q32_dilate_q3(
        gate_w->low2[GROUP][lane], gate_w->high[GROUP][lane]);
    operands.up_qw = sm75_q32_dilate_q3(
        up_w->low2[GROUP][lane], up_w->high[GROUP][lane]);
    operands.lo = a->low[GROUP][lane4];
    operands.hi = a->high_signed[GROUP][lane4];
    return operands;
}

template <uint32_t GROUP>
__device__ __forceinline__ static void sm75_q3a4_k4_consume_group(
        sm75_q3a4_k4_group_operands operands,
        const cuda_sm75_q3a4_tile *gate_w,
        const cuda_sm75_q3a4_tile *up_w,
        const cuda_sm75_native_q8_K *a,
        uint32_t row8, uint32_t lane4,
        int *gate_total, int *up_total,
        int *gate_correction, int *up_correction) {
    static_assert(GROUP < 8u, "Q3A4 group must be in [0, 8)");
    const uint32_t nibble_mask = 0x0f0f0f0fu;
    const uint32_t a_even =
        (operands.lo & nibble_mask) |
        ((operands.hi & nibble_mask) << 4u);
    const uint32_t a_odd =
        ((operands.lo >> 4u) & nibble_mask) |
        (operands.hi & 0xf0f0f0f0u);
    const uint32_t gate_even = operands.gate_qw & nibble_mask;
    const uint32_t gate_odd = (operands.gate_qw >> 4u) & nibble_mask;
    const uint32_t up_even = operands.up_qw & nibble_mask;
    const uint32_t up_odd = (operands.up_qw >> 4u) & nibble_mask;
    int gate_sum = __dp4a((int)a_even, (int)gate_even, 0);
    gate_sum = __dp4a((int)a_odd, (int)gate_odd, gate_sum);
    int up_sum = __dp4a((int)a_even, (int)up_even, 0);
    up_sum = __dp4a((int)a_odd, (int)up_odd, up_sum);
    const int gate1 = __shfl_sync(0xffffffffu, gate_sum, 1, 4);
    const int gate2 = __shfl_sync(0xffffffffu, gate_sum, 2, 4);
    const int gate3 = __shfl_sync(0xffffffffu, gate_sum, 3, 4);
    const int up1 = __shfl_sync(0xffffffffu, up_sum, 1, 4);
    const int up2 = __shfl_sync(0xffffffffu, up_sum, 2, 4);
    const int up3 = __shfl_sync(0xffffffffu, up_sum, 3, 4);
    if (lane4 == 0u) {
        const int gate_group = ((gate_sum + gate1) + gate2) + gate3;
        const int up_group = ((up_sum + up1) + up2) + up3;
        const int bsum = (int)a->bsums[2u * GROUP] +
                         (int)a->bsums[2u * GROUP + 1u];
        *gate_total +=
            (int)sm75_q32_u4(gate_w->scales[row8], GROUP) * gate_group;
        *up_total +=
            (int)sm75_q32_u4(up_w->scales[row8], GROUP) * up_group;
        *gate_correction +=
            (int)sm75_q32_u4(gate_w->mins[row8], GROUP) * bsum;
        *up_correction +=
            (int)sm75_q32_u4(up_w->mins[row8], GROUP) * bsum;
    }
}

template <uint32_t PREFETCH_DEPTH>
__global__ __launch_bounds__(512, 2) static void
moe_gate_up_mid_decode_sm75_q3a4_tile32_k4_prefetch_owned_kernel(
        float *mid_out, const char *gate_base, const char *up_base,
        const cuda_sm75_native_q8_K *xq,
        const int32_t *selected, const float *weights,
        uint32_t xq_blocks, uint32_t expert_mid_dim,
        uint32_t expert_base, uint32_t expert_count, float clamp) {
    static_assert(PREFETCH_DEPTH == 1u || PREFETCH_DEPTH == 2u,
                  "Q3A4 K4 prefetch depth must be 1 or 2");
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t split = warp >> 2u;
    const uint32_t row_group = warp & 3u;
    const uint32_t row8 = lane >> 2u;
    const uint32_t lane4 = lane & 3u;
    const uint32_t block_row0 = blockIdx.x * 32u;
    const uint32_t row0 = block_row0 + row_group * 8u;
    const uint32_t slot = blockIdx.y;
    __shared__ float gate_part[4][16][8];
    __shared__ float up_part[4][16][8];

    if (slot >= 6u || block_row0 >= expert_mid_dim || xq_blocks != 16u)
        return;
    uint32_t expert = 0;
    if (!moe_owned_local_expert(selected[slot], expert_base,
                                expert_count, &expert)) {
        if (split == 0u && lane < 8u && row0 + lane < expert_mid_dim)
            mid_out[(uint64_t)slot * expert_mid_dim + row0 + lane] = 0.0f;
        return;
    }

    const bool row_group_valid = row0 < expert_mid_dim;
    if (row_group_valid) {
        for (uint32_t b = split; b < 16u; b += 4u) {
            const cuda_sm75_q3a4_tile *gate_w =
                sm75_q32_ops<true>::record(gate_base, expert, row0,
                                           expert_mid_dim, xq_blocks, b);
            const cuda_sm75_q3a4_tile *up_w =
                sm75_q32_ops<true>::record(up_base, expert, row0,
                                           expert_mid_dim, xq_blocks, b);
            const cuda_sm75_native_q8_K *a = xq + b;
            int gate_total = 0, up_total = 0;
            int gate_correction = 0, up_correction = 0;
            sm75_q3a4_k4_group_operands op0 =
                sm75_q3a4_k4_load_group<0u>(gate_w, up_w, a, lane, lane4);
            sm75_q3a4_k4_group_operands op1 =
                sm75_q3a4_k4_load_group<1u>(gate_w, up_w, a, lane, lane4);
            if (PREFETCH_DEPTH == 1u) {
                sm75_q3a4_k4_consume_group<0u>(
                    op0, gate_w, up_w, a, row8, lane4, &gate_total,
                    &up_total, &gate_correction, &up_correction);
                op0 = sm75_q3a4_k4_load_group<2u>(
                    gate_w, up_w, a, lane, lane4);
                sm75_q3a4_k4_consume_group<1u>(
                    op1, gate_w, up_w, a, row8, lane4, &gate_total,
                    &up_total, &gate_correction, &up_correction);
                op1 = sm75_q3a4_k4_load_group<3u>(
                    gate_w, up_w, a, lane, lane4);
                sm75_q3a4_k4_consume_group<2u>(
                    op0, gate_w, up_w, a, row8, lane4, &gate_total,
                    &up_total, &gate_correction, &up_correction);
                op0 = sm75_q3a4_k4_load_group<4u>(
                    gate_w, up_w, a, lane, lane4);
                sm75_q3a4_k4_consume_group<3u>(
                    op1, gate_w, up_w, a, row8, lane4, &gate_total,
                    &up_total, &gate_correction, &up_correction);
                op1 = sm75_q3a4_k4_load_group<5u>(
                    gate_w, up_w, a, lane, lane4);
                sm75_q3a4_k4_consume_group<4u>(
                    op0, gate_w, up_w, a, row8, lane4, &gate_total,
                    &up_total, &gate_correction, &up_correction);
                op0 = sm75_q3a4_k4_load_group<6u>(
                    gate_w, up_w, a, lane, lane4);
                sm75_q3a4_k4_consume_group<5u>(
                    op1, gate_w, up_w, a, row8, lane4, &gate_total,
                    &up_total, &gate_correction, &up_correction);
                op1 = sm75_q3a4_k4_load_group<7u>(
                    gate_w, up_w, a, lane, lane4);
                sm75_q3a4_k4_consume_group<6u>(
                    op0, gate_w, up_w, a, row8, lane4, &gate_total,
                    &up_total, &gate_correction, &up_correction);
                sm75_q3a4_k4_consume_group<7u>(
                    op1, gate_w, up_w, a, row8, lane4, &gate_total,
                    &up_total, &gate_correction, &up_correction);
            } else {
                sm75_q3a4_k4_group_operands op2 =
                    sm75_q3a4_k4_load_group<2u>(
                        gate_w, up_w, a, lane, lane4);
                sm75_q3a4_k4_consume_group<0u>(
                    op0, gate_w, up_w, a, row8, lane4, &gate_total,
                    &up_total, &gate_correction, &up_correction);
                op0 = sm75_q3a4_k4_load_group<3u>(
                    gate_w, up_w, a, lane, lane4);
                sm75_q3a4_k4_consume_group<1u>(
                    op1, gate_w, up_w, a, row8, lane4, &gate_total,
                    &up_total, &gate_correction, &up_correction);
                op1 = sm75_q3a4_k4_load_group<4u>(
                    gate_w, up_w, a, lane, lane4);
                sm75_q3a4_k4_consume_group<2u>(
                    op2, gate_w, up_w, a, row8, lane4, &gate_total,
                    &up_total, &gate_correction, &up_correction);
                op2 = sm75_q3a4_k4_load_group<5u>(
                    gate_w, up_w, a, lane, lane4);
                sm75_q3a4_k4_consume_group<3u>(
                    op0, gate_w, up_w, a, row8, lane4, &gate_total,
                    &up_total, &gate_correction, &up_correction);
                op0 = sm75_q3a4_k4_load_group<6u>(
                    gate_w, up_w, a, lane, lane4);
                sm75_q3a4_k4_consume_group<4u>(
                    op1, gate_w, up_w, a, row8, lane4, &gate_total,
                    &up_total, &gate_correction, &up_correction);
                op1 = sm75_q3a4_k4_load_group<7u>(
                    gate_w, up_w, a, lane, lane4);
                sm75_q3a4_k4_consume_group<5u>(
                    op2, gate_w, up_w, a, row8, lane4, &gate_total,
                    &up_total, &gate_correction, &up_correction);
                sm75_q3a4_k4_consume_group<6u>(
                    op0, gate_w, up_w, a, row8, lane4, &gate_total,
                    &up_total, &gate_correction, &up_correction);
                sm75_q3a4_k4_consume_group<7u>(
                    op1, gate_w, up_w, a, row8, lane4, &gate_total,
                    &up_total, &gate_correction, &up_correction);
            }
            if (lane4 == 0u) {
                gate_part[row_group][b][row8] = a->d * (
                    dev_f16_to_f32(gate_w->d[row8]) * (float)gate_total -
                    dev_f16_to_f32(gate_w->dmin[row8]) *
                        (float)gate_correction);
                up_part[row_group][b][row8] = a->d * (
                    dev_f16_to_f32(up_w->d[row8]) * (float)up_total -
                    dev_f16_to_f32(up_w->dmin[row8]) *
                        (float)up_correction);
            }
        }
    }
    __syncthreads();

    if (split != 0u || !row_group_valid) return;
    /* This is the production K4/K1 16-leaf reduction, intentionally exact. */
    float gate = __fadd_rn(
        __fadd_rn(gate_part[row_group][lane4][row8],
                  gate_part[row_group][lane4 + 8u][row8]),
        __fadd_rn(gate_part[row_group][lane4 + 4u][row8],
                  gate_part[row_group][lane4 + 12u][row8]));
    float up = __fadd_rn(
        __fadd_rn(up_part[row_group][lane4][row8],
                  up_part[row_group][lane4 + 8u][row8]),
        __fadd_rn(up_part[row_group][lane4 + 4u][row8],
                  up_part[row_group][lane4 + 12u][row8]));
    gate = __fadd_rn(gate, __shfl_down_sync(0xffffffffu, gate, 2, 4));
    up = __fadd_rn(up, __shfl_down_sync(0xffffffffu, up, 2, 4));
    gate = __fadd_rn(gate, __shfl_down_sync(0xffffffffu, gate, 1, 4));
    up = __fadd_rn(up, __shfl_down_sync(0xffffffffu, up, 1, 4));
    if (lane4 == 0u && row0 + row8 < expert_mid_dim) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        mid_out[(uint64_t)slot * expert_mid_dim + row0 + row8] =
            (gate / (1.0f + expf(-gate))) * up * weights[slot];
    }
}

/* Single-launch low-register gate/up candidates.  Unlike the rejected split
 * experiment these preserve one owned-slot traversal, keep the gate and up
 * outputs in registers, and avoid global intermediates and a combine launch.
 * The sweep intentionally restores ILP in bounded steps; launch_bounds makes
 * any register excess visible as spills rather than silently dropping below
 * the two-CTA/SM resource target. */
template <bool Q3A, uint32_t UNROLL>
__global__ __launch_bounds__(256, 2) static void
moe_gate_up_mid_decode_sm75_q32_fused_lowreg_owned_kernel(
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
        gate += sm75_q32_ops<Q3A>::template dot_decode_partial<UNROLL>(
            sm75_q32_ops<Q3A>::record(gate_base, expert, row,
                                      expert_mid_dim, xq_blocks, b),
            row & 7u, xq + b);
        up += sm75_q32_ops<Q3A>::template dot_decode_partial<UNROLL>(
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

/* Decode-only packed-INT4 mapping for the production eight-record Q4-32
 * down projection.  A warp follows one native eight-row tile.  Each MMA
 * consumes one K256 record and produces the eight output rows; the four
 * lane leaders stage its two-row fragments.  Keeping all eight float leaves
 * in shared memory lets the leaders reproduce quarter_warp_sum_f32's exact
 * 4/2/1 tree without carrying sixteen live float accumulators through the
 * packed-INT4 MMA loop.
 *
 * Four warps cover the same 32 output rows as the 256-thread scalar fallback.
 * The 128-thread launch bound intentionally caps allocation at 64 registers,
 * so eight CTAs can supply all 32 resident SM75 warps if the compiler keeps
 * the production kernel spill-free. */
__device__ __forceinline__ static float sm75_q4_32_down_reduce8_exact(
        const float leaf[8][8], uint32_t row8) {
    const float a0 = __fadd_rn(leaf[0][row8], leaf[4][row8]);
    const float a1 = __fadd_rn(leaf[1][row8], leaf[5][row8]);
    const float a2 = __fadd_rn(leaf[2][row8], leaf[6][row8]);
    const float a3 = __fadd_rn(leaf[3][row8], leaf[7][row8]);
    return __fadd_rn(__fadd_rn(a0, a2), __fadd_rn(a1, a3));
}

template <uint32_t PREFETCH_DEPTH = 0u>
__global__ __launch_bounds__(128, 8) static void
moe_down_sm75_q4_32_tile32_owned_slots_kernel(
        float *out, const char *down_base,
        const cuda_sm75_native_q8_K *midq, const int32_t *selected,
        uint32_t midq_blocks, uint32_t out_dim,
        uint32_t expert_base, uint32_t expert_count) {
    static_assert(PREFETCH_DEPTH <= 2u,
                  "Q4-32 down prefetch depth must be 0, 1, or 2");
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane4 = lane & 3u;
    const uint32_t n0 = lane4 * 2u;
    const uint32_t row0 = blockIdx.x * 32u + warp * 8u;
    const uint32_t slot = blockIdx.y;
    __shared__ float leaf[4][8][8];

    /* Dispatch admits this candidate only for the production K=8 shape.
     * Keep the condition explicit so an accidental direct launch cannot read
     * uninitialized leaves. */
    if (midq_blocks != 8u || row0 >= out_dim || slot >= 6u) return;
    uint32_t expert = 0;
    if (!moe_owned_local_expert(selected[slot], expert_base,
                                expert_count, &expert)) {
        if (lane < 4u) {
            if (row0 + n0 < out_dim)
                out[(uint64_t)slot * out_dim + row0 + n0] = 0.0f;
            if (row0 + n0 + 1u < out_dim)
                out[(uint64_t)slot * out_dim + row0 + n0 + 1u] = 0.0f;
        }
        return;
    }

#pragma unroll
    for (uint32_t b = 0; b < 8u; b++) {
        const cuda_sm75_q4_32_tile *w =
            sm75_q32_ops<false>::record(
                down_base, expert, row0, out_dim, midq_blocks, b);
        const cuda_sm75_native_q8_K *a =
            midq + (uint64_t)slot * midq_blocks + b;
        float v0 = 0.0f, v1 = 0.0f;
        if (PREFETCH_DEPTH == 1u) {
            sm75_q32_ops<false>::mma_block_prefetch<1u>(
                w, a, lane, n0, &v0, &v1);
        } else if (PREFETCH_DEPTH == 2u) {
            sm75_q32_ops<false>::mma_block_prefetch<2u>(
                w, a, lane, n0, &v0, &v1);
        } else {
            sm75_q32_ops<false>::mma_block(w, a, lane, n0, &v0, &v1);
        }
        /* Every four-lane MMA token group sees the same decode activation.
         * The first group owns the unique output fragments.  Normalize each
         * staged leaf with the control's initial +0 addition.  This is
         * observable for a raw -0 MMA product and is part of byte exactness. */
        if (lane < 4u) {
            leaf[warp][b][n0] = __fadd_rn(0.0f, v0);
            leaf[warp][b][n0 + 1u] = __fadd_rn(0.0f, v1);
        }
    }
    __syncwarp();
    if (lane < 4u) {
        const float v0 = sm75_q4_32_down_reduce8_exact(leaf[warp], n0);
        const float v1 = sm75_q4_32_down_reduce8_exact(
            leaf[warp], n0 + 1u);
        if (row0 + n0 < out_dim)
            out[(uint64_t)slot * out_dim + row0 + n0] = v0;
        if (row0 + n0 + 1u < out_dim)
            out[(uint64_t)slot * out_dim + row0 + n0 + 1u] = v1;
    }
}

template <uint32_t PREFETCH_DEPTH = 0u>
__global__ __launch_bounds__(128, 8) static void
moe_down_sm75_q4_32_tile32_owned_packed_kernel(
        float *out, const char *down_base,
        const cuda_sm75_native_q8_K *midq, const int32_t *selected,
        uint32_t midq_blocks, uint32_t out_dim,
        uint32_t expert_base, uint32_t expert_count) {
    static_assert(PREFETCH_DEPTH <= 2u,
                  "Q4-32 down prefetch depth must be 0, 1, or 2");
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane4 = lane & 3u;
    const uint32_t n0 = lane4 * 2u;
    const uint32_t row0 = blockIdx.x * 32u + warp * 8u;
    const uint32_t packed_slot = blockIdx.y;
    __shared__ float leaf[4][8][8];
    /* Keep the first expert's exact prefix total out of the register file
     * while the fully unrolled second eight-record MMA pass is live.  Volatile
     * makes this an intentional shared-memory handoff rather than allowing
     * ptxas to promote the values back into the two local spill slots this
     * buffer replaces. */
    __shared__ volatile float prefix_total[4][8];

    if (midq_blocks != 8u || row0 >= out_dim || packed_slot >= 4u) return;
    bool prefix_pair = false;
    const int first = moe_owned_packed_component(
        selected, packed_slot / 2u, packed_slot & 1u,
        expert_base, expert_count, &prefix_pair);
    if (first < 0) {
        if (lane < 4u) {
            if (row0 + n0 < out_dim)
                out[(uint64_t)packed_slot * out_dim + row0 + n0] = 0.0f;
            if (row0 + n0 + 1u < out_dim)
                out[(uint64_t)packed_slot * out_dim + row0 + n0 + 1u] = 0.0f;
        }
        return;
    }

    const uint32_t count = prefix_pair ? 2u : 1u;
#pragma unroll
    for (uint32_t i = 0; i < 2u; i++) {
        if (i >= count) break;
        const uint32_t slot = (uint32_t)first + i;
        uint32_t expert = 0;
        if (!moe_owned_local_expert(selected[slot], expert_base,
                                    expert_count, &expert)) continue;
#pragma unroll
        for (uint32_t b = 0; b < 8u; b++) {
            const cuda_sm75_q4_32_tile *w =
                sm75_q32_ops<false>::record(
                    down_base, expert, row0, out_dim, midq_blocks, b);
            const cuda_sm75_native_q8_K *a =
                midq + (uint64_t)slot * midq_blocks + b;
            float v0 = 0.0f, v1 = 0.0f;
            if (PREFETCH_DEPTH == 1u) {
                sm75_q32_ops<false>::mma_block_prefetch<1u>(
                    w, a, lane, n0, &v0, &v1);
            } else if (PREFETCH_DEPTH == 2u) {
                sm75_q32_ops<false>::mma_block_prefetch<2u>(
                    w, a, lane, n0, &v0, &v1);
            } else {
                sm75_q32_ops<false>::mma_block(
                    w, a, lane, n0, &v0, &v1);
            }
            if (lane < 4u) {
                leaf[warp][b][n0] = __fadd_rn(0.0f, v0);
                leaf[warp][b][n0 + 1u] = __fadd_rn(0.0f, v1);
            }
        }
        __syncwarp();
        if (lane < 4u) {
            const float acc0 =
                sm75_q4_32_down_reduce8_exact(leaf[warp], n0);
            const float acc1 = sm75_q4_32_down_reduce8_exact(
                leaf[warp], n0 + 1u);
            if (prefix_pair && i == 0u) {
                /* Match the control's initial total=+0 addition exactly. */
                prefix_total[warp][n0] = __fadd_rn(0.0f, acc0);
                prefix_total[warp][n0 + 1u] = __fadd_rn(0.0f, acc1);
            } else {
                const float total0 = prefix_pair
                    ? __fadd_rn(prefix_total[warp][n0], acc0) : acc0;
                const float total1 = prefix_pair
                    ? __fadd_rn(prefix_total[warp][n0 + 1u], acc1) : acc1;
                if (row0 + n0 < out_dim)
                    out[(uint64_t)packed_slot * out_dim + row0 + n0] =
                        total0;
                if (row0 + n0 + 1u < out_dim)
                    out[(uint64_t)packed_slot * out_dim + row0 + n0 + 1u] =
                        total1;
            }
        }
        /* Prevent the next expert's MMA leaders from overwriting shared
         * leaves while the current leaders still consume the exact tree. */
        __syncwarp();
    }
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
