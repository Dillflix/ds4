/* Production consumers for the explicitly tagged GGUF routed-Q4 layout
 * sm75_m8n8k32_native_aw_v1.  This file is included by ds4_cuda.cu after the
 * standard quant/dot helpers.  It is deliberately not a fallback: ordinary
 * Q4_K files continue through the row-major kernels in ds4_cuda.cu. */

__device__ __forceinline__ static uint32_t sm75_q4_pack_nibbles8(
        uint32_t lo, uint32_t hi, uint32_t shift) {
    uint32_t out = 0;
#pragma unroll
    for (uint32_t i = 0; i < 4u; i++) {
        out |= ((lo >> (8u * i + shift)) & 0x0fu) << (4u * i);
        out |= ((hi >> (8u * i + shift)) & 0x0fu) << (4u * (i + 4u));
    }
    return out;
}

/* Q8_K and the native activation record are both 292 bytes.  Every source
 * byte is first captured in registers, then the warp collectively overwrites
 * the record in place. */
__global__ static void q8_K_pack_sm75_native_inplace_kernel(
        cuda_block_q8_K *io, uint64_t n_blocks) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint64_t block = (uint64_t)blockIdx.x * 8u + warp;
    if (block >= n_blocks) return;
    const cuda_block_q8_K *src = io + block;
    const uint32_t group = lane >> 2u;
    const uint32_t lane4 = lane & 3u;
    const uint32_t off = group * 32u + lane4 * 8u;
    const uint32_t q0 = *(const uint32_t *)(src->qs + off);
    const uint32_t q1 = *(const uint32_t *)(src->qs + off + 4u);
    const float d = src->d;
    const int16_t bsum = lane < 16u ? src->bsums[lane] : 0;
    __syncwarp();
    cuda_sm75_native_q8_K *dst = (cuda_sm75_native_q8_K *)io + block;
    dst->low[group][lane4] = sm75_q4_pack_nibbles8(q0, q1, 0u);
    dst->high_signed[group][lane4] =
        sm75_q4_pack_nibbles8(q0, q1, 4u);
    if (lane == 0u) dst->d = d;
    if (lane < 16u) dst->bsums[lane] = bsum;
}

static int sm75_q4_pack_activations_inplace(
        cuda_block_q8_K *io, uint64_t n_blocks, const char *what) {
    if (!n_blocks) return 1;
    q8_K_pack_sm75_native_inplace_kernel<<<
        (unsigned)((n_blocks + 7u) / 8u), 256>>>(io, n_blocks);
    return cuda_ok(cudaGetLastError(), what);
}

/* Scalar reference for decode.  The packed weight word is the exact B
 * fragment owned by one MMA lane; the packed high activation nibble is
 * interpreted as signed s4, hence q8 = low + 16 * signed(high). */
__device__ __forceinline__ static float dev_dot_sm75_native_q4_q8_block(
        const cuda_sm75_native_q4_tile *w,
        uint32_t row8,
        const cuda_sm75_native_q8_K *a) {
    const uint4 hdr = w->hdr[row8];
    int isum = 0;
    int summs = 0;
#pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        int dot = 0;
#pragma unroll
        for (uint32_t lane4 = 0; lane4 < 4u; lane4++) {
            const uint32_t qw = w->b[j][row8 * 4u + lane4];
            const uint32_t lo = a->low[j][lane4];
            const uint32_t hi = a->high_signed[j][lane4];
#pragma unroll
            for (uint32_t i = 0; i < 8u; i++) {
                const int q4 = (int)((qw >> (4u * i)) & 0x0fu);
                const int qlo = (int)((lo >> (4u * i)) & 0x0fu);
                const int qhi4 = (int)((hi >> (4u * i)) & 0x0fu);
                const int qhi = (qhi4 ^ 8) - 8;
                dot += q4 * (qlo + 16 * qhi);
            }
        }
        uint8_t sc, mn;
        dev_q4_K_get_scale_min(j, (const uint8_t *)&hdr.y, &sc, &mn);
        isum += (int)sc * dot;
        summs += (int)mn *
            ((int)a->bsums[2u * j] + (int)a->bsums[2u * j + 1u]);
    }
    const float yd = a->d;
    return yd * dev_f16_to_f32((uint16_t)(hdr.x & 0xffffu)) *
               (float)isum -
           yd * dev_f16_to_f32((uint16_t)(hdr.x >> 16u)) *
               (float)summs;
}

__device__ __forceinline__ static const cuda_sm75_native_q4_tile *
sm75_native_q4_record(
        const char *base, uint32_t expert, uint32_t row,
        uint32_t rows, uint32_t blocks, uint32_t block) {
    return (const cuda_sm75_native_q4_tile *)base +
        (((uint64_t)expert * (rows / 8u) + row / 8u) * blocks + block);
}

/* Decode uses the same lane-to-block assignment and reduction tree as the
 * ordinary warp32 Q4 path, so the layout change does not change output bits. */
__global__ static void moe_gate_up_mid_decode_sm75_native_q4_kernel(
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
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const cuda_sm75_native_q8_K *a = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f, up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        gate += dev_dot_sm75_native_q4_q8_block(
            sm75_native_q4_record(gate_base, expert, row,
                                  expert_mid_dim, xq_blocks, b),
            row & 7u, a + b);
        up += dev_dot_sm75_native_q4_q8_block(
            sm75_native_q4_record(up_base, expert, row,
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
        if (write_aux) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
            weights[(uint64_t)tok * n_expert + slot];
    }
}

template <uint32_t NSLOT>
__global__ static void moe_down_sm75_native_q4_sum_kernel(
        float *out, const char *down_base,
        const cuda_sm75_native_q8_K *midq, const int32_t *selected,
        uint32_t midq_blocks, uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
#pragma unroll
    for (uint32_t slot = 0; slot < NSLOT; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const uint32_t expert = (uint32_t)expert_i;
        const cuda_sm75_native_q8_K *a = midq +
            (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            acc += dev_dot_sm75_native_q4_q8_block(
                sm75_native_q4_record(down_base, expert, row,
                                      out_dim, midq_blocks, b),
                row & 7u, a + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0u) total += acc;
    }
    if (lane == 0u) out[row] = total;
}

__global__ static void moe_down_sm75_native_q4_pair_kernel(
        float *down_out, const char *down_base,
        const cuda_sm75_native_q8_K *midq, const int32_t *selected,
        uint32_t midq_blocks, uint32_t out_dim, uint32_t n_expert) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const cuda_sm75_native_q8_K *a = midq +
        (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        acc += dev_dot_sm75_native_q4_q8_block(
            sm75_native_q4_record(down_base, expert, row,
                                  out_dim, midq_blocks, b),
            row & 7u, a + b);
    }
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0u)
        down_out[(uint64_t)pair * out_dim + row] = acc;
    (void)n_expert;
}

__global__ static void moe_gate_up_mid_decode_sm75_native_q4_owned_kernel(
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
        gate += dev_dot_sm75_native_q4_q8_block(
            sm75_native_q4_record(gate_base, expert, row,
                                  expert_mid_dim, xq_blocks, b),
            row & 7u, xq + b);
        up += dev_dot_sm75_native_q4_q8_block(
            sm75_native_q4_record(up_base, expert, row,
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

__global__ static void moe_down_sm75_native_q4_owned_slots_kernel(
        float *down_out, const char *down_base,
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
        if (lane == 0u)
            down_out[(uint64_t)slot * out_dim + row] = 0.0f;
        return;
    }
    float acc = 0.0f;
    const cuda_sm75_native_q8_K *a = midq +
        (uint64_t)slot * midq_blocks;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        acc += dev_dot_sm75_native_q4_q8_block(
            sm75_native_q4_record(down_base, expert, row,
                                  out_dim, midq_blocks, b),
            row & 7u, a + b);
    }
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0u)
        down_out[(uint64_t)slot * out_dim + row] = acc;
}

__global__ static void moe_down_sm75_native_q4_owned_packed_kernel(
        float *packed_out, const char *down_base,
        const cuda_sm75_native_q8_K *midq, const int32_t *selected,
        uint32_t midq_blocks, uint32_t out_dim,
        uint32_t expert_base, uint32_t expert_count) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t packed_slot = blockIdx.y;
    if (row >= out_dim || packed_slot >= 4u) return;
    bool prefix_pair = false;
    const int first_slot = moe_owned_packed_component(
        selected, packed_slot / 2u, packed_slot & 1u,
        expert_base, expert_count, &prefix_pair);
    if (first_slot < 0) {
        if (lane == 0u)
            packed_out[(uint64_t)packed_slot * out_dim + row] = 0.0f;
        return;
    }
    float packed = 0.0f;
    const uint32_t n_slots = prefix_pair ? 2u : 1u;
#pragma unroll
    for (uint32_t i = 0; i < 2u; i++) {
        if (i >= n_slots) break;
        const uint32_t slot = (uint32_t)first_slot + i;
        uint32_t expert = 0;
        if (!moe_owned_local_expert(selected[slot], expert_base,
                                    expert_count, &expert)) continue;
        float acc = 0.0f;
        const cuda_sm75_native_q8_K *a = midq +
            (uint64_t)slot * midq_blocks;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            acc += dev_dot_sm75_native_q4_q8_block(
                sm75_native_q4_record(down_base, expert, row,
                                      out_dim, midq_blocks, b),
                row & 7u, a + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0u)
            packed = prefix_pair ? __fadd_rn(packed, acc) : acc;
    }
    if (lane == 0u)
        packed_out[(uint64_t)packed_slot * out_dim + row] = packed;
}

#define DS4_SM75_NATIVE_SLOT4_DECL(S) \
    float s0_##S = 0.0f, s1_##S = 0.0f, \
          s2_##S = 0.0f, s3_##S = 0.0f
#define DS4_SM75_NATIVE_SLOT4_ADD(S, V0, V1, V2, V3) \
    case S: s0_##S += (V0); s1_##S += (V1); \
            s2_##S += (V2); s3_##S += (V3); break
#define DS4_SM75_NATIVE_REDUCE(P, O) do { \
    const float _a0 = P##_0 + P##_4; \
    const float _a1 = P##_1 + P##_5; \
    const float _a2 = P##_2 + P##_6; \
    const float _a3 = P##_3 + P##_7; \
    (O) = (_a0 + _a2) + (_a1 + _a3); \
} while (0)

/* Packed-A/W, scalar-slot, 16-warp gate/up tile8.  TILE_DELTA is 0 or 8
 * when a real tile16 entry is consumed and 0 for a real tail entry. */
template <uint32_t ROW_SPAN, uint32_t TILE_DELTA>
__global__ static void moe_gate_up_mid_sm75_native_q4_tile8_kernel(
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
    const uint32_t local_start = tile_starts[tile] + TILE_DELTA;
    __shared__ cuda_sm75_native_q8_K sxq[8][16];
    __shared__ uint32_t s_pair[8], s_tok[8], s_slot[8], s_np;
    if (threadIdx.x == 0u) {
        uint32_t np = 0;
        for (; np < 8u; np++) {
            const uint32_t local = local_start + np;
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
    const uint32_t words = (uint32_t)(sizeof(cuda_sm75_native_q8_K) / 4u);
    const uint32_t words_per_tok = xq_blocks * words;
    for (uint32_t i = threadIdx.x; i < 8u * words_per_tok;
         i += blockDim.x) {
        const uint32_t p = i / words_per_tok;
        const uint32_t w = i - p * words_per_tok;
        ((uint32_t *)sxq[p])[w] = p < np ?
            ((const uint32_t *)(xq +
                (uint64_t)s_tok[p] * xq_blocks))[w] : 0u;
    }
    __syncthreads();
    const uint32_t mtok = lane >> 2u;
    const uint32_t n0 = (lane & 3u) * 2u;
    for (uint32_t rr = 0; rr < ROW_SPAN / 128u; rr++) {
        const uint32_t row0 = blockIdx.x * ROW_SPAN +
            rr * 128u + warp * 8u;
        if (row0 >= expert_mid_dim) continue;
        DS4_SM75_NATIVE_SLOT4_DECL(0); DS4_SM75_NATIVE_SLOT4_DECL(1);
        DS4_SM75_NATIVE_SLOT4_DECL(2); DS4_SM75_NATIVE_SLOT4_DECL(3);
        DS4_SM75_NATIVE_SLOT4_DECL(4); DS4_SM75_NATIVE_SLOT4_DECL(5);
        DS4_SM75_NATIVE_SLOT4_DECL(6); DS4_SM75_NATIVE_SLOT4_DECL(7);
        const uint64_t native_tile =
            (uint64_t)expert * (expert_mid_dim / 8u) + row0 / 8u;
        const cuda_sm75_native_q8_K *a = &sxq[mtok][0];
        for (uint32_t b = 0; b < xq_blocks; b++) {
            const cuda_sm75_native_q4_tile *gw =
                (const cuda_sm75_native_q4_tile *)gate_base +
                native_tile * xq_blocks + b;
            const cuda_sm75_native_q4_tile *uw =
                (const cuda_sm75_native_q4_tile *)up_base +
                native_tile * xq_blocks + b;
            const uint4 gh0 = gw->hdr[n0], gh1 = gw->hdr[n0 + 1u];
            const uint4 uh0 = uw->hdr[n0], uh1 = uw->hdr[n0 + 1u];
            int gi0 = 0, gi1 = 0, gm0 = 0, gm1 = 0;
            int ui0 = 0, ui1 = 0, um0 = 0, um1 = 0;
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                const uint32_t al = a[b].low[j][lane & 3u];
                const uint32_t ah = a[b].high_signed[j][lane & 3u];
                int32_t gl0 = 0, gl1 = 0, ghh0 = 0, ghh1 = 0;
                int32_t ul0 = 0, ul1 = 0, uhh0 = 0, uhh1 = 0;
                mma_m8n8k32_u4_u4(gl0, gl1, al, gw->b[j][lane]);
                mma_m8n8k32_s4_u4(ghh0, ghh1, ah, gw->b[j][lane]);
                mma_m8n8k32_u4_u4(ul0, ul1, al, uw->b[j][lane]);
                mma_m8n8k32_s4_u4(uhh0, uhh1, ah, uw->b[j][lane]);
                const int gc0 = gl0 + 16 * ghh0;
                const int gc1 = gl1 + 16 * ghh1;
                const int uc0 = ul0 + 16 * uhh0;
                const int uc1 = ul1 + 16 * uhh1;
                const int bs = (int)a[b].bsums[2u * j] +
                               (int)a[b].bsums[2u * j + 1u];
                uint8_t sc, mn;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&gh0.y, &sc, &mn);
                gi0 += (int)sc * gc0; gm0 += (int)mn * bs;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&gh1.y, &sc, &mn);
                gi1 += (int)sc * gc1; gm1 += (int)mn * bs;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&uh0.y, &sc, &mn);
                ui0 += (int)sc * uc0; um0 += (int)mn * bs;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&uh1.y, &sc, &mn);
                ui1 += (int)sc * uc1; um1 += (int)mn * bs;
            }
            const float yd = a[b].d;
            const float vg0 = yd * dev_f16_to_f32((uint16_t)gh0.x) *
                (float)gi0 - yd * dev_f16_to_f32((uint16_t)(gh0.x >> 16u)) *
                (float)gm0;
            const float vg1 = yd * dev_f16_to_f32((uint16_t)gh1.x) *
                (float)gi1 - yd * dev_f16_to_f32((uint16_t)(gh1.x >> 16u)) *
                (float)gm1;
            const float vu0 = yd * dev_f16_to_f32((uint16_t)uh0.x) *
                (float)ui0 - yd * dev_f16_to_f32((uint16_t)(uh0.x >> 16u)) *
                (float)um0;
            const float vu1 = yd * dev_f16_to_f32((uint16_t)uh1.x) *
                (float)ui1 - yd * dev_f16_to_f32((uint16_t)(uh1.x >> 16u)) *
                (float)um1;
            switch (b & 7u) {
                DS4_SM75_NATIVE_SLOT4_ADD(0, vg0, vg1, vu0, vu1);
                DS4_SM75_NATIVE_SLOT4_ADD(1, vg0, vg1, vu0, vu1);
                DS4_SM75_NATIVE_SLOT4_ADD(2, vg0, vg1, vu0, vu1);
                DS4_SM75_NATIVE_SLOT4_ADD(3, vg0, vg1, vu0, vu1);
                DS4_SM75_NATIVE_SLOT4_ADD(4, vg0, vg1, vu0, vu1);
                DS4_SM75_NATIVE_SLOT4_ADD(5, vg0, vg1, vu0, vu1);
                DS4_SM75_NATIVE_SLOT4_ADD(6, vg0, vg1, vu0, vu1);
                DS4_SM75_NATIVE_SLOT4_ADD(7, vg0, vg1, vu0, vu1);
            }
        }
        float gate0, gate1, up0, up1;
        DS4_SM75_NATIVE_REDUCE(s0, gate0);
        DS4_SM75_NATIVE_REDUCE(s1, gate1);
        DS4_SM75_NATIVE_REDUCE(s2, up0);
        DS4_SM75_NATIVE_REDUCE(s3, up1);
        if (mtok < np) {
            const uint32_t pair = s_pair[mtok];
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
                const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
                if (write_aux) { gate_out[off] = g; up_out[off] = u; }
                mid_out[off] = (g / (1.0f + expf(-g))) * u *
                    weights[(uint64_t)s_tok[mtok] * n_expert +
                            s_slot[mtok]];
            }
        }
    }
}

/* Only these 256 bytes feed MMA. Keeping d/bsums on the read-only path lets
 * the next designs stage more useful activation payload below 32 KiB. */
typedef struct {
    uint32_t low[8][4];
    uint32_t high_signed[8][4];
} cuda_sm75_native_q8_payload;
static_assert(sizeof(cuda_sm75_native_q8_payload) == 256u,
              "unexpected native Q8 MMA payload size");

/* One packed-Q4 block for two m8 token halves. A single loaded weight
 * fragment feeds low8 and high8; metadata can remain in L1/read-only cache. */
__device__ __forceinline__ static void
moe_sm75_native_q4_pair_block(
        const char *base,
        const uint32_t (*low0)[4], const uint32_t (*high0)[4],
        const uint32_t (*low1)[4], const uint32_t (*high1)[4],
        const cuda_sm75_native_q8_K *meta0,
        const cuda_sm75_native_q8_K *meta1,
        bool have0, bool have1, uint32_t b,
        uint32_t xq_blocks, uint32_t expert, uint32_t matrix_rows,
        uint32_t row0, uint32_t lane,
        float &v00, float &v01, float &v10, float &v11) {
    const uint32_t n0 = (lane & 3u) * 2u;
    const uint64_t native_tile =
        (uint64_t)expert * (matrix_rows / 8u) + row0 / 8u;
    const cuda_sm75_native_q4_tile *w =
        (const cuda_sm75_native_q4_tile *)base +
        native_tile * xq_blocks + b;
    const uint4 h0 = w->hdr[n0], h1 = w->hdr[n0 + 1u];
    int i00=0,i01=0,i10=0,i11=0,m00=0,m01=0,m10=0,m11=0;
#pragma unroll
    for (uint32_t j=0;j<8u;j++) {
        const uint32_t wf=w->b[j][lane];
        int32_t l00=0,l01=0,h00=0,h01=0,l10=0,l11=0,h10=0,h11=0;
        const uint32_t al0=have0?low0[j][lane&3u]:0u;
        const uint32_t ah0=have0?high0[j][lane&3u]:0u;
        const uint32_t al1=have1?low1[j][lane&3u]:0u;
        const uint32_t ah1=have1?high1[j][lane&3u]:0u;
        mma_m8n8k32_u4_u4(l00,l01,al0,wf);
        mma_m8n8k32_s4_u4(h00,h01,ah0,wf);
        mma_m8n8k32_u4_u4(l10,l11,al1,wf);
        mma_m8n8k32_s4_u4(h10,h11,ah1,wf);
        const int c00=l00+16*h00,c01=l01+16*h01;
        const int c10=l10+16*h10,c11=l11+16*h11;
        const int bs0=have0?
            (int)meta0->bsums[2u*j]+(int)meta0->bsums[2u*j+1u]:0;
        const int bs1=have1?
            (int)meta1->bsums[2u*j]+(int)meta1->bsums[2u*j+1u]:0;
        uint8_t sc0,mn0,sc1,mn1;
        dev_q4_K_get_scale_min(j,(const uint8_t *)&h0.y,&sc0,&mn0);
        dev_q4_K_get_scale_min(j,(const uint8_t *)&h1.y,&sc1,&mn1);
        i00+=(int)sc0*c00;i01+=(int)sc1*c01;
        i10+=(int)sc0*c10;i11+=(int)sc1*c11;
        m00+=(int)mn0*bs0;m01+=(int)mn1*bs0;
        m10+=(int)mn0*bs1;m11+=(int)mn1*bs1;
    }
    const float d0=dev_f16_to_f32((uint16_t)h0.x);
    const float z0=dev_f16_to_f32((uint16_t)(h0.x>>16u));
    const float d1=dev_f16_to_f32((uint16_t)h1.x);
    const float z1=dev_f16_to_f32((uint16_t)(h1.x>>16u));
    const float yd0=have0?meta0->d:0.0f;
    const float yd1=have1?meta1->d:0.0f;
    v00=yd0*d0*(float)i00-yd0*z0*(float)m00;
    v01=yd0*d1*(float)i01-yd0*z1*(float)m01;
    v10=yd1*d0*(float)i10-yd1*z0*(float)m10;
    v11=yd1*d1*(float)i11-yd1*z1*(float)m11;
}

template <uint32_t STAGED_ROWS>
__device__ __forceinline__ static void
moe_sm75_native_q4_stream_matrix(
        const char *base, const cuda_sm75_native_q8_payload *staged,
        const cuda_sm75_native_q8_K *xq, const uint32_t *tok,
        uint32_t np, uint32_t xq_blocks, uint32_t expert,
        uint32_t matrix_rows, uint32_t row0, uint32_t lane,
        float &r0, float &r1, float &r2, float &r3) {
    const uint32_t p0=lane>>2u,p1=p0+8u;
    const bool have0=p0<np,have1=p1<np;
    DS4_SM75_NATIVE_SLOT4_DECL(0); DS4_SM75_NATIVE_SLOT4_DECL(1);
    DS4_SM75_NATIVE_SLOT4_DECL(2); DS4_SM75_NATIVE_SLOT4_DECL(3);
    DS4_SM75_NATIVE_SLOT4_DECL(4); DS4_SM75_NATIVE_SLOT4_DECL(5);
    DS4_SM75_NATIVE_SLOT4_DECL(6); DS4_SM75_NATIVE_SLOT4_DECL(7);
    for (uint32_t b=0;b<xq_blocks;b++) {
        const cuda_sm75_native_q8_K *m0=have0?
            xq+(uint64_t)tok[p0]*xq_blocks+b:xq;
        const cuda_sm75_native_q8_K *m1=have1?
            xq+(uint64_t)tok[p1]*xq_blocks+b:xq;
        const cuda_sm75_native_q8_payload *a0=p0<STAGED_ROWS?
            staged+p0*16u+b:NULL;
        const cuda_sm75_native_q8_payload *a1=p1<STAGED_ROWS?
            staged+p1*16u+b:NULL;
        const uint32_t (*low0)[4]=a0?a0->low:m0->low;
        const uint32_t (*high0)[4]=a0?a0->high_signed:m0->high_signed;
        const uint32_t (*low1)[4]=a1?a1->low:m1->low;
        const uint32_t (*high1)[4]=a1?a1->high_signed:m1->high_signed;
        float v00,v01,v10,v11;
        moe_sm75_native_q4_pair_block(
            base,low0,high0,low1,high1,m0,m1,have0,have1,b,
            xq_blocks,expert,matrix_rows,row0,lane,v00,v01,v10,v11);
        switch(b&7u) {
            DS4_SM75_NATIVE_SLOT4_ADD(0,v00,v01,v10,v11);
            DS4_SM75_NATIVE_SLOT4_ADD(1,v00,v01,v10,v11);
            DS4_SM75_NATIVE_SLOT4_ADD(2,v00,v01,v10,v11);
            DS4_SM75_NATIVE_SLOT4_ADD(3,v00,v01,v10,v11);
            DS4_SM75_NATIVE_SLOT4_ADD(4,v00,v01,v10,v11);
            DS4_SM75_NATIVE_SLOT4_ADD(5,v00,v01,v10,v11);
            DS4_SM75_NATIVE_SLOT4_ADD(6,v00,v01,v10,v11);
            DS4_SM75_NATIVE_SLOT4_ADD(7,v00,v01,v10,v11);
        }
    }
    DS4_SM75_NATIVE_REDUCE(s0,r0);DS4_SM75_NATIVE_REDUCE(s1,r1);
    DS4_SM75_NATIVE_REDUCE(s2,r2);DS4_SM75_NATIVE_REDUCE(s3,r3);
}

/* Streamed tile16 replacement for the rejected six-row gate kernel. Seven
 * complete MMA payload rows stay shared across every output-row group; the
 * remaining rows and the small d/bsums metadata use the read-only path. Gate
 * scratch uses the existing gate tensor, leaving declared shared memory below
 * 32 KiB while each packed weight fragment feeds both 8-pair halves. */
template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_sm75_native_q4_tile16_stream7_kernel(
        float *gate_out, float *up_out, float *mid_out,
        const char *gate_base, const char *up_base,
        const cuda_sm75_native_q8_K *xq,
        const uint32_t *sorted_pairs, const uint32_t *offsets,
        const uint32_t *counts, const uint32_t *tile_total,
        const uint32_t *tile_experts, const uint32_t *tile_starts,
        const float *weights, uint32_t xq_blocks,
        uint32_t expert_mid_dim, uint32_t n_expert,
        uint32_t write_aux, float clamp) {
    static_assert(7u * 16u * sizeof(cuda_sm75_native_q8_payload) +
                      3u * 16u * sizeof(uint32_t) + sizeof(uint32_t) <
                  32768u, "native Q4 gate stream7 must stay below 32 KiB");
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t expert = tile_experts[tile];
    const uint32_t local_start = tile_starts[tile];
    __shared__ cuda_sm75_native_q8_payload sxq[7][16];
    __shared__ uint32_t s_pair[16], s_tok[16], s_slot[16], s_np;
    if (threadIdx.x == 0u) {
        uint32_t np = 0u;
        for (; np < 16u; np++) {
            const uint32_t local = local_start + np;
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
    const uint32_t staged_rows=np<7u?np:7u;
    for (uint32_t i=threadIdx.x;i<staged_rows*xq_blocks*64u;
         i+=blockDim.x) {
        const uint32_t p=i/(xq_blocks*64u);
        const uint32_t q=i-p*xq_blocks*64u;
        const uint32_t b=q/64u,w=q-b*64u;
        const cuda_sm75_native_q8_K *src=
            xq+(uint64_t)s_tok[p]*xq_blocks+b;
        ((uint32_t *)&sxq[p][b])[w]=w<32u?
            ((const uint32_t *)src->low)[w]:
            ((const uint32_t *)src->high_signed)[w-32u];
    }
    __syncthreads();
    const uint32_t mtok_a = lane >> 2u;
    const uint32_t mtok_b = mtok_a + 8u;
    const uint32_t n0 = (lane & 3u) * 2u;
    for (uint32_t rr = 0; rr < ROW_SPAN / 64u; rr++) {
        const uint32_t row0 = blockIdx.x * ROW_SPAN +
            rr * 64u + warp * 8u;
        /* Every thread must reach the phase barrier even for a partial row
         * group. Inactive warps compute against row zero and discard it. */
        const bool row_group_active = row0 < expert_mid_dim;
        const uint32_t compute_row0 = row_group_active ? row0 : 0u;
        {
        float g0,g1,g2,g3;
        moe_sm75_native_q4_stream_matrix<7u>(
            gate_base,&sxq[0][0],xq,s_tok,np,xq_blocks,expert,
            expert_mid_dim,compute_row0,lane,g0,g1,g2,g3);
#pragma unroll
        for (uint32_t e = 0; e < 4u; e++) {
            const uint32_t p = e < 2u ? mtok_a : mtok_b;
            const uint32_t row_delta = n0 + (e & 1u);
            const uint32_t row = row0 + row_delta;
            if (!row_group_active || p >= np || row >= expert_mid_dim) continue;
            float gate=e==0u?g0:(e==1u?g1:(e==2u?g2:g3));
            if (clamp > 1.0e-6f && gate > clamp) gate = clamp;
            gate_out[(uint64_t)s_pair[p] * expert_mid_dim + row] = gate;
        }
        }
        __syncthreads();
        {
        float u0,u1,u2,u3;
        moe_sm75_native_q4_stream_matrix<7u>(
            up_base,&sxq[0][0],xq,s_tok,np,xq_blocks,expert,
            expert_mid_dim,compute_row0,lane,u0,u1,u2,u3);
#pragma unroll
        for (uint32_t e = 0; e < 4u; e++) {
            const uint32_t p = e < 2u ? mtok_a : mtok_b;
            const uint32_t row_delta = n0 + (e & 1u);
            const uint32_t row = row0 + row_delta;
            if (!row_group_active || p >= np || row >= expert_mid_dim) continue;
            const uint64_t off =
                (uint64_t)s_pair[p] * expert_mid_dim + row;
            const float gate = gate_out[off];
            float up=e==0u?u0:(e==1u?u1:(e==2u?u2:u3));
            if (clamp > 1.0e-6f) {
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            if (write_aux) up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
                weights[(uint64_t)s_tok[p] * n_expert + s_slot[p]];
        }
        }
    }
}

/* Native down consumer. TILE_PAIRS is instantiated as 16, 8, and 4. */
template <uint32_t ROW_SPAN, uint32_t TILE_PAIRS>
__global__ static void moe_down_sm75_native_q4_tile_kernel(
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
    const uint32_t local_start = tile_starts[tile];
    __shared__ cuda_sm75_native_q8_K sxq[TILE_PAIRS][8];
    __shared__ uint32_t s_pair[TILE_PAIRS], s_np;
    if (threadIdx.x == 0u) {
        uint32_t np = 0;
        for (; np < TILE_PAIRS; np++) {
            const uint32_t local = local_start + np;
            if (local >= counts[expert]) break;
            s_pair[np] = sorted_pairs[offsets[expert] + local];
        }
        s_np = np;
    }
    __syncthreads();
    const uint32_t np = s_np;
    const uint32_t words = (uint32_t)(sizeof(cuda_sm75_native_q8_K) / 4u);
    const uint32_t words_per_pair = midq_blocks * words;
    for (uint32_t i = threadIdx.x; i < TILE_PAIRS * words_per_pair;
         i += blockDim.x) {
        const uint32_t p = i / words_per_pair;
        const uint32_t w = i - p * words_per_pair;
        ((uint32_t *)sxq[p])[w] = p < np ?
            ((const uint32_t *)(midq +
                (uint64_t)s_pair[p] * midq_blocks))[w] : 0u;
    }
    __syncthreads();
    const uint32_t p0 = lane >> 2u;
    const uint32_t p1 = p0 + 8u;
    const uint32_t n0 = (lane & 3u) * 2u;
    for (uint32_t rr = 0; rr < ROW_SPAN / 64u; rr++) {
        const uint32_t row0 = blockIdx.x * ROW_SPAN +
            rr * 64u + warp * 8u;
        if (row0 >= out_dim) continue;
        DS4_SM75_NATIVE_SLOT4_DECL(0); DS4_SM75_NATIVE_SLOT4_DECL(1);
        DS4_SM75_NATIVE_SLOT4_DECL(2); DS4_SM75_NATIVE_SLOT4_DECL(3);
        DS4_SM75_NATIVE_SLOT4_DECL(4); DS4_SM75_NATIVE_SLOT4_DECL(5);
        DS4_SM75_NATIVE_SLOT4_DECL(6); DS4_SM75_NATIVE_SLOT4_DECL(7);
        const uint64_t native_tile =
            (uint64_t)expert * (out_dim / 8u) + row0 / 8u;
#pragma unroll
        for (uint32_t b = 0; b < 8u; b++) {
            if (b >= midq_blocks) continue;
            const cuda_sm75_native_q4_tile *w =
                (const cuda_sm75_native_q4_tile *)down_base +
                native_tile * midq_blocks + b;
            const uint4 h0 = w->hdr[n0], h1 = w->hdr[n0 + 1u];
            const cuda_sm75_native_q8_K *a0=
                p0<TILE_PAIRS?&sxq[p0][b]:&sxq[0][b];
            const cuda_sm75_native_q8_K *a1=
                TILE_PAIRS>8u?&sxq[p1][b]:a0;
            int i00 = 0, i01 = 0, i10 = 0, i11 = 0;
            int m00 = 0, m01 = 0, m10 = 0, m11 = 0;
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                const uint32_t wf = w->b[j][lane];
                int32_t l00=0,l01=0,h00=0,h01=0;
                int32_t l10=0,l11=0,h10=0,h11=0;
                mma_m8n8k32_u4_u4(l00,l01,a0->low[j][lane&3u],wf);
                mma_m8n8k32_s4_u4(h00,h01,a0->high_signed[j][lane&3u],wf);
                if (TILE_PAIRS > 8u) {
                    mma_m8n8k32_u4_u4(l10,l11,a1->low[j][lane&3u],wf);
                    mma_m8n8k32_s4_u4(h10,h11,a1->high_signed[j][lane&3u],wf);
                }
                const int c00=l00+16*h00, c01=l01+16*h01;
                const int c10=l10+16*h10, c11=l11+16*h11;
                const int bs0=(int)a0->bsums[2u*j]+(int)a0->bsums[2u*j+1u];
                const int bs1=TILE_PAIRS > 8u ?
                    (int)a1->bsums[2u*j]+(int)a1->bsums[2u*j+1u] : 0;
                uint8_t sc0,mn0,sc1,mn1;
                dev_q4_K_get_scale_min(j,(const uint8_t *)&h0.y,&sc0,&mn0);
                dev_q4_K_get_scale_min(j,(const uint8_t *)&h1.y,&sc1,&mn1);
                i00+=(int)sc0*c00; i01+=(int)sc1*c01;
                i10+=(int)sc0*c10; i11+=(int)sc1*c11;
                m00+=(int)mn0*bs0; m01+=(int)mn1*bs0;
                m10+=(int)mn0*bs1; m11+=(int)mn1*bs1;
            }
            const float d0=dev_f16_to_f32((uint16_t)h0.x);
            const float z0=dev_f16_to_f32((uint16_t)(h0.x>>16u));
            const float d1=dev_f16_to_f32((uint16_t)h1.x);
            const float z1=dev_f16_to_f32((uint16_t)(h1.x>>16u));
            const float v00=a0->d*d0*(float)i00-a0->d*z0*(float)m00;
            const float v01=a0->d*d1*(float)i01-a0->d*z1*(float)m01;
            const float v10=TILE_PAIRS > 8u ?
                a1->d*d0*(float)i10-a1->d*z0*(float)m10 : 0.0f;
            const float v11=TILE_PAIRS > 8u ?
                a1->d*d1*(float)i11-a1->d*z1*(float)m11 : 0.0f;
            switch (b) {
                DS4_SM75_NATIVE_SLOT4_ADD(0,v00,v01,v10,v11);
                DS4_SM75_NATIVE_SLOT4_ADD(1,v00,v01,v10,v11);
                DS4_SM75_NATIVE_SLOT4_ADD(2,v00,v01,v10,v11);
                DS4_SM75_NATIVE_SLOT4_ADD(3,v00,v01,v10,v11);
                DS4_SM75_NATIVE_SLOT4_ADD(4,v00,v01,v10,v11);
                DS4_SM75_NATIVE_SLOT4_ADD(5,v00,v01,v10,v11);
                DS4_SM75_NATIVE_SLOT4_ADD(6,v00,v01,v10,v11);
                DS4_SM75_NATIVE_SLOT4_ADD(7,v00,v01,v10,v11);
            }
        }
        float o0,o1,o2,o3;
        DS4_SM75_NATIVE_REDUCE(s0,o0); DS4_SM75_NATIVE_REDUCE(s1,o1);
        DS4_SM75_NATIVE_REDUCE(s2,o2); DS4_SM75_NATIVE_REDUCE(s3,o3);
        if (p0 < TILE_PAIRS && p0 < np) {
            const uint32_t r0=row0+n0, r1=r0+1u;
            if (r0<out_dim) down_out[(uint64_t)s_pair[p0]*out_dim+r0]=o0;
            if (r1<out_dim) down_out[(uint64_t)s_pair[p0]*out_dim+r1]=o1;
        }
        if (p1 < TILE_PAIRS && p1 < np) {
            const uint32_t r0=row0+n0, r1=r0+1u;
            if (r0<out_dim) down_out[(uint64_t)s_pair[p1]*out_dim+r0]=o2;
            if (r1<out_dim) down_out[(uint64_t)s_pair[p1]*out_dim+r1]=o3;
        }
    }
}

/* Compact tile16 down stages the MMA payload for seven of eight K blocks once
 * per CTA (28 KiB). The final block and all d/bsums metadata use the read-only
 * path. This preserves low8/high8 weight reuse and exact slot order without
 * either the rejected half-row global traffic or per-row-group restaging. */
template <uint32_t ROW_SPAN>
__global__ static void moe_down_sm75_native_q4_tile16_compact7_kernel(
        float *down_out, const char *down_base,
        const cuda_sm75_native_q8_K *midq,
        const uint32_t *sorted_pairs, const uint32_t *offsets,
        const uint32_t *counts, const uint32_t *tile_total,
        const uint32_t *tile_experts, const uint32_t *tile_starts,
        uint32_t midq_blocks, uint32_t out_dim) {
    static_assert(16u * 7u * sizeof(cuda_sm75_native_q8_payload) +
                      16u * sizeof(uint32_t) + sizeof(uint32_t) < 32768u,
                  "native Q4 down compact7 must stay below 32 KiB");
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t expert = tile_experts[tile];
    const uint32_t local_start = tile_starts[tile];
    __shared__ cuda_sm75_native_q8_payload sxq[16][7];
    __shared__ uint32_t s_pair[16], s_np;
    if (threadIdx.x == 0u) {
        uint32_t np = 0u;
        for (; np < 16u; np++) {
            const uint32_t local = local_start + np;
            if (local >= counts[expert]) break;
            s_pair[np] = sorted_pairs[offsets[expert] + local];
        }
        s_np = np;
    }
    __syncthreads();
    const uint32_t np = s_np;
    const uint32_t staged_blocks=midq_blocks<7u?midq_blocks:7u;
    for (uint32_t i=threadIdx.x;i<16u*staged_blocks*64u;
         i+=blockDim.x) {
        const uint32_t p=i/(staged_blocks*64u);
        const uint32_t q=i-p*staged_blocks*64u;
        const uint32_t b=q/64u,w=q-b*64u;
        const cuda_sm75_native_q8_K *src=p<np?
            midq+(uint64_t)s_pair[p]*midq_blocks+b:midq;
        ((uint32_t *)&sxq[p][b])[w]=p<np?(w<32u?
            ((const uint32_t *)src->low)[w]:
            ((const uint32_t *)src->high_signed)[w-32u]):0u;
    }
    __syncthreads();
    const uint32_t p0 = lane >> 2u;
    const uint32_t p1 = p0 + 8u;
    const uint32_t n0 = (lane & 3u) * 2u;
    for (uint32_t rr = 0; rr < ROW_SPAN / 64u; rr++) {
        const uint32_t row0 = blockIdx.x * ROW_SPAN + rr * 64u + warp * 8u;
        const bool row_group_active = row0 < out_dim;
        const uint32_t compute_row0 = row_group_active ? row0 : 0u;
        DS4_SM75_NATIVE_SLOT4_DECL(0); DS4_SM75_NATIVE_SLOT4_DECL(1);
        DS4_SM75_NATIVE_SLOT4_DECL(2); DS4_SM75_NATIVE_SLOT4_DECL(3);
        DS4_SM75_NATIVE_SLOT4_DECL(4); DS4_SM75_NATIVE_SLOT4_DECL(5);
        DS4_SM75_NATIVE_SLOT4_DECL(6); DS4_SM75_NATIVE_SLOT4_DECL(7);
        for (uint32_t b=0;b<midq_blocks;b++) {
            const bool have0=p0<np,have1=p1<np;
            const cuda_sm75_native_q8_K *m0=have0?
                midq+(uint64_t)s_pair[p0]*midq_blocks+b:midq;
            const cuda_sm75_native_q8_K *m1=have1?
                midq+(uint64_t)s_pair[p1]*midq_blocks+b:midq;
            const cuda_sm75_native_q8_payload *a0=b<staged_blocks?
                &sxq[p0][b]:NULL;
            const cuda_sm75_native_q8_payload *a1=b<staged_blocks?
                &sxq[p1][b]:NULL;
            const uint32_t (*low0)[4]=a0?a0->low:m0->low;
            const uint32_t (*high0)[4]=a0?a0->high_signed:m0->high_signed;
            const uint32_t (*low1)[4]=a1?a1->low:m1->low;
            const uint32_t (*high1)[4]=a1?a1->high_signed:m1->high_signed;
            float v00,v01,v10,v11;
            moe_sm75_native_q4_pair_block(
                down_base,low0,high0,low1,high1,m0,m1,have0,have1,b,
                midq_blocks,expert,out_dim,compute_row0,lane,
                v00,v01,v10,v11);
            switch (b & 7u) {
                DS4_SM75_NATIVE_SLOT4_ADD(0,v00,v01,v10,v11);
                DS4_SM75_NATIVE_SLOT4_ADD(1,v00,v01,v10,v11);
                DS4_SM75_NATIVE_SLOT4_ADD(2,v00,v01,v10,v11);
                DS4_SM75_NATIVE_SLOT4_ADD(3,v00,v01,v10,v11);
                DS4_SM75_NATIVE_SLOT4_ADD(4,v00,v01,v10,v11);
                DS4_SM75_NATIVE_SLOT4_ADD(5,v00,v01,v10,v11);
                DS4_SM75_NATIVE_SLOT4_ADD(6,v00,v01,v10,v11);
                DS4_SM75_NATIVE_SLOT4_ADD(7,v00,v01,v10,v11);
            }
        }
        float o0, o1, o2, o3;
        DS4_SM75_NATIVE_REDUCE(s0,o0); DS4_SM75_NATIVE_REDUCE(s1,o1);
        DS4_SM75_NATIVE_REDUCE(s2,o2); DS4_SM75_NATIVE_REDUCE(s3,o3);
        if (row_group_active && p0 < np) {
            const uint32_t r0 = row0 + n0, r1 = r0 + 1u;
            if (r0 < out_dim)
                down_out[(uint64_t)s_pair[p0] * out_dim + r0] = o0;
            if (r1 < out_dim)
                down_out[(uint64_t)s_pair[p0] * out_dim + r1] = o1;
        }
        if (row_group_active && p1 < np) {
            const uint32_t r0 = row0 + n0, r1 = r0 + 1u;
            if (r0 < out_dim)
                down_out[(uint64_t)s_pair[p1] * out_dim + r0] = o2;
            if (r1 < out_dim)
                down_out[(uint64_t)s_pair[p1] * out_dim + r1] = o3;
        }
    }
}

#undef DS4_SM75_NATIVE_REDUCE
#undef DS4_SM75_NATIVE_SLOT4_ADD
#undef DS4_SM75_NATIVE_SLOT4_DECL
