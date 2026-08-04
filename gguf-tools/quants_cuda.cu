/*
 * CUDA encoders for the routed-MoE quantization formats used by DS4.
 *
 * One CUDA thread owns one 256-value GGML block.  The searches within a block
 * are intentionally serial so their decisions follow the reference C
 * quantizer, while thousands of independent blocks execute concurrently.
 * Host threads keep a stream and device buffers alive across experts.
 */

#include "quants_cuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define QK_K 256
#define GROUP_EPS 1e-15f

static __device__ __forceinline__ int d_min_i(int a, int b) { return a < b ? a : b; }
static __device__ __forceinline__ int d_max_i(int a, int b) { return a > b ? a : b; }
static __device__ __forceinline__ float d_max_f(float a, float b) { return a > b ? a : b; }
static __device__ __forceinline__ int d_nearest(float x) { return __float2int_rn(x); }

static __device__ __forceinline__ uint16_t d_f16_bits(float x) {
    return __half_as_ushort(__float2half_rn(x));
}

static __device__ __forceinline__ float d_f16_value(uint16_t x) {
    return __half2float(__ushort_as_half(x));
}

static __device__ __forceinline__ void d_store_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
}

static __device__ __forceinline__ void d_store_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static __device__ float d_make_qkx3_quants(int n, int nmax,
                                            const float *x, const float *weights,
                                            uint8_t *L, float *the_min,
                                            uint8_t *Laux,
                                            float rmin, float rdelta, int nstep) {
    float minv = x[0];
    float maxv = x[0];
    float sum_w = weights ? weights[0] : x[0] * x[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; i++) {
        minv = fminf(minv, x[i]);
        maxv = fmaxf(maxv, x[i]);
        const float w = weights ? weights[i] : x[i] * x[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (minv > 0) minv = 0;
    if (maxv <= minv) {
        for (int i = 0; i < n; i++) L[i] = 0;
        *the_min = -minv;
        return 0;
    }

    float iscale = nmax / (maxv - minv);
    float scale = 1 / iscale;
    float best = 0;
    for (int i = 0; i < n; i++) {
        int l = d_nearest(iscale * (x[i] - minv));
        l = d_max_i(0, d_min_i(nmax, l));
        L[i] = (uint8_t)l;
        const float diff = scale * l + minv - x[i];
        const float w = weights ? weights[i] : x[i] * x[i];
        best += w * diff * diff;
    }

    for (int is = 0; is <= nstep; is++) {
        iscale = (rmin + rdelta * is + nmax) / (maxv - minv);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; i++) {
            int l = d_nearest(iscale * (x[i] - minv));
            l = d_max_i(0, d_min_i(nmax, l));
            Laux[i] = (uint8_t)l;
            const float w = weights ? weights[i] : x[i] * x[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        const float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float error = 0;
            for (int i = 0; i < n; i++) {
                const float diff = this_scale * Laux[i] + this_min - x[i];
                const float w = weights ? weights[i] : x[i] * x[i];
                error += w * diff * diff;
            }
            if (error < best) {
                for (int i = 0; i < n; i++) L[i] = Laux[i];
                best = error;
                scale = this_scale;
                minv = this_min;
            }
        }
    }
    *the_min = -minv;
    return scale;
}

static __device__ float d_make_qp_quants(int n, int nmax,
                                          const float *x, uint8_t *L,
                                          const float *weights) {
    float maxv = 0;
    for (int i = 0; i < n; i++) maxv = d_max_f(maxv, x[i]);
    if (maxv < GROUP_EPS) {
        for (int i = 0; i < n; i++) L[i] = 0;
        return 0;
    }

    float iscale = nmax / maxv;
    for (int i = 0; i < n; i++) L[i] = (uint8_t)d_nearest(iscale * x[i]);
    float scale = 1 / iscale;
    float best_mse = 0;
    for (int i = 0; i < n; i++) {
        const float diff = x[i] - scale * L[i];
        best_mse += weights[i] * diff * diff;
    }
    for (int is = -4; is <= 4; is++) {
        if (is == 0) continue;
        const float iscale_is = (0.1f * is + nmax) / maxv;
        const float scale_is = 1 / iscale_is;
        float mse = 0;
        for (int i = 0; i < n; i++) {
            int l = d_nearest(iscale_is * x[i]);
            l = d_min_i(nmax, l);
            const float diff = x[i] - scale_is * l;
            mse += weights[i] * diff * diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }

    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; i++) {
        int l = d_nearest(iscale * x[i]);
        l = d_min_i(nmax, l);
        L[i] = (uint8_t)l;
        const float w = weights[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; itry++) {
        int changed = 0;
        for (int i = 0; i < n; i++) {
            const float w = weights[i];
            float slx = sumlx - w * x[i] * L[i];
            float sl2 = suml2 - w * L[i] * L[i];
            if (slx > 0 && sl2 > 0) {
                int nl = d_nearest(x[i] * sl2 / slx);
                nl = d_min_i(nmax, nl);
                if (nl != L[i]) {
                    slx += w * x[i] * nl;
                    sl2 += w * nl * nl;
                    if (slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i] = (uint8_t)nl;
                        sumlx = slx;
                        suml2 = sl2;
                        changed++;
                    }
                }
            }
        }
        if (!changed) break;
    }
    return suml2 > 0 ? sumlx / suml2 : 0;
}

static __device__ __forceinline__ void d_get_scale_min_k4(
        int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static __device__ void d_write_q4_k_weighted(const float *x, uint8_t *y,
                                               const float *quant_weights) {
    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[8], Lm[8];
    float weights[32];
    float sw[8], mins[8], scales[8];
    uint8_t packed_scales[12] = {0};

    float sum_x2 = 0;
    for (int i = 0; i < QK_K; i++) sum_x2 += x[i] * x[i];
    const float sigma2 = 2 * sum_x2 / QK_K;

    for (int j = 0; j < 8; j++) {
        float sumw = 0;
        for (int l = 0; l < 32; l++) {
            weights[l] = quant_weights[32 * j + l] *
                         sqrtf(sigma2 + x[32 * j + l] * x[32 * j + l]);
            sumw += weights[l];
        }
        sw[j] = sumw;
        scales[j] = d_make_qkx3_quants(32, 15, x + 32 * j, weights,
                                        L + 32 * j, &mins[j], Laux,
                                        -0.9f, 0.05f, 36);
    }

    const float d_block = d_make_qp_quants(8, 63, scales, Ls, sw);
    const float m_block = d_make_qp_quants(8, 63, mins, Lm, sw);
    for (int j = 0; j < 8; j++) {
        const uint8_t ls = Ls[j];
        const uint8_t lm = Lm[j];
        if (j < 4) {
            packed_scales[j] = ls;
            packed_scales[j + 4] = lm;
        } else {
            packed_scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            packed_scales[j - 4] |= (ls >> 4) << 6;
            packed_scales[j] |= (lm >> 4) << 6;
        }
    }

    const uint16_t hd = d_f16_bits(d_block);
    const uint16_t hm = d_f16_bits(m_block);
    d_store_u16(y, hd);
    d_store_u16(y + 2, hm);
    for (int i = 0; i < 12; i++) y[4 + i] = packed_scales[i];

    for (int j = 0; j < 8; j++) {
        uint8_t sc, m;
        d_get_scale_min_k4(j, packed_scales, &sc, &m);
        const float dd = d_f16_value(hd) * sc;
        if (!dd) continue;
        const float dm = d_f16_value(hm) * m;
        for (int i = 0; i < 32; i++) {
            int l = d_nearest((x[32 * j + i] + dm) / dd);
            L[32 * j + i] = (uint8_t)d_max_i(0, d_min_i(15, l));
        }
    }
    uint8_t *q = y + 16;
    for (int j = 0; j < QK_K; j += 64) {
        for (int l = 0; l < 32; l++) q[l] = L[j + l] | (L[j + l + 32] << 4);
        q += 32;
    }
}

static __device__ void d_write_q2_k_weighted(const float *x, uint8_t *y,
                                               const float *quant_weights) {
    uint8_t L[QK_K];
    uint8_t Laux[16];
    uint8_t Ls[16], Lm[16];
    float mins[16], scales[16], sw[16], weights[16];

    float sum_x2 = 0;
    for (int i = 0; i < QK_K; i++) sum_x2 += x[i] * x[i];
    const float sigma2 = sum_x2 / QK_K;
    for (int j = 0; j < 16; j++) {
        float sumw = 0;
        for (int l = 0; l < 16; l++) {
            weights[l] = quant_weights[16 * j + l] *
                         sqrtf(sigma2 + x[16 * j + l] * x[16 * j + l]);
            sumw += weights[l];
        }
        sw[j] = sumw;
        scales[j] = d_make_qkx3_quants(16, 3, x + 16 * j, weights,
                                        L + 16 * j, &mins[j], Laux,
                                        -0.9f, 0.05f, 36);
    }

    const float d_block = d_make_qp_quants(16, 15, scales, Ls, sw);
    const float m_block = d_make_qp_quants(16, 15, mins, Lm, sw);
    const uint16_t hd = d_f16_bits(d_block);
    const uint16_t hm = d_f16_bits(m_block);
    const float d = d_f16_value(hd);
    const float m = d_f16_value(hm);
    for (int j = 0; j < 16; j++) y[j] = Ls[j] | (Lm[j] << 4);
    d_store_u16(y + 80, hd);
    d_store_u16(y + 82, hm);

    for (int j = 0; j < 16; j++) {
        const float ds = d * (y[j] & 0xF);
        if (!ds) continue;
        const float ms = m * (y[j] >> 4);
        for (int i = 0; i < 16; i++) {
            int l = d_nearest((x[16 * j + i] + ms) / ds);
            L[16 * j + i] = (uint8_t)d_max_i(0, d_min_i(3, l));
        }
    }
    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; l++) {
            y[16 + j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) |
                                 (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
        }
    }
}

static __device__ int d_iq2_best_neighbour(const uint16_t *neighbours,
                                            const uint64_t *grid,
                                            const float *xval,
                                            const float *weight,
                                            float scale,
                                            uint8_t *L) {
    const int n = neighbours[0];
    float best = FLT_MAX;
    int best_index = -1;
    for (int j = 1; j <= n; j++) {
        const int index = neighbours[j];
        const int8_t *pg = (const int8_t *)(grid + index);
        float d2 = 0;
        for (int i = 0; i < 8; i++) {
            const float diff = scale * pg[i] - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best) {
            best = d2;
            best_index = index;
        }
    }
    const int8_t *pg = (const int8_t *)(grid + best_index);
    for (int i = 0; i < 8; i++) L[i] = (uint8_t)((pg[i] - 1) / 2);
    return best_index;
}

static __device__ void d_write_iq2_xxs(const float *x, uint8_t *y,
                                        const float *quant_weights,
                                        const uint64_t *grid,
                                        const int *map,
                                        const uint16_t *neighbours) {
    uint32_t q2[16] = {0};
    float scales[8];
    float weight[32], xval[32], waux[32];
    uint8_t L[32], Laux[32], signs[4];
    d_store_u16(y, d_f16_bits(0));

    float sum_x2 = 0;
    for (int i = 0; i < QK_K; i++) sum_x2 += x[i] * x[i];
    const float sigma2 = sum_x2 / QK_K;
    float max_scale = 0;

    for (int ib = 0; ib < 8; ib++) {
        const float *xb = x + 32 * ib;
        const float *qw = quant_weights + 32 * ib;
        for (int i = 0; i < 32; i++) {
            weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            waux[i] = sqrtf(weight[i]);
        }
        for (int k = 0; k < 4; k++) {
            int nflip = 0;
            uint8_t s = 0;
            for (int i = 0; i < 8; i++) {
                const float v = xb[8 * k + i];
                xval[8 * k + i] = v >= 0 ? v : -v;
                if (v < 0) {
                    nflip++;
                    s |= (uint8_t)(1u << i);
                }
            }
            if (nflip & 1) {
                int imin = 0;
                float minv = weight[8 * k] * xb[8 * k] * xb[8 * k];
                for (int i = 1; i < 8; i++) {
                    const float v = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                    if (v < minv) {
                        minv = v;
                        imin = i;
                    }
                }
                xval[8 * k + imin] = -xval[8 * k + imin];
                s ^= (uint8_t)(1u << imin);
            }
            signs[k] = s & 127;
        }

        float maxv = xval[0];
        for (int i = 1; i < 32; i++) maxv = d_max_f(maxv, xval[i]);
        if (maxv < GROUP_EPS) {
            scales[ib] = 0;
            for (int i = 0; i < 32; i++) L[i] = 0;
            continue;
        }

        float scale = d_make_qp_quants(32, 4, xval, L, weight);
        const float eff_max = scale * 3;
        if (eff_max <= 0) {
            scales[ib] = 0;
            for (int i = 0; i < 32; i++) L[i] = 0;
            continue;
        }

        float best = 0;
        for (int is = -6; is <= 6; is++) {
            const float id = (5 + is * 0.1f) / eff_max;
            const float this_scale = 1 / id;
            for (int k = 0; k < 4; k++) {
                uint16_t u = 0;
                for (int i = 0; i < 8; i++) {
                    int l = d_nearest(0.5f * (id * xval[8 * k + i] - 1));
                    l = d_max_i(0, d_min_i(2, l));
                    Laux[8 * k + i] = (uint8_t)l;
                    u |= (uint16_t)(l << (2 * i));
                }
                if (map[u] < 0) {
                    const uint16_t *nbs = neighbours - map[u] - 1;
                    d_iq2_best_neighbour(nbs, grid, xval + 8 * k,
                                         waux + 8 * k, this_scale, Laux + 8 * k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 32; i++) {
                const float q = 2 * Laux[i] + 1;
                sumqx += weight[i] * xval[i] * q;
                sumq2 += weight[i] * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                for (int i = 0; i < 32; i++) L[i] = Laux[i];
            }
        }

        if (scale > 0) {
            const float id = 1 / scale;
            for (int k = 0; k < 4; k++) {
                uint16_t u = 0;
                for (int i = 0; i < 8; i++) {
                    int l = d_nearest(0.5f * (id * xval[8 * k + i] - 1));
                    l = d_max_i(0, d_min_i(2, l));
                    u |= (uint16_t)(l << (2 * i));
                }
                int grid_index = map[u];
                if (grid_index < 0) {
                    const uint16_t *nbs = neighbours - map[u] - 1;
                    grid_index = d_iq2_best_neighbour(nbs, grid, xval + 8 * k,
                                                       waux + 8 * k, scale, L + 8 * k);
                }
                const int8_t *pg = (const int8_t *)(grid + grid_index);
                for (int i = 0; i < 8; i++) L[8 * k + i] = (uint8_t)((pg[i] - 1) / 2);
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 32; i++) {
                const float q = 2 * L[i] + 1;
                sumqx += weight[i] * xval[i] * q;
                sumq2 += weight[i] * q * q;
            }
            if (sumq2 > 0) scale = sumqx / sumq2;
        }

        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < 4; k++) signs[k] = (~signs[k]) & 127;
        }
        for (int k = 0; k < 4; k++) {
            uint16_t u = 0;
            for (int i = 0; i < 8; i++) u |= (uint16_t)(L[8 * k + i] << (2 * i));
            const int grid_index = map[u];
            q2[2 * ib] |= (uint32_t)grid_index << (8 * k);
            q2[2 * ib + 1] |= (uint32_t)signs[k] << (7 * k);
        }
        scales[ib] = scale;
        max_scale = d_max_f(max_scale, scale);
    }

    if (!max_scale) {
        for (int i = 0; i < 64; i++) y[2 + i] = 0;
        return;
    }
    const float d = max_scale / 31;
    d_store_u16(y, d_f16_bits(d));
    const float id = 1 / d;
    for (int ib = 0; ib < 8; ib++) {
        int l = d_nearest(0.5f * (id * scales[ib] - 1));
        l = d_max_i(0, d_min_i(15, l));
        q2[2 * ib + 1] |= (uint32_t)l << 28;
    }
    for (int i = 0; i < 16; i++) d_store_u32(y + 2 + 4 * i, q2[i]);
}

static __global__ __launch_bounds__(64) void q4_k_kernel(
        const float *src, uint8_t *dst, const float *imatrix,
        int64_t nblocks, int64_t blocks_per_row) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    d_write_q4_k_weighted(src + b * QK_K, dst + b * 144,
                           imatrix + (b % blocks_per_row) * QK_K);
}

static __global__ __launch_bounds__(64) void q2_k_kernel(
        const float *src, uint8_t *dst, const float *imatrix,
        int64_t nblocks, int64_t blocks_per_row) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    d_write_q2_k_weighted(src + b * QK_K, dst + b * 84,
                           imatrix + (b % blocks_per_row) * QK_K);
}

static __global__ __launch_bounds__(64) void iq2_xxs_kernel(
        const float *src, uint8_t *dst, const float *imatrix,
        int64_t nblocks, int64_t blocks_per_row,
        const uint64_t *grid, const int *map, const uint16_t *neighbours) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    d_write_iq2_xxs(src + b * QK_K, dst + b * 66,
                     imatrix + (b % blocks_per_row) * QK_K,
                     grid, map, neighbours);
}

/* One CTA transforms one eight-row by one-Q4-block record. The first warp
 * copies the eight 16-byte headers; all eight warps emit one packed m8n8k32
 * B-fragment word apiece. */
static __global__ __launch_bounds__(256) void repack_sm75_native_q4_kernel(
        const uint8_t *src, uint8_t *dst,
        uint64_t records, uint32_t blocks_per_row) {
    const uint64_t record = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (record >= records) return;
    const uint64_t tile = record / blocks_per_row;
    const uint32_t block = (uint32_t)(record - tile * blocks_per_row);
    const uint32_t group = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t row = lane >> 2u;
    const uint32_t lane4 = lane & 3u;
    const uint64_t src_block =
        ((tile * 8u + row) * blocks_per_row + block) * 144u;
    uint8_t *record_dst = dst + record * (8u * 144u);
    if (tid < 32u) {
        ((uint32_t *)record_dst)[tid] =
            ((const uint32_t *)(src + src_block))[tid & 3u];
    }
    const uint8_t *qs = src + src_block + 16u;
    const uint32_t off = (group >> 1u) * 32u + lane4 * 8u;
    const uint32_t shift = (group & 1u) ? 4u : 0u;
    uint32_t packed = 0u;
#pragma unroll
    for (uint32_t i = 0; i < 4u; i++) {
        packed |= ((uint32_t)((qs[off + i] >> shift) & 15u)) << (4u * i);
        packed |= ((uint32_t)((qs[off + 4u + i] >> shift) & 15u))
                  << (4u * (i + 4u));
    }
    ((uint32_t *)(record_dst + 8u * 16u))[group * 32u + lane] = packed;
}

struct cuda_thread_state {
    int device;
    cudaStream_t stream;
    float *src;
    float *imatrix;
    uint8_t *dst;
    size_t src_cap, imatrix_cap, dst_cap;
    uint64_t *grid;
    int *map;
    uint16_t *neighbours;
};

static __thread cuda_thread_state tls;
static __thread bool tls_initialized;

static void tls_prepare(void) {
    if (tls_initialized) return;
    memset(&tls, 0, sizeof(tls));
    tls.device = -1;
    tls_initialized = true;
}

static void tls_release(void) {
    if (!tls_initialized) return;
    if (tls.device >= 0) cudaSetDevice(tls.device);
    if (tls.src) cudaFree(tls.src);
    if (tls.imatrix) cudaFree(tls.imatrix);
    if (tls.dst) cudaFree(tls.dst);
    if (tls.grid) cudaFree(tls.grid);
    if (tls.map) cudaFree(tls.map);
    if (tls.neighbours) cudaFree(tls.neighbours);
    if (tls.stream) cudaStreamDestroy(tls.stream);
    memset(&tls, 0, sizeof(tls));
    tls.device = -1;
}

static void set_cuda_error(char *error, size_t cap, const char *where, cudaError_t status) {
    if (!error || !cap) return;
    snprintf(error, cap, "%s: %s", where, cudaGetErrorString(status));
}

static bool ensure_allocation(void **ptr, size_t *cap, size_t need,
                              char *error, size_t error_cap) {
    if (*cap >= need) return true;
    if (*ptr) {
        const cudaError_t st = cudaFree(*ptr);
        if (st != cudaSuccess) {
            set_cuda_error(error, error_cap, "cudaFree", st);
            return false;
        }
        *ptr = NULL;
        *cap = 0;
    }
    const cudaError_t st = cudaMalloc(ptr, need);
    if (st != cudaSuccess) {
        set_cuda_error(error, error_cap, "cudaMalloc", st);
        return false;
    }
    *cap = need;
    return true;
}

static bool ensure_device(int device, bool need_iq2, char *error, size_t error_cap) {
    tls_prepare();
    if (tls.device != device) {
        tls_release();
        cudaError_t st = cudaSetDevice(device);
        if (st != cudaSuccess) {
            set_cuda_error(error, error_cap, "cudaSetDevice", st);
            return false;
        }
        st = cudaStreamCreateWithFlags(&tls.stream, cudaStreamNonBlocking);
        if (st != cudaSuccess) {
            set_cuda_error(error, error_cap, "cudaStreamCreate", st);
            return false;
        }
        tls.device = device;
    }
    if (!need_iq2 || (tls.grid && tls.map && tls.neighbours)) return true;

    const uint64_t *h_grid = NULL;
    const int *h_map = NULL;
    const uint16_t *h_neighbours = NULL;
    size_t grid_len = 0, map_len = 0, neighbours_len = 0;
    if (!ds4q_iq2_xxs_tables(&h_grid, &grid_len, &h_map, &map_len,
                              &h_neighbours, &neighbours_len)) {
        if (error && error_cap) snprintf(error, error_cap, "failed to initialize IQ2 tables");
        return false;
    }
    cudaError_t st = cudaMalloc((void **)&tls.grid, grid_len * sizeof(*h_grid));
    if (st == cudaSuccess) st = cudaMalloc((void **)&tls.map, map_len * sizeof(*h_map));
    if (st == cudaSuccess) st = cudaMalloc((void **)&tls.neighbours,
                                           neighbours_len * sizeof(*h_neighbours));
    if (st == cudaSuccess) st = cudaMemcpyAsync(tls.grid, h_grid,
                                                grid_len * sizeof(*h_grid),
                                                cudaMemcpyHostToDevice, tls.stream);
    if (st == cudaSuccess) st = cudaMemcpyAsync(tls.map, h_map,
                                                map_len * sizeof(*h_map),
                                                cudaMemcpyHostToDevice, tls.stream);
    if (st == cudaSuccess) st = cudaMemcpyAsync(tls.neighbours, h_neighbours,
                                                neighbours_len * sizeof(*h_neighbours),
                                                cudaMemcpyHostToDevice, tls.stream);
    if (st == cudaSuccess) st = cudaStreamSynchronize(tls.stream);
    if (st != cudaSuccess) {
        set_cuda_error(error, error_cap, "copy IQ2 tables", st);
        tls_release();
        return false;
    }
    return true;
}

bool ds4q_cuda_type_supported(ds4q_type type) {
    return type == DS4Q_TYPE_IQ2_XXS || type == DS4Q_TYPE_Q4_K || type == DS4Q_TYPE_Q2_K;
}

int ds4q_cuda_device_count(char *error, size_t error_cap) {
    int count = 0;
    const cudaError_t st = cudaGetDeviceCount(&count);
    if (st != cudaSuccess) {
        set_cuda_error(error, error_cap, "cudaGetDeviceCount", st);
        return -1;
    }
    return count;
}

size_t ds4q_cuda_quantize_chunk(ds4q_type type,
                                const float *src,
                                void *dst,
                                int64_t nrows,
                                int64_t ncols,
                                const float *imatrix,
                                int device,
                                char *error,
                                size_t error_cap) {
    if (error && error_cap) error[0] = '\0';
    if (!ds4q_cuda_type_supported(type) || !src || !dst || !imatrix ||
        nrows <= 0 || ncols <= 0 || ncols % QK_K != 0) {
        if (error && error_cap) snprintf(error, error_cap, "invalid CUDA quantization request");
        return 0;
    }
    if (!ensure_device(device, type == DS4Q_TYPE_IQ2_XXS, error, error_cap)) return 0;

    const int64_t blocks_per_row = ncols / QK_K;
    const int64_t nblocks = nrows * blocks_per_row;
    const size_t src_bytes = (size_t)nrows * (size_t)ncols * sizeof(float);
    const size_t imatrix_bytes = (size_t)ncols * sizeof(float);
    const size_t block_bytes = type == DS4Q_TYPE_IQ2_XXS ? 66 :
                               type == DS4Q_TYPE_Q4_K ? 144 : 84;
    const size_t dst_bytes = (size_t)nblocks * block_bytes;
    if (!ensure_allocation((void **)&tls.src, &tls.src_cap, src_bytes, error, error_cap) ||
        !ensure_allocation((void **)&tls.imatrix, &tls.imatrix_cap, imatrix_bytes, error, error_cap) ||
        !ensure_allocation((void **)&tls.dst, &tls.dst_cap, dst_bytes, error, error_cap)) {
        return 0;
    }

    cudaError_t st = cudaMemcpyAsync(tls.src, src, src_bytes,
                                     cudaMemcpyHostToDevice, tls.stream);
    if (st == cudaSuccess) st = cudaMemcpyAsync(tls.imatrix, imatrix, imatrix_bytes,
                                                cudaMemcpyHostToDevice, tls.stream);
    if (st != cudaSuccess) {
        set_cuda_error(error, error_cap, "copy quantization input", st);
        return 0;
    }

    const int threads = 64;
    const unsigned blocks = (unsigned)((nblocks + threads - 1) / threads);
    if (type == DS4Q_TYPE_IQ2_XXS) {
        iq2_xxs_kernel<<<blocks, threads, 0, tls.stream>>>(
                tls.src, tls.dst, tls.imatrix, nblocks, blocks_per_row,
                tls.grid, tls.map, tls.neighbours);
    } else if (type == DS4Q_TYPE_Q4_K) {
        q4_k_kernel<<<blocks, threads, 0, tls.stream>>>(
                tls.src, tls.dst, tls.imatrix, nblocks, blocks_per_row);
    } else {
        q2_k_kernel<<<blocks, threads, 0, tls.stream>>>(
                tls.src, tls.dst, tls.imatrix, nblocks, blocks_per_row);
    }
    st = cudaGetLastError();
    if (st == cudaSuccess) st = cudaMemcpyAsync(dst, tls.dst, dst_bytes,
                                                cudaMemcpyDeviceToHost, tls.stream);
    if (st == cudaSuccess) st = cudaStreamSynchronize(tls.stream);
    if (st != cudaSuccess) {
        set_cuda_error(error, error_cap, "CUDA quantization", st);
        return 0;
    }
    return dst_bytes;
}

size_t ds4q_cuda_repack_sm75_native_q4(const void *src,
                                       void *dst,
                                       int64_t nrows,
                                       int64_t ncols,
                                       int device,
                                       char *error,
                                       size_t error_cap) {
    if (error && error_cap) error[0] = '\0';
    if (!src || !dst || nrows <= 0 || ncols <= 0 ||
        nrows % 8 != 0 || ncols % QK_K != 0) {
        if (error && error_cap)
            snprintf(error, error_cap, "invalid CUDA SM75-Q4 repack request");
        return 0;
    }
    if (!ensure_device(device, false, error, error_cap)) return 0;
    const uint32_t blocks_per_row = (uint32_t)(ncols / QK_K);
    const uint64_t records = (uint64_t)(nrows / 8) * blocks_per_row;
    const size_t bytes = (size_t)nrows * blocks_per_row * 144u;
    if (!ensure_allocation((void **)&tls.src, &tls.src_cap, bytes,
                           error, error_cap) ||
        !ensure_allocation((void **)&tls.dst, &tls.dst_cap, bytes,
                           error, error_cap)) {
        return 0;
    }
    cudaError_t st = cudaMemcpyAsync(tls.src, src, bytes,
                                     cudaMemcpyHostToDevice, tls.stream);
    if (st == cudaSuccess) {
        repack_sm75_native_q4_kernel<<<
            (unsigned)records, 256, 0, tls.stream>>>(
                (const uint8_t *)tls.src, tls.dst,
                records, blocks_per_row);
        st = cudaGetLastError();
    }
    if (st == cudaSuccess) {
        st = cudaMemcpyAsync(dst, tls.dst, bytes,
                             cudaMemcpyDeviceToHost, tls.stream);
    }
    if (st == cudaSuccess) st = cudaStreamSynchronize(tls.stream);
    if (st != cudaSuccess) {
        set_cuda_error(error, error_cap, "CUDA SM75-Q4 repack", st);
        return 0;
    }
    return bytes;
}

void ds4q_cuda_thread_shutdown(void) {
    tls_release();
}
