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

static __device__ float d_make_qx_quants(int n, int nmax,
                                         const float *x, int8_t *L,
                                         const float *weights) {
    float maxv = 0.0f, amax = 0.0f;
    for (int i = 0; i < n; i++) {
        const float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; maxv = x[i]; }
    }
    if (amax < GROUP_EPS) {
        for (int i = 0; i < n; i++) L[i] = 0;
        return 0.0f;
    }
    float iscale = -(float)nmax / maxv;
    float sumlx = 0.0f, suml2 = 0.0f;
    for (int i = 0; i < n; i++) {
        int l = d_nearest(iscale * x[i]);
        l = d_max_i(-nmax, d_min_i(nmax - 1, l));
        L[i] = (int8_t)(l + nmax);
        const float w = weights ? weights[i] : x[i] * x[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    float scale = suml2 ? sumlx / suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; is++) {
        if (!is) continue;
        iscale = -(nmax + 0.1f * is) / maxv;
        sumlx = 0.0f; suml2 = 0.0f;
        for (int i = 0; i < n; i++) {
            int l = d_nearest(iscale * x[i]);
            l = d_max_i(-nmax, d_min_i(nmax - 1, l));
            const float w = weights ? weights[i] : x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; i++) {
                int l = d_nearest(iscale * x[i]);
                l = d_max_i(-nmax, d_min_i(nmax - 1, l));
                L[i] = (int8_t)(l + nmax);
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

static __device__ void d_pack_signed_scales8(const int8_t in[8],
                                              uint8_t out[6]) {
    for (int i = 0; i < 6; i++) out[i] = 0;
    for (int j = 0; j < 8; j++) {
        const unsigned code = (unsigned)(uint8_t)in[j] & 63u;
        if (j < 4) out[j] |= (uint8_t)(code & 15u);
        else out[j - 4] |= (uint8_t)((code & 15u) << 4);
        out[4 + j / 4] |= (uint8_t)((code >> 4) << (2 * (j % 4)));
    }
}

static __device__ int d_unpack_signed_scale8(const uint8_t in[6], int j) {
    const unsigned low = j < 4 ? in[j] & 15u : in[j - 4] >> 4;
    const unsigned high = (in[4 + j / 4] >> (2 * (j % 4))) & 3u;
    return (int)(low | (high << 4)) - 32;
}

static __device__ void d_pack_bits(const uint8_t *values, int n, int bits,
                                   uint8_t *out) {
    const int bytes = (n * bits + 7) / 8;
    for (int i = 0; i < bytes; i++) out[i] = 0;
    for (int i = 0, bit = 0; i < n; i++, bit += bits) {
        const int byte = bit >> 3;
        const int shift = bit & 7;
        const unsigned value = values[i] & ((1u << bits) - 1u);
        out[byte] |= (uint8_t)(value << shift);
        if (shift + bits > 8)
            out[byte + 1] |= (uint8_t)(value >> (8 - shift));
    }
}

static __device__ unsigned d_unpack_bits(const uint8_t *data, int index,
                                         int bits) {
    const unsigned bit = (unsigned)index * (unsigned)bits;
    const unsigned byte = bit >> 3;
    const unsigned shift = bit & 7u;
    unsigned value = data[byte] >> shift;
    if (shift + (unsigned)bits > 8u)
        value |= (unsigned)data[byte + 1] << (8u - shift);
    return value & ((1u << bits) - 1u);
}

static __device__ void d_write_sm75_q4_32(const float *x, uint8_t *y,
                                           const float *quant_weights) {
    int8_t L[QK_K], Ls[8];
    float scales[8], weights[32], sw[8];
    float sumx2 = 0.0f;
    for (int i = 0; i < QK_K; i++) sumx2 += x[i] * x[i];
    const float sigma2 = 2.0f * sumx2 / QK_K;
    for (int g = 0; g < 8; g++) {
        sw[g] = 0.0f;
        for (int i = 0; i < 32; i++) {
            const float xv = x[32 * g + i];
            weights[i] = quant_weights
                ? quant_weights[32 * g + i] * sqrtf(sigma2 + xv * xv)
                : xv * xv;
            sw[g] += weights[i];
        }
        scales[g] = d_make_qx_quants(32, 8, x + 32 * g,
                                      L + 32 * g, weights);
    }
    const float dblock = d_make_qx_quants(8, 32, scales, Ls, sw);
    const uint16_t hd = d_f16_bits(dblock);
    d_store_u16(y, hd);
    d_pack_signed_scales8(Ls, y + 2);
    const float d = d_f16_value(hd);
    for (int g = 0; g < 8; g++) {
        const float scale = d * d_unpack_signed_scale8(y + 2, g);
        if (!scale) continue;
        for (int i = 0; i < 32; i++) {
            int l = d_nearest(x[32 * g + i] / scale);
            l = d_max_i(-8, d_min_i(7, l));
            L[32 * g + i] = (int8_t)(l + 8);
        }
    }
    for (int i = 0; i < 128; i++) y[8 + i] = 0;
    for (int k = 0; k < QK_K; k++) {
        const uint8_t code = (uint8_t)(L[k] - 8) & 15u;
        y[8 + k / 2] |= (uint8_t)(code << (4 * (k & 1)));
    }
}

static __device__ void d_write_sm75_q3a4(const float *x, uint8_t *y,
                                          const float *quant_weights) {
    uint8_t codes[QK_K], scratch[32], scales_q[8], mins_q[8];
    float scales[8], mins[8], sw[8], weights[32];
    float sumx2 = 0.0f;
    for (int i = 0; i < QK_K; i++) sumx2 += x[i] * x[i];
    const float sigma2 = 2.0f * sumx2 / QK_K;
    for (int g = 0; g < 8; g++) {
        sw[g] = 0.0f;
        for (int i = 0; i < 32; i++) {
            const float xv = x[32 * g + i];
            weights[i] = quant_weights
                ? quant_weights[32 * g + i] * sqrtf(sigma2 + xv * xv)
                : xv * xv;
            sw[g] += weights[i];
        }
        scales[g] = d_make_qkx3_quants(32, 7, x + 32 * g, weights,
            codes + 32 * g, &mins[g], scratch, -0.9f, 0.05f, 36);
    }
    const float dblock = d_make_qp_quants(8, 15, scales, scales_q, sw);
    const float mblock = d_make_qp_quants(8, 15, mins, mins_q, sw);
    const uint16_t hd = d_f16_bits(dblock), hm = d_f16_bits(mblock);
    d_store_u16(y, hd); d_store_u16(y + 2, hm);
    d_pack_bits(scales_q, 8, 4, y + 4);
    d_pack_bits(mins_q, 8, 4, y + 8);
    const float d = d_f16_value(hd), m = d_f16_value(hm);
    for (int g = 0; g < 8; g++) {
        const float scale = d * d_unpack_bits(y + 4, g, 4);
        const float minv = m * d_unpack_bits(y + 8, g, 4);
        for (int i = 0; i < 32; i++) {
            int q = scale ? d_nearest((x[32 * g + i] + minv) / scale) : 0;
            codes[32 * g + i] = (uint8_t)d_max_i(0, d_min_i(7, q));
        }
    }
    d_pack_bits(codes, QK_K, 3, y + 12);
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

static __global__ __launch_bounds__(64) void sm75_q4_32_kernel(
        const float *src, uint8_t *dst, const float *imatrix,
        int64_t nblocks, int64_t blocks_per_row) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    d_write_sm75_q4_32(src + b * QK_K, dst + b * 136,
                       imatrix + (b % blocks_per_row) * QK_K);
}

static __global__ __launch_bounds__(64) void sm75_q3a4_kernel(
        const float *src, uint8_t *dst, const float *imatrix,
        int64_t nblocks, int64_t blocks_per_row) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    d_write_sm75_q3a4(src + b * QK_K, dst + b * 108,
                      imatrix + (b % blocks_per_row) * QK_K);
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

static __global__ __launch_bounds__(256) void repack_sm75_q4_32_kernel(
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
        ((tile * 8u + row) * blocks_per_row + block) * 136u;
    uint8_t *out = dst + record * 1088u;
    if (tid < 8u) {
        const uint8_t *h = src +
            ((tile * 8u + tid) * blocks_per_row + block) * 136u;
        ((uint16_t *)out)[tid] = *(const uint16_t *)h;
        for (uint32_t i = 0; i < 6u; i++)
            out[16u + tid * 6u + i] = h[2u + i];
    }
    const uint8_t *qs = src + src_block + 8u;
    uint32_t packed = 0u;
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) {
        const uint32_t k = group * 32u + lane4 * 8u + i;
        const uint32_t code = (qs[k >> 1u] >> (4u * (k & 1u))) & 15u;
        packed |= code << (4u * i);
    }
    ((uint32_t *)(out + 64u))[group * 32u + lane] = packed;
}

static __global__ __launch_bounds__(256) void repack_sm75_q3a4_kernel(
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
        ((tile * 8u + row) * blocks_per_row + block) * 108u;
    uint8_t *out = dst + record * 864u;
    if (tid < 8u) {
        const uint8_t *h = src +
            ((tile * 8u + tid) * blocks_per_row + block) * 108u;
        ((uint16_t *)out)[tid] = *(const uint16_t *)h;
        ((uint16_t *)(out + 16u))[tid] = *(const uint16_t *)(h + 2u);
        for (uint32_t i = 0; i < 4u; i++) {
            out[32u + tid * 4u + i] = h[4u + i];
            out[64u + tid * 4u + i] = h[8u + i];
        }
    }
    const uint8_t *qs = src + src_block + 12u;
    uint16_t low = 0u;
    uint8_t high = 0u;
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) {
        const uint32_t k = group * 32u + lane4 * 8u + i;
        const uint32_t code = d_unpack_bits(qs, (int)k, 3);
        low |= (uint16_t)(code & 3u) << (2u * i);
        high |= (uint8_t)((code >> 2u) & 1u) << i;
    }
    ((uint16_t *)(out + 96u))[group * 32u + lane] = low;
    out[608u + group * 32u + lane] = high;
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
    return type == DS4Q_TYPE_IQ2_XXS || type == DS4Q_TYPE_Q4_K ||
           type == DS4Q_TYPE_Q2_K || type == DS4Q_TYPE_SM75_Q4_32 ||
           type == DS4Q_TYPE_SM75_Q3A4;
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
                               type == DS4Q_TYPE_Q4_K ? 144 :
                               type == DS4Q_TYPE_Q2_K ? 84 :
                               type == DS4Q_TYPE_SM75_Q4_32 ? 136 : 108;
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
    } else if (type == DS4Q_TYPE_SM75_Q4_32) {
        sm75_q4_32_kernel<<<blocks, threads, 0, tls.stream>>>(
                tls.src, tls.dst, tls.imatrix, nblocks, blocks_per_row);
    } else if (type == DS4Q_TYPE_SM75_Q3A4) {
        sm75_q3a4_kernel<<<blocks, threads, 0, tls.stream>>>(
                tls.src, tls.dst, tls.imatrix, nblocks, blocks_per_row);
    } else {
        q2_k_kernel<<<blocks, threads, 0, tls.stream>>>(
                tls.src, tls.dst, tls.imatrix, nblocks, blocks_per_row);
    }
    st = cudaGetLastError();
    const uint8_t *device_output = tls.dst;
    if (st == cudaSuccess && (type == DS4Q_TYPE_SM75_Q4_32 ||
                             type == DS4Q_TYPE_SM75_Q3A4)) {
        if ((nrows & 7) != 0) {
            if (error && error_cap)
                snprintf(error, error_cap,
                         "SM75 Q32 output requires a multiple of eight rows");
            return 0;
        }
        const uint64_t records = (uint64_t)(nrows / 8) *
                                 (uint64_t)blocks_per_row;
        if (type == DS4Q_TYPE_SM75_Q4_32) {
            repack_sm75_q4_32_kernel<<<
                (unsigned)records, 256, 0, tls.stream>>>(
                    tls.dst, (uint8_t *)tls.src, records,
                    (uint32_t)blocks_per_row);
        } else {
            repack_sm75_q3a4_kernel<<<
                (unsigned)records, 256, 0, tls.stream>>>(
                    tls.dst, (uint8_t *)tls.src, records,
                    (uint32_t)blocks_per_row);
        }
        st = cudaGetLastError();
        device_output = (const uint8_t *)tls.src;
    }
    if (st == cudaSuccess) st = cudaMemcpyAsync(dst, device_output, dst_bytes,
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
