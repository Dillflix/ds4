#include "quants.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t next_u32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static int check_zero_blocks(void) {
    float source[256] = {0};
    float weights[256];
    float decoded[256];
    _Alignas(16) uint8_t encoded[DS4Q_EXPERIMENT_MAX_BLOCK_BYTES];
    for (int i = 0; i < 256; i++) weights[i] = 1.0f + (float)(i % 17) / 17.0f;
    for (int f = 0; f < DS4Q_EXPERIMENT_COUNT; f++) {
        memset(encoded, 0xa5, sizeof(encoded));
        if (!ds4q_experimental_quantize_block(
                (ds4q_experimental_format)f, source, weights, encoded) ||
            !ds4q_experimental_dequantize_block(
                (ds4q_experimental_format)f, encoded, decoded)) {
            fprintf(stderr, "zero encode/decode failed for format %d\n", f);
            return 1;
        }
        for (int i = 0; i < 256; i++) {
            if (decoded[i] != 0.0f) {
                fprintf(stderr, "nonzero zero-block decode: format=%d index=%d value=%g\n",
                        f, i, decoded[i]);
                return 1;
            }
        }
    }
    return 0;
}

static int check_q4_control_parity(void) {
    float source[2 * 256];
    float weights[256];
    uint8_t production[2 * 144];
    _Alignas(16) uint8_t experimental[DS4Q_EXPERIMENT_MAX_BLOCK_BYTES];
    uint32_t state = 0x243f6a88u;
    for (int i = 0; i < 2 * 256; i++) {
        source[i] = ((int32_t)(next_u32(&state) % 20001u) - 10000) / 4096.0f;
    }
    for (int i = 0; i < 256; i++) {
        weights[i] = 0.125f + (float)(next_u32(&state) % 4096u) / 1024.0f;
    }
    const size_t written = ds4q_quantize_chunk(
        DS4Q_TYPE_Q4_K, source, production, 0, 2, 256, weights);
    if (written != sizeof(production)) {
        fprintf(stderr, "unexpected production Q4_K byte count: %zu\n", written);
        return 1;
    }
    for (int row = 0; row < 2; row++) {
        if (!ds4q_experimental_quantize_block(
                DS4Q_EXPERIMENT_Q4_K, source + row * 256, weights,
                experimental) ||
            memcmp(experimental, production + row * 144, 144) != 0) {
            fprintf(stderr, "experimental Q4_K differs from production row %d\n", row);
            return 1;
        }
    }
    return 0;
}

static int check_shipping_control_parity(ds4q_type production_type,
                                         ds4q_experimental_format format,
                                         size_t block_bytes) {
    float source[2 * 256];
    float weights[256];
    uint8_t production[2 * 144];
    _Alignas(16) uint8_t experimental[DS4Q_EXPERIMENT_MAX_BLOCK_BYTES];
    uint32_t state = 0xa4093822u ^ (uint32_t)format;
    for (int i = 0; i < 2 * 256; i++) {
        source[i] = ((int32_t)(next_u32(&state) % 20001u) - 10000) / 4096.0f;
    }
    for (int i = 0; i < 256; i++) {
        weights[i] = 0.125f + (float)(next_u32(&state) % 4096u) / 1024.0f;
    }
    ds4q_quantize_init(production_type);
    const size_t written = ds4q_quantize_chunk(
        production_type, source, production, 0, 2, 256, weights);
    if (written != 2 * block_bytes) {
        fprintf(stderr, "unexpected production byte count: type=%d got=%zu\n",
                production_type, written);
        return 1;
    }
    for (int row = 0; row < 2; row++) {
        if (!ds4q_experimental_quantize_block(
                format, source + row * 256, weights, experimental) ||
            memcmp(experimental, production + row * block_bytes,
                   block_bytes) != 0) {
            fprintf(stderr, "experimental shipping control differs: format=%d row=%d\n",
                    format, row);
            return 1;
        }
    }
    return 0;
}

static int check_iq2_runtime_codebook(void) {
    _Alignas(16) uint8_t encoded[66] = {0};
    float decoded[256];
    const float one = 1.0f;
    uint16_t one_f16;
    ds4q_f32_to_f16_row(&one, &one_f16, 1);
    memcpy(encoded, &one_f16, sizeof(one_f16));
    /* Every zero grid index selects eight symbolic level-1 entries; zero
       sign/scale words select positive signs and the first odd scale.  The
       runtime codebook maps level 1 to 8, which cancels its 1/8 factor. */
    if (!ds4q_experimental_dequantize_block(
            DS4Q_EXPERIMENT_IQ2_XXS, encoded, decoded)) {
        fprintf(stderr, "IQ2 runtime-codebook decode failed\n");
        return 1;
    }
    for (int i = 0; i < 256; i++) {
        if (decoded[i] != 1.0f) {
            fprintf(stderr,
                    "IQ2 runtime-codebook mismatch: index=%d expected=1 got=%g\n",
                    i, decoded[i]);
            return 1;
        }
    }
    return 0;
}

static int check_determinism_and_finite_decode(void) {
    float source[256], weights[256], decoded[256];
    _Alignas(16) uint8_t first[DS4Q_EXPERIMENT_MAX_BLOCK_BYTES];
    _Alignas(16) uint8_t second[DS4Q_EXPERIMENT_MAX_BLOCK_BYTES];
    uint32_t state = 0x13198a2eu;
    for (int sample = 0; sample < 64; sample++) {
        for (int i = 0; i < 256; i++) {
            const float envelope = 0.1f + (float)((i / 32) + 1);
            source[i] = envelope *
                ((int32_t)(next_u32(&state) % 65535u) - 32767) / 16384.0f;
            weights[i] = 0.01f + (float)(next_u32(&state) % 8192u) / 1024.0f;
        }
        for (int f = 0; f < DS4Q_EXPERIMENT_COUNT; f++) {
            const ds4q_experimental_format format = (ds4q_experimental_format)f;
            const size_t bytes = ds4q_experimental_block_bytes(format);
            memset(first, 0x5a, sizeof(first));
            memset(second, 0xa5, sizeof(second));
            if (!ds4q_experimental_quantize_block(
                    format, source, weights, first) ||
                !ds4q_experimental_quantize_block(
                    format, source, weights, second) ||
                memcmp(first, second, bytes) != 0) {
                fprintf(stderr, "nondeterministic encoding: sample=%d format=%d\n",
                        sample, f);
                return 1;
            }
            for (size_t i = bytes; i < sizeof(first); i++) {
                if (first[i] != 0x5a || second[i] != 0xa5) {
                    fprintf(stderr,
                            "encoding exceeded block: sample=%d format=%d byte=%zu\n",
                            sample, f, i);
                    return 1;
                }
            }
            if (!ds4q_experimental_dequantize_block(format, first, decoded)) {
                fprintf(stderr, "decode failed: sample=%d format=%d\n", sample, f);
                return 1;
            }
            for (int i = 0; i < 256; i++) {
                if (!isfinite(decoded[i])) {
                    fprintf(stderr, "non-finite decode: sample=%d format=%d index=%d\n",
                            sample, f, i);
                    return 1;
                }
            }
        }
    }
    return 0;
}

static int popcount8(uint8_t value) {
    int count = 0;
    for (; value; value &= (uint8_t)(value - 1)) count++;
    return count;
}

static int check_adaptive_promotion_counts(void) {
    float source[256], weights[256];
    _Alignas(16) uint8_t encoded[DS4Q_EXPERIMENT_MAX_BLOCK_BYTES];
    uint32_t state = 0x082efa98u;
    for (int i = 0; i < 256; i++) {
        source[i] = ((int32_t)(next_u32(&state) % 20001u) - 10000) / 4096.0f;
        weights[i] = 0.125f + (float)(next_u32(&state) % 4096u) / 1024.0f;
    }
    static const struct {
        ds4q_experimental_format format;
        size_t mask_offset;
        int expected;
    } cases[] = {
        {DS4Q_EXPERIMENT_SM75_Q3Q4_32_25, 104, 2},
        {DS4Q_EXPERIMENT_SM75_Q3Q4_32_50, 104, 4},
        {DS4Q_EXPERIMENT_SM75_Q2Q3_32_50, 72, 4},
        {DS4Q_EXPERIMENT_SM75_Q2Q3_32_75, 72, 6},
    };
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        memset(encoded, 0, sizeof(encoded));
        if (!ds4q_experimental_quantize_block(cases[c].format, source,
                                               weights, encoded) ||
            popcount8(encoded[cases[c].mask_offset]) != cases[c].expected) {
            fprintf(stderr, "adaptive promotion count failed: format=%d\n",
                    cases[c].format);
            return 1;
        }
    }
    return 0;
}

int main(void) {
    static const size_t expected[DS4Q_EXPERIMENT_COUNT] = {
        144, 110, 104, 136, 66, 104, 108, 112, 113, 121,
        140, 142, 168, 89, 97,
    };
    for (int f = 0; f < DS4Q_EXPERIMENT_COUNT; f++) {
        if (!ds4q_experimental_format_name((ds4q_experimental_format)f) ||
            ds4q_experimental_block_bytes((ds4q_experimental_format)f) != expected[f]) {
            fprintf(stderr, "experimental format metadata failed: %d\n", f);
            return 1;
        }
    }
    if (check_zero_blocks() ||
        check_q4_control_parity() ||
        check_shipping_control_parity(DS4Q_TYPE_IQ2_XXS,
                                      DS4Q_EXPERIMENT_IQ2_XXS, 66) ||
        check_iq2_runtime_codebook() ||
        check_adaptive_promotion_counts() ||
        check_determinism_and_finite_decode()) {
        return 1;
    }
    printf("experimental routed-quant tests: OK\n");
    return 0;
}
