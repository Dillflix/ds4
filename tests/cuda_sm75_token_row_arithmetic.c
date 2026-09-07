#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Bounded single-GPU diagnostic for the arithmetic shape change made by the
 * token-row prototype.  Production dispatch is not modified: every boundary
 * is invoked explicitly once as N=512 and once as two ordered N=256 views. */
#define N_TOK 512u
#define HALF_TOK (N_TOK / 2u)
#define IN_DIM 1024u
#define N_HEAD 64u
#define HEAD_DIM 512u
#define Q_DIM ((uint64_t)N_HEAD * HEAD_DIM)
#define N_ROT 64u
#define N_GROUP 8u
#define GROUP_DIM 4096u
#define RANK 1024u
#define LOW_DIM ((uint64_t)N_GROUP * RANK)
#define OUT_DIM 4096u
#define N_COMP 128u
#define ATTN_WINDOW 128u
#define ATTN_RATIO 4u
#define POS0 0u
#define ORIG_CTX 65536u
#define ROPE_BASE 160000.0f
#define ROPE_SCALE 0.0625f
#define ROPE_EXT 1.0f
#define BETA_FAST 32.0f
#define BETA_SLOW 1.0f
#define EPSILON 1.0e-6f
#define PROJECTION_PROBE_GUARD_BYTES 4096u

static const int b_algorithms[] = {
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    99,
    100, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115,
};

typedef struct {
    uint64_t mismatches;
    uint64_t first;
    double max_abs;
    double rmse;
} diff_metrics;

static uint16_t float_to_half_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16u) & 0x8000u;
    const uint32_t exponent = (bits >> 23u) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu)
        return (uint16_t)(sign | (mantissa ? 0x7e00u : 0x7c00u));
    int32_t half_exponent = (int32_t)exponent - 127 + 15;
    if (half_exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return (uint16_t)sign;
        mantissa |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - half_exponent);
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((UINT32_C(1) << shift) - 1u);
        const uint32_t halfway = UINT32_C(1) << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u)))
            rounded++;
        return (uint16_t)(sign | rounded);
    }
    uint32_t rounded = mantissa >> 13u;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        rounded++;
        if (rounded == 0x400u) {
            rounded = 0u;
            half_exponent++;
            if (half_exponent >= 31)
                return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)half_exponent << 10u) | rounded);
}

static void build_q8_rows(unsigned char *dst, uint64_t rows,
                          uint64_t columns, uint64_t seed) {
    const uint64_t blocks = columns / 32u;
    for (uint64_t row = 0u; row < rows; row++) {
        for (uint64_t block = 0u; block < blocks; block++) {
            unsigned char *packed = dst + (row * blocks + block) * 34u;
            const float scale =
                (float)(1u + ((row * 11u + block * 7u + seed) % 13u)) /
                256.0f;
            const uint16_t scale_bits = float_to_half_bits(scale);
            packed[0] = (unsigned char)(scale_bits & 0xffu);
            packed[1] = (unsigned char)(scale_bits >> 8u);
            for (uint64_t lane = 0u; lane < 32u; lane++) {
                const uint64_t column = block * 32u + lane;
                const int value = (int)((row * 19u + column * 23u +
                    block * 29u + (row >> 3u) * 5u + seed) % 127u) - 63;
                packed[2u + lane] = (unsigned char)(int8_t)value;
            }
        }
    }
}

/* Match the tagged GGUF row-warp32 encoding exactly: for each 32-block
 * group, store 32 scales followed by eight lane-major int8x4 word planes. */
static void pack_q8_rows_warp32_slice(
        unsigned char *dst, const unsigned char *src, uint64_t rows,
        uint64_t source_blocks, uint64_t source_block_start,
        uint64_t blocks) {
    const uint64_t source_row_bytes = source_blocks * 34u;
    const uint64_t output_row_bytes = blocks * 34u;
    for (uint64_t row = 0u; row < rows; row++) {
        const unsigned char *src_row = src + row * source_row_bytes +
            source_block_start * 34u;
        unsigned char *dst_row = dst + row * output_row_bytes;
        for (uint64_t group = 0u; group < blocks / 32u; group++) {
            const unsigned char *src_group = src_row + group * 1088u;
            unsigned char *dst_group = dst_row + group * 1088u;
            for (uint64_t lane = 0u; lane < 32u; lane++) {
                const unsigned char *block = src_group + lane * 34u;
                dst_group[2u * lane] = block[0];
                dst_group[2u * lane + 1u] = block[1];
                for (uint64_t word = 0u; word < 8u; word++) {
                    memcpy(dst_group + 64u + (word * 32u + lane) * 4u,
                           block + 2u + word * 4u, 4u);
                }
            }
        }
    }
}

static void pack_q8_rows_warp32(unsigned char *dst,
                                const unsigned char *src,
                                uint64_t rows, uint64_t columns) {
    const uint64_t blocks = columns / 32u;
    pack_q8_rows_warp32_slice(dst, src, rows, blocks, 0u, blocks);
}

/* Match the tagged attention-output B encoding: the two K halves are
 * contiguous matrix shards, and rows inside each shard use warp32 planes. */
static void pack_q8_rows_b_kshards_warp32(
        unsigned char *dst, const unsigned char *src,
        uint64_t rows, uint64_t columns) {
    const uint64_t blocks = columns / 32u;
    const uint64_t half_blocks = blocks / 2u;
    const uint64_t shard_bytes = rows * half_blocks * 34u;
    pack_q8_rows_warp32_slice(
        dst, src, rows, blocks, 0u, half_blocks);
    pack_q8_rows_warp32_slice(
        dst + shard_bytes, src, rows, blocks, half_blocks, half_blocks);
}

static diff_metrics compare_f32(const float *reference,
                                const float *candidate, uint64_t count) {
    diff_metrics result = {0u, UINT64_MAX, 0.0, 0.0};
    double squared = 0.0;
    for (uint64_t i = 0u; i < count; i++) {
        uint32_t a, b;
        memcpy(&a, reference + i, sizeof(a));
        memcpy(&b, candidate + i, sizeof(b));
        if (a != b) {
            if (result.first == UINT64_MAX) result.first = i;
            result.mismatches++;
        }
        const double delta = (double)reference[i] - (double)candidate[i];
        const double abs_delta = fabs(delta);
        if (abs_delta > result.max_abs) result.max_abs = abs_delta;
        squared += delta * delta;
    }
    result.rmse = count ? sqrt(squared / (double)count) : 0.0;
    return result;
}

static diff_metrics compare_u16(const uint16_t *reference,
                                const uint16_t *candidate, uint64_t count) {
    diff_metrics result = {0u, UINT64_MAX, 0.0, 0.0};
    for (uint64_t i = 0u; i < count; i++) {
        if (reference[i] != candidate[i]) {
            if (result.first == UINT64_MAX) result.first = i;
            result.mismatches++;
        }
    }
    return result;
}

static void report_diff(const char *boundary, const char *type,
                        uint64_t count, diff_metrics diff) {
    printf("boundary=%s,type=%s,values=%llu,status=%s,bit_mismatches=%llu,"
           "first_index=%lld,max_abs=%.9g,rmse=%.9g\n",
           boundary, type, (unsigned long long)count,
           diff.mismatches ? "different" : "exact",
           (unsigned long long)diff.mismatches,
           diff.first == UINT64_MAX ? -1ll : (long long)diff.first,
           diff.max_abs, diff.rmse);
}

static float rope_attention_factor(void) {
    return 1.0f / (1.0f + 0.1f * logf(1.0f / ROPE_SCALE));
}

static int launch_q_b(ds4_gpu_tensor *out, ds4_gpu_tensor *half,
                      const unsigned char *model, uint64_t model_bytes,
                      uint64_t q_b_offset, const ds4_gpu_tensor *input,
                      uint32_t n_tokens, uint32_t pos0) {
    return ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        out, half, model, model_bytes, q_b_offset, IN_DIM, Q_DIM, input,
        n_tokens, N_HEAD, HEAD_DIM, N_ROT, pos0, ORIG_CTX, false,
        ROPE_BASE, ROPE_SCALE, ROPE_EXT, rope_attention_factor(),
        BETA_FAST, BETA_SLOW, EPSILON);
}

static int launch_output(ds4_gpu_tensor *out, ds4_gpu_tensor *low,
                         const unsigned char *model, uint64_t model_bytes,
                         uint64_t out_a_offset, uint64_t out_b_offset,
                         const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    return ds4_gpu_attention_output_q8_batch_tensor(
        out, low, NULL, NULL, model, model_bytes, out_a_offset, out_b_offset,
        GROUP_DIM, RANK, N_GROUP, OUT_DIM, heads, n_tokens);
}

static int compare_double(const void *a, const void *b) {
    const double lhs = *(const double *)a;
    const double rhs = *(const double *)b;
    return (lhs > rhs) - (lhs < rhs);
}

static double monotonic_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return -1.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static uint32_t positive_env_u32(const char *name, uint32_t fallback,
                                 uint32_t maximum) {
    const char *text = getenv(name);
    if (!text || !text[0]) return fallback;
    char *end = NULL;
    const unsigned long parsed = strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0ul || parsed > maximum) {
        fprintf(stderr, "error: %s must be an integer in [1,%u]\n",
                name, maximum);
        return 0u;
    }
    return (uint32_t)parsed;
}

static void select_b_algorithm(int algorithm) {
    if (algorithm < 0) {
        (void)unsetenv("DS4_CUDA_ATTN_OUTPUT_B_F16_GEMM_ALGO_DIAGNOSTIC");
        return;
    }
    char text[32];
    snprintf(text, sizeof(text), "%d", algorithm);
    (void)setenv("DS4_CUDA_ATTN_OUTPUT_B_F16_GEMM_ALGO_DIAGNOSTIC", text, 1);
}

static int time_output_b(double *median_ms, ds4_gpu_tensor *out,
                         const unsigned char *model, uint64_t model_bytes,
                         uint64_t out_b_offset, const ds4_gpu_tensor *low,
                         uint32_t n_tokens, int algorithm, uint32_t rounds,
                         uint32_t repeats, uint32_t warmups) {
    double *samples = (double *)malloc((size_t)rounds * sizeof(*samples));
    if (!samples) return 0;
    select_b_algorithm(algorithm);
    for (uint32_t i = 0u; i < warmups; i++) {
        if (!ds4_gpu_attention_output_q8_batch_b_tensor(
                out, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
                low, n_tokens)) {
            free(samples);
            return 0;
        }
    }
    if (!ds4_gpu_synchronize()) {
        free(samples);
        return 0;
    }
    for (uint32_t round = 0u; round < rounds; round++) {
        const double begin = monotonic_seconds();
        if (begin < 0.0) {
            free(samples);
            return 0;
        }
        for (uint32_t repeat = 0u; repeat < repeats; repeat++) {
            if (!ds4_gpu_attention_output_q8_batch_b_tensor(
                    out, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
                    low, n_tokens)) {
                free(samples);
                return 0;
            }
        }
        if (!ds4_gpu_synchronize()) {
            free(samples);
            return 0;
        }
        const double end = monotonic_seconds();
        if (end < begin) {
            free(samples);
            return 0;
        }
        samples[round] = (end - begin) * 1000.0 / (double)repeats;
    }
    qsort(samples, rounds, sizeof(*samples), compare_double);
    *median_ms = (rounds & 1u)
        ? samples[rounds / 2u]
        : 0.5 * (samples[rounds / 2u - 1u] + samples[rounds / 2u]);
    free(samples);
    return 1;
}

static int run_output_b_single_launch(
        const unsigned char *model, uint64_t model_bytes,
        uint64_t out_b_offset, int native_stream) {
    const uint64_t low_count = (uint64_t)HALF_TOK * LOW_DIM;
    const uint64_t low_bytes = low_count * sizeof(float);
    const uint64_t out_count = (uint64_t)HALF_TOK * OUT_DIM;
    const uint64_t out_bytes = out_count * sizeof(float);
    const uint64_t guarded_bytes =
        PROJECTION_PROBE_GUARD_BYTES + out_bytes +
        PROJECTION_PROBE_GUARD_BYTES;
    float *low_host = NULL;
    unsigned char *guarded_host = NULL;
    ds4_gpu_tensor *low = NULL;
    ds4_gpu_tensor *guarded = NULL;
    ds4_gpu_tensor *out = NULL;
    int status = 0;

    low_host = (float *)malloc((size_t)low_bytes);
    guarded_host = (unsigned char *)malloc((size_t)guarded_bytes);
    if (!low_host || !guarded_host) {
        fprintf(stderr, "error: output-B probe host allocation failed\n");
        goto cleanup;
    }
    for (uint64_t i = 0u; i < low_count; i++) {
        const int value =
            (int)((i * 43u + (i >> 6u) * 17u + 61u) % 509u) - 254;
        low_host[i] = (float)value / 4096.0f;
    }
    memset(guarded_host, 0xa5, (size_t)guarded_bytes);

    low = ds4_gpu_tensor_alloc(low_bytes);
    guarded = ds4_gpu_tensor_alloc(guarded_bytes);
    out = guarded ? ds4_gpu_tensor_view(
        guarded, PROJECTION_PROBE_GUARD_BYTES, out_bytes) : NULL;
    if (!low || !guarded || !out ||
        !ds4_gpu_tensor_write(low, 0u, low_host, low_bytes) ||
        !ds4_gpu_tensor_write(guarded, 0u, guarded_host, guarded_bytes) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: output-B probe setup failed\n");
        goto cleanup;
    }

    printf("diagnostic_scope=output-b-%s-single-launch\n"
           "n_tokens=%u\ninput_dim=%llu\noutput_dim=%u\n"
           "algorithm=CUBLAS_GEMM_ALGO3_TENSOR_OP\n"
           "projection_launches=1\npeer_access=none\n"
           "native_stream=%s\n",
           native_stream ? "native" : "canonical",
           HALF_TOK, (unsigned long long)LOW_DIM, OUT_DIM,
           native_stream ? "on" : "off");
    fflush(stdout);

    select_b_algorithm(103);
    if (!ds4_gpu_attention_output_q8_batch_b_tensor(
            out, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
            low, HALF_TOK) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(guarded, 0u, guarded_host, guarded_bytes)) {
        fprintf(stderr,
                "error: %s output-B single-launch probe failed\n",
                native_stream ? "native-stream" : "canonical");
        goto cleanup;
    }

    uint64_t prefix_mismatches = 0u;
    uint64_t suffix_mismatches = 0u;
    for (uint64_t i = 0u; i < PROJECTION_PROBE_GUARD_BYTES; i++) {
        prefix_mismatches += guarded_host[i] != 0xa5u;
        suffix_mismatches +=
            guarded_host[PROJECTION_PROBE_GUARD_BYTES + out_bytes + i] !=
            0xa5u;
    }
    const float *output = (const float *)(
        guarded_host + PROJECTION_PROBE_GUARD_BYTES);
    uint64_t finite = 0u;
    uint64_t nonzero = 0u;
    uint64_t output_hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0u; i < out_count; i++) {
        finite += isfinite(output[i]) != 0;
        nonzero += output[i] != 0.0f;
    }
    for (uint64_t i = 0u; i < out_bytes; i++) {
        output_hash ^= guarded_host[PROJECTION_PROBE_GUARD_BYTES + i];
        output_hash *= UINT64_C(1099511628211);
    }
    printf("canary_prefix_mismatches=%llu\n"
           "canary_suffix_mismatches=%llu\n"
           "output_finite=%llu\noutput_nonzero=%llu\n"
           "output_fnv1a64=%016llx\n",
           (unsigned long long)prefix_mismatches,
           (unsigned long long)suffix_mismatches,
           (unsigned long long)finite,
           (unsigned long long)nonzero,
           (unsigned long long)output_hash);
    if (prefix_mismatches || suffix_mismatches || finite != out_count ||
        nonzero != out_count) {
        fprintf(stderr, "error: output-B probe validation failed\n");
        goto cleanup;
    }
    printf("diagnostic_conclusion=%s-output-b-single-launch-clean\n"
           "harness_status=ok\n",
           native_stream ? "native-stream" : "canonical");
    status = 1;

cleanup:
    select_b_algorithm(-1);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(guarded);
    ds4_gpu_tensor_free(low);
    free(guarded_host);
    free(low_host);
    return status;
}

static int output_b_replay_step(
        const char *stage, int algorithm, ds4_gpu_tensor *out,
        const ds4_gpu_tensor *low, uint32_t n_tokens,
        const unsigned char *model, uint64_t model_bytes,
        uint64_t out_b_offset, ds4_gpu_tensor *guarded,
        unsigned char *guarded_host, uint64_t guarded_bytes,
        uint64_t output_offset, uint64_t output_bytes,
        const unsigned char *expected_output) {
    memset(guarded_host, 0xa5, (size_t)guarded_bytes);
    if (!ds4_gpu_tensor_write(
            guarded, 0u, guarded_host, guarded_bytes) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr,
                "error: canonical output-B replay reset failed at %s\n",
                stage);
        return 0;
    }
    select_b_algorithm(algorithm);
    printf("replay_step=%s,event=submit,algorithm=%s,algorithm_id=%d,rows=%u\n",
           stage, algorithm < 0 ? "default" : "algo3-tensor-op",
           algorithm, n_tokens);
    fflush(stdout);
    if (!ds4_gpu_attention_output_q8_batch_b_tensor(
            out, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
            low, n_tokens)) {
        fprintf(stderr,
                "error: canonical output-B replay submit failed at %s\n",
                stage);
        return 0;
    }
    if (!ds4_gpu_synchronize()) {
        fprintf(stderr,
                "error: canonical output-B replay synchronize failed at %s\n",
                stage);
        return 0;
    }
    if (!ds4_gpu_tensor_read(
            guarded, 0u, guarded_host, guarded_bytes)) {
        fprintf(stderr,
                "error: canonical output-B replay readback failed at %s\n",
                stage);
        return 0;
    }

    uint64_t prefix_mismatches = 0u;
    uint64_t suffix_mismatches = 0u;
    uint64_t untouched_payload_mismatches = 0u;
    uint64_t selected_poison_words = 0u;
    uint64_t expected_bit_mismatches = 0u;
    for (uint64_t i = 0u; i < PROJECTION_PROBE_GUARD_BYTES; i++) {
        prefix_mismatches += guarded_host[i] != 0xa5u;
        suffix_mismatches += guarded_host[
            PROJECTION_PROBE_GUARD_BYTES +
            (uint64_t)N_TOK * OUT_DIM * sizeof(float) + i] != 0xa5u;
    }
    const unsigned char *payload =
        guarded_host + PROJECTION_PROBE_GUARD_BYTES;
    for (uint64_t i = 0u; i < output_offset; i++) {
        untouched_payload_mismatches += payload[i] != 0xa5u;
    }
    for (uint64_t i = output_offset + output_bytes;
         i < (uint64_t)N_TOK * OUT_DIM * sizeof(float); i++) {
        untouched_payload_mismatches += payload[i] != 0xa5u;
    }
    const float *output = (const float *)(
        guarded_host + PROJECTION_PROBE_GUARD_BYTES + output_offset);
    const uint64_t output_count = output_bytes / sizeof(float);
    uint64_t finite = 0u;
    uint64_t nonzero = 0u;
    uint64_t output_hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0u; i < output_count; i++) {
        finite += isfinite(output[i]) != 0;
        nonzero += output[i] != 0.0f;
    }
    const unsigned char *output_raw = (const unsigned char *)output;
    for (uint64_t i = 0u; i < output_count; i++) {
        uint32_t bits = 0u;
        memcpy(&bits, output_raw + i * sizeof(bits), sizeof(bits));
        selected_poison_words += bits == UINT32_C(0xa5a5a5a5);
    }
    for (uint64_t i = 0u; i < output_bytes; i++) {
        output_hash ^= output_raw[i];
        output_hash *= UINT64_C(1099511628211);
    }
    if (expected_output) {
        const float *expected = (const float *)expected_output;
        for (uint64_t i = 0u; i < output_count; i++) {
            uint32_t actual_bits = 0u;
            uint32_t expected_bits = 0u;
            memcpy(&actual_bits, output + i, sizeof(actual_bits));
            memcpy(&expected_bits, expected + i, sizeof(expected_bits));
            expected_bit_mismatches += actual_bits != expected_bits;
        }
    }
    const int expected_match_required = expected_output && algorithm == 103;
    const int valid = prefix_mismatches == 0u &&
        suffix_mismatches == 0u && untouched_payload_mismatches == 0u &&
        selected_poison_words == 0u &&
        (!expected_match_required || expected_bit_mismatches == 0u) &&
        finite == output_count && nonzero != 0u;
    printf("replay_step=%s,event=complete,status=%s,"
           "canary_prefix_mismatches=%llu,"
           "canary_suffix_mismatches=%llu,"
           "untouched_payload_mismatches=%llu,selected_poison_words=%llu,"
           "expected_bit_mismatches=%llu,expected_match_required=%d,"
           "finite=%llu,nonzero=%llu,"
           "fnv1a64=%016llx\n",
           stage, valid ? "ok" : "failed",
           (unsigned long long)prefix_mismatches,
           (unsigned long long)suffix_mismatches,
           (unsigned long long)untouched_payload_mismatches,
           (unsigned long long)selected_poison_words,
           (unsigned long long)expected_bit_mismatches,
           expected_match_required,
           (unsigned long long)finite,
           (unsigned long long)nonzero,
           (unsigned long long)output_hash);
    fflush(stdout);
    if (!valid) {
        fprintf(stderr,
                "error: canonical output-B replay validation failed at %s\n",
                stage);
        return 0;
    }
    return 1;
}

static int run_output_b_canonical_replay(
        const unsigned char *model, uint64_t model_bytes,
        uint64_t out_b_offset) {
    const uint64_t low_count = (uint64_t)N_TOK * LOW_DIM;
    const uint64_t low_bytes = low_count * sizeof(float);
    const uint64_t low_half_bytes = low_bytes / 2u;
    const uint64_t out_bytes = (uint64_t)N_TOK * OUT_DIM * sizeof(float);
    const uint64_t out_half_bytes = out_bytes / 2u;
    const uint64_t guarded_bytes =
        PROJECTION_PROBE_GUARD_BYTES + out_bytes +
        PROJECTION_PROBE_GUARD_BYTES;
    float *low_host = NULL;
    unsigned char *full_host = NULL;
    unsigned char *split_host = NULL;
    ds4_gpu_tensor *low = NULL;
    ds4_gpu_tensor *low0 = NULL;
    ds4_gpu_tensor *low1 = NULL;
    ds4_gpu_tensor *guarded_full = NULL;
    ds4_gpu_tensor *guarded_split = NULL;
    ds4_gpu_tensor *out_full = NULL;
    ds4_gpu_tensor *out_split = NULL;
    ds4_gpu_tensor *out0 = NULL;
    ds4_gpu_tensor *out1 = NULL;
    int status = 0;

    low_host = (float *)malloc((size_t)low_bytes);
    full_host = (unsigned char *)malloc((size_t)guarded_bytes);
    split_host = (unsigned char *)malloc((size_t)guarded_bytes);
    if (!low_host || !full_host || !split_host) {
        fprintf(stderr,
                "error: canonical output-B replay host allocation failed\n");
        goto cleanup;
    }
    for (uint64_t i = 0u; i < low_count; i++) {
        const int value =
            (int)((i * 43u + (i >> 6u) * 17u + 61u) % 509u) - 254;
        low_host[i] = (float)value / 4096.0f;
    }
    memset(full_host, 0xa5, (size_t)guarded_bytes);
    memset(split_host, 0xa5, (size_t)guarded_bytes);

    low = ds4_gpu_tensor_alloc(low_bytes);
    low0 = low ? ds4_gpu_tensor_view(low, 0u, low_half_bytes) : NULL;
    low1 = low ? ds4_gpu_tensor_view(
        low, low_half_bytes, low_half_bytes) : NULL;
    guarded_full = ds4_gpu_tensor_alloc(guarded_bytes);
    guarded_split = ds4_gpu_tensor_alloc(guarded_bytes);
    out_full = guarded_full ? ds4_gpu_tensor_view(
        guarded_full, PROJECTION_PROBE_GUARD_BYTES, out_bytes) : NULL;
    out_split = guarded_split ? ds4_gpu_tensor_view(
        guarded_split, PROJECTION_PROBE_GUARD_BYTES, out_bytes) : NULL;
    out0 = out_split ? ds4_gpu_tensor_view(
        out_split, 0u, out_half_bytes) : NULL;
    out1 = out_split ? ds4_gpu_tensor_view(
        out_split, out_half_bytes, out_half_bytes) : NULL;
    if (!low || !low0 || !low1 || !guarded_full || !guarded_split ||
        !out_full || !out_split || !out0 || !out1 ||
        !ds4_gpu_tensor_write(low, 0u, low_host, low_bytes) ||
        !ds4_gpu_tensor_write(
            guarded_full, 0u, full_host, guarded_bytes) ||
        !ds4_gpu_tensor_write(
            guarded_split, 0u, split_host, guarded_bytes) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: canonical output-B replay setup failed\n");
        goto cleanup;
    }

    printf("diagnostic_scope=output-b-canonical-replay\n"
           "fidelity=bounded-synchronized-transition-probe\n"
           "source_failure_archive=sm75-token-row-arithmetic-20260906T213618Z\n"
           "n_tokens_full=%u\nn_tokens_half=%u\n"
           "input_dim=%llu\noutput_dim=%u\n"
           "resident_f16_cache_bytes=%llu\n"
           "peer_access=none\nnative_stream=off\n"
           "selected_algorithm=CUBLAS_GEMM_ALGO3_TENSOR_OP\n"
           "exhaustive_algorithm_sweep=off\n"
           "original_device_working_set_reproduced=0\n"
           "original_cumulative_launch_history_reproduced=0\n"
           "original_unfenced_submission_suffix_reproduced=0\n"
           "synchronization=after-every-projection\n",
           N_TOK, HALF_TOK, (unsigned long long)LOW_DIM, OUT_DIM,
           (unsigned long long)(2u * (Q_DIM * IN_DIM +
               LOW_DIM * GROUP_DIM + OUT_DIM * LOW_DIM)));
    fflush(stdout);

#define REPLAY_STEP(label, algorithm, output, input, rows, guarded, host, offset, bytes, expected) \
    do { \
        if (!output_b_replay_step( \
                label, algorithm, output, input, rows, model, model_bytes, \
                out_b_offset, guarded, host, guarded_bytes, offset, bytes, \
                expected)) { \
            goto cleanup; \
        } \
    } while (0)

    REPLAY_STEP("default-prefix-full512", -1, out_full, low, N_TOK,
                guarded_full, full_host, 0u, out_bytes, NULL);
    REPLAY_STEP("default-prefix-row0-256", -1, out0, low0, HALF_TOK,
                guarded_split, split_host, 0u, out_half_bytes,
                full_host + PROJECTION_PROBE_GUARD_BYTES);
    REPLAY_STEP("default-prefix-row1-256", -1, out1, low1, HALF_TOK,
                guarded_split, split_host, out_half_bytes, out_half_bytes,
                full_host + PROJECTION_PROBE_GUARD_BYTES + out_half_bytes);
    REPLAY_STEP("algo3-full512", 103, out_full, low, N_TOK,
                guarded_full, full_host, 0u, out_bytes, NULL);
    REPLAY_STEP("algo3-row0-256", 103, out0, low0, HALF_TOK,
                guarded_split, split_host, 0u, out_half_bytes,
                full_host + PROJECTION_PROBE_GUARD_BYTES);
    REPLAY_STEP("algo3-row1-256", 103, out1, low1, HALF_TOK,
                guarded_split, split_host, out_half_bytes, out_half_bytes,
                full_host + PROJECTION_PROBE_GUARD_BYTES + out_half_bytes);
    REPLAY_STEP("default-suffix-full512", -1, out_full, low, N_TOK,
                guarded_full, full_host, 0u, out_bytes, NULL);
    REPLAY_STEP("default-suffix-row0-256", -1, out0, low0, HALF_TOK,
                guarded_split, split_host, 0u, out_half_bytes,
                full_host + PROJECTION_PROBE_GUARD_BYTES);
    REPLAY_STEP("default-suffix-row1-256", -1, out1, low1, HALF_TOK,
                guarded_split, split_host, out_half_bytes, out_half_bytes,
                full_host + PROJECTION_PROBE_GUARD_BYTES + out_half_bytes);
#undef REPLAY_STEP

    printf("diagnostic_conclusion=canonical-output-b-replay-clean\n"
           "harness_status=ok\n");
    status = 1;

cleanup:
    select_b_algorithm(-1);
    ds4_gpu_tensor_free(out1);
    ds4_gpu_tensor_free(out0);
    ds4_gpu_tensor_free(out_split);
    ds4_gpu_tensor_free(out_full);
    ds4_gpu_tensor_free(guarded_split);
    ds4_gpu_tensor_free(guarded_full);
    ds4_gpu_tensor_free(low1);
    ds4_gpu_tensor_free(low0);
    ds4_gpu_tensor_free(low);
    free(split_host);
    free(full_host);
    free(low_host);
    return status;
}

static int run_output_a_native_single_launch(
        const unsigned char *model, uint64_t model_bytes,
        uint64_t out_a_offset, uint64_t out_b_offset) {
    const uint64_t heads_count =
        (uint64_t)HALF_TOK * N_GROUP * GROUP_DIM;
    const uint64_t heads_bytes = heads_count * sizeof(float);
    const uint64_t low_count = (uint64_t)HALF_TOK * LOW_DIM;
    const uint64_t low_bytes = low_count * sizeof(float);
    const uint64_t out_bytes = (uint64_t)HALF_TOK * OUT_DIM * sizeof(float);
    const uint64_t guarded_bytes =
        PROJECTION_PROBE_GUARD_BYTES + low_bytes +
        PROJECTION_PROBE_GUARD_BYTES;
    float *heads_host = NULL;
    unsigned char *guarded_host = NULL;
    ds4_gpu_tensor *heads = NULL;
    ds4_gpu_tensor *guarded = NULL;
    ds4_gpu_tensor *low = NULL;
    ds4_gpu_tensor *out = NULL;
    int status = 0;

    heads_host = (float *)malloc((size_t)heads_bytes);
    guarded_host = (unsigned char *)malloc((size_t)guarded_bytes);
    if (!heads_host || !guarded_host) {
        fprintf(stderr, "error: output-A probe host allocation failed\n");
        goto cleanup;
    }
    for (uint64_t i = 0u; i < heads_count; i++) {
        const int value =
            (int)((i * 47u + (i >> 7u) * 19u + 73u) % 521u) - 260;
        heads_host[i] = (float)value / 4096.0f;
    }
    memset(guarded_host, 0xa5, (size_t)guarded_bytes);

    heads = ds4_gpu_tensor_alloc(heads_bytes);
    guarded = ds4_gpu_tensor_alloc(guarded_bytes);
    low = guarded ? ds4_gpu_tensor_view(
        guarded, PROJECTION_PROBE_GUARD_BYTES, low_bytes) : NULL;
    out = ds4_gpu_tensor_alloc(out_bytes);
    if (!heads || !guarded || !low || !out ||
        !ds4_gpu_tensor_write(heads, 0u, heads_host, heads_bytes) ||
        !ds4_gpu_tensor_write(guarded, 0u, guarded_host, guarded_bytes) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: output-A probe setup failed\n");
        goto cleanup;
    }

    printf("diagnostic_scope=output-a-native-single-launch\n"
           "n_tokens=%u\ngroups=%u\ngroup_dim=%u\nrank=%u\n"
           "low_dim=%llu\nprojection_launches=1\npeer_access=none\n"
           "native_stream=on\noutput_b=not-entered\n",
           HALF_TOK, N_GROUP, GROUP_DIM, RANK,
           (unsigned long long)LOW_DIM);
    fflush(stdout);

    if (!ds4_gpu_attention_output_q8_batch_row_owned_sm75_tensor(
            out, low, NULL, NULL, model, model_bytes,
            out_a_offset, out_b_offset, GROUP_DIM, RANK, N_GROUP,
            OUT_DIM, heads, HALF_TOK) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(guarded, 0u, guarded_host, guarded_bytes)) {
        fprintf(stderr, "error: native-stream output-A single-launch probe failed\n");
        goto cleanup;
    }

    uint64_t prefix_mismatches = 0u;
    uint64_t suffix_mismatches = 0u;
    for (uint64_t i = 0u; i < PROJECTION_PROBE_GUARD_BYTES; i++) {
        prefix_mismatches += guarded_host[i] != 0xa5u;
        suffix_mismatches +=
            guarded_host[PROJECTION_PROBE_GUARD_BYTES + low_bytes + i] !=
            0xa5u;
    }
    const float *low_output = (const float *)(
        guarded_host + PROJECTION_PROBE_GUARD_BYTES);
    uint64_t finite = 0u;
    uint64_t nonzero = 0u;
    uint64_t low_hash = UINT64_C(1469598103934665603);
    for (uint64_t i = 0u; i < low_count; i++) {
        finite += isfinite(low_output[i]) != 0;
        nonzero += low_output[i] != 0.0f;
    }
    for (uint64_t i = 0u; i < low_bytes; i++) {
        low_hash ^= guarded_host[PROJECTION_PROBE_GUARD_BYTES + i];
        low_hash *= UINT64_C(1099511628211);
    }
    printf("canary_prefix_mismatches=%llu\n"
           "canary_suffix_mismatches=%llu\n"
           "low_finite=%llu\nlow_nonzero=%llu\n"
           "low_fnv1a64=%016llx\n",
           (unsigned long long)prefix_mismatches,
           (unsigned long long)suffix_mismatches,
           (unsigned long long)finite,
           (unsigned long long)nonzero,
           (unsigned long long)low_hash);
    if (prefix_mismatches || suffix_mismatches || finite != low_count ||
        nonzero == 0u) {
        fprintf(stderr, "error: output-A probe validation failed\n");
        goto cleanup;
    }
    printf("diagnostic_conclusion=native-stream-output-a-single-launch-clean\n"
           "harness_status=ok\n");
    status = 1;

cleanup:
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(low);
    ds4_gpu_tensor_free(guarded);
    ds4_gpu_tensor_free(heads);
    free(guarded_host);
    free(heads_host);
    return status;
}

static int run_output_ab_native_calls(
        const unsigned char *model, uint64_t model_bytes,
        uint64_t q_b_offset, uint64_t out_a_offset, uint64_t out_b_offset,
        uint32_t stress_calls, int q_b_chain) {
    const uint64_t heads_count =
        (uint64_t)HALF_TOK * N_GROUP * GROUP_DIM;
    const uint64_t heads_bytes = heads_count * sizeof(float);
    const uint64_t low_count = (uint64_t)HALF_TOK * LOW_DIM;
    const uint64_t low_bytes = low_count * sizeof(float);
    const uint64_t out_count = (uint64_t)HALF_TOK * OUT_DIM;
    const uint64_t out_bytes = out_count * sizeof(float);
    const uint64_t guarded_low_bytes =
        PROJECTION_PROBE_GUARD_BYTES + low_bytes +
        PROJECTION_PROBE_GUARD_BYTES;
    const uint64_t guarded_out_bytes =
        PROJECTION_PROBE_GUARD_BYTES + out_bytes +
        PROJECTION_PROBE_GUARD_BYTES;
    float *heads_host = NULL;
    float *input_host = NULL;
    unsigned char *guarded_low_host = NULL;
    unsigned char *guarded_out_host = NULL;
    unsigned char *reference_low_host = NULL;
    unsigned char *reference_out_host = NULL;
    ds4_gpu_tensor *heads = NULL;
    ds4_gpu_tensor *input = NULL;
    ds4_gpu_tensor *guarded_low = NULL;
    ds4_gpu_tensor *guarded_out = NULL;
    ds4_gpu_tensor *low = NULL;
    ds4_gpu_tensor *out = NULL;
    int status = 0;

    heads_host = (float *)malloc((size_t)heads_bytes);
    if (q_b_chain) {
        input_host = (float *)malloc(
            (size_t)((uint64_t)HALF_TOK * IN_DIM * sizeof(float)));
    }
    guarded_low_host = (unsigned char *)malloc((size_t)guarded_low_bytes);
    guarded_out_host = (unsigned char *)malloc((size_t)guarded_out_bytes);
    if (stress_calls > 1u) {
        reference_low_host = (unsigned char *)malloc((size_t)low_bytes);
        reference_out_host = (unsigned char *)malloc((size_t)out_bytes);
    }
    if (!heads_host || (q_b_chain && !input_host) ||
        !guarded_low_host || !guarded_out_host ||
        (stress_calls > 1u &&
         (!reference_low_host || !reference_out_host))) {
        fprintf(stderr, "error: output A-to-B probe host allocation failed\n");
        goto cleanup;
    }
    for (uint64_t i = 0u; i < heads_count; i++) {
        const int value =
            (int)((i * 47u + (i >> 7u) * 19u + 73u) % 521u) - 260;
        heads_host[i] = (float)value / 4096.0f;
    }
    if (q_b_chain) {
        for (uint64_t i = 0u; i < (uint64_t)HALF_TOK * IN_DIM; i++) {
            const int value = (int)((i * 29u + (i >> 5u) * 17u +
                (i / IN_DIM) * 7u + 23u) % 257u) - 128;
            input_host[i] = (float)value / 128.0f;
        }
    }
    memset(guarded_low_host, 0xa5, (size_t)guarded_low_bytes);
    memset(guarded_out_host, 0x5a, (size_t)guarded_out_bytes);

    heads = ds4_gpu_tensor_alloc(heads_bytes);
    input = q_b_chain
        ? ds4_gpu_tensor_alloc(
              (uint64_t)HALF_TOK * IN_DIM * sizeof(float))
        : NULL;
    guarded_low = ds4_gpu_tensor_alloc(guarded_low_bytes);
    guarded_out = ds4_gpu_tensor_alloc(guarded_out_bytes);
    low = guarded_low ? ds4_gpu_tensor_view(
        guarded_low, PROJECTION_PROBE_GUARD_BYTES, low_bytes) : NULL;
    out = guarded_out ? ds4_gpu_tensor_view(
        guarded_out, PROJECTION_PROBE_GUARD_BYTES, out_bytes) : NULL;
    if (!heads || (q_b_chain && !input) || !guarded_low || !guarded_out ||
        !low || !out ||
        (!q_b_chain &&
         !ds4_gpu_tensor_write(heads, 0u, heads_host, heads_bytes)) ||
        (q_b_chain &&
         !ds4_gpu_tensor_write(
             input, 0u, input_host,
             (uint64_t)HALF_TOK * IN_DIM * sizeof(float))) ||
        !ds4_gpu_tensor_write(
            guarded_low, 0u, guarded_low_host, guarded_low_bytes) ||
        !ds4_gpu_tensor_write(
            guarded_out, 0u, guarded_out_host, guarded_out_bytes) ||
        !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: output A-to-B probe setup failed\n");
        goto cleanup;
    }

    const int repeated_chain = q_b_chain && stress_calls > 1u;
    const uint32_t total_calls =
        stress_calls + (stress_calls > 1u ? 1u : 0u);
    printf("diagnostic_scope=%s\n"
           "n_tokens=%u\ngroups=%u\ngroup_dim=%u\nrank=%u\n"
           "low_dim=%llu\noutput_dim=%u\nreference_calls=%u\n"
           "stress_calls=%u\nattention_output_calls=%u\n"
           "output_a_launches=%u\noutput_b_launches=%u\npeer_access=none\n"
           "native_stream=on\nproduction_order=unfenced-a-to-b\n"
           "handoff_sync_before_b=0\n",
           repeated_chain ? "projection-chain-native-repeat" :
           q_b_chain ? "projection-chain-native" :
           stress_calls > 1u ? "output-ab-native-repeat" :
                               "output-ab-native-single-call",
           HALF_TOK, N_GROUP, GROUP_DIM, RANK,
           (unsigned long long)LOW_DIM, OUT_DIM,
           stress_calls > 1u ? 1u : 0u, stress_calls,
           stress_calls + (stress_calls > 1u ? 1u : 0u),
           stress_calls + (stress_calls > 1u ? 1u : 0u),
           stress_calls + (stress_calls > 1u ? 1u : 0u));
    fflush(stdout);

    if (q_b_chain) {
        printf("q_b_launches=%u\nq_b_to_a_sync=0\n",
               total_calls);
        fflush(stdout);
    }

    if (stress_calls > 1u) {
        if ((q_b_chain &&
             !launch_q_b(heads, NULL, model, model_bytes, q_b_offset,
                         input, HALF_TOK, POS0)) ||
            !ds4_gpu_attention_output_q8_batch_row_owned_sm75_tensor(
                out, low, NULL, NULL, model, model_bytes,
                out_a_offset, out_b_offset, GROUP_DIM, RANK, N_GROUP,
                OUT_DIM, heads, HALF_TOK) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(
                low, 0u, reference_low_host, low_bytes) ||
            !ds4_gpu_tensor_read(
                out, 0u, reference_out_host, out_bytes)) {
            fprintf(stderr,
                    "error: native-stream output A-to-B reference failed\n");
            goto cleanup;
        }
    }
    for (uint32_t call = 0u; call < stress_calls; call++) {
        if ((q_b_chain &&
             !launch_q_b(heads, NULL, model, model_bytes, q_b_offset,
                         input, HALF_TOK, POS0)) ||
            !ds4_gpu_attention_output_q8_batch_row_owned_sm75_tensor(
                out, low, NULL, NULL, model, model_bytes,
                out_a_offset, out_b_offset, GROUP_DIM, RANK, N_GROUP,
                OUT_DIM, heads, HALF_TOK)) {
            fprintf(stderr,
                    "error: native-stream output A-to-B call %u/%u failed\n",
                    call + 1u, stress_calls);
            goto cleanup;
        }
    }
    if (!ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(
            guarded_low, 0u, guarded_low_host, guarded_low_bytes) ||
        !ds4_gpu_tensor_read(
            guarded_out, 0u, guarded_out_host, guarded_out_bytes)) {
        fprintf(stderr,
                "error: native-stream output A-to-B execution probe failed\n");
        goto cleanup;
    }

    uint64_t low_prefix_mismatches = 0u;
    uint64_t low_suffix_mismatches = 0u;
    uint64_t out_prefix_mismatches = 0u;
    uint64_t out_suffix_mismatches = 0u;
    for (uint64_t i = 0u; i < PROJECTION_PROBE_GUARD_BYTES; i++) {
        low_prefix_mismatches += guarded_low_host[i] != 0xa5u;
        low_suffix_mismatches +=
            guarded_low_host[PROJECTION_PROBE_GUARD_BYTES + low_bytes + i] !=
            0xa5u;
        out_prefix_mismatches += guarded_out_host[i] != 0x5au;
        out_suffix_mismatches +=
            guarded_out_host[PROJECTION_PROBE_GUARD_BYTES + out_bytes + i] !=
            0x5au;
    }
    const float *low_output = (const float *)(
        guarded_low_host + PROJECTION_PROBE_GUARD_BYTES);
    const float *output = (const float *)(
        guarded_out_host + PROJECTION_PROBE_GUARD_BYTES);
    uint64_t low_finite = 0u;
    uint64_t low_nonzero = 0u;
    uint64_t out_finite = 0u;
    uint64_t out_nonzero = 0u;
    uint64_t low_hash = UINT64_C(1469598103934665603);
    uint64_t out_hash = UINT64_C(1469598103934665603);
    uint64_t low_repeat_bit_mismatches = 0u;
    uint64_t out_repeat_bit_mismatches = 0u;
    for (uint64_t i = 0u; i < low_count; i++) {
        low_finite += isfinite(low_output[i]) != 0;
        low_nonzero += low_output[i] != 0.0f;
    }
    for (uint64_t i = 0u; i < out_count; i++) {
        out_finite += isfinite(output[i]) != 0;
        out_nonzero += output[i] != 0.0f;
    }
    for (uint64_t i = 0u; i < low_bytes; i++) {
        low_hash ^= guarded_low_host[PROJECTION_PROBE_GUARD_BYTES + i];
        low_hash *= UINT64_C(1099511628211);
    }
    for (uint64_t i = 0u; i < out_bytes; i++) {
        out_hash ^= guarded_out_host[PROJECTION_PROBE_GUARD_BYTES + i];
        out_hash *= UINT64_C(1099511628211);
    }
    if (stress_calls > 1u) {
        for (uint64_t i = 0u; i < low_count; i++) {
            low_repeat_bit_mismatches += memcmp(
                reference_low_host + i * sizeof(float),
                guarded_low_host + PROJECTION_PROBE_GUARD_BYTES +
                    i * sizeof(float), sizeof(float)) != 0;
        }
        for (uint64_t i = 0u; i < out_count; i++) {
            out_repeat_bit_mismatches += memcmp(
                reference_out_host + i * sizeof(float),
                guarded_out_host + PROJECTION_PROBE_GUARD_BYTES +
                    i * sizeof(float), sizeof(float)) != 0;
        }
    }
    printf("low_canary_prefix_mismatches=%llu\n"
           "low_canary_suffix_mismatches=%llu\n"
           "out_canary_prefix_mismatches=%llu\n"
           "out_canary_suffix_mismatches=%llu\n"
           "low_finite=%llu\nlow_nonzero=%llu\n"
           "output_finite=%llu\noutput_nonzero=%llu\n"
           "low_fnv1a64=%016llx\noutput_fnv1a64=%016llx\n"
           "low_repeat_bit_mismatches=%llu\n"
           "output_repeat_bit_mismatches=%llu\n",
           (unsigned long long)low_prefix_mismatches,
           (unsigned long long)low_suffix_mismatches,
           (unsigned long long)out_prefix_mismatches,
           (unsigned long long)out_suffix_mismatches,
           (unsigned long long)low_finite,
           (unsigned long long)low_nonzero,
           (unsigned long long)out_finite,
           (unsigned long long)out_nonzero,
           (unsigned long long)low_hash,
           (unsigned long long)out_hash,
           (unsigned long long)low_repeat_bit_mismatches,
           (unsigned long long)out_repeat_bit_mismatches);
    if (low_prefix_mismatches || low_suffix_mismatches ||
        out_prefix_mismatches || out_suffix_mismatches ||
        low_finite != low_count || low_nonzero == 0u ||
        out_finite != out_count || out_nonzero == 0u ||
        low_repeat_bit_mismatches || out_repeat_bit_mismatches) {
        fprintf(stderr, "error: output A-to-B probe validation failed\n");
        goto cleanup;
    }
    printf("diagnostic_conclusion=%s\n"
           "harness_status=ok\n",
           repeated_chain ? "native-stream-projection-chain-repeat-clean" :
           q_b_chain ? "native-stream-projection-chain-clean" :
           stress_calls > 1u ? "native-stream-output-ab-repeat-clean" :
                               "native-stream-output-ab-single-call-clean");
    status = 1;

cleanup:
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(low);
    ds4_gpu_tensor_free(guarded_out);
    ds4_gpu_tensor_free(guarded_low);
    ds4_gpu_tensor_free(input);
    ds4_gpu_tensor_free(heads);
    free(guarded_out_host);
    free(guarded_low_host);
    free(reference_out_host);
    free(reference_low_host);
    free(input_host);
    free(heads_host);
    return status;
}

int main(void) {
    const int native_q_b_diagnostic =
        getenv("DS4_TOKEN_ROW_ARITHMETIC_NATIVE_Q_B") != NULL;
    const int output_b_canonical_diagnostic =
        getenv("DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_CANONICAL") != NULL;
    const int output_b_canonical_replay_diagnostic =
        getenv("DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_CANONICAL_REPLAY") != NULL;
    const int output_b_canonical_suffix_replay_diagnostic =
        getenv("DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_CANONICAL_SUFFIX_REPLAY") !=
        NULL;
    const int output_b_production103_replay_diagnostic =
        getenv("DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_PRODUCTION103_REPLAY") !=
        NULL;
    const int output_b_production103_pinned_half_diagnostic =
        getenv(
            "DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_PRODUCTION103_PINNED_HALF") !=
        NULL;
    const int output_b_production103_burnin_diagnostic =
        output_b_production103_replay_diagnostic ||
        output_b_production103_pinned_half_diagnostic;
    const int output_b_working_set_replay_diagnostic =
        output_b_canonical_suffix_replay_diagnostic ||
        output_b_production103_burnin_diagnostic;
    const int output_b_native_diagnostic =
        getenv("DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_NATIVE") != NULL;
    const int output_a_native_diagnostic =
        getenv("DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_A_NATIVE") != NULL;
    const int output_ab_native_diagnostic =
        getenv("DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_AB_NATIVE") != NULL;
    const int output_ab_native_repeat_diagnostic =
        getenv("DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_AB_NATIVE_REPEAT") != NULL;
    const int projection_chain_native_diagnostic =
        getenv("DS4_TOKEN_ROW_ARITHMETIC_PROJECTION_CHAIN_NATIVE") != NULL;
    const int projection_chain_native_repeat_diagnostic =
        getenv("DS4_TOKEN_ROW_ARITHMETIC_PROJECTION_CHAIN_NATIVE_REPEAT") !=
        NULL;
    const int output_ab_native_required =
        output_ab_native_diagnostic || output_ab_native_repeat_diagnostic ||
        projection_chain_native_diagnostic ||
        projection_chain_native_repeat_diagnostic;
    const int native_q_b_required =
        native_q_b_diagnostic || projection_chain_native_diagnostic ||
        projection_chain_native_repeat_diagnostic;
    const int output_a_native_required =
        output_a_native_diagnostic || output_ab_native_required;
    const int output_b_native_required =
        output_b_native_diagnostic || output_ab_native_required;
    const int output_b_single_diagnostic =
        output_b_canonical_diagnostic || output_b_native_diagnostic;
    const int output_b_diagnostic =
        output_b_single_diagnostic || output_b_canonical_replay_diagnostic;
    const int projection_diagnostic =
        output_a_native_required || output_b_diagnostic;
    const uint64_t q_b_bytes = Q_DIM * (IN_DIM / 32u) * 34u;
    const uint64_t sinks_offset = q_b_bytes;
    const uint64_t sinks_bytes = N_HEAD * sizeof(float);
    const uint64_t out_a_offset = sinks_offset + sinks_bytes;
    const uint64_t out_a_bytes = LOW_DIM * (GROUP_DIM / 32u) * 34u;
    const uint64_t out_b_offset = out_a_offset + out_a_bytes;
    const uint64_t out_b_bytes = OUT_DIM * (LOW_DIM / 32u) * 34u;
    const uint64_t native_q_b_offset = out_b_offset + out_b_bytes;
    const uint64_t native_out_a_offset = native_q_b_offset +
        (native_q_b_required ? q_b_bytes : 0u);
    const uint64_t native_out_b_offset = native_out_a_offset +
        (output_a_native_required ? out_a_bytes : 0u);
    const uint64_t model_bytes = native_out_b_offset +
        (output_b_native_required ? out_b_bytes : 0u);
    const uint64_t input_count = (uint64_t)N_TOK * IN_DIM;
    const uint64_t q_count = (uint64_t)N_TOK * Q_DIM;
    const uint64_t heads_count = q_count;
    const uint64_t low_count = (uint64_t)N_TOK * LOW_DIM;
    const uint64_t out_count = (uint64_t)N_TOK * OUT_DIM;
    const uint64_t raw_count = (uint64_t)N_TOK * HEAD_DIM;
    const uint64_t comp_count = (uint64_t)N_COMP * HEAD_DIM;
    const uint64_t input_bytes = input_count * sizeof(float);
    const uint64_t q_bytes = q_count * sizeof(float);
    const uint64_t q_half_bytes = q_count * sizeof(uint16_t);
    const uint64_t heads_bytes = heads_count * sizeof(float);
    const uint64_t low_bytes = low_count * sizeof(float);
    const uint64_t out_bytes = out_count * sizeof(float);
    const uint64_t input_half_bytes = input_bytes / 2u;
    const uint64_t q_row_half_bytes = q_bytes / 2u;
    const uint64_t qh_row_half_bytes = q_half_bytes / 2u;
    const uint64_t heads_row_half_bytes = heads_bytes / 2u;
    const uint64_t low_row_half_bytes = low_bytes / 2u;
    const uint64_t out_row_half_bytes = out_bytes / 2u;

    unsigned char *model = NULL;
    float *input_host = NULL, *reference = NULL, *candidate = NULL;
    float *actual_low_host = NULL;
    float *combined_full_host = NULL, *combined_split_host = NULL;
    float *shipping_b_host = NULL;
    float *raw_host = NULL, *comp_host = NULL;
    uint16_t *reference_half = NULL, *candidate_half = NULL;
    ds4_gpu_tensor *input = NULL, *q_full = NULL, *q_split = NULL;
    ds4_gpu_tensor *qh_full = NULL, *qh_split = NULL;
    ds4_gpu_tensor *heads_full = NULL, *heads_split = NULL;
    ds4_gpu_tensor *low_full = NULL, *low_split = NULL;
    ds4_gpu_tensor *out_full = NULL, *out_split = NULL;
    ds4_gpu_tensor *raw = NULL, *comp = NULL;
    ds4_gpu_tensor *input0 = NULL, *input1 = NULL;
    ds4_gpu_tensor *q0 = NULL, *q1 = NULL, *qh0 = NULL, *qh1 = NULL;
    ds4_gpu_tensor *q_ref0 = NULL, *q_ref1 = NULL;
    ds4_gpu_tensor *heads0 = NULL, *heads1 = NULL;
    ds4_gpu_tensor *heads_ref0 = NULL, *heads_ref1 = NULL;
    ds4_gpu_tensor *low0 = NULL, *low1 = NULL, *out0 = NULL, *out1 = NULL;
    ds4_gpu_tensor *low_ref0 = NULL, *low_ref1 = NULL;
    int initialized = 0;
    int status = 1;
    const uint32_t output_ab_stress_calls =
        (output_ab_native_repeat_diagnostic ||
         projection_chain_native_repeat_diagnostic)
            ? positive_env_u32(
                  "DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_AB_REPEAT_CALLS",
                  256u, 4096u)
            : 1u;
    const uint32_t output_b_production103_calls =
        output_b_production103_burnin_diagnostic ?
            positive_env_u32(
                "DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_PRODUCTION103_CALLS",
                1024u, 16384u) : 1u;
    const uint32_t output_b_production103_batch =
        output_b_production103_burnin_diagnostic ?
            positive_env_u32(
                "DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_PRODUCTION103_BATCH",
                10u, 1024u) : 1u;

    if (!output_b_production103_calls || !output_b_production103_batch ||
        output_b_production103_batch > output_b_production103_calls) {
        if (output_b_production103_calls &&
            output_b_production103_batch > output_b_production103_calls) {
            fprintf(stderr,
                    "error: production-103 batch must not exceed call count\n");
        }
        goto cleanup;
    }
    if (output_ab_stress_calls == 0u ||
        ((output_ab_native_repeat_diagnostic ||
          projection_chain_native_repeat_diagnostic) &&
         output_ab_stress_calls < 2u)) {
        if (output_ab_stress_calls != 0u) {
            fprintf(stderr,
                    "error: repeated output A-to-B probe requires at least "
                    "two stress calls\n");
        }
        goto cleanup;
    }

    model = (unsigned char *)malloc((size_t)model_bytes);
    input_host = (float *)malloc((size_t)input_bytes);
    reference = (float *)malloc((size_t)q_bytes);
    candidate = (float *)malloc((size_t)q_bytes);
    reference_half = (uint16_t *)malloc((size_t)q_half_bytes);
    candidate_half = (uint16_t *)malloc((size_t)q_half_bytes);
    raw_host = (float *)malloc((size_t)raw_count * sizeof(float));
    comp_host = (float *)malloc((size_t)comp_count * sizeof(float));
    if (!model || !input_host || !reference || !candidate ||
        !reference_half || !candidate_half || !raw_host || !comp_host) {
        fprintf(stderr, "error: host allocation failed\n");
        goto cleanup;
    }
    /* q_count is four times low_count, leaving room in the comparison slabs
     * for one real A output and one combined-output snapshot per arm. */
    actual_low_host = reference + low_count;
    combined_full_host = reference + 2u * low_count;
    combined_split_host = candidate + 2u * low_count;
    shipping_b_host = reference + 2u * low_count + out_count;

    build_q8_rows(model, Q_DIM, IN_DIM, 17u);
    if (native_q_b_required) {
        pack_q8_rows_warp32(
            model + native_q_b_offset, model, Q_DIM, IN_DIM);
    }
    for (uint32_t h = 0u; h < N_HEAD; h++) {
        const float sink = (float)((int)(h % 11u) - 5) / 32.0f;
        memcpy(model + sinks_offset + (uint64_t)h * sizeof(float),
               &sink, sizeof(sink));
    }
    build_q8_rows(model + out_a_offset, LOW_DIM, GROUP_DIM, 37u);
    build_q8_rows(model + out_b_offset, OUT_DIM, LOW_DIM, 53u);
    if (output_a_native_required) {
        pack_q8_rows_warp32(
            model + native_out_a_offset, model + out_a_offset,
            LOW_DIM, GROUP_DIM);
    }
    if (output_b_native_required) {
        pack_q8_rows_b_kshards_warp32(
            model + native_out_b_offset, model + out_b_offset,
            OUT_DIM, LOW_DIM);
    }
    for (uint64_t i = 0u; i < input_count; i++) {
        const int value = (int)((i * 29u + (i >> 5u) * 17u +
            (i / IN_DIM) * 7u + 23u) % 257u) - 128;
        input_host[i] = (float)value / 128.0f;
    }
    for (uint64_t i = 0u; i < raw_count; i++) {
        const int value = (int)((i * 13u + (i >> 5u) * 17u + 37u) % 257u) - 128;
        raw_host[i] = (float)value / 2048.0f;
    }
    for (uint64_t i = 0u; i < comp_count; i++) {
        const int value = (int)((i * 31u + (i >> 4u) * 7u + 41u) % 263u) - 131;
        comp_host[i] = (float)value / 2048.0f;
    }

    (void)setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    (void)setenv("DS4_CUDA_Q8_F16_CACHE_MB", "512", 1);
    (void)setenv("DS4_CUDA_Q8_F16_CACHE_RESERVE_MB", "1", 1);
    (void)setenv("DS4_CUDA_NO_Q8_F32_CACHE", "1", 1);
    (void)setenv("DS4_CUDA_ATTENTION_OUTPUT_PRELOAD", "1", 1);
    (void)setenv("DS4_CUDA_NO_TF32", "1", 1);
    (void)setenv("DS4_CUDA_T32_F16_FUSED", "1", 1);
    (void)unsetenv("DS4_CUDA_NO_T32_F16_FUSED");
    (void)unsetenv("DS4_CUDA_NO_Q8_F16_CACHE");
    (void)unsetenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE");
    (void)unsetenv("DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A");
    (void)unsetenv("DS4_CUDA_NO_CUBLAS_ATTENTION");
    (void)unsetenv("DS4_CUDA_NO_WINDOW_ATTENTION");
    (void)unsetenv("DS4_CUDA_T32_F16_GEMM_ALGO_DIAGNOSTIC");
    (void)unsetenv("DS4_CUDA_ATTN_OUTPUT_B_F16_GEMM_ALGO_DIAGNOSTIC");
    (void)unsetenv("DS4_CUDA_TP_PREFILL_ATTN_TOKEN_ROWS_WEIGHT_MODE");
    (void)unsetenv("DS4_CUDA_TP_PREFILL_ATTN_TOKEN_ROWS_PIPELINE_PAIRS");
    if (output_ab_native_required) {
        (void)setenv("DS4_CUDA_Q8_NATIVE_REBASE_AUDIT", "1", 1);
    } else {
        (void)unsetenv("DS4_CUDA_Q8_NATIVE_REBASE_AUDIT");
    }
    if (native_q_b_required) {
        (void)setenv(
            "DS4_CUDA_TOKEN_ROWS_NATIVE_STREAM_LOCAL_DIAGNOSTIC", "1", 1);
    } else {
        (void)unsetenv(
            "DS4_CUDA_TOKEN_ROWS_NATIVE_STREAM_LOCAL_DIAGNOSTIC");
    }
    if (projection_chain_native_diagnostic ||
        projection_chain_native_repeat_diagnostic) {
        (void)setenv(
            "DS4_CUDA_TOKEN_ROWS_NATIVE_STREAM_LOCAL_NO_CHECKPOINT", "1", 1);
    } else {
        (void)unsetenv(
            "DS4_CUDA_TOKEN_ROWS_NATIVE_STREAM_LOCAL_NO_CHECKPOINT");
    }
    if (output_b_single_diagnostic || output_ab_native_diagnostic) {
        (void)setenv("DS4_CUDA_OUTPUT_B_LOCAL_DIAGNOSTIC", "1", 1);
    } else {
        (void)unsetenv("DS4_CUDA_OUTPUT_B_LOCAL_DIAGNOSTIC");
    }
    if (output_a_native_diagnostic) {
        (void)setenv("DS4_CUDA_OUTPUT_A_LOCAL_DIAGNOSTIC", "1", 1);
        (void)setenv("DS4_CUDA_OUTPUT_A_ONLY_LOCAL_DIAGNOSTIC", "1", 1);
    } else {
        (void)unsetenv("DS4_CUDA_OUTPUT_A_LOCAL_DIAGNOSTIC");
        (void)unsetenv("DS4_CUDA_OUTPUT_A_ONLY_LOCAL_DIAGNOSTIC");
    }

    if (!ds4_gpu_init()) {
        fprintf(stderr, "error: CUDA initialization failed\n");
        goto cleanup;
    }
    initialized = 1;
    if (!ds4_gpu_set_model_map(model, model_bytes) ||
        ((!projection_diagnostic || output_b_canonical_replay_diagnostic) &&
         (!ds4_gpu_cache_q8_f16_range_on_device(
              model, model_bytes, 0u, q_b_bytes, IN_DIM, Q_DIM, 0,
              "attn_q_b") ||
          !ds4_gpu_cache_q8_f16_range_on_device(
              model, model_bytes, out_a_offset, out_a_bytes,
              GROUP_DIM, LOW_DIM, 0, "attn_output_a"))) ||
        (!output_b_native_required && !output_a_native_required &&
         !ds4_gpu_cache_q8_f16_range_on_device(
             model, model_bytes, out_b_offset, out_b_bytes,
             LOW_DIM, OUT_DIM, 0, "attn_output_b"))) {
        fprintf(stderr, "error: model/cache installation failed\n");
        goto cleanup;
    }
    if (native_q_b_required) {
        const ds4_tensor_range native_source = {
            native_q_b_offset, q_b_bytes, 0};
        const ds4_q8_native_range native_range = {
            native_q_b_offset, q_b_bytes,
            native_q_b_offset, q_b_bytes,
            native_q_b_offset, IN_DIM / 32u, 0u,
            IN_DIM / 32u, Q_DIM,
            DS4_Q8_NATIVE_LAYOUT_ROW_WARP32, 1, 0};
        if (ds4_gpu_device_cache_tensors(0, &native_source, 1) != 0 ||
            ds4_gpu_device_cache_q8_native_tensors(
                0, &native_range, 1) != 0) {
            fprintf(stderr,
                    "error: native q_b source installation failed\n");
            goto cleanup;
        }
    }
    if (output_a_native_required) {
        const ds4_tensor_range native_source = {
            native_out_a_offset, out_a_bytes, 0};
        const ds4_q8_native_range native_range = {
            native_out_a_offset, out_a_bytes,
            native_out_a_offset, out_a_bytes,
            native_out_a_offset, GROUP_DIM / 32u, 0u,
            GROUP_DIM / 32u, LOW_DIM,
            DS4_Q8_NATIVE_LAYOUT_ROW_WARP32, 1, 0};
        if (ds4_gpu_device_cache_tensors(0, &native_source, 1) != 0 ||
            ds4_gpu_device_cache_q8_native_tensors(
                0, &native_range, 1) != 0) {
            fprintf(stderr,
                    "error: native output-A source installation failed\n");
            goto cleanup;
        }
        (void)setenv("DS4_CUDA_NO_Q8_F16_CACHE", "1", 1);
        (void)setenv("DS4_CUDA_TP_PREFILL_ATTN_TOKEN_ROWS_WEIGHT_MODE",
                     "native-stream", 1);
        (void)setenv("DS4_CUDA_TP_PREFILL_ATTN_TOKEN_ROWS_PIPELINE_PAIRS",
                     "0", 1);
        (void)setenv(
            "DS4_CUDA_TOKEN_ROWS_NATIVE_STREAM_LOCAL_DIAGNOSTIC", "1", 1);
    }
    if (output_b_native_required) {
        const ds4_tensor_range native_source = {
            native_out_b_offset, out_b_bytes, 0};
        const ds4_q8_native_range native_range = {
            native_out_b_offset, out_b_bytes,
            native_out_b_offset, out_b_bytes,
            native_out_b_offset, LOW_DIM / 32u, 0u,
            LOW_DIM / 32u, OUT_DIM,
            DS4_Q8_NATIVE_LAYOUT_B_KSHARDS_WARP32, 1, 0};
        if (ds4_gpu_device_cache_tensors(0, &native_source, 1) != 0 ||
            ds4_gpu_device_cache_q8_native_tensors(
                0, &native_range, 1) != 0) {
            fprintf(stderr,
                    "error: native output-B source installation failed\n");
            goto cleanup;
        }
        (void)setenv("DS4_CUDA_NO_Q8_F16_CACHE", "1", 1);
        (void)setenv("DS4_CUDA_TP_PREFILL_ATTN_TOKEN_ROWS_WEIGHT_MODE",
                     "native-stream", 1);
        (void)setenv("DS4_CUDA_TP_PREFILL_ATTN_TOKEN_ROWS_PIPELINE_PAIRS",
                     "0", 1);
        (void)setenv(
            "DS4_CUDA_TOKEN_ROWS_NATIVE_STREAM_LOCAL_DIAGNOSTIC", "1", 1);
    }
    if (output_b_canonical_replay_diagnostic) {
        if (!ds4_gpu_synchronize() ||
            !run_output_b_canonical_replay(
                model, model_bytes, out_b_offset)) {
            goto cleanup;
        }
        status = 0;
        goto cleanup;
    }
    if (output_b_single_diagnostic) {
        if (!ds4_gpu_synchronize() ||
            !run_output_b_single_launch(
                model, model_bytes,
                output_b_native_diagnostic
                    ? native_out_b_offset : out_b_offset,
                output_b_native_diagnostic)) {
            goto cleanup;
        }
        status = 0;
        goto cleanup;
    }
    if (output_a_native_diagnostic) {
        if (!ds4_gpu_synchronize() ||
            !run_output_a_native_single_launch(
                model, model_bytes, native_out_a_offset, out_b_offset)) {
            goto cleanup;
        }
        status = 0;
        goto cleanup;
    }
    if (output_ab_native_required) {
        if (!ds4_gpu_synchronize() ||
            !run_output_ab_native_calls(
                model, model_bytes,
                native_q_b_offset, native_out_a_offset, native_out_b_offset,
                output_ab_stress_calls,
                projection_chain_native_diagnostic ||
                projection_chain_native_repeat_diagnostic)) {
            goto cleanup;
        }
        status = 0;
        goto cleanup;
    }

    input = ds4_gpu_tensor_alloc(input_bytes);
    q_full = ds4_gpu_tensor_alloc(q_bytes);
    q_split = ds4_gpu_tensor_alloc(q_bytes);
    qh_full = ds4_gpu_tensor_alloc(q_half_bytes);
    qh_split = ds4_gpu_tensor_alloc(q_half_bytes);
    heads_full = ds4_gpu_tensor_alloc(heads_bytes);
    heads_split = ds4_gpu_tensor_alloc(heads_bytes);
    low_full = ds4_gpu_tensor_alloc(low_bytes);
    low_split = ds4_gpu_tensor_alloc(low_bytes);
    out_full = ds4_gpu_tensor_alloc(out_bytes);
    out_split = ds4_gpu_tensor_alloc(out_bytes);
    raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    if (!input || !q_full || !q_split || !qh_full || !qh_split ||
        !heads_full || !heads_split || !low_full || !low_split ||
        !out_full || !out_split || !raw || !comp ||
        !ds4_gpu_tensor_write(input, 0u, input_host, input_bytes) ||
        !ds4_gpu_tensor_write(raw, 0u, raw_host,
                              raw_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(comp, 0u, comp_host,
                              comp_count * sizeof(float))) {
        fprintf(stderr, "error: device allocation/setup failed\n");
        goto cleanup;
    }

    input0 = ds4_gpu_tensor_view(input, 0u, input_half_bytes);
    input1 = ds4_gpu_tensor_view(input, input_half_bytes, input_half_bytes);
    q0 = ds4_gpu_tensor_view(q_split, 0u, q_row_half_bytes);
    q1 = ds4_gpu_tensor_view(q_split, q_row_half_bytes, q_row_half_bytes);
    qh0 = ds4_gpu_tensor_view(qh_split, 0u, qh_row_half_bytes);
    qh1 = ds4_gpu_tensor_view(qh_split, qh_row_half_bytes, qh_row_half_bytes);
    q_ref0 = ds4_gpu_tensor_view(q_full, 0u, q_row_half_bytes);
    q_ref1 = ds4_gpu_tensor_view(q_full, q_row_half_bytes, q_row_half_bytes);
    heads0 = ds4_gpu_tensor_view(heads_split, 0u, heads_row_half_bytes);
    heads1 = ds4_gpu_tensor_view(heads_split, heads_row_half_bytes,
                                 heads_row_half_bytes);
    heads_ref0 = ds4_gpu_tensor_view(heads_full, 0u, heads_row_half_bytes);
    heads_ref1 = ds4_gpu_tensor_view(heads_full, heads_row_half_bytes,
                                     heads_row_half_bytes);
    low0 = ds4_gpu_tensor_view(low_split, 0u, low_row_half_bytes);
    low1 = ds4_gpu_tensor_view(low_split, low_row_half_bytes,
                               low_row_half_bytes);
    out0 = ds4_gpu_tensor_view(out_split, 0u, out_row_half_bytes);
    out1 = ds4_gpu_tensor_view(out_split, out_row_half_bytes,
                               out_row_half_bytes);
    low_ref0 = ds4_gpu_tensor_view(low_full, 0u, low_row_half_bytes);
    low_ref1 = ds4_gpu_tensor_view(low_full, low_row_half_bytes,
                                   low_row_half_bytes);
    if (!input0 || !input1 || !q0 || !q1 || !qh0 || !qh1 || !q_ref0 ||
        !q_ref1 || !heads0 ||
        !heads1 || !heads_ref0 || !heads_ref1 || !low0 || !low1 || !out0 ||
        !out1 || !low_ref0 || !low_ref1) {
        fprintf(stderr, "error: tensor-view construction failed\n");
        goto cleanup;
    }

    printf("scenario=sm75-token-row-arithmetic-full512-vs-row256x2\n"
           "scope=bounded-single-gpu-production-shapes\n"
           "production_default_changed=0\n"
           "n_tokens=%u\nhalf_tokens=%u\nq_dim=%llu\nlow_dim=%llu\n"
           "output_dim=%u\nf16_cache_bytes=%llu\n",
           N_TOK, HALF_TOK, (unsigned long long)Q_DIM,
           (unsigned long long)LOW_DIM, OUT_DIM,
           (unsigned long long)(2u * (Q_DIM * IN_DIM +
               LOW_DIM * GROUP_DIM + OUT_DIM * LOW_DIM)));
    if (output_b_canonical_suffix_replay_diagnostic) {
        printf("diagnostic_scope=output-b-canonical-suffix-replay\n"
               "fidelity=working-set-and-suffix-replay-without-burn-in\n"
               "source_failure_archive=sm75-token-row-arithmetic-20260906T213618Z\n"
               "resident_f16_cache_bytes=%llu\n"
               "device_working_set_bytes=%llu\n"
               "peer_access=none\nnative_stream=off\n"
               "pre_suffix_q_b_attention_output_a=on\n"
               "exhaustive_algorithm_sweep=off\n"
               "historical_timing_burn_in=off\n"
               "suffix_submission_fencing=original-phase-boundaries\n",
               (unsigned long long)(2u * (Q_DIM * IN_DIM +
                   LOW_DIM * GROUP_DIM + OUT_DIM * LOW_DIM)),
               (unsigned long long)(input_bytes + 2u * q_bytes +
                   2u * q_half_bytes + 2u * heads_bytes + 2u * low_bytes +
                   2u * out_bytes + raw_count * sizeof(float) +
                   comp_count * sizeof(float)));
    } else if (output_b_production103_burnin_diagnostic) {
        printf("diagnostic_scope=%s\n"
               "fidelity=%s\n"
               "source_failure_archive=sm75-token-row-arithmetic-20260906T213618Z\n"
               "resident_f16_cache_bytes=%llu\n"
               "device_working_set_bytes=%llu\n"
               "peer_access=none\nnative_stream=off\n"
               "pre_suffix_q_b_attention_output_a=on\n"
               "production_b_algorithm=103\n"
               "production_b_algorithm_name=CUBLAS_GEMM_ALGO3_TENSOR_OP\n"
               "production_b_rows=256\n"
               "production_b_burnin_calls=%u\n"
               "production_b_burnin_batch=%u\n"
               "exhaustive_algorithm_sweep=off\n"
               "historical_mixed_algorithm_timing=off\n"
               "suffix_half_algorithm=%s\n"
               "suffix_submission_fencing=checkpointed-default-transitions\n",
               output_b_production103_pinned_half_diagnostic ?
                   "output-b-production103-pinned-half" :
                   "output-b-production103-replay",
               output_b_production103_pinned_half_diagnostic ?
                   "working-set-production103-default512-pinned103-half" :
                   "working-set-production-algorithm-and-suffix-replay",
               (unsigned long long)(2u * (Q_DIM * IN_DIM +
                   LOW_DIM * GROUP_DIM + OUT_DIM * LOW_DIM)),
               (unsigned long long)(input_bytes + 2u * q_bytes +
                   2u * q_half_bytes + 2u * heads_bytes + 2u * low_bytes +
                   2u * out_bytes + raw_count * sizeof(float) +
                   comp_count * sizeof(float)),
               output_b_production103_calls,
               output_b_production103_batch,
               output_b_production103_pinned_half_diagnostic ?
                   "103:CUBLAS_GEMM_ALGO3_TENSOR_OP" : "DEFAULT");
        fflush(stdout);
    }

    /* q_b: identical rows and weights, only the cuBLAS N dimension and
     * positional row origin differ. */
    if (!launch_q_b(q_full, qh_full, model, model_bytes, 0u, input,
                    N_TOK, POS0) ||
        !launch_q_b(q0, qh0, model, model_bytes, 0u, input0,
                    HALF_TOK, POS0) ||
        !launch_q_b(q1, qh1, model, model_bytes, 0u, input1,
                    HALF_TOK, POS0 + HALF_TOK) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(qh_full, 0u, reference_half, q_half_bytes) ||
        !ds4_gpu_tensor_read(qh_split, 0u, candidate_half, q_half_bytes) ||
        !ds4_gpu_tensor_read(q_full, 0u, reference, q_bytes) ||
        !ds4_gpu_tensor_read(q_split, 0u, candidate, q_bytes)) {
        fprintf(stderr, "error: q_b boundary runtime failed\n");
        goto cleanup;
    }
    const diff_metrics qh_diff = compare_u16(
        reference_half, candidate_half, q_count);
    const diff_metrics q_diff = compare_f32(reference, candidate, q_count);
    report_diff("q-b-f16-projection-full512-vs-row256x2", "f16", q_count,
                qh_diff);
    report_diff("q-b-rms-rope-full512-vs-row256x2", "f32", q_count, q_diff);

    /* Match the current internal token-row candidate as well: control supplies
     * its persistent q_half, while each candidate call asks the helper to use
     * its internal temporary q_half. */
    if (!launch_q_b(q0, NULL, model, model_bytes, 0u, input0,
                    HALF_TOK, POS0) ||
        !launch_q_b(q1, NULL, model, model_bytes, 0u, input1,
                    HALF_TOK, POS0 + HALF_TOK) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(q_split, 0u, candidate, q_bytes)) {
        fprintf(stderr, "error: internal token-row q_b scratch runtime failed\n");
        goto cleanup;
    }
    const diff_metrics q_scratch_diff = compare_f32(
        reference, candidate, q_count);
    report_diff("q-b-control-persistent-vs-internal-row-scratch256x2", "f32",
                q_count, q_scratch_diff);
    diff_metrics native_qh_diff = {0u, UINT64_MAX, 0.0, 0.0};
    diff_metrics native_q_diff = {0u, UINT64_MAX, 0.0, 0.0};
    diff_metrics native_scratch_diff = {0u, UINT64_MAX, 0.0, 0.0};
    if (native_q_b_diagnostic) {
        /* Keep the canonical control cache installed above, but prevent the
         * lazy cache path from treating the independently packed native range
         * as canonical Q8_0 and admitting a new FP16 entry before the native
         * source lookup runs.  Production gets this lookup-only behavior from
         * its finalized cache plan; the bounded single-GPU harness has no
         * startup planner, so make that boundary explicit here. */
        (void)setenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE", "1", 1);
        (void)setenv("DS4_CUDA_TP_PREFILL_ATTN_TOKEN_ROWS_WEIGHT_MODE",
                     "native-stream", 1);
        (void)setenv("DS4_CUDA_TP_PREFILL_ATTN_TOKEN_ROWS_PIPELINE_PAIRS",
                     "0", 1);
        if (!launch_q_b(q0, qh0, model, model_bytes, native_q_b_offset,
                        input0, HALF_TOK, POS0) ||
            !launch_q_b(q1, qh1, model, model_bytes, native_q_b_offset,
                        input1, HALF_TOK, POS0 + HALF_TOK) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(
                qh_split, 0u, candidate_half, q_half_bytes) ||
            !ds4_gpu_tensor_read(q_split, 0u, candidate, q_bytes)) {
            fprintf(stderr, "error: native-stream q_b runtime failed\n");
            goto cleanup;
        }
        native_qh_diff = compare_u16(
            reference_half, candidate_half, q_count);
        native_q_diff = compare_f32(reference, candidate, q_count);
        report_diff("q-b-native-f16-control-vs-row256x2", "f16", q_count,
                    native_qh_diff);
        report_diff("q-b-native-rms-rope-control-vs-row256x2", "f32",
                    q_count, native_q_diff);

        if (!launch_q_b(q0, NULL, model, model_bytes, native_q_b_offset,
                        input0, HALF_TOK, POS0) ||
            !launch_q_b(q1, NULL, model, model_bytes, native_q_b_offset,
                        input1, HALF_TOK, POS0 + HALF_TOK) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(q_split, 0u, candidate, q_bytes)) {
            fprintf(stderr,
                    "error: native-stream internal q_b scratch failed\n");
            goto cleanup;
        }
        native_scratch_diff = compare_f32(reference, candidate, q_count);
        report_diff("q-b-native-control-vs-internal-scratch256x2", "f32",
                    q_count, native_scratch_diff);
    }
    if (getenv("DS4_TOKEN_ROW_ARITHMETIC_STOP_AFTER_Q_B") ||
        getenv("DS4_TOKEN_ROW_ARITHMETIC_SANITIZER_SMOKE")) {
        printf("diagnostic_scope=%s\n",
               native_q_b_diagnostic ? "q-b-native-only" : "q-b-only");
        printf("boundary=static-mixed-attention-shipping-vs-full-range512,status=skipped-by-q-b-scope\n");
        printf("boundary=static-mixed-attention-full-range512-vs-row256x2,status=skipped-by-q-b-scope\n");
        printf("boundary=inverse-rope-full512-vs-row256x2,status=skipped-by-q-b-scope\n");
        printf("boundary=output-a-full512-vs-row256x2,status=skipped-by-q-b-scope\n");
        printf("boundary=output-a-plus-b-full512-vs-row256x2,status=skipped-by-q-b-scope\n");
        printf("boundary=output-b-from-identical-actual-a-low-full512-vs-row256x2,status=skipped-by-q-b-scope\n");
        printf("boundary=output-a-to-b-chain-reference,status=skipped-by-q-b-scope\n");
        printf("boundary=output-b-structured-control-full512-vs-row256x2,status=skipped-by-q-b-scope\n");
        printf("diagnostic_conclusion=%s\n",
               qh_diff.mismatches ? "first-divergence-q-b-f16-projection" :
               q_diff.mismatches ? "first-divergence-q-b-postprocess" :
               q_scratch_diff.mismatches ?
                   "first-divergence-q-b-internal-scratch-shape" :
               native_qh_diff.mismatches ?
                   "first-divergence-q-b-native-f16-projection" :
               native_q_diff.mismatches ?
                   "first-divergence-q-b-native-postprocess" :
               native_scratch_diff.mismatches ?
                   "first-divergence-q-b-native-internal-scratch" :
               "q-b-boundaries-bit-exact");
        printf("harness_status=ok\n");
        status = 0;
        goto cleanup;
    }

    /* Static mixed attention sees the same full q_b result and the same full
     * current/compressed KV tensors in every arm.  First distinguish the
     * shipping full wrapper from a 512-row launch of the range kernel.  Then
     * keep the range kernel fixed and isolate its 512 versus 256+256 extent. */
    if (!ds4_gpu_attention_prefill_static_mixed_heads_tensor(
            heads_full, model, model_bytes, sinks_offset, q_full, raw, comp,
            0u, N_TOK, N_COMP, ATTN_WINDOW, ATTN_RATIO, N_HEAD, HEAD_DIM) ||
        !ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(
            heads_split, model, model_bytes, sinks_offset, q_full, raw, comp,
            0u, 0u, N_TOK, N_TOK, N_COMP, ATTN_WINDOW, ATTN_RATIO,
            N_HEAD, HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads_full, 0u, reference, heads_bytes) ||
        !ds4_gpu_tensor_read(heads_split, 0u, candidate, heads_bytes)) {
        fprintf(stderr, "error: static-mixed full-range boundary runtime failed\n");
        goto cleanup;
    }
    const diff_metrics attention_wrapper_diff = compare_f32(
        reference, candidate, heads_count);
    report_diff("static-mixed-attention-shipping-vs-full-range512", "f32",
                heads_count, attention_wrapper_diff);

    /* Save the full-range result as the baseline before its output tensor is
     * overwritten by the two ordered rectangular launches. */
    memcpy(reference, candidate, (size_t)heads_bytes);
    if (!ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(
            heads0, model, model_bytes, sinks_offset, q_ref0, raw, comp, 0u,
            0u, HALF_TOK, N_TOK, N_COMP, ATTN_WINDOW, ATTN_RATIO,
            N_HEAD, HEAD_DIM) ||
        !ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(
            heads1, model, model_bytes, sinks_offset, q_ref1, raw, comp, 0u,
            HALF_TOK, HALF_TOK, N_TOK, N_COMP, ATTN_WINDOW, ATTN_RATIO,
            N_HEAD, HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads_split, 0u, candidate, heads_bytes)) {
        fprintf(stderr, "error: static-mixed row-extent boundary runtime failed\n");
        goto cleanup;
    }
    const diff_metrics attention_extent_diff = compare_f32(
        reference, candidate, heads_count);
    report_diff("static-mixed-attention-full-range512-vs-row256x2", "f32",
                heads_count, attention_extent_diff);

    /* Inverse RoPE is pointwise, but explicitly verify that changing the row
     * launch extent and pos0 preserves its ordered arithmetic.  Reset the
     * split input to the identical full-reference attention output first. */
    if (!ds4_gpu_tensor_write(heads_full, 0u, reference, heads_bytes) ||
        !ds4_gpu_tensor_write(heads_split, 0u, reference, heads_bytes) ||
        !ds4_gpu_rope_tail_tensor(heads_full, N_TOK, N_HEAD, HEAD_DIM, N_ROT,
            POS0, ORIG_CTX, true, ROPE_BASE, ROPE_SCALE, ROPE_EXT,
            rope_attention_factor(), BETA_FAST, BETA_SLOW) ||
        !ds4_gpu_rope_tail_tensor(heads0, HALF_TOK, N_HEAD, HEAD_DIM, N_ROT,
            POS0, ORIG_CTX, true, ROPE_BASE, ROPE_SCALE, ROPE_EXT,
            rope_attention_factor(), BETA_FAST, BETA_SLOW) ||
        !ds4_gpu_rope_tail_tensor(heads1, HALF_TOK, N_HEAD, HEAD_DIM, N_ROT,
            POS0 + HALF_TOK, ORIG_CTX, true, ROPE_BASE, ROPE_SCALE, ROPE_EXT,
            rope_attention_factor(), BETA_FAST, BETA_SLOW) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads_full, 0u, reference, heads_bytes) ||
        !ds4_gpu_tensor_read(heads_split, 0u, candidate, heads_bytes)) {
        fprintf(stderr, "error: inverse-RoPE boundary runtime failed\n");
        goto cleanup;
    }
    const diff_metrics rope_diff = compare_f32(
        reference, candidate, heads_count);
    report_diff("inverse-rope-full512-vs-row256x2", "f32", heads_count,
                rope_diff);

    /* Feed the identical full-reference heads into both output shapes.  low
     * exposes output A, while out exposes output B; this keeps an upstream
     * difference from being misattributed to either GEMM. */
    if (output_b_production103_pinned_half_diagnostic) {
        select_b_algorithm(-1);
        printf("pre_suffix_transition_phase=default-full512-submit\n");
        fflush(stdout);
        if (!launch_output(out_full, low_full, model, model_bytes,
                           out_a_offset, out_b_offset, heads_full, N_TOK) ||
            !ds4_gpu_synchronize()) {
            fprintf(stderr,
                    "error: pre-suffix DEFAULT full512 A+B failed\n");
            goto cleanup;
        }
        printf("pre_suffix_transition_phase=default-full512-complete\n"
               "pre_suffix_transition_phase=algo103-half0-256-submit\n");
        fflush(stdout);
        select_b_algorithm(103);
        if (!launch_output(out0, low0, model, model_bytes, out_a_offset,
                           out_b_offset, heads_ref0, HALF_TOK) ||
            !ds4_gpu_synchronize()) {
            fprintf(stderr,
                    "error: pre-suffix algorithm-103 half0-256 A+B failed\n");
            goto cleanup;
        }
        printf("pre_suffix_transition_phase=algo103-half0-256-complete\n"
               "pre_suffix_transition_phase=algo103-half1-256-submit\n");
        fflush(stdout);
        if (!launch_output(out1, low1, model, model_bytes, out_a_offset,
                           out_b_offset, heads_ref1, HALF_TOK) ||
            !ds4_gpu_synchronize()) {
            fprintf(stderr,
                    "error: pre-suffix algorithm-103 half1-256 A+B failed\n");
            goto cleanup;
        }
        printf("pre_suffix_transition_phase=algo103-half1-256-complete\n");
        fflush(stdout);
        select_b_algorithm(-1);
    } else if (!launch_output(out_full, low_full, model, model_bytes,
                              out_a_offset, out_b_offset, heads_full, N_TOK) ||
               !launch_output(out0, low0, model, model_bytes, out_a_offset,
                              out_b_offset, heads_ref0, HALF_TOK) ||
               !launch_output(out1, low1, model, model_bytes, out_a_offset,
                              out_b_offset, heads_ref1, HALF_TOK) ||
               !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: output A+B boundary runtime failed\n");
        goto cleanup;
    }
    if (!ds4_gpu_tensor_read(low_full, 0u, reference, low_bytes) ||
        !ds4_gpu_tensor_read(low_split, 0u, candidate, low_bytes)) {
        fprintf(stderr, "error: output-A boundary readback failed\n");
        goto cleanup;
    }
    const diff_metrics out_a_diff = compare_f32(
        reference, candidate, low_count);
    report_diff("output-a-full512-vs-row256x2", "f32", low_count,
                out_a_diff);
    memcpy(actual_low_host, reference, (size_t)low_bytes);

    if (!ds4_gpu_tensor_read(out_full, 0u, reference, out_bytes) ||
        !ds4_gpu_tensor_read(out_split, 0u, candidate, out_bytes)) {
        fprintf(stderr, "error: output-B boundary readback failed\n");
        goto cleanup;
    }
    const diff_metrics out_b_diff = compare_f32(
        reference, candidate, out_count);
    report_diff("output-a-plus-b-full512-vs-row256x2", "f32", out_count,
                out_b_diff);
    memcpy(combined_full_host, reference, (size_t)out_bytes);
    memcpy(combined_split_host, candidate, (size_t)out_bytes);

    /* Re-run B only after the exact A result has been synchronized and copied
     * identically into both low tensors.  This distinguishes row-count-sensitive
     * B arithmetic from an A-producer/B-consumer ordering defect in the wrapper.
     * The two additional comparisons identify which composed arm, if either,
     * differs from its synchronized B reference. */
    if (!ds4_gpu_tensor_write(low_full, 0u, actual_low_host, low_bytes) ||
        !ds4_gpu_tensor_write(low_split, 0u, actual_low_host, low_bytes) ||
        !ds4_gpu_attention_output_q8_batch_b_tensor(
            out_full, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
            low_full, N_TOK) ||
        !ds4_gpu_attention_output_q8_batch_b_tensor(
            out0, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
            low_ref0, HALF_TOK) ||
        !ds4_gpu_attention_output_q8_batch_b_tensor(
            out1, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
            low_ref1, HALF_TOK) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(out_full, 0u, reference, out_bytes) ||
        !ds4_gpu_tensor_read(out_split, 0u, candidate, out_bytes)) {
        fprintf(stderr, "error: actual-A-low output-B boundary runtime failed\n");
        goto cleanup;
    }
    const diff_metrics out_b_actual_diff = compare_f32(
        reference, candidate, out_count);
    const diff_metrics out_b_full_chain_diff = compare_f32(
        combined_full_host, reference, out_count);
    const diff_metrics out_b_split_chain_diff = compare_f32(
        combined_split_host, candidate, out_count);
    memcpy(shipping_b_host, reference, (size_t)out_bytes);
    report_diff("output-b-from-identical-actual-a-low-full512-vs-row256x2",
                "f32", out_count, out_b_actual_diff);
    report_diff("output-a-plus-b-full512-vs-synchronized-b-full512", "f32",
                out_count, out_b_full_chain_diff);
    report_diff("output-a-plus-b-row256x2-vs-synchronized-b-row256x2", "f32",
                out_count, out_b_split_chain_diff);

    if (!output_b_working_set_replay_diagnostic) {
        int exact_b_algorithm = -1;
        int exact_b_algorithms[sizeof(b_algorithms) / sizeof(b_algorithms[0])];
        size_t exact_b_algorithm_count = 0u;
        for (size_t ai = 0u;
             ai < sizeof(b_algorithms) / sizeof(b_algorithms[0]); ai++) {
            const int algorithm = b_algorithms[ai];
            char text[32];
            snprintf(text, sizeof(text), "%d", algorithm);
            (void)setenv("DS4_CUDA_ATTN_OUTPUT_B_F16_GEMM_ALGO_DIAGNOSTIC",
                         text, 1);
            if (!ds4_gpu_attention_output_q8_batch_b_tensor(
                    out_full, model, model_bytes, out_b_offset,
                    LOW_DIM, OUT_DIM, low_full, N_TOK)) {
                printf("b_algorithm=%d,status=full-unsupported\n", algorithm);
                continue;
            }
            if (!ds4_gpu_attention_output_q8_batch_b_tensor(
                    out0, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
                    low_ref0, HALF_TOK) ||
                !ds4_gpu_attention_output_q8_batch_b_tensor(
                    out1, model, model_bytes, out_b_offset, LOW_DIM, OUT_DIM,
                    low_ref1, HALF_TOK)) {
                printf("b_algorithm=%d,status=row256-unsupported\n",
                       algorithm);
                continue;
            }
            if (!ds4_gpu_synchronize() ||
                !ds4_gpu_tensor_read(out_full, 0u, reference, out_bytes) ||
                !ds4_gpu_tensor_read(out_split, 0u, candidate, out_bytes)) {
                fprintf(stderr, "error: B algorithm %d readback failed\n",
                        algorithm);
                goto cleanup;
            }
            const diff_metrics full_vs_split = compare_f32(
                reference, candidate, out_count);
            const diff_metrics shipping_vs_full = compare_f32(
                shipping_b_host, reference, out_count);
            const diff_metrics shipping_vs_split = compare_f32(
                shipping_b_host, candidate, out_count);
            printf("b_algorithm=%d,status=ok,"
                   "full_vs_row256x2_mismatches=%llu,"
                   "full_vs_row256x2_max_abs=%.9g,"
                   "shipping_vs_full_mismatches=%llu,"
                   "shipping_vs_full_max_abs=%.9g,"
                   "shipping_vs_row256x2_mismatches=%llu,"
                   "shipping_vs_row256x2_max_abs=%.9g\n",
                   algorithm,
                   (unsigned long long)full_vs_split.mismatches,
                   full_vs_split.max_abs,
                   (unsigned long long)shipping_vs_full.mismatches,
                   shipping_vs_full.max_abs,
                   (unsigned long long)shipping_vs_split.mismatches,
                   shipping_vs_split.max_abs);
            if (full_vs_split.mismatches == 0u &&
                shipping_vs_full.mismatches == 0u &&
                shipping_vs_split.mismatches == 0u) {
                if (exact_b_algorithm < 0) exact_b_algorithm = algorithm;
                exact_b_algorithms[exact_b_algorithm_count++] = algorithm;
            }
        }
        (void)unsetenv("DS4_CUDA_ATTN_OUTPUT_B_F16_GEMM_ALGO_DIAGNOSTIC");
        printf("first_shipping_exact_b_algorithm=%d\n", exact_b_algorithm);
        printf("b_algorithm_conclusion=%s\n",
               exact_b_algorithm < 0 ?
                   "no-legacy-algorithm-preserved-shipping-output" :
                   "explicit-algorithm-preserved-shipping-output");

        const uint32_t timing_rounds = positive_env_u32(
            "B_TIMING_ROUNDS", 7u, 99u);
        const uint32_t timing_repeats = positive_env_u32(
            "B_TIMING_REPEATS", 10u, 1000u);
        const uint32_t timing_warmups = positive_env_u32(
            "B_TIMING_WARMUPS", 3u, 100u);
        if (!timing_rounds || !timing_repeats || !timing_warmups)
            goto cleanup;
        printf("b_timing_scope=bounded-host-wall-clock-batched-launches\n"
               "b_timing_rounds=%u\nb_timing_repeats=%u\n"
               "b_timing_warmups=%u\n",
               timing_rounds, timing_repeats, timing_warmups);
        double shipping_full_ms = 0.0, shipping_half_ms = 0.0;
        if (!time_output_b(&shipping_full_ms, out_full, model, model_bytes,
                           out_b_offset, low_full, N_TOK, -1, timing_rounds,
                           timing_repeats, timing_warmups) ||
            !time_output_b(&shipping_half_ms, out0, model, model_bytes,
                           out_b_offset, low_ref0, HALF_TOK, -1, timing_rounds,
                           timing_repeats, timing_warmups)) {
            fprintf(stderr, "error: shipping output-B timing failed\n");
            goto cleanup;
        }
        printf("b_timing=shipping-default,rows=%u,median_ms=%.9g\n",
               N_TOK, shipping_full_ms);
        printf("b_timing=shipping-default,rows=%u,median_ms=%.9g\n",
               HALF_TOK, shipping_half_ms);

        int fastest_exact_b_algorithm = -1;
        double fastest_exact_b_half_ms = HUGE_VAL;
        for (size_t i = 0u; i < exact_b_algorithm_count; i++) {
            const int algorithm = exact_b_algorithms[i];
            double full_ms = 0.0, half_ms = 0.0;
            if (!time_output_b(&full_ms, out_full, model, model_bytes,
                               out_b_offset, low_full, N_TOK, algorithm,
                               timing_rounds, timing_repeats, timing_warmups) ||
                !time_output_b(&half_ms, out0, model, model_bytes,
                               out_b_offset, low_ref0, HALF_TOK, algorithm,
                               timing_rounds, timing_repeats, timing_warmups)) {
                fprintf(stderr,
                        "error: exact output-B algorithm %d timing failed\n",
                        algorithm);
                goto cleanup;
            }
            printf("b_timing=explicit,algorithm=%d,rows=%u,median_ms=%.9g,"
                   "shipping_full_over_arm=%.9g\n",
                   algorithm, N_TOK, full_ms, shipping_full_ms / full_ms);
            printf("b_timing=explicit,algorithm=%d,rows=%u,median_ms=%.9g,"
                   "shipping_full_over_parallel_half_envelope=%.9g\n",
                   algorithm, HALF_TOK, half_ms,
                   shipping_full_ms / half_ms);
            if (half_ms < fastest_exact_b_half_ms) {
                fastest_exact_b_algorithm = algorithm;
                fastest_exact_b_half_ms = half_ms;
            }
        }
        select_b_algorithm(-1);
        printf("fastest_shipping_exact_b_algorithm=%d\n",
               fastest_exact_b_algorithm);
        printf("fastest_shipping_exact_b_half_median_ms=%.9g\n",
               fastest_exact_b_half_ms);
        printf("fastest_shipping_exact_b_projected_parallel_speedup=%.9g\n",
               fastest_exact_b_algorithm < 0 ? 0.0 :
                   shipping_full_ms / fastest_exact_b_half_ms);
    } else {
        if (output_b_production103_burnin_diagnostic) {
            select_b_algorithm(103);
            printf("production103_replay_phase=burnin,event=submit,"
                   "calls=0,total=%u\n", output_b_production103_calls);
            fflush(stdout);
            for (uint32_t call = 0u;
                 call < output_b_production103_calls; call++) {
                if (!ds4_gpu_attention_output_q8_batch_b_tensor(
                        out0, model, model_bytes, out_b_offset,
                        LOW_DIM, OUT_DIM, low_ref0, HALF_TOK)) {
                    fprintf(stderr,
                            "error: production-103 output-B burn-in launch "
                            "%u failed\n", call + 1u);
                    goto cleanup;
                }
                const uint32_t completed = call + 1u;
                if (completed % output_b_production103_batch == 0u ||
                    completed == output_b_production103_calls) {
                    if (!ds4_gpu_synchronize()) {
                        fprintf(stderr,
                                "error: production-103 output-B burn-in "
                                "synchronize failed after %u calls\n",
                                completed);
                        goto cleanup;
                    }
                    printf("production103_replay_phase=burnin,event=complete,"
                           "calls=%u,total=%u\n", completed,
                           output_b_production103_calls);
                    fflush(stdout);
                }
            }
            if (!ds4_gpu_tensor_read(
                    out0, 0u, candidate, out_row_half_bytes)) {
                fprintf(stderr,
                        "error: production-103 output-B burn-in readback failed\n");
                goto cleanup;
            }
            const diff_metrics production103_diff = compare_f32(
                shipping_b_host, candidate, out_count / 2u);
            report_diff("output-b-production103-burnin-vs-shipping-half0",
                        "f32", out_count / 2u, production103_diff);
            if (production103_diff.mismatches) {
                fprintf(stderr,
                        "error: production-103 output-B burn-in diverged\n");
                goto cleanup;
            }
            printf("production103_replay_conclusion=burnin-clean\n");
            fflush(stdout);
        }
        select_b_algorithm(-1);
        printf("suffix_replay_phase=preconditioning-complete\n"
               "suffix_replay_phase=production-row-owned-pair-submit\n");
        fflush(stdout);
    }

    /* Exercise the dedicated production-candidate entry point itself.  Its
     * two calls must reproduce both the already-exact A result and shipping
     * DEFAULT's full-row A+B result; testing only the diagnostic environment
     * selector would not prove that the graph-facing helper is wired safely. */
    if (!ds4_gpu_attention_output_q8_batch_row_owned_sm75_tensor(
            out0, low0, NULL, NULL, model, model_bytes, out_a_offset,
            out_b_offset, GROUP_DIM, RANK, N_GROUP, OUT_DIM, heads_ref0,
            HALF_TOK) ||
        !ds4_gpu_attention_output_q8_batch_row_owned_sm75_tensor(
            out1, low1, NULL, NULL, model, model_bytes, out_a_offset,
            out_b_offset, GROUP_DIM, RANK, N_GROUP, OUT_DIM, heads_ref1,
            HALF_TOK) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(low_split, 0u, candidate, low_bytes)) {
        fprintf(stderr, "error: dedicated row-owned output helper failed\n");
        goto cleanup;
    }
    const diff_metrics row_owned_low_diff = compare_f32(
        actual_low_host, candidate, low_count);
    report_diff("output-a-shipping-full512-vs-row-owned-algo3-256x2",
                "f32", low_count, row_owned_low_diff);
    if (!ds4_gpu_tensor_read(out_split, 0u, candidate, out_bytes)) {
        fprintf(stderr, "error: dedicated row-owned output readback failed\n");
        goto cleanup;
    }
    const diff_metrics row_owned_output_diff = compare_f32(
        shipping_b_host, candidate, out_count);
    report_diff("output-a-plus-b-shipping-full512-vs-row-owned-algo3-256x2",
                "f32", out_count, row_owned_output_diff);
    if (row_owned_low_diff.mismatches || row_owned_output_diff.mismatches) {
        fprintf(stderr,
                "error: dedicated row-owned output helper failed exactness\n");
        goto cleanup;
    }
    if (output_b_working_set_replay_diagnostic) {
        printf("suffix_replay_phase=production-row-owned-pair-complete\n"
               "suffix_replay_phase=%s\n",
               output_b_production103_pinned_half_diagnostic ?
                   "default-full-pinned103-split-group-submit" :
                   "default-full-split-group-submit");
        fflush(stdout);
    }

    /* Keep the original structured B-only fixture as a control.  Its exactness
     * is not sufficient to clear B for arbitrary A-produced inputs. */
    for (uint64_t i = 0u; i < low_count; i++) {
        const int value = (int)((i * 43u + (i >> 6u) * 17u + 61u) % 509u) - 254;
        actual_low_host[i] = (float)value / 4096.0f;
    }
    if (!ds4_gpu_tensor_write(low_full, 0u, actual_low_host, low_bytes) ||
        !ds4_gpu_tensor_write(low_split, 0u, actual_low_host, low_bytes)) {
        fprintf(stderr, "error: isolated output-B input reset failed\n");
        goto cleanup;
    }
    if (output_b_production103_burnin_diagnostic) {
        printf("suffix_transition_phase=default-full512-submit\n");
        fflush(stdout);
        if (!ds4_gpu_attention_output_q8_batch_b_tensor(
                out_full, model, model_bytes, out_b_offset,
                LOW_DIM, OUT_DIM, low_full, N_TOK) ||
            !ds4_gpu_synchronize()) {
            fprintf(stderr,
                    "error: production-103 to DEFAULT full512 transition "
                    "failed\n");
            goto cleanup;
        }
        printf("suffix_transition_phase=default-full512-complete\n"
               "suffix_transition_phase=%s-half0-256-submit\n",
               output_b_production103_pinned_half_diagnostic ?
                   "algo103" : "default");
        fflush(stdout);
        if (output_b_production103_pinned_half_diagnostic) {
            select_b_algorithm(103);
        }
        if (!ds4_gpu_attention_output_q8_batch_b_tensor(
                out0, model, model_bytes, out_b_offset,
                LOW_DIM, OUT_DIM, low_ref0, HALF_TOK) ||
            !ds4_gpu_synchronize()) {
            fprintf(stderr,
                    "error: production-103 to %s half0-256 transition "
                    "failed\n",
                    output_b_production103_pinned_half_diagnostic ?
                        "algorithm-103" : "DEFAULT");
            goto cleanup;
        }
        printf("suffix_transition_phase=%s-half0-256-complete\n"
               "suffix_transition_phase=%s-half1-256-submit\n",
               output_b_production103_pinned_half_diagnostic ?
                   "algo103" : "default",
               output_b_production103_pinned_half_diagnostic ?
                   "algo103" : "default");
        fflush(stdout);
        if (!ds4_gpu_attention_output_q8_batch_b_tensor(
                out1, model, model_bytes, out_b_offset,
                LOW_DIM, OUT_DIM, low_ref1, HALF_TOK) ||
            !ds4_gpu_synchronize()) {
            fprintf(stderr,
                    "error: production-103 to %s half1-256 transition "
                    "failed\n",
                    output_b_production103_pinned_half_diagnostic ?
                        "algorithm-103" : "DEFAULT");
            goto cleanup;
        }
        printf("suffix_transition_phase=%s-half1-256-complete\n",
               output_b_production103_pinned_half_diagnostic ?
                   "algo103" : "default");
        fflush(stdout);
    } else if (!ds4_gpu_attention_output_q8_batch_b_tensor(
                   out_full, model, model_bytes, out_b_offset,
                   LOW_DIM, OUT_DIM, low_full, N_TOK) ||
               !ds4_gpu_attention_output_q8_batch_b_tensor(
                   out0, model, model_bytes, out_b_offset,
                   LOW_DIM, OUT_DIM, low_ref0, HALF_TOK) ||
               !ds4_gpu_attention_output_q8_batch_b_tensor(
                   out1, model, model_bytes, out_b_offset,
                   LOW_DIM, OUT_DIM, low_ref1, HALF_TOK) ||
               !ds4_gpu_synchronize()) {
        fprintf(stderr, "error: isolated output-B boundary runtime failed\n");
        goto cleanup;
    }
    if (!ds4_gpu_tensor_read(out_full, 0u, reference, out_bytes) ||
        !ds4_gpu_tensor_read(out_split, 0u, candidate, out_bytes)) {
        fprintf(stderr, "error: isolated output-B boundary runtime failed\n");
        goto cleanup;
    }
    const diff_metrics out_b_isolated_diff = compare_f32(
        reference, candidate, out_count);
    report_diff("output-b-structured-control-full512-vs-row256x2", "f32",
                out_count, out_b_isolated_diff);
    if (output_b_working_set_replay_diagnostic) {
        printf("suffix_replay_phase=%s\n",
               output_b_production103_pinned_half_diagnostic ?
                   "default-full-pinned103-split-group-complete" :
                   "default-full-split-group-complete");
        const int suffix_replay_ok =
            qh_diff.mismatches == 0u && q_diff.mismatches == 0u &&
            q_scratch_diff.mismatches == 0u &&
            attention_wrapper_diff.mismatches == 0u &&
            attention_extent_diff.mismatches == 0u &&
            rope_diff.mismatches == 0u && out_a_diff.mismatches == 0u &&
            out_b_full_chain_diff.mismatches == 0u &&
            out_b_split_chain_diff.mismatches == 0u &&
            row_owned_low_diff.mismatches == 0u &&
            row_owned_output_diff.mismatches == 0u &&
            out_b_isolated_diff.mismatches == 0u;
        printf("suffix_replay_conclusion=%s\n",
               suffix_replay_ok ?
                   (output_b_production103_burnin_diagnostic ?
                       (output_b_production103_pinned_half_diagnostic ?
                           "working-set-production103-default512-"
                           "pinned103-halves-clean" :
                           "working-set-production103-burnin-and-suffix-clean") :
                       "working-set-and-suffix-clean-without-burn-in") :
                   "unexpected-boundary-divergence");
        fflush(stdout);
        if (!suffix_replay_ok) {
            fprintf(stderr,
                    "error: canonical output-B suffix replay diverged\n");
            goto cleanup;
        }
    }

    printf("diagnostic_conclusion=%s\n",
           qh_diff.mismatches ? "first-divergence-q-b-f16-projection" :
           q_diff.mismatches ? "first-divergence-q-b-postprocess" :
           q_scratch_diff.mismatches ?
               "first-divergence-q-b-internal-scratch-shape" :
           attention_wrapper_diff.mismatches ?
               "first-divergence-static-attention-wrapper" :
           attention_extent_diff.mismatches ?
               "first-divergence-static-attention-row-extent" :
           rope_diff.mismatches ? "first-divergence-inverse-rope" :
           out_a_diff.mismatches ? "first-divergence-output-a" :
           row_owned_low_diff.mismatches ?
               "dedicated-row-owned-helper-diverged-at-output-a" :
           row_owned_output_diff.mismatches ?
               "dedicated-row-owned-helper-diverged-at-output-b" :
           out_b_full_chain_diff.mismatches || out_b_split_chain_diff.mismatches ?
               "first-divergence-output-a-to-b-dependency" :
           out_b_actual_diff.mismatches ?
               "first-divergence-output-b-row-extent" :
           out_b_diff.mismatches ? "combined-output-diverged-only" :
           out_b_isolated_diff.mismatches ?
               "structured-output-b-control-diverged" :
           "all-tested-row-arithmetic-boundaries-bit-exact");
    printf("harness_status=ok\n");
    status = 0;

cleanup:
    (void)unsetenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE");
    (void)unsetenv("DS4_CUDA_TP_PREFILL_ATTN_TOKEN_ROWS_WEIGHT_MODE");
    (void)unsetenv("DS4_CUDA_TP_PREFILL_ATTN_TOKEN_ROWS_PIPELINE_PAIRS");
    (void)unsetenv("DS4_CUDA_TOKEN_ROWS_NATIVE_STREAM_LOCAL_DIAGNOSTIC");
    (void)unsetenv(
        "DS4_CUDA_TOKEN_ROWS_NATIVE_STREAM_LOCAL_NO_CHECKPOINT");
    (void)unsetenv("DS4_CUDA_OUTPUT_A_LOCAL_DIAGNOSTIC");
    (void)unsetenv("DS4_CUDA_OUTPUT_A_ONLY_LOCAL_DIAGNOSTIC");
    (void)unsetenv("DS4_CUDA_OUTPUT_B_LOCAL_DIAGNOSTIC");
    (void)unsetenv("DS4_CUDA_ATTN_OUTPUT_B_F16_GEMM_ALGO_DIAGNOSTIC");
    (void)unsetenv("DS4_CUDA_Q8_NATIVE_REBASE_AUDIT");
    ds4_gpu_tensor_free(low_ref1);
    ds4_gpu_tensor_free(low_ref0);
    ds4_gpu_tensor_free(out1);
    ds4_gpu_tensor_free(out0);
    ds4_gpu_tensor_free(low1);
    ds4_gpu_tensor_free(low0);
    ds4_gpu_tensor_free(heads_ref1);
    ds4_gpu_tensor_free(heads_ref0);
    ds4_gpu_tensor_free(heads1);
    ds4_gpu_tensor_free(heads0);
    ds4_gpu_tensor_free(qh1);
    ds4_gpu_tensor_free(qh0);
    ds4_gpu_tensor_free(q_ref1);
    ds4_gpu_tensor_free(q_ref0);
    ds4_gpu_tensor_free(q1);
    ds4_gpu_tensor_free(q0);
    ds4_gpu_tensor_free(input1);
    ds4_gpu_tensor_free(input0);
    ds4_gpu_tensor_free(out_split);
    ds4_gpu_tensor_free(out_full);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(low_split);
    ds4_gpu_tensor_free(low_full);
    ds4_gpu_tensor_free(heads_split);
    ds4_gpu_tensor_free(heads_full);
    ds4_gpu_tensor_free(qh_split);
    ds4_gpu_tensor_free(qh_full);
    ds4_gpu_tensor_free(q_split);
    ds4_gpu_tensor_free(q_full);
    ds4_gpu_tensor_free(input);
    if (initialized) ds4_gpu_cleanup();
    free(candidate_half);
    free(reference_half);
    free(comp_host);
    free(raw_host);
    free(candidate);
    free(reference);
    free(input_host);
    free(model);
    return status;
}
