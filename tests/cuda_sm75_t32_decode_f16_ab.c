#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A bounded, one-device diagnostic for the per-rank decode q_b shape.  T32
 * means 1024 / 32 Q8 blocks, not 32 tokens.  Nothing in this executable is
 * selected by the production engine. */
#define T32_IN_DIM 1024u
#define T32_N_HEAD 32u
#define T32_HEAD_DIM 512u
#define T32_OUT_DIM ((uint64_t)T32_N_HEAD * T32_HEAD_DIM)
#define T32_N_ROT 64u
#define T32_POS 32768u
#define T32_ORIG_CTX 65536u
#define T32_ROPE_BASE 160000.0f
#define T32_ROPE_SCALE 0.0625f
#define T32_ROPE_EXT 1.0f
#define T32_BETA_FAST 32.0f
#define T32_BETA_SLOW 1.0f
#define T32_EPS 1.0e-6f
#define T32_Q8_SCALE_BITS 0x2400u /* binary16 1/64 */

enum t32_arm {
    T32_ARM_NATIVE_F32 = 0,
    T32_ARM_F16_COMPUTE_F32 = 1,
    T32_ARM_F16_COMPUTE_F16 = 2,
    T32_ARM_COUNT = 3,
};

static const char *const arm_names[T32_ARM_COUNT] = {
    "native-q8-f32",
    "f16-compute-f32-output",
    "f16-compute-f16-output",
};

typedef struct {
    const unsigned char *model;
    uint64_t model_bytes;
    uint64_t f16_offset;
    ds4_gpu_tensor *input;
    ds4_gpu_tensor *output[T32_ARM_COUNT];
    ds4_gpu_tensor *q_half;
} t32_context;

typedef struct {
    double max_abs;
    double rmse;
    double rel_l2;
    double cosine;
    uint64_t bit_mismatches;
} comparison_metrics;

static uint32_t positive_env_u32(const char *name, uint32_t fallback) {
    const char *text = getenv(name);
    if (!text || !text[0]) return fallback;
    char *end = NULL;
    const unsigned long value = strtoul(text, &end, 10);
    if (end == text || *end || value == 0u || value > UINT32_MAX)
        return fallback;
    return (uint32_t)value;
}

static int compare_float(const void *lhs, const void *rhs) {
    const float a = *(const float *)lhs;
    const float b = *(const float *)rhs;
    return (a > b) - (a < b);
}

/* Round-to-nearest binary32 -> binary16.  The generated fixture uses only
 * exactly representable values; the complete conversion keeps the test easy
 * to extend without depending on CUDA C++ half helpers. */
static uint16_t float_to_half_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16u) & 0x8000u;
    const uint32_t exponent = (bits >> 23u) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;

    if (exponent == 0xffu) {
        if (mantissa == 0u) return (uint16_t)(sign | 0x7c00u);
        return (uint16_t)(sign | 0x7e00u);
    }

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

static float half_bits_to_float(uint16_t half) {
    const uint32_t sign = (uint32_t)(half & 0x8000u) << 16u;
    uint32_t exponent = (half >> 10u) & 0x1fu;
    uint32_t mantissa = half & 0x03ffu;
    uint32_t bits;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            bits = sign;
        } else {
            uint32_t e = 113u;
            while ((mantissa & 0x0400u) == 0u) {
                mantissa <<= 1u;
                e--;
            }
            bits = sign | (e << 23u) | ((mantissa & 0x03ffu) << 13u);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        bits = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int8_t fixture_q(uint64_t row, uint64_t column) {
    return (int8_t)((int)((row * 17u + column * 5u +
                           (row >> 3u) * 7u + 11u) % 15u) - 7);
}

static int build_model(unsigned char **model_out, uint64_t *model_bytes_out,
                       uint64_t *q8_bytes_out, uint64_t *f16_offset_out) {
    const uint64_t blocks = T32_IN_DIM / 32u;
    const uint64_t q8_bytes = T32_OUT_DIM * blocks * 34u;
    const uint64_t f16_offset = (q8_bytes + 255u) & ~UINT64_C(255);
    const uint64_t f16_bytes = T32_OUT_DIM * T32_IN_DIM * sizeof(uint16_t);
    if (f16_offset > UINT64_MAX - f16_bytes ||
        f16_offset + f16_bytes > (uint64_t)SIZE_MAX) {
        return 0;
    }
    const uint64_t model_bytes = f16_offset + f16_bytes;
    unsigned char *model = (unsigned char *)calloc(1u, (size_t)model_bytes);
    if (!model) return 0;
    uint16_t *expanded = (uint16_t *)(model + f16_offset);

    for (uint64_t row = 0u; row < T32_OUT_DIM; row++) {
        for (uint64_t block = 0u; block < blocks; block++) {
            unsigned char *packed =
                model + (row * blocks + block) * 34u;
            packed[0] = (unsigned char)(T32_Q8_SCALE_BITS & 0xffu);
            packed[1] = (unsigned char)(T32_Q8_SCALE_BITS >> 8u);
            for (uint64_t lane = 0u; lane < 32u; lane++) {
                const uint64_t column = block * 32u + lane;
                const int8_t q = fixture_q(row, column);
                packed[2u + lane] = (unsigned char)q;
                expanded[row * T32_IN_DIM + column] =
                    float_to_half_bits((float)q * (1.0f / 64.0f));
            }
        }
    }
    *model_out = model;
    *model_bytes_out = model_bytes;
    *q8_bytes_out = q8_bytes;
    *f16_offset_out = f16_offset;
    return 1;
}

static float rope_attention_factor(void) {
    return 1.0f /
        (1.0f + 0.1f * logf(1.0f / T32_ROPE_SCALE));
}

static int postprocess_f32(ds4_gpu_tensor *tensor) {
    return ds4_gpu_head_rms_norm_rope_tail_tensor(
        tensor, 1u, T32_N_HEAD, T32_HEAD_DIM, T32_N_ROT,
        T32_POS, T32_ORIG_CTX, false,
        T32_ROPE_BASE, T32_ROPE_SCALE, T32_ROPE_EXT,
        rope_attention_factor(), T32_BETA_FAST, T32_BETA_SLOW, T32_EPS);
}

static int launch_arm(const t32_context *context, enum t32_arm arm) {
    switch (arm) {
        case T32_ARM_NATIVE_F32:
            return ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                       context->output[arm], context->model,
                       context->model_bytes, 0u, T32_IN_DIM, T32_OUT_DIM,
                       context->input, 1u) &&
                   postprocess_f32(context->output[arm]);
        case T32_ARM_F16_COMPUTE_F32:
            return ds4_gpu_matmul_f16_tensor(
                       context->output[arm], context->model,
                       context->model_bytes, context->f16_offset,
                       T32_IN_DIM, T32_OUT_DIM, context->input, 1u) &&
                   postprocess_f32(context->output[arm]);
        case T32_ARM_F16_COMPUTE_F16:
            return ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
                context->output[arm], context->q_half,
                context->model, context->model_bytes, 0u,
                T32_IN_DIM, T32_OUT_DIM, context->input,
                1u, T32_N_HEAD, T32_HEAD_DIM, T32_N_ROT,
                T32_POS, T32_ORIG_CTX, false,
                T32_ROPE_BASE, T32_ROPE_SCALE, T32_ROPE_EXT,
                rope_attention_factor(), T32_BETA_FAST, T32_BETA_SLOW,
                T32_EPS);
        default:
            return 0;
    }
}

static int all_finite_nonzero(const float *values, uint64_t count,
                              const char *label) {
    uint64_t nonzero = 0u;
    for (uint64_t i = 0u; i < count; i++) {
        if (!isfinite(values[i])) {
            fprintf(stderr, "error: %s output[%llu] is non-finite\n",
                    label, (unsigned long long)i);
            return 0;
        }
        nonzero += values[i] != 0.0f;
    }
    if (nonzero == 0u) {
        fprintf(stderr, "error: %s output is entirely zero\n", label);
        return 0;
    }
    printf("validation,arm=%s,finite=%llu,nonzero=%llu\n",
           label, (unsigned long long)count, (unsigned long long)nonzero);
    return 1;
}

static comparison_metrics compare_outputs(const float *reference,
                                          const float *candidate,
                                          uint64_t count) {
    comparison_metrics result;
    memset(&result, 0, sizeof(result));
    double square_error = 0.0;
    double square_reference = 0.0;
    double square_candidate = 0.0;
    double dot = 0.0;
    for (uint64_t i = 0u; i < count; i++) {
        const double r = reference[i];
        const double c = candidate[i];
        const double error = c - r;
        const double abs_error = fabs(error);
        if (abs_error > result.max_abs) result.max_abs = abs_error;
        square_error += error * error;
        square_reference += r * r;
        square_candidate += c * c;
        dot += r * c;
        uint32_t rb, cb;
        memcpy(&rb, reference + i, sizeof(rb));
        memcpy(&cb, candidate + i, sizeof(cb));
        result.bit_mismatches += rb != cb;
    }
    result.rmse = sqrt(square_error / (double)count);
    result.rel_l2 = square_reference > 0.0
        ? sqrt(square_error / square_reference) : 0.0;
    result.cosine = square_reference > 0.0 && square_candidate > 0.0
        ? dot / sqrt(square_reference * square_candidate) : 0.0;
    return result;
}

static void print_comparison(const char *reference_name,
                             const char *candidate_name,
                             comparison_metrics metrics) {
    printf("comparison,reference=%s,candidate=%s,max_abs=%.9g,rmse=%.9g,"
           "relative_l2=%.9g,cosine=%.12g,bit_mismatches=%llu\n",
           reference_name, candidate_name, metrics.max_abs, metrics.rmse,
           metrics.rel_l2, metrics.cosine,
           (unsigned long long)metrics.bit_mismatches);
}

static int validate_outputs(t32_context *context, float **host_output) {
    const uint64_t output_bytes = T32_OUT_DIM * sizeof(float);
    float *repeat = (float *)malloc((size_t)output_bytes);
    float *raw_f16_f32 = (float *)malloc((size_t)output_bytes);
    float *half_widened = (float *)malloc((size_t)output_bytes);
    uint16_t *half =
        (uint16_t *)malloc((size_t)T32_OUT_DIM * sizeof(uint16_t));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(output_bytes);
    ds4_gpu_tensor *half_reference = ds4_gpu_tensor_alloc(output_bytes);
    int ok = repeat && raw_f16_f32 && half_widened && half && raw &&
             half_reference;
    if (!ok) {
        fprintf(stderr, "error: validation allocation failed\n");
        goto cleanup;
    }

    for (int arm = 0; arm < T32_ARM_COUNT; arm++) {
        if (!launch_arm(context, (enum t32_arm)arm) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(context->output[arm], 0u,
                                 host_output[arm], output_bytes) ||
            !all_finite_nonzero(host_output[arm], T32_OUT_DIM,
                                arm_names[arm])) {
            fprintf(stderr, "error: %s validation launch failed\n",
                    arm_names[arm]);
            ok = 0;
            goto cleanup;
        }
    }

    /* Each arm must be deterministic before any cross-arm numerical drift is
     * interpreted.  Run in the same order, leaving the FP16 intermediate from
     * the candidate arm available for the storage-boundary check below. */
    for (int arm = 0; arm < T32_ARM_COUNT; arm++) {
        if (!launch_arm(context, (enum t32_arm)arm) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(context->output[arm], 0u,
                                 repeat, output_bytes) ||
            memcmp(repeat, host_output[arm], (size_t)output_bytes) != 0) {
            fprintf(stderr, "error: %s is not byte-deterministic\n",
                    arm_names[arm]);
            ok = 0;
            goto cleanup;
        }
    }
    printf("determinism=byte-exact-all-arms\n");

    /* Test whether this cuBLAS build lets the output-storage axis be isolated.
     * Both calls use identical F16 inputs/weights, but CUDA does not promise
     * that changing C from F32 to F16 preserves GEMM algorithm or reduction
     * order.  A mismatch is therefore reported, not rejected: when nonzero,
     * the B/C timing includes an output-type-dependent compute change and is
     * not evidence for storage alone. */
    if (!ds4_gpu_matmul_f16_tensor(
            raw, context->model, context->model_bytes,
            context->f16_offset, T32_IN_DIM, T32_OUT_DIM,
            context->input, 1u) ||
        !ds4_gpu_tensor_read(raw, 0u, raw_f16_f32, output_bytes) ||
        !ds4_gpu_tensor_read(context->q_half, 0u, half,
                             T32_OUT_DIM * sizeof(uint16_t))) {
        fprintf(stderr, "error: projection storage-boundary read failed\n");
        ok = 0;
        goto cleanup;
    }
    uint64_t half_mismatches = 0u;
    double half_max_abs = 0.0;
    for (uint64_t i = 0u; i < T32_OUT_DIM; i++) {
        const uint16_t rounded = float_to_half_bits(raw_f16_f32[i]);
        half_mismatches += rounded != half[i];
        half_widened[i] = half_bits_to_float(half[i]);
        const double error = fabs((double)half_widened[i] - raw_f16_f32[i]);
        if (error > half_max_abs) half_max_abs = error;
    }
    printf("projection_storage,reference=f16-compute-f32-output,"
           "candidate=f16-compute-f16-output,rounded_half_mismatches=%llu,"
           "widened_max_abs=%.9g\n",
           (unsigned long long)half_mismatches, half_max_abs);
    printf("projection_compute_equivalence=%s\n"
           "storage_axis_isolated=%u\n"
           "rounded_half_match_is_acceptance_gate=0\n",
           half_mismatches == 0u ? "f32-output-rounds-to-f16-output"
                                 : "output-type-changed-cublas-result",
           half_mismatches == 0u ? 1u : 0u);

    /* The half-input postprocess is accepted only if it is byte-identical to
     * the established F32 postprocess after exact half widening. */
    if (!ds4_gpu_tensor_write(half_reference, 0u, half_widened,
                              output_bytes) ||
        !postprocess_f32(half_reference) ||
        !ds4_gpu_tensor_read(half_reference, 0u, repeat, output_bytes) ||
        memcmp(repeat, host_output[T32_ARM_F16_COMPUTE_F16],
               (size_t)output_bytes) != 0) {
        fprintf(stderr,
                "error: FP16-output postprocess differs from exact-half "
                "widening plus the established F32 postprocess\n");
        ok = 0;
        goto cleanup;
    }
    printf("half_postprocess_validation=byte-exact\n");

    print_comparison(
        arm_names[T32_ARM_NATIVE_F32],
        arm_names[T32_ARM_F16_COMPUTE_F32],
        compare_outputs(host_output[T32_ARM_NATIVE_F32],
                        host_output[T32_ARM_F16_COMPUTE_F32], T32_OUT_DIM));
    print_comparison(
        arm_names[T32_ARM_F16_COMPUTE_F32],
        arm_names[T32_ARM_F16_COMPUTE_F16],
        compare_outputs(host_output[T32_ARM_F16_COMPUTE_F32],
                        host_output[T32_ARM_F16_COMPUTE_F16], T32_OUT_DIM));
    print_comparison(
        arm_names[T32_ARM_NATIVE_F32],
        arm_names[T32_ARM_F16_COMPUTE_F16],
        compare_outputs(host_output[T32_ARM_NATIVE_F32],
                        host_output[T32_ARM_F16_COMPUTE_F16], T32_OUT_DIM));

cleanup:
    ds4_gpu_tensor_free(half_reference);
    ds4_gpu_tensor_free(raw);
    free(half);
    free(half_widened);
    free(raw_f16_f32);
    free(repeat);
    return ok;
}

static int time_arms(const t32_context *context, uint32_t rounds,
                     uint32_t repeats, uint32_t warmups) {
    float *samples[T32_ARM_COUNT] = {NULL, NULL, NULL};
    ds4_gpu_timer *timer = NULL;
    int ok = 0;
    for (int arm = 0; arm < T32_ARM_COUNT; arm++) {
        samples[arm] = (float *)malloc((size_t)rounds * sizeof(float));
        if (!samples[arm]) goto cleanup;
    }
    timer = ds4_gpu_timer_create();
    if (!timer) goto cleanup;

    for (uint32_t warmup = 0u; warmup < warmups; warmup++) {
        for (int arm = 0; arm < T32_ARM_COUNT; arm++) {
            if (!launch_arm(context, (enum t32_arm)arm)) {
                fprintf(stderr, "error: %s warmup failed\n", arm_names[arm]);
                goto cleanup;
            }
        }
    }
    if (!ds4_gpu_synchronize()) goto cleanup;

    /* Rotate all three orders.  Each recorded sample includes projection,
     * activation conversion/quantization, output storage, and RMS/RoPE.  The
     * B/C ratio isolates storage only when validation printed
     * storage_axis_isolated=1. */
    for (uint32_t round = 0u; round < rounds; round++) {
        for (uint32_t slot = 0u; slot < T32_ARM_COUNT; slot++) {
            const enum t32_arm arm =
                (enum t32_arm)((round + slot) % T32_ARM_COUNT);
            if (!ds4_gpu_timer_record_start(timer)) goto cleanup;
            for (uint32_t repeat_index = 0u;
                 repeat_index < repeats; repeat_index++) {
                if (!launch_arm(context, arm)) {
                    fprintf(stderr, "error: %s timing launch failed\n",
                            arm_names[arm]);
                    goto cleanup;
                }
            }
            if (!ds4_gpu_timer_record_end(timer)) goto cleanup;
            float elapsed_ms = 0.0f;
            if (!ds4_gpu_timer_elapsed_ms(timer, &elapsed_ms)) goto cleanup;
            samples[arm][round] = elapsed_ms / (float)repeats;
        }
    }

    for (int arm = 0; arm < T32_ARM_COUNT; arm++) {
        qsort(samples[arm], rounds, sizeof(float), compare_float);
        printf("timing,arm=%s,rounds=%u,repeats=%u,median_ms=%.9g\n",
               arm_names[arm], rounds, repeats, samples[arm][rounds / 2u]);
    }
    printf("speedup,reference=%s,candidate=%s,value=%.9g\n",
           arm_names[T32_ARM_NATIVE_F32],
           arm_names[T32_ARM_F16_COMPUTE_F32],
           samples[T32_ARM_NATIVE_F32][rounds / 2u] /
               samples[T32_ARM_F16_COMPUTE_F32][rounds / 2u]);
    printf("speedup,reference=%s,candidate=%s,value=%.9g\n",
           arm_names[T32_ARM_F16_COMPUTE_F32],
           arm_names[T32_ARM_F16_COMPUTE_F16],
           samples[T32_ARM_F16_COMPUTE_F32][rounds / 2u] /
               samples[T32_ARM_F16_COMPUTE_F16][rounds / 2u]);
    printf("speedup,reference=%s,candidate=%s,value=%.9g\n",
           arm_names[T32_ARM_NATIVE_F32],
           arm_names[T32_ARM_F16_COMPUTE_F16],
           samples[T32_ARM_NATIVE_F32][rounds / 2u] /
               samples[T32_ARM_F16_COMPUTE_F16][rounds / 2u]);
    ok = 1;

cleanup:
    ds4_gpu_timer_free(timer);
    for (int arm = 0; arm < T32_ARM_COUNT; arm++) free(samples[arm]);
    return ok;
}

int main(void) {
    const uint32_t rounds_env =
        positive_env_u32("DS4_T32_DECODE_ROUNDS", 9u);
    const uint32_t rounds = (rounds_env & 1u) ? rounds_env : rounds_env + 1u;
    const uint32_t repeats =
        positive_env_u32("DS4_T32_DECODE_REPEATS", 100u);
    const uint32_t warmups =
        positive_env_u32("DS4_T32_DECODE_WARMUPS", 5u);
    unsigned char *model = NULL;
    uint64_t model_bytes = 0u;
    uint64_t q8_bytes = 0u;
    uint64_t f16_offset = 0u;
    float *host_output[T32_ARM_COUNT] = {NULL, NULL, NULL};
    t32_context context;
    memset(&context, 0, sizeof(context));
    int initialized = 0;
    int status = 1;

    if (!build_model(&model, &model_bytes, &q8_bytes, &f16_offset)) {
        fprintf(stderr, "error: host model construction failed\n");
        goto cleanup;
    }
    float input_host[T32_IN_DIM];
    for (uint32_t i = 0u; i < T32_IN_DIM; i++) {
        const int value = (int)((i * 13u + (i >> 3u) * 7u + 5u) % 33u) - 16;
        input_host[i] = (float)value * (1.0f / 16.0f);
    }

    /* Normalize every relevant switch.  The decode admission variable is
     * deliberately absent for the first eligibility assertion below. */
    (void)setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    (void)setenv("DS4_CUDA_Q8_F16_CACHE_MB", "64", 1);
    (void)setenv("DS4_CUDA_Q8_F16_CACHE_RESERVE_MB", "1", 1);
    (void)setenv("DS4_CUDA_NO_TF32", "1", 1);
    (void)setenv("DS4_CUDA_T32_F16_FUSED", "1", 1);
    (void)setenv("DS4_CUDA_F16_CUBLAS_ONE", "1", 1);
    (void)unsetenv("DS4_CUDA_NO_T32_F16_FUSED");
    (void)unsetenv("DS4_CUDA_NO_Q8_F16_CACHE");
    (void)unsetenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE");
    (void)unsetenv("DS4_CUDA_Q8_F32_PRELOAD");
    (void)unsetenv("DS4_CUDA_ATTN_Q_B_F32_CACHE");
    (void)unsetenv("DS4_CUDA_NO_Q8_DP4A");
    (void)unsetenv("DS4_CUDA_NO_F16_CUBLAS_ONE");
    (void)unsetenv("DS4_CUDA_SERIAL_F16_MATMUL");
    (void)unsetenv("DS4_CUDA_SERIAL_ROUTER");
    (void)unsetenv("DS4_CUDA_NO_ORDERED_F16_MATMUL");
    (void)unsetenv("DS4_CUDA_T32_F16_DECODE_PROBE");

    if (!ds4_gpu_init()) {
        fprintf(stderr, "error: CUDA backend initialization failed\n");
        goto cleanup;
    }
    initialized = 1;
    if (!ds4_gpu_set_model_map(model, model_bytes) ||
        !ds4_gpu_cache_q8_f16_range(model, model_bytes, 0u, q8_bytes,
                                     T32_IN_DIM, T32_OUT_DIM,
                                     "attn_q_b")) {
        fprintf(stderr, "error: model/cache installation failed\n");
        goto cleanup;
    }

    context.model = model;
    context.model_bytes = model_bytes;
    context.f16_offset = f16_offset;
    context.input = ds4_gpu_tensor_alloc(T32_IN_DIM * sizeof(float));
    for (int arm = 0; arm < T32_ARM_COUNT; arm++) {
        context.output[arm] =
            ds4_gpu_tensor_alloc(T32_OUT_DIM * sizeof(float));
        host_output[arm] =
            (float *)malloc((size_t)T32_OUT_DIM * sizeof(float));
    }
    context.q_half =
        ds4_gpu_tensor_alloc(T32_OUT_DIM * sizeof(uint16_t));
    if (!context.input || !context.q_half || !context.output[0] ||
        !context.output[1] || !context.output[2] || !host_output[0] ||
        !host_output[1] || !host_output[2] ||
        !ds4_gpu_tensor_write(context.input, 0u, input_host,
                              sizeof(input_host))) {
        fprintf(stderr, "error: tensor allocation/setup failed\n");
        goto cleanup;
    }

    /* Assert that production behavior remains unchanged: without the probe
     * variable, the prefill-only helper must still reject one token. */
    if (ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
            context.output[T32_ARM_F16_COMPUTE_F16], context.q_half,
            model, model_bytes, 0u, T32_IN_DIM, T32_OUT_DIM,
            context.input, 1u, T32_N_HEAD, T32_HEAD_DIM, T32_N_ROT,
            T32_POS, T32_ORIG_CTX, false,
            T32_ROPE_BASE, T32_ROPE_SCALE, T32_ROPE_EXT,
            rope_attention_factor(), T32_BETA_FAST, T32_BETA_SLOW,
            T32_EPS)) {
        fprintf(stderr,
                "error: one-token helper was admitted without the diagnostic "
                "switch\n");
        goto cleanup;
    }
    printf("production_default_n_tok1=rejected\n");
    (void)setenv("DS4_CUDA_T32_F16_DECODE_PROBE", "1", 1);

    printf("scenario=sm75-t32-single-token-f16-output-ab\n"
           "scope=bounded-single-device-production-rank-shape\n"
           "production_default_changed=0\n"
           "n_tokens=1\nin_dim=%u\nout_dim=%llu\n"
           "n_head=%u\nhead_dim=%u\nq8_blocks_per_row=%u\n"
           "position=%u\nq8_model_bytes=%llu\nf16_model_bytes=%llu\n"
           "projection_result_bytes_native_f32=%llu\n"
           "projection_result_bytes_f16_compute_f32=%llu\n"
           "projection_result_bytes_f16_compute_f16=%llu\n"
           "final_postprocess_bytes_each_arm=%llu\n",
           T32_IN_DIM, (unsigned long long)T32_OUT_DIM,
           T32_N_HEAD, T32_HEAD_DIM, T32_IN_DIM / 32u, T32_POS,
           (unsigned long long)q8_bytes,
           (unsigned long long)(T32_OUT_DIM * T32_IN_DIM * sizeof(uint16_t)),
           (unsigned long long)(T32_OUT_DIM * sizeof(float)),
           (unsigned long long)(T32_OUT_DIM * sizeof(float)),
           (unsigned long long)(T32_OUT_DIM * sizeof(uint16_t)),
           (unsigned long long)(T32_OUT_DIM * sizeof(float)));

    if (!validate_outputs(&context, host_output) ||
        !time_arms(&context, rounds, repeats, warmups)) {
        goto cleanup;
    }
    printf("q8_f16_cache_dispatch=validated-by-local-helper-success\n");
    printf("promotion_eligible=0\n"
           "promotion_requirement=four-gpu-production-ab-and-logit-exactness\n"
           "harness_status=ok\n");
    status = 0;

cleanup:
    ds4_gpu_tensor_free(context.q_half);
    for (int arm = 0; arm < T32_ARM_COUNT; arm++) {
        ds4_gpu_tensor_free(context.output[arm]);
        free(host_output[arm]);
    }
    ds4_gpu_tensor_free(context.input);
    if (initialized) ds4_gpu_cleanup();
    free(model);
    return status;
}
