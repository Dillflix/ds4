#include "quants.h"
#include "quants_cuda.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_input(float *x, float *weights, int64_t nrows, int64_t ncols) {
    static const float levels[4] = {1.0f, 3.0f, 5.0f, 7.0f};
    for (int64_t c = 0; c < ncols; c++) {
        weights[c] = 0.75f + 0.125f * (float)(c % 7);
    }
    for (int64_t r = 0; r < nrows; r++) {
        for (int64_t c = 0; c < ncols; c++) {
            const int group = (int)((c / 32) & 7);
            float v = levels[(c + 3 * r) & 3] * (0.125f + 0.03125f * group);
            if (((c / 3 + r) & 1) != 0) v = -v;
            x[r * ncols + c] = v;
        }
    }
}

static int test_type(ds4q_type type, int device,
                     const float *src, const float *weights,
                     int64_t nrows, int64_t ncols) {
    const size_t bytes = (size_t)nrows * ds4q_row_size(type, ncols);
    uint8_t *cpu = calloc(1, bytes);
    uint8_t *gpu = calloc(1, bytes);
    if (!cpu || !gpu) {
        fprintf(stderr, "allocation failed\n");
        free(cpu);
        free(gpu);
        return 1;
    }
    ds4q_quantize_init(type);
    size_t cpu_bytes = 0;
    if (type == DS4Q_TYPE_SM75_Q4_32 ||
        type == DS4Q_TYPE_SM75_Q3A4) {
        uint8_t *canonical = calloc(1, bytes);
        if (!canonical) {
            fprintf(stderr, "canonical allocation failed\n");
            free(cpu);
            free(gpu);
            return 1;
        }
        const size_t encoded = ds4q_quantize_chunk(
            type, src, canonical, 0, nrows, ncols, weights);
        cpu_bytes = encoded == bytes
            ? (type == DS4Q_TYPE_SM75_Q4_32
                ? ds4q_repack_sm75_q4_32(canonical, cpu, nrows, ncols)
                : ds4q_repack_sm75_q3a4(canonical, cpu, nrows, ncols))
            : 0;
        free(canonical);
    } else {
        cpu_bytes = ds4q_quantize_chunk(type, src, cpu, 0,
                                        nrows, ncols, weights);
    }
    char error[256];
    const size_t gpu_bytes = ds4q_cuda_quantize_chunk(type, src, gpu,
                                                       nrows, ncols, weights,
                                                       device, error, sizeof(error));
    if (cpu_bytes != bytes || gpu_bytes != bytes) {
        fprintf(stderr, "device %d %s size failure: cpu=%zu gpu=%zu expected=%zu (%s)\n",
                device, ds4q_type_name(type), cpu_bytes, gpu_bytes, bytes,
                error[0] ? error : "no CUDA detail");
        free(cpu);
        free(gpu);
        return 1;
    }
    size_t mismatch = 0;
    size_t first = bytes;
    for (size_t i = 0; i < bytes; i++) {
        if (cpu[i] != gpu[i]) {
            if (first == bytes) first = i;
            mismatch++;
        }
    }
    if (mismatch) {
        fprintf(stderr,
                "device %d %s byte mismatch: %zu/%zu bytes, first=%zu cpu=%02x gpu=%02x\n",
                device, ds4q_type_name(type), mismatch, bytes, first,
                cpu[first], gpu[first]);
        free(cpu);
        free(gpu);
        return 1;
    }
    fprintf(stderr, "device %d %s CPU/CUDA byte check: OK (%zu bytes)\n",
            device, ds4q_type_name(type), bytes);
    free(cpu);
    free(gpu);
    return 0;
}

static void reference_sm75_q4_repack(uint8_t *dst, const uint8_t *src,
                                      int64_t nrows, int64_t ncols) {
    const size_t blocks = (size_t)ncols / 256u;
    const size_t row_bytes = blocks * 144u;
    for (size_t tile = 0; tile < (size_t)nrows / 8u; tile++) {
        for (size_t block = 0; block < blocks; block++) {
            uint8_t *record = dst + (tile * blocks + block) * 8u * 144u;
            for (size_t row = 0; row < 8u; row++) {
                const uint8_t *q4 = src + (tile * 8u + row) * row_bytes +
                    block * 144u;
                memcpy(record + row * 16u, q4, 16u);
            }
            uint32_t *mma = (uint32_t *)(record + 8u * 16u);
            for (size_t lane = 0; lane < 32u; lane++) {
                const size_t row = lane >> 2u, lane4 = lane & 3u;
                const uint8_t *qs = src + (tile * 8u + row) * row_bytes +
                    block * 144u + 16u;
                for (size_t group = 0; group < 8u; group++) {
                    const size_t off = (group >> 1u) * 32u + lane4 * 8u;
                    const unsigned shift = (group & 1u) ? 4u : 0u;
                    uint32_t packed = 0u;
                    for (size_t i = 0; i < 4u; i++) {
                        packed |= ((uint32_t)((qs[off + i] >> shift) & 15u))
                                  << (4u * i);
                        packed |= ((uint32_t)((qs[off + 4u + i] >> shift) & 15u))
                                  << (4u * (i + 4u));
                    }
                    mma[group * 32u + lane] = packed;
                }
            }
        }
    }
}

static int test_sm75_q4_repack(int device) {
    const int64_t nrows = 16, ncols = 512;
    const size_t bytes = (size_t)nrows * (size_t)(ncols / 256) * 144u;
    uint8_t *src = malloc(bytes), *cpu = malloc(bytes), *gpu = malloc(bytes);
    if (!src || !cpu || !gpu) {
        free(gpu); free(cpu); free(src);
        return 1;
    }
    for (size_t i = 0; i < bytes; i++)
        src[i] = (uint8_t)(i * 73u + (i >> 4u) * 19u + 11u);
    reference_sm75_q4_repack(cpu, src, nrows, ncols);
    char error[256] = {0};
    const size_t wrote = ds4q_cuda_repack_sm75_native_q4(
        src, gpu, nrows, ncols, device, error, sizeof(error));
    if (wrote != bytes || memcmp(cpu, gpu, bytes) != 0) {
        fprintf(stderr, "device %d SM75 Q4 repack mismatch (%s)\n",
                device, error[0] ? error : "byte comparison failed");
        free(gpu); free(cpu); free(src);
        return 1;
    }
    fprintf(stderr, "device %d SM75 Q4 repack byte check: OK (%zu bytes)\n",
            device, bytes);
    free(gpu); free(cpu); free(src);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s CUDA_DEVICE_CSV\n", argv[0]);
        return 2;
    }
    const int64_t nrows = 8;
    const int64_t ncols = 512;
    float *src = malloc((size_t)nrows * (size_t)ncols * sizeof(float));
    float *weights = malloc((size_t)ncols * sizeof(float));
    if (!src || !weights) return 2;
    fill_input(src, weights, nrows, ncols);

    int failed = 0;
    char *list = strdup(argv[1]);
    char *save = NULL;
    for (char *item = strtok_r(list, ",", &save);
         item;
         item = strtok_r(NULL, ",", &save)) {
        char *end = NULL;
        const long device = strtol(item, &end, 10);
        if (end == item || *end || device < 0) {
            fprintf(stderr, "bad CUDA device: %s\n", item);
            failed = 1;
            break;
        }
        failed |= test_type(DS4Q_TYPE_IQ2_XXS, (int)device,
                            src, weights, nrows, ncols);
        failed |= test_type(DS4Q_TYPE_Q4_K, (int)device,
                            src, weights, nrows, ncols);
        failed |= test_type(DS4Q_TYPE_Q2_K, (int)device,
                            src, weights, nrows, ncols);
        failed |= test_type(DS4Q_TYPE_SM75_Q4_32, (int)device,
                            src, weights, nrows, ncols);
        failed |= test_type(DS4Q_TYPE_SM75_Q3A4, (int)device,
                            src, weights, nrows, ncols);
        failed |= test_sm75_q4_repack((int)device);
    }
    ds4q_cuda_thread_shutdown();
    free(list);
    free(src);
    free(weights);
    return failed ? 1 : 0;
}
