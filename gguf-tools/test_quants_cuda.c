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
    const size_t cpu_bytes = ds4q_quantize_chunk(type, src, cpu, 0,
                                                 nrows, ncols, weights);
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

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s CUDA_DEVICE_CSV\n", argv[0]);
        return 2;
    }
    const int64_t nrows = 2;
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
    }
    ds4q_cuda_thread_shutdown();
    free(list);
    free(src);
    free(weights);
    return failed ? 1 : 0;
}
