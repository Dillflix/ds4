#ifndef DS4_QUANTS_H
#define DS4_QUANTS_H

/*
 * Narrow quantization API used by the DS4 GGUF writer.
 *
 * The enum values intentionally match GGUF/GGML type IDs so template metadata
 * can be copied without translation.  Only the formats used by the DS4 Flash
 * quantization recipes are implemented as output targets.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4Q_MAX_DIMS 4

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DS4Q_TYPE_F32     = 0,
    DS4Q_TYPE_F16     = 1,
    DS4Q_TYPE_Q4_0    = 2,
    DS4Q_TYPE_Q4_1    = 3,
    DS4Q_TYPE_Q5_0    = 6,
    DS4Q_TYPE_Q5_1    = 7,
    DS4Q_TYPE_Q8_0    = 8,
    DS4Q_TYPE_Q8_1    = 9,
    DS4Q_TYPE_Q2_K    = 10,
    DS4Q_TYPE_Q3_K    = 11,
    DS4Q_TYPE_Q4_K    = 12,
    DS4Q_TYPE_Q5_K    = 13,
    DS4Q_TYPE_Q6_K    = 14,
    DS4Q_TYPE_Q8_K    = 15,
    DS4Q_TYPE_IQ2_XXS = 16,
    DS4Q_TYPE_IQ2_XS  = 17,
    DS4Q_TYPE_IQ3_XXS = 18,
    DS4Q_TYPE_IQ1_S   = 19,
    DS4Q_TYPE_IQ4_NL  = 20,
    DS4Q_TYPE_IQ3_S   = 21,
    DS4Q_TYPE_IQ2_S   = 22,
    DS4Q_TYPE_IQ4_XS  = 23,
    DS4Q_TYPE_I8      = 24,
    DS4Q_TYPE_I16     = 25,
    DS4Q_TYPE_I32     = 26,
    DS4Q_TYPE_I64     = 27,
    DS4Q_TYPE_F64     = 28,
    DS4Q_TYPE_IQ1_M   = 29,
    DS4Q_TYPE_BF16    = 30,
    DS4Q_TYPE_TQ1_0   = 34,
    DS4Q_TYPE_TQ2_0   = 35,
    DS4Q_TYPE_MXFP4   = 39,
    DS4Q_TYPE_NVFP4   = 40,
    DS4Q_TYPE_Q1_0    = 41,
    /* DS4-private, architecture-tagged routed-expert formats in this revision.
     * They must never be read as standard GGUF quant types. */
    DS4Q_TYPE_SM75_Q4_32 = 42,
    DS4Q_TYPE_SM75_Q3A4  = 43,
    DS4Q_TYPE_COUNT      = 44,
} ds4q_type;

static inline size_t ds4q_pad(size_t x, size_t n) {
    return ((x + n - 1) / n) * n;
}

const char *ds4q_type_name(ds4q_type type);
bool ds4q_can_quantize(ds4q_type type);
int64_t ds4q_block_size(ds4q_type type);
size_t ds4q_row_size(ds4q_type type, int64_t ne);
bool ds4q_requires_imatrix(ds4q_type type);
void ds4q_quantize_init(ds4q_type type);
size_t ds4q_quantize_chunk(ds4q_type type, const float *src, void *dst,
                           int64_t start, int64_t nrows, int64_t ncols,
                           const float *imatrix);

float ds4q_f16_to_f32(uint16_t bits);
float ds4q_bf16_to_f32(uint16_t bits);
void ds4q_f32_to_f16_row(const float *src, uint16_t *dst, int64_t n);
void ds4q_f32_to_bf16_row(const float *src, uint16_t *dst, int64_t n);

/* Internal IQ2_XXS search tables shared with the optional CUDA encoder. */
bool ds4q_iq2_xxs_tables(const uint64_t **grid, size_t *grid_len,
                         const int **map, size_t *map_len,
                         const uint16_t **neighbours, size_t *neighbours_len);

/*
 * Bounded research API for the SM75 Q3/Q4-32 experiment.  The shipping
 * IQ2_XXS gate/up format is included as a comparison control.  The new
 * Q4-32 and Q3A4 were subsequently promoted to architecture-tagged GGUF
 * output types; they remain here as controls for the bounded format sweep.
 */
typedef enum {
    DS4Q_EXPERIMENT_Q4_K = 0,
    DS4Q_EXPERIMENT_Q3_K,
    DS4Q_EXPERIMENT_SM75_Q3_32,
    DS4Q_EXPERIMENT_SM75_Q4_32,
    DS4Q_EXPERIMENT_IQ2_XXS,
    DS4Q_EXPERIMENT_SM75_IQ3_32,
    DS4Q_EXPERIMENT_SM75_Q3A_32_4,
    DS4Q_EXPERIMENT_SM75_Q3A_32_6,
    DS4Q_EXPERIMENT_SM75_Q3Q4_32_25,
    DS4Q_EXPERIMENT_SM75_Q3Q4_32_50,
    DS4Q_EXPERIMENT_SM75_Q4A_32_4,
    DS4Q_EXPERIMENT_SM75_Q4A_32_5,
    DS4Q_EXPERIMENT_SM75_Q5_32,
    DS4Q_EXPERIMENT_SM75_Q2Q3_32_50,
    DS4Q_EXPERIMENT_SM75_Q2Q3_32_75,
    DS4Q_EXPERIMENT_COUNT,
} ds4q_experimental_format;

#define DS4Q_EXPERIMENT_MAX_BLOCK_BYTES 168

const char *ds4q_experimental_format_name(ds4q_experimental_format format);
size_t ds4q_experimental_block_bytes(ds4q_experimental_format format);
bool ds4q_experimental_quantize_block(ds4q_experimental_format format,
                                      const float src[256],
                                      const float *imatrix,
                                      void *dst);
bool ds4q_experimental_dequantize_block(ds4q_experimental_format format,
                                        const void *src,
                                        float dst[256]);

/* Size-neutral canonical-row to SM75 m8n8k32 tile transforms.  nrows must be
 * a multiple of eight and ncols a multiple of 256. */
size_t ds4q_repack_sm75_q4_32(const void *src, void *dst,
                              int64_t nrows, int64_t ncols);
size_t ds4q_repack_sm75_q3a4(const void *src, void *dst,
                             int64_t nrows, int64_t ncols);

#ifdef __cplusplus
}
#endif

#endif
