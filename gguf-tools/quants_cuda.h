#ifndef DS4_QUANTS_CUDA_H
#define DS4_QUANTS_CUDA_H

#include "quants.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The CUDA encoder intentionally covers the routed-MoE matrix only. */
bool ds4q_cuda_type_supported(ds4q_type type);
int ds4q_cuda_device_count(char *error, size_t error_cap);

/* Returns the number of bytes written, or zero and an error string. */
size_t ds4q_cuda_quantize_chunk(ds4q_type type,
                                const float *src,
                                void *dst,
                                int64_t nrows,
                                int64_t ncols,
                                const float *imatrix,
                                int device,
                                char *error,
                                size_t error_cap);

/* Releases buffers and the stream cached by the calling host thread. */
void ds4q_cuda_thread_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
