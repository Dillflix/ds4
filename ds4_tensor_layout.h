#ifndef DS4_TENSOR_LAYOUT_H
#define DS4_TENSOR_LAYOUT_H

#include <stdint.h>

/* Internal routed-weight layout bit carried alongside the ordinary GGUF
 * tensor type.  It never changes the GGUF type id: untagged Q4_K remains the
 * portable row-major format on every backend. */
#define DS4_TENSOR_LAYOUT_SM75_NATIVE_Q4 UINT32_C(0x80000000)
#define DS4_TENSOR_LAYOUT_SM75_Q4_32      UINT32_C(0x40000000)
#define DS4_TENSOR_LAYOUT_SM75_Q3A4       UINT32_C(0x20000000)

#define DS4_KV_SM75_ROUTED_LAYOUT \
    "ds4.routed_expert.sm75.layout"
#define DS4_KV_SM75_ROUTED_LAYOUT_VERSION \
    "ds4.routed_expert.sm75.layout_version"
#define DS4_SM75_ROUTED_LAYOUT_Q4_32_Q3A4 \
    "sm75_m8n8k32_q4_32_q3a4_native_aw_v1"

#endif
