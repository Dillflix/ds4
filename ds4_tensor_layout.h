#ifndef DS4_TENSOR_LAYOUT_H
#define DS4_TENSOR_LAYOUT_H

#include <stdint.h>

/* Internal routed-weight layout bit carried alongside the ordinary GGUF
 * tensor type.  It never changes the GGUF type id: untagged Q4_K remains the
 * portable row-major format on every backend. */
#define DS4_TENSOR_LAYOUT_SM75_NATIVE_Q4 UINT32_C(0x80000000)
#define DS4_TENSOR_LAYOUT_SM75_Q4_32      UINT32_C(0x40000000)
#define DS4_TENSOR_LAYOUT_SM75_Q3A4       UINT32_C(0x20000000)
#define DS4_TENSOR_LAYOUT_SM75_Q8_WARP32  UINT32_C(0x10000000)

/* Source encodings for ds4_q8_native_range.  These values are shared by the
 * C planner and the CUDA backend, which intentionally do not share the
 * descriptor typedef itself. */
#define DS4_Q8_NATIVE_LAYOUT_NONE              0u
#define DS4_Q8_NATIVE_LAYOUT_ROW_WARP32        1u
#define DS4_Q8_NATIVE_LAYOUT_B_KSHARDS_WARP32  2u

#define DS4_KV_SM75_ROUTED_LAYOUT \
    "ds4.routed_expert.sm75.layout"
#define DS4_KV_SM75_ROUTED_LAYOUT_VERSION \
    "ds4.routed_expert.sm75.layout_version"
#define DS4_SM75_ROUTED_LAYOUT_Q4_32_Q3A4 \
    "sm75_m8n8k32_q4_32_q3a4_native_aw_v1"

/* DS4-private dense-Q8 layout.  T32 and attention-output A are stored as
 * size-neutral 32-block scale/word planes per row.  Attention-output B uses
 * the same row encoding inside two contiguous K halves so the fixed 2-way TP
 * recipe can map one ordinary GGUF span per GPU without a runtime repack or a
 * second persistent allocation. */
#define DS4_KV_SM75_DENSE_Q8_LAYOUT \
    "ds4.dense_q8.sm75.layout"
#define DS4_KV_SM75_DENSE_Q8_LAYOUT_VERSION \
    "ds4.dense_q8.sm75.layout_version"
#define DS4_SM75_DENSE_Q8_LAYOUT_WARP32_TP2 \
    "sm75_q8_0_warp32_t32_a_rows_b_kshards_tp2_v1"

#endif
