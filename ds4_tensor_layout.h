#ifndef DS4_TENSOR_LAYOUT_H
#define DS4_TENSOR_LAYOUT_H

#include <stdint.h>

/* Internal routed-weight layout bit carried alongside the ordinary GGUF
 * tensor type.  It never changes the GGUF type id: untagged Q4_K remains the
 * portable row-major format on every backend. */
#define DS4_TENSOR_LAYOUT_SM75_NATIVE_Q4 UINT32_C(0x80000000)

#endif
