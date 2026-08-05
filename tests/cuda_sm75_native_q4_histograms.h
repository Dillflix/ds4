#ifndef DS4_CUDA_SM75_NATIVE_Q4_HISTOGRAMS_H
#define DS4_CUDA_SM75_NATIVE_Q4_HISTOGRAMS_H

#include <stdint.h>

/* Exact home-half expert histograms from the fixed full-Q4 512-token prompt
 * capture.  Native layout changes are bit-exact, so routing is unchanged. */
static const uint16_t ds4_native_q4_early_counts[128] = {
    18u, 16u, 0u, 3u, 3u, 5u, 15u, 0u,
    0u, 0u, 2u, 10u, 2u, 4u, 5u, 1u,
    19u, 26u, 12u, 6u, 15u, 10u, 1u, 35u,
    4u, 8u, 28u, 17u, 3u, 0u, 2u, 2u,
    0u, 0u, 4u, 4u, 1u, 22u, 0u, 0u,
    8u, 8u, 1u, 0u, 5u, 12u, 8u, 0u,
    6u, 0u, 0u, 0u, 0u, 2u, 19u, 6u,
    1u, 7u, 0u, 0u, 478u, 18u, 1u, 0u,
    2u, 3u, 6u, 0u, 4u, 27u, 4u, 15u,
    0u, 24u, 449u, 11u, 0u, 2u, 3u, 7u,
    3u, 12u, 0u, 6u, 24u, 24u, 13u, 34u,
    0u, 3u, 4u, 3u, 1u, 20u, 2u, 0u,
    2u, 3u, 0u, 9u, 54u, 0u, 2u, 9u,
    4u, 20u, 2u, 10u, 6u, 3u, 0u, 32u,
    1u, 21u, 13u, 33u, 6u, 52u, 3u, 0u,
    3u, 0u, 5u, 12u, 0u, 0u, 0u, 0u,
};

static const uint16_t ds4_native_q4_late_counts[128] = {
    0u, 4u, 122u, 0u, 3u, 13u, 0u, 3u,
    1u, 0u, 0u, 0u, 1u, 4u, 0u, 0u,
    498u, 0u, 0u, 11u, 0u, 0u, 0u, 96u,
    1u, 466u, 4u, 0u, 0u, 2u, 7u, 22u,
    0u, 0u, 1u, 2u, 0u, 0u, 13u, 0u,
    1u, 1u, 28u, 2u, 1u, 0u, 0u, 0u,
    13u, 8u, 0u, 0u, 3u, 0u, 0u, 1u,
    2u, 9u, 0u, 6u, 6u, 1u, 5u, 5u,
    11u, 0u, 0u, 1u, 5u, 0u, 2u, 0u,
    0u, 1u, 0u, 0u, 0u, 4u, 1u, 0u,
    0u, 0u, 14u, 11u, 0u, 1u, 0u, 0u,
    0u, 5u, 2u, 1u, 0u, 0u, 9u, 141u,
    2u, 6u, 0u, 0u, 2u, 1u, 1u, 1u,
    3u, 3u, 2u, 87u, 22u, 0u, 0u, 0u,
    0u, 301u, 0u, 14u, 15u, 1u, 8u, 4u,
    0u, 1u, 34u, 39u, 3u, 0u, 0u, 1u,
};

#endif
