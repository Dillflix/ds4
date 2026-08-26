#include "quants.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int expect_rejected(float value) {
    uint16_t dst = UINT16_C(0xdead);
    int64_t bad_index = -1;
    if (ds4q_f32_to_f16_checked_row(&value, &dst, 1, &bad_index) ||
        bad_index != 0) {
        fprintf(stderr, "expected checked F16 conversion to reject %g\n",
                (double)value);
        return 0;
    }
    return 1;
}

int main(void) {
    const float finite[] = {
        0.0f, -0.0f, 1.0f, -2.0f, 65504.0f, 0x1.0p-24f,
    };
    uint16_t converted[sizeof(finite) / sizeof(finite[0])] = {0};
    int64_t bad_index = 123;
    if (!ds4q_f32_to_f16_checked_row(
            finite, converted,
            (int64_t)(sizeof(finite) / sizeof(finite[0])), &bad_index) ||
        bad_index != -1) {
        fputs("checked F16 conversion rejected a valid finite row\n", stderr);
        return 1;
    }
    for (size_t i = 0; i < sizeof(converted) / sizeof(converted[0]); i++) {
        if (!isfinite(ds4q_f16_to_f32(converted[i]))) {
            fprintf(stderr, "checked F16 conversion produced non-finite output at %zu\n", i);
            return 1;
        }
    }
    const uint16_t expected[] = {
        UINT16_C(0x0000), UINT16_C(0x8000), UINT16_C(0x3c00),
        UINT16_C(0xc000), UINT16_C(0x7bff), UINT16_C(0x0001),
    };
    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); i++) {
        if (converted[i] != expected[i]) {
            fprintf(stderr,
                    "checked F16 conversion mismatch at %zu: got=%04x expected=%04x\n",
                    i, converted[i], expected[i]);
            return 1;
        }
    }

    if (!expect_rejected(NAN) || !expect_rejected(INFINITY) ||
        !expect_rejected(-INFINITY) || !expect_rejected(70000.0f) ||
        !expect_rejected(-70000.0f)) {
        return 1;
    }

    puts("source-F16 checked conversion tests passed");
    return 0;
}
