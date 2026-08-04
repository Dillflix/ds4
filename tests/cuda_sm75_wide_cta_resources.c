#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int ds4_cuda_sm75_wide_cta_resource_matrix(void);

#ifdef __cplusplus
}
#endif

int main(void) {
    if (!ds4_cuda_sm75_wide_cta_resource_matrix()) {
        fprintf(stderr, "error: SM75 wide-CTA resource query failed\n");
        return 1;
    }
    return 0;
}
