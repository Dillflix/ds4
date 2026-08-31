#include <cuda_runtime.h>

#include <cstdio>

static void format_uuid(char out[41], const cudaUUID_t *uuid) {
    const unsigned char *b =
        uuid ? reinterpret_cast<const unsigned char *>(uuid->bytes) : nullptr;
    if (!b) {
        std::snprintf(out, 41, "unavailable");
        return;
    }
    std::snprintf(out, 41,
                  "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
                  "%02x%02x%02x%02x%02x%02x",
                  b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                  b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
}

int main(void) {
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n",
                     cudaGetErrorString(error));
        return 1;
    }

    std::puts("cuda_ordinal,pci_bus_id,uuid");
    for (int ordinal = 0; ordinal < count; ordinal++) {
        cudaDeviceProp prop = {};
        char pci_bus_id[32] = {};
        error = cudaGetDeviceProperties(&prop, ordinal);
        if (error != cudaSuccess) {
            std::fprintf(stderr,
                         "cudaGetDeviceProperties(%d) failed: %s\n",
                         ordinal, cudaGetErrorString(error));
            return 1;
        }
        error = cudaDeviceGetPCIBusId(pci_bus_id,
                                      static_cast<int>(sizeof(pci_bus_id)),
                                      ordinal);
        if (error != cudaSuccess) {
            std::fprintf(stderr,
                         "cudaDeviceGetPCIBusId(%d) failed: %s\n",
                         ordinal, cudaGetErrorString(error));
            return 1;
        }
        char uuid[41] = {};
        format_uuid(uuid, &prop.uuid);
        std::printf("%d,%s,%s\n", ordinal, pci_bus_id, uuid);
    }
    return 0;
}
