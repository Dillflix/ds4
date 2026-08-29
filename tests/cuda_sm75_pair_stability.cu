#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#if !defined(_WIN32)
#include <unistd.h>
#endif

static int cuda_ok(cudaError_t rc, const char *what) {
    if (rc == cudaSuccess) return 1;
    std::fprintf(stderr, "error: %s: %s\n", what, cudaGetErrorString(rc));
    std::fflush(stderr);
    return 0;
}

static int cublas_ok(cublasStatus_t rc, const char *what) {
    if (rc == CUBLAS_STATUS_SUCCESS) return 1;
    std::fprintf(stderr, "error: %s: cuBLAS status %d\n", what, (int)rc);
    std::fflush(stderr);
    return 0;
}

static unsigned long long env_u64(const char *name,
                                  unsigned long long fallback,
                                  unsigned long long lo,
                                  unsigned long long hi) {
    const char *value = std::getenv(name);
    if (!value || !*value) return fallback;
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno || end == value || !end || *end || parsed < lo || parsed > hi) {
        std::fprintf(stderr, "error: %s must be %llu..%llu\n", name, lo, hi);
        std::exit(2);
    }
    return parsed;
}

static double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static int journal_note(const char *path,
                        const char *scenario,
                        unsigned long long round,
                        double elapsed,
                        const char *status) {
    if (!path || !*path) return 1;
    FILE *fp = std::fopen(path, "ab");
    if (!fp) return 0;
    const int wrote = std::fprintf(fp, "%s,%llu,%.6f,%s\n",
                                   scenario,
                                   round,
                                   elapsed,
                                   status);
    int ok = wrote > 0 && std::fflush(fp) == 0;
#if !defined(_WIN32)
    if (ok && fsync(fileno(fp)) != 0) ok = 0;
#endif
    if (std::fclose(fp) != 0) ok = 0;
    return ok;
}

struct device_state {
    void *resident = nullptr;
    __half *a = nullptr;
    __half *b = nullptr;
    __half *c = nullptr;
    void *copy_src = nullptr;
    void *copy_dst = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudaStream_t copy_stream = nullptr;
    cublasHandle_t blas = nullptr;
};

static int select_device(int device) {
    return cuda_ok(cudaSetDevice(device), "cudaSetDevice");
}

static int enable_peer_pair() {
    for (int src = 0; src < 2; src++) {
        const int dst = 1 - src;
        int can = 0;
        if (!select_device(src) ||
            !cuda_ok(cudaDeviceCanAccessPeer(&can, src, dst),
                     "cudaDeviceCanAccessPeer") || !can) {
            std::fprintf(stderr,
                         "error: visible devices %d and %d lack peer access\n",
                         src, dst);
            return 0;
        }
        const cudaError_t rc = cudaDeviceEnablePeerAccess(dst, 0);
        if (rc != cudaSuccess && rc != cudaErrorPeerAccessAlreadyEnabled) {
            return cuda_ok(rc, "cudaDeviceEnablePeerAccess");
        }
        if (rc == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
    }
    return 1;
}

static int allocate_device(device_state *s,
                           int device,
                           size_t resident_bytes,
                           size_t matrix_bytes,
                           size_t copy_bytes,
                           int need_compute,
                           int need_copy) {
    if (!select_device(device) ||
        !cuda_ok(cudaStreamCreateWithFlags(&s->compute_stream,
                                           cudaStreamNonBlocking),
                 "create compute stream") ||
        !cuda_ok(cudaStreamCreateWithFlags(&s->copy_stream,
                                           cudaStreamNonBlocking),
                 "create copy stream") ||
        !cuda_ok(cudaMalloc(&s->resident, resident_bytes),
                 "allocate resident slab") ||
        !cuda_ok(cudaMemsetAsync(s->resident, 0xa5, resident_bytes,
                                s->compute_stream),
                 "touch resident slab")) return 0;

    if (need_compute) {
        if (!cuda_ok(cudaMalloc((void **)&s->a, matrix_bytes), "allocate A") ||
            !cuda_ok(cudaMalloc((void **)&s->b, matrix_bytes), "allocate B") ||
            !cuda_ok(cudaMalloc((void **)&s->c, matrix_bytes), "allocate C") ||
            !cuda_ok(cudaMemsetAsync(s->a, 0, matrix_bytes, s->compute_stream),
                     "initialize A") ||
            !cuda_ok(cudaMemsetAsync(s->b, 0, matrix_bytes, s->compute_stream),
                     "initialize B") ||
            !cublas_ok(cublasCreate(&s->blas), "cublasCreate") ||
            !cublas_ok(cublasSetStream(s->blas, s->compute_stream),
                       "cublasSetStream") ||
            !cublas_ok(cublasSetMathMode(s->blas, CUBLAS_TENSOR_OP_MATH),
                       "cublasSetMathMode")) return 0;
    }
    if (need_copy) {
        if (!cuda_ok(cudaMalloc(&s->copy_src, copy_bytes),
                     "allocate copy source") ||
            !cuda_ok(cudaMalloc(&s->copy_dst, copy_bytes),
                     "allocate copy destination") ||
            !cuda_ok(cudaMemsetAsync(s->copy_src, 0x30 + device, copy_bytes,
                                    s->copy_stream),
                     "initialize copy source")) return 0;
    }
    return cuda_ok(cudaStreamSynchronize(s->compute_stream),
                   "initial compute synchronize") &&
           cuda_ok(cudaStreamSynchronize(s->copy_stream),
                   "initial copy synchronize");
}

static void release_device(device_state *s, int device) {
    if (!select_device(device)) return;
    if (s->blas) (void)cublasDestroy(s->blas);
    if (s->a) (void)cudaFree(s->a);
    if (s->b) (void)cudaFree(s->b);
    if (s->c) (void)cudaFree(s->c);
    if (s->copy_src) (void)cudaFree(s->copy_src);
    if (s->copy_dst) (void)cudaFree(s->copy_dst);
    if (s->resident) (void)cudaFree(s->resident);
    if (s->compute_stream) (void)cudaStreamDestroy(s->compute_stream);
    if (s->copy_stream) (void)cudaStreamDestroy(s->copy_stream);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: %s residency|compute|p2p|combined\n", argv[0]);
        return 2;
    }
    const std::string scenario = argv[1];
    const int do_compute = scenario == "compute" || scenario == "combined";
    const int do_copy = scenario == "p2p" || scenario == "combined";
    if (scenario != "residency" && !do_compute && !do_copy) {
        std::fprintf(stderr, "error: unknown scenario: %s\n", argv[1]);
        return 2;
    }

    int count = 0;
    if (!cuda_ok(cudaGetDeviceCount(&count), "cudaGetDeviceCount") || count != 2) {
        std::fprintf(stderr,
                     "error: pair stability harness requires exactly two visible GPUs\n");
        return 2;
    }
    const unsigned long long duration =
        env_u64("DS4_PAIR_STRESS_SECONDS", 60, 5, 3600);
    const unsigned long long resident_mib =
        env_u64("DS4_PAIR_STRESS_RESIDENT_MIB", 43008, 1024, 45000);
    const unsigned long long copy_mib =
        env_u64("DS4_PAIR_STRESS_COPY_MIB", 128, 16, 1024);
    const unsigned long long gemm_n =
        env_u64("DS4_PAIR_STRESS_GEMM_N", 4096, 1024, 8192);
    const unsigned long long gemms_per_round =
        env_u64("DS4_PAIR_STRESS_GEMMS_PER_ROUND", 4, 1, 64);
    const unsigned long long copies_per_round =
        env_u64("DS4_PAIR_STRESS_COPIES_PER_ROUND", 4, 1, 64);
    const char *journal = std::getenv("DS4_PAIR_STRESS_JOURNAL");

    char bus0[32] = {0};
    char bus1[32] = {0};
    if (!cuda_ok(cudaDeviceGetPCIBusId(bus0, sizeof(bus0), 0), "PCI bus ID 0") ||
        !cuda_ok(cudaDeviceGetPCIBusId(bus1, sizeof(bus1), 1), "PCI bus ID 1")) {
        return 1;
    }
    std::printf("scenario=%s\nvisible_device_0=%s\nvisible_device_1=%s\n"
                "duration_seconds=%llu\nresident_mib_per_gpu=%llu\n"
                "copy_mib_per_buffer=%llu\ngemm_n=%llu\n",
                scenario.c_str(), bus0, bus1, duration, resident_mib,
                copy_mib, gemm_n);
    std::fflush(stdout);

    if (journal && *journal) {
        FILE *fp = std::fopen(journal, "wb");
        if (!fp) {
            std::fprintf(stderr, "error: cannot create journal %s\n", journal);
            return 1;
        }
        std::fprintf(fp, "scenario,round,elapsed_sec,status\n");
        std::fflush(fp);
#if !defined(_WIN32)
        (void)fsync(fileno(fp));
#endif
        std::fclose(fp);
        if (!journal_note(journal, scenario.c_str(), 0, 0.0, "allocating")) {
            std::fprintf(stderr, "error: failed to initialize %s\n", journal);
            return 1;
        }
    }

    if ((do_copy || scenario == "combined") && !enable_peer_pair()) return 1;
    const size_t resident_bytes = (size_t)resident_mib * 1024u * 1024u;
    const size_t copy_bytes = (size_t)copy_mib * 1024u * 1024u;
    const size_t matrix_bytes = (size_t)gemm_n * (size_t)gemm_n * sizeof(__half);
    device_state state[2];
    for (int device = 0; device < 2; device++) {
        if (!allocate_device(&state[device], device, resident_bytes,
                             matrix_bytes, copy_bytes, do_compute, do_copy)) {
            return 1;
        }
    }
    if (!journal_note(journal, scenario.c_str(), 0, 0.0, "resident")) return 1;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const double started = now_sec();
    double next_progress = 0.0;
    unsigned long long round = 0;
    while (now_sec() - started < (double)duration) {
        round++;
        if (do_compute) {
            for (int device = 0; device < 2; device++) {
                if (!select_device(device)) return 1;
                for (unsigned long long op = 0; op < gemms_per_round; op++) {
                    if (!cublas_ok(cublasGemmEx(
                            state[device].blas,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            (int)gemm_n, (int)gemm_n, (int)gemm_n,
                            &alpha,
                            state[device].a, CUDA_R_16F, (int)gemm_n,
                            state[device].b, CUDA_R_16F, (int)gemm_n,
                            &beta,
                            state[device].c, CUDA_R_16F, (int)gemm_n,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                        "cublasGemmEx")) return 1;
                }
            }
        }
        if (do_copy) {
            for (unsigned long long op = 0; op < copies_per_round; op++) {
                if (!select_device(0) ||
                    !cuda_ok(cudaMemcpyPeerAsync(
                        state[0].copy_dst, 0, state[1].copy_src, 1,
                        copy_bytes, state[0].copy_stream),
                        "peer copy 1->0") ||
                    !select_device(1) ||
                    !cuda_ok(cudaMemcpyPeerAsync(
                        state[1].copy_dst, 1, state[0].copy_src, 0,
                        copy_bytes, state[1].copy_stream),
                        "peer copy 0->1")) return 1;
            }
        }
        if (!do_compute && !do_copy) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        for (int device = 0; device < 2; device++) {
            if (!select_device(device) ||
                !cuda_ok(cudaStreamSynchronize(state[device].compute_stream),
                         "compute stream synchronize") ||
                !cuda_ok(cudaStreamSynchronize(state[device].copy_stream),
                         "copy stream synchronize")) return 1;
        }
        const double elapsed = now_sec() - started;
        if (elapsed >= next_progress || elapsed >= (double)duration) {
            if (!journal_note(journal, scenario.c_str(), round,
                              elapsed, "running")) {
                std::fprintf(stderr, "error: failed to update %s\n", journal);
                return 1;
            }
            std::printf("progress_round=%llu elapsed_sec=%.3f\n", round, elapsed);
            std::fflush(stdout);
            next_progress = elapsed + 1.0;
        }
    }

    if (do_copy) {
        for (int device = 0; device < 2; device++) {
            unsigned char observed = 0;
            const unsigned char expected = (unsigned char)(0x30 + (1 - device));
            if (!select_device(device) ||
                !cuda_ok(cudaMemcpy(&observed, state[device].copy_dst, 1,
                                    cudaMemcpyDeviceToHost),
                         "validate peer-copy destination") ||
                observed != expected) {
                std::fprintf(stderr,
                             "error: peer-copy validation device=%d expected=%u observed=%u\n",
                             device, (unsigned)expected, (unsigned)observed);
                return 1;
            }
        }
        std::printf("peer_copy_validation=exact\n");
    }
    if (!journal_note(journal, scenario.c_str(), round,
                      now_sec() - started, "completed")) return 1;
    std::printf("completed_rounds=%llu\nharness_status=ok\n", round);
    std::fflush(stdout);
    for (int device = 1; device >= 0; device--) release_device(&state[device], device);
    return 0;
}
