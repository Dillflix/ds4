#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kIn = 4096u;
constexpr uint32_t kWidth = 1024u;
constexpr uint32_t kRatio = 4u;
constexpr uint32_t kStateRows = 8u;
constexpr uint32_t kLanes = 32u;
constexpr uint32_t kChunk = kIn / kLanes;
constexpr uint32_t kStageTile = 32u;
constexpr uint32_t kStagePad = kStageTile + 1u;
constexpr uint32_t kStageTiles = kChunk / kStageTile;
constexpr uint32_t kGuard = 32u;
constexpr uint32_t kPos = 32767u;

enum class Arm { Control, CanonicalStaged, LaneMajor, TwoWarp };

const char *arm_name(Arm arm) {
    switch (arm) {
        case Arm::Control: return "control";
        case Arm::CanonicalStaged: return "canonical-staged";
        case Arm::LaneMajor: return "lane-major";
        case Arm::TwoWarp: return "lane-major-two-warp";
    }
    return "invalid";
}

bool cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return true;
    std::fprintf(stderr, "error: %s: %s\n", what, cudaGetErrorString(err));
    return false;
}

__global__ void sm75_compressor_weight_lane_major_pack_kernel(
        __half *dst, const __half *src) {
    const uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t count = (uint64_t)kWidth * kIn;
    if (index >= count) return;
    const uint32_t row = (uint32_t)(index / kIn);
    const uint32_t k = (uint32_t)(index - (uint64_t)row * kIn);
    const uint32_t lane = k / kChunk;
    const uint32_t step = k - lane * kChunk;
    dst[(uint64_t)row * kIn + (uint64_t)step * kLanes + lane] = src[index];
}

__global__ void sm75_compressor_activation_lane_major_pack_kernel(
        float *dst, const float *src) {
    const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= kIn) return;
    const uint32_t lane = k / kChunk;
    const uint32_t step = k - lane * kChunk;
    dst[step * kLanes + lane] = src[k];
}

__device__ __forceinline__ void store_projection_state(
        float *out, float *state, const __half *ape, uint32_t row,
        uint32_t pos, float value, bool score) {
    out[row] = value;
    const uint32_t pos_mod = pos % kRatio;
    const uint32_t dst_row = kRatio + pos_mod;
    const float adjusted = score
        ? value + __half2float(ape[(uint64_t)pos_mod * kWidth + row])
        : value;
    state[(uint64_t)dst_row * kWidth + row] = adjusted;
}

__global__ void sm75_compressor_pair_control_kernel(
        float *out_kv, float *out_score,
        float *state_kv, float *state_score,
        const __half *weight_kv, const __half *weight_score,
        const float *x, const __half *ape, uint32_t pos) {
    const uint32_t row = blockIdx.x;
    if (row >= kWidth) return;
    __shared__ float partial_kv[kLanes];
    __shared__ float partial_score[kLanes];
    const uint32_t lane = threadIdx.x;
    const uint32_t k0 = lane * kChunk;
    const __half *wkv = weight_kv + (uint64_t)row * kIn;
    const __half *wscore = weight_score + (uint64_t)row * kIn;
    float sum_kv = 0.0f;
    float sum_score = 0.0f;
    for (uint32_t step = 0; step < kChunk; ++step) {
        const uint32_t k = k0 + step;
        const float xv = x[k];
        sum_kv += __half2float(wkv[k]) * xv;
        sum_score += __half2float(wscore[k]) * xv;
    }
    partial_kv[lane] = sum_kv;
    partial_score[lane] = sum_score;
    __syncthreads();
    if (lane == 0u) {
        float total_kv = 0.0f;
        float total_score = 0.0f;
        for (uint32_t i = 0; i < kLanes; ++i) {
            total_kv += partial_kv[i];
            total_score += partial_score[i];
        }
        store_projection_state(out_kv, state_kv, ape, row, pos,
                               total_kv, false);
        store_projection_state(out_score, state_score, ape, row, pos,
                               total_score, true);
    }
}

/* Keep the canonical row-major model representation.  Each CTA cooperatively
 * loads a 32x32 slice from every lane's original 128-element K interval, so
 * global reads are contiguous.  Padding the shared rows keeps the subsequent
 * per-lane reads from collapsing onto the same banks.  Tiles and elements are
 * consumed in the original order, preserving the production accumulation. */
__global__ void sm75_compressor_pair_canonical_staged_kernel(
        float *out_kv, float *out_score,
        float *state_kv, float *state_score,
        const __half *weight_kv, const __half *weight_score,
        const float *x, const __half *ape, uint32_t pos) {
    const uint32_t row = blockIdx.x;
    if (row >= kWidth) return;
    __shared__ __half staged_kv[kLanes][kStagePad];
    __shared__ __half staged_score[kLanes][kStagePad];
    __shared__ float partial_kv[kLanes];
    __shared__ float partial_score[kLanes];
    const uint32_t lane = threadIdx.x;
    const __half *wkv = weight_kv + (uint64_t)row * kIn;
    const __half *wscore = weight_score + (uint64_t)row * kIn;
    float sum_kv = 0.0f;
    float sum_score = 0.0f;
    for (uint32_t tile = 0; tile < kStageTiles; ++tile) {
        for (uint32_t source_lane = 0; source_lane < kLanes;
             ++source_lane) {
            const uint32_t k = source_lane * kChunk +
                               tile * kStageTile + lane;
            staged_kv[source_lane][lane] = wkv[k];
            staged_score[source_lane][lane] = wscore[k];
        }
        __syncthreads();
        for (uint32_t step = 0; step < kStageTile; ++step) {
            const uint32_t k = lane * kChunk + tile * kStageTile + step;
            const float xv = x[k];
            sum_kv += __half2float(staged_kv[lane][step]) * xv;
            sum_score += __half2float(staged_score[lane][step]) * xv;
        }
        __syncthreads();
    }
    partial_kv[lane] = sum_kv;
    partial_score[lane] = sum_score;
    __syncthreads();
    if (lane == 0u) {
        float total_kv = 0.0f;
        float total_score = 0.0f;
        for (uint32_t i = 0; i < kLanes; ++i) {
            total_kv += partial_kv[i];
            total_score += partial_score[i];
        }
        store_projection_state(out_kv, state_kv, ape, row, pos,
                               total_kv, false);
        store_projection_state(out_score, state_score, ape, row, pos,
                               total_score, true);
    }
}

__global__ void sm75_compressor_pair_lane_major_kernel(
        float *out_kv, float *out_score,
        float *state_kv, float *state_score,
        const __half *weight_kv, const __half *weight_score,
        const float *x, const __half *ape, uint32_t pos) {
    const uint32_t row = blockIdx.x;
    if (row >= kWidth) return;
    __shared__ float partial_kv[kLanes];
    __shared__ float partial_score[kLanes];
    const uint32_t lane = threadIdx.x;
    const uint64_t base = (uint64_t)row * kIn + lane;
    float sum_kv = 0.0f;
    float sum_score = 0.0f;
    for (uint32_t step = 0; step < kChunk; ++step) {
        const uint64_t index = base + (uint64_t)step * kLanes;
        const float xv = x[step * kLanes + lane];
        sum_kv += __half2float(weight_kv[index]) * xv;
        sum_score += __half2float(weight_score[index]) * xv;
    }
    partial_kv[lane] = sum_kv;
    partial_score[lane] = sum_score;
    __syncthreads();
    if (lane == 0u) {
        float total_kv = 0.0f;
        float total_score = 0.0f;
        for (uint32_t i = 0; i < kLanes; ++i) {
            total_kv += partial_kv[i];
            total_score += partial_score[i];
        }
        store_projection_state(out_kv, state_kv, ape, row, pos,
                               total_kv, false);
        store_projection_state(out_score, state_score, ape, row, pos,
                               total_score, true);
    }
}

__global__ void sm75_compressor_pair_lane_major_two_warp_kernel(
        float *out_kv, float *out_score,
        float *state_kv, float *state_score,
        const __half *weight_kv, const __half *weight_score,
        const float *x, const __half *ape, uint32_t pos) {
    const uint32_t row = blockIdx.x;
    if (row >= kWidth) return;
    __shared__ float partial[2][kLanes];
    const uint32_t warp = threadIdx.x / kLanes;
    const uint32_t lane = threadIdx.x % kLanes;
    const __half *weight = warp == 0u ? weight_kv : weight_score;
    const uint64_t base = (uint64_t)row * kIn + lane;
    float sum = 0.0f;
    for (uint32_t step = 0; step < kChunk; ++step) {
        const uint64_t index = base + (uint64_t)step * kLanes;
        sum += __half2float(weight[index]) * x[step * kLanes + lane];
    }
    partial[warp][lane] = sum;
    __syncthreads();
    if (lane == 0u) {
        float total = 0.0f;
        for (uint32_t i = 0; i < kLanes; ++i) total += partial[warp][i];
        if (warp == 0u) {
            store_projection_state(out_kv, state_kv, ape, row, pos,
                                   total, false);
        } else {
            store_projection_state(out_score, state_score, ape, row, pos,
                                   total, true);
        }
    }
}

struct Guarded {
    float *device = nullptr;
    uint64_t payload = 0u;
    std::vector<float> initial;

    float *data() const { return device + kGuard; }
};

struct Outputs {
    Guarded out_kv;
    Guarded out_score;
    Guarded state_kv;
    Guarded state_score;
};

struct Fixture {
    __half *weight_kv = nullptr;
    __half *weight_score = nullptr;
    __half *packed_kv = nullptr;
    __half *packed_score = nullptr;
    __half *ape = nullptr;
    float *x = nullptr;
    float *packed_x = nullptr;
    Outputs output;
    uint32_t pos = kPos;
};

bool guarded_alloc(Guarded *g, uint64_t payload, uint32_t salt) {
    g->payload = payload;
    g->initial.resize((size_t)payload + 2u * kGuard);
    for (uint64_t i = 0; i < g->initial.size(); ++i) {
        const int v = (int)((i * 37u + salt * 19u) % 251u) - 125;
        g->initial[(size_t)i] = (float)v / 64.0f;
    }
    return cuda_ok(cudaMalloc((void **)&g->device,
                              g->initial.size() * sizeof(float)),
                   "guarded cudaMalloc") &&
           cuda_ok(cudaMemcpy(g->device, g->initial.data(),
                              g->initial.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "guarded initialize");
}

bool outputs_alloc(Outputs *o) {
    return guarded_alloc(&o->out_kv, kWidth, 11u) &&
           guarded_alloc(&o->out_score, kWidth, 13u) &&
           guarded_alloc(&o->state_kv, (uint64_t)kStateRows * kWidth, 17u) &&
           guarded_alloc(&o->state_score, (uint64_t)kStateRows * kWidth, 19u);
}

bool outputs_reset(Outputs *o) {
    for (Guarded *g : {&o->out_kv, &o->out_score, &o->state_kv,
                       &o->state_score}) {
        if (!cuda_ok(cudaMemcpy(g->device, g->initial.data(),
                                g->initial.size() * sizeof(float),
                                cudaMemcpyHostToDevice), "output reset")) return false;
    }
    return true;
}

void outputs_free(Outputs *o) {
    for (Guarded *g : {&o->out_kv, &o->out_score, &o->state_kv,
                       &o->state_score}) {
        cudaFree(g->device);
        g->device = nullptr;
    }
}

std::vector<unsigned char> outputs_snapshot(const Outputs &o) {
    std::vector<unsigned char> bytes;
    for (const Guarded *g : {&o.out_kv, &o.out_score, &o.state_kv,
                             &o.state_score}) {
        const size_t old = bytes.size();
        const size_t add = g->initial.size() * sizeof(float);
        bytes.resize(old + add);
        if (!cuda_ok(cudaMemcpy(bytes.data() + old, g->device, add,
                                cudaMemcpyDeviceToHost), "output snapshot")) {
            bytes.clear();
            return bytes;
        }
    }
    return bytes;
}

bool canaries_ok(const Outputs &o, const std::vector<unsigned char> &snapshot) {
    size_t offset = 0u;
    for (const Guarded *g : {&o.out_kv, &o.out_score, &o.state_kv,
                             &o.state_score}) {
        const unsigned char *got = snapshot.data() + offset;
        const uint64_t suffix = kGuard + g->payload;
        if (std::memcmp(got, g->initial.data(), kGuard * sizeof(float)) != 0 ||
            std::memcmp(got + suffix * sizeof(float),
                        g->initial.data() + suffix,
                        kGuard * sizeof(float)) != 0) return false;
        offset += g->initial.size() * sizeof(float);
    }
    return true;
}

bool written_regions_changed(const Outputs &o,
                             const std::vector<unsigned char> &snapshot) {
    size_t offset = 0u;
    for (const Guarded *g : {&o.out_kv, &o.out_score}) {
        const unsigned char *got = snapshot.data() + offset;
        if (std::memcmp(got + kGuard * sizeof(float),
                        g->initial.data() + kGuard,
                        g->payload * sizeof(float)) == 0) return false;
        offset += g->initial.size() * sizeof(float);
    }
    constexpr uint32_t target_row = kRatio + (kPos % kRatio);
    for (const Guarded *g : {&o.state_kv, &o.state_score}) {
        const unsigned char *got = snapshot.data() + offset;
        const uint64_t first = kGuard + (uint64_t)target_row * kWidth;
        if (std::memcmp(got + first * sizeof(float),
                        g->initial.data() + first,
                        kWidth * sizeof(float)) == 0) return false;
        offset += g->initial.size() * sizeof(float);
    }
    return true;
}

bool pack_activation(Fixture *f) {
    sm75_compressor_activation_lane_major_pack_kernel<<<16, 256>>>(
        f->packed_x, f->x);
    return cuda_ok(cudaGetLastError(), "activation pack launch");
}

bool launch_arm(Fixture *f, Arm arm) {
    float *out_kv = f->output.out_kv.data();
    float *out_score = f->output.out_score.data();
    float *state_kv = f->output.state_kv.data();
    float *state_score = f->output.state_score.data();
    if (arm == Arm::Control) {
        sm75_compressor_pair_control_kernel<<<kWidth, 32>>>(
            out_kv, out_score, state_kv, state_score,
            f->weight_kv, f->weight_score, f->x, f->ape, f->pos);
    } else if (arm == Arm::CanonicalStaged) {
        sm75_compressor_pair_canonical_staged_kernel<<<kWidth, 32>>>(
            out_kv, out_score, state_kv, state_score,
            f->weight_kv, f->weight_score, f->x, f->ape, f->pos);
    } else if (arm == Arm::LaneMajor) {
        sm75_compressor_pair_lane_major_kernel<<<kWidth, 32>>>(
            out_kv, out_score, state_kv, state_score,
            f->packed_kv, f->packed_score, f->packed_x, f->ape, f->pos);
    } else {
        sm75_compressor_pair_lane_major_two_warp_kernel<<<kWidth, 64>>>(
            out_kv, out_score, state_kv, state_score,
            f->packed_kv, f->packed_score, f->packed_x, f->ape, f->pos);
    }
    return cuda_ok(cudaGetLastError(), "projection launch");
}

uint16_t patterned_half(uint64_t i, uint32_t salt) {
    static const uint16_t pattern[] = {
        0x3000u, 0xb400u, 0x3800u, 0xb800u,
        0x3a00u, 0xba00u, 0x3400u, 0xb000u,
    };
    return pattern[(i * 5u + salt) & 7u];
}

bool fixture_init(Fixture *f) {
    const uint64_t weight_count = (uint64_t)kWidth * kIn;
    std::vector<uint16_t> h_kv((size_t)weight_count);
    std::vector<uint16_t> h_score((size_t)weight_count);
    std::vector<uint16_t> h_ape((size_t)kRatio * kWidth);
    std::vector<float> h_x(kIn);
    for (uint64_t i = 0; i < weight_count; ++i) {
        h_kv[(size_t)i] = patterned_half(i, 1u);
        h_score[(size_t)i] = patterned_half(i, 3u);
    }
    for (uint64_t i = 0; i < h_ape.size(); ++i)
        h_ape[(size_t)i] = patterned_half(i, 5u);
    for (uint32_t i = 0; i < kIn; ++i) {
        const int v = (int)((i * 37u + 7u * 19u) % 251u) - 125;
        h_x[i] = (float)v / 512.0f;
    }
    const size_t weight_bytes = (size_t)weight_count * sizeof(__half);
    if (!cuda_ok(cudaMalloc((void **)&f->weight_kv, weight_bytes), "weight kv alloc") ||
        !cuda_ok(cudaMalloc((void **)&f->weight_score, weight_bytes), "weight score alloc") ||
        !cuda_ok(cudaMalloc((void **)&f->packed_kv, weight_bytes), "packed kv alloc") ||
        !cuda_ok(cudaMalloc((void **)&f->packed_score, weight_bytes), "packed score alloc") ||
        !cuda_ok(cudaMalloc((void **)&f->ape,
                            h_ape.size() * sizeof(__half)), "ape alloc") ||
        !cuda_ok(cudaMalloc((void **)&f->x, kIn * sizeof(float)), "x alloc") ||
        !cuda_ok(cudaMalloc((void **)&f->packed_x,
                            kIn * sizeof(float)), "packed x alloc") ||
        !outputs_alloc(&f->output)) return false;
    if (!cuda_ok(cudaMemcpy(f->weight_kv, h_kv.data(), weight_bytes,
                            cudaMemcpyHostToDevice), "weight kv copy") ||
        !cuda_ok(cudaMemcpy(f->weight_score, h_score.data(), weight_bytes,
                            cudaMemcpyHostToDevice), "weight score copy") ||
        !cuda_ok(cudaMemcpy(f->ape, h_ape.data(), h_ape.size() * sizeof(__half),
                            cudaMemcpyHostToDevice), "ape copy") ||
        !cuda_ok(cudaMemcpy(f->x, h_x.data(), kIn * sizeof(float),
                            cudaMemcpyHostToDevice), "x copy")) return false;
    const uint32_t blocks = (uint32_t)((weight_count + 255u) / 256u);
    sm75_compressor_weight_lane_major_pack_kernel<<<blocks, 256>>>(
        f->packed_kv, f->weight_kv);
    sm75_compressor_weight_lane_major_pack_kernel<<<blocks, 256>>>(
        f->packed_score, f->weight_score);
    return cuda_ok(cudaGetLastError(), "weight pack launch") &&
           pack_activation(f) && cuda_ok(cudaDeviceSynchronize(), "fixture sync");
}

void fixture_free(Fixture *f) {
    outputs_free(&f->output);
    cudaFree(f->packed_x);
    cudaFree(f->x);
    cudaFree(f->ape);
    cudaFree(f->packed_score);
    cudaFree(f->packed_kv);
    cudaFree(f->weight_score);
    cudaFree(f->weight_kv);
}

bool validate_pack(const Fixture &f) {
    const uint64_t count = (uint64_t)kWidth * kIn;
    std::vector<uint16_t> canonical((size_t)count);
    std::vector<uint16_t> packed((size_t)count);
    std::vector<float> x(kIn), packed_x(kIn);
    if (!cuda_ok(cudaMemcpy(canonical.data(), f.weight_kv,
                            canonical.size() * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost), "canonical read") ||
        !cuda_ok(cudaMemcpy(packed.data(), f.packed_kv,
                            packed.size() * sizeof(uint16_t),
                            cudaMemcpyDeviceToHost), "packed read") ||
        !cuda_ok(cudaMemcpy(x.data(), f.x, x.size() * sizeof(float),
                            cudaMemcpyDeviceToHost), "x read") ||
        !cuda_ok(cudaMemcpy(packed_x.data(), f.packed_x,
                            packed_x.size() * sizeof(float),
                            cudaMemcpyDeviceToHost), "packed x read")) return false;
    for (uint32_t row = 0; row < kWidth; ++row) {
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
            for (uint32_t step = 0; step < kChunk; ++step) {
                const uint64_t src = (uint64_t)row * kIn + lane * kChunk + step;
                const uint64_t dst = (uint64_t)row * kIn + step * kLanes + lane;
                if (canonical[(size_t)src] != packed[(size_t)dst]) return false;
            }
        }
    }
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        for (uint32_t step = 0; step < kChunk; ++step) {
            if (std::memcmp(&x[lane * kChunk + step],
                            &packed_x[step * kLanes + lane],
                            sizeof(float)) != 0) return false;
        }
    }
    return true;
}

bool run_correctness(Fixture *f) {
    if (!validate_pack(*f)) {
        std::fprintf(stderr, "error: lane-major repack roundtrip failed\n");
        return false;
    }
    if (!outputs_reset(&f->output) || !launch_arm(f, Arm::Control) ||
        !cuda_ok(cudaDeviceSynchronize(), "control correctness sync")) return false;
    const std::vector<unsigned char> reference = outputs_snapshot(f->output);
    if (reference.empty() || !canaries_ok(f->output, reference) ||
        !written_regions_changed(f->output, reference)) {
        std::fprintf(stderr,
                     "error: control did not overwrite every required region\n");
        return false;
    }
    for (Arm arm : {Arm::CanonicalStaged, Arm::LaneMajor, Arm::TwoWarp}) {
        if (!outputs_reset(&f->output) || !pack_activation(f) ||
            !launch_arm(f, arm) ||
            !cuda_ok(cudaDeviceSynchronize(), "candidate correctness sync")) return false;
        const std::vector<unsigned char> candidate = outputs_snapshot(f->output);
        if (candidate != reference || !canaries_ok(f->output, candidate)) {
            std::fprintf(stderr, "error: %s is not bit-exact\n", arm_name(arm));
            return false;
        }
        std::printf("correctness_result_%s=bit-exact\n", arm_name(arm));
    }
    std::printf("weight_layout_size_neutral=yes\n"
                "weight_repack_roundtrip=byte-exact\n"
                "activation_repack_roundtrip=byte-exact\n"
                "control_required_regions=overwritten\n"
                "correctness_full_output_state=bit-exact\n"
                "correctness_canaries=ok\n");
    return true;
}

bool time_sequence(Fixture *f, Arm arm, bool include_pack,
                   uint32_t launches, float *per_launch_ms) {
    cudaEvent_t start = nullptr, stop = nullptr;
    if (!cuda_ok(cudaEventCreate(&start), "timer start create") ||
        !cuda_ok(cudaEventCreate(&stop), "timer stop create") ||
        !cuda_ok(cudaEventRecord(start), "timer start record")) return false;
    for (uint32_t i = 0; i < launches; ++i) {
        if (include_pack && arm != Arm::Control && !pack_activation(f)) return false;
        if (!launch_arm(f, arm)) return false;
    }
    if (!cuda_ok(cudaEventRecord(stop), "timer stop record") ||
        !cuda_ok(cudaEventSynchronize(stop), "timer stop sync")) return false;
    float elapsed = 0.0f;
    const bool ok = cuda_ok(cudaEventElapsedTime(&elapsed, start, stop),
                            "timer elapsed");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    *per_launch_ms = elapsed / launches;
    return ok;
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2u];
}

bool run_benchmark(Fixture *f, uint32_t rounds, uint32_t launches) {
    std::vector<float> control(rounds), staged(rounds);
    std::vector<float> lane(rounds), lane_inclusive(rounds);
    std::vector<float> two(rounds), two_inclusive(rounds);
    for (uint32_t warm = 0; warm < 5u; ++warm) {
        if (!launch_arm(f, Arm::Control) ||
            !launch_arm(f, Arm::CanonicalStaged) ||
            !launch_arm(f, Arm::LaneMajor) ||
            !launch_arm(f, Arm::TwoWarp)) return false;
    }
    if (!cuda_ok(cudaDeviceSynchronize(), "benchmark warmup")) return false;
    for (uint32_t round = 0; round < rounds; ++round) {
        const bool reverse = (round & 1u) != 0u;
        if (!reverse) {
            if (!time_sequence(f, Arm::Control, false, launches, &control[round]) ||
                !time_sequence(f, Arm::CanonicalStaged, false, launches,
                               &staged[round]) ||
                !time_sequence(f, Arm::LaneMajor, false, launches, &lane[round]) ||
                !time_sequence(f, Arm::TwoWarp, false, launches, &two[round]) ||
                !time_sequence(f, Arm::LaneMajor, true, launches, &lane_inclusive[round]) ||
                !time_sequence(f, Arm::TwoWarp, true, launches, &two_inclusive[round])) return false;
        } else {
            if (!time_sequence(f, Arm::TwoWarp, true, launches, &two_inclusive[round]) ||
                !time_sequence(f, Arm::LaneMajor, true, launches, &lane_inclusive[round]) ||
                !time_sequence(f, Arm::TwoWarp, false, launches, &two[round]) ||
                !time_sequence(f, Arm::LaneMajor, false, launches, &lane[round]) ||
                !time_sequence(f, Arm::CanonicalStaged, false, launches,
                               &staged[round]) ||
                !time_sequence(f, Arm::Control, false, launches, &control[round])) return false;
        }
    }
    std::vector<float> staged_ratio(rounds), lane_ratio(rounds);
    std::vector<float> lane_inc_ratio(rounds);
    std::vector<float> two_ratio(rounds), two_inc_ratio(rounds);
    for (uint32_t i = 0; i < rounds; ++i) {
        staged_ratio[i] = control[i] / staged[i];
        lane_ratio[i] = control[i] / lane[i];
        lane_inc_ratio[i] = control[i] / lane_inclusive[i];
        two_ratio[i] = control[i] / two[i];
        two_inc_ratio[i] = control[i] / two_inclusive[i];
    }
    std::printf("benchmark_scope=width1024-paired-f16-projection-state-store\n"
                "weight_repack_scope=one-time-excluded\n"
                "activation_pack_scope=per-token-included-separately\n"
                "timing_rounds=%u\ntiming_launches=%u\n"
                "control_median_ms=%.9g\n"
                "canonical_staged_median_ms=%.9g\n"
                "canonical_staged_speedup=%.9g\n"
                "lane_major_consumer_median_ms=%.9g\n"
                "lane_major_consumer_speedup=%.9g\n"
                "lane_major_inclusive_median_ms=%.9g\n"
                "lane_major_inclusive_speedup=%.9g\n"
                "two_warp_consumer_median_ms=%.9g\n"
                "two_warp_consumer_speedup=%.9g\n"
                "two_warp_inclusive_median_ms=%.9g\n"
                "two_warp_inclusive_speedup=%.9g\n",
                rounds, launches, median(control), median(staged),
                median(staged_ratio), median(lane), median(lane_ratio),
                median(lane_inclusive),
                median(lane_inc_ratio), median(two), median(two_ratio),
                median(two_inclusive), median(two_inc_ratio));
    return run_correctness(f) &&
           std::printf("post_timing_exactness=bit-exact\n"
                       "post_timing_canaries=ok\n"
                       "benchmark_status=ok\n") > 0;
}

template <typename Kernel>
bool print_resources(const char *name, Kernel kernel, int block_size) {
    cudaFuncAttributes attr;
    int active = 0;
    if (!cuda_ok(cudaFuncGetAttributes(&attr, kernel), "resource query") ||
        !cuda_ok(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                     &active, kernel, block_size, 0), "occupancy query")) return false;
    std::printf("resources,variant=%s,registers=%d,static_shared_bytes=%zu,"
                "local_bytes=%zu,max_threads=%d,active_blocks_per_sm=%d\n",
                name, attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes,
                attr.maxThreadsPerBlock, active);
    return true;
}

uint32_t parse_u32(const char *text, const char *name) {
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (!text[0] || !end || *end || value == 0u || value > UINT32_MAX) {
        std::fprintf(stderr, "error: invalid %s: %s\n", name, text);
        std::exit(2);
    }
    return (uint32_t)value;
}

int parse_device(const char *text) {
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!text[0] || !end || *end || value < 0 || value > INT32_MAX) {
        std::fprintf(stderr, "error: invalid device: %s\n", text);
        std::exit(2);
    }
    return (int)value;
}

}  // namespace

int main(int argc, char **argv) {
    int device = 0;
    uint32_t rounds = 9u, launches = 25u;
    bool correctness = true, benchmark = true;
    std::string profile;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) device = parse_device(argv[++i]);
        else if (arg == "--rounds" && i + 1 < argc) rounds = parse_u32(argv[++i], "rounds");
        else if (arg == "--launches" && i + 1 < argc) launches = parse_u32(argv[++i], "launches");
        else if (arg == "--correctness-only") { correctness = true; benchmark = false; }
        else if (arg == "--benchmark-only") { correctness = false; benchmark = true; }
        else if (arg == "--profile" && i + 1 < argc) {
            profile = argv[++i]; correctness = benchmark = false;
        } else {
            std::fprintf(stderr, "error: unknown or incomplete argument: %s\n", arg.c_str());
            return 2;
        }
    }
    if (!cuda_ok(cudaSetDevice(device), "select device")) return 1;
    cudaDeviceProp prop;
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties") ||
        prop.major != 7 || prop.minor != 5) {
        std::fprintf(stderr, "error: this diagnostic requires SM75\n");
        return 1;
    }
    Fixture fixture;
    if (!fixture_init(&fixture)) return 1;
    bool ok = print_resources("control", sm75_compressor_pair_control_kernel, 32) &&
              print_resources("canonical-staged",
                              sm75_compressor_pair_canonical_staged_kernel, 32) &&
              print_resources("lane-major", sm75_compressor_pair_lane_major_kernel, 32) &&
              print_resources("lane-major-two-warp",
                              sm75_compressor_pair_lane_major_two_warp_kernel, 64);
    if (ok && !profile.empty()) {
        Arm arm;
        if (profile == "control") arm = Arm::Control;
        else if (profile == "canonical-staged") arm = Arm::CanonicalStaged;
        else if (profile == "lane-major") arm = Arm::LaneMajor;
        else if (profile == "two-warp") arm = Arm::TwoWarp;
        else {
            std::fprintf(stderr, "error: unknown profile arm: %s\n", profile.c_str());
            ok = false;
            arm = Arm::Control;
        }
        for (uint32_t i = 0; ok && i < launches; ++i) ok = launch_arm(&fixture, arm);
        ok = ok && cuda_ok(cudaDeviceSynchronize(), "profile sync");
        if (ok) std::printf("profile_status=ok\n");
    }
    if (ok && correctness) ok = run_correctness(&fixture);
    if (ok && benchmark) ok = run_benchmark(&fixture, rounds | 1u, launches);
    if (ok) std::printf("harness_status=ok\n");
    fixture_free(&fixture);
    return ok ? 0 : 1;
}
