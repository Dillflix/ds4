// Linux x86-64, CUDA 13.2 host-side API interposer. No GPU work is launched here.
// Build against the installed NVIDIA headers; do not redeclare vendor structs.
// This is an INSTRUMENTED experiment, not a transparent or complete GPU trace.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cublas_v2.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

#if !defined(__linux__) || !defined(__x86_64__) || CUDART_VERSION != 13020
#error This interposer requires Linux x86-64 and the official CUDA 13.2 headers.
#endif
#if CUBLAS_VER_MAJOR != 13
#error This interposer requires the official cuBLAS 13 headers.
#endif
static_assert(sizeof(void *) == 8 && sizeof(size_t) == 8, "64-bit ABI required");
// CUDA 13.2 adds eight reserved longs after the familiar 24-byte prefix.
static_assert(sizeof(cudaPointerAttributes) == 88, "Unexpected CUDA pointer ABI");
static_assert(sizeof(dim3) == 12, "Unexpected CUDA dim3 ABI");

namespace {
using S = std::string;
pthread_mutex_t log_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_once_t log_once = PTHREAD_ONCE_INIT;
int log_fd = -1;
std::atomic<pid_t> owner_pid{0};
uint64_t record_seq = 0, written_bytes = 0;
constexpr uint64_t limit_bytes = 64ULL * 1024 * 1024;
std::atomic<uint64_t> next_call{1};
std::atomic<bool> launch_runtime_validated{false};
thread_local unsigned depth = 0;
pthread_mutex_t kernel_lock=PTHREAD_MUTEX_INITIALIZER;
struct KernelEntry { void **owner; const void *host; cudaKernel_t kernel; bool conversion; };
KernelEntry kernels[4096]{};
constexpr char conversion_name[]="_Z17f32_to_f16_kernelP6__halfPKfm";

[[noreturn]] void fatal(const char *message) {
    // Fixed diagnostics only: never dump the environment or tensor data.
    (void)!write(STDERR_FILENO, "runtime-contract-trace: ", 24);
    (void)!write(STDERR_FILENO, message, strlen(message));
    (void)!write(STDERR_FILENO, "\n", 1);
    _exit(125);
}
void initialize_log() {
    const char *path=getenv("DS4_RUNTIME_TRACE_LOG");
    if (!path || path[0]!='/') fatal("DS4_RUNTIME_TRACE_LOG must name a new absolute file");
    owner_pid=getpid();
    log_fd=open(path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
    if (log_fd<0) fatal("cannot exclusively create trace file");
    struct stat st{};
    if (fstat(log_fd,&st) || !S_ISREG(st.st_mode) || st.st_uid!=geteuid() ||
        st.st_nlink!=1 || (st.st_mode & 0777)!=0600) fatal("unsafe trace file");
}
S number(uint64_t n) { return std::to_string(n); }
S integer(long long n) { return std::to_string(n); }
S pointer(const void *p) {
    char b[32]; snprintf(b, sizeof b, "\"0x%llx\"", (unsigned long long)(uintptr_t)p);
    return b;
}
S ptrnum(CUdeviceptr p) { return pointer(reinterpret_cast<const void *>(uintptr_t(p))); }
S field(const char *name, const S &value) { return S("\"") + name + "\":" + value; }
S dims(dim3 d) { return "[" + number(d.x) + "," + number(d.y) + "," + number(d.z) + "]"; }
uint64_t clock_ns(clockid_t kind) {
    timespec t{};
    if (clock_gettime(kind, &t)) fatal("clock query failed");
    return uint64_t(t.tv_sec) * 1000000000ULL + uint64_t(t.tv_nsec);
}
void emit_locked(const char *event, const char *api, uint64_t call, const S &extra) {
    S line = "{\"schema\":\"ds4-runtime-contract-v1\",\"seq\":" + number(++record_seq) +
        ",\"event\":\"" + event + "\",\"api\":\"" + api + "\",\"call_id\":" + number(call) +
        ",\"pid\":" + integer(owner_pid) + ",\"tid\":" + integer(syscall(SYS_gettid)) +
        ",\"monotonic_ns\":" + number(clock_ns(CLOCK_MONOTONIC)) +
        ",\"realtime_ns\":" + number(clock_ns(CLOCK_REALTIME)) + extra + "}\n";
    if (line.size() > 65536 || written_bytes + line.size() > limit_bytes)
        fatal("trace byte limit exceeded; workload stopped before next API");
    const char *p = line.data(); size_t remain = line.size();
    while (remain) {
        ssize_t n = write(log_fd, p, remain);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) fatal("trace write failed; workload stopped");
        p += n; remain -= size_t(n);
    }
    written_bytes += line.size();
}
void emit(const char *event,const char *api,uint64_t call,const S &extra) {
    // A linked CUDA DSO can register kernels before this DSO's constructor.
    // Initialization is host-only, idempotent and safe at that earlier point.
    if (owner_pid.load() && getpid()!=owner_pid.load()) fatal("unsupported fork");
    if (pthread_once(&log_once,initialize_log)) fatal("log initialization failed");
    if (getpid()!=owner_pid.load()) fatal("unsupported fork");
    if (pthread_mutex_lock(&log_lock)) fatal("log mutex failed");
    if (log_fd<0 || getpid()!=owner_pid) fatal("missing log or unsupported fork");
    if (!record_seq) emit_locked("meta","trace_start",0,
        ",\"result\":{\"instrumented\":true,\"abi_version\":1,\"cuda_header_version\":13020,"
        "\"log_limit_bytes\":67108864,\"extra_synchronization\":false,\"constructor_cuda_calls\":0,"
        "\"kernel_argument_policy\":\"recognized-main-ELF-f32-to-f16-scalars-only\","
        "\"coverage\":\"interposed-host-api-only\",\"queries_may_surface_prior_async_errors\":true}");
    if (strcmp(api,"trace_start")) emit_locked(event,api,call,extra);
    if (pthread_mutex_unlock(&log_lock)) fatal("log mutex unlock failed");
}
struct Guard {
    bool active;
    Guard() : active(depth++ == 0) {}
    ~Guard() { --depth; }
};
struct Call {
    const char *api; uint64_t id;
    Call(const char *name, const S &args, uint64_t parent = 0) : api(name), id(next_call.fetch_add(1)) {
        emit("enter", api, id, ",\"parent_call_id\":" + number(parent) + ",\"args\":{" + args + "}");
    }
    void done(int status, const S &result = "") {
        emit("exit", api, id, ",\"status\":" + integer(status) + ",\"result\":{" + result + "}");
    }
};
template<typename F> F resolve(const char *name) {
    // No dlopen: the target must already link the matching real library.
    dlerror(); void *p = dlsym(RTLD_NEXT, name);
    if (!p || dlerror()) fatal("required real API symbol unavailable");
    return reinterpret_cast<F>(p);
}
template<typename F> F optional(const char *name) {
    dlerror(); void *p = dlsym(RTLD_NEXT, name);
    if (dlerror()) return nullptr;
    return reinterpret_cast<F>(p);
}
template<typename F, typename... Args> auto query(const char *api, uint64_t parent, F real, Args... args)
    -> decltype(real(args...)) {
    Call c(api,"\"tracer_added\":true",parent);
    auto status=real(args...); c.done(int(status));
    if (int(status) != 0) fatal("added metadata query failed; no further GPU API forwarded");
    return status;
}
template<typename F> F driver_entrypoint(const char *symbol,uint64_t parent) {
    // Documented runtime lookup sees an RTLD_LOCAL driver without dlopen or
    // promoting it into the global link scope. Never assume dlsym sees libcuda.
    using Fn=decltype(&::cudaGetDriverEntryPointByVersion);
    static Fn real=resolve<Fn>("cudaGetDriverEntryPointByVersion");
    void *entry=nullptr;
    cudaDriverEntryPointQueryResult result=cudaDriverEntryPointSymbolNotFound;
    Call c("trace.cudaGetDriverEntryPointByVersion",S("\"symbol\":\"") + symbol +
        "\",\"cuda_version\":13020,\"flags\":1,\"tracer_added\":true",parent);
    auto st=real(symbol,&entry,13020,cudaEnableLegacyStream,&result);
    c.done(st,field("query_result",integer(result)) + "," + field("function",pointer(entry)));
    if (st != cudaSuccess) fatal("added driver-entrypoint query failed; no GEMM forwarded");
    if (result != cudaDriverEntryPointSuccess || !entry) return nullptr;
    return reinterpret_cast<F>(entry);
}
void validate_runtime_before_launch(uint64_t parent) {
    // No process-wide lock around a real API. Concurrent first callers may each
    // make the same cheap version query; neither may forward an unsupported ABI.
    if (launch_runtime_validated.load()) return;
    using Fn=cudaError_t (*)(int *); static Fn real=resolve<Fn>("cudaRuntimeGetVersion");
    int version=0;
    query("trace.cudaRuntimeGetVersion",parent,real,&version);
    if (version!=CUDART_VERSION) fatal("unsupported runtime version; kernel launch not forwarded");
    launch_runtime_validated.store(true);
}
S current_device() {
    using Fn=cudaError_t (*)(int *); static Fn real=resolve<Fn>("cudaGetDevice");
    int device=-1; auto s=query("trace.cudaGetDevice",0,real,&device);
    S result=field("device_query_status",integer(s));
    if (s == cudaSuccess) result += "," + field("device",integer(device));
    auto getcontext=driver_entrypoint<PFN_cuCtxGetCurrent_v4000>("cuCtxGetCurrent",0);
    if (getcontext) {
        CUcontext context=nullptr;
        auto cs=query("trace.cuCtxGetCurrent",0,getcontext,&context);
        result += "," + field("context_query_status",integer(cs)) + "," + field("context",pointer(context));
    } else result += ",\"context_query_status\":null";
    return result;
}
S operand(const void *p, uint64_t parent) {
    using Attr = cudaError_t (*)(cudaPointerAttributes *, const void *);
    using Range = PFN_cuMemGetAddressRange_v3020;
    static Attr attr = resolve<Attr>("cudaPointerGetAttributes");
    Range range = driver_entrypoint<Range>("cuMemGetAddressRange",parent);
    cudaPointerAttributes a{};
    Call aq("trace.cudaPointerGetAttributes",field("ptr",pointer(p)) + ",\"tracer_added\":true",parent);
    cudaError_t st = attr(&a, p);
    aq.done(st);
    if (st != cudaSuccess) fatal("added pointer query failed; GEMM not forwarded");
    S s = field("attributes_status", integer(st));
    if (st == cudaSuccess) {
        s += "," + field("type", integer(a.type)) + "," + field("device", integer(a.device)) +
             "," + field("device_pointer", pointer(a.devicePointer)) +
             "," + field("host_pointer", pointer(a.hostPointer));
    }
    if (range && st == cudaSuccess && a.type == cudaMemoryTypeDevice) {
        CUdeviceptr base = 0; size_t bytes = 0;
        Call rq("trace.cuMemGetAddressRange_v2",field("ptr",pointer(p)) + ",\"tracer_added\":true",parent);
        CUresult rs = range(&base, &bytes, CUdeviceptr(uintptr_t(p)));
        rq.done(rs,rs == CUDA_SUCCESS ? field("base",ptrnum(base)) + "," + field("bytes",number(bytes)) : "");
        if (rs != CUDA_SUCCESS) fatal("added range query failed; GEMM not forwarded");
        s += "," + field("range_status", integer(rs));
        if (rs == CUDA_SUCCESS) s += "," + field("base", ptrnum(base)) + "," + field("bytes", number(bytes));
    } else s += ",\"range_status\":null";
    return "{" + s + "}";
}
S scalar(const void *p) {
    // Read only a known host-mode FP32 scalar, without directly dereferencing
    // arbitrary caller addresses. Failure is evidence, not a fallback memcpy.
    unsigned char bytes[4]{};
    iovec local{bytes, 4}, remote{const_cast<void *>(p), 4};
    ssize_t n = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
    S result = field("host_read_bytes", integer(n));
    if (n == 4) {
        char b[9]; for (int i = 0; i < 4; ++i) snprintf(b + 2*i, 3, "%02x", unsigned(bytes[i]));
        result += ",\"bits_hex\":\"" + S(b) + "\"";
    }
    return "{" + result + "}";
}
bool read_host(void *out,const void *from,size_t size) {
    iovec local{out,size},remote{const_cast<void *>(from),size};
    return process_vm_readv(getpid(),&local,1,&remote,1,0)==ssize_t(size);
}
bool main_elf_function(const void *host) {
    Dl_info info{}; struct stat exe{},object{};
    return dladdr(host,&info) && info.dli_fname && !stat("/proc/self/exe",&exe) &&
        !stat(info.dli_fname,&object) && exe.st_dev==object.st_dev && exe.st_ino==object.st_ino;
}
void register_kernel(void **owner,const void *host,const char *device_name) {
    char name[sizeof conversion_name]{};
    bool recognized=main_elf_function(host) && read_host(name,device_name,sizeof name) &&
        !memcmp(name,conversion_name,sizeof name);
    if (pthread_mutex_lock(&kernel_lock)) fatal("kernel table lock failed");
    KernelEntry *entry=nullptr;
    for (auto &k:kernels) {
        if (k.owner==owner && k.host==host) { entry=&k; break; }
        if (!k.host && !entry) entry=&k;
    }
    if (!entry) fatal("kernel registration table full; no silent coverage loss");
    *entry={owner,host,nullptr,recognized};
    if (pthread_mutex_unlock(&kernel_lock)) fatal("kernel table unlock failed");
}
void map_kernel(const void *host,cudaKernel_t handle) {
    if (pthread_mutex_lock(&kernel_lock)) fatal("kernel table lock failed");
    // Retire any older use of an opaque handle before assigning its new source.
    for (auto &k:kernels) if (k.kernel==handle) k.kernel=nullptr;
    for (auto &k:kernels) if (k.host==host) k.kernel=handle;
    if (pthread_mutex_unlock(&kernel_lock)) fatal("kernel table unlock failed");
}
void unregister_kernels(void **owner) {
    if (pthread_mutex_lock(&kernel_lock)) fatal("kernel table lock failed");
    for (auto &k:kernels) if (k.owner==owner) k={};
    if (pthread_mutex_unlock(&kernel_lock)) fatal("kernel table unlock failed");
}
void observe_conversion(uint64_t call,const void *host,cudaKernel_t handle,void **args) {
    bool recognized=false;
    if (pthread_mutex_lock(&kernel_lock)) fatal("kernel table lock failed");
    for (const auto &k:kernels)
        if (k.conversion && ((host && k.host==host) || (handle && k.kernel==handle))) recognized=true;
    if (pthread_mutex_unlock(&kernel_lock)) fatal("kernel table unlock failed");
    if (!recognized) return;
    void *slots[3]{}; void *dst=nullptr; const void *src=nullptr; size_t count=0;
    bool read=read_host(slots,args,sizeof slots) && read_host(&dst,slots[0],sizeof dst) &&
        read_host(&src,slots[1],sizeof src) && read_host(&count,slots[2],sizeof count);
    S result=S("\"recognized\":true,\"kernel_name\":\"") + conversion_name +
        "\",\"argument_read_status\":\"" + (read ? "complete" : "unreadable") + "\"";
    result += "," + current_device();
    if (read) result += "," + field("dst",pointer(dst)) + "," + field("src",pointer(src)) + "," + field("count",number(count));
    emit("observation","conversion_contract",call,",\"parent_call_id\":" + number(call) + ",\"result\":{" + result + "}");
}
void forward_marker(uint64_t call) {
    emit("observation","forward_to_real",call,",\"parent_call_id\":" + number(call) + ",\"result\":{}");
}
void observe_gemm(uint64_t id, cublasHandle_t h, const void *a, const void *b,
                  const void *c, const void *alpha, const void *beta, cublasComputeType_t ct) {
    using GD = cudaError_t (*)(int *);
    using PM = cublasStatus_t (*)(cublasHandle_t, cublasPointerMode_t *);
    using MM = cublasStatus_t (*)(cublasHandle_t, cublasMath_t *);
    using GS = cublasStatus_t (*)(cublasHandle_t, cudaStream_t *);
    using RV = cudaError_t (*)(int *);
    using BV = cublasStatus_t (*)(cublasHandle_t,int *);
    using GP = decltype(&::cublasGetProperty);
    static GD getdev = resolve<GD>("cudaGetDevice");
    static PM getpm = resolve<PM>("cublasGetPointerMode_v2");
    static MM getmm = resolve<MM>("cublasGetMathMode");
    static GS getstream = resolve<GS>("cublasGetStream_v2");
    static RV runtimeversion = resolve<RV>("cudaRuntimeGetVersion");
    static BV blasversion = resolve<BV>("cublasGetVersion_v2");
    static GP blasproperty = resolve<GP>("cublasGetProperty");
    int device = -1; cudaStream_t stream = nullptr;
    cublasPointerMode_t pm{}; cublasMath_t mm{};
    int ds = query("trace.cudaGetDevice",id,getdev,&device);
    int ps = query("trace.cublasGetPointerMode_v2",id,getpm,h,&pm);
    int ms = query("trace.cublasGetMathMode",id,getmm,h,&mm);
    int ss = query("trace.cublasGetStream_v2",id,getstream,h,&stream);
    int rv=0,bv=0,major=0;
    int rvs=query("trace.cudaRuntimeGetVersion",id,runtimeversion,&rv);
    int bvs=query("trace.cublasGetVersion_v2",id,blasversion,h,&bv);
    int majors=query("trace.cublasGetProperty",id,blasproperty,MAJOR_VERSION,&major);
    auto getcontext=driver_entrypoint<PFN_cuCtxGetCurrent_v4000>("cuCtxGetCurrent",id);
    CUcontext context=nullptr;
    S context_result=",\"context_query_status\":null";
    if (getcontext) {
        auto cs=query("trace.cuCtxGetCurrent",id,getcontext,&context);
        context_result="," + field("context_query_status",integer(cs)) + "," + field("context",pointer(context));
    }
    S handle = field("pointer_mode_query_status", integer(ps)) + "," + field("math_mode_query_status", integer(ms)) +
               "," + field("stream_query_status", integer(ss));
    if (!ps) handle += "," + field("pointer_mode", integer(pm));
    if (!ms) handle += "," + field("math_mode", integer(mm));
    if (!ss) handle += "," + field("stream", pointer(stream));
    S result = field("device_query_status", integer(ds));
    result += context_result;
    result += "," + field("runtime_version_status",integer(rvs)) + "," + field("runtime_version",integer(rv)) +
              "," + field("cublas_version_status",integer(bvs)) + "," + field("cublas_version",integer(bv)) +
              "," + field("cublas_major_status",integer(majors)) + "," + field("cublas_major",integer(major));
    if (!ds) result += "," + field("device", integer(device));
    result += ",\"handle\":{" + handle + "},\"operands\":{\"A\":" + operand(a,id) +
              ",\"B\":" + operand(b,id) + ",\"C\":" + operand(c,id) + "}";
    if (!ps && pm == CUBLAS_POINTER_MODE_HOST &&
        (ct == CUBLAS_COMPUTE_32F || ct == CUBLAS_COMPUTE_32F_PEDANTIC))
        result += ",\"scalars\":{\"alpha\":" + scalar(alpha) + ",\"beta\":" + scalar(beta) + "}";
    emit("observation", "gemm_contract", id,
         ",\"parent_call_id\":" + number(id) + ",\"result\":{" + result + "}");
    // A failed metadata query can surface an earlier async failure. Do not
    // mislabel it as invalid memory or silently continue toward another GEMM.
    if (rvs || rv != CUDART_VERSION || bvs || majors || major != 13)
        fatal("runtime/cuBLAS ABI query failed or unsupported version; GEMM not forwarded");
}
} // namespace

#pragma GCC visibility push(default)
extern "C" __attribute__((visibility("default"))) int ds4_runtime_trace_abi_version() { return 1; }
__attribute__((constructor)) static void trace_start() {
    emit("meta", "trace_start", 0, "");
}
__attribute__((destructor)) static void trace_end() {
    if (log_fd >= 0 && getpid() == owner_pid) {
        emit("meta", "trace_end", 0,
             ",\"status\":0,\"result\":{\"meaning\":\"finalizer-reached-not-guaranteed-last-record\"}");
        // Another linked DSO can unregister kernels later in loader teardown.
        // Keep this one CLOEXEC descriptor until the OS closes it at process
        // exit, so late teardown is recorded rather than causing a fake fault.
    }
}

#define STR1(x) #x
#define STR(x) STR1(x)
#define WRAP(ret, name, params, pass, args) \
extern "C" ret name params { \
    using Fn = ret (*) params; static Fn real = resolve<Fn>(STR(name)); \
    Guard guard; if (!guard.active) return real pass; \
    Call call(STR(name), args); auto status = real pass; call.done(int(status)); return status; \
}

extern "C" cudaError_t cudaMalloc(void **p, size_t size) {
    using Fn = cudaError_t (*)(void **, size_t); static Fn real = resolve<Fn>("cudaMalloc");
    Guard g; if (!g.active) return real(p, size);
    Call c("cudaMalloc", field("size", number(size)) + "," + current_device()); auto s = real(p, size);
    c.done(s, s == cudaSuccess && p ? field("ptr", pointer(*p)) : ""); return s;
}
extern "C" cudaError_t cudaMallocManaged(void **p, size_t size, unsigned flags) {
    using Fn = cudaError_t (*)(void **, size_t, unsigned); static Fn real = resolve<Fn>("cudaMallocManaged");
    Guard g; if (!g.active) return real(p, size, flags);
    Call c("cudaMallocManaged", field("size", number(size)) + "," + field("flags", number(flags)) + "," + current_device());
    auto s = real(p, size, flags); c.done(s, s == cudaSuccess && p ? field("ptr", pointer(*p)) : ""); return s;
}
WRAP(cudaError_t, cudaFree, (void *p), (p), field("ptr", pointer(p)))
WRAP(cudaError_t, cudaMemcpy, (void *dst, const void *src, size_t size, cudaMemcpyKind kind),
     (dst, src, size, kind), field("dst", pointer(dst)) + "," + field("src", pointer(src)) + "," + field("size", number(size)) + "," + field("kind", integer(kind)))
WRAP(cudaError_t, cudaMemcpyAsync, (void *dst, const void *src, size_t size, cudaMemcpyKind kind, cudaStream_t stream),
     (dst, src, size, kind, stream), field("dst", pointer(dst)) + "," + field("src", pointer(src)) + "," + field("size", number(size)) + "," + field("kind", integer(kind)) + "," + field("stream", pointer(stream)))
WRAP(cudaError_t, cudaMemset, (void *p, int value, size_t size), (p, value, size),
     field("ptr", pointer(p)) + "," + field("value", integer(value)) + "," + field("size", number(size)))
WRAP(cudaError_t, cudaMemsetAsync, (void *p, int value, size_t size, cudaStream_t stream), (p, value, size, stream),
     field("ptr", pointer(p)) + "," + field("value", integer(value)) + "," + field("size", number(size)) + "," + field("stream", pointer(stream)))
WRAP(cudaError_t, cudaSetDevice, (int device), (device), field("device", integer(device)))
WRAP(cudaError_t, cudaGetLastError, (void), (), "")
WRAP(cudaError_t, cudaPeekAtLastError, (void), (), "")
WRAP(cudaError_t, cudaDeviceSynchronize, (void), (), current_device())
WRAP(cudaError_t, cudaStreamSynchronize, (cudaStream_t stream), (stream), field("stream", pointer(stream)))
WRAP(cudaError_t, cudaStreamQuery, (cudaStream_t stream), (stream), field("stream", pointer(stream)))
WRAP(cudaError_t, cudaStreamWaitEvent, (cudaStream_t stream, cudaEvent_t event, unsigned flags), (stream,event,flags),
     field("stream", pointer(stream)) + "," + field("event", pointer(event)) + "," + field("flags", number(flags)))
WRAP(cudaError_t, cudaEventRecord, (cudaEvent_t event, cudaStream_t stream), (event,stream),
     field("event", pointer(event)) + "," + field("stream", pointer(stream)))
WRAP(cudaError_t, cudaEventSynchronize, (cudaEvent_t event), (event), field("event", pointer(event)))
WRAP(cudaError_t, cudaEventQuery, (cudaEvent_t event), (event), field("event", pointer(event)))
WRAP(cudaError_t, cudaEventDestroy, (cudaEvent_t event), (event), field("event", pointer(event)))
WRAP(cudaError_t, cudaStreamDestroy, (cudaStream_t stream), (stream), field("stream", pointer(stream)))
extern "C" cudaError_t cudaLaunchKernel(const void *function,dim3 grid,dim3 block,void **args,size_t shared,cudaStream_t stream) {
    using Fn=cudaError_t (*)(const void *,dim3,dim3,void **,size_t,cudaStream_t);
    static Fn real=resolve<Fn>("cudaLaunchKernel");
    Guard g; if (!g.active) return real(function,grid,block,args,shared,stream);
    Call c("cudaLaunchKernel",field("function",pointer(function)) + "," + field("grid",dims(grid)) + "," +
        field("block",dims(block)) + "," + field("args_pointer",pointer(args)) + "," +
        field("shared_bytes",number(shared)) + "," + field("stream",pointer(stream)));
    validate_runtime_before_launch(c.id); observe_conversion(c.id,function,nullptr,args); forward_marker(c.id);
    auto s=real(function,grid,block,args,shared,stream); c.done(s); return s;
}

// The frozen CUDA 13.2 ELF uses these compiler-runtime entrypoints instead of
// cudaLaunchKernel. Exact prototypes verified in official CRT 13.2.78:
// crt/device_functions.h:2932-2947 and crt/host_runtime.h:90-95. The host-only
// build script validates those installed declarations before compilation.
// Do not generalize this private ABI to other CUDA releases.
WRAP(unsigned, __cudaPushCallConfiguration,
     (dim3 grid, dim3 block, size_t shared, struct CUstream_st *stream),
     (grid,block,shared,stream), field("grid",dims(grid)) + "," + field("block",dims(block)) +
     "," + field("shared_bytes",number(shared)) + "," + field("stream",pointer(stream)))
extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *grid, dim3 *block, size_t *shared, void *stream) {
    using Fn = cudaError_t (*)(dim3 *,dim3 *,size_t *,void *);
    static Fn real=resolve<Fn>("__cudaPopCallConfiguration");
    Guard g; if (!g.active) return real(grid,block,shared,stream);
    Call c("__cudaPopCallConfiguration", ""); auto s=real(grid,block,shared,stream);
    S result;
    if (s == cudaSuccess && grid && block && shared && stream)
        result=field("grid",dims(*grid)) + "," + field("block",dims(*block)) + "," +
            field("shared_bytes",number(*shared)) + "," + field("stream",pointer(*static_cast<cudaStream_t *>(stream)));
    c.done(s,result); return s;
}
extern "C" cudaError_t __cudaGetKernel(cudaKernel_t *kernel, const void *function) {
    using Fn=cudaError_t (*)(cudaKernel_t *,const void *);
    static Fn real=resolve<Fn>("__cudaGetKernel");
    Guard g; if (!g.active) return real(kernel,function);
    Dl_info info{}; S args=field("function",pointer(function));
    if (dladdr(function,&info) && info.dli_fbase)
        args += "," + field("module_base",pointer(info.dli_fbase)) + "," +
            field("function_offset",number(uintptr_t(function)-uintptr_t(info.dli_fbase)));
    Call c("__cudaGetKernel",args); auto s=real(kernel,function);
    if (s == cudaSuccess && kernel) map_kernel(function,*kernel);
    c.done(s,s == cudaSuccess && kernel ? field("kernel",pointer(*kernel)) : ""); return s;
}
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel,dim3 grid,dim3 block,void **args,size_t shared,cudaStream_t stream) {
    using Fn=cudaError_t (*)(cudaKernel_t,dim3,dim3,void **,size_t,cudaStream_t);
    static Fn real=resolve<Fn>("__cudaLaunchKernel");
    Guard g; if (!g.active) return real(kernel,grid,block,args,shared,stream);
    Call c("__cudaLaunchKernel",field("kernel",pointer(kernel)) + "," + field("grid",dims(grid)) + "," +
        field("block",dims(block)) + "," + field("args_pointer",pointer(args)) + "," +
        field("shared_bytes",number(shared)) + "," + field("stream",pointer(stream)));
    validate_runtime_before_launch(c.id); observe_conversion(c.id,nullptr,kernel,args); forward_marker(c.id);
    auto s=real(kernel,grid,block,args,shared,stream); c.done(s); return s;
}
// CRT host_runtime.h:246-257,197-199. Registration is not a GPU launch.
extern "C" void __cudaRegisterFunction(void **owner,const char *host,char *device_function,const char *device_name,
        int limit,uint3 *tid,uint3 *bid,dim3 *block,dim3 *grid,int *warp) {
    using Fn=void (*)(void **,const char *,char *,const char *,int,uint3 *,uint3 *,dim3 *,dim3 *,int *);
    static Fn real=resolve<Fn>("__cudaRegisterFunction");
    Guard g;
    if (!g.active) { real(owner,host,device_function,device_name,limit,tid,bid,block,grid,warp); return; }
    Call c("__cudaRegisterFunction",field("owner",pointer(owner)) + "," + field("function",pointer(host)));
    register_kernel(owner,host,device_name);
    real(owner,host,device_function,device_name,limit,tid,bid,block,grid,warp); c.done(0);
}
extern "C" void __cudaUnregisterFatBinary(void **owner) {
    using Fn=void (*)(void **); static Fn real=resolve<Fn>("__cudaUnregisterFatBinary");
    Guard g; if (!g.active) { real(owner); return; }
    Call c("__cudaUnregisterFatBinary",field("owner",pointer(owner)));
    unregister_kernels(owner); real(owner); c.done(0);
}

#define CREATE_WRAPPER(name, type, params, pass, argjson, out) \
extern "C" cudaError_t name params { \
    using Fn = cudaError_t (*) params; static Fn real = resolve<Fn>(STR(name)); \
    Guard g; if (!g.active) return real pass; Call c(STR(name), argjson); auto s = real pass; \
    c.done(s, s == cudaSuccess && out ? field(type, pointer(*out)) : ""); return s; \
}
CREATE_WRAPPER(cudaStreamCreate, "stream", (cudaStream_t *p), (p), "", p)
CREATE_WRAPPER(cudaStreamCreateWithFlags, "stream", (cudaStream_t *p, unsigned flags), (p,flags), field("flags",number(flags)), p)
CREATE_WRAPPER(cudaStreamCreateWithPriority, "stream", (cudaStream_t *p, unsigned flags, int priority), (p,flags,priority), field("flags",number(flags)) + "," + field("priority",integer(priority)), p)
CREATE_WRAPPER(cudaEventCreate, "event", (cudaEvent_t *p), (p), "", p)
CREATE_WRAPPER(cudaEventCreateWithFlags, "event", (cudaEvent_t *p, unsigned flags), (p,flags), field("flags",number(flags)), p)
CREATE_WRAPPER(cudaMallocHost, "ptr", (void **p, size_t size), (p,size), field("size",number(size)), p)
CREATE_WRAPPER(cudaHostAlloc, "ptr", (void **p, size_t size, unsigned flags), (p,size,flags), field("size",number(size)) + "," + field("flags",number(flags)), p)
WRAP(cudaError_t, cudaFreeHost, (void *p), (p), field("ptr", pointer(p)))
WRAP(cudaError_t, cudaHostRegister, (void *p, size_t size, unsigned flags), (p,size,flags), field("ptr",pointer(p)) + "," + field("size",number(size)) + "," + field("flags",number(flags)))
WRAP(cudaError_t, cudaHostUnregister, (void *p), (p), field("ptr",pointer(p)))

extern "C" cublasStatus_t cublasCreate_v2(cublasHandle_t *p) {
    using Fn = cublasStatus_t (*)(cublasHandle_t *); static Fn real = resolve<Fn>("cublasCreate_v2");
    Guard g; if (!g.active) return real(p); Call c("cublasCreate_v2", ""); auto s = real(p);
    c.done(s, s == CUBLAS_STATUS_SUCCESS && p ? field("handle",pointer(*p)) : ""); return s;
}
WRAP(cublasStatus_t, cublasDestroy_v2, (cublasHandle_t h), (h), field("handle",pointer(h)))
WRAP(cublasStatus_t, cublasSetMathMode, (cublasHandle_t h, cublasMath_t mode), (h,mode), field("handle",pointer(h)) + "," + field("math_mode",integer(mode)))
WRAP(cublasStatus_t, cublasSetStream_v2, (cublasHandle_t h, cudaStream_t stream), (h,stream), field("handle",pointer(h)) + "," + field("stream",pointer(stream)))
WRAP(cublasStatus_t, cublasSetPointerMode_v2, (cublasHandle_t h, cublasPointerMode_t mode), (h,mode), field("handle",pointer(h)) + "," + field("pointer_mode",integer(mode)))
WRAP(cublasStatus_t, cublasSetWorkspace, (cublasHandle_t h, void *workspace, size_t size), (h,workspace,size), field("handle",pointer(h)) + "," + field("workspace",pointer(workspace)) + "," + field("size",number(size)))
extern "C" cublasStatus_t cublasGetMathMode(cublasHandle_t h, cublasMath_t *mode) {
    using Fn = cublasStatus_t (*)(cublasHandle_t,cublasMath_t *); static Fn real = resolve<Fn>("cublasGetMathMode");
    Guard g; if (!g.active) return real(h,mode); Call c("cublasGetMathMode",field("handle",pointer(h))); auto s=real(h,mode);
    c.done(s,s == CUBLAS_STATUS_SUCCESS && mode ? field("math_mode",integer(*mode)) : ""); return s;
}

namespace {
S gemm_args(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
            const void *alpha,const void *a,cudaDataType_t at,int lda,const void *b,cudaDataType_t bt,int ldb,
            const void *beta,void *c,cudaDataType_t ct,int ldc,cublasComputeType_t compute,cublasGemmAlgo_t algo) {
    return field("handle",pointer(h)) + "," + field("transa",integer(ta)) + "," + field("transb",integer(tb)) +
        "," + field("m",integer(m)) + "," + field("n",integer(n)) + "," + field("k",integer(k)) +
        "," + field("alpha_ptr",pointer(alpha)) + "," + field("beta_ptr",pointer(beta)) +
        "," + field("A",pointer(a)) + "," + field("B",pointer(b)) + "," + field("C",pointer(c)) +
        "," + field("atype",integer(at)) + "," + field("btype",integer(bt)) + "," + field("ctype",integer(ct)) +
        "," + field("lda",integer(lda)) + "," + field("ldb",integer(ldb)) + "," + field("ldc",integer(ldc)) +
        "," + field("compute_type",integer(compute)) + "," + field("algorithm",integer(algo));
}
}
#define GEMM_PARAMS cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,const void *alpha,const void *a,cudaDataType_t at,int lda,const void *b,cudaDataType_t bt,int ldb,const void *beta,void *c,cudaDataType_t ct,int ldc,cublasComputeType_t compute,cublasGemmAlgo_t algo
#define GEMM_PASS h,ta,tb,m,n,k,alpha,a,at,lda,b,bt,ldb,beta,c,ct,ldc,compute,algo
extern "C" cublasStatus_t cublasGemmEx(GEMM_PARAMS) {
    using Fn = cublasStatus_t (*)(GEMM_PARAMS); static Fn real = resolve<Fn>("cublasGemmEx");
    Guard g; if (!g.active) return real(GEMM_PASS);
    Call call("cublasGemmEx",gemm_args(GEMM_PASS)); observe_gemm(call.id,h,a,b,c,alpha,beta,compute);
    forward_marker(call.id); auto status=real(GEMM_PASS); call.done(status); return status;
}
#define BATCH_PARAMS cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,const void *alpha,const void *a,cudaDataType_t at,int lda,long long int sa,const void *b,cudaDataType_t bt,int ldb,long long int sb,const void *beta,void *c,cudaDataType_t ct,int ldc,long long int sc,int batches,cublasComputeType_t compute,cublasGemmAlgo_t algo
#define BATCH_PASS h,ta,tb,m,n,k,alpha,a,at,lda,sa,b,bt,ldb,sb,beta,c,ct,ldc,sc,batches,compute,algo
extern "C" cublasStatus_t cublasGemmStridedBatchedEx(BATCH_PARAMS) {
    using Fn = cublasStatus_t (*)(BATCH_PARAMS); static Fn real = resolve<Fn>("cublasGemmStridedBatchedEx");
    Guard g; if (!g.active) return real(BATCH_PASS);
    Call call("cublasGemmStridedBatchedEx",gemm_args(GEMM_PASS) + "," + field("stride_a",integer(sa)) +
        "," + field("stride_b",integer(sb)) + "," + field("stride_c",integer(sc)) + "," + field("batch_count",integer(batches)));
    observe_gemm(call.id,h,a,b,c,alpha,beta,compute);
    forward_marker(call.id); auto status=real(BATCH_PASS); call.done(status); return status;
}

// Other actual cuBLAS imports of the frozen binary. The non-Ex API does not
// expose an algorithm parameter; -1 below is a schema placeholder, explicitly
// marked non-explicit, not evidence of which internal kernel was selected.
#define SGEMM_PARAMS cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,const float *alpha,const float *a,int lda,const float *b,int ldb,const float *beta,float *c,int ldc
#define SGEMM_PASS h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc
extern "C" cublasStatus_t cublasSgemm_v2(SGEMM_PARAMS) {
    using Fn=cublasStatus_t (*)(SGEMM_PARAMS); static Fn real=resolve<Fn>("cublasSgemm_v2");
    Guard g; if (!g.active) return real(SGEMM_PASS);
    Call call("cublasSgemm_v2",gemm_args(h,ta,tb,m,n,k,alpha,a,CUDA_R_32F,lda,b,CUDA_R_32F,ldb,
        beta,c,CUDA_R_32F,ldc,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT) + ",\"algorithm_explicit\":false");
    observe_gemm(call.id,h,a,b,c,alpha,beta,CUBLAS_COMPUTE_32F);
    forward_marker(call.id); auto s=real(SGEMM_PASS); call.done(s); return s;
}
#define SGEMM_BATCH_PARAMS cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,const float *alpha,const float *a,int lda,long long int sa,const float *b,int ldb,long long int sb,const float *beta,float *c,int ldc,long long int sc,int batches
#define SGEMM_BATCH_PASS h,ta,tb,m,n,k,alpha,a,lda,sa,b,ldb,sb,beta,c,ldc,sc,batches
extern "C" cublasStatus_t cublasSgemmStridedBatched(SGEMM_BATCH_PARAMS) {
    using Fn=cublasStatus_t (*)(SGEMM_BATCH_PARAMS); static Fn real=resolve<Fn>("cublasSgemmStridedBatched");
    Guard g; if (!g.active) return real(SGEMM_BATCH_PASS);
    Call call("cublasSgemmStridedBatched",gemm_args(h,ta,tb,m,n,k,alpha,a,CUDA_R_32F,lda,b,CUDA_R_32F,ldb,
        beta,c,CUDA_R_32F,ldc,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT) + ",\"algorithm_explicit\":false," +
        field("stride_a",integer(sa)) + "," + field("stride_b",integer(sb)) + "," +
        field("stride_c",integer(sc)) + "," + field("batch_count",integer(batches)));
    observe_gemm(call.id,h,a,b,c,alpha,beta,CUBLAS_COMPUTE_32F);
    forward_marker(call.id); auto s=real(SGEMM_BATCH_PASS); call.done(s); return s;
}
#pragma GCC visibility pop
