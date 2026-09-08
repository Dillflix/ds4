// Host-only fake provider. MUST NEVER link to libcudart/cuBLAS/libcuda.
// Used to verify interposer pass-through and log ownership without a GPU.
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <map>
#include <string>

namespace {
std::map<uintptr_t,size_t> allocations;
int device=0, gemms=0, launches=0, syncs=0;
cublasMath_t math_mode=CUBLAS_DEFAULT_MATH;
cublasPointerMode_t pointer_mode=CUBLAS_POINTER_MODE_HOST;
cudaStream_t selected_stream=nullptr;
}
extern "C" int sm75_mock_counts(int kind) { return kind==0 ? gemms : kind==1 ? launches : syncs; }
extern "C" cudaError_t cudaGetDevice(int *d) { *d=device; return cudaSuccess; }
extern "C" cudaError_t cudaSetDevice(int d) { device=d; return cudaSuccess; }
extern "C" cudaError_t cudaRuntimeGetVersion(int *v) { *v=CUDART_VERSION; return cudaSuccess; }
extern "C" cudaError_t cudaMalloc(void **p,size_t size) {
    *p=malloc(size); if (!*p) return cudaErrorMemoryAllocation;
    allocations[uintptr_t(*p)]=size; return cudaSuccess;
}
extern "C" cudaError_t cudaFree(void *p) { allocations.erase(uintptr_t(p)); free(p); return cudaSuccess; }
extern "C" cudaError_t cudaMemcpy(void *d,const void *s,size_t n,cudaMemcpyKind) { memcpy(d,s,n); return cudaSuccess; }
extern "C" cudaError_t cudaMemcpyAsync(void *d,const void *s,size_t n,cudaMemcpyKind,cudaStream_t) { memcpy(d,s,n); return cudaSuccess; }
extern "C" cudaError_t cudaDeviceSynchronize() { ++syncs; return cudaSuccess; }
extern "C" cudaError_t cudaPointerGetAttributes(cudaPointerAttributes *a,const void *p) {
    if (getenv("DS4_TRACE_MOCK_FAIL_POINTER_QUERY")) return cudaErrorInvalidValue;
    *a={}; a->type=cudaMemoryTypeDevice; a->device=device; a->devicePointer=const_cast<void *>(p); return cudaSuccess;
}
extern "C" CUresult cuMemGetAddressRange_v2(CUdeviceptr *base,size_t *bytes,CUdeviceptr p) {
    auto it=allocations.upper_bound(uintptr_t(p));
    if (it==allocations.begin()) return CUDA_ERROR_INVALID_VALUE;
    --it; if (p-it->first>=it->second) return CUDA_ERROR_INVALID_VALUE;
    *base=it->first; *bytes=it->second; return CUDA_SUCCESS;
}
extern "C" CUresult cuCtxGetCurrent(CUcontext *context) {
    *context=reinterpret_cast<CUcontext>(uintptr_t(0x9abc)); return CUDA_SUCCESS;
}
extern "C" cudaError_t cudaGetDriverEntryPointByVersion(const char *name,void **entry,unsigned version,
        unsigned long long flags,cudaDriverEntryPointQueryResult *result) {
    if (version!=13020 || flags!=cudaEnableLegacyStream) return cudaErrorInvalidValue;
    if (!strcmp(name,"cuMemGetAddressRange")) *entry=reinterpret_cast<void *>(&cuMemGetAddressRange_v2);
    else if (!strcmp(name,"cuCtxGetCurrent")) *entry=reinterpret_cast<void *>(&cuCtxGetCurrent);
    else { *entry=nullptr; *result=cudaDriverEntryPointSymbolNotFound; return cudaSuccess; }
    *result=cudaDriverEntryPointSuccess; return cudaSuccess;
}
extern "C" cublasStatus_t cublasCreate_v2(cublasHandle_t *h) { *h=reinterpret_cast<cublasHandle_t>(uintptr_t(0x1234)); return CUBLAS_STATUS_SUCCESS; }
extern "C" cublasStatus_t cublasDestroy_v2(cublasHandle_t) { return CUBLAS_STATUS_SUCCESS; }
extern "C" cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t,cublasPointerMode_t *m) { *m=pointer_mode; return CUBLAS_STATUS_SUCCESS; }
extern "C" cublasStatus_t cublasGetMathMode(cublasHandle_t,cublasMath_t *m) { *m=math_mode; return CUBLAS_STATUS_SUCCESS; }
extern "C" cublasStatus_t cublasSetMathMode(cublasHandle_t,cublasMath_t m) { math_mode=m; return CUBLAS_STATUS_SUCCESS; }
extern "C" cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t,cublasPointerMode_t m) { pointer_mode=m; return CUBLAS_STATUS_SUCCESS; }
extern "C" cublasStatus_t cublasSetStream_v2(cublasHandle_t,cudaStream_t s) { selected_stream=s; return CUBLAS_STATUS_SUCCESS; }
extern "C" cublasStatus_t cublasGetStream_v2(cublasHandle_t,cudaStream_t *s) { *s=selected_stream; return CUBLAS_STATUS_SUCCESS; }
extern "C" cublasStatus_t cublasGetVersion_v2(cublasHandle_t,int *v) { *v=CUBLAS_VERSION; return CUBLAS_STATUS_SUCCESS; }
extern "C" cublasStatus_t cublasGetProperty(libraryPropertyType p,int *v) { *v=p==MAJOR_VERSION ? 13 : 0; return CUBLAS_STATUS_SUCCESS; }
extern "C" cudaError_t cudaLaunchKernel(const void *,dim3 grid,dim3 block,void **,size_t shared,cudaStream_t stream) {
    if (grid.x!=2 || block.x!=32 || shared!=64 || stream!=selected_stream) return cudaErrorInvalidValue;
    ++launches; return cudaSuccess;
}
extern "C" void __cudaRegisterFunction(void **,const char *,char *,const char *,int,uint3 *,uint3 *,dim3 *,dim3 *,int *) {}
extern "C" void __cudaUnregisterFatBinary(void **) {}
namespace {
void *early_fat=nullptr;
void early_stub() {}
__attribute__((constructor)) void early_registration() {
    // This provider constructor can precede the preload constructor; its
    // destructor can follow trace_end. Both are host-only ordering probes.
    __cudaRegisterFunction(&early_fat,reinterpret_cast<const char *>(&early_stub),nullptr,
        "mock-non-main-function",0,nullptr,nullptr,nullptr,nullptr,nullptr);
}
__attribute__((destructor)) void late_unregister() { __cudaUnregisterFatBinary(&early_fat); }
}
extern "C" cudaError_t __cudaGetKernel(cudaKernel_t *kernel,const void *) {
    *kernel=reinterpret_cast<cudaKernel_t>(uintptr_t(0xbeef)); return cudaSuccess;
}
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t,dim3 grid,dim3 block,void **args,size_t shared,cudaStream_t stream) {
    return cudaLaunchKernel(nullptr,grid,block,args,shared,stream);
}
extern "C" cublasStatus_t cublasGemmEx(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,
        int m,int n,int k,const void *alpha,const void *a,cudaDataType_t at,int lda,
        const void *b,cudaDataType_t bt,int ldb,const void *beta,void *c,cudaDataType_t ct,
        int ldc,cublasComputeType_t compute,cublasGemmAlgo_t algo) {
    // Deliberate small host pointers. This provider never computes a GEMM.
    if (h!=reinterpret_cast<cublasHandle_t>(uintptr_t(0x1234)) || ta!=CUBLAS_OP_T || tb!=CUBLAS_OP_N ||
        m!=4 || n!=2 || k!=8 || !a || !b || !c || at!=CUDA_R_16F || bt!=CUDA_R_16F ||
        ct!=CUDA_R_32F || lda!=8 || ldb!=8 || ldc!=4 || compute!=CUBLAS_COMPUTE_32F ||
        algo!=CUBLAS_GEMM_ALGO3_TENSOR_OP) return CUBLAS_STATUS_INVALID_VALUE;
    if (pointer_mode==CUBLAS_POINTER_MODE_HOST &&
        (*static_cast<const float *>(alpha)!=1.0f || *static_cast<const float *>(beta)!=0.0f))
        return CUBLAS_STATUS_INVALID_VALUE;
    ++gemms; return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,
        int m,int n,int k,const void *alpha,const void *a,cudaDataType_t at,int lda,long long int sa,
        const void *b,cudaDataType_t bt,int ldb,long long int sb,const void *beta,void *c,cudaDataType_t ct,
        int ldc,long long int sc,int batches,cublasComputeType_t compute,cublasGemmAlgo_t algo) {
    // One batch means these deliberately >32-bit strides never form an offset;
    // they detect ABI truncation/reordered arguments without giant allocations.
    if (sa!=(1LL<<34)+32 || sb!=(1LL<<35)+16 || sc!=(1LL<<36)+8 || batches!=1)
        return CUBLAS_STATUS_INVALID_VALUE;
    return cublasGemmEx(h,ta,tb,m,n,k,alpha,a,at,lda,b,bt,ldb,beta,c,ct,ldc,compute,algo);
}
