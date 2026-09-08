// CPU mock API calls only; linked solely against sm75-runtime-contract-mock.
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(expression) do { if (!(expression)) { \
    std::fprintf(stderr,"host-mock CHECK failed at line %d\n",__LINE__); std::abort(); \
} } while (0)
#include <cstdint>
extern "C" int sm75_mock_counts(int);
extern "C" void __cudaRegisterFunction(void **,const char *,char *,const char *,int,uint3 *,uint3 *,dim3 *,dim3 *,int *);
extern "C" void __cudaUnregisterFatBinary(void **);
extern "C" cudaError_t __cudaGetKernel(cudaKernel_t *,const void *);
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t,dim3,dim3,void **,size_t,cudaStream_t);
static void conversion_stub() {}
int main() {
    CHECK(sm75_mock_counts(0)==0 && sm75_mock_counts(1)==0 && sm75_mock_counts(2)==0);
    void *a=nullptr,*b=nullptr,*c=nullptr;
    CHECK(cudaSetDevice(0)==cudaSuccess);
    CHECK(cudaMalloc(&a,64)==cudaSuccess);
    CHECK(cudaMalloc(&b,32)==cudaSuccess);
    CHECK(cudaMalloc(&c,32)==cudaSuccess);
    cublasHandle_t h=nullptr; float alpha=1.0f,beta=0.0f;
    CHECK(cublasCreate(&h)==CUBLAS_STATUS_SUCCESS);
    CHECK(cublasSetMathMode(h,CUBLAS_DEFAULT_MATH)==CUBLAS_STATUS_SUCCESS);
    CHECK(cublasSetStream(h,nullptr)==CUBLAS_STATUS_SUCCESS);
    CHECK(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,4,2,8,&alpha,a,CUDA_R_16F,8,
        b,CUDA_R_16F,8,&beta,c,CUDA_R_32F,4,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_ALGO3_TENSOR_OP)==CUBLAS_STATUS_SUCCESS);
    CHECK(cublasGemmStridedBatchedEx(h,CUBLAS_OP_T,CUBLAS_OP_N,4,2,8,&alpha,a,CUDA_R_16F,8,(1LL<<34)+32,
        b,CUDA_R_16F,8,(1LL<<35)+16,&beta,c,CUDA_R_32F,4,(1LL<<36)+8,1,
        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_ALGO3_TENSOR_OP)==CUBLAS_STATUS_SUCCESS);
    // Invalid host addresses in DEVICE pointer mode must never be inspected.
    CHECK(cublasSetPointerMode(h,CUBLAS_POINTER_MODE_DEVICE)==CUBLAS_STATUS_SUCCESS);
    CHECK(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,4,2,8,reinterpret_cast<void *>(uintptr_t(1)),a,CUDA_R_16F,8,
        b,CUDA_R_16F,8,reinterpret_cast<void *>(uintptr_t(2)),c,CUDA_R_32F,4,
        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_ALGO3_TENSOR_OP)==CUBLAS_STATUS_SUCCESS);
    CHECK(cudaLaunchKernel(reinterpret_cast<void *>(uintptr_t(0x5678)),dim3(2),dim3(32),
        reinterpret_cast<void **>(uintptr_t(1)),64,nullptr)==cudaSuccess);
    void *fat=nullptr;
    __cudaRegisterFunction(&fat,reinterpret_cast<const char *>(&conversion_stub),nullptr,
        "_Z17f32_to_f16_kernelP6__halfPKfm",0,nullptr,nullptr,nullptr,nullptr,nullptr);
    cudaKernel_t kernel=nullptr;
    CHECK(__cudaGetKernel(&kernel,reinterpret_cast<const void *>(&conversion_stub))==cudaSuccess);
    size_t count=8; void *conversion_args[]={&b,&a,&count};
    CHECK(__cudaLaunchKernel(kernel,dim3(2),dim3(32),conversion_args,64,nullptr)==cudaSuccess);
    // The same recognized kernel with unreadable argument-array storage records
    // unknown metadata; it is never directly dereferenced by the interposer.
    CHECK(__cudaLaunchKernel(kernel,dim3(2),dim3(32),reinterpret_cast<void **>(uintptr_t(1)),64,nullptr)==cudaSuccess);
    __cudaUnregisterFatBinary(&fat);
    // An unregistered stale handle must no longer receive a recognized decoder.
    CHECK(__cudaLaunchKernel(kernel,dim3(2),dim3(32),reinterpret_cast<void **>(uintptr_t(1)),64,nullptr)==cudaSuccess);
    CHECK(cudaDeviceSynchronize()==cudaSuccess);
    CHECK(sm75_mock_counts(0)==3 && sm75_mock_counts(1)==4 && sm75_mock_counts(2)==1);
    CHECK(cublasDestroy(h)==CUBLAS_STATUS_SUCCESS);
    CHECK(cudaFree(a)==cudaSuccess && cudaFree(b)==cudaSuccess && cudaFree(c)==cudaSuccess);
    return 0;
}
