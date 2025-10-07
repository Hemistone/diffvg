#pragma once

// CUDA runtime headers can trigger toolchain issues in some units (e.g.,
// NVCC 12.8 with certain includes). Allow opting out per-translation-unit by
// defining DIFFVG_NO_CUDA_RUNTIME_INCLUDES before including this header.
#if !defined(DIFFVG_NO_CUDA_RUNTIME_INCLUDES)
#ifdef __CUDACC__
    #include <cuda.h>
    #include <cuda_runtime.h>
#elif defined(COMPILE_WITH_CUDA)
    #include <cuda_runtime_api.h>
#endif
#else
#if defined(COMPILE_WITH_CUDA)
extern "C" {
    typedef enum cudaError cudaError_t;
    const char* cudaGetErrorString(cudaError_t error);
    cudaError_t cudaDeviceSynchronize(void);
}
#endif
#endif
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <limits>

#if (defined(__CUDACC__) || defined(COMPILE_WITH_CUDA)) && !defined(DIFFVG_NO_CUDA_RUNTIME_INCLUDES)
#define checkCuda(x) do { if((x)!=cudaSuccess) { \
    std::fprintf(stderr, "CUDA Runtime Error: %s at %s:%d\n",\
    cudaGetErrorString(x), __FILE__, __LINE__);\
    std::abort();}} while(0)
#elif defined(DIFFVG_NO_CUDA_RUNTIME_INCLUDES) && defined(COMPILE_WITH_CUDA)
inline void diffvg_check_cuda(cudaError_t result, const char *file, int line) {
    if (result != 0) {
        const char *msg = cudaGetErrorString(result);
        if (msg == nullptr) {
            msg = "Unknown CUDA error";
        }
        std::fprintf(stderr, "CUDA Runtime Error: %s at %s:%d\n", msg, file, line);
        std::abort();
    }
}
#define checkCuda(x) diffvg_check_cuda((x), __FILE__, __LINE__)
#else
// Fallback no-op checker when CUDA runtime is not included in this TU
#ifndef checkCuda
#define checkCuda(x) do { auto _diffvg_cuda_result = (x); (void)_diffvg_cuda_result; } while(0)
#endif
#endif

template <typename T>
DEVICE
inline T infinity() {
#ifdef __CUDA_ARCH__
    const unsigned long long ieee754inf = 0x7ff0000000000000;
    return __longlong_as_double(ieee754inf);
#else
    return std::numeric_limits<T>::infinity();
#endif
}

template <>
DEVICE
inline double infinity() {
#ifdef __CUDA_ARCH__
    return __longlong_as_double(0x7ff0000000000000ULL);
#else
    return std::numeric_limits<double>::infinity();
#endif
}

template <>
DEVICE
inline float infinity() {
#ifdef __CUDA_ARCH__
    return __int_as_float(0x7f800000);
#else
    return std::numeric_limits<float>::infinity();
#endif
}

inline void cuda_synchronize() {
#if (defined(__CUDACC__) || defined(COMPILE_WITH_CUDA)) && !defined(DIFFVG_NO_CUDA_RUNTIME_INCLUDES)
    checkCuda(cudaDeviceSynchronize());
#elif defined(DIFFVG_NO_CUDA_RUNTIME_INCLUDES) && defined(COMPILE_WITH_CUDA)
    checkCuda(cudaDeviceSynchronize());
#endif
}
