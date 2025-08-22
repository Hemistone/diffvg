#pragma once

// CUDA runtime headers can trigger toolchain issues in some units (e.g.,
// NVCC 12.8 with certain includes). Allow opting out per-translation-unit by
// defining DIFFVG_NO_CUDA_RUNTIME_INCLUDES before including this header.
#if defined(__CUDACC__) && !defined(DIFFVG_NO_CUDA_RUNTIME_INCLUDES)
    #include <cuda.h>
    #include <cuda_runtime.h>
#endif
#include <cstdio>
#include <cassert>
#include <limits>

#if defined(__CUDACC__) && !defined(DIFFVG_NO_CUDA_RUNTIME_INCLUDES)
#define checkCuda(x) do { if((x)!=cudaSuccess) { \
    printf("CUDA Runtime Error: %s at %s:%d\n",\
    cudaGetErrorString(x),__FILE__,__LINE__);\
    exit(1);}} while(0)
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
#if defined(__CUDACC__) && !defined(DIFFVG_NO_CUDA_RUNTIME_INCLUDES)
    checkCuda(cudaDeviceSynchronize());
#endif
}
