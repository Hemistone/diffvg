// Keep the GPU sort implementation minimal and independent of host-only
// headers to avoid triggering NVCC toolchain issues on some setups (e.g.,
// WSL2 + CUDA 12.x). Use CUB's DeviceRadixSort, which is part of the CUDA
// Toolkit (CCCL) and generally compiles more reliably than the full Thrust
// sort stack in isolated translation units.

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

extern "C" void diffvg_gpu_sort_by_key_uint_uint(uint32_t* keys, int* vals, size_t n) {
    if (n <= 1) return;

    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    uint32_t* keys_out = nullptr;
    int* vals_out = nullptr;

    // Allocate output buffers on device. Keys/vals may be Unified Memory; CUB
    // supports device-accessible pointers. Using separate outputs avoids the
    // need for in-place ping-pong buffers.
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&keys_out), n * sizeof(uint32_t));
    if (err != cudaSuccess) return; // bail in release; upstream has assert builds
    err = cudaMalloc(reinterpret_cast<void**>(&vals_out), n * sizeof(int));
    if (err != cudaSuccess) { cudaFree(keys_out); return; }

    // Query temp storage size, then allocate and perform the sort.
    cub::DeviceRadixSort::SortPairs(
        temp_storage, temp_storage_bytes,
        keys, keys_out, vals, vals_out,
        n, /*begin_bit*/ 0, /*end_bit*/ 32);

    err = cudaMalloc(&temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) { cudaFree(keys_out); cudaFree(vals_out); return; }

    cub::DeviceRadixSort::SortPairs(
        temp_storage, temp_storage_bytes,
        keys, keys_out, vals, vals_out,
        n, /*begin_bit*/ 0, /*end_bit*/ 32);

    // Copy results back to the original buffers (device-to-device is fine for
    // both device and managed memory).
    cudaMemcpy(keys, keys_out, n * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vals, vals_out, n * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaFree(temp_storage);
    cudaFree(keys_out);
    cudaFree(vals_out);
}

