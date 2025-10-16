#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cstdint>

extern "C" void diffvg_gpu_sort_by_key_uint_uint(uint32_t* keys, int* vals, size_t n) {
    auto d_keys = thrust::device_pointer_cast(keys);
    auto d_vals = thrust::device_pointer_cast(vals);
    thrust::sort_by_key(thrust::device, d_keys, d_keys + n, d_vals);
}

