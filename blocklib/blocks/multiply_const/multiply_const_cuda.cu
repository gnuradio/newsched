#include "multiply_const_cuda.cuh"
#include <cuComplex.h>
#include <complex>

namespace gr {
namespace blocks {
namespace multiply_const_cu {

template <typename T>
__global__ void multiply_const_kernel(const T* in, T* out, T k, size_t n)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = k * in[i];
}

template <>
__global__ void multiply_const_kernel(const cuFloatComplex* in,
                                      cuFloatComplex* out,
                                      cuFloatComplex k,
                                      size_t n)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i].x = k.x * in[i].x - k.y * in[i].y;
        out[i].y = k.x * in[i].y + k.y * in[i].x;
    }
}

template <typename T>
void exec_kernel(
    const T* in, T* out, T k, int grid_size, int block_size, cudaStream_t stream)
{
    multiply_const_kernel<T>
        <<<grid_size, block_size, 0, stream>>>(in, out, k, block_size * grid_size);
}

// template <>
// void exec_kernel(const std::complex<float>* in,
//                  std::complex<float>* out,
//                  std::complex<float> k,
//                  int grid_size,
//                  int block_size,
//                  cudaStream_t stream)
// {
//     multiply_const_kernel<cuFloatComplex>
//         <<<grid_size, block_size, 0, stream>>>(reinterpret_cast<const cuFloatComplex*>(in),
//                                                reinterpret_cast<cuFloatComplex*>(out),
//                                                make_cuFloatComplex(real(k), imag(k)),
//                                                block_size * grid_size);
// }

template <typename T>
void get_block_and_grid(int* minGrid, int* minBlock)
{
    // https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, multiply_const_kernel<T>, 0, 0);
}

#define IMPLEMENT_KERNEL(T)                                                            \
    template void exec_kernel<T>(                                                      \
        const T* in, T* out, T k, int grid_size, int block_size, cudaStream_t stream); \
    template void get_block_and_grid<T>(int* minGrid, int* minBlock);


IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(float)
IMPLEMENT_KERNEL(cuFloatComplex)

} // namespace multiply_const_cu
} // namespace blocks
} // namespace gr