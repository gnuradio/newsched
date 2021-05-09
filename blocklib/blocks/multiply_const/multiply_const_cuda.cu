#include <cuComplex.h>

namespace gr {
namespace blocks {

template <typename T>
__global__ 
void multiply_const_kernel(const T* in, T* out, T k, size_t n)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = k * in[i];
}

template <>
__global__ 
void multiply_const_kernel(const cuFloatComplex* in, cuFloatComplex* out, cuFloatComplex k, size_t n)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i].x = k.x * in[i].x - k.y*in[i].y;
        out[i].y = k.x * in[i].y + k.y*in[i].x;
    }
}

template <typename T>
void exec_multiply_const_kernel(
    const T* in, T* out, T k, int grid_size, int block_size, cudaStream_t stream)
{
    multiply_const_kernel<T><<<grid_size, block_size, 0, stream>>>(in, out, k, block_size * grid_size);
} 

template <typename T>
void get_block_and_grid_multiply_const(int* minGrid, int* minBlock)
{
    // https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, multiply_const_kernel<T>, 0, 0);
}

template void exec_multiply_const_kernel<float>(
    const float* in, float* out, float k, int grid_size, int block_size, cudaStream_t stream);
template void get_block_and_grid_multiply_const<float>(int* minGrid, int* minBlock);

} // namespace blocks
} // namespace gr