#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

// The block cuda file is just a wrapper for the kernels that will be launched in the work
// function
namespace gr {
namespace blocks {
__global__ void apply_copy_kernel(const uint8_t* in, uint8_t* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i];
    }
}

void apply_copy(
    const uint8_t* in, uint8_t* out, int n, int grid_size, int block_size, cudaStream_t stream)
{
    apply_copy_kernel<<<grid_size, block_size, 0, stream>>>(in, out, n);
}

void get_block_and_grid(int* minGrid, int* minBlock)
{
    // https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, apply_copy_kernel, 0, 0);
}
} // namespace blocks
} // namespace gr