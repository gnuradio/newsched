namespace gr {
namespace blocks {
__global__ void multiply_const_kernel(const float* in, float* out, float k, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = k * in[i];
}

void exec_multiply_const_kernel(const float* in, float* out, int grid_size, int block_size, int N, float k)
{
    // multiply_const_kernel<<<(N + nthreads - 1) / nthreads, nthreads>>>(N, k, in, out);
    multiply_const_kernel<<<grid_size, block_size>>>(in, out, k, N);
}

void get_block_and_grid_multiply_const(int* minGrid, int* minBlock)
{
    // https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, multiply_const_kernel, 0, 0);
}

} // namespace blocks
} // namespace gr